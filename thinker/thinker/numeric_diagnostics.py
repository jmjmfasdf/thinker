"""Detached, bounded failure captures for dynamic imitation training.

No values are clipped, replaced, or skipped. The optional stop threshold is
for diagnostic training attempts; the default only records unusual batches.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "_asdict"):
        return {k: cpu_copy(v) for k, v in value._asdict().items()}
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def tensor_range(value):
    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        return {}
    with torch.no_grad():
        v = value.detach()
        return {"shape": list(v.shape), "dtype": str(v.dtype),
                "absmax": float(v.abs().max()),
                "finite": bool(torch.isfinite(v).all())}


class ImitationNumericTrace:
    def __init__(self, learner, batch):
        self.learner = learner
        self.batch = batch
        self.events = []
        self.handles = []
        self.maxima = {}
        self.rng = {"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        self.capture_max = float(getattr(learner, "_numeric_bc_capture_max", 0.))
        self.trigger = max(1e4, self.capture_max * 10.)
        self.stop_at = float(os.environ.get("DI72_NUMERIC_STOP_AT_LOGIT", "0"))
        self.trigger_event = None

    def __enter__(self):
        model = self.learner.bc_model_net
        for prefix in ("sr_net", "vp_net"):
            net = getattr(model, prefix, None)
            for name in ("encoder", "RNN"):
                module = getattr(net, name, None)
                if isinstance(module, torch.nn.Module):
                    self.handles.append(module.register_forward_hook(
                        self._hook(prefix + "." + name), with_kwargs=True))
        actor = self.learner.actor_net
        for name in ("h_encoder", "tree_rep_encoder", "final_layer", "policy"):
            module = getattr(actor, name, None)
            if isinstance(module, torch.nn.Module):
                self.handles.append(module.register_forward_hook(
                    self._hook("actor." + name), with_kwargs=True))
        self.handles.append(actor.register_forward_pre_hook(self._actor_before))
        self.handles.append(actor.register_forward_hook(self._actor_after))
        return self

    def _hook(self, name):
        def record(module, args, kwargs, output):
            del module
            source = kwargs.get("h", args[0] if args else None)
            target = output[0] if isinstance(output, (tuple, list)) else output
            row = {"module": name, "input": tensor_range(source),
                   "output": tensor_range(target)}
            self.events.append(row)
            value = row["output"].get("absmax", 0.)
            self.maxima[name] = max(self.maxima.get(name, 0.), value)
        return record

    def _actor_before(self, module, args):
        del module
        self.last_actor_input = args

    def _actor_after(self, module, args, output):
        del module
        out = output[0]
        env = args[0]
        value = tensor_range(getattr(out, "pri_param", None))
        self.events.append({"module": "actor", "logits": value,
                            "hs": tensor_range(getattr(env, "hs", None)),
                            "tree": tensor_range(getattr(env, "tree_reps", None)),
                            "phase": cpu_copy(getattr(env, "phase", None)),
                            "search_steps": cpu_copy(getattr(env, "search_steps", None))})
        magnitude = value.get("absmax", 0.)
        previous = self.maxima.get("actor.logits", 0.)
        if (magnitude >= self.trigger and magnitude > previous) or not value.get("finite", True):
            self.trigger_event = {"env_out": cpu_copy(env),
                                  "actor_state": cpu_copy(args[1]),
                                  "actor_out": cpu_copy(out)}
        self.maxima["actor.logits"] = max(previous, magnitude)

    def save(self, reason, result=None):
        learner = self.learner
        target = Path(learner.flags.ckpdir) / "numeric_diagnostics"
        target.mkdir(exist_ok=True)
        step = int(getattr(learner, "real_step", 0))
        name = "failure" if reason != "large_finite_bc" else "large"
        path = target / f"bc-{name}-{step}.pt"
        payload = {"schema": 1, "reason": reason, "real_step": step,
                   "batch": cpu_copy(self.batch), "rng_before": self.rng,
                   "events": self.events, "maxima": self.maxima,
                   "trigger_event": self.trigger_event,
                   "actor_state_dict": cpu_copy(learner.actor_net.state_dict()),
                   "model_state_dict": cpu_copy(learner.bc_model_net.state_dict()),
                   "actor_optimizer_state": cpu_copy(learner.optimizer.state_dict()),
                   "flags": dict(vars(learner.flags)),
                   "actor_device": str(getattr(learner, "device", "unknown")),
                   "stop_at_logit": self.stop_at,
                   "attempt": os.environ.get("SLURM_ARRAY_JOB_ID", "") + "_" + os.environ.get("SLURM_ARRAY_TASK_ID", "")}
        if result is not None:
            payload["result"] = {key: cpu_copy(getattr(result, key)) for key in
                                 ("loss", "logits", "targets", "normalized_ce", "margin_loss", "pvp_loss")}
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        print("[numeric-capture] " + json.dumps({"path": str(path), "reason": reason,
                                                "step": step, "maxima": self.maxima}), flush=True)
        learner._numeric_bc_capture_max = max(self.maxima.get("actor.logits", 0.), self.capture_max)

    def finish(self, result):
        magnitude = self.maxima.get("actor.logits", 0.)
        self.learner._last_bc_numeric_maxima = dict(self.maxima)
        if not bool(torch.isfinite(result.loss)) or not bool(torch.isfinite(result.logits).all()):
            self.save("nonfinite_bc", result)
        elif magnitude >= self.trigger:
            self.save("large_finite_bc", result)
        if self.stop_at > 0. and magnitude >= self.stop_at:
            raise FloatingPointError(
                f"diagnostic stop: BC actor logits {magnitude} >= {self.stop_at}; exact batch saved")

    def __exit__(self, kind, exception, tb):
        for handle in self.handles:
            handle.remove()
        if exception is not None:
            self.save(repr(exception))


def save_actor_failure(learner, data, exception, losses=None):
    target = Path(learner.flags.ckpdir) / "numeric_diagnostics"
    target.mkdir(exist_ok=True)
    path = target / f"actor-failure-{int(getattr(learner, 'real_step', 0))}.pt"
    payload = {"schema": 1, "reason": repr(exception), "data": cpu_copy(data),
               "losses": cpu_copy(losses), "flags": dict(vars(learner.flags)),
               "actor_state_dict": cpu_copy(learner.actor_net.state_dict()),
               "optimizer_state": cpu_copy(learner.optimizer.state_dict()),
               "torch_rng_state": torch.get_rng_state(),
               "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save(payload, path.with_suffix(".tmp"))
    os.replace(path.with_suffix(".tmp"), path)
    print(f"[numeric-capture] actor failure: {path}", flush=True)
