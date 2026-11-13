"""T1 content control and T2 EMA gate intervention on paired human prefixes.

These outcomes measure computational influence and human fit, NOT reward VOC.
Checkpoints and the frozen capture runtime are immutable inputs.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from forced_budget_probe import (ROOT, RUNTIME, BudgetProbe, ReadoutReady,
                                 DynamicImitationRunner, digest, util, verify_pairing)
from thinker.dynamic_imitation import _voc_trace_diagnostics

# Preserve phase, validity/reset masks and timing. Remove updated content and
# imagined action/reward feedback, including at the final real-head readout.
FROZEN_FIELDS = ("tree_reps", "xs", "hs", "reward", "last_pri", "last_reset",
                 "last_search_control", "done", "real_done", "truncated_done")


def select_targets(actions, max_edge=256):
    actions = np.asarray(actions).reshape(-1)
    found = {}
    for edge in range(1, min(len(actions), max_edge + 1)):
        category = ("NOOP" if actions[edge-1] == 0 else "active") + "_to_" + ("NOOP" if actions[edge] == 0 else "active")
        found.setdefault(category, edge)
    return sorted(found.values()), sorted(set(["NOOP_to_NOOP", "NOOP_to_active",
                                               "active_to_NOOP", "active_to_active"]) - set(found))


def gate_choice(probability, uniform, conditional_logits):
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid continue probability")
    return int(np.argmax(conditional_logits)) if uniform < probability else util.STOP


class InterventionProbe(BudgetProbe):
    def __init__(self, target, budget, mode, flags, ema, gate_seed, schedule=None):
        super().__init__(target, budget)
        self.mode, self.flags, self.ema = mode, flags, ema
        self.schedule = schedule
        self.draws = random.Random(gate_seed)
        self.content = None
        self.features = None
        self.calls = 0
        self.tokens = 0
        self.choices = []
        self.action_schedule = []
        self.token_masks = []
        self.content_hashes = []

    def before(self, module, args, kwargs):
        super().before(module, args, kwargs)
        if self.cursor != self.target:
            return None
        env = kwargs.get("env_out", args[0] if args else None)
        self.calls += 1
        mask = bool(env.tree_token_valid.reshape(-1)[0])
        self.tokens += int(mask)
        self.token_masks.append(mask)
        if self.content is None:
            self.content = {k: copy.deepcopy(getattr(env, k)) for k in FROZEN_FIELDS}
        if self.mode == "frozen":
            env = env._replace(**copy.deepcopy(self.content))
        self.content_hashes.append(digest({k: getattr(env, k) for k in FROZEN_FIELDS}))
        if self.mode == "frozen":
            if "env_out" in kwargs:
                return args, dict(kwargs, env_out=env)
            return (env, *args[1:]), kwargs
        return None

    def capture_features(self, module, args):
        self.features = args[0].detach()

    def after(self, module, args, kwargs, output):
        actor, state = output
        if self.cursor != self.target:
            return None
        policy = int(actor.policy_type[-1].reshape(-1)[0])
        valid = bool(actor.policy_valid[-1].reshape(-1)[0])
        if not valid:
            return None
        if policy == util.POLICY_REAL:
            if self.mode in ("fixed", "frozen") and self.computations != self.budget:
                raise AssertionError("Fixed budget was not executed")
            logits = actor.pri_param[-1].reshape(-1, module.num_actions)[0].detach().double()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite real-head logits")
            self.result = dict(**self.entry, target_edge=self.target, mode=self.mode,
                               budget=self.budget, actual_computations=self.computations,
                               probability=torch.softmax(logits, -1).cpu().tolist(), argmax=int(logits.argmax()),
                               actor_calls=self.calls, recurrent_tokens=self.tokens, token_masks=self.token_masks,
                               content_hashes=self.content_hashes, choices=self.choices,
                               action_schedule=self.action_schedule, target_human_action_not_executed=True)
            if self.mode == "frozen" and len(set(self.content_hashes)) != 1:
                raise AssertionError("New model content leaked into the frozen-input control")
            raise ReadoutReady()
        if policy != util.POLICY_SEARCH:
            return None
        primary, control = actor.action
        if actor.voc_q is not None:
            diag = _voc_trace_diagnostics(actor, self.features, flags=self.flags,
                                         ema_head_state=self.ema, effective_think_cost=None)
            q = diag["ema_q"].reshape(-1, 2)[0]
            online = diag["online_q"].reshape(-1, 2)[0]
            delta = float(q[0] - q[1])
            learned = float(diag["soft_continue_probability"].reshape(-1)[0])
            teacher = float(torch.sigmoid((q[0] - q[1]) / float(self.flags.voc_gate_q_temperature)))
            diagnostic = dict(ema_delta_q=delta, online_delta_q=float(online[0]-online[1]),
                              teacher_probability=teacher, learned_probability=learned)
        else:
            if self.mode in ("learned", "ema"):
                raise ValueError("Gate interventions require a VOC head")
            diagnostic = {}
        conditional = actor.search_control_logits[-1].reshape(-1, 3)[0, :2].detach().cpu().numpy()
        if self.mode in ("fixed", "frozen"):
            chosen = util.STOP if self.computations >= self.budget else int(np.argmax(conditional))
        else:
            uniform = self.draws.random()
            prob = learned if self.mode == "learned" else teacher
            chosen = gate_choice(prob, uniform, conditional)
            diagnostic.update(uniform=uniform, executed_probability=prob)
        if self.mode == "frozen":
            saved = self.schedule[len(self.action_schedule)]
            chosen = saved["control"]
            primary = torch.as_tensor(saved["primary"], dtype=primary.dtype, device=primary.device).reshape_as(primary)
        self.action_schedule.append(dict(primary=primary.detach().cpu().tolist(), control=chosen))
        self.choices.append(dict(computation_index=self.computations, chosen=chosen, **diagnostic))
        return actor._replace(action=(primary, torch.full_like(control, chosen))), state


def run_intervention(actor, model, flags, batch, *, target, budget, mode, seed, ema, gate_seed=0, schedule=None):
    if mode not in ("fixed", "frozen", "learned", "ema"):
        raise ValueError(mode)
    if mode in ("fixed", "frozen") and not 0 <= budget < int(flags.max_search_steps):
        raise ValueError("Fixed budget must be below the safety cap")
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    runner = DynamicImitationRunner(actor, model, flags, device="cpu")
    probe = InterventionProbe(target, budget, mode, flags, ema, gate_seed, schedule)
    probe.runner = runner
    handles = [actor.register_forward_pre_hook(probe.before, with_kwargs=True),
               actor.register_forward_hook(probe.after, with_kwargs=True)]
    if hasattr(actor, "voc_head"):
        handles.append(actor.voc_head.register_forward_pre_hook(probe.capture_features))
    try:
        with torch.no_grad():
            runner.stream_rollout_episode(batch, event_sink=probe, tree_carry=True,
                                          policy_mode="checkpoint", voc_ema_head_state_dict=ema)
    except ReadoutReady:
        pass
    finally:
        for handle in handles:
            handle.remove()
        runner.close()
    if probe.result is None:
        raise RuntimeError("Episode ended before target readout")
    return probe.result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "behavioral_data_block")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget", type=int, default=3)
    p.add_argument("--max-target-edge", type=int, default=256)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.use_deterministic_algorithms(True)
    import visual_behav as vb
    bundle = vb.resolve_checkpoint_bundle(args.checkpoint, requested_step=5_000_000)
    config = SimpleNamespace(checkpoint=bundle, torch_threads_per_worker=args.threads,
                             data_root=str(args.data_root), base_seed=args.seed, episode_indices=(0,))
    runtime = vb._CaptureRuntime(config, vb.WorkerSpec(rank=0, device="cpu"))
    sources = vb.enumerate_source_bundles(args.data_root, selectors=[args.source], hash_workers=1)
    tasks = vb._coordinator_source_tasks(config, sources)
    if len(tasks) != 1:
        raise ValueError("Select one source")
    task = tasks[0]
    expected = {task.source.path: {k: getattr(task.source, k) for k in ("size", "sha256", "device", "inode", "mtime_ns")}}
    source = runtime.trace_api.BehaviorEpisodeSource(vb._SourceLoaderView(task), expected_sources=expected,
                allowed_source_paths=[task.source.path], scratch_dir=args.output / "scratch")
    batch = None
    rows = []
    try:
        descriptor = next(d for d in source.episodes if d.local_episode_index == 0)
        if int(descriptor.session) not in {int(s) for s in str(runtime.flags.icopro_holdout_sessions).split(",")}:
            raise ValueError("Source is not held out")
        batch = vb._episode_batch(source, descriptor)
        targets, missing = select_targets(batch["actions_seq"][0], args.max_target_edge)
        runtime._ensure_runner(batch); runtime.runner.close()
        before = digest(dict(actor=runtime.actor_net.state_dict(), model=runtime.model_net.state_dict()))
        for target in targets:
            group = []
            # Common random numbers for T2; independent of torch/planner RNG.
            gate_seed = args.seed * 1000003 + target * 9176 + int(descriptor.subject) * 31 + int(descriptor.game)
            for name, mode, budget in [("stop", "fixed", 0), ("full", "fixed", args.budget),
                                       ("frozen", "frozen", args.budget), ("learned", "learned", -1),
                                       ("ema", "ema", -1), ("stop_repeat", "fixed", 0)]:
                result = run_intervention(runtime.actor_net, runtime.model_net, runtime.flags, batch,
                    target=target, budget=budget, mode=mode, seed=args.seed,
                    ema=runtime.voc_ema_head_state_dict, gate_seed=gate_seed,
                    schedule=group[1]["action_schedule"] if name == "frozen" else None)
                human, previous = (int(batch["actions_seq"][0, i]) for i in (target, target-1))
                result.update(condition=name, human_action=human, previous_action=previous,
                    transition=("NOOP" if previous == 0 else "active") + "_to_" + ("NOOP" if human == 0 else "active"),
                    observation_source_index=int(batch["observation_source_index"][target]), gate_seed=gate_seed)
                group.append(result)
                print(f"edge={target} {name}: computations={result['actual_computations']} argmax={result['argmax']}", flush=True)
            verify_pairing(group)
            assert group[1]["token_masks"] == group[2]["token_masks"]
            assert group[1]["action_schedule"] == group[2]["action_schedule"]
            rows.extend(group)
            (args.output / "decisions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        assert before == digest(dict(actor=runtime.actor_net.state_dict(), model=runtime.model_net.state_dict()))
        manifest = dict(status="COMPLETE", checkpoint=asdict(bundle), source=asdict(task.source), runtime=str(RUNTIME),
            subject=int(descriptor.subject), game=int(descriptor.game), session=int(descriptor.session),
            episode=0, seed=args.seed, budget=args.budget, targets=targets, missing_strata=missing,
            target_selection=f"first of each human transition/stay category within zero-based edges 1..{args.max_target_edge}",
            frozen_fields=FROZEN_FIELDS, frozen_schedule="same imagined primary/control sequence as full arm",
            t2_sampling="one shared uniform stream per paired decision; conditional PROCEED/RESET argmax",
            t2_compute_matching="not matched; report calculation-count difference separately",
            primary_outcome="computational influence, not reward utility", reward_voc_validated=False,
            state_pairing="exact common-prefix input/core/RNG/tree-observation hashes; repeated stop readout",
            parameter_sha256=before, runtime_visual_sha256=hashlib.sha256((RUNTIME/'visual_behav.py').read_bytes()).hexdigest(),
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            helper_sha256=hashlib.sha256((Path(__file__).parent/'forced_budget_probe.py').read_bytes()).hexdigest(),
            limitations=["one early episode per held-out checkpoint; selected strata, not natural event frequency",
                         "frozen root content is an artificial intervention, not an independently trained baseline",
                         "T2 is a one-decision intervention with one random stream, not whole-policy evaluation",
                         "human fit and internal critic estimates do not validate reward-minus-cost VOC"])
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
        for filename in ("intervention_probe.py", "forced_budget_probe.py"):
            (args.output/filename).write_bytes((Path(__file__).parent/filename).read_bytes())
    finally:
        if batch is not None:
            batch["observation_provider"].close()
        if runtime.runner is not None:
            runtime.runner.close()
        source.close()


if __name__ == "__main__":
    main()
