"""One-step BC meta-learning of discount and computation cost.

The real learner owns ordinary actor/Q/world-model updates. This module owns
only two meta parameters. Virtual updates are stateless, use detached input
data and frozen optimizer history, and never step a live network optimizer.

Protocol (see config/planning_meta.yaml for a fresh-run example)
-------------------------------------------------------------
Let phi=(raw_gamma, log_lambda), theta be the current actor/critic weights,
and D_in the ordinary online unroll plus its original-order BC window.
The conditional virtual update is

    theta_plus(phi) = Adam(theta, grad_theta L_RL+BC(theta, phi; D_in)).

Its critic step, selective VOC->GRU VJP, clipping, frozen Adam moments,
EMA update and exact gate projection match the ordinary learner. The
bootstrap/network targets retain their normal semi-gradient convention,
but gamma/lambda dependence in the return targets is not detached. BC tree
arithmetic in the reconstruction also retains gamma derivatives. The
observed inner trajectories, optimizer history, recurrent initial state
and world-model parameters are conditioned upon, not differentiated as a
full history of training. There is no virtual ModelNet optimizer step.

An independently sampled human window D_out supplies mean action NLL after
the virtual update. It is a fitting batch from the SAME subject/game and
sessions, not a validation/test split. Human REAL actions are overrides;
SEARCH primary actions, branch controls and gate decisions are stochastic.
For each of two independent planning replicas, the derivative combines
pathwise dNLL/dphi with (suffix_NLL - peer_suffix_NLL) * d log P(SEARCH)/dphi.
This includes burn-in decisions' effects on the scored suffix, but no WAIT
dummies or human-action likelihood in the planning score term. Legal masks,
epsilon mixing, forced computation and forced stopping use native execution
probabilities. The estimator is unbiased for this CONDITIONAL outer planning
objective, not for the complete stochastic training process.

Only that outer BC objective is differentiated with respect to phi; neither
the real actor optimizer nor the real world-model optimizer owns phi.
There is no human-switch label, computation-count target, VoC-sign target,
or value-change regularizer. gamma is bounded in (1e-4,1-1e-4); lambda=exp
is positive. Nonfinite gradients fail before real updates; invalid meta
Adam candidates restore the previous meta state and stop the run.

Defaults: original-order burn-in 16 + scored 8, every 32 successful actor
updates after 100k online real steps, outer batch 4 with 2 replicas, meta
Adam lr 1e-3 and norm clip 1. Unsupported/frozen protocols fail closed.
No extra dataset or intermediate analysis output is read by this module.

At a committed meta step, actor buffers carry gamma/lambda/version with the
weights. Self-play installs them only at unroll boundaries, revaluing the
carried tree and its last emitted token without resetting GRU history.
The ordinary ModelLearner refreshes gamma at minibatch boundaries. Normal
off-policy actor/model lag remains; this is not an atomic cluster barrier.
Checkpoint state includes meta Adam, raw parameters, counters and outer RNG.
visual_behav uses the learned checkpoint values, not initial config values.

Interpretation: fitted parameters are conditional, effective model settings,
not uniquely identified psychological constants. Their usefulness or human
switch associations require empirical validation; this code does not impose
them. Existing checkpoint/output/frozen campaign artifacts are not rewritten.
"""
from contextlib import contextmanager
import copy
import math
import time

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from thinker.planning_meta_context import meta_scope


META_DEFAULTS = dict(
    planning_meta_enabled=False,
    planning_backend="cython",
    planning_meta_every=32,
    planning_meta_warmup_steps=100000,
    planning_meta_batch_size=4,
    planning_meta_samples=2,
    planning_meta_lr=1e-3,
    planning_meta_grad_clip=1.,
)
GAMMA_EPSILON = 1e-4
META_SCHEMA = 1


def configure_meta(flags):
    if not getattr(flags, "planning_meta_enabled", False):
        if getattr(flags, "planning_backend", "cython") not in ("cython", "torch"):
            raise ValueError("planning_backend must be cython or torch")
        return flags
    for key, default in META_DEFAULTS.items():
        if not hasattr(flags, key):
            setattr(flags, key, default)
    flags.planning_backend = "torch"
    required = dict(dynamic_search=True, dynamic_factorized_control=True,
                    dynamic_voc_mode="control", voc_gate_exact_projection=True,
                    voc_gru_connected=True, float16=False, actor_use_rms=False,
                    voc_critic_parameterization="common_contrast_v1",
                    ppo_k=1, return_norm_type=-1, wrapper_type=0,
                    critic_enc_type=0, think_cost_anneal=False, sep_actor_critic=False,
                    voc_gate_epsilon_greedy_execution=False, autotune=False,
                    noise_enable=False)
    for key, expected in required.items():
        if getattr(flags, key, None) != expected:
            raise ValueError(f"planning meta requires {key}={expected!r}")
    if getattr(flags, "voc_actor_policy_version_barrier", False):
        raise ValueError("planning meta does not alter frozen atomic VOC protocols")
    if not str(getattr(flags, "icopro_data_path", "")).strip():
        raise ValueError("planning meta requires behavioral data")
    if len(str(flags.icopro_subjects).split(",")) != 1:
        raise ValueError("planning meta fits one subject x game per run")
    if str(getattr(flags, "icopro_holdout_sessions", "4")).strip():
        raise ValueError("planning meta uses all-session fitting, not held-out validation")
    if int(getattr(flags, "icopro_supervised_freq", 1)) != 1:
        raise ValueError("planning meta requires an inner BC batch on every actor update")
    if int(getattr(flags, "icopro_burn_in_steps", 1)) != 16 or int(flags.batch_length) != 8:
        raise ValueError("planning meta requires burn-in 16 and scored length 8")
    for key in ("planning_meta_every", "planning_meta_batch_size"):
        if type(getattr(flags, key)) is not int or getattr(flags, key) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if flags.planning_meta_samples != 2:
        raise ValueError("planning meta uses exactly two independent paths per window")
    if flags.planning_meta_warmup_steps < 0:
        raise ValueError("planning_meta_warmup_steps must be nonnegative")
    for key in ("planning_meta_lr", "planning_meta_grad_clip"):
        if not math.isfinite(getattr(flags, key)) or getattr(flags, key) <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not GAMMA_EPSILON < flags.discounting < 1-GAMMA_EPSILON or not 0 < flags.think_cost < math.inf:
        raise ValueError("invalid initial meta gamma or computation cost")
    return flags


class PlanningMetaParameters(nn.Module):
    def __init__(self, gamma, cost, *, device=None, dtype=torch.float32):
        super().__init__()
        probability = (float(gamma)-GAMMA_EPSILON)/(1-2*GAMMA_EPSILON)
        if not 0 < probability < 1 or not math.isfinite(cost) or cost <= 0:
            raise ValueError("invalid meta parameter initialization")
        self.raw = nn.Parameter(torch.tensor([math.log(probability/(1-probability)),
                                             math.log(cost)], device=device, dtype=dtype))

    def values(self):
        return (GAMMA_EPSILON + (1-2*GAMMA_EPSILON)*self.raw[0].sigmoid(), self.raw[1].exp())


def clipped_gradients(gradients, limit):
    used = [g for g in gradients if g is not None]
    if not used or limit <= 0:
        return gradients
    norm = torch.stack([g.double().square().sum() for g in used]).sum().clamp_min(1e-60).sqrt()
    factor = (limit/(norm+1e-6)).clamp(max=1.)
    return tuple(None if g is None else g*factor.to(g.dtype) for g in gradients)


def functional_adam(parameters, gradients, optimizer, names_by_id, *, clip_limit=0):
    """An ordinary Adam step, including its frozen moment history, without I/O."""
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError("meta functional optimizer supports ordinary Adam only")
    result = dict(parameters)
    names, owned = [], []
    for group in optimizer.param_groups:
        for p in group["params"]:
            name = names_by_id[id(p)]
            names.append(name)
            owned.append(gradients.get(name))
    clipped = dict(zip(names, clipped_gradients(tuple(owned), clip_limit)))
    for group in optimizer.param_groups:
        if group.get("amsgrad", False) or group.get("maximize", False) or group.get("weight_decay", 0):
            raise ValueError("unsupported Adam variant in planning meta")
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            name = names_by_id[id(p)]
            g = clipped[name]
            if g is None:
                continue
            state = optimizer.state.get(p, {})
            m0 = state.get("exp_avg", torch.zeros_like(g)).detach()
            v0 = state.get("exp_avg_sq", torch.zeros_like(g)).detach()
            step = int(state.get("step", 0)) + 1
            m = beta1*m0 + (1-beta1)*g
            v = beta2*v0 + (1-beta2)*g.square()
            # A smooth finite extension at exactly zero avoids 0 * inf in a
            # mixed derivative; its FP32 forward difference is negligible.
            denominator = v.clamp_min(1e-30).sqrt()/math.sqrt(1-beta2**step) + group["eps"]
            result[name] = parameters[name] - (group["lr"]/(1-beta1**step))*m/denominator
    return result


def paired_bc_surrogate(results):
    """Unbiased pathwise + score-function gradient for the finite BC window.

Each replica is independent conditional on the same human window. Its peer's
suffix NLL is an action-independent baseline. Burn-in choices receive credit
for later scored actions; no human action or scheduler dummy is scored.
"""
    if len(results) != 2:
        raise ValueError("paired BC estimator requires two independent replicas")
    costs = [r.meta_nll * r.score_mask.to(r.meta_nll.dtype) for r in results]
    count = results[0].score_mask.sum().clamp_min(1)
    suffix = [x.flip(1).cumsum(1).flip(1) for x in costs]
    pathwise = sum(x.sum()/count for x in costs)/2
    score = pathwise.new_zeros(())
    for index, result in enumerate(results):
        for cursor, log_prob in result.meta_search_log_probs:
            advantage = (suffix[index][:, cursor] - suffix[1-index][:, cursor]).detach()
            score = score + (advantage*log_prob).sum()/(2*count)
    # Numerically equal to NLL: the extra term carries only the likelihood
    # gradient, not an additional cost or a computation-count regularizer.
    return pathwise + score - score.detach(), pathwise, score


class FunctionalActor:
    def __init__(self, actor, parameters, buffers, mode):
        self.actor = actor
        self.parameter_map = parameters
        self.buffer_map = buffers
        self.mode = mode

    def __getattr__(self, name):
        return getattr(self.actor, name)

    def __call__(self, *args, **kwargs):
        with meta_scope(self.mode):
            return functional_call(self.actor, (self.parameter_map, self.buffer_map), args, kwargs, strict=True)


@contextmanager
def functional_voc_features(actor):
    """Capture selective GRU VJP inputs without the live learner's hooks."""
    records, live = [], []
    def capture(module, args):
        if torch.is_grad_enabled():
            live[:] = [args[0]]
    def leaf(module, args):
        if not torch.is_grad_enabled() or not live:
            return
        value = args[0].detach().requires_grad_(True)
        records.append((live[0], value))
        return (value,)+args[1:]
    handles = [actor.policy.register_forward_pre_hook(capture), actor.voc_head.register_forward_pre_hook(leaf)]
    try:
        yield records
    finally:
        for h in handles:
            h.remove()


def selective_meta_gradients(loss, records, parameters):
    leaves = [leaf for _, leaf in records]
    upstream = torch.autograd.grad(loss, leaves, create_graph=True, retain_graph=True, allow_unused=True)
    selected = [(live, g) for (live, _), g in zip(records, upstream) if g is not None]
    if not selected:
        raise RuntimeError("meta VOC -> GRU path is disconnected")
    return torch.autograd.grad([x for x, _ in selected], parameters,
                               grad_outputs=[g for _, g in selected], create_graph=True,
                               retain_graph=True, allow_unused=True)


def sync_actor_planning(actor, flags):
    if not hasattr(actor, "planning_gamma"):
        return None
    gamma, cost = float(actor.planning_gamma), float(actor.planning_cost)
    if not 0 < gamma < 1 or not math.isfinite(cost) or cost <= 0:
        raise ValueError("invalid published planning parameters")
    flags.discounting, flags.think_cost = gamma, cost
    if getattr(flags, "reward_clip", -1) > 0:
        actor.baseline_clamp = flags.reward_clip/(1-gamma)
    return dict(gamma=gamma, think_cost=cost, version=int(actor.planning_version))


def restore_checkpoint_planning(flags, checkpoint):
    state = checkpoint.get("planning_meta_state")
    if state is None:
        if getattr(flags, "planning_meta_enabled", False):
            raise ValueError("meta run cannot resume/evaluate a legacy checkpoint")
        return
    if state.get("schema") != META_SCHEMA or not getattr(flags, "planning_meta_enabled", False):
        raise ValueError("planning meta protocol mismatch")
    parameters = PlanningMetaParameters(state["gamma"], state["think_cost"])
    parameters.load_state_dict(state["parameters"])
    gamma, cost = parameters.values()
    published = checkpoint.get("actor_net_state_dict", {})
    for key, expected in (("planning_gamma", gamma), ("planning_cost", cost),
                          ("planning_version", torch.tensor(state["updates"]))):
        actual = published.get(key)
        if actual is None or not torch.allclose(actual.cpu(), expected.detach().cpu(), rtol=1e-6, atol=0):
            raise ValueError("checkpoint planning parameters disagree with actor buffers: "+key)
    if not math.isclose(float(gamma.detach()), state["gamma"], rel_tol=1e-6) or not math.isclose(float(cost.detach()), state["think_cost"], rel_tol=1e-6):
        raise ValueError("checkpoint planning parameter metadata is inconsistent")
    flags.planning_backend = "torch"
    flags.discounting = float(state["gamma"])
    flags.think_cost = float(state["think_cost"])


class PlanningMetaLearner:
    def __init__(self, learner, checkpoint=None):
        configure_meta(learner.flags)
        self.learner = learner
        self.parameters = PlanningMetaParameters(learner.flags.discounting, learner.flags.think_cost,
                                                  device=learner.device)
        self.optimizer = torch.optim.Adam(self.parameters.parameters(), lr=learner.flags.planning_meta_lr)
        self.actor_updates = self.updates = 0
        self.rng = np.random.default_rng(int(learner.flags.base_seed)+720001)
        self.metrics = {}
        self.pending = None
        if checkpoint is not None:
            if checkpoint.get("schema") != META_SCHEMA:
                raise ValueError("invalid meta checkpoint schema")
            self.parameters.load_state_dict(checkpoint["parameters"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.actor_updates, self.updates = checkpoint["actor_updates"], checkpoint["updates"]
            self.rng.bit_generator.state = copy.deepcopy(checkpoint["rng"])
        self.publish()

    def publish(self):
        actor = self.learner.actor_net
        gamma, cost = self.parameters.values()
        with torch.no_grad():
            actor.planning_gamma.copy_(gamma)
            actor.planning_cost.copy_(cost)
            actor.planning_version.fill_(self.updates)
        sync_actor_planning(actor, self.learner.flags)
        self.learner.im_discounting = float(gamma.detach()) ** (1/self.learner.flags.rec_t)
        if hasattr(self.learner, "actor_param_buffer"):
            self.learner.actor_param_buffer.set_data.remote(
                "planning_parameters", sync_actor_planning(actor, self.learner.flags))

    def state_dict(self):
        gamma, cost = self.parameters.values()
        return dict(schema=META_SCHEMA, gamma=float(gamma.detach()), think_cost=float(cost.detach()),
                    parameters=self.parameters.state_dict(), optimizer=self.optimizer.state_dict(),
                    actor_updates=self.actor_updates, updates=self.updates,
                    rng=copy.deepcopy(self.rng.bit_generator.state))

    def due(self):
        flags = self.learner.flags
        return (self.learner.real_step >= flags.planning_meta_warmup_steps
                and (self.actor_updates+1) % flags.planning_meta_every == 0)

    def prepare(self, data, inner_batch, inner_rng):
        self.pending = None
        if not self.due():
            return
        if inner_batch is None:
            raise RuntimeError("meta step requires a scheduled inner BC batch")
        started = time.monotonic()
        learner = self.learner
        flags = learner.flags
        if learner.bc_runner is None or learner.bc_model_net is None:
            raise RuntimeError("meta reconstruction requires the real BC model snapshot")
        from thinker.dynamic_imitation import DynamicImitationRunner
        device = learner.device
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            # Model refresh has already completed for the real BC forward.
            # The same frozen weights serve inner reconstruction and both
            # outer paths. No model optimizer operation occurs here.
            gamma, cost = self.parameters.values()
            names_by_id = {id(p): n for n, p in learner.actor_net.named_parameters()}
            parameters = {n: p.detach().clone().requires_grad_(p.requires_grad)
                          for n, p in learner.actor_net.named_parameters()}
            buffers = {n: b.detach().clone() for n, b in learner.actor_net.named_buffers()}
            buffers["planning_gamma"] = gamma
            buffers["planning_cost"] = cost
            shadow = copy.copy(learner)
            shadow.flags = copy.copy(flags)
            shadow._meta_gamma, shadow._meta_lambda = gamma, cost
            shadow.norm_stats = copy.deepcopy(learner.norm_stats)
            shadow._voc_pending_update = False
            shadow._assert_voc_gate_exact_projection_invariant = lambda **kw: None
            shadow._stage_schema13_replay_batch = lambda *a, **kw: None
            shadow.actor_net = FunctionalActor(learner.actor_net, parameters, buffers, "inner")
            inner_runner = DynamicImitationRunner(shadow.actor_net, learner.bc_model_net, shadow.flags,
                                                 device=device, noop_action_index=learner.imitation_noop_action_index)
            inner_runner.planning_gamma = gamma
            def inner_imitation():
                torch.set_rng_state(inner_rng[0])
                if device.type == "cuda":
                    torch.cuda.set_rng_state(inner_rng[1], device)
                return inner_runner.rollout(inner_batch, tree_carry=bool(flags.tree_carry), training=True)
            shadow._maybe_compute_imitation = inner_imitation
            try:
                with functional_voc_features(learner.actor_net) as records, meta_scope("inner"):
                    # Bypass instance wrappers and their pending live gradients.
                    losses, _ = type(learner).compute_losses(shadow, *data)
                    actor_names = [names_by_id[id(p)] for group in learner.optimizer.param_groups for p in group["params"]]
                    grads = torch.autograd.grad(losses["total_loss"], [parameters[n] for n in actor_names],
                                                create_graph=True, retain_graph=True, allow_unused=True)
                    actor_gradients = dict(zip(actor_names, grads))
                    q_loss = losses.get("_voc_total_loss")
                    if q_loss is None:
                        self.metrics = dict(skipped_no_voc_support=1.)
                        return
                    q_names = [names_by_id[id(p)] for p in learner.voc_parameters]
                    q_gradients = dict(zip(q_names, torch.autograd.grad(q_loss, [parameters[n] for n in q_names],
                                                     create_graph=True, retain_graph=True, allow_unused=True)))
                    gru_names = [n for n in actor_names if n.startswith("tree_rep_encoder.")]
                    extra = selective_meta_gradients(losses["_voc_gru_loss"], records, [parameters[n] for n in gru_names])
                    for name, gradient in zip(gru_names, extra):
                        if gradient is not None:
                            addition = float(flags.voc_gru_gradient_scale)*gradient
                            actor_gradients[name] = addition if actor_gradients[name] is None else actor_gradients[name]+addition
            finally:
                inner_runner.close()
            t, b = data[0].done.shape
            clip = flags.actor_grad_norm_clipping*t*b
            fast = functional_adam(parameters, actor_gradients, learner.optimizer, names_by_id, clip_limit=clip)
            fast = functional_adam(fast, q_gradients, learner.voc_optimizer, names_by_id, clip_limit=clip)
            tau = learner.voc_gate_target_tau
            ema_w = (1-tau)*learner.voc_ema_gate_weight.detach() + tau*fast["voc_head.weight"]
            ema_b = (1-tau)*learner.voc_ema_gate_bias.detach() + tau*fast["voc_head.bias"]
            scale = flags.voc_gate_temperature/flags.voc_gate_q_temperature
            fast["voc_gate_head.weight"] = scale*(ema_w[0:1]-ema_w[1:2])
            fast["voc_gate_head.bias"] = scale*(ema_b[0:1]-ema_b[1:2])
            buffers["planning_gamma"] = gamma
            buffers["planning_cost"] = cost
            outer_actor = FunctionalActor(learner.actor_net, fast, buffers, "outer")
            outer_runner = DynamicImitationRunner(outer_actor, learner.bc_model_net, flags,
                                                 device=device, noop_action_index=learner.imitation_noop_action_index)
            outer_runner.planning_gamma = gamma
            outer_runner.collect_meta = True
            old_rng = learner.bc_loader.rng
            try:
                learner.bc_loader.rng = self.rng
                batch = learner.bc_loader.get_sequence_batch(batch_size=flags.planning_meta_batch_size,
                                                             sequence_length=flags.batch_length)
            finally:
                learner.bc_loader.rng = old_rng
            torch.manual_seed(int(self.rng.integers(0, 2**31-1)))
            try:
                results = [outer_runner.rollout(batch, tree_carry=bool(flags.tree_carry), training=True) for _ in range(2)]
                objective, pathwise, score = paired_bc_surrogate(results)
                gradient = torch.autograd.grad(objective, self.parameters.raw, retain_graph=True)[0]
                path_gradient = torch.autograd.grad(pathwise, self.parameters.raw, allow_unused=True)[0]
            finally:
                outer_runner.close()
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("nonfinite BC meta-gradient; no parameter update applied")
            self.pending = gradient.detach()
            self.metrics = dict(meta_nll=float(pathwise.detach()),
                                gamma_gradient=float(gradient[0]), lambda_gradient=float(gradient[1]),
                                pathwise_gradient_norm=float(path_gradient.norm()),
                                score_gradient_norm=float((gradient-path_gradient).norm()),
                                seconds=time.monotonic()-started)

    def commit(self, actor_stepped, voc_stepped):
        if actor_stepped:
            self.actor_updates += 1
        if self.pending is not None and actor_stepped and voc_stepped:
            previous_raw = self.parameters.raw.detach().clone()
            previous_optimizer = copy.deepcopy(self.optimizer.state_dict())
            self.optimizer.zero_grad(set_to_none=True)
            self.parameters.raw.grad = clipped_gradients(
                (self.pending,), self.learner.flags.planning_meta_grad_clip)[0]
            self.optimizer.step()
            gamma, cost = self.parameters.values()
            if (not torch.isfinite(self.parameters.raw).all() or not torch.isfinite(cost)
                    or cost <= 0 or not GAMMA_EPSILON < gamma < 1-GAMMA_EPSILON):
                with torch.no_grad():
                    self.parameters.raw.copy_(previous_raw)
                self.optimizer.load_state_dict(previous_optimizer)
                self.pending = None
                raise FloatingPointError("invalid meta parameter update")
            self.updates += 1
            self.publish()
        self.pending = None
