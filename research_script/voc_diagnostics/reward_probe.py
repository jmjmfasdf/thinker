"""Paired online STOP/CONTINUE evaluation of reward-based VOC.

The environment and planner are rebuilt from the same seeded policy prefix.
Human observations/actions are never used to advance either branch. Current
stage costs and future task rewards have distinct termination/discount rules.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import yaml

from forced_budget_probe import ROOT, RUNTIME, digest, util
from thinker.actor_net import ActorNet
from thinker.model_net import ModelNet
from thinker.cenv import cModelWrapper
from thinker.gym_add import wrapper
from thinker.dynamic_imitation import _voc_trace_diagnostics, _VoCFeatureObserver


@dataclass
class ReturnAccumulator:
    gamma: float
    cost: float
    task_return: float = 0.0
    discount: float = 1.0
    stage_cost: float = 0.0
    all_discounted_cost: float = 0.0
    stage_open: bool = True
    terminal: bool = False
    real_steps: int = 0
    stage_computations: int = 0
    all_computations: int = 0

    def observe(self, *, reward, think_reward, real_transition, stage_end, done):
        if self.terminal:
            raise ValueError("Cannot accumulate after termination")
        if not np.isfinite(reward) or not np.isfinite(think_reward):
            raise ValueError("Nonfinite return input")
        self.task_return += self.discount * reward
        compute_cost = -self.cost * think_reward
        self.all_discounted_cost += self.discount * compute_cost
        self.all_computations += int(think_reward == -1)
        if self.stage_open:
            self.stage_cost += compute_cost
            self.stage_computations += int(think_reward == -1)
        if stage_end or done:
            self.stage_open = False
        if real_transition:
            self.real_steps += 1
            self.discount *= self.gamma
        if done:
            self.terminal = True
            self.discount = 0.0

    def scores(self, task_bootstrap=0.0):
        return dict(task_truncated=self.task_return,
                    current_stage_cost=self.stage_cost,
                    net_truncated=self.task_return-self.stage_cost,
                    task_bootstrap_contribution=self.discount*task_bootstrap,
                    net_bootstrapped=self.task_return-self.stage_cost+self.discount*task_bootstrap,
                    all_future_cost_net_truncated=self.task_return-self.all_discounted_cost,
                    **asdict(self))


def keyed_seed(seed, decision, depth, phase, continuation_seed=0):
    payload = f"{seed}:{decision}:{depth}:{phase}:{continuation_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def seed_policy(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preserve_tokens(previous, current, info):
    primary = current.last_pri.clone()
    control = current.last_search_control.clone()
    accepted = info["accepted_primary_action"]
    real = info["real_transition"].bool()
    invalid = (accepted < 0) & ~real
    primary[0] = torch.where(invalid, previous.last_pri[0], primary[0])
    invalid_control = info["accepted_control"] < 0
    control[0] = torch.where(invalid_control, previous.last_search_control[0], control[0])
    return current._replace(last_pri=primary, last_search_control=control)


class OnlineRuntime:
    def __init__(self, flags, actor_checkpoint, model_checkpoint, seed, device):
        import envpool
        self.flags = copy.deepcopy(flags)
        self.flags.train_actor = False
        self.flags.train_model = False
        self.flags.parallel = False
        self.flags.parallel_actor = False
        self.flags.env_n = 1
        self.flags.ckp = False
        self.flags.use_wandb = False
        self.device = torch.device(device)
        seed_policy(seed)
        raw = envpool.make_gymnasium(self.flags.name, num_envs=1, batch_size=1, num_threads=1,
                    seed=seed, gray_scale=self.flags.grayscale,
                    stack_num=self.flags.frame_stack_n, episodic_life=True)
        self.raw = raw
        base = wrapper.VectorWrap(wrapper.EnvPoolWrap(raw, 1, frame_stack_n=self.flags.frame_stack_n), self.flags)
        self.model = ModelNet(obs_space=base.get_wrapper_attr("single_observation_space"),
                             action_space=base.get_wrapper_attr("single_action_space"),
                             flags=self.flags, frame_stack_n=self.flags.frame_stack_n).to(self.device)
        self.model.set_weights(model_checkpoint["model_net_state_dict"])
        self.model.eval(); self.model.requires_grad_(False)
        self.planner = cModelWrapper(env=base, env_n=1, flags=self.flags,
                                    model_net=self.model, device=self.device, timing=False)
        self.env = wrapper.PostWrapper(self.planner, self.flags)
        self.actor = ActorNet(obs_space=self.env.observation_space, action_space=self.env.action_space,
                             flags=self.flags, tree_rep_meaning=util.get_tree_rep_meaning(
                                 int(raw.action_space.n), 1, self.flags)).to(self.device)
        self.actor.set_weights(actor_checkpoint["actor_net_state_dict"], strict=True)
        self.actor.eval(); self.actor.requires_grad_(False)
        self.ema = actor_checkpoint["voc_ema_gate_head_state_dict"]
        if bool(self.flags.think_cost_anneal):
            raise ValueError("An exact annealed cost must be supplied, not inferred")
        if not self.actor.voc_eval_stochastic or self.actor.voc_gate_epsilon_greedy_execution:
            raise ValueError("This protocol requires stochastic soft gate execution")

    def close(self):
        self.env.close()


def run_arm(flags, actor_checkpoint, model_checkpoint, *, seed, prefix_steps, horizon,
            arm, continuation_seed=0, device="cpu", policy_epsilon=False):
    if arm not in ("stop", "continue") or horizon < 1 or prefix_steps < 0:
        raise ValueError("Invalid arm, horizon or prefix")
    rt = OnlineRuntime(flags, actor_checkpoint, model_checkpoint, seed, device)
    rt.actor.train_actor_enabled = bool(policy_epsilon)
    parameter_before = digest(dict(actor=rt.actor.state_dict(), model=rt.model.state_dict()))
    try:
        state, info = rt.env.reset(rt.model, seed=[seed])
        env_out = util.init_env_out(state, info, rt.flags, rt.actor.dim_actions, rt.actor.tuple_action)
        core = rt.actor.initial_state(batch_size=1, device=rt.device)
        real_count = 0
        prefix = hashlib.sha256()
        entry = None
        accumulator = ReturnAccumulator(float(flags.discounting), float(flags.think_cost))
        trajectory = []
        bootstrap = 0.0
        task_index = util.get_reward_names(rt.flags).index("re")
        max_calls = (prefix_steps+horizon+1) * (int(flags.max_search_steps)+4)
        with torch.no_grad(), _VoCFeatureObserver(rt.actor) as observer:
            for call in range(max_calls):
                phase = int(env_out.phase.reshape(-1)[0])
                depth = int(env_out.search_steps.reshape(-1)[0])
                current_seed = keyed_seed(seed, real_count, depth, phase,
                    continuation_seed if real_count >= prefix_steps else 0)
                seed_policy(current_seed)
                pre = dict(env=env_out, core=core, torch_rng=torch.get_rng_state(),
                           policy_seed=current_seed)
                if rt.device.type == "cuda":
                    pre["cuda_rng"] = torch.cuda.get_rng_state(rt.device)
                if entry is None and real_count < prefix_steps:
                    prefix.update(digest(pre).encode())
                is_entry = entry is None and real_count == prefix_steps and phase == 0 and depth == 0
                if is_entry:
                    entry = dict(prefix_sha256=prefix.hexdigest(), entry_sha256=digest(pre),
                                 tree_sha256=digest(rt.planner.dynamic_trace_snapshot(include_xs=True, include_hs=True)))
                observer.begin_row()
                out, core = rt.actor(env_out, core, compute_loss=False, greedy=False)
                features = observer.take(out.voc_q)
                if is_entry:
                    diag = _voc_trace_diagnostics(out, features, flags=rt.flags,
                            ema_head_state=rt.ema, effective_think_cost=float(flags.think_cost))
                    ema_q = diag["ema_q"].reshape(-1, 2)[0]
                    online_q = diag["online_q"].reshape(-1, 2)[0]
                    entry.update(ema_q=ema_q.cpu().tolist(), online_q=online_q.cpu().tolist(),
                                 ema_delta_q=float(ema_q[0]-ema_q[1]), online_delta_q=float(online_q[0]-online_q[1]),
                                 learned_continue_probability=float(diag["soft_continue_probability"].reshape(-1)[0]))
                if entry is not None and accumulator.real_steps >= horizon:
                    if phase != 0 or depth != 0:
                        raise AssertionError("Horizon must end at a real-decision boundary")
                    bootstrap = float(out.baseline[-1, 0, task_index])
                    break
                primary, control = out.action
                if is_entry:
                    if not bool(out.control_valid[-1, 0]):
                        raise AssertionError("Intervention is not a valid control choice")
                    if arm == "stop":
                        chosen = util.STOP
                    else:
                        logits = out.search_control_logits[-1, 0, :2].double()
                        # Sample conditional PROCEED/RESET, not its argmax.
                        # The independent draw does not shift later keyed streams.
                        probability = torch.softmax(logits, -1)
                        gen = torch.Generator(device=rt.device).manual_seed(current_seed ^ 0x32A12)
                        chosen = int(torch.multinomial(probability, 1, generator=gen))
                    control = torch.full_like(control, chosen)
                action = (primary, control)
                state, reward, done, truncated, info = rt.env.step(action, rt.model)
                next_env = util.create_env_out(action, state, reward, done, truncated, info, rt.flags)
                next_env = preserve_tokens(env_out, next_env, info)
                real = bool(info["real_transition"].reshape(-1)[0])
                terminal = bool((done | truncated.bool()).reshape(-1)[0])
                if entry is not None:
                    task_reward = float(reward.reshape(-1)[0])
                    if not real and np.isnan(task_reward):
                        task_reward = 0.0
                    think_reward = float(info["think_reward"].reshape(-1)[0])
                    event = dict(call=call, real_index=real_count-prefix_steps,
                        real_transition=real, reward=task_reward, think_reward=think_reward,
                        stage_end=bool(info["stage_end"].reshape(-1)[0]), done=terminal,
                        accepted_control=int(info["accepted_control"].reshape(-1)[0]),
                        executed_action=int(info["executed_primary_action"].reshape(-1)[0]),
                        observation_sha256=digest(state["real_states"]) if real else None)
                    trajectory.append(event)
                    accumulator.observe(**{k:event[k] for k in ("reward", "think_reward", "real_transition", "stage_end", "done")})
                real_count += int(real)
                env_out = next_env
                if entry is not None and terminal:
                    break
            else:
                raise RuntimeError("Online probe exceeded bounded augmented calls")
        if entry is None or accumulator.stage_open:
            raise AssertionError("Intervention never completed its initial search stage")
        if arm == "stop" and accumulator.stage_computations != 0:
            raise AssertionError("STOP incurred computation cost")
        if arm == "continue" and accumulator.stage_computations < 1:
            raise AssertionError("CONTINUE did not execute a computation")
        parameter_after = digest(dict(actor=rt.actor.state_dict(), model=rt.model.state_dict()))
        if parameter_before != parameter_after:
            raise AssertionError("Evaluation changed parameters or buffers")
        return dict(arm=arm, seed=seed, continuation_seed=continuation_seed, prefix_steps=prefix_steps,
                    horizon=horizon, entry=entry, returns=accumulator.scores(bootstrap), trajectory=trajectory,
                    parameter_sha256=parameter_before, primary_and_bout_policy="sampled",
                    policy_epsilon=float(flags.voc_train_epsilon) if policy_epsilon else 0.0,
                    full_return_observed=accumulator.terminal)
    finally:
        rt.close()


def verify_arms(stop, proceed, repeat):
    for result in (proceed, repeat):
        for key in ("prefix_sha256", "entry_sha256", "tree_sha256"):
            if result["entry"][key] != stop["entry"][key]:
                raise AssertionError(f"Unpaired online intervention: {key}")
    if stop["returns"] != repeat["returns"] or stop["trajectory"] != repeat["trajectory"]:
        raise AssertionError("Repeated STOP trajectory did not reproduce exactly")


def load_checkpoints(args):
    import visual_behav as vb
    if args.bundle:
        binding = json.loads(args.bundle.read_text())
        files = binding["files"]
        for name in ("actor", "model", "config"):
            record = files[name]
            if hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Bundle {name} hash mismatch")
        actor = torch.load(files["actor"]["path"], map_location="cpu", weights_only=False)
        model = torch.load(files["model"]["path"], map_location="cpu", weights_only=False)
        flags = argparse.Namespace(**yaml.safe_load(Path(files["config"]["path"]).read_text()))
        schema = actor.get("voc_gate_policy_schema_version")
        util.validate_voc_gate_policy_checkpoint(actor, flags=flags, label="reward-evaluation actor")
        # Model checkpoints have no top-level gate schema in schemas 1/4.
        # Bind their embedded protocol rather than inventing that metadata.
        if model.get("voc_gate_policy_schema_version", schema) != schema:
            raise ValueError("Actor/model schema mismatch")
        for name, value in vars(flags).items():
            if name in {"cmd", "git_revision", "ckp", "ckpdir", "savedir"}:
                continue
            if actor["flags"].get(name) != value or model["flags"].get(name) != value:
                raise ValueError(f"Actor/model/config mismatch: {name}")
        flags.voc_gate_policy_schema_version = schema
        return flags, actor, model, binding
    bundle = vb.resolve_checkpoint_bundle(args.checkpoint, requested_step=5_000_000)
    config = SimpleNamespace(checkpoint=bundle, torch_threads_per_worker=args.threads,
                             data_root=str(ROOT/"behavioral_data_block"), base_seed=0, episode_indices=(0,))
    rt = vb._CaptureRuntime(config, vb.WorkerSpec(rank=0, device="cpu"))
    return rt.flags, rt.actor_checkpoint, rt.model_checkpoint, asdict(bundle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--bundle", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260925, 20260926])
    p.add_argument("--prefix-steps", type=int, nargs="+", default=[32, 128])
    p.add_argument("--repetitions", type=int, default=2)
    p.add_argument("--horizon", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--policy-epsilon", action="store_true")
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    flags, actor, model, binding = load_checkpoints(args)
    records = []
    for seed in args.seeds:
        for prefix in args.prefix_steps:
            for repetition in range(args.repetitions):
                arms = []
                for arm in ("stop", "continue", "stop"):
                    result = run_arm(flags, actor, model, seed=seed, prefix_steps=prefix,
                        horizon=args.horizon, arm=arm, continuation_seed=repetition,
                        device=args.device, policy_epsilon=args.policy_epsilon)
                    arms.append(result)
                    print(f"seed={seed} prefix={prefix} rep={repetition} {arm}: "
                          f"net={result['returns']['net_truncated']:.5f} steps={result['returns']['real_steps']}", flush=True)
                verify_arms(*arms)
                row = dict(seed=seed, prefix_steps=prefix, repetition=repetition,
                           ema_delta_q=arms[0]["entry"]["ema_delta_q"],
                           online_delta_q=arms[0]["entry"]["online_delta_q"],
                           delta_net_truncated=arms[1]["returns"]["net_truncated"]-arms[0]["returns"]["net_truncated"],
                           delta_net_bootstrapped=arms[1]["returns"]["net_bootstrapped"]-arms[0]["returns"]["net_bootstrapped"],
                           full_return_observed=all(a["full_return_observed"] for a in arms[:2]), arms=arms)
                records.append(row)
                (args.output/"pairs.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    manifest = dict(status="PAIRED_ONLINE_PASS", game=flags.name, n_pairs=len(records), checkpoint=binding,
                    arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    runtime=str(RUNTIME), gamma=float(flags.discounting), think_cost=float(flags.think_cost),
                    reward_clip=float(flags.reward_clip), cost_horizon="remaining current search stage only",
                    task_horizon="terminal/life-loss or requested finite real-step horizon",
                    policy="sampled primary, sampled conditional PROCEED/RESET, stochastic learned gate",
                    bootstrap_warning="Same critic tail estimate is not independent validation; truncated and bootstrapped targets are separate",
                    scope="paired evaluation plumbing; small samples do not establish critic utility",
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (args.output/"reward_probe.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
