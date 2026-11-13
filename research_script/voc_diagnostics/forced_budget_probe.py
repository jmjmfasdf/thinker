"""Bounded T1 probe using deterministic prefix replay, without modifying cenv.

A fresh planner replays the same episode prefix and RNG seed for each arm.
Only the target decision's gate token is overridden. Prefix actor inputs,
recurrent state, current tree diagnostics, and RNG must hash identically.
The real-head probabilities are intercepted BEFORE the target human action
is passed to the execution adapter. This is a one-decision probe, not a
whole-sequence intervention or an estimate of task reward improvement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path(os.environ.get("VOC_DIAGNOSTIC_RUNTIME", str(ROOT / "thinker"))).resolve()
sys.dont_write_bytecode = True  # Frozen capture runtimes are read-only inputs.
sys.path.insert(0, str(RUNTIME))
from thinker import util
from thinker.dynamic_imitation import DynamicImitationRunner


def digest(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            visit(x.detach().cpu().contiguous().numpy())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype.str, x.shape)).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x):
                h.update(str(key).encode()); visit(x[key])
        elif isinstance(x, (tuple, list)):
            h.update(str(type(x)).encode())
            for item in x:
                visit(item)
        elif x is None or isinstance(x, (str, int, float, bool, np.generic)):
            h.update(repr(x).encode())
        else:
            raise TypeError(f"Cannot fingerprint {type(x)}")
    visit(value)
    return h.hexdigest()


class ReadoutReady(Exception):
    pass


class BudgetProbe:
    image_mode = "none"

    def __init__(self, target, budget):
        self.target = target
        self.budget = budget
        self.cursor = 0
        self.computations = 0
        self.prefix = hashlib.sha256()
        self.prefix_calls = 0
        self.entry = None
        self.result = None
        self.runner = None

    def before(self, module, args, kwargs):
        env = kwargs.get("env_out", args[0] if args else None)
        core = kwargs.get("core_state", args[1] if len(args) > 1 else None)
        state = dict(env=env, core=core, torch_rng=torch.get_rng_state(),
                     numpy_rng=np.random.get_state(), python_rng=random.getstate())
        if self.cursor < self.target:
            self.prefix.update(digest(state).encode())
            self.prefix_calls += 1
        elif self.cursor == self.target and self.entry is None:
            tree = self.runner._planner.dynamic_trace_snapshot(include_xs=True, include_hs=True)
            self.entry = dict(prefix_sha256=self.prefix.hexdigest(), prefix_calls=self.prefix_calls,
                              entry_sha256=digest(state), tree_observation_sha256=digest(tree))

    def after(self, module, args, kwargs, output):
        actor, state = output
        if self.cursor != self.target:
            return None
        policy = int(actor.policy_type[-1].reshape(-1)[0])
        valid = bool(actor.policy_valid[-1].reshape(-1)[0])
        if valid and policy == util.POLICY_REAL:
            logits = actor.pri_param[-1].reshape(-1, module.num_actions)[0].detach().double()
            if self.computations != self.budget:
                raise AssertionError(f"Requested {self.budget}, executed {self.computations}")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite real-head logits")
            self.result = dict(**self.entry, target_edge=self.target, budget=self.budget,
                               actual_computations=self.computations,
                               probability=torch.softmax(logits, -1).cpu().tolist(),
                               argmax=int(logits.argmax()), target_human_action_not_executed=True)
            raise ReadoutReady()
        if valid and policy == util.POLICY_SEARCH:
            primary, control = actor.action
            if self.computations >= self.budget:
                chosen = util.STOP
            else:
                # Evaluation already uses greedy conditional PROCEED/RESET.
                # Preserve that conditional policy even when gate sampled STOP.
                logits = actor.search_control_logits[-1].reshape(-1, 3)[0]
                chosen = int(logits[:2].argmax())
            replacement = torch.full_like(control, chosen)
            return actor._replace(action=(primary, replacement)), state
        return None

    def write_event(self, event):
        if int(event["cursor_before"]) == self.target:
            self.computations += int(event["search_computation"])
        self.cursor = int(event["cursor_after"])


def run_probe(actor, model, flags, batch, *, target, budget, seed, ema):
    if not 0 <= budget < int(flags.max_search_steps):
        raise ValueError("Budget must be below the safety cap to avoid forced-stop semantics")
    if target < 1:
        raise ValueError("Target must follow the unscored burn-in edge")
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    runner = DynamicImitationRunner(actor, model, flags, device="cpu")
    probe = BudgetProbe(target, budget)
    probe.runner = runner
    pre = actor.register_forward_pre_hook(probe.before, with_kwargs=True)
    post = actor.register_forward_hook(probe.after, with_kwargs=True)
    try:
        with torch.no_grad():
            runner.stream_rollout_episode(batch, event_sink=probe, tree_carry=True,
                                          policy_mode="checkpoint", voc_ema_head_state_dict=ema)
    except ReadoutReady:
        pass
    finally:
        pre.remove(); post.remove(); runner.close()
    if probe.result is None:
        raise RuntimeError("Episode ended before target real-head readout")
    return probe.result


def verify_pairing(results):
    reference = results[0]
    for row in results:
        for key in ["prefix_sha256", "prefix_calls", "entry_sha256", "tree_observation_sha256"]:
            if row[key] != reference[key]:
                raise AssertionError(f"Unpaired counterfactual: {key}")
    zero = [row for row in results if row["budget"] == 0]
    if len(zero) < 2 or zero[0]["probability"] != zero[-1]["probability"]:
        raise AssertionError("Repeated zero-budget arm did not reproduce exactly")


def first_transition_targets(actions):
    """Return zero-based action edges, never cache real_step (one-based)."""
    if torch.is_tensor(actions):
        actions = actions.detach().cpu().numpy()
    actions = np.asarray(actions)
    if actions.ndim != 1 or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("Expected one integer action sequence")
    active = actions != 0
    onsets = np.flatnonzero(~active[:-1] & active[1:]) + 1
    offsets = np.flatnonzero(active[:-1] & ~active[1:]) + 1
    if not len(onsets) or not len(offsets):
        raise ValueError("Episode must contain both NOOP-to-active and active-to-NOOP")
    return sorted([int(onsets[0]), int(offsets[0])])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "behavioral_data_block")
    p.add_argument("--source", required=True, help="One NPZ path relative to data root")
    p.add_argument("--episode", type=int, default=0)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--targets", type=int, nargs="+",
                           help="Zero-based action edges (default: 4 16 32)")
    selection.add_argument("--first-transitions", action="store_true",
                           help="Select the first boundary in each direction from original action labels")
    p.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 3, 5])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.budgets[0] != 0:
        p.error("First budget must be zero")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    import visual_behav as vb
    bundle = vb.resolve_checkpoint_bundle(args.checkpoint, requested_step=5_000_000)
    config = SimpleNamespace(checkpoint=bundle, torch_threads_per_worker=args.threads,
                             data_root=str(args.data_root), base_seed=args.seed,
                             episode_indices=(args.episode,))
    runtime = vb._CaptureRuntime(config, vb.WorkerSpec(rank=0, device="cpu"))
    sources = vb.enumerate_source_bundles(args.data_root, selectors=[args.source], hash_workers=1)
    tasks = vb._coordinator_source_tasks(config, sources)
    if len(tasks) != 1:
        raise ValueError("Select exactly one source")
    task = tasks[0]
    expected = {task.source.path: {k: getattr(task.source, k) for k in ["size", "sha256", "device", "inode", "mtime_ns"]}}
    source = runtime.trace_api.BehaviorEpisodeSource(vb._SourceLoaderView(task), expected_sources=expected,
                allowed_source_paths=[task.source.path], scratch_dir=args.output / "scratch")
    all_results, batch = [], None
    try:
        descriptor = next(x for x in source.episodes if x.local_episode_index == args.episode)
        holdouts = {int(x) for x in str(runtime.flags.icopro_holdout_sessions).split(",")}
        if int(descriptor.session) not in holdouts:
            raise ValueError("This smoke test must use a held-out human session")
        batch = vb._episode_batch(source, descriptor)
        targets = (first_transition_targets(batch["actions_seq"][0]) if args.first_transitions
                   else (args.targets if args.targets is not None else [4, 16, 32]))
        if min(targets) < 1 or max(targets) >= batch["actions_seq"].shape[-1]:
            raise ValueError("Target exceeds episode length")
        print(f"Selected zero-based action edges: {targets}", flush=True)
        runtime._ensure_runner(batch)
        runtime.runner.close()
        parameter_hash_before = digest(dict(actor=runtime.actor_net.state_dict(), model=runtime.model_net.state_dict()))
        for target in targets:
            rows = []
            for budget in args.budgets + [0]:
                row = run_probe(runtime.actor_net, runtime.model_net, runtime.flags, batch,
                                target=target, budget=budget, seed=args.seed,
                                ema=runtime.voc_ema_head_state_dict)
                rows.append(row)
                print(f"Target {target}, budget {budget}, argmax {row['argmax']}", flush=True)
            verify_pairing(rows)
            human = int(batch["actions_seq"][0, target])
            previous = int(batch["actions_seq"][0, target-1])
            base = np.asarray(rows[0]["probability"])
            for i, row in enumerate(rows):
                q = np.asarray(row["probability"])
                row.update(human_action=human, previous_human_action=previous,
                           observation_source_index=int(batch["observation_source_index"][target]),
                           repeat_control=i == len(rows)-1,
                           total_variation=float(np.abs(q-base).sum()/2),
                           argmax_changed=row["argmax"] != rows[0]["argmax"],
                           delta_human_log_likelihood=float(np.log(max(q[human],1e-12))-np.log(max(base[human],1e-12))),
                           transition=("NOOP" if previous == 0 else "active") + "_to_" + ("NOOP" if human == 0 else "active"))
                all_results.append(row)
            (args.output / "decisions.jsonl").write_text("".join(json.dumps(row)+"\n" for row in all_results))
        parameter_hash_after = digest(dict(actor=runtime.actor_net.state_dict(), model=runtime.model_net.state_dict()))
        if parameter_hash_before != parameter_hash_after:
            raise AssertionError("Probe changed network weights or buffers")
        manifest = dict(status="SMOKE_PASS", checkpoint=asdict(bundle), source=asdict(task.source),
                        targets=targets, budgets=args.budgets, episode=args.episode,
                        target_selection="first_boundary_each_direction" if args.first_transitions else "explicit_edges",
                        target_index_convention="zero-based actions_seq edge; cache real_step is edge + 1",
                        subject=int(descriptor.subject), game=int(descriptor.game), session=int(descriptor.session),
                        seed=args.seed, device="cpu", deterministic_algorithms=True,
                        runtime=str(RUNTIME),
                        runtime_visual_sha256=hashlib.sha256((RUNTIME / "visual_behav.py").read_bytes()).hexdigest(),
                        state_equivalence="deterministic common-prefix replay; input/core/RNG/tree-observation hashes",
                        cenv_full_state_clone=False, parameter_sha256=parameter_hash_before,
                        limitations=["Selected early decisions from one held-out episode; no population inference",
                                     "One-decision effect with existing tree carry held common",
                                     "Includes extra actor recurrent updates; not isolated tree-information effect",
                                     "Human likelihood is not task-reward VOC validation"],
                        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (args.output / "probe_source.py").write_bytes(Path(__file__).read_bytes())
    finally:
        if batch is not None:
            batch["observation_provider"].close()
        if runtime.runner is not None:
            runtime.runner.close()
        source.close()


if __name__ == "__main__":
    main()
