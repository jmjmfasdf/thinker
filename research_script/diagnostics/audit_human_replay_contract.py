"""Exercise human replay with the real planner and a deterministic model.

This audit does not load training checkpoints or rewrite existing captures.
Test fixtures supply small observation spaces; the planner, replay runner,
canonical image sink, and asynchronous event freezer are production code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "thinker"))

import visual_behav as recorder
from thinker import util
from thinker.cenv import cModelWrapper
from thinker.dataset_env import LazyBehaviorEpisodeVectorEnv
from thinker.dynamic_imitation import DynamicImitationRunner
from tests.test_dynamic_imitation_trace import (
    _ListSink, _ProceedResetStopActor, _TraceModel, _trace_batch, _trace_flags,
    _make_trace_runner,
)


class ObservationModel(_TraceModel):
    """Distinct actual and imagined states expose unintended reuse/leakage."""

    def __init__(self, num_actions=3):
        super().__init__(num_actions)
        self.real_calls = []
        self.search_calls = []

    def output(self, core):
        n = len(core)
        offset = core.reshape(1, n, 1, 1, 1)
        return SimpleNamespace(
            vs=core.reshape(1, n, 1),
            policy=torch.zeros(1, n, 1, self.num_actions),
            xs=offset.expand(1, n, 1, 2, 2).clone(),
            hs=offset + torch.arange(5).reshape(1, 1, 5, 1, 1),
            state={"core": core.reshape(n, 1).clone()},
            rs=torch.ones(1, n, 1),
            dones=torch.zeros(1, n, 1, dtype=torch.bool),
        )

    def forward(self, env_state, done, actions, state, **kwargs):
        self.real_calls.append({
            "observation": env_state.detach().cpu().tolist(),
            "action": actions.detach().cpu().reshape(-1).tolist(),
        })
        return self.output(env_state[:, 0, 0, 0].float())

    def forward_single(self, state, action, **kwargs):
        self.search_calls.append({
            "parent_core": state["core"].detach().reshape(-1).tolist(),
            "action": action.detach().reshape(-1).tolist(),
        })
        return self.output(state["core"].reshape(-1) + action.reshape(-1) + 1)


def direct_carry(human_action=1, tree_carry=True, terminal=False):
    batch, provider = _trace_batch(edge_count=1 if terminal else 2)
    provider.values[0].fill(10)
    provider.values[1].fill(80)
    model = ObservationModel()
    flags = _trace_flags(max_search_steps=8, max_depth=8)
    flags.tree_carry = tree_carry
    actions = np.array([human_action, 0])
    if terminal:
        actions = actions[:1]
    env = LazyBehaviorEpisodeVectorEnv(
        provider, actions_seq=actions,
        done_seq=np.array([True]) if terminal else np.array([False, True]),
        initial_prev_action=np.array([0]), num_actions=3,
        observation_space=batch["observation_space"],
    )
    planner = cModelWrapper(env, 1, flags, model, device=torch.device("cpu"))
    planner.reset(model)

    def step(action, control):
        return planner.step((torch.tensor([action]), torch.tensor([control])), model)

    step(1, util.PROCEED)
    first = planner.dynamic_trace_snapshot()
    first_frozen = recorder._freeze_event(first)
    step(2, util.PROCEED)
    step(0, util.STOP)
    assert provider.calls == [0], "Imagination must not read a future human frame"
    _, _, _, _, info = step(human_action, util.PROCEED)
    carried = bool(info["root_carried"][0])
    useful = bool(info["useful_carry"][0])
    root = planner.dynamic_trace_snapshot()
    assert float(root["current_node_hs"][0][0, 0, 0]) == 80
    assert model.real_calls[-1]["action"] == ([0] if terminal else [human_action])
    refreshed = None
    if not terminal:
        before_calls = len(model.search_calls)
        step(2, util.PROCEED)
        refreshed = len(model.search_calls) == before_calls + 1
        assert refreshed, "Even a carried child needs refresh after a real observation"
        assert model.search_calls[-1]["parent_core"] == [80.0]
    after = recorder._freeze_event(first)
    for key in first_frozen:
        np.testing.assert_equal(after[key], first_frozen[key])
    result = {
        "human_action": human_action, "carry_enabled": tree_carry,
        "terminal": terminal, "root_carried": carried, "useful_carry": useful,
        "retained_expanded_children": int(info["carried_descendant_expanded_count"][0]),
        "root_actual_observation": 80, "carried_child_recomputed_on_visit": refreshed,
        "provider_calls": provider.calls, "search_calls": model.search_calls,
        "earlier_snapshot_unchanged": True,
    }
    planner.close()
    assert carried == (tree_carry and human_action == 1 and not terminal)
    assert useful == carried
    return result


class ScriptedActor(_ProceedResetStopActor):
    def __init__(self, *args):
        super().__init__(*args)
        self.input_records = []

    def forward(self, env_out, state, compute_loss=False, greedy=False):
        out, state = super().forward(env_out, state, compute_loss, greedy)
        k = env_out.search_steps[-1]
        search = env_out.phase[-1] == util.SEARCH_PHASE
        primary = torch.where(k == 0, torch.ones_like(k), torch.full_like(k, 2))
        control = torch.where(k < 2, torch.full_like(k, util.PROCEED), torch.full_like(k, util.STOP))
        self.input_records.append(recorder._freeze_event({
            "real": env_out.real_states, "tree": env_out.tree_reps,
            "xs": env_out.xs, "hs": env_out.hs, "last_pri": env_out.last_pri,
            "greedy": np.bool_(greedy),
        }))
        return out._replace(
            pri=primary.unsqueeze(0), reset=control.unsqueeze(0),
            action=(primary, control), search_control=control.unsqueeze(0),
        ), state


def runner_case(human_action=1, future_pixel=80, policy_mode="checkpoint"):
    batch, provider = _trace_batch(edge_count=2)
    batch["actions_seq"] = np.array([human_action, 0])
    provider.values[0].fill(10)
    provider.values[1:].fill(future_pixel)
    flags = _trace_flags(max_search_steps=8, max_depth=8)
    actor = ScriptedActor(3, len(util.get_reward_names(flags)))
    model = ObservationModel()
    runner = DynamicImitationRunner(actor, model, flags, device="cpu")
    sink = _ListSink()
    runner.stream_rollout_episode(batch, event_sink=sink, tree_carry=True, policy_mode=policy_mode)
    runner.close()
    return sink.events, actor.input_records, provider.calls


def audit_no_leakage():
    base, inputs, calls = runner_case()
    changed_action, action_inputs, _ = runner_case(human_action=0)
    changed_future, future_inputs, _ = runner_case(future_pixel=160)
    # Two computations, STOP, then REAL proposal: all precede execution.
    for other in (action_inputs, future_inputs):
        for row in range(4):
            for key in inputs[row]:
                np.testing.assert_equal(inputs[row][key], other[row][key])
    real = [event for event in base if event["real_transition"]]
    assert [int(e["executed_primary_action"]) for e in real] == [1, 0]
    assert [int(e["thinker_action"]) for e in real] == [2, 2]
    assert calls == [0, 1, 2]
    assert all(bool(row["greedy"]) for row in inputs)
    _, sampled_inputs, _ = runner_case(policy_mode="sampled")
    assert not any(bool(row["greedy"]) for row in sampled_inputs)
    return {
        "changed_current_target_does_not_change_pre_execution_inputs": True,
        "changed_future_observation_does_not_change_pre_execution_inputs": True,
        "real_executions": [1, 0], "model_real_proposals": [2, 2],
        "first_search_actions": [int(e["accepted_primary_action"]) for e in base if e["search_computation"]][:2],
        "provider_calls": calls, "checkpoint_mode_primary_greedy": True,
        "sampled_mode_primary_greedy": False,
    }


def audit_reset_fields():
    runner = _make_trace_runner()
    batch, _ = _trace_batch()
    sink = _ListSink()
    runner.stream_rollout_episode(batch, event_sink=sink)
    runner.close()
    rows = []
    for i, event in enumerate(sink.events):
        if event["search_computation"]:
            rows.append({
                "event": i, "reset": bool(event["actor_reset"]),
                "predicted_image_offset": float(event["predicted_xs"].reshape(-1)[0]),
                "stored_sr_hidden_offset": float(event["im_vectors"].reshape(-1)[0]),
                "post_internal_node_depth": int(event["current_node_depth_post"]),
                "evaluated_child_depth": int(event["current_node_depth_pre"]) + 1,
            })
    return {
        "rows": rows,
        "reset_image_and_hidden_describe_same_node": all(
            r["predicted_image_offset"] == r["stored_sr_hidden_offset"]
            for r in rows if r["reset"]
        ),
        "meaning": "reset_mode=0 returns the evaluated child token before moving the internal pointer to root; recorder hidden vectors currently use the latter",
    }


def audit_display_timing():
    runner = _make_trace_runner()
    batch, provider = _trace_batch()
    output = _ListSink()
    sink = recorder._CanonicalEventSink(
        output, images="all", source_frame_indices=batch["observation_source_index"],
        decision_times=np.array([0., 1., 2.]), grayscale=True,
        observation_provider=provider.values,
    )
    runner.stream_rollout_episode(batch, event_sink=sink)
    runner.close()
    real_rows = [e for e in output.events if e["real_transition"]]
    for cursor, event in enumerate(real_rows):
        assert int(event["source_frame_index"]) == int(batch["observation_source_index"][cursor + 1])
        assert float(event["decision_time"]) == cursor
        np.testing.assert_equal(event["real_image"], provider.values[cursor + 1])
    return {
        "native_rows": len(output.events),
        "actual_search_computations": sum(bool(e["search_computation"]) for e in output.events),
        "valid_imagined_display_rows": sum(bool(e["imagined_image_valid"]) for e in output.events),
        "valid_real_display_rows": sum(bool(e["real_image_valid"]) for e in output.events),
        "real_rows": [{
            "action_time": float(e["decision_time"]),
            "pre_observation_source_index": int(e["pre_observation_source_index"]),
            "post_observation_source_index": int(e["post_observation_source_index"]),
            "display_source_frame_index": int(e["source_frame_index"]),
        } for e in real_rows],
        "meaning": "Real-edge image is post-action, policy is pre-action. Imagined display also includes root reconstructions and STOP duplicates; count computations with search_computation, not displayed frames.",
    }


def audit_async_freezing(output):
    entered, release = threading.Event(), threading.Event()
    stored = []

    class DelayedWriter:
        def __init__(self, *args, **kwargs):
            pass

        def write_event(self, event):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("audit writer timed out")
            stored.append(event)

        def finalize(self, *args):
            return {"count": len(stored)}

        def abort(self):
            release.set()

    writer = recorder._BoundedWriter(
        SimpleNamespace(EpisodeZarrWriter=DelayedWriter),
        partial_path=output / ".unused.partial", final_path=output / "unused.zarr",
        metadata={}, num_actions=3, tree_rep_schema={}, chunk_events=2,
        images="none", queue_size=1,
    )
    payload = {"tensor": torch.tensor([1., 2.]), "nested": {"array": np.array([3., 4.])}}
    writer.write_event(payload)
    try:
        assert entered.wait(10)
        payload["tensor"].fill_(90)
        payload["nested"]["array"][:] = 80
    finally:
        release.set()
    writer.finalize()
    np.testing.assert_equal(stored[0]["tensor"], [1., 2.])
    np.testing.assert_equal(stored[0]["nested"]["array"], [3., 4.])
    return {"queued_tensor_and_array_independent_of_later_mutations": True}


def audit_existing_cache_sample():
    """Inspect subject 001 / leave01, all stored runs, in each game."""
    import pandas as pd

    cache = ROOT / "research_script/outputs/00_dynamic_cache_10m_20261110/checkpoint_leave01"
    columns = [
        "real_transition", "search_computation", "actor_reset", "human_action",
        "accepted_primary_action", "root_carried", "useful_carry", "done",
        "truncated", "pre_observation_source_index", "post_observation_source_index",
        "current_node_depth_pre", "current_node_depth_post", "rollout_depth_post",
    ]
    result = []
    for game, name in [(0, "Enduro"), (1, "Pong"), (2, "Space Invaders")]:
        path = cache / f"sub001_game{game}/native_events.csv.gz"
        if not path.exists():
            result.append({"game": name, "missing": str(path)})
            continue
        manifest = json.loads(path.with_name("manifest.json").read_text())
        n_actions = int(manifest["num_actions"])
        visit_columns = [f"root_ns_pre_{a}" for a in range(n_actions)]
        probability_columns = [f"actor_action_probability_{a}" for a in range(n_actions)]
        frame = pd.read_csv(path, usecols=columns + visit_columns + probability_columns)
        real = frame.loc[frame.real_transition.eq(1)]
        search = frame.loc[frame.search_computation.eq(1)]
        reset = search.loc[search.actor_reset.eq(1)]
        root_search = search.loc[search.current_node_depth_pre.eq(0)]
        human = real.human_action.to_numpy(dtype=int)
        visits = real[visit_columns].to_numpy()
        expected = (visits[np.arange(len(real)), human] > 0) & ~real.done.eq(1).to_numpy() & ~real.truncated.eq(1).to_numpy()
        probs = search[probability_columns].to_numpy()
        result.append({
            "game": name, "scope": "sub001 leave01 checkpoint, all captured runs",
            "real_decisions": len(real), "search_computations": len(search),
            "real_execution_human_mismatches": int(real.accepted_primary_action.ne(real.human_action).sum()),
            "search_advances_human_observation": int(search.pre_observation_source_index.ne(search.post_observation_source_index).sum()),
            "carry_differs_from_visited_human_child_and_nonterminal": int(np.count_nonzero(expected != real.root_carried.eq(1).to_numpy())),
            "root_carried": int(real.root_carried.eq(1).sum()),
            "useful_carry": int(real.useful_carry.eq(1).sum()),
            "search_primary_equals_policy_argmax": int(np.count_nonzero(search.accepted_primary_action.to_numpy() == np.argmax(probs, axis=1))),
            "root_search_action_counts": {str(int(k)): int(v) for k, v in root_search.accepted_primary_action.value_counts().items()},
            "human_action_counts": {str(int(k)): int(v) for k, v in real.human_action.value_counts().items()},
            "reset_rows": len(reset),
            "reset_rows_internal_depth_differs_from_returned_token": int(reset.current_node_depth_post.ne(reset.rollout_depth_post).sum()),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "research_script/outputs/00/replay_contract_audit_20261113")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "current source, deterministic instrumented models, actual Cython planner and production streaming recorder; not a historical capture rerun",
        "carry_cases": [direct_carry(), direct_carry(human_action=0), direct_carry(tree_carry=False), direct_carry(terminal=True)],
        "human_trajectory_contract": audit_no_leakage(),
        "recording_independence": audit_async_freezing(args.output),
        "reset_record_alignment": audit_reset_fields(),
        "display_timing": audit_display_timing(),
        "existing_cache_sample": audit_existing_cache_sample(),
    }
    import thinker.cenv as cenv
    sources = [Path(__file__), ROOT / "thinker/visual_behav.py", ROOT / "thinker/thinker/dynamic_imitation.py", ROOT / "thinker/thinker/dataset_env.py", ROOT / "thinker/thinker/cenv.pyx", Path(cenv.__file__)]
    result["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    target = args.output / "contract_audit.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
