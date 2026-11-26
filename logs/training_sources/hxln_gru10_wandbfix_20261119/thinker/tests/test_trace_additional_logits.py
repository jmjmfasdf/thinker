"""Additive logit recording must not change Actor policy or replay execution."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thinker import util
from thinker.dynamic_imitation import _ModelDoneLogitObserver
from test_dynamic_imitation_trace import (
    _ListSink, _ProceedResetStopActor, _TraceModel, _make_trace_runner, _trace_batch,
)


class _DoneHead(torch.nn.Module):
    def forward(self, h):
        logits = h[:, 0, 0, 0] - 3.5
        return SimpleNamespace(done_logits=logits, dones=logits.sigmoid() > 0.5)


class _DoneTraceModel(_TraceModel):
    def __init__(self, num_actions):
        super().__init__(num_actions)
        self.dual_net = True
        self.sr_net.out = _DoneHead()
        self.expansion_calls = 0
        self.root_calls = 0

    def forward(self, *args, **kwargs):
        self.root_calls += 1
        return super().forward(*args, **kwargs)

    def forward_single(self, *args, **kwargs):
        self.expansion_calls += 1
        result = super().forward_single(*args, **kwargs)
        # Like production ModelNet, discard the raw logit at the wrapper, but
        # keep the boolean prediction.  The observer must capture it upstream.
        output = self.sr_net.out(result.hs[-1])
        result.dones = output.dones.unsqueeze(0)
        return result


def _runner(**kwargs):
    runner = _make_trace_runner(**kwargs)
    runner.model_net = _DoneTraceModel(runner.actor_net.num_actions)
    return runner


@pytest.mark.parametrize("budget", [0, 1, 2, 29, 30])
def test_logits_at_every_budget_and_forced_cap(budget):
    runner = _runner(
        control_mode="stop" if budget == 0 else "proceed",
        max_search_steps=max(1, budget), max_depth=64,
    )
    batch, _ = _trace_batch(edge_count=1)
    sink = _ListSink()
    summary = runner.stream_rollout_episode(batch, event_sink=sink)
    assert summary["search_computation_count"] == budget
    assert runner.model_net.expansion_calls == budget
    assert runner.model_net.root_calls == 2
    assert sum(e["model_done_logit_valid"] for e in sink.events) == budget
    for event in sink.events:
        if event["model_done_logit_valid"]:
            assert event["model_done_logit"] == np.float32(-1.5)
            assert event["model_done_target_depth"] == event["current_node_depth_pre"] + 1
            assert event["model_done_target_action"] == event["accepted_primary_action"]
        else:
            assert event["model_done_logit"] == 0
            assert event["model_done_target_depth"] == -1
            assert event["model_done_target_action"] == -1
        assert bool(event["actor_real_logits_pre_override_valid"]) == bool(event["real_transition"])
        if event["real_transition"]:
            np.testing.assert_array_equal(event["actor_real_logits_pre_override"], event["actor_policy"])
            assert event["execution_primary"] == batch["actions_seq"][0]
            assert event["thinker_action"] != event["human_action"]
        else:
            np.testing.assert_array_equal(event["actor_real_logits_pre_override"], np.zeros(3))
    assert not runner.model_net.sr_net.out._forward_hooks
    runner.close()


def test_reset_logit_belongs_to_evaluated_child_not_post_reset_root():
    runner = _runner()
    batch, _ = _trace_batch(edge_count=2)
    sink = _ListSink()
    runner.stream_rollout_episode(batch, event_sink=sink)
    resets = [e for e in sink.events if e["actor_reset"]]
    assert len(resets) == 2
    for event in resets:
        assert event["model_done_logit_valid"]
        assert event["current_node_depth_post"] == 0
        assert event["model_done_target_depth"] == 2
        assert event["model_done_logit"] == -1.5
        # The old latent snapshot is the reset root: it must NOT be used.
        assert event["im_vectors"].reshape(-1)[0] - 3.5 != event["model_done_logit"]
    runner.close()


def test_revisit_does_not_reuse_previous_model_logit():
    class ResetActor(_ProceedResetStopActor):
        def forward(self, *args, **kwargs):
            output, state = super().forward(*args, **kwargs)
            control = torch.full_like(output.action[1], util.RESET)
            return output._replace(
                action=(output.action[0], control),
                reset=control.unsqueeze(0), search_control=control.unsqueeze(0),
            ), state
    runner = _runner(max_search_steps=4, max_depth=2)
    runner.actor_net = ResetActor(3, len(util.get_reward_names(runner.flags)))
    batch, _ = _trace_batch(edge_count=1)
    sink = _ListSink()
    runner.stream_rollout_episode(batch, event_sink=sink)
    revisits = [e for e in sink.events if e["search_revisit"]]
    assert revisits
    assert all(not e["model_done_logit_valid"] for e in revisits)
    assert runner.model_net.expansion_calls == sum(e["model_done_logit_valid"] for e in sink.events)
    runner.close()


@pytest.mark.parametrize("dual", [False, True])
def test_observer_uses_correct_head_and_discards_root_evaluations(dual):
    sr, vp = _DoneHead(), _DoneHead()
    model = SimpleNamespace(dual_net=dual, sr_net=SimpleNamespace(out=sr), vp_net=SimpleNamespace(out=vp))
    selected, other = (sr, vp) if dual else (vp, sr)
    with _ModelDoneLogitObserver(model) as observer:
        for value in (1., 5.):
            observer.begin_step()
            expected = selected(torch.full((1, 1, 1, 1), value))
            other(torch.full((1, 1, 1, 1), 19.))
            actual = observer.take(model_expansion=True)
            assert torch.equal(actual, expected.done_logits)
            assert torch.equal(actual.sigmoid() > 0.5, expected.dones)
        observer.begin_step()
        selected(torch.ones(1, 1, 1, 1))
        assert observer.take(model_expansion=False) is None
        observer.begin_step()
        assert observer.take(model_expansion=True) is None
    assert not sr._forward_hooks and not vp._forward_hooks


@pytest.mark.parametrize("enabled", [False, True])
def test_production_done_head_is_captured_without_another_forward(enabled):
    from gymnasium import spaces
    from thinker.model_net import OutputNet
    head = OutputNet(
        action_space=spaces.Discrete(3), input_shape=(8,), reward_n=1,
        enc_type=0, enc_f_type=0, zero_init=False, size_nn=1,
        predict_v_pi=False, predict_done=enabled,
    ).eval()
    model = SimpleNamespace(dual_net=True, sr_net=SimpleNamespace(out=head))
    h = torch.ones(1, 8)
    with torch.no_grad(), _ModelDoneLogitObserver(model) as observer:
        observer.begin_step()
        rng_before = torch.get_rng_state().clone()
        output = head(h)
        logit = observer.take(model_expansion=True)
        assert torch.equal(torch.get_rng_state(), rng_before)
        if enabled:
            assert torch.equal(logit, output.done_logits)
            assert torch.equal(logit.sigmoid() > 0.5, output.dones)
        else:
            assert logit is None


def test_model_hook_removed_after_sink_error():
    class BrokenSink:
        def write_event(self, _event):
            raise RuntimeError("test sink failure")
    runner = _runner()
    batch, _ = _trace_batch()
    with pytest.raises(RuntimeError, match="test sink failure"):
        runner.stream_rollout_episode(batch, event_sink=BrokenSink())
    assert not runner.model_net.sr_net.out._forward_hooks
    runner.close()


def test_recording_does_not_change_any_existing_field_or_model_calls(monkeypatch):
    import thinker.dynamic_imitation as dynamic
    outputs, counts = [], []
    for observed in (True, False):
        runner = _runner()
        if not observed:
            # Disable only the new hook, retaining the same streaming path.
            original = dynamic._ModelDoneLogitObserver
            def without_hook(model):
                observer = original(model)
                observer.module = None
                return observer
            monkeypatch.setattr(dynamic, "_ModelDoneLogitObserver", without_hook)
        batch, _ = _trace_batch()
        sink = _ListSink()
        runner.stream_rollout_episode(batch, event_sink=sink)
        outputs.append(sink.events)
        counts.append((runner.model_net.root_calls, runner.model_net.expansion_calls))
        runner.close()
    assert counts[0] == counts[1]
    assert len(outputs[0]) == len(outputs[1])
    for observed, plain in zip(*outputs):
        for name in observed:
            if not name.startswith("model_done_"):
                np.testing.assert_array_equal(observed[name], plain[name], err_msg=name)


def test_new_logits_survive_native_zarr_roundtrip(tmp_path):
    from thinker.behavior_trace import EpisodeZarrWriter, open_video_stat
    runner = _runner()
    batch, _ = _trace_batch()
    sink = _ListSink()
    runner.stream_rollout_episode(batch, event_sink=sink)
    writer = EpisodeZarrWriter(
        tmp_path / "partial", metadata={}, num_actions=3,
        tree_rep_schema={"all": [0, 44]}, chunk_events=2, images="none",
    )
    for event in sink.events:
        writer.write_event(event)
    writer.finalize(tmp_path / "episode.zarr")
    native = open_video_stat(tmp_path / "episode.zarr").native
    for name in ("actor_policy", "actor_real_logits_pre_override", "actor_real_logits_pre_override_valid",
                 "model_done_logit", "model_done_logit_valid", "model_done_target_action", "model_done_target_depth"):
        np.testing.assert_array_equal(native.field(name)[:], np.asarray([e[name] for e in sink.events]))
    runner.close()


def test_v2_read_compatibility_and_v3_rejects_missing_additions(tmp_path):
    import visual_behav as recorder
    from thinker import behavior_trace as trace
    runner = _runner()
    batch, _ = _trace_batch()
    sink = _ListSink()
    runner.stream_rollout_episode(batch, event_sink=sink)
    writer = trace.EpisodeZarrWriter(
        tmp_path / "partial", metadata={}, num_actions=3,
        tree_rep_schema={"all": [0, 44]}, chunk_events=2, images="none",
    )
    for event in sink.events:
        old = {k: v for k, v in event.items()
               if not k.startswith(("actor_real_logits_pre_override", "model_done_"))}
        old["trace_schema_version"] = np.int16(2)
        writer.write_event(old)
    path = tmp_path / "old.zarr"
    writer.finalize(path)
    manifest = trace.validate_episode_store(path)
    recorder._validate_native_trace_contract(trace, path, manifest, 2)
    np.testing.assert_array_equal(
        trace.open_video_stat(path).native.field("actor_policy")[:],
        np.asarray([e["actor_policy"] for e in sink.events]),
    )
    with pytest.raises(recorder.VisualBehaviorError, match="lacks trace-v3 fields"):
        recorder._validate_native_trace_contract(trace, path, manifest, 3)
    runner.close()
