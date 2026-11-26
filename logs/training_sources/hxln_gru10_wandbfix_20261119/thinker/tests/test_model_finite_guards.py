import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from thinker import util
from thinker.buffer import SModelBuffer
from thinker.learn_model import (
    GradientStepResult,
    SModelLearner,
    _empty_model_observability,
    _record_tensor_scale,
)
from thinker.model_net import ModelNet


def _buffer(batch_size=2, alpha=1.0):
    return SModelBuffer(
        buffer_n=16,
        max_rank=1,
        batch_size=batch_size,
        alpha=alpha,
        warm_up_n=0,
    )


def _buffer_data(batch_size=2):
    return {"value": np.arange(batch_size, dtype=np.float32)[:, None]}


@pytest.mark.parametrize("bad_priority", [np.nan, np.inf, -1.0])
def test_model_buffer_write_rejects_invalid_priority_atomically(bad_priority):
    buffer = _buffer()

    with pytest.raises(ValueError, match="ModelBuffer.write priority"):
        buffer.write(
            _buffer_data(),
            rank=0,
            priority=np.array([1.0, bad_priority], dtype=np.float64),
        )

    assert buffer.processed_n == 0
    assert not buffer.initialized


def test_model_buffer_default_write_rejects_poisoned_stored_priority():
    buffer = _buffer()
    buffer.write(_buffer_data(), rank=0)
    processed_n = buffer.processed_n
    buffer.priority[0, 0] = np.nan

    with pytest.raises(ValueError, match="stored priority before write"):
        buffer.write(_buffer_data(), rank=0)

    assert buffer.processed_n == processed_n


@pytest.mark.parametrize("bad_priority", [np.nan, -0.1])
def test_model_buffer_update_rejects_invalid_priority_without_mutation(bad_priority):
    buffer = _buffer()
    buffer.write(_buffer_data(), rank=0)
    before = buffer.priority.copy()
    idx = (
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([0.0, 0.0]),
    )

    with pytest.raises(ValueError, match="ModelBuffer.update_priority priority"):
        buffer.update_priority(
            idx, np.array([1.0, bad_priority], dtype=np.float64)
        )

    np.testing.assert_array_equal(buffer.priority, before)


@pytest.mark.parametrize("bad_priority", [np.nan, -0.1])
def test_model_buffer_read_rejects_poisoned_priority(bad_priority):
    buffer = _buffer(batch_size=1)
    buffer.write(_buffer_data(batch_size=1), rank=0)
    buffer.priority[0, 0] = bad_priority

    with pytest.raises(ValueError, match="ModelBuffer.read stored priority"):
        buffer.read(t=1, b=1)


def test_model_buffer_alpha_zero_samples_only_available_entries():
    buffer = _buffer(batch_size=1, alpha=0.0)
    buffer.write(_buffer_data(batch_size=1), rank=0)
    buffer.priority[:] = 0.0
    buffer.priority[0, 0] = 2.0

    sample = buffer.read(t=1, b=1, beta=0.4)

    assert sample["idx"][0].item() == 0
    np.testing.assert_allclose(sample["weights"], np.ones(1))


def test_model_buffer_rejects_overflowed_probability():
    buffer = _buffer(batch_size=1, alpha=2.0)
    buffer.write(_buffer_data(batch_size=1), rank=0)
    buffer.priority[:] = 0.0
    buffer.priority[0, 0] = 1e308

    with pytest.raises(ValueError, match="invalid unnormalized sampling"):
        buffer.read(t=1, b=1)


class _SkipScaler:
    def __init__(self, scale=8.0):
        self._scale = scale

    def get_scale(self):
        return self._scale

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        # Deliberately skip optimizer.step(), like GradScaler on found_inf.
        return None

    def update(self):
        self._scale /= 2.0


def _gradient_learner():
    learner = object.__new__(SModelLearner)
    learner.flags = SimpleNamespace(
        model_optimizer="adam",
        model_grad_norm_clipping=0.0,
    )
    learner.numel_per_step = 1
    learner.real_step = 7
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return learner, parameter, optimizer, scheduler


def test_gradient_step_reports_amp_skip_and_does_not_mutate_parameter():
    learner, parameter, optimizer, scheduler = _gradient_learner()
    before = parameter.detach().clone()

    result = learner.gradient_step(
        parameter.square(), optimizer, scheduler, scaler=_SkipScaler()
    )

    assert isinstance(result, GradientStepResult)
    assert not result.optimizer_stepped
    assert result.amp_scale_before == 8.0
    assert result.amp_scale_after == 4.0
    torch.testing.assert_close(parameter, before)


def test_gradient_step_rejects_nonfinite_gradient_before_optimizer_step():
    learner, parameter, optimizer, scheduler = _gradient_learner()
    parameter.data.zero_()
    before = parameter.detach().clone()

    with pytest.raises(FloatingPointError, match="gradient norm"):
        learner.gradient_step(torch.sqrt(parameter), optimizer, scheduler)

    torch.testing.assert_close(parameter, before)


def test_model_scale_observability_is_detached_and_materialized_for_logs():
    observability = _empty_model_observability()
    value = torch.tensor([-3.0, 4.0], dtype=torch.float16, requires_grad=True)

    _record_tensor_scale(observability, "pred_sr_hs", value)

    assert observability["pred_sr_hs_abs_max"].item() == pytest.approx(4.0)
    assert observability["pred_sr_hs_rms"].item() == pytest.approx(
        np.sqrt(12.5)
    )
    assert not observability["pred_sr_hs_abs_max"].requires_grad
    learner = object.__new__(SModelLearner)
    learner._pending_model_observability = observability
    stats = learner._model_observability_log_stats()
    assert stats["model/pred_sr_hs_abs_max"] == pytest.approx(4.0)
    assert stats["model/pred_policy_logits_abs_max"] is None


def test_model_gradient_clip_counters_use_preclip_norm_and_resume():
    learner = object.__new__(SModelLearner)
    learner.flags = SimpleNamespace(model_grad_norm_clipping=10.0)
    learner.device = torch.device("cpu")
    learner._initialize_gradient_clip_counters()

    first_m = learner._record_gradient_clipping("m", torch.tensor(12.0))
    last_m = learner._record_gradient_clipping("m", torch.tensor(3.0))
    last_p = learner._record_gradient_clipping("p", torch.tensor(10.01))
    stats = learner._gradient_clip_log_stats(last_m, last_p)

    assert first_m.item() is True
    assert last_m.item() is False
    assert stats["model/m_grad_clipped"] == 0
    assert stats["model/m_grad_clip_count"] == 1
    assert stats["model/m_grad_step_count"] == 2
    assert stats["model/m_grad_clip_rate"] == pytest.approx(0.5)
    assert stats["model/p_grad_clipped"] == 1
    assert stats["model/p_grad_clip_count"] == 1
    assert stats["model/p_grad_step_count"] == 1
    assert stats["model/p_grad_clip_rate"] == pytest.approx(1.0)

    checkpoint_state = learner._gradient_clip_checkpoint_state()
    resumed = object.__new__(SModelLearner)
    resumed.device = torch.device("cpu")
    resumed._initialize_gradient_clip_counters(checkpoint_state)
    assert resumed._gradient_clip_checkpoint_state() == checkpoint_state

    legacy = object.__new__(SModelLearner)
    legacy.device = torch.device("cpu")
    legacy._initialize_gradient_clip_counters({})
    assert legacy._gradient_clip_checkpoint_state() == {
        "model_grad_clip_count_m": 0,
        "model_grad_step_count_m": 0,
        "model_grad_clip_count_p": 0,
        "model_grad_step_count_p": 0,
    }


class _PriorityRecorder:
    def __init__(self):
        self.calls = []

    def update_priority(self, idx, priorities):
        self.calls.append((idx, np.asarray(priorities).copy()))


def _consume_learner(*, target=1.0, loss_kind="finite", priorities=None):
    learner = object.__new__(SModelLearner)
    learner.flags = SimpleNamespace(
        dual_net=False,
        float16=False,
        priority_alpha=1.0,
        model_optimizer="adam",
        model_grad_norm_clipping=0.0,
        xpid="finite-guard-test",
    )
    learner.device = torch.device("cpu")
    learner.model_float16 = False
    learner.model_net = SimpleNamespace(state_dict=lambda: {})
    learner.n = 0
    learner.real_step = 10
    learner.update_real_step = lambda data: None
    learner.prepare_data = lambda train_model_out: {
        "target": torch.tensor([target], dtype=torch.float32)
    }
    learner.parameter = torch.nn.Parameter(torch.tensor(2.0))
    learner.optimizer_p = torch.optim.SGD([learner.parameter], lr=0.1)
    learner.scheduler_p = torch.optim.lr_scheduler.LambdaLR(
        learner.optimizer_p, lambda _: 1.0
    )
    learner.scaler_p = None
    if priorities is None:
        priorities = np.array([1.0], dtype=np.float32)

    def compute_losses_p(train_model_out, prepared_target, is_weights, pred_xs):
        if loss_kind == "nonfinite":
            loss = learner.parameter * torch.tensor(float("nan"))
        else:
            loss = learner.parameter.square()
        return {"total_loss_p": loss}, priorities

    learner.compute_losses_p = compute_losses_p
    learner.timing = None
    learner.step = 0
    learner.numel_per_step = 1
    learner.timer = time.monotonic
    learner.start_time = learner.timer()
    learner.sps_buffer = [(0, learner.start_time)] * 36
    learner.sps_buffer_n = 0
    learner.sps_start_time = learner.start_time
    learner.sps_start_step = 0
    learner.ckp_start_time = int(time.strftime("%M")) // 10
    return learner


def _consume_data(value=1.0):
    return {
        "processed_n": 1,
        "data": {"input": np.array([value], dtype=np.float32)},
        "weights": np.array([1.0], dtype=np.float32),
        "idx": (np.array([0]), np.array([0]), np.array([0.0])),
    }


def test_consume_data_rejects_nonfinite_input_before_optimizer_or_per():
    learner = _consume_learner()
    recorder = _PriorityRecorder()
    before = learner.parameter.detach().clone()

    with pytest.raises(FloatingPointError, match="model input.*input"):
        learner.consume_data(_consume_data(np.nan), model_buffer=recorder)

    torch.testing.assert_close(learner.parameter, before)
    assert recorder.calls == []


def test_consume_data_rejects_nonfinite_target_before_optimizer_or_per():
    learner = _consume_learner(target=np.nan)
    recorder = _PriorityRecorder()
    before = learner.parameter.detach().clone()

    with pytest.raises(FloatingPointError, match="model target.*target"):
        learner.consume_data(_consume_data(), model_buffer=recorder)

    torch.testing.assert_close(learner.parameter, before)
    assert recorder.calls == []


def test_consume_data_rejects_nonfinite_loss_before_optimizer_or_per():
    learner = _consume_learner(loss_kind="nonfinite")
    recorder = _PriorityRecorder()
    before = learner.parameter.detach().clone()

    with pytest.raises(FloatingPointError, match="model VP losses.*total_loss_p"):
        learner.consume_data(_consume_data(), model_buffer=recorder)

    torch.testing.assert_close(learner.parameter, before)
    assert recorder.calls == []


def test_consume_data_rejects_nonfinite_priority_before_optimizer_or_per():
    learner = _consume_learner(priorities=np.array([np.nan]))
    recorder = _PriorityRecorder()
    before = learner.parameter.detach().clone()

    with pytest.raises(ValueError, match="model computed priority"):
        learner.consume_data(_consume_data(), model_buffer=recorder)

    torch.testing.assert_close(learner.parameter, before)
    assert recorder.calls == []


def test_consume_data_does_not_update_per_after_amp_skip():
    learner = _consume_learner()
    learner.gradient_step = lambda *args, **kwargs: GradientStepResult(
        torch.tensor(1.0), False, 8.0, 4.0
    )
    recorder = _PriorityRecorder()

    with pytest.raises(FloatingPointError, match="skipped the model VP"):
        learner.consume_data(_consume_data(), model_buffer=recorder)

    assert recorder.calls == []
    assert learner.step == 0


def test_consume_data_finite_path_steps_and_updates_per():
    learner = _consume_learner()
    recorder = _PriorityRecorder()
    before = learner.parameter.detach().clone()

    assert learner.consume_data(_consume_data(), model_buffer=recorder) is True

    assert not torch.equal(learner.parameter.detach(), before)
    assert learner.step == 1
    assert len(recorder.calls) == 1
    np.testing.assert_array_equal(recorder.calls[0][1], np.array([1.0]))


@pytest.mark.parametrize("disable_bn", [True, False])
def test_model_disable_bn_flag_reaches_both_frame_encoders(disable_bn):
    flags = util.create_setting(
        args=[
            "--parallel", "false",
            "--float16", "false",
            "--model_disable_bn", str(disable_bn).lower(),
            "--model_size_nn", "1",
        ],
        save_flags=False,
    )
    model = ModelNet(
        obs_space=spaces.Box(0, 255, shape=(12, 84, 84), dtype=np.uint8),
        action_space=spaces.Discrete(9),
        flags=flags,
        frame_stack_n=4,
    )
    batch_norms = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    ]

    assert bool(batch_norms) is (not disable_bn)


def _projection_model(mode="clamp", range_cost=1.0):
    flags = util.create_setting(
        args=[
            "--parallel", "false",
            "--float16", "false",
            "--model_float16", "false",
            "--model_disable_bn", "true",
            "--model_size_nn", "1",
            "--model_state_projection", mode,
            "--model_state_range_loss_cost", str(range_cost),
        ],
        save_flags=False,
    )
    return ModelNet(
        obs_space=spaces.Box(0, 255, shape=(12, 16, 16), dtype=np.uint8),
        action_space=spaces.Discrete(3),
        flags=flags,
        frame_stack_n=4,
    )


def test_depth_zero_projection_clamps_and_checks_raw_before_projection():
    model = _projection_model()
    sr_net = model.sr_net
    raw = torch.tensor([-2.0, 0.25, 2.0])

    projected = sr_net._project_decoded_state(
        raw, check_raw_finite=True
    )

    torch.testing.assert_close(projected, torch.tensor([0.0, 0.25, 1.0]))
    with pytest.raises(FloatingPointError, match="before projection"):
        sr_net._project_decoded_state(
            torch.tensor([0.0, float("inf")])
        )

    legacy_sr_net = _projection_model(
        mode="none", range_cost=0.0
    ).sr_net
    legacy_raw = torch.tensor([float("-inf"), float("inf")])
    torch.testing.assert_close(
        legacy_sr_net._project_decoded_state(legacy_raw), legacy_raw
    )
    with pytest.raises(FloatingPointError, match="before projection"):
        legacy_sr_net._project_decoded_state(
            legacy_raw, check_raw_finite=True
        )


def test_clamp_inference_rejects_raw_nonfinite_before_it_reaches_vp(monkeypatch):
    model = _projection_model()
    model.eval()
    raw_frame = torch.full((1, 3, 16, 16), float("inf"))

    def fake_decode(hidden, flatten=False):
        if flatten:
            return raw_frame.expand(
                hidden.shape[0], hidden.shape[1], -1, -1, -1
            )
        return raw_frame.expand(hidden.shape[0], -1, -1, -1)

    monkeypatch.setattr(model.sr_net.encoder, "decode", fake_decode)
    root = torch.zeros((1, 12, 16, 16), dtype=torch.uint8)
    actions = torch.zeros((2, 1, 1), dtype=torch.long)

    with torch.no_grad(), pytest.raises(
        FloatingPointError, match="before projection"
    ):
        model(
            env_state=root,
            done=torch.zeros(1, dtype=torch.bool),
            actions=actions,
            state=model.initial_state(batch_size=1),
            training=False,
        )


def test_state_range_loss_uses_edge_mask_and_importance_weights():
    learner = object.__new__(SModelLearner)
    raw = torch.tensor(
        [
            [[[[ -1.0, 0.5]]], [[[ -2.0, 2.0]]]],
            [[[[ 2.0, 0.5]]], [[[ -0.5, 1.5]]]],
        ],
        requires_grad=True,
    )
    edge_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    is_weights = torch.tensor([0.5, 2.0])

    loss = learner.compute_state_range_loss(raw, edge_mask, is_weights)
    loss.backward()

    assert loss.item() == pytest.approx(0.5)
    assert raw.grad[0, 0, 0, 0, 0] < 0
    assert raw.grad[1, 0, 0, 0, 0] > 0
    assert raw.grad[1, 1, 0, 0, 0] < 0
    assert raw.grad[1, 1, 0, 0, 1] > 0
    assert torch.count_nonzero(raw.grad[0, 1]) == 0


def test_projected_state_reaches_frame_carry_and_vp_consistently(monkeypatch):
    model = _projection_model()
    model.eval()
    raw_frame = torch.empty(1, 3, 16, 16)
    raw_frame[:, 0].fill_(-2.0)
    raw_frame[:, 1].fill_(0.5)
    raw_frame[:, 2].fill_(2.0)

    def fake_decode(hidden, flatten=False):
        if flatten:
            return raw_frame.expand(hidden.shape[0], hidden.shape[1], -1, -1, -1)
        return raw_frame.expand(hidden.shape[0], -1, -1, -1)

    monkeypatch.setattr(model.sr_net.encoder, "decode", fake_decode)
    vp_inputs = []
    original_vp_forward = model.vp_net.encoder.forward

    def capture_vp_input(x, *args, **kwargs):
        vp_inputs.append(x.detach().clone())
        return original_vp_forward(x, *args, **kwargs)

    monkeypatch.setattr(model.vp_net.encoder, "forward", capture_vp_input)
    root = torch.full((1, 12, 16, 16), 64, dtype=torch.uint8)
    actions = torch.zeros(2, 1, 1, dtype=torch.long)

    with torch.no_grad():
        root_out = model(
            env_state=root,
            done=torch.zeros(1, dtype=torch.bool),
            actions=actions,
            state=model.initial_state(batch_size=1),
            training=True,
        )
        next_out = model.forward_single(
            state=root_out.state,
            action=torch.zeros(1, 1, dtype=torch.long),
            training=True,
        )

    for value in (root_out.xs, next_out.xs, *vp_inputs):
        assert torch.isfinite(value).all()
        assert value.min().item() >= 0.0
        assert value.max().item() <= 1.0
    torch.testing.assert_close(
        root_out.xs[-1, :, -3:],
        torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1, 1).expand(1, 3, 16, 16),
    )
    torch.testing.assert_close(
        next_out.state["last_x"][:, -3:], root_out.xs[-1, :, -3:]
    )
    assert torch.isfinite(next_out.policy).all()


def test_projection_mode_does_not_change_model_state_dict_schema():
    legacy = _projection_model(mode="none", range_cost=0.0)
    projected = _projection_model(mode="clamp", range_cost=1.0)

    legacy_state = legacy.state_dict()
    projected_state = projected.state_dict()

    assert list(legacy_state) == list(projected_state)
    assert {
        key: tuple(value.shape) for key, value in legacy_state.items()
    } == {
        key: tuple(value.shape) for key, value in projected_state.items()
    }
    projected.load_state_dict(legacy_state, strict=True)


def test_projection_rejects_unbounded_observation_contract():
    flags = util.create_setting(
        args=[
            "--parallel", "false",
            "--float16", "false",
            "--model_state_projection", "clamp",
            "--model_state_range_loss_cost", "1.0",
        ],
        save_flags=False,
    )

    with pytest.raises(ValueError, match="finite, non-degenerate"):
        ModelNet(
            obs_space=spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32
            ),
            action_space=spaces.Discrete(3),
            flags=flags,
            frame_stack_n=1,
        )
