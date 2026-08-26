from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from thinker import util
from thinker.learn_actor import (
    ActorGradientStepResult,
    SActorLearner,
    _validate_model_state_dict_compatibility,
)
from thinker.learn_model import SModelLearner
from thinker.self_play import SelfPlayWorker
from thinker.dynamic_imitation import (
    DynamicImitationRunner,
    HumanActionExecutionAdapter,
    compute_imitation_objective,
    compute_masked_imitation_objective,
    detached_imitation_logit_metrics,
    imitation_checkpoint_state,
    scale_imitation_for_online_rows,
    validate_behavior_batch,
)


class _FrozenFakeModel(torch.nn.Module):
    hidden_shape = (1,)

    def __init__(self, num_actions, observation_space=None):
        super().__init__()
        self.num_actions = num_actions
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.observation_space = observation_space or spaces.Box(
            0, 255, shape=(4, 8, 8), dtype=np.uint8
        )

    def initial_state(self, batch_size, device=None):
        return {"core": torch.zeros(batch_size, 1, device=device)}

    def forward(self, env_state, done, actions, state, **_kwargs):
        assert env_state.dtype == torch.from_numpy(
            np.empty((), dtype=self.observation_space.dtype)
        ).dtype
        batch_size = env_state.shape[0]
        device = env_state.device
        return SimpleNamespace(
            vs=torch.zeros(1, batch_size, 1, device=device),
            policy=torch.zeros(
                1, batch_size, 1, self.num_actions, device=device
            ),
            xs=None,
            hs=None,
            state={"core": torch.zeros(batch_size, 1, device=device)},
        )

    def forward_single(self, state, action, **_kwargs):
        batch_size = action.shape[0]
        device = action.device
        return SimpleNamespace(
            rs=torch.zeros(1, batch_size, 1, device=device),
            vs=torch.zeros(1, batch_size, 1, device=device),
            policy=torch.zeros(
                1, batch_size, 1, self.num_actions, device=device
            ),
            xs=None,
            hs=None,
            dones=torch.zeros(
                1, batch_size, 1, dtype=torch.bool, device=device
            ),
            state={"core": state["core"]},
        )


class _StopThenProposeActor(torch.nn.Module):
    """STOP at SEARCH and propose argmax at every NEED_REAL call."""

    def __init__(self, num_actions):
        super().__init__()
        self.real_logits = torch.nn.Parameter(torch.arange(float(num_actions)))
        self.num_actions = num_actions
        self.dim_actions = 1
        self.tuple_action = False
        self.discrete_action = True

    def initial_state(self, batch_size, device=None):
        del batch_size, device
        return ()

    def forward(self, env_out, state, compute_loss=False, greedy=False):
        del compute_loss, greedy
        phase = env_out.phase[-1]
        batch_size = phase.shape[0]
        policy_type = torch.where(
            phase == util.SEARCH_PHASE,
            torch.full_like(phase, util.POLICY_SEARCH),
            torch.where(
                phase == util.NEED_REAL_ACTION_PHASE,
                torch.full_like(phase, util.POLICY_REAL),
                torch.full_like(phase, util.POLICY_NONE),
            ),
        )
        logits = self.real_logits.view(1, 1, -1).expand(batch_size, 1, -1)
        primary = logits[:, 0].argmax(dim=-1)
        control = torch.where(
            phase == util.SEARCH_PHASE,
            torch.full_like(phase, util.STOP),
            torch.full_like(phase, util.PROCEED),
        )
        return SimpleNamespace(
            action=(primary, control),
            pri_param=logits.unsqueeze(0),
            policy_type=policy_type.unsqueeze(0),
            policy_valid=(policy_type != util.POLICY_NONE).unsqueeze(0),
        ), state


class _StaggeredCarryActor(torch.nn.Module):
    """Expand human branch 1, then stop at different times across rows."""

    def __init__(self, num_actions, edge_count):
        super().__init__()
        self.stage_logits = torch.nn.Parameter(
            torch.arange(float(num_actions)).repeat(edge_count, 1)
        )
        self.num_actions = num_actions
        self.dim_actions = 1
        self.tuple_action = False
        self.discrete_action = True
        self.phase_history = []
        self.last_pri_history = []

    def initial_state(self, batch_size, device=None):
        return (torch.zeros(batch_size, dtype=torch.long, device=device),)

    def forward(self, env_out, state, compute_loss=False, greedy=False):
        del compute_loss, greedy
        phase = env_out.phase[-1]
        self.phase_history.append(phase.detach().clone())
        self.last_pri_history.append(env_out.last_pri[-1].detach().clone())
        stage = state[0] + env_out.real_transition[-1].long()
        stage = stage.clamp_max(self.stage_logits.shape[0] - 1)
        batch_size = phase.shape[0]
        batch_index = torch.arange(batch_size, device=phase.device)
        search_steps = env_out.search_steps[-1]
        search = phase == util.SEARCH_PHASE
        need_real = phase == util.NEED_REAL_ACTION_PHASE
        policy_type = torch.where(
            search,
            torch.full_like(phase, util.POLICY_SEARCH),
            torch.where(
                need_real,
                torch.full_like(phase, util.POLICY_REAL),
                torch.full_like(phase, util.POLICY_NONE),
            ),
        )
        logits = self.stage_logits.index_select(0, stage)
        primary = logits.argmax(dim=-1)
        # The first SEARCH call expands the human-action child.  Environment
        # one deepens once more, making environment zero wait at the barrier.
        primary = torch.where(
            search & (search_steps == 0), torch.ones_like(primary), primary
        )
        primary = torch.where(
            search & (search_steps == 1) & (batch_index == 1),
            torch.full_like(primary, 2),
            primary,
        )
        control = torch.full_like(phase, util.PROCEED)
        control = torch.where(
            search & (search_steps == 1) & (batch_index == 0),
            torch.full_like(control, util.STOP),
            control,
        )
        control = torch.where(
            search & (search_steps >= 2),
            torch.full_like(control, util.STOP),
            control,
        )
        return SimpleNamespace(
            action=(primary, control),
            pri_param=logits[:, None, :].unsqueeze(0),
            policy_type=policy_type.unsqueeze(0),
            policy_valid=(policy_type != util.POLICY_NONE).unsqueeze(0),
        ), (stage,)


def _batch(batch_size=2, scored_length=4):
    edges = scored_length + 1
    return {
        "obs_seq": np.zeros(
            (batch_size, edges + 1, 4, 8, 8), dtype=np.uint8
        ),
        "actions_seq": np.zeros((batch_size, edges), dtype=np.int64),
        "initial_prev_action": np.ones(batch_size, dtype=np.int64),
        "rewards_seq": np.zeros((batch_size, edges), dtype=np.float32),
        "done_seq": np.zeros((batch_size, edges), dtype=np.bool_),
        "truncated_seq": np.zeros((batch_size, edges), dtype=np.bool_),
        "score_mask": np.array([False] + [True] * scored_length),
    }


def _actor_out(policy_type):
    batch_size = len(policy_type)
    primary = torch.tensor([2, 1, 0][:batch_size], dtype=torch.long)
    control = torch.tensor([util.STOP, util.PROCEED, util.RESET][:batch_size])
    logits = torch.tensor(
        [[[0.0, 1.0, 2.0]], [[3.0, 2.0, 1.0]], [[0.0, 4.0, 1.0]]]
    )[:batch_size]
    return SimpleNamespace(
        action=(primary, control),
        pri_param=logits.unsqueeze(0),
        policy_type=torch.tensor([policy_type]),
        policy_valid=torch.ones((1, batch_size), dtype=torch.bool),
    )


def test_behavior_mask_contract_rejects_scored_burnin():
    batch = _batch()
    assert validate_behavior_batch(batch) == (2, 5)
    batch["score_mask"] = np.ones(5, dtype=np.bool_)
    with pytest.raises(ValueError, match="burn-in"):
        validate_behavior_batch(batch)


def test_burnin_logits_have_zero_supervised_gradient():
    logits = torch.randn(2, 5, 3, requires_grad=True)
    targets = torch.zeros(2, 5, dtype=torch.long)
    mask = torch.tensor([False, True, True, True, True])
    objective = compute_masked_imitation_objective(
        logits,
        targets,
        mask,
        ce_coef=1.0,
        margin_coef=0.0,
        pvp_coef=0.0,
    )
    objective["loss"].backward()
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, 1:]) > 0


def test_teacher_force_changes_only_real_rows_without_mutating_actor_out():
    actor_out = _actor_out(
        [util.POLICY_REAL, util.POLICY_SEARCH, util.POLICY_NONE]
    )
    original_primary = actor_out.action[0].clone()
    original_control = actor_out.action[1].clone()
    adapter = HumanActionExecutionAdapter(torch.tensor([1, 1, 1]))
    decision = adapter.prepare(actor_out, torch.tensor([0, 0, 0]))

    assert torch.equal(actor_out.action[0], original_primary)
    assert torch.equal(actor_out.action[1], original_control)
    assert decision.execution_primary.tolist() == [0, 1, 0]
    assert torch.equal(decision.execution_control, original_control)
    assert decision.primary_proposal.tolist() == [2, 1, 0]
    assert decision.primary_argmax.tolist() == [2, 0, 1]


def test_wait_dummy_does_not_overwrite_effective_human_action():
    adapter = HumanActionExecutionAdapter(torch.tensor([2, 3]))
    info = adapter.observe({
        "accepted_primary_action": torch.tensor([4, -1]),
        "accepted_control": torch.tensor([util.STOP, -1]),
        "executed_primary_action": torch.tensor([4, -1]),
        "real_transition": torch.tensor([True, False]),
    })
    assert info["effective_primary_action"].tolist() == [4, 3]
    assert info["effective_search_control"].tolist() == [util.STOP, 0]

    wait_info = adapter.observe({
        "accepted_primary_action": torch.tensor([-1, -1]),
        "accepted_control": torch.tensor([-1, -1]),
        "executed_primary_action": torch.tensor([-1, -1]),
        "real_transition": torch.tensor([False, False]),
    })
    assert wait_info["effective_primary_action"].tolist() == [4, 3]
    assert wait_info["effective_search_control"].tolist() == [util.STOP, 0]


def test_objective_has_no_search_control_gradient_and_coefficients_are_exact():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]], requires_grad=True)
    targets = torch.tensor([0, 0])
    control_logits = torch.randn(2, 3, requires_grad=True)
    objective = compute_imitation_objective(
        logits,
        targets,
        ce_coef=2.0,
        margin=1.0,
        margin_coef=3.0,
        pvp_coef=0.0,
        overall_coef=4.0,
    )
    expected = 4.0 * (
        2.0 * objective["normalized_ce"]
        + 3.0 * objective["margin_loss"]
    )
    assert torch.allclose(objective["loss"], expected)
    logit_grad, control_grad = torch.autograd.grad(
        objective["loss"], (logits, control_logits), allow_unused=True
    )
    assert logit_grad is not None
    assert control_grad is None


def test_detached_bc_logit_metrics_are_stable_for_extreme_finite_rows():
    logits = torch.tensor(
        [
            [1.0e20, -1.0e20, 0.0],
            [1.0e20, -1.0e20, 0.0],
            [0.0, 3.0, -2.0],
            [4.0, 1.0, 5.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 0, 2])

    metrics = detached_imitation_logit_metrics(logits, targets)

    work = logits.detach().double()
    row_index = torch.arange(work.shape[0])
    target_logits = work[row_index, targets]
    expected_nll = torch.logsumexp(work, dim=-1) - target_logits
    other = work.clone()
    other[row_index, targets] = -torch.inf
    expected_gap = target_logits - other.max(dim=-1).values
    expected_rms = torch.sqrt(torch.mean(work.square()))
    assert metrics["nll_max"] == pytest.approx(expected_nll.max().item())
    assert metrics["nll_p99"] == pytest.approx(
        torch.quantile(expected_nll, 0.99).item()
    )
    assert metrics[
        "target_vs_best_other_logit_gap_max"
    ] == pytest.approx(expected_gap.max().item())
    assert metrics[
        "target_vs_best_other_logit_gap_p99"
    ] == pytest.approx(torch.quantile(expected_gap, 0.99).item())
    assert metrics["scored_logits_absmax"] == pytest.approx(1.0e20)
    assert metrics["scored_logits_rms"] == pytest.approx(
        expected_rms.item()
    )
    assert all(np.isfinite(value) for value in metrics.values())
    assert logits.grad is None


def test_bc_mean_is_scaled_by_online_real_policy_rows_only():
    mean_loss = torch.tensor(2.5, requires_grad=True)
    real_mask = torch.tensor([[True, False, False], [False, True, False]])
    scaled = scale_imitation_for_online_rows(mean_loss, real_mask)
    assert scaled.item() == 5.0
    scaled.backward()
    assert mean_loss.grad.item() == 2.0


def test_action_prior_uses_online_real_policy_rows_only():
    learner = SActorLearner.__new__(SActorLearner)
    learner.flags = SimpleNamespace(
        action_prior_weight=1.0, action_prior_ema=0.05
    )
    learner.action_prior = torch.tensor([0.7, 0.2, 0.1])
    learner.action_prior_ema = None
    learner._pending_action_prior_ema = None
    logits = torch.randn(2, 2, 1, 3, requires_grad=True)
    real_mask = torch.tensor([[True, False], [False, False]])
    losses = {}

    total = learner._add_online_action_prior(
        torch.zeros((), requires_grad=True),
        losses,
        SimpleNamespace(pri_param=logits),
        real_mask,
    )
    total.backward()

    assert "action_prior_loss" in losses
    assert torch.count_nonzero(logits.grad[0, 0]) > 0
    assert torch.count_nonzero(logits.grad[~real_mask]) == 0
    assert learner.action_prior_ema is None
    assert learner._pending_action_prior_ema is not None
    assert not learner._pending_action_prior_ema.requires_grad


def test_action_prior_rejects_nonfinite_logits_without_poisoning_ema():
    learner = SActorLearner.__new__(SActorLearner)
    learner.flags = SimpleNamespace(
        action_prior_weight=1.0, action_prior_ema=0.05
    )
    original_ema = torch.tensor([0.6, 0.3, 0.1])
    learner.action_prior = torch.tensor([0.7, 0.2, 0.1])
    learner.action_prior_ema = original_ema.clone()
    learner._pending_action_prior_ema = None
    logits = torch.zeros(1, 1, 1, 3)
    logits[0, 0, 0, 1] = float("nan")

    with pytest.raises(FloatingPointError, match="action-prior batch"):
        learner._add_online_action_prior(
            torch.zeros(()),
            {},
            SimpleNamespace(pri_param=logits),
            torch.ones(1, 1, dtype=torch.bool),
        )

    torch.testing.assert_close(learner.action_prior_ema, original_ema)
    assert learner._pending_action_prior_ema is None


class _AdaptiveActorScaler:
    def __init__(self, scale=256.0):
        self._scale = float(scale)
        self._found_inf = False

    def get_scale(self):
        return self._scale

    def step(self, optimizer):
        self._found_inf = any(
            parameter.grad is not None
            and not torch.isfinite(parameter.grad).all().item()
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        if not self._found_inf:
            optimizer.step()

    def update(self):
        if self._found_inf:
            self._scale /= 2.0


def _actor_gradient_learner(*, float16=True, max_skips=8):
    learner = SActorLearner.__new__(SActorLearner)
    learner.flags = SimpleNamespace(
        float16=float16,
        actor_grad_norm_clipping=0.5,
        actor_amp_max_consecutive_skips=max_skips,
    )
    learner.actor_net = torch.nn.Linear(1, 1, bias=False)
    learner.optimizer = torch.optim.SGD(learner.actor_net.parameters(), lr=0.1)
    learner.scaler = _AdaptiveActorScaler()
    learner.actor_amp_skip_count = 0
    learner.actor_amp_consecutive_skips = 0
    learner._logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None)
    return learner, next(learner.actor_net.parameters())


def test_actor_amp_overflow_skips_then_recovers_without_mutating_failed_step():
    learner, parameter = _actor_gradient_learner()
    before = parameter.detach().clone()
    parameter.grad = torch.full_like(parameter, float("inf"))

    skipped = learner._step_actor_optimizer([parameter], T=1, B=1)

    assert isinstance(skipped, ActorGradientStepResult)
    assert not skipped.optimizer_stepped
    assert skipped.total_norm == 0.0
    assert skipped.amp_scale_before == 256.0
    assert skipped.amp_scale_after == 128.0
    assert learner.actor_amp_skip_count == 1
    assert learner.actor_amp_consecutive_skips == 1
    torch.testing.assert_close(parameter, before)

    parameter.grad = torch.ones_like(parameter)
    recovered = learner._step_actor_optimizer([parameter], T=1, B=1)

    assert recovered.optimizer_stepped
    assert learner.actor_amp_skip_count == 1
    assert learner.actor_amp_consecutive_skips == 0
    assert not torch.equal(parameter.detach(), before)


def test_actor_fp32_nonfinite_gradient_remains_fatal():
    learner, parameter = _actor_gradient_learner(float16=False)
    parameter.grad = torch.full_like(parameter, float("inf"))

    with pytest.raises(FloatingPointError, match="FP32"):
        learner._step_actor_optimizer([parameter], T=1, B=1)


def test_actor_amp_repeated_overflow_is_fatal_at_configured_limit():
    learner, parameter = _actor_gradient_learner(max_skips=3)

    for _ in range(2):
        parameter.grad = torch.full_like(parameter, float("inf"))
        result = learner._step_actor_optimizer([parameter], T=1, B=1)
        assert not result.optimizer_stepped

    parameter.grad = torch.full_like(parameter, float("inf"))
    with pytest.raises(FloatingPointError, match="persisted for 3"):
        learner._step_actor_optimizer([parameter], T=1, B=1)


def test_imitation_model_refresh_rejects_nonfinite_weights():
    model = torch.nn.Linear(3, 2)
    weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
    weights["weight"][0, 0] = float("nan")

    with pytest.raises(ValueError, match="nonfinite"):
        _validate_model_state_dict_compatibility(model, weights, "behavioral ModelNet")


def test_old_checkpoint_has_backward_compatible_imitation_defaults():
    state = imitation_checkpoint_state({"step": 10})
    assert state == {
        "update_count": 0,
        "schedule_step": 0,
        "rng_state": None,
        "data_signature": None,
        "action_prior_ema": None,
    }


def test_actor_learner_resume_rejects_control_objective_mismatch(tmp_path):
    checkpoint_path = tmp_path / "ckp_actor.tar"
    torch.save(
        {
            "dynamic_search": True,
            "dynamic_factorized_control": False,
            "flags": {
                "dynamic_search": True,
                "dynamic_factorized_control": False,
            },
        },
        checkpoint_path,
    )
    learner = SActorLearner.__new__(SActorLearner)
    learner.dynamic_search = True
    learner.dynamic_factorized_control = True

    with pytest.raises(ValueError, match="control objectives"):
        learner.load_checkpoint(str(checkpoint_path))


def test_self_play_resume_rejects_control_objective_mismatch(tmp_path):
    torch.save(
        {
            "dynamic_search": True,
            "dynamic_factorized_control": False,
            "flags": {
                "dynamic_search": True,
                "dynamic_factorized_control": False,
            },
        },
        tmp_path / "ckp_actor.tar",
    )
    worker_class = SelfPlayWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.rank = 0
    worker.dynamic_search = True
    worker.flags = SimpleNamespace(
        ckp=True,
        ckpdir=str(tmp_path),
        preload_actor="",
        dynamic_factorized_control=True,
    )

    with pytest.raises(ValueError, match="control objectives"):
        worker._load_net()


def test_imitation_checkpoint_cannot_resume_with_behavior_disabled():
    learner = SActorLearner.__new__(SActorLearner)
    learner.imitation_enabled = False
    learner.imitation_update_count = 3
    learner._checkpoint_imitation_data_signature = "b" * 64

    with pytest.raises(ValueError, match="icopro_data_path is empty"):
        learner._init_imitation_components()


def test_float_observations_keep_unit_scale_through_real_cenv():
    flags = util.create_setting(
        args=[
            "--dynamic_search", "true",
            "--max_search_steps", "2",
            "--max_depth", "2",
            "--see_h", "false",
            "--see_x", "false",
            "--parallel", "false",
            "--float16", "false",
        ],
        save_flags=False,
    )
    flags.icopro_action_diff_coef = 1.0
    flags.icopro_margin = 1.0
    flags.icopro_margin_coef = 1.0
    flags.icopro_pvp_coef = 0.0
    flags.icopro_coef = 1.0
    batch = _batch(batch_size=1)
    batch["obs_seq"] = batch["obs_seq"].astype(np.float32) / 255.0
    model = _FrozenFakeModel(
        5, spaces.Box(0.0, 1.0, shape=(4, 8, 8), dtype=np.float32)
    )
    runner = DynamicImitationRunner(
        _StopThenProposeActor(5), model, flags, device="cpu"
    )

    result = runner.rollout(batch, tree_carry=True, training=True)

    assert result.count == 4
    assert runner._behavior_env.obs_seq.dtype == np.float32
    assert runner._behavior_env.observation_space.high.max() == 1.0


def test_actor_learner_exception_is_re_raised_and_marks_unsuccessful(monkeypatch):
    class _RemoteMethod:
        def remote(self, *_args, **_kwargs):
            return object()

    learner = SActorLearner.__new__(SActorLearner)
    learner.time = False
    learner.real_step = 0
    learner.flags = SimpleNamespace(total_steps=1)
    learner.actor_buffer = SimpleNamespace(read=_RemoteMethod())
    learner.queue_n = 0.0
    learner._logger = SimpleNamespace(error=lambda *_args: None)
    learner.consume_data = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("bc failure")
    )
    close_calls = []
    learner.close = lambda successful=True: close_calls.append(successful)
    monkeypatch.setattr("thinker.learn_actor.ray.get", lambda _ref: ((), ()))
    monkeypatch.setattr("thinker.learn_actor.ray.internal.free", lambda _ref: None)
    monkeypatch.setattr("thinker.learn_actor.util.tuple_map", lambda value, _fn: value)

    with pytest.raises(RuntimeError, match="bc failure"):
        learner.learn_data()
    assert close_calls == [False]


def test_model_learner_exception_is_re_raised_and_marks_unsuccessful(monkeypatch):
    class _RemoteMethod:
        def __init__(self):
            self.calls = 0

        def remote(self, *_args, **_kwargs):
            self.calls += 1
            return object()

    set_finish = _RemoteMethod()
    learner = SModelLearner.__new__(SModelLearner)
    learner.time = False
    learner.timing = None
    learner.real_step = 0
    learner.replay_ratio = 0.0
    learner.finish = False
    learner.flags = SimpleNamespace(total_steps=1, max_replay_ratio=10.0)
    learner.model_buffer = SimpleNamespace(set_finish=set_finish)
    learner.read_buffer_ptr = lambda: object()
    learner.init_psteps = lambda _data: None
    learner._logger = SimpleNamespace(error=lambda *_args: None)
    learner.consume_data = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("model failure")
    )
    close_calls = []
    learner.close = lambda successful=True: close_calls.append(successful)
    monkeypatch.setattr(
        "thinker.learn_model.ray.get", lambda _ref: {"replay_ratio": 0.0}
    )
    monkeypatch.setattr("thinker.learn_model.ray.internal.free", lambda _ref: None)

    with pytest.raises(RuntimeError, match="model failure"):
        learner.learn_data()
    assert set_finish.calls == 1
    assert close_calls == [False]


def test_imitation_model_refresh_waits_for_initial_publication(monkeypatch):
    class _RemoteMethod:
        def remote(self, *_args, **_kwargs):
            return object()

    class _Model(torch.nn.Linear):
        def set_weights(self, value):
            self.load_state_dict(value)

    model = _Model(3, 2)
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    replies = iter((None, None, weights))
    learner = SActorLearner.__new__(SActorLearner)
    learner.model_param_buffer = SimpleNamespace(get_data=_RemoteMethod())
    learner.flags = SimpleNamespace(ckp=False, preload="")
    learner.bc_model_net = model
    monkeypatch.setattr("thinker.learn_actor.ray.get", lambda _ref: next(replies))
    monkeypatch.setattr("thinker.learn_actor.time.sleep", lambda _seconds: None)

    assert learner._refresh_imitation_model(require_weights=True) is True
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_runner_executes_burnin_then_four_human_actions_with_current_cenv():
    flags = util.create_setting(
        args=[
            "--dynamic_search", "true",
            "--max_search_steps", "2",
            "--max_depth", "2",
            "--see_h", "false",
            "--see_x", "false",
            "--parallel", "false",
            "--float16", "false",
        ],
        save_flags=False,
    )
    flags.icopro_action_diff_coef = 1.0
    flags.icopro_margin = 1.0
    flags.icopro_margin_coef = 1.0
    flags.icopro_pvp_coef = 0.0
    flags.icopro_coef = 1.0
    batch = _batch()
    batch["actions_seq"] = np.asarray(
        [[1, 2, 3, 4, 0], [2, 3, 4, 0, 1]], dtype=np.int64
    )
    batch["initial_prev_action"] = np.asarray([4, 3], dtype=np.int64)
    actor = _StopThenProposeActor(num_actions=5)
    model = _FrozenFakeModel(num_actions=5)
    runner = DynamicImitationRunner(actor, model, flags, device="cpu")

    result = runner.rollout(batch, tree_carry=True, training=True)

    assert result.count == 8
    assert result.augmented_steps == 10
    assert result.burnin_executed.tolist() == [1, 2]
    assert result.executed.tolist() == [[2, 3, 4, 0], [3, 4, 0, 1]]
    assert result.proposal.tolist() == [[4] * 4, [4] * 4]
    assert not result.root_carried.any()  # no human child was searched
    detached_metrics = result.detached_metrics()
    for metric_name in (
        "nll_max",
        "nll_p99",
        "target_vs_best_other_logit_gap_max",
        "target_vs_best_other_logit_gap_p99",
        "scored_logits_absmax",
        "scored_logits_rms",
    ):
        assert metric_name in detached_metrics
        assert np.isfinite(detached_metrics[metric_name])
    result.loss.backward()
    assert actor.real_logits.grad is not None
    assert all(not parameter.requires_grad for parameter in model.parameters())

    # A same-shape batch must reuse the planner while replacing only sequence
    # data.  Constructor-only action/observation contracts are retained by the
    # existing BehaviorSequenceVectorEnv rather than forwarded to its updater.
    planner = runner._planner
    second_batch = _batch()
    second_batch["actions_seq"] = np.asarray(
        [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=np.int64
    )
    second_batch["initial_prev_action"] = np.asarray([0, 4], dtype=np.int64)
    second_result = runner.rollout(second_batch, tree_carry=True, training=True)

    assert runner._planner is planner
    assert second_result.burnin_executed.tolist() == [0, 4]
    assert second_result.executed.tolist() == [[1, 2, 3, 4], [3, 2, 1, 0]]
    runner.close()


def test_runner_associates_burnin_carry_and_preserves_human_last_pri_in_wait():
    flags = util.create_setting(
        args=[
            "--dynamic_search", "true",
            "--max_search_steps", "3",
            "--max_depth", "3",
            "--tree_carry", "true",
            "--see_h", "false",
            "--see_x", "false",
            "--parallel", "false",
            "--float16", "false",
        ],
        save_flags=False,
    )
    flags.icopro_action_diff_coef = 1.0
    flags.icopro_margin = 1.0
    flags.icopro_margin_coef = 1.0
    flags.icopro_pvp_coef = 0.0
    flags.icopro_coef = 1.0
    batch = _batch()
    # Human action 1 is explicitly expanded during every SEARCH stage.
    batch["actions_seq"][:] = 1
    batch["initial_prev_action"] = np.asarray([3, 2], dtype=np.int64)
    actor = _StaggeredCarryActor(num_actions=5, edge_count=5)
    model = _FrozenFakeModel(num_actions=5)
    runner = DynamicImitationRunner(actor, model, flags, device="cpu")

    result = runner.rollout(batch, tree_carry=True, training=True)

    assert result.count == 2 * 4  # burn-in is deliberately excluded
    assert result.augmented_steps == 5 * 4
    assert result.burnin_executed.tolist() == [1, 1]
    assert result.executed.tolist() == [[1] * 4, [1] * 4]
    assert actor.last_pri_history[0].tolist() == [3, 2]
    # Carry generated by the burn-in edge belongs to the first scored root.
    assert result.root_carried.tolist() == [[True] * 4, [True] * 4]
    assert result.carried_descendant_visit_count.tolist() == [
        [0] * 4,
        [1] * 4,
    ]
    assert result.carried_descendant_expanded_count.tolist() == [
        [0] * 4,
        [1] * 4,
    ]
    assert result.useful_carry.tolist() == [
        [False] * 4,
        [True] * 4,
    ]
    result.loss.backward()
    assert torch.count_nonzero(actor.stage_logits.grad[0]) == 0
    assert torch.count_nonzero(actor.stage_logits.grad[1:]) > 0

    # Row zero enters WAIT while row one still needs its real action.  The
    # dummy action on that WAIT call must not replace the executed human token.
    wait_observations = [
        last_pri[phase == util.WAIT_PHASE]
        for phase, last_pri in zip(
            actor.phase_history, actor.last_pri_history
        )
        if torch.any(phase == util.WAIT_PHASE)
    ]
    assert wait_observations
    assert torch.cat(wait_observations).tolist() == [1] * len(wait_observations)
    runner.close()
