from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thinker import util
from thinker.actor_net import compute_voc_gate_distribution
from thinker.bc_loader import FrameStackedBehavioralDataLoader, behavioral_data_signature
from thinker.dynamic_imitation import validate_behavior_batch
from thinker.learn_actor import compute_dynamic_voc_loss, SActorLearner
from thinker.planning_protocol import (
    planning_training_contract, validate_planning_training_checkpoint,
)
from thinker.voc_gru_gradient import SelectiveVOCGradient
from thinker.voc_training_metrics import TrainingMetrics
from test_dynamic_imitation_data import _archive, _modern_path
from test_dynamic_imitation_actor import _batch, _FrozenFakeModel, _StopThenProposeActor
from thinker.dynamic_imitation import DynamicImitationRunner


def test_fresh_training_defaults_to_eight_scored_steps_and_keeps_burnin_16():
    args = ['--dynamic_search', 'true', '--icopro_burn_in_steps', '16',
            '--icopro_sampling_context', '16']
    flags = util.create_setting(args=args, save_flags=False)
    assert flags.batch_length == 8
    assert flags.icopro_burn_in_steps == flags.icopro_sampling_context == 16
    legacy = util.create_setting(args=args + ['--batch_length', '4'], save_flags=False)
    assert legacy.batch_length == 4


def _loss(raw, target=None, mode="common_contrast_v1"):
    n = raw.shape[1]
    policy = torch.tensor([.009, .001, .99]).log().expand(1, n, 3)
    return compute_dynamic_voc_loss(
        voc_q=raw, target_control_logits=policy, behavior_control_logits=policy,
        control_action=torch.tensor([[util.PROCEED, util.STOP]]).expand(1, n),
        control_valid=torch.ones(1, n, dtype=torch.bool),
        voc_target=torch.full((1, n), .1) if target is None else target,
        mode="shadow", dueling_q=True, voc_state_value=torch.zeros(1, n),
        critic_parameterization=mode,
    )


def test_common_value_error_does_not_require_a_false_contrast():
    raw = torch.full((1, 2, 2), .1, requires_grad=True)
    result = _loss(raw)
    torch.testing.assert_close(result.q_values, torch.full_like(raw, .1))
    assert result.q_loss.item() == pytest.approx(0, abs=1e-12)
    assert result.delta_q.abs().sum() == 0
    assert _loss(raw, mode="policy_centered_v1").q_loss.item() > .009


def test_gru_gradient_excludes_common_direction_but_head_still_learns_it():
    raw = torch.zeros(1, 2, 2, requires_grad=True)
    result = _loss(raw)
    full = torch.autograd.grad(result.q_loss, raw, retain_graph=True)[0]
    contrast = torch.autograd.grad(result.gru_q_loss, raw)[0]
    torch.testing.assert_close(contrast.sum(-1), torch.zeros(1, 2), atol=1e-8, rtol=0)
    assert full.sum(-1).abs().sum() > 0
    assert contrast.abs().sum() > 0
    torch.testing.assert_close(result.gru_q_loss, result.q_loss)


def test_online_and_ema_reconstruct_identical_new_critic():
    learner = object.__new__(SActorLearner)
    learner.flags = SimpleNamespace(voc_critic_parameterization="common_contrast_v1")
    learner.dynamic_voc_mode = "shadow"
    learner.voc_ema_gate_weight = torch.tensor([[.4, .1], [-.2, .3]])
    learner.voc_ema_gate_bias = torch.tensor([.1, .2])
    features = torch.tensor([[[.3, .7], [.4, .2]]])
    raw = torch.nn.functional.linear(features, learner.voc_ema_gate_weight, learner.voc_ema_gate_bias)
    policy = torch.tensor([.009, .001, .99]).log().expand(1, 2, 3)
    _, ema = learner._compute_ema_gate_loss(features=features, logits=policy,
        valid=torch.ones(1, 2, dtype=torch.bool), state_value=torch.zeros(1, 2))
    torch.testing.assert_close(ema, _loss(raw).q_values)


def test_epsilon_changes_support_not_q_temperature():
    raw = torch.zeros(1, 2, 3)
    odds = torch.tensor([[-100., 100.]])
    train = compute_voc_gate_distribution(raw, raw_gate_log_odds=odds, epsilon=.1)
    evaluation = compute_voc_gate_distribution(raw, raw_gate_log_odds=odds, epsilon=0)
    torch.testing.assert_close(train.continue_prob, torch.tensor([[.05, .95]]))
    assert evaluation.continue_prob[0, 0] < 1e-10
    assert evaluation.continue_prob[0, 1] > .999


@pytest.mark.parametrize('scored_length', [4, 8])
def test_burnin_conditions_score_identical_targets_and_preserve_boundaries(tmp_path, scored_length):
    _archive(_modern_path(tmp_path, 1), length=100, times=False,
             episode_starts=(0, 50), episode_ends=(49, 99))
    batches, signatures = [], []
    for burn in (1, 8, 16):
        loader = FrameStackedBehavioralDataLoader(tmp_path, num_actions=9,
            sessions=(1,), scored_length=scored_length, burn_in_steps=burn, sampling_context=16,
            frame_stack_n=1, target_size=(6, 7), seed=123)
        batch = loader.get_sequence_batch(16)
        assert validate_behavior_batch(batch) == (16, burn+scored_length)
        assert batch['score_mask'].sum() == scored_length
        assert not np.any(batch['done_seq'][:, :-1])
        assert not np.any(batch['truncated_seq'][:, :-1])
        batches.append(batch)
        signatures.append(behavioral_data_signature(loader, tmp_path))
    for batch in batches[1:]:
        for name in ('actions_seq', 'rewards_seq'):
            np.testing.assert_array_equal(batch[name][:, -scored_length:], batches[0][name][:, -scored_length:])
        np.testing.assert_array_equal(batch['observation_source_index'][:, -(scored_length+1):],
                                      batches[0]['observation_source_index'][:, -(scored_length+1):])
        np.testing.assert_array_equal(batch['obs_seq'][:, -(scored_length+1):], batches[0]['obs_seq'][:, -(scored_length+1):])
    assert len(set(signatures)) == 3


@pytest.mark.parametrize('burn', [1, 8, 16])
@pytest.mark.parametrize('scored_length', [4, 8])
def test_burnin_coverage_and_validation(tmp_path, burn, scored_length):
    _archive(_modern_path(tmp_path, 1), length=50, times=False)
    loader = FrameStackedBehavioralDataLoader(tmp_path, num_actions=9,
        sessions=(1,), scored_length=scored_length, burn_in_steps=burn, sampling_context=16, target_size=(6, 7))
    batches = list(loader.iter_batches(batch_size=2))
    targets = [tuple(row) for b in batches for row in b['observation_source_index'][:, -(scored_length+1):-1]]
    coverage = loader.evaluation_coverage()
    assert len(targets)*scored_length == coverage['unique_scored_targets']
    batch = loader.get_sequence_batch(1)
    batch['score_mask'][-2] = False
    with pytest.raises(ValueError, match='after burn-in'):
        validate_behavior_batch(batch)


@pytest.mark.parametrize('burn', [8, 16])
@pytest.mark.parametrize('scored_length', [4, 8])
def test_full_planner_replay_processes_warmup_and_scores_actions(burn, scored_length):
    flags = util.create_setting(args=["--dynamic_search", "true", "--max_search_steps", "2",
        "--max_depth", "2", "--see_h", "false", "--see_x", "false",
        "--parallel", "false", "--float16", "false"], save_flags=False)
    batch = _batch(scored_length=burn+scored_length-1)
    batch['score_mask'] = np.array([False]*burn + [True]*scored_length)
    batch['actions_seq'][:] = np.arange(burn+scored_length) % 5
    actor = _StopThenProposeActor(num_actions=5)
    runner = DynamicImitationRunner(actor, _FrozenFakeModel(num_actions=5), flags, device='cpu')
    try:
        result = runner.rollout(batch, tree_carry=True, training=True)
        assert result.count == 2*scored_length
        assert result.augmented_steps == 2*(burn+scored_length)
        result.loss.backward()
        assert torch.isfinite(actor.real_logits.grad).all()
    finally:
        runner.close()


@pytest.mark.parametrize('key,value', [('icopro_burn_in_steps', 8),
    ('voc_gru_gradient_scale', 10.), ('voc_critic_parameterization', 'common_contrast_v1')])
def test_resume_rejects_changed_training_contract(key, value):
    flags = dict(voc_gru_connected=True, icopro_sampling_context=16)
    checkpoint = dict(flags=flags, planning_training_contract=planning_training_contract(flags))
    validate_planning_training_checkpoint(checkpoint, flags)
    with pytest.raises(ValueError, match='different'):
        validate_planning_training_checkpoint(checkpoint, {**flags, key: value})
    checkpoint.pop('planning_training_contract')
    with pytest.raises(ValueError, match='missing'):
        validate_planning_training_checkpoint(checkpoint)


@pytest.mark.parametrize('scale', [1., 10.])
def test_multiplier_is_applied_to_gru_before_actor_clipping(scale):
    actor = torch.nn.Module()
    actor.tree_rep_encoder = torch.nn.Linear(1, 1, bias=False)
    actor.policy = torch.nn.Linear(1, 1)
    actor.voc_head = torch.nn.Linear(1, 2)
    parameter = actor.tree_rep_encoder.weight
    observed = []
    def step(*args):
        observed.append(parameter.grad.item())
        torch.nn.utils.clip_grad_norm_([parameter], .5)
        return SimpleNamespace(optimizer_stepped=True)
    learner = SimpleNamespace(actor_net=actor, flags=SimpleNamespace(voc_gru_gradient_scale=scale),
        optimizer=torch.optim.SGD(actor.parameters(), lr=.1), compute_losses=lambda: None,
        _step_actor_optimizer=step)
    adapter = SelectiveVOCGradient(learner, True)
    try:
        parameter.grad = torch.full_like(parameter, 2.)
        adapter.pending = (torch.full_like(parameter, .1),)
        adapter._step([parameter], 1, 1)
        assert observed == pytest.approx([2+.1*scale])
        assert parameter.grad.abs().item() <= .5
        assert adapter.last_metrics['voc_gru_applied_norm'] == pytest.approx(.1*scale)
    finally:
        adapter.close()


def test_collector_counts_cap_transition_once_and_records_recent_histogram(tmp_path):
    actor = SimpleNamespace(num_actions=3, voc_gate_head=torch.nn.Linear(1, 1))
    flags = SimpleNamespace(env_n=1, max_search_steps=2, min_search_steps=1,
                            voc_gate_q_temperature=.01)
    metrics = TrainingMetrics(actor, flags, tmp_path)
    metrics.current[:] = 1
    metrics.gate_logits = torch.tensor([[1.]])
    previous = SimpleNamespace(forced_stop=torch.tensor([False]),
        phase=torch.tensor([util.SEARCH_PHASE]), search_steps=torch.tensor([1]),
        legal_control_mask=torch.tensor([[True, True, True]]))
    out = SimpleNamespace(control_valid=torch.tensor([True]),
        search_control_logits=torch.tensor([[[0., -100., -1.]]]))
    info = dict(real_transition=torch.tensor([False]), think_reward=torch.tensor([-1.]),
                accepted_control=torch.tensor([util.PROCEED]), stage_end=torch.tensor([True]),
                forced_stop=torch.tensor([True]))
    try:
        metrics.observe(previous, out, info, real_step=1, attempt=1)
        assert metrics.forced == 1
        previous.phase[:] = util.NEED_REAL_ACTION_PHASE
        out.control_valid[:] = False
        info.update(real_transition=torch.tensor([True]), think_reward=torch.tensor([0.]),
            stage_end=torch.tensor([False]), forced_stop=torch.tensor([False]),
            episode_return=torch.tensor([0.]), real_done=torch.tensor([False]))
        metrics.observe(previous, out, info, real_step=1, attempt=1)
        assert metrics.forced == 1
        assert metrics.metrics()['window_cap_fraction'] == 1
        assert metrics.metrics()['window_ema_positive_continue_fraction'] == 1
        metrics.publish()
        assert metrics.metrics()['window_decisions'] == 0
        assert metrics.hist.sum() == 1
    finally:
        metrics.close()
