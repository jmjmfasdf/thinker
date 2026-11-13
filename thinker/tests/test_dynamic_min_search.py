"""Minimum computation, lower cost, and all-session training contracts."""
import pytest
import torch

from thinker import util
from thinker.cenv import cModelWrapper, cPerfectWrapper
from thinker.learn_actor import SActorLearner
from tests.test_dynamic_cenv import FakeVectorEnv, FakeModel, _flags, _step
from tests.test_dynamic_voc_learner import _learner_flags
from tests.test_dynamic_learner_integration import _rollout, _NullWriter


@pytest.mark.parametrize('wrapper_type', [0, 2])
@pytest.mark.parametrize('first_control', [util.PROCEED, util.RESET])
def test_first_query_required_charged_and_reenabled_per_real_decision(wrapper_type, first_control):
    flags = _flags(cap=3, wrapper_type=wrapper_type)
    flags.min_search_steps = 1
    env = FakeVectorEnv(env_n=1)
    model = FakeModel(env.num_actions)
    cls = cModelWrapper if wrapper_type == 0 else cPerfectWrapper
    wrapper = cls(env, 1, flags, model, device=torch.device('cpu'))
    try:
        _, info = wrapper.reset(model)
        assert info['search_steps'].item() == 0
        assert not info['legal_control_mask'][0, util.STOP]
        with pytest.raises(AssertionError, match='min_search_steps'):
            _step(wrapper, model, [0], [util.STOP])
        _, *_, info = _step(wrapper, model, [1], [first_control])
        assert info['think_reward'].item() == -1
        assert info['search_steps'].item() == 1
        assert info['legal_control_mask'][0, util.STOP]
        assert not info['real_transition'].item()
        # A perfect model queries a cloned emulator; this is still imagination.
        assert len(env.step_calls) == (1 if wrapper_type == 2 else 0)
        _, *_, info = _step(wrapper, model, [1], [util.STOP])
        assert info['think_reward'].item() == 0
        _, *_, info = _step(wrapper, model, [2], [util.PROCEED])
        assert info['real_transition'].item()
        assert info['search_steps'].item() == 0
        assert not info['legal_control_mask'][0, util.STOP]
        assert len(env.step_calls) == (2 if wrapper_type == 2 else 1)
    finally:
        wrapper.close()


def settings(**extra):
    args = ['--dynamic_search', 'true', '--dynamic_factorized_control', 'true',
            '--dynamic_voc_mode', 'control', '--think_cost', '0.0001',
            '--think_cost_anneal', 'false', '--min_search_steps', '1',
            '--max_search_steps', '30', '--parallel', 'false']
    for k, v in extra.items():
        args += ['--' + k, str(v)]
    return util.create_setting(args=args, save_flags=False)


def test_config_accepts_lower_cost_and_minimum():
    flags = settings()
    assert flags.think_cost == 0.0001
    assert flags.min_search_steps == 1
    assert flags.dynamic_search_state_reset == 'episode'


@pytest.mark.parametrize('cost', [-0.1, 'nan', 'inf'])
def test_cost_must_be_finite_nonnegative(cost):
    with pytest.raises(ValueError, match='think_cost'):
        settings(think_cost=cost)


@pytest.mark.parametrize('minimum', [-1, 31])
def test_minimum_must_fit_budget(minimum):
    with pytest.raises(ValueError, match='min_search_steps'):
        settings(min_search_steps=minimum)


def test_only_holdout_session_list_can_be_empty():
    parse = SActorLearner._parse_int_list
    assert parse('', 'icopro_holdout_sessions') == ()
    assert parse([], 'icopro_holdout_sessions') == ()
    assert parse('1,2,3,4', 'icopro_train_sessions') == (1,2,3,4)
    for name in ('icopro_subjects', 'icopro_train_sessions'):
        with pytest.raises(ValueError, match='empty'):
            parse('', name)


def test_learner_excludes_mandatory_query_from_voc_not_branch_rl(tmp_path, monkeypatch):
    flags = _learner_flags(tmp_path, 'min1-cost0001')
    flags.min_search_steps = 1
    flags.think_cost = 0.0001
    actor, train_out, initial_state = _rollout(flags)
    monkeypatch.setattr('thinker.learn_actor.FileWriter', lambda **_: _NullWriter())
    learner = SActorLearner(ray_obj=None, actor_param={}, flags=flags,
                            actor_net=actor, device=torch.device('cpu'))
    losses, shifted = learner.compute_losses(train_out, initial_state)
    legal = train_out.legal_control_mask[:-1]
    required = shifted.control_valid.bool() & ~legal[..., util.STOP]
    optional = shifted.control_valid.bool() & legal[..., util.STOP] & legal[..., :util.STOP].any(-1)
    assert required.any() and optional.any()
    assert losses['voc_mandatory_excluded_count'].item() == required.sum().item()
    assert losses['voc_optional_choice_count'].item() == optional.sum().item()
    assert shifted.primary_valid[required].all()
    assert torch.isfinite(losses['total_loss'])
    losses['total_loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.tree_rep_encoder.parameters())
