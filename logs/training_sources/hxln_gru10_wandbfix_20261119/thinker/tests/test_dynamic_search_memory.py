"""Real-time search memory must cross observations, but never episodes."""
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from thinker import util

_spec = importlib.util.spec_from_file_location(
    'dynamic_actor_memory_helpers', Path(__file__).with_name('test_dynamic_actor.py')
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def network(policy='episode'):
    torch.manual_seed(20261111)
    flags = helpers._flags(True)
    if policy is None:
        del flags.dynamic_search_state_reset
    else:
        flags.dynamic_search_state_reset = policy
    return flags, helpers._network(flags)


def root(flags, *, done=False, truncated=False, transition=True):
    env = helpers._env_out(flags, [util.SEARCH_PHASE],
        reset_mask=torch.ones(1, dtype=torch.bool))
    return env._replace(
        real_transition=torch.full_like(env.real_transition, transition),
        real_done=torch.full_like(env.real_done, done),
        done=torch.full_like(env.done, done),
        truncated_done=torch.full_like(env.truncated_done, truncated),
    )


def test_memory_and_gradient_cross_nonterminal_real_observation():
    flags, actor = network()
    seed = torch.randn_like(actor.initial_state(1)[0], requires_grad=True)
    env = root(flags)
    _, carried = actor(env, (seed,))
    _, fresh = actor(env, actor.initial_state(1))
    assert not torch.equal(carried[0], fresh[0])
    gradient, = torch.autograd.grad(carried[0].sum(), seed)
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize('boundary', ['done', 'truncated', 'fresh_run'])
def test_episode_boundaries_clear_memory_and_cross_boundary_gradient(boundary):
    flags, actor = network()
    seed = torch.randn_like(actor.initial_state(1)[0], requires_grad=True)
    env = root(flags, done=boundary=='done', truncated=boundary=='truncated',
        transition=boundary!='fresh_run')
    _, carried = actor(env, (seed,))
    _, fresh = actor(env, actor.initial_state(1))
    torch.testing.assert_close(carried[0], fresh[0], rtol=0, atol=0)
    gradient, = torch.autograd.grad(carried[0].sum(), seed)
    assert torch.count_nonzero(gradient) == 0


def test_life_loss_done_without_real_episode_end_preserves_memory():
    flags, actor = network()
    env = root(flags)._replace(done=torch.ones(1,1,dtype=torch.bool))
    seed = torch.randn_like(actor.initial_state(1)[0])
    _, carried = actor(env, (seed,))
    _, fresh = actor(env, actor.initial_state(1))
    assert not torch.equal(carried[0], fresh[0])


@pytest.mark.parametrize('policy', ['observation', None])
def test_explicit_and_unversioned_legacy_checkpoints_keep_observation_reset(policy):
    flags, actor = network(policy)
    env = root(flags)
    seed = torch.randn_like(actor.initial_state(1)[0])
    _, carried = actor(env, (seed,))
    _, fresh = actor(env, actor.initial_state(1))
    torch.testing.assert_close(carried[0], fresh[0], rtol=0, atol=0)


def test_chunked_search_and_real_observations_match_continuous_evaluation():
    flags, actor = network()
    rows = [root(flags, transition=False),
        helpers._env_out(flags, [util.SEARCH_PHASE]),
        helpers._env_out(flags, [util.WAIT_PHASE], token_valid=torch.zeros(1,dtype=torch.bool)),
        root(flags), helpers._env_out(flags, [util.SEARCH_PHASE]),
        root(flags, truncated=True), root(flags)]
    continuous = util.EnvOut(*(None if rows[0][i] is None else
        torch.cat([r[i] for r in rows], dim=0) for i in range(len(rows[0]))))
    _, expected = actor(continuous, actor.initial_state(1))
    state = actor.initial_state(1)
    for env in rows:
        _, state = actor(env, state)
    torch.testing.assert_close(state[0], expected[0], rtol=1e-6, atol=1e-7)


def test_new_run_default_and_explicit_cli_policy():
    flags = util.create_setting(args=['--dynamic_search','true','--parallel','false'], save_flags=False)
    assert flags.dynamic_search_state_reset == 'episode'
    flags = util.create_setting(args=['--dynamic_search','true','--parallel','false',
        '--dynamic_search_state_reset','observation'], save_flags=False)
    assert flags.dynamic_search_state_reset == 'observation'


@pytest.mark.parametrize('policy', ['episode', None])
def test_resume_restores_saved_policy_or_legacy_behavior(tmp_path, policy):
    flags = util.create_setting(args=['--dynamic_search','true','--parallel','false'], save_flags=False)
    cfg = dict(vars(flags))
    if policy is None:
        cfg.pop('dynamic_search_state_reset')
    directory = tmp_path / 'memory-test'
    directory.mkdir()
    (directory/'config_c.yaml').write_text(yaml.safe_dump(cfg))
    resumed = util.create_setting(args=['--ckp','true','--savedir',str(tmp_path),
        '--xpid','memory-test','--parallel','false'],save_flags=False)
    assert getattr(resumed,'dynamic_search_state_reset','observation') == (policy or 'observation')


def test_state_dict_round_trip_preserves_persistent_prediction():
    flags, actor = network()
    restored = helpers._network(flags)
    restored.load_state_dict(actor.state_dict())
    first, second = root(flags, transition=False), root(flags)
    _, state = actor(first, actor.initial_state(1))
    out, next_state = actor(second, state)
    other, other_state = restored(second, state)
    torch.testing.assert_close(out.pri_param, other.pri_param, rtol=0, atol=0)
    torch.testing.assert_close(next_state[0], other_state[0], rtol=0, atol=0)
