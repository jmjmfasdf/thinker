import hashlib

import numpy as np
import pytest
import torch

from thinker import util
from thinker.cenv import cModelWrapper, cPerfectWrapper
from tests.test_dynamic_cenv import FakeModel, FakeVectorEnv, _flags, _step


class InitialActionCaptureModel(FakeModel):
    def __init__(self, num_actions):
        super().__init__(num_actions)
        self.input_actions = []

    def __call__(self, env_state, done, actions, state):
        self.input_actions.append(actions.detach().cpu().clone())
        return super().__call__(env_state, done, actions, state)


def _learned_wrapper(*, tree_carry=True, truncate_on_step=False):
    env = FakeVectorEnv(truncate_on_step=truncate_on_step)
    model = InitialActionCaptureModel(env.num_actions)
    flags = _flags(tree_carry=tree_carry)
    wrapper = cModelWrapper(
        env, env.env_n, flags, model, device=torch.device("cpu")
    )
    return wrapper, env, model, flags


def test_initial_action_seeds_model_root_and_initial_env_out():
    wrapper, env, model, flags = _learned_wrapper()
    initial_action = torch.tensor([1, 3, 4], dtype=torch.long)

    state, info = wrapper.reset(model, initial_action=initial_action)

    assert model.input_actions[0].shape == (1, env.env_n, 1)
    torch.testing.assert_close(
        model.input_actions[0][0, :, 0], initial_action
    )
    root_action = util.slice_dynamic_tree_reps(env.num_actions)["root_action"]
    torch.testing.assert_close(
        state["tree_reps"][:, root_action],
        torch.nn.functional.one_hot(
            initial_action, num_classes=env.num_actions
        ).float(),
    )
    torch.testing.assert_close(info["last_pri"], initial_action)
    assert not info["root_carried"].any()
    assert not info["carried_descendant_visit_count"].any()
    assert not info["carried_descendant_expanded_count"].any()
    assert not info["useful_carry"].any()

    env_out = util.init_env_out(
        state, info, flags, dim_actions=1, tuple_action=False
    )
    torch.testing.assert_close(env_out.last_pri[0], initial_action)
    assert env_out.root_carried.shape == (1, env.env_n)
    assert not env_out.root_carried.any()
    assert not env_out.carried_descendant_visit_count.any()
    assert not env_out.carried_descendant_expanded_count.any()
    assert not env_out.useful_carry.any()
    wrapper.close()


@pytest.mark.parametrize(
    ("initial_action", "expected"),
    [
        (None, [0, 0, 0]),
        (np.int64(2), [2, 2, 2]),
        ([4, 1, 0], [4, 1, 0]),
    ],
)
def test_initial_action_default_scalar_and_batch_contract(
    initial_action, expected
):
    wrapper, env, model, _flags_ = _learned_wrapper()

    _state, info = wrapper.reset(model, initial_action=initial_action)

    assert model.input_actions[0][0, :, 0].tolist() == expected
    assert info["last_pri"].tolist() == expected
    wrapper.close()


@pytest.mark.parametrize(
    ("initial_action", "error"),
    [
        ([[1], [2], [3]], ValueError),
        ([0, 1], ValueError),
        ([0, 1, 5], ValueError),
        ([-1, 1, 2], ValueError),
        ([0.0, 1.0, 2.0], TypeError),
        (True, TypeError),
    ],
)
def test_initial_action_rejects_bad_shape_type_and_range(
    initial_action, error
):
    wrapper, _env, model, _flags_ = _learned_wrapper()

    with pytest.raises(error):
        wrapper.reset(model, initial_action=initial_action)

    wrapper.close()


@pytest.mark.parametrize(
    ("wrapper_cls", "wrapper_type"),
    [(cModelWrapper, 0), (cPerfectWrapper, 2)],
)
def test_root_carried_reports_actual_per_environment_promotion(
    wrapper_cls, wrapper_type
):
    env = FakeVectorEnv()
    model = FakeModel(env.num_actions)
    flags = _flags(wrapper_type=wrapper_type, tree_carry=True)
    wrapper = wrapper_cls(
        env, env.env_n, flags, model, device=torch.device("cpu")
    )
    _state, info = wrapper.reset(model)
    assert not info["root_carried"].any()

    # Expand one child per environment, but execute an unexpanded action in
    # environment 1.  Carry is therefore a per-environment event.
    _state, _reward, _done, _truncated, info = _step(
        wrapper, model, [1, 2, 3], [util.PROCEED] * env.env_n
    )
    assert not info["root_carried"].any()
    _step(wrapper, model, [0, 0, 0], [util.STOP] * env.env_n)
    state, reward, done, truncated, info = _step(
        wrapper, model, [1, 4, 3], [util.PROCEED] * env.env_n
    )

    assert info["real_transition"].all()
    assert info["root_carried"].tolist() == [True, False, True]
    assert info["carried_descendant_visit_count"].tolist() == [0, 0, 0]
    assert info["carried_descendant_expanded_count"].tolist() == [0, 0, 0]
    assert info["useful_carry"].tolist() == [False, False, False]
    info_without_episode_stats = dict(info)
    info_without_episode_stats.pop("episode_return", None)
    env_out = util.create_env_out(
        (torch.tensor([1, 4, 3]), torch.zeros(env.env_n, dtype=torch.long)),
        state,
        reward,
        done,
        truncated,
        info_without_episode_stats,
        flags,
    )
    assert env_out.root_carried.tolist() == [[True, False, True]]
    assert env_out.carried_descendant_visit_count.tolist() == [[0, 0, 0]]
    assert env_out.carried_descendant_expanded_count.tolist() == [[0, 0, 0]]
    assert env_out.useful_carry.tolist() == [[False, False, False]]
    wrapper.close()


@pytest.mark.parametrize(
    ("tree_carry", "truncate_on_step"),
    [(False, False), (True, True)],
)
def test_root_carried_is_false_when_disabled_or_terminal(
    tree_carry, truncate_on_step
):
    wrapper, env, model, _flags_ = _learned_wrapper(
        tree_carry=tree_carry, truncate_on_step=truncate_on_step
    )
    wrapper.reset(model)
    _step(wrapper, model, [1, 2, 3], [util.PROCEED] * env.env_n)
    _step(wrapper, model, [0, 0, 0], [util.STOP] * env.env_n)
    _state, _reward, _done, _truncated, info = _step(
        wrapper, model, [1, 2, 3], [util.PROCEED] * env.env_n
    )

    assert info["real_transition"].all()
    assert not info["root_carried"].any()
    assert not info["carried_descendant_visit_count"].any()
    assert not info["carried_descendant_expanded_count"].any()
    assert not info["useful_carry"].any()
    wrapper.close()


def _run_scripted_one_edge(tree_carry):
    env = FakeVectorEnv(env_n=1)
    model = FakeModel(env.num_actions)
    flags = _flags(tree_carry=tree_carry)
    wrapper = cModelWrapper(
        env, env.env_n, flags, model, device=torch.device("cpu")
    )
    wrapper.reset(model)
    _step(wrapper, model, [1], [util.PROCEED])
    _step(wrapper, model, [0], [util.STOP])
    state, reward, done, truncated, info = _step(
        wrapper, model, [1], [util.PROCEED]
    )
    wrapper.close()
    return state, reward, done, truncated, info


def test_one_edge_promotion_is_root_carried_but_not_useful():
    carry_on = _run_scripted_one_edge(True)
    carry_off = _run_scripted_one_edge(False)

    assert carry_on[4]["root_carried"].tolist() == [True]
    assert carry_off[4]["root_carried"].tolist() == [False]
    for result in (carry_on, carry_off):
        info = result[4]
        assert info["carried_descendant_visit_count"].tolist() == [0]
        assert info["carried_descendant_expanded_count"].tolist() == [0]
        assert info["useful_carry"].tolist() == [False]
    torch.testing.assert_close(
        carry_on[0]["tree_reps"], carry_off[0]["tree_reps"]
    )


def _run_scripted_burn_in(tree_carry):
    env = FakeVectorEnv(env_n=1)
    model = FakeModel(env.num_actions)
    flags = _flags(tree_carry=tree_carry)
    wrapper = cModelWrapper(
        env, env.env_n, flags, model, device=torch.device("cpu")
    )
    wrapper.reset(model, initial_action=torch.tensor([4]))

    # The first search edge expands the human-action child.  The second edge
    # expands one of its descendants, giving carry-on retained subtree state
    # that a fresh root cannot contain.
    _step(wrapper, model, [1], [util.PROCEED])
    _step(wrapper, model, [2], [util.PROCEED])
    pre_barrier_state, _reward, _done, _truncated, pre_barrier_info = _step(
        wrapper, model, [0], [util.STOP]
    )
    state, reward, done, truncated, info = _step(
        wrapper, model, [1], [util.PROCEED]
    )

    info_without_episode_stats = dict(info)
    info_without_episode_stats.pop("episode_return", None)
    env_out = util.create_env_out(
        (torch.tensor([1]), torch.tensor([util.PROCEED])),
        state,
        reward,
        done,
        truncated,
        info_without_episode_stats,
        flags,
    )
    result = {
        "env": env,
        "pre_state": pre_barrier_state,
        "pre_info": pre_barrier_info,
        "state": state,
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "info": info,
        "env_out": env_out,
    }
    wrapper.close()
    return result


def test_carry_on_off_changes_only_the_promoted_first_scored_root():
    carry_on = _run_scripted_burn_in(True)
    carry_off = _run_scripted_burn_in(False)

    # Before teacher-forcing the burn-in action, both searches and all inputs
    # visible to an actor are identical.
    torch.testing.assert_close(
        carry_on["pre_state"]["real_states"],
        carry_off["pre_state"]["real_states"],
    )
    torch.testing.assert_close(
        carry_on["pre_state"]["tree_reps"],
        carry_off["pre_state"]["tree_reps"],
    )
    for field in ("phase", "legal_control_mask", "search_steps"):
        torch.testing.assert_close(
            carry_on["pre_info"][field], carry_off["pre_info"][field]
        )

    # The recorded real transition and the previous-action token supplied to
    # the next actor call stay fixed.  Only the tree root differs.
    assert carry_on["env"].step_calls[-1].tolist() == [1]
    assert carry_off["env"].step_calls[-1].tolist() == [1]
    for field in ("reward", "done", "truncated"):
        torch.testing.assert_close(carry_on[field], carry_off[field])
    torch.testing.assert_close(
        carry_on["state"]["real_states"], carry_off["state"]["real_states"]
    )
    assert carry_on["info"]["executed_primary_action"].tolist() == [1]
    assert carry_off["info"]["executed_primary_action"].tolist() == [1]
    assert carry_on["env_out"].last_pri.tolist() == [[1]]
    assert carry_off["env_out"].last_pri.tolist() == [[1]]
    assert carry_on["info"]["root_carried"].tolist() == [True]
    assert carry_off["info"]["root_carried"].tolist() == [False]
    assert carry_on["info"]["carried_descendant_visit_count"].tolist() == [1]
    assert carry_on["info"]["carried_descendant_expanded_count"].tolist() == [1]
    assert carry_on["info"]["useful_carry"].tolist() == [True]
    assert carry_off["info"]["carried_descendant_visit_count"].tolist() == [0]
    assert carry_off["info"]["carried_descendant_expanded_count"].tolist() == [0]
    assert carry_off["info"]["useful_carry"].tolist() == [False]

    on_tree = carry_on["state"]["tree_reps"]
    off_tree = carry_off["state"]["tree_reps"]
    on_hash = hashlib.sha256(on_tree.numpy().tobytes()).hexdigest()
    off_hash = hashlib.sha256(off_tree.numpy().tobytes()).hexdigest()
    assert on_hash != off_hash
    root_ns = util.slice_dynamic_tree_reps(5)["root_ns"]
    assert on_tree[0, root_ns][2] > 0
    assert off_tree[0, root_ns][2] == 0
