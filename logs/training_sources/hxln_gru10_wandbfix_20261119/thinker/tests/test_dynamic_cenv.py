import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch
from gymnasium import spaces

from thinker import util
from thinker.cenv import cModelWrapper, cPerfectWrapper


class FakeVectorEnv:
    def __init__(self, env_n=3, num_actions=5, truncate_on_step=False):
        self.env_n = env_n
        self.num_actions = num_actions
        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(num_actions) for _ in range(env_n))
        )
        self.observation_space = spaces.Box(
            -100, 100, shape=(env_n, 2), dtype=np.float32
        )
        self.reward_range = (-float("inf"), float("inf"))
        self.metadata = {}
        self.values = np.zeros((env_n, 2), dtype=np.float32)
        self.step_calls = []
        self.step_pre_values = []
        self.snapshot_slots = [dict() for _ in range(env_n)]
        self.truncate_on_step = truncate_on_step
        self.reset_value = 0.0
        self.close_calls = 0

    def reset(self, reset_stat=True, seed=None):
        self.values.fill(self.reset_value)
        return self.values.copy(), self._info(self.env_n)

    def step(self, actions, env_id=None):
        env_id = list(range(self.env_n)) if env_id is None else list(env_id)
        actions = np.asarray(actions, dtype=np.int32)
        self.step_calls.append(actions.copy())
        self.step_pre_values.append(self.values[env_id, 0].copy())
        self.values[env_id, 0] += actions + 1
        reward = actions.astype(np.float32)
        done = np.zeros(len(env_id), dtype=np.bool_)
        truncated = np.zeros(len(env_id), dtype=np.bool_)
        if self.truncate_on_step:
            truncated[:] = True
        return (
            self.values[env_id].copy(), reward, done, truncated,
            self._info(len(env_id)),
        )

    def quick_save(self, env_id=None, slot_id=0):
        env_id = list(range(self.env_n)) if env_id is None else list(env_id)
        for index in env_id:
            self.snapshot_slots[index][int(slot_id)] = self.values[index].copy()

    def quick_load(self, env_id=None, slot_id=0):
        env_id = list(range(self.env_n)) if env_id is None else list(env_id)
        for index in env_id:
            self.values[index] = self.snapshot_slots[index][int(slot_id)]

    def quick_delete(self, env_id=None, slot_id=0):
        env_id = list(range(self.env_n)) if env_id is None else list(env_id)
        for index in env_id:
            self.snapshot_slots[index].pop(int(slot_id), None)

    def quick_save_slots(self, env_id, slot_ids):
        for index, slot_id in zip(env_id, slot_ids):
            self.snapshot_slots[index][int(slot_id)] = self.values[index].copy()

    def quick_load_slots(self, env_id, slot_ids):
        for index, slot_id in zip(env_id, slot_ids):
            self.values[index] = self.snapshot_slots[index][int(slot_id)]

    def quick_delete_slots(self, env_id, slot_ids):
        for index, slot_id in zip(env_id, slot_ids):
            self.snapshot_slots[index].pop(int(slot_id), None)

    @staticmethod
    def _info(n):
        return {
            "real_done": np.zeros(n, dtype=np.bool_),
            "episode_return": np.zeros(n, dtype=np.float32),
            "episode_step": np.zeros(n, dtype=np.int64),
        }

    def close(self):
        self.close_calls += 1


class FakeModel:
    hidden_shape = (1,)

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def initial_state(self, batch_size, device=None):
        return {"core": torch.zeros(batch_size, 1, device=device)}

    def __call__(self, env_state, done, actions, state):
        batch_size = env_state.shape[0]
        device = env_state.device
        return SimpleNamespace(
            vs=torch.zeros(1, batch_size, 1, device=device),
            policy=torch.zeros(
                1, batch_size, 1, self.num_actions, device=device
            ),
            xs=None,
            hs=None,
            state={"core": env_state[:, :1].float()},
        )

    def forward_single(self, state, action):
        batch_size = action.shape[0]
        device = action.device
        return SimpleNamespace(
            rs=action.float().view(1, batch_size, 1),
            vs=torch.zeros(1, batch_size, 1, device=device),
            policy=torch.zeros(
                1, batch_size, 1, self.num_actions, device=device
            ),
            xs=None,
            hs=None,
            dones=torch.zeros(1, batch_size, 1, dtype=torch.bool, device=device),
            state={"core": state["core"] + action.float()},
        )


class TerminalFakeModel(FakeModel):
    def __init__(self, num_actions):
        super().__init__(num_actions)
        self.forward_single_calls = 0

    def forward_single(self, state, action):
        out = super().forward_single(state, action)
        self.forward_single_calls += 1
        out.dones = torch.ones_like(out.dones)
        return out


class LifetimeTrackingModel(FakeModel):
    """Expose whether a root Node still owns its reset observation tensor."""

    def __init__(self, num_actions):
        super().__init__(num_actions)
        self.last_env_state_ref = None

    def __call__(self, env_state, done, actions, state):
        self.last_env_state_ref = weakref.ref(env_state)
        out = super().__call__(env_state, done, actions, state)
        # Do not retain env_state through the fake recurrent state; the root
        # Node's encoded real-state view should be its only remaining owner.
        out.state = {
            "core": torch.zeros(env_state.shape[0], 1, device=env_state.device)
        }
        return out


def _flags(*, cap=-1, max_depth=40, tree_carry=True, wrapper_type=0):
    return util.create_setting(
        args=[
            "--dynamic_search", "true",
            "--wrapper_type", str(wrapper_type),
            "--max_search_steps", str(cap),
            "--max_depth", str(max_depth),
            "--tree_carry", str(tree_carry).lower(),
            "--see_h", "false",
            "--see_x", "false",
            "--model_done_loss_cost", "0",
            "--parallel", "false",
        ],
        save_flags=False,
    )


def _fixed_flags(*, rec_t=3, max_depth=40, wrapper_type=0):
    return util.create_setting(
        args=[
            "--dynamic_search", "false",
            "--wrapper_type", str(wrapper_type),
            "--rec_t", str(rec_t),
            "--test_rec_t", "-1",
            "--max_depth", str(max_depth),
            "--see_h", "false",
            "--see_x", "false",
            "--model_done_loss_cost", "0",
            "--parallel", "false",
        ],
        save_flags=False,
    )


def _wrapper(*, cap=-1, max_depth=40, truncate_on_step=False):
    env = FakeVectorEnv(truncate_on_step=truncate_on_step)
    model = FakeModel(env.num_actions)
    wrapper = cModelWrapper(
        env, env.env_n, _flags(cap=cap, max_depth=max_depth), model,
        device=torch.device("cpu"),
    )
    state, info = wrapper.reset(model)
    return wrapper, env, model, state, info


def _perfect_wrapper(env_n=1):
    env = FakeVectorEnv(env_n=env_n, num_actions=5)
    model = FakeModel(env.num_actions)
    wrapper = cPerfectWrapper(
        env,
        env.env_n,
        _flags(wrapper_type=2, tree_carry=True),
        model,
        device=torch.device("cpu"),
    )
    state, info = wrapper.reset(model)
    return wrapper, env, model, state, info


def _step(wrapper, model, primary, controls):
    return wrapper.step(
        (torch.tensor(primary), torch.tensor(controls)), model
    )


def test_fixed_mode_keeps_binary_control_and_rec_t_schedule():
    env = FakeVectorEnv()
    model = FakeModel(env.num_actions)
    flags = _fixed_flags(rec_t=3)
    wrapper = cModelWrapper(
        env, env.env_n, flags, model, device=torch.device("cpu")
    )
    _state, info = wrapper.reset(model)

    assert wrapper.action_space[1][0].n == 2
    assert info["step_status"].tolist() == [0, 0, 0]

    for expected_status in [1, 2]:
        _state, reward, done, truncated, info = _step(
            wrapper, model, [0, 1, 2], [0, 0, 0]
        )
        assert info["step_status"].tolist() == [expected_status] * 3
        assert reward.tolist() == [0.0, 0.0, 0.0]
        assert len(env.step_calls) == 0

    _state, reward, done, truncated, info = _step(
        wrapper, model, [4, 3, 2], [0, 0, 0]
    )
    assert len(env.step_calls) == 1
    assert env.step_calls[0].tolist() == [4, 3, 2]
    assert info["step_status"].tolist() == [0, 0, 0]
    assert reward.tolist() == [4.0, 3.0, 2.0]


def test_independent_stop_times_release_cached_real_actions_once():
    wrapper, env, model, state, info = _wrapper()
    root0 = state["tree_reps"][0].clone()

    state, reward, done, truncated, info = _step(
        wrapper, model, [0, 1, 2], [util.STOP, util.PROCEED, util.PROCEED]
    )
    assert info["phase"].tolist() == [
        util.NEED_REAL_ACTION_PHASE, util.SEARCH_PHASE, util.SEARCH_PHASE
    ]
    assert info["think_reward"].tolist() == [0.0, -1.0, -1.0]
    assert len(env.step_calls) == 0

    state, *_rest, info = _step(
        wrapper, model, [4, 0, 1], [0, util.STOP, util.PROCEED]
    )
    waiting_tree = state["tree_reps"][0].clone()
    assert info["stored_action_mask"].tolist() == [True, False, False]

    state, *_rest, info = _step(
        wrapper, model, [0, 3, 0], [0, 0, util.STOP]
    )
    assert torch.equal(state["tree_reps"][0], waiting_tree)
    assert info["stored_action_mask"].tolist() == [False, True, False]
    assert len(env.step_calls) == 0

    state, reward, done, truncated, info = _step(
        wrapper, model, [0, 0, 2], [0, 0, 0]
    )
    assert len(env.step_calls) == 1
    assert env.step_calls[0].tolist() == [4, 3, 2]
    assert info["real_transition"].all()
    assert info["executed_primary_action"].tolist() == [4, 3, 2]
    assert info["phase"].tolist() == [util.SEARCH_PHASE] * 3
    assert info["search_steps"].tolist() == [0, 0, 0]
    assert info["tree_token_valid"].all()
    assert info["search_state_reset"].all()
    assert reward.tolist() == [4.0, 3.0, 2.0]
    assert not torch.equal(state["tree_reps"][0], root0)


def test_positive_cap_ends_stage_on_nth_planning_action_without_extra_stop():
    wrapper, env, model, _state, _info = _wrapper(cap=1)
    _state, reward, _done, _truncated, info = _step(
        wrapper, model, [0, 1, 2], [util.PROCEED] * 3
    )

    assert info["phase"].tolist() == [util.NEED_REAL_ACTION_PHASE] * 3
    assert info["stage_end"].all()
    assert info["forced_stop"].all()
    assert info["search_steps"].tolist() == [1, 1, 1]
    assert info["think_reward"].tolist() == [-1.0, -1.0, -1.0]
    assert not info["legal_control_mask"].any()
    assert len(env.step_calls) == 0


def test_max_depth_masks_and_rejects_proceed_but_allows_reset():
    wrapper, _env, model, _state, info = _wrapper(max_depth=1)
    assert info["legal_control_mask"].tolist() == [
        [False, True, True], [False, True, True], [False, True, True]
    ]
    with pytest.raises(AssertionError):
        _step(wrapper, model, [0, 0, 0], [util.PROCEED] * 3)

    wrapper, _env, model, _state, _info = _wrapper(max_depth=1)
    state, _reward, _done, _truncated, info = _step(
        wrapper, model, [0, 1, 2], [util.RESET] * 3
    )
    tree_reset = util.slice_dynamic_tree_reps(5)["tree_reset"]
    assert torch.all(state["tree_reps"][:, tree_reset] == 1)
    assert info["think_reward"].tolist() == [-1.0, -1.0, -1.0]


def test_real_truncation_is_an_authoritative_terminal():
    wrapper, env, model, _state, _info = _wrapper(truncate_on_step=True)
    _step(wrapper, model, [0, 0, 0], [util.STOP] * 3)
    _state, reward, done, truncated, info = _step(
        wrapper, model, [1, 2, 3], [0, 0, 0]
    )

    assert info["real_transition"].all()
    assert truncated.all()
    assert (done | truncated).all()
    assert len(env.step_calls) == 1
    np.testing.assert_array_equal(env.step_calls[0], np.array([1, 2, 3]))


def test_imagined_terminal_stays_inside_search_and_stop_is_non_mutating():
    env = FakeVectorEnv()
    model = TerminalFakeModel(env.num_actions)
    flags = _flags()
    flags.model_done_loss_cost = 1.0
    wrapper = cModelWrapper(
        env, env.env_n, flags, model, device=torch.device("cpu")
    )
    wrapper.reset(model)

    state, reward, done, truncated, info = _step(
        wrapper, model, [0, 1, 2], [util.PROCEED] * 3
    )
    cur_done = util.slice_dynamic_tree_reps(env.num_actions)["cur_d"]
    assert torch.all(state["tree_reps"][:, cur_done] == 1)
    assert info["phase"].tolist() == [util.SEARCH_PHASE] * env.env_n
    assert model.forward_single_calls == 1
    assert len(env.step_calls) == 0

    terminal_tree = state["tree_reps"].clone()
    state, reward, done, truncated, info = _step(
        wrapper, model, [4, 4, 4], [util.STOP] * 3
    )
    assert torch.equal(state["tree_reps"], terminal_tree)
    assert info["phase"].tolist() == [util.NEED_REAL_ACTION_PHASE] * env.env_n
    assert model.forward_single_calls == 1
    assert len(env.step_calls) == 0


def test_perfect_tree_carry_preserves_descendant_snapshot_slots():
    wrapper, env, model, _state, _info = _perfect_wrapper()

    # Build root --0--> child --1--> grandchild using perfect snapshots.
    _step(wrapper, model, [0], [util.PROCEED])
    _step(wrapper, model, [1], [util.PROCEED])
    _step(wrapper, model, [0], [util.STOP])
    _state, _reward, _done, _truncated, carry_info = _step(
        wrapper, model, [0], [util.PROCEED]
    )  # real action; carries child 0
    assert carry_info["root_carried"].tolist() == [True]
    assert carry_info["carried_descendant_visit_count"].tolist() == [1]
    assert carry_info["carried_descendant_expanded_count"].tolist() == [1]
    assert carry_info["useful_carry"].tolist() == [True]

    # Revisit the retained grandchild (no simulator step), then expand from it.
    step_calls_before_revisit = len(env.step_calls)
    _step(wrapper, model, [1], [util.PROCEED])
    assert len(env.step_calls) == step_calls_before_revisit
    _step(wrapper, model, [0], [util.PROCEED])

    # The final expansion must restore the grandchild snapshot whose state was
    # 3, not the post-real root (1) or any unrelated simulator state.
    assert env.step_pre_values[-1].tolist() == [3.0]


def test_learned_and_perfect_dynamic_wrappers_match_public_transitions():
    def make(wrapper_cls, wrapper_type):
        env = FakeVectorEnv(env_n=2)
        model = FakeModel(env.num_actions)
        flags = _flags(wrapper_type=wrapper_type, tree_carry=False)
        wrapper = wrapper_cls(
            env, env.env_n, flags, model, device=torch.device("cpu")
        )
        state, info = wrapper.reset(model)
        return wrapper, model, state, info

    learned = make(cModelWrapper, 0)
    perfect = make(cPerfectWrapper, 2)
    torch.testing.assert_close(
        learned[2]["tree_reps"], perfect[2]["tree_reps"]
    )

    sequence = [
        ([0, 1], [util.PROCEED, util.PROCEED]),
        ([2, 3], [util.RESET, util.RESET]),
        ([0, 0], [util.STOP, util.STOP]),
        ([4, 2], [util.PROCEED, util.PROCEED]),
    ]
    event_fields = [
        "phase", "legal_control_mask", "tree_token_valid",
        "search_state_reset", "real_transition", "stage_end",
        "forced_stop", "stored_action_mask", "executed_primary_action",
        "accepted_primary_action", "accepted_control", "search_steps",
        "think_reward",
    ]
    for primary, control in sequence:
        learned_out = _step(learned[0], learned[1], primary, control)
        perfect_out = _step(perfect[0], perfect[1], primary, control)
        torch.testing.assert_close(
            learned_out[0]["tree_reps"], perfect_out[0]["tree_reps"]
        )
        for index in [1, 2, 3]:
            torch.testing.assert_close(learned_out[index], perfect_out[index])
        for field in event_fields:
            torch.testing.assert_close(
                learned_out[4][field], perfect_out[4][field]
            )


@pytest.mark.parametrize(
    ("wrapper_cls", "wrapper_type"),
    [(cModelWrapper, 0), (cPerfectWrapper, 2)],
)
def test_repeated_reset_replaces_the_active_tree(wrapper_cls, wrapper_type):
    env = FakeVectorEnv(env_n=2)
    model = FakeModel(env.num_actions)
    wrapper = wrapper_cls(
        env,
        env.env_n,
        _flags(wrapper_type=wrapper_type),
        model,
        device=torch.device("cpu"),
    )
    wrapper.reset(model)
    _step(wrapper, model, [0, 1], [util.PROCEED, util.PROCEED])
    if wrapper_type == 2:
        assert all(1 in slots for slots in env.snapshot_slots)

    env.reset_value = 9.0
    state, info = wrapper.reset(model)
    assert state["real_states"][:, 0].tolist() == [9.0, 9.0]
    assert info["phase"].tolist() == [util.SEARCH_PHASE] * 2
    tree_reset = util.slice_dynamic_tree_reps(env.num_actions)["tree_reset"]
    assert torch.all(state["tree_reps"][:, tree_reset] == 1)
    if wrapper_type == 2:
        assert all(set(slots) == {0} for slots in env.snapshot_slots)

    _step(wrapper, model, [0, 0], [util.STOP, util.STOP])
    state, _reward, _done, _truncated, info = _step(
        wrapper, model, [1, 2], [util.PROCEED, util.PROCEED]
    )
    assert env.step_pre_values[-1].tolist() == [9.0, 9.0]
    assert info["real_transition"].all()
    assert torch.all(state["tree_reps"][:, tree_reset] == 1)
    wrapper.close()


@pytest.mark.parametrize(
    ("wrapper_cls", "wrapper_type", "dynamic"),
    [
        (cModelWrapper, 0, False),
        (cPerfectWrapper, 2, False),
        (cModelWrapper, 0, True),
        (cPerfectWrapper, 2, True),
    ],
)
def test_close_and_dealloc_release_node_owned_tensors(
    wrapper_cls, wrapper_type, dynamic
):
    flags = (
        _flags(wrapper_type=wrapper_type)
        if dynamic
        else _fixed_flags(wrapper_type=wrapper_type)
    )
    env = FakeVectorEnv(env_n=1)
    model = LifetimeTrackingModel(env.num_actions)
    wrapper = wrapper_cls(
        env,
        env.env_n,
        flags,
        model,
        device=torch.device("cpu"),
    )
    state, info = wrapper.reset(model)
    encoded_owner_ref = model.last_env_state_ref
    assert encoded_owner_ref() is not None
    del state, info

    wrapper.close()
    gc.collect()
    assert encoded_owner_ref() is None
    assert env.close_calls == 1
    wrapper.close()
    assert env.close_calls == 1

    env_2 = FakeVectorEnv(env_n=1)
    model_2 = LifetimeTrackingModel(env_2.num_actions)
    flags_2 = (
        _flags(wrapper_type=wrapper_type)
        if dynamic
        else _fixed_flags(wrapper_type=wrapper_type)
    )
    wrapper_2 = wrapper_cls(
        env_2,
        env_2.env_n,
        flags_2,
        model_2,
        device=torch.device("cpu"),
    )
    state_2, info_2 = wrapper_2.reset(model_2)
    dealloc_owner_ref = model_2.last_env_state_ref
    assert dealloc_owner_ref() is not None
    del state_2, info_2, wrapper_2
    gc.collect()
    assert dealloc_owner_ref() is None


@pytest.mark.parametrize(
    ("wrapper_cls", "wrapper_type"),
    [(cModelWrapper, 0), (cPerfectWrapper, 2)],
)
def test_fixed_repeated_reset_replaces_the_active_tree(wrapper_cls, wrapper_type):
    env = FakeVectorEnv(env_n=2)
    model = FakeModel(env.num_actions)
    wrapper = wrapper_cls(
        env,
        env.env_n,
        _fixed_flags(rec_t=2, wrapper_type=wrapper_type),
        model,
        device=torch.device("cpu"),
    )
    wrapper.reset(model)
    _step(wrapper, model, [0, 1], [0, 0])

    env.reset_value = 9.0
    state, info = wrapper.reset(model)
    assert state["real_states"][:, 0].tolist() == [9.0, 9.0]
    assert wrapper.action_space[1][0].n == 2
    assert info["step_status"].tolist() == [0, 0]

    _step(wrapper, model, [0, 0], [0, 0])
    _state, reward, _done, _truncated, info = _step(
        wrapper, model, [1, 2], [0, 0]
    )
    if wrapper_type == 2:
        # The legacy perfect wrapper has one mutable snapshot per env: the
        # imagination call starts at the new reset state, then the real call
        # continues from that saved imagined state.
        assert env.step_pre_values[-2].tolist() == [9.0, 9.0]
    else:
        assert env.step_pre_values[-1].tolist() == [9.0, 9.0]
    assert reward.tolist() == [1.0, 2.0]
    assert info["step_status"].tolist() == [0, 0]
    wrapper.close()
