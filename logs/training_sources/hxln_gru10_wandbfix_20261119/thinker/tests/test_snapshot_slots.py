from argparse import Namespace

import gymnasium as gym
import numpy as np
import pytest

from thinker.gym_add.asyn_vector_env import AsyncVectorEnv
from thinker.gym_add.wrapper import TimeLimitExtended, VectorWrap


class SlotEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 100, shape=(1,), dtype=np.int64)
    action_space = gym.spaces.Discrete(2)

    def __init__(self):
        self.value = 0
        self.slots = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.value = 0
        return np.asarray([self.value]), {}

    def step(self, action):
        self.value += int(action) + 1
        return np.asarray([self.value]), 0.0, False, False, {}

    def quick_save(self, slot_id=0):
        self.slots[int(slot_id)] = self.value

    def quick_load(self, slot_id=0):
        slot_id = int(slot_id)
        if slot_id not in self.slots:
            raise ValueError(f"No state in slot {slot_id}")
        self.value = self.slots[slot_id]

    def quick_delete(self, slot_id=0):
        self.slots.pop(int(slot_id), None)

    def get_value(self):
        return self.value


class LegacySaveEnv(gym.Env):
    """Historical singleton snapshot interface with no slot argument."""

    observation_space = SlotEnv.observation_space
    action_space = SlotEnv.action_space

    def __init__(self):
        self.value = 0
        self.saved_value = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.value = 0
        return np.asarray([self.value]), {}

    def step(self, action):
        self.value += int(action) + 1
        return np.asarray([self.value]), 0.0, False, False, {}

    def quick_save(self):
        self.saved_value = self.value

    def quick_load(self):
        self.value = self.saved_value


def _make_async_env(env_n=3):
    return AsyncVectorEnv(
        [SlotEnv for _ in range(env_n)],
        shared_memory=False,
        context="fork",
    )


def test_named_snapshot_restores_wrapper_and_environment_state():
    base = SlotEnv()
    env = TimeLimitExtended(base, max_episode_steps=20)
    env.reset()
    env.step(0)
    env.quick_save(11)
    env.step(1)
    env.quick_save(29)
    env.step(1)

    env.quick_load(11)
    assert base.value == 1
    assert env._elapsed_steps == 1

    env.quick_load(29)
    assert base.value == 3
    assert env._elapsed_steps == 2

    env.quick_delete(11)
    assert 11 not in base.slots
    with pytest.raises(ValueError):
        env.quick_load(11)


def test_slot_zero_keeps_legacy_no_argument_snapshot_api():
    base = LegacySaveEnv()
    env = TimeLimitExtended(base, max_episode_steps=20)
    env.reset()
    env.step(0)
    env.quick_save()
    env.step(1)

    env.quick_load()

    assert base.value == 1
    assert env._elapsed_steps == 1


def test_async_zipped_slots_follow_env_id_order_and_keep_workers_isolated():
    env = _make_async_env(3)
    try:
        env.reset()
        env.step(np.asarray([0, 1, 1]))  # values: [1, 2, 2]
        env.quick_save_each(env_id=[2, 0], slot_ids=[17, 31])
        env.step(np.asarray([1, 1, 1]))  # values: [3, 4, 4]

        env.quick_load_each(env_id=[0, 2], slot_ids=[31, 17])
        assert list(env.call("get_value", env_id=[0, 2])) == [1, 2]

        # The same numeric slot is local to each worker, not a batch-global
        # snapshot that can collide between envs.
        env.quick_save_each(env_id=[0, 1], slot_ids=[5, 5])
        env.step(np.asarray([1, 1, 1]))
        env.quick_load_each(env_id=[0, 1], slot_ids=[5, 5])
        assert list(env.call("get_value", env_id=[0, 1])) == [1, 4]

        env.quick_delete_each(env_id=[0], slot_ids=[5])
        env.step(np.asarray([0, 0, 0]))
        env.quick_load_each(env_id=[1], slot_ids=[5])
        assert list(env.call("get_value", env_id=[1])) == [4]

        # Legacy all-worker calls still address default slot zero.
        before = list(env.call("get_value"))
        env.quick_save()
        env.step(np.asarray([1, 1, 1]))
        env.quick_load()
        assert list(env.call("get_value")) == before

        with pytest.raises(ValueError, match="duplicates"):
            env.quick_save_each(env_id=[0, 0], slot_ids=[1, 2])
        with pytest.raises(ValueError, match="selected environment count"):
            env.quick_save_each(env_id=[0, 1], slot_ids=[1])
    finally:
        env.close()


def test_vector_slot_stats_survive_partial_same_slot_deletion():
    base = _make_async_env(2)
    flags = Namespace(
        obs_clip=-1,
        reward_clip=-1,
        obs_norm=False,
        reward_norm=False,
    )
    env = VectorWrap(base, flags)
    try:
        env.reset(reset_stat=True)
        env.step(np.asarray([0, 1]))  # worker values: [1, 2]
        env.episode_return[:] = [10.0, 20.0]
        env.episode_step[:] = [3, 7]
        env.quick_save_slots(env_id=[0, 1], slot_ids=[5, 5])

        env.step(np.asarray([1, 1]))
        env.episode_return[:] = [-1.0, -1.0]
        env.episode_step[:] = [-1, -1]
        env.quick_delete_slots(env_id=[0], slot_ids=[5])

        assert env.save_states[5]["valid"].tolist() == [False, True]
        env.quick_load_slots(env_id=[1], slot_ids=[5])
        assert env.episode_return[1] == 20.0
        assert env.episode_step[1] == 7
        assert list(base.call("get_value", env_id=[1])) == [2]

        with pytest.raises(ValueError, match="env_id.*0.*slot 5"):
            env.quick_load_slots(env_id=[0], slot_ids=[5])

        # The fixed path's unnamed singleton snapshot remains full-batch and
        # restores both vector statistics and worker state.
        env.episode_return[:] = [30.0, 40.0]
        env.episode_step[:] = [8, 9]
        before = list(base.call("get_value"))
        env.quick_save()
        env.step(np.asarray([1, 1]))
        env.episode_return[:] = [0.0, 0.0]
        env.episode_step[:] = [0, 0]
        env.quick_load()
        assert env.episode_return.tolist() == [30.0, 40.0]
        assert env.episode_step.tolist() == [8, 9]
        assert list(base.call("get_value")) == before
    finally:
        env.close()
