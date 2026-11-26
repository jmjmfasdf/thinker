from types import SimpleNamespace

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

from thinker import util
from thinker.actor_net import ActorBaseNet
from thinker.gym_add.wrapper import EnvPoolWrap, create_envpool
from thinker.main import (
    _runtime_primary_action_meanings,
    _validate_online_env_contract,
)
from thinker.model_net import ModelNet


class _SpaceOnlyEnv(gym.Env):
    metadata = {}

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space


@pytest.mark.parametrize(
    ("name", "action_n"),
    (("Enduro-v5", 9), ("Pong-v5", 6)),
)
def test_envpool_preserves_atari_frame_stack_metadata(name, action_n):
    pytest.importorskip("envpool")
    flags = SimpleNamespace(grayscale=False, frame_stack_n=4)
    env = create_envpool(name, flags, env_n=2)
    try:
        observation, _ = env.reset()
        assert observation.shape == (2, 12, 84, 84)
        assert observation.dtype == np.uint8
        assert env.single_observation_space.shape == (12, 84, 84)
        assert isinstance(env.single_action_space, spaces.Discrete)
        assert env.single_action_space.n == action_n
        assert env.frame_stack_n == 4
        meanings = _runtime_primary_action_meanings(
            env,
            env.single_action_space,
            name=name,
            envpool=True,
        )
        assert len(meanings) == action_n
        assert meanings.count("NOOP") == 1
        assert _validate_online_env_contract(
            env.single_observation_space,
            env.single_action_space,
            env.frame_stack_n,
            expected_frame_stack_n=flags.frame_stack_n,
            require_discrete=True,
        ) == 3
    finally:
        env.close()


def test_vector_runtime_action_tables_must_agree_and_receive_no_env_kwargs():
    class _ActionTableVector:
        def __init__(self, tables):
            self.tables = tables
            self.calls = []

        def call(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.tables

    vector = _ActionTableVector(
        [("FIRE", "NOOP", "LEFT"), ("FIRE", "NOOP", "LEFT")]
    )
    meanings = _runtime_primary_action_meanings(
        vector, spaces.Discrete(3), envpool=False
    )
    assert meanings == ("FIRE", "NOOP", "LEFT")
    assert vector.calls == [(("get_action_meanings",), {})]

    disagreeing = _ActionTableVector(
        [("FIRE", "NOOP"), ("NOOP", "FIRE")]
    )
    with pytest.raises(ValueError, match="disagree"):
        _runtime_primary_action_meanings(
            disagreeing, spaces.Discrete(2), envpool=False
        )


def test_envpool_wrapper_requires_explicit_valid_stack_metadata():
    valid_obs = spaces.Box(0, 255, shape=(12, 84, 84), dtype=np.uint8)
    with pytest.raises(TypeError, match="frame_stack_n"):
        EnvPoolWrap(
            _SpaceOnlyEnv(valid_obs, spaces.Discrete(9)),
            num_envs=2,
        )

    invalid_obs = spaces.Box(0, 255, shape=(10, 84, 84), dtype=np.uint8)
    with pytest.raises(ValueError, match="divisible"):
        EnvPoolWrap(
            _SpaceOnlyEnv(invalid_obs, spaces.Discrete(9)),
            num_envs=2,
            frame_stack_n=4,
        )

    with pytest.raises(TypeError, match="Discrete"):
        EnvPoolWrap(
            _SpaceOnlyEnv(
                valid_obs,
                spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            ),
            num_envs=2,
            frame_stack_n=4,
        )


def test_online_contract_fails_on_stack_or_action_mismatch():
    obs_space = spaces.Box(0, 255, shape=(12, 84, 84), dtype=np.uint8)
    with pytest.raises(ValueError, match="disagrees with flags"):
        _validate_online_env_contract(
            obs_space,
            spaces.Discrete(9),
            1,
            expected_frame_stack_n=4,
            require_discrete=True,
        )

    with pytest.raises(TypeError, match="Discrete"):
        _validate_online_env_contract(
            obs_space,
            spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            4,
            expected_frame_stack_n=4,
            require_discrete=True,
        )


def test_model_net_preserves_and_validates_frame_stack_metadata():
    flags = util.create_flags("default_thinker.yaml", save_flags=False)
    obs_space = spaces.Box(0, 255, shape=(12, 84, 84), dtype=np.uint8)
    model = ModelNet(
        obs_space=obs_space,
        action_space=spaces.Discrete(9),
        flags=flags,
        frame_stack_n=4,
    )
    assert model.obs_shape == (12, 84, 84)
    assert model.obs_dtype == np.dtype(np.uint8)
    assert np.array_equal(model.obs_low, obs_space.low)
    assert np.array_equal(model.obs_high, obs_space.high)
    assert model.frame_stack_n == 4
    assert model.frame_ch == 3
    assert model.sr_net.frame_stack_n == 4

    invalid_obs = spaces.Box(0, 255, shape=(10, 84, 84), dtype=np.uint8)
    with pytest.raises(ValueError, match="divisible"):
        ModelNet(
            obs_space=invalid_obs,
            action_space=spaces.Discrete(9),
            flags=flags,
            frame_stack_n=4,
        )


def test_actor_retains_single_online_observation_box_contract():
    flags = util.create_setting(
        args=["--wrapper_type", "1", "--parallel", "false", "--float16", "false"],
        save_flags=False,
    )
    single = spaces.Box(0, 255, shape=(12, 84, 84), dtype=np.uint8)
    vector = spaces.Box(
        low=np.broadcast_to(single.low, (2,) + single.shape),
        high=np.broadcast_to(single.high, (2,) + single.shape),
        dtype=single.dtype,
    )
    actor = ActorBaseNet(
        obs_space=spaces.Dict({"real_states": vector}),
        action_space=spaces.Tuple((spaces.Discrete(9),)),
        flags=flags,
    )

    retained = actor.online_real_state_space
    assert retained.shape == single.shape
    assert retained.dtype == single.dtype
    assert np.array_equal(retained.low, single.low)
    assert np.array_equal(retained.high, single.high)
