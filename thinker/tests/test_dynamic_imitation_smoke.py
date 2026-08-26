from argparse import Namespace

from gymnasium import spaces
import numpy as np
import pytest
import torch

import smoke_dynamic_imitation as smoke
from thinker.learn_actor import _validate_model_state_dict_compatibility


def test_vector_actor_observation_space_adds_only_missing_batch_axes():
    template = spaces.Dict(
        {
            "real_states": spaces.Box(
                0, 255, shape=(12, 84, 84), dtype=np.uint8
            ),
            "xs": spaces.Box(0.0, 1.0, shape=(3, 84, 84), dtype=np.float32),
            "tree_reps": spaces.Box(
                -np.inf, np.inf, shape=(2, 104), dtype=np.float32
            ),
            "hs": spaces.Box(
                -np.inf, np.inf, shape=(2, 64, 6, 6), dtype=np.float32
            ),
        }
    )

    vector = smoke._vector_actor_observation_space(template, batch_size=2)

    assert vector["real_states"].shape == (2, 12, 84, 84)
    assert vector["xs"].shape == (2, 3, 84, 84)
    assert vector["tree_reps"].shape == (2, 104)
    assert vector["hs"].shape == (2, 64, 6, 6)


def test_smoke_state_dict_validation_reports_shape_and_key_errors():
    module = torch.nn.Linear(3, 2)
    valid = module.state_dict()
    smoke._validate_state_dict(module, valid, "test")

    invalid = {"weight": torch.zeros(4, 3), "extra": torch.zeros(1)}
    with pytest.raises(ValueError) as error:
        smoke._validate_state_dict(module, invalid, "test")
    message = str(error.value)
    assert "missing=['bias']" in message
    assert "unexpected=['extra']" in message
    assert "weight: incoming(4, 3) != expected(2, 3)" in message

    with pytest.raises(ValueError, match="shape_mismatch"):
        _validate_model_state_dict_compatibility(module, invalid, "test")


def test_fresh_smoke_flags_use_dynamic_20_20_20_protocol():
    args = Namespace(
        config=None,
        checkpoint_dir=None,
        env_name="Enduro-v5",
        rec_t=None,
        max_search_steps=None,
        max_depth=None,
        model_unroll_len=None,
        think_cost=None,
        model_size_nn=None,
        frame_stack_n=None,
        grayscale=None,
        tree_carry=None,
        scored_length=4,
        device=torch.device("cpu"),
    )

    flags = smoke._load_flags(args)

    assert flags.name == "Enduro-v5"
    assert flags.dynamic_search is True
    assert flags.sep_im_head is True
    assert flags.envpool is True
    assert (flags.rec_t, flags.max_search_steps, flags.max_depth) == (20, 20, 20)
    assert flags.model_unroll_len == 20
    assert flags.think_cost == pytest.approx(0.0005)
    assert flags.model_size_nn == 2
    assert flags.frame_stack_n == 4
    assert flags.batch_length == 4
