from argparse import Namespace

import pytest
import torch

from thinker import util


def _flags(**overrides):
    values = dict(
        wrapper_type=0,
        dynamic_search=True,
        max_search_steps=-1,
        reset_mode=0,
        rec_t=40,
        has_action_seq=True,
        im_cost=1.0,
        cur_cost=0.0,
        think_cost=0.002,
        think_cost_anneal=False,
        dynamic_search_hidden_dim=100,
        dual_net=True,
        cur_enable=False,
        model_rs_loss_cost=1.0,
        model_img_loss_cost=0.0,
        model_done_loss_cost=1.0,
    )
    values.update(overrides)
    return Namespace(**values)


def test_dynamic_public_enum_values_are_stable():
    assert (util.PROCEED, util.RESET, util.STOP) == (0, 1, 2)
    assert (
        util.SEARCH_PHASE,
        util.NEED_REAL_ACTION_PHASE,
        util.WAIT_PHASE,
    ) == (0, 1, 2)


@pytest.mark.parametrize("num_actions", [2, 5, 11])
def test_dynamic_tree_schema_is_budget_independent(num_actions):
    expected_width = 10 * num_actions + 14
    schemas = []
    for max_search_steps, max_depth in [(-1, 5), (8, 40), (40, 100)]:
        flags = _flags(max_search_steps=max_search_steps, max_depth=max_depth)
        util.process_flags(flags)
        schema = util.get_tree_rep_meaning(num_actions, 1, flags)
        schemas.append(schema)
        assert schema["search_start"].stop == expected_width
    assert schemas[0] == schemas[1] == schemas[2]


def test_dynamic_reward_channel_is_appended():
    flags = _flags(im_cost=1.0, cur_cost=1.0)
    assert util.get_reward_names(flags) == ["re", "im", "cur", "think"]


def test_dynamic_budget_stats_use_only_completed_search_stages():
    search_steps = torch.tensor([
        [0, 99, 2],
        [4, 7, 6],
    ])
    stage_end = torch.tensor([
        [True, False, True],
        [True, False, True],
    ])

    stats = util.get_search_budget_stats(search_steps, stage_end)

    # The unfinished 99/7 stages do not contribute.  A zero-step STOP does.
    assert stats["max_budget"] == 6.0
    assert stats["mean_budget"] == 3.0
    assert stats["search/mean_steps"] == stats["mean_budget"]
    assert stats["search/median_steps"] == 2.0
    assert stats["search/p95_steps"] == pytest.approx(5.7)


def test_dynamic_budget_stats_are_zero_without_completed_stage():
    stats = util.get_search_budget_stats(
        torch.tensor([[8, 13]]), torch.tensor([[False, False]])
    )

    assert stats["max_budget"] == 0.0
    assert stats["mean_budget"] == 0.0


@pytest.mark.parametrize("bad_cap", [0, -2])
def test_dynamic_rejects_invalid_search_cap(bad_cap):
    with pytest.raises(ValueError):
        util.process_flags(_flags(max_search_steps=bad_cap))


def test_dynamic_requires_reset_mode_zero():
    with pytest.raises(ValueError):
        util.process_flags(_flags(reset_mode=1))


@pytest.mark.parametrize("wrapper_type", [1, 3, 4])
def test_dynamic_rejects_unsupported_wrapper_types(wrapper_type):
    with pytest.raises(ValueError):
        util.process_flags(_flags(wrapper_type=wrapper_type))


def test_dynamic_rejects_mcts_actor():
    with pytest.raises(ValueError):
        util.process_flags_actor(_flags(drc=False, mcts=True))


def test_fixed_mode_keeps_action_sequence_setting():
    flags = _flags(dynamic_search=False, has_action_seq=True)
    util.process_flags(flags)
    assert flags.has_action_seq is True


def test_initial_env_out_uses_wrapper_phase_and_masks_over_defaults():
    flags = _flags()
    state = {
        "real_states": torch.zeros(2, 1),
        "phase": torch.tensor([util.WAIT_PHASE, util.SEARCH_PHASE]),
        "tree_token_valid": torch.tensor([False, True]),
    }
    info = {
        "legal_control_mask": torch.tensor(
            [[False, False, False], [True, True, True]]
        ),
        "search_state_reset": torch.tensor([False, True]),
    }

    env_out = util.init_env_out(
        state, info, flags, dim_actions=1, tuple_action=False
    )

    assert env_out.phase.tolist() == [[util.WAIT_PHASE, util.SEARCH_PHASE]]
    assert env_out.tree_token_valid.tolist() == [[False, True]]
    assert env_out.legal_control_mask.tolist() == [
        [[False, False, False], [True, True, True]]
    ]


def test_dynamic_decoder_uses_budget_independent_schema():
    num_actions = 5
    width = 10 * num_actions + 14
    token = torch.arange(width, dtype=torch.float32).view(1, width)

    decoded = util.decode_dynamic_tree_reps(token, num_actions)

    assert decoded["search_start"].item() == width - 1
    assert torch.equal(decoded["cur_reset"], decoded["tree_reset"])
