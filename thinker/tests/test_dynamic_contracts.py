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


def test_factorized_control_legacy_default_is_disabled():
    flags = _flags()
    util.process_flags(flags)
    assert flags.dynamic_factorized_control is False


def test_factorized_control_opt_in_requires_dynamic_search():
    with pytest.raises(ValueError, match="requires dynamic_search=true"):
        util.process_flags(_flags(
            dynamic_search=False, dynamic_factorized_control=True
        ))


@pytest.mark.parametrize("bad_value", [1, 0, "true", None])
def test_factorized_control_rejects_non_boolean_values(bad_value):
    with pytest.raises(ValueError, match="must be boolean"):
        util.process_flags(_flags(dynamic_factorized_control=bad_value))


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
    assert stats["search/budget_bin_0_count"] == 1
    assert stats["search/budget_bin_0_fraction"] == pytest.approx(0.25)
    assert stats["search/budget_bin_1_count"] == 0
    assert stats["search/budget_bin_1_fraction"] == 0.0
    assert stats["search/budget_bin_2_3_count"] == 1
    assert stats["search/budget_bin_2_3_fraction"] == pytest.approx(0.25)
    assert stats["search/budget_bin_4_7_count"] == 2
    assert stats["search/budget_bin_4_7_fraction"] == pytest.approx(0.5)
    assert stats["search/budget_bin_8_15_count"] == 0
    assert stats["search/budget_bin_16_cap_count"] == 0


def test_dynamic_budget_stats_are_zero_without_completed_stage():
    stats = util.get_search_budget_stats(
        torch.tensor([[8, 13]]), torch.tensor([[False, False]])
    )

    assert stats["max_budget"] == 0.0
    assert stats["mean_budget"] == 0.0
    for label in ("0", "1", "2_3", "4_7", "8_15", "16_cap"):
        assert stats[f"search/budget_bin_{label}_count"] == 0
        assert stats[f"search/budget_bin_{label}_fraction"] == 0.0


def test_dynamic_depth_stop_stats_use_pre_decision_depth_and_stable_empty_bins():
    search_steps = torch.tensor([[0, 1, 1, 3, 4, 8, 16, 99]])
    controls = torch.tensor([[
        util.STOP,
        util.PROCEED,
        util.STOP,
        util.RESET,
        util.STOP,
        util.STOP,
        util.STOP,
        util.PROCEED,
    ]])
    valid = torch.tensor([[True, True, True, True, True, True, True, False]])
    stop_probability = torch.tensor([[
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0,
    ]])

    stats = util.get_search_depth_stop_stats(
        search_steps, controls, valid, stop_probability
    )

    # Accepted PROCEED/RESET rows are reported at their pre-increment depth.
    assert stats["search/depth_bin_0_count"] == 2
    assert stats["search/depth_bin_0_stop_probability"] == pytest.approx(0.15)
    assert stats["search/depth_bin_1_count"] == 1
    assert stats["search/depth_bin_1_stop_probability"] == pytest.approx(0.3)
    assert stats["search/depth_bin_2_3_count"] == 1
    assert stats["search/depth_bin_2_3_stop_probability"] == pytest.approx(0.4)
    assert stats["search/depth_bin_4_7_count"] == 1
    assert stats["search/depth_bin_4_7_stop_probability"] == pytest.approx(0.5)
    assert stats["search/depth_bin_8_15_count"] == 1
    assert stats["search/depth_bin_8_15_stop_probability"] == pytest.approx(0.6)
    assert stats["search/depth_bin_16_plus_count"] == 1
    assert stats["search/depth_bin_16_plus_stop_probability"] == pytest.approx(0.7)
    assert stats["search/depth_stop_probability_count"] == 7
    assert stats["search/depth_stop_probability_slope"] > 0.0

    empty = util.get_search_depth_stop_stats(
        search_steps,
        controls,
        torch.zeros_like(valid),
        stop_probability,
    )
    assert empty["search/depth_stop_probability_count"] == 0
    assert empty["search/depth_stop_probability_slope"] == 0.0
    for label in ("0", "1", "2_3", "4_7", "8_15", "16_plus"):
        assert empty[f"search/depth_bin_{label}_count"] == 0
        assert empty[f"search/depth_bin_{label}_stop_probability"] == 0.0


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


@pytest.mark.parametrize(
    ("model_setting", "global_setting", "expected"),
    [("inherit", True, True), ("inherit", False, False), ("false", True, False),
     ("true", False, True), (False, True, False), (True, False, True)],
)
def test_model_float16_can_override_or_inherit_actor_precision(
        model_setting, global_setting, expected):
    flags = _flags(
        dynamic_search=False,
        model_float16=model_setting,
        float16=global_setting,
    )
    util.process_flags(flags)
    assert flags.model_float16 is expected


def test_model_float16_rejects_unknown_setting():
    with pytest.raises(ValueError, match="model_float16"):
        util.process_flags(
            _flags(dynamic_search=False, model_float16="sometimes", float16=True)
        )


def test_model_state_projection_legacy_defaults_are_backward_compatible():
    flags = _flags(dynamic_search=False)

    util.process_flags(flags)

    assert flags.model_state_projection == "none"
    assert flags.model_state_range_loss_cost == 0.0


@pytest.mark.parametrize("bad_mode", [None, "sigmoid", "Clamp", True, 1])
def test_model_state_projection_rejects_unknown_mode(bad_mode):
    with pytest.raises(ValueError, match="model_state_projection"):
        util.process_flags(
            _flags(dynamic_search=False, model_state_projection=bad_mode)
        )


@pytest.mark.parametrize("bad_cost", [-0.1, float("nan"), float("inf"), True])
def test_model_state_range_loss_rejects_invalid_cost(bad_cost):
    with pytest.raises(ValueError, match="model_state_range_loss_cost"):
        util.process_flags(
            _flags(
                dynamic_search=False,
                model_state_projection="clamp",
                model_state_range_loss_cost=bad_cost,
            )
        )


def test_model_state_range_loss_requires_projection():
    with pytest.raises(ValueError, match="requires"):
        util.process_flags(
            _flags(
                dynamic_search=False,
                model_state_projection="none",
                model_state_range_loss_cost=1.0,
            )
        )


def test_model_state_projection_rejects_latent_decoder_depth():
    with pytest.raises(ValueError, match="model_decoder_depth=0"):
        util.process_flags(
            _flags(
                dynamic_search=False,
                model_decoder_depth=1,
                model_state_projection="clamp",
            )
        )


def test_schedule_horizon_inherits_total_steps():
    flags = _flags(dynamic_search=False, total_steps=30_000)
    util.process_flags(flags)
    assert flags.schedule_total_steps == 30_000
    assert util.schedule_progress(flags, 15_000) == pytest.approx(0.5)


def test_schedule_horizon_can_outlive_bounded_run():
    flags = _flags(
        dynamic_search=False,
        total_steps=30_000,
        schedule_total_steps=100_000_000,
    )
    util.process_flags(flags)
    assert flags.total_steps == 30_000
    assert flags.schedule_total_steps == 100_000_000
    assert util.schedule_progress(flags, 30_000) == pytest.approx(0.0003)
    assert util.schedule_progress(flags, 200_000_000) == 1.0


@pytest.mark.parametrize("bad_horizon", [0, -2, 1.5, True, "100000000"])
def test_schedule_horizon_rejects_invalid_values(bad_horizon):
    with pytest.raises(ValueError, match="schedule_total_steps"):
        util.process_flags(
            _flags(
                dynamic_search=False,
                total_steps=30_000,
                schedule_total_steps=bad_horizon,
            )
        )


@pytest.mark.parametrize("bad_limit", [0, -1, 1.5, True, "8"])
def test_actor_amp_skip_limit_rejects_invalid_values(bad_limit):
    with pytest.raises(ValueError, match="actor_amp_max_consecutive_skips"):
        util.process_flags(
            _flags(
                dynamic_search=False,
                actor_amp_max_consecutive_skips=bad_limit,
            )
        )


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
