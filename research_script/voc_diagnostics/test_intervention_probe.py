import numpy as np
import pytest

from intervention_probe import run_intervention, select_targets, gate_choice
from test_forced_budget_probe import fixture_module


def test_frozen_content_matches_actual_computations_tokens_and_search_schedule():
    f = fixture_module()
    flags = f._trace_flags(max_search_steps=8, max_depth=8)
    model = f._TraceModel(3)
    actor = f._ProceedResetStopActor(3, len(f.util.get_reward_names(flags)), control_mode="schedule")
    batch, _ = f._trace_batch(edge_count=4, num_actions=3)
    full = run_intervention(actor, model, flags, batch, target=2, budget=3, mode="fixed", seed=11, ema=None)
    frozen = run_intervention(actor, model, flags, batch, target=2, budget=3, mode="frozen", seed=11, ema=None,
                              schedule=full["action_schedule"])
    assert full["actual_computations"] == frozen["actual_computations"] == 3
    assert full["token_masks"] == frozen["token_masks"]
    assert full["actor_calls"] == frozen["actor_calls"]
    assert full["action_schedule"] == frozen["action_schedule"]
    assert full["entry_sha256"] == frozen["entry_sha256"]
    assert len(set(frozen["content_hashes"])) == 1
    assert len(set(full["content_hashes"])) > 1


def test_strata_selection_is_label_only_and_reports_absence():
    assert select_targets([0, 0, 2, 2, 0, 0]) == ([1, 2, 3, 4], [])
    targets, missing = select_targets([0, 0, 2, 2, 0], max_edge=2)
    assert targets == [1, 2]
    assert missing == ["active_to_NOOP", "active_to_active"]


def test_common_uniform_preserves_bernoulli_gate_and_conditional_action():
    draws = (np.arange(1000) + .5) / 1000
    controls = [gate_choice(.3, u, [0., 1.]) for u in draws]
    assert controls.count(1) == 300
    assert controls.count(2) == 700
    assert gate_choice(0., 0., [1., 0.]) == 2
    assert gate_choice(1., .999, [1., 0.]) == 0
    with pytest.raises(ValueError):
        gate_choice(float("nan"), .5, [0., 0.])
