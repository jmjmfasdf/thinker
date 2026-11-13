import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from forced_budget_probe import ROOT, first_transition_targets, run_probe, verify_pairing


def fixture_module():
    spec = importlib.util.spec_from_file_location("trace_test_fixtures", ROOT / "thinker/tests/test_dynamic_imitation_trace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_cenv_forced_budget_and_repeat_pairing():
    f = fixture_module()
    flags = f._trace_flags(max_search_steps=8, max_depth=8)
    model = f._TraceModel(3)
    actor = f._ProceedResetStopActor(3, len(f.util.get_reward_names(flags)), control_mode="schedule")
    batch, _ = f._trace_batch(edge_count=4, num_actions=3)
    results = [run_probe(actor, model, flags, batch, target=2, budget=k, seed=11, ema=None) for k in [0, 1, 3, 0]]
    verify_pairing(results)
    assert [r['actual_computations'] for r in results] == [0, 1, 3, 0]
    assert all(r['target_human_action_not_executed'] for r in results)


def test_pairing_rejects_different_preintervention_state():
    good = dict(prefix_sha256='a', prefix_calls=2, entry_sha256='b', tree_observation_sha256='c', budget=0, probability=[.2,.8])
    bad = dict(good, entry_sha256='different')
    with pytest.raises(AssertionError, match='Unpaired'):
        verify_pairing([good,bad])


def test_transition_selection_uses_current_action_zero_based_edge():
    actions = torch.tensor([0, 0, 2, 3, 0, 1, 0])
    assert first_transition_targets(actions) == [2, 4]
    assert first_transition_targets(np.array([2, 0, 0, 3])) == [1, 3]
    with pytest.raises(ValueError, match="both"):
        first_transition_targets(np.array([0, 0, 2, 2]))


def test_current_and_future_human_labels_cannot_change_target_readout():
    f = fixture_module()
    flags = f._trace_flags(max_search_steps=8, max_depth=8)
    model = f._TraceModel(3)
    actor = f._ProceedResetStopActor(3, len(f.util.get_reward_names(flags)), control_mode="schedule")
    batch, _ = f._trace_batch(edge_count=4, num_actions=3)
    changed = dict(batch)
    changed["actions_seq"] = batch["actions_seq"].copy()
    changed["actions_seq"][..., 2:] = (changed["actions_seq"][..., 2:] + 1) % 3
    original = run_probe(actor, model, flags, batch, target=2, budget=3, seed=11, ema=None)
    counterfactual = run_probe(actor, model, flags, changed, target=2, budget=3, seed=11, ema=None)
    for key in ("prefix_sha256", "entry_sha256", "tree_observation_sha256", "probability"):
        assert original[key] == counterfactual[key]
