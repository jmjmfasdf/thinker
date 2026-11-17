import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('selected_utility',
    Path(__file__).parents[1]/'diagnostics/evaluate_planning_selected_utility.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rows():
    return [dict(seed=1,stream=0,decision=i,episode=0,episode_step=i,
        first_optional_probability=p,first_optional_delta=p-.5)
        for i,p in enumerate([.05,.2,.09,.6,.08,.9])]


def test_first_positive_and_latest_earlier_low_are_selected_not_maximum():
    chosen = MODULE.select_states(rows()[::-1])
    assert [(r['cohort'],r['prefix']) for r in chosen] == [('low',2),('high',3)]


def test_reference_never_crosses_episode_boundary():
    sample = rows()
    for row in sample[3:]:
        row['episode'] = 1
    with pytest.raises(ValueError,match='same episode'):
        MODULE.select_states(sample)


def test_duplicate_time_identity_is_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        MODULE.select_states(rows()+rows()[:1])


def test_different_prefixes_remain_distinct_and_repetitions_average_before_comparison():
    selected = MODULE.select_states(rows())
    pairs = []
    for selection in selected:
        for rep in range(2):
            value = (2 if selection['cohort']=='high' else -1)+rep
            pairs.append(dict(selection,repetition=rep,entry_hash=str(selection['prefix']),
                ema_delta=selection['online_delta'],delta_net=value,delta_task=value+.0001,
                added_queries=1,pair_tail_bound=.00004))
    summary = MODULE.summarize_pairs(pairs,selected,2,bootstrap=10)
    assert summary['selected_states'] == 2
    assert summary['independent_streams'] == 1
    assert summary['high_mean_net_utility'] == 2.5
    assert summary['low_mean_net_utility'] == -.5
    assert summary['high_minus_low_mean'] == 3
    assert summary['high_minus_low_ci95'] == [3,3]
    with pytest.raises(ValueError,match='repetition'):
        MODULE.summarize_pairs(pairs[:-1],selected,2,bootstrap=10)


def test_exact_natural_state_guard_rejects_changed_hidden_state_hash():
    MODULE.verify_reference([{'hash':'a'}],{'entries':[{'hash':'a'}]})
    with pytest.raises(ValueError,match='exact natural'):
        MODULE.verify_reference([{'hash':'a'}],{'entries':[{'hash':'b'}]})
