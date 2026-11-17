import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location('planning_queries',
    Path(__file__).parents[1]/'diagnostics/probe_adaptive_planning_queries.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def example(length, probabilities, cap=4):
    allocation = dict(seed=1, stream=0, decision=128, length=length,
        optional_choices=len(probabilities), first_optional_probability=probabilities[0],
        sign_changes=sum((a-.5)*(b-.5)<0 for a,b in zip(probabilities, probabilities[1:])))
    rows = [dict(seed=1, stream=0, decision=128, count=i+1, probability=p,
        online_delta=0., real_episode=0, episode_age=128,
        **{'continue': length == cap or i+1 < length}) for i,p in enumerate(probabilities)]
    return rows, [allocation]


def test_cap_stop_is_not_a_learned_stop_and_within_stage_change_is_measured():
    rows, allocation = example(4, [.6, .8, .99])
    report, stages = PROBE.query_report(rows, allocation, cap=4)
    assert len(report['by_count']) == 3
    assert all(r['observed_continue_fraction'] == 1. for r in report['by_count'])
    assert stages[0]['last_minus_first'] == pytest.approx(.39)
    assert report['by_realized_length'][2]['fraction'] == 1.


def test_minimum_and_interior_stops_include_the_optional_stop_choice():
    for length, probabilities in [(1, [.2]), (3, [.8, .6, .3])]:
        report, stages = PROBE.query_report(*example(length, probabilities), cap=4)
        assert report['by_count'][-1]['observed_continue_fraction'] == 0.
        assert stages[0]['optional_choices'] == length


def test_incomplete_trace_and_false_stopping_record_fail():
    rows, allocation = example(3, [.8, .6, .3])
    with pytest.raises(ValueError, match='missing or duplicate'):
        PROBE.query_report(rows[:-1], allocation, cap=4)
    rows[-1]['continue'] = True
    with pytest.raises(ValueError, match='controls disagree'):
        PROBE.query_report(rows, allocation, cap=4)


def test_observer_overlap_is_exact_and_required():
    _, allocations = example(3, [.8, .6, .3])
    assert PROBE.exact_overlap(allocations, allocations) == 1
    with pytest.raises(ValueError, match='changed frozen allocation'):
        PROBE.exact_overlap([dict(allocations[0], length=4)], allocations)
    with pytest.raises(ValueError, match='no overlap'):
        PROBE.exact_overlap([dict(allocations[0], seed=2)], allocations)
