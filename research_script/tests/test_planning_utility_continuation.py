import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('utility_continuation',
    Path(__file__).parents[1] / 'diagnostics/continue_planning_utility.py')
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def group_fixture():
    result = {}
    for label, arm, queries in [('stop', 'stop', 0), ('continue_', 'continue', 1)]:
        result[label] = dict(seed=7, repetition=0, prefix=5, arm=arm, targets=[1],
            real_action_rng='seed/stream/real_index/repetition; independent of search count',
            entries=[dict(hash='same-state', count=1)], tail_bounds=[.1], real_steps=4,
            returns=[dict(task_truncated=.3, current_stage_cost=queries*.001,
                net_truncated=.3-queries*.001, cost=.001, gamma=.99,
                stage_computations=queries, all_future_cost_net_truncated=.2,
                terminal=False, real_steps=4)])
    return result


def validate(group):
    MODULE.validate_group(group, seed=7, count=1, repetition=0, prefix=5,
                          batch=1, horizon=4, gamma=.99, think_cost=.001)


def test_cached_groups_require_matching_entry_protocol_and_accounting():
    group = group_fixture(); validate(group)
    bad = copy.deepcopy(group); bad['continue_']['entries'][0]['hash'] = 'other-state'
    with pytest.raises(ValueError, match='entries differ'): validate(bad)
    bad = copy.deepcopy(group); bad['stop']['prefix'] = 6
    with pytest.raises(ValueError, match='prefix'): validate(bad)
    bad = copy.deepcopy(group); bad['continue_']['returns'][0]['current_stage_cost'] = .02
    with pytest.raises(ValueError, match='cost differs'): validate(bad)
    bad = copy.deepcopy(group); bad['stop']['returns'][0]['real_steps'] = 3
    with pytest.raises(ValueError, match='horizon'): validate(bad)


def test_cached_allocations_cannot_be_partial_or_duplicated():
    rows = [dict(seed=7, stream=i, decision=5) for i in range(2)]
    assert MODULE.checked_allocation(rows, seed=7, batch=2, prefix=5, decisions=1) == rows
    assert MODULE.checked_allocation(rows, seed=8, batch=2, prefix=5, decisions=1) == []
    for invalid in [rows[:1], rows + rows[:1], [rows[0], rows[0]]]:
        with pytest.raises(ValueError, match='allocation'):
            MODULE.checked_allocation(invalid, seed=7, batch=2, prefix=5, decisions=1)


def test_only_old_complete_groups_are_reused_new_stop_repeat_really_runs():
    group = group_fixture()
    plan = dict(batch=1, prefix=5, horizon=4, decisions=1)
    original_calls = []
    def original_arm(*args, **kwargs):
        original_calls.append(kwargs['arm']); return dict(call=len(original_calls))
    allocation_calls = []
    def original_allocation(*args, **kwargs):
        allocation_calls.append(kwargs['seed']); return ['fresh']
    cached_rows = [dict(seed=7, stream=0, decision=5)]
    arm, allocation, reused, computed = MODULE.make_adapters(
        plan, {(7, 1, 0): group}, cached_rows, original_arm, original_allocation)
    kwargs = dict(seed=7, targets=np.array([1]), repetition=0, prefix=5, horizon=4, arm='stop', device='fake')
    first = arm(None, None, None, **kwargs)
    first['entries'][0]['hash'] = 'mutated-consumer-copy'
    assert arm(None, None, None, **kwargs)['entries'][0]['hash'] == 'same-state'
    assert original_calls == []
    kwargs['seed'] = 8
    assert arm(None, None, None, **kwargs) == {'call': 1}
    kwargs['arm'] = 'continue'; arm(None, None, None, **kwargs)
    kwargs['arm'] = 'stop'
    assert arm(None, None, None, **kwargs) == {'call': 3}
    assert original_calls == ['stop', 'continue', 'stop']
    assert len(reused) == 2 and len(computed) == 3
    akw = dict(seed=7, batch=1, decisions=1, warmup=5, device='fake')
    assert allocation(None, None, None, **akw) == cached_rows
    akw['seed'] = 8
    assert allocation(None, None, None, **akw) == ['fresh']
    assert allocation_calls == [8]
