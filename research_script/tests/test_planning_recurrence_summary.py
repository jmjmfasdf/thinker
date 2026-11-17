import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location('recurrence_summary',
    Path(__file__).parents[1]/'diagnostics/summarize_planning_recurrence.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_later_queries_are_paired_with_their_own_first_query():
    rows = [dict(seed=1, stream=0, decision=d, count=c, actual=p,
                 first_probability=first, frozen_hidden=first, repeated_token=first)
            for d, c, p, first in [(0, 1, .8, .8), (1, 1, .2, .2), (0, 2, .9, .8)]]
    observed = [dict(r, probability=r['actual'], online_delta=r['actual']-.5) for r in rows]
    report, _ = MODULE.summarize(rows, observed)
    later = report['by_count'][1]
    assert later['first_probability_same_survivors'] == .8
    assert later['actual_paired_probability_change'] == pytest.approx(.1)
    assert report['first_query_online_gate_sign']['decisions'] == 2
    with pytest.raises(ValueError, match='coverage'):
        MODULE.summarize(rows, observed[:-1])


def test_log_odds_preserve_contributions_hidden_by_saturated_probabilities():
    rows = []
    for count, actual, other in [(1, 20., 20.), (2, 30., 25.)]:
        row = dict(seed=1, stream=0, decision=0, count=count,
                   first_probability=float(1/(1+np.exp(-20))), first_log_odds=20.)
        for name, value in [('actual', actual), ('frozen_hidden', other),
                            ('frozen_other', other), ('repeated_token', other)]:
            row[name] = float(1/(1+np.exp(-value)))
            row[name+'_log_odds'] = value
        rows.append(row)
    observed = [dict(r, probability=r['actual'], online_delta=1.) for r in rows]
    report, _ = MODULE.summarize(rows, observed)
    later = report['by_count'][1]
    assert later['actual_paired_probability_change'] < 1e-8
    assert later['actual_paired_log_odds_change'] == 10.
    assert later['tree_log_odds_contribution'] == 5.
    assert later['other_log_odds_contribution'] == 5.
    rows[1]['frozen_other_log_odds'] = 26.
    rows[1]['frozen_other'] = float(1/(1+np.exp(-26.)))
    with pytest.raises(ValueError, match='decomposition'):
        MODULE.summarize(rows, observed)
