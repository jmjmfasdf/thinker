import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('planning_action',
    Path(__file__).parents[1]/'diagnostics/probe_planning_real_action.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def test_known_distribution_changes_and_zero_probabilities():
    identical=MODULE.policy_metrics([1,0],[1,0])
    assert identical['total_variation']==0 and identical['jensen_shannon_nat']==0
    opposed=MODULE.policy_metrics([1,0],[0,1])
    assert opposed['total_variation']==1
    assert opposed['jensen_shannon_nat']==pytest.approx(np.log(2))
    assert opposed['argmax_changed']
    subtle=MODULE.policy_metrics([.6,.4],[.9,.1])
    assert subtle['total_variation']==pytest.approx(.3)
    assert not subtle['argmax_changed']


@pytest.mark.parametrize('bad',[[.3,.3],[-.1,1.1],[float('nan'),0],[[1,0]]])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):MODULE.policy_metrics(bad,[1,0])


def test_summary_requires_complete_repeats_and_averages_before_resampling():
    rows=[dict(seed=1,stream=0,prefix=10,repetition=i,entry_hash='same',total_variation=v,
        jensen_shannon_nat=0.,argmax_changed=False,sampled_action_changed=bool(i),added_queries=1)
        for i,v in enumerate([.1,.5])]
    r=MODULE.summarize(rows,1,2,bootstrap=10)
    assert r['independent_streams']==1
    assert r['metrics']['total_variation']['mean']==pytest.approx(.3)
    assert r['metrics']['sampled_action_changed']['mean']==.5
    with pytest.raises(ValueError,match='repetition'):MODULE.summarize(rows[:1],1,2)
    with pytest.raises(ValueError,match='coverage'):MODULE.summarize(rows,2,2)
    rows[1]['entry_hash']='different'
    with pytest.raises(ValueError,match='entry'):MODULE.summarize(rows,1,2)
