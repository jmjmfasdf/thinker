import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'diagnostics'))
from compare_online_ema_single_choice import metrics, summarize


def states(utility):
    return [dict(utility=y,task_utility=y,count=1,probability=p,
        ema_delta=.01*np.log(p/(1-p)),online_delta=.01*np.log(q/(1-q)))
        for y,p,q in zip(utility,[.2,.8],[.8,.2])]


def test_one_choice_expectation_matches_hand_calculation():
    result=metrics(states([1.,-1.]),.01)
    assert result['ema_gain_over_stop']==pytest.approx(-.3)
    assert result['online_gain_over_stop']==pytest.approx(.3)
    assert result['online_minus_ema_gain']==pytest.approx(.6)
    assert result['online_minus_ema_matched_advantage']==pytest.approx(.6)


def test_constant_utility_has_no_choice_matched_advantage():
    result=metrics(states([-.0001,-.0001]),.01)
    assert result['ema_choice_matched_advantage']==0.
    assert result['online_choice_matched_advantage']==0.
    assert result['online_minus_ema_gain']==pytest.approx(0.,abs=1e-18)


def test_incomplete_registered_scope_is_rejected():
    with pytest.raises(ValueError,match='all64states'):
        summarize([dict(seed=8301,stream=0,count=1,repetitions=2)],.01,bootstrap=2)
