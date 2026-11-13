import pytest
import torch
import numpy as np
from reward_learning_eval import categorical, tail_bound, aggregate_pairs, verify_pair


def test_sampling_uses_full_policy_not_argmax():
    logits=torch.log(torch.tensor([[.25,.75],[.25,.75],[.25,.75]]))
    assert categorical(logits,np.array([.1,.5,.99])).tolist()==[0,1,1]


def test_discounted_tail_bound_is_smaller_than_one_compute_cost():
    assert 2*tail_bound(.99,1536,1)<.00005
    assert tail_bound(.99,8,1,True)==0
    assert tail_bound(.9,2,1)==pytest.approx(8.1)


def test_pair_checks_full_prefix_and_repeated_returns():
    a=dict(entry=dict(prefix_sha256='p',entry_sha256='e',tree_sha256='t'),trace_sha256='x',returns=[1])
    verify_pair(a,a,a)
    with pytest.raises(AssertionError):verify_pair(a,dict(a,entry=dict(a['entry'],tree_sha256='different')))
    with pytest.raises(AssertionError):verify_pair(a,a,dict(a,trace_sha256='different'))


def test_repeat_aggregation_and_compute_matched_allocation():
    rows=[]
    for stream,delta,p,cost in [(0,1.,.9,.001),(1,-1.,.1,.003)]:
        for rep in range(2):rows.append(dict(seed=1,stream=stream,prefix=128,delta=delta,
            ema_delta=delta,p_continue=p,continue_stage_cost=cost,tail_bound=0.,repetition=rep))
    result=aggregate_pairs(rows,bootstrap=30)
    assert result['states']==2 and result['pairs']==4
    assert result['random_continue_probability']==pytest.approx(.3)
    assert result['allocation_gain_vs_compute_matched_random']==pytest.approx(.4)
    assert result['rmse']==0 and result['zero_predictor_rmse']==1
