import numpy as np
import pytest
from gru_ablation_eval import prescribed_budget,geometric_survival,paired_summary


@pytest.mark.parametrize('mean',[0.,.4,1.4,7.,30.])
def test_prescribed_budget_has_requested_mean_and_bounds(mean):
    cap=30;n=10000
    fixed=np.array([prescribed_budget('fixed',mean,cap,j,.5) for j in range(n)])
    random=np.array([prescribed_budget('random',mean,cap,0,(j+.5)/n) for j in range(n)])
    assert abs(fixed.mean()-mean)<1e-10
    assert abs(random.mean()-mean)<.002
    assert fixed.min()>=0 and random.min()>=0
    assert fixed.max()<=cap and random.max()<=cap
    r=geometric_survival(mean,cap)
    assert abs(sum(r**k for k in range(1,cap+1))-mean)<1e-10


def test_paired_summary_keeps_counts_separate_and_averages_repetitions():
    rows=[]
    for stream in range(4):
        for count in [0,3]:
            for rep in range(2):
                target=(stream+count)/10
                rows.append(dict(seed=1,stream=stream,prefix=128,count=count,repetition=rep,
                    delta=target+(-.1 if rep==0 else .1),online=target,ema=target,
                    probability=.5,cost=.001,tail_bound=0.,task_delta=target+.001))
    result=paired_summary(rows)
    assert result['states']==8 and result['pairs']==16 and result['independent_streams']==4
    assert result['ema_rmse']<1e-12 and result['online_rmse']<1e-12
    assert result['allocation_gain_vs_cost_matched_random']==0
