import pytest
from reward_probe import ReturnAccumulator, keyed_seed, verify_arms


def test_cost_stops_at_first_stage_end_but_task_rewards_continue():
    acc = ReturnAccumulator(gamma=.9, cost=.1)
    acc.observe(reward=0, think_reward=-1, real_transition=False, stage_end=False, done=False)
    acc.observe(reward=0, think_reward=0, real_transition=False, stage_end=True, done=False)
    assert acc.discount == 1
    acc.observe(reward=2, think_reward=0, real_transition=True, stage_end=False, done=False)
    acc.observe(reward=0, think_reward=-1, real_transition=False, stage_end=False, done=False)
    acc.observe(reward=3, think_reward=0, real_transition=True, stage_end=False, done=True)
    scores = acc.scores(task_bootstrap=999)
    assert scores['task_truncated'] == pytest.approx(4.7)
    assert scores['current_stage_cost'] == pytest.approx(.1)
    assert scores['net_truncated'] == pytest.approx(4.6)
    assert scores['all_future_cost_net_truncated'] == pytest.approx(4.51)
    assert scores['net_bootstrapped'] == scores['net_truncated']
    assert acc.stage_computations == 1 and acc.all_computations == 2


def test_forced_final_computation_is_charged_and_bootstrap_discount_is_real_time():
    acc = ReturnAccumulator(gamma=.9, cost=.1)
    acc.observe(reward=0, think_reward=-1, real_transition=False, stage_end=True, done=False)
    for _ in range(2):
        acc.observe(reward=0, think_reward=0, real_transition=True, stage_end=False, done=False)
    assert acc.scores(10)['net_bootstrapped'] == pytest.approx(-.1+.9**2*10)
    assert not acc.terminal and not acc.stage_open


def test_pairing_rejects_state_or_repeated_return_mismatch():
    row = dict(entry=dict(prefix_sha256='a',entry_sha256='b',tree_sha256='c'), returns={'v':1}, trajectory=[1])
    verify_arms(row,row,row)
    with pytest.raises(AssertionError,match='Unpaired'):
        verify_arms(row,dict(row,entry=dict(row['entry'],tree_sha256='x')),row)
    with pytest.raises(AssertionError,match='Repeated'):
        verify_arms(row,row,dict(row,returns={'v':2}))
    assert keyed_seed(1,2,3,0,1) == keyed_seed(1,2,3,0,1)
    assert keyed_seed(1,2,3,0,1) != keyed_seed(1,3,3,0,1)
