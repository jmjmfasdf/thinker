from types import SimpleNamespace as NS
import numpy as np
import torch
from thinker import util
from voc10m_metrics import TrainingMetrics


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.voc_gate_head=torch.nn.Linear(2,1)
        self.num_actions=6


def event(m, *, think=False, real=False, done=False, reward=0., control=util.STOP, forced=False):
    previous=NS(phase=torch.tensor([[util.SEARCH_PHASE]]),forced_stop=torch.tensor([[forced]]),
        tree_reps=torch.zeros(1,1,74),search_steps=torch.zeros(1,1,dtype=torch.int64))
    out=NS(control_valid=torch.tensor([[not forced]]),
        search_control_logits=torch.zeros(1,1,3),voc_q=torch.tensor([[[.3,.1]]]))
    info=dict(real_transition=torch.tensor([real]),think_reward=torch.tensor([-int(think)]),
        accepted_control=torch.tensor([control]),real_done=torch.tensor([done]),
        episode_return=torch.tensor([reward]))
    m.gate_logits=torch.tensor([[[2.]]])
    m.observe(previous,out,info,real_step=m.steps,attempt=0)


def test_negative_episode_returns_and_reward_increments(tmp_path):
    flags=NS(env_n=1,max_search_steps=30,voc_gate_q_temperature=.05)
    m=TrainingMetrics(Actor(),flags,tmp_path)
    assert m.metrics()['max_completed_episode_return'] is None
    event(m,real=True,reward=-1.)
    event(m,real=True,done=True,reward=-3.)
    event(m,real=True,done=True,reward=-2.)
    assert m.metrics()['mean_last_100_episode_return']==-2.5
    assert m.metrics()['max_completed_episode_return']==-2.
    assert m.raw_reward==-5.
    assert m.episodes==2
    m.close()


def test_imagination_counts_forced_stops_and_restore(tmp_path):
    flags=NS(env_n=1,max_search_steps=30,voc_gate_q_temperature=.05)
    m=TrainingMetrics(Actor(),flags,tmp_path)
    event(m,think=True,control=util.PROCEED)
    event(m,think=True,control=util.RESET)
    event(m,real=True,forced=True)
    event(m,real=True)
    assert m.hist[2]==1 and m.hist[0]==1
    assert m.think==2 and m.steps==2 and m.forced==1
    assert m.resets==1 and m.stops==1
    assert abs(m.rows[0]['ema_delta_q']-.1)<1e-7
    clone=TrainingMetrics(Actor(),flags,tmp_path,m.state())
    assert clone.metrics()==m.metrics()
    np.testing.assert_array_equal(clone.current,[0])
    m.close();clone.close()
