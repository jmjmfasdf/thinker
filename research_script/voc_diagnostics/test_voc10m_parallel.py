from types import SimpleNamespace as NS
import json
import pytest
import torch
from thinker import util
from thinker.voc_training_metrics import TrainingMetrics
from voc10m_parallel_monitor import first_checkpoint


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.voc_gate_head=torch.nn.Linear(2,1)
        self.num_actions=6


def test_parallel_physical_counts_negative_scores_and_publication(tmp_path):
    flags=NS(env_n=2,max_search_steps=30,voc_gate_q_temperature=.05)
    m=TrainingMetrics(Actor(),flags,tmp_path)
    previous=NS(phase=torch.tensor([[util.SEARCH_PHASE]*2]),forced_stop=torch.zeros(1,2,dtype=torch.bool),
                tree_reps=torch.zeros(1,2,74),search_steps=torch.zeros(1,2,dtype=torch.long))
    out=NS(control_valid=torch.ones(1,2,dtype=torch.bool),
           search_control_logits=torch.zeros(1,2,3),voc_q=torch.zeros(1,2,2))
    m.gate_logits=torch.zeros(1,2,1)
    info=dict(real_transition=torch.zeros(2,dtype=torch.bool),think_reward=torch.tensor([-1.,0.]),
              accepted_control=torch.tensor([util.PROCEED,util.STOP]),real_done=torch.zeros(2,dtype=torch.bool),
              episode_return=torch.zeros(2))
    m.observe(previous,out,info,real_step=0,attempt=0)
    assert m.steps==0 and m.think==1
    info.update(real_transition=torch.ones(2,dtype=torch.bool),think_reward=torch.zeros(2),
                episode_return=torch.tensor([-21.,-19.]),real_done=torch.ones(2,dtype=torch.bool))
    m.observe(previous,out,info,real_step=0,attempt=0)
    m.publish(terminal=True)
    data=json.loads((tmp_path/'collector_metrics.json').read_text())
    assert data['physical_env_steps']==2 and data['imagined_transitions']==1
    assert data['max_completed_episode_return']==-19.
    assert data['mean_last_100_episode_return']==-20.
    assert data['raw_reward_sum']==-40.
    assert data['count_0_decisions']==data['count_1_decisions']==1
    assert data['terminal'] is True
    m.close()


def test_pair_selection_never_crosses_milestone_window(tmp_path):
    for step in (199984,200064,299999,300000):
        (tmp_path/f'ckp_actor.tar_step_{step}').touch()
    assert first_checkpoint(tmp_path,'actor',200000).name.endswith('200064')
    assert first_checkpoint(tmp_path,'model',200000) is None
    (tmp_path/'ckp_model.tar_step_300000').touch()
    assert first_checkpoint(tmp_path,'model',200000) is None
