"""Reproduce the collector's missed cap event with the real Cython planner."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'thinker'))
from thinker import util
from thinker.cenv import cModelWrapper
from thinker.voc_training_metrics import TrainingMetrics
from tests.test_dynamic_cenv import FakeModel,FakeVectorEnv,_flags,_step


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    flags=_flags();flags.max_search_steps=1;flags.min_search_steps=1;flags.env_n=1
    env=FakeVectorEnv(env_n=1);model=FakeModel(env.num_actions)
    planner=cModelWrapper(env,1,flags,model,device=torch.device('cpu'))
    actor=SimpleNamespace(num_actions=env.num_actions,voc_gate_head=torch.nn.Linear(1,1))
    metrics=TrainingMetrics(actor,flags,a.output.parent)
    try:
        _,info=planner.reset(model)
        def previous(info):
            return SimpleNamespace(phase=info['phase'].unsqueeze(0),forced_stop=info['forced_stop'].unsqueeze(0),
                legal_control_mask=info['legal_control_mask'].unsqueeze(0))
        root=previous(info)
        _,_,_,_,cap_info=_step(planner,model,[0],[util.PROCEED])
        assert bool(cap_info['forced_stop'][0]) and int(cap_info['phase'][0])==util.NEED_REAL_ACTION_PHASE
        metrics.observe(root,SimpleNamespace(control_valid=torch.ones(1,1,dtype=torch.bool)),cap_info,real_step=1,attempt=0)
        _,_,_,_,real_info=_step(planner,model,[0],[util.PROCEED])
        metrics.observe(previous(cap_info),SimpleNamespace(control_valid=torch.zeros(1,1,dtype=torch.bool)),real_info,real_step=1,attempt=0)
        assert metrics.hist[-1]==1 and metrics.forced==0
        expected=int(cap_info['forced_stop'].sum())+int(real_info['forced_stop'].sum())
        assert expected==1
        result=dict(status='REPRODUCED',planner_cap_events=expected,completed_cap_decisions=int(metrics.hist[-1]),
                    collector_forced_stop_count=metrics.forced,
                    cause='Collector checks previous SEARCH phase AND previous forced_stop. Cap event belongs to post-step info and its next phase is NEED_REAL_ACTION.',
                    proposed_fix='Accumulate info[forced_stop] from the returned transition instead of previous phase/flag.',
                    learning_effect='Counter only; does not control action sampling, rewards, or gradient. Learner search/forced_stop_count is a separate event-based metric.')
        a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    finally:
        metrics.hook.remove();planner.close()


if __name__=='__main__':main()
