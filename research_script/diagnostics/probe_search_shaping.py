"""Production-planner control: optimistic model value can earn intrinsic reward.

A deterministic zero-reward environment isolates the objective, not Atari
accuracy or policy quality. No learning is run.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'thinker'))
from thinker.cenv import cModelWrapper
from thinker import util
from thinker.learn_actor import dynamic_voc_policy_log_probs
from tests.test_dynamic_cenv import FakeModel, FakeVectorEnv, _flags, _step


class ZeroRewardEnv(FakeVectorEnv):
    def step(self,actions,env_id=None):
        obs,reward,done,truncated,info=super().step(actions,env_id)
        reward.fill(0.)
        return obs,reward,done,truncated,info


class BiasedValue(FakeModel):
    def __init__(self,num_actions,bias):
        super().__init__(num_actions);self.bias=bias
    def forward_single(self,state,action):
        out=super().forward_single(state,action)
        out.rs.zero_();out.vs.fill_(self.bias)
        return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);rows=[]
    for bias in [-10.,0.,10.]:
        env=ZeroRewardEnv(env_n=1);model=BiasedValue(env.num_actions,bias)
        flags=_flags();flags.im_enable=True;flags.im_cost=1.;flags.discounting=.99
        planner=cModelWrapper(env,1,flags,model,device=torch.device('cpu'))
        try:
            planner.reset(model)
            _,_,_,_,info=_step(planner,model,[0],[util.PROCEED])
            intrinsic=float(info['im_reward'].reshape(-1)[0])
            _step(planner,model,[0],[util.STOP])
            _,reward,_,_,info=_step(planner,model,[0],[util.PROCEED])
            assert info['real_transition'].all() and reward.eq(0).all()
            assert abs(intrinsic-max(0.,.99*bias))<1e-5
            rows.append(dict(imagined_value_bias=bias,intrinsic_reward=intrinsic,actual_reward=0.,true_future_return=0.))
        finally:planner.close()
    # Existing task-reward learning of imagined primary/bout choices remains.
    gate=torch.tensor(-.4,requires_grad=True);primary=torch.tensor(-1.,requires_grad=True);bout=torch.tensor(-.2,requires_grad=True)
    out=dynamic_voc_policy_log_probs(dict(re=primary+gate+bout,im=primary+bout,think=gate),'control')
    gradients={}
    for name in ['re','im']:
        grads=torch.autograd.grad(out[name],(gate,primary,bout),allow_unused=True,retain_graph=True)
        values=[0. if g is None else float(g) for g in grads]
        assert values==[0.,1.,1.]
        gradients[name]=dict(zip(['gate','imagined_primary','bout'],values))
    result=dict(status='PASS',cases=rows,logprob_backward_routes=gradients,
        conclusion='Search shaping rewards model-value optimism even when true return is zero; task RL still trains imagined primary and bout choices. Shaping does not directly train the dedicated gate.',
        limitation='Constructed model-error example, not measured error in trained Enduro. No claim optimism is the main failure cause.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
