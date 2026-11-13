"""Counterexample for a detached, policy-centered, one-contrast VOC critic.

Artificial targets isolate representational/optimization behavior. This does
NOT establish that the production checkpoint has learned a spurious contrast.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'thinker'))
from thinker import util
from thinker.learn_actor import compute_dynamic_voc_loss


def evaluate(p,b,delta,baseline=0.,common=0.,common_mode=False):
    count=100
    control=torch.full((1,count),util.STOP,dtype=torch.long)
    control[:,:round(b*count)]=util.PROCEED
    def logits(prob):
        return torch.tensor([prob/2,prob/2,1-prob]).log().view(1,1,3).expand(1,count,3)
    d=torch.tensor(float(delta),requires_grad=True)
    c=torch.tensor(float(common),requires_grad=True)
    q=torch.stack([c+d/2,c-d/2]).view(1,1,2).expand(1,count,2)
    result=compute_dynamic_voc_loss(voc_q=q,target_control_logits=logits(p),
        behavior_control_logits=logits(b),control_action=control,
        control_valid=torch.ones_like(control,dtype=torch.bool),
        voc_target=torch.full((1,count),.1),mode='shadow',dueling_q=True,
        voc_state_value=torch.full((1,count),baseline),
        gate_policy_schema_version=(util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION if common_mode
                                   else util.VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION))
    mean_loss=result.q_loss/count
    derivative,=torch.autograd.grad(mean_loss,d)
    return dict(p_current=p,p_behavior=b,identical_target_for_both_actions=.1,
                detached_baseline=baseline,contrast=float(d),common_correction=common,
                common_mode=common_mode,loss=float(mean_loss),d_loss_d_contrast=float(derivative))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[]
    for current,behavior in [(.5,.5),(.9,.9),(.9,.99),(.99,.9)]:
        optimum=(behavior-current)*.1/(behavior*(1-current)**2+(1-behavior)*current**2)
        start=evaluate(current,behavior,0.)
        end=evaluate(current,behavior,optimum)
        assert abs(end['d_loss_d_contrast'])<1e-6
        assert end['loss'] <= start['loss']+1e-7
        rows.extend([dict(case='centered_delta_zero',**start),dict(case='centered_optimum',**end)])
        correct=evaluate(current,behavior,0.,baseline=.1)
        flexible=evaluate(current,behavior,0.,common=.1,common_mode=True)
        assert correct['loss']<1e-10 and flexible['loss']<1e-10
        rows.extend([dict(case='correct_baseline_control',**correct),dict(case='common_mode_control',**flexible)])
    result=dict(status='PASS',cases=rows,
        finding='With baseline error and different behavior/current action probabilities, selected-action regression can prefer nonzero DeltaQ even with identical targets for both choices.',
        limitation='Synthetic fixed-policy loss slices. Real targets and policies move during learning; occurrence and magnitude in training require rowwise telemetry.',
        fix_not_applied='Common-mode or Q-regression importance weighting are hypotheses, not validated training modifications.')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
