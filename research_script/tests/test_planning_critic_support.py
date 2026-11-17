import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SPEC=importlib.util.spec_from_file_location('critic_support',
    Path(__file__).parents[1]/'diagnostics/audit_planning_critic_support.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('epsilon',[0.,.02,.1,1.])
def test_reference_moments_match_exact_two_action_derivatives(epsilon):
    # Independent derivative calculation, then enumerate each action exactly.
    p=torch.tensor([0.,.01,.2,.5,.8,.99,1.],dtype=torch.float64)
    d=torch.tensor(.03,dtype=torch.float64,requires_grad=True)
    qc=.7+(1-p)*d;qs=.7-p*d
    jc=torch.stack([torch.autograd.grad(q,d,retain_graph=True)[0] for q in qc])
    js=torch.stack([torch.autograd.grad(q,d,retain_graph=True)[0] for q in qs])
    b=(1-epsilon)*p+epsilon/2
    exact=[(b*jc+(1-b)*js).mean(),(b*jc.abs()+(1-b)*js.abs()).mean(),
           (b*jc.square()+(1-b)*js.square()).mean()]
    result=MODULE.contrast_moments(p.mean().item(),(p*(1-p)).mean().item(),epsilon)
    assert list(result.values())==pytest.approx([v.item() for v in exact],abs=1e-12)


def test_saturated_gate_with_epsilon_does_not_imply_zero_contrast_derivative():
    result=MODULE.contrast_moments(0,0,.1)
    assert result['reference_mean_j_squared']==.05
    assert result['reference_mean_j']==.05
    assert MODULE.contrast_moments(0,0,0)['reference_mean_j_squared']==0


@pytest.mark.parametrize('p',[.01,.5,.99])
def test_actual_learner_gru_loss_has_the_audited_contrast_derivative(p):
    from thinker import util
    from thinker.learn_actor import compute_dynamic_voc_loss
    d=torch.full((1,2),.03,requires_grad=True)
    c=torch.full((1,2),.1,requires_grad=True)
    raw=torch.stack((c+d/2,c-d/2),dim=-1)
    policy=torch.tensor([.9*p,.1*p,1-p]).log().expand(1,2,3)
    selected=torch.tensor([[.8+(1-p)*.03,.8-p*.03]])
    result=compute_dynamic_voc_loss(voc_q=raw,target_control_logits=policy,
        behavior_control_logits=policy,control_action=torch.tensor([[util.PROCEED,util.STOP]]),
        control_valid=torch.ones(1,2,dtype=torch.bool),voc_target=selected+.1,
        mode='shadow',dueling_q=True,voc_state_value=torch.full((1,2),.7),
        critic_parameterization='common_contrast_v1')
    gd,gc=torch.autograd.grad(result.gru_q_loss,(d,c))
    torch.testing.assert_close(gd,torch.tensor([[-.1*(1-p),.1*p]]),atol=1e-6,rtol=0)
    torch.testing.assert_close(gc,torch.zeros_like(gc),atol=1e-7,rtol=0)


def test_invalid_moments_rejected():
    with pytest.raises(ValueError,match='moments'):
        MODULE.contrast_moments(.1,.2,.1)


def test_independently_logged_fp32_means_allow_only_small_roundoff():
    MODULE.contrast_moments(.9999951124191284,5.020830485591432e-6,.02)
    with pytest.raises(ValueError,match='moments'):
        MODULE.contrast_moments(.9999951124191284,.0001,.02)


def test_summaries_weight_by_training_support_and_keep_holdout_separate():
    rows=[]
    for count,p,c,hc in [(1,.1,0,100),(9,.9,9,1)]:
        row={k:0. for k in MODULE.FIELDS}
        row.update({'actor/voc_continue_support':c,'actor/voc_stop_support':count-c,
            'actor/voc_holdout_continue_support':hc,
            'actor/voc_optional_choice_count':count+hc,
            'actor/voc_gate_student_continue_probability':p,
            'actor/voc_gate_student_saturation_jacobian':p*(1-p),
            'actor/voc_optimizer_stepped':1,'gradient/voc_gru_gradient_scale':1})
        MODULE.validate_row(row,1);rows.append(row)
    result=MODULE.summarize(rows,condition='test',first=0,last=1,epsilon=.1)
    assert result['train_continue_fraction']==.9
    assert result['mean_soft_continue_probability']==pytest.approx(.82)
    assert result['holdout_continue_fraction']==1
    assert result['batches_without_continue']==1
    assert result['batches_without_stop']==1
    rows[0]['actor/voc_optional_choice_count']+=1
    with pytest.raises(ValueError,match='partition'):
        MODULE.validate_row(rows[0],1)
