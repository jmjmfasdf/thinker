"""Numerical structural controls and frozen log audit, excluding direct action shortcuts."""
import argparse
import io
import json
from pathlib import Path
import sys
import hashlib

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'thinker'))
from thinker.learn_actor import compute_dynamic_voc_loss, compute_baseline_loss
from thinker.core.vtrace import compute_v_trace
from thinker import util


def critic(p, b, residual, delta, weighted=False, common_mode=False):
    n=1000
    actions=torch.full((1,n), util.STOP, dtype=torch.long)
    actions[:,:round(b*n)]=util.PROCEED
    def logit(x): return torch.tensor([x/2,x/2,1-x]).log().reshape(1,1,3).expand(1,n,3)
    d=torch.tensor(float(delta),requires_grad=True)
    q=torch.stack((d/2,-d/2)).reshape(1,1,2).expand(1,n,2)
    r=compute_dynamic_voc_loss(voc_q=q,target_control_logits=logit(p),behavior_control_logits=logit(b),
        control_action=actions,control_valid=torch.ones_like(actions).bool(),voc_target=torch.full((1,n),residual),
        mode='shadow',dueling_q=True,voc_state_value=torch.zeros(1,n),
        gate_policy_schema_version=4)
    loss=r.q_loss/n
    if weighted:
        # Deliberately untruncated importance weights: a diagnostic control,
        # not a proposed ready-to-use training patch.
        weight=torch.where(actions==util.STOP, torch.tensor((1-p)/(1-b)), torch.tensor(p/b))
        selected=r.q_values.gather(-1,r.gate_action.unsqueeze(-1)).squeeze(-1)
        loss=(torch.nn.functional.smooth_l1_loss(selected,r.target,reduction='none')*weight).mean()
    g,=torch.autograd.grad(loss,d)
    return dict(p_current=p,p_behavior=b,common_error=residual,delta=delta,weighted=weighted,loss=float(loss),gradient=float(g))


def controls():
    rows=[]
    for p,b in [(.5,.5),(.9,.9),(.9,.99),(.99,.9),(.01,.02)]:
        for e in [-.1,.1]:
            d=(b-p)*e/(b*(1-p)**2+(1-b)*p*p)
            a=critic(p,b,e,0.); z=critic(p,b,e,d); w=critic(p,b,e,0.,weighted=True)
            assert abs(z['gradient'])<1e-6 and abs(w['gradient'])<1e-6
            assert abs(a['gradient'] + (b-p)*e)<1e-6
            rows.extend([dict(case='uncorrected_zero',**a),dict(case='uncorrected_optimum',**z),dict(case='importance_control',**w)])
    # WAIT identity rows preserve targets at lambda=1, but add value-loss weight.
    waits=[]
    for count in [0,1,10,30]:
        theta=torch.tensor(.1,requires_grad=True)
        values=theta*torch.ones(count+2,1)
        r=torch.zeros_like(values);r[-1]=1.
        discounts=torch.ones_like(values);discounts[-1]=.99
        v=compute_v_trace(torch.zeros_like(values), discounts,r,values,torch.tensor([.3]),-1)
        assert abs(float(v.vs[0])-1.297)<1e-5
        wait_mask=torch.zeros_like(values);wait_mask[1:1+count]=1
        loss=compute_baseline_loss(values,v.vs,wait_mask)
        grad,=torch.autograd.grad(loss,theta)
        waits.append(dict(wait_rows=count,active_target=float(v.vs[0]),wait_loss=float(loss),wait_shared_value_gradient=float(grad)))
    # Zero environment time does not remove importance-weight attenuation.
    attenuation=[]
    for steps in [1,7,15,30]:
        for rho in [1.,.99,.95,.9]:
            rewards=torch.zeros(steps+1,1);rewards[-1]=1.
            log_rhos=torch.full_like(rewards,float(np.log(rho)));log_rhos[-1]=0.
            v=compute_v_trace(log_rhos,torch.ones_like(rewards),rewards,torch.zeros_like(rewards),torch.zeros(1),-1)
            assert abs(float(v.vs[0])-rho**steps)<1e-5
            attenuation.append(dict(imagined_choices=steps,rho_per_choice=rho,reward_correction_at_start=float(v.vs[0])))
    return rows,waits,attenuation


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    rows,waits,att=controls()
    pd.DataFrame(rows).to_csv(a.output/'critic_bias_controls.csv',index=False)
    pd.DataFrame(waits).to_csv(a.output/'wait_controls.csv',index=False)
    pd.DataFrame(att).to_csv(a.output/'vtrace_attenuation_controls.csv',index=False)
    path=a.run/'logs.csv'
    with path.open('rb') as f: blob=f.read(path.stat().st_size)
    blob=blob[:blob.rfind(b'\n')+1]
    keys=['real_step','step','actor/voc_gate_rho','actor/voc_td_bias','actor/voc_td_mae','actor/voc_td_rmse',
          'actor/voc_gate_holdout_td_bias','actor/voc_gate_holdout_td_rmse','actor/voc_q_loss',
          'actor/voc_continue_support','actor/voc_stop_support','search/wait_fraction',
          'search/real_transition_fraction','actor/im_pg_loss','actor/im_baseline_loss','actor/pg_loss',
          'actor/baseline_loss','actor/icopro_scaled_loss','actor/think_baseline_loss']
    d=pd.read_csv(io.BytesIO(blob),comment='#',usecols=lambda c:c in keys)
    d.to_csv(a.output/'training_log_snapshot.csv',index=False)
    d['interval']=np.minimum((d.real_step//1000000).astype(int),6)
    summaries=d.groupby('interval').median(numeric_only=True)
    summaries.to_csv(a.output/'training_intervals.csv')
    cp=a.run/'control_samples.jsonl'
    with cp.open('rb') as f: cb=f.read(cp.stat().st_size)
    cb=cb[:cb.rfind(b'\n')+1]
    c=pd.read_json(io.BytesIO(cb),lines=True)
    c=c[(c.real_step<=d.real_step.max()) & c.valid]
    c.to_csv(a.output/'controls_snapshot.csv',index=False)
    decisions=[]
    for key,g in c.groupby(['attempt','stream','decision']):
        g=g.sort_values('count');end=g.iloc[-1]
        length=int(end['count'])+int(end.control!=util.STOP)
        if end.control!=util.STOP and length!=30:continue
        opt=g[g.gate_choice_valid & ~g.forced]
        if not len(opt) or int(opt.iloc[0]['count'])!=1:continue
        p0=float(opt.iloc[0].p_continue)
        chosen=opt.control.to_numpy()!=util.STOP
        p=opt.p_continue.to_numpy()
        likelihood=np.log(np.where(chosen,p,1-p)).sum()
        frozen_ll=np.log(np.where(chosen,p0,1-p0)).sum()
        decisions.append(dict(real_step=int(end.real_step),attempt=key[0],stream=key[1],decision=key[2],length=length,
            first_probability=p0,actual_cap=length==30,frozen_probability_cap=p0**29,
            frozen_expected_length=(1-p0**30)/(1-p0),
            sequential_ll=likelihood,frozen_ll=frozen_ll,ll_gain=likelihood-frozen_ll,
            sign_changed=bool(((p>.5)!=(p0>.5)).any()),optional_rows=len(opt),
            final_p=float(end.p_continue),ended_stop=end.control==util.STOP))
    s=pd.DataFrame(decisions);s.to_csv(a.output/'stage_hazard.csv',index=False)
    hazard=[]
    for lo,hi in [(0,1e6),(1e6,2e6),(2e6,3e6),(3e6,4e6),(4e6,5e6),(5e6,6e6),(6e6,1e8)]:
        x=s[(s.real_step>=lo)&(s.real_step<hi)];long=x[x.length>=8]
        hazard.append(dict(start=lo,end=min(int(d.real_step.max()),hi),decisions=len(x),
            observed_mean_length=x.length.mean(),frozen_expected_mean_length=x.frozen_expected_length.mean(),
            observed_cap_fraction=x.actual_cap.mean(),frozen_expected_cap_fraction=x.frozen_probability_cap.mean(),
            stage_sign_change_fraction=x.sign_changed.mean(),long_stage_sign_change_fraction=long.sign_changed.mean(),
            mean_sequential_logscore_gain=x.ll_gain.mean(),long_decisions=len(long),
            long_stop_count=int(long.ended_stop.sum()),long_stop_with_p_above_095=int((long.ended_stop & (long.final_p>.95)).sum())))
    pd.DataFrame(hazard).to_csv(a.output/'hazard_intervals.csv',index=False)
    metadata=dict(last_real_step=int(d.real_step.max()),updates=len(d),learner_log_sha256=hashlib.sha256(blob).hexdigest(),
        control_log_sha256=hashlib.sha256(cb).hexdigest(),scope='1,2,3,4,6,7; item5 excluded',
        tests='30 critic controls; 4 WAIT controls; 16 V-trace controls; assertions passed.',
        limitations=['Gate rho summaries do not identify the magnitude or sign of common-error leakage.',
                     'V-trace attenuation is a controlled numerical example, not measured actual trace products.',
                     'Hazard control fixes first optional probability; no reward/utility optimality claim.',
                     'Control rows are a stream-0 subsample, not all-stream independent decisions.'])
    (a.output/'manifest.json').write_text(json.dumps(metadata,indent=2))
    print(summaries[['actor/voc_gate_rho','actor/voc_td_bias','actor/voc_td_mae','actor/voc_td_rmse']].to_string())
    print(pd.DataFrame(hazard).to_string(index=False));print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
