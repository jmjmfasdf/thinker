"""Offline critic controls and count-conditioned gate support; no model changes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'thinker'))
from thinker import util
from thinker.learn_actor import compute_dynamic_voc_loss


def critic_controls(out):
    rows=[]
    for p,b in [(.01,.01),(.01,.02),(.1,.2),(.5,.6),(.9,.8),(.99,.98),(.99,.99)]:
        for e in [-.1,0.,.1]:
            for true_d in [0.,.02]:
                ca=torch.tensor([[util.PROCEED,util.STOP]])
                target=torch.tensor([[e+(1-p)*true_d,e-p*true_d]])
                weights=torch.tensor([[b,1-b]])
                def logits(prob):
                    return torch.tensor([prob/2,prob/2,1-prob]).log().reshape(1,1,3).expand(1,2,3)
                leak=(b-p)*e/(b*(1-p)**2+(1-b)*p*p)
                for mode,delta,common in [('original',true_d+leak,0.),('importance',true_d,0.),('common_mode',true_d,e)]:
                    d=torch.tensor(delta,requires_grad=True);c=torch.tensor(common,requires_grad=True)
                    q=torch.stack((c+d/2,c-d/2)).reshape(1,1,2).expand(1,2,2)
                    result=compute_dynamic_voc_loss(voc_q=q,target_control_logits=logits(p),
                        behavior_control_logits=logits(b),control_action=ca,control_valid=torch.ones_like(ca).bool(),
                        voc_target=target,mode='shadow',dueling_q=True,voc_state_value=torch.zeros_like(target),
                        gate_policy_schema_version=(util.VOC_GATE_POLICY_HUBER_COMMON_Q_SCHEMA_VERSION if mode=='common_mode' else 4))
                    lr=torch.nn.functional.smooth_l1_loss(result.selected_q,result.target,reduction='none')
                    if mode=='importance':weights_use=torch.tensor([[p,1-p]])
                    else:weights_use=weights
                    loss=(lr*weights_use).sum()
                    dg,=torch.autograd.grad(loss,d,retain_graph=True)
                    assert abs(float(dg))<1e-6
                    if mode=='common_mode':
                        cg,=torch.autograd.grad(loss,c);assert abs(float(cg))<1e-6
                    rows.append(dict(p_current=p,p_behavior=b,common_error=e,true_delta=true_d,mode=mode,
                        fitted_delta=delta,contrast_error=delta-true_d,derivative=float(dg),
                        balanced_q_rmse=float((result.selected_q-target).square().mean().sqrt())))
    pd.DataFrame(rows).to_csv(out/'critic_population_controls.csv',index=False)
    # Synthetic finite-sample sensitivity with known contrast and common error.
    # Free common-mode fit is undefined when one action has no observations.
    rng=np.random.default_rng(20261114);sim=[]
    for p,b in [(.01,.01),(.01,.02),(.1,.1),(.5,.5),(.99,.99),(.99,.98)]:
        for n in [32,256,1024]:
            count=2000;c=rng.random((count,n))<b;x=c.astype(float)-p
            target=.1+x*.02+rng.normal(0,.05,(count,n))
            original=(x*target).sum(1)/(x*x).sum(1)
            w=np.where(c,p/b,(1-p)/(1-b))
            importance=(w*x*target).sum(1)/(w*x*x).sum(1)
            nc=c.sum(1);ns=n-nc;valid=(nc>0)&(ns>0)
            common=np.full(count,np.nan)
            common[valid]=(target*c).sum(1)[valid]/nc[valid]-(target*~c).sum(1)[valid]/ns[valid]
            for mode,d in [('original',original),('importance',importance),('common_mode',common)]:
                good=np.isfinite(d)
                sim.append(dict(p_current=p,p_behavior=b,sample_size=n,repetitions=count,mode=mode,
                    identifiable_fraction=float(good.mean()),both_choices_fraction=float(valid.mean()),
                    mean_contrast_error=float(np.mean(d[good]-.02)),contrast_rmse=float(np.sqrt(np.mean((d[good]-.02)**2))),
                    correct_positive_sign_fraction=float(np.mean(d[good]>0)),
                    noiseless_true_delta=.02,common_error=.1,noise_sd=.05))
    pd.DataFrame(sim).to_csv(out/'critic_sampling_controls.csv',index=False)


def snapshot(path):
    with path.open('rb') as f:data=f.read(path.stat().st_size)
    return data[:data.rfind(b'\n')+1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);torch.set_num_threads(1)
    critic_controls(a.output)
    blob=snapshot(a.run/'control_samples.jsonl');d=pd.read_json(io.BytesIO(blob),lines=True)
    d=d[d.valid].copy();d.to_csv(a.output/'controls_snapshot.csv',index=False)
    q=d[d.gate_choice_valid & ~d.forced].copy()
    q['interval']=pd.cut(q.real_step,[-1,2e6,4e6,6e6,8e6,np.inf],labels=['0–2M','2–4M','4–6M','6–8M','8–10M'])
    q['count_group']=pd.cut(q['count'],[0,1,3,7,15,29],labels=['1','2–3','4–7','8–15','16–29'])
    q['continue']=q.control!=util.STOP
    q['saturated']=(q.p_continue<.05)|(q.p_continue>.95)
    q['opposite_choice']=q['continue']!=(q.p_continue>=.5)
    q['expected_opposite']=np.minimum(q.p_continue,1-q.p_continue)
    q['contrast_information']=q.p_continue*(1-q.p_continue)
    q['calibration_residual']=q['continue'].astype(float)-q.p_continue
    q['choice_variance']=q.p_continue*(1-q.p_continue)
    rows=[]
    for (interval,count),g in q.groupby(['interval','count_group'],observed=True):
        # Calibration residual descriptive only, observations within stages dependent.
        rows.append(dict(interval=interval,count_group=count,queries=len(g),
            decisions=g[['attempt','stream','decision']].drop_duplicates().shape[0],
            mean_p_continue=g.p_continue.mean(),actual_continue_fraction=g['continue'].mean(),
            saturation_fraction=g.saturated.mean(),observed_opposite=int(g.opposite_choice.sum()),
            expected_opposite=g.expected_opposite.sum(),mean_information=g.contrast_information.mean(),
            information_relative_to_p05=g.contrast_information.mean()/.25,
            mean_execution_calibration_residual=g.calibration_residual.mean()))
    pd.DataFrame(rows).to_csv(a.output/'gate_count_support.csv',index=False)
    q.to_csv(a.output/'optional_rows.csv',index=False)
    alt=[]
    for tau in [.01,.025,.05]:
        for eps in [.02,.1,.2]:
            pc=eps/2+(1-eps)/(1+np.exp(-np.clip(q.ema_delta_q.to_numpy()/tau,-700,700)))
            if tau==.01 and eps==.02:np.testing.assert_allclose(pc,q.p_continue,rtol=1e-5,atol=2e-6)
            for interval in q.interval.cat.categories:
                sel=(q.interval==interval).to_numpy();pp=pc[sel]
                alt.append(dict(interval=interval,temperature=tau,epsilon=eps,queries=len(pp),
                    mean_continue_probability=float(pp.mean()),
                    saturation_fraction=float(((pp<.05-1e-12)|(pp>.95+1e-12)).mean()),
                    expected_opposite_fraction=float(np.minimum(pp,1-pp).mean()),
                    contrast_information=float((pp*(1-pp)).mean())))
    pd.DataFrame(alt).to_csv(a.output/'gate_same_rows_alternatives.csv',index=False)
    stages=[]
    for key,g in d.groupby(['attempt','stream','decision']):
        g=g.sort_values('count');end=g.iloc[-1]
        length=int(end['count'])+int(end.control!=util.STOP)
        if end.control!=util.STOP and length!=30:continue
        opt=g[g.gate_choice_valid & ~g.forced]
        if not len(opt) or int(opt.iloc[0]['count'])!=1:continue
        if not np.array_equal(g['count'].to_numpy(),np.arange(length+int(end.control==util.STOP))):
            raise AssertionError('Incomplete/duplicate decision sequence')
        pc=opt.p_continue.to_numpy();first=pc[0]
        stages.append(dict(real_step=int(end.real_step),attempt=key[0],stream=key[1],decision=key[2],
            length=length,cap=length==30,first_probability=first,first_delta=float(opt.iloc[0].ema_delta_q),
            sign_changed=bool(((pc>.5)!=(first>.5)).any()),stopped=bool(end.control==util.STOP),final_p=float(end.p_continue)))
    s=pd.DataFrame(stages);s.to_csv(a.output/'complete_stages.csv',index=False)
    hist=[]
    for lo,hi in [(0,2e6),(2e6,4e6),(4e6,6e6),(6e6,8e6),(8e6,11e6)]:
        x=s[(s.real_step>=lo)&(s.real_step<hi)];long=x[x.length>=8]
        hist.append(dict(start=lo,end=hi,decisions=len(x),mean_length=x.length.mean(),cap_count=int(x.cap.sum()),
            cap_fraction=x.cap.mean(),sign_change_fraction=x.sign_changed.mean(),long_decisions=len(long),
            long_sign_change_fraction=long.sign_changed.mean(),long_stops=int(long.stopped.sum()),
            long_stop_despite_p095=int((long.stopped & (long.final_p>.95)).sum())))
    pd.DataFrame(hist).to_csv(a.output/'stage_intervals.csv',index=False)
    logs=snapshot(a.run/'logs.csv')
    keys=['real_step','search/wait_fraction','search/real_transition_fraction','search/forced_stop_count',
        'actor/voc_gate_rho','actor/voc_td_rmse','actor/voc_td_bias','actor/voc_continue_support','actor/voc_stop_support']
    ld=pd.read_csv(io.BytesIO(logs),comment='#',usecols=lambda c:c in keys)
    ld.to_csv(a.output/'training_snapshot.csv',index=False)
    ld['interval']=pd.cut(ld.real_step,[-1,2e6,4e6,6e6,8e6,np.inf],labels=['0–2M','2–4M','4–6M','6–8M','8–10M'])
    ld.groupby('interval',observed=True).median(numeric_only=True).to_csv(a.output/'training_intervals.csv')
    manifest=dict(last_training_step=int(ld.real_step.max()),last_control_step=int(d.real_step.max()),
        controls_sha256=hashlib.sha256(blob).hexdigest(),training_sha256=hashlib.sha256(logs).hexdigest(),
        optional_rows=len(q),complete_decisions=len(s),cap_decisions=int(s.cap.sum()),
        assertions=['Population critic stationary points verified through production Q reconstruction and Huber loss.',
            'Stored gate probabilities reproduced from EMA delta at tau=.01 and epsilon=.02.',
            'Completed stage counts consecutive; mandatory decisions excluded from optional support.'],
        limitations=['Stream-0 subsample; within-stage and temporal dependence.',
            'Teacher/gate execution calibration does not measure the reward benefit of computation.',
            'Alternative gate settings evaluated on fixed historical rows, not their changed occupancy.',
            'Critic controls synthetic with known truth; no actual learned contrast bias identified.'])
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
