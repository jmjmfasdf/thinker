"""Read-only structural diagnostics for adaptive computation learning.

Full-population learner counters and the deterministic stream-0 control sample
are kept separate. No return evaluation, checkpoint update, or job submission.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'thinker'))
from thinker import util
from thinker.learn_actor import compute_dynamic_voc_loss


def snapshot(path):
    with path.open('rb') as f:
        blob = f.read(path.stat().st_size)
    n = blob.rfind(b'\n')+1
    return blob[:n], dict(path=str(path), complete_bytes=n,
                         omitted_tail_bytes=len(blob)-n,
                         sha256=hashlib.sha256(blob[:n]).hexdigest())


def q_derivative_probe():
    """Same TD residual; demonstrate the exact centered-advantage derivative.

    This is a mathematical property, not a measured training-gradient effect.
    It does not imply a biased optimum: rare opposite choices supply gradients.
    """
    rows=[]
    for probability in [.5, .9, .99]:
        for choice in [util.PROCEED, util.STOP]:
            p=torch.tensor(probability)
            logits=torch.stack([p/2,p/2,1-p]).log().view(1,1,3)
            delta=.01*torch.log(p/(1-p))
            q=torch.tensor([float(delta/2),float(-delta/2)]).view(1,1,2).requires_grad_(True)
            value=torch.zeros(1,1,requires_grad=True)
            selected=(1-p)*delta if choice==util.PROCEED else -p*delta
            result=compute_dynamic_voc_loss(voc_q=q,target_control_logits=logits,
                behavior_control_logits=logits,control_action=torch.tensor([[choice]]),
                control_valid=torch.ones(1,1,dtype=torch.bool),voc_target=(selected+.1).view(1,1),
                mode='shadow',dueling_q=True,voc_state_value=value,
                gate_policy_schema_version=util.VOC_GATE_POLICY_EXACT_PROJECTION_SCHEMA_VERSION)
            actual,base_grad=torch.autograd.grad(result.q_loss,(q,value),allow_unused=True)
            expected=torch.tensor([-(1-probability),1-probability])*.1 if choice==util.PROCEED else torch.tensor([probability,-probability])*.1
            torch.testing.assert_close(actual.flatten(),expected,atol=1e-7,rtol=1e-5)
            assert base_grad is None
            rows.append(dict(p_continue=probability,choice='continue' if choice==util.PROCEED else 'stop',
                             td_residual=.1,gradient_q_continue=float(actual[0,0,0]),
                             gradient_q_stop=float(actual[0,0,1]),norm=float(actual.norm())))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    cfg_blob=(a.run/'config_c.yaml').read_bytes();cfg=yaml.safe_load(cfg_blob)
    (a.output/'config_snapshot.yaml').write_bytes(cfg_blob)
    keys=['real_step','step','search/mean_steps','search/wait_fraction','search/real_transition_fraction',
          'search/forced_stop_count','search/stage_end_count','search/mean_continue_probability',
          'actor/voc_q_loss','actor/voc_td_rmse','actor/voc_holdout_td_rmse','actor/voc_delta_q',
          'actor/voc_gate_delta_q','actor/voc_continue_support','actor/voc_stop_support',
          'actor/voc_holdout_count','actor/voc_think_target','actor/voc_task_target',
          'actor/think_baseline_loss','actor/baseline_loss','actor/im_baseline_loss','actor/im_pg_loss',
          'actor/icopro_nll','actor/voc_gate_teacher_confidence','actor/voc_gate_teacher_student_gap_abs',
          'actor/voc_gate_projection_post_error_norm']
    bins=['0','1','2_3','4_7','8_15','16_cap']
    keys += [f'search/budget_bin_{b}_count' for b in bins]
    blob,log_meta=snapshot(a.run/'logs.csv')
    d=pd.read_csv(io.BytesIO(blob),comment='#',usecols=lambda c:c in keys)
    assert d.real_step.is_monotonic_increasing and not d.real_step.duplicated().any()
    # The real_step increments count learner-consumed real transitions.
    d['real_actions_per_stream_unroll']=d.real_step.diff().fillna(d.real_step.iloc[0])/cfg['actor_batch_size']
    assert (d.step.diff().dropna()==cfg['actor_unroll_len']*cfg['actor_batch_size']).all()
    d.to_csv(a.output/'learner_snapshot.csv',index=False)
    blob,control_meta=snapshot(a.run/'control_samples.jsonl')
    controls=pd.read_json(io.BytesIO(blob),lines=True)
    controls=controls[controls.real_step<=d.real_step.max()].copy()
    controls.to_csv(a.output/'control_sample_snapshot.csv',index=False)
    row_summaries=[];control_summaries=[];decisions=[]
    for key,g in controls[controls.valid].groupby(['attempt','stream','decision']):
        g=g.sort_values('count')
        assert not g['count'].duplicated().any()
        last=g.iloc[-1]
        k=int(last['count'])+int(last.control!=util.STOP)
        if last.control!=util.STOP and k!=cfg['max_search_steps']:
            continue  # incomplete last sampled decision
        opt=g[g.gate_choice_valid & ~g.forced]
        if not len(opt):continue
        decisions.append(dict(attempt=int(key[0]),stream=int(key[1]),decision=int(key[2]),
            real_step=int(last.real_step),length=k,p_first=float(opt.iloc[0].p_continue),
            p_last=float(last.p_continue),last_control=int(last.control),
            all_optional_ema_delta_positive=bool((opt.ema_delta_q>0).all())))
    dec=pd.DataFrame(decisions);dec.to_csv(a.output/'sampled_decisions.csv',index=False)
    last=int(d.real_step.max())
    windows=[(0,1000000),(1000000,2000000),(2000000,3000000),(3000000,4000000),
             (4000000,5000000),(5000000,6000000),(6000000,last+1)]
    for lo,hi in windows:
        x=d[(d.real_step>=lo)&(d.real_step<hi)]
        if not len(x):continue
        r=dict(start=lo,end=min(last,hi),n_updates=len(x))
        r.update({f'{k}_median':float(v) for k,v in x.drop(columns=['step','real_step']).median().items()})
        total=x['search/stage_end_count'].sum()
        r['forced_stop_fraction_pooled']=float(x['search/forced_stop_count'].sum()/total)
        histogram_total=x[[f'search/budget_bin_{b}_count' for b in bins]].sum().sum()
        r['length1_fraction_pooled']=float(x['search/budget_bin_1_count'].sum()/histogram_total)
        r['completed_stage_support']=float(total)
        r['budget_histogram_support']=float(histogram_total)
        row_summaries.append(r)
        x=controls[(controls.real_step>=lo)&(controls.real_step<hi)&controls.gate_choice_valid&~controls.forced]
        first=x[x['count']==1]
        stage=dec[(dec.real_step>=lo)&(dec.real_step<hi)]
        long=stage[stage.length>=8]
        long_stopped=long[long.last_control==util.STOP]
        pred=.01+.98/(1+np.exp(-np.clip(x.ema_delta_q.to_numpy()/.01,-700,700)))
        control_summaries.append(dict(start=lo,end=min(last,hi),optional_rows=len(x),
            first_optional_decisions=len(first),complete_decisions=len(stage),
            query_saturated_fraction=float(((x.p_continue<.05)|(x.p_continue>.95)).mean()),
            first_p_below_005_fraction=float((first.p_continue<.05).mean()),
            first_p_above_095_fraction=float((first.p_continue>.95).mean()),
            online_ema_sign_disagreement=float((np.sign(x.online_delta_q)!=np.sign(x.ema_delta_q)).mean()),
            gate_projection_max_error=float(np.max(np.abs(pred-x.p_continue))),
            length1_fraction=float((stage.length==1).mean()),cap_fraction=float((stage.length==cfg['max_search_steps']).mean()),
            long_decisions=len(long),long_all_positive_fraction=float(long.all_optional_ema_delta_positive.mean()),
            long_learned_stop_decisions=len(long_stopped),
            long_learned_stop_p_continue_median=float(long_stopped.p_last.median())))
    pd.DataFrame(row_summaries).to_csv(a.output/'learner_intervals.csv',index=False)
    pd.DataFrame(control_summaries).to_csv(a.output/'control_intervals.csv',index=False)
    derivative=q_derivative_probe()
    pd.DataFrame(derivative).to_csv(a.output/'dueling_selected_gradient_probe.csv',index=False)
    source_files=['actor_net.py','learn_actor.py','self_play.py','cenv.pyx','dynamic_imitation.py',
                  'learn_model.py','model_net.py','voc_gru_gradient.py','voc_training_metrics.py','core/rnn.py']
    frozen=ROOT/'research_script/outputs/00/enduro_sub001_all_min1_cost0001_20261111/source/thinker/thinker'
    source={}
    for f in source_files:
        path=ROOT/'thinker/thinker'/f
        h=hashlib.sha256(path.read_bytes()).hexdigest()
        source[f]=dict(path=str(path),sha256=h,frozen_identical=(frozen/f).exists() and hashlib.sha256((frozen/f).read_bytes()).hexdigest()==h)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),run=str(a.run),first_step=int(d.real_step.min()),
        last_step=last,learner_updates=len(d),log=log_meta,controls=control_meta,sources=source,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Current Enduro min1/cost0.0001/tau0.01/episode-memory run; not the old 72-fold models.',
        control_sample_design='All queries for stream 0 every 128th decision. Stream 0 is held out of direct VOC regression, but not shared actor training. Not population-independent trials.',
        q_probe='Six selected-action Smooth-L1 derivative cases match analytical centered-advantage derivatives; baseline detached.',
        cautions=['Scalar Q TD RMSE is not the error of DeltaQ and is not divided by DeltaQ SD here.',
                  'Learner real-transition counts are rollout exposure, not a hard maximum reward horizon; bootstrap extends credit.',
                  'Gate saturation or variable length does not establish utility or failure.',
                  'No training/job/checkpoint modifications or game-score evaluation.'])
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Last step:',last,'learner updates:',len(d))
    print(pd.DataFrame(row_summaries)[['start','end','search/wait_fraction_median',
        'real_actions_per_stream_unroll_median','length1_fraction_pooled','forced_stop_fraction_pooled']].to_string(index=False))
    print(pd.DataFrame(control_summaries).to_string(index=False))
    print('Frozen source matches:',all(v['frozen_identical'] for v in source.values()))
    print('Selected-action derivative checks: PASS (6 cases)')


if __name__=='__main__':main()
