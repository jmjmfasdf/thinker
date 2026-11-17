"""Reweight one observed STOP/CONTINUE choice with online versus EMA gates.

All subsequent choices retain the evaluated EMA policy. This is a conditional
single-choice comparison, not evaluation of an online-gated full policy or
of a training change to the EMA update rate.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.special import expit
import yaml

from summarize_adaptive_planning_evidence import paired_states, matched_choice_advantage


def metrics(states, temperature):
    y=np.array([r['utility'] for r in states])
    task=np.array([r['task_utility'] for r in states])
    counts=np.array([r['count'] for r in states])
    ema=np.array([r['ema_delta'] for r in states])
    online=np.array([r['online_delta'] for r in states])
    p=np.array([r['probability'] for r in states])
    q=expit(online/temperature)
    if not all(np.isfinite(a).all() for a in (y,task,ema,online,p,q)):
        raise ValueError('nonfinite state data')
    ema_allocation,_=matched_choice_advantage(p,y,counts)
    online_allocation,_=matched_choice_advantage(q,y,counts)
    return dict(ema_mean_continue_probability=float(p.mean()),
        online_mean_continue_probability=float(q.mean()),
        ema_gain_over_stop=float(np.mean(p*y)),
        online_gain_over_stop=float(np.mean(q*y)),
        online_minus_ema_gain=float(np.mean((q-p)*y)),
        online_minus_ema_task_gain=float(np.mean((q-p)*task)),
        ema_choice_matched_advantage=ema_allocation,
        online_choice_matched_advantage=online_allocation,
        online_minus_ema_matched_advantage=online_allocation-ema_allocation,
        ema_utility_rmse=float(np.sqrt(np.mean((ema-y)**2))),
        online_utility_rmse=float(np.sqrt(np.mean((online-y)**2))),
        online_minus_ema_utility_rmse=float(np.sqrt(np.mean((online-y)**2))-np.sqrt(np.mean((ema-y)**2))))


def summarize(states, temperature, bootstrap=4000):
    expected={(s,i,c) for s in (8301,8401) for i in range(8) for c in (1,3,6,12)}
    keys=[(r['seed'],r['stream'],r['count']) for r in states]
    if len(keys)!=64 or set(keys)!=expected or any(r['repetitions']!=2 for r in states):
        raise ValueError('requires all64states and128paired repetitions')
    point=metrics(states,temperature)
    groups=sorted({key[:2] for key in keys})
    grouped=[[r for r in states if (r['seed'],r['stream'])==g] for g in groups]
    rng=np.random.default_rng(20261115)
    draws={k:[] for k in point}
    for _ in range(bootstrap):
        sample=[r for i in rng.integers(0,len(groups),len(groups)) for r in grouped[i]]
        values=metrics(sample,temperature)
        for key,value in values.items():draws[key].append(value)
    return dict(states=64,paired_repetitions=128,independent_streams=16,metrics=point,
        descriptive_cluster_ci95={key:np.quantile(value,[.025,.975]).tolist() for key,value in draws.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.base.resolve();sources={}
    def bind(path):
        path=Path(path).resolve();sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return path
    def read(path):return json.loads(bind(path).read_text())
    conditions=dict(burn8ln='adaptive_planning_encoder_burn8_matched_20261114c',
                    burn16ln='adaptive_planning_encoder_stability_20261114c')
    args.output.mkdir(parents=True,exist_ok=False)
    plan=dict(created_at=time.time(),conditions=list(conditions),scope=__doc__,
        target='Discounted clippedtask reward minus remainingcurrentstagecost.',
        comparison='Change only the designated intervention gate probability to sigmoid(onlineDelta/Qtemperature); all subsequent choices use the originalEMApolicy.',
        sampling='All64states,counts1/3/6/12,seeds8301/8401,prefix128,batch8,repetitions2.',
        uncertainty='Average repetitions first,bootstrap16whole seed/streamgroups withallfour counts.4000draws,seed20261115.',
        restrictions='No outcome-based subset,temperature tuning or full-policy improvement claim.')
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    reports={};state_rows=[];table=[]
    for condition,name in conditions.items():
        root=base/name;folder=root/f'evaluation_{condition}'
        summary=read(folder/'summary.json')
        if summary['status']!='COMPLETE':raise ValueError('incomplete utility')
        for info in summary['checkpoint'].values():
            if hashlib.sha256(bind(info['path']).read_bytes()).hexdigest()!=info['sha256']:
                raise ValueError('checkpoint changed')
        flags=yaml.safe_load(bind(root/condition/'config.yaml').read_text())
        if flags['voc_gate_temperature']!=1 or not flags['voc_gate_exact_projection']:
            raise ValueError('unexpected gate formula')
        temperature=flags['voc_gate_q_temperature'];pairs=read(folder/'pairs.json');arms={}
        for row in pairs:
            key=row['seed'],row['count'],row['repetition']
            if key not in arms:arms[key]=read(folder/('arms_%s_%s_%s.json'%key))
        states=paired_states(pairs,arms)
        if any(r['intervention_real_decision_index']!=128 for r in states):
            raise ValueError('unregistered prefix')
        for row in states:
            if not np.isclose(expit(row['ema_delta']/temperature),row['probability'],rtol=0,atol=2e-5):
                raise ValueError('saved EMAprobability does not follow registered formula')
            state_rows.append(dict(condition=condition,**row,
                online_probability=float(expit(row['online_delta']/temperature))))
        report=summarize(states,temperature);report['checkpoint']=summary['checkpoint'];reports[condition]=report
        for metric,value in report['metrics'].items():
            lo,hi=report['descriptive_cluster_ci95'][metric]
            table.append(dict(condition=condition,metric=metric,value=value,ci_low=lo,ci_high=hi))
    for filename,rows in [('summary.csv',table),('states.csv',state_rows)]:
        with (args.output/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(status='COMPLETE_SINGLE_CHOICE_REWEIGHTING',completed_at=time.time(),reports=reports,
        sources_sha256=sources,limitations=[
            'Single-choice conditional effect atforcedcountstates; suffix and state distribution remain originalEMApolicy.',
            'Policyfrequency can change. Choice-matched advantage controls meanCONTINUE probability percount,not totalcomputation.',
            'Utility estimates are noisy;16streams andone trainedseed percondition. Bootstrap holds checkpointsfixed.',
            'Online andEMA critics were trained under exploration; epsilon-zero utility is a policy-shifted target.',
            'No claim that changingEMA tau improves learning or full-episode returns.'])
    (args.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    source=args.output/'analysis_sources';source.mkdir()
    for name in [Path(__file__).name,'summarize_adaptive_planning_evidence.py']:
        shutil.copy2(Path(__file__).parent/name,source/name)
    print(json.dumps({c:{'gain':r['metrics']['online_minus_ema_gain'],
        'ci95':r['descriptive_cluster_ci95']['online_minus_ema_gain']} for c,r in reports.items()}))


if __name__=='__main__':main()
