"""Rebuild a bounded reward-learning pilot report from immutable evaluations.

Missing results remain incomplete. Confidence intervals resample environment
streams, not training seeds; this single-seed pilot is exploratory.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import io
import json
from pathlib import Path
import numpy as np


CELLS=[(g,m) for g in ('enduro','pong') for m in ('control','projection')]


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def save(path,text):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)


def write_csv(path,rows):
    if not rows:return save(path,'')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    f=io.StringIO();w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    save(path,f.getvalue())


def mean_ci(values):
    x=np.asarray(values,dtype=float)
    if x.size<2:return float(x.mean()),None
    rng=np.random.default_rng(7001)
    ci=np.quantile(x[rng.integers(0,len(x),(4000,len(x)))].mean(1),[.025,.975])
    return float(x.mean()),ci.tolist()


def evidence(ci):
    if ci is None:return 'INSUFFICIENT'
    if ci[0]>0:return 'POSITIVE (exploratory)'
    if ci[1]<0:return 'NEGATIVE (exploratory)'
    return 'INCONCLUSIVE'


def generate(root):
    status=[];scores=[];voc=[];comparisons=[];summaries={};initials={}
    for game,mode in CELLS:
        directory=root/f'{game}_{mode}';s=read(directory/'status.json',{})
        initials[game,mode]=read(directory/'initialization.json',{}).get('initial')
        status.append(dict(game=game,mode=mode,status=s.get('phase','NOT_STARTED'),
            env_steps=s.get('env_steps',0),target=s.get('total_env_steps',100000),
            actor_updates=s.get('actor_updates',0),wandb=s.get('wandb_url',''),
            error=s.get('error',''),evaluation_step=s.get('evaluation_step','')))
        for milestone in read(directory/'milestones.json',[]):
            r=read(Path(milestone['summary']))
            if r is None:continue
            step=milestone['env_steps'];summaries[game,mode,step]=r
            mean,ci=mean_ci(np.array(r['score_raw_per_stream'])/r['args']['score_horizon']*1000)
            scores.append(dict(game=game,mode=mode,env_steps=step,score_raw_per_1000=mean,
                ci95_low=ci[0] if ci else None,ci95_high=ci[1] if ci else None,
                complete_episodes=r['complete_episodes'],mean_complete_episode_return=r['mean_complete_episode_return'],
                computations_per_real_step=r['score_compute_per_real_step'],
                eval_real_steps=r['score_real_steps'],checkpoint=milestone['bundle']))
            if 'voc' in r:
                v=r['voc'];row=dict(game=game,mode=mode,env_steps=step)
                row.update({k:value for k,value in v.items() if k!='state_scores'})
                for key in ('rmse_improvement','allocation_gain_vs_compute_matched_random'):
                    row[key+'_evidence']=evidence(v[key+'_ci95'])
                voc.append(row)
    for (g,m,step),r in summaries.items():
        if step==0 or (g,m,0) not in summaries:continue
        baseline=summaries[g,m,0]
        if r['args']['seed']!=baseline['args']['seed'] or r['args']['score_horizon']!=baseline['args']['score_horizon']:
            raise ValueError('Score evaluation configuration changed')
        mean,ci=mean_ci((np.array(r['score_raw_per_stream'])-baseline['score_raw_per_stream'])/r['args']['score_horizon']*1000)
        comparisons.append(dict(game=g,comparison=m+' minus initialization',env_steps=step,
            mean_score_difference_per_1000=mean,ci95=ci,evidence=evidence(ci)))
        if m=='projection' and (g,'control',step) in summaries:
            control=summaries[g,'control',step]
            mean,ci=mean_ci((np.array(r['score_raw_per_stream'])-control['score_raw_per_stream'])/r['args']['score_horizon']*1000)
            comparisons.append(dict(game=g,comparison='projection minus control',env_steps=step,
                mean_score_difference_per_1000=mean,ci95=ci,evidence=evidence(ci)))
    matching={g:(initials[g,'control']==initials[g,'projection']) if initials[g,'control'] and initials[g,'projection'] else None for g in ('enduro','pong')}
    complete=all(s['status']=='COMPLETE' for s in status)
    result=dict(status='COMPLETE' if complete else 'INCOMPLETE',cells=status,
        identical_initial_weights=matching,scores=scores,voc=voc,comparisons=comparisons,
        scope='One subject and training seed. CIs reflect evaluation streams only; no population or multi-seed inference.')
    save(root/'learning_report.json',json.dumps(result,indent=2,allow_nan=False))
    for name,rows in [('training_status',status),('score_curve',scores),('voc_curve',voc),('reward_comparisons',comparisons)]:
        write_csv(root/(name+'.csv'),rows)
    lines=[f'# VOC reward-learning pilot: {result["status"]}','',result['scope'],'',
        '| Game | Gate | Status | Environment steps | Actor updates | W&B |',
        '|---|---|---|---:|---:|---|']
    for s in status:lines.append(f'| {s["game"]} | {s["mode"]} | {s["status"]} | {s["env_steps"]:,}/{s["target"]:,} | {s["actor_updates"]} | {s["wandb"]} |')
    lines+=['',f'Identical initial actor/world-model weights: {matching}','',
        'Independent scores use fixed real-step windows; completed full-game returns are reported separately.',
        'Reward differences and VOC allocation gains require a positive 95% CI for exploratory positive evidence. An interval crossing zero is inconclusive.',
        'VOC targets use actual clipped task return minus current-stage computation cost, without critic bootstrap. Allocation gain is a local STOP/CONTINUE comparison at matched expected current-stage cost; it is not a full-policy randomized stopping trial.','',
        '| Game | Comparison | Steps | Reward difference / 1000 decisions | 95% CI | Evidence |',
        '|---|---|---:|---:|---|---|']
    for r in comparisons:
        lines.append(f'| {r["game"]} | {r["comparison"]} | {r["env_steps"]:,} | {r["mean_score_difference_per_1000"]:.4f} | {r["ci95"]} | {r["evidence"]} |')
    lines+=['','| Game | Gate | Steps | VOC RMSE | Zero-predictor RMSE | Allocation gain | 95% CI | Evidence |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for r in voc:
        k='allocation_gain_vs_compute_matched_random'
        lines.append(f'| {r["game"]} | {r["mode"]} | {r["env_steps"]:,} | {r["rmse"]:.5f} | {r["zero_predictor_rmse"]:.5f} | {r[k]:.6f} | {r[k+"_ci95"]} | {r[k+"_evidence"]} |')
    save(root/'learning_report.md','\n'.join(lines)+'\n')
    if scores:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(14,7),squeeze=False)
        for i,game in enumerate(('enduro','pong')):
            for mode in ('control','projection'):
                ss=sorted([r for r in scores if r['game']==game and r['mode']==mode],key=lambda r:r['env_steps'])
                vv=sorted([r for r in voc if r['game']==game and r['mode']==mode],key=lambda r:r['env_steps'])
                if ss:
                    y=np.array([r['score_raw_per_1000'] for r in ss]);ci=np.array([[r['ci95_low'],r['ci95_high']] for r in ss])
                    axes[i,0].errorbar([r['env_steps'] for r in ss],y,yerr=np.maximum(0,np.array([y-ci[:,0],ci[:,1]-y])),marker='o',label=mode)
                for j,key in [(1,'rmse_improvement'),(2,'allocation_gain_vs_compute_matched_random')]:
                    if vv:
                        y=np.array([r[key] for r in vv]);ci=np.array([r[key+'_ci95'] for r in vv])
                        axes[i,j].errorbar([r['env_steps'] for r in vv],y,yerr=np.maximum(0,np.array([y-ci[:,0],ci[:,1]-y])),marker='o',label=mode)
            for j,title in enumerate(('Raw score / 1000 decisions','VOC RMSE improvement over zero','Local allocation gain over matched random')):
                ax=axes[i,j];ax.set_title(game+': '+title);ax.set_xlabel('Training environment steps');ax.grid(alpha=.2)
                if j:ax.axhline(0,color='gray',linewidth=.8)
                if ax.get_legend_handles_labels()[0]:ax.legend()
        fig.suptitle('Exploratory pilot — '+result['status']);fig.tight_layout()
        fig.savefig(root/'learning_curves.png',dpi=150);plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'.report.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);generate(args.root)


if __name__=='__main__':main()
