"""Milestone paired VOC utility evaluation; separate game scoring is disabled."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from reward_probe import load_checkpoints
from gru_ablation_eval import paired_summary
from gru_reward_readout import arm_rollout, verify
from train_pilot import atomic_json


def score_metrics(score):
    completed=[x for stream in score['completed_episode_returns'] for x in stream]
    hist=np.asarray(score['budget_histogram'])
    return dict(raw_reward_per_1000_actions=float(np.mean(score['score_per_1000_actions'])),
        computations_per_action=score['mean_computations_per_action'],
        zero_search_fraction=float(hist[0]/hist.sum()),
        completed_episode_count=len(completed),
        completed_episode_return_mean=float(np.mean(completed)) if completed else None,
        streams_without_completed_episode=sum(not x for x in score['completed_episode_returns']),
        evaluation_real_actions=score['real_actions_per_stream']*len(score['raw_window']))


def evaluate(task, output):
    if task['kind'] != 'benefit':
        raise ValueError('Score evaluation disabled: only paired VOC benefit is enabled')
    args=argparse.Namespace(bundle=Path(task['bundle']),checkpoint=None,threads=1)
    flags,actor,model,binding=load_checkpoints(args)
    steps=task['real_step']
    progress=lambda phase,**kw:atomic_json(output/'status.json',dict(phase=phase,updated_at=time.time(),**kw))
    smoke=task.get('smoke',False)
    rows=[];ema=actor['voc_ema_gate_head_state_dict']
    ew=ema['weight'].double().cpu().numpy();eb=ema['bias'].double().cpu().numpy()
    # At the final two main milestones broaden to 16 independent streams,
    # four entry strata and four continuation replicates = 256 paired returns.
    seeds=[33101,43101] if steps>=5_000_000 and not smoke else [33101]
    reps=1 if smoke else (4 if steps>=5_000_000 else 2)
    prefixes=[8] if smoke else [128,1024]
    counts=[0] if smoke else [0,3]
    for seed in seeds:
      for prefix in prefixes:
       for k in counts:
        for rep in range(reps):
            progress('PAIRED',seed=seed,prefix=prefix,count=k,rep=rep,pairs=len(rows))
            kwargs=dict(seed=seed,repetition=rep,prefix=prefix,targets=np.full(8,k),
                horizon=32 if smoke else 1536,device='cuda:0')
            arms=[]
            for mode in ('stop','continue'):
                result=arm_rollout(flags,actor,model,arm=mode,**kwargs)
                arms.append(result)
                atomic_json(output/f'pair_s{seed}_p{prefix}_k{k}_r{rep}_{mode}.json',result)
            verify(*arms)
            if rep==0:
                repeat=arm_rollout(flags,actor,model,arm='stop',**kwargs)
                verify(arms[0],repeat)
                if repeat['returns']!=arms[0]['returns']:raise AssertionError('STOP repeat differs')
            for i,(entry,stop,cont) in enumerate(zip(arms[0]['entries'],arms[0]['returns'],arms[1]['returns'])):
                q=np.asarray(entry['features'])@ew.T+eb
                rows.append(dict(seed=seed,stream=i,prefix=prefix,count=k,repetition=rep,
                    entry_hash=entry['hash'],delta=cont['net_truncated']-stop['net_truncated'],
                    task_delta=cont['task_truncated']-stop['task_truncated'],
                    cost=cont['current_stage_cost']-stop['current_stage_cost'],online=entry['delta'],
                    ema=float(q[0]-q[1]),probability=entry['p_continue'],
                    tail_bound=arms[0]['tail_bounds'][i]+arms[1]['tail_bounds'][i]))
            atomic_json(output/'pairs.json',rows)
    voc=paired_summary(rows)
    metrics={**{'voc/'+k:v for k,v in voc.items() if isinstance(v,(int,float)) or v is None}}
    return dict(status='COMPLETE',task=task,metrics=metrics,voc=voc,budget_comparisons={},
        score_evaluation='DISABLED_BY_USER',evaluation_mode='paired_utility_only',
        checkpoint=binding,scope='actual discounted reward minus current-stage cost; finite-horizon tails bounded; one training seed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--task',type=Path,required=True)
    args=p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('SLURM GPU required')
    task=json.loads(args.task.read_text());output=Path(task['output']);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'evaluation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'summary.json').exists():
        if json.loads((output/'summary.json').read_text())['status']=='COMPLETE':return
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    import wandb
    run=wandb.init(project='thinker',entity=os.getenv('WANDB_USER','jmme425'),mode='online',
        group=args.campaign.name,job_type='evaluation',name=f"{task['cell']}-{task['real_step']}-{task['kind']}",
        dir=str(output),config=task)
    if run.settings.mode!='online':raise RuntimeError('Online W&B required')
    started=time.monotonic()
    try:
        result=evaluate(task,output)
        result.update(wandb_url=run.url,seconds=time.monotonic()-started,job_id=os.getenv('SLURM_JOB_ID'))
        run.log(dict(real_step=task['real_step'],**{k:v for k,v in result['metrics'].items() if v is not None}))
        run.summary.update(result['metrics']);run.summary['phase']='COMPLETE'
        atomic_json(output/'summary.json',result)
        atomic_json(output/'status.json',dict(phase='COMPLETE',wandb_url=run.url,updated_at=time.time()))
        run.finish(exit_code=0)
    except BaseException as exc:
        atomic_json(output/'status.json',dict(phase='FAILED',error=f'{type(exc).__name__}: {exc}',wandb_url=run.url))
        run.finish(exit_code=1);raise


if __name__=='__main__':main()
