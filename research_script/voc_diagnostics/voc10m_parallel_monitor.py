"""Publish bounded-lag parallel checkpoint pairs and schedule independent GPU evaluations.

Training is submitted explicitly once. Numerical failures are never retried by
this monitor. Evaluation milestones and a separate GPU partition are configurable.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from voc10m_controller import NODES, ACTIVE, atomic, read, job_states

MILESTONES = (200_000,500_000,1_000_000,2_000_000,3_000_000,5_000_000,7_500_000,10_000_000)
EVALUATION_KINDS = {'score','score_sign','score_zero','benefit','episodes'}
DEFAULT_DISABLED_KINDS = ['episodes','score','score_sign','score_zero']
LEGACY_BENEFIT_MILESTONES = (1_000_000,3_000_000,5_000_000,10_000_000)


def evaluation_policy(campaign):
    policy=read(campaign/'evaluation_policy.json',{})
    policy.setdefault('evaluation_partition','rtx2080')
    # Only paired VOC utility is scheduled by default, per user request.
    disabled=policy.setdefault('disabled_kinds',list(DEFAULT_DISABLED_KINDS))
    if not isinstance(disabled,list) or any(not isinstance(k,str) or k not in EVALUATION_KINDS for k in disabled):
        raise ValueError('Invalid disabled evaluation kinds')
    start=policy.get('benefit_from_milestone',{})
    if not isinstance(start,dict) or any(not isinstance(cell,str) or type(step) is not int or step<0
                                       for cell,step in start.items()):
        raise ValueError('Invalid benefit start milestones')
    if policy.get('evaluation_partition')!='rtx2080':
        raise ValueError('Separate evaluation partition must be rtx2080')
    limit=policy.get('evaluation_max_jobs',4)
    if type(limit) is not int or limit<1:raise ValueError('Invalid evaluation concurrency')
    return policy


def evaluation_milestones(manifest):
    steps=manifest.get('evaluation_milestones',list(MILESTONES))
    if not isinstance(steps,list) or not steps or any(type(s) is not int or s<=0 for s in steps):
        raise ValueError('Invalid evaluation milestones')
    if steps!=sorted(set(steps)):raise ValueError('Evaluation milestones must increase without duplicates')
    return tuple(steps)


def allocated_nodes():
    text=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%N'],text=True)
    return {jid:node for jid,node in (line.split('|',1) for line in text.splitlines())
            if node.startswith('bml-compute') and ',' not in node and '[' not in node}


def evaluation_kinds(cell,milestone,policy):
    kinds=['score','score_sign','score_zero']
    # A per-cell cutoff enables prospective rollout on an existing campaign,
    # preserving its original benefit tasks without backfilling old checkpoints.
    start=policy.get('benefit_from_milestone',{}).get(cell,0)
    if milestone in LEGACY_BENEFIT_MILESTONES or milestone>=start:kinds.append('benefit')
    if milestone in (3_000_000,5_000_000,10_000_000):kinds.append('episodes')
    return [kind for kind in kinds if kind not in policy.get('disabled_kinds',DEFAULT_DISABLED_KINDS)]


def partition_evaluations(tasks,registry,policy):
    """Exclude disabled work from submission/completion, retaining its history."""
    enabled=[];disabled={}
    for task in tasks:
        payload=read(task)
        if payload['kind'] not in policy.get('disabled_kinds',[]):
            enabled.append(task)
            continue
        record=registry['evaluation'].get(task.stem)
        disabled[task.stem]=dict(kind=payload['kind'],game=payload['game'],
            job_id=record.get('job_id') if record else None,
            state=record.get('state') if record else 'NOT_SUBMITTED',
            reason=policy.get('reason','Disabled by campaign evaluation policy'),
            output=payload['output'])
    return enabled,disabled


def first_checkpoint(directory, role, milestone, window=100_000):
    candidates=[]
    for path in directory.glob(f'ckp_{role}.tar_step_*'):
        tail=path.name.rsplit('_',1)[-1]
        if tail.isdigit() and milestone<=int(tail)<milestone+window:
            candidates.append((int(tail),path))
    return min(candidates,default=(None,None))[1]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def publish(campaign,cell,milestone):
    from reward_probe import load_checkpoints
    import torch
    run=campaign/'cells'/cell['name']
    target=campaign/'bundles'/cell['name']/str(milestone)
    binding=target/'bundle.json'
    if binding.exists():return binding
    actor=first_checkpoint(run,'actor',milestone)
    model=first_checkpoint(run,'model',milestone)
    if actor is None or model is None:return None
    target.mkdir(parents=True,exist_ok=True)
    files={}
    for kind,path in [('actor',actor),('model',model),('config',run/'config_c.yaml')]:
        dest=target/path.name
        shutil.copyfile(path,str(dest)+'.tmp')
        os.replace(str(dest)+'.tmp',dest)
        files[kind]=dict(path=str(dest),sha256=sha(dest))
    actor_data=torch.load(files['actor']['path'],map_location='cpu',weights_only=False)
    model_data=torch.load(files['model']['path'],map_location='cpu',weights_only=False)
    steps={role:int(data['real_step']) for role,data in [('actor',actor_data),('model',model_data)]}
    if any(not milestone<=s<milestone+100_000 for s in steps.values()):
        raise ValueError('Checkpoint counters outside predeclared milestone window')
    if int(actor.name.rsplit('_',1)[-1])!=steps['actor'] or int(model.name.rsplit('_',1)[-1])!=steps['model']:
        raise ValueError('Checkpoint filename/counter mismatch')
    payload=dict(files=files,nominal_milestone=milestone,real_step=steps['actor'],
        actor_real_step=steps['actor'],model_real_step=steps['model'],
        pair_step_difference=steps['model']-steps['actor'],
        pairing='first checkpoint per learner at or after milestone, each within +100k; asynchronous, not simultaneous',
        source_sha256=sha(campaign/'source/SOURCE.sha256'),created_at=time.time())
    pending=target/'bundle.pending.json';atomic(pending,payload)
    load_checkpoints(argparse.Namespace(bundle=pending,checkpoint=None,threads=1))
    pending.replace(binding)
    return binding


def queue_evaluations(campaign,cell,milestone,bundle,policy=None):
    if policy is None:policy=evaluation_policy(campaign)
    for kind in evaluation_kinds(cell['name'],milestone,policy):
        name=f"{cell['name']}_{milestone:08d}_{kind}"
        task=campaign/'eval_tasks'/f'{name}.json'
        payload=dict(cell=cell['name'],game=cell['game'],real_step=milestone,
            step_label='nominal milestone; actual actor/model counters in bundle',kind=kind,
            bundle=str(bundle),output=str(campaign/'evaluations'/name))
        if task.exists():
            if read(task)!=payload:raise ValueError('Evaluation task identity changed')
        else:atomic(task,payload)


def submit_eval(campaign,task,node=None):
    # Recheck immediately before submission, including preexisting task files.
    policy=evaluation_policy(campaign)
    if read(task)['kind'] in policy.get('disabled_kinds',[]):
        raise ValueError('Evaluation disabled by campaign policy: '+task.stem)
    separate=policy.get('evaluation_partition')
    partition=separate or NODES[node][0]
    script=Path(policy.get('evaluation_script',campaign/'source/research_script/voc_diagnostics/voc10m.slurm'))
    cmd=['sbatch','--parsable','--chdir='+str(campaign/'source'),
         '--job-name=v10e-'+task.stem[-28:],'--partition='+partition,
         '--nodes=1','--gres=gpu:'+partition+':1','--cpus-per-task=4',
         '--mem=24G','--time=2-00:00:00','--output='+str(campaign/'eval-%j.out'),
         *([] if separate else ['--nodelist='+node]),
         str(script),str(campaign),'evaluate',str(task)]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Invalid SLURM job id')
    return dict(job_id=job,node=None if separate else node,partition=partition,
                gpus=1,submitted_at=time.time(),state='PENDING',command=cmd)


def training_summary(run):
    path=run/'logs.csv'
    summary={}
    if path.exists():
        with path.open('rb') as stream:
            header=stream.readline().decode()
            size=stream.seek(0,2)
            stream.seek(max(0,size-2**20))
            lines=stream.read().decode(errors='replace').splitlines()
        fields=next(csv.reader([header]));last=None
        # A writer may be midway through the last row; take the most recent
        # complete row without rescanning the growing, wide training CSV.
        for line in reversed(lines[-3:]):
            values=next(csv.reader([line]))
            if len(values)==len(fields) and values[0] not in {'_tick','# _tick'}:
                last=dict(zip(fields,values));break
        if last:
            keys=['real_step','_time','rmean_episode_return','max_episode_return',
                  'training/real_actions_per_second','training/bc_targets_committed',
                  'collector/physical_env_steps','collector/zero_search_fraction',
                  'gradient/voc_gru_applied_norm','gradient/gru_update_norm']
            summary={k:last.get(k) for k in keys}
    summary['collector']=read(run/'collector_metrics.json',{})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--once',action='store_true')
    args=p.parse_args();c=args.campaign.resolve()
    lock=(c/'monitor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(c/'campaign.json')
    milestones=evaluation_milestones(manifest)
    registry=read(c/'jobs.json')
    registry.setdefault('evaluation',{})
    (c/'eval_tasks').mkdir(exist_ok=True)
    while True:
        policy=evaluation_policy(c)
        training=list(registry['training'].values())
        evaluations=list(registry['evaluation'].values())
        states=job_states([j['job_id'] for j in training+evaluations])
        locations=allocated_nodes() if policy.get('evaluation_partition') else {}
        live={n:0 for n in NODES};errors=[];training_complete=0
        for j in training+evaluations:
            state=states.get(j['job_id'],{}).get('state','UNKNOWN')
            j['state']=state
            if j['job_id'] in locations:j['node']=locations[j['job_id']]
            if state in ACTIVE|{'UNKNOWN'}:
                node=j.get('node') or 'PENDING_PLACEMENT'
                live[node]=live.get(node,0)+j['gpus']
        for cell in manifest['cells']:
            j=registry['training'][cell['name']]
            run=c/'cells'/cell['name']
            j['progress']=training_summary(run)
            published=[]
            for milestone in milestones:
                try:
                    bundle=publish(c,cell,milestone)
                    if bundle:
                        queue_evaluations(c,cell,milestone,bundle,policy)
                        published.append(milestone)
                except Exception as e:
                    errors.append(dict(cell=cell['name'],milestone=milestone,error=str(e)))
            j['published_milestones']=published
            if j['state']=='COMPLETED':
                if len(published)==len(milestones) and (run/'finish').exists():training_complete+=1
                else:errors.append(dict(cell=cell['name'],error='Job completed without all verified milestones/finish'))
            elif j['state'] not in ACTIVE|{'UNKNOWN'}:
                errors.append(dict(cell=cell['name'],error='Training failed; no automatic restart',job=j['job_id'],state=j['state']))
        complete_eval=0
        tasks,disabled=partition_evaluations(sorted((c/'eval_tasks').glob('*.json')),registry,policy)
        for name,record in disabled.items():
            if record['job_id'] and record['state'] in ACTIVE|{'UNKNOWN'}:
                errors.append(dict(evaluation=name,state=record['state'],
                    error='Disabled evaluation still active; explicit cancellation required'))
        for task in tasks:
            record=registry['evaluation'].get(task.stem)
            if record:
                summary=read(Path(record.get('output') or read(task)['output'])/'summary.json',{})
                if record['state']=='COMPLETED' and summary.get('status')=='COMPLETE':complete_eval+=1
                elif record['state'] not in ACTIVE|{'UNKNOWN'}:
                    errors.append(dict(evaluation=task.stem,state=record['state'],error='Evaluation missing successful result'))
                continue
            if policy.get('evaluation_partition'):
                in_flight=sum(1 for j in registry['evaluation'].values() if j['state'] in ACTIVE|{'UNKNOWN'})
                if in_flight>=policy.get('evaluation_max_jobs',4):continue
                record=submit_eval(c,task)
                registry['evaluation'][task.stem]=record
                live['PENDING_PLACEMENT']=live.get('PENDING_PLACEMENT',0)+1
                atomic(c/'jobs.json',registry)
                continue
            # Reserve two GPUs for each unfinished training job even if it
            # failed: the monitor never fills those with unrelated training.
            candidates=[]
            for node,(_,capacity) in NODES.items():
                reserved=sum(2 for x in manifest['cells'] if x['node']==node and
                             registry['training'][x['name']]['state'] not in {'COMPLETED'})
                active_train=sum(j['gpus'] for j in training if j['node']==node and j['state'] in ACTIVE|{'UNKNOWN'})
                used=live[node]+max(0,reserved-active_train)
                if used<capacity:candidates.append(node)
            if not candidates:continue
            node=min(candidates,key=lambda n:(live[n]/NODES[n][1],n!='bml-compute08',n))
            record=submit_eval(c,task,node)
            registry['evaluation'][task.stem]=record;live[node]+=1
            atomic(c/'jobs.json',registry)
        done=training_complete==len(manifest['cells']) and complete_eval==len(tasks) and not errors
        report=dict(updated_at=time.time(),phase='COMPLETE' if done else ('NEEDS_ATTENTION' if errors else 'RUNNING'),
            training_complete=training_complete,training_total=len(manifest['cells']),
            evaluation_milestones=list(milestones),evaluation_partition=policy.get('evaluation_partition'),
            evaluation_complete=complete_eval,evaluation_total=len(tasks),
            evaluation_disabled=len(disabled),disabled_evaluation_kinds=policy.get('disabled_kinds',[]),
            disabled_evaluations=disabled,gpu_jobs_by_node=live,errors=errors)
        atomic(c/'jobs.json',registry);atomic(c/'progress.json',report)
        print(json.dumps(report),flush=True)
        # Fail visibly once no live or unsubmitted work can make progress.
        active_any=any(j['state'] in ACTIVE|{'UNKNOWN'} for j in list(registry['training'].values())+list(registry['evaluation'].values()))
        unsent=any(t.stem not in registry['evaluation'] for t in tasks)
        if args.once or done:break
        if errors and not active_any and not unsent:raise RuntimeError('Campaign needs attention; see progress.json')
        time.sleep(45)


if __name__=='__main__':main()
