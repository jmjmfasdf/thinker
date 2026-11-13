"""Bounded SLURM submission/monitor: three training runs and up to seven eval GPUs.

Only orderly PAUSED training is resumed automatically. Failed scientific jobs
remain failed for inspection. The controller never cancels another job.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

NODES={'bml-compute07':('titanrtx',4),'bml-compute08':('rtx3090',4),'bml-compute10':('rtx4090',2)}
ACTIVE={'PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED','REQUEUED'}


def read(path,default=None):
    if not path.exists():return default
    return json.loads(path.read_text())


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def job_states(ids):
    if not ids:return {}
    text=subprocess.check_output(['sacct','-n','-X','-P','-j',','.join(ids),'-o','JobIDRaw,State,ExitCode'],text=True)
    rows={}
    for line in text.splitlines():
        fields=line.split('|')
        if len(fields)>=3:rows[fields[0]]=dict(state=fields[1].split()[0],exit=fields[2])
    # squeue wins for live jobs, including a recently submitted job whose
    # accounting record has not propagated yet.
    text=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%T'],text=True)
    for line in text.splitlines():
        jid,state=line.split('|',1)
        if jid in ids:rows[jid]=dict(state=state,exit=None)
    return rows


def submit(campaign,kind,payload,node):
    partition,_=NODES[node]
    script=campaign/'source/research_script/voc_diagnostics/voc10m.slurm'
    cmd=['sbatch','--parsable','--chdir='+str(campaign/'source'),
        '--job-name=v10-'+(payload if kind=='train' else Path(payload).stem[-24:]),
        '--partition='+partition,'--nodelist='+node,'--gres=gpu:'+partition+':1',
        '--cpus-per-task=4','--mem=28G','--time=6-00:00:00','--signal=B:USR1@600',
        '--output='+str(campaign/'slurm-%j.out'),str(script),str(campaign),kind,str(payload)]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Invalid sbatch reply')
    return dict(job_id=job,node=node,kind=kind,payload=str(payload),submitted_at=time.time())


def submit_controller(campaign):
    script=campaign/'source/research_script/voc_diagnostics/voc10m_controller.slurm'
    return subprocess.check_output(['sbatch','--parsable','--chdir='+str(campaign/'source'),
        '--job-name=voc10m-monitor','--partition=rtx3090','--nodelist=bml-compute08',
        '--cpus-per-task=1','--mem=1G','--time=7-00:00:00',
        '--output='+str(campaign/'controller-%j.out'),str(script),str(campaign)],text=True).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--once',action='store_true')
    args=p.parse_args();c=args.campaign.resolve()
    lock=(c/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(c/'campaign.json');registry=read(c/'jobs.json',dict(training={},evaluation={}))
    started=time.monotonic()
    while True:
        all_jobs=[j for group in registry.values() for attempts in group.values() for j in attempts]
        states=job_states([j['job_id'] for j in all_jobs])
        active_per_node={n:0 for n in NODES}
        errors=[]
        for j in all_jobs:
            state=states.get(j['job_id'],{}).get('state','UNKNOWN')
            j['state']=state
            if state in ACTIVE or state=='UNKNOWN':active_per_node[j['node']]+=1
        finished_training=0
        for cell in manifest['cells']:
            name=cell['name'];output=c/'cells'/name
            status=read(output/'status.json',{})
            history=registry['training'].setdefault(name,[])
            if history and history[-1]['state'] in ACTIVE|{'UNKNOWN'}:continue
            if status.get('phase')=='COMPLETE':
                if status['env_steps']!=10_000_000:raise RuntimeError('Invalid completed budget')
                finished_training+=1;continue
            if history and (history[-1]['state']!='COMPLETED' or status.get('phase')!='PAUSED'):
                errors.append(dict(cell=name,status=status,job=history[-1]));continue
            j=submit(c,'train',cell['game'],cell['node']);j['state']='PENDING'
            history.append(j);active_per_node[j['node']]+=1
            atomic(c/'jobs.json',registry)
        tasks=sorted((c/'eval_tasks').glob('*.json'))
        eval_complete=0
        for task_path in tasks:
            name=task_path.stem;task=read(task_path)
            history=registry['evaluation'].setdefault(name,[])
            if history and history[-1]['state'] in ACTIVE|{'UNKNOWN'}:continue
            summary=read(Path(task['output'])/'summary.json',{})
            if summary.get('status')=='COMPLETE':eval_complete+=1;continue
            if history:
                errors.append(dict(evaluation=name,job=history[-1]));continue
            eligible=[n for n,(_,cap) in NODES.items() if active_per_node[n]<cap]
            if not eligible:continue
            # Fill each named node's declared share, with compute08 first for
            # initial score tasks. SLURM arbitrates actual availability.
            node=min(eligible,key=lambda n:(active_per_node[n]/NODES[n][1],n!='bml-compute08',n))
            j=submit(c,'evaluate',task_path,node);j['state']='PENDING'
            history.append(j);active_per_node[node]+=1
            atomic(c/'jobs.json',registry)
        atomic(c/'jobs.json',registry)
        complete=finished_training==len(manifest['cells']) and eval_complete==len(tasks)
        report=dict(updated_at=time.time(),phase='COMPLETE' if complete else ('NEEDS_ATTENTION' if errors else 'RUNNING'),
            training_complete=finished_training,training_total=len(manifest['cells']),
            evaluation_complete=eval_complete,evaluation_total=len(tasks),
            active_or_pending_by_node=active_per_node,errors=errors)
        atomic(c/'progress.json',report)
        print(json.dumps(report),flush=True)
        if args.once or complete:break
        if time.monotonic()-started>6*86400:
            atomic(c/'controller_successor.json',dict(job_id=submit_controller(c),at=time.time()))
            break
        time.sleep(30)


if __name__=='__main__':main()
