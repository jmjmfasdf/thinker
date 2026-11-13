"""CPU-only, at most two GPU shards; wait for SI 4M without reserving a GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from expanded_voc_sampling import atomic, sampling_shards

ACTIVE = {'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','REQUEUED','UNKNOWN'}


def states(ids):
    if not ids:return {}
    result = subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
        '--format=JobIDRaw,State'], text=True)
    return {row.split('|')[0]: row.split('|')[1].split()[0].rstrip('+')
            for row in result.splitlines() if row}


def aggregate(root, tasks, registry, game):
    selected = [task for task in tasks if task['game'] == game]
    if len(selected) != 8:return False
    out = root/'aggregate'/game
    if (out/'summary.json').exists():return True
    summaries = []
    rows = []
    for task in selected:
        if registry.get(task['name'],{}).get('state') != 'COMPLETED':return False
        path = Path(task['output'])
        if not (path/'summary.json').exists():return False
        summary=json.loads((path/'summary.json').read_text())
        if summary.get('status') != 'COMPLETE':return False
        summaries.append(summary);rows.extend(json.loads((path/'pairs.json').read_text()))
    assert all(s['checkpoint']==summaries[0]['checkpoint'] for s in summaries)
    identities=[(r['seed'],r['stream'],r['prefix'],r['count'],r['repetition']) for r in rows]
    assert len(rows)==512 and len(set(identities))==512
    source=Path(selected[0]['source'])
    sys.path.insert(0,str(source/'thinker'));sys.path.insert(0,str(source/'research_script/voc_diagnostics'))
    from gru_ablation_eval import paired_summary
    voc=paired_summary(rows)
    assert voc['states']==256 and voc['independent_streams']==32
    atomic(out/'pairs.json',rows)
    atomic(out/'summary.json',dict(status='COMPLETE',task=dict(game=game,real_step=4_000_000),
        voc=voc,checkpoint=summaries[0]['checkpoint'],
        wall_seconds_sum=sum(s['seconds'] for s in summaries),
        shards=[dict(job_id=s['job_id'],wandb_url=s['wandb_url']) for s in summaries],
        scope='Expanded predetermined paired sampling, temperature .01; 128 scenes, 256 states, 512 pairs'))
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();plan=json.loads((root/'plan.json').read_text())
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registry=json.loads((root/'jobs.json').read_text()) if (root/'jobs.json').exists() else {}
    campaign=Path(plan['campaign']);source=campaign/'source'
    while True:
        tasks=[];waiting=[]
        for shard in sampling_shards():
          for game in plan['games']:
            bundle=campaign/'bundles'/f'v10t001-20260926-{game}_s701'/'4000000'/'bundle.json'
            if not bundle.exists():
                if game not in waiting:waiting.append(game)
                continue
            name=f'{game}_4M_shard{shard["shard"]}'
            task=dict(name=name,game=game,real_step=4_000_000,temperature=.01,
                code_sha256=plan['source_sha256'],base_source_sha256=plan['base_source_sha256'],
                bundle=str(bundle),source=str(source),output=str(root/'shards'/name),**shard)
            tf=root/'tasks'/(name+'.json')
            if tf.exists():
                assert json.loads(tf.read_text())==task
            else:atomic(tf,task)
            tasks.append(task)
        status=states([j['job_id'] for j in registry.values()])
        errors=[]
        for name,record in registry.items():
            record['state']=status.get(record['job_id'],'UNKNOWN')
            if record['state'] not in ACTIVE|{'COMPLETED'}:
                errors.append(dict(task=name,job_id=record['job_id'],state=record['state']))
            elif record['state']=='COMPLETED' and not (root/'shards'/name/'summary.json').exists():
                errors.append(dict(task=name,error='Completed without summary'))
        active=sum(j['state'] in ACTIVE for j in registry.values())
        for task in tasks:
            if active>=plan['max_active_jobs'] or errors:break
            name=task['name']
            if name in registry:continue
            cmd=['sbatch','--parsable','--partition=rtx2080','--exclude=bml-compute02',
                '--nodes=1','--gres=gpu:rtx2080:1','--cpus-per-task=4','--mem=24G',
                '--time=12:00:00','--job-name=vocx-'+name,
                '--chdir='+str(source),'--output='+str(root/'logs'/'%j.out'),
                str(root/'evaluate.slurm'),str(root),str(root/'tasks'/(name+'.json'))]
            job=subprocess.check_output(cmd,cwd='/home/jeongmin/slurm_submit',text=True).strip().split(';')[0]
            assert job.isdigit()
            registry[name]=dict(job_id=job,state='PENDING',command=cmd,submitted_at=time.time())
            atomic(root/'jobs.json',registry);active+=1
        completed_games=[g for g in plan['games'] if aggregate(root,tasks,registry,g)]
        done=len(completed_games)==len(plan['games'])
        report=dict(updated_at=time.time(),phase='COMPLETE' if done else 'NEEDS_ATTENTION' if errors else 'RUNNING',
            completed_shards=sum(j['state']=='COMPLETED' for j in registry.values()),total_shards=24,
            active_jobs=active,waiting_for_4m_checkpoint=waiting,completed_games=completed_games,errors=errors)
        atomic(root/'jobs.json',registry);atomic(root/'progress.json',report)
        print(json.dumps(report),flush=True)
        if done:break
        if errors and not active:raise RuntimeError('Explicit recovery required; see progress.json')
        time.sleep(45)


if __name__=='__main__':main()
