"""Run the 10M thematic campaign with bounded local CPU workers and resume.

Local execution does not submit PBS jobs or start RL/score evaluation. Each
subject/game is sequential; independent units use a bounded process pool.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import sys
import time
import traceback

for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import INPUT_CACHE,OUTPUTS,output
from common.io import write_json

def worker(args,s,g):
    from analyses.runner import run_unit
    statuses={}
    task_path=output(0,args.outputs_root)/'campaign/tasks'/f'sub{s:03d}_game{g}.json'
    for number in args.themes:
        if number=='10':continue
        print(json.dumps(dict(event='start',theme=number,subject=s,game=g,pid=os.getpid())),flush=True)
        write_json(task_path,dict(subject=s,game=g,pid=os.getpid(),current_theme=number,status='RUNNING',finished=statuses,updated_at=time.time()))
        try:status=run_unit(number,args.cache_root,args.outputs_root,s,g)
        except Exception:
            status='FAILED'
            write_json(output(number,args.outputs_root)/f'sub{s:03d}_game{g}'/'failure.json',dict(error=traceback.format_exc(),at=time.time()))
            print(traceback.format_exc(),flush=True)
        statuses[number]=status
        write_json(task_path,dict(subject=s,game=g,pid=os.getpid(),current_theme=number,status=status,finished=statuses,updated_at=time.time()))
        print(json.dumps(dict(event='finish',theme=number,subject=s,game=g,status=status)),flush=True)
    return s,g,statuses

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root',type=Path,default=INPUT_CACHE)
    p.add_argument('--outputs-root',type=Path,default=OUTPUTS)
    p.add_argument('--subjects',type=int,nargs='+',default=list(range(1,7)))
    p.add_argument('--games',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--themes',nargs='+',default=['02','03','04','05','09','12','07','08','06','10','13'])
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();args.themes=[f'{int(n):02d}' for n in args.themes]
    if args.workers<1 or args.workers>8:p.error('Use 1--8 bounded CPU workers')
    manifest=json.loads((args.cache_root/'manifest.json').read_text())
    expected={(s,g,c) for s in args.subjects for g in args.games for c in range(1,5)}
    available={(e['subject'],e['game'],e['leave_session']) for e in manifest['caches']}
    if manifest['status']!='COMPLETED' or not expected.issubset(available):raise ValueError('Incomplete requested 10M cache cohort')
    root=output(0,args.outputs_root)/'campaign';root.mkdir(parents=True,exist_ok=True)
    state=dict(status='RUNNING',pid=os.getpid(),started_at=time.time(),subjects=args.subjects,games=args.games,
               themes=args.themes,workers=args.workers,cache=str(args.cache_root),tasks={})
    write_json(root/'progress.json',state)
    if '10' in args.themes:
        from analyses.optional import utility
        try:state['utility']=utility(args.outputs_root)
        except Exception:state['utility']='FAILED';state['utility_error']=traceback.format_exc()
        write_json(root/'progress.json',state)
    phases=[('behavior_control_prediction',[n for n in args.themes if n not in ['06','07','08','10','13']]),
            ('neural',[n for n in ['07','08','06'] if n in args.themes]),
            ('exploratory_modes',[n for n in args.themes if n=='13'])]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for phase,numbers in phases:
            if not numbers:continue
            phase_args=copy.copy(args);phase_args.themes=numbers;state['phase']=phase
            write_json(root/'progress.json',state)
            futures={pool.submit(worker,phase_args,s,g):(s,g) for s in args.subjects for g in args.games}
            for future in as_completed(futures):
                s,g=futures[future]
                try:_,_,statuses=future.result()
                except Exception:statuses={'worker':'FAILED','error':traceback.format_exc()}
                state['tasks'].setdefault(f'sub{s:03d}_game{g}',{}).update(statuses);state['updated_at']=time.time()
                write_json(root/'progress.json',state)
            from common.reporting import aggregate
            for number in numbers:
                if number=='13':continue
                try:aggregate(number,args.outputs_root,args.subjects,args.games)
                except Exception:state.setdefault('group_errors',{})[number]=traceback.format_exc()
    from common.reporting import aggregate
    for number in args.themes:
        if number=='10':
            pass
        elif number!='13':
            try:aggregate(number,args.outputs_root,args.subjects,args.games)
            except Exception:state.setdefault('group_errors',{})[number]=traceback.format_exc()
    values=[v for t in state['tasks'].values() for v in t.values()]
    state['status']='FINISHED_WITH_PENDING_OR_FAILURES' if any(v in ['FAILED','PARTIAL_FITS','BLOCKED_INPUT','BLOCKED_RUNTIME','READY_OPTIONAL'] for v in values) or state.get('group_errors') else 'COMPLETED'
    state['completed_at']=time.time();write_json(root/'progress.json',state)
    print(json.dumps(state),flush=True)

if __name__=='__main__':main()
