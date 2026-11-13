"""Recompute non-neural COSYNE panels from the dynamic 10M cache.

Existing validated regression/readout results are reused with explicit source
hashes. New matching levels, surfaces, endpoint sensitivities and curves are
computed. All outputs stay immediately below outputs/NN/. Neural is excluded.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import INPUT_CACHE, OUTPUTS, output
from common.io import write_json


def job(cache, outputs, subject, game):
    from panels.runner import run
    try:
        return run(cache, outputs, subject, game)
    except Exception:
        record = dict(status='FAILED', subject=subject, game=game, error=traceback.format_exc())
        write_json(output(0, outputs)/'panel_campaign'/f'failure_sub{subject:03d}_game{game}.json', record)
        print(record['error'], flush=True)
        return record


def raw_job(cache,outputs,subject,game):
    from panels.raw_behavior import run
    return dict(subject=subject,game=game,status=run(cache,outputs,subject,game))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root', type=Path, default=INPUT_CACHE)
    p.add_argument('--outputs-root', type=Path, default=OUTPUTS)
    p.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 7)))
    p.add_argument('--games', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--render-only', action='store_true')
    args = p.parse_args()
    if not 1 <= args.workers <= 8: p.error('Use 1–8 CPU workers')
    root = output(0, args.outputs_root)/'panel_campaign'; root.mkdir(parents=True, exist_ok=True)
    state = dict(status='RUNNING', pid=os.getpid(), started_at=time.time(), subjects=args.subjects,
        games=args.games, workers=args.workers, neural=False, new_score_evaluation=False, tasks={})
    if args.render_only and (root/'progress.json').exists():
        previous=json.loads((root/'progress.json').read_text())
        state['tasks']=previous.get('tasks',{})
        state['analysis_started_at']=previous.get('analysis_started_at',previous.get('started_at'))
        if (root/'human_refresh.json').exists():
            refresh=json.loads((root/'human_refresh.json').read_text())
            state['original_human_behavior']={k:dict(status=v) for k,v in refresh.get('tasks',{}).items()}
    write_json(root/'progress.json', state)
    if not args.render_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(job, args.cache_root, args.outputs_root, s, g): (s, g)
                       for s in args.subjects for g in args.games}
            for future in as_completed(futures):
                s, g = futures[future]
                state['tasks'][f'sub{s:03d}_game{g}'] = future.result()
                state['updated_at'] = time.time(); write_json(root/'progress.json', state)
        state['phase']='original_human_behavior';write_json(root/'progress.json',state)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(raw_job,args.cache_root,args.outputs_root,s,g):(s,g) for s in args.subjects for g in args.games}
            state['original_human_behavior']={}
            for future in as_completed(futures):
                s,g=futures[future]
                try:state['original_human_behavior'][f'sub{s:03d}_game{g}']=future.result()
                except Exception:state['original_human_behavior'][f'sub{s:03d}_game{g}']=dict(status='FAILED',error=traceback.format_exc())
                write_json(root/'progress.json',state)
    state['phase'] = 'rendering'; write_json(root/'progress.json', state)
    from panels.render import aggregate
    try:
        coverage = aggregate(args.outputs_root, args.subjects, args.games)
        coverage.to_csv(root/'coverage.csv', index=False)
        state['coverage'] = coverage.to_dict('records')
    except Exception:
        state['render_error'] = traceback.format_exc(); print(state['render_error'], flush=True)
    # HSMM has its own already-running campaign; record its actual state instead
    # of launching a competing writer or pretending its panels were recomputed.
    state['behavioral_modes'] = {}
    for s in args.subjects:
        for g in args.games:
            f = output(13, args.outputs_root)/f'sub{s:03d}_game{g}'/'manifest.json'
            state['behavioral_modes'][f'sub{s:03d}_game{g}'] = json.loads(f.read_text()).get('status') if f.exists() else 'WAITING_EXISTING_CAMPAIGN'
    state['status'] = 'FAILED' if state.get('render_error') or any(v['status'] == 'FAILED' for v in state['tasks'].values()) or any(v['status']=='FAILED' for v in state.get('original_human_behavior',{}).values()) else 'COMPLETED'
    if state.get('coverage') and any(v['status'] == 'INCOMPLETE_COHORT' for v in state['coverage']): state['status'] = 'PARTIAL'
    state['completed_at'] = time.time(); write_json(root/'progress.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__': main()
