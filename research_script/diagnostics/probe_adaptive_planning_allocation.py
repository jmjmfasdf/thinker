"""Fixed-checkpoint imagination allocation; no separate game scoring or training."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def summarize(rows, minimum, cap):
    length = np.array([r['length'] for r in rows])
    probabilities = np.array([r['first_optional_probability'] for r in rows], dtype=float)
    valid = np.isfinite(probabilities)
    p = probabilities[valid]
    # The first optional query is at min_search_steps. Continuing it is exactly
    # the event that the completed length exceeds the mandatory minimum.
    actual = (length[valid] > minimum).astype(float)
    floor = 1e-12
    ll = actual*np.log(np.maximum(p, floor))+(1-actual)*np.log(np.maximum(1-p, floor))
    rate = np.clip(actual.mean(), floor, 1-floor)
    constant_ll = actual*np.log(rate)+(1-actual)*np.log(1-rate)
    groups = sorted(set((r['seed'], r['stream']) for r in rows))
    group_stats = []
    for group in groups:
        mask = np.array([(r['seed'],r['stream']) == group for r in rows])
        group_stats.append([np.mean(length[mask] == minimum),
                            np.mean((length[mask] > minimum)&(length[mask] < cap)),
                            np.mean(length[mask] == cap),np.mean(length[mask])])
    group_stats = np.array(group_stats)
    rng = np.random.default_rng(91273)
    boot = group_stats[rng.integers(0,len(groups),(2000,len(groups)))].mean(axis=1)
    return dict(decisions=len(rows), independent_streams=len(groups),
        histogram=np.bincount(length,minlength=cap+1).tolist(),
        min_fraction=float(np.mean(length==minimum)),
        interior_fraction=float(np.mean((length>minimum)&(length<cap))),
        cap_fraction=float(np.mean(length==cap)), mean_length=float(length.mean()),
        stream_bootstrap_ci95=dict(zip(('min','interior','cap','mean'),np.quantile(boot,[.025,.975],axis=0).T.tolist())),
        first_optional_probability_sd=float(p.std()),
        first_optional_probability_p05_p50_p95=np.quantile(p,[.05,.5,.95]).tolist(),
        first_optional_probability_min=float(p.min()), first_optional_probability_max=float(p.max()),
        within_decision_sign_change_fraction=float(np.mean([r['sign_changes']>0 for r in rows])),
        first_query_execution_log_score=float(ll.mean()),
        first_query_execution_log_score_gain_over_constant=float((ll-constant_ll).mean()),
        interpretation='Execution calibration/state variation only; no claim that additional computation improves reward.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--conditions',nargs='+',default=['control','common','explore','burn8'])
    ap.add_argument('--milestone',type=int,default=50_000)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--decisions',type=int,default=512)
    ap.add_argument('--seeds',type=int,nargs='+',default=[8301,8401])
    args = ap.parse_args()
    campaign=args.campaign.resolve()
    root=campaign.parents[3]
    args.output.mkdir(parents=True,exist_ok=False)
    gpu=torch.cuda.get_device_name(0)
    if '2080 Ti' not in gpu:
        raise RuntimeError('Use RTX 2080 Ti for this evaluation')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source=campaign/'evaluation_source_v2/research_script/diagnostics/evaluate_adaptive_planning_pilot.py'
    spec=importlib.util.spec_from_file_location('bound_pilot_eval',source)
    evaluation=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    manifest=json.loads((campaign/'manifest.json').read_text())
    summaries=[]
    for condition in args.conditions:
        item=next(r for r in manifest['runs'] if r['condition']==condition)
        run=root/'logs/thinker'/item['xpid']
        checkpoints=[]
        for kind in ('actor','model'):
            paths=[p for p in run.glob(f'ckp_{kind}.tar_step_*')
                   if args.milestone <= int(p.name.rsplit('_',1)[1]) < args.milestone+2000]
            if not paths:
                raise FileNotFoundError(f'No {condition} {kind} checkpoint near {args.milestone}')
            checkpoints.append(min(paths,key=lambda p:int(p.name.rsplit('_',1)[1])))
        flags,actor,model,binding=evaluation.load_pair(*checkpoints)
        eval_flags=copy.deepcopy(flags)
        eval_flags.voc_train_epsilon=0.
        eval_flags.voc_eval_stochastic=True
        rows=[]
        for seed in args.seeds:
            rows.extend(evaluation.allocation(eval_flags,actor,model,seed=seed,
                batch=args.batch,decisions=args.decisions,warmup=128,device='cuda:0'))
            evaluation.atomic_json(args.output/(condition+'_allocation.json'),rows)
            print(json.dumps(dict(condition=condition,seed=seed,decisions=len(rows))),flush=True)
        summary=dict(condition=condition,status='COMPLETE',checkpoint=binding,device=gpu,
            training_epsilon=float(flags.voc_train_epsilon),evaluation_epsilon=0.,
            **summarize(rows,flags.min_search_steps,flags.max_search_steps))
        evaluation.atomic_json(args.output/(condition+'_summary.json'),summary)
        summaries.append(summary)
        evaluation.atomic_json(args.output/'summaries.json',summaries)
        print(json.dumps({k:summary[k] for k in ['condition','mean_length','min_fraction','interior_fraction','cap_fraction','first_optional_probability_sd']}),flush=True)
    evaluation.atomic_json(args.output/'completion.json',dict(status='COMPLETE',conditions=args.conditions,
        decisions_per_condition=args.batch*args.decisions*len(args.seeds),device=gpu,
        scope='fixed checkpoint allocation; paired utility still pending'))


if __name__=='__main__':
    main()
