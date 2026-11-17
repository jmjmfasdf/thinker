"""Record optional-query dynamics for a completed fixed-checkpoint allocation.

Reuse the original allocation loop and observer. Every observed decision must
match the original allocation byte for byte after JSON decoding. These are the
same trajectories, not another independent evaluation sample.
"""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time

from compare_planning_allocation_milestones import load_sample
from probe_adaptive_planning_queries import exact_overlap, observe, query_report


def validate_observed(rows, reference, seed):
    expected = {(seed, stream, decision) for stream in range(8) for decision in range(128,1664)}
    identities = [(r['seed'],r['stream'],r['decision']) for r in rows]
    if len(rows) != len(expected) or set(identities) != expected:
        raise ValueError('missing or duplicated registered observation')
    matched = exact_overlap(rows, reference)
    if matched != len(expected):
        raise ValueError('observer reference does not cover every decision')
    return matched


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--condition',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-summary',type=Path)
    parser.add_argument('--reference-allocation',type=Path)
    args=parser.parse_args()
    import torch
    gpu=torch.cuda.get_device_name(0)
    if '2080 Ti' not in gpu:
        raise ValueError('query evaluation requires RTX2080Ti')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    campaign=args.campaign.resolve()
    if bool(args.reference_summary)!=bool(args.reference_allocation):
        raise ValueError('provide both explicit reference artifacts')
    if args.reference_summary:
        summary=json.loads(args.reference_summary.read_text())
        if summary['status']!='COMPLETE' or summary.get('evaluation_epsilon',summary.get('evaluation_train_epsilon'))!=0:
            raise ValueError('reference allocation must be complete and epsilon-zero')
        reference_path=args.reference_allocation
        reference=json.loads(reference_path.read_text())
        expected={(s,i,d) for s in (8301,8401) for i in range(8) for d in range(128,1664)}
        keys=[(r['seed'],r['stream'],r['decision']) for r in reference]
        if len(keys)!=len(expected) or set(keys)!=expected:
            raise ValueError('explicit reference has missing/duplicate decisions')
        verified=dict(checkpoint=summary['checkpoint'],sources=[dict(path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in
            (args.reference_summary,args.reference_allocation)])
    else:
        reference_root=campaign/f'allocation_1m_{args.condition}'
        verified=load_sample([reference_root],args.condition,registered=True,final=True)
        reference_path=reference_root/f'{args.condition}_allocation.json'
        reference=json.loads(reference_path.read_text())
    source=campaign/'evaluation_source_v2/research_script/diagnostics/evaluate_adaptive_planning_pilot.py'
    spec=importlib.util.spec_from_file_location('registered_query_evaluation',source)
    evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
    flags,actor,model,binding=evaluation.load_pair(
        Path(verified['checkpoint']['actor']['path']),Path(verified['checkpoint']['model']['path']))
    if binding!=verified['checkpoint']:
        raise ValueError('loaded checkpoint differs from completed reference')
    manifest=json.loads((campaign/'manifest.json').read_text())
    registered=next(r for r in manifest['runs'] if r['condition']==args.condition)
    if flags.xpid!=registered['xpid']:
        raise ValueError('reference checkpoint belongs to a different condition')
    flags=copy.deepcopy(flags)
    flags.voc_train_epsilon,flags.voc_eval_stochastic=0.,True
    if flags.min_search_steps!=1 or flags.max_search_steps!=30:
        raise ValueError('unexpected planning limits')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    allocations,queries,overlaps=[],[],[]
    for seed in (8301,8401):
        rows,observed=observe(evaluation,flags,actor,model,seed=seed,batch=8,decisions=1536,warmup=128)
        matched=validate_observed(rows,reference,seed)
        # Validate complete optional-choice sequences before persisting them.
        query_report(observed,rows,minimum=1,cap=30)
        allocations.extend(rows);queries.extend(observed)
        overlaps.append(dict(seed=seed,matched_decisions=matched,queries=len(observed)))
        with gzip.open(args.output/f'queries_seed{seed}.json.gz','wt') as stream:
            json.dump(observed,stream,allow_nan=False)
        evaluation.atomic_json(args.output/f'allocations_seed{seed}.json',rows)
        print(json.dumps(overlaps[-1]),flush=True)
    report,stages=query_report(queries,allocations,minimum=1,cap=30)
    if report['decisions']!=24576 or report['independent_streams']!=16:
        raise ValueError('incomplete registered query sample')
    report.update(status='COMPLETE',condition=args.condition,checkpoint=binding,gpu=gpu,
        evaluation_epsilon=0.,seconds=time.monotonic()-started,reference_overlap=overlaps,
        verified_reference_sources=verified['sources'],observed_decision_range=[128,1663],
        independent_sample=False,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Within-stage diagnostics of the already evaluated trajectories; no separate score or reward-utility evaluation.')
    with gzip.open(args.output/(args.condition+'_queries.json.gz'),'wt') as stream:
        json.dump(queries,stream,allow_nan=False)
    evaluation.atomic_json(args.output/(args.condition+'_allocation.json'),allocations)
    evaluation.atomic_json(args.output/(args.condition+'_stages.json'),stages)
    evaluation.atomic_json(args.output/(args.condition+'_summary.json'),report)
    evaluation.atomic_json(args.output/'completion.json',dict(status='COMPLETE',conditions=[args.condition],
        decisions_per_condition=24576,optional_queries=len(queries),independent_sample=False,
        checkpoint=binding,exact_reference_decisions=sum(r['matched_decisions'] for r in overlaps)))
    print(json.dumps(dict(status='EXACT_REGISTERED_QUERY_OBSERVATION_PASS',decisions=24576,queries=len(queries))),flush=True)


if __name__=='__main__':
    main()
