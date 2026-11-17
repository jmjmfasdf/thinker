"""Validate every registered selected-state pair, replacing recovered shards."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.campaign.resolve()/'selected_utility_500k'
    sources={}
    def read(path):
        raw=path.read_bytes();sources[str(path)]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    plan=read(root/'plan.json');plan_hash=sources[str(root/'plan.json')]
    recovery=read(args.campaign/'selected_utility_recoveries.json')['recoveries']
    replacement={r['shard']:Path(r['output']) for r in recovery}
    if len(replacement)!=len(recovery):raise ValueError('duplicate shard replacement')
    spec=importlib.util.spec_from_file_location('frozen_selected',root/'evaluate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sources[str(root/'evaluate.py')]=hashlib.sha256((root/'evaluate.py').read_bytes()).hexdigest()
    all_pairs=[]
    for shard in range(plan['shards']):
        path=replacement.get(shard,root/f'shard_{shard}')
        done=read(path/'completion.json')
        if done['status']!='COMPLETE' or done['plan_sha256']!=plan_hash or done['checkpoint']!=plan['checkpoint']:
            raise ValueError('missing or mismatched shard completion')
        pairs=read(path/'primary_pairs.json')
        selected=[r for r in plan['selected'] if r['shard']==shard]
        expected={(module.state_key(r),rep) for r in selected for rep in range(plan['repetitions'])}
        actual=[(module.state_key(r),r['repetition']) for r in pairs]
        if len(actual)!=len(expected) or set(actual)!=expected or done['selected_pairs']!=len(pairs):
            raise ValueError('missing or duplicated selected pair')
        refs=read(root/f'shard_{shard}'/'natural_references.json')
        for row in pairs:
            arms=read(path/f'arms_{row["stream"]}_{row["cohort"]}_{row["repetition"]}.json')
            for name in ('stop','continue'):
                arm=arms[name]
                if arm['entries']!=refs[str(row['prefix'])]:
                    raise ValueError('pair does not start at exact natural-policy entry')
                if (arm['seed'],arm['repetition'],arm['prefix'])!=(row['seed'],row['repetition'],row['prefix']):
                    raise ValueError('saved arm identity mismatch')
            entry=arms['stop']['entries'][row['stream']]
            if entry['hash']!=row['entry_hash'] or entry['p_continue']!=row['probability']:
                raise ValueError('registered focal entry mismatch')
            a,b=[arms[name]['returns'][row['stream']] for name in ('stop','continue')]
            if a['stage_computations']!=0 or b['stage_computations']<1:
                raise ValueError('intervention not executed')
            for result in (a,b):
                if not np.isclose(result['net_truncated'],result['task_truncated']-result['current_stage_cost'],rtol=0,atol=1e-12):
                    raise ValueError('return accounting mismatch')
            expected_values=dict(delta_net=b['net_truncated']-a['net_truncated'],
                delta_task=b['task_truncated']-a['task_truncated'],
                added_queries=b['stage_computations']-a['stage_computations'],
                pair_tail_bound=sum(arms[name]['tail_bounds'][row['stream']] for name in arms))
            if not all(np.isclose(row[k],v,rtol=0,atol=1e-12) for k,v in expected_values.items()):
                raise ValueError('saved focal outcomes disagree with arms')
        if shard in replacement:
            original=read(root/f'shard_{shard}'/'primary_pairs.json')
            if not all(row in pairs for row in original):raise ValueError('valid original pair changed')
        all_pairs.extend(pairs)
    report=module.summarize_pairs(all_pairs,plan['selected'],plan['repetitions'])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (args.output/'primary_pairs.json').write_text(json.dumps(all_pairs,indent=2,allow_nan=False)+'\n')
    with (args.output/'selected_states.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(report['states'][0]));w.writeheader();w.writerows(report['states'])
    manifest=dict(status='ALL_REGISTERED_PAIRS_EXACT_REFERENCE_AND_ACCOUNTING_PASS',
        completed_at=time.time(),plan_sha256=plan_hash,sources=sources,
        paired_repetitions=len(all_pairs),selected_states=len(report['states']),
        replacements={str(k):str(v) for k,v in replacement.items()},double_counted=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'completion.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(__file__,args.output/Path(__file__).name)
    print(json.dumps({k:v for k,v in report.items() if k!='states'}),flush=True)


if __name__=='__main__':main()
