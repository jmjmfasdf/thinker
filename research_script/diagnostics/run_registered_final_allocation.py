"""Run the already registered 1M allocation sample before long utility completes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--condition',choices=['control','burn8','gru10'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    campaign=args.campaign.resolve()
    plan=json.loads((campaign/'fast_allocation_1m/plan.json').read_text())
    manifest=json.loads((campaign/'manifest.json').read_text())
    item=next(r for r in manifest['runs'] if r['condition']==args.condition)
    if item['training_steps']!=plan['milestone'] or plan['milestone']!=1_000_000:
        raise ValueError('final registered training budget differs')
    run=Path(item['run_directory'])
    if not (run/'finish').exists():
        raise ValueError('successful training finish is required')
    checkpoint={}
    for kind in ('actor','model'):
        paths=[p for p in run.glob(f'ckp_{kind}.tar_step_*')
               if int(p.name.rsplit('_',1)[1])>=plan['milestone']]
        if not paths:
            raise ValueError('missing final checkpoint')
        path=min(paths,key=lambda p:int(p.name.rsplit('_',1)[1]))
        if int(path.name.rsplit('_',1)[1])>=plan['milestone']+2000:
            raise ValueError('final checkpoint outside registered allocation tolerance')
        checkpoint[kind]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              real_step=int(path.name.rsplit('_',1)[1]))
    program=campaign/'allocation_midpoint.py'
    if hashlib.sha256(program.read_bytes()).hexdigest()!=plan['allocation_script_sha256']:
        raise ValueError('frozen allocation implementation changed')
    binding_path=campaign/f'fast_allocation_1m/{args.condition}_binding.json'
    if binding_path.exists() or args.output.exists():
        raise ValueError('allocation output/binding already exists; do not overwrite')
    binding_path.write_text(json.dumps(dict(checkpoint=checkpoint,plan=plan),indent=2)+'\n')
    subprocess.run([sys.executable,str(program),'--campaign',str(campaign),
        '--output',str(args.output),'--conditions',args.condition,'--milestone',str(plan['milestone']),
        '--batch','8','--decisions','1536','--seeds','8301','8401'],check=True)
    summary=json.loads((args.output/f'{args.condition}_summary.json').read_text())
    rows=json.loads((args.output/f'{args.condition}_allocation.json').read_text())
    if summary['status']!='COMPLETE' or summary['checkpoint']!=checkpoint or summary['evaluation_epsilon']!=0:
        raise ValueError('result differs from the fixed final checkpoint/policy')
    identity=lambda r:(r['seed'],r['stream'],r['decision'])
    expected={(s,i,t) for s in (8301,8401) for i in range(8) for t in range(128,1664)}
    if len(rows)!=24576 or {identity(r) for r in rows}!=expected:
        raise ValueError('missing or duplicated registered decisions')
    hist=[0]*31
    for row in rows:
        if type(row['length']) is not int or not 1<=row['length']<=30:
            raise ValueError('invalid planning length')
        hist[row['length']]+=1
    if hist!=summary['histogram'] or sum(hist)!=summary['decisions']:
        raise ValueError('histogram differs from per-decision records')
    overlap=0
    primary=campaign/f'evaluation_{args.condition}/allocation.json'
    if primary.exists():
        indexed={identity(r):r for r in rows}
        primary_rows=json.loads(primary.read_text())
        for row in primary_rows:
            if indexed.get(identity(row))!=row:
                raise ValueError('fast and primary registered allocations differ')
            overlap+=1
    (args.output/'final_binding_audit.json').write_text(json.dumps(dict(
        status='REGISTERED_FINAL_CHECKPOINT_AND_DECISIONS_PASS',checkpoint=checkpoint,
        decisions=len(rows),exact_primary_overlap=overlap,independent_sample=False,
        role='Early availability of the same registered1M allocation, not an additional independent experiment.'),indent=2)+'\n')
    print(json.dumps(dict(status='FINAL_ALLOCATION_AUDIT_PASS',condition=args.condition,decisions=len(rows),
        exact_primary_overlap=overlap)),flush=True)


if __name__=='__main__':main()
