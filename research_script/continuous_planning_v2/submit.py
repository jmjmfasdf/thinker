"""Immutable snapshots, pilot-first submission, explicit review gate for expansion."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

REPO=Path('/home/jeongmin/thinker-dynamic-imitation')
HERE=Path(__file__).resolve().parent
DEFAULT_OUT=REPO/'research_script/outputs/continuous_planning_v2'
DEFAULT_V1=REPO/'research_script/outputs/continuous_planning_v1'
FIGURES=('02-F1','02-F2','02-F3','02-F4','02-F5','02-F6','02-F7',
         '03-F1','03-F2','03-F3','03-S1','04-F1','04-F1b','04-F2','04-F3','N1','N2','N3','02-S1')

NODES=['bml2-slave5','bml2-slave6','bml2-slave7','bml2-slave8']


def hash_file(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze_snapshot(snapshot):
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+list(HERE.glob('*.pbs'))
    paths+=[REPO/'thinker/thinker/behavior_trace.py']
    hashes=[]
    for source in sorted(paths):
        relative=source.relative_to(REPO);target=snapshot/relative
        if target.exists():
            if hash_file(target)!=hash_file(source):raise ValueError('Immutable snapshot differs: '+str(target))
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);target.chmod(0o444)
        hashes.append(hash_file(target)+'  '+str(relative))
    sums='\n'.join(hashes)+'\n';manifest=snapshot/'SHA256SUMS'
    if manifest.exists() and manifest.read_text()!=sums:raise ValueError('Snapshot inventory changed')
    if not manifest.exists():manifest.write_text(sums);manifest.chmod(0o444)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUT)
    p.add_argument('--v1-root',type=Path,default=DEFAULT_V1)
    p.add_argument('--phase',choices=('pilot','full'),default='pilot')
    p.add_argument('--submit',action='store_true')
    p.add_argument('--resume',action='store_true',help='Resume interrupted submission without duplicating recorded jobs')
    args=p.parse_args()
    out=args.output_root.resolve();v1=args.v1_root.resolve()
    if out==v1 or v1 in out.parents:raise ValueError('Cannot write V2 inside V1')
    jobs=[(1,0),(1,1)] if args.phase=='pilot' else [(s,g) for s in range(2,7) for g in (0,1)]
    print(json.dumps(dict(phase=args.phase,jobs=jobs,nodes=NODES,walltime='168:00:00',
        review_gate='all 19 figures in both sub001 games must be manually reviewed before full expansion'),indent=2))
    if not args.submit:return
    if not shutil.which('qsub'):
        raise RuntimeError('Run submit.py using host /usr/bin/python3; qsub is outside the Singularity container')
    snapshot=out/('source' if args.phase=='pilot' else 'source_full')
    if args.phase=='pilot' and not args.resume:
        if out.exists():raise FileExistsError('Preserve existing campaign; use another output root')
        (out/'jobs').mkdir(parents=True)
        freeze_snapshot(snapshot)
    elif args.phase=='full':
        review_path=out/'visual_review.json'
        if not review_path.exists():raise ValueError('Manual visual review required; no full-cohort submission')
        review=json.loads(review_path.read_text())
        expected={f'sub001_game{g}/{fig}.png' for g in (0,1) for fig in FIGURES}
        if review.get('status')!='APPROVED' or set(review.get('reviewed',{}))!=expected:
            raise ValueError('Review must cover every specified pilot figure in both games')
        for key,sha in review['reviewed'].items():
            tag,file=key.split('/')
            if hash_file(out/tag/'figures'/file)!=sha:raise ValueError('Pilot figure changed since review')
        for g in (0,1):
            if not (out/f'sub001_game{g}/completion.json').exists():raise ValueError('Pilot not completed')
        for name in ('render.py','panels.py','group_report.py'):
            if review.get('reviewed_sources',{}).get(name)!=hash_file(HERE/name):
                raise ValueError('Reviewed rendering / panel source changed: '+name)
        freeze_snapshot(snapshot)
    ledger_path=out/'jobs'/(args.phase+'_submission.json')
    if ledger_path.exists():
        if not args.resume:raise FileExistsError('Already submitted; use --resume for an interrupted submission')
        ledger=json.loads(ledger_path.read_text())
        if ledger['status']=='SUBMITTED':raise ValueError('Submission already completed')
    else:
        ledger=dict(status='SUBMITTING',time=datetime.now(timezone.utc).isoformat(),phase=args.phase,
            source=str(snapshot),v1=str(v1),jobs=[])
    def save():ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    save();ids=[r['job_id'] for r in ledger['jobs'] if r['subject']]
    for i,(subject,game) in enumerate(jobs+([(0,0)] if args.phase=='full' else [])):
        if any(r['subject']==subject and r['game']==game for r in ledger['jobs']):continue
        node=NODES[i%4];tag=f'sub{subject:03d}_game{game}'
        cmd=['qsub','-N','cpv2-'+tag,'-l',f'select=1:ncpus=8:mem=32gb:host={node}',
             '-l','walltime=168:00:00','-o',str(out/'jobs'/(tag+'.pbs.log')),
             '-v',f'V2_OUTPUT={out},V2_INPUT={v1},V2_SOURCE={snapshot},V2_SUBJECT={subject},V2_GAME={game}']
        dependencies=[]
        if subject==0:
            dependencies=ids.copy()
            # V1 group must finish before its canonical primary file is copied.
            for record in json.loads((v1/'jobs/submission.json').read_text())['jobs']:
                if record.get('subject')==0:
                    check=subprocess.run(['qstat',record['job_id']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    if check.returncode==0:dependencies.append(record['job_id'])
            cmd+=['-W','depend=afterok:'+':'.join(dependencies)]
        cmd.append(str(snapshot/'research_script/continuous_planning_v2/campaign.pbs'))
        job=subprocess.check_output(cmd,universal_newlines=True).strip()
        if subject:ids.append(job)
        ledger['jobs'].append(dict(subject=subject,game=game,node=node,job_id=job,dependencies=dependencies));save()
        print('submitted',tag,job,node,flush=True)
    ledger['status']='SUBMITTED';save()


if __name__=='__main__':main()
