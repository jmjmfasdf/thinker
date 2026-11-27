#!/usr/bin/env python3
"""Frozen, original-order re-capture of sub001–004 × three games at 10M.

Reuses the audited checkpoint/source preparation and guarded capture runner
from the previous campaign, copied into this new campaign (never edited).
Only recorder/trace source is overlaid on the original inference runtime.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path('/home/jeongmin/thinker-dynamic-imitation')
CAMPAIGNS = REPO/'logs/visualization/campaigns'
ROOT = CAMPAIGNS/'visualization_hxln_10m_allframes_20261126'
SOURCE = ROOT/'runtime_source'
PREVIOUS = CAMPAIGNS/'visualization_hxln_10m_sub001_004_cpu_20261126'
ORIGINAL_RUNTIME = CAMPAIGNS/'visualization_hxln_10m_sub001_20261123/runtime_source'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def save(path, value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def freeze():
    if ROOT.exists():raise FileExistsError('Refusing to overwrite a campaign: '+str(ROOT))
    ROOT.mkdir()
    shutil.copytree(ORIGINAL_RUNTIME,SOURCE)
    for name in ('thinker/visual_behav.py','thinker/thinker/behavior_trace.py'):
        destination=SOURCE/name
        destination.chmod(0o644)
        shutil.copyfile(REPO/name,destination)
        destination.chmod(0o444)
    sums=SOURCE/'SOURCE.sha256';sums.chmod(0o644)
    files=sorted(p for p in SOURCE.rglob('*') if p.is_file() and p!=sums and '__pycache__' not in p.parts)
    sums.write_text(''.join(f'{sha(p)}  {p.relative_to(SOURCE)}\n' for p in files));sums.chmod(0o444)
    shutil.copyfile(PREVIOUS/'cpu_capture.py',ROOT/'base_capture.py')
    shutil.copyfile(Path(__file__),ROOT/'campaign.py')
    shutil.copyfile(REPO/'thinker/tools/visual_allframes_job.pbs',ROOT/'job.pbs')
    for name in ('base_capture.py','campaign.py','job.pbs'):(ROOT/name).chmod(0o444)


def base():
    spec=importlib.util.spec_from_file_location('allframes_base',ROOT/'base_capture.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=ROOT;module.SOURCE=SOURCE;module.CAPTURES=REPO/'logs/visualization/hxln_10m_allframes'
    sys.path.insert(0,str(SOURCE/'thinker'))
    return module


def audit_original_order(module, manifest):
    import numpy as np
    from thinker.bc_loader import FrameStackedBehavioralDataLoader
    reports=[]
    for task in manifest['runs']:
        loader=FrameStackedBehavioralDataLoader(base_path=module.DATA,subjects=[task['subject']],
            game_id=task['game_id'],sessions=None,split=None,num_actions=9 if task['game_id']==0 else 6,
            source_paths=[module.DATA/s['source_id'] for s in task['sources']],
            decision_hz=None,max_time_gap=2.5/15.)
        observations=edges=0
        for index,record in enumerate(loader.file_records):
            with np.load(record.path,allow_pickle=False) as raw:n=len(raw['action'])
            episodes=[e for e in loader._episodes if e.file_index==index]
            observed=np.concatenate([e.observation_source_index for e in episodes])
            if not np.array_equal(observed,np.arange(n)):
                raise ValueError('Original observation coverage is not exhaustive: '+str(record.path))
            for ep in episodes:
                if not np.all(np.diff(ep.observation_source_index)==1):raise ValueError('Skipped source frame')
            observations+=n;edges+=sum(len(e.actions) for e in episodes)
        reports.append(dict(task=task['key'],observations=observations,transitions=edges,
            episodes=len(loader._episodes),all_original_observations_verified=True))
    save(ROOT/'reports/original_order_audit.json',reports)


def prepare():
    freeze();module=base();module.prepare()
    manifest=json.loads((ROOT/'manifest.json').read_text())
    if len(manifest['runs'])!=12 or manifest['reused']:raise ValueError('Expected 12 new captures')
    audit_original_order(module,manifest)
    manifest.update(replay_sampling='original-order-v1',decision_hz=None,
        initial_transition_retained=True,unscored_initial_edges=2,
        backend_note='All 12 captures use CPU, full original chronology; previous captures are unchanged.',
        driver_hashes={name:sha(ROOT/name) for name in ('base_capture.py','campaign.py','job.pbs')})
    manifest['resources']['allowed_hosts']=[f'bml2-slave{n}' for n in (5,6,7,8)]
    manifest['resources']['node_constraint']='explicit allowed host per job'
    save(ROOT/'manifest.json',manifest)


def submit():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    if (ROOT/'submission.json').exists() and json.loads((ROOT/'submission.json').read_text()):
        raise FileExistsError('Nonempty submission record exists; inspect before resubmitting')
    submitted=[]
    save(ROOT/'submission.json',submitted)
    for task in manifest['runs']:
        host=f"bml2-slave{5+task['index']%4}"
        cmd=['/opt/pbs/bin/qsub','-N',f"raw-{task['game'][:3]}-{task['subject']:03}",
            '-l',f'select=1:ncpus=16:mem=32gb:host={host}',
            '-o',str(ROOT/'pbs'),'-v',f"VISUAL_INDEX={task['index']}",str(ROOT/'job.pbs')]
        job=subprocess.check_output(cmd,universal_newlines=True).strip()
        submitted.append(dict(index=task['index'],task=task['key'],job_id=job,host=host,command=cmd))
        save(ROOT/'submission.json',submitted)
        print(task['key'],job,host,flush=True)


def run(index):
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,digest in manifest['driver_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Campaign driver changed: '+name)
    module=base();module.run(index)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','submit','run']);p.add_argument('--index',type=int)
    args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='submit':submit()
    else:
        if args.index is None or not 0<=args.index<12:p.error('--index must be 0..11')
        run(args.index)
