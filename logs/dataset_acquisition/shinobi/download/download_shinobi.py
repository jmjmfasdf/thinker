#!/usr/bin/env python3
"""Resume public Shinobi BOLD and replay assets using git-annex checksums."""
from pathlib import Path
import argparse
import concurrent.futures
import csv
import datetime
import fcntl
import json
import os
import re
import subprocess
import time

ROOT = Path('/home/jeongmin/fmri/shinobi')
OUT = Path(__file__).resolve().parent
BEHAV = Path('/home/jeongmin/thinker-dynamic-imitation/behavioral_data_shinobi')
ANNEX = '/home/jeongmin/miniconda3/envs/thinker/bin/git-annex'
PUBLIC = ('sub-01/', 'sub-02/', 'sub-06/')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def tracked(repo):
    return subprocess.check_output(['git', 'ls-files', '-z'], cwd=repo).decode().rstrip('\0').split('\0')


def plan():
    rows = []
    for name in ('bids', 'fmriprep'):
        repo = ROOT / name
        for rel in tracked(repo):
            if not rel.startswith(PUBLIC):
                continue
            path = repo / rel
            group = None
            if name == 'bids':
                if rel.endswith('.bk2') or rel.endswith('_events.tsv') or rel.endswith('_scans.tsv'):
                    group = 'behavior'
                elif rel.endswith('_bold.nii.gz'):
                    group = 'raw_bold'
            else:
                if '_space-MNI152NLin2009cAsym_' in rel and rel.endswith('_desc-preproc_bold.nii.gz'):
                    group = 'mni_bold'
                elif '_space-MNI152NLin2009cAsym_' in rel and (rel.endswith('_desc-brain_mask.nii.gz') or rel.endswith('_boldref.nii.gz')):
                    group = 'mni_support'
                elif '_desc-confounds_timeseries.' in rel:
                    group = 'mni_support'
            if not group:
                continue
            key = os.path.basename(os.readlink(path)) if path.is_symlink() else None
            match = re.search(r'-s(\d+)--', key or '')
            expected = int(match.group(1)) if match else path.stat().st_size
            rows.append(dict(repo=name, path=rel, group=group, key=key, bytes=expected))
            if group == 'behavior':
                dest = BEHAV / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.is_symlink() and not dest.exists():
                    dest.symlink_to(path)
    commits = {name: subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT/name).decode().strip()
               for name in ('bids', 'fmriprep', 'bids/stimuli')}
    write_json(OUT/'download_plan.json', dict(created_at=now(), public_subjects=['01','02','06'],
               commits=commits, remote='conp-ria-storage-http', files=rows,
               scope='In-scanner replay/events + raw functional BOLD + MNI preprocessed BOLD/masks/boldref/confounds; no duplicate T1w/CIFTI outputs or physiology.'))
    for name in ('stimuli', 'source_dataset'):
        dest = BEHAV/name
        target = ROOT/'bids'/('stimuli' if name=='stimuli' else '')
        if not dest.is_symlink() and not dest.exists():
            dest.symlink_to(target, target_is_directory=True)
    write_json(BEHAV/'provenance.json', dict(created_at=now(), source=str(ROOT/'bids'), public_subjects=['01','02','06'],
               source_commit=commits['bids'], source_url='https://github.com/courtois-neuromod/shinobi',
               note='Links to original in-scanner BK2, events and scan timing; missing original recordings are preserved as missing in events.'))
    return rows


def available(row):
    p = ROOT/row['repo']/row['path']
    try:
        return p.stat().st_size == row['bytes']
    except FileNotFoundError:
        return False


def snapshot(rows, state, start):
    groups = {}
    for row in rows:
        g = groups.setdefault(row['group'], dict(files=0, available_files=0, bytes=0, available_bytes=0))
        g['files'] += 1
        g['bytes'] += row['bytes']
        if available(row):
            g['available_files'] += 1
            g['available_bytes'] += row['bytes']
    data = dict(updated_at=now(), started_at=start, pid=os.getpid(), state=state, groups=groups,
                verification='git-annex validates payload checksums; these progress counts count complete files, not partial transfers.')
    write_json(OUT/'status.json', data)
    return data


def worker(rows, groups, label):
    with (OUT/(label+'.jsonl')).open('a', buffering=1) as logfile:
        for group in groups:
            selected = [r for r in rows if r['group']==group and r['key']]
            if not selected:
                continue
            for attempt in range(1, 5):
                pending = [r for r in selected if not available(r)]
                if not pending:
                    break
                logfile.write(json.dumps(dict(event='attempt',group=group,attempt=attempt,pending=len(pending),time=now()))+'\n')
                args = [ANNEX,'get','--from=conp-ria-storage-http','--jobs=3','--json','--']+[r['path'] for r in pending]
                subprocess.run(args, cwd=ROOT/pending[0]['repo'], stdout=logfile, stderr=logfile)
                if attempt < 4 and any(not available(r) for r in pending):
                    time.sleep(30)
    return label


def run(rows):
    with (OUT/'download.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start=now()
        (OUT/'pid').write_text(str(os.getpid())+'\n')
        snapshot(rows,'RUNNING',start)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,rows,['behavior','raw_bold'],'bids'),
                     pool.submit(worker,rows,['mni_support','mni_bold'],'fmriprep')]
            while not all(f.done() for f in futures):
                time.sleep(20)
                snapshot(rows,'RUNNING',start)
            for f in futures:
                f.result()
        missing=[r for r in rows if not available(r)]
        write_json(OUT/'missing_after_download.json',missing)
        snapshot(rows,'COMPLETE' if not missing else 'PARTIAL_RETRY_REQUIRED',start)
        print(json.dumps(dict(state='COMPLETE' if not missing else 'PARTIAL_RETRY_REQUIRED',missing=len(missing),time=now())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    rows=plan()
    if args.plan_only:
        print(json.dumps(snapshot(rows,'PLANNED',now()),indent=2))
    else:
        run(rows)
