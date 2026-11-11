#!/usr/bin/env python3
"""Move not-yet-started retry tasks to slave1--4 without stopping captures.

The old frozen driver has an in-memory executor queue and no cancellation API.
An atomic directory fence at its initial running-report path makes queued
dispatch fail *before* opening a capture. Existing running reports cannot be
fenced. ThreadPoolExecutor drains running captures before the old driver exits.
Its dispatch-fence exit is expected, not a capture failure; reconcile checks
individual capture receipts after all eight jobs, regardless of job exit code.
Frozen old source is never edited or injected into a running process.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

from visualization_10m import REPO, read, write, sha, run_task
from visualization_10m_retry import checked_run_task

PARENT = REPO / 'research_script/outputs/allcheckpoints_allsessions_10m_20261110_voc_retry'
DEFAULT_OUT = PARENT / 'redistribution_20261110'


def fence(parent, index):
    """mkdir and the old driver's atomic file replace cannot both succeed."""
    reports = parent / 'reports'
    if any((reports / ('task-%02d.%s.json' % (index, s))).exists()
           for s in ('completed', 'failed')):
        return None
    path = reports / ('task-%02d.running.json' % index)
    try:
        path.mkdir()
    except FileExistsError:
        return None  # The old worker already started (or someone else fenced it).
    return path


def self_test():
    with tempfile.TemporaryDirectory(prefix='dispatch-fence-test-') as tmp:
        root = Path(tmp)
        (root / 'reports').mkdir()
        # A started task must never be fenced or changed.
        started = root / 'reports/task-00.running.json'
        write(started, {'status': 'RUNNING'})
        before = sha(started)
        assert fence(root, 0) is None and sha(started) == before
        barrier = fence(root, 1)
        assert barrier is not None and barrier.is_dir()
        with patch('subprocess.Popen', side_effect=AssertionError('Duplicate capture launched')) as popen:
            try:
                run_task(root, {'index': 1, 'capture': '/not-used'})
            except IsADirectoryError:
                pass
            else:
                raise AssertionError('Original dispatcher bypassed the fence')
            assert not popen.called
        # Racing atomic dispatch vs. fence has exactly one winner.
        for i in range(2, 34):
            path = root / 'reports' / ('task-%02d.running.json' % i)
            def dispatch():
                try:
                    write(path, {'status': 'RUNNING'})
                    return True
                except IsADirectoryError:
                    return False
            with ThreadPoolExecutor(max_workers=2) as pool:
                a = pool.submit(dispatch)
                b = pool.submit(fence, root, i)
                assert int(a.result()) + int(b.result() is not None) == 1
    print('DISPATCH_FENCE_TESTS_PASSED', flush=True)


def prepare(out):
    self_test()
    if out.exists():
        raise FileExistsError(out)
    parent = read(PARENT / 'campaign.json')
    verification = read(PARENT / 'reports/verification.json')
    if verification['status'] != 'PASSED' or verification['source_sha256'] != sha(PARENT / 'source/SHA256SUMS'):
        raise RuntimeError('Parent source verification missing')
    for name in ('reports', 'pbs'):
        (out / name).mkdir(parents=True)
    shutil.copytree(PARENT / 'source', out / 'source')
    for name in ('visualization_10m_redistribute.py', 'visualization_10m_redistribute.pbs'):
        shutil.copy2(REPO / 'research_script' / name, out / 'source/research_script' / name)
    sums = out / 'source/SHA256SUMS'
    sums.unlink()
    entries = []
    for path in sorted((out / 'source').rglob('*')):
        if path.is_file():
            entries.append('%s  %s\n' % (sha(path), path.relative_to(out / 'source')))
            path.chmod(0o444)
    sums.write_text(''.join(entries))
    sums.chmod(0o444)
    tasks, retained = [], []
    nodes = [dict(index=n, host='bml2-slave%d' % n, ncpus=28,
                  concurrency=3, tasks=[]) for n in range(1, 5)]
    for task in parent['tasks']:
        barrier = fence(PARENT, task['index'])
        if barrier is None:
            retained.append(task['index'])
            continue
        target = task['node'] - 4
        moved = dict(task, previous_node=task['node'], node=target)
        write(barrier / 'DELEGATED.json', dict(status='DELEGATED', index=task['index'],
            from_node=task['node'], to_node=target, campaign=str(out), created_at=time.time(),
            explanation='Atomic pre-capture dispatch fence; old driver drains active captures then exits'))
        nodes[target - 1]['tasks'].append(task['index'])
        tasks.append(moved)
        print('FENCED_AND_MOVED', task['index'], task['node'], target, flush=True)
    campaign = dict(parent, kind='queued-retry-redistribution', created_at=time.time(),
        parent_retry=str(PARENT), tasks=tasks, nodes=[n for n in nodes if n['tasks']],
        retained_indices=retained, checkpoint_count=len(tasks),
        checkpoint_source_pairs=sum(len(t['sources']) for t in tasks),
        source_sha256=sha(sums), parent_source_sha256=verification['source_sha256'],
        placement_authority='User explicitly requested redistribution to slave1-4',
        dispatch_fence='Atomic directory at original not-yet-created running report',
        status='PREPARED')
    write(out / 'campaign.json', campaign)
    write(PARENT / 'redistribution_index.json', dict(campaign=str(out / 'campaign.json'),
        moved_indices=[t['index'] for t in tasks], retained_indices=retained,
        original_driver_exit_note='Dispatch-fence IsADirectoryError is expected after active captures drain'))
    print('PREPARED', len(tasks), 'moved;', len(retained), 'untouched', flush=True)


def run_node(out, index):
    campaign = read(out / 'campaign.json')
    node = next(n for n in campaign['nodes'] if n['index'] == index)
    if index not in range(1, 5) or os.environ.get('HOSTNAME', '').split('.')[0] != node['host']:
        raise RuntimeError('Explicit slave1-4 placement required')
    tasks = {t['index']: t for t in campaign['tasks']}
    for i in node['tasks']:
        barrier = PARENT / 'reports' / ('task-%02d.running.json' % i)
        if not barrier.is_dir() or read(barrier / 'DELEGATED.json')['campaign'] != str(out):
            raise RuntimeError('Original dispatch is not fenced; refusing duplicate capture')
        if (out / 'reports' / ('task-%02d.running.json' % i)).exists():
            raise RuntimeError('Redistributed capture already started')
    write(out / 'reports' / ('node-%d.running.json' % index), dict(status='RUNNING',
        node=node, job_id=os.environ.get('PBS_JOBID'), started_at=time.time()))
    with ThreadPoolExecutor(max_workers=node['concurrency']) as pool:
        futures = [pool.submit(checked_run_task, out, tasks[i]) for i in node['tasks']]
        ok = [f.result() for f in as_completed(futures)]
    write(out / 'reports' / ('node-%d.completed.json' % index), dict(
        status='COMPLETED' if all(ok) else 'FAILED', succeeded=sum(ok), failed=len(ok)-sum(ok),
        node=node, completed_at=time.time()))
    if not all(ok):
        raise RuntimeError('Redistributed capture failure; inspect task reports')


def reconcile(out):
    campaign = read(out / 'campaign.json')
    parent = read(PARENT / 'campaign.json')
    moved = {t['index'] for t in campaign['tasks']}
    errors, receipts = [], []
    for task in parent['tasks']:
        owner = out if task['index'] in moved else PARENT
        report = owner / 'reports' / ('task-%02d.completed.json' % task['index'])
        capture_receipt = Path(task['capture']) / 'COMPLETED.json'
        if not report.is_file() or not capture_receipt.is_file():
            errors.append(dict(index=task['index'], reason='Capture completion proof missing', owner=str(owner)))
            continue
        result = read(report)
        if result['status'] != 'COMPLETED' or result['completion_sha256'] != sha(capture_receipt):
            errors.append(dict(index=task['index'], reason='Completion proof mismatch'))
        else:
            receipts.append(dict(index=task['index'], report=str(report), sha256=sha(capture_receipt)))
    for item in parent['preserved_completed'] + parent['preserved_episode_manifests']:
        if sha(item['path']) != item['sha256']:
            errors.append(dict(reason='Preserved result changed', path=item['path']))
    report = dict(status='FAILED' if errors else 'COMPLETED', completed_at=time.time(),
        errors=errors, checkpoint_count=72, retried_checkpoint_count=24,
        preserved_checkpoint_count=48, redistributed_checkpoint_count=len(moved),
        retry_campaign=str(PARENT), redistribution_campaign=str(out),
        visual_root=parent['visual_root'], cache_generated=False, receipts=receipts,
        old_dispatch_fences='Expected driver exits; capture receipts checked independently')
    write(out / 'reports/reconciliation.json', report)
    if errors:
        raise RuntimeError('Capture reconciliation failed; see reconciliation.json')
    write(out / 'COMPLETED.json', report)
    write(PARENT / 'COMPLETED.json', report)
    write(Path(parent['visual_root']) / 'RECOVERY_COMPLETED.json', report)
    print('ALL_72_CAPTURES_COMPLETE', flush=True)


def submit(out):
    campaign = read(out / 'campaign.json')
    if (out / 'submission.json').exists():
        raise FileExistsError('Refusing duplicate redistribution submission')
    if not campaign['tasks']:
        raise RuntimeError('No queued tasks remained')
    self_test()
    ledger = dict(status='SUBMITTING', created_at=time.time(), jobs=[])
    write(out / 'submission.json', ledger)
    script = str(out / 'source/research_script/visualization_10m_redistribute.pbs')
    for node in campaign['nodes']:
        job = subprocess.check_output(['qsub', '-N', 'vis10x-n%d' % node['index'],
            '-l', 'select=1:ncpus=28:mem=112gb:host=%s' % node['host'],
            '-l', 'place=excl', '-l', 'walltime=168:00:00',
            '-o', str(out / 'pbs' / ('node-%d.pbs.log' % node['index'])),
            '-v', 'CAMPAIGN_OUTPUT=%s,CAMPAIGN_NODE=%d,CAMPAIGN_MODE=run-node' % (out, node['index']),
            script], text=True).strip()
        ledger['jobs'].append(dict(node, job_id=job))
        write(out / 'submission.json', ledger)
        print('SUBMITTED', job, node['host'], node['tasks'], flush=True)
    dependencies = [j['job_id'] for j in read(PARENT / 'submission.json')['jobs'] + ledger['jobs']]
    job = subprocess.check_output(['qsub', '-N', 'vis10x-check',
        '-l', 'select=1:ncpus=1:mem=2gb:host=bml2-slave1', '-l', 'walltime=168:00:00',
        '-W', 'depend=afterany:' + ':'.join(dependencies),
        '-o', str(out / 'pbs/reconcile.pbs.log'),
        '-v', 'CAMPAIGN_OUTPUT=%s,CAMPAIGN_NODE=1,CAMPAIGN_MODE=reconcile' % out,
        script], text=True).strip()
    ledger.update(status='SUBMITTED', reconcile_job_id=job, reconcile_afterany=dependencies)
    write(out / 'submission.json', ledger)
    print('RECONCILIATION_SUBMITTED', job, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('self-test', 'prepare', 'submit', 'run-node', 'reconcile'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.command == 'self-test': self_test()
    elif args.command == 'prepare': prepare(args.output)
    elif args.command == 'submit': submit(args.output)
    elif args.command == 'reconcile': reconcile(args.output)
    else: run_node(args.output, args.index)


if __name__ == '__main__':
    main()
