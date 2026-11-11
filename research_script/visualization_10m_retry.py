#!/usr/bin/env python3
"""Resume failed 10M captures with recording-only probability stabilization.

Original sources, checkpoint weights, successful captures, and old failure
reports remain intact.  This attempt has its own frozen source and ledger.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from visualization_10m import REPO, read, write, sha, run_task

BASE = REPO / 'research_script/outputs/allcheckpoints_allsessions_10m_20261109'
DEFAULT_OUT = REPO / 'research_script/outputs/allcheckpoints_allsessions_10m_20261110_voc_retry'
PATCHED = ('thinker/thinker/dynamic_imitation.py',
           'thinker/tests/test_dynamic_imitation_trace.py')
CASES = (
    (54, 'sub-005/ses-04/sub005-ses04-block3-game0.npz', 0, 259),
    (43, 'sub-004/ses-03/sub004-ses03-block6-game1.npz', 0, 575),
    (32, 'sub-003/ses-01/sub003-ses01-block1-game2.npz', 3, 626),
)


def non_recording_ast(text):
    tree = ast.parse(text)
    tree.body = [node for node in tree.body if getattr(node, 'name', '') not in
                 ('_trace_unit_probability', '_voc_trace_diagnostics')]
    return ast.dump(tree, include_attributes=False)


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    parent = read(BASE / 'campaign.json')
    tasks, preserved = [], []
    for task in parent['tasks']:
        report = BASE / 'reports' / ('task-%02d' % task['index'])
        capture = Path(task['capture'])
        if Path(str(report) + '.completed.json').exists():
            receipt = capture / 'COMPLETED.json'
            if not receipt.is_file():
                raise RuntimeError('Completed task lacks capture receipt')
            preserved.append(dict(index=task['index'], path=str(receipt), sha256=sha(receipt)))
        elif Path(str(report) + '.failed.json').exists():
            records = read(capture / 'capture_result.json')['records']
            failures = [r for r in records if r.get('ok') is False]
            if not failures or any(r.get('error') != 'VoC soft gate probability is invalid'
                                   for r in failures):
                raise RuntimeError('Unexpected failure type; inspect before resuming')
            tasks.append(dict(task, failed_sources=len(failures)))
        else:
            raise RuntimeError('Parent task not terminal; do not overlap campaigns')
    if len(tasks) != 24 or len(preserved) != 48:
        raise RuntimeError('Expected exactly 24 failed and 48 completed checkpoints')
    old = (BASE / 'source' / PATCHED[0]).read_text()
    new = (REPO / PATCHED[0]).read_text()
    if non_recording_ast(old) != non_recording_ast(new):
        raise RuntimeError('Changes outside telemetry functions require a separate audit')
    nodes = [dict(index=n, host='bml2-slave%d' % n, ncpus=28,
                  concurrency=3, tasks=[], weight=0) for n in range(5, 9)]
    for task in sorted(tasks, key=lambda t: (-t['failed_sources'], t['index'])):
        node = min((n for n in nodes if len(n['tasks']) < 6), key=lambda n: n['weight'])
        node['tasks'].append(task['index'])
        node['weight'] += task['failed_sources']
        task['node'] = node['index']
    for name in ('reports', 'pbs', 'previous_capture_results'):
        (out / name).mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASE / 'source', out / 'source')
    for name in PATCHED:
        target = out / 'source' / name
        target.unlink()  # Only the new, unpublished copy is replaced.
        shutil.copy2(REPO / name, target)
    for name in ('visualization_10m_retry.py', 'visualization_10m_retry.pbs'):
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
    (out / 'recording_only.patch').write_text(''.join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True), fromfile='original', tofile='retry')))
    existing = []
    for task in tasks:
        capture = Path(task['capture'])
        shutil.copy2(capture / 'capture_result.json', out / 'previous_capture_results' /
                     ('task-%02d.json' % task['index']))
        for manifest in sorted((capture / 'episodes').glob('*.zarr/manifest.json')):
            existing.append(dict(task=task['index'], path=str(manifest), sha256=sha(manifest)))
    campaign = dict(parent, kind='10m-recording-probability-retry',
        base_campaign=str(BASE), created_at=time.time(), tasks=sorted(tasks, key=lambda t: t['index']),
        nodes=nodes, checkpoint_count=len(tasks), total_checkpoint_count=72,
        checkpoint_source_pairs=sum(len(t['sources']) for t in tasks),
        preserved_completed=preserved, preserved_episode_manifests=existing,
        source_sha256=sha(sums), changed_source_files=list(PATCHED),
        placement_authority='Default slave5-8 placement for retry',
        recording_probability='float64 softmax/sum; <=4 precision eps roundoff clamp; float32 storage',
        status='PREPARED')
    write(out / 'campaign.json', campaign)
    write(BASE / 'retry_index.json', dict(retry_campaign=str(out / 'campaign.json'),
          source_sha256=campaign['source_sha256'], resumed_indices=[t['index'] for t in tasks]))
    print('PREPARED', len(tasks), 'retry checkpoints;', len(preserved),
          'completed captures and', len(existing), 'existing retry episodes preserved', flush=True)


def verify(out):
    """Test the frozen code, then replay the original three real failures."""
    tests = ['test_dynamic_imitation_trace.py', 'test_trace_additional_logits.py',
             'test_visual_compact_storage.py', 'test_visual_behav_recorder.py',
             'test_behavior_trace.py', 'test_visual_10m_bundle.py']
    result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider'] +
        [str(out / 'source/thinker/tests' / name) for name in tests],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    write(out / 'reports/unit_tests.json', dict(returncode=result.returncode, output=result.stdout))
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError('Frozen-source regression tests failed')
    import numpy as np
    import visual_behav as v
    from thinker import dynamic_imitation as d
    if Path(d.__file__).resolve() != out / 'source/thinker/thinker/dynamic_imitation.py':
        raise RuntimeError('Verification must import the frozen retry source')
    fixed = d._voc_trace_diagnostics
    old_tree = ast.parse((BASE / 'source' / PATCHED[0]).read_text())
    old_node = next(n for n in old_tree.body if getattr(n, 'name', '') == '_voc_trace_diagnostics')
    namespace = dict(vars(d))
    exec(compile(ast.Module(body=[old_node], type_ignores=[]), '<original-telemetry>', 'exec'), namespace)
    original = namespace['_voc_trace_diagnostics']
    parent = read(BASE / 'campaign.json')
    results = []

    class ReplayedPastFailure(Exception):
        pass

    class Sink:
        images = 'none'
        def __init__(self, limit):
            self.rows, self.limit = [], limit
        def write_event(self, row):
            digest = hashlib.sha256()
            for key in ('accepted_primary_action', 'accepted_control', 'actor_policy',
                        'actor_control_logits', 'actor_action_probability',
                        'actor_real_logits_pre_override', 'internal_status', 'stop',
                        'real_cursor', 'root_carried'):
                value = np.asarray(row[key])
                digest.update(key.encode() + str(value.dtype).encode() + value.tobytes())
            self.rows.append(digest.hexdigest())
            if len(self.rows) >= self.limit:
                raise ReplayedPastFailure()

    for index, selector, episode, expected_failure_count in CASES:
        task = next(t for t in parent['tasks'] if t['index'] == index)
        print('REAL_REPLAY', index, selector, episode, flush=True)
        runtime = behavior = batch = None
        with tempfile.TemporaryDirectory(prefix='voc-retry-verify-') as scratch:
            try:
                bundle = v.resolve_checkpoint_bundle(task['checkpoint'], requested_step=10000000,
                                                     stable_polls=1)
                if bundle.bundle_sha256 != task['bundle_sha256']:
                    raise RuntimeError('Checkpoint changed')
                worker = v.WorkerSpec(0, 'cpu')
                config = v.CaptureConfig(checkpoint=bundle, data_root=parent['data_root'],
                    output_root=scratch, scratch_root=scratch, base_seed=0, queue_size=2,
                    chunk_events=256, images='all', tree_carry=True, policy_mode='checkpoint',
                    determinism_backend_fingerprint=v.resolve_determinism_backend_fingerprint([worker]),
                    resume=True, torch_threads_per_worker=2, source_hash_workers=1,
                    episode_indices=(), compression_level=9, store_predicted_xs=False)
                sources = v.enumerate_source_bundles(parent['data_root'], selectors=[selector],
                                                     stable_polls=1, hash_workers=1)
                source = sources[0]
                if source.sha256 != next(s['sha256'] for s in task['sources'] if s['source_id'] == selector):
                    raise RuntimeError('Source changed')
                source_task = v._coordinator_source_tasks(config, sources)[0]
                runtime = v._CaptureRuntime(config, worker)
                behavior = runtime.trace_api.BehaviorEpisodeSource(v._SourceLoaderView(source_task),
                    scratch_dir=scratch, allowed_source_paths=[source.path], expected_sources={
                        source.path: {key: getattr(source, key) for key in
                                      ('size', 'sha256', 'device', 'inode', 'mtime_ns')}})
                descriptor = next(e for e in behavior.episodes if int(e.local_episode_index) == episode)
                batch = v._episode_batch(behavior, descriptor)
                runtime._ensure_runner(batch)
                seed = v.episode_seed(0, source, episode)
                v._set_episode_seed(seed, runtime.torch, runtime.numpy)
                old_sink = Sink(expected_failure_count + 1)
                d._voc_trace_diagnostics = original
                try:
                    runtime._stream_episode(batch, old_sink)
                except RuntimeError as error:
                    if str(error) != 'VoC soft gate probability is invalid':
                        raise
                else:
                    raise RuntimeError('Original failure did not reproduce')
                if len(old_sink.rows) != expected_failure_count:
                    raise RuntimeError('Original failure location differs')
                d._voc_trace_diagnostics = fixed
                v._set_episode_seed(seed, runtime.torch, runtime.numpy)
                new_sink = Sink(expected_failure_count + 128)
                try:
                    runtime._stream_episode(batch, new_sink)
                except ReplayedPastFailure:
                    pass
                if new_sink.rows[:expected_failure_count] != old_sink.rows:
                    raise RuntimeError('Actor/control trajectory changed')
                if len(new_sink.rows) != expected_failure_count + 128:
                    raise RuntimeError('Fixed replay did not pass the original failure')
                record = dict(index=index, source=selector, episode=episode, status='PASSED',
                    original_failure_events=expected_failure_count, fixed_events=len(new_sink.rows),
                    actor_and_control_prefix_bitwise_equal=True)
                results.append(record)
                write(out / 'reports' / ('real-replay-%02d.json' % index), record)
                print('REAL_REPLAY_PASSED', record, flush=True)
            finally:
                d._voc_trace_diagnostics = fixed
                if batch is not None:
                    close = getattr(batch['observation_provider'], 'close', None)
                    if close:
                        close()
                if behavior is not None:
                    behavior.close()
                if runtime is not None:
                    runtime.close()
    write(out / 'reports/verification.json', dict(status='PASSED', completed_at=time.time(),
        source_sha256=sha(out / 'source/SHA256SUMS'), unit_tests='121 passed', real_replays=results))


def checked_run_task(out, task):
    if not run_task(out, task):
        return False
    # The recorder validates and reuses complete Zarr episodes on --resume.
    campaign = read(out / 'campaign.json')
    for item in campaign['preserved_episode_manifests']:
        if item['task'] == task['index'] and sha(item['path']) != item['sha256']:
            report = out / 'reports' / ('task-%02d' % task['index'])
            Path(str(report) + '.completed.json').rename(str(report) + '.preservation-check-failed.json')
            write(str(report) + '.failed.json', dict(status='FAILED',
                  error='Previously completed episode manifest changed', path=item['path']))
            return False
    return True


def run_node(out, index):
    campaign = read(out / 'campaign.json')
    node = next(n for n in campaign['nodes'] if n['index'] == index)
    if index not in range(5, 9) or os.environ.get('HOSTNAME', '').split('.')[0] != node['host']:
        raise RuntimeError('Retry requires the explicitly assigned slave5-8 host')
    tasks = {t['index']: t for t in campaign['tasks']}
    write(out / 'reports' / ('node-%d.running.json' % index), dict(
        status='RUNNING', started_at=time.time(), node=node, job_id=os.environ.get('PBS_JOBID')))
    with ThreadPoolExecutor(max_workers=node['concurrency']) as pool:
        futures = [pool.submit(checked_run_task, out, tasks[i]) for i in node['tasks']]
        ok = [future.result() for future in as_completed(futures)]
    write(out / 'reports' / ('node-%d.completed.json' % index), dict(
        status='COMPLETED' if all(ok) else 'FAILED', completed_at=time.time(),
        succeeded=sum(ok), failed=len(ok)-sum(ok), node=node))
    if not all(ok):
        raise RuntimeError('Retry failures; see per-checkpoint reports')
    if all((out / 'reports' / ('task-%02d.completed.json' % i)).is_file() for i in tasks):
        for item in campaign['preserved_completed']:
            if sha(item['path']) != item['sha256']:
                raise RuntimeError('An original completed capture receipt changed')
        receipt = dict(status='COMPLETED', completed_at=time.time(), checkpoint_count=72,
            retried_checkpoint_count=24, preserved_checkpoint_count=48, cache_generated=False,
            base_campaign=str(BASE), retry_campaign=str(out), visual_root=campaign['visual_root'])
        write(out / 'COMPLETED.json', receipt)
        write(Path(campaign['visual_root']) / 'RECOVERY_COMPLETED.json', receipt)


def submit(out):
    campaign = read(out / 'campaign.json')
    verification = read(out / 'reports/verification.json')
    if verification['status'] != 'PASSED' or verification['source_sha256'] != sha(out / 'source/SHA256SUMS'):
        raise RuntimeError('Frozen source must pass verification before submission')
    if (out / 'submission.json').exists():
        raise FileExistsError('Refusing duplicate retry submission')
    ledger = dict(status='SUBMITTING', created_at=time.time(), jobs=[])
    write(out / 'submission.json', ledger)
    for node in campaign['nodes']:
        job = subprocess.check_output(['qsub', '-N', 'vis10r-n%d' % node['index'],
            '-l', 'select=1:ncpus=28:mem=112gb:host=%s' % node['host'],
            '-l', 'place=excl', '-l', 'walltime=168:00:00',
            '-o', str(out / 'pbs' / ('node-%d.pbs.log' % node['index'])),
            '-v', 'CAMPAIGN_OUTPUT=%s,CAMPAIGN_NODE=%d' % (out, node['index']),
            str(out / 'source/research_script/visualization_10m_retry.pbs')], text=True).strip()
        ledger['jobs'].append(dict(node, job_id=job))
        write(out / 'submission.json', ledger)
        print('SUBMITTED', job, node['host'], node['tasks'], flush=True)
    ledger['status'] = 'SUBMITTED'
    write(out / 'submission.json', ledger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'verify', 'submit', 'run-node'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.output)
    elif args.command == 'verify': verify(args.output)
    elif args.command == 'submit': submit(args.output)
    else: run_node(args.output, args.index)


if __name__ == '__main__':
    main()
