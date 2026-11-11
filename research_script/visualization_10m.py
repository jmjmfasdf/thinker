#!/usr/bin/env python3
"""All-session 10M capture campaign; no training, cache, or analysis jobs.

The 2026-11-09 user request explicitly allows slave1--8 for this campaign.
Eight exclusive node jobs use the published asynchronous milestone pairs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

REPO = Path('/home/jeongmin/thinker-dynamic-imitation')
TAG = 'allcheckpoints_allsessions_10m_20261109'
DEFAULT_OUT = REPO / 'research_script/outputs' / TAG
VISUAL = REPO / 'logs/visual_behav' / TAG
DATA = REPO / 'behavioral_data_block'
GAMES = ('enduro', 'pong', 'spaceinvaders')
MIN_FREE = 120 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-%d' % os.getpid())
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def api(out=None):
    base = out / 'source/thinker' if out is not None else REPO / 'thinker'
    sys.path.insert(0, str(base))
    import visual_behav
    return visual_behav


def prepare(out):
    v = api()
    if out.exists() or VISUAL.exists():
        raise FileExistsError('New campaign paths must not already exist')
    if shutil.disk_usage(REPO).free < 500 * 1024**3:
        raise RuntimeError('Less than 500 GiB free before campaign')
    tasks, source_sets = [], {}
    lanes = [1, 2, 3, 3, 4, 5, 6, 7, 8]
    for subject in range(1, 7):
        for leave in range(1, 5):
            for game, name in enumerate(GAMES):
                cp = REPO / 'logs/thinker' / (
                    'dynamic-imitation-%s-voc-tau001-gru-sub%d-leavesession%d-10m-s701'
                    % (name, subject, leave))
                bundle = v.resolve_checkpoint_bundle(cp, requested_step=10_000_000,
                                                      stable_polls=1)
                payload = v.validate_checkpoint_payload_bundle(bundle)
                flags = v._raw_config_mapping(bundle.config)
                if (payload['game_id'], str(flags['icopro_subjects']), str(flags['icopro_holdout_sessions'])) != (game, str(subject), str(leave)):
                    raise ValueError('Checkpoint subject/game/leave identity mismatch: ' + str(cp))
                key = (subject, game)
                if key not in source_sets:
                    paths = sorted((DATA / ('sub-%03d' % subject)).glob('ses-*/*game%d.npz' % game))
                    if len(paths) != 11 or {p.parent.name for p in paths} != {'ses-%02d' % s for s in range(1, 5)}:
                        raise ValueError('Incomplete four-session behavioral sources: %r' % (key,))
                    source_sets[key] = [dict(path=str(p), source_id=str(p.relative_to(DATA)),
                        sha256=sha(p), size=p.stat().st_size) for p in paths]
                index = len(tasks)
                tasks.append(dict(index=index, game=game, game_name=name,
                    subject=subject, leave_session=leave,
                    checkpoint=str(cp), bundle_sha256=bundle.bundle_sha256,
                    bound_checkpoint=asdict(bundle), node=lanes[index % len(lanes)],
                    sources=source_sets[key],
                    capture=str(VISUAL / 'captures' / ('%s-sub%03d-leave%02d-all' % (name, subject, leave)))))
                print('CHECKPOINT_VERIFIED', index, name, subject, leave,
                      payload['actor_real_step'], payload['model_real_step'], flush=True)
    out.mkdir(parents=True)
    VISUAL.mkdir(parents=True)
    for name in ('pbs', 'reports', 'source/research_script'):
        (out / name).mkdir(parents=True, exist_ok=True)
    # Copy the updated live implementation, not the old 5M inference snapshot.
    shutil.copytree(REPO / 'thinker', out / 'source/thinker', ignore=shutil.ignore_patterns(
        '__pycache__', '.pytest_cache', 'build', '*.egg-info', '*.pyc', '*.cpp', '*.c',
        '*.cpython-312-*.so'))
    for name in ('visualization_10m.py', 'visualization_10m.pbs'):
        shutil.copy2(REPO / 'research_script' / name, out / 'source/research_script' / name)
    hashes = []
    for path in sorted((out / 'source').rglob('*')):
        if path.is_file():
            hashes.append('%s  %s\n' % (sha(path), path.relative_to(out / 'source')))
            path.chmod(0o444)
    (out / 'source/SHA256SUMS').write_text(''.join(hashes))
    (out / 'source/SHA256SUMS').chmod(0o444)
    nodes = [dict(index=n, host='bml2-slave%d' % n, ncpus=56 if n == 3 else 28,
                  concurrency=6 if n == 3 else 3,
                  tasks=[t['index'] for t in tasks if t['node'] == n]) for n in range(1, 9)]
    campaign = dict(status='PREPARED', created_at=time.time(), tasks=tasks, nodes=nodes,
        checkpoint_count=len(tasks), checkpoint_source_pairs=sum(len(t['sources']) for t in tasks),
        unique_source_count=sum(len(s) for s in source_sets.values()),
        visual_root=str(VISUAL), data_root=str(DATA), all_sessions=True, cache_requested=False,
        source_sha256=sha(out / 'source/SHA256SUMS'),
        placement_authority='User explicitly requested all eight slaves on 2026-11-09',
        storage=dict(compression='blosc-zstd', compression_level=9, shuffle='bitshuffle',
                     images='all', store_predicted_xs=False, trace_event_schema_version=3),
        resources=dict(walltime='168:00:00', mem_per_node='112gb', exclusive=True,
                       workers_per_checkpoint=3, threads_per_worker=2), min_free_bytes=MIN_FREE)
    write(out / 'campaign.json', campaign)
    write(VISUAL / 'campaign_index.json', dict(campaign=str(out / 'campaign.json'),
                                             captures=tasks))
    print('PREPARED', len(tasks), 'checkpoints', flush=True)


def probe(out, index):
    """Load real checkpoint weights and perform a small, synthetic trace QA."""
    v = api(out)
    import numpy as np
    from gymnasium import spaces
    task = read(out / 'campaign.json')['tasks'][index]
    bundle = v.resolve_checkpoint_bundle(task['checkpoint'], requested_step=10_000_000, stable_polls=1)
    if bundle.bundle_sha256 != task['bundle_sha256']:
        raise RuntimeError('Checkpoint changed since preparation')
    worker = v.WorkerSpec(0, 'cpu')
    config = v.CaptureConfig(checkpoint=bundle, data_root=str(DATA), output_root=task['capture'],
        scratch_root=None, base_seed=0, queue_size=2, chunk_events=256, images='all',
        tree_carry=True, policy_mode='checkpoint',
        determinism_backend_fingerprint=v.resolve_determinism_backend_fingerprint([worker]),
        resume=True, torch_threads_per_worker=2, source_hash_workers=1, episode_indices=(),
        compression_level=9, store_predicted_xs=False)
    runtime = v._CaptureRuntime(config, worker)
    spec = runtime.spec
    shape = (spec.frame_stack_n * (1 if spec.grayscale else 3),) + tuple(spec.target_size)
    batch = dict(observation_provider=np.zeros((4,) + shape, dtype=np.uint8),
        observation_space=spaces.Box(0, 255, shape=shape, dtype=np.uint8),
        actions_seq=np.zeros((1, 3), dtype=np.int64),
        rewards_seq=np.zeros((1, 3), dtype=np.float32),
        done_seq=np.array([[False, False, True]]), truncated_seq=np.zeros((1, 3), dtype=bool),
        initial_prev_action=np.zeros(1, dtype=np.int64), score_mask=np.array([[False, True, True]]))
    class Sink:
        images = 'none'
        def __init__(self): self.events = []
        def write_event(self, event): self.events.append(event)
    sink = Sink()
    try:
        runtime._ensure_runner(batch)
        summary = runtime._stream_episode(batch, sink)
        for e in sink.events:
            if e['real_transition']:
                assert e['actor_real_logits_pre_override_valid']
                np.testing.assert_array_equal(e['actor_real_logits_pre_override'], e['actor_policy'])
            assert bool(e['model_done_logit_valid']) == (e['internal_status'] == 4)
        assert summary['real_transition_count'] == 3
        result = dict(status='PASSED', kind='synthetic network/trace probe; not behavioral output',
                      index=index, game=task['game'], num_actions=spec.num_actions,
                      bundle_sha256=bundle.bundle_sha256, summary=summary)
        write(out / 'reports' / ('probe-%02d.json' % index), result)
        print('PROBE_PASSED', index, task['game_name'], summary, flush=True)
    finally:
        runtime.close()


def capture_command(out, task):
    result = [sys.executable, '-B', '-u', str(out / 'source/thinker/visual_behav.py'), 'capture',
        '--checkpoint-dir', task['checkpoint'], '--checkpoint-step', '10000000',
        '--data-root', str(DATA), '--output', task['capture'],
        '--devices', 'cpu', '--cpu-workers', '3', '--torch-threads-per-worker', '2',
        '--images', 'all', '--compression-level', '9', '--omit-predicted-xs',
        '--policy-mode', 'checkpoint', '--base-seed', '0', '--resume']
    for source in task['sources']:
        result += ['--source', source['source_id']]
    return result


def run_task(out, task):
    index = task['index']
    report = out / 'reports' / ('task-%02d' % index)
    started = time.time()
    status = dict(index=index, started_at=started, job_id=os.environ.get('PBS_JOBID'),
                  node=os.environ.get('HOSTNAME'), capture=task['capture'])
    write(str(report) + '.running.json', dict(status, status='RUNNING'))
    try:
        if shutil.disk_usage(out).free < MIN_FREE:
            raise RuntimeError('Less than 120 GiB free; refusing to begin another checkpoint')
        for role in ('config', 'actor', 'model', 'milestone'):
            record = task['bound_checkpoint'][role]
            if sha(record['path']) != record['sha256']:
                raise RuntimeError('Bound checkpoint input changed: ' + role)
        # Rehash the bound inputs in the actual recorder too.  Captures keep
        # independent source hashes, source IDs, sessions and checkpoint IDs.
        command = capture_command(out, task)
        with (out / 'pbs' / ('task-%02d.log' % index)).open('a') as log:
            with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True) as process:
                while True:
                    try:
                        code = process.wait(timeout=30)
                        if code:
                            raise subprocess.CalledProcessError(code, command)
                        break
                    except subprocess.TimeoutExpired:
                        if shutil.disk_usage(out).free < MIN_FREE:
                            os.killpg(process.pid, signal.SIGTERM)
                            try: process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                            raise RuntimeError('Stopped own capture below 120 GiB free; partial files preserved')
        # The recorder validates every completed episode and the capture as a
        # whole before publishing COMPLETED.json; retain its checksummed receipt.
        completed = Path(task['capture']) / 'COMPLETED.json'
        if not completed.is_file():
            raise RuntimeError('Recorder exited without a completed capture')
        manifest = read(Path(task['capture']) / 'manifest.json')
        semantic = manifest['semantic']
        if semantic['checkpoint']['bundle_sha256'] != task['bundle_sha256']:
            raise RuntimeError('Completed capture checkpoint identity differs from campaign')
        if {s['source_id']: s['sha256'] for s in semantic['sources']} != {
                s['source_id']: s['sha256'] for s in task['sources']}:
            raise RuntimeError('Completed capture source set differs from campaign')
        write(str(report) + '.completed.json', dict(status, status='COMPLETED',
            completed_at=time.time(), elapsed_seconds=time.time()-started,
            completion_receipt=read(completed), completion_sha256=sha(completed)))
        print('CAPTURE_COMPLETED', index, task['capture'], flush=True)
        return True
    except Exception:
        write(str(report) + '.failed.json', dict(status, status='FAILED',
              failed_at=time.time(), error=traceback.format_exc()))
        print('CAPTURE_FAILED', index, traceback.format_exc(), flush=True)
        return False


def run_node(out, node_index):
    campaign = read(out / 'campaign.json')
    node = campaign['nodes'][node_index - 1]
    if os.environ.get('HOSTNAME', '').split('.')[0] != node['host']:
        raise RuntimeError('PBS host does not match explicitly assigned node')
    write(out / 'reports' / ('node-%d.running.json' % node_index), dict(
        node=node, job_id=os.environ.get('PBS_JOBID'), started_at=time.time(), status='RUNNING'))
    with ThreadPoolExecutor(max_workers=node['concurrency']) as pool:
        futures = [pool.submit(run_task, out, campaign['tasks'][i]) for i in node['tasks']]
        ok = [future.result() for future in as_completed(futures)]
    write(out / 'reports' / ('node-%d.completed.json' % node_index), dict(
        status='COMPLETED' if all(ok) else 'FAILED', completed_at=time.time(),
        succeeded=sum(ok), failed=len(ok)-sum(ok), node=node))
    if not all(ok):
        raise RuntimeError('One or more checkpoint captures failed; see reports')
    if all((out / 'reports' / ('task-%02d.completed.json' % t['index'])).is_file()
           for t in campaign['tasks']):
        write(out / 'COMPLETED.json', dict(status='COMPLETED', checkpoint_count=72,
              completed_at=time.time(), visual_root=campaign['visual_root'], cache_generated=False))


def submit(out):
    campaign = read(out / 'campaign.json')
    if (out / 'submission.json').exists():
        raise FileExistsError('Refusing duplicate campaign submission')
    for index in (0, 1, 2):
        if read(out / 'reports' / ('probe-%02d.json' % index))['status'] != 'PASSED':
            raise RuntimeError('Representative network probes must pass before submission')
    ledger = dict(status='SUBMITTING', created_at=time.time(), jobs=[])
    write(out / 'submission.json', ledger)
    for node in campaign['nodes']:
        result = subprocess.check_output(['qsub', '-N', 'vis10m-n%d' % node['index'],
            '-l', 'select=1:ncpus=%d:mem=112gb:host=%s' % (node['ncpus'], node['host']),
            '-l', 'place=excl', '-l', 'walltime=168:00:00',
            '-o', str(out / 'pbs' / ('node-%d.pbs.log' % node['index'])),
            '-v', 'CAMPAIGN_OUTPUT=%s,CAMPAIGN_NODE=%d' % (out, node['index']),
            str(out / 'source/research_script/visualization_10m.pbs')], text=True).strip()
        ledger['jobs'].append(dict(node, job_id=result))
        write(out / 'submission.json', ledger)
        print('SUBMITTED', result, node['host'], len(node['tasks']), 'checkpoints', flush=True)
    ledger['status'] = 'SUBMITTED'
    write(out / 'submission.json', ledger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'probe', 'submit', 'run-node'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.output)
    elif args.command == 'probe': probe(args.output, args.index)
    elif args.command == 'submit': submit(args.output)
    else: run_node(args.output, args.index)


if __name__ == '__main__':
    main()
