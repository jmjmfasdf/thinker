#!/usr/bin/env python3
"""Build all 72 10M numeric caches from completed, read-only native captures.

No checkpoint inference, behavioral NPZ reads, or analysis fits are submitted.
Eight explicitly placed PBS workers share an atomic longest-first task queue.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

REPO = Path('/home/jeongmin/thinker-dynamic-imitation')
VISUAL = REPO / 'logs/visual_behav/allcheckpoints_allsessions_10m_20261109'
BASE = REPO / 'research_script/outputs/allcheckpoints_allsessions_10m_20261109'
DEFAULT_OUTPUT = REPO / 'research_script/outputs/00_dynamic_cache_10m_20261110'
RUNTIME_SUMS = Path('/home/jeongmin/visual-behav-assets/shared-runtime-25d766c04fe7-core.sha256')
SOURCE_FILES = (
    'research_script/00_cache.py', 'research_script/dynamic_analysis_cache.py',
    'research_script/dynamic_native_features.py', 'research_script/shared_analysis_cache.py',
    'research_script/shared_actor_policy_cache.py',
    'research_script/legacy/02_persistence_action_geometry.py',
    'research_script/legacy/02_actor_policy.py', 'thinker/thinker/behavior_trace.py',
    'research_script/dynamic_cache_10m.py', 'research_script/dynamic_cache_10m.pbs',
    'research_script/test_dynamic_cache_10m.py', 'research_script/test_dynamic_native_features.py',
)
MIN_FREE_BYTES = 120 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-%d' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def expected_keys():
    return {(s, g, l) for s in range(1, 7) for g in range(3) for l in range(1, 5)}


def validate_capture(task):
    capture = Path(task['capture']).resolve()
    m, proof = read(capture / 'manifest.json'), read(capture / 'COMPLETED.json')
    unsigned = {k: v for k, v in m.items() if k != 'manifest_sha256'}
    digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()
    require(proof['status'] == 'COMPLETED' and digest == proof['manifest_sha256'] == m['manifest_sha256'],
            'Capture completion or manifest hash mismatch: %s' % capture)
    cp = m['semantic']['checkpoint']
    require(cp['requested_step'] == 10000000 and cp['logical_xpid'] == Path(task['checkpoint']).name
            and cp['bundle_sha256'] == task['bundle_sha256'], 'Wrong checkpoint identity')
    episodes = m['episodes']
    require(len(episodes) == proof['episode_count'] and len(episodes) > 0, 'Wrong episode count')
    sources = {s['source_id']: s['sha256'] for s in m['semantic']['sources']}
    require(sources == {s['source_id']: s['sha256'] for s in task['sources']} and len(sources) == 11,
            'Source identity/coverage mismatch')
    require({e['metadata']['session'] for e in episodes} == {1, 2, 3, 4}, 'Missing source session')
    seen = set()
    for ep in episodes:
        meta = ep['metadata']
        require((meta['subject'], meta['game']) == (task['subject'], task['game']), 'Wrong subject/game')
        require(meta['checkpoint_bundle_sha256'] == task['bundle_sha256']
                and meta['checkpoint_real_step'] == 10000000, 'Episode checkpoint mismatch')
        require(sources.get(meta['source_id']) == meta['source_sha256'], 'Episode source mismatch')
        key = meta['source_id'], meta['source_episode_index']
        require(key not in seen, 'Duplicate source episode')
        seen.add(key)
        path = (capture / ep['path']).resolve()
        require(path.is_relative_to(capture) and (path / 'manifest.json').is_file(), 'Missing native episode')
        require(meta['images'] == 'all' and meta['store_predicted_xs'] is False, 'Unexpected image profile')
    if 'capture_manifest_sha256' in task:
        require(task['capture_manifest_sha256'] == digest, 'Capture changed after preparation')
    return m


def make_command(out, task, cache_root=None, max_files=None):
    command = [sys.executable, '-B', '-u', str(out / 'source/research_script/00_cache.py'),
               '--subject', str(task['subject']), '--game', str(task['game']),
               '--capture-dir', task['capture'], '--cache-root', str(cache_root or task['cache_root']),
               '--include-prediction-summaries', '--include-latent-summaries',
               '--native-summary-batch-size', '64']
    if max_files is not None:
        command += ['--max-files', str(max_files)]
    return command


def prepare(out):
    require(not out.exists(), 'Refusing to overwrite existing cache campaign')
    require(shutil.disk_usage(REPO).free >= MIN_FREE_BYTES, 'Less than 120 GiB free')
    final = read(VISUAL / 'RECOVERY_COMPLETED.json')
    require(final['status'] == 'COMPLETED' and final['checkpoint_count'] == 72 and not final['errors'],
            'Visualization campaign is not verified complete')
    tasks, keys, source_identities = [], set(), {}
    for original in read(BASE / 'campaign.json')['tasks']:
        task = {k: original[k] for k in ('index', 'subject', 'game', 'game_name', 'leave_session',
                                          'checkpoint', 'capture', 'bundle_sha256', 'sources')}
        key = task['subject'], task['game'], task['leave_session']
        require(key in expected_keys() and key not in keys, 'Duplicate or unexpected checkpoint')
        keys.add(key)
        m = validate_capture(task)
        for item in task['sources']:
            old = source_identities.setdefault(item['source_id'], item['sha256'])
            require(old == item['sha256'], 'Same trajectory differs between checkpoints')
        task.update(capture_manifest_sha256=m['manifest_sha256'],
                    capture_completion_sha256=sha(Path(task['capture']) / 'COMPLETED.json'),
                    expected_real_steps=sum(e['metadata']['replay_edge_count'] for e in m['episodes']),
                    expected_scored_steps=sum(e['metadata']['scored_edge_count'] for e in m['episodes']),
                    expected_native_events=sum(e['event_count'] for e in m['episodes']),
                    expected_episode_count=len(m['episodes']), num_actions=9 if task['game'] == 0 else 6)
        task['cache_root'] = str(out / ('checkpoint_leave%02d' % task['leave_session']))
        task['cache'] = str(Path(task['cache_root']) / ('sub%03d_game%d' % (task['subject'], task['game'])))
        tasks.append(task)
    require(keys == expected_keys() and len(source_identities) == 198, 'Incomplete 72-checkpoint/198-source coverage')
    # Biggest captures begin first; free workers on any of the eight nodes then
    # claim the next task. No fixed per-node tail of queued checkpoints.
    tasks.sort(key=lambda t: (-t['expected_native_events'], t['index']))
    for name in ('source', 'reports/claims', 'pbs', 'subjects'):
        (out / name).mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, dest)
    write(out / 'source/provenance.json', dict(created_at=time.time(), copied_from=str(REPO),
          input_format='native Zarr only; no behavioral NPZ or legacy NPY inputs',
          source_files={name: sha(REPO / name) for name in SOURCE_FILES}))
    files = sorted(p for p in (out / 'source').rglob('*') if p.is_file())
    (out / 'source/SHA256SUMS').write_text(''.join('%s  %s\n' % (sha(p), p.relative_to(out / 'source')) for p in files))
    for p in files + [out / 'source/SHA256SUMS']:
        p.chmod(0o444)
    campaign = dict(status='PREPARED', created_at=time.time(), target_step=10000000,
                    visual_root=str(VISUAL), tasks=tasks, checkpoint_count=72, unique_source_count=198,
                    checkpoint_source_pairs=792, cache_requested=True, visualization_requested=False,
                    analyses_requested=False, all_sessions=True, source_sha256=sha(out / 'source/SHA256SUMS'),
                    runtime_sums_sha256=sha(RUNTIME_SUMS),
                    placement_authority='User explicitly requested all eight slaves for this cache campaign',
                    nodes=[dict(index=i, host='bml2-slave%d' % i, ncpus=56 if i == 3 else 28,
                                memory='112gb', concurrency=3) for i in range(1, 9)],
                    walltime='168:00:00', queue_policy='atomic longest-native-event-count first',
                    predicted_xs='Not recorded; unavailable summaries remain NaN with validity=0',
                    repeated_trajectory_warning='Four checkpoint views of each trajectory, not four independent behavioral samples')
    write(out / 'campaign.json', campaign)
    print('PREPARED', out, '72 caches; 8 nodes; 24 concurrent capture caches', flush=True)


def claim_next(out, tasks, worker):
    for task in tasks:
        claim = out / 'reports/claims' / ('task-%02d' % task['index'])
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        write(claim / 'owner.json', dict(node=os.getenv('HOSTNAME'), job_id=os.getenv('PBS_JOBID'),
                                       worker=worker, claimed_at=time.time(), index=task['index']))
        return task
    return None


def verify_cache(task, capture):
    import numpy as np
    import pandas as pd
    from shared_analysis_cache import cache_paths
    p = cache_paths(task['cache_root'], task['subject'], task['game'])
    m = read(p.manifest)
    require(m['status'] == 'COMPLETED' and m['adapter_version'] == 3, 'Incomplete cache')
    require((m['subject'], m['game'], m['num_actions']) == (task['subject'], task['game'], task['num_actions']),
            'Cache identity mismatch')
    require(m['source_count'] == m['runs'] == 11 and m['real_steps'] == task['expected_real_steps']
            and m['episode_count'] == task['expected_episode_count'], 'Cache coverage mismatch')
    require(m['captures'][0]['manifest_sha256'] == capture['manifest_sha256'] and len(m['captures']) == 1,
            'Wrong capture binding')
    require(m['native_feature_options'] == dict(images=True, latents=True, batch_size=64), 'Wrong summary options')
    for name, path in vars(p).items():
        if name in ('root', 'root_state_dir', 'root_state_index', 'representation_dir'):
            continue
        require(path.is_file() and path.stat().st_size > 0, 'Missing/empty cache artifact: %s' % path)
    decisions = pd.read_csv(p.native_decisions, usecols=['real_step_id', 'stop', 'forced_stop', 'additional_computations'])
    require(len(decisions) == m['real_steps'] and decisions.real_step_id.is_unique, 'Invalid decision keys')
    require((decisions.stop + decisions.forced_stop).eq(1).all(), 'Invalid STOP/censoring coverage')
    require(decisions.additional_computations.sum() == m['search_events'], 'Lost computation events')
    windows = pd.read_csv(p.persistence_windows, usecols=['real_step_id', 'scored', 'session', 'run_label'])
    require(windows.real_step_id.tolist() == decisions.real_step_id.tolist(), 'Decision table keys/order differ')
    require(int(windows.scored.sum()) == task['expected_scored_steps'] and set(windows.session) == {1, 2, 3, 4}
            and windows.run_label.nunique() == 11,
            'Scored-step or session coverage differs')
    qa = read(p.native_features_qa)
    require(sum(ep['native_events'] for ep in qa['episodes']) == task['expected_native_events'], 'Native event coverage differs')
    require(qa['paths'] == m['search_events'] and qa['decisions'] == m['real_steps'], 'Path/decision QA differs')
    with np.load(p.imagination_paths, allow_pickle=False) as a:
        require(len(a['path_lengths']) == m['search_events'] and len(a['path_offsets']) == m['search_events'] + 1,
                'Ragged path lengths differ')
        require(np.array_equal(np.diff(a['path_offsets']), a['path_lengths'])
                and a['path_offsets'][-1] == len(a['path_actions']), 'Invalid ragged offsets')
        require(((a['path_actions'] >= 0) & (a['path_actions'] < task['num_actions'])).all(), 'Invalid imagined action')
    # Stream the large CSVs; do not count metadata alone as successful output.
    count, search = 0, 0
    for chunk in pd.read_csv(p.native_events, usecols=['search_computation'], chunksize=100000):
        count += len(chunk)
        search += int(chunk.search_computation.sum())
    require(count == task['expected_native_events'] and search == m['search_events'], 'Event CSV count differs')
    count = sum(len(chunk) for chunk in pd.read_csv(p.forecast_outcomes, usecols=['native_event'], chunksize=100000))
    require(count == m['search_events'], 'Forecast CSV count differs')
    return m


def run_task(out, task, worker):
    index = task['index']
    report = out / 'reports' / ('task-%02d' % index)
    status = dict(index=index, node=os.getenv('HOSTNAME'), job_id=os.getenv('PBS_JOBID'),
                  worker=worker, started_at=time.time(), capture=task['capture'], cache=task['cache'])
    def stage(value):
        write(str(report) + '.running.json', dict(status, status='RUNNING', stage=value, updated_at=time.time()))
    try:
        require(not Path(task['cache']).exists(), 'Refusing to overwrite existing cache')
        require(shutil.disk_usage(out).free >= MIN_FREE_BYTES, 'Free space below 120 GiB reserve')
        stage('input_validation')
        capture = validate_capture(task)
        require(sha(Path(task['capture']) / 'COMPLETED.json') == task['capture_completion_sha256'], 'Input completion changed')
        stage('cache_extraction')
        with (out / 'pbs' / ('task-%02d.log' % index)).open('x') as log:
            subprocess.run(make_command(out, task), stdout=log, stderr=subprocess.STDOUT, check=True)
        stage('cache_validation')
        m = verify_cache(task, capture)
        require(validate_capture(task)['manifest_sha256'] == capture['manifest_sha256'], 'Input changed during extraction')
        cache = Path(task['cache'])
        artifacts = {str(p.relative_to(cache)): dict(size=p.stat().st_size, sha256=sha(p))
                     for p in sorted(cache.rglob('*')) if p.is_file()}
        result = dict(status, status='COMPLETED', completed_at=time.time(), subject=task['subject'],
                      game=task['game'], leave_session=task['leave_session'], checkpoint=task['checkpoint'],
                      checkpoint_real_step=10000000, capture_manifest_sha256=capture['manifest_sha256'],
                      cache_manifest_sha256=sha(cache / 'manifest.json'), real_steps=m['real_steps'],
                      scored_steps=task['expected_scored_steps'], search_events=m['search_events'],
                      source_count=11, episode_count=m['episode_count'], artifacts=artifacts,
                      session_roles={str(s): 'held_out' if s == task['leave_session'] else 'training' for s in range(1, 5)})
        write(cache / 'checkpoint_binding.json', result)
        write(str(report) + '.completed.json', result)
        print('CACHE_VERIFIED', index, task['cache'], flush=True)
        return True
    except Exception:
        write(str(report) + '.failed.json', dict(status, status='FAILED', failed_at=time.time(), error=traceback.format_exc()))
        traceback.print_exc()
        return False


def run_node(out, index):
    campaign = read(out / 'campaign.json')
    require(sha(out / 'source/SHA256SUMS') == campaign['source_sha256']
            and sha(RUNTIME_SUMS) == campaign['runtime_sums_sha256'], 'Source/runtime manifest changed')
    node = next(n for n in campaign['nodes'] if n['index'] == index)
    require(os.getenv('HOSTNAME') == node['host'], 'Wrong PBS host')
    def worker(number):
        ok = True
        while True:
            task = claim_next(out, campaign['tasks'], number)
            if task is None:
                return ok
            ok = run_task(out, task, number) and ok
    with ThreadPoolExecutor(max_workers=node['concurrency']) as pool:
        results = list(pool.map(worker, range(node['concurrency'])))
    write(out / 'reports' / ('node-%d.json' % index), dict(status='COMPLETED' if all(results) else 'FAILED',
          node=node['host'], finished_at=time.time(), job_id=os.getenv('PBS_JOBID')))
    if not all(results):
        raise RuntimeError('One or more cache tasks failed; see task reports')


def reconcile(out):
    campaign = read(out / 'campaign.json')
    completed, errors = [], []
    for task in campaign['tasks']:
        try:
            p = out / 'reports' / ('task-%02d.completed.json' % task['index'])
            result = read(p)
            require(result['status'] == 'COMPLETED', 'Task not complete')
            require(result == read(Path(task['cache']) / 'checkpoint_binding.json'), 'Binding/report mismatch')
            validate_capture(task)
            for name, artifact in result['artifacts'].items():
                path = Path(task['cache']) / name
                require(path.is_file() and path.stat().st_size == artifact['size'] and sha(path) == artifact['sha256'],
                        'Cache artifact hash mismatch: %s' % path)
            completed.append(result)
        except Exception as exc:
            errors.append(dict(index=task['index'], reason=str(exc)))
    report = dict(status='FAILED' if errors else 'COMPLETED', completed_at=time.time(), errors=errors,
                  cache_count=len(completed), expected_caches=72, cache_generated=not errors, visual_root=str(VISUAL),
                  real_steps=sum(r['real_steps'] for r in completed), unique_source_count=198,
                  checkpoint_source_pairs=792, duplicate_trajectory_warning=campaign['repeated_trajectory_warning'])
    write(out / 'reports/reconciliation.json', report)
    if errors:
        raise RuntimeError('Incomplete cache campaign: %d failed/missing tasks' % len(errors))
    columns = ['subject', 'game', 'leave_session', 'checkpoint_real_step', 'source_count', 'episode_count',
               'real_steps', 'scored_steps', 'search_events', 'capture', 'cache', 'cache_manifest_sha256']
    with (out / 'cache_index.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(completed, key=lambda r: (r['game'], r['subject'], r['leave_session'])))
    for subject in range(1, 7):
        for game in range(3):
            group = [r for r in completed if (r['subject'], r['game']) == (subject, game)]
            write(out / 'subjects' / ('sub%03d_game%d.json' % (subject, game)),
                  dict(status='COMPLETED', subject=subject, game=game,
                       checkpoints=[{k: r[k] for k in columns} for r in group]))
    write(out / 'manifest.json', dict(report, caches=[{k: r[k] for k in columns} for r in completed],
                                    source_sha256=campaign['source_sha256']))
    write(out / 'COMPLETED.json', report)
    print('ALL_72_CACHES_COMPLETE', flush=True)


def submit(out):
    campaign = read(out / 'campaign.json')
    require(not (out / 'submission.json').exists(), 'Already submitted; inspect before retry')
    jobs = []
    script = str(out / 'source/research_script/dynamic_cache_10m.pbs')
    for node in campaign['nodes']:
        command = ['qsub', '-N', 'cache10-n%d' % node['index'], '-l',
                   'select=1:ncpus=%d:mem=%s:host=%s' % (node['ncpus'], node['memory'], node['host']),
                   '-l', 'place=excl', '-l', 'walltime=168:00:00', '-o',
                   str(out / 'pbs' / ('node-%d.pbs.log' % node['index'])), '-v',
                   'CAMPAIGN_OUTPUT=%s,CAMPAIGN_NODE=%d,CAMPAIGN_MODE=run-node' % (out, node['index']), script]
        job = subprocess.check_output(command, text=True).strip()
        jobs.append(dict(node=node['host'], job_id=job))
        write(out / 'submission.json', dict(status='SUBMITTING', jobs=jobs, submitted_at=time.time()))
        print('SUBMITTED', node['host'], job, flush=True)
    job = subprocess.check_output(['qsub', '-N', 'cache10-check', '-l',
        'select=1:ncpus=1:mem=4gb:host=bml2-slave1', '-l', 'walltime=168:00:00', '-W',
        'depend=afterany:' + ':'.join(j['job_id'] for j in jobs), '-o', str(out / 'pbs/reconcile.pbs.log'),
        '-v', 'CAMPAIGN_OUTPUT=%s,CAMPAIGN_NODE=1,CAMPAIGN_MODE=reconcile' % out, script], text=True).strip()
    write(out / 'submission.json', dict(status='SUBMITTED', jobs=jobs, verification_job=job, submitted_at=time.time()))
    print('VERIFICATION_JOB', job, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'submit', 'run-node', 'reconcile'))
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--index', type=int, choices=range(1, 9))
    args = p.parse_args()
    out = args.output.resolve()
    if args.command == 'run-node':
        if args.index is None:
            p.error('--index required')
        run_node(out, args.index)
    else:
        {'prepare': prepare, 'submit': submit, 'reconcile': reconcile}[args.command](out)


if __name__ == '__main__':
    main()
