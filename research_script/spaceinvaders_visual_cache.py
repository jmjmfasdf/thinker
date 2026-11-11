#!/usr/bin/env python3
"""Reuse 20 completed captures and chain four new captures into 24 caches.

No existing result or executed source snapshot is modified. The cache code is
frozen from the completed Enduro/Pong campaign. Tasks 0-3 run the new subject
first, overlapping its visualization with the cache-only tasks on slave5-8.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from spaceinvaders_visualization import digest, read, write, monitor, command, preflight
from all_session_visual_cache import verify_cache

REPO = Path('/home/jeongmin/thinker-dynamic-imitation')
REFERENCE = REPO/'research_script/outputs/allcheckpoints_allsessions_5m_20260923/source'
OLD_VISUAL = REPO/'logs/visual_behav/spaceinvaders_allcheckpoints_allsessions_5m_20260924'
NEW_VISUAL = REPO/'logs/visual_behav/spaceinvaders_allcheckpoints_allsessions_5m_20260928'
DEFAULT_OUTPUT = REPO/'research_script/outputs/spaceinvaders_allcheckpoints_allsessions_5m_20260928'
RUNTIME_SUMS = Path('/home/jeongmin/visual-behav-assets/shared-runtime-25d766c04fe7-core.sha256')
CACHE_FILES = (
    'research_script/00_cache.py', 'research_script/dynamic_analysis_cache.py',
    'research_script/dynamic_native_features.py', 'research_script/shared_analysis_cache.py',
    'research_script/shared_actor_policy_cache.py', 'research_script/legacy/02_persistence_action_geometry.py',
    'research_script/legacy/02_actor_policy.py', 'thinker/thinker/behavior_trace.py',
    'research_script/all_session_visual_cache.py',
)


def completed_capture(path, task):
    """Lightweight preparation check; run() also verifies native data hashes."""
    import hashlib
    capture, proof = read(path/'manifest.json'), read(path/'COMPLETED.json')
    unsigned = {k: v for k, v in capture.items() if k != 'manifest_sha256'}
    sha = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    if not (proof['status'] == 'COMPLETED' and sha == proof['manifest_sha256'] == capture['manifest_sha256']):
        raise ValueError('Capture completion/hash mismatch: %s' % path)
    cp = capture['semantic']['checkpoint']
    if cp['logical_xpid'] != Path(task['checkpoint']).name or cp['milestone_sha256'] != task['milestone_sha256']:
        raise ValueError('Capture belongs to another checkpoint: %s' % path)
    if {s['source_id']: s['sha256'] for s in capture['semantic']['sources']} != {
            s['source_id']: s['sha256'] for s in task['sources']}:
        raise ValueError('Source coverage/hash mismatch: %s' % path)
    if len(capture['episodes']) != proof['episode_count']:
        raise ValueError('Episode count mismatch')
    for ep in capture['episodes']:
        meta = ep['metadata']
        if (meta['subject'], meta['game']) != (task['subject'], 2):
            raise ValueError('Episode identity mismatch')
        if not (path/ep['path']/'manifest.json').is_file():
            raise ValueError('Missing native episode')
    if {e['metadata']['session'] for e in capture['episodes']} != {1, 2, 3, 4}:
        raise ValueError('Capture does not cover all sessions')
    return capture


def make_cache_command(out, task):
    return [sys.executable, '-B', '-u', str(out/'source/research_script/00_cache.py'),
            '--subject', str(task['subject']), '--game', '2', '--capture-dir', task['capture'],
            '--cache-root', task['cache_root'], '--include-prediction-summaries',
            '--include-latent-summaries', '--native-summary-batch-size', '64']


def prepare(out):
    if out.exists() or NEW_VISUAL.exists():
        raise FileExistsError('Refusing to overwrite output or visualization campaign')
    if shutil.disk_usage(REPO).free < 160*1024**3:
        raise RuntimeError('Insufficient storage including 80 GiB reserve')
    for root in (REFERENCE, OLD_VISUAL/'source'):
        subprocess.run(['sha256sum', '-c', 'SHA256SUMS'], cwd=str(root),
                       stdout=subprocess.DEVNULL, check=True)
    old_tasks = {(t['subject'], t['leave_session']): t for t in read(OLD_VISUAL/'campaign.json')['tasks']}
    assert len(old_tasks) == 20 and read(OLD_VISUAL/'COMPLETED.json')['checkpoint_count'] == 20
    tasks = []
    for subject in (6, 1, 2, 3, 4, 5):
        sources = []
        files = sorted((REPO/'behavioral_data_block'/('sub-%03d' % subject)).glob('ses-*/*game2.npz'))
        if len(files) != 11 or {p.parent.name for p in files} != {'ses-%02d' % s for s in range(1, 5)}:
            raise ValueError('Incomplete subject data: %d' % subject)
        for path in files:
            sources.append(dict(source_id=str(path.relative_to(REPO/'behavioral_data_block')),
                                sha256=digest(path), size=path.stat().st_size))
        for leave in range(1, 5):
            cp = REPO/'logs/thinker'/('dynamic-imitation-spaceinvaders-voc-sub%d-leavesession%d' % (subject, leave))
            m = read(cp/'milestone_5000000.json')
            if m['status'] != 'VERIFIED_5M_STOPPED' or (
                    m['logical']['subject'], m['logical']['game_id'], m['logical']['leave_session']) != (subject, 2, leave):
                raise ValueError('Checkpoint identity mismatch')
            for item in list(m['artifacts'].values()) + [m['config']]:
                if digest(cp/item['filename']) != item['sha256']:
                    raise ValueError('Checkpoint hash mismatch: %s' % cp)
            source_manifest = {l.split(maxsplit=1)[1].lstrip('./'): l.split()[0]
                for l in (cp/'behavior.sha256').read_text(encoding='utf8').splitlines() if l.strip()}
            if any(source_manifest.get(s['source_id']) != s['sha256'] for s in sources):
                raise ValueError('Behavior bytes differ from checkpoint data manifest')
            if subject == 6:
                code_hashes = {l.split(maxsplit=1)[1].lstrip('./'): l.split()[0]
                    for l in (cp/'source.sha256').read_text(encoding='utf8').splitlines() if l.strip()}
                for name in ('actor_net.py', 'model_net.py', 'util.py'):
                    if code_hashes.get('thinker/thinker/'+name) != digest(REFERENCE/'runtimes/group/thinker/thinker'/name):
                        raise ValueError('Unrecognized new checkpoint runtime: %s' % name)
            reused = (subject, leave) in old_tasks
            cap = Path(old_tasks[(subject, leave)]['capture']) if reused else NEW_VISUAL/'captures'/(
                'spaceinvaders-sub%03d-leave%02d-all' % (subject, leave))
            index = len(tasks)
            task = dict(index=index, subject=subject, game=2, game_name='spaceinvaders', leave_session=leave,
                checkpoint=str(cp), milestone_sha256=digest(cp/'milestone_5000000.json'), sources=sources,
                runtime='group', node='bml2-slave%d' % (5+index % 4), capture=str(cap), reuse_visualization=reused,
                num_actions=6, cache_root=str(out/'00_dynamic_cache'/('checkpoint_leave%02d' % leave)))
            task['cache'] = str(Path(task['cache_root'])/('sub%03d_game2' % subject))
            if reused:
                capture = completed_capture(cap, task)
                task['capture_manifest_sha256'] = capture['manifest_sha256']
                task['expected_real_steps'] = sum(e['metadata']['replay_edge_count'] for e in capture['episodes'])
                task['expected_episode_count'] = len(capture['episodes'])
            tasks.append(task)
    assert len(tasks) == 24 and sum(not t['reuse_visualization'] for t in tasks) == 4
    out.mkdir(parents=True)
    for directory in ('reports', 'pbs', 'subjects', 'source/research_script'):
        (out/directory).mkdir(parents=True, exist_ok=True)
    (NEW_VISUAL/'captures').mkdir(parents=True)
    for relative in CACHE_FILES:
        dest = out/'source'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(REFERENCE/relative), str(dest))
    shutil.copytree(str(REFERENCE/'runtimes/group'), str(out/'source/runtimes/group'),
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(str(OLD_VISUAL/'source/research_script/spaceinvaders_visualization.py'),
                 str(out/'source/research_script/spaceinvaders_visualization.py'))
    for name in ('spaceinvaders_visual_cache.py', 'spaceinvaders_visual_cache.pbs'):
        shutil.copy2(str(REPO/'research_script'/name), str(out/'source/research_script'/name))
    write(out/'source/provenance.json', dict(cache_source=str(REFERENCE),
        visualization_helper_source=str(OLD_VISUAL/'source'),
        new_capture_runtime='Exact published GroupNorm actor/model/util, existing native replay instrumentation',
        reuse_existing_captures_read_only=True))
    checksums = []
    for p in sorted((out/'source').rglob('*')):
        if p.is_file():
            checksums.append('%s  %s\n' % (digest(p), p.relative_to(out/'source')))
            p.chmod(0o444)
    (out/'source/SHA256SUMS').write_text(''.join(checksums), encoding='utf8')
    (out/'source/SHA256SUMS').chmod(0o444)
    write(out/'campaign.json', dict(status='PREPARED', tasks=tasks, created_at=time.time(),
        source=str(out/'source'), new_capture_root=str(NEW_VISUAL), reused_capture_root=str(OLD_VISUAL),
        checkpoint_count=24, source_count=66, checkpoint_source_pairs=264, new_visualizations=4,
        reused_visualizations=20, cache_count=24, all_sessions=True, analyses_requested=False,
        resources=dict(ncpus=8, memory='24gb', walltime='168:00:00'),
        storage=dict(compression='blosc-zstd9-bitshuffle', predicted_xs=False, min_free_gib=80)))
    print('PREPARED: 4 visualization-to-cache chains + 20 cache-only tasks', flush=True)


def aggregate(out):
    campaign = read(out/'campaign.json')
    done = []
    for t in campaign['tasks']:
        p = out/'reports'/('task-%02d.completed.json' % t['index'])
        if p.is_file():
            done.append(read(p))
    for subject in range(1, 7):
        group = [r for r in done if r['subject'] == subject]
        if len(group) == 4 and {r['leave_session'] for r in group} == {1, 2, 3, 4}:
            write(out/'subjects'/('sub%03d_game2.json' % subject), dict(
                status='COMPLETED', subject=subject, game=2, checkpoints=group))
    if len(done) == 24:
        write(out/'COMPLETED.json', dict(status='COMPLETED', checkpoint_count=24, source_count=66,
            checkpoint_source_pairs=264, completed_at=time.time(), new_visualizations=4,
            reused_visualizations=20, caches=24, real_steps=sum(r['real_steps'] for r in done)))


def run(out, task):
    index = task['index']
    if (out/'reports'/('task-%02d.completed.json' % index)).exists() or Path(task['cache']).exists():
        raise FileExistsError('Refusing to overwrite task cache; inspect before any retry')
    if digest(Path(task['checkpoint'])/'milestone_5000000.json') != task['milestone_sha256']:
        raise ValueError('Checkpoint changed since preparation')
    start = time.time()
    def stage(value):
        write(out/'reports'/('task-%02d.running.json' % index), dict(index=index,
            stage=value, started_at=start, updated_at=time.time(), job_id=os.getenv('PBS_JOBID'), task=task))
    if not task['reuse_visualization']:
        preflight(out, task)
        stage('visualization')
        monitor(command(out, task), out)
    stage('capture_validation')
    sys.path.insert(0, str(out/'source/runtimes/group/thinker'))
    import visual_behav
    capture = visual_behav.validate_capture_output(task['capture'])
    completed_capture(Path(task['capture']), task)
    if task['reuse_visualization'] and capture['manifest_sha256'] != task['capture_manifest_sha256']:
        raise ValueError('Reused capture changed since preparation')
    task['expected_real_steps'] = sum(e['metadata']['replay_edge_count'] for e in capture['episodes'])
    assert len({e['metadata']['source_id'] for e in capture['episodes']}) == 11
    assert all(e['metadata']['store_predicted_xs'] is False and e['metadata']['compression_level'] == 9
               for e in capture['episodes'])
    write(out/'reports'/('task-%02d.visual.completed.json' % index), dict(status='COMPLETED', index=index,
        reused=task['reuse_visualization'], capture=task['capture'], manifest_sha256=capture['manifest_sha256'],
        real_steps=task['expected_real_steps'], episode_count=len(capture['episodes']), completed_at=time.time()))
    print('CAPTURE_VERIFIED', index, 'reused='+str(task['reuse_visualization']), flush=True)
    stage('cache')
    monitor(make_cache_command(out, task), out)
    stage('cache_validation')
    verify_cache(out, task, capture)
    aggregate(out)


def submit(out):
    receipt = out/'submission.json'
    if receipt.exists():
        raise FileExistsError('Refusing duplicate submission')
    if shutil.disk_usage(out).free < 160*1024**3:
        raise RuntimeError('Insufficient storage reserve')
    ledger = dict(status='SUBMITTING', created_at=time.time(), jobs=[])
    write(receipt, ledger)
    for t in read(out/'campaign.json')['tasks']:
        job = subprocess.check_output(['qsub', '-N', 'sc-spa-s%d-l%d' % (t['subject'], t['leave_session']),
            '-l', 'select=1:ncpus=8:mem=24gb:host='+t['node'],
            '-o', str(out/'pbs'/('task-%02d.pbs.log' % t['index'])),
            '-v', 'CAMPAIGN_OUTPUT=%s,CAMPAIGN_INDEX=%d,NEW_CAPTURE_ROOT=%s,RUNTIME_SUMS_SHA=%s' % (
                out, t['index'], NEW_VISUAL, digest(RUNTIME_SUMS)),
            str(out/'source/research_script/spaceinvaders_visual_cache.pbs')], universal_newlines=True).strip()
        ledger['jobs'].append(dict(index=t['index'], job_id=job, node=t['node'], subject=t['subject'],
            game=2, leave_session=t['leave_session'], reuse_visualization=t['reuse_visualization']))
        write(receipt, ledger)
        print('SUBMITTED', t['index'], job, t['node'], flush=True)
    ledger['status'] = 'SUBMITTED'
    write(receipt, ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'submit', 'run'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.action == 'prepare':
        prepare(out)
    elif args.action == 'submit':
        submit(out)
    else:
        tasks = read(out/'campaign.json')['tasks']
        if args.action == 'preflight':
            for t in tasks:
                if not t['reuse_visualization']:
                    preflight(out, t)
        else:
            if args.index is None:
                parser.error('run requires --index')
            try:
                run(out, tasks[args.index])
            except Exception as error:
                write(out/'reports'/('task-%02d.failed.json' % args.index), dict(
                    status='FAILED', index=args.index, at=time.time(), error=repr(error)))
                raise


if __name__ == '__main__':
    main()
