#!/usr/bin/env python3
"""Submit the currently published Space Invaders checkpoints, visualization only.

Does not delete captures, regenerate Enduro/Pong, or launch numeric cache jobs.
Host-side preparation uses the standard library; inference uses the sealed
Python 3.10 runtime and previously validated native replay implementation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

REPO = Path('/home/jeongmin/thinker-dynamic-imitation')
REFERENCE = REPO/'research_script/outputs/allcheckpoints_allsessions_5m_20260923/source'
DEFAULT_OUTPUT = REPO/'logs/visual_behav/spaceinvaders_allcheckpoints_allsessions_5m_20260924'
RUNTIME_SUMS = Path('/home/jeongmin/visual-behav-assets/shared-runtime-25d766c04fe7-core.sha256')
DEPTH_ACTOR_SHA = 'd26fdca8d2fb31622adb70cbe59d70b04c66b5ce22770a063f123961092cded2'
MIN_FREE = 80 * 1024**3


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-%d' % os.getpid())
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    temp.replace(path)


def published_hashes(checkpoint):
    return {line.split(maxsplit=1)[1].lstrip('./'): line.split()[0]
            for line in (checkpoint/'source.sha256').read_text(encoding='utf8').splitlines()
            if line.strip()}


def depth_actor():
    """Recover the published intermediate actor byte-for-byte, hash or fail.

    Its inference-side singleton legal-control stabilization already exists in
    the GroupNorm runtime; its training loss is the legacy implementation.
    No guessed implementation is accepted: the result must match the training
    source SHA256 published with the checkpoint.
    """
    legacy = (REFERENCE/'runtimes/legacy/thinker/thinker/actor_net.py').read_text(encoding='utf8')
    group = (REFERENCE/'runtimes/group/thinker/thinker/actor_net.py').read_text(encoding='utf8')
    a = legacy.index('        if compute_loss:\n            reg_loss = 1e-6')
    b = group.index('        if compute_loss:\n            from thinker.numeric_stability import masked_squared_sum')
    result = group[:b] + legacy[a:]
    if hashlib.sha256(result.encode('utf8')).hexdigest() != DEPTH_ACTOR_SHA:
        raise RuntimeError('Cannot recover exact published depth-support actor')
    return result


def select_runtime(checkpoint):
    hashes = published_hashes(checkpoint)
    for name in ('legacy', 'group', 'depth'):
        base = REFERENCE/'runtimes'/('legacy' if name == 'depth' else name)/'thinker/thinker'
        actor = DEPTH_ACTOR_SHA if name == 'depth' else digest(base/'actor_net.py')
        if (hashes.get('thinker/thinker/actor_net.py') == actor and
                hashes.get('thinker/thinker/model_net.py') == digest(base/'model_net.py')):
            return name
    raise RuntimeError('Unrecognized actor/model training source: %s' % checkpoint)


def prepare(out):
    if out.exists():
        raise FileExistsError('Refusing existing campaign: %s' % out)
    if shutil.disk_usage(REPO).free < 200*1024**3:
        raise RuntimeError('Insufficient free storage for new visualization campaign')
    subprocess.run(['sha256sum', '-c', 'SHA256SUMS'], cwd=str(REFERENCE),
                   stdout=subprocess.DEVNULL, check=True)
    candidates = sorted((REPO/'logs/thinker').glob('dynamic-imitation-spaceinvaders-voc-sub*-leavesession*'))
    if not candidates:
        raise RuntimeError('No Space Invaders checkpoints')
    tasks = []
    source_sets = {}
    for cp in candidates:
        match = re.fullmatch(r'dynamic-imitation-spaceinvaders-voc-sub(\d+)-leavesession([1-4])', cp.name)
        if not match:
            raise ValueError(cp)
        subject, leave = map(int, match.groups())
        m = read(cp/'milestone_5000000.json')
        if m['status'] != 'VERIFIED_5M_STOPPED' or (
                m['logical']['subject'], m['logical']['game_id'], m['logical']['leave_session']) != (subject, 2, leave):
            raise ValueError('Checkpoint identity/status mismatch: %s' % cp)
        for artifact in list(m['artifacts'].values()) + [m['config']]:
            if digest(cp/artifact['filename']) != artifact['sha256']:
                raise ValueError('Checkpoint artifact hash mismatch: %s' % cp)
        if subject not in source_sets:
            sources = sorted((REPO/'behavioral_data_block'/('sub-%03d' % subject)).glob('ses-*/*game2.npz'))
            if len(sources) != 11 or {p.parent.name for p in sources} != {'ses-%02d' % i for i in range(1, 5)}:
                raise ValueError('Incomplete four-session coverage: subject %d' % subject)
            source_sets[subject] = [dict(source_id=str(p.relative_to(REPO/'behavioral_data_block')),
                                         sha256=digest(p), size=p.stat().st_size) for p in sources]
        index = len(tasks)
        tasks.append(dict(index=index, subject=subject, game=2, leave_session=leave,
                          checkpoint=str(cp), milestone_sha256=digest(cp/'milestone_5000000.json'),
                          runtime=select_runtime(cp), node='bml2-slave%d' % (5+index % 4),
                          capture=str(out/'captures'/('spaceinvaders-sub%03d-leave%02d-all' % (subject, leave))),
                          sources=source_sets[subject]))
    out.mkdir(parents=True)
    for subdir in ('captures', 'reports', 'pbs', 'source/research_script', 'source/runtimes'):
        (out/subdir).mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name in sorted({t['runtime'] for t in tasks}):
        base_name = 'legacy' if name == 'depth' else name
        src = REFERENCE/'runtimes'/base_name
        dst = out/'source/runtimes'/name
        shutil.copytree(str(src), str(dst), ignore=shutil.ignore_patterns('__pycache__'))
        provenance[name] = dict(base_runtime=str(src),
            note='Existing instrumented native replay runtime; not an unmodified training source tree')
        if name == 'depth':
            actor = dst/'thinker/thinker/actor_net.py'
            actor.chmod(0o644)
            actor.write_text(depth_actor(), encoding='utf8')
            provenance[name].update(actor_sha256=digest(actor),
                util_compatibility='Legacy evaluation utility; not byte-identical to depth-support training util. '
                                   'No changes to inference flags or budget-independent tree schema.')
    for name in ('spaceinvaders_visualization.py', 'spaceinvaders_visualization.pbs'):
        shutil.copy2(str(REPO/'research_script'/name), str(out/'source/research_script'/name))
    write(out/'source/provenance.json', provenance)
    checksums = []
    for path in sorted((out/'source').rglob('*')):
        if path.is_file():
            checksums.append('%s  %s\n' % (digest(path), path.relative_to(out/'source')))
            path.chmod(0o444)
    (out/'source/SHA256SUMS').write_text(''.join(checksums), encoding='utf8')
    (out/'source/SHA256SUMS').chmod(0o444)
    write(out/'campaign.json', dict(status='PREPARED', created_at=time.time(),
        kind='visualization-only', data_root=str(REPO/'behavioral_data_block'), tasks=tasks,
        checkpoint_count=len(tasks), source_count=sum(map(len, source_sets.values())),
        checkpoint_source_pairs=sum(len(t['sources']) for t in tasks),
        subjects=sorted(source_sets), source=str(out/'source'),
        omitted_subjects=sorted(set(range(1, 7))-set(source_sets)),
        storage=dict(compression='blosc-zstd', level=9, shuffle='bitshuffle',
                     images='all', store_predicted_xs=False, min_free_bytes=MIN_FREE),
        resources=dict(ncpus=8, mem='24gb', walltime='168:00:00'),
        cache_requested=False, all_sessions=True))
    print('PREPARED', len(tasks), 'visualization-only tasks', flush=True)


def command(out, task):
    runtime = out/'source/runtimes'/task['runtime']/'thinker'
    args = [sys.executable, '-B', '-u', str(runtime/'visual_behav.py'), 'capture',
            '--checkpoint-dir', task['checkpoint'], '--checkpoint-step', '5000000',
            '--data-root', str(REPO/'behavioral_data_block'), '--output', task['capture'],
            '--devices', 'cpu', '--cpu-workers', '3', '--torch-threads-per-worker', '2',
            '--images', 'all', '--compression-level', '9', '--omit-predicted-xs',
            '--policy-mode', 'checkpoint', '--base-seed', '0', '--resume']
    for source in task['sources']:
        args += ['--source', source['source_id']]
    return args


def preflight(out, task):
    if digest(Path(task['checkpoint'])/'milestone_5000000.json') != task['milestone_sha256']:
        raise RuntimeError('Checkpoint changed since preparation')
    report = out/'reports'/('task-%02d.dry-run.json' % task['index'])
    with report.open('w') as stream:
        subprocess.run(command(out, task)+['--dry-run'], stdout=stream, check=True)
    dry = read(report)
    semantic = dry['plan']['semantic']
    assert dry['checkpoint_payload_validation']['game_id'] == 2
    assert semantic['checkpoint']['logical_xpid'] == Path(task['checkpoint']).name
    assert semantic['compression_level'] == 9 and semantic['store_predicted_xs'] is False
    assert {s['source_id']: s['sha256'] for s in semantic['sources']} == {
        s['source_id']: s['sha256'] for s in task['sources']}
    print('PREFLIGHT_OK', task['index'], task['runtime'], flush=True)


def monitor(args, out):
    with subprocess.Popen(args, start_new_session=True) as process:
        while True:
            try:
                code = process.wait(timeout=30)
                if code:
                    raise subprocess.CalledProcessError(code, args)
                return
            except subprocess.TimeoutExpired:
                if shutil.disk_usage(out).free < MIN_FREE:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError('Stopped own subprocess below 80 GiB free; partial outputs retained')


def run(out, task):
    preflight(out, task)
    write(out/'reports'/('task-%02d.running.json' % task['index']), dict(
        index=task['index'], stage='visualization', started_at=time.time(),
        job_id=os.getenv('PBS_JOBID'), node=os.getenv('HOSTNAME'), task=task))
    monitor(command(out, task), out)
    sys.path.insert(0, str(out/'source/runtimes'/task['runtime']/'thinker'))
    import visual_behav
    capture = visual_behav.validate_capture_output(task['capture'])
    assert capture['semantic']['checkpoint']['logical_xpid'] == Path(task['checkpoint']).name
    assert {s['source_id']: s['sha256'] for s in capture['semantic']['sources']} == {
        s['source_id']: s['sha256'] for s in task['sources']}
    for episode in capture['episodes']:
        metadata = episode['metadata']
        assert (metadata['subject'], metadata['game']) == (task['subject'], 2)
        assert metadata['store_predicted_xs'] is False and metadata['compression_level'] == 9
    result = dict(status='COMPLETED', index=task['index'], subject=task['subject'], game=2,
        leave_session=task['leave_session'], capture=task['capture'], completed_at=time.time(),
        manifest_sha256=capture['manifest_sha256'], episode_count=len(capture['episodes']),
        real_steps=sum(e['metadata']['replay_edge_count'] for e in capture['episodes']),
        source_count=len(task['sources']), cache_generated=False,
        session_roles={str(s): 'held_out' if s == task['leave_session'] else 'training' for s in range(1, 5)})
    write(out/'reports'/('task-%02d.completed.json' % task['index']), result)
    tasks = read(out/'campaign.json')['tasks']
    completed = [out/'reports'/('task-%02d.completed.json' % t['index']) for t in tasks]
    if all(p.is_file() for p in completed):
        rows = list(map(read, completed))
        write(out/'COMPLETED.json', dict(status='COMPLETED', kind='visualization-only',
            checkpoint_count=len(rows), completed_at=time.time(),
            real_steps=sum(r['real_steps'] for r in rows),
            episode_count=sum(r['episode_count'] for r in rows), cache_generated=False))
    print('VISUALIZATION_VERIFIED', task['index'], task['capture'], flush=True)


def submit(out):
    receipt = out/'submission.json'
    if receipt.exists():
        raise FileExistsError('Refusing duplicate submission')
    campaign = read(out/'campaign.json')
    if shutil.disk_usage(out).free < 200*1024**3:
        raise RuntimeError('Insufficient free space')
    ledger = dict(status='SUBMITTING', created_at=time.time(), jobs=[])
    write(receipt, ledger)
    for task in campaign['tasks']:
        job = subprocess.check_output(['qsub', '-N', 'visual-spa-s%d-l%d' % (task['subject'], task['leave_session']),
            '-l', 'select=1:ncpus=8:mem=24gb:host='+task['node'],
            '-o', str(out/'pbs'/('task-%02d.pbs.log' % task['index'])),
            '-v', 'CAMPAIGN_OUTPUT=%s,CAMPAIGN_INDEX=%d,RUNTIME_SUMS_SHA=%s' % (out, task['index'], digest(RUNTIME_SUMS)),
            str(out/'source/research_script/spaceinvaders_visualization.pbs')], universal_newlines=True).strip()
        ledger['jobs'].append(dict(index=task['index'], job_id=job, node=task['node'],
                                  subject=task['subject'], leave_session=task['leave_session']))
        write(receipt, ledger)
        print('SUBMITTED', task['index'], job, task['node'], flush=True)
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
        selected = tasks if args.index is None and args.action == 'preflight' else [tasks[args.index]]
        for task in selected:
            (run if args.action == 'run' else preflight)(out, task)


if __name__ == '__main__':
    main()
