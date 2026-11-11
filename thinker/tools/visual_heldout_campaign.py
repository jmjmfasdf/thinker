#!/usr/bin/env python3
"""Session-matched, source-exact visualization campaign (no training writes)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import yaml

REPO = Path('/home/jeongmin/thinker-dynamic-imitation')
DEFAULT_ROOT = REPO / 'logs/visual_behav/heldout_5m_20260921'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def integer_ids(value):
    """Normalize the CLI's comma-separated IDs and YAML list equivalents."""
    if isinstance(value, str):
        value = value.split(',')
    elif isinstance(value, int):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f'Invalid ID list: {value!r}')
    return [int(item) for item in value]


def prepare(root):
    data = REPO / 'behavioral_data_block'
    tasks, seen = [], set()
    for game, name, actions in ((0, 'enduro', 9), (1, 'pong', 6)):
        for subject in range(1, 7):
            for session in range(1, 5):
                checkpoint = REPO / 'logs/thinker' / f'dynamic-imitation-{name}-voc-sub{subject}-leavesession{session}'
                config = yaml.safe_load((checkpoint / 'config_c.yaml').read_text())
                marker = json.loads((checkpoint / 'milestone_5000000.json').read_text())
                logical = marker['logical']
                assert (logical['subject'], logical['game_id'], logical['leave_session']) == (subject, game, session)
                assert integer_ids(config['icopro_subjects']) == [subject]
                assert integer_ids(config['icopro_holdout_sessions']) == [session]
                assert set(integer_ids(config['icopro_train_sessions'])) == set(range(1, 5)) - {session}
                assert config['icopro_game_id'] == game
                files = sorted(data.glob(f'sub-{subject:03d}/ses-{session:02d}/*-game{game}.npz'))
                assert len(files) == (2 if session == 1 else 3), (checkpoint, files)
                sources = []
                for path in files:
                    source_id = str(path.relative_to(data))
                    assert source_id not in seen
                    seen.add(source_id)
                    sources.append(dict(source_id=source_id, size=path.stat().st_size, sha256=digest(path)))
                tasks.append(dict(index=len(tasks), game=game, game_name=name, subject=subject,
                    session=session, num_actions=actions, checkpoint=str(checkpoint),
                    milestone_sha256=digest(checkpoint / 'milestone_5000000.json'),
                    dynamics_norm=config.get('model_sr_dynamics_norm', 'legacy-default'),
                    capture=str(root / 'captures' / f'{name}-sub{subject:03d}-ses{session:02d}'), sources=sources))
    assert len(tasks) == 48 and len(seen) == 132
    manifest = dict(schema_version=1, data_root=str(data), target_step=5_000_000,
                    images='all', policy_mode='checkpoint', base_seed=0,
                    assignment='same subject/game; leavesession N -> session N only', tasks=tasks)
    path = root / 'campaign.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'Existing campaign has different inputs'
    else:
        write_json(path, manifest)
    for name in ('reports', 'pbs', 'captures', 'subjects'):
        (root / name).mkdir(exist_ok=True)
    print(f'Prepared {len(tasks)} held-out folds, {len(seen)} unique sources', flush=True)


def verify_task(root, manifest, task):
    import visual_behav as visual
    capture = visual.validate_capture_output(task['capture'])
    assert capture['semantic']['checkpoint']['logical_xpid'] == Path(task['checkpoint']).name
    expected = {s['source_id']: s['sha256'] for s in task['sources']}
    actual = {s['source_id']: s['sha256'] for s in capture['semantic']['sources']}
    assert actual == expected
    observed = set()
    for episode in capture['episodes']:
        meta = episode['metadata']
        assert (meta['subject'], meta['game'], meta['session']) == (task['subject'], task['game'], task['session'])
        assert meta['source_sha256'] == expected[meta['source_id']]
        assert episode['images'] == 'all'
        observed.add(meta['source_id'])
    assert observed == set(expected)
    result = dict(status='COMPLETED', index=task['index'], subject=task['subject'],
                  game=task['game'], session=task['session'], source_count=len(expected),
                  episode_count=len(capture['episodes']), capture=task['capture'],
                  manifest_sha256=capture['manifest_sha256'], completed_at=time.time())
    write_json(root / 'reports' / f"fold-{task['index']:02d}.completed.json", result)
    print(json.dumps(result), flush=True)


def run(root, index):
    manifest = json.loads((root / 'campaign.json').read_text())
    task = manifest['tasks'][index]
    checkpoint = Path(task['checkpoint'])
    assert digest(checkpoint / 'milestone_5000000.json') == task['milestone_sha256']
    import visual_behav as visual
    # The CPU job binds a frozen source snapshot, not a mutable worktree.
    source = Path(visual.__file__)
    args = [sys.executable, '-u', str(source), 'capture', '--checkpoint-dir', str(checkpoint),
        '--checkpoint-step', '5000000', '--data-root', manifest['data_root'],
        '--output', task['capture'], '--devices', 'cpu', '--cpu-workers', str(len(task['sources'])),
        '--torch-threads-per-worker', '2', '--images', 'all', '--policy-mode', 'checkpoint',
        '--base-seed', '0', '--resume']
    for item in task['sources']:
        args += ['--source', item['source_id']]
    if task['dynamics_norm'] == 'group':
        from thinker import model_net
        if 'model_sr_dynamics_norm' not in Path(model_net.__file__).read_text():
            raise RuntimeError('GroupNorm training source is required; refusing legacy model fallback')
    dry_path = root / 'reports' / f'fold-{index:02d}.dry-run.json'
    with dry_path.open('w') as output:
        subprocess.run(args + ['--dry-run'], stdout=output, check=True)
    dry = json.loads(dry_path.read_text())
    assert dry['checkpoint_payload_validation']['game_id'] == task['game']
    assert {s['source_id']:s['sha256'] for s in dry['plan']['semantic']['sources']} == {
        s['source_id']:s['sha256'] for s in task['sources']}
    print(f"preflight-ok fold={index} game={task['game_name']} sub={task['subject']} session={task['session']} sources={len(task['sources'])}", flush=True)
    write_json(root / 'reports' / f'fold-{index:02d}.running.json', dict(
        task=task, started_at=time.time(), job_id=os.getenv('PBS_JOBID'),
        source=str(source), source_sha256=digest(source), checkpoint_validation=dry['checkpoint_payload_validation']))
    # Preserve headroom even if new policies imagine far more than the pilot.
    # Only terminate this capture's own process group, leaving other jobs alone.
    with subprocess.Popen(args, start_new_session=True) as process:
        while True:
            try:
                code = process.wait(timeout=30)
                if code:
                    raise subprocess.CalledProcessError(code, args)
                break
            except subprocess.TimeoutExpired:
                if shutil.disk_usage(root).free < 80 * 1024**3:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError('Capture stopped: shared storage has less than 80 GiB free; partial output retained for resume')
    verify_task(root, manifest, task)
    status(root)


def status(root):
    manifest = json.loads((root / 'campaign.json').read_text())
    completed = []
    for task in manifest['tasks']:
        proof = root / 'reports' / f"fold-{task['index']:02d}.completed.json"
        if proof.exists():
            completed.append(task)
    print(f'Verified complete: {len(completed)}/48 folds; {sum(len(t["sources"]) for t in completed)}/132 sources')
    # Publish subject-level indexes only when all four matching checkpoints finish.
    # These indexes reference existing stores; they do not duplicate image data or
    # pretend that four checkpoints are a single-checkpoint capture manifest.
    for game in (0, 1):
        for subject in range(1, 7):
            tasks = [t for t in completed if t['game'] == game and t['subject'] == subject]
            if len(tasks) != 4:
                continue
            sources = [s['source_id'] for t in tasks for s in t['sources']]
            assert len(sources) == len(set(sources)) == 11
            write_json(root / 'subjects' / f'sub{subject:03d}_game{game}.json', dict(
                status='COMPLETED', subject=subject, game=game, source_count=11,
                capture_roots=[t['capture'] for t in tasks], checkpoints=[t['checkpoint'] for t in tasks],
                assignment=manifest['assignment']))
    if len(completed) == 48:
        write_json(root / 'COMPLETED.json', dict(status='COMPLETED', fold_count=48, source_count=132))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'verify', 'status'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.root)
    elif args.command == 'status':
        status(args.root)
    else:
        if args.index is None or not 0 <= args.index < 48:
            parser.error('--index must be in 0..47')
        if args.command == 'run':
            run(args.root, args.index)
        else:
            manifest = json.loads((args.root / 'campaign.json').read_text())
            verify_task(args.root, manifest, manifest['tasks'][args.index])


if __name__ == '__main__':
    main()
