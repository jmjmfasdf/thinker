#!/usr/bin/env python3
"""Verify published milestones and clean only the current job's owned scratch.

Training/publication failure is fatal. Failure to delete already-published
scratch is a durable maintenance warning and never changes training success.
This module uses only the standard library, including during EXIT handling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import tempfile
import time

import campaign_5m_controller as c


def require(value, message):
    if not value:
        raise c.CampaignError(message)


def sha256(path):
    return c._sha256_file(Path(path))


def milestone_files(target, info):
    result = [target / 'milestone_5000000.json']
    for artifact in info.get('artifacts', {}).values():
        name = artifact['filename']
        require(Path(name).name == name, 'artifact path must be a basename')
        result.append(target / name)
    result.append(target / 'config_c.yaml')
    for key in ('source', 'behavior'):
        name = info[key]['published_manifest']
        require(Path(name).name == name, 'manifest path must be a basename')
        result.append(target / name)
    return result


def verify_milestone(target, campaign_root, row, expected_job=None):
    target, campaign_root = Path(target), Path(campaign_root)
    info = json.loads((target / 'milestone_5000000.json').read_text())
    require(info['status'] == 'VERIFIED_5M_STOPPED', 'milestone not verified')
    require(info['task_id'] == row['task_id'], 'milestone task mismatch')
    logical = {k: v for k, v in row['logical'].items() if k != 'num_actions'}
    require(info['logical'] == logical, 'milestone fold mapping mismatch')
    require(info['array_task_id'] == row['array_task_id'], 'milestone array index mismatch')
    if expected_job is not None:
        require(info['job_key'] == expected_job, 'milestone belongs to a different attempt')
    require(set(info['artifacts']) == {'actor', 'model'}, 'missing checkpoint role')
    for alias in ('finish', 'ckp_actor.tar', 'ckp_model.tar'):
        require(not os.path.lexists(target / alias), f'forbidden alias: {alias}')
    for path in milestone_files(target, info):
        require(path.is_file() and not path.is_symlink(), f'not a regular artifact: {path}')
    for role, artifact in info['artifacts'].items():
        step = artifact['step']
        require(c.MILESTONE_MIN_STEP <= step < c.MILESTONE_MAX_STEP_EXCLUSIVE,
                'checkpoint outside milestone window')
        require(artifact['filename'] == f'ckp_{role}.tar_step_{step}', 'checkpoint filename/step mismatch')
        p = target / artifact['filename']
        require(p.stat().st_size == artifact['size'], f'{role} size mismatch')
        require(sha256(p) == artifact['sha256'], f'{role} hash mismatch')
    require(sha256(target / 'config_c.yaml') == info['config']['sha256'], 'config hash mismatch')
    for key, digest in [('source', c.SOURCE_MANIFEST_SHA256), ('behavior', c.BEHAVIOR_MANIFEST_SHA256)]:
        require(info[key]['manifest_sha256'] == digest, f'{key} provenance mismatch')
        require(sha256(target / info[key]['published_manifest']) == digest, f'{key} manifest hash mismatch')
    require(info['behavior']['imitation_data_signature'] == row['behavior']['behavioral_data_signature'],
            'training data signature mismatch')
    index = campaign_root / 'milestones' / f"{row['task_id']}.json"
    expected_index = {**info, 'canonical_path': str(target.resolve()),
                      'canonical_milestone_sha256': sha256(target / 'milestone_5000000.json')}
    require(json.loads(index.read_text()) == expected_index, 'external milestone index mismatch')
    return info


def verification_fingerprint(target, campaign_root, row):
    target = Path(target)
    info = json.loads((target / 'milestone_5000000.json').read_text())
    files = milestone_files(target, info) + [Path(campaign_root) / 'milestones' / f"{row['task_id']}.json"]
    for alias in ('finish', 'ckp_actor.tar', 'ckp_model.tar'):
        require(not os.path.lexists(target / alias), f'forbidden alias: {alias}')
    return tuple((str(p), p.lstat().st_ino, p.lstat().st_size, p.lstat().st_mtime_ns,
                  p.lstat().st_mode) for p in files)


def _make_owned_directories_writable(fd):
    """Traverse directory descriptors, never following symlinks or chmod'ing files."""
    st = os.fstat(fd)
    require(st.st_uid == os.getuid(), 'scratch directory is not owned by this user')
    os.fchmod(fd, stat.S_IMODE(st.st_mode) | stat.S_IRWXU)
    for name in os.listdir(fd):
        entry = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISDIR(entry.st_mode):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                require(os.fstat(child).st_dev == st.st_dev, 'refusing nested filesystem')
                _make_owned_directories_writable(child)
            finally:
                os.close(child)


def remove_owned_tree(path, parent, expected_name):
    path, parent = Path(path), Path(parent)
    require(path == parent / expected_name and path.parent == parent, 'unexpected scratch path')
    require(parent.is_absolute() and parent.resolve() == parent, 'scratch parent must be canonical')
    require(expected_name not in ('', '.', '..') and '/' not in expected_name, 'invalid scratch basename')
    if not os.path.lexists(path):
        return 'absent'
    require(shutil.rmtree.avoids_symlink_attacks, 'safe rmtree unavailable')
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        original = os.fstat(fd)
        _make_owned_directories_writable(fd)
        current = path.lstat()
        require((current.st_dev, current.st_ino) == (original.st_dev, original.st_ino),
                'scratch root changed during cleanup')
        shutil.rmtree(path)
    finally:
        os.close(fd)
    return 'removed'


def clean_job_paths(job_key, scratch_parent=None, tmp_parent=Path('/tmp')):
    require(re.fullmatch(r'\d+_\d+', job_key), 'invalid job key')
    if scratch_parent is None:
        scratch_parent = Path('/scratch') / pwd.getpwuid(os.getuid()).pw_name
    result = []
    for parent, name in [(Path(scratch_parent), f'di72-{job_key}'),
                         (Path(tmp_parent), f'r72{job_key}'),
                         (Path(tmp_parent), f't72{job_key}')]:
        try:
            outcome = {'path': str(parent / name), 'status': remove_owned_tree(parent / name, parent, name)}
        except Exception as exc:
            outcome = {'path': str(parent / name), 'status': 'warning', 'error': repr(exc)}
        result.append(outcome)
    return result


def preflight(parent):
    """Use the same cleanup on real storage and prove symlink targets stay intact."""
    parent = Path(parent)
    fixture = Path(tempfile.mkdtemp(prefix='.di72-lifecycle-', dir=parent))
    sentinel = fixture / 'immutable'
    sentinel.mkdir()
    (sentinel / 'data').write_bytes(b'unchanged immutable input\n')
    sentinel.chmod(0o555)
    before = (sentinel.stat().st_mode, sha256(sentinel / 'data'))
    scratch = fixture / 'scratch'
    (scratch / 'readonly' / 'nested').mkdir(parents=True)
    (scratch / 'readonly' / 'nested' / 'data').write_text('discard scratch only')
    (scratch / 'source-link').symlink_to(sentinel, target_is_directory=True)
    for p in [scratch / 'readonly' / 'nested', scratch / 'readonly', scratch]:
        p.chmod(0o555)
    remove_owned_tree(scratch, fixture, 'scratch')
    require(not scratch.exists(), 'scratch cleanup failed')
    require(before == (sentinel.stat().st_mode, sha256(sentinel / 'data')), 'symlink target was modified')
    remove_owned_tree(fixture, parent, fixture.name)
    return {'readonly_cleanup': 'passed', 'symlink_target_preserved': True, 'storage': str(parent)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    pre = sub.add_parser('preflight')
    pre.add_argument('--parent', required=True, type=Path)
    finish = sub.add_parser('finish')
    finish.add_argument('--campaign-root', required=True, type=Path)
    finish.add_argument('--publish-root', required=True, type=Path)
    finish.add_argument('--job-key', required=True)
    args = parser.parse_args()
    if args.command == 'preflight':
        print(json.dumps(preflight(args.parent)))
        return 0
    index = int(args.job_key.split('_')[1])
    rows = [json.loads(line) for line in (args.campaign_root / 'pending_tasks.jsonl').read_text().splitlines()]
    matches = [row for row in rows if row['array_task_id'] == index]
    require(len(matches) == 1, 'job not in frozen ledger')
    row = matches[0]
    target = args.publish_root / row['logical']['xpid']
    info = verify_milestone(target, args.campaign_root, row, args.job_key)
    report = {'job_key': args.job_key, 'verified_publication': True,
              'milestone_sha256': sha256(target / 'milestone_5000000.json'),
              'checkpoint_steps': {k: v['step'] for k, v in info['artifacts'].items()},
              'checked_at_unix': time.time(), 'cleanup': clean_job_paths(args.job_key)}
    report['cleanup_ok'] = all(item['status'] != 'warning' for item in report['cleanup'])
    # Stdout is durable in the SLURM log even if the secondary report cannot be written.
    print(json.dumps(report), flush=True)
    try:
        report_dir = args.campaign_root / 'lifecycle'
        report_dir.mkdir(parents=True, exist_ok=True)
        c._write_json_atomic(report_dir / f'{args.job_key}.json', report)
    except OSError as exc:
        print(f'[cleanup-warning] could not write secondary report: {exc}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
