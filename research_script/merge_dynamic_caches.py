#!/usr/bin/env python3
"""Publish a standalone 72-checkpoint cache collection without changing inputs.

Uses independent copies (copy-on-write reflinks where supported), never
symlinks/hardlinks. Numeric files and leaf manifests must match byte-for-byte.
Only copied checkpoint_binding.json files receive the new canonical path.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

REPO = Path('/home/jeongmin/thinker-dynamic-imitation')
OUTPUTS = REPO/'research_script/outputs'
SOURCES = (
    OUTPUTS/'allcheckpoints_allsessions_5m_20260923',
    OUTPUTS/'spaceinvaders_allcheckpoints_allsessions_5m_20260928',
)
GAMES = {0: 'Enduro', 1: 'Pong', 2: 'SpaceInvaders'}
REQUIRED = {'manifest.json', 'checkpoint_binding.json', 'native_decisions.csv.gz',
    'native_events.csv.gz', 'imagination_paths.npz', 'forecast_outcomes.csv.gz',
    'representation_summaries.csv.gz', 'native_features_schema.json', 'native_features_qa.json',
    'actor_metrics.csv.gz', 'model_control_windows.csv.gz', 'persistence_windows.csv.gz',
    'rollout_windows.csv.gz', 'visual_controls.npz'}


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf8')


def expected_keys():
    return {(subject, game, leave) for subject in range(1, 7)
            for game in range(3) for leave in range(1, 5)}


def inspect_sources():
    entries = []
    seen = set()
    for campaign in SOURCES:
        proof = read(campaign/'COMPLETED.json')
        if proof['status'] != 'COMPLETED':
            raise ValueError('Incomplete source campaign')
        paths = sorted((campaign/'00_dynamic_cache').glob('checkpoint_leave*/sub*_game*'))
        if len(paths) != proof['checkpoint_count']:
            raise ValueError('Source campaign cache count mismatch')
        for path in paths:
            binding, manifest = read(path/'checkpoint_binding.json'), read(path/'manifest.json')
            key = binding['subject'], binding['game'], binding['leave_session']
            if key not in expected_keys() or key in seen:
                raise ValueError('Unexpected or duplicate key: %s' % (key,))
            seen.add(key)
            subject, game, leave = key
            relative = Path('checkpoint_leave%02d' % leave)/('sub%03d_game%d' % (subject, game))
            if path.relative_to(campaign/'00_dynamic_cache') != relative:
                raise ValueError('Folder/metadata identity mismatch')
            if binding['status'] != 'COMPLETED' or manifest['status'] != 'COMPLETED':
                raise ValueError('Incomplete leaf cache')
            if (manifest['subject'], manifest['game'], manifest['num_actions']) != (subject, game, 9 if game == 0 else 6):
                raise ValueError('Subject/game/action count mismatch')
            if manifest['adapter_version'] != 3 or manifest['source_count'] != 11 or manifest['runs'] != 11:
                raise ValueError('Cache schema/source coverage mismatch')
            if manifest['real_steps'] != binding['real_steps']:
                raise ValueError('Real step count mismatch')
            manifest_sha = sha256(path/'manifest.json')
            if manifest_sha != binding['cache_manifest_sha256']:
                raise ValueError('Cache manifest hash mismatch')
            if str(path) != binding['cache']:
                raise ValueError('Source cache path mismatch')
            if binding['session_roles'] != {str(s): 'held_out' if s == leave else 'training' for s in range(1, 5)}:
                raise ValueError('Wrong training/held-out session roles')
            capture = read(Path(binding['capture'])/'manifest.json')
            if capture['manifest_sha256'] != binding['capture_manifest_sha256']:
                raise ValueError('Capture binding mismatch')
            if {e['metadata']['session'] for e in capture['episodes']} != {1, 2, 3, 4}:
                raise ValueError('Missing source session')
            current_cp = Path(binding['checkpoint'])/'milestone_5000000.json'
            if sha256(current_cp) != capture['semantic']['checkpoint']['milestone_sha256']:
                raise ValueError('Checkpoint changed since visualization')
            files = sorted(p for p in path.rglob('*') if p.is_file())
            if any(p.is_symlink() for p in path.rglob('*')) or path.is_symlink():
                raise ValueError('Refusing linked source cache')
            if not REQUIRED <= {p.name for p in files} or any(p.stat().st_size == 0 for p in files):
                raise ValueError('Missing/empty cache files')
            entries.append(dict(subject=subject, game=game, game_name=GAMES[game], leave_session=leave,
                relative_path=str(relative), source_cache=str(path), source_campaign=str(campaign),
                cache_manifest_sha256=manifest_sha, source_binding_sha256=sha256(path/'checkpoint_binding.json'),
                capture=binding['capture'], capture_manifest_sha256=binding['capture_manifest_sha256'],
                checkpoint=binding['checkpoint'], real_steps=manifest['real_steps'],
                episode_count=manifest['episode_count'], source_count=manifest['source_count'],
                bytes=sum(p.stat().st_size for p in files), file_count=len(files),
                session_roles=binding['session_roles'], source_runs=manifest['source_runs']))
    if seen != expected_keys():
        raise ValueError('Incomplete 6 subject x 3 game x 4 checkpoint coverage')
    return sorted(entries, key=lambda e: (e['leave_session'], e['subject'], e['game']))


def copy_leaf(entry, staging, destination):
    src = Path(entry['source_cache'])
    target = staging/entry['relative_path']
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['cp', '-a', '--reflink=auto', '--', str(src), str(target)], check=True)
    original_files = {str(p.relative_to(src)): p for p in src.rglob('*') if p.is_file()}
    copied_files = {str(p.relative_to(target)): p for p in target.rglob('*') if p.is_file()}
    if set(original_files) != set(copied_files):
        raise ValueError('Copy file coverage mismatch')
    checksums = {}
    for name, original in sorted(original_files.items()):
        copied = copied_files[name]
        before = original.stat()
        original_sha = sha256(original)
        after = original.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed during merge')
        if before.st_ino == copied.stat().st_ino and before.st_dev == copied.stat().st_dev:
            raise ValueError('Refusing hardlinked copies')
        if copied.is_symlink() or original_sha != sha256(copied):
            raise ValueError('Copied bytes differ: %s' % copied)
        checksums[name] = original_sha
    if checksums['manifest.json'] != entry['cache_manifest_sha256'] or checksums['checkpoint_binding.json'] != entry['source_binding_sha256']:
        raise ValueError('Metadata changed after preparation')
    # Leaf data and analysis manifest stay untouched. Relocate only its binding.
    binding = read(target/'checkpoint_binding.json')
    binding.update(cache=str(destination/entry['relative_path']),
                   source_cache=entry['source_cache'], source_binding_sha256=entry['source_binding_sha256'])
    write(target/'checkpoint_binding.json', binding)
    checksums['checkpoint_binding.json'] = sha256(target/'checkpoint_binding.json')
    result = dict(entry, cache=binding['cache'], file_sha256=checksums)
    print('VERIFIED', entry['relative_path'], entry['file_count'], 'files', flush=True)
    return result


def merge(destination, workers=2):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Refusing existing destination: %s' % destination)
    entries = inspect_sources()
    size = sum(e['bytes'] for e in entries)
    if shutil.disk_usage(destination.parent).free < size+80*1024**3:
        raise RuntimeError('Insufficient space for independent copies plus 80 GiB reserve')
    staging = destination.with_name('.'+destination.name+'.merging-%d' % os.getpid())
    staging.mkdir()
    write(staging/'MERGE_IN_PROGRESS.json', dict(destination=str(destination), created_at=time.time(),
        sources=list(map(str, SOURCES)), expected_caches=len(entries), expected_bytes=size))
    print('MERGING', len(entries), 'caches', '%.2f GiB' % (size/1024**3), 'to', destination, flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        merged = list(pool.map(lambda e: copy_leaf(e, staging, destination), entries))
    # Ensure the immutable source manifests/bindings still match after all copies.
    for e in merged:
        if sha256(Path(e['source_cache'])/'manifest.json') != e['cache_manifest_sha256'] or sha256(
                Path(e['source_cache'])/'checkpoint_binding.json') != e['source_binding_sha256']:
            raise ValueError('Source metadata changed during merge')
    source_runs = {run for e in merged for run in e['source_runs']}
    manifest = dict(status='COMPLETED', kind='unified-checkpoint-specific-dynamic-cache', schema_version=1,
        created_at=time.time(), cache_count=len(merged), subjects=list(range(1, 7)), games=GAMES,
        leave_sessions=list(range(1, 5)), source_sessions=list(range(1, 5)),
        unique_source_runs=len(source_runs), checkpoint_source_pairs=sum(e['source_count'] for e in merged),
        real_steps=sum(e['real_steps'] for e in merged), episode_count=sum(e['episode_count'] for e in merged),
        source_roots=list(map(str, SOURCES)), numeric_data_unchanged=True, original_caches_preserved=True,
        standalone_copies=True, interpretation='leave_session identifies the checkpoint; every cache contains all four source sessions',
        caches=merged)
    write(staging/'manifest.json', manifest)
    columns = ['subject', 'game', 'game_name', 'leave_session', 'relative_path', 'cache', 'source_cache',
               'real_steps', 'episode_count', 'source_count', 'cache_manifest_sha256']
    with (staging/'cache_index.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(merged)
    (staging/'README.md').write_text(
        '# Unified dynamic cache\n\n'
        '72 caches: 6 subjects x 3 games x 4 leave-session checkpoints.\n\n'
        'Layout: `checkpoint_leave01..04/sub001..006_game0..2/`.\n'
        'Game 0 = Enduro; 1 = Pong; 2 = Space Invaders.\n\n'
        'Each checkpoint cache contains all four behavioral sessions; the\n'
        'leave-session identifies the held-out training fold, not a data filter.\n'
        'Never pool the four checkpoint replicas as independent human data.\n\n'
        'For existing analysis readers, pass `--cache-root` pointing to one\n'
        '`checkpoint_leaveNN` directory, plus `--subject` and `--game`.\n\n'
        'All numeric files and leaf manifests are byte-identical to the originals.\n'
        'Only copied checkpoint bindings use the new path, preserving the source\n'
        'path/hash. Files are standalone copies, not symbolic or hard links.\n'
        'The dated source campaigns, captures and frozen sources are unchanged.\n'
        'The top-level manifest inventories every file SHA256; cache_index.csv\n'
        'provides all 72 identities and relative paths. Native capture references\n'
        'still point to logs/visual_behav and are not copied here.\n', encoding='utf8')
    write(staging/'COMPLETED.json', dict(status='COMPLETED', cache_count=72,
        completed_at=time.time(), manifest_sha256=sha256(staging/'manifest.json'),
        verified_files=sum(e['file_count'] for e in merged),
        real_steps=manifest['real_steps'], episode_count=manifest['episode_count']))
    (staging/'MERGE_IN_PROGRESS.json').unlink()
    if destination.exists():
        raise FileExistsError('Destination appeared during merge')
    staging.rename(destination)
    print('COMPLETED', destination, '72 caches, original files preserved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=OUTPUTS/'00_dynamic_cache')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    merge(args.destination.resolve(), args.workers)
