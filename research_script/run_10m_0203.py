"""Frozen, bounded-CPU 02/03 campaign for the imported November 2026 caches."""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback

REPO = Path('/work/jeongmin/thinker-dynamic-imitation')
SCRIPTS = REPO / 'research_script'
CACHE = SCRIPTS / 'outputs/00_dynamic_cache_10m_20261110'
OUT = SCRIPTS / 'outputs/analysis_0203_10m_20261111'
SOURCE = OUT / 'source/research_script'
PYTHON = Path('/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python')
ROOTS = {'02': OUT / '02_evaluation_revision_geometry',
         '03': OUT / '03_dynamic_voc_mechanisms'}
FILES = ('model_control_windows.csv.gz', 'actor_metrics.csv.gz',
         'persistence_windows.csv.gz', 'persistence_events.csv.gz',
         'native_decisions.csv.gz', 'native_events.csv.gz',
         'visual_controls.npz', 'visual_alignment.csv.gz')
SOURCE_FILES = ('02_evaluation_revision_geometry.py', 'shared_analysis_cache.py',
                '03_dynamic_voc_mechanisms.py', '03_actor_integration_mechanisms.py',
                'dynamic_voc_session_fixed.py', 'tests/test_dynamic_voc_session_fixed.py')
LOCK = threading.Lock()
INPUT_LOCKS = {(s, g): threading.Lock() for s in range(1, 7) for g in range(3)}
STATE = {}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(p, obj):
    p = Path(p)
    temp = p.with_name(p.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def event(tag, **values):
    with LOCK:
        STATE.setdefault('tasks', {}).setdefault(tag, {}).update(values)
        STATE['updated_at'] = time.time()
        write(OUT / 'progress.json', STATE)
        print(json.dumps(dict(task=tag, at=time.time(), **values)), flush=True)


def prepare():
    if OUT.exists():
        raise FileExistsError(f'Campaign already exists: {OUT}')
    completed = read(CACHE / 'COMPLETED.json')
    manifest = read(CACHE / 'manifest.json')
    assert completed['status'] == manifest['status'] == 'COMPLETED'
    assert completed['cache_count'] == len(manifest['caches']) == 72
    expected = {(s, g, c) for s in range(1, 7) for g in range(3) for c in range(1, 5)}
    assert {(e['subject'], e['game'], e['leave_session']) for e in manifest['caches']} == expected
    checked = []
    for e in manifest['caches']:
        p = CACHE / f"checkpoint_leave{e['leave_session']:02d}" / f"sub{e['subject']:03d}_game{e['game']}"
        assert Path(e['cache']).resolve() == p.resolve()
        m, b = read(p / 'manifest.json'), read(p / 'checkpoint_binding.json')
        assert m['status'] == b['status'] == 'COMPLETED'
        assert sha(p / 'manifest.json') == e['cache_manifest_sha256'] == b['cache_manifest_sha256']
        assert (m['subject'], m['game'], b['leave_session']) == (e['subject'], e['game'], e['leave_session'])
        assert b['checkpoint_real_step'] == 10000000
        assert m['num_actions'] == (9 if e['game'] == 0 else 6)
        assert m['real_steps'] == e['real_steps']
        assert b['session_roles'][str(e['leave_session'])] == 'held_out'
        for n, rec in b['artifacts'].items():
            assert (p / n).is_file() and (p / n).stat().st_size == rec['size'], p / n
        for n in FILES:
            assert n in b['artifacts']
        config = Path(b['checkpoint']) / 'config_c.yaml'
        assert config.is_file(), config
        import yaml
        flags = yaml.safe_load(config.read_text())
        assert flags['voc_gate_q_temperature'] == .01 and flags['voc_gru_connected'] is True
        assert str(flags['icopro_subjects']) == str(e['subject'])
        assert int(flags['icopro_game_id']) == e['game']
        assert str(flags['icopro_holdout_sessions']) == str(e['leave_session'])
        checked.append(dict(subject=e['subject'], game=e['game'], checkpoint=e['leave_session'],
                            path=str(p), manifest_sha256=sha(p / 'manifest.json'),
                            binding_sha256=sha(p / 'checkpoint_binding.json'),
                            config=str(config), config_sha256=sha(config), real_steps=m['real_steps']))
    SOURCE.mkdir(parents=True)
    (OUT / 'jobs').mkdir()
    hashes = {}
    for name in SOURCE_FILES:
        dest = SOURCE / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SCRIPTS / name, dest)
        hashes[name] = sha(dest)
    verifier = SCRIPTS / 'outputs/02_evaluation_revision_geometry_20260929_v10/run_campaign.py'
    shutil.copy2(verifier, SOURCE / 'analysis02_verifier.py')
    hashes['analysis02_verifier.py'] = sha(SOURCE / 'analysis02_verifier.py')
    if (REPO / 'ARIAL.TTF').is_file():
        shutil.copy2(REPO / 'ARIAL.TTF', SOURCE.parent / 'ARIAL.TTF')
    shutil.copy2(__file__, OUT / 'run_campaign.py')
    protocol = dict(created_at=time.time(), cache=str(CACHE), output=str(OUT),
                    cache_manifest_sha256=sha(CACHE / 'manifest.json'),
                    source_sha256=hashes, inputs=checked, subjects=list(range(1, 7)),
                    games=[0, 1, 2], checkpoints=[1, 2, 3, 4], python=str(PYTHON), workers=3,
                    cpu_only=True, analysis02_schema=10, analysis03_version='dynamic-voc-session-fixed-v2',
                    analysis02='Current .py v10; checkpoint estimates averaged within run; held-out sensitivity retained.',
                    analysis03='Generate fresh Q4 predictions and descriptive curves from 10M caches with native loader; '
                               'then run current session-fixed-effects analysis on one held-out checkpoint per session.',
                    scientific_source_changes=[], old_result_reuse=False,
                    planned_subject_analyses=36, planned_group_analyses=6,
                    final_session_fe_subjects=18, final_session_fe_groups=3)
    write(OUT / 'campaign.json', protocol)
    write(OUT / 'preflight.json', dict(status='PASS', caches=72, entries=checked,
          checks='All artifact sizes, manifest/binding hashes, checkpoint configs and complete 6x3x4 coverage. '
                 'Consumed data file hashes checked per subject/game before computation.'))
    print(json.dumps(dict(preflight='PASS', caches=72, output=str(OUT))), flush=True)


def validate_inputs(subject, game, protocol):
    with INPUT_LOCKS[subject, game]:
        output = OUT / 'jobs' / f'sub{subject:03d}_game{game}.inputs.json'
        if output.exists():
            assert read(output)['status'] == 'PASS'
            return
        checks = []
        for e in protocol['inputs']:
            if (e['subject'], e['game']) != (subject, game):
                continue
            p = Path(e['path'])
            assert sha(p / 'manifest.json') == e['manifest_sha256']
            assert sha(p / 'checkpoint_binding.json') == e['binding_sha256']
            assert sha(e['config']) == e['config_sha256']
            b = read(p / 'checkpoint_binding.json')
            for name in FILES:
                actual = sha(p / name)
                assert actual == b['artifacts'][name]['sha256'], p / name
                checks.append(dict(path=str(p / name), sha256=actual))
        assert len(checks) == 4 * len(FILES)
        write(output, dict(status='PASS', at=time.time(), files=checks))


def validate03(subject, game, protocol):
    import pandas as pd
    tag = f'sub{subject:03d}_game{game}' if subject else f'group_game{game}'
    root = ROOTS['03'] / tag
    m = read(root / 'analysis_manifest.json')
    assert m['status'] == 'COMPLETED' and m['game'] == game
    assert m['script_sha256'] == protocol['source_sha256']['03_dynamic_voc_mechanisms.py']
    if subject:
        assert m['subject'] == subject and m['prediction_requested']
        assert {s['checkpoint'] for s in m['source_caches']} == {1, 2, 3, 4}
        assert all(Path(s['cache']).resolve().is_relative_to(CACHE.resolve()) for s in m['source_caches'])
    else:
        assert m['subjects'] == list(range(1, 7))
    inference = 'run_inference.csv' if subject else 'group_inference.csv'
    tables = {}
    for name in ['run_effects_by_checkpoint.csv', 'curves_by_checkpoint.csv', inference]:
        frame = pd.read_csv(root / 'tables' / name)
        assert len(frame), (tag, name)
        tables[name] = dict(rows=len(frame), sha256=sha(root / 'tables' / name))
    inf = pd.read_csv(root / 'tables' / inference)
    assert 'Q4_prediction' in set(inf.family), f'No new-cache Q4 prediction: {tag}'
    figures = sorted((root / 'figures').glob('*.png'))
    assert len(figures) == 4 and all(p.stat().st_size > 0 for p in figures), tag
    return dict(status='PASS', manifest_sha256=sha(root / 'analysis_manifest.json'),
                tables=tables, figures=[p.name for p in figures])


def run_task(analysis, subject, game, protocol, verifier02):
    tag = f'sub{subject:03d}_game{game}' if subject else f'group_game{game}'
    task = analysis + '_' + tag
    event(task, status='VALIDATING_INPUTS', started_at=time.time(), subject=subject, game=game)
    try:
        if subject:
            validate_inputs(subject, game, protocol)
        name = '02_evaluation_revision_geometry.py' if analysis == '02' else '03_dynamic_voc_mechanisms.py'
        command = [str(PYTHON), '-B', '-u', str(SOURCE / name), '--game', str(game),
                   '--shared-cache-root', str(CACHE), '--output-root', str(ROOTS[analysis])]
        command += ['--subject', str(subject)] if subject else ['--aggregate-subjects', '1', '2', '3', '4', '5', '6']
        if analysis == '03':
            command.append('--runwise-legacy')
        logfile = OUT / 'jobs' / (task + '.log')
        with logfile.open('x') as f:
            child = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, cwd=SOURCE)
            event(task, status='RUNNING', pid=child.pid, command=command, log=str(logfile))
            code = child.wait()
        if code:
            raise RuntimeError(f'Analysis exited {code}; see {logfile}')
        if analysis == '02':
            report = verifier02.verify_outputs(tag, game, subject,
                      protocol['source_sha256']['02_evaluation_revision_geometry.py'])
        else:
            report = validate03(subject, game, protocol)
        write(OUT / 'jobs' / (task + '.verified.json'), report)
        event(task, status='COMPLETE', completed_at=time.time())
        return True
    except Exception as e:
        event(task, status='FAILED', completed_at=time.time(), error=str(e), traceback=traceback.format_exc())
        return False


def final_session_fixed(protocol):
    tag = '03_session_fixed_all'
    command = [str(PYTHON), '-B', '-u', str(SOURCE / '03_dynamic_voc_mechanisms.py'),
               '--subjects', '1', '2', '3', '4', '5', '6', '--games', '0', '1', '2',
               '--workers', '3', '--shared-cache-root', str(CACHE), '--output-root', str(ROOTS['03'])]
    logfile = OUT / 'jobs' / (tag + '.log')
    try:
        with logfile.open('x') as f:
            child = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, cwd=SOURCE)
            event(tag, status='RUNNING', pid=child.pid, command=command, log=str(logfile))
            code = child.wait()
        if code:
            raise RuntimeError(f'Session-FE analysis exited {code}; see {logfile}')
        import pandas as pd
        verified = []
        for game in range(3):
            for subject in [*range(1, 7), None]:
                folder = ROOTS['03'] / (f'sub{subject:03d}_game{game}' if subject else f'group_game{game}')
                m = read(folder / 'analysis_manifest.json')
                assert m['status'] == 'COMPLETED' and m['version'] == 'dynamic-voc-session-fixed-v2'
                assert m['game'] == game
                if subject:
                    assert m['subject'] == subject and m['sessions'] == [1, 2, 3, 4]
                    assert m['source_hashes']['dynamic_voc_session_fixed.py'] == protocol['source_sha256']['dynamic_voc_session_fixed.py']
                    assert all(Path(e['path']).resolve().is_relative_to(CACHE.resolve()) for e in m['source_caches'])
                else:
                    assert m['subjects'] == list(range(1, 7))
                effect = 'pooled_session_fixed_effects.csv' if subject else 'group_inference.csv'
                assert len(pd.read_csv(folder / 'tables' / effect))
                pred = pd.read_csv(folder / 'tables/prediction_inference.csv')
                assert len(pred) and set(pred.family) == {'Q4_prediction'}
                figures = sorted((folder / 'figures').glob('*.png'))
                assert len(figures) == 4 and all(p.stat().st_size > 0 for p in figures)
                verified.append(dict(folder=str(folder), manifest_sha256=sha(folder / 'analysis_manifest.json')))
        installation = read(ROOTS['03'] / 'session_fixed_installation.json')
        assert installation['protected_files_unchanged']
        write(OUT / 'jobs' / (tag + '.verified.json'), dict(status='PASS', entries=verified,
              q4_predictions_preserved=True, inference='Current session-FE primary; old runwise stage archived.'))
        event(tag, status='COMPLETE', completed_at=time.time(), subjects=18, groups=3)
        return True
    except Exception as e:
        event(tag, status='FAILED', error=str(e), traceback=traceback.format_exc(), completed_at=time.time())
        return False


def run():
    if (OUT / 'progress.json').exists():
        raise FileExistsError('Campaign already started; inspect before recovery.')
    protocol = read(OUT / 'campaign.json')
    assert sha(CACHE / 'manifest.json') == protocol['cache_manifest_sha256']
    for name, digest in protocol['source_sha256'].items():
        assert sha(SOURCE / name) == digest
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1',
                      MPLCONFIGDIR=str(OUT / 'matplotlib'))
    for p in ROOTS.values():
        p.mkdir()
        (p / 'jobs').mkdir()
    spec = importlib.util.spec_from_file_location('verifier02', SOURCE / 'analysis02_verifier.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.OUT = ROOTS['02']
    STATE.update(status='RUNNING', host=socket.gethostname(), pid=os.getpid(), started_at=time.time(), tasks={})
    event('campaign', status='RUNNING')
    tasks = [(a, s, g) for s in range(1, 7) for g in range(3) for a in ['02', '03']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda t: run_task(*t, protocol, verifier), tasks))
    if all(results):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            groups = list(pool.map(lambda t: run_task(*t, protocol, verifier),
                                  [(a, None, g) for g in range(3) for a in ['02', '03']]))
        ok = all(groups) and final_session_fixed(protocol)
    else:
        ok = False
        event('groups', status='BLOCKED', reason='At least one subject analysis failed; no partial group inference.')
    for name, digest in protocol['source_sha256'].items():
        assert sha(SOURCE / name) == digest
    assert sha(CACHE / 'manifest.json') == protocol['cache_manifest_sha256']
    STATE.update(status='COMPLETE' if ok else 'FAILED', completed_at=time.time())
    event('campaign', status=STATE['status'])
    if not ok:
        raise SystemExit(1)
    write(OUT / 'COMPLETED.json', dict(status='COMPLETE', completed_at=time.time(),
          subject_analyses=36, group_analyses=6, cache_manifest_sha256=protocol['cache_manifest_sha256'],
          source_sha256=protocol['source_sha256'], analysis03_primary='session-fixed-v2'))


if __name__ == '__main__':
    if sys.argv[1:] == ['prepare']:
        prepare()
    elif sys.argv[1:] == ['run']:
        run()
    else:
        raise SystemExit('Use prepare or run')
