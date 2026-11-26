"""Deterministic, resumable publication of numbered research results.

The public unit is ``outputs/NN/subNNN/game`` (or ``outputs/NN/group/game``). The output base may have any directory name.
Only figures, reusable numeric tables and two diagnostic logs are published.
Hashes identify stage contents in ``logs/run.json``; they never name a public
result directory. Work and locks live below ``outputs/.work``.

Example::

    with ResultStore(unit, {'analysis': 'allocation-v1', 'input': input_hash},
                     resume=True) as store:
        with store.stage('prepare', preparation_signature) as stage:
            if not stage.reused:
                stage.write_table('decisions', decisions)
        with store.stage('analyze', analysis_signature) as stage:
            if not stage.reused:
                data = stage.read_table('decisions')
                stage.write_table('gate_curves', fit(data))
        with store.stage('render', render_signature) as stage:
            if not stage.reused:
                curves = stage.read_table('gate_curves')
                figure.savefig(stage.figure_path('gate_curves', 'png'))
        store.publish()

For plot-only work, open the same unit and identity with ``resume=True`` and
enter only the render stage. Existing prepare/analyze outputs are verified,
then copied into scratch; neither raw inputs nor fitted models are reopened.

A stage commits to the scratch draft only after successful exit. ``publish``
validates the whole draft, then atomically exchanges it with the public unit
(on Linux). A journaled rename fallback recovers interrupted publication.
Failures leave the previous successful public result intact. An unfinished
verified draft can be resumed; successful publication removes its scratch.
Persistent lock files are private coordination state, never research output.

``blocked_input(reason)`` records an unsuccessful attempt without changing the
success status or numerical outputs of an older completed result. Callers
must not label missing inputs as zero-valued measurements.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import traceback as traceback_module
from typing import Mapping

STAGES = ('prepare', 'analyze', 'render')
GAMES = ('pong', 'enduro', 'spaceinvaders', 'mario')
SCHEMA_VERSION = 1


class ResultStoreError(RuntimeError):
    """Invalid result lifecycle, signature, or artifact."""


class OutputBusyError(ResultStoreError):
    """Another process owns this exact subject/game output unit."""


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, 'item'):
        return value.item()
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            result.update(block)
    return result.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _revision(meta):
    if not meta.get('stages'):
        return _digest({})
    return _digest({k: meta.get(k) for k in ('schema_version', 'identity', 'stages')})


def _artifact_key(relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or len(path.parts) != 2:
        raise ResultStoreError(f'Artifact must be a direct tables/ or figures/ child: {relative}')
    folder, name = path.parts
    good = ((folder == 'tables' and name.endswith(('.csv', '.csv.gz', '.npz')))
            or (folder == 'figures' and name.endswith(('.png', '.pdf'))))
    if not good:
        raise ResultStoreError(f'Unsupported public artifact: {relative}')
    return path.as_posix()


def _artifacts(directory):
    result = {}
    for path in Path(directory).rglob('*'):
        if path.is_symlink():
            raise ResultStoreError(f'Symlinks are not result artifacts: {path}')
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in ('logs/run.json', 'logs/run.log'):
            continue
        result[_artifact_key(relative)] = _sha(path)
    return result


def _verify(directory, meta, required=()):
    stages = meta.get('stages', {})
    owned = {}
    for name in STAGES:
        if name not in stages:
            if name in required:
                raise ResultStoreError(f'Missing completed stage: {name}')
            continue
        record = stages[name]
        if record.get('status') != 'COMPLETED':
            raise ResultStoreError(f'Incomplete stage: {name}')
        for relative, expected in record['artifacts'].items():
            _artifact_key(relative)
            if relative in owned:
                raise ResultStoreError(f'Multiple stage owners: {relative}')
            owned[relative] = expected
    actual = _artifacts(directory)
    if actual != owned:
        differences = sorted(set(actual) ^ set(owned) |
                             {k for k in actual.keys() & owned.keys() if actual[k] != owned[k]})
        raise ResultStoreError('Result artifact verification failed: ' + ', '.join(differences[:8]))


def _exchange(left, right):
    """Use Linux atomic directory exchange; return False if unsupported."""
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, 'renameat2', None)
    if function is None:
        return False
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(left), -100, os.fsencode(right), 2) == 0:
        return True
    number = ctypes.get_errno()
    if number in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP, errno.EXDEV):
        return False
    raise OSError(number, os.strerror(number))


class Stage:
    """One stage's private filesystem and CSV/figure convenience interface."""
    def __init__(self, path, name, reused):
        self.path = Path(path)
        self.name = name
        self.reused = reused

    def table_path(self, name):
        if not str(name).endswith(('.csv', '.csv.gz', '.npz')):
            name = str(name) + '.csv.gz'
        return self.path / _artifact_key('tables/' + str(name))

    def write_table(self, name, frame):
        """Write plot-ready data as compressed CSV by default."""
        if self.reused:
            raise ResultStoreError('A reused stage must remain read-only')
        path = self.table_path(name)
        if path.suffix == '.npz':
            raise ResultStoreError('Use write_arrays for named numeric NPZ results')
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, compression='infer')
        return path

    def read_table(self, name, **kwargs):
        import pandas as pd
        return pd.read_csv(self.table_path(name), **kwargs)

    def write_arrays(self, name, **arrays):
        """Retain necessary numeric results such as ACE bases, never pickles."""
        import numpy as np
        if self.reused:
            raise ResultStoreError('A reused stage must remain read-only')
        if not str(name).endswith('.npz'):
            name = str(name) + '.npz'
        path = self.table_path(name)
        if not arrays or any(np.asarray(x).dtype.kind not in 'biufc' for x in arrays.values()):
            raise ResultStoreError('NPZ artifacts must contain named non-object numeric arrays')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path

    def figure_path(self, name, extension='png'):
        if self.reused:
            raise ResultStoreError('A reused stage must remain read-only')
        relative = _artifact_key(f'figures/{name}.{extension}')
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def import_artifacts(self, mapping: Mapping[str, Path]):
        """Copy only explicit final files from an existing analysis backend.

        Mapping keys are canonical relative names, e.g. tables/curves.csv.gz.
        The caller selects relevant results; directories and snapshots are
        intentionally never imported wholesale.
        """
        if self.reused:
            raise ResultStoreError('A reused stage must remain read-only')
        for relative, source in mapping.items():
            target = self.path / _artifact_key(relative)
            source = Path(source)
            if not source.is_file():
                raise ResultStoreError(f'Missing selected backend artifact: {source}')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


class ResultStore:
    """Transaction for one deterministic numbered subject/game result.

    ``identity`` contains scientific input/protocol identifiers, not a random
    run ID, subject list, timestamp, or whole-script render hash. Give each
    stage its own source/config signature via ``stage``. By default existing
    verified results require ``resume=True``. Changed signatures then rebuild
    only that stage and its descendants in private scratch.
    """
    def __init__(self, root, identity, resume=False):
        self.root = Path(root).expanduser().resolve()
        if (self.root.name not in GAMES or
            not (re.fullmatch(r'sub\d{3,}', self.root.parent.name) or self.root.parent.name == 'group') or
            not re.fullmatch(r'\d{2}', self.root.parent.parent.name)):
            raise ValueError('Expected outputs/NN/subNNN/game or outputs/NN/group/game')
        self.outputs_root = self.root.parents[2]
        self.identity = _jsonable(identity)
        _digest(self.identity)
        self.resume = bool(resume)
        self.key = _digest(str(self.root))[:20]
        self.work_root = self.outputs_root / '.work'
        self.work = self.work_root / self.key
        self.draft = self.work / 'draft'
        self.journal = self.work / 'publication.json'
        self._lock = None
        self._active_stage = None
        self._published = False
        self._blocked = False

    def __enter__(self):
        locks = self.work_root / '.locks'
        locks.mkdir(parents=True, exist_ok=True)
        self._lock = (locks / (self.key + '.lock')).open('a+')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            self._lock = None
            raise OutputBusyError(f'Output is already being written: {self.root}') from None
        try:
            self.work.mkdir(parents=True, exist_ok=True)
            self._recover_publication()
            public = self.root / 'logs/run.json'
            previous = _read(public) if public.exists() else {}
            if self.root.exists() and not public.exists():
                raise ResultStoreError('Existing output has no ResultStore manifest; migrate explicitly')
            if previous.get('stages'):
                _verify(self.root, previous)
                if not self.resume:
                    raise ResultStoreError('Completed results exist; use resume=True')
            self.base_revision = _revision(previous)
            state_path = self.work / 'state.json'
            can_resume = self.resume and self.draft.exists() and state_path.exists()
            if can_resume:
                saved = _read(state_path)
                can_resume = (saved.get('base_revision') == self.base_revision and
                              saved.get('identity') == self.identity)
            if can_resume:
                self.meta = _read(self.draft / 'logs/run.json')
                _verify(self.draft, self.meta)
            else:
                if self.draft.exists():
                    shutil.rmtree(self.draft)
                if previous.get('identity') == self.identity and previous.get('stages'):
                    shutil.copytree(self.root, self.draft)
                    self.meta = previous
                else:
                    self.draft.mkdir()
                    self.meta = dict(schema_version=SCHEMA_VERSION, identity=self.identity, stages={})
                self.meta['status'] = 'RUNNING'
                self.meta.pop('latest_attempt', None)
                self._save_meta()
            abandoned = self.work / 'previous-draft'
            if abandoned.exists():
                shutil.rmtree(abandoned)
            for orphan in self.work.glob('stage-*'):
                if orphan.is_dir():
                    shutil.rmtree(orphan)
            _atomic_json(state_path, dict(base_revision=self.base_revision, identity=self.identity))
            self.log('Opened result transaction')
            return self
        except BaseException:
            self._release()
            raise

    def _release(self):
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def _save_meta(self):
        self.meta['updated_at'] = time.time()
        _atomic_json(self.draft / 'logs/run.json', self.meta)

    def log(self, message):
        target = self.draft if self.draft.exists() and not self._published else self.root
        path = target / 'logs/run.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(time.strftime('%Y-%m-%dT%H:%M:%S%z ') + str(message) + '\n')

    @contextmanager
    def stage(self, name, signature):
        if self._lock is None or self._published or self._blocked:
            raise ResultStoreError('Stages require an open unpublished, unblocked transaction')
        if name not in STAGES or self._active_stage is not None:
            raise ResultStoreError('Stages must be named prepare/analyze/render and cannot overlap')
        position = STAGES.index(name)
        required = STAGES[:position]
        _verify(self.draft, self.meta, required)
        effective = _digest(dict(identity=self.identity, stage=name, signature=signature,
                                 parents={k: self.meta['stages'][k]['signature'] for k in required}))
        record = self.meta['stages'].get(name)
        if record and record['signature'] == effective:
            if not self.resume:
                raise ResultStoreError('Existing stage requires resume=True')
            self.log(f'Reused {name}: verified artifacts')
            before = _artifacts(self.draft)
            yield Stage(self.draft, name, True)
            if before != _artifacts(self.draft):
                raise ResultStoreError('Reused stage modified verified artifacts')
            return
        self._active_stage = name
        stage_path = Path(tempfile.mkdtemp(prefix='stage-', dir=self.work))
        shutil.rmtree(stage_path)
        shutil.copytree(self.draft, stage_path)
        stage_meta = json.loads(json.dumps(self.meta))
        for descendant in STAGES[position:]:
            old = stage_meta['stages'].pop(descendant, {})
            for relative in old.get('artifacts', {}):
                (stage_path / relative).unlink(missing_ok=True)
        retained = _artifacts(stage_path)
        started = time.time()
        self.log(f'Started {name}')
        try:
            yield Stage(stage_path, name, False)
            all_files = _artifacts(stage_path)
            if any(all_files.get(k) != v for k, v in retained.items()):
                raise ResultStoreError(f'{name} modified an upstream stage artifact')
            owned = {k: v for k, v in all_files.items() if k not in retained}
            stage_meta['stages'][name] = dict(status='COMPLETED', signature=effective,
                requested_signature=_jsonable(signature), artifacts=owned,
                elapsed_seconds=time.time() - started)
            stage_meta['status'] = 'STAGE_COMPLETE'
            stage_meta['updated_at'] = time.time()
            # Keep diagnostics written while the isolated stage was running.
            shutil.copy2(self.draft / 'logs/run.log', stage_path / 'logs/run.log')
            _atomic_json(stage_path / 'logs/run.json', stage_meta)
            _verify(stage_path, stage_meta, STAGES[:position+1])
            if not _exchange(stage_path, self.draft):
                previous = self.work / 'previous-draft'
                if previous.exists():
                    shutil.rmtree(previous)
                os.replace(self.draft, previous)
                os.replace(stage_path, self.draft)
                shutil.rmtree(previous)
            else:
                shutil.rmtree(stage_path)
            self.meta = stage_meta
            self.log(f'Completed {name}')
        finally:
            self._active_stage = None
            if stage_path.exists():
                shutil.rmtree(stage_path)

    def publish(self, required_stages=STAGES):
        """Publish a verified completed result; previous success survives errors."""
        if self._lock is None or self._active_stage or self._published or self._blocked:
            raise ResultStoreError('Cannot publish in the current transaction state')
        _verify(self.draft, self.meta, required_stages)
        final_status = 'COMPLETED' if all(name in self.meta['stages'] for name in STAGES) else 'STAGE_COMPLETE'
        self.meta.update(status=final_status, completed_at=time.time())
        self._save_meta()
        self.log('Publishing verified result')
        self.root.parent.mkdir(parents=True, exist_ok=True)
        backup = self.work / 'previous-public'
        _atomic_json(self.journal, dict(root=str(self.root), draft=str(self.draft), backup=str(backup)))
        if self.root.exists() and _exchange(self.draft, self.root):
            shutil.rmtree(self.draft)
        else:
            if self.root.exists():
                os.replace(self.root, backup)
            os.replace(self.draft, self.root)
            if backup.exists():
                shutil.rmtree(backup)
        self.journal.unlink(missing_ok=True)
        self._published = True
        shutil.rmtree(self.work)
        return self.root

    def _recover_publication(self):
        if not self.journal.exists():
            previous = self.work / 'previous-draft'
            if not self.draft.exists() and previous.exists():
                os.replace(previous, self.draft)
            return
        journal = _read(self.journal)
        if journal['root'] != str(self.root):
            raise ResultStoreError('Publication journal belongs to another output')
        backup = Path(journal['backup'])
        if not self.root.exists() and backup.exists():
            os.replace(backup, self.root)
        elif not self.root.exists() and self.draft.exists():
            meta = _read(self.draft / 'logs/run.json')
            _verify(self.draft, meta)
            os.replace(self.draft, self.root)
        if backup.exists():
            shutil.rmtree(backup)
        self.journal.unlink(missing_ok=True)

    def _record_attempt(self, status, reason):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / 'logs/run.json'
        prior = _read(path) if path.exists() else dict(schema_version=SCHEMA_VERSION,
                     identity=self.identity, stages={}, status=status)
        prior['latest_attempt'] = dict(status=status, reason=str(reason),
                                      identity=self.identity, time=time.time())
        if not prior.get('stages'):
            prior['status'] = status
        _atomic_json(path, prior)
        logfile = self.root / 'logs/run.log'
        with logfile.open('a') as stream:
            stream.write(time.strftime('%Y-%m-%dT%H:%M:%S%z ') + status + ': ' + str(reason) + '\n')

    def blocked_input(self, reason):
        """Record missing input; keep an older COMPLETED result and its tables."""
        if self._lock is None or self._published or self._active_stage:
            raise ResultStoreError('Record blocked input outside a stage in an open transaction')
        self._blocked = True
        self.log('BLOCKED_INPUT: ' + str(reason))
        self._record_attempt('BLOCKED_INPUT', reason)

    def __exit__(self, kind, error, traceback):
        try:
            if error is not None and not self._published:
                self.log('FAILED: ' + ''.join(traceback_module.format_exception(kind, error, traceback)))
                self._record_attempt('FAILED', repr(error))
        finally:
            self._release()
        return False
