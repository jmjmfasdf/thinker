"""Read-only V1 bridge and versioned V2 artifact contracts.

V1's frozen preprocessing/CV implementation is imported, never modified. New
fits always write below V2. Primary estimates are copied, not recomputed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path('/home/jeongmin/thinker-dynamic-imitation')
HERE = Path(__file__).resolve().parent
VERSION = 'continuous-planning-v2.0'
DEFAULT_V1 = REPO / 'research_script/outputs/continuous_planning_v1'
DEFAULT_OUT = REPO / 'research_script/outputs/continuous_planning_v2'
FIGURES = ('02-F1', '02-F2', '02-F3', '02-F4', '02-F5', '02-F6', '02-F7',
           '03-F1', '03-F2', '03-F3', '03-S1', '04-F1', '04-F1b', '04-F2',
           '04-F3', 'N1', 'N2', 'N3', '02-S1')
CAUSES = ('NOOP_to_active', 'active_to_NOOP', 'active_to_other_active')
CAUSE_LABELS = ('NOOP → active', 'Active → NOOP', 'Active → other active')
BUDGET_EDGES = [-.5, .5, 1.5, 2.5, 4.5, 8.5, 15.5, np.inf]
BUDGET_LABELS = ['0', '1', '2', '3–4', '5–8', '9–15', '16+']
AGE_EDGES = np.array([0, .125, .25, .5, 1, 2, 4, 8, 16, 32, 64, 128, 512.])
MARGIN_EDGES = np.array([-32., -8., -4., -2., -1., 0., 1., 2., 4., 8., 32.])
PRIMARY = {'H1': 'current_compute_switch', 'H2': 'current_compute_destination',
           'H3': 'all_earlier_computation_increment'}


def parser():
    p = argparse.ArgumentParser(description='Legacy-layout dynamic planning analysis, read-only V1 inputs')
    p.add_argument('--v1-root', type=Path, default=DEFAULT_V1)
    p.add_argument('--capture-index', type=Path)
    p.add_argument('--subject', type=int, choices=range(1, 7), default=1)
    p.add_argument('--game', type=int, choices=(0, 1), default=0)
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    p.add_argument('--config', type=Path)
    p.add_argument('--render-only', action='store_true')
    p.add_argument('--stage', choices=('all', 'derive', 'fit', 'tables', 'render', 'verify', 'group'), default='all')
    return p


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def bridge(root):
    frozen = root / 'source/research_script/continuous_planning'
    if not frozen.is_dir():
        raise ValueError('V1 frozen source is required; live source is not a fallback')
    c = module('common', frozen / 'common.py')
    m = module('models', frozen / 'models.py')
    a = module('v1_analysis', frozen / 'analysis.py')
    return c, m, a


class Context:
    def __init__(self, args):
        self.args = args
        self.C, self.M, self.A = bridge(args.v1_root)
        self.tag = f'sub{args.subject:03d}_game{args.game}'
        self.root = args.output_root / self.tag
        if args.output_root.resolve() == args.v1_root.resolve() or args.v1_root.resolve() in args.output_root.resolve().parents:
            raise ValueError('V2 output cannot be V1 or a directory inside V1')
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = args.v1_root / '00_native_cache' / self.tag
        self.inputs = {}
        self.manifest = self.verified(self.cache / 'manifest.json')
        if self.manifest['status'] != 'COMPLETED' or self.manifest['smoke']:
            raise ValueError('Only complete full-data native caches are allowed')
        if (self.manifest['subject'], self.manifest['game']) != (args.subject, args.game):
            raise ValueError('Subject/game mismatch')
        if self.manifest['source_runs'] != 11:
            raise ValueError('Expected 11 source runs')
        self.actions = self.manifest['num_actions']
        frozen = args.v1_root/'source/research_script/continuous_planning'
        for name in ('common.py', 'models.py', 'analysis.py'):
            if self.C.sha(frozen/name) != self.manifest['sources'][name]:
                raise ValueError(f'Frozen V1 implementation no longer matches original result: {name}')
        self.config = dict(self.manifest['config'])
        if args.config:
            supplied = json.loads(args.config.read_text())
            permitted = {'c_grid', 'max_iter', 'tolerance', 'pca_components', 'pca_fit_rows', 'seed'}
            if set(supplied) - permitted:
                raise ValueError('Only new-fit tuning configuration can be overridden')
            self.config.update(supplied)
        self.source_hashes = {p.name: self.C.sha(p) for p in sorted(HERE.glob('*.py'))}

    def verified(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Awaiting V1 artifact: {path}')
        self.inputs[str(path)] = self.C.sha(path)
        if path.suffix == '.json':
            return json.loads(path.read_text())
        return pd.read_csv(path)

    def table(self, module_id, name):
        path = self.args.v1_root / self.C.MODULES[module_id] / self.tag
        manifest = self.verified(path / 'manifest.json')
        if manifest['status'] != 'COMPLETED':
            raise ValueError(f'Awaiting completed V1 module {module_id}')
        value = self.verified(path / name)
        if self.inputs[str(path/name)] != manifest['artifacts'][name]:
            raise ValueError(f'Changed V1 artifact: {path/name}')
        return value

    def decisions(self, supplemented=True):
        d = self.verified(self.cache/'decisions.csv.gz')
        if self.inputs[str(self.cache/'decisions.csv.gz')] != self.manifest['artifacts']['decisions.csv.gz']:
            raise ValueError('Changed V1 decisions')
        if supplemented:
            extra = pd.read_csv(self.root/'derived/decision_extra.csv.gz')
            d = d.merge(extra, on='decision_id', validate='one_to_one')
        return d

    def prediction(self, name, outcome='switch'):
        # Identify the canonical model context through module 02, not a slope
        # subset / delayed-history sensitivity model with the same spec name.
        qa = self.table('02', 'qa.json')['models']
        canonical = next(v for k, v in qa.items() if k.startswith('baseline/switch@'))
        key = next(k for k, v in qa.items() if v is canonical)
        prefix = key.split('@')[-1]
        stores = list((self.args.v1_root/'model_store'/self.tag).glob(prefix+'*'))
        candidates = []
        for store in stores:
            candidates += list(store.glob(f'{outcome}_{name}_*/completed.json'))
        if len(candidates) != 1:
            raise ValueError(f'Expected one canonical V1 model {name}/{outcome}, got {len(candidates)}')
        report = self.verified(candidates[0])
        pred = self.verified(candidates[0].parent/'predictions.csv.gz')
        return pred, report

    def pixels(self):
        with np.load(self.cache/'visual.npz', allow_pickle=False) as z:
            return z['lowres'].astype(np.float32)

    def engine_args(self):
        # Engine uses V1 settings/schema, with its own output root. No V1 writes.
        config_path = self.root/'new_fit_config.json'
        self.C.write_json(config_path, self.config)
        return SimpleNamespace(subject=self.args.subject, game=self.args.game,
            output_root=self.args.output_root, config=config_path, max_episodes=None)

    def seal(self, directory, **metadata):
        directory = Path(directory)
        artifacts = {str(p.relative_to(directory)): self.C.sha(p) for p in sorted(directory.rglob('*'))
                     if p.is_file() and p.name != 'manifest.json'}
        self.C.write_json(directory/'manifest.json', dict(version=VERSION, status='COMPLETED',
            subject=self.args.subject, game=self.args.game, inputs=self.inputs,
            sources=self.source_hashes, config=self.config, artifacts=artifacts, **metadata))

    def verify_seal(self, directory):
        directory = Path(directory)
        m = json.loads((directory/'manifest.json').read_text())
        if m['status'] != 'COMPLETED':
            raise ValueError(f'Incomplete {directory}')
        dependencies = {'derived': ('core.py', 'derive.py'),
                        'fits': ('core.py', 'fitting.py'),
                        'panels': ('core.py', 'panels.py')}
        for name in dependencies.get(directory.name, ()):
            if m.get('sources', {}).get(name) != self.source_hashes.get(name):
                raise ValueError(f'Source changed since producing {directory}: {name}; preserve this output and select a new root')
        if m.get('config') != self.config:
            raise ValueError(f'Configuration changed for {directory}')
        for name, expected in m['artifacts'].items():
            if self.C.sha(directory/name) != expected:
                raise ValueError(f'Changed V2 artifact: {directory/name}')
        return m


def risk(d):
    return d[d.scored.astype(bool) & (d.incumbent >= 0)].copy()


def primary_lock(ctx):
    out = ctx.root/'provenance'
    out.mkdir(exist_ok=True)
    rows = []
    for mod in ('02', '03'):
        e = ctx.table(mod, 'effects.csv')
        rows.append(e[e.primary.isin(PRIMARY)])
        for filename in ('effects.csv', 'run_effects.csv'):
            source = ctx.args.v1_root/ctx.C.MODULES[mod]/ctx.tag/filename
            target = out/f'v1_{mod}_{filename}'
            data = source.read_bytes()
            if target.exists() and target.read_bytes() != data:
                raise ValueError('Previously locked primary input changed')
            target.write_bytes(data)
    primary = pd.concat(rows, ignore_index=True)
    if len(primary) != 3 or set(primary.primary) != set(PRIMARY):
        raise ValueError('Expected unchanged H1, H2, H3')
    primary.to_csv(out/'primary_subject.csv', index=False)
    path = ctx.args.v1_root/'08_group_report/primary_max_t.csv'
    if path.exists():
        target = out/'v1_primary_max_t.csv'
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError('V1 primary group inference changed')
        target.write_bytes(path.read_bytes())
        ctx.inputs[str(path)] = ctx.C.sha(path)
    ctx.C.write_json(out/'primary_lock.json', dict(inputs=ctx.inputs,
        primary_group_status='LOCKED' if path.exists() else 'AWAITING_V1_GROUP',
        method='Byte-identical primary source tables; no primary refitting or re-testing'))
