"""Current native-capture neural analyses, independent of legacy model regressors.

run_current_neural(number, decisions, captures_root, subject, game,
                   prepared_root=None) returns (tables, qa).
11=encoding/localization, 12=decoding, 13=RSA. Missing verified BOLD returns
BLOCKED_INPUT. prepared_root optionally holds compact HRF design matrices;
the caller owns result tables. Inputs are read-only. Neural session holdout
is not unseen-network-session validation for an all-session-trained Thinker.

Optional directional/motif columns require explicit provenance in
DataFrame.attrs['neural_optional_features']; matching column names alone never
cause inclusion. Missing optional features are recorded, not synthesized.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from common.io import sha
from .neural import standardize, residual

REPO = Path(__file__).resolve().parents[2]
GAMES = {'enduro': 0, 'pong': 1, 'spaceinvaders': 2}
FOCAL = ['M', 'log_budget', 'ema_initial', 'A_V', 'U_V', 'D_V']
REQUIRED = ['episode', 'run', 'session', 'source_frame', 'time', 'human_action',
            'incumbent_action', 'budget', 'M', 'root_value', 'root_entropy',
            'age', 'prior_reward', 'carry', 'ema_initial']
ACQUISITION = dict(behavior_time_zero_assumed_at_fmri_volume=60, fmri_tr_s=1.,
                   analyzed_fmri_volume_range_inclusive=[60, 539])


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _blocked(reason, **details):
    return [], dict(status='BLOCKED_INPUT', reason=reason, **details)


def _brain_root():
    relative = 'thinker/research_script/outputs/00_brain_cache/time_aligned_cache/v4_native_time_hrf'
    candidate = REPO.parent / relative
    default = candidate if candidate.exists() else Path.home() / relative
    return Path(os.environ.get('THINKER_BRAIN_CACHE', str(default)))



def input_fingerprint(subject, game):
    """Content identity of the BOLD files read by current 11–13/preparation.

    Call before deciding whether a prior analysis can be resumed. This performs
    no fitting, writes no cache, and does not rely on file modification times.
    A newly added/deleted run, alignment, ROI or voxel array changes the identity.
    We include every available run of the requested participant/game, a safe
    superset if the supplied native capture covers fewer runs. Actual acquisition
    alignment, behavior binding and voxel shape/order remain prepare_current's
    responsibility. Missing inputs have an explicit, reproducible identity.
    """
    game = str(game).lower()
    if game.isdigit():
        game = {v:k for k,v in GAMES.items()}.get(int(game), game)
    if game not in GAMES:
        raise ValueError('Unknown game: ' + game)
    brain = (_brain_root()/f'sub{int(subject):03d}_game{GAMES[game]}').resolve()
    files = {}
    runs = []
    if brain.is_dir():
        for run in sorted(p for p in brain.iterdir() if p.is_dir()):
            alignments = sorted(run.glob('time_alignment*.json'))
            arrays = sorted((run/'roi_bold_maxall').glob('*.npz'))
            if not alignments and not arrays:
                continue
            runs.append(dict(run=run.name, alignments=len(alignments), roi_arrays=len(arrays)))
            for path in alignments + arrays:
                files[str(path.relative_to(brain))] = sha(path)
    ready = bool(runs) and all(r['alignments'] == 1 and r['roi_arrays'] > 0 for r in runs)
    result = dict(version='current-bold-content-v1', subject=int(subject), game=game,
                  brain_cache=str(brain), status='AVAILABLE' if ready else 'BLOCKED_INPUT',
                  runs=runs, files=files)
    if not ready:
        result['reason'] = 'No complete BOLD run alignment/ROI-array inventory for this participant/game'
    result['fingerprint_sha256'] = _digest(result)
    return result


def _canonical(decisions):
    d = decisions.copy()
    missing = sorted(set(REQUIRED) - set(d.columns))
    if missing:
        raise ValueError('Native neural decisions missing columns: ' + ', '.join(missing))
    if ((d.budget < 0).any() or not np.isfinite(d.budget).all()
            or not np.equal(d.budget, np.floor(d.budget)).all()):
        raise ValueError('Invalid native planning budgets')
    if (not np.isfinite(d.source_frame).all() or (d.source_frame < 0).any()
            or not np.equal(d.source_frame, np.floor(d.source_frame)).all()):
        raise ValueError('Invalid original source-frame indices')
    if 'V_total' in d:
        if 'A_V' in d and not np.allclose(d.A_V, d.V_total, equal_nan=True):
            raise ValueError('A_V differs from cumulative absolute V_total')
        d['A_V'] = d.V_total
    if 'A_V' not in d:
        raise ValueError('A_V or V_total is required; legacy Q-based U cannot substitute')
    u = np.where(d.budget > 0, d.A_V / np.maximum(d.budget, 1), 0.)
    if 'U_V' in d and not np.allclose(d.U_V, u, equal_nan=True):
        raise ValueError('U_V differs from A_V divided by actual budget')
    d['U_V'] = u
    if {'V_initial', 'V_final'}.issubset(d.columns):
        delta = d.V_final - d.V_initial
        if 'D_V' in d and not np.allclose(d.D_V, delta, equal_nan=True):
            raise ValueError('D_V differs from final minus initial value')
        d['D_V'] = delta
    if 'D_V' not in d:
        raise ValueError('D_V or both V_initial/V_final are required')
    if (d.A_V < 0).any() or (d.loc[d.budget.eq(0), 'A_V'].fillna(0) != 0).any():
        raise ValueError('Cumulative V revision must be nonnegative and zero without computation')
    if (d.D_V.abs() > d.A_V + 1e-7).any():
        raise ValueError('Net value change cannot exceed cumulative absolute revision')
    d['log_budget'] = np.log1p(d.budget)
    d['log_age'] = np.log1p(d.age)
    optional = decisions.attrs.get('neural_optional_features', {})
    if not isinstance(optional, dict):
        raise ValueError('neural_optional_features must be a mapping')
    for name, meta in optional.items():
        if (name not in d or name in FOCAL or meta.get('kind') not in ('hold', 'impulse')
                or meta.get('availability') != 'past_or_current'
                or meta.get('fit_scope') not in ('unfitted', 'session_cross_fitted')
                or not meta.get('provenance')):
            raise ValueError('Missing/unsafe optional neural feature provenance: ' + name)
    for name in FOCAL + list(optional):
        if not np.isfinite(d[name].to_numpy(float)).any():
            raise ValueError('No finite observations for neural feature: ' + name)
    return d, optional


def _visual(images, frames):
    """Fixed spatial grayscale averages; no fitted visual representation."""
    chosen = np.asarray(images)[np.asarray(frames, int)]
    if chosen.ndim != 4 or chosen.shape[-1] not in (1, 3, 4):
        raise ValueError('Expected channels-last original behavioral images')
    gray = chosen[..., :3].astype(np.float32).mean(-1) / 255.
    rows = np.array_split(np.arange(gray.shape[1]), 7)
    cols = np.array_split(np.arange(gray.shape[2]), 7)
    return np.column_stack([gray[:, rr][:, :, cc].mean((1, 2)) for rr in rows for cc in cols])


def _forward(times, values, episodes, impulse_cols=(), n_tr=480, dt=.02):
    """Canonical HRF; no persistent holds across capture gaps/episode boundaries.

    Rows already represent unique human timestamps, averaging replay duplicates.
    Missing capture intervals have missingness regressors, not invented states.
    """
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Neural timestamps must be finite and strictly increasing')
    grid = np.arange(0, n_tr, dt)
    ix = np.searchsorted(times, grid, side='right') - 1
    clipped = np.clip(ix, 0, len(times)-1)
    ends = np.minimum(np.r_[times[1:], times[-1]+.25], times+.25)
    boundary = np.r_[np.asarray(episodes)[1:] != np.asarray(episodes)[:-1], False]
    ends[boundary] = times[boundary]
    valid = (ix >= 0) & (grid < ends[clipped])
    finite = np.isfinite(values)
    x = np.where(finite, values, 0.)[clipped].copy()
    missing = (~finite)[clipped].astype(float)
    x[~valid] = 0.
    missing[~valid] = 1.
    for col in impulse_cols:
        x[:, col] = 0.
        positions = np.ceil(times/dt-1e-12).astype(int)
        good = (positions >= 0) & (positions < len(grid)) & finite[:, col]
        np.add.at(x[:, col], positions[good], values[good, col]/dt)
    design = np.column_stack([x, missing])
    t = np.arange(0, 32, dt)
    h = stats.gamma.pdf(t, 6) - stats.gamma.pdf(t, 16)/6
    h /= h.sum()
    convolved = signal.fftconvolve(design, h[:, None], mode='full', axes=0)[:len(grid)]
    return convolved[np.rint(np.arange(n_tr)/dt).astype(int)]


def prepare_current(decisions, captures_root, subject, game, prepared_root=None):
    """Return verified BOLD/design runs plus compact provenance; no fitting."""
    game = str(game).lower()
    if game.isdigit():
        game = {v: k for k, v in GAMES.items()}[int(game)]
    if game not in GAMES:
        raise ValueError('Unknown game: ' + game)
    brain = _brain_root()/f'sub{int(subject):03d}_game{GAMES[game]}'
    if not brain.is_dir():
        return _blocked('No verified BOLD/ROI cache for this subject/game', required=str(brain))
    capture = Path(captures_root)/f'sub{int(subject):03d}'/game
    if not (capture/'COMPLETED.json').is_file():
        return _blocked('Completed native visualization capture is missing', required=str(capture))
    manifest = _read(capture/'manifest.json')
    completion = _read(capture/'COMPLETED.json')
    unsigned = dict(manifest)
    claimed = unsigned.pop('manifest_sha256')
    if (completion.get('status') != 'COMPLETED' or _digest(unsigned) != claimed
            or completion['manifest_sha256'] != claimed
            or _digest(manifest['episodes']) != manifest['episodes_sha256']
            or _digest(manifest['semantic']) != manifest['semantic_sha256']):
        raise ValueError('Native capture manifest/completion hash mismatch')
    metadata = {e['episode_id']: e['metadata'] for e in manifest['episodes']}
    if decisions.empty:
        return _blocked('No current native decisions supplied')
    d, optional = _canonical(decisions)
    if not set(d.episode).issubset(metadata):
        raise ValueError('Decisions contain episodes absent from the current capture')
    if 'subject' in d and not d.subject.eq(int(subject)).all():
        raise ValueError('Subject differs in neural decisions')
    columns = FOCAL + list(optional)
    controls = ['log_age', 'prior_reward', 'carry', 'root_value', 'root_entropy']
    if 'V_initial' in d:
        controls.append('V_initial')
    for action in sorted(set(d.incumbent_action.astype(int))):
        if action >= 0:
            name = f'previous_action_{action}'
            d[name] = d.incumbent_action.eq(action).astype(float)
            controls.append(name)
    visual = [c for c in d if c.startswith('visual_') and c != 'visual_change']
    visual_names = visual or [f'visual_{j}' for j in range(49)]
    names = columns + controls + visual_names
    impulses = [names.index(c) for c in ['log_budget', 'A_V', 'U_V', 'D_V']]
    impulses += [names.index(c) for c, meta in optional.items() if meta['kind'] == 'impulse']
    runs, checks = [], []
    for source_id, part in d.groupby('run', sort=True):
        metas = [metadata[e] for e in part.episode.unique()]
        if any(m['source_id'] != source_id or m['subject'] != int(subject)
               or m['game'] != GAMES[game] for m in metas):
            raise ValueError('Capture run/subject/game metadata disagree')
        expected = {m['source_sha256'] for m in metas}
        sessions = {int(m['session']) for m in metas}
        blocks = {int(m['block']) for m in metas}
        if len(expected) != 1 or len(sessions) != 1 or len(blocks) != 1:
            raise ValueError('Inconsistent source identity across capture episodes')
        digest, session, block = expected.pop(), sessions.pop(), blocks.pop()
        if not part.session.eq(session).all():
            raise ValueError('Decision session differs from capture metadata')
        label = f'sub{int(subject):03d}_ses{session:02d}_block{block:02d}_game{GAMES[game]}'
        br = brain/label
        align = list(br.glob('time_alignment*.json'))
        if len(align) != 1:
            return _blocked('Missing/ambiguous BOLD run alignment', run=label, required=str(br))
        am = _read(align[0])
        if am['source_signature']['behavior']['sha256'] != digest:
            raise ValueError('BOLD and native capture behavioral SHA differ: ' + label)
        timebase = am.get('fmri_behavior_timebase', {})
        if any(timebase.get(k) != v for k, v in ACQUISITION.items()):
            raise ValueError('BOLD acquisition alignment differs from the declared 480-TR protocol')
        raw = Path(metas[0]['source_path'])
        if not raw.is_file():
            raw = REPO/'behavioral_data_block'/source_id
        if not raw.is_file() or sha(raw) != digest:
            raise ValueError('Missing or changed original behavior: ' + str(raw))
        part = part.sort_values(['source_frame', 'episode', 'time']).copy()
        frames = part.source_frame.to_numpy(int)
        with np.load(raw, allow_pickle=False) as z:
            if (frames < 0).any() or (frames >= len(z['time'])).any():
                raise ValueError('Source frame outside original behavior')
            times = np.asarray(z['time'][frames], float)
            raw_action = z['action']
            action = raw_action.argmax(1) if raw_action.ndim == 2 else raw_action
            if not np.array_equal(action[frames], part.human_action.to_numpy(int)):
                raise ValueError('Native human action differs from original behavioral frame')
            if not np.allclose(times, part.time, atol=1e-6, rtol=0):
                raise ValueError('Native time differs from original behavioral timestamp')
            if not visual:
                if 'image' not in z:
                    return _blocked('Original image/explicit visual nuisances unavailable', run=label)
                part[visual_names] = _visual(z['image'], frames)
        if not np.isfinite(times).all() or times.min() < 0 or times.max() > 480.1:
            raise ValueError('Behavior outside canonical 480-second acquisition')
        if part.groupby('time').episode.nunique().gt(1).any():
            raise ValueError('Multiple capture episodes overlap at the same human time')
        # Multiple replay evaluations at one frame are not multiple brain events.
        unique = part.groupby('time', sort=True).agg({**{c: 'mean' for c in names}, 'episode': 'first'})
        x = _forward(unique.index.to_numpy(float), unique[names].to_numpy(float),
                     unique.episode.to_numpy(), impulse_cols=impulses)
        design_names = names + [c+'_missing' for c in names]
        x = np.column_stack([x, np.linspace(-1, 1, 480), np.linspace(-1, 1, 480)**2])
        design_names += ['drift', 'drift_squared']
        roi = br/'roi_bold_maxall'
        if not roi.is_dir() or not any(roi.glob('*.npz')):
            return _blocked('Full-voxel BOLD cache absent', run=label, required=str(roi))
        runs.append(dict(run=label, session=session, x=x, names=design_names, roi=roi))
        checks.append(dict(run=label, behavior_sha256=digest, behavior=str(raw),
                           alignment_sha256=sha(align[0]), decisions=len(part),
                           unique_human_times=len(unique), repeated_times_collapsed=len(part)-len(unique),
                           raw_time_range=[float(times.min()), float(times.max())], roi=str(roi)))
    if len({r['session'] for r in runs}) < 2:
        return _blocked('At least two human sessions are required for held-out neural readouts')
    qa = dict(status='COMPLETED', capture_manifest_sha256=claimed, runs=checks,
              focal_columns=columns, optional_features=optional,
              optional_features_absent=not bool(optional), design_columns=runs[0]['names'],
              visual_source='supplied current-frame descriptors' if visual else '7x7 current-image grayscale block means',
              acquisition_assumption={**ACQUISITION, 'scanner_triggers_available': False},
              timebase='SHA-verified original behavior time[source_frame]; no imagined-event timestamps',
              missing_capture='Causal holds stop at episode boundaries or gaps >250 ms; missingness regressors retained',
              hold_features=['M', 'ema_initial']+[c for c in controls+visual_names],
              impulse_features=[names[i] for i in impulses],
              network_training_scope='all human sessions; neural readout holdout does not imply unseen Thinker training sessions')
    if prepared_root is not None:
        target = Path(prepared_root)/f'sub{int(subject):03d}'/game
        target.mkdir(parents=True, exist_ok=True)
        for r in runs:
            np.savez_compressed(target/(r['run']+'.npz'), x=r['x'], names=np.asarray(r['names']))
        (target/'provenance.json').write_text(json.dumps(qa, indent=2)+'\n')
        qa['prepared_root'] = str(target)
    return runs, qa


def _roi_arrays(runs, roi):
    arrays, coords = [], None
    for r in runs:
        path = r['roi']/roi
        with np.load(path, allow_pickle=False) as z:
            values, current_coords = z['patterns'].astype(float), z['coords']
        if values.ndim != 2 or values.shape[0] != len(r['x']) or not np.isfinite(values).all():
            raise ValueError('Invalid BOLD shape/content: ' + str(path))
        if coords is not None and not np.array_equal(coords, current_coords):
            raise ValueError('ROI voxel order differs across runs: ' + roi)
        arrays.append(values)
        coords = current_coords
    return arrays


def _roi_names(runs):
    sets = [set(p.name for p in r['roi'].glob('*.npz')) for r in runs]
    common = sorted(set.intersection(*sets))
    if not common:
        raise ValueError('No ROI is present in every included BOLD run')
    return common, sorted(set.union(*sets)-set(common))


def _encoding(runs, focal, decode=False):
    names = runs[0]['names']
    x = np.concatenate([r['x'] for r in runs])
    sessions = np.concatenate([np.full(len(r['x']), r['session']) for r in runs])
    labels = np.concatenate([np.full(len(r['x']), r['run']) for r in runs])
    controls = [i for i, c in enumerate(names) if c not in focal]
    rois, dropped = _roi_names(runs)
    rows, fits, localization = [], [], []
    for roi in rois:
        yy = _roi_arrays(runs, roi)
        y = np.concatenate(yy)
        if not decode:
            for r, brain in zip(runs, yy):
                z = np.column_stack([np.ones(len(brain)), standardize(r['x'][:, controls])])
                ry = residual(brain, z).mean(1)
                for name in focal:
                    fi = names.index(name)
                    rx = residual(r['x'][:, fi:fi+1], z).ravel()
                    den = np.linalg.norm(rx)*np.linalg.norm(ry)
                    value = float(rx@ry/den) if den > 1e-10 else np.nan
                    localization.append(dict(run_label=r['run'], session=r['session'], roi=Path(roi).stem,
                                             term=name, value=value, status='ok' if np.isfinite(value) else 'constant'))
        for held in sorted(set(sessions)):
            train, test = sessions != held, sessions == held
            if decode:
                count = min(32, y.shape[1], int(train.sum())-1)
                pca = PCA(n_components=count, svd_solver='randomized', random_state=20261111)
                brain_train = pca.fit_transform(y[train])
                brain_test = pca.transform(y[test])
                xx = standardize(x[:, controls], train)
                for name in focal:
                    target = x[:, names.index(name)]
                    sd = target[train].std()
                    if sd < 1e-10:
                        fits.append(dict(roi=Path(roi).stem, heldout_session=int(held), term=name, status='constant_target'))
                        continue
                    target = (target-target[train].mean())/sd
                    base = Ridge(alpha=100.).fit(xx[train], target[train]).predict(xx[test])
                    full = Ridge(alpha=100.).fit(np.column_stack([xx[train], brain_train]), target[train]).predict(
                        np.column_stack([xx[test], brain_test]))
                    for run in np.unique(labels[test]):
                        take = labels[test] == run
                        value = float(((target[test][take]-base[take])**2-(target[test][take]-full[take])**2).mean())
                        rows.append(dict(run_label=run, session=int(held), roi=Path(roi).stem, term=name,
                                         value=value, n_voxels=y.shape[1], status='ok'))
            else:
                base_cols = controls + [names.index('M'), names.index('log_budget')]
                full_cols = base_cols + [names.index(c) for c in focal if c not in ('M', 'log_budget')]
                predictions = {}
                for label, cols in [('controls', base_cols), ('full', full_cols)]:
                    xx = standardize(x[:, cols], train)
                    predictions[label] = Ridge(alpha=100.).fit(xx[train], y[train]).predict(xx[test])
                for run in np.unique(labels[test]):
                    take = labels[test] == run
                    truth = y[test][take]
                    sse0 = ((truth-predictions['controls'][take])**2).sum(0)
                    sse1 = ((truth-predictions['full'][take])**2).sum(0)
                    denominator = ((truth-y[train].mean(0))**2).sum(0)
                    valid = denominator > 1e-10
                    value = float(((sse0-sse1)[valid]/denominator[valid]).mean()) if valid.any() else np.nan
                    rows.append(dict(run_label=run, session=int(held), roi=Path(roi).stem,
                                     term='evaluation_VOC_beyond_initial_quantity', value=value,
                                     n_voxels=int(valid.sum()), status='ok' if valid.any() else 'constant'))
            fits.append(dict(roi=Path(roi).stem, heldout_session=int(held), n_train=int(train.sum()),
                             n_test=int(test.sum()), alpha=100., status='ok'))
    tables = {('decoding_run_effects' if decode else 'encoding_run_effects'): pd.DataFrame(rows),
              'neural_fit_qa': pd.DataFrame(fits)}
    if not decode:
        tables['localization_run_effects'] = pd.DataFrame(localization)
    return tables, dropped


def _rsa(runs, focal):
    rows = []
    rois, dropped = _roi_names(runs)
    for roi in rois:
        _roi_arrays(runs, roi)
    for r in runs:
        x, names = r['x'], r['names']
        i, j = np.triu_indices(len(x), 1)
        keep = (j-i) >= 8
        def distance(cols):
            y = standardize(x[:, [names.index(c) for c in cols]])
            return stats.rankdata(pdist(y, 'euclidean')[keep]) if y.shape[1] else np.zeros(keep.sum())
        controls = [distance([c for c in names if c.startswith('visual_') and not c.endswith('_missing')]),
                    distance([c for c in names if c.startswith('previous_action_') and not c.endswith('_missing')]),
                    distance(['log_age', 'prior_reward', 'carry']), stats.rankdata((j-i)[keep]),
                    distance([c for c in names if c.endswith('_missing')])]
        values = {k: distance([k]) for k in focal}
        plans = {}
        for name in focal:
            for stage in ['behavior_visual_history', 'initial_and_quantity']:
                zz = list(controls)
                if stage == 'initial_and_quantity':
                    zz += [values[k] for k in ['M', 'log_budget'] if k != name]
                    initial = ['root_value', 'root_entropy'] + (['V_initial'] if 'V_initial' in names else [])
                    zz += [distance(initial)]
                z = np.column_stack([np.ones(keep.sum()), standardize(np.column_stack(zz))])
                u, s, _ = np.linalg.svd(z, full_matrices=False)
                q = u[:, s > s[0]*1e-10]
                fx = values[name]-q@(q.T@values[name])
                plans[name, stage] = q, fx, np.linalg.norm(fx)
        for roi in rois:
            with np.load(r['roi']/roi, allow_pickle=False) as z:
                y = z['patterns'].astype(float)
            brain = stats.rankdata(pdist(y, 'correlation')[keep])
            for (name, stage), (q, fx, norm) in plans.items():
                by = brain-q@(q.T@brain)
                den = norm*np.linalg.norm(by)
                value = float(fx@by/den) if den > 1e-8 else np.nan
                rows.append(dict(run_label=r['run'], session=r['session'], roi=Path(roi).stem,
                                 stage=stage, term=name, value=value, n_pairs=int(keep.sum()),
                                 n_voxels=y.shape[1], status='ok' if np.isfinite(value) else 'constant'))
    return {'rsa_run_effects': pd.DataFrame(rows)}, dropped


def run_current_neural(number, decisions, captures_root, subject, game, prepared_root=None):
    """Return tables/QA or empty tables and BLOCKED_INPUT; no fallback dataset."""
    number = f'{int(number):02d}'
    if number not in ('11', '12', '13'):
        raise ValueError('Current neural analysis number must be 11, 12 or 13')
    runs, qa = prepare_current(decisions, captures_root, subject, game, prepared_root)
    if qa['status'] != 'COMPLETED':
        return {}, qa
    focal = qa['focal_columns']
    tables, dropped = _rsa(runs, focal) if number == '13' else _encoding(runs, focal, decode=number == '12')
    qa.update(analysis=number, method={'11': 'fixed-ridge encoding and separate ROI-mean localization',
        '12': 'training-session PCA plus fixed-ridge incremental decoding',
        '13': 'partial rank RSA; exclude pairs separated by <8 TR'}[number],
        noncommon_rois_excluded=dropped, readout_holdout='whole human session',
        causal_claim=False, population_inference=False,
        value_definitions={'A_V': 'sum absolute within-imagination V changes',
            'U_V': 'A_V / actual computation count; 0 at B=0', 'D_V': 'V_final - V_initial'})
    for frame in tables.values():
        frame['subject'] = int(subject)
        frame['game'] = str(game)
    return tables, qa
