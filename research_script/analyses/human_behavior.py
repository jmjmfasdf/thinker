"""Original human behavior, independent of Thinker checkpoints and replay caches.

All recorded observations are retained. Atari uses its exact native action set;
Mario uses original BK2 button combinations and retains the canonical-nine
mapping as a sensitivity analysis. Plotting consumes only the returned tables.
Group curves normalize within participant before averaging participants.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import warnings
import zipfile

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
NAMES = {'enduro': 'Enduro', 'pong': 'Pong', 'spaceinvaders': 'Space Invaders',
         'mario': 'Super Mario Bros.'}
ATARI = {
    'enduro': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'DOWN', 'DOWNRIGHT', 'DOWNLEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'pong': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'spaceinvaders': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
}
BUTTONS = ['A', 'Right', 'Left', 'Down', 'B', 'Select', 'Start', 'Up']
VOCAB = [[], ['Right'], ['Right', 'B'], ['A', 'Right'], ['A', 'Right', 'B'],
         ['Left'], ['A'], ['B'], ['Left', 'B']]
VOCAB_SETS = [set(x) for x in VOCAB]
LOG_EDGES = np.linspace(np.log10(.003), np.log10(180), 91)
LOG_CENTERS = (LOG_EDGES[:-1] + LOG_EDGES[1:]) / 2
WIDTH = np.diff(LOG_EDGES)[0]
AGE_EDGES = np.r_[0., np.geomspace(.01, 10, 16), np.inf]
POPS = ('overall', 'noop_to_active', 'active_to_other', 'active_to_noop')


def normalize_game(game):
    key = re.sub(r'[^a-z0-9]', '', str(game).lower())
    aliases = {'si': 'spaceinvaders', 'supermario': 'mario', 'supermariobros': 'mario'}
    key = aliases.get(key, key)
    if key not in NAMES:
        raise ValueError(f'Unsupported human behavior game: {game}')
    return key


def _subject(subject):
    return int(re.sub(r'[^0-9]', '', str(subject)))


def discover(subject=None, game=None, behavior_root=None):
    """Discover original files directly; never inspect model-cache manifests.

    behavior_root may be the repository root or a game's behavioral_data_* root.
    With no filters the currently available Atari and Mario sources are returned.
    """
    base = Path(behavior_root) if behavior_root else REPO
    atari_root = base / 'behavioral_data_block' if (base / 'behavioral_data_block').is_dir() else base
    mario_root = base / 'behavioral_data_mario' if (base / 'behavioral_data_mario').is_dir() else base
    selected = normalize_game(game) if game is not None else None
    sub = _subject(subject) if subject is not None else None
    tasks = []
    if selected != 'mario':
        for p in sorted(atari_root.glob('sub-*/ses-*/*.npz')):
            m = re.fullmatch(r'sub(\d+)-ses(\d+)-block(\d+)-game(\d+)\.npz', p.name)
            if not m or int(m[4]) not in range(3):
                continue
            g = ('enduro', 'pong', 'spaceinvaders')[int(m[4])]
            if (selected is not None and selected != g) or (sub is not None and sub != int(m[1])):
                continue
            tasks.append(dict(path=str(p), game=g, subject=int(m[1]), session=int(m[2]),
                              run_label=p.stem, stage='', bk2_path=''))
    if selected in (None, 'mario'):
        manifest = mario_root / 'manifest_canonical.tsv'
        if manifest.exists():
            rows = pd.read_csv(manifest, sep='\t')
            if rows.output_path.duplicated().any():
                raise ValueError('Duplicate Mario source in canonical behavioral manifest')
            for r in rows.to_dict('records'):
                s = _subject(r['sub'])
                if sub is not None and s != sub:
                    continue
                p = Path(r['output_path'])
                if not p.exists():
                    p = mario_root / f'sub-{s:03d}' / f"ses-{_subject(r['ses']):02d}" / p.name
                bk = Path(r['bk2_path'])
                if not bk.exists():
                    # The manifest remains portable when its source dataset is moved.
                    suffix = str(bk).split('cneuromod.mario/', 1)[-1]
                    candidate = mario_root / 'source_dataset' / suffix
                    if candidate.exists():
                        bk = candidate
                tasks.append(dict(path=str(p), game='mario', subject=s, session=_subject(r['ses']),
                                  run_label=p.stem, stage=f"w{r['world']}l{r['level']}", bk2_path=str(bk)))
    for i, task in enumerate(tasks):
        task['file_id'] = i
    if (subject is not None or game is not None) and not tasks:
        raise FileNotFoundError(f'No original behavioral sources: subject={subject}, game={game}, root={base}')
    paths = [str(Path(t['path']).resolve()) for t in tasks]
    if len(paths) != len(set(paths)):
        raise ValueError('One original human recording was discovered more than once')
    return tasks


def action_name(game, action, representation='exact'):
    if game != 'mario':
        return ATARI[game][int(action)]
    buttons = (b for i, b in enumerate(BUTTONS) if int(action) & (1 << i)) if representation == 'exact' else VOCAB[int(action)]
    return '+'.join(buttons) or 'NOOP'


def _canon(keys):
    if keys in VOCAB_SETS:
        return VOCAB_SETS.index(keys)
    reduced = keys - {'Up', 'Down', 'Select', 'Start'}
    return min(range(len(VOCAB)), key=lambda i: len(reduced ^ VOCAB_SETS[i]))


def _bk2(path, n):
    with zipfile.ZipFile(path) as z:
        payload = z.read('Input Log.txt')
    order, patterns, active = BUTTONS, [], False
    for line in payload.decode('utf-8').splitlines():
        line = line.strip()
        if line == '[Input]':
            active = True
        elif line == '[/Input]':
            break
        elif active and line.startswith('P1 '):
            order = [v.strip().split(' ', 1)[1] for v in line.split('|') if v.strip()]
        elif active and line.startswith('|'):
            patterns.append(line.split('|')[2])
    selected = patterns[1:1+n]  # Behavioral converter skips the saved initial state.
    if len(selected) != n or set(order) != set(BUTTONS):
        raise ValueError(f'BK2 alignment or controller header mismatch: {path}')
    lookup = {}
    for p in set(selected):
        if len(p) != len(order):
            raise ValueError(f'Invalid BK2 controller pattern: {p!r}')
        keys = {b for c, b in zip(p, order) if c != '.'}
        lookup[p] = (sum(1 << BUTTONS.index(b) for b in keys), _canon(keys), keys not in VOCAB_SETS)
    raw = np.array([lookup[p][0] for p in selected], dtype=np.int16)
    mapped = np.array([lookup[p][1] for p in selected], dtype=np.int16)
    return raw, mapped, sum(lookup[p][2] for p in selected), hashlib.sha256(payload).hexdigest()


def _bouts(a, t, boundary, meta, representation):
    onset = boundary.copy()
    onset[1:] |= a[1:] != a[:-1]
    starts = np.flatnonzero(onset)
    ends = np.r_[starts[1:], len(a)]
    right = np.ones(len(starts), bool)
    valid = ends < len(a)
    right[valid] = boundary[ends[valid]]
    left = boundary[starts]
    duration = t[np.where(right, ends-1, ends)] - t[starts]
    previous, following = np.full(len(starts), -1), np.full(len(starts), -1)
    previous[~left] = a[starts[~left]-1]
    following[~right] = a[ends[~right]]
    d = pd.DataFrame(dict(start=starts, end=ends, action=a[starts], previous_action=previous,
                          next_action=following, n_samples=ends-starts, duration_s=duration,
                          left_censored=left, right_censored=right,
                          segment=np.cumsum(boundary)[starts], representation=representation))
    if d.n_samples.sum() != len(a) or (duration < 0).any():
        raise ValueError('Human bout partition failed')
    for key, value in meta.items():
        d[key] = value
    return d


def _source(task, sample_mode='all'):
    with np.load(task['path'], allow_pickle=False) as z:
        action = z['action']
        t = z['time'].reshape(-1).astype(float)
        first = z['is_first'].reshape(-1).astype(bool)
        terminal = z['is_terminal'].reshape(-1).astype(bool)
    n = len(t)
    if not n or not (len(action) == n == len(first) == len(terminal)) or not np.isfinite(t).all():
        raise ValueError(f'Invalid human timestamps: {task["path"]}')
    if action.ndim != 2 or not np.isfinite(action).all() or not ((action == 0) | (action == 1)).all() or not (action.sum(1) == 1).all():
        raise ValueError(f'Human actions must be one-hot: {task["path"]}')
    canonical = action.argmax(1).astype(np.int16)
    dt = np.diff(t)
    positive = dt[dt > 0]
    median_dt = float(np.median(positive)) if len(positive) else np.nan
    gap = max(.25, 10*median_dt) if len(positive) else .25
    boundary = first.copy()
    boundary[0] = True
    boundary[1:] |= terminal[:-1] | (dt <= 0) | (dt > gap)
    elapsed = np.r_[np.where(~boundary[1:], dt, 0.), 0.]
    h = hashlib.sha256()
    for arr in [action, t, first, terminal]:
        h.update(str((arr.shape, arr.dtype)).encode())
        h.update(arr.tobytes())
    audit = dict(task, n_frames=n, n_segments=int(boundary.sum()), n_first=int(first.sum()),
                 n_terminal=int(terminal.sum()), n_clock_reversals=int((dt <= 0).sum()),
                 n_large_gaps=int((dt > gap).sum()), dt_median=median_dt, gap_limit_s=gap,
                 observed_seconds=float(elapsed.sum()), behavior_fields_sha256=h.hexdigest(),
                 switches_canonical=int(((canonical[1:] != canonical[:-1]) & ~boundary[1:]).sum()))
    meta = {k: task[k] for k in ['file_id', 'game', 'subject', 'session', 'run_label', 'stage']}
    representations = [('exact', canonical)]
    if task['game'] == 'mario':
        raw, mapped, nonvocab, digest = _bk2(task['bk2_path'], n)
        if not np.array_equal(canonical, mapped):
            raise ValueError(f'Mario exact inputs do not match canonical recording: {task["path"]}')
        representations = [('exact', raw), ('canonical9', canonical)]
        audit.update(nonvocab_frames=nonvocab, mapped_mismatch_frames=0, bk2_input_sha256=digest)
    elif action.shape[1] != len(ATARI[task['game']]):
        raise ValueError(f'Atari action vocabulary mismatch: {task["path"]}')
    original_n = n
    source_frames = np.arange(n)
    if sample_mode == '15hz':
        from common.sampling import causal_grid
        segments = np.cumsum(boundary)
        selected = []
        for segment in np.unique(segments):
            ids = np.flatnonzero(segments == segment)
            _, rows = causal_grid(t[ids])
            selected.extend(ids[np.unique(rows)])
        source_frames = np.asarray(selected, int)
        representations = [(rep, a[source_frames]) for rep,a in representations]
        t, canonical = t[source_frames], canonical[source_frames]
        boundary = np.r_[True, np.diff(segments[source_frames]) != 0]
        n = len(t)
        elapsed = np.r_[np.where(~boundary[1:], np.diff(t), 0.), 0.]
        audit.update(n_frames=n, n_segments=int(boundary.sum()), observed_seconds=float(elapsed.sum()),
            switches_canonical=int(((canonical[1:] != canonical[:-1]) & ~boundary[1:]).sum()))
    elif sample_mode != 'all':
        raise ValueError('sample_mode must be all or 15hz')
    audit.update(sample_mode=sample_mode, n_original_frames=original_n)
    exact = representations[0][1]
    audit['switches_exact'] = int(((exact[1:] != exact[:-1]) & ~boundary[1:]).sum())
    bouts = pd.concat([_bouts(a, t, boundary, meta, rep) for rep, a in representations], ignore_index=True)
    onset = boundary.copy()
    onset[1:] |= exact[1:] != exact[:-1]
    starts = np.maximum.accumulate(np.where(onset, np.arange(n), 0))
    before = np.r_[0, starts[:-1]]
    age = t-t[before]
    age[boundary] = np.nan
    previous = np.r_[-1, exact[:-1]]
    previous[boundary] = -1
    observations = pd.DataFrame(dict(frame=source_frames, time_seconds=t, human_action=exact,
        incumbent_action=previous, canonical_action=canonical, boundary=boundary,
        segment=np.cumsum(boundary), age_seconds=age, elapsed_seconds=elapsed,
        unknown_bout_onset=boundary[before]))
    for key, value in meta.items():
        observations[key] = value
    return bouts, observations, audit


def _risk(d, population):
    d = d.loc[d.incumbent_action.ge(0)].copy()
    if population == 'noop_to_active':
        d = d.loc[d.incumbent_action.eq(0)].copy()
    elif population in ('active_to_noop', 'active_to_other'):
        d = d.loc[d.incumbent_action.ne(0)].copy()
    d['outcome'] = (d.human_action.eq(0) if population == 'active_to_noop'
                    else d.human_action.ne(d.incumbent_action)).astype(int)
    return d


def _density(values, smooth=.1):
    values = np.asarray(values)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return np.full(len(LOG_CENTERS), np.nan)
    y = np.histogram(np.log10(values), LOG_EDGES)[0] / len(values) / WIDTH
    return gaussian_filter1d(y, smooth/WIDTH) if smooth else y


def _survival(b, grid):
    if not len(b):
        return pd.DataFrame()
    # Aalen--Johansen, accounting for right censoring and tied event times.
    times, inverse = np.unique(b.duration_s, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(times))
    risk = np.cumsum(counts[::-1])[::-1]
    dn = np.bincount(inverse, weights=(~b.right_censored & b.next_action.eq(0)), minlength=len(times))
    da = np.bincount(inverse, weights=(~b.right_censored & b.next_action.ne(0)), minlength=len(times))
    survival_after = np.cumprod(1-(dn+da)/risk)
    prior = np.r_[1., survival_after[:-1]]
    noop, active = np.cumsum(prior*dn/risk), np.cumsum(prior*da/risk)
    ix = np.searchsorted(times, grid, side='right')
    at_risk = len(b)-np.r_[0, np.cumsum(counts)][np.searchsorted(times, grid, side='left')]
    out = pd.DataFrame(dict(seconds=grid, survival=np.r_[1., survival_after][ix],
                           to_noop=np.r_[0., noop][ix], to_active=np.r_[0., active][ix], at_risk=at_risk))
    out.loc[out.at_risk.lt(5), ['survival', 'to_noop', 'to_active']] = np.nan
    return out


def _duration_tables(d):
    curves, scores, qa = [], [], []
    for pop in POPS:
        r = _risk(d, pop).dropna(subset=['age_seconds']).reset_index(drop=True)
        if not len(r):
            continue
        r['bin'] = np.searchsorted(AGE_EDGES, r.age_seconds, side='right')-1
        grouped = r.groupby(['run_label', 'session', 'bin']).agg(
            age_seconds=('age_seconds', 'mean'), probability=('outcome', 'mean'), n=('outcome', 'size')).reset_index()
        grouped['population'] = pop
        grouped['lower_seconds'] = AGE_EDGES[grouped.bin]
        grouped['upper_seconds'] = AGE_EDGES[grouped.bin+1]
        curves.append(grouped)
        age = np.log(r.age_seconds.clip(lower=.001)).to_numpy()
        # Fixed categorical vocabulary; fit scaling and coefficients on training sessions only.
        acts = [a for a in sorted(r.incumbent_action.unique()) if a != 0]
        x_action = np.column_stack([(r.incumbent_action.to_numpy() == a).astype(float) for a in acts]) if acts else np.zeros((len(r), 0))
        designs = {'previous_action': x_action,
                   'action_and_duration': np.column_stack([x_action, age, x_action*age[:, None]])}
        y = r.outcome.to_numpy(int)
        sessions = sorted(r.session.unique())
        for held in sessions:
            train, test = r.session.ne(held).to_numpy(), r.session.eq(held).to_numpy()
            for name, x in designs.items():
                audit = dict(population=pop, session=int(held), model=name, n_train=int(train.sum()), n_test=int(test.sum()))
                if train.sum()<100 or not test.any():
                    qa.append(dict(audit, status='insufficient_training_or_test')); continue
                if len(np.unique(y[train]))<2:
                    qa.append(dict(audit, status='single_training_class')); continue
                sd = x[train].std(0)
                valid = sd > 1e-10
                if not valid.any():
                    p = np.full(test.sum(), y[train].mean())
                    audit.update(status='ok', estimator='training-only intercept', n_features=0)
                else:
                    xx = (x[:, valid]-x[train][:, valid].mean(0))/sd[valid]
                    estimator = LogisticRegression(C=1., solver='newton-cholesky', max_iter=200, tol=1e-6)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        estimator.fit(xx[train], y[train])
                    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                        qa.append(dict(audit, status='nonconverged')); continue
                    p = estimator.predict_proba(xx[test])[:, 1]
                    audit.update(status='ok', estimator='logistic', n_features=int(valid.sum()), iterations=int(estimator.n_iter_.max()))
                qa.append(audit)
                selected = r.loc[test, ['run_label', 'outcome']].reset_index(drop=True)
                for run, indexes in selected.groupby('run_label').groups.items():
                    truth, pred = selected.loc[indexes, 'outcome'].to_numpy(), p[indexes]
                    prob_truth = np.where(truth == 1, pred, 1-pred)
                    scores.append(dict(run_label=run, population=pop, session=int(held), model=name,
                        n=len(truth), log_score=float(np.log(prob_truth.clip(1e-12)).mean()),
                        accuracy=float(((pred>.5)==truth).mean()), brier=float(np.square(pred-truth).mean()),
                        auroc=float(roc_auc_score(truth,pred)) if len(np.unique(truth))==2 else np.nan))
    return {'duration_curves': pd.concat(curves,ignore_index=True) if curves else pd.DataFrame(),
            'duration_prediction': pd.DataFrame(scores), 'duration_fit_qa': pd.DataFrame(qa)}


def _summary_tables(b, obs, audit, game):
    completed = b.loc[~b.left_censored & ~b.right_censored & b.duration_s.gt(0)]
    rows, densities, peaks, sensitivity, survival, transitions = [], [], [], [], [], []
    for (rep, action), d in completed.groupby(['representation', 'action']):
        quant = d.duration_s.quantile([.1,.25,.5,.75,.9,.95])
        rows.append(dict(representation=rep,action=int(action),action_name=action_name(game,action,rep),
            n_bouts=len(d), mean_s=d.duration_s.mean(), median_s=quant.loc[.5],q10_s=quant.loc[.1],
            q25_s=quant.loc[.25],q75_s=quant.loc[.75],q90_s=quant.loc[.9],q95_s=quant.loc[.95],
            fraction_over_1s=d.duration_s.gt(1).mean(),fraction_over_2s=d.duration_s.gt(2).mean()))
    for rep, rd in completed.groupby('representation'):
        subsets = [('all', rd), ('noop', rd[rd.action.eq(0)]), ('active', rd[rd.action.ne(0)])]
        subsets += [(f'action_{int(a)}', g) for a,g in rd.groupby('action')]
        for population,d in subsets:
            raw, smooth = _density(d.duration_s,0), _density(d.duration_s)
            for x, y, z in zip(LOG_CENTERS,raw,smooth):
                densities.append(dict(representation=rep,population=population,log10_seconds=x,
                    seconds=10**x,density=y,smoothed_density=z,n_bouts=len(d)))
            if population.startswith('action_') and len(d):
                for width in [.08,.10,.15]:
                    sm = _density(d.duration_s,width)
                    loc,_ = find_peaks(sm,prominence=float(np.max(sm))*.12,distance=max(1,int(.3/WIDTH)))
                    peaks.append(dict(representation=rep,population=population,bandwidth_dex=width,
                        n_bouts=len(d),n_visible_peaks=len(loc),peak_seconds=';'.join(f'{v:.4g}' for v in 10**LOG_CENTERS[loc])))
    for rep, rd in b.groupby('representation'):
        for name,sd in [('observed_onset_and_exit',rd[~rd.left_censored & ~rd.right_censored]),
                        ('include_boundary_starts',rd[~rd.right_censored]),('all_lower_bounds',rd)]:
            for x,y in zip(LOG_CENTERS,_density(sd.duration_s)):
                sensitivity.append(dict(representation=rep,cohort=name,seconds=10**x,density=y))
        sw = rd.loc[~rd.right_censored]
        for (source,target),z in sw.groupby(['action','next_action']):
            transitions.append(dict(representation=rep,source=int(source),target=int(target),
                source_name=action_name(game,source,rep),target_name=action_name(game,target,rep),count=len(z)))
    tr=pd.DataFrame(transitions)
    if len(tr):
        tr['conditional_probability']=tr['count']/tr.groupby(['representation','source'])['count'].transform('sum')
        def keys(name):
            return set() if name=='NOOP' else ({v for v in ('RIGHT','LEFT','DOWN','FIRE') if v in name} if game!='mario' else set(name.split('+')))
        tr['button_changes']=[len(keys(a)^keys(b)) for a,b in zip(tr.source_name,tr.target_name)]
        tr['most_frequent_destination']=tr['count'].eq(tr.groupby(['representation','source'])['count'].transform('max'))
    exact=b[b.representation.eq('exact')]
    for (run,active),rd in exact.assign(active=exact.action.ne(0).astype(int)).groupby(['run_label','active']):
        for name,z in [('observed_onset',rd[~rd.left_censored]),('include_boundary_starts',rd)]:
            curve=_survival(z,np.linspace(0,5,101))
            if len(curve):
                survival.append(curve.assign(run_label=run,active=active,cohort=name))
    distribution=[]
    for representation,col in [('exact','human_action'),('canonical9','canonical_action')]:
        if game!='mario' and representation!='exact':
            continue
        for a,d in obs.groupby(col):
            distribution.append(dict(representation=representation,action=int(a),action_name=action_name(game,a,representation),
                n_observations=len(d),observation_fraction=len(d)/len(obs),
                observed_seconds=d.elapsed_seconds.sum(),time_fraction=d.elapsed_seconds.sum()/obs.elapsed_seconds.sum() if obs.elapsed_seconds.sum()>0 else np.nan))
    strict=exact.loc[~exact.left_censored & ~exact.right_censored & exact.duration_s.gt(0)]
    return_rows=[]
    for action,d in strict.groupby('action'):
        expected=0.
        for _,r in d.groupby('file_id'):
            prev=r.previous_action.value_counts();nxt=r.next_action.value_counts()
            expected += (prev*nxt.reindex(prev.index,fill_value=0)).sum()/len(r)
        return_rows.append(dict(action=int(action),action_name=action_name(game,action),n_bouts=len(d),
            return_fraction=d.previous_action.eq(d.next_action).mean(),conditional_reference=expected/len(d)))
    returns=pd.DataFrame(return_rows)
    if len(returns):
        returns['excess']=returns.return_fraction-returns.conditional_reference
    stage=completed.groupby(['representation','stage','action']).agg(n_bouts=('duration_s','size'),
        median_seconds=('duration_s','median'),q90_seconds=('duration_s',lambda x:x.quantile(.9))).reset_index()
    switched=exact.loc[~exact.right_censored]
    behavior=pd.DataFrame([dict(n_observations=len(obs),n_files=len(audit),n_bouts=len(exact),
        n_complete_bouts=len(strict),observed_seconds=float(obs.elapsed_seconds.sum()),
        exact_switches=len(switched),switches_per_second=len(switched)/obs.elapsed_seconds.sum() if obs.elapsed_seconds.sum()>0 else np.nan,
        fraction_one_sample=float(strict.n_samples.eq(1).mean()),fraction_under_100ms=float(strict.duration_s.lt(.1).mean()),
        noop_bout_fraction=float(strict.action.eq(0).mean()),
        switch_involving_noop_fraction=float((switched.action.eq(0)|switched.next_action.eq(0)).mean()),
        single_button_change_fraction=float(tr.loc[tr.representation.eq('exact')&tr.button_changes.eq(1),'count'].sum()/len(switched)) if len(switched) else np.nan,
        return_fraction=float(strict.previous_action.eq(strict.next_action).mean()),
        conditional_return_reference=float(np.average(returns.conditional_reference,weights=returns.n_bouts)) if len(returns) else np.nan)])
    native=[]
    for (rep,a,n),d in b.loc[~b.right_censored].groupby(['representation','action','n_samples']):
        native.append(dict(representation=rep,action=int(a),action_name=action_name(game,a,rep),n_samples=int(n),count=len(d)))
    result=dict(action_summary=pd.DataFrame(rows),action_distribution=pd.DataFrame(distribution),
        duration_density=pd.DataFrame(densities),density_peaks=pd.DataFrame(peaks),
        boundary_sensitivity=pd.DataFrame(sensitivity),transitions=tr,
        persistence_curves=pd.concat(survival,ignore_index=True) if survival else pd.DataFrame(),
        return_patterns=returns,stage_summary=stage,behavior_summary=behavior,native_sample_counts=pd.DataFrame(native))
    result.update(_duration_tables(obs))
    return result


def analyze(subject, game, behavior_root=None, sample_mode='all'):
    """Return reproducible human-only tables plus QA without writing artifacts."""
    game, subject = normalize_game(game), _subject(subject)
    sources=discover(subject,game,behavior_root)
    bout_frames,observation_frames,audits=[],[],[]
    for task in sources:
        bouts,observations,audit=_source(task,sample_mode=sample_mode)
        bout_frames.append(bouts);observation_frames.append(observations);audits.append(audit)
    b=pd.concat(bout_frames,ignore_index=True)
    obs=pd.concat(observation_frames,ignore_index=True)
    audit=pd.DataFrame(audits)
    tables={'observations':obs,'bouts':b,'source_alignment':audit}
    tables.update(_summary_tables(b,obs,audit,game))
    checks=[]
    for rep,d in b.groupby('representation'):
        expected=int(audit['switches_exact' if rep=='exact' else 'switches_canonical'].sum())
        actual=int((~d.right_censored).sum())
        if actual!=expected or int(d.n_samples.sum())!=len(obs):
            raise ValueError(f'Behavioral partition/switch mismatch: {rep}')
        checks.append(dict(representation=rep,switches=actual,partition_observations=int(d.n_samples.sum())))
    for frame in tables.values():
        frame['subject']=subject;frame['game']=game
    qa=dict(status='completed',subject=subject,game=game,n_files=len(sources),n_observations=len(obs),
        uses_replay_sampling=False,model_dependencies=False,image_arrays_read=False,
        n_original_observations=int(audit.n_original_frames.sum()),duplicated_checkpoint_copies=0,switch_checks=checks,
        sample_mode=sample_mode,
        cohort=f'{sample_mode} observations; exact actions primary; complete-bout distributions exclude both boundary ends',
        boundaries='file, is_first, previous terminal, nonincreasing clock, gap > max(0.25 s, 10 median dt)',
        duration_prediction='leave-session-out; predictor scaling fit on training sessions; probabilities per native observation',
        survival='COSYNE boundary-start-inclusive panel preserved; observed-onset sensitivity also shown; right-censoring respected',
        group_averaging='equal runs then participants for survival/duration; equal participants for normalized action/bout distributions',
        peak_caution='Descriptive density peaks, not evidence for two latent control regimes',
        return_caution='A-B-A descriptive excess conditional on file/current action; mechanics and screen state uncontrolled')
    return tables,qa


def _subject_curves(d,x,y,keys=()):
    # A source run gets one vote within a participant; participants get equal votes.
    group=list(keys)+['subject',x]
    return d.groupby(group,dropna=False)[y].mean().reset_index()


def render(root,tables,title):
    """Render title-free paper figures without changing other analyses' style.

    ``title`` remains accepted for the shared CLI, but is deliberately not drawn.
    """
    import matplotlib
    matplotlib.use('Agg')
    with matplotlib.rc_context({
        'font.size': 22,
        'axes.labelsize': 26,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
        'legend.title_fontsize': 22,
        'axes.labelpad': 12,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.2,
        'lines.markersize': 7,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }):
        return _render_figures(root,tables)


def _render_figures(root,tables):
    """Render participant or group panels using stored tables only; no source IO."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    def save(fig,name):
        fig.tight_layout(pad=1.2,w_pad=2.,h_pad=2.)
        for suffix in ('png','pdf'):
            path=out/f'{name}.{suffix}';fig.savefig(path,dpi=300,bbox_inches='tight');artifacts.append(str(path))
        plt.close(fig)
    def curve(ax,d,x,y,label,**kwargs):
        if not len(d):return
        s=_subject_curves(d,x,y)
        for _,r in s.groupby('subject'):
            if s.subject.nunique()>1:ax.plot(r[x],r[y],alpha=.18,lw=.7,color=kwargs.get('color','#5c7890'))
        z=s.groupby(x)[y].agg(['mean','sem']).reset_index()
        ax.plot(z[x],z['mean'],label=label,**kwargs)
        if s.subject.nunique()>1:ax.fill_between(z[x],z['mean']-z['sem'],z['mean']+z['sem'],alpha=.15,color=kwargs.get('color'))
    def style_time(ax):
        ax.set_xscale('log');ax.set(xlabel='Unchanged action duration (s)',ylabel='Density per log10(s)',xlim=(.005,100));ax.grid(alpha=.15)
    if len(tables.get('action_distribution',[])):
        d=tables['action_distribution'];d=d.loc[d.representation.eq('exact')]
        # Absent actions receive zero probability for participants with recordings.
        subjects=sorted(d.subject.unique());actions=sorted(d.action.unique())
        index=pd.MultiIndex.from_product([subjects,actions],names=['subject','action'])
        values=d.set_index(['subject','action'])[['observation_fraction','time_fraction']].reindex(index,fill_value=0).reset_index()
        labels=d.drop_duplicates('action').set_index('action').action_name
        z=values.groupby('action')[['observation_fraction','time_fraction']].mean()
        fig,ax=plt.subplots(figsize=(max(10,len(z)*.9),6.5));x=np.arange(len(z));w=.38
        ax.bar(x-w/2,z.observation_fraction,w,label='Recorded observations');ax.bar(x+w/2,z.time_fraction,w,label='Recorded time')
        ax.set(xticks=x,xticklabels=[labels.loc[a] for a in z.index],ylabel='Fraction\n(equal participant weight)')
        ax.tick_params(axis='x',rotation=45);ax.legend(frameon=False);save(fig,'action_distribution')
    d=tables.get('duration_density',pd.DataFrame())
    if len(d):
        exact=d.loc[d.representation.eq('exact')]
        fig,ax=plt.subplots(figsize=(10,6.5))
        for pop,label,c in [('all','All exact actions','#566573'),('noop','NOOP','#2e86c1'),('active','Active inputs','#d98d25')]:
            curve(ax,exact[exact.population.eq(pop)],'seconds','smoothed_density',label,color=c)
        style_time(ax);ax.legend(frameon=False);save(fig,'bout_duration_noop_active')
        actions=sorted(exact.loc[exact.population.str.startswith('action_'),'population'].unique(),key=lambda v:int(v.split('_')[1]))
        if actions:
            fig,axes=plt.subplots((len(actions)+2)//3,3,figsize=(21,6*((len(actions)+2)//3)),squeeze=False)
            names=tables['action_summary'].loc[lambda x:x.representation.eq('exact')].drop_duplicates('action').set_index('action').action_name
            for pop,ax in zip(actions,axes.flat):
                r=exact[exact.population.eq(pop)];curve(ax,r,'seconds','density','_nolegend_',color='#a5b3bd');curve(ax,r,'seconds','smoothed_density',names.get(int(pop.split('_')[1]),pop),color='#176b98')
                style_time(ax);ax.legend(frameon=False)
            for ax in axes.flat[len(actions):]:ax.axis('off')
            save(fig,'duration_by_action')
    d=tables.get('transitions',pd.DataFrame())
    if len(d):
        d=d[d.representation.eq('exact')];order=sorted(set(d.source)|set(d.target))
        mat=np.full((len(order),len(order)),np.nan)
        for i,a in enumerate(order):
            eligible=d.loc[d.source.eq(a),'subject'].unique()
            for j,b in enumerate(order):
                if a==b or not len(eligible):continue
                p=d.loc[d.source.eq(a)&d.target.eq(b)].set_index('subject').conditional_probability
                mat[i,j]=p.reindex(eligible,fill_value=0).mean()
        name_map=dict(zip(d.source,d.source_name));name_map.update(zip(d.target,d.target_name))
        fig,ax=plt.subplots(figsize=(max(10,len(order)*.9),max(8.5,len(order)*.8)))
        im=ax.imshow(mat,vmin=0,vmax=1,cmap='Blues');names=[name_map[a] for a in order]
        ax.set(xticks=range(len(order)),xticklabels=names,yticks=range(len(order)),yticklabels=names,xlabel='Next exact action',ylabel='Current exact action')
        ax.tick_params(axis='x',rotation=70);fig.colorbar(im,ax=ax,label='P(next action | switch)');save(fig,'switch_destinations')
    d=tables.get('boundary_sensitivity',pd.DataFrame())
    if len(d):
        fig,ax=plt.subplots(figsize=(12,8))
        for (rep,cohort),r in d.groupby(['representation','cohort']):curve(ax,r,'seconds','density',f'{rep}: {cohort.replace("_"," ")}')
        style_time(ax);ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.02));save(fig,'boundary_vocabulary_sensitivity')
    d=tables.get('native_sample_counts',pd.DataFrame())
    if len(d):
        d=d.loc[d.representation.eq('exact')&d.action.eq(0)].copy()
        if len(d):
            edges=np.geomspace(.95,5000,65);records=[]
            for sub,r in d.groupby('subject'):
                counts=np.histogram(r.n_samples,bins=edges,weights=r['count'])[0]
                records.append(counts/counts.sum() if counts.sum() else counts)
            fig,ax=plt.subplots(figsize=(10,6.5));ax.stairs(np.mean(records,axis=0),edges,fill=True,alpha=.6)
            ax.set(xscale='log',xlabel='Native recorded samples in NOOP bout',ylabel='Bout fraction\n(equal participant weight)')
            save(fig,'noop_native_samples')
    d=tables.get('persistence_curves',pd.DataFrame())
    if len(d):
        for cohort,suffix in [('include_boundary_starts',''),('observed_onset','_observed_onset')]:
            part=d.loc[d.cohort.eq(cohort)]
            fig,axes=plt.subplots(1,2,figsize=(17,6.5))
            for active,label,c in [(0,'No input','#2e86c1'),(1,'Active input','#d98d25')]:curve(axes[0],part[part.active.eq(active)],'seconds','survival',label,color=c)
            for column,label,c in [('to_noop','Release input','#2e86c1'),('to_active','Different active action','#d98d25')]:curve(axes[1],part[part.active.eq(1)],'seconds',column,label,color=c)
            for ax in axes:ax.set(xlabel='Elapsed bout duration (s)',ylim=(0,1));ax.legend(frameon=False)
            axes[0].set(ylabel='Survival probability')
            axes[1].set(ylabel='Cumulative incidence')
            save(fig,'cosyne_fig2A_action_persistence'+suffix)
    d=tables.get('duration_prediction',pd.DataFrame())
    if len(d):
        fig,axes=plt.subplots(1,3,figsize=(21,7))
        labels={'overall':'Any action switch','noop_to_active':'NOOP → active','active_to_noop':'Active → NOOP'}
        for ax,pop in zip(axes,['overall','noop_to_active','active_to_noop']):
            z=d[d.population.eq(pop)].groupby(['subject','model']).log_score.mean().unstack('model')
            columns=['previous_action','action_and_duration']
            if all(c in z for c in columns):
                for i,(_,r) in enumerate(z.iterrows()):ax.plot([0,1],r[columns],'-o',alpha=.6,color='#5c7890',label=labels[pop] if i==0 else '_nolegend_')
                ax.plot([0,1],z[columns].mean(),'k_',ms=14)
                ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.02))
            ax.set(xticks=[0,1],xticklabels=['Previous\naction','+ Duration'],ylabel='Held-out log score\n(nat / observation)')
        save(fig,'cosyne_fig2B_duration_prediction')
    d=tables.get('duration_curves',pd.DataFrame())
    if len(d):
        d=d.copy();d.loc[d.n.lt(20),'probability']=np.nan
        d['seconds']=np.sqrt(d.lower_seconds.clip(lower=.0025)*d.upper_seconds.replace(np.inf,20))
        fig,axes=plt.subplots(1,3,figsize=(21,7))
        labels={'noop_to_active':'NOOP → active','active_to_other':'Active → any other action','active_to_noop':'Active → NOOP'}
        for ax,pop in zip(axes,['noop_to_active','active_to_other','active_to_noop']):
            curve(ax,d[d.population.eq(pop)],'seconds','probability',labels[pop],color='#176b98')
            ax.set(xscale='log',xlabel='Elapsed action duration (s)',ylabel='Probability per\nrecorded observation')
            ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.02))
        save(fig,'cosyne_fig2C_duration_switch_probability')
    d=tables.get('return_patterns',pd.DataFrame())
    if len(d):
        z=d.groupby('action_name')[['return_fraction','conditional_reference']].mean()
        fig,ax=plt.subplots(figsize=(max(10,len(z)*.9),7));x=np.arange(len(z));w=.38
        ax.bar(x-w/2,z.return_fraction,w,label='Observed A → B → A');ax.bar(x+w/2,z.conditional_reference,w,label='Conditional independence reference')
        ax.set(xticks=x,xticklabels=z.index,ylabel='Return fraction',xlabel='Middle action B');ax.tick_params(axis='x',rotation=50);ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.02))
        save(fig,'descriptive_return_patterns')
    return artifacts
