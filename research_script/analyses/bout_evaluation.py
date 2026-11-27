#!/usr/bin/env python3
"""Supplement to theme13: prospective within-bout evaluation and matched boundaries.

Fix an evaluation direction near the START of each raw human action bout, then
track it without resetting at sliding-window boundaries. Separately predict
next-observation human switching and REAL-policy change. Event-aligned matched
plots are descriptive, not the prospective test. No human goal labels, training,
or score evaluation. Long-bout eligibility and NOOP dominance are reported.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

RESEARCH = Path(__file__).resolve().parents[1]
REPO = RESEARCH.parent
ACTION_NAMES = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
VERSION = 'bout-direction-v1'


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def build_risk_set(d, actor, args, parent):
    """Only the current bout's observed prefix enters each prospective row."""
    from common.sampling import sequence_index
    ts = d.time.to_numpy(); dec = sequence_index(d).to_numpy(); ep = d.episode.to_numpy()
    bouts, risk = [], []
    coverage = dict(raw_observed_bouts=0, left_censored=0, too_short=0,
                    gap_truncated=0, no_valid_future_interval=0)
    for (episode, raw_bout), rows in d.groupby(['episode', 'raw_bout'], sort=False):
        coverage['raw_observed_bouts'] += 1
        ix = rows.index.to_numpy()
        if not np.all(np.diff(ix) == 1): raise ValueError('Noncontiguous episode/bout rows')
        if rows.age.iloc[0] > args.onset_tolerance:
            coverage['left_censored'] += 1; continue
        bad = np.flatnonzero((np.diff(dec[ix]) != 1) | (np.diff(ts[ix]) <= 0)
                             | (np.diff(ts[ix]) > args.max_gap))
        if len(bad):
            ix = ix[:bad[0]+1]; coverage['gap_truncated'] += 1
        start = int(ix[0]); onset = ts[start]-float(d.age.iloc[start])
        pos = int(np.searchsorted(ts[ix], onset+args.anchor_seconds))
        if pos < 2 or pos+args.minimum_late_steps >= len(ix):
            coverage['too_short'] += 1; continue
        anchor = int(ix[pos])
        if ts[anchor]-(onset+args.anchor_seconds) > args.onset_tolerance:
            coverage['too_short'] += 1; continue
        bout_id = len(bouts); begin = len(risk)
        incumbent = int(d.human_action.iloc[start])
        for end in ix[pos+args.minimum_late_steps:]:
            future = int(end)+1
            if (future >= len(d) or ep[future] != episode or dec[future] != dec[end]+1
                    or not 0 < ts[future]-ts[end] <= args.max_gap):
                coverage['no_valid_future_interval'] += 1; continue
            switch_time = float(d.next_human_switch_time.iloc[end])
            if switch_time <= ts[end]: raise ValueError('Nonprospective switch timestamp')
            risk.append(dict(row_id=len(risk), bout_id=bout_id, episode=episode,
                run=d.run.iloc[start], session=int(d.session.iloc[start]), action=incumbent,
                onset=onset, start=start, anchor=anchor, end=int(end), future=future,
                time=ts[end], dt=ts[future]-ts[end], age=ts[end]-onset,
                since_anchor=ts[end]-ts[anchor], next_switch_time=switch_time,
                human_switch=float(switch_time <= ts[future]),
                policy_js=float(parent.js_distance(actor[end], actor[future])),
                policy_away=float(actor[end, incumbent]-actor[future, incumbent])))
        if len(risk) == begin:
            coverage['too_short'] += 1; continue
        bouts.append(dict(bout_id=bout_id, episode=episode, raw_bout=int(raw_bout),
            run=d.run.iloc[start], session=int(d.session.iloc[start]), action=incumbent,
            onset=onset, start=start, anchor=anchor, last=int(ix[-1]),
            next_switch_time=float(d.next_human_switch_time.iloc[start]),
            n_rows=len(risk)-begin))
    w = pd.DataFrame(risk); b = pd.DataFrame(bouts)
    coverage.update(eligible_bouts=len(b), risk_rows=len(w), events=int(w.human_switch.sum()) if len(w) else 0)
    return b, w, coverage


def cumulative_geometry(z, ts, anchor, eps=1e-8):
    """Past-only features; anchor index is fixed for the whole bout.

Progress is net projection, not a measured neural accumulator. Positive-time
fraction and current streak add path-history information beyond net progress.
"""
    early = z[anchor]-z[0]; strength = np.linalg.norm(early)
    direction = early/max(strength, eps)
    delta = np.diff(z[anchor:], axis=0); dt = np.diff(ts[anchor:])
    lengths = np.linalg.norm(delta, axis=1); projection = delta@direction
    positive = projection > eps
    streak = np.zeros(len(dt))
    for k in range(len(dt)):
        streak[k] = (streak[k-1] if k else 0)+dt[k] if positive[k] else 0
    path = np.cumsum(lengths); elapsed = np.cumsum(dt)
    progress = np.cumsum(projection)
    metrics = dict(progress=progress, coherence=progress/np.maximum(path, eps),
        positive_fraction=np.cumsum(dt*positive)/np.maximum(elapsed, eps), streak=streak,
        path=path, rms=np.sqrt(np.cumsum(lengths**2)/np.arange(1, len(lengths)+1)),
        max_increment=np.maximum.accumulate(lengths), last_increment=lengths,
        early_strength=np.full(len(dt), strength), defined=np.full(len(dt), float(strength > eps)))
    return metrics, direction


def coordinates(d, a, held, args, parent):
    train = np.flatnonzero(d.session.to_numpy() != held)
    root, _ = parent.fit_coordinates(a['root'], train, args.root_rank, args.seed+held)
    onehot = np.eye(a['actor_p'].shape[1])
    nuisance = np.column_stack([root, a['prior'], onehot[d.first_action.to_numpy(int)],
                                d[['root_value', 'reward_prior', 'carry']].to_numpy()])
    nuisance = StandardScaler().fit(nuisance[train]).transform(nuisance)
    ev = StandardScaler().fit(a['heads'][train]).transform(a['heads'])
    fit = Ridge(alpha=args.penalty*len(train)).fit(nuisance[train], ev[train])
    residual = ev-fit.predict(nuisance)
    z = StandardScaler().fit(residual[train]).transform(residual)
    context = np.column_stack([root, a['actor_p']])
    context = StandardScaler().fit(context[train]).transform(context)
    return root, z, context


def features(d, a, b, w, root, z):
    nact = a['actor_p'].shape[1]; onehot = np.eye(nact)
    history = np.column_stack([root, a['prior'], a['actor_p'],
        onehot[d.first_action.to_numpy(int)], d[['root_value', 'reward_prior', 'carry', 'budget']]])
    baseline = np.empty((len(w), history.shape[1]*4+nact+12))
    levels = np.empty((len(w), z.shape[1]*4))
    magnitude = np.empty((len(w), 5)); directed = np.empty((len(w), 4))
    for bout in b.itertuples():
        r = w.loc[w.bout_id.eq(bout.bout_id)]
        idx = np.arange(bout.start, bout.last+1); local = r.end.to_numpy()-bout.start
        anchor = bout.anchor-bout.start
        zz = z[idx]; h = history[idx]
        hm = np.cumsum(h, axis=0)/np.arange(1, len(h)+1)[:, None]
        zm = np.cumsum(zz, axis=0)/np.arange(1, len(zz)+1)[:, None]
        g, _ = cumulative_geometry(zz, d.time.to_numpy()[idx], anchor)
        k = local-anchor-1; rid = r.row_id.to_numpy()
        age = r.age.to_numpy(); logage = np.log1p(age)
        knots = np.log1p([.25, .5, 1., 2., 4., 8.])
        agebasis = np.maximum(0, logage[:, None]-knots)
        structure = np.column_stack([logage, agebasis, np.log(r.dt.to_numpy()),
            r.since_anchor, local+1, g['defined'][k], np.log1p(d.budget.to_numpy()[r.end])])
        baseline[rid] = np.column_stack([h[local], np.broadcast_to(h[anchor], h[local].shape),
            hm[local], h[local]-h[anchor], onehot[r.action.to_numpy()], structure])
        levels[rid] = np.column_stack([zz[local], np.broadcast_to(zz[anchor], zz[local].shape),
                                      zm[local], zz[local]-zz[local-1]])
        magnitude[rid] = np.column_stack([g[n][k] for n in ['path', 'rms', 'max_increment', 'last_increment', 'early_strength']])
        directed[rid] = np.column_stack([g[n][k] for n in ['progress', 'positive_fraction', 'streak', 'coherence']])
    level = np.column_stack([baseline, levels])
    nonlinear = np.column_stack([np.log1p(magnitude), magnitude**2])
    amplitude = np.column_stack([level, magnitude, nonlinear])
    designs = dict(baseline=baseline, level=level, amplitude=amplitude,
        progress=np.column_stack([amplitude, directed[:, 0]]),
        persistence=np.column_stack([amplitude, directed[:, :3]]))
    for x in designs.values():
        if not np.isfinite(x).all(): raise ValueError('Nonfinite design')
    return designs, directed


def episode_samples(d, episode_indices, episode, center, grid, max_gap):
    ix = episode_indices[episode]; ts = d.time.to_numpy()[ix]; target = center+grid
    if target[0] < ts[0] or target[-1] > ts[-1]: return None
    local = np.searchsorted(ts, target, side='right')-1
    span = slice(local[0], local[-1]+1)
    if (np.any(np.diff(ts[span]) <= 0) or np.any(np.diff(ts[span]) > max_gap)
            or np.any(np.diff((d['sample_index'] if 'sample_index' in d else d.decision).to_numpy()[ix][span]) != 1)
            or np.any(target-ts[local] > max_gap)):
        return None
    # Last-observed values: interpolation could leak post-switch values leftward.
    return ix[local]


def matched_boundaries(d, a, raw, b, w, z, context, held, args):
    grid = np.linspace(-args.event_window, args.event_window, 21)
    episode_indices = {ep: r.index.to_numpy() for ep, r in d.groupby('episode', sort=False)}
    held_rows = w[w.session.eq(held)]
    cases = held_rows[held_rows.human_switch.eq(1)].sort_values(['run', 'time'])
    used_controls = set(); pairs = []; curves = []; skipped = {}
    for case in cases.itertuples():
        center = case.next_switch_time
        if d.time.iloc[case.anchor] > center-args.event_window:
            skipped['anchor_after_plot_start'] = skipped.get('anchor_after_plot_start', 0)+1; continue
        case_ix = episode_samples(d, episode_indices, case.episode, center, grid, args.max_gap)
        if case_ix is None or case_ix[0] < case.anchor:
            skipped['incomplete_case_window'] = skipped.get('incomplete_case_window', 0)+1; continue
        lag = center-case.time
        pool = held_rows[(held_rows.run==case.run) & (held_rows.action==case.action)
            & (held_rows.bout_id!=case.bout_id) & (~held_rows.bout_id.isin(used_controls))
            & ((np.log1p(held_rows.age)-np.log1p(case.age)).abs() <= args.age_caliper)
            & (held_rows.next_switch_time > held_rows.time+lag+args.event_window)]
        if not len(pool):
            skipped['no_age_action_control'] = skipped.get('no_age_action_control', 0)+1; continue
        ends = pool.end.to_numpy(int)
        distance = np.sqrt(np.mean((context[ends]-context[case.end])**2, axis=1))
        order = np.argsort(distance, kind='stable'); matched = None
        for k in order:
            if distance[k] > args.context_caliper: break
            control = pool.iloc[k]; control_center = float(control.time)+lag
            if d.time.iloc[int(control.anchor)] > control_center-args.event_window: continue
            control_ix = episode_samples(d, episode_indices, control.episode, control_center, grid, args.max_gap)
            if control_ix is None or control_ix[0] < int(control.anchor): continue
            matched = (control, control_center, control_ix, float(distance[k])); break
        if matched is None:
            skipped['no_complete_context_control'] = skipped.get('no_complete_context_control', 0)+1; continue
        control, control_center, control_ix, distance = matched
        used_controls.add(int(control.bout_id))
        pair_id = f'{held}_{len(pairs)}'
        pairs.append(dict(pair_id=pair_id, session=held, run=case.run, time=center,
            case_bout=case.bout_id, control_bout=int(control.bout_id), action=case.action,
            case_age=case.age, control_age=float(control.age), context_rms_distance=distance,
            case_budget=float(d.budget.iloc[case.end]), control_budget=float(d.budget.iloc[int(control.end)])))
        for label, row, ix, ct in [('switch', case, case_ix, center), ('persist', control, control_ix, control_center)]:
            start = int(row.start); anchor = int(row.anchor)
            direction = z[anchor]-z[start]; direction /= max(np.linalg.norm(direction), 1e-8)
            projected = (z[ix]-z[ix[0]])@direction
            actor_away = a['actor_p'][ix[0], case.action]-a['actor_p'][ix, case.action]
            rawtime, rawaction = raw[case.run]
            ri = np.searchsorted(rawtime, ct+grid, side='right')-1
            human_away = (rawaction[ri] != case.action).astype(float)
            zero = int(np.argmin(abs(grid)))
            if label=='switch' and human_away[zero] != 1:
                raise ValueError('Raw switch is not aligned exactly at lag zero')
            if label=='persist' and human_away[grid>=0].any():
                raise ValueError('Persistence control switches in the plotted future')
            for j, relative_time in enumerate(grid):
                curves.append(dict(pair_id=pair_id, session=held, run=case.run, time=center,
                    condition=label, lag=relative_time, action=case.action,
                    evaluation_change=projected[j], policy_away=actor_away[j],
                    human_away=human_away[j], budget=float(d.budget.iloc[ix[j]])))
    qa = dict(session=int(held), available_events=len(cases), matched_pairs=len(pairs), exclusions=skipped)
    return pd.DataFrame(pairs), pd.DataFrame(curves), qa


def comparisons(predictions, w, args, parent):
    results = []
    contrasts = [('level', 'baseline'), ('amplitude', 'level'),
                 ('progress', 'amplitude'), ('persistence', 'progress')]
    for subgroup in ['all', 'noop', 'active']:
        mask = np.ones(len(predictions), bool)
        if subgroup != 'all': mask = predictions.action.eq(0).to_numpy() == (subgroup=='noop')
        for target, part in predictions.loc[mask].groupby('target'):
            wide = part.pivot(index='row_id', columns='model', values='loss')
            for model, reference in contrasts:
                if model not in wide or reference not in wide: continue
                losses = wide[[model, reference]].dropna()
                rows = w.set_index('row_id').loc[losses.index].copy()
                if len(rows) < 20: continue
                gain = (losses[reference]-losses[model]).to_numpy()
                # Assign a whole bout to the block containing its onset.
                rows['time'] = rows.onset
                interval = parent.block_interval(rows, gain, args.bootstrap, args.seed)
                per = rows[['session', 'run']].copy(); per['gain'] = gain
                session_gain = per.groupby(['session', 'run']).gain.mean().groupby('session').mean()
                results.append(dict(subgroup=subgroup, target=target, model=model, reference=reference,
                    rows=len(rows), bouts=rows.bout_id.nunique(), sessions=rows.session.nunique(),
                    events=int(rows.human_switch.sum()), positive_sessions=int((session_gain>0).sum()), **interval))
    return pd.DataFrame(results)


def summarize_curves(curves, args, parent):
    records = []
    for lag, rows in curves.groupby('lag'):
        for metric in ['evaluation_change', 'policy_away', 'human_away', 'budget']:
            wide = rows.pivot(index='pair_id', columns='condition', values=metric)
            meta = rows.drop_duplicates('pair_id').set_index('pair_id').loc[wide.index]
            for condition, values in [('switch', wide.switch), ('persist', wide.persist),
                                      ('difference', wide.switch-wide.persist)]:
                result = parent.block_interval(meta, values.to_numpy(), args.bootstrap, args.seed)
                records.append(dict(lag=lag, metric=metric, condition=condition, pairs=len(wide), **result))
    return pd.DataFrame(records)


def render(d, a, b, w, display, pairs, curves_summary, scores, out, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = out/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    def save(fig, name):
        for ext in ['png', 'pdf']: fig.savefig(figures/(name+'.'+ext), dpi=160, bbox_inches='tight')
        plt.close(fig)
    z, held = display
    available = b[b.session.eq(held)].sort_values('start')
    chosen = available.groupby('action', sort=True).head(1).head(3)
    examples = []
    if len(chosen):
        fig, axes = plt.subplots(4, len(chosen), figsize=(5*len(chosen), 10), squeeze=False, layout='constrained')
        for col, bout in enumerate(chosen.itertuples()):
            ix = np.arange(bout.start, bout.last+1); t = d.time.to_numpy()[ix]-bout.onset
            metrics, direction = cumulative_geometry(z[ix], d.time.to_numpy()[ix], bout.anchor-bout.start)
            axes[0, col].plot(t, (z[ix]-z[bout.anchor])@direction, lw=1.3)
            axes[0, col].set_title(f'{ACTION_NAMES[bout.action]} · first eligible bout\nsession {held}, {bout.run.split("/")[-1]}', fontsize=9)
            late_time = t[bout.anchor-bout.start+1:]
            axes[1, col].plot(late_time, metrics['coherence'], label='Directional coherence')
            axes[1, col].plot(late_time, metrics['positive_fraction'], label='Fraction of time moving forward')
            axes[1, col].set_ylim(-1.05, 1.05)
            axes[1, col].legend(fontsize=7)
            axes[2, col].plot(t, a['actor_p'][ix, bout.action], label='Maintained human action')
            alternatives = np.array(a['actor_p'][ix]); alternatives[:, bout.action] = 0
            axes[2, col].plot(t, alternatives.max(1), label='Largest alternative probability')
            axes[2, col].set_ylim(0, 1.02); axes[2, col].legend(fontsize=7)
            axes[3, col].step(t, d.budget.to_numpy()[ix], where='post')
            axes[3, col].set_xlabel('Seconds from human bout onset')
            for ax in axes[:, col]:
                ax.axvspan(0, d.time.iloc[bout.anchor]-bout.onset, color='.9', label='Direction-estimation period')
                if np.isfinite(bout.next_switch_time): ax.axvline(bout.next_switch_time-bout.onset, color='black', ls='--')
            examples.append(dict(bout_id=bout.bout_id, action=bout.action, session=held,
                                 selection='first eligible bout per action in first analysis-held session'))
        for ax, label in zip(axes[:, 0], ['Projected evaluation\n(relative to anchor)', 'Path history', 'REAL policy probability', 'Computations']): ax.set_ylabel(label)
        fig.suptitle('Pong sub001 · direction fixed once per human action bout\nGrey = reference period; dashed = next raw human switch. Descriptive examples.', fontsize=12)
        save(fig, 'whole_bout_examples')
    pd.DataFrame(examples).to_csv(out/'examples.csv', index=False)
    if len(curves_summary):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
        labels = {'evaluation_change':'Evaluation change along the bout direction',
                  'policy_away':'Drop in probability of the maintained action',
                  'human_away':'Fraction no longer using the maintained action', 'budget':'Imagination count'}
        for ax, metric in zip(axes.flat, labels):
            for condition, color in [('switch', '#b44c3a'), ('persist', '#2b739b')]:
                r = curves_summary[(curves_summary.metric==metric)&(curves_summary.condition==condition)].sort_values('lag')
                if metric=='human_away':
                    ax.step(r.lag, r.gain, where='post', color=color, label=condition)
                    ax.fill_between(r.lag, r.ci_low, r.ci_high, step='post', color=color, alpha=.15)
                else:
                    ax.plot(r.lag, r.gain, color=color, label=condition)
                    ax.fill_between(r.lag, r.ci_low, r.ci_high, color=color, alpha=.15)
            ax.axvline(0, color='black', ls='--'); ax.set(xlabel='Seconds from actual / matched pseudo switch', ylabel=labels[metric])
            if metric in ['evaluation_change', 'policy_away']: ax.axhline(0, color='.7', lw=.7)
            ax.legend()
        distribution = pairs.action.value_counts().sort_index()
        counts = ', '.join(f'{ACTION_NAMES[int(k)]}: {v}' for k, v in distribution.items())
        fig.suptitle(f'Pong sub001 · {len(pairs)} matched pairs ({counts})\nSame run/action, similar age and pre-event context. Human separation follows selection by definition.', fontsize=11)
        save(fig, 'matched_switch_persist')
    body = '<html><meta charset="utf-8"><body style="font-family:sans-serif;max-width:1200px;margin:auto">'
    body += '<h1>Pong sub001: whole-bout evaluation dynamics</h1>'
    body += '<p>Exploratory supplement. Direction is fixed near 0.25 s after raw human bout onset. Short/left-censored bouts are excluded; NOOP and active-action support are reported separately. All sessions trained Thinker; readout folds are analysis holdouts only.</p>'
    body += '<p>Primary contrast: persistence versus progress, predicting the NEXT replay observation. Added variables are positive-time fraction and current positive streak. Progress alone is net displacement. Policy probability change and human switching are separate outcomes; no causal sequence or internal accumulator is established.</p>'
    body += '<p>Reference state, current evaluation, latest evaluation increment, action age, policy, branch, tree carry, budget and unsigned motion are controlled. No direction is defined using future outcomes. Block intervals are conditional on the participant/fitted models; not population inference or multiplicity-adjusted.</p>'
    body += '<h2>Prospective tests</h2>'+scores[scores.model.isin(['progress', 'persistence'])].to_html(index=False, float_format=lambda x:f'{x:.5g}')
    body += '<h2>Support by action</h2>'+pd.read_csv(out/'support.csv').to_html(index=False)
    body += '<p>Boundary matching is descriptive and selects controls with no future switch; the human-action panel therefore differs by construction. Evaluation/policy curves are centered at the first plotted pre-event sample. Values are sampled causally (last observation), without interpolation across transitions. Axis remains fixed after the switch. Pair CIs do not account fully for bouts reused in both case and control roles.</p>'
    for p in sorted(figures.glob('*.png')): body += f'<img style="width:100%" src="figures/{p.name}">'
    body += '<p>See comparisons.csv, fit_status.csv, matched_pairs.csv, matching_qa.json and matched_curve_summary.csv for all tests/support. Coefficient sign and prediction improvement are distinct checks.</p></body></html>'
    (out/'index.html').write_text(body)


def self_test():
    ts = np.arange(8.)/10
    z = np.column_stack([np.arange(8.), np.zeros(8)])
    g, u = cumulative_geometry(z, ts, 2)
    assert g['coherence'][-1] > .999 and g['positive_fraction'][-1] > .999
    np.testing.assert_allclose(g['streak'][-1], .5)
    changed = z.copy(); changed[6:] -= 100
    gg, uu = cumulative_geometry(changed, ts, 2)
    np.testing.assert_allclose(u, uu)
    for key in g: np.testing.assert_allclose(g[key][:3], gg[key][:3])
    # Same endpoints/path length but different ordering gives different streak.
    x = np.array([0., 1., 2., 3., 4., 3., 4., 5.])[:, None]
    y = np.array([0., 1., 2., 1., 2., 3., 4., 5.])[:, None]
    gx, _ = cumulative_geometry(x, ts, 2); gy, _ = cumulative_geometry(y, ts, 2)
    np.testing.assert_allclose(gx['progress'][-1], gy['progress'][-1])
    np.testing.assert_allclose(gx['path'][-1], gy['path'][-1])
    assert gx['streak'][-1] != gy['streak'][-1]
    flat, _ = cumulative_geometry(np.ones((8, 2)), ts, 2)
    assert not flat['defined'].any() and not flat['progress'].any()
    print('PASS: fixed anchor, prospective feature prefixes, endpoint-matched path history, flat direction')

