"""Theme 09: prospective evaluation direction and whole-bout dynamics.

``analyze(args, subject, game, workdir)`` preserves the statistical tests from
``evaluation_accumulation`` and ``bout_evaluation``. Workdir is disposable:
full feature matrices and backend diagnostics are never final artifacts.
The return value is (plot/statistics tables, diagnostic metadata, arrays).
All required figure trajectories, including exact raw human action samples,
are returned as tables. No full activation matrix or fitted model is retained.

``render(root, tables, title)`` reads only these in-memory/saved tables. It
never reads behavioral sources, visualization captures, or neural checkpoints.
This lets a style-only rerun use CSVs without re-extracting or refitting.

Primary and supplementary analyses remain observational and participant
specific. Model training used every session; readout analysis holds out one
session. Event-aligned matching is descriptive, not a prospective test.
"""
from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluation_accumulation as accumulation
from . import bout_evaluation as bout

GAMES = {0: 'enduro', 1: 'pong', 2: 'spaceinvaders'}
DEFAULTS = dict(window=.5, horizon=.5, root_rank=12, activation_rank=8,
    penalty=.1, permutations=10, bootstrap=1000, minimum_train=200,
    seed=20261125, representations=['heads', 'activation'], smoke=False,
    smoke_decisions=512, reuse_from=[], anchor_seconds=.25,
    onset_tolerance=.12, max_gap=.25, minimum_late_steps=2,
    event_window=.5, age_caliper=.15, context_caliper=1.5)


def settings(args):
    """Backends' existing defaults; explicit caller settings take precedence."""
    values = {key: getattr(args, key, default) for key, default in DEFAULTS.items()}
    values['input_root'] = Path(getattr(args, 'input_root',
        accumulation.REPO / 'logs/visualization/hxln_10m'))
    values['reuse_from'] = [Path(p) for p in (values['reuse_from'] or [])]
    values['representations'] = list(values['representations'])
    if not values['representations'] or values['representations'][0] != 'heads':
        raise ValueError('heads must remain the prespecified primary representation')
    if any(x not in ('heads', 'activation') for x in values['representations']):
        raise ValueError('Unknown temporal representation')
    for name in ('window', 'horizon', 'penalty', 'anchor_seconds', 'onset_tolerance',
                 'max_gap', 'event_window', 'age_caliper', 'context_caliper'):
        if not np.isfinite(values[name]) or values[name] <= 0:
            raise ValueError(name + ' must be positive and finite')
    for name in ('root_rank', 'activation_rank', 'permutations', 'minimum_train',
                 'minimum_late_steps', 'smoke_decisions'):
        if values[name] < 1:
            raise ValueError(name + ' must be positive')
    if values['bootstrap'] < 20:
        raise ValueError('At least 20 bootstrap resamples are required')
    return Namespace(**values)


def _empty(columns=()):
    return pd.DataFrame(columns=list(columns) or ['status'])


def _concat(parts, columns=()):
    return pd.concat(parts, ignore_index=True) if parts else _empty(columns)


def _read_optional(path):
    if not Path(path).exists():
        return _empty()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return _empty()


def _records(rows):
    """Keep nested diagnostics readable in CSV rather than separate JSONs."""
    return pd.DataFrame([{k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                          for k, v in row.items()} for row in rows]) if rows else _empty()


def _load_raw(manifest):
    raw = {}
    sources = {e['metadata']['source_id']: e['metadata'] for e in manifest['episodes']}
    for run, metadata in sources.items():
        path = Path(metadata['source_path'])
        if not path.exists():
            path = accumulation.REPO / 'behavioral_data_block' / run
        if accumulation.sha(path) != metadata['source_sha256']:
            raise ValueError('Behavioral source changed: ' + str(path))
        with np.load(path, allow_pickle=False) as data:
            actions = data['action']
            actions = actions.argmax(1) if actions.ndim == 2 else actions
            times = data['time'].astype(float)
        if len(times) != len(actions) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Invalid behavioral timestamps: ' + run)
        raw[run] = (times, actions.astype(int))
    return raw


def _human_samples(raw, run, start, stop, center, example):
    times, actions = raw[run]
    # Match the original descriptive panels: exact logged samples inside the
    # selected interval, without interpolation or assuming one decision/step.
    keep = (times >= start) & (times <= stop)
    return pd.DataFrame(dict(example=example, relative_time=times[keep]-center,
                             action=actions[keep], source_time=times[keep]))


def _primary_examples(d, arrays, windows, display, raw):
    metadata, traces, humans = [], [], []
    if display is None or windows.empty:
        return dict(primary_examples=_empty(), primary_example_trajectories=_empty(),
                    primary_example_human=_empty())
    z, directions, held = display
    eligible = np.flatnonzero(windows.session.eq(held).to_numpy() &
        windows.nonoverlap.to_numpy(bool) & (directions[:, 4] > 0))
    if len(eligible):
        ordered = eligible[np.argsort(directions[eligible, 1])]
        chosen = [ordered[int(q*(len(ordered)-1))] for q in (.1, .5, .9)]
    else:
        chosen = []
    for label, i in zip(('low', 'middle', 'high'), chosen):
        row = windows.iloc[i]
        index = np.arange(int(row.start), int(row.future)+1)
        times = d.loc[index, 'time'].to_numpy()
        axis = z[int(row.middle)]-z[int(row.start)]
        axis /= max(np.linalg.norm(axis), 1e-12)
        frame = pd.DataFrame(dict(example=label, decision_row=index,
            relative_time=times-row.time, projection=(z[index]-z[int(row.start)])@axis,
            budget=d.loc[index, 'budget'].to_numpy()))
        for action in range(arrays['actor_p'].shape[1]):
            frame[f'policy_action_{action}'] = arrays['actor_p'][index, action]
        traces.append(frame)
        humans.append(_human_samples(raw, row.run, times[0], times[-1], row.time, label))
        metadata.append(dict(example=label, window_id=int(row.window_id), session=int(row.session),
            run=row.run, start_time=float(row.start_time), prediction_time=float(row.time),
            early_direction_fixed=float(d.loc[int(row.middle), 'time']-row.time),
            future_end=float(times[-1]-row.time), coherence=float(directions[i, 1]),
            selection='past coherence quantile; no future outcome used'))
    return dict(primary_examples=_records(metadata),
                primary_example_trajectories=_concat(traces), primary_example_human=_concat(humans))


def _bout_examples(d, arrays, bouts, display, raw):
    metadata, traces, humans = [], [], []
    if display is None or bouts.empty:
        return dict(bout_examples=_empty(), bout_example_trajectories=_empty(), bout_example_human=_empty())
    z, held = display
    selected = bouts[bouts.session.eq(held)].sort_values('start').groupby('action', sort=True).head(1).head(3)
    for row in selected.itertuples():
        index = np.arange(row.start, row.last+1)
        times = d.time.to_numpy()[index]
        local_anchor = row.anchor-row.start
        metrics, axis = bout.cumulative_geometry(z[index], times, local_anchor)
        example = f'bout_{row.bout_id}'
        alternatives = np.array(arrays['actor_p'][index], copy=True)
        alternatives[:, row.action] = 0
        frame = pd.DataFrame(dict(example=example, decision_row=index, relative_time=times-row.onset,
            projection=(z[index]-z[row.anchor])@axis,
            incumbent_probability=arrays['actor_p'][index, row.action],
            alternative_probability=alternatives.max(1), budget=d.budget.to_numpy()[index]))
        for name in ('coherence', 'positive_fraction'):
            frame[name] = np.r_[np.full(local_anchor+1, np.nan), metrics[name]]
        traces.append(frame)
        humans.append(_human_samples(raw, row.run, times[0], times[-1], row.onset, example))
        metadata.append(dict(example=example, bout_id=row.bout_id, action=row.action, session=int(held),
            run=row.run, anchor_time=float(d.time.iloc[row.anchor]-row.onset),
            next_switch_time=float(row.next_switch_time-row.onset),
            selection='first eligible bout per action in first analysis-held session'))
    return dict(bout_examples=_records(metadata), bout_example_trajectories=_concat(traces),
                bout_example_human=_concat(humans))


def _supplement(d, arrays, raw, args):
    bouts, risk, coverage = bout.build_risk_set(d, arrays['actor_p'], args, accumulation)
    tables = {'bout_bouts': bouts, 'bout_risk_rows': risk}
    qa = dict(coverage=coverage, matching=[], status='INSUFFICIENT_SUPPORT')
    if risk.empty:
        tables.update(bout_support=_empty(), bout_fit_status=_empty(), bout_predictions=_empty(),
            bout_metrics=_empty(), bout_comparisons=_empty(), matched_pairs=_empty(),
            matched_curves=_empty(), matched_curve_summary=_empty(), matching_qa=_empty(),
            **_bout_examples(d, arrays, bouts, None, raw))
        return tables, qa
    tables['bout_support'] = risk.groupby(['session', 'action']).agg(
        rows=('row_id', 'size'), bouts=('bout_id', 'nunique'), events=('human_switch', 'sum'),
        median_age=('age', 'median')).reset_index()
    predictions, statuses, metrics, pair_tables, curve_tables, matching = [], [], [], [], [], []
    display = None
    for held in sorted(risk.session.unique()):
        held = int(held)
        accumulation.log(f'09: analysis-held session {held}, fixed-bout readouts')
        root, z, context = bout.coordinates(d, arrays, held, args, accumulation)
        designs, directed = bout.features(d, arrays, bouts, risk, root, z)
        selected = risk.session.to_numpy() == held
        train = np.flatnonzero(~selected)
        test = np.flatnonzero(selected)
        frame = risk.iloc[test].copy()
        for j, name in enumerate(('progress', 'positive_fraction', 'positive_streak_seconds', 'coherence')):
            frame[name] = directed[test, j]
        metrics.append(frame)
        if display is None:
            display = (z.copy(), held)
        for target in ('human_switch', 'policy_js', 'policy_away'):
            y = risk[target].to_numpy()
            binary = target == 'human_switch'
            for model, design in designs.items():
                prediction, status, coefficient = accumulation.predict_model(
                    design, y, train, test, binary, args.penalty, args.minimum_train)
                record = dict(session=held, target=target, model=model, status=status,
                    n_train=len(train), n_test=len(test), events=int(risk.human_switch.iloc[test].sum()))
                if coefficient is not None and model in ('progress', 'persistence'):
                    names = ['progress'] if model == 'progress' else ['progress', 'positive_fraction', 'positive_streak']
                    record.update({name+'_coefficient': float(value)
                                   for name, value in zip(names, coefficient[-len(names):])})
                statuses.append(record)
                if prediction is None:
                    continue
                frame = risk.iloc[test][['row_id', 'bout_id', 'session', 'run', 'time', 'action']].copy()
                frame['target'] = target
                frame['model'] = model
                frame['truth'] = y[test]
                frame['prediction'] = prediction
                if binary:
                    frame['loss'] = -(y[test]*np.log(np.clip(prediction, 1e-10, 1-1e-10)) +
                        (1-y[test])*np.log(np.clip(1-prediction, 1e-10, 1)))
                else:
                    frame['loss'] = (prediction-y[test])**2
                predictions.append(frame)
        pairs, curves, audit = bout.matched_boundaries(
            d, arrays, raw, bouts, risk, z, context, held, args)
        if not pairs.empty:
            pair_tables.append(pairs)
            curve_tables.append(curves)
        matching.append(audit)
    predicted = _concat(predictions)
    scores = bout.comparisons(predicted, risk, args, accumulation) if predictions else _empty()
    pairs = _concat(pair_tables)
    curves = _concat(curve_tables)
    summary = bout.summarize_curves(curves, args, accumulation) if curve_tables else _empty()
    tables.update(bout_fit_status=pd.DataFrame(statuses), bout_predictions=predicted,
        bout_metrics=_concat(metrics), bout_comparisons=scores, matched_pairs=pairs,
        matched_curves=curves, matched_curve_summary=summary, matching_qa=_records(matching),
        **_bout_examples(d, arrays, bouts, display, raw))
    qa.update(matching=matching, status='COMPLETED' if predictions else 'INSUFFICIENT_SUPPORT',
        fit_status_counts=pd.Series([r['status'] for r in statuses]).value_counts().to_dict())
    return tables, qa


def analyze(args, subject, game, workdir):
    """Run both temporal analyses; return only reusable scientific outputs.

    Missing inputs or integrity failures raise rather than producing artificial
    empty measurements. Insufficient statistical support is instead reported
    explicitly in QA and fit-status tables, preserving any descriptive data.
    """
    args = settings(args)
    game = GAMES[int(game)] if isinstance(game, (int, np.integer)) else str(game).lower()
    if game not in GAMES.values():
        raise ValueError('Unknown game: ' + game)
    subject = int(subject)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    primary_dir = workdir / 'primary'
    primary_dir.mkdir(exist_ok=True)
    capture = args.input_root / f'sub{subject:03d}' / game
    manifest = accumulation.validate_capture(capture)
    d, arrays = accumulation.extract(args, subject, game, workdir, manifest)
    raw = _load_raw(manifest)
    windows, coverage = accumulation.make_windows(d, arrays['actor_p'], args.window, args.horizon, args.max_gap)
    tables = {'decisions': d, 'primary_windows': windows}
    if not windows.empty:
        predictions, comparisons, display = accumulation.fit_readouts(d, arrays, windows, args, primary_dir)
        fold_qa = accumulation.read_json(primary_dir/'fold_qa.json')
    else:
        predictions, comparisons, display, fold_qa = _empty(), _empty(), None, []
    tables.update(primary_predictions=predictions, primary_comparisons=comparisons,
        primary_fit_status=_read_optional(primary_dir/'fit_status.csv'),
        primary_directional_features=_read_optional(primary_dir/'heldout_directional_features.csv.gz'),
        primary_fold_qa=_records(fold_qa), **_primary_examples(d, arrays, windows, display, raw))
    extra, supplementary_qa = _supplement(d, arrays, raw, args)
    tables.update(extra)
    # The published tables retain identity explicitly even when plotted alone.
    for table in tables.values():
        table['subject'] = subject
        table['game'] = game
    primary_supported = False
    if len(comparisons):
        primary = comparisons[(comparisons.representation == 'heads') &
            comparisons.sampling.eq('nonoverlap') & comparisons.target.eq('human_switch') &
            comparisons.model.eq('directional') & comparisons.reference.eq('magnitude')]
        primary_supported = bool(len(primary) and primary.sessions.min() == 4)
    extraction = accumulation.read_json(workdir/'coverage.json') if (workdir/'coverage.json').exists() else {}
    qa = dict(subject=subject, game=game, status='COMPLETED' if primary_supported else 'INSUFFICIENT_SUPPORT',
        primary_supported=primary_supported, decisions=len(d), sessions=sorted(map(int, d.session.unique())),
        extraction=extraction, windows=coverage, primary_fold_qa=fold_qa,
        supplementary=supplementary_qa, protocol=accumulation.protocol(args),
        bout_protocol={k: getattr(args, k) for k in ('anchor_seconds', 'onset_tolerance', 'max_gap',
            'minimum_late_steps', 'event_window', 'age_caliper', 'context_caliper')},
        limitations=[
            'Session holdout applies to readout fitting; Thinker trained on all four sessions.',
            'Projection is net directional change, not evidence of an internal neural accumulator.',
            'Matched switch/persist curves are descriptive; human separation follows selection.',
            'Intervals condition on this participant and fitted models; no population or multiplicity-adjusted inference.',
            'Supplement excludes short and left-censored bouts; action-specific support is explicit.'])
    # All rendered trajectories are tabular. There are no necessary persistent
    # high-dimensional arrays for this theme; upstream matrices stay in scratch.
    return tables, qa, {}


def render(root, tables, title):
    """Reproduce all five original panel types solely from saved table data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    figures = root/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    written = []
    palette = {'switch': '#b44c3a', 'persist': '#2b739b'}
    def save(figure, name):
        for extension in ('png', 'pdf'):
            path = figures/f'{name}.{extension}'
            figure.savefig(path, dpi=160, bbox_inches='tight')
            written.append(path)
        plt.close(figure)
    def table(name):
        return tables.get(name, _empty())
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42}):
        examples = table('primary_examples')
        traces = table('primary_example_trajectories')
        human = table('primary_example_human')
        for metadata in examples.itertuples():
            frame = traces[traces.example.eq(metadata.example)].sort_values('relative_time')
            observed = human[human.example.eq(metadata.example)].sort_values('relative_time')
            fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True, layout='constrained')
            axes[0].plot(frame.relative_time, frame.projection, 'o-', ms=3, label='Evaluation onto early direction')
            axes[0].axvline(metadata.early_direction_fixed, color='grey', ls=':', label='Direction fixed')
            axes[0].set_ylabel('Projection (a.u.)')
            axes[0].legend(fontsize=8)
            for column in sorted((c for c in frame if c.startswith('policy_action_')),
                                 key=lambda c: int(c.rsplit('_', 1)[-1])):
                axes[1].plot(frame.relative_time, frame[column], label=column.rsplit('_', 1)[-1], lw=1)
            axes[1].set_ylabel('REAL policy')
            axes[1].legend(title='Action ID', ncol=6, fontsize=7)
            axes[2].step(observed.relative_time, observed.action, where='post')
            axes[2].set_ylabel('Human action ID')
            axes[3].step(frame.relative_time, frame.budget, where='post')
            axes[3].set(ylabel='Computations', xlabel='Seconds relative to prediction time')
            for ax in axes:
                ax.axvline(0, color='black', ls='--')
                ax.axvspan(0, metadata.future_end, color='#eee', alpha=.5)
            fig.suptitle(f'{title} · past-coherence {metadata.example} example\nShaded = future; selected without future outcome')
            save(fig, 'timeline_'+metadata.example)
        examples = table('bout_examples')
        traces = table('bout_example_trajectories')
        if len(examples):
            fig, axes = plt.subplots(4, len(examples), figsize=(5*len(examples), 10),
                                     squeeze=False, layout='constrained')
            for column, metadata in enumerate(examples.itertuples()):
                frame = traces[traces.example.eq(metadata.example)].sort_values('relative_time')
                axes[0, column].plot(frame.relative_time, frame.projection, lw=1.3)
                axes[0, column].set_title(f'Action {metadata.action} · first eligible bout\nsession {metadata.session}', fontsize=9)
                axes[1, column].plot(frame.relative_time, frame.coherence, label='Directional coherence')
                axes[1, column].plot(frame.relative_time, frame.positive_fraction, label='Fraction of time moving forward')
                axes[1, column].set_ylim(-1.05, 1.05)
                axes[1, column].legend(fontsize=7)
                axes[2, column].plot(frame.relative_time, frame.incumbent_probability, label='Maintained human action')
                axes[2, column].plot(frame.relative_time, frame.alternative_probability, label='Largest alternative probability')
                axes[2, column].set_ylim(0, 1.02)
                axes[2, column].legend(fontsize=7)
                axes[3, column].step(frame.relative_time, frame.budget, where='post')
                axes[3, column].set_xlabel('Seconds from human bout onset')
                for ax in axes[:, column]:
                    ax.axvspan(0, metadata.anchor_time, color='.9')
                    if np.isfinite(metadata.next_switch_time):
                        ax.axvline(metadata.next_switch_time, color='black', ls='--')
            for ax, label in zip(axes[:, 0], ('Projected evaluation\n(relative to anchor)',
                'Path history', 'REAL policy probability', 'Computations')):
                ax.set_ylabel(label)
            fig.suptitle(f'{title} · direction fixed once per human action bout\nGrey = reference period; dashed = next human switch. Descriptive examples.', fontsize=12)
            save(fig, 'whole_bout_examples')
        curves = table('matched_curve_summary')
        pairs = table('matched_pairs')
        if len(curves):
            labels = {'evaluation_change': 'Evaluation change along the bout direction',
                'policy_away': 'Drop in probability of maintained action',
                'human_away': 'Fraction no longer using maintained action', 'budget': 'Imagination count'}
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
            for ax, (metric, label) in zip(axes.flat, labels.items()):
                for condition, color in palette.items():
                    rows = curves[curves.metric.eq(metric) & curves.condition.eq(condition)].sort_values('lag')
                    if metric == 'human_away':
                        ax.step(rows.lag, rows.gain, where='post', color=color, label=condition)
                        ax.fill_between(rows.lag, rows.ci_low, rows.ci_high, step='post', color=color, alpha=.15)
                    else:
                        ax.plot(rows.lag, rows.gain, color=color, label=condition)
                        ax.fill_between(rows.lag, rows.ci_low, rows.ci_high, color=color, alpha=.15)
                ax.axvline(0, color='black', ls='--')
                ax.set(xlabel='Seconds from actual / matched pseudo switch', ylabel=label)
                if metric in ('evaluation_change', 'policy_away'):
                    ax.axhline(0, color='.7', lw=.7)
                ax.legend()
            counts = ', '.join(f'action {int(k)}: {v}' for k, v in pairs.action.value_counts().sort_index().items())
            fig.suptitle(f'{title} · {len(pairs)} matched pairs ({counts})\nSame run/action, similar age and context; human separation follows selection.', fontsize=11)
            save(fig, 'matched_switch_persist')
    return written
