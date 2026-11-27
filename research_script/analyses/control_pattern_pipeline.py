"""Theme 10 adapter for verified, session-held-out control-pattern analyses.

The statistical backend is unchanged: functional ACE representation, nested
session mixture selection (including K=1), temporal windows, and readout/null
validation run in disposable scratch. Only explicit tables, plot-source arrays
and diagnostic metadata are returned for deterministic publication.

analyze(args, subject, game, prepared, workdir) -> (tables, qa, arrays)
render(root, tables, title) -> figure paths

``prepared`` may provide ``rich_path`` pointing at a completed old extraction.
The old extraction's marker and every reused artifact are verified against a
fresh input audit. A running downstream stage does not invalidate a completed,
immutable extract stage. No old source or result is modified.

Arrays are a mapping of NPZ name to named numeric arrays. Only selected screen
montages and all-switch plot samples are retained by default. ``retain_bases``
explicitly retains functional basis parameters for later projection; complete
activation matrices and nested fitted-mixture directories are never exported.
The renderer opens only these saved tables and montage arrays, never raw data.
"""
from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from common import native_replay as native
from . import control_patterns as backend

GAMES = {0: 'enduro', 1: 'pong', 2: 'spaceinvaders'}
DEFAULTS = dict(windows_seconds=[.25, .5, 1.], horizons_seconds=[.25, .5, 1.],
    basis_samples=2048, jacobian_samples=512, energy=.99, max_rank=128,
    chunk_size=256, threads=4, seed=701, bootstrap=1000, null_repeats=100,
    smoke=False, smoke_decisions=96, retain_bases=False)
RICH_FILES = ('decisions.csv.gz', 'computations.csv.gz', 'actor_features.npy',
    'policies.npy', 'actual_policy.npy', 'screen.npy', 'tree_summary.npy',
    'screens.npy', 'extraction.json')
ACE_LABELS = ('vp_value', 'actor_voc', 'actor_policy')
ALIGN_METRICS = ('K', 'U', 'D', 'policy_change',
                 'vp_value_ACE1', 'actor_voc_ACE1', 'actor_policy_ACE1')
PROFILE_METRICS = ('K', 'V_initial', 'D', 'U', 'voc_initial', 'policy_change')
FINAL_TABLES = ('decisions', 'computations', 'windows', 'switches', 'representation_summary',
    'motif_summary', 'predictions', 'budget_predictions', 'comparisons', 'continuity',
    'profiles', 'examples', 'voluntary_choices')


def settings(args, subject):
    values = {name: getattr(args, name, default) for name, default in DEFAULTS.items()}
    values['input_root'] = Path(getattr(args, 'input_root', native.DEFAULT_INPUT))
    values['subject'] = int(subject)
    values['resume'] = False
    for name in ('windows_seconds', 'horizons_seconds'):
        values[name] = sorted(set(map(float, values[name])))
        if not values[name] or not all(np.isfinite(x) and x > 0 for x in values[name]):
            raise ValueError(name + ' must contain positive finite durations')
    if .5 not in values['windows_seconds'] or .25 not in values['horizons_seconds']:
        raise ValueError('The prespecified .5 s window and .25 s horizon are required')
    if not 0 < values['energy'] <= 1:
        raise ValueError('ACE energy must be in (0, 1]')
    for name in ('basis_samples', 'jacobian_samples', 'max_rank', 'chunk_size', 'threads',
                 'bootstrap', 'null_repeats', 'smoke_decisions'):
        if values[name] < 1:
            raise ValueError(name + ' must be positive')
    if values['smoke']:
        for name, maximum in [('basis_samples', 64), ('jacobian_samples', 16),
                              ('max_rank', 8), ('bootstrap', 20), ('null_repeats', 3)]:
            values[name] = min(values[name], maximum)
    return Namespace(**values)


def _empty():
    return pd.DataFrame(columns=['status'])


def _concat(parts):
    return pd.concat(parts, ignore_index=True) if parts else _empty()


def _read(path):
    path = Path(path)
    if not path.exists():
        return _empty()
    try:
        return pd.read_csv(path, float_precision='round_trip')
    except pd.errors.EmptyDataError:
        return _empty()


def _flat_rows(records):
    return pd.DataFrame([{key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in records]) if records else _empty()


def _candidate_path(args, prepared, subject, game):
    value = getattr(args, 'rich_reuse', None)
    if value is None and isinstance(prepared, dict):
        value = prepared.get('rich_path') or prepared.get('rich_reuse')
    if value is None:
        return None
    path = Path(value)
    for candidate in (path/f'sub{subject:03d}'/game, path/game, path):
        if (candidate/'extract.done.json').is_file():
            return candidate
    raise ValueError('Requested rich reuse has no completed extraction marker: ' + str(path))


def _reuse_extract(source, destination, audit, smoke):
    """Bind copied numeric artifacts to the same episodes and checkpoint."""
    old = native.read_json(source/'audit.json')
    marker = native.read_json(source/'extract.done.json')
    if marker.get('status') not in ('COMPLETE', 'COMPLETED'):
        raise ValueError('Rich extraction is not complete')
    for key in ('subject', 'game', 'num_actions'):
        if old.get(key) != audit.get(key):
            raise ValueError('Rich extraction identity mismatch: ' + key)
    if sorted(old['checkpoint_hashes'].values()) != sorted(audit['checkpoint_hashes'].values()):
        raise ValueError('Rich extraction checkpoint mismatch')
    old_episodes = {e['episode']: (e['manifest_sha256'], e['session'], e['run']) for e in old['episodes']}
    current_episodes = {e['episode']: (e['manifest_sha256'], e['session'], e['run']) for e in audit['episodes']}
    if old_episodes != current_episodes or sorted(old['sources'].values()) != sorted(audit['sources'].values()):
        raise ValueError('Rich extraction input capture mismatch')
    extraction = native.read_json(source/'extraction.json')
    if extraction.get('sample_mode','all') != 'all':
        raise ValueError('Rich reuse must be a full extraction; selection is applied after provenance validation')
    if bool(extraction.get('smoke')) != bool(smoke):
        raise ValueError('Cannot mix smoke and full rich extractions')
    for name in RICH_FILES:
        expected = marker.get('artifacts', {}).get(name)
        if not expected or native.sha(source/name) != expected:
            raise ValueError('Changed/unverified rich artifact: ' + name)
        shutil.copy2(source/name, destination/name)
        if native.sha(destination/name) != expected:
            raise ValueError('Rich artifact changed while copying: ' + name)
    return dict(source=str(source), artifacts={name: marker['artifacts'][name] for name in RICH_FILES},
                checkpoint_bound=True, episode_bound=True)


def _nested_tables(out):
    assignments, first, selection, first_selection = [], [], [], []
    for folder in sorted((out/'motifs').glob('held_*/w*/*')):
        meta = native.read_json(folder/'fit.json')
        identity = {name: meta[name] for name in ('held', 'width', 'view')}
        for filename, collection in [('assignments.csv.gz', assignments),
            ('first_only_assignments.csv.gz', first), ('selection.csv', selection),
            ('first_only_selection.csv', first_selection)]:
            if (folder/filename).exists():
                frame = _read(folder/filename)
                for key, value in identity.items():
                    # The inner held session in selection.csv is a distinct
                    # identifier and must not be overwritten by the outer fold.
                    target = 'outer_held' if key == 'held' and filename.endswith('selection.csv') else key
                    frame[target] = value
                collection.append(frame)
    return dict(candidate_assignments=_concat(assignments), first_only_assignments=_concat(first),
                candidate_selection=_concat(selection), first_only_selection=_concat(first_selection))


def _computation_sources(events):
    curves = []
    for session, frame in events.groupby('session'):
        for metric in ('value', 'voc_ema', 'policy_change'):
            summary = frame.groupby('k')[metric].agg(['mean', 'count']).reset_index()
            summary['session'] = session
            summary['metric'] = metric
            curves.append(summary)
    bins = []
    choices = events[events.voluntary.astype(bool)]
    diagnostics = []
    for metric in ('voc_ema', 'delta_V', 'gate_probability'):
        valid = choices[np.isfinite(choices[metric]) & np.isfinite(choices.continue_choice)]
        unique = valid[metric].nunique()
        diagnostics.append(dict(metric=metric, rows=len(valid), unique_values=unique))
        if unique < 2:
            continue
        label = pd.qcut(valid[metric], min(10, unique), duplicates='drop')
        grouped = valid.groupby(label, observed=True).agg(x=(metric, 'mean'),
            y=('continue_choice', 'mean'), n=('row', 'size')).reset_index(drop=True)
        grouped['metric'] = metric
        bins.append(grouped)
    counts = events[events.k.le(30)].groupby('k').size().rename('n').reset_index()
    return dict(computation_curves=_concat(curves), stopping_curves=_concat(bins),
                computation_support=counts, stopping_support=pd.DataFrame(diagnostics))


def _alignment_sources(args, out, decisions, switches):
    grid = np.arange(-.5, .5001, .05)
    curves, coverage, directions, coordinates = [], [], [], []
    samples = {'grid': grid}
    for held in sorted(decisions.session.unique()):
        fit = tuple(s for s in sorted(decisions.session.unique()) if s != held)
        folder = out/'representation'/backend.fit_key(fit)
        first = np.minimum(decisions.start.to_numpy(int)+1, decisions.end.to_numpy(int)-1)
        axes = []
        for label in ACE_LABELS:
            latent = np.load(folder/(label+'_events.npy'), mmap_mode='r')
            axes.append(latent[first, 0] if latent.shape[1] else np.full(len(decisions), np.nan))
        values = np.column_stack([decisions[list(ALIGN_METRICS[:4])].to_numpy(float), *axes])
        held_rows = decisions[decisions.session.eq(held)]
        coords = held_rows[['row', 'session', 'segment', 'time']].copy()
        for j, name in enumerate(ALIGN_METRICS[4:]):
            coords[name] = values[held_rows.index, j+4]
        coordinates.append(coords)
        observed, shifted = [], []
        rng = np.random.default_rng(args.seed)
        transition_samples = {}
        for segment, frame in held_rows.groupby('segment'):
            events = switches[switches.segment.eq(segment)]
            times = frame.time.to_numpy()
            indices = frame.index.to_numpy()
            if events.empty or times[-1] <= times[0]:
                continue
            sampled = native.sa_indices(times, events.time.to_numpy(), grid)
            observed.append(native.sa_sample(values[indices], sampled))
            shifted_times = times[0] + np.mod(events.time.to_numpy()-times[0] +
                rng.uniform(.2, .8)*(times[-1]-times[0]), times[-1]-times[0])
            shifted.append(native.sa_sample(values[indices], native.sa_indices(times, shifted_times, grid)))
            for lag, count in zip(grid, (sampled >= 0).sum(0)):
                coverage.append(dict(held=held, segment=segment, raw_switches=len(events),
                    complete=int((sampled >= 0).all(1).sum()), lag=lag, n=int(count)))
            if held == 4:
                for transition, part in events.groupby(['action_from', 'action_to']):
                    selected = native.sa_indices(times, part.time.to_numpy(), grid)
                    transition_samples.setdefault(transition, []).append(native.sa_sample(values[indices], selected))
        if observed:
            observed = np.concatenate(observed)
            shifted = np.concatenate(shifted)
            samples[f'held{int(held)}_observed'] = observed
            samples[f'held{int(held)}_shifted'] = shifted
            for j, metric in enumerate(ALIGN_METRICS):
                mean = native.sa_mean(observed[:, :, j], axis=0)
                null = native.sa_mean(shifted[:, :, j], axis=0)
                for lag, value, reference, count in zip(grid, mean, null, np.isfinite(observed[:, :, j]).sum(0)):
                    curves.append(dict(held=held, metric=metric, lag=lag, mean=value, shifted=reference, n=int(count)))
        for (action_from, action_to), parts in sorted(transition_samples.items()):
            merged = np.concatenate(parts)
            for metric in ('vp_value_ACE1', 'actor_policy_ACE1'):
                j = ALIGN_METRICS.index(metric)
                for lag, value, count in zip(grid, native.sa_mean(merged[:, :, j], axis=0),
                                             np.isfinite(merged[:, :, j]).sum(0)):
                    directions.append(dict(held=held, metric=metric, action_from=action_from,
                        action_to=action_to, lag=lag, mean=value, n=int(count)))
    tables = dict(switch_curves=_flat_rows(curves), switch_coverage=_flat_rows(coverage),
        transition_curves=_flat_rows(directions), ace_first_coordinates=_concat(coordinates),
        aligned_metric_index=pd.DataFrame(dict(metric_index=np.arange(len(ALIGN_METRICS)), metric=ALIGN_METRICS)))
    return tables, samples


def _timeline_sources(out, decisions, examples, assignments):
    metadata, trajectories, candidate_rows, policy_rows = [], [], [], []
    montages = {}
    screens = np.load(out/'screens.npy', mmap_mode='r')
    policies = np.load(out/'actual_policy.npy', mmap_mode='r')
    selected = examples[examples.held.eq(4) & examples.width.eq(.5) & examples.view.isin(['evaluation', 'joint'])]
    for number, example in enumerate(selected.itertuples()):
        row = decisions.loc[example.row]
        ids = decisions.index[decisions.segment.eq(row.segment) & decisions.time.between(row.time-1, row.time+1)].to_numpy()
        if not len(ids):
            continue
        key = f'example_{number}'
        frame = decisions.loc[ids, ['row', 'time', 'human_action', 'K', 'D', 'U', 'voc_initial', 'voc_final']].copy()
        frame['relative_time'] = frame.time-row.time
        frame['example'] = key
        trajectories.append(frame)
        shown = ids[np.linspace(0, len(ids)-1, min(6, len(ids))).astype(int)]
        montages[key] = np.concatenate([screens[i].transpose(1, 2, 0) for i in shown], axis=1)
        probability = pd.DataFrame(policies[ids], columns=[f'p{i}' for i in range(policies.shape[1])])
        probability['example'] = key
        probability['relative_time'] = frame.relative_time.to_numpy()
        policy_rows.append(probability)
        candidates = assignments[assignments.held.eq(4) & assignments.width.eq(.5) &
            assignments.view.eq(example.view) & assignments.row.isin(ids)].copy()
        for column in [c for c in candidates if c.startswith('p') and c[1:].isdigit()]:
            part = candidates[['time', column]].rename(columns={column: 'probability'})
            part['relative_time'] = part.time-row.time
            part['candidate'] = int(column[1:])
            part['example'] = key
            candidate_rows.append(part)
        metadata.append(dict(example=key, held=4, view=example.view, state=int(example.state),
            kind=example.kind, time=float(row.time), row=int(example.row),
            selection='held-out centrality or seeded random; not outcome amplitude',
            figure=f'01_timeline_{example.view}_c{example.state+1}_{example.kind}'))
    return dict(timeline_examples=_flat_rows(metadata), timeline_trajectories=_concat(trajectories),
        timeline_policy=_concat(policy_rows), timeline_candidates=_concat(candidate_rows)), montages


def _profile_sources(decisions, profiles):
    records = []
    training = decisions[decisions.session.ne(4)]
    for view in ('evaluation', 'joint'):
        selected = profiles[profiles.held.eq(4) & profiles.width.eq(.5) & profiles.view.eq(view) &
                            profiles.metric.isin(PROFILE_METRICS)]
        for row in selected.itertuples():
            mean = float(training[row.metric].mean())
            sd = float(training[row.metric].std())
            sd = 1. if sd == 0 else sd
            records.append(dict(held=4, view=view, state=int(row.state), metric=row.metric,
                raw_mean=float(row.mean), training_mean=mean, training_sd=sd,
                standardized_mean=(float(row.mean)-mean)/sd, n=int(row.n)))
    return _flat_rows(records)


def _within_across_sources(decisions, assignments):
    frame = assignments[assignments.held.eq(4) & assignments.width.eq(.5) &
                        assignments.view.eq('evaluation')].reset_index(drop=True)
    if frame.empty:
        return dict(control_examples=_empty(), control_example_trajectories=_empty())
    good, action_change = backend.raw_pair_changes(frame, decisions, {})
    state_change = np.r_[False, np.diff(frame.state) != 0]
    different_action = np.r_[False, np.diff(frame.human_action) != 0]
    descriptions = [('same_action_changed_candidate', 'Same action, different candidate', good & state_change & ~action_change),
        ('changed_action_same_candidate', 'Different action, same candidate', good & ~state_change & different_action)]
    examples, traces = [], []
    for key, label, mask in descriptions:
        candidates = np.flatnonzero(mask)
        record = dict(example=key, label=label, candidates=len(candidates), supported=bool(len(candidates)))
        if len(candidates):
            center = frame.iloc[candidates[len(candidates)//2]]
            chosen = frame[frame.segment.eq(center.segment) & frame.time.between(center.time-.5, center.time+.5)]
            ids = chosen.row.to_numpy(int)
            part = chosen[['row', 'human_action', 'state']].copy()
            part['relative_time'] = chosen.time-center.time
            part['U'] = decisions.loc[ids, 'U'].to_numpy()
            part['example'] = key
            traces.append(part)
            record.update(row=int(center.row), time=float(center.time), selection='middle eligible occurrence; no outcome maximization')
        examples.append(record)
    return dict(control_examples=_flat_rows(examples), control_example_trajectories=_concat(traces))


def collect_plot_sources(args, out, tables):
    """Calculate original render-stage statistics once; save their full inputs."""
    decisions, events = tables['decisions'], tables['computations']
    added = _computation_sources(events)
    alignment, samples = _alignment_sources(args, out, decisions, tables['switches'])
    added.update(alignment)
    timelines, montages = _timeline_sources(out, decisions, tables['examples'], tables['candidate_assignments'])
    added.update(timelines)
    added['profile_display'] = _profile_sources(decisions, tables['profiles'])
    added.update(_within_across_sources(decisions, tables['candidate_assignments']))
    arrays = {'switch_aligned_samples': samples}
    if montages:
        arrays['timeline_montages'] = montages
    return added, arrays


def _select_rich(out, prepared):
    """Subset disposable extraction and remap every endpoint/array together."""
    from common.sampling import select_backend
    d = _read(out/'decisions.csv.gz')
    rows, selected = select_backend(d, prepared)
    if prepared is None or prepared.get('sample_mode','all') == 'all':
        return
    if not len(selected): raise ValueError('No supported 15-Hz control-pattern decisions')
    endpoints = np.concatenate([np.arange(int(r.start),int(r.end)) for r in selected.itertuples()])
    e = _read(out/'computations.csv.gz').iloc[endpoints].copy().reset_index(drop=True)
    counts = (selected.end-selected.start).to_numpy(int)
    selected['row'] = np.arange(len(selected))
    selected['end'] = np.cumsum(counts)
    selected['start'] = selected.end-counts
    e['row'] = np.repeat(selected.row.to_numpy(), counts)
    selected = native.timeline_segments(selected)
    selected.to_csv(out/'decisions.csv.gz',index=False)
    e.to_csv(out/'computations.csv.gz',index=False)
    for name in ('actor_features','policies','actual_policy','screen','tree_summary','screens'):
        arr = np.load(out/(name+'.npy'))
        ids = endpoints if name in ('actor_features','policies') else rows
        np.save(out/(name+'.npy'),arr[ids])
    metadata = native.read_json(out/'extraction.json')
    metadata.update(sample_mode='15hz', full_decisions=len(d), n=len(selected), endpoints=len(e))
    native.write_json(out/'extraction.json',metadata)


def analyze(args, subject, game, prepared, workdir):
    """Preserve original computations; flatten only explicitly selected results."""
    original = args
    args = settings(args, subject)
    game = GAMES[int(game)] if isinstance(game, (int, np.integer)) else str(game).lower()
    if game not in native.GAME_IDS:
        raise ValueError('Unknown game: ' + game)
    out = Path(workdir)
    out.mkdir(parents=True, exist_ok=True)
    native.audit_game(args, game, out)
    audit = native.read_json(out/'audit.json')
    # Actor checkpoints also pickle a return-stat buffer. Load its owning frozen
    # module before extract_game/representation_game unpickle the linear heads,
    # including when the shared VoC cache was reused in this fresh process.
    native.frozen_import(Path(audit['checkpoint']).parents[3], 'buffer')
    reused = _candidate_path(original, prepared, int(subject), game)
    reuse = _reuse_extract(reused, out, audit, args.smoke) if reused else None
    if reused is None:
        native.extract_game(args, game, out)
    _select_rich(out, prepared)
    decisions = _read(out/'decisions.csv.gz')
    if sorted(decisions.session.unique()) != [1, 2, 3, 4]:
        raise ValueError('Full control-pattern analysis requires all four sessions')
    for name in ('representation', 'temporal', 'motifs', 'validate'):
        native.log(f'09/10 adapter: {game} sub{int(subject):03d} {name}')
        getattr(backend, name+'_game')(args, game, out)
    tables = {}
    for name in FINAL_TABLES:
        plain, compressed = out/(name+'.csv'), out/(name+'.csv.gz')
        tables[name] = _read(compressed if compressed.exists() else plain)
    tables.update(_nested_tables(out))
    plot_tables, arrays = collect_plot_sources(args, out, tables)
    tables.update(plot_tables)
    if args.retain_bases:
        for path in sorted((out/'representation').glob('train_*/*_basis.npz')):
            key = 'basis_'+path.parent.name+'_'+path.stem.removesuffix('_basis')
            with np.load(path, allow_pickle=False) as archive:
                arrays[key] = {name: np.array(archive[name]) for name in archive.files}
    for frame in tables.values():
        frame['subject'] = int(subject)
        frame['game'] = game
    qa = dict(subject=int(subject), game=game, status='COMPLETED', version=backend.VERSION,
        settings=native.clean_json(vars(args)), extraction=native.read_json(out/'extraction.json'),
        temporal=native.read_json(out/'temporal.json'), validation=native.read_json(out/'validation.json'),
        reuse=reuse, decisions=len(tables['decisions']), computation_endpoints=len(tables['computations']),
        windows=len(tables['windows']), raw_switches=len(tables['switches']),
        preserved_methods=['representation_game', 'temporal_game', 'motifs_game', 'validate_game'],
        basis_arrays_retained=bool(args.retain_bases),
        limitations=['Candidate labels are fold-specific evaluation patterns, not inferred human goals.',
            'Analysis fits hold out sessions; Thinker trained on all four human sessions.',
            'Late computation endpoints condition on searches that continued.',
            'ACE axis signs are fixed using training features, never held-out switch labels.',
            'Q revisions exclude unvisited branches; absence is not zero utility.',
            'All comparisons are observational; no forced-budget causal intervention.'])
    return tables, qa, arrays


def render(root, tables, title):
    """Render original panels using published CSVs and selected montage NPZ."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    figures = root/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    written = []
    def table(name):
        return tables.get(name, _empty())
    def save(figure, name):
        figure.suptitle(title, fontsize=12)
        figure.tight_layout()
        for extension in ('png', 'pdf'):
            path = figures/(name+'.'+extension)
            figure.savefig(path, dpi=160)
            written.append(path)
        plt.close(figure)
    with plt.rc_context({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'}):
        examples = table('timeline_examples')
        if len(examples):
            with np.load(root/'tables/timeline_montages.npz', allow_pickle=False) as images:
                for row in examples.itertuples():
                    traces = table('timeline_trajectories')
                    traces = traces[traces.example.eq(row.example)].sort_values('relative_time')
                    policy = table('timeline_policy')
                    policy = policy[policy.example.eq(row.example)].sort_values('relative_time')
                    candidate = table('timeline_candidates')
                    candidate = candidate[candidate.example.eq(row.example)]
                    fig, axes = plt.subplots(7, 1, figsize=(10, 10),
                        gridspec_kw={'height_ratios': [1.5, 1, 1, 1, 1, 1, 1]})
                    axes[0].imshow(images[row.example])
                    axes[0].axis('off')
                    axes[0].set_title(f'Held session {row.held}: {row.view}, C{row.state+1}, {row.kind}; t={row.time:.2f}s')
                    axes[1].step(traces.relative_time, traces.human_action, where='post')
                    axes[1].set_ylabel('Human action')
                    axes[2].step(traces.relative_time, traces.K, where='post')
                    axes[2].set_ylabel('Computations')
                    axes[3].plot(traces.relative_time, traces.D, label='signed net')
                    axes[3].plot(traces.relative_time, traces.U, label='absolute / K')
                    axes[3].legend()
                    axes[3].set_ylabel('Value update')
                    axes[4].plot(traces.relative_time, traces.voc_initial, label='initial')
                    axes[4].plot(traces.relative_time, traces.voc_final, label='final')
                    axes[4].legend()
                    axes[4].set_ylabel('EMA VoC')
                    columns = sorted([c for c in policy if c.startswith('p') and c[1:].isdigit()], key=lambda c: int(c[1:]))
                    axes[5].imshow(policy[columns].to_numpy().T, aspect='auto', origin='lower',
                        extent=[policy.relative_time.iloc[0], policy.relative_time.iloc[-1], -.5, len(columns)-.5], vmin=0, vmax=1)
                    axes[5].set_ylabel('Real policy')
                    for number, curve in candidate.groupby('candidate'):
                        axes[6].plot(curve.relative_time, curve.probability, label=f'C{number+1}')
                    axes[6].legend(ncol=4)
                    axes[6].set(ylabel='Candidate P', xlabel='Seconds from selected window endpoint')
                    for ax in axes[1:]:
                        ax.axvline(0, color='gray', ls=':', lw=.7)
                    save(fig, row.figure)
        curves = table('computation_curves')
        stopping = table('stopping_curves')
        if len(curves):
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            for column, metric in enumerate(('value', 'voc_ema', 'policy_change')):
                for session, frame in curves[curves.metric.eq(metric)].groupby('session'):
                    frame = frame.sort_values('k')
                    axes[0, column].plot(frame.k, frame['mean'], label=f'Session {session}')
                axes[0, column].set(xlabel='Completed computations k', ylabel=metric)
            axes[0, 0].legend()
            for column, metric in enumerate(('voc_ema', 'delta_V', 'gate_probability')):
                frame = stopping[stopping.metric.eq(metric)] if len(stopping) else _empty()
                if len(frame):
                    axes[1, column].plot(frame.x, frame.y, 'o-')
                axes[1, column].set(xlabel=metric, ylabel='Observed voluntary continue')
            save(fig, '02_computation_and_stopping')
        curves = table('switch_curves')
        if len(curves):
            sessions = sorted(curves.held.unique())
            fig, axes = plt.subplots(len(sessions), len(ALIGN_METRICS), figsize=(19, 2.5*len(sessions)), sharex=True, squeeze=False)
            for row, held in enumerate(sessions):
                for column, metric in enumerate(ALIGN_METRICS):
                    frame = curves[curves.held.eq(held) & curves.metric.eq(metric)].sort_values('lag')
                    axes[row, column].plot(frame.lag, frame['mean'], label='human switch')
                    axes[row, column].plot(frame.lag, frame.shifted, ls='--', label='time-shifted')
                    axes[row, column].axvline(0, color='gray', ls=':')
                    if row == 0:
                        axes[row, column].set_title(metric)
                    if column == 0:
                        axes[row, column].set_ylabel(f'Held session {held}')
                    if row == len(sessions)-1:
                        axes[row, column].set_xlabel('Seconds')
            axes[0, 0].legend(fontsize=7)
            save(fig, '03_all_switches')
        curves = table('transition_curves')
        if len(curves):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, metric in zip(axes, ('vp_value_ACE1', 'actor_policy_ACE1')):
                for key, frame in curves[curves.metric.eq(metric)].groupby(['action_from', 'action_to']):
                    frame = frame.sort_values('lag')
                    ax.plot(frame.lag, frame['mean'], label=f'{key[0]}>{key[1]}')
                ax.axvline(0, color='gray', ls=':')
                ax.set(title=metric+', held session 4', xlabel='Seconds')
                ax.legend(fontsize=6, ncol=3)
            save(fig, '04_transition_directions')
        profiles = table('profile_display')
        if len(profiles):
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            for ax, view in zip(axes, ('evaluation', 'joint')):
                frame = profiles[profiles.view.eq(view)]
                if frame.empty:
                    ax.text(.5, .5, 'No supported candidates', ha='center')
                    ax.set_axis_off()
                    continue
                matrix = frame.pivot(index='state', columns='metric', values='standardized_mean').reindex(columns=PROFILE_METRICS)
                image = ax.imshow(matrix, aspect='auto', cmap='coolwarm', vmin=-2, vmax=2)
                ax.set_xticks(range(len(PROFILE_METRICS)), PROFILE_METRICS, rotation=45, ha='right')
                ax.set_yticks(range(len(matrix)), [f'C{i+1}' for i in matrix.index])
                ax.set_title(view+' (held session 4)')
                fig.colorbar(image, ax=ax, label='Training SD')
            save(fig, '05_candidate_profiles')
        comparisons = table('comparisons')
        if len(comparisons):
            frame = comparisons[comparisons.width.eq(.5) & comparisons.horizon.eq(.25) &
                                comparisons.block_seconds.eq(10) & comparisons.target.eq('switch')]
            if len(frame):
                fig, ax = plt.subplots(figsize=(12, 5))
                names = list(frame.model.unique())
                for held, rows in frame.groupby('held'):
                    x = np.array([names.index(name) for name in rows.model]) + (held-2.5)*.12
                    ax.errorbar(x, rows['mean'], yerr=[np.maximum(0, rows['mean']-rows.low),
                        np.maximum(0, rows.high-rows['mean'])], fmt='o', label=f'Session {held}', capsize=2)
                ax.axhline(0, color='gray')
                ax.set_xticks(range(len(names)), names, rotation=35, ha='right')
                ax.set_ylabel('Held-out log-loss gain over listed parent')
                ax.legend()
                save(fig, '06_heldout_comparisons')
        examples = table('control_examples')
        traces = table('control_example_trajectories')
        if len(examples):
            fig, axes = plt.subplots(len(examples), 3, figsize=(13, 3*len(examples)), squeeze=False)
            for row, metadata in enumerate(examples.itertuples()):
                if not metadata.supported:
                    for ax in axes[row]:
                        ax.text(.5, .5, 'No supported example', ha='center')
                        ax.set_axis_off()
                    continue
                frame = traces[traces.example.eq(metadata.example)].sort_values('relative_time')
                axes[row, 0].step(frame.relative_time, frame.human_action, where='post')
                axes[row, 0].set(title=metadata.label, ylabel='Human action')
                axes[row, 1].step(frame.relative_time, frame.state+1, where='post')
                axes[row, 1].set_ylabel('Candidate')
                axes[row, 2].plot(frame.relative_time, frame.U)
                axes[row, 2].set_ylabel('Value update / K')
            save(fig, '07_within_across_action_examples')
    return written


def render_group(root, tables, title):
    """Saved-table-only group panels; fitted coordinates and labels stay local.

    Only K, U, D and policy-change curves have common scalar definitions. These
    are summarized with equal held sessions within participant, then equal
    participants. ACE coordinates are shown separately for each fitted fold;
    candidate identities are shown separately for subject/held session/view.
    No basis alignment, clustering, montage IO or model fitting is performed.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);folder=root/'figures';folder.mkdir(parents=True,exist_ok=True)
    written=[]
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            path=folder/(name+'.'+ext);fig.savefig(path,dpi=170,bbox_inches='tight');written.append(path)
        plt.close(fig)
    curves=tables.get('switch_curves',_empty())
    if len(curves):
        required={'subject','held','metric','lag','mean','shifted'}
        if not required.issubset(curves):raise ValueError('Group switch curves lack '+str(sorted(required-set(curves))))
        scalar=curves.loc[curves.metric.isin(ALIGN_METRICS[:4])].copy()
        if len(scalar):
            participant=scalar.groupby(['subject','metric','lag'],dropna=False)[['mean','shifted']].mean().reset_index()
            group_scalar=participant.groupby(['metric','lag'])[['mean','shifted']].mean().reset_index()
            group_scalar['n_subjects']=participant.groupby(['metric','lag']).subject.nunique().to_numpy()
            if 'control_switch_scalar_participant' in tables:
                participant=tables['control_switch_scalar_participant'].copy()
            if 'control_switch_scalar_group' in tables:
                group_scalar=tables['control_switch_scalar_group'].copy()
            fig,axes=plt.subplots(1,4,figsize=(16,4),sharex=True)
            colors=plt.get_cmap('tab10')
            subjects=sorted(participant.subject.unique())
            for ax,metric in zip(axes,ALIGN_METRICS[:4]):
                d=participant.loc[participant.metric.eq(metric)]
                for i,subject in enumerate(subjects):
                    z=d.loc[d.subject.eq(subject)].sort_values('lag')
                    ax.plot(z.lag,z['mean'],color=colors(i%10),alpha=.65,lw=1,label=f'sub{int(subject):03d}')
                    ax.plot(z.lag,z.shifted,color=colors(i%10),alpha=.35,lw=.8,ls='--')
                summary=group_scalar.loc[group_scalar.metric.eq(metric)].sort_values('lag')
                ax.plot(summary.lag,summary['mean'],color='black',lw=2,label='Equal participant mean')
                ax.plot(summary.lag,summary.shifted,color='black',lw=1.5,ls='--',label='Time-shifted mean')
                ax.axvline(0,color='gray',ls=':',lw=.8)
                ax.set(title=metric,xlabel='Seconds from human action switch',ylabel='Recorded scalar value')
            axes[0].legend(fontsize=7)
            fig.suptitle(title+' · shared scalar definitions\nEqual held sessions within participant, then equal participants; value scales may differ across trained models',fontsize=11)
            save(fig,'control_switch_scalar_group')
        ace=curves.loc[curves.metric.isin(ALIGN_METRICS[4:])].copy()
        if len(ace):
            for subject,d in ace.groupby('subject',sort=True):
                helds=sorted(d.held.unique());fig,axes=plt.subplots(len(helds),3,figsize=(12,2.6*len(helds)),squeeze=False,sharex=True)
                for row,held in enumerate(helds):
                    for col,metric in enumerate(ALIGN_METRICS[4:]):
                        z=d.loc[d.held.eq(held)&d.metric.eq(metric)].sort_values('lag')
                        if z.duplicated('lag').any():raise ValueError('ACE curves must have one estimate per participant/fold/metric/lag')
                        ax=axes[row,col];ax.plot(z.lag,z['mean'],label='Human switch')
                        ax.plot(z.lag,z.shifted,ls='--',label='Time shifted');ax.axvline(0,color='gray',ls=':',lw=.8)
                        if row==0:ax.set_title(metric)
                        if col==0:ax.set_ylabel(f'Held session {int(held)}\nLocal ACE coordinate')
                        if row==len(helds)-1:ax.set_xlabel('Seconds from human action switch')
                axes[0,0].legend(fontsize=7)
                fig.suptitle(f'{title} · sub{int(subject):03d}\nEach held-session basis shown independently; no averaging or axis alignment',fontsize=11)
                save(fig,f'control_ace_sub{int(subject):03d}_separate_folds')
    profiles=tables.get('profile_display',_empty())
    if len(profiles):
        required={'subject','held','view','state','metric','standardized_mean'}
        if not required.issubset(profiles):raise ValueError('Group profiles lack '+str(sorted(required-set(profiles))))
        subsets=['held']+(['width'] if 'width' in profiles else [])
        grouped=profiles.groupby(subsets,dropna=False,sort=True)
        for key,d in grouped:
            key=key if isinstance(key,tuple) else (key,)
            held=int(key[0]);subjects=sorted(d.subject.unique());views=sorted(d.view.unique())
            fig,axes=plt.subplots(len(subjects),len(views),figsize=(6.4*len(views),3.4*len(subjects)),squeeze=False)
            for row,subject in enumerate(subjects):
                for col,view in enumerate(views):
                    z=d.loc[d.subject.eq(subject)&d.view.eq(view)];ax=axes[row,col]
                    if not len(z):ax.set_axis_off();continue
                    if z.duplicated(['state','metric']).any():raise ValueError('Candidate profiles cannot mix fitted bases or folds')
                    matrix=z.pivot(index='state',columns='metric',values='standardized_mean').reindex(columns=PROFILE_METRICS)
                    image=ax.imshow(np.ma.masked_invalid(matrix.to_numpy()),aspect='auto',cmap='coolwarm',vmin=-2,vmax=2)
                    ax.set(xticks=np.arange(len(matrix.columns)),xticklabels=matrix.columns,
                        yticks=np.arange(len(matrix)),yticklabels=[f'Local C{int(k)+1}' for k in matrix.index],
                        title=f'sub{int(subject):03d} · {view}')
                    ax.tick_params(axis='x',rotation=45);fig.colorbar(image,ax=ax,label='Training SD')
            suffix=f'held{held}'+(f'_w{key[1]:g}' if len(key)>1 else '')
            fig.suptitle(f'{title} · held session {held}\nCandidate labels are local to each participant and view; row identities are not matched',fontsize=11)
            save(fig,'control_candidate_profiles_'+suffix)
    examples=tables.get('control_examples',_empty());traces=tables.get('control_example_trajectories',_empty())
    if len(examples) and len(traces):
        for subject,metadata in examples.groupby('subject',sort=True):
            fig,axes=plt.subplots(len(metadata),3,figsize=(13,3*len(metadata)),squeeze=False)
            for row,item in enumerate(metadata.itertuples()):
                frame=traces.loc[traces.subject.eq(subject)&traces.example.eq(item.example)].sort_values('relative_time')
                if not bool(item.supported) or not len(frame):
                    for ax in axes[row]:ax.text(.5,.5,'No supported example',ha='center');ax.set_axis_off()
                    continue
                axes[row,0].step(frame.relative_time,frame.human_action,where='post');axes[row,0].set(title=item.label,ylabel='Human action')
                axes[row,1].step(frame.relative_time,frame.state+1,where='post');axes[row,1].set_ylabel('Local candidate')
                axes[row,2].plot(frame.relative_time,frame.U);axes[row,2].set_ylabel('Value update / K')
                for ax in axes[row]:ax.axvline(0,color='gray',ls=':',lw=.8);ax.set_xlabel('Seconds from selected endpoint')
            fig.suptitle(f'{title} · sub{int(subject):03d} · descriptive examples\nCandidate identity remains specific to this participant and fitted held session',fontsize=11)
            save(fig,f'control_within_across_action_sub{int(subject):03d}')
    return written
