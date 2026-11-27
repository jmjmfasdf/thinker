"""Analysis-only sampling. Never changes replay, model state, or raw outcomes."""
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = 'causal-15hz-unique-v1'
KEYS = ['episode', 'real_event']
RULE = dict(version=VERSION, hz=15., origin='first recorded observation of each episode',
    selection='latest observation at or before each grid time; no interpolation or future frames',
    duplicates='retain grid mapping; use each selected real decision once in analysis',
    boundaries='restart grid at each episode; never cross reset/terminal/clock/gap boundaries',
    eligibility='apply scored/quality masks after selection; never substitute another eligible frame',
    interpretation='subsample of full replay, not a replay executed at 15 Hz')


def mode(args):
    value = getattr(args, 'sample_mode', 'all')
    if value not in ('all', '15hz'):
        raise ValueError('sample_mode must be all or 15hz')
    return value


def output_base(args):
    # Preserve direct legacy API callers; all numbered CLI calls are explicit.
    root = Path(args.outputs_root)
    if getattr(args, 'legacy_output_layout', False):
        if mode(args) != 'all':
            raise ValueError('Legacy output layout requires sample_mode all; sampled results must stay separate')
        return root
    return root/'samples'/mode(args) if hasattr(args, 'sample_mode') else root


def causal_grid(times, origin=None, hz=15.):
    """Return every grid time and its causal source row, including repeats."""
    times = np.asarray(times, float)
    if not len(times):
        return np.empty(0), np.empty(0, dtype=int)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Sampling requires a strictly increasing, finite episode clock')
    origin = times[0] if origin is None else float(origin)
    count = int(np.floor((times[-1]-origin)*hz + 1e-9)) + 1
    grid = origin + np.arange(max(0, count))/hz
    rows = np.searchsorted(times, grid, side='right')-1
    valid = rows >= 0
    return grid[valid], rows[valid]


def inventory(metadata):
    """All stored real transitions, including unscored initialization edges."""
    from analyses.replay_themes import PackedTrace
    frames = []
    for ep in metadata['episodes']:
        trace = PackedTrace(Path(metadata['capture'])/ep['path'])
        real = np.flatnonzero(trace.read('real_transition').reshape(-1))
        src = trace.read('pre_observation_source_index', real).reshape(-1).astype(int)
        with np.load(trace.meta['source_path'], allow_pickle=False) as raw:
            times = raw['time'].reshape(-1)[src].astype(float)
        frame = pd.DataFrame(dict(episode=trace.meta['episode_id'], run=trace.meta['source_id'],
            real_event=real, source_frame=src, time=times,
            scored=trace.read('score_mask', real).reshape(-1).astype(bool)))
        frame['sampling_origin'] = trace.meta.get('source_first_time', times[0])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def prepare_tables(decisions, choices, real_steps):
    """Keep the full tables, plus a grid-to-source audit and selected views."""
    d = decisions.copy()
    if d.duplicated(KEYS).any() or real_steps.duplicated(KEYS).any():
        raise ValueError('Nonunique real decision identity')
    maps = []
    for episode, part in real_steps.groupby('episode', sort=False):
        # Old 15-Hz captures can repeat raw frames: preserve their logical rows
        # but choose the last row available at a timestamp, without upsampling.
        part = part.drop_duplicates('time', keep='last').reset_index(drop=True)
        grid, rows = causal_grid(part.time, part.sampling_origin.iloc[0])
        selected = part.iloc[rows].copy().reset_index(drop=True)
        selected['grid_time'] = grid
        selected['grid_index'] = np.arange(len(grid))
        selected['lag_seconds'] = grid-selected.time.to_numpy()
        maps.append(selected)
    mapping = pd.concat(maps, ignore_index=True)
    unique = mapping.drop_duplicates(KEYS).copy()
    unique['sample_index_15hz'] = unique.groupby('episode', sort=False).cumcount()
    ordinal = unique.set_index(KEYS)['sample_index_15hz']
    d['sample_index_15hz'] = ordinal.reindex(pd.MultiIndex.from_frame(d[KEYS])).to_numpy()
    eligible = pd.MultiIndex.from_frame(d[KEYS])
    mapping['analysis_eligible'] = pd.MultiIndex.from_frame(mapping[KEYS]).isin(eligible)
    chosen = pd.MultiIndex.from_frame(mapping.loc[mapping.analysis_eligible, KEYS].drop_duplicates())
    d['selected_15hz'] = pd.MultiIndex.from_frame(d[KEYS]).isin(chosen)
    d['sample_mode'] = 'all'
    sampled = select_decisions(d, '15hz')
    e15 = select_choices(choices, sampled)
    return dict(decisions=d, choices=choices, real_steps=real_steps,
                sampling_15hz=mapping, decisions_15hz=sampled, choices_15hz=e15)


def select_decisions(d, selected_mode):
    result = d.loc[d.selected_15hz].copy() if selected_mode == '15hz' else d.copy()
    result['sample_mode'] = selected_mode
    result['sample_index'] = (result.sample_index_15hz.astype(int) if selected_mode == '15hz'
                              else result.groupby('episode', sort=False).cumcount())
    return result.reset_index(drop=True)


def select_choices(choices, decisions):
    keys = ['episode', 'decision']
    if choices.empty:
        return choices.copy()
    keep = pd.MultiIndex.from_frame(choices[keys]).isin(pd.MultiIndex.from_frame(decisions[keys]))
    return choices.loc[keep].copy().reset_index(drop=True)


def select_backend(d, prepared):
    """Align a backend's supported rows to 00, keeping missing-support gaps."""
    if prepared is None or prepared.get('sample_mode', 'all') == 'all':
        return np.arange(len(d)), d.copy()
    targets = prepared['decisions']
    wanted = pd.MultiIndex.from_frame(targets[KEYS])
    rows = np.flatnonzero(pd.MultiIndex.from_frame(d[KEYS]).isin(wanted))
    result = d.iloc[rows].copy()
    lookup = targets.set_index(KEYS)['sample_index']
    result['sample_index'] = lookup.reindex(pd.MultiIndex.from_frame(result[KEYS])).to_numpy()
    result['sample_mode'] = '15hz'
    return rows, result.reset_index(drop=True)


def sequence_index(d):
    return d['sample_index'] if 'sample_index' in d else d['decision']
