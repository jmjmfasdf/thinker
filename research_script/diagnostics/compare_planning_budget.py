"""Compare completed planning stages at identical physical environment counts.

Uses differences of cumulative budget histograms, never unweighted averages of
log rows. The first-query-adjusted metric is E[max(B-1, 0)] for EVERY run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


HIST = [f'collector/count_{k}_decisions' for k in range(31)]
STEP = 'collector/physical_env_steps'


def load_prefix(directory: Path, limit: int) -> pd.DataFrame:
    columns = ['real_step', STEP, 'collector/imagined_transitions', *HIST]
    frames = []
    for chunk in pd.read_csv(directory/'logs.csv', comment='#', usecols=columns,
                             chunksize=1000):
        frames.append(chunk[chunk[STEP] <= limit])
        if (chunk[STEP] > limit).any():
            break
    frame = (pd.concat(frames).dropna(subset=columns)
             .drop_duplicates(STEP, keep='last').set_index(STEP).sort_index())
    hist = frame[HIST].to_numpy()
    if not np.array_equal(hist.sum(1), frame.index.to_numpy()):
        raise ValueError(f'Histogram/physical action mismatch: {directory}')
    if not np.array_equal(hist @ np.arange(31), frame['collector/imagined_transitions']):
        raise ValueError(f'Histogram/imagination mismatch: {directory}')
    if np.any(np.diff(hist, axis=0) < 0):
        raise ValueError(f'Counter reset within comparison prefix: {directory}')
    return frame


def score(frame: pd.DataFrame, lo: int, hi: int) -> dict:
    h = frame.loc[hi, HIST].to_numpy(float)
    if lo:
        h = h - frame.loc[lo, HIST].to_numpy(float)
    n = h.sum()
    if n != hi-lo or np.any(h < 0):
        raise ValueError('Invalid matched interval')
    k = np.arange(31)
    return dict(start_env_step=lo, end_env_step=hi, decisions=int(n),
                mean_budget=float(h @ k/n),
                mean_queries_beyond_first=float(h @ np.maximum(k-1, 0)/n),
                zero_fraction=float(h[0]/n), one_fraction=float(h[1]/n),
                two_plus_fraction=float(h[2:].sum()/n),
                cap_fraction=float(h[-1]/n),
                cap_share_of_all_queries=float(30*h[-1]/(h @ k)) if h @ k else 0.)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--current', type=Path, required=True)
    ap.add_argument('--baselines', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-steps', type=int, default=350_000)
    ap.add_argument('--recent-steps', type=int, default=100_000)
    ap.add_argument('--bin-steps', type=int, default=50_000)
    args = ap.parse_args()
    if min(args.max_steps, args.recent_steps, args.bin_steps) <= 0:
        ap.error('step counts must be positive')
    paths = [args.current, *args.baselines]
    frames = {p.name: load_prefix(p, args.max_steps) for p in paths}
    common = set.intersection(*(set(f.index) for f in frames.values()))
    end = int(max(common))
    start = int(max(s for s in common if s <= end-args.recent_steps))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in paths:
        config = yaml.safe_load((p/'config_c.yaml').read_text())
        metadata = dict(run=p.name, kind='current' if p == args.current else 'previous',
                        holdout_sessions=config.get('icopro_holdout_sessions'),
                        training_sessions=config.get('icopro_train_sessions'),
                        temperature=config.get('voc_gate_q_temperature'),
                        min_search_steps=config.get('min_search_steps', 0),
                        think_cost=config.get('think_cost'),
                        gru_connected=config.get('voc_gru_connected', False),
                        gru_reset=config.get('dynamic_search_state_reset', 'observation'))
        for label, lo in [('cumulative', 0), ('recent', start)]:
            rows.append(dict(**metadata, interval=label, **score(frames[p.name], lo, end)))
        frames[p.name].to_csv(args.output/(p.name+'_counters.csv.gz'))
    result = pd.DataFrame(rows)
    result.to_csv(args.output/'run_comparison.csv', index=False)
    metrics = ['mean_budget', 'mean_queries_beyond_first', 'zero_fraction',
               'one_fraction', 'two_plus_fraction', 'cap_fraction', 'cap_share_of_all_queries']
    summary = result.groupby(['interval', 'kind'])[metrics].agg(['mean', 'min', 'max'])
    summary.to_csv(args.output/'comparison_summary.csv')

    # Bin edges are observed counts shared by all runs; no interpolation.
    edges = sorted({0, end, *(int(max(s for s in common if s <= t))
                             for t in range(args.bin_steps, end, args.bin_steps))})
    trace = pd.DataFrame([dict(run=name, **score(f, lo, hi))
                          for name, f in frames.items()
                          for lo, hi in zip(edges[:-1], edges[1:])])
    trace.to_csv(args.output/'planning_curve.csv', index=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharex=True)
    for ax, metric, title in zip(axes, metrics[:2],
                                 ['All imagined transitions', 'Queries after the first']):
        old = trace[trace.run != args.current.name].groupby('end_env_step')[metric]
        x = old.mean().index.to_numpy()/1e6
        ax.fill_between(x, old.min(), old.max(), color='#777777', alpha=.17,
                        label='Previous four folds: range')
        ax.plot(x, old.mean(), color='#6B6B6B', marker='o', ms=4,
                label='Previous four folds: mean')
        new = trace[trace.run == args.current.name]
        ax.plot(new.end_env_step/1e6, new[metric], color='#186CA0', marker='o', ms=4,
                label='Current: min 1, cost 0.0001')
        ax.set(title=title, xlabel='Training environment actions (million)',
               ylabel='Mean queries per real decision', ylim=(0, None))
        ax.grid(axis='y', alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
               loc='upper center', bbox_to_anchor=(.5, .925))
    fig.suptitle('Enduro, subject 001: matched training intervals')
    fig.text(.5, .01, 'Interval means from completed-decision counters. Previous folds are not independent seeds.',
             ha='center', fontsize=8, color='#555555')
    fig.tight_layout(rect=[0, .04, 1, .86])
    fig.savefig(args.output/'planning_comparison.png', dpi=180)
    fig.savefig(args.output/'planning_comparison.pdf')
    plt.close(fig)
    manifest = dict(current=str(args.current.resolve()), baselines=[str(p.resolve()) for p in args.baselines],
                    end_env_step=end, recent_start_env_step=start, plot_bin_steps=args.bin_steps,
                    unit='imagined transitions per real environment action during training',
                    method='Differences of cumulative histograms at identical observed physical action counts; equal weight per previous fold.',
                    first_query_adjustment='E[max(B-1,0)] in every run; not a causal subtraction of the intervention.',
                    caveats=['Different human training sessions, cost, minimum budget and memory-reset policy.',
                             'Training-policy allocation, not matched-state evaluation or planning utility.',
                             'Current run is early in 10M training; four previous folds share subject and seed.'],
                    qa=dict(histogram_sums_match_physical_actions=True,
                            weighted_histograms_match_query_counter=True, monotone_counters=True,
                            matched_interval_boundaries=True))
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(result[['run', 'interval', *metrics]].to_string(index=False))


if __name__ == '__main__':
    main()
