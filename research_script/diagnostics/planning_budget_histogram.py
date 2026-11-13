"""Plot decision-weighted planning counts between two cumulative snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--counter-history', type=Path, required=True)
    parser.add_argument('--start-step', type=int, default=6_000_000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Fail rather than replace a completed analysis or its source snapshots.
    args.output.mkdir(parents=True, exist_ok=False)
    end_bytes = (args.run / 'collector_metrics.json').read_bytes()
    end = json.loads(end_bytes)
    (args.output / 'end_collector_metrics.json').write_bytes(end_bytes)
    history = pd.read_csv(args.counter_history)
    step_key = 'collector/physical_env_steps'
    keys = [f'count_{k}_decisions' for k in range(31)]
    columns = [f'collector/{key}' for key in keys]
    valid = history.dropna(subset=[step_key, 'collector/imagined_transitions', *columns])
    valid = valid.sort_values(step_key).drop_duplicates(step_key, keep='last')
    candidates = valid[valid[step_key] >= args.start_step]
    if candidates.empty:
        raise ValueError('History contains no snapshot at or after the requested boundary')
    row = candidates.iloc[0]
    start = dict(physical_env_steps=int(row[step_key]),
                 imagined_transitions=int(row['collector/imagined_transitions']),
                 **{key: int(row[f'collector/{key}']) for key in keys})
    (args.output / 'start_counters.json').write_text(json.dumps(start, indent=2))
    k = np.arange(31, dtype=np.int64)
    for snapshot in (start, end):
        values = np.array([snapshot[key] for key in keys], dtype=np.int64)
        if values.sum() != snapshot['physical_env_steps']:
            raise ValueError('Histogram sum differs from real decision count')
        if values @ k != snapshot['imagined_transitions']:
            raise ValueError('Weighted histogram differs from imagined transition count')
    counts = np.array([end[key] - start[key] for key in keys], dtype=np.int64)
    n = end['physical_env_steps'] - start['physical_env_steps']
    if n <= 0 or (counts < 0).any() or counts.sum() != n:
        raise ValueError('Invalid interval or nonmonotone counters')
    percentage = 100 * counts / n
    if not np.isclose(percentage.sum(), 100):
        raise ValueError('Percentages do not sum to 100')
    table = pd.DataFrame(dict(imagination_count=k, decisions=counts, percentage=percentage))
    table.to_csv(args.output / 'imagination_histogram.csv', index=False)
    summary = dict(start_env_step=start['physical_env_steps'],
                   end_env_step=end['physical_env_steps'], decisions=int(n),
                   mean_imagination_count=float(counts @ k / n),
                   zero_percentage=float(percentage[0]),
                   one_percentage=float(percentage[1]),
                   two_plus_percentage=float(percentage[2:].sum()),
                   cap30_percentage=float(percentage[-1]))
    manifest = dict(
        run=str(args.run.resolve()), requested_start_step=args.start_step,
        counter_history=str(args.counter_history.resolve()),
        history_sha256=hashlib.sha256(args.counter_history.read_bytes()).hexdigest(),
        end_snapshot_sha256=hashlib.sha256(end_bytes).hexdigest(),
        end_snapshot_updated_at_utc=datetime.fromtimestamp(end['updated_at'], timezone.utc).isoformat(),
        summary=summary,
        definition='Total imagined transitions before a completed real action, including the mandatory first transition; not search depth.',
        boundary_rule='Subtract the first recorded cumulative snapshot at or after the requested start from the frozen latest snapshot. No interpolation.',
        skipped_boundary_actions=start['physical_env_steps'] - args.start_step,
        denominator='All completed real decisions in this interval, identical in both panels.',
        qa=dict(histogram_sum_matches_decisions=True, weighted_sum_matches_imagined_transitions=True,
                nonnegative_interval_counts=True, percentages_sum_to_100=True))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.2))
    colors = ['#397ea8'] * 31
    colors[1] = '#7495a9'
    colors[30] = '#d58135'
    for ax, first, title in zip(axes, [0, 2],
                                ['All decisions: 0–30 imagined transitions',
                                 'Detail: 2–30 imagined transitions (same denominator)']):
        ax.bar(k[first:], percentage[first:], width=.8, color=colors[first:], zorder=3)
        ax.set(xticks=k[first:], xlim=(first-.65, 30.65),
               ylabel='Percentage of all decisions (%)', title=title)
        ax.grid(axis='y', alpha=.18, zorder=0)
        ax.tick_params(axis='x', labelsize=9)
        maximum = percentage[first:].max()
        ax.set_ylim(0, maximum * 1.17)
        labelled = [0, 1, 30] if first == 0 else list(range(2, 31))
        for i in labelled:
            ax.text(i, percentage[i] + maximum*.025, f'{percentage[i]:.2f}',
                    ha='center', va='bottom', fontsize=8)
    axes[1].set_xlabel('Imagined transitions before one real action (30 = maximum allowed)')
    axes[0].text(.97, .83,
                 f'Mean = {summary["mean_imagination_count"]:.3f}\n'
                 f'At least 2 transitions = {summary["two_plus_percentage"]:.2f}%',
                 transform=axes[0].transAxes, ha='right', va='top', fontsize=12)
    fig.suptitle('Enduro · subject 001 · imagination count after 6M training actions', fontsize=15)
    fig.text(.5, .925,
             f'{start["physical_env_steps"]:,} → {end["physical_env_steps"]:,} environment actions'
             f'  |  N = {n:,} real decisions', ha='center', color='#444444')
    fig.text(.5, .015, 'Minimum 1 transition · cost 0.0001 · temperature 0.01 · training-policy data',
             ha='center', color='#555555', fontsize=10)
    fig.tight_layout(rect=[0, .045, 1, .90], h_pad=2.5)
    fig.savefig(args.output / 'imagination_histogram.png', dpi=180)
    fig.savefig(args.output / 'imagination_histogram.pdf')
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
