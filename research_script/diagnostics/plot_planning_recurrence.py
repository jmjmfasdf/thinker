"""Plot paired, survivor-matched readout sensitivity without causal utility claims."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    reports = {r['condition']: r for r in evidence['reports']}
    conditions = ['control', 'burn8', 'gru10']
    if any(reports[c]['status'] != 'COMPLETE' for c in conditions):
        raise ValueError('all three complete paired diagnostics are required')
    args.output.mkdir(parents=True, exist_ok=False)
    titles = ['Original critic, burn-in 1', 'Revised critic, burn-in 8',
              'Revised critic, burn-in 8, GRU gradient ×10']
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey='row')
    snapshots = []
    for column, (condition, title) in enumerate(zip(conditions, titles)):
        report = reports[condition]
        rows = report['by_count']
        counts = [r['count'] for r in rows]
        for key, label, color, style in [
            ('actual_mean_probability', 'Observed readout', '#202020', '-'),
            ('first_probability_same_survivors', 'First query: same surviving decisions', '#888888', ':'),
            ('freeze_progress_mean_probability', 'Progress features held at first query', '#c86720', '--'),
        ]:
            axes[0, column].plot(counts, [r[key] for r in rows], label=label,
                                 color=color, linestyle=style, linewidth=1.8)
        for key, label, color, style in [
            ('actual_paired_log_odds_change', 'Observed readout', '#202020', '-'),
            ('freeze_progress_paired_log_odds_change', 'Progress features held at first query', '#c86720', '--'),
            ('repeated_token_paired_log_odds_change', 'First tree token repeated', '#237eab', '-.'),
        ]:
            axes[1, column].plot(counts, [r[key] for r in rows], label=label,
                                 color=color, linestyle=style, linewidth=1.8)
        axes[0, column].set_title(title, fontsize=10)
        axes[0, column].set_ylim(-.03, 1.03)
        axes[1, column].axhline(0, color='#aaaaaa', linewidth=.8)
        axes[1, column].set_xlabel('Completed imagination steps')
        for ax in axes[:, column]:
            ax.set_xticks([1, 8, 16, 24, 29])
            ax.grid(alpha=.2)
        final = rows[-1]
        axes[0, column].text(.04, .05,
            f"Decisions at query 1: {rows[0]['at_risk_decisions']:,}\n"
            f"Decisions at query 29: {final['at_risk_decisions']:,}",
            transform=axes[0, column].transAxes, fontsize=8)
        snapshots.append(dict(condition=condition, final_count=final,
            first_query_online_gate_sign=report['first_query_online_gate_sign']))
    axes[0, 0].set_ylabel('Mean probability of continuing')
    axes[1, 0].set_ylabel('Change in log-odds from first query\n(same surviving decisions)')
    axes[0, 0].legend(fontsize=7, loc='center right')
    axes[1, 0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Enduro, 250k: planning readout sensitivity', fontsize=14)
    fig.text(.5, .015,
        'One training seed; eight evaluation streams. Each query conditions on reaching it.\n'
        'Feature manipulations keep the observed trajectory fixed; they are not executed policies or utility estimates.',
        ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/('paired_readout_sensitivity.'+suffix), dpi=180)
    plt.close(fig)
    (args.output/'evidence.json').write_text(json.dumps(dict(
        source=str(args.evidence.resolve()),
        source_sha256=hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
        snapshots=snapshots, limitations=evidence['limitations'],
    ), indent=2)+'\n')
    shutil.copy2(__file__, args.output/Path(__file__).name)
    print(args.output/'paired_readout_sensitivity.png')


if __name__ == '__main__':
    main()
