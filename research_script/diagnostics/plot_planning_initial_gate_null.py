"""Compare completed final allocations with a per-decision fixed-gate model.

The model holds each observed first optional probability fixed across queries.
It diagnoses within-planning probability changes, not reward usefulness. Missing
conditions remain visible; this report cannot stand in for a full comparison.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_planning_allocation_milestones import load_sample
from summarize_adaptive_planning_evidence import checked_probabilities, initial_gate_null


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    conditions = ('control', 'burn8', 'gru10')
    labels = ('Control', 'Burn-in 8', 'Burn-in 8 + GRU gradient x10')
    records, tables, histograms, missing = [], [], [], []
    for condition in conditions:
        root = campaign / ('allocation_1m_' + condition)
        if not (root / 'final_binding_audit.json').exists():
            missing.append(condition)
            continue
        verified = load_sample([root], condition, registered=True, final=True)
        rows = json.loads((root / (condition + '_allocation.json')).read_text())
        lengths = np.array([r['length'] for r in rows], dtype=int)
        probability, roundoff = checked_probabilities(
            [r['first_optional_probability'] for r in rows])
        # Exact capped geometric distribution: queries 1..29 can terminate;
        # reaching query 30 ends the stage regardless of the next gate draw.
        pmf = np.column_stack([
            probability ** (length - 1) * (1 - probability)
            for length in range(1, 30)
        ] + [probability ** 29])
        np.testing.assert_allclose(pmf.sum(axis=1), 1, rtol=0, atol=2e-14)
        predicted_mean, predicted_cap = initial_gate_null(probability, 1, 30)
        np.testing.assert_allclose(pmf @ np.arange(1, 31), predicted_mean,
                                   rtol=0, atol=2e-13)
        np.testing.assert_array_equal(pmf[:, -1], predicted_cap)
        clusters = sorted({(r['seed'], r['stream']) for r in rows})
        excess = []
        for seed, stream in clusters:
            use = np.array([r['seed'] == seed and r['stream'] == stream for r in rows])
            excess.append([(lengths[use] - predicted_mean[use]).mean(),
                           ((lengths[use] == 30) - predicted_cap[use]).mean()])
        rng = np.random.default_rng(20261114)
        boot = np.array(excess)[rng.integers(0, len(clusters), (4000, len(clusters)))].mean(axis=1)
        intervals = np.quantile(boot, [.025, .975], axis=0)
        observed_histogram = np.bincount(lengths, minlength=31)[1:] / len(rows)
        model_histogram = pmf.mean(axis=0)
        summary = dict(condition=condition, decisions=len(rows), clusters=len(clusters),
            actual_mean=float(lengths.mean()), fixed_gate_mean=float(predicted_mean.mean()),
            mean_excess=float((lengths - predicted_mean).mean()),
            mean_excess_ci_low=float(intervals[0, 0]), mean_excess_ci_high=float(intervals[1, 0]),
            actual_cap_fraction=float(np.mean(lengths == 30)),
            fixed_gate_cap_fraction=float(predicted_cap.mean()),
            cap_excess=float(np.mean((lengths == 30) - predicted_cap)),
            cap_excess_ci_low=float(intervals[0, 1]), cap_excess_ci_high=float(intervals[1, 1]),
            actual_continued_decisions=int(np.sum(lengths > 1)),
            actual_mean_given_continue=float(lengths[lengths > 1].mean()),
            fixed_gate_mean_given_continue=float(1 + (predicted_mean.mean() - 1) / probability.mean()))
        tables.append(summary)
        for length in range(1, 31):
            histograms.append(dict(condition=condition, length=length,
                actual_count=int(np.sum(lengths == length)),
                actual_fraction=float(observed_histogram[length - 1]),
                fixed_gate_fraction=float(model_histogram[length - 1])))
        records.append(dict(**summary, checkpoint=verified['checkpoint'],
            sources=verified['sources'], probability_roundoff=roundoff,
            actual_histogram=observed_histogram.tolist(),
            fixed_gate_histogram=model_histogram.tolist()))

    if not records:
        raise ValueError('No complete registered final allocations')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in [('summary.csv', tables), ('histograms.csv', histograms)]:
        with (args.output / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), sharey='row')
    for column, (condition, label) in enumerate(zip(conditions, labels)):
        axes[0, column].set_title(label)
        if condition in missing:
            for ax in axes[:, column]:
                ax.text(.5, .5, 'Evaluation pending', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
            continue
        row = next(r for r in records if r['condition'] == condition)
        actual, fixed = np.array(row['actual_histogram']), np.array(row['fixed_gate_histogram'])
        x = np.arange(1, 31)
        axes[0, column].bar(x - .2, actual, width=.4, color='#168d9a', label='Observed')
        axes[0, column].bar(x + .2, fixed, width=.4, color='#a2a8b1', label='First probability held fixed')
        axes[0, column].set_xlim(.5, 8.5)
        axes[0, column].set_xticks(range(1, 9))
        axes[0, column].set_ylim(0, 1)
        axes[0, column].set_xlabel('Imagination queries (first eight counts)')
        tail = x >= 2
        actual_tail = tail & (actual > 0)
        fixed_tail = tail & (fixed > 0)
        axes[1, column].plot(x[actual_tail], actual[actual_tail], 'o-', color='#168d9a', ms=3)
        axes[1, column].plot(x[fixed_tail], fixed[fixed_tail], '--', color='#727982', lw=1.5)
        axes[1, column].set_yscale('log')
        axes[1, column].set_ylim(1e-6, 1)
        axes[1, column].set_xticks([2, 5, 10, 15, 20, 25, 30])
        axes[1, column].set_xlim(1.5, 30.5)
        axes[1, column].set_xlabel('Imagination queries (2–30, log frequency)')
        for ax in axes[:, column]:
            ax.grid(axis='y', alpha=.15)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel('Fraction of all real decisions')
    axes[1, 0].set_ylabel('Fraction of all real decisions')
    fig.suptitle('Enduro sub001 at 1M: within-planning change versus a fixed initial gate', fontsize=12)
    fig.text(.5, .015, '24,576 decisions per completed condition; 16 seed/stream clusters; evaluation epsilon 0.\n'
             'Zero observed frequencies omitted on log axes. Diagnostic model; no reward-benefit claim.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output / ('initial_gate_model.' + suffix), dpi=180)
    plt.close(fig)
    frozen = args.output / 'analysis_sources'
    frozen.mkdir()
    for name in (Path(__file__).name, 'compare_planning_allocation_milestones.py',
                 'summarize_adaptive_planning_evidence.py'):
        shutil.copy2(Path(__file__).parent / name, frozen / name)
    result = dict(status='PARTIAL_CONDITION_COVERAGE' if missing else 'COMPLETE_ALLOCATION_DIAGNOSTIC',
        missing_conditions=missing, records=records, uncertainty='4000 seed/stream cluster resamples.',
        inference='Within-planning probability changes relative to each recorded first optional probability. '
                  'States vary across conditions. Does not isolate burn-in effects or establish reward usefulness.',
        sources_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.iterdir()})
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(tables, indent=2))


if __name__ == '__main__':
    main()
