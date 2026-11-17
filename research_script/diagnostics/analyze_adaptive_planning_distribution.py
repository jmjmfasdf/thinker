"""Distinguish a broad length histogram from evidence for adaptive computation.

Fit a constant continuation probability separately within each collector
interval. This is a descriptive stochastic-gate null, not a significance test:
decisions are clustered in streams and the policy changes during training.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def constant_gate_distribution(probability, minimum=1, cap=30):
    if not 0 <= probability <= 1 or not 0 <= minimum < cap:
        raise ValueError('invalid gate probability or support')
    values = np.zeros(cap+1, dtype=float)
    lengths = np.arange(minimum, cap)
    values[lengths] = (1-probability)*probability**(lengths-minimum)
    values[cap] = probability**(cap-minimum)
    return values


def fit_constant_gate(histogram, minimum=1):
    histogram = np.asarray(histogram, dtype=np.int64)
    cap = len(histogram)-1
    if np.any(histogram < 0) or histogram[:minimum].sum() or not histogram.sum():
        raise ValueError('invalid/empty length histogram')
    continues = int(np.dot(histogram, np.maximum(np.arange(cap+1)-minimum, 0)))
    stops = int(histogram[minimum:cap].sum())
    p = continues/(continues+stops) if continues+stops else 0.
    expected = constant_gate_distribution(p, minimum, cap)
    empirical = histogram/histogram.sum()
    valid = empirical > 0
    return dict(p_continue_fit=p, decisions=int(histogram.sum()),
        mean_length=float(np.dot(empirical, np.arange(cap+1))),
        min_fraction=float(empirical[minimum]), interior_fraction=float(empirical[minimum+1:cap].sum()),
        cap_fraction=float(empirical[cap]), null_cap_fraction=float(expected[cap]),
        total_variation=float(.5*np.abs(empirical-expected).sum()),
        kl_to_constant_gate=float(np.sum(empirical[valid]*np.log(empirical[valid]/expected[valid]))))


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    # Training may be appending the final line. Only complete JSON rows count.
    for line in path.open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if line.endswith('\n'):
                raise
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    campaign = args.campaign.resolve()
    root = campaign.parents[3]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((campaign/'manifest.json').read_text())
    interval_rows, sampled_stages, latest = [], [], []
    for item in manifest['runs']:
        condition = item['condition']
        if condition == 'smoke':
            continue
        run = root/'logs/thinker'/item['xpid']
        intervals = read_jsonl(run/'planning_intervals.jsonl')
        selected = []
        for interval in intervals:
            if not interval.get('window_decisions'):
                continue
            hist = np.array([interval[f'window_count_{k}_decisions'] for k in range(31)])
            row = dict(condition=condition, physical_env_steps=interval['physical_env_steps'],
                **fit_constant_gate(hist), sign_change_fraction=interval['window_sign_change_fraction'],
                saturated_optional_fraction=interval['window_saturated_fraction'],
                mean_completed_training_return=interval['mean_last_100_episode_return'])
            interval_rows.append(row)
            selected.append((interval, hist))
        if selected:
            # At least the most recent interval, and up to ten complete windows.
            tail = selected[-10:]
            pooled = sum((h for _, h in tail), np.zeros(31, dtype=int))
            fit = fit_constant_gate(pooled)
            interval_tv = np.average([fit_constant_gate(h)['total_variation'] for _, h in tail],
                                     weights=[h.sum() for _, h in tail])
            latest.append(dict(condition=condition,
                start_physical_env_steps=tail[0][0]['physical_env_steps']-tail[0][0]['window_decisions'],
                end_physical_env_steps=tail[-1][0]['physical_env_steps'], **fit,
                mean_within_interval_total_variation=float(interval_tv)))
        groups = defaultdict(list)
        for row in read_jsonl(run/'control_samples.jsonl'):
            if row['valid']:
                groups[(row['attempt'], row['stream'], row['decision'])].append(row)
        for identity, records in groups.items():
            records.sort(key=lambda r:r['count'])
            counts = [r['count'] for r in records]
            if counts != list(range(len(counts))):
                continue
            last = records[-1]
            if last['control'] == 2:  # learned STOP
                length = last['count']
            elif last['count'] == 29:  # final computation is capped without a STOP sample
                length = 30
            else:
                continue  # partially flushed stage
            choices = [r for r in records if r['gate_choice_valid']]
            if not choices:
                continue
            probs = np.array([r['p_continue'] for r in choices])
            signs = np.sign(probs-.5)
            sampled_stages.append(dict(condition=condition, physical_env_steps=records[0]['real_step'],
                stream=identity[1], decision=identity[2], length=length, optional_queries=len(choices),
                first_probability=float(probs[0]), last_probability=float(probs[-1]),
                probability_range=float(np.ptp(probs)),
                sign_changes=int(np.count_nonzero(signs[1:]*signs[:-1]<0)),
                first_ema_delta=choices[0]['ema_delta_q'], last_ema_delta=choices[-1]['ema_delta_q']))
    write_csv(args.output/'intervals.csv', interval_rows)
    write_csv(args.output/'recent_distribution.csv', latest)
    write_csv(args.output/'sampled_stages.csv', sampled_stages)
    theory = []
    for epsilon in (.02, .10):
        p = 1-epsilon/2
        hist = constant_gate_distribution(p)
        theory.append(dict(train_epsilon=epsilon, saturated_continue_probability=p,
            cap_probability=float(hist[-1]), mean_length=float(np.dot(hist,np.arange(31)))))
    (args.output/'interpretation.json').write_text(json.dumps(dict(
        scope='interim training diagnostics; training exploration included; unequal achieved steps',
        recent=latest, constant_gate_limits=theory, sampled_complete_stages=len(sampled_stages),
        caution='Small departure from a constant stochastic gate does not show state-dependent utility. Large departure may reflect changing policy over time. Fixed-checkpoint evaluation and paired actual utility remain required.'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for condition in sorted(set(r['condition'] for r in interval_rows)):
        data = [r for r in interval_rows if r['condition']==condition]
        x = np.array([r['physical_env_steps'] for r in data])/1000
        for ax, key, label in zip(axes.ravel(),
            ['mean_length','cap_fraction','total_variation','sign_change_fraction'],
            ['Mean imagination length','Fraction reaching 30','Distance from constant gate (TV)','Within-decision sign-change fraction']):
            ax.plot(x, [r[key] for r in data], label=condition, lw=1.4)
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel('Collected real environment actions (thousands)')
    axes[0,0].legend(ncol=2, fontsize=8)
    fig.suptitle('Interim training — exploration included; curves are not fixed-policy utility evidence')
    fig.tight_layout()
    fig.savefig(args.output/'training_distribution.png', dpi=160)
    fig.savefig(args.output/'training_distribution.pdf')
    plt.close(fig)
    for row in latest:
        print(row['condition'], 'through',row['end_physical_env_steps'],
              'mean',round(row['mean_length'],3),'cap',round(row['cap_fraction'],3),
              'within-window TV',round(row['mean_within_interval_total_variation'],3))


if __name__ == '__main__':
    main()
