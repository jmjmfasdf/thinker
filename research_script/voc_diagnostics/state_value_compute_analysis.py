"""Relate pre-search state value to subsequent computation in training logs.

Primary analysis: sub001, tau .01, 7--8M collected real actions. A search stage,
not each query within it, is one observation. Include stages reaching the cap.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
import yaml

GAMES = [('pong', 'Pong', '#0072B2'), ('enduro', 'Enduro', '#D55E00'),
         ('spaceinvaders', 'Space Invaders', '#009E73')]


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def rho(x, y):
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def partial_rho(value, outcome, controls):
    x, y = rankdata(value), rankdata(outcome)
    design = np.column_stack([np.ones(len(x)), controls])
    x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def reconstruct(campaign, maximum):
    manifest = read(campaign / 'campaign.json')
    assert manifest['subject'] == 1 and manifest['temperature'] == .01
    resumes = read(campaign / 'training_resume_paths.json')
    decisions, audit = [], []
    for cell in manifest['cells']:
        flags = yaml.safe_load(Path(cell['config']).read_text())
        # root_value_token is untransformed world-model V, not a root-Q margin.
        assert flags['model_enc_type'] == 0 and flags['voc_gru_connected']
        cap = int(flags['max_search_steps'])
        segments = [(campaign / 'cells' / cell['name'], 0)]
        if cell['name'] in resumes:
            directory = Path(resumes[cell['name']])
            segments.append((directory, read(directory / 'resume_provenance.json')['parent_collector_steps']))
        for segment, (directory, offset) in enumerate(segments):
            path = directory / 'control_samples.jsonl'
            raw = path.read_bytes(); complete = raw[:raw.rfind(b'\n')+1]
            groups = defaultdict(list); identities = set(); excluded = Counter(); retained = Counter()
            for line in complete.splitlines():
                r = json.loads(line)
                identity = (r['attempt'], r['decision'], r['stream'], r['count'])
                assert identity not in identities, identity
                identities.add(identity); groups[identity[:3]].append(r)
            for key, seq in groups.items():
                step = seq[0]['real_step'] + offset
                if step > maximum:
                    continue
                assert len({r['real_step'] for r in seq}) == 1
                assert [r['count'] for r in seq] == list(range(len(seq)))
                assert not any(r['control'] == 2 for r in seq[:-1])
                if any(not r['valid'] or r['forced'] for r in seq):
                    excluded['invalid_or_forced_record'] += 1; continue
                n = sum(r['control'] in [0, 1] for r in seq)
                capped = seq[-1]['control'] != 2 and n == cap and seq[-1]['count'] == cap-1
                if seq[-1]['control'] != 2 and not capped:
                    excluded['incomplete_uncapped_stage'] += 1; continue
                assert n == (cap if capped else seq[-1]['count'])
                assert all(0 <= r['depth'] <= r['count'] for r in seq)
                retained['capped' if capped else 'observed_stop'] += 1
                first = seq[0]
                decisions.append(dict(game=cell['game'], segment=segment,
                    attempt=key[0], decision=key[1], stream=key[2], global_step=step,
                    initial_value=first['root_value_token'], initial_voc=first['ema_delta_q'],
                    initial_policy_margin=first['root_policy_margin'],
                    initial_continue_probability=first['p_continue'], computations=n,
                    started=int(n > 0), capped=int(capped),
                    max_observed_depth=max(r['depth'] for r in seq),
                    depth_is_lower_bound=int(capped), resets=sum(r['control'] == 1 for r in seq)))
            audit.append(dict(path=str(path), segment=segment, offset=offset,
                snapshot_sha256=hashlib.sha256(complete).hexdigest(), snapshot_bytes=len(complete),
                incomplete_final_bytes=len(raw)-len(complete), groups=len(groups),
                included_to_maximum=dict(retained), excluded_to_maximum=dict(excluded)))
    return decisions, audit


def stats(rows, start):
    value = np.array([r['initial_value'] for r in rows])
    count = np.array([r['computations'] for r in rows])
    voc = np.array([r['initial_voc'] for r in rows])
    margin = np.array([r['initial_policy_margin'] for r in rows])
    time = np.array([(r['global_step']-start)//250_000 for r in rows])
    segment = np.array([r['segment'] for r in rows])
    controls = np.column_stack([rankdata(margin)] + [(time == x).astype(float) for x in np.unique(time)[1:]]
                              + [(segment == x).astype(float) for x in np.unique(segment)[1:]])
    active = count > 0
    return dict(rho_value_count=rho(value, count), rho_value_count_started=rho(value[active], count[active]),
        rho_value_start=rho(value, active.astype(int)), rho_voc_count=rho(voc, count),
        partial_rho_value_count_time_margin=partial_rho(value, count, controls),
        partial_rho_value_count_time_margin_voc=partial_rho(value, count, np.column_stack([controls, rankdata(voc)])))


def summarize(rows, game, start, end, bootstrap, seed):
    results = stats(rows, start)
    blocks = defaultdict(list)
    # Each block spans approximately 16 sampled decisions. Preserve segment boundaries.
    for i, r in enumerate(rows):
        blocks[r['segment'], (r['global_step']-start)//32_768].append(i)
    blocks = [np.array(ids) for ids in blocks.values()]
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    for _ in range(bootstrap):
        ids = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        estimate = stats([rows[i] for i in ids], start)
        for name, val in estimate.items():
            draws[name].append(val)
    out = dict(game=game, start=start, end=end, n=len(rows),
        n_started=sum(r['started'] for r in rows), capped=sum(r['capped'] for r in rows),
        mean_computations=float(np.mean([r['computations'] for r in rows])),
        n_time_blocks=len(blocks), **results)
    for name, vals in draws.items():
        vals = np.asarray(vals); vals = vals[np.isfinite(vals)]
        assert len(vals) > .95*bootstrap
        out[name+'_ci_low'], out[name+'_ci_high'] = np.quantile(vals, [.025, .975]).tolist()
    order = np.argsort([r['initial_value'] for r in rows])
    low, high = np.array_split(order, 4)[0], np.array_split(order, 4)[-1]
    for label, ids in [('low_value_quartile', low), ('high_value_quartile', high)]:
        rr = [rows[i] for i in ids]
        out[label+'_n'] = len(rr)
        out[label+'_mean_count'] = float(np.mean([r['computations'] for r in rr]))
        out[label+'_start_fraction'] = float(np.mean([r['started'] for r in rr]))
    return out


def figure(rows, summaries, output):
    fig, axes = plt.subplots(2, 3, figsize=(12.9, 8.1), sharex='col')
    fig.subplots_adjust(left=.085, right=.985, top=.82, bottom=.16, hspace=.40, wspace=.22)
    fig.text(.085, .965, 'Do high-value states receive more imagination?', fontsize=18, weight='bold', va='top')
    fig.text(.085, .914, 'sub001 | temperature 0.01 | 7–8M collected environment actions | value measured before search', fontsize=11)
    for col, (game, label, color) in enumerate(GAMES):
        group = [r for r in rows if r['game'] == game]
        summary = next(r for r in summaries if r['game'] == game)
        for row in range(2):
            selected = [r for r in group if row == 0 or r['started']]
            x = np.array([r['initial_value'] for r in selected])
            y = np.log1p([r['computations'] for r in selected])
            cap = np.array([r['capped'] for r in selected], dtype=bool)
            ax = axes[row, col]
            ax.scatter(x[~cap], y[~cap], s=13, color=color, alpha=.25, edgecolors='none')
            if cap.any():
                ax.scatter(x[cap], y[cap], s=26, marker='^', color=color, alpha=.85, edgecolors='none')
            trend = lowess(y, x, frac=.65, it=0, return_sorted=True)
            ax.plot(trend[:, 0], trend[:, 1], color=color, lw=2.2)
            metric = 'rho_value_count' if row == 0 else 'rho_value_count_started'
            ax.set_title(f'{label} | n={len(selected)}\nrho={summary[metric]:+.2f} [{summary[metric+"_ci_low"]:+.2f}, {summary[metric+"_ci_high"]:+.2f}]', fontsize=10.5)
            ticks = [0, 1, 3, 10, 30] if row == 0 else [1, 3, 10, 30]
            ax.set_yticks(np.log1p(ticks), [str(t) for t in ticks])
            ax.set_ylim(-.1 if row == 0 else .5, np.log1p(30)+.2)
            ax.grid(axis='y', alpha=.17)
            ax.set_xlabel('Initial predicted state value V(s)')
    axes[0, 0].set_ylabel('Executed imaginations (log1p scale)\nAll decisions, including zero search')
    axes[1, 0].set_ylabel('Executed imaginations (log1p scale)\nOnly decisions that started search')
    fig.text(.085, .087, 'Dots: individual decisions. Triangles: 30-computation cap. Lines: continuous LOWESS trends on log1p(count).', fontsize=10)
    fig.text(.085, .047, 'Spearman rho with 95% time-block bootstrap intervals. V(s) is a model prediction, not measured future reward.', fontsize=10)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'state_value_imagination.{ext}', dpi=200, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args(); args.output.mkdir(exist_ok=True, parents=True)
    all_rows, audit = reconstruct(args.campaign, 8_000_000)
    primary, summaries, sensitivity = [], [], []
    for gi, (game, _, _) in enumerate(GAMES):
        selected = [r for r in all_rows if r['game'] == game and 7_000_000 < r['global_step'] <= 8_000_000]
        primary.extend(selected)
        summaries.append(summarize(selected, game, 7_000_000, 8_000_000, args.bootstrap, 30092026+gi))
        for start, end in [(0, 2_000_000), (2_000_000, 4_000_000), (4_000_000, 6_000_000),
                           (6_000_000, 8_000_000), (7_500_000, 8_000_000)]:
            rr = [r for r in all_rows if r['game'] == game and start < r['global_step'] <= end]
            sensitivity.append(dict(game=game, start=start, end=end, n=len(rr),
                n_started=sum(r['started'] for r in rr), capped=sum(r['capped'] for r in rr), **stats(rr, start)))
    write_csv(args.output / 'decisions_all_to_8M.csv', all_rows)
    write_csv(args.output / 'decisions_7_to_8M.csv', primary)
    write_csv(args.output / 'summary.csv', summaries)
    write_csv(args.output / 'window_sensitivity.csv', sensitivity)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    figure(primary, summaries, args.output)
    manifest = dict(created_at=datetime.datetime.now().astimezone().isoformat(),
        subject=1, temperature=.01, primary_window=[7_000_000, 8_000_000],
        predictor='count-0 root_value_token: untransformed world-model state value before this search stage',
        outcome='Number of actually accepted PROCEED/RESET controls before STOP or the fixed cap of 30',
        depth='Recorded separately as max_observed_depth; cap-stage final depth not imputed',
        high_value_definition='High predicted V(s), not high VOC and not verified high realized reward',
        weighting='One sampled decision per row, stream0 every128 decisions; not one query per row',
        resume='Enduro local counters offset by recorded parent_collector_steps',
        uncertainty=f'{args.bootstrap} non-overlapping 32768-action time-block bootstrap draws; segment boundaries retained',
        adjustment='Partial rank correlations residualize 250k-step strata, segment, and initial policy margin; VOC added only in the separately named additional adjustment',
        interpretation='Observational association in one evolving training trajectory per game. Started-only subset and VOC adjustment do not identify causal effects.',
        sources=audit, source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
