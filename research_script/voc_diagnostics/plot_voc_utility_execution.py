"""Plot paired environment utility and *observed* training control choices.

Read-only analysis of the sub001 tau=.01 pilot. Resume segment counters are
converted to cumulative collected environment actions using recorded provenance.
No jobs, evaluations, or W&B runs are started by this script.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import yaml

GAMES = [('pong', 'Pong', '#0072B2'), ('enduro', 'Enduro', '#D55E00'),
         ('spaceinvaders', 'Space Invaders', '#009E73')]
SIGNS = [('positive', 1, '#2563A6'), ('negative', -1, '#CF6030')]


def read(path):
    return json.loads(Path(path).read_text())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty output: ' + str(path))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output, stem):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'{stem}.{ext}', dpi=200, facecolor='white')
    plt.close(fig)


def collect_utility(campaign, expansion, milestones):
    registry = read(campaign / 'jobs.json')['evaluation']
    rows, states, sources = [], [], []
    for game, _, _ in GAMES:
        inputs = []
        for step in milestones:
            key = f'v10t001-20260926-{game}_s701_{step:08d}_benefit'
            record = registry[key]
            assert record['state'] == 'COMPLETED', (key, record['state'])
            directory = Path(record.get('output') or campaign / 'evaluations' / key)
            inputs.append(('standard', step, directory))
        if expansion:
            inputs.append(('expanded_4M', 4_000_000, expansion / 'aggregate' / game))
        for protocol, step, directory in inputs:
            summary_path, pairs_path = directory / 'summary.json', directory / 'pairs.json'
            raw_summary, raw_pairs = summary_path.read_bytes(), pairs_path.read_bytes()
            summary, pairs = json.loads(raw_summary), json.loads(raw_pairs)
            assert summary['status'] == 'COMPLETE'
            voc = summary['voc']
            groups = {}
            identities = set()
            for pair in pairs:
                key = tuple(pair[k] for k in ['seed', 'stream', 'prefix', 'count'])
                identity = key + (pair['repetition'],)
                assert identity not in identities, identity
                identities.add(identity)
                assert np.isclose(pair['delta'], pair['task_delta'] - pair['cost'], atol=1e-10)
                groups.setdefault(key, []).append(pair)
            assert len(groups) == voc['states'] and len(pairs) == voc['pairs']
            computed = []
            for key, repeats in groups.items():
                assert len({r['entry_hash'] for r in repeats}) == 1
                state = dict(game=game, protocol=protocol, nominal_step=step,
                    seed=key[0], stream=key[1], prefix=key[2], count=key[3],
                    repetitions=len(repeats),
                    utility=float(np.mean([r['delta'] for r in repeats])),
                    task_reward_difference=float(np.mean([r['task_delta'] for r in repeats])),
                    computation_cost=float(np.mean([r['cost'] for r in repeats])),
                    predicted_voc=float(np.mean([r['ema'] for r in repeats])),
                    continue_probability=float(np.mean([r['probability'] for r in repeats])))
                computed.append(state)
            states.extend(computed)
            u = np.array([s['utility'] for s in computed])
            cost = np.array([s['computation_cost'] for s in computed])
            prob = np.array([s['continue_probability'] for s in computed])
            assert np.isfinite(u).all() and np.all(cost > 0)
            random_probability = float(np.sum(prob * cost) / np.sum(cost))
            allocation = float(np.mean((prob - random_probability) * u))
            assert np.isclose(u.mean(), voc['mean_net_benefit'], atol=1e-12)
            assert np.isclose(allocation, voc['allocation_gain_vs_cost_matched_random'], atol=1e-12)
            row = dict(game=game, protocol=protocol, nominal_step=step,
                actor_step=summary['checkpoint']['actor_real_step'],
                model_step=summary['checkpoint']['model_real_step'],
                mean_net_utility=float(u.mean()),
                utility_ci_low=voc['mean_net_benefit_ci95'][0],
                utility_ci_high=voc['mean_net_benefit_ci95'][1],
                allocation_gain=allocation,
                allocation_ci_low=voc['allocation_gain_vs_cost_matched_random_ci95'][0],
                allocation_ci_high=voc['allocation_gain_vs_cost_matched_random_ci95'][1],
                mean_task_reward_difference=float(np.mean([s['task_reward_difference'] for s in computed])),
                mean_computation_cost=float(cost.mean()),
                random_continue_probability=random_probability,
                states=len(computed), pairs=len(pairs), independent_streams=voc['independent_streams'],
                task_changed_pairs=sum(abs(r['task_delta']) > 1e-12 for r in pairs),
                max_pair_tail_bound=max(r['tail_bound'] for r in pairs),
                source=str(summary_path))
            rows.append(row)
            sources.append(dict(path=str(summary_path), sha256=sha(raw_summary),
                pairs_path=str(pairs_path), pairs_sha256=sha(raw_pairs)))
    return rows, states, sources


def collect_controls(campaign, max_step):
    manifest = read(campaign / 'campaign.json')
    resumes = read(campaign / 'training_resume_paths.json')
    samples, sources, qa = [], [], {}
    for cell in manifest['cells']:
        game = cell['game']
        flags = yaml.safe_load(Path(cell['config']).read_text())
        assert str(flags['icopro_subjects']) == '1'
        assert flags['voc_gate_exact_projection'] and flags['voc_gru_connected']
        assert flags['voc_gate_q_temperature'] == .01
        segments = [(campaign / 'cells' / cell['name'], 0)]
        if cell['name'] in resumes:
            directory = Path(resumes[cell['name']])
            provenance = read(directory / 'resume_provenance.json')
            # These logs count collected interactions, not actor learner updates.
            segments.append((directory, provenance['parent_collector_steps']))
        game_rows, identities = [], set()
        for segment, (directory, offset) in enumerate(segments):
            path = directory / 'control_samples.jsonl'
            raw = path.read_bytes()
            complete = raw[:raw.rfind(b'\n') + 1]
            records = [json.loads(line) for line in complete.splitlines() if line]
            sources.append(dict(path=str(path), snapshot_sha256=sha(complete),
                snapshot_bytes=len(complete), incomplete_bytes=len(raw) - len(complete),
                segment=segment, cumulative_collector_offset=offset,
                offset_source=str(directory / 'resume_provenance.json') if segment else None))
            for r in records:
                global_step = int(r['real_step']) + offset
                if global_step > max_step:
                    continue
                key = (segment, r['attempt'], r['real_step'], r['stream'], r['count'])
                assert key not in identities, key
                identities.add(key)
                if not r['valid'] or r['forced']:
                    continue
                assert r['control'] in [0, 1, 2]
                row = dict(game=game, segment=segment, global_step=global_step,
                    local_step=r['real_step'], attempt=r['attempt'], stream=r['stream'],
                    decision=r['decision'], count=r['count'], depth=r['depth'],
                    ema_voc=r['ema_delta_q'], online_voc=r['online_delta_q'],
                    continue_probability=r['p_continue'], actual_continue=int(r['control'] in [0, 1]),
                    control=r['control'])
                game_rows.append(row)
        q = np.array([r['ema_voc'] for r in game_rows])
        p = np.array([r['continue_probability'] for r in game_rows])
        assert np.isfinite(q).all() and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
        epsilon = flags['voc_train_epsilon']
        expected = epsilon / 2 + (1 - epsilon) / (1 + np.exp(-np.clip(q / .01 / flags['voc_gate_temperature'], -700, 700)))
        projection_error = float(np.max(np.abs(p - expected)))
        assert projection_error < 1e-5, (game, projection_error)
        samples.extend(game_rows)
        qa[game] = dict(valid_controls=len(game_rows), segments=len(segments),
            exact_projection_max_absolute_error=projection_error,
            sampled_decisions=len({(r['segment'], r['attempt'], r['decision'], r['stream']) for r in game_rows}))
    return samples, sources, qa


def summarize_controls(samples, max_step, window, stride):
    rows = []
    for game, _, _ in GAMES:
        data = [r for r in samples if r['game'] == game]
        for step in range(window, max_step + 1, stride):
            selection = [r for r in data if step - window < r['global_step'] <= step]
            for stage in ['all_choices', 'start_count0', 'continue_count_positive']:
                selected = [r for r in selection if stage == 'all_choices'
                    or (stage == 'start_count0' and r['count'] == 0)
                    or (stage == 'continue_count_positive' and r['count'] > 0)]
                for sign, direction, _ in SIGNS:
                    rr = [r for r in selected if direction * r['ema_voc'] > 1e-6]
                    rows.append(dict(game=game, step=step, window_start=step - window,
                        stage=stage, voc_sign=sign, n_choices=len(rr),
                        n_decisions=len({(r['segment'], r['attempt'], r['decision'], r['stream']) for r in rr}),
                        actual_continue_count=sum(r['actual_continue'] for r in rr),
                        actual_continue_rate=float(np.mean([r['actual_continue'] for r in rr])) if rr else None,
                        mean_continue_probability=float(np.mean([r['continue_probability'] for r in rr])) if rr else None,
                        reset_count=sum(r['control'] == 1 for r in rr)))
    return rows


def utility_figure(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4))
    fig.subplots_adjust(left=.08, right=.98, bottom=.24, top=.73, wspace=.32)
    fig.text(.08, .96, 'Does additional imagination improve real-environment outcomes?',
             fontsize=18, weight='bold', va='top')
    fig.text(.08, .905, 'sub001  |  temperature 0.01  |  VOC gradients connected to GRU  |  training seed 701', fontsize=11)
    keys = [('mean_net_utility', 'utility_ci_low', 'utility_ci_high',
             'A. CONTINUE versus STOP', 'Mean net utility per evaluated state'),
            ('allocation_gain', 'allocation_ci_low', 'allocation_ci_high',
             'B. Does the gate allocate computation usefully?', 'Utility gain over cost-matched random allocation')]
    for ax, (value, low, high, title, ylabel) in zip(axes, keys):
        ax.set_title(title, fontsize=12, pad=12)
        for game, label, color in GAMES:
            standard = sorted([r for r in rows if r['game'] == game and r['protocol'] == 'standard'], key=lambda r: r['nominal_step'])
            xx = np.array([r['nominal_step'] / 1e6 for r in standard])
            # Display round-off at zero as zero; CSV preserves the raw numbers.
            yy = np.array([r[value] for r in standard]); yy[np.abs(yy) < 1e-12] = 0
            lo = np.array([r[low] for r in standard]); lo[np.abs(lo) < 1e-12] = 0
            hi = np.array([r[high] for r in standard]); hi[np.abs(hi) < 1e-12] = 0
            ax.fill_between(xx, lo, hi, color=color, alpha=.12, linewidth=0)
            ax.plot(xx, yy, '-o', color=color, lw=2.1, ms=6, label=label)
            for r in rows:
                if r['game'] == game and r['protocol'] == 'expanded_4M':
                    x = 4.08 + [g[0] for g in GAMES].index(game) * .055
                    ax.errorbar(x, r[value], yerr=[[max(0, r[value]-r[low])], [max(0, r[high]-r[value])]],
                        fmt='D', color=color, mfc='white', mec=color, ms=6, capsize=3, elinewidth=1.1)
        ax.axhline(0, color='#657080', ls='--', lw=1, zorder=0)
        ax.set(xlim=(1.75, 8.3), xlabel='Training environment steps (millions)', ylabel=ylabel)
        ax.set_xticks([2, 4, 6, 8]); ax.grid(axis='y', alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.07, .87), ncol=3, frameon=False)
    fig.text(.08, .145, 'Lines / shading: standard diagnostic and 95% stream-bootstrap intervals. Hollow diamonds: separate expanded 4M diagnostic.', fontsize=10)
    fig.text(.08, .10, 'Net utility = discounted future environment reward difference minus extra current-stage computation cost. Zero is the reference.', fontsize=10)
    fig.text(.08, .055, 'Local paired interventions, not whole-game score improvement. One trained seed; intervals describe sampled evaluation streams.', fontsize=10, color='#52606D')
    save_figure(fig, output, '01_real_environment_utility')


def execution_figure(rows, output, max_step, window):
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.98, bottom=.24, top=.72, wspace=.13)
    fig.text(.065, .96, 'When the model favors CONTINUE, does it actually continue?', fontsize=18, weight='bold', va='top')
    fig.text(.065, .903, f'sub001  |  temperature 0.01  |  trailing {window/1e6:g}M-action windows  |  observed collector choices', fontsize=11)
    for ax, (game, label, _) in zip(axes, GAMES):
        ax.set_title(label, fontsize=13, weight='bold', pad=12)
        for sign, _, color in SIGNS:
            rr = [r for r in rows if r['game'] == game and r['stage'] == 'all_choices' and r['voc_sign'] == sign]
            xx = [r['step']/1e6 for r in rr]
            actual = [r['actual_continue_rate'] if r['actual_continue_rate'] is not None else np.nan for r in rr]
            expected = [r['mean_continue_probability'] if r['mean_continue_probability'] is not None else np.nan for r in rr]
            ax.plot(xx, expected, ls=':', lw=1.9, color=color, alpha=.8)
            ax.plot(xx, actual, lw=2.3, color=color)
            ax.plot(xx[-1], actual[-1], 'o', ms=6, color=color)
            ax.annotate(f'{actual[-1]:.1%}', (xx[-1], actual[-1]), xytext=(-5, 10 if sign == 'positive' else -17),
                        textcoords='offset points', ha='right', fontsize=11, color=color, weight='bold')
        ax.axhline(.5, color='#A2AAB4', ls='--', lw=.9)
        ax.set(xlim=(.85, max_step/1e6+.15), ylim=(-.03, 1.05), xlabel='Collected environment actions (millions)')
        ax.set_xticks([1, 2, 4, 6, 8]); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Fraction of actual CONTINUE choices')
    handles = [Line2D([0], [0], color=color, lw=2.3,
               label='EMA VOC > 0: favors CONTINUE' if sign == 'positive' else 'EMA VOC < 0: favors STOP')
               for sign, _, color in SIGNS]
    handles += [Line2D([0], [0], color='#454D58', lw=2.3, label='Solid: observed choices'),
                Line2D([0], [0], color='#454D58', lw=1.9, ls=':', label='Dotted: logged probability')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.055, .864), ncol=2, frameon=False, columnspacing=2.5)
    fig.text(.065, .145, 'CONTINUE = PROCEED or RESET; all valid search choices, including count 0. Forced stops and near-zero VOC ties excluded.', fontsize=10)
    fig.text(.065, .102, 'Stream 0 sampled every 128 decisions. Enduro resume counters joined using recorded collector offsets; overlapping windows are descriptive.', fontsize=10)
    fig.text(.065, .059, 'The sign is the model\'s own value estimate. Following that sign is separate from improving real-environment reward.', fontsize=10, color='#52606D')
    save_figure(fig, output, '02_predicted_voc_actual_continuation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--expansion', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-step', type=int, default=8_000_000)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    window, stride = 1_000_000, 250_000
    milestones = list(range(2_000_000, args.max_step+1, 2_000_000))
    utility, states, utility_sources = collect_utility(args.campaign, args.expansion, milestones)
    controls, control_sources, qa = collect_controls(args.campaign, args.max_step)
    curves = summarize_controls(controls, args.max_step, window, stride)
    write_csv(args.output / 'utility_scores.csv', utility)
    write_csv(args.output / 'utility_states.csv', states)
    write_csv(args.output / 'control_samples.csv', controls)
    write_csv(args.output / 'continuation_curves.csv', curves)
    latest = [r for r in curves if r['step'] == args.max_step]
    write_csv(args.output / 'continuation_latest_window.csv', latest)
    combined = []
    for game, _, _ in GAMES:
        result = next(r for r in utility if r['game'] == game
                      and r['protocol'] == 'standard' and r['nominal_step'] == args.max_step)
        row = {key: result[key] for key in ['game', 'mean_net_utility', 'utility_ci_low',
            'utility_ci_high', 'allocation_gain', 'allocation_ci_low', 'allocation_ci_high',
            'mean_task_reward_difference', 'mean_computation_cost', 'task_changed_pairs', 'states', 'pairs']}
        for sign, _, _ in SIGNS:
            control = next(r for r in latest if r['game'] == game
                           and r['stage'] == 'all_choices' and r['voc_sign'] == sign)
            for key in ['actual_continue_count', 'n_choices', 'n_decisions',
                        'actual_continue_rate', 'mean_continue_probability']:
                row[f'{sign}_{key}'] = control[key]
        combined.append(row)
    write_csv(args.output / f'summary_{args.max_step//1_000_000}M.csv', combined)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    utility_figure(utility, args.output)
    execution_figure(curves, args.output, args.max_step, window)
    manifest = dict(created_at=datetime.datetime.now().astimezone().isoformat(),
        campaign=str(args.campaign), expansion=str(args.expansion), subject=1, temperature=.01,
        max_step=args.max_step, checkpoints=milestones, window=window, stride=stride,
        utility_definition='E[sum_t gamma^t (r_continue-r_stop) - extra current-search-stage computation cost]',
        reward='clipped environment task reward; gamma=.99, horizon<=1536 real actions, stop at terminal, no value bootstrap',
        intervention='Force STOP now versus force one PROCEED/RESET, then use the native stochastic policy; current stage may contain additional computations',
        allocation_gain='mean((p_continue - sum(p_continue*cost)/sum(cost))*net_utility)',
        utility_uncertainty='Existing 2000 stream-cluster bootstrap draws; not a population or multi-training-seed interval',
        standard_sampling='2/4M: 32 states x 2 repetitions in 8 streams; 6/8M: 64 states x 4 repetitions in 16 streams',
        expansion_sampling='4M only: 256 states x 2 repetitions in 32 streams, four rather than two prefix strata; shown separately, never pooled',
        execution_definition='Accepted control in {PROCEED=0, RESET=1} conditional on same-time EMA VOC sign; STOP=2',
        execution_axis='Cumulative physically collected real actions. Resume offset=parent_collector_steps, not reset local counter or actor learner counter.',
        execution_scope='Sampled stream0 controls during training; all choices equally weighted, multiple dependent choices per decision. Curves descriptive, not frozen evaluation or causal training-time effect.',
        epsilon_q_tie=1e-6, utility_plot_zero_tolerance=1e-12,
        qa=qa, utility_sources=utility_sources, control_sources=control_sources,
        code_sha256=sha(Path(__file__).read_bytes()))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',utility_results=len(utility),control_qa=qa,output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
