"""Audit every registered burn16 continuation endpoint before reporting it."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from finalize_normalized_planning_extension import check_binding, check_scope
from summarize_adaptive_planning_evidence import allocation_report, paired_states, utility_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    sources = {}

    def bind(path):
        path = Path(path).resolve()
        if str(path) not in sources:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            sources[str(path)] = digest.hexdigest()
        return path

    def read(path):
        return json.loads(bind(path).read_text())

    def checkpoint_bytes(binding):
        for info in binding.values():
            if sources[str(bind(info['path']))] != info['sha256']:
                raise ValueError('checkpoint content changed')

    manifest = read(campaign / 'manifest.json')
    if (len(manifest['runs']) != 1 or manifest['runs'][0]['condition'] != 'burn16ln'
            or manifest['milestones'] != [1500000, 2000000] or manifest['utility_endpoint'] != 2000000):
        raise ValueError('unregistered continuation scope')
    item = manifest['runs'][0]
    parent = read(item['parent_completion_audit'])
    if (parent != read(campaign / 'parent_completion.json')
            or parent['status'] != 'CONTINUATION_CHECKPOINT_OPTIMIZER_COUNTERS_WANDB_PASS'):
        raise ValueError('parent audit mismatch')
    checkpoint_bytes(parent['checkpoint'])
    for kind in ('actor', 'model'):
        for key in ('path', 'real_step', 'sha256'):
            if parent['checkpoint'][kind][key] != item['parent_checkpoint'][kind][key]:
                raise ValueError('registered parent differs from completed training')
    if args.preflight:
        print(json.dumps(dict(status='PERSISTENCE_REGISTRY_AND_PARENT_PASS', milestones=manifest['milestones'])))
        return
    if args.output is None or args.output.exists():
        raise ValueError('a new output directory is required')
    final = read(campaign / 'burn16ln/completion.json')
    if (final['status'] != 'CONTINUATION_CHECKPOINT_OPTIMIZER_COUNTERS_WANDB_PASS'
            or final['health']['bc_history_status'] != 'NO_LARGE_NLL_OBSERVED'
            or final['contract'] != parent['contract'] or final['wandb']['state'] != 'finished'):
        raise ValueError('continuation completion audit failed')
    checkpoint_bytes(final['checkpoint'])
    run = Path(item['run_directory'])
    if sources[str(bind(run / 'logs.csv'))] != final['logs_sha256']:
        raise ValueError('training log bytes changed')
    if read(run / 'collector_metrics.json')['physical_env_steps'] != final['physical_collected_transitions']:
        raise ValueError('physical interaction counter mismatch')
    parent_root = Path(manifest['parent_campaign'])
    table, histograms, utilities, details, states_table = [], [], [], [], []
    allocation_rows = {}
    for milestone in (1000000, 1500000, 2000000):
        root = parent_root if milestone == 1000000 else campaign
        folder = root / f'allocation_{milestone}_burn16ln'
        summary = read(folder / 'burn16ln_summary.json')
        expected = parent['checkpoint'] if milestone == 1000000 else final['checkpoint'] if milestone == 2000000 else None
        check_binding(summary, milestone, expected)
        checkpoint_bytes(summary['checkpoint'])
        rows = read(folder / 'burn16ln_allocation.json')
        check_scope(rows)
        allocation_rows[milestone] = rows
        values = allocation_report(rows)
        lengths = np.asarray([r['length'] for r in rows])
        if not np.equal(lengths, np.floor(lengths)).all():
            raise ValueError('noninteger length')
        histogram = np.bincount(lengths.astype(int), minlength=31).tolist()
        if histogram != summary['histogram']:
            raise ValueError('histogram mismatch')
        fields = ('mean_length', 'min_fraction', 'interior_fraction', 'cap_fraction')
        for key in fields:
            if not np.isclose(values[key], summary[key], rtol=0, atol=1e-12):
                raise ValueError('allocation summary mismatch')
        table.append(dict(milestone=milestone, actor_step=summary['checkpoint']['actor']['real_step'],
                          decisions=len(rows), **{k:values[k] for k in fields}))
        histograms.extend(dict(milestone=milestone, length=k, count=n, fraction=n/len(rows))
                          for k, n in enumerate(histogram) if k)
        details.append(dict(kind='allocation', milestone=milestone, report=values))
    for milestone in (1000000, 2000000):
        folder = (parent_root if milestone == 1000000 else campaign) / 'evaluation_burn16ln'
        summary = read(folder / 'summary.json')
        check_binding(summary, milestone, parent['checkpoint'] if milestone == 1000000 else final['checkpoint'])
        if read(folder / 'allocation.json') != allocation_rows[milestone]:
            raise ValueError('utility and standalone allocation differ')
        pairs = read(folder / 'pairs.json')
        check_scope(pairs, paired=True)
        arms = {}
        for row in pairs:
            key = row['seed'], row['count'], row['repetition']
            if key not in arms:
                arms[key] = read(folder / ('arms_%s_%s_%s.json' % key))
        states = paired_states(pairs, arms)
        report = utility_report(states)
        if report['paired_repetitions'] != 128 or report['intervention_real_decision_indices'] != [128]:
            raise ValueError('utility scope mismatch')
        result = dict(milestone=milestone, states=report['states'], streams=report['independent_streams'])
        for key in ('mean_net_utility', 'choice_matched_advantage', 'ema_rmse_improvement_over_zero'):
            result[key] = report[key]
            result[key+'_ci_low'], result[key+'_ci_high'] = report[key+'_ci95']
        utilities.append(result)
        states_table.extend(dict(milestone=milestone, **r) for r in states)
        details.append(dict(kind='utility', milestone=milestone, report=report))
    training = [dict(parent_actor_step=parent['checkpoint']['actor']['real_step'],
                     final_actor_step=final['checkpoint']['actor']['real_step'],
                     learner_transitions=final['actor_added_transitions'],
                     physical_actions=final['physical_collected_transitions'],
                     bc_targets=final['bc_targets'],
                     actor_updates=final['counters']['imitation_update_count']['logged_delta'],
                     max_bc_nll=final['health']['maximum_bc_nll'], wandb_url=final['wandb']['url'])]
    args.output.mkdir(parents=True)
    for name, rows in [('allocation_summary.csv', table), ('histograms.csv', histograms),
                       ('utility_summary.csv', utilities), ('paired_states.csv', states_table),
                       ('training_summary.csv', training)]:
        with (args.output / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / 'details.json').write_text(json.dumps(details, indent=2, allow_nan=False)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, layout='constrained')
    for ax, milestone in zip(axes, (1000000, 1500000, 2000000)):
        rows = [r for r in histograms if r['milestone'] == milestone]
        ax.bar([r['length'] for r in rows], [100*r['fraction'] for r in rows], color='#218c8d')
        ax.set(title=f'{milestone/1e6:g}M', xlabel='Imagination queries',
               xticks=[1, 5, 10, 15, 20, 25, 30], ylim=(0, 100))
    axes[0].set_ylabel('Decisions (%)')
    fig.suptitle('Enduro sub001 · Burn-in 16 · Registered persistence checkpoints')
    for extension in ('png', 'pdf'):
        fig.savefig(args.output / f'length_histograms.{extension}', dpi=180)
    plt.close(fig)
    result = dict(status='PERSISTENCE_EVIDENCE_ASSEMBLED_NOT_GOAL_COMPLETION', goal_complete=False,
                  checked_at=time.time(), sources_sha256=sources, limitations=manifest['limitations'],
                  interpretation='Evaluate both registered endpoints; completion is not evidence of useful planning.')
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], output=str(args.output))))


if __name__ == '__main__':
    main()
