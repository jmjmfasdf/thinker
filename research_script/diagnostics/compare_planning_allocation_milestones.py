"""Audit matched evaluation scopes and compare complete checkpoint distributions.

Matching seeds/decision indices does not match observations across policies.
The figure is descriptive; it establishes neither utility nor a causal effect
of a single training change. Source artifacts and checkpoints remain untouched.
"""
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


CONDITIONS = ('control', 'burn8', 'gru10')
LABELS = ('Control', 'Common + exploration + burn-in 8',
          'Common + exploration + burn-in 8 + GRU 10x')


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_sample(roots, condition, registered=False, final=False):
    matches = [root for root in roots if (root/f'{condition}_summary.json').exists()]
    if len(matches) != 1:
        raise ValueError(f'expected one complete source for {condition}, got {matches}')
    root = matches[0]
    summary_path, rows_path = (root/f'{condition}_summary.json',
                               root/f'{condition}_allocation.json')
    summary = json.loads(summary_path.read_text())
    rows = json.loads(rows_path.read_text())
    if summary['status'] != 'COMPLETE' or summary['evaluation_epsilon'] != 0:
        raise ValueError('only complete evaluation with training epsilon disabled is eligible')
    identities = [(r['seed'], r['stream'], r['decision']) for r in rows]
    expected = {(seed, stream, decision) for seed in (8301, 8401)
                for stream in range(8) for decision in range(128, 1664)}
    if len(rows) != len(expected) or set(identities) != expected:
        raise ValueError('duplicate/missing decision or unmatched evaluation scope')
    if summary['decisions'] != len(rows) or summary['independent_streams'] != 16:
        raise ValueError('summary coverage differs from underlying rows')
    length = np.asarray([r['length'] for r in rows])
    if not np.isfinite(length).all() or not np.equal(length, np.floor(length)).all():
        raise ValueError('nonintegral or nonfinite planning length')
    if np.any((length < 1) | (length > 30)):
        raise ValueError('planning length outside configured limits')
    hist = np.bincount(length.astype(int), minlength=31)
    statistics = dict(mean_length=float(length.mean()),
        min_fraction=float(np.mean(length == 1)),
        interior_fraction=float(np.mean((length > 1) & (length < 30))),
        cap_fraction=float(np.mean(length == 30)))
    for name, value in statistics.items():
        if not np.isclose(value, summary[name], atol=1e-12, rtol=0):
            raise ValueError('reported summary disagrees with rows: '+name)
    if not np.array_equal(hist, summary['histogram']):
        raise ValueError('reported histogram disagrees with rows')
    sources = [summary_path, rows_path]
    binding = summary['checkpoint']
    for role in ('actor', 'model'):
        if fingerprint(Path(binding[role]['path'])) != binding[role]['sha256']:
            raise ValueError('checkpoint content changed: '+role)
    if registered:
        registry = root.parent/(root.name+'_job.json')
        saved = json.loads(registry.read_text())
        if final:
            binding_path = root.parent/f'fast_allocation_1m/{condition}_binding.json'
            audit_path = root/'final_binding_audit.json'
            registration = json.loads(binding_path.read_text())
            audit = json.loads(audit_path.read_text())
            if (saved['milestone'] != 1_000_000 or registration['plan']['milestone'] != 1_000_000
                    or audit['status'] != 'REGISTERED_FINAL_CHECKPOINT_AND_DECISIONS_PASS'
                    or audit['checkpoint'] != binding or audit['independent_sample'] is not False):
                raise ValueError('final sample lacks the registered endpoint audit')
            registered_binding = registration['checkpoint']
            sources.extend([binding_path,audit_path])
        else:
            registered_binding = saved['checkpoint']
        if registered_binding != binding or saved['decisions_per_condition'] != len(rows):
            raise ValueError('completed sample differs from submitted checkpoint/scope')
        completion = root/'completion.json'
        completed = json.loads(completion.read_text())
        if completed['status'] != 'COMPLETE' or completed['conditions'] != [condition]:
            raise ValueError('missing or inconsistent completion record')
        sources.extend([registry, completion])
    streams = []
    for seed in (8301,8401):
        for stream in range(8):
            values = np.array([r['length'] for r in rows if r['seed']==seed and r['stream']==stream])
            streams.append(dict(seed=seed,stream=stream,decisions=len(values),
                mean_length=float(values.mean()),min_fraction=float(np.mean(values==1)),
                interior_fraction=float(np.mean((values>1)&(values<30))),cap_fraction=float(np.mean(values==30))))
    # Recompute cluster intervals instead of trusting the saved summary alone.
    stream_values = np.array([[r[k] for k in ('min_fraction','interior_fraction','cap_fraction','mean_length')] for r in streams])
    rng = np.random.default_rng(91273)
    boot = stream_values[rng.integers(0,16,(2000,16))].mean(axis=1)
    intervals = dict(zip(('min','interior','cap','mean'),np.quantile(boot,[.025,.975],axis=0).T.tolist()))
    for key in intervals:
        if not np.allclose(intervals[key],summary['stream_bootstrap_ci95'][key],rtol=0,atol=1e-12):
            raise ValueError('saved cluster uncertainty disagrees with per-stream records')
    return dict(condition=condition, checkpoint=binding, decisions=len(rows),
        streams=16, observed_decision_range=[128,1663], **statistics,
        histogram=hist.tolist(), per_stream=streams, stream_bootstrap_ci95=intervals,
        sources=[dict(path=str(path.resolve()), sha256=fingerprint(path)) for path in sources])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--earlier-root', type=Path, action='append', required=True)
    parser.add_argument('--later-root', type=Path, action='append', required=True)
    parser.add_argument('--final-root', type=Path, action='append',
                        help='Optional complete registered1M roots; require all three conditions.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    samples, table = {}, []
    milestones = [(250000,args.earlier_root),(500000,args.later_root)]
    if args.final_root:
        milestones.append((1_000_000,args.final_root))
    ticks = [f'{step//1000}k' if step<1_000_000 else '1M' for step,_ in milestones]
    for condition in CONDITIONS:
        samples[condition] = []
        for milestone, roots in milestones:
            sample = load_sample(roots,condition,registered=milestone>=500000,final=milestone==1_000_000)
            # Batches cross the exact milestone by a small amount; report it.
            step = sample['checkpoint']['actor']['real_step']
            tolerance = 1999 if milestone==1_000_000 else 1000
            if not milestone <= step <= milestone+tolerance:
                raise ValueError('checkpoint outside the declared milestone')
            samples[condition].append(sample)
            table.append(dict(condition=condition, milestone=milestone,
                actor_env_steps=step, model_env_steps=sample['checkpoint']['model']['real_step'],
                decisions=sample['decisions'], independent_streams=sample['streams'],
                **{k:sample[k] for k in ('mean_length','min_fraction','interior_fraction','cap_fraction')}))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'allocation_milestones.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    stream_rows = [dict(condition=condition,milestone=milestones[i][0],**row)
                   for condition in CONDITIONS for i,sample in enumerate(samples[condition])
                   for row in sample['per_stream']]
    with (args.output/'allocation_streams.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(stream_rows[0]));writer.writeheader();writer.writerows(stream_rows)
    fig, axes = plt.subplots(2,3,figsize=(13,7.4),sharey='row')
    colors = ('#b7bcc4','#168d9a','#d76d45')
    for column, (condition, label) in enumerate(zip(CONDITIONS, LABELS)):
        series = samples[condition]
        ax = axes[0,column]
        bottom = np.zeros(len(series))
        for key, color, legend in zip(('min_fraction','interior_fraction','cap_fraction'),
                                     colors, ('1 query','2–29 queries','30 queries')):
            values = np.array([sample[key] for sample in series])
            ax.bar(np.arange(len(series)),values,bottom=bottom,width=.55,color=color,label=legend)
            bottom += values
        ax.set_title(label,fontsize=10)
        ax.set_xticks(np.arange(len(series)),ticks)
        ax.set_ylim(0,1.13)
        ax.set_xlabel('Training environment steps')
        for x, sample in enumerate(series):
            ax.text(x,1.03,f"Mean {sample['mean_length']:.2f}",ha='center',fontsize=9)
        ax = axes[1,column]
        for sample,color,legend in zip(series,('#3465a4','#ad3a44','#3d8b49'),ticks):
            ax.plot(range(1,31),np.array(sample['histogram'])[1:]/sample['decisions'],
                    marker='o',ms=2.5,lw=1,color=color,label=legend)
        ax.set_xlim(.5,30.5)
        ax.set_ylim(0,1.02)
        ax.set_xticks([1,5,10,15,20,25,30])
        ax.set_xlabel('Imagination queries per real decision')
        ax.grid(axis='y',alpha=.2)
    axes[0,0].set_ylabel('Fraction of decisions')
    axes[1,0].set_ylabel('Fraction of decisions')
    axes[0,0].legend(fontsize=8,loc='center left')
    axes[1,0].legend(fontsize=8)
    fig.suptitle('Enduro sub001: planning allocation at '+', '.join(ticks),fontsize=13)
    fig.text(.5,.015,'Each sample: 24,576 decisions, 16 seed/stream clusters; stochastic gate, exploration epsilon 0.\n'
             'Same seeds and decision window; visited states differ. Distribution diversity does not establish utility.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95))
    for suffix in ('png','pdf'):
        fig.savefig(args.output/f'allocation_milestones.{suffix}',dpi=180)
    plt.close(fig)
    report = dict(checked_at=time.time(),status='COMPLETE_MATCHED_SCOPE_ALLOCATION_AUDIT',
        samples=samples,script_sha256=fingerprint(Path(__file__)),
        declared_milestones=[step for step,_ in milestones],
        cluster_intervals='Recomputed from16 seed/stream clusters; not independent-decision bootstrap.',
        comparison_scope='Identical seeds, stream count and decision-index window; not matched observations.',
        continuation='Resumed at 250k with weights/optimizers/counters restored; environment and recurrent state restarted.',
        inference='Descriptive fixed-policy allocation. Neither useful planning nor durable target distribution established.',
        utility='Not included; ongoing paired task-return evaluations are separate.')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(table,indent=2))


if __name__ == '__main__':
    main()
