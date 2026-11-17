"""Complete pilot follow-up with separate allocation and reward-utility tables.

Repeated states and streams are not pooled across checkpoints/protocols.
Only complete, bound results enter tables; unhealthy raw burn16 is excluded.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from summarize_adaptive_planning_evidence import paired_states, utility_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    names = dict(parent='adaptive_planning_pilot_20261114b',
        extension='adaptive_planning_extension_1m_20261114',
        persistence='adaptive_planning_persistence_15m_20261114',
        burn16ln='adaptive_planning_encoder_stability_20261114c',
        burn8ln='adaptive_planning_encoder_burn8_matched_20261114c')
    camps = {k:base/v for k,v in names.items()}
    sources, hashes = {}, {}

    def sha(path):
        path = Path(path).resolve()
        if path not in hashes:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024), b''):
                    h.update(block)
            hashes[path] = h.hexdigest()
        return hashes[path]

    def read(path):
        path = Path(path).resolve()
        sources[str(path)] = sha(path)
        return json.loads(path.read_text())

    def bind(summary):
        if summary['status'] != 'COMPLETE':
            raise ValueError('incomplete result')
        for info in summary['checkpoint'].values():
            if sha(info['path']) != info['sha256']:
                raise ValueError('checkpoint binding changed')

    args.output.mkdir(parents=True, exist_ok=False)
    allocation, histograms, utilities, states_all, utility_details = [], [], [], [], []
    expected = {(s,i,d) for s in (8301,8401) for i in range(8) for d in range(128,1664)}
    specs = []
    for condition in ('control','common','explore','burn8','gru10'):
        folder = 'allocation_250k_burn8' if condition=='burn8' else 'allocation_250k_long'
        specs.append(('parent',condition,250000,camps['parent']/folder,condition+'_'))
    for condition in ('control','burn8','gru10'):
        for step,label in [(500000,'500k'),(1000000,'1m')]:
            specs.append(('extension',condition,step,camps['extension']/f'allocation_{label}_{condition}',condition+'_'))
    for step in (1250000,1500000):
        specs.append(('persistence','gru10',step,camps['persistence']/f'allocation_{step}_gru10','gru10_'))
    for condition in ('burn8ln','burn16ln'):
        specs.append((condition,condition,250000,camps[condition]/f'evaluation_{condition}',''))
    for campaign,condition,milestone,folder,prefix in specs:
        summary = read(folder/f'{prefix}summary.json')
        bind(summary)
        if summary.get('evaluation_epsilon',summary.get('evaluation_train_epsilon')) != 0:
            raise ValueError('allocation includes training exploration')
        rows = read(folder/f'{prefix}allocation.json')
        keys = [(r['seed'],r['stream'],r['decision']) for r in rows]
        if len(keys)!=len(expected) or set(keys)!=expected:
            raise ValueError('mismatched/duplicated evaluation scope')
        lengths = np.array([r['length'] for r in rows])
        if not np.isfinite(lengths).all() or np.any(lengths!=np.floor(lengths)) or np.any((lengths<1)|(lengths>30)):
            raise ValueError('invalid planning length')
        stats = dict(mean_length=float(lengths.mean()),min_fraction=float(np.mean(lengths==1)),
            interior_fraction=float(np.mean((lengths>1)&(lengths<30))),cap_fraction=float(np.mean(lengths==30)))
        for key,value in stats.items():
            if not np.isclose(value,summary[key],rtol=0,atol=1e-12):
                raise ValueError('summary differs from underlying allocation')
        histogram = np.bincount(lengths.astype(int),minlength=31)
        if histogram.tolist()!=summary.get('histogram',summary.get('length_histogram')):
            raise ValueError('histogram differs from underlying allocation')
        allocation.append(dict(campaign=campaign,condition=condition,milestone=milestone,
            actor_env_steps=summary['checkpoint']['actor']['real_step'],decisions=len(rows),
            independent_streams=16,**stats))
        histograms.extend(dict(condition=condition,milestone=milestone,length=k,count=int(n),
            fraction=float(n/len(rows))) for k,n in enumerate(histogram) if k)

    jobs = []
    for condition in ('control','common','explore','burn8','gru10'):
        jobs.append(('parent',condition,250000,'primary',camps['parent']/f'evaluation_{condition}'))
    for campaign,milestone in [('parent',250000),('extension',1000000)]:
        for condition in ('control','burn8','gru10'):
            suffix = '_continued_001' if campaign=='parent' and condition=='burn8' else ''
            jobs.append((campaign,condition,milestone,'late',camps[campaign]/f'utility_late_prefix896_{condition}{suffix}'))
            if campaign=='extension':
                jobs.append((campaign,condition,milestone,'primary',camps[campaign]/f'evaluation_{condition}'))
    jobs.append(('extension','gru10',500000,'count1',camps['extension']/'utility_500k_gru10'))
    for condition in ('burn8ln','burn16ln'):
        jobs.append((condition,condition,250000,'primary',camps[condition]/f'evaluation_{condition}'))
    for campaign,condition,milestone,cohort,folder in jobs:
        summary = read(folder/'summary.json'); bind(summary)
        pairs = read(folder/'pairs.json'); arms = {}
        for row in pairs:
            key = row['seed'],row['count'],row['repetition']
            if key not in arms: arms[key] = read(folder/('arms_%s_%s_%s.json'%key))
        states = paired_states(pairs,arms)
        expected_count = {1,3,6,12} if cohort=='primary' else {1}
        if {(r['seed'],r['stream'],r['count'],r['repetition']) for r in pairs} != {
                (seed,stream,count,rep) for seed in (8301,8401) for stream in range(8)
                for count in expected_count for rep in range(2)}:
            raise ValueError('incomplete registered utility cohort')
        report = utility_report(states)
        meta = dict(campaign=campaign,condition=condition,milestone=milestone,cohort=cohort)
        utility_details.append(dict(**meta,report=report,checkpoint=summary['checkpoint']))
        states_all.extend(dict(**meta,**row) for row in states)
        fields = ['states','paired_repetitions','independent_streams','mean_net_utility',
            'choice_matched_advantage','mean_task_utility','mean_additional_current_stage_computations',
            'ema_utility_rmse','zero_predictor_utility_rmse','ema_rmse_improvement_over_zero',
            'states_with_any_task_difference_exceeding_tail_bound']
        row = dict(**meta,**{k:report[k] for k in fields})
        for field in ('mean_net_utility','choice_matched_advantage','ema_rmse_improvement_over_zero'):
            row[field+'_ci_low'],row[field+'_ci_high'] = report[field+'_ci95']
        utilities.append(row)

    # Continuation54818 replaces incomplete54501; its24old pairs are not new samples.
    recovered = camps['parent']/'utility_late_prefix896_burn8_continued_001'
    audit = read(recovered/'continuation_audit.json')
    current_pairs = read(recovered/'pairs.json')
    old_pairs = read(camps['parent']/'utility_late_prefix896_burn8/pairs.json')
    if len(current_pairs)!=32 or len(old_pairs)!=24 or not all(r in current_pairs for r in old_pairs):
        raise ValueError('late recovery altered original pairs')
    for record in audit['source_group_hashes']:
        if sha(record['path'])!=record['sha256']:
            raise ValueError('reused group changed')
        if Path(record['path']).read_bytes() != (recovered/Path(record['path']).name).read_bytes():
            raise ValueError('recovered group is not identical to original')
    audits = [read(camps['persistence']/'completion_gru10_001/audit.json')]
    audits.extend(read(camps[c]/'completion_audit_001/audit.json') for c in ('burn8ln','burn16ln'))
    audits.extend(read(camps['extension']/f'completion_{c}_001/audit.json') for c in ('control','burn8','gru10'))

    def csv_out(name,rows):
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (args.output/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    csv_out('allocation_summary.csv',allocation)
    csv_out('allocation_histograms.csv',histograms)
    csv_out('utility_summary.csv',utilities)
    csv_out('paired_states.csv',states_all)
    (args.output/'utility_details.json').write_text(json.dumps(utility_details,indent=2,allow_nan=False)+'\n')
    (args.output/'training_audits.json').write_text(json.dumps(audits,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    sample=sorted([r for r in allocation if r['condition']=='gru10'],key=lambda r:r['milestone'])
    x=[r['milestone']/1e6 for r in sample]
    axes[0].plot(x,[r['mean_length'] for r in sample],'-o',color='#0072B2')
    axes[0].set(xlabel='Training milestone (million learner steps)',ylabel='Mean imagination length',
                title='GRU contrast gradient 10x: improvement did not persist',xticks=x,ylim=(0,6))
    for field,label,color in [('min_fraction','1 query','#D55E00'),('interior_fraction','2–29 queries','#0072B2'),('cap_fraction','30 queries','#777777')]:
        axes[1].plot(x,[100*r[field] for r in sample],'-o',label=label,color=color)
    axes[1].set(xlabel='Training milestone (million learner steps)',ylabel='Decisions (%)',
                title='Same registered evaluation window; epsilon = 0',xticks=x,ylim=(-2,102))
    axes[1].legend(frameon=False)
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(args.output/'gru10_persistence.png',dpi=180);fig.savefig(args.output/'gru10_persistence.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for c,color,label in [('burn8ln','#0072B2','Burn-in 8 + normalization'),('burn16ln','#D55E00','Burn-in 16 + normalization')]:
        sample=[r for r in histograms if r['condition']==c]
        axes[0].plot([r['length'] for r in sample],[100*r['fraction'] for r in sample],'-o',ms=3,label=label,color=color)
    axes[0].set(xlabel='Imagination queries per decision',ylabel='Decisions (%)',title='250k: numerical stability with normalization',xticks=[1,5,10,15,20,25,30])
    axes[0].legend(frameon=False,fontsize=9)
    chosen=[next(r for r in utilities if r['condition']==c and r['milestone']==m and r['cohort']=='primary')
            for c,m in [('control',1000000),('burn8',1000000),('gru10',1000000),('burn8ln',250000),('burn16ln',250000)]]
    labels=['Control, 1M','Burn-in 8, 1M','GRU 10x, 1M','Normalized burn-in 8, 250k','Normalized burn-in 16, 250k']
    # Points are separate conditional estimates, not a pooled condition contrast.
    for i,row in enumerate(chosen):
        y=row['choice_matched_advantage'];lo=row['choice_matched_advantage_ci_low'];hi=row['choice_matched_advantage_ci_high']
        axes[1].errorbar(y,i,xerr=[[y-lo],[hi-y]],fmt='o',color='#333333',capsize=3)
    axes[1].axvline(0,color='#888888',ls='--');axes[1].set(yticks=range(len(labels)),yticklabels=labels,
        xlabel='Net utility gain over matched CONTINUE frequency',title='Useful allocation remains unestablished (95% CI)')
    axes[1].invert_yaxis()
    fig.savefig(args.output/'normalization_and_utility.png',dpi=180);fig.savefig(args.output/'normalization_and_utility.pdf');plt.close(fig)
    manifest=dict(status='COMPLETE_FOR_LISTED_COHORTS',created_at=time.time(),sources=sources,
        checkpoint_hashes={str(p):h for p,h in hashes.items() if 'ckp_' in p.name},
        allocation_cohorts=len(allocation),utility_cohorts=len(utilities),
        recovered_late_pairs=dict(total=32,reused=24,new=8,double_counted=False),
        scope='Enduro sub001 all sessions; one training seed. Evaluations reuse seeds/streams. ',
        limitations=['No inference that broad distributions establish useful computation.',
            '1.25M and1.5M were cold collector/GRU/model-replay continuations with restored weights/optimizers.',
            'Matched time/seed does not mean the same observation across policies.',
            'Primary and late utility share16streams and must not be treated as independent cohorts.',
            'Burn-in8/16 normalized pilots share architecture/seed but had559/593updates and35776/37952scored BCtargets.',
            'Raw burn16 diverged and is excluded; raw burn8 recovered from a sustained292512–309840NLLspike.',
            'Selected500k utility missing shard4 remains a separate recovery; excluded here.',
            'Bootstrap intervals are conditional on each trained checkpoint, not across training seeds.'],
        decisions=['Do not call1M histogram a stable solution.',
            'Normalization resolved observed250k encoder/loss instability; durability remains untested.',
            'Do not tune gate to force a broad histogram before reward utility is demonstrated.'])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    shutil.copy2(__file__,args.output/Path(__file__).name)
    print(json.dumps(dict(output=str(args.output),allocation_cohorts=len(allocation),utility_cohorts=len(utilities))))


if __name__=='__main__':main()
