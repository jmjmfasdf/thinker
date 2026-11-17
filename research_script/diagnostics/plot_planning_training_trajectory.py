"""Counter-validated training allocation over actual cumulative environment actions.

Training exploration remains enabled. These changing-policy observations are
not a substitute for the separately registered fixed-checkpoint evaluation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_intervals(path, offset=0):
    raw = path.read_bytes()
    complete = raw[:raw.rfind(b'\n')+1]
    rows = [json.loads(line) for line in complete.splitlines() if line]
    prior_hist = np.zeros(31,dtype=np.int64)
    prior_step = 0
    result = []
    for row in rows:
        hist = np.array([row[f'count_{k}_decisions'] for k in range(31)],dtype=np.int64)
        local = row['physical_env_steps']
        if hist.sum()!=local or (hist<prior_hist).any() or local<prior_step:
            raise ValueError('cumulative histogram does not match monotonic physical actions')
        delta = hist-prior_hist
        window = np.array([row[f'window_count_{k}_decisions'] for k in range(31)],dtype=np.int64)
        if not np.array_equal(delta,window) or delta.sum()!=row['window_decisions']:
            raise ValueError('window counter does not match cumulative histogram difference')
        n = int(delta.sum())
        if n:
            if delta[0] or not np.isclose(float(np.dot(delta,np.arange(31))/n),row['window_mean_length']):
                raise ValueError('invalid minimum or mean planning length')
            result.append(dict(first_physical_step=offset+prior_step,last_physical_step=offset+local,
                decisions=n,mean_length=row['window_mean_length'],min_fraction=float(delta[1]/n),
                interior_fraction=float(delta[2:30].sum()/n),cap_fraction=float(delta[30]/n),hist=delta.tolist()))
        prior_hist,prior_step = hist,local
    return result,dict(path=str(path),read_sha256=hashlib.sha256(raw).hexdigest(),
        complete_prefix_sha256=hashlib.sha256(complete).hexdigest(),complete_prefix_bytes=len(complete),
        ignored_partial_line_bytes=len(raw)-len(complete),physical_env_steps=prior_step,offset=offset)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    camp=args.campaign.resolve();repo=camp.parents[3]
    manifest=json.loads((camp/'manifest.json').read_text())
    sources,intervals,bins,resumes=[],[],[],{}
    conditions=('control','burn8','gru10')
    for item in manifest['runs']:
        condition=item['condition']
        if condition not in conditions:
            continue
        parent=Path(item['parent_checkpoint']['actor']['path']).parent
        before,source=read_intervals(parent/'planning_intervals.jsonl')
        sources.append(dict(condition=condition,segment='parent',**source))
        offset=source['physical_env_steps'];resumes[condition]=offset
        after,source=read_intervals(repo/'logs/thinker'/item['xpid']/'planning_intervals.jsonl',offset)
        sources.append(dict(condition=condition,segment='continuation',**source))
        combined=before+after
        intervals.extend(dict(condition=condition,**r) for r in combined)
        groups={}
        for row in combined:
            groups.setdefault((row['last_physical_step']-1)//50000,[]).append(row)
        for index,rows in sorted(groups.items()):
            hist=np.sum([r['hist'] for r in rows],axis=0)
            n=int(hist.sum());lengths=np.array([r['mean_length'] for r in rows])
            bins.append(dict(condition=condition,bin_index=index,nominal_end_step=(index+1)*50000,
                first_observed_step=min(r['first_physical_step'] for r in rows),
                last_observed_step=max(r['last_physical_step'] for r in rows),decisions=n,
                mean_length=float(hist@np.arange(31)/n),min_fraction=float(hist[1]/n),
                interior_fraction=float(hist[2:30].sum()/n),cap_fraction=float(hist[30]/n),
                interval_mean_p10=float(np.quantile(lengths,.1)),interval_mean_p90=float(np.quantile(lengths,.9)),
                complete_bin=max(r['last_physical_step'] for r in combined)>=(index+1)*50000))
    args.output.mkdir(parents=True,exist_ok=False)
    labels=dict(control='Control (training epsilon 0.02)',burn8='Burn-in 8 (epsilon 0.10)',
                gru10='Burn-in 8 + GRU gradient 10x (epsilon 0.10)')
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
    for row,condition in enumerate(conditions):
        sample=[r for r in bins if r['condition']==condition and r['complete_bin']]
        fine=[r for r in intervals if r['condition']==condition]
        x=np.array([r['last_observed_step'] for r in sample])/1e6
        ax=axes[row,0]
        ax.plot([r['last_physical_step']/1e6 for r in fine],[r['mean_length'] for r in fine],
            color='#b6c9db',alpha=.45,lw=.6,label='Counter interval')
        ax.plot(x,[r['mean_length'] for r in sample],color='#175f94',lw=2,label='50k-bin weighted mean')
        ax.set_ylim(0,30);ax.set_ylabel('Mean imagination length');ax.set_title(labels[condition],fontsize=10)
        axes[row,1].stackplot(x,[[r[k] for r in sample] for k in ('min_fraction','interior_fraction','cap_fraction')],
            colors=['#c7c7c7','#2a9d8f','#df8456'],labels=['Length 1','Length 2–29','Length 30'],alpha=.9)
        axes[row,1].set_ylim(0,1);axes[row,1].set_ylabel('Fraction of actual actions')
        for current in axes[row]:
            current.axvline(resumes[condition]/1e6,color='.3',ls=':',lw=1)
            current.grid(axis='y',alpha=.15)
    axes[0,0].legend(fontsize=8,loc='upper right');axes[0,1].legend(fontsize=8,loc='upper right')
    for ax in axes[-1]:ax.set_xlabel('Cumulative actual environment actions (millions)')
    fig.suptitle('Training allocation trajectory — changing policy, exploration enabled',fontsize=13)
    fig.text(.5,.018,'Enduro sub001, one seed. Dotted line: cold environment/replay restart at continuation.\n'
        '50k summaries show completed bins; the pale line includes the newest intervals.\n'
        'Burn-in 8 had a transient large BC loss spike. Training distributions do not establish computation utility.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.96))
    for suffix in ('png','pdf'):fig.savefig(args.output/f'training_allocation.{suffix}',dpi=180)
    plt.close(fig)
    with (args.output/'bins.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(bins[0]));writer.writeheader();writer.writerows(bins)
    with (args.output/'intervals.csv').open('w') as stream:
        flat=[{k:v for k,v in r.items() if k!='hist'} for r in intervals]
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    audit=dict(status='CUMULATIVE_AND_WINDOW_COUNTERS_AGREE',sources=sources,resume_offsets=resumes,
        total_actions_by_condition={c:sum(r['decisions'] for r in intervals if r['condition']==c) for c in conditions},
        binning='Whole counter intervals assigned by endpoint to nominal 50k physical-action bins. Last bins may be incomplete.',
        maximum_interval_actions=max(r['decisions'] for r in intervals),
        scope='Actual physical environment-action clock, not learner update clock. Includes training exploration and evolving weights.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit['total_actions_by_condition']))
    for r in bins:
        if r['nominal_end_step']>=500000:print(json.dumps(r))


if __name__=='__main__':main()
