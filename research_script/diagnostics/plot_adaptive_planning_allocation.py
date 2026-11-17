"""Plot completed fixed-checkpoint allocation samples with missing panels explicit."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_adaptive_planning_distribution import fit_constant_gate, constant_gate_distribution
from summarize_adaptive_planning_evidence import checked_probabilities


CONDITIONS = ['control','common','explore','burn8','burn16','gru10']
LABELS = dict(control='Original critic', common='Common value + contrast',
              explore='Higher training exploration', burn8='Burn-in 8',
              burn16='Burn-in 16', gru10='Burn-in 8 + GRU gradient 10x')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--allocation-root', action='append', type=Path, required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    data, sources = {}, []
    for condition in CONDITIONS:
        matches = [root for root in args.allocation_root if (root/f'{condition}_summary.json').exists()]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError('ambiguous completed sample: '+condition)
        root = matches[0]
        summary_bytes = (root/f'{condition}_summary.json').read_bytes()
        rows_bytes = (root/f'{condition}_allocation.json').read_bytes()
        summary, rows = json.loads(summary_bytes), json.loads(rows_bytes)
        if summary['status'] != 'COMPLETE' or summary['evaluation_epsilon'] != 0:
            raise ValueError('only complete, exploration-free allocation is eligible')
        if len(rows) != summary['decisions']:
            raise ValueError('sample count differs from completion record')
        for name, blob in [('summary',summary_bytes),('allocation',rows_bytes)]:
            sources.append(dict(path=str((root/f'{condition}_{name}.json').resolve()),
                sha256=hashlib.sha256(blob).hexdigest()))
        length = np.array([row['length'] for row in rows])
        probability, roundoff = checked_probabilities([row['first_optional_probability'] for row in rows])
        identities = [(r['seed'], r['stream'], r['decision']) for r in rows]
        if len(set(identities)) != len(rows):
            raise ValueError('duplicate allocation decisions')
        scope = dict(decisions=len(rows), independent_streams=len({key[:2] for key in identities}),
            seeds=sorted({key[0] for key in identities}),
            observed_decision_range=[min(key[2] for key in identities), max(key[2] for key in identities)],
            actor_env_step=summary['checkpoint']['actor']['real_step'])
        histogram = np.bincount(length,minlength=31)
        fit = fit_constant_gate(histogram)
        data[condition] = dict(summary=summary,length=length,probability=probability,
            histogram=histogram,constant_gate=fit,scope=scope,roundoff=roundoff)
        outcome = root.parent/f'training_outcome_{condition}.json'
        if outcome.exists():
            raw = outcome.read_bytes()
            data[condition]['training_outcome'] = json.loads(raw)
            sources.append(dict(path=str(outcome.resolve()), sha256=hashlib.sha256(raw).hexdigest()))
    for kind in ('length','first_probability'):
        fig, axes = plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
        for ax,condition in zip(axes.flat,CONDITIONS):
            ax.set_title(LABELS[condition],fontsize=11)
            ax.set_ylim(0,1)
            ax.grid(axis='y',alpha=.15)
            if condition not in data:
                ax.text(.5,.5,'Evaluation pending',transform=ax.transAxes,ha='center',color='.5')
                continue
            value = data[condition];summary=value['summary']
            if value.get('training_outcome', {}).get('status') == 'SUSTAINED_TRAINING_DIVERGENCE_WITH_FINITE_CHECKPOINT':
                ax.set_title(LABELS[condition]+' — training diverged', fontsize=11, color='#b22222')
            if kind == 'length':
                empirical = value['histogram']/len(value['length'])
                ax.bar(np.arange(1,31),empirical[1:],color='#377eb8',width=.85,label='Observed')
                fitted=constant_gate_distribution(value['constant_gate']['p_continue_fit'])
                ax.plot(np.arange(1,31),fitted[1:],color='#333333',lw=1.4,label='Constant-probability gate')
                note=f"Mean {summary['mean_length']:.2f}; min {summary['min_fraction']:.1%}; cap {summary['cap_fraction']:.1%}"
                ax.set_xlim(.4,30.6)
            else:
                bins=np.linspace(0,1,21)
                hist,_=np.histogram(value['probability'],bins=bins)
                ax.bar((bins[:-1]+bins[1:])/2,hist/len(value['probability']),width=.045,color='#4daf4a')
                ax.axvline(.5,color='#333333',ls='--',lw=1)
                note=f"SD {value['probability'].std():.3f}; sign changes {summary['within_decision_sign_change_fraction']:.1%}"
                ax.set_xlim(0,1)
            step=summary['checkpoint']['actor']['real_step']
            first,last=value['scope']['observed_decision_range']
            ax.text(.03,.96,note+f"\nStep {step:,}; n={len(value['length']):,}; decisions {first}\u2013{last}",
                transform=ax.transAxes,va='top',fontsize=8)
        for ax in axes[:,0]:
            ax.set_ylabel('Fraction of decisions')
        for ax in axes[-1]:
            ax.set_xlabel('Imagination computations' if kind=='length' else 'P(continue) at first optional query')
        if kind=='length' and data:
            next(ax for ax,c in zip(axes.flat,CONDITIONS) if c in data).legend(fontsize=8,loc='center right')
        fig.suptitle('Fixed checkpoint allocation: stochastic gate, training exploration disabled',fontsize=13)
        fig.text(.5,.015,'Enduro sub001 pilot; completed samples shown. Allocation diversity does not establish reward utility.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.95))
        for ext in ('png','pdf'):
            fig.savefig(args.output/f'{kind}.{ext}',dpi=180)
        plt.close(fig)
    manifest=dict(completed_conditions=list(data),pending_conditions=[c for c in CONDITIONS if c not in data],
        sources=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sampling={condition:value['scope'] for condition,value in data.items()},
        probability_roundoff={condition:value['roundoff'] for condition,value in data.items()},
        utility_evaluation='Not shown; paired reward utility is a separate evaluation',
        training_outcomes={condition:value.get('training_outcome', {}).get('status', 'NOT_INDEPENDENTLY_CLASSIFIED')
                           for condition,value in data.items()},
        constant_gate_null='One fitted continuation probability per checkpoint, capped at 30, minimum 1')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['completed_conditions','pending_conditions']}))


if __name__=='__main__':
    main()
