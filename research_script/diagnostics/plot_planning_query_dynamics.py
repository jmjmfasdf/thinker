"""Plot completed within-search diagnostics without interpreting them as utility."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.campaign.resolve()
    source=base/'query_dynamics_1m_gru10/gru10_summary.json'
    regression=base/'query_predictors_1m_gru10_001/audit.json'
    completion=json.loads((source.parent/'completion.json').read_text())
    summary=json.loads(source.read_text());audit=json.loads(regression.read_text())
    if completion['exact_reference_decisions']!=24576 or summary['status']!='COMPLETE':
        raise ValueError('complete registered1M query observation required')
    if summary['checkpoint']!=audit['checkpoint'] or audit['independent_streams']!=16:
        raise ValueError('regression and query checkpoints/coverage differ')
    rows=summary['by_count']
    counts=np.array([r['count'] for r in rows])
    support=np.array([r['at_risk_decisions'] for r in rows])
    now=np.array([r['predicted_continue_mean'] for r in rows])
    initial=now-np.array([r['mean_probability_change_from_first'] for r in rows])
    if np.any((initial<0)|(initial>1)):
        raise ValueError('invalid paired initial probabilities')
    shown=support>=500
    names=['count_structure_initial','plus_value_updates']
    mse=np.array([audit['stream_equal_rmse'][k]**2 for k in names])
    improvement=1-mse[1]/mse[0]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),gridspec_kw={'width_ratios':[1.55,1]})
    axes[0].plot(counts[shown],initial[shown],'o--',color='#5a708b',label='First probability, same surviving decisions')
    axes[0].plot(counts[shown],now[shown],'o-',color='#118c9c',label='Current probability')
    axes[0].set(xlabel='Queries already performed (k)',ylabel='Mean probability of continuing',ylim=(0,1),
        title='Continuing searches start with stronger support')
    axes[0].set_xticks([1,2,4,6,8,10,12]);axes[0].legend(loc='lower right',fontsize=8)
    axes[0].grid(axis='y',alpha=.2)
    axes[1].bar([0,1],mse,color=['#9da7b4','#118c9c'],width=.6)
    for i,value in enumerate(mse):axes[1].text(i,value+.025,f'{value:.3f}',ha='center')
    axes[1].set_xticks([0,1],['Count, structure,\ninitial state','Also include\nvalue updates'])
    axes[1].set(ylabel='Held-out MSE of gate log-odds change',ylim=(0,float(mse.max()*1.25)),
        title=f'Value updates reduce MSE by {improvement:.1%}')
    axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Enduro sub001, 1M, GRU gradient 10x: within-search control',fontsize=13)
    fig.text(.5,.015,'24,576 decisions; 16 seed/stream groups; no training epsilon. Left: risk sets change with k; shown where n >= 500.\n'
        'Right: four folds withholding entire streams; each multi-query decision equally weighted. Descriptive, not reward utility.',
        ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.1,1,.94))
    args.output.mkdir(parents=True,exist_ok=False)
    for suffix in ('png','pdf'):fig.savefig(args.output/f'within_search_control.{suffix}',dpi=180)
    plt.close(fig)
    with (args.output/'query_counts.csv').open('w') as stream:
        records=[dict(r,paired_initial_probability=float(initial[i]),shown=bool(shown[i])) for i,r in enumerate(rows)]
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    report=dict(status='COMPLETE',checkpoint=summary['checkpoint'],total_decisions=24576,
        queried_decisions=audit['multi_query_decisions'],later_optional_queries=audit['later_optional_queries'],
        baseline_mse=float(mse[0]),plus_values_mse=float(mse[1]),relative_mse_improvement=float(improvement),
        inference='Descriptive survivor-conditioned gate dynamics. Regression association is not causal use of value or useful computation.',
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,regression,Path(__file__)]})
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(__file__,args.output/Path(__file__).name)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
