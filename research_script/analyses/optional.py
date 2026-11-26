"""Explicit additional-input analyses; never fabricate real utility from replay."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from common.paths import OUTPUTS,SCRIPTS,REPO,output,GAMES
from common.io import sha,write_json
from common.reporting import save_tables

def utility(outputs):
    root=output(10,outputs);root.mkdir(parents=True,exist_ok=True)
    source=OUTPUTS/'voc_utility_execution_sub001_tau001_20260930'
    if not (source/'utility_states.csv').exists():
        write_json(root/'manifest.json',dict(status='BLOCKED_INPUT',reason='Human replay cannot supply paired environment STOP/CONTINUE outcomes',required=str(source)))
        return 'BLOCKED_INPUT'
    d=pd.read_csv(source/'utility_states.csv');scores=pd.read_csv(source/'utility_scores.csv')
    provenance=json.loads((source/'manifest.json').read_text());verified=[]
    for rec in provenance['utility_sources']:
        for path_key,sha_key in [('path','sha256'),('pairs_path','pairs_sha256')]:
            f=Path(rec[path_key])
            if not f.exists() or sha(f)!=rec[sha_key]:raise ValueError('Paired evaluation provenance mismatch: '+str(f))
            verified.append(str(f))
    if not np.allclose(d.utility,d.task_reward_difference-d.computation_cost,atol=1e-10):raise ValueError('Utility decomposition mismatch')
    rows=[];rng=np.random.default_rng(20261111)
    for (game,protocol,step),part in d.groupby(['game','protocol','nominal_step']):
        part=part.copy();part['cluster']=part.seed.astype(str)+'_'+part.stream.astype(str)
        part=part.dropna(subset=['predicted_voc','utility'])
        x=part.predicted_voc.to_numpy();y=part.utility.to_numpy();sd=x.std()
        if sd<1e-10:beta=np.nan
        else:beta=float(np.linalg.lstsq(np.column_stack([np.ones(len(x)),(x-x.mean())/sd]),y,rcond=None)[0][1])
        estimates=[];clusters=part.cluster.unique()
        if sd>1e-10 and len(clusters)>1:
            for _ in range(1000):
                sample=pd.concat([part.loc[part.cluster.eq(c)] for c in rng.choice(clusters,len(clusters),replace=True)])
                xx=(sample.predicted_voc-x.mean())/sd
                if xx.std()>1e-10:estimates.append(np.linalg.lstsq(np.column_stack([np.ones(len(xx)),xx]),sample.utility,rcond=None)[0][1])
        ci=np.quantile(estimates,[.025,.975]) if estimates else [np.nan,np.nan]
        rows.append(dict(game=game,protocol=protocol,nominal_step=step,n_states=len(part),n_streams=len(clusters),
            utility_mean=y.mean(),utility_voc_slope=beta,slope_ci_low=ci[0],slope_ci_high=ci[1],
            spearman=float(stats.spearmanr(x,y).statistic) if sd>0 and y.std()>0 else np.nan))
    tables={'state_outcomes':d,'previous_pilot_scores':scores,'voc_utility_regression':pd.DataFrame(rows)}
    save_tables(root,tables)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for game,part in scores.loc[scores.protocol.eq('standard')].groupby('game'):
        part=part.sort_values('nominal_step')
        axes[0].plot(part.nominal_step/1e6,part.mean_task_reward_difference,'o-',label=game)
        axes[1].plot(part.nominal_step/1e6,part.mean_net_utility,'o-',label=game)
    for ax,title in zip(axes,['Observed discounted reward difference','Net utility after computation cost']):
        ax.axhline(0,color='gray',lw=.6);ax.set(xlabel='Training environment steps (millions)',ylabel=title);ax.legend()
    fig.suptitle('Existing sub001 paired environment pilot · temperature 0.01');fig.tight_layout()
    (root/'figures').mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(root/'figures'/f'paired_utility.{ext}',dpi=180)
    plt.close(fig)
    write_json(root/'manifest.json',dict(status='COMPLETED_SUPPLEMENTARY',scope='Existing sub001 paired pilot; not new 72-checkpoint evaluations',
        source=str(source),source_hashes={n:sha(source/n) for n in ['utility_states.csv','utility_scores.csv','manifest.json']},
        verified_paired_source_files=verified,
        raw_environment_evaluation_launched=False,score_evaluation_launched=False,
        independent_resampling='seed x environment stream; paired repeats remain within state',
        limitations=['Human replay cache alone cannot estimate causal task-reward benefit.',
                     'Finite-horizon pilot outcomes differ from the critic long-horizon target.',
                     'No inference from sub001 to six-subject training population.']))
    return 'COMPLETED_SUPPLEMENTARY'
