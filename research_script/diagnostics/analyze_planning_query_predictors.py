"""Does value information describe within-search gate changes beyond count?

Exploratory regression of recorded gate log-odds changes. Entire seed/stream
groups are withheld; each multi-query decision has equal weight. Optional risk
sets select surviving searches, so this does not estimate a causal effect of
value updating, nor the reward utility of more computation.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from probe_adaptive_planning_queries import query_report


def logodds(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p)-np.log1p(-p)


def value_summary(row):
    q=np.asarray(row['root_q_mean'],dtype=float)
    visits=np.asarray(row['root_visits'],dtype=float)
    if q.ndim!=1 or q.shape!=visits.shape or not np.isfinite(q).all() or not np.isfinite(visits).all() or np.any(visits<0):
        raise ValueError('invalid root action values or visits')
    seen=visits>0
    total=visits.sum()
    mass=visits/total if total>0 else np.zeros_like(visits)
    entropy=-np.sum(mass[mass>0]*np.log(mass[mass>0]))
    return q,seen,np.array([seen.sum(),np.log1p(total),mass.max(),entropy])


def design(queries, response_scale='logodds'):
    if response_scale not in ('logodds', 'probability'):
        raise ValueError('unknown response scale')
    # The probability sensitivity keeps every observed row, including saturated
    # gates. Only FP32 probability-sum roundoff is bounded to [0, 1].
    transform = logodds if response_scale == 'logodds' else lambda p: np.clip(p, 0., 1.)
    grouped=defaultdict(list)
    for row in queries:
        grouped[(row['seed'],row['stream'],row['decision'])].append(row)
    base,extra,target,metadata,weights=[],[],[],[],[]
    clipped=0
    for key,rows in sorted(grouped.items()):
        rows=sorted(rows,key=lambda r:r['count'])
        if rows[0]['count']!=1 or [r['count'] for r in rows]!=list(range(1,len(rows)+1)):
            raise ValueError('missing or duplicated query count')
        if len(rows)<2:
            continue
        first=rows[0];q0,seen0,structure0=value_summary(first)
        p0=float(first['probability'])
        if not np.isfinite(p0) or not -1e-6<=p0<=1+1e-6:
            raise ValueError('invalid first probability')
        l0=transform(p0)
        for row in rows[1:]:
            p=float(row['probability']);count=row['count']
            if not np.isfinite(p) or not -1e-6<=p<=1+1e-6 or not 2<=count<=29:
                raise ValueError('invalid later probability/count')
            q,seen,structure=value_summary(row)
            common=seen&seen0
            difference=q[common]-q0[common]
            value_features=[difference.mean() if common.any() else 0.,
                np.abs(difference).mean() if common.any() else 0.,
                (q[seen].max() if seen.any() else 0.)-(q0[seen0].max() if seen0.any() else 0.),
                row['rollout_return']-first['rollout_return']]
            # Initial values are controls. Added features describe updates.
            context=[l0,l0*np.log1p(count),row['depth'],row['depth']/count,
                *structure,*structure0,np.count_nonzero(seen&~seen0),common.sum(),
                q0[seen0].mean() if seen0.any() else 0.,q0[seen0].max() if seen0.any() else 0.]
            base.append([*np.eye(28)[count-2],*context]);extra.append(value_features)
            target.append(transform(p)-l0)
            metadata.append(dict(seed=key[0],stream=key[1],decision=key[2],count=count))
            weights.append(1/(len(rows)-1))
            clipped+=int(p<1e-6 or p>1-1e-6 or p0<1e-6 or p0>1-1e-6)
    arrays=[np.asarray(x,dtype=float) for x in (base,extra,target,weights)]
    if not metadata or any(not np.isfinite(x).all() for x in arrays):
        raise ValueError('no valid multi-query decisions')
    return *arrays,metadata,clipped


def crossfit(x,y,weights,groups):
    unique=np.unique(groups)
    if len(unique)<4:
        raise ValueError('need at least four independent stream groups')
    predictions=np.full(len(y),np.nan);folds=[]
    for train,test in GroupKFold(n_splits=4).split(x,y,groups):
        if set(groups[train])&set(groups[test]):
            raise ValueError('train/test stream overlap')
        w=weights[train]/weights[train].mean()
        scale=StandardScaler().fit(x[train],sample_weight=w)
        fit=Ridge(alpha=1.).fit(scale.transform(x[train]),y[train],sample_weight=w)
        predictions[test]=fit.predict(scale.transform(x[test]))
        folds.append(dict(train_groups=sorted(set(groups[train].tolist())),test_groups=sorted(set(groups[test].tolist()))))
    if not np.isfinite(predictions).all():
        raise ValueError('incomplete crossfit predictions')
    return predictions,folds


def restore_allocations(queries,stages):
    """Stages store clipped probabilities; the verifier expects raw FP32 sums."""
    grouped=defaultdict(list)
    for row in queries:grouped[(row['seed'],row['stream'],row['decision'])].append(row)
    allocations=[]
    for row in stages:
        observed=sorted(grouped[(row['seed'],row['stream'],row['decision'])],key=lambda r:r['count'])
        if not observed:raise ValueError('stage has no queries')
        p=np.array([r['probability'] for r in observed])
        if np.clip(p[0],0,1)!=row['first_probability']:
            raise ValueError('stage first probability differs from query record')
        signs=np.sign(np.clip(p,0,1)-.5)
        flips=int(np.sum((signs[1:]!=signs[:-1])&(signs[1:]!=0)&(signs[:-1]!=0)))
        allocations.append(dict(row,first_optional_probability=float(p[0]),sign_changes=flips))
    return allocations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--condition',default='gru10')
    parser.add_argument('--response-scale',choices=['logodds','probability'],default='logodds')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.input.resolve();condition=args.condition
    summary=json.loads((root/f'{condition}_summary.json').read_text())
    completion=json.loads((root/'completion.json').read_text())
    if summary['status']!='COMPLETE' or completion['status']!='COMPLETE' or condition not in completion['conditions']:
        raise ValueError('complete query observation required')
    with gzip.open(root/f'{condition}_queries.json.gz','rt') as stream:queries=json.load(stream)
    stages=json.loads((root/f'{condition}_stages.json').read_text())
    allocations=restore_allocations(queries,stages)
    sequence_report,_=query_report(queries,allocations,minimum=1,cap=30)
    if sequence_report['decisions']!=summary['decisions'] or sequence_report['optional_queries']!=summary['optional_queries']:
        raise ValueError('query coverage differs from summary')
    base,extra,y,weight,metadata,clipped=design(queries,args.response_scale)
    group=np.array([f"{r['seed']}:{r['stream']}" for r in metadata])
    predictions,folds={},None
    for name,x in [('count_structure_initial',base),('plus_value_updates',np.column_stack([base,extra]))]:
        predictions[name],split=crossfit(x,y,weight,group)
        if folds is not None and split!=folds:raise ValueError('models used different test folds')
        folds=split
    decisions=defaultdict(list)
    for i,row in enumerate(metadata):decisions[(row['seed'],row['stream'],row['decision'])].append(i)
    decision_rows=[]
    for key,index in decisions.items():
        decision_rows.append(dict(seed=key[0],stream=key[1],decision=key[2],later_queries=len(index),
            **{name+'_mse':float(np.mean((y[index]-p[index])**2)) for name,p in predictions.items()}))
    stream_rows=[]
    for key in sorted({(r['seed'],r['stream']) for r in decision_rows}):
        rows=[r for r in decision_rows if (r['seed'],r['stream'])==key]
        values={name+'_mse':float(np.mean([r[name+'_mse'] for r in rows])) for name in predictions}
        stream_rows.append(dict(seed=key[0],stream=key[1],decisions=len(rows),**values,
            mse_improvement=values['count_structure_initial_mse']-values['plus_value_updates_mse']))
    improvement=np.array([r['mse_improvement'] for r in stream_rows])
    rng=np.random.default_rng(20261114);draws=improvement[rng.integers(0,len(improvement),(4000,len(improvement)))].mean(axis=1)
    result=dict(status='COMPLETE_EXPLORATORY_STREAM_HELDOUT_REGRESSION',condition=condition,checkpoint=summary['checkpoint'],
        all_decisions=summary['decisions'],multi_query_decisions=len(decision_rows),later_optional_queries=len(y),
        independent_streams=len(stream_rows),
        response_scale=args.response_scale,
        clipped_logodds_rows=clipped if args.response_scale=='logodds' else 0,
        logodds_clipping_candidate_rows=clipped,folds=folds,
        stream_equal_rmse={name:float(np.sqrt(np.mean([r[name+'_mse'] for r in stream_rows]))) for name in predictions},
        mean_mse_improvement=float(improvement.mean()),descriptive_cluster_ci95=np.quantile(draws,[.025,.975]).tolist(),
        model='Fixed Ridge alpha1; weighted standardization trained inside each fold. No hyperparameter search.',
        baseline='Query-count indicators, initial gate '+args.response_scale+' and count interaction, current depth, visit structure, initial value/structure, newly visited action count.',
        added='Signed and absolute root-Q change for commonly visited actions; maximum root-Q change; rollout-return change.',
        limitations=['Only observed later optional queries: searches that already stopped have no later outcome.',
            'Equal weights per multi-query decision; stream-equal reported MSE. Four folds with whole seed/stream groups held out.',
            'Bootstrap describes variation of fixed crossfit errors, not uncertainty from refitting all models.',
            'This predicts the model gate itself; it establishes neither causal use of a feature nor computation reward utility.',
            'Value-update features may still reflect expansion mechanics, other changing state variables, or common network inputs.',
            'Probability scale is an exploratory sensitivity to logodds clipping; report both scales without selecting favorable results. Ridge predictions are not constrained probabilities.'])
    args.output.mkdir(parents=True,exist_ok=False)
    for filename,rows in [('decision_errors.csv',decision_rows),('stream_errors.csv',stream_rows),('query_count_summary.csv',sequence_report['by_count'])]:
        with (args.output/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    source=args.output/'analysis_sources';source.mkdir()
    for name in [Path(__file__).name,'probe_adaptive_planning_queries.py']:
        shutil.copy2(Path(__file__).parent/name,source/name)
    result['inputs_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        root/f'{condition}_summary.json',root/f'{condition}_queries.json.gz',root/f'{condition}_stages.json',root/'completion.json')}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('multi_query_decisions','later_optional_queries','stream_equal_rmse','mean_mse_improvement','descriptive_cluster_ci95')},indent=2))


if __name__=='__main__':main()
