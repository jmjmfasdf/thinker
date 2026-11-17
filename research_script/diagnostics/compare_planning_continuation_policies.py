"""Compare completed epsilon-zero efficacy and training-policy Q calibration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from summarize_adaptive_planning_evidence import paired_states, utility_report, read_bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    sources, loaded = [], {}
    for label,folder in [('epsilon0',args.reference),('epsilon_training',args.calibration)]:
        summary = read_bound(folder/'summary.json',sources)
        if summary['status'] != 'COMPLETE':
            raise ValueError('only complete evaluations may be compared')
        pairs = read_bound(folder/'pairs.json',sources)
        arms = {}
        for row in pairs:
            key = (row['seed'],row['count'],row['repetition'])
            if key not in arms:
                arms[key] = read_bound(folder/('arms_%s_%s_%s.json'%key),sources)
        states = paired_states(pairs,arms)
        if len(pairs)!=summary['pairs'] or len(states)!=summary['paired_states']:
            raise ValueError('summary counts differ from paired observations')
        loaded[label] = dict(summary=summary,arms=arms,states=states)
    old,new = loaded['epsilon0'],loaded['epsilon_training']
    if old['summary']['checkpoint'] != new['summary']['checkpoint']:
        raise ValueError('checkpoint mismatch')
    if old['summary']['evaluation_train_epsilon'] != 0:
        raise ValueError('reference must be epsilon zero')
    if new['summary']['prefix_epsilon'] != 0 or new['summary']['entry_epsilon'] != 0:
        raise ValueError('calibration altered prefix or entry policy')
    audit = read_bound(args.calibration/'matched_prefix_audit.json',sources)
    if audit['status'] != 'SAME_INTERVENTION_ENTRIES_ACROSS_CONTINUATION_POLICIES_PASS':
        raise ValueError('incomplete cross-policy entry audit')
    for entry in old['summary']['checkpoint'].values():
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('checkpoint bytes no longer match the evaluation')
    epsilon = new['summary']['continuation_train_epsilon']
    if set(old['arms']) != set(new['arms']):
        raise ValueError('different intervention coverage')
    checked = []
    for key,a in old['arms'].items():
        b = new['arms'][key]
        for arm in ('stop','continue_'):
            if a[arm]['entries'] != b[arm]['entries'] or a[arm]['prefix'] != b[arm]['prefix']:
                raise ValueError('different exact intervention entries')
            check = b[arm]['continuation_policy_check']
            if not check['switched'] or check['future_epsilon'] != epsilon:
                raise ValueError('continuation policy was not applied')
            checked.append(dict(key=key,arm=arm,matched_entries=len(a[arm]['entries']),
                maximum_probability_error=check['maximum_probability_error']))
    def identity(row):
        return (row['seed'],row['stream'],row['count'],row['intervention_real_decision_index'])
    indexed = {label:{identity(r):r for r in data['states']} for label,data in loaded.items()}
    if set(indexed['epsilon0']) != set(indexed['epsilon_training']):
        raise ValueError('different state coverage')
    comparisons = []
    for key,a in sorted(indexed['epsilon0'].items()):
        b = indexed['epsilon_training'][key]
        for field in ('probability','online_delta','ema_delta','repetitions'):
            if a[field] != b[field]:
                raise ValueError('changed prediction or sampling coverage')
        comparisons.append(dict(seed=key[0],stream=key[1],count=key[2],prefix=key[3],
            p_continue=a['probability'],ema_prediction=a['ema_delta'],
            epsilon0_net_utility=a['utility'],training_policy_net_utility=b['utility'],
            net_utility_difference=b['utility']-a['utility']))
    # Resample streams, keeping all counts and both policies together.
    groups = sorted({(r['seed'],r['stream']) for r in comparisons})
    indices = [np.array([i for i,r in enumerate(comparisons) if (r['seed'],r['stream'])==g]) for g in groups]
    differences = np.array([r['net_utility_difference'] for r in comparisons])
    rng = np.random.default_rng(20261114)
    boot = [float(differences[np.concatenate([indices[j] for j in rng.integers(0,len(groups),len(groups))])].mean()) for _ in range(4000)]
    reports = {label:utility_report(data['states'],bootstrap=4000) for label,data in loaded.items()}
    result = dict(status='COMPLETE',checkpoint=old['summary']['checkpoint'],
        future_epsilon=epsilon,exact_entry_checks=checked,reports=reports,
        paired_policy_target_difference=dict(mean=float(differences.mean()),
            ci95=np.quantile(boot,[.025,.975]).tolist(),independent_streams=len(groups)),
        sources=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).with_name('summarize_adaptive_planning_evidence.py').read_bytes()).hexdigest(),
        interpretation='Identical fixed entry states, different future execution policies. Epsilon-training is calibration of the training critic; epsilon-zero is evaluation efficacy. '
                       'Targets include current-stage computation cost. Different post-entry visitation/noise prevents interpreting smaller RMSE alone as improved value learning. '
                       'These specified states do not cover all rare positive predictions; no population utility or general success claim.')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (args.output/'state_comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    for label,report in reports.items():
        print(label,json.dumps({k:report[k] for k in ('mean_net_utility','mean_net_utility_ci95',
            'choice_matched_advantage','choice_matched_advantage_ci95','ema_rmse_improvement_over_zero_ci95')}))
    print('policy target difference',result['paired_policy_target_difference'])


if __name__=='__main__':
    main()
