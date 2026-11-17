"""Bind completed next-action readouts to the original long-horizon utility.

Do not interpret an unchanged first action as absence of later recurrent-state
effects. Complete states/repetitions and exact original entry identities are
required. No partial-sample report is written.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np

from summarize_adaptive_planning_evidence import paired_states


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve()
    sources=[]
    def read(path):
        raw=path.read_bytes()
        sources.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest()))
        return json.loads(raw)
    plan=read(root/'plan.json');saved=read(root/'result/summary.json')
    if saved['status']!='COMPLETE' or saved['plan_sha256']!=sources[0]['sha256']:
        raise ValueError('incomplete or mismatched readout result')
    if saved['checkpoint']!=plan['checkpoint'] or saved['paired_repetitions']!=plan['expected_pairs']:
        raise ValueError('checkpoint/count mismatch')
    reference=Path(plan['reference']);old_summary=read(reference/'summary.json')
    if sources[-1]['sha256']!=plan['reference_summary_sha256']:
        raise ValueError('reference summary changed')
    if old_summary['status']!='COMPLETE' or old_summary['checkpoint']!=plan['checkpoint']:
        raise ValueError('utility reference is not complete/matched')
    code=root/'evaluate.py'
    if hashlib.sha256(code.read_bytes()).hexdigest()!=plan['script_sha256']:
        raise ValueError('frozen readout code changed')
    spec=importlib.util.spec_from_file_location('frozen_readout',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    readouts=read(root/'result/pairs.json');original=read(reference/'pairs.json')
    lookup={(r['seed'],r['stream'],r['repetition']):r for r in readouts}
    expected={(s,i,r) for s in plan['seeds'] for i in range(plan['batch']) for r in range(plan['repetitions'])}
    if len(readouts)!=plan['expected_pairs'] or set(lookup)!=expected:
        raise ValueError('missing or duplicate readout identity')
    recomputed=[];old_arms={}
    for seed in plan['seeds']:
        for rep in range(plan['repetitions']):
            observed=read(root/f'result/arms_{seed}_{rep}.json')
            old_arms[(seed,1,rep)]=read(reference/f'arms_{seed}_1_{rep}.json')
            prior=old_arms[(seed,1,rep)]
            for key in ('stop','continue_'):
                if observed[key]['entries']!=prior[key]['entries'] or observed[key]['real_steps']!=1:
                    raise ValueError('entry/horizon mismatch')
                if observed[key]['prefix']!=plan['prefix'] or prior[key]['prefix']!=plan['prefix']:
                    raise ValueError('prefix mismatch')
            if observed['stop']['entries']!=observed['continue_']['entries']:
                raise ValueError('paired entry mismatch')
            for i in range(plan['batch']):
                row=lookup[(seed,i,rep)]
                a,b=[observed[key]['next_real_actions'][i] for key in ('stop','continue_')]
                if a['uniform']!=b['uniform']:raise ValueError('different real-action random draws')
                for key,r in [('stop',a),('continue_',b)]:
                    if r['stream']!=i or r['real_decision_index']!=plan['prefix']:
                        raise ValueError('wrong real action recorded')
                    expected_action=min(int((r['uniform']>np.cumsum(r['probability'])).sum()),len(r['probability'])-1)
                    if r['action']!=expected_action:raise ValueError('real action inconsistent with probability/draw')
                    if observed[key]['returns'][i]['stage_computations']!=prior[key]['returns'][i]['stage_computations']:
                        raise ValueError('query count differs from original utility intervention')
                metrics=module.policy_metrics(a['probability'],b['probability'])
                if any(row[k]!=v for k,v in metrics.items()):raise ValueError('saved action metrics differ')
                if row['entry_hash']!=observed['stop']['entries'][i]['hash']:
                    raise ValueError('readout row entry hash mismatch')
                changed=a['action']!=b['action']
                if row['sampled_action_changed']!=changed:raise ValueError('saved action change differs')
                added=observed['continue_']['returns'][i]['stage_computations']-observed['stop']['returns'][i]['stage_computations']
                if row['added_queries']!=added:raise ValueError('saved added query count differs')
                recomputed.append(dict(row,**metrics,sampled_action_changed=changed))
    reproduced=module.summarize(recomputed,plan['expected_states'],plan['repetitions'])
    if reproduced['states']!=saved['states'] or reproduced['metrics']!=saved['metrics']:
        raise ValueError('saved state aggregation or uncertainty is not reproducible')
    utility=paired_states(original,old_arms)
    utility_index={(r['seed'],r['stream'],r['intervention_real_decision_index']):r for r in utility}
    action_keys={(r['seed'],r['stream'],r['prefix']) for r in saved['states']}
    if set(utility_index)!=action_keys:raise ValueError('state coverage differs between action and utility')
    states=[]
    for state in saved['states']:
        key=(state['seed'],state['stream'],state['prefix']);u=utility_index[key]
        states.append(dict(state,p_continue=u['probability'],ema_prediction=u['ema_delta'],
            net_utility=u['utility'],task_utility=u['task_utility'],
            any_task_difference_exceeding_tail_bound=u['task_difference_exceeds_tail_bound']>0,
            unchanged_sampled_action_in_both_repetitions=state['sampled_action_changed']==0,
            maximum_next_policy_tv=max(r['total_variation'] for r in recomputed if (r['seed'],r['stream'],r['prefix'])==key)))
    report=dict(status='COMPLETE_EXACT_ACTION_UTILITY_JOIN',checked_at=time.time(),
        checkpoint=plan['checkpoint'],metrics=saved['metrics'],states=states,sources=sources,
        paired_repetitions=len(recomputed),independent_streams=saved['independent_streams'],
        states_with_next_policy_tv_at_most_1e_minus6_in_all_repetitions=sum(r['maximum_next_policy_tv']<=1e-6 for r in states),
        states_with_any_sampled_action_change=sum(r['sampled_action_changed']>0 for r in states),
        states_with_any_argmax_change=sum(r['argmax_changed']>0 for r in states),
        states_with_task_difference_despite_same_first_sampled_action=sum(
            r['any_task_difference_exceeding_tail_bound'] and r['unchanged_sampled_action_in_both_repetitions'] for r in states),
        limitations=[
            'Only16 entry states atprefix896 of one500k Enduro checkpoint; no population claim.',
            'Policy sensitivity is not reward improvement and cannot itself validate VOC.',
            'An unchanged first sampled action does not imply identical subsequent policies: imagination can change carried recurrent state.',
            'No direct attribution to tree GRU versus world-model features; only the full additional-computation intervention is compared.',
            'Argmax and sample changes complement full-distribution TV/JS and must not substitute for them.',
            'All32 paired readouts and both repetitions are retained; no favorable-subset selection.'])
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'state_action_utility.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(states[0]));writer.writeheader();writer.writerows(states)
    snapshot=args.output/'analysis_sources';snapshot.mkdir()
    for path in (Path(__file__),Path(__file__).with_name('summarize_adaptive_planning_evidence.py'),code):
        shutil.copy2(path,snapshot/path.name)
    report['analysis_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'evidence.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('states','sources','checkpoint')},indent=2))


if __name__=='__main__':main()
