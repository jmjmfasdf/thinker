"""Paired next-action policy sensitivity at registered utility entry states.

Observe the existing paired evaluator without changing its sampling or planner.
Stop after one actual environment action. Reward over this truncated horizon is
not an efficacy metric. State coverage is fixed independently of these results.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import types

import numpy as np


def policy_metrics(stop, cont):
    a,b=np.asarray(stop,dtype=float),np.asarray(cont,dtype=float)
    for p in (a,b):
        if p.ndim!=1 or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or not np.isclose(p.sum(),1.,atol=1e-7,rtol=0):
            raise ValueError('invalid action distribution')
    if a.shape!=b.shape:raise ValueError('action dimension mismatch')
    m=(a+b)/2
    def kl(p):
        positive=p>0
        return np.sum(p[positive]*np.log(p[positive]/m[positive]))
    return dict(total_variation=float(np.abs(a-b).sum()/2),
        jensen_shannon_nat=float((kl(a)+kl(b))/2),
        argmax_changed=bool(a.argmax()!=b.argmax()),
        maximum_probability_change=float(np.abs(a-b).max()))


def observed_arm(evaluation, flags, actor, model, **kwargs):
    import torch
    from reward_probe import keyed_seed
    original=evaluation.arm_rollout
    runtime_class=original.__globals__['BatchRuntime']
    observations=[]
    prefix=kwargs['prefix'];repetition=kwargs['repetition'];seed=kwargs['seed']

    class ObservedRuntime(runtime_class):
        def __init__(self,*args,**kw):
            super().__init__(*args,**kw)
            self.real_index=-1;self.current=None;self.next_actions={}
            self.handles=[self.actor.register_forward_pre_hook(self.before_actor),
                          self.actor.register_forward_hook(self.after_actor)]
            self.saved_step=self.env.step
            self.env.step=self.observed_step

        def before_actor(self,module,args):
            self.current=args[0]
            if self.real_index<0:self.real_index=0
            elif bool(self.current.real_transition[0].all()):self.real_index+=1

        def after_actor(self,module,args,result):
            if self.real_index!=prefix:return
            out=result[0]
            real=self.current.phase[0]==evaluation.util.NEED_REAL_ACTION_PHASE
            probabilities=out.pri_param[-1].reshape(self.batch_size,-1).double().softmax(-1)
            for i in real.nonzero().flatten().tolist():
                if i in self.next_actions:raise ValueError('duplicate real-head readout')
                p=probabilities[i].cpu().numpy()
                policy_metrics(p,p)
                self.next_actions[i]=dict(stream=i,probability=p.tolist(),
                    planning_count=int(self.current.search_steps[0,i]),
                    real_decision_index=self.real_index)

        def observed_step(self,action,model):
            if self.real_index==prefix:
                real=self.current.phase[0]==evaluation.util.NEED_REAL_ACTION_PHASE
                for i in real.nonzero().flatten().tolist():
                    row=self.next_actions[i]
                    actual=int(action[0].reshape(self.batch_size,-1)[i,0])
                    u=keyed_seed(seed+i,prefix,0,evaluation.util.NEED_REAL_ACTION_PHASE,repetition)/2**32
                    expected=min(int((u>np.cumsum(row['probability'])).sum()),len(row['probability'])-1)
                    if actual!=expected:raise ValueError('real action does not match count-independent random draw')
                    row.update(action=actual,uniform=u)
            return self.saved_step(action,model)

        def close(self):
            self.env.step=self.saved_step
            for handle in self.handles:handle.remove()
            observations.extend(self.next_actions[i] for i in sorted(self.next_actions))
            super().close()

    observed=types.FunctionType(original.__code__,
        dict(original.__globals__,BatchRuntime=ObservedRuntime),original.__name__,
        original.__defaults__,original.__closure__)
    observed.__kwdefaults__=original.__kwdefaults__
    result=observed(flags,actor,model,**kwargs)
    if len(observations)!=len(kwargs['targets']) or any('action' not in r for r in observations):
        raise ValueError('missing next actual action')
    result['next_real_actions']=observations
    return result


def summarize(rows, expected_states, repetitions, bootstrap=4000):
    grouped={}
    for row in rows:grouped.setdefault((row['seed'],row['stream'],row['prefix']),[]).append(row)
    if len(grouped)!=expected_states:raise ValueError('incomplete state coverage')
    states=[]
    metrics=('total_variation','jensen_shannon_nat','argmax_changed','sampled_action_changed','added_queries')
    for key,values in sorted(grouped.items()):
        if sorted(r['repetition'] for r in values)!=list(range(repetitions)):
            raise ValueError('missing or duplicate repetition')
        if len({r['entry_hash'] for r in values})!=1:raise ValueError('changed entry across repeats')
        states.append(dict(seed=key[0],stream=key[1],prefix=key[2],
            **{name:float(np.mean([r[name] for r in values])) for name in metrics}))
    groups=sorted({(r['seed'],r['stream']) for r in states})
    matrix=np.array([[np.mean([r[k] for r in states if (r['seed'],r['stream'])==g])
                     for k in metrics] for g in groups])
    rng=np.random.default_rng(20261114)
    samples=matrix[rng.integers(0,len(groups),(bootstrap,len(groups)))].mean(axis=1)
    return dict(status='COMPLETE',states=states,paired_repetitions=len(rows),
        independent_streams=len(groups),
        metrics={k:dict(mean=float(matrix[:,i].mean()),ci95=np.quantile(samples[:,i],[.025,.975]).tolist())
                 for i,k in enumerate(metrics)},
        scope='Paired local action sensitivity, not reward efficacy. No population estimate from a single fixed prefix. '
              'STOP and CONTINUE share entry state and real-action random draws; CONTINUE forces one additional query then follows the policy. '
              'Repetitions average within states; seed/stream clusters are resampled.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();root=args.plan.resolve().parent
    plan=json.loads(args.plan.read_bytes())
    path=Path(plan['campaign'])/'evaluation_source_v2/research_script/diagnostics/evaluate_adaptive_planning_pilot.py'
    spec=importlib.util.spec_from_file_location('action_probe_bound_evaluation',path)
    evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
    torch=evaluation.torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    gpu=torch.cuda.get_device_name(0)
    if '2080 Ti' not in gpu:raise ValueError('RTX2080Ti required')
    flags,actor,model,binding=evaluation.load_pair(Path(plan['checkpoint']['actor']['path']),Path(plan['checkpoint']['model']['path']))
    if binding!=plan['checkpoint']:raise ValueError('checkpoint mismatch')
    flags=copy.deepcopy(flags);flags.voc_train_epsilon=0.;flags.voc_eval_stochastic=True
    output=root/('smoke' if args.smoke else 'result');output.mkdir(exist_ok=False)
    if args.smoke:
        kwargs=dict(seed=8301,repetition=0,prefix=4,targets=np.ones(8,dtype=int),horizon=1,device='cuda:0')
        checked=[]
        for arm in ('stop','continue'):
            plain=evaluation.arm_rollout(flags,actor,model,arm=arm,**kwargs)
            seen=observed_arm(evaluation,flags,actor,model,arm=arm,**kwargs)
            if {k:v for k,v in seen.items() if k not in ('seconds','next_real_actions')} != {k:v for k,v in plain.items() if k!='seconds'}:
                raise ValueError('observer changed original rollout')
            repeated=observed_arm(evaluation,flags,actor,model,arm=arm,**kwargs)
            if seen['next_real_actions']!=repeated['next_real_actions']:raise ValueError('readout repeat mismatch')
            checked.append(seen)
        evaluation.verify(*checked)
        evaluation.atomic_json(output/'smoke.json',dict(status='EXACT_UNOBSERVED_ROLLOUT_AND_REPEAT_PASS',
            checkpoint=binding,device=gpu,all_eight_entries_equal=True,
            count_independent_real_action_rng=True,arms=checked))
        print('Read-only next-action observer smoke PASS',flush=True)
        return
    smoke=json.loads((root/'smoke/smoke.json').read_text())
    if smoke['status']!='EXACT_UNOBSERVED_ROLLOUT_AND_REPEAT_PASS' or smoke['checkpoint']!=binding:
        raise ValueError('mandatory smoke missing/mismatched')
    reference=Path(plan['reference'])
    summary=json.loads((reference/'summary.json').read_text())
    if summary['status']!='COMPLETE' or summary['checkpoint']!=binding:
        raise ValueError('reference utility not complete/mismatched')
    pairs=[]
    for seed in plan['seeds']:
        for rep in range(plan['repetitions']):
            original=reference/f'arms_{seed}_1_{rep}.json'
            payload=json.loads(original.read_text())
            arms={}
            for arm,key in [('stop','stop'),('continue','continue_')]:
                result=observed_arm(evaluation,flags,actor,model,seed=seed,repetition=rep,prefix=plan['prefix'],
                    targets=np.ones(plan['batch'],dtype=int),horizon=1,arm=arm,device='cuda:0')
                if result['entries']!=payload[key]['entries']:raise ValueError('entry differs from original utility evaluation')
                if [r['stage_computations'] for r in result['returns']]!=[r['stage_computations'] for r in payload[key]['returns']]:
                    raise ValueError('current-stage computation changed from original utility evaluation')
                arms[key]=result
            evaluation.verify(arms['stop'],arms['continue_'])
            evaluation.atomic_json(output/f'arms_{seed}_{rep}.json',arms)
            for i,(a,b) in enumerate(zip(arms['stop']['next_real_actions'],arms['continue_']['next_real_actions'])):
                if a['uniform']!=b['uniform']:raise ValueError('paired real-action draws differ')
                entry=arms['stop']['entries'][i]
                added=arms['continue_']['returns'][i]['stage_computations']-arms['stop']['returns'][i]['stage_computations']
                pairs.append(dict(seed=seed,stream=i,prefix=plan['prefix'],repetition=rep,
                    entry_hash=entry['hash'],p_continue=entry['p_continue'],
                    sampled_action_changed=bool(a['action']!=b['action']),added_queries=added,
                    reference_path=str(original),reference_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                    **policy_metrics(a['probability'],b['probability'])))
            evaluation.atomic_json(output/'pairs.json',pairs)
            print(json.dumps(dict(seed=seed,repetition=rep,paired_readouts=len(pairs))),flush=True)
    report=summarize(pairs,plan['expected_states'],plan['repetitions'])
    report.update(checkpoint=binding,device=gpu,plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        score_evaluation='DISABLED',utility_evaluation='NOT_REESTIMATED')
    evaluation.atomic_json(output/'summary.json',report)
    print(json.dumps(report['metrics']),flush=True)


if __name__=='__main__':main()
