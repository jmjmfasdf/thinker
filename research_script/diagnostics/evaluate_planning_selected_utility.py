"""Outcome-free selection and paired utility at rare positive-VOC states.

Keep the original eight-stream runtime and homogeneous intervention counts.
Changing batch shapes or intervening on streams at different times can change
floating-point model outputs. A separate natural-policy replay therefore binds
the exact input, recurrent state and tree for every intervention before scoring.
Only the prespecified focal stream enters the primary conditional analysis.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np


def select_states(rows):
    groups, seen = {}, set()
    for row in rows:
        key = (row['seed'], row['stream'], row['decision'])
        if key in seen:
            raise ValueError('duplicate allocation identity')
        seen.add(key)
        p = row['first_optional_probability']
        if p is None or not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('invalid first optional probability')
        groups.setdefault(key[:2], []).append(row)
    selected = []
    for (seed, stream), sample in sorted(groups.items()):
        sample.sort(key=lambda r: r['decision'])
        high = next((r for r in sample if r['first_optional_probability'] > .5), None)
        if high is None:
            raise ValueError(f'no positive-VOC state for {seed}:{stream}')
        low = [r for r in sample if r['decision'] < high['decision']
               and r['episode'] == high['episode'] and r['first_optional_probability'] <= .1]
        if not low:
            raise ValueError(f'no preceding low-VOC state in the same episode for {seed}:{stream}')
        for cohort, row in [('low', low[-1]), ('high', high)]:
            selected.append(dict(seed=seed, stream=stream, prefix=row['decision'],
                episode=row['episode'], episode_step=row['episode_step'], cohort=cohort,
                probability=row['first_optional_probability'], online_delta=row['first_optional_delta']))
    return selected


def state_key(row):
    return (row['seed'], row['stream'], row['prefix'], row['cohort'])


def summarize_pairs(rows, selected, repetitions, bootstrap=4000):
    expected = {state_key(r) for r in selected}
    groups = {}
    for row in rows:
        key = state_key(row)
        if key not in expected:
            raise ValueError('unregistered primary state')
        groups.setdefault(key, []).append(row)
    if set(groups) != expected:
        raise ValueError('missing primary state')
    states = []
    for key, sample in sorted(groups.items()):
        if sorted(r['repetition'] for r in sample) != list(range(repetitions)):
            raise ValueError('missing or duplicated repetition')
        for field in ('entry_hash', 'probability', 'online_delta', 'ema_delta'):
            if len({r[field] for r in sample}) != 1:
                raise ValueError('intervention changed across repetitions')
        states.append(dict(zip(('seed','stream','prefix','cohort'), key),
            probability=sample[0]['probability'], ema_delta=sample[0]['ema_delta'],
            utility=float(np.mean([r['delta_net'] for r in sample])),
            task_utility=float(np.mean([r['delta_task'] for r in sample])),
            added_queries=float(np.mean([r['added_queries'] for r in sample])),
            maximum_pair_tail_bound=max(r['pair_tail_bound'] for r in sample)))
    streams = sorted({(r['seed'], r['stream']) for r in states})
    pairs = []
    for identity in streams:
        sample = {r['cohort']: r for r in states if (r['seed'],r['stream']) == identity}
        if set(sample) != {'low','high'}:
            raise ValueError('each stream must include both cohorts')
        pairs.append([sample['low']['utility'], sample['high']['utility']])
    values = np.asarray(pairs)
    rng = np.random.default_rng(20261114)
    draws = values[rng.integers(0,len(values),(bootstrap,len(values)))].mean(axis=1)
    ci = np.quantile(draws, [.025,.975],axis=0).T
    difference = values[:,1]-values[:,0]
    return dict(status='COMPLETE', selected_states=len(states), paired_repetitions=len(rows),
        independent_streams=len(streams), low_mean_net_utility=float(values[:,0].mean()),
        high_mean_net_utility=float(values[:,1].mean()),
        low_mean_net_utility_ci95=ci[0].tolist(), high_mean_net_utility_ci95=ci[1].tolist(),
        high_minus_low_mean=float(difference.mean()),
        high_minus_low_ci95=np.quantile(draws[:,1]-draws[:,0],[.025,.975]).tolist(),
        maximum_pair_tail_bound=max(r['maximum_pair_tail_bound'] for r in states),
        states=states,
        interpretation='Selected-state conditional efficacy, not population mean or causal effect of predicted VOC. '
                       'High/low are different observations; only STOP/CONTINUE within each state is an intervention. '
                       'Uncertainty resamples entire seed/stream with both cohorts and averaged repetitions. '
                       'Epsilon-zero efficacy is distinct from training-policy critic calibration.')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def natural_entries(evaluation, flags, actor, model, *, seed, prefixes, batch=8):
    """Capture the same full entry representation as the frozen arm evaluator."""
    from gru_reward_readout import core_stream, tree_stream
    from reward_probe import digest
    import torch
    original = evaluation.allocation
    runtime_class = original.__globals__['BatchRuntime']
    references = {}

    class ObservedRuntime(runtime_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.real_index = -1
            self.capture = {}
            self.handles = [self.actor.register_forward_pre_hook(self.before_actor),
                self.actor.tree_rep_encoder.register_forward_hook(
                    lambda m,a,o: self.capture.update(gru=o[0].detach())),
                self.actor.voc_head.register_forward_pre_hook(
                    lambda m,a: self.capture.update(features=a[0].detach())),
                self.actor.register_forward_hook(self.after_actor)]

        def before_actor(self, module, args):
            env, core = args[:2]
            if self.real_index < 0:
                self.real_index = 0
            elif bool(env.real_transition[0].all()):
                self.real_index += 1
            self.is_entry = (self.real_index in prefixes and
                bool(((env.phase[0] == 0) & (env.search_steps[0] == 1)).any()))
            if not self.is_entry:
                return
            if not bool(((env.phase[0] == 0) & (env.search_steps[0] == 1)).all()):
                raise ValueError('first optional entry must be simultaneous')
            self.entry_env = env
            snapshot = self.planner.dynamic_trace_snapshot(include_xs=True, include_hs=True)
            self.identities = []
            for i in range(batch):
                selected_env = {key:getattr(env,key)[:,i:i+1] for key in env._fields
                                if torch.is_tensor(getattr(env,key))}
                self.identities.append(digest(dict(env=selected_env,
                    core=core_stream(core,i,batch), tree=tree_stream(snapshot,i))))

        def after_actor(self, module, args, result):
            if not self.is_entry:
                return
            out = result[0]
            if not bool(out.control_valid[0].all()):
                raise ValueError('reference entry lacks valid control')
            schema = self.actor.tree_rep_meaning
            entries = []
            for i in range(batch):
                token = self.entry_env.tree_reps[0,i].cpu().numpy()
                entries.append(dict(hash=self.identities[i], count=1,
                    depth=int(np.rint(np.expm1(token[schema['rollout_depth']][0]))),
                    features=self.capture['features'][i].cpu().tolist(),
                    gru=self.capture['gru'][0,i].cpu().tolist(),
                    delta=float(out.voc_q[0,i,0]-out.voc_q[0,i,1]),
                    p_continue=float(torch.softmax(out.search_control_logits[0,i],-1)[:2].sum())))
            if self.real_index in references:
                raise ValueError('duplicate natural entry')
            references[self.real_index] = entries

        def close(self):
            for handle in self.handles:
                handle.remove()
            super().close()

    observed = types.FunctionType(original.__code__,
        dict(original.__globals__,BatchRuntime=ObservedRuntime),original.__name__,
        original.__defaults__,original.__closure__)
    observed.__kwdefaults__ = original.__kwdefaults__
    observed(flags,actor,model,seed=seed,batch=batch,decisions=max(prefixes)+1,warmup=0,device='cuda:0')
    if set(references) != set(prefixes):
        raise ValueError('missing natural reference entries')
    return references


def verify_reference(reference, arm):
    if reference != arm['entries']:
        raise ValueError('intervention does not start from the exact natural-policy entry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--shard',type=int)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.plan.resolve().parent
    if args.summarize:
        all_rows = []
        for shard in range(plan['shards']):
            path = root/f'shard_{shard}'
            done = json.loads((path/'completion.json').read_text())
            if done['status'] != 'COMPLETE' or done['plan_sha256'] != hashlib.sha256(args.plan.read_bytes()).hexdigest():
                raise ValueError('incomplete or mismatched shard')
            all_rows.extend(json.loads((path/'primary_pairs.json').read_text()))
        summary = summarize_pairs(all_rows,plan['selected'],plan['repetitions'])
        (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k!='states'}),flush=True)
        return
    if not args.smoke and (args.shard is None or not 0<=args.shard<plan['shards']):
        raise ValueError('choose a valid shard or smoke')
    campaign = Path(plan['campaign'])
    evaluation = load_module(campaign/'evaluation_source_v2/research_script/diagnostics/evaluate_adaptive_planning_pilot.py','selected_bound_evaluation')
    torch = evaluation.torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gpu = torch.cuda.get_device_name(0)
    if '2080 Ti' not in gpu:
        raise ValueError('utility evaluation requires RTX2080Ti')
    flags,actor,model,binding = evaluation.load_pair(
        Path(plan['checkpoint']['actor']['path']),Path(plan['checkpoint']['model']['path']))
    if binding != plan['checkpoint'] or flags.min_search_steps != 1:
        raise ValueError('checkpoint/first optional count mismatch')
    flags = copy.deepcopy(flags)
    flags.voc_train_epsilon,flags.voc_eval_stochastic = 0.,True
    output = root/('smoke' if args.smoke else f'shard_{args.shard}')
    output.mkdir(exist_ok=False)
    if args.smoke:
        reference = natural_entries(evaluation,flags,actor,model,seed=8301,prefixes=[4])[4]
        results = []
        for arm in ('stop','continue','stop'):
            result = evaluation.arm_rollout(flags,actor,model,seed=8301,repetition=0,prefix=4,
                targets=np.ones(8,dtype=int),horizon=32,arm=arm,device='cuda:0')
            verify_reference(reference,result)
            results.append(result)
        if results[0]['returns'] != results[2]['returns']:
            raise ValueError('STOP repeat differs')
        evaluation.atomic_json(output/'smoke.json',dict(status='EXACT_NATURAL_ENTRY_AND_REPEAT_PASS',
            checkpoint=binding,device=gpu,batch=8,search_count=1,horizon=32))
        print('Exact natural entry and repeat smoke PASS',flush=True)
        return
    selected = [r for r in plan['selected'] if r['shard']==args.shard]
    seeds = {r['seed'] for r in selected}
    if len(seeds) != 1:
        raise ValueError('one seed per shard required')
    seed = next(iter(seeds))
    references = natural_entries(evaluation,flags,actor,model,seed=seed,
        prefixes=sorted({r['prefix'] for r in selected}))
    for row in selected:
        entry = references[row['prefix']][row['stream']]
        if entry['p_continue'] != row['probability'] or entry['delta'] != row['online_delta']:
            raise ValueError('reference differs from the prespecified allocation row')
    evaluation.atomic_json(output/'natural_references.json',references)
    print(json.dumps(dict(shard=args.shard,reference_states=len(selected),status='EXACT_SELECTION_PASS')),flush=True)
    primary = []
    ew = actor['voc_ema_gate_head_state_dict']['weight'].double().numpy()
    eb = actor['voc_ema_gate_head_state_dict']['bias'].double().numpy()
    for selection in selected:
        prefix,stream = selection['prefix'],selection['stream']
        for repetition in range(plan['repetitions']):
            arms = {}
            for arm in ('stop','continue'):
                result = evaluation.arm_rollout(flags,actor,model,seed=seed,repetition=repetition,
                    prefix=prefix,targets=np.ones(8,dtype=int),horizon=plan['horizon'],arm=arm,device='cuda:0')
                verify_reference(references[prefix],result)
                arms[arm] = result
            evaluation.verify(arms['stop'],arms['continue'])
            evaluation.atomic_json(output/f'arms_{stream}_{selection["cohort"]}_{repetition}.json',arms)
            entry = arms['stop']['entries'][stream]
            a,b = [arms[arm]['returns'][stream] for arm in ('stop','continue')]
            q = np.asarray(entry['features'])@ew.T+eb
            primary.append(dict(selection,repetition=repetition,entry_hash=entry['hash'],
                ema_delta=float(q[0]-q[1]),delta_net=b['net_truncated']-a['net_truncated'],
                delta_task=b['task_truncated']-a['task_truncated'],
                added_queries=b['stage_computations']-a['stage_computations'],
                pair_tail_bound=sum(arms[arm]['tail_bounds'][stream] for arm in arms)))
            evaluation.atomic_json(output/'primary_pairs.json',primary)
            print(json.dumps(dict(shard=args.shard,selected_pairs=len(primary),stream=stream,
                cohort=selection['cohort'],repetition=repetition)),flush=True)
    evaluation.atomic_json(output/'completion.json',dict(status='COMPLETE',checkpoint=binding,
        selected_states=len(selected),selected_pairs=len(primary),gpu=gpu,
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),score_evaluation='DISABLED'))


if __name__ == '__main__':
    main()
