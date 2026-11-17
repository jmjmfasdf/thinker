"""Recover registered selected-state utility without changing mandatory draws.

The frozen arm evaluator resampled PROCEED/RESET before its forced entry,
including when STOP was already illegal. Keep the natural joint draw in that
case. Exact natural input/state/tree equality is still required, not relaxed.
All old artifacts remain untouched; reuse is validated against natural entries.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import time

import numpy as np


def force_pending(control, conditional, pending, stop_legal, *, preserve, trace):
    control = control.copy()
    pending = np.asarray(pending, dtype=bool)
    stop_legal = np.asarray(stop_legal, dtype=bool)
    mandatory = pending & ~stop_legal
    changed = np.flatnonzero(mandatory & (control != conditional))
    if len(changed):
        trace.append(dict(streams=changed.tolist(), natural=control[changed].tolist(),
                          legacy=conditional[changed].tolist()))
    forced = pending & stop_legal if preserve else pending
    control[forced] = conditional[forced]
    return control


def patched_arm(evaluation, *, preserve, trace):
    original = evaluation.arm_rollout
    source = inspect.getsource(original)
    before = 'control[pending] = conditional[pending]'
    if source.count(before) != 1:
        raise ValueError('frozen arm source differs from audited implementation')
    after = ('control = _force_pending(control, conditional, pending, '
             'env.legal_control_mask[0, :, util.STOP].cpu().numpy(), '
             'preserve=_preserve, trace=_trace)')
    namespace = dict(original.__globals__, _force_pending=force_pending,
                     _preserve=preserve, _trace=trace)
    exec(compile(source.replace(before, after), '<mandatory-draw-repair>', 'exec'), namespace)
    return namespace[original.__name__]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_cached(arms, reference, evaluation):
    for arm in ('stop', 'continue'):
        if arms[arm]['entries'] != reference:
            raise ValueError('cached arm differs from exact natural reference')
    evaluation.verify(arms['stop'], arms['continue'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=4)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--probe-result', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.plan.resolve().parent
    original = load_module(root/'evaluate.py', 'frozen_selected_utility')
    campaign = Path(plan['campaign'])
    evaluation = load_module(campaign/'evaluation_source_v2/research_script/diagnostics/evaluate_adaptive_planning_pilot.py', 'frozen_selected_evaluation')
    torch = evaluation.torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gpu = torch.cuda.get_device_name(0)
    if '2080 Ti' not in gpu:
        raise ValueError('RTX2080Ti required')
    flags, actor, model, binding = evaluation.load_pair(
        Path(plan['checkpoint']['actor']['path']), Path(plan['checkpoint']['model']['path']))
    if binding != plan['checkpoint'] or flags.min_search_steps != 1:
        raise ValueError('registered checkpoint/minimum changed')
    flags = copy.deepcopy(flags)
    flags.voc_train_epsilon, flags.voc_eval_stochastic = 0., True
    selected = [r for r in plan['selected'] if r['shard'] == args.shard]
    cached = root/f'shard_{args.shard}'
    references = json.loads((cached/'natural_references.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    audit = dict(plan_sha256=sha(args.plan), script_sha256=sha(__file__),
                 checkpoint=binding, shard=args.shard, gpu=gpu, checked_at=time.time(),
                 original_reference_sha256=sha(cached/'natural_references.json'))
    for row in selected:
        ref = references[str(row['prefix'])][row['stream']]
        if ref['p_continue'] != row['probability'] or ref['delta'] != row['online_delta']:
            raise ValueError('reference is inconsistent with registered selection')
    if args.probe:
        # The failed first missing group is fixed from the original partial log.
        first = next(r for r in selected if not (cached/f'arms_{r["stream"]}_{r["cohort"]}_0.json').exists())
        kwargs = dict(seed=first['seed'], repetition=0, prefix=first['prefix'],
                      targets=np.ones(8, dtype=int), horizon=1, arm='stop', device='cuda:0')
        outputs, traces = {}, {}
        for name, preserve in [('legacy', False), ('repaired', True), ('repeat', True)]:
            traces[name] = []
            outputs[name] = patched_arm(evaluation, preserve=preserve, trace=traces[name])(
                flags, actor, model, **kwargs)
            evaluation.atomic_json(args.output/f'{name}.json', outputs[name])
        reference = references[str(first['prefix'])]
        matches = {name: out['entries'] == reference for name, out in outputs.items()}
        audit.update(selection=first, matches=matches, mandatory_draw_changes=traces)
        evaluation.atomic_json(args.output/'diagnostic.json', audit)
        if matches['legacy'] or not matches['repaired'] or not matches['repeat']:
            raise ValueError('hypothesized cause not uniquely reproduced/repaired')
        if outputs['repaired']['returns'] != outputs['repeat']['returns']:
            raise ValueError('repaired STOP is not exactly reproducible')
        audit.update(status='MANDATORY_RNG_CAUSE_AND_EXACT_REPAIR_PASS')
        evaluation.atomic_json(args.output/'completion.json', audit)
        print(json.dumps(audit), flush=True)
        return
    if args.probe_result is None:
        raise ValueError('recovery requires successful causal probe')
    probe = json.loads(args.probe_result.read_text())
    if (probe['status'] != 'MANDATORY_RNG_CAUSE_AND_EXACT_REPAIR_PASS'
            or probe['plan_sha256'] != sha(args.plan) or probe['checkpoint'] != binding
            or probe['script_sha256'] != sha(__file__)):
        raise ValueError('probe is missing or does not bind this repair')
    trace, reused, new, primary = [], [], [], []
    arm_rollout = patched_arm(evaluation, preserve=True, trace=trace)
    ew = actor['voc_ema_gate_head_state_dict']['weight'].double().numpy()
    eb = actor['voc_ema_gate_head_state_dict']['bias'].double().numpy()
    for selection in selected:
        prefix, stream = selection['prefix'], selection['stream']
        ref = references[str(prefix)]
        for repetition in range(plan['repetitions']):
            filename = f'arms_{stream}_{selection["cohort"]}_{repetition}.json'
            saved = cached/filename
            if saved.exists():
                arms = json.loads(saved.read_text())
                validate_cached(arms, ref, evaluation)
                reused.append(dict(path=str(saved), sha256=sha(saved)))
            else:
                arms = {}
                for name in ('stop', 'continue'):
                    arms[name] = arm_rollout(flags, actor, model,
                        seed=selection['seed'], repetition=repetition, prefix=prefix,
                        targets=np.ones(8, dtype=int), horizon=plan['horizon'],
                        arm=name, device='cuda:0')
                    original.verify_reference(ref, arms[name])
                validate_cached(arms, ref, evaluation)
                new.append(filename)
            # Verify repeat STOP for every selected state, including reused arms.
            if repetition == 0:
                repeat = arm_rollout(flags, actor, model,
                    seed=selection['seed'], repetition=0, prefix=prefix,
                    targets=np.ones(8, dtype=int), horizon=plan['horizon'],
                    arm='stop', device='cuda:0')
                original.verify_reference(ref, repeat)
                if arms['stop']['returns'] != repeat['returns']:
                    raise ValueError('repeated STOP outcome differs')
                evaluation.atomic_json(args.output/f'repeat_{stream}_{selection["cohort"]}.json', repeat)
            evaluation.atomic_json(args.output/filename, arms)
            entry = arms['stop']['entries'][stream]
            a, b = [arms[name]['returns'][stream] for name in ('stop', 'continue')]
            q = np.asarray(entry['features']) @ ew.T + eb
            primary.append(dict(selection, repetition=repetition, entry_hash=entry['hash'],
                ema_delta=float(q[0]-q[1]), delta_net=b['net_truncated']-a['net_truncated'],
                delta_task=b['task_truncated']-a['task_truncated'],
                added_queries=b['stage_computations']-a['stage_computations'],
                pair_tail_bound=sum(arms[name]['tail_bounds'][stream] for name in arms)))
            evaluation.atomic_json(args.output/'primary_pairs.json', primary)
            print(json.dumps(dict(shard=args.shard, pairs=len(primary), reused=len(reused))), flush=True)
    old_pairs = json.loads((cached/'primary_pairs.json').read_text())
    if not all(row in primary for row in old_pairs):
        raise ValueError('previously valid pair changed')
    audit.update(status='COMPLETE', selected_states=len(selected), selected_pairs=len(primary),
        reused=reused, newly_computed=new, mandatory_draw_changes=trace,
        original_pair_overlap=len(old_pairs), independent_additional_sample=False,
        probe_sha256=sha(args.probe_result), score_evaluation='DISABLED')
    evaluation.atomic_json(args.output/'completion.json', audit)


if __name__ == '__main__':
    main()
