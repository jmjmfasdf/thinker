"""Continue a frozen utility evaluation by reusing complete paired groups.

The original evaluator, rollout, repeated-STOP check and summary calculations
are unchanged. Only complete, already verified groups from a terminal source
job are reused. New groups run the original rollout including its repeat test.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_group(group, *, seed, count, repetition, prefix, batch, horizon, gamma, think_cost):
    for label, arm in [('stop', 'stop'), ('continue_', 'continue')]:
        saved = group[label]
        for key, expected in dict(seed=seed, repetition=repetition, prefix=prefix, arm=arm).items():
            if saved[key] != expected:
                raise ValueError('cached arm protocol differs: ' + key)
        if saved['targets'] != [count] * batch:
            raise ValueError('cached forced counts differ')
        if saved['real_action_rng'] != 'seed/stream/real_index/repetition; independent of search count':
            raise ValueError('cached real-action randomness differs')
        if any(len(saved[key]) != batch for key in ('entries', 'returns', 'tail_bounds')):
            raise ValueError('cached batch size differs')
        if not 1 <= saved['real_steps'] <= horizon:
            raise ValueError('cached rollout length invalid')
        for entry, result, bound in zip(saved['entries'], saved['returns'], saved['tail_bounds']):
            if entry['count'] != count or not entry['hash']:
                raise ValueError('cached entry identity invalid')
            for key in ('task_truncated', 'current_stage_cost', 'net_truncated', 'cost',
                        'stage_computations', 'all_future_cost_net_truncated', 'gamma'):
                if not math.isfinite(result[key]):
                    raise ValueError('nonfinite cached return')
            if not result['terminal'] and result['real_steps'] != horizon:
                raise ValueError('incomplete nonterminal cached horizon')
            if result['gamma'] != gamma or result['cost'] != think_cost:
                raise ValueError('cached discount/computation cost differs')
            if result['cost'] < 0 or result['stage_computations'] < (0 if arm == 'stop' else 1):
                raise ValueError('cached intervention not executed')
            if arm == 'stop' and result['stage_computations'] != 0:
                raise ValueError('cached STOP contains computation')
            if not math.isclose(result['current_stage_cost'], result['cost'] * result['stage_computations'], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('cached computation cost differs')
            if not math.isclose(result['net_truncated'], result['task_truncated'] - result['current_stage_cost'], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('cached net return differs')
            if not math.isfinite(bound) or bound < 0:
                raise ValueError('invalid cached tail bound')
    if group['stop']['entries'] != group['continue_']['entries']:
        raise ValueError('cached paired entries differ')


def checked_allocation(rows, *, seed, batch, prefix, decisions):
    selected = [r for r in rows if r['seed'] == seed]
    identities = [(r['stream'], r['decision']) for r in selected]
    expected = {(s, t) for s in range(batch) for t in range(prefix, prefix + decisions)}
    if selected and (len(identities) != len(expected) or set(identities) != expected):
        raise ValueError('incomplete or duplicated cached allocation')
    return selected


def make_adapters(plan, groups, allocations, original_arm, original_allocation):
    reused_calls, computed_calls = [], []

    def arm_adapter(flags, actor, model, **kwargs):
        targets = kwargs['targets'].tolist()
        if len(set(targets)) != 1 or targets != [targets[0]] * plan['batch']:
            raise ValueError('runtime target group differs')
        if kwargs['prefix'] != plan['prefix'] or kwargs['horizon'] != plan['horizon']:
            raise ValueError('runtime prefix/horizon differs')
        key = kwargs['seed'], targets[0], kwargs['repetition']
        if key in groups:
            # Source files are written only after the original repeated-STOP
            # test. Do not cache new calls: they must run that test again.
            reused_calls.append(dict(seed=key[0], count=key[1], repetition=key[2], arm=kwargs['arm']))
            return copy.deepcopy(groups[key]['stop' if kwargs['arm'] == 'stop' else 'continue_'])
        computed_calls.append(dict(seed=key[0], count=key[1], repetition=key[2], arm=kwargs['arm']))
        return original_arm(flags, actor, model, **kwargs)

    def allocation_adapter(flags, actor, model, **kwargs):
        if (kwargs['batch'], kwargs['decisions'], kwargs['warmup']) != (plan['batch'], plan['decisions'], plan['prefix']):
            raise ValueError('runtime allocation scope differs')
        cached = checked_allocation(allocations, seed=kwargs['seed'], batch=kwargs['batch'],
                                    prefix=kwargs['warmup'], decisions=kwargs['decisions'])
        return copy.deepcopy(cached) if cached else original_allocation(flags, actor, model, **kwargs)
    return arm_adapter, allocation_adapter, reused_calls, computed_calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    plan_path = args.plan.resolve(); plan = json.loads(plan_path.read_text())
    campaign, original, output = map(Path, (plan['campaign'], plan['source_output'], plan['output']))
    for name, expected in plan['source_files'].items():
        if fingerprint(campaign / name) != expected:
            raise ValueError('frozen evaluation source changed: ' + name)
    registry = Path(plan['registry'])
    if fingerprint(registry) != plan['registry_sha256']:
        raise ValueError('original job registry changed')
    for binding in plan['checkpoint'].values():
        if fingerprint(Path(binding['path'])) != binding['sha256']:
            raise ValueError('checkpoint changed')
    sources, groups = [], {}
    for seed in plan['seeds']:
        for count in plan['counts']:
            for repetition in range(plan['repetitions']):
                path = original / f'arms_{seed}_{count}_{repetition}.json'
                if not path.exists():
                    continue
                raw = path.read_bytes(); group = json.loads(raw)
                validate_group(group, seed=seed, count=count, repetition=repetition,
                    prefix=plan['prefix'], batch=plan['batch'], horizon=plan['horizon'],
                    gamma=plan['gamma'], think_cost=plan['think_cost'])
                groups[seed, count, repetition] = group
                sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
    allocation_path = original / 'allocation.json'
    allocations = json.loads(allocation_path.read_text()) if allocation_path.exists() else []
    for seed in plan['seeds']:
        checked_allocation(allocations, seed=seed, batch=plan['batch'],
                           prefix=plan['prefix'], decisions=plan['decisions'])
    audit = dict(checked_at=time.time(), reused_groups=len(groups), source_group_hashes=sources,
        cached_allocation_decisions=len(allocations), plan_sha256=fingerprint(plan_path),
        independent_sample=False, status='CACHED_GROUP_PROTOCOL_PASS')
    if args.audit_only:
        print(json.dumps(audit, indent=2)); return
    scheduler = subprocess.run(['sacct', '-j', str(plan['source_job']), '--noheader', '--parsable2',
        '--format=JobID,State,ExitCode'], capture_output=True, text=True, check=True)
    records = [line.split('|') for line in scheduler.stdout.splitlines()]
    record = next((r for r in records if r[0] == str(plan['source_job'])), None)
    terminal = {'COMPLETED', 'TIMEOUT', 'FAILED', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE'}
    if not record or record[1].split()[0] not in terminal:
        raise ValueError('source job is not confirmed terminal; never duplicate live evaluation')
    audit['source_scheduler_record'] = record
    if (original / 'summary.json').exists():
        summary = json.loads((original / 'summary.json').read_text())
        expected = len(plan['seeds']) * len(plan['counts']) * plan['repetitions']
        if summary['status'] != 'COMPLETE' or len(groups) != expected or summary['pairs'] != expected * plan['batch']:
            raise ValueError('original completion is inconsistent')
        expected_binding = {role: {k: value[k] for k in ('path', 'sha256', 'real_step')}
                            for role, value in plan['checkpoint'].items()}
        if summary['checkpoint'] != expected_binding or summary['evaluation_train_epsilon'] != 0.:
            raise ValueError('original completed checkpoint/policy differs')
        prior = json.loads((original / 'pairs.json').read_text())
        identities = {(r['seed'], r['count'], r['repetition'], r['stream']) for r in prior}
        expected_ids = {(s, c, r, b) for s in plan['seeds'] for c in plan['counts']
                        for r in range(plan['repetitions']) for b in range(plan['batch'])}
        if len(prior) != len(expected_ids) or identities != expected_ids:
            raise ValueError('original completed paired coverage differs')
        (plan_path.parent / 'completion.json').write_text(json.dumps(dict(audit,
            status='ORIGINAL_COMPLETED_CONTINUATION_NOT_NEEDED', source_summary=str(original/'summary.json')), indent=2) + '\n')
        print('Original evaluation complete; no extra rollout required.', flush=True); return
    if output.exists():
        raise ValueError('continuation output exists; do not overwrite')
    frozen = campaign / 'evaluation_source_v2/research_script/diagnostics/evaluate_adaptive_planning_pilot.py'
    spec = importlib.util.spec_from_file_location('original_utility', frozen)
    evaluation = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluation)
    arm_adapter, allocation_adapter, reused_calls, computed_calls = make_adapters(
        plan, groups, allocations, evaluation.arm_rollout, evaluation.allocation)
    evaluation.arm_rollout = arm_adapter
    evaluation.allocation = allocation_adapter
    sys.argv = [str(frozen), '--actor', plan['checkpoint']['actor']['path'],
        '--model', plan['checkpoint']['model']['path'], '--output', str(output),
        '--prefix', str(plan['prefix']), '--batch', str(plan['batch']),
        '--decisions', str(plan['decisions']), '--horizon', str(plan['horizon']),
        '--repetitions', str(plan['repetitions']), '--counts', *map(str, plan['counts']),
        '--seeds', *map(str, plan['seeds'])]
    (plan_path.parent / 'source_reuse_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    evaluation.main()
    result_pairs = json.loads((output / 'pairs.json').read_text())
    key = lambda r: (r['seed'], r['count'], r['repetition'], r['stream'])
    index = {key(r): r for r in result_pairs}
    expected_count = len(plan['seeds']) * len(plan['counts']) * plan['repetitions'] * plan['batch']
    if len(index) != expected_count or len(result_pairs) != expected_count:
        raise ValueError('completed continuation coverage differs')
    original_pairs = original / 'pairs.json'
    prior = json.loads(original_pairs.read_text()) if original_pairs.exists() else []
    for row in prior:
        if index.get(key(row)) != row:
            raise ValueError('reused paired result changed')
    for source in sources:
        if fingerprint(Path(source['path'])) != source['sha256']:
            raise ValueError('original group changed during continuation')
    completion = dict(audit, status='COMPLETE_REUSED_AND_NEW_GROUPS_PASS',
        pairs=len(result_pairs), exact_original_pair_overlap=len(prior),
        reused_calls=reused_calls, newly_computed_calls=computed_calls,
        output=str(output), independent_sample=False)
    (plan_path.parent / 'completion.json').write_text(json.dumps(completion, indent=2) + '\n')
    (output / 'continuation_audit.json').write_text(json.dumps(completion, indent=2) + '\n')
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == '__main__':
    main()
