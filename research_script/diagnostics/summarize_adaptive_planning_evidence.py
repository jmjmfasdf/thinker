"""Assess fixed-policy allocation separately from paired computation utility.

The allocation log-score measures execution calibration, not useful planning.
The paired utility contrast compares state-dependent continuation with a
constant continuation probability matched separately at each forced count.
This matches the expected number of CONTINUE choices at the intervention,
not total future planning work. Uncertainty resamples complete seed/stream
clusters; repetitions and forced-count states are not independent samples.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def checked_probabilities(values):
    """Tolerate only float32 summation roundoff, preserving correction counts.

    The saved gate probability sums PROCEED and RESET softmax probabilities;
    that float32 sum can exceed one by one ULP. Raw source files stay intact.
    """
    raw = np.asarray(values, dtype=float)
    tolerance = 2*np.finfo(np.float32).eps
    if not np.isfinite(raw).all() or np.any((raw < -tolerance)|(raw > 1+tolerance)):
        raise ValueError('invalid continuation probability beyond float32 roundoff')
    clipped = np.clip(raw, 0., 1.)
    correction = np.abs(clipped-raw)
    return clipped, dict(corrected_values=int(np.count_nonzero(correction)),
                         maximum_correction=float(correction.max(initial=0.)),
                         tolerance=float(tolerance))


def numerically_constant(values):
    values = np.asarray(values, dtype=float)
    return np.ptp(values) <= 1e-12 + 1e-10*np.max(np.abs(values))


def correlation(x, y, rank=False):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or numerically_constant(x) or numerically_constant(y):
        return None
    value = spearmanr(x, y).statistic if rank else np.corrcoef(x, y)[0, 1]
    return float(value) if np.isfinite(value) else None


def initial_gate_null(probability, minimum, cap):
    """Length moments if each decision retained its initial gate probability."""
    p=np.asarray(probability,dtype=float)
    expected=np.full_like(p,float(minimum))
    for extra in range(1,cap-minimum+1):
        expected += p**extra
    return expected,p**(cap-minimum)


def remove_stream_and_decision_effects(values, streams, decisions):
    """Descriptive two-way fixed-effect residuals, including unbalanced panels.

    Removes a separate intercept per stream and arbitrary shared changes at
    each real-decision index. It does not remove stream-specific temporal
    drift or establish a causal effect of observation content.
    """
    residual = np.asarray(values, dtype=float).copy()
    factors = [np.unique(factor, return_inverse=True)[1] for factor in (streams, decisions)]
    for _ in range(1000):
        before = residual.copy()
        for factor in factors:
            mean = np.bincount(factor, weights=residual)/np.bincount(factor)
            residual -= mean[factor]
        if np.max(np.abs(residual-before)) <= 1e-12:
            return residual
    raise ValueError('stream/decision fixed effects did not converge')


def matched_choice_advantage(probability, utility, counts):
    """Covariance within count, equally weighting the observed states.

    E[p(s) (U_continue-U_stop)] - E[p_bar(count) (U_continue-U_stop)].
    No fitting to realized reward or choice of a favorable cutoff is involved.
    """
    probability, utility = np.asarray(probability), np.asarray(utility)
    counts = np.asarray(counts)
    reference = np.zeros_like(probability, dtype=float)
    centered_probability = np.zeros_like(probability, dtype=float)
    centered_utility = np.zeros_like(utility, dtype=float)
    for count in np.unique(counts):
        selected = counts == count
        reference[selected] = probability[selected].mean()
        if not numerically_constant(probability[selected]):
            centered_probability[selected] = probability[selected]-reference[selected]
        if not numerically_constant(utility[selected]):
            centered_utility[selected] = utility[selected]-utility[selected].mean()
    # Center both vectors to avoid attributing subtraction roundoff (~1e-16)
    # to useful allocation when all observed gains equal minus one query cost.
    return float(np.mean(centered_probability*centered_utility)), reference


def paired_return_components(row, arms):
    """Validate saved reward accounting and expose the actual intervention work.

    CONTINUE forces its first query, then follows the checkpoint's gate until
    the real action. It can therefore incur more than one additional query.
    """
    stop = arms['stop']['returns'][row['stream']]
    continued = arms['continue_']['returns'][row['stream']]
    for result in (stop, continued):
        fields = ('task_truncated', 'net_truncated', 'current_stage_cost',
                  'all_future_cost_net_truncated', 'stage_computations', 'cost')
        if not all(np.isfinite(result[key]) for key in fields):
            raise ValueError('nonfinite paired return accounting')
        computations = result['stage_computations']
        if computations < 0 or int(computations) != computations:
            raise ValueError('invalid current-stage computation count')
        if result['cost'] < 0 or not np.isclose(
                result['current_stage_cost'], result['cost']*computations,
                rtol=1e-9, atol=1e-12):
            raise ValueError('current-stage cost differs from query count')
        if not np.isclose(result['net_truncated'],
                          result['task_truncated']-result['current_stage_cost'],
                          rtol=1e-9, atol=1e-12):
            raise ValueError('net return differs from task reward minus stage cost')
    if stop['cost'] != continued['cost']:
        raise ValueError('STOP/CONTINUE computation costs differ')
    if stop['stage_computations'] != 0 or continued['stage_computations'] < 1:
        raise ValueError('STOP/CONTINUE intervention was not executed')
    mapping = dict(delta_task='task_truncated', delta_net='net_truncated',
                   delta_all_future_cost_net='all_future_cost_net_truncated')
    for delta, source in mapping.items():
        if not np.isclose(row[delta], continued[source]-stop[source],
                          rtol=1e-9, atol=1e-12):
            raise ValueError('saved paired contrast disagrees with arm returns: '+delta)
    return dict(
        additional_current_stage_computations=continued['stage_computations'],
        additional_current_stage_cost=continued['current_stage_cost'],
        all_future_cost_net_utility=row['delta_all_future_cost_net'],
        task_difference_exceeds_tail_bound=(
            abs(row['delta_task']) > row['pair_task_tail_bound']+1e-12))


def paired_states(pairs, arms):
    groups = defaultdict(list)
    identities = set()
    for row in pairs:
        identity = (row['seed'], row['stream'], row['count'], row['repetition'])
        if identity in identities:
            raise ValueError('duplicate paired observation')
        identities.add(identity)
        arm = arms[(row['seed'], row['count'], row['repetition'])]
        prefix = arm['stop'].get('prefix')
        if arm['continue_'].get('prefix') != prefix:
            raise ValueError('STOP/CONTINUE intervention prefixes differ')
        entry = arm['stop']['entries'][row['stream']]
        continued = arm['continue_']['entries'][row['stream']]
        if entry != continued or entry['hash'] != row['entry_hash']:
            raise ValueError('STOP/CONTINUE states differ')
        probabilities, roundoff = checked_probabilities([entry['p_continue']])
        probability = float(probabilities[0])
        components = paired_return_components(row, arms[(row['seed'], row['count'], row['repetition'])])
        groups[identity[:3]].append(dict(row, probability=probability,
            intervention_real_decision_index=prefix,
            raw_probability=float(entry['p_continue']),
            probability_roundoff_correction=roundoff['maximum_correction'], **components))
    states = []
    for (seed, stream, count), rows in sorted(groups.items()):
        if len({r['entry_hash'] for r in rows}) != 1:
            raise ValueError('repetitions do not share the same intervention state')
        for key in ('raw_probability', 'online_delta', 'ema_delta', 'intervention_real_decision_index'):
            if any(r[key] != rows[0][key] for r in rows):
                raise ValueError('pre-intervention prediction changed across repetitions')
        states.append(dict(seed=seed, stream=stream, count=count, repetitions=len(rows),
            intervention_real_decision_index=rows[0]['intervention_real_decision_index'],
            probability=rows[0]['probability'], online_delta=rows[0]['online_delta'],
            ema_delta=rows[0]['ema_delta'],
            probability_roundoff_correction=rows[0]['probability_roundoff_correction'],
            utility=float(np.mean([r['delta_net'] for r in rows])),
            task_utility=float(np.mean([r['delta_task'] for r in rows])),
            task_tail_bound=float(np.mean([r['pair_task_tail_bound'] for r in rows])),
            **{key: float(np.mean([r[key] for r in rows])) for key in (
                'additional_current_stage_computations', 'additional_current_stage_cost',
                'all_future_cost_net_utility', 'task_difference_exceeds_tail_bound')}))
    return states


def utility_report(states, bootstrap=2000):
    if not states:
        raise ValueError('empty paired-state sample')
    p = np.array([r['probability'] for r in states], dtype=float)
    y = np.array([r['utility'] for r in states], dtype=float)
    task = np.array([r['task_utility'] for r in states], dtype=float)
    count = np.array([r['count'] for r in states])
    x = np.array([r['ema_delta'] for r in states], dtype=float)
    tail = np.array([r['task_tail_bound'] for r in states], dtype=float)
    computations = np.array([r['additional_current_stage_computations'] for r in states])
    costs = np.array([r['additional_current_stage_cost'] for r in states])
    groups = sorted({(r['seed'], r['stream']) for r in states})
    indices = [np.array([i for i, r in enumerate(states)
                        if (r['seed'], r['stream']) == group]) for group in groups]
    advantage, reference = matched_choice_advantage(p, y, count)
    if not np.allclose(y, task-costs, rtol=1e-9, atol=1e-12):
        raise ValueError('state utility differs from task benefit minus computation cost')
    task_advantage, _ = matched_choice_advantage(p, task, count)
    cost_advantage, _ = matched_choice_advantage(p, -costs, count)
    if not np.isclose(advantage, task_advantage+cost_advantage, rtol=1e-9, atol=1e-12):
        raise ValueError('choice-matched utility components do not sum to net advantage')
    leave_one_stream_out = []
    if len(groups) > 1:
        for omitted in indices:
            keep = np.ones(len(states), dtype=bool)
            keep[omitted] = False
            gain, _ = matched_choice_advantage(p[keep], y[keep], count[keep])
            leave_one_stream_out.append(gain)
    rng = np.random.default_rng(20261114)
    sampled = []
    for _ in range(bootstrap):
        take = np.concatenate([indices[i] for i in rng.integers(0, len(groups), len(groups))])
        gain, _ = matched_choice_advantage(p[take], y[take], count[take])
        sampled.append([gain, np.mean(y[take]), np.mean(x[take]-y[take]),
            np.sqrt(np.mean(y[take]**2))-np.sqrt(np.mean((x[take]-y[take])**2))])
    ci = np.quantile(sampled, [.025, .975], axis=0).T
    residual_x, residual_y = x.copy(), y.copy()
    for k in np.unique(count):
        selected = count == k
        residual_x[selected] -= residual_x[selected].mean()
        residual_y[selected] -= residual_y[selected].mean()
    per_count = []
    for k in np.unique(count):
        take = count == k
        gain, _ = matched_choice_advantage(p[take], y[take], count[take])
        per_count.append(dict(count=int(k), states=int(take.sum()),
            p_continue_mean=float(p[take].mean()), p_continue_sd=float(p[take].std()),
            mean_net_utility=float(y[take].mean()), choice_matched_advantage=gain,
            mean_task_utility=float(task[take].mean()),
            mean_additional_current_stage_computations=float(computations[take].mean()),
            mean_additional_current_stage_cost=float(costs[take].mean()),
            mean_ema_prediction=float(x[take].mean()),
            ema_utility_spearman=correlation(x[take], y[take], rank=True)))
    return dict(states=len(states), independent_streams=len(groups),
        intervention_real_decision_indices=sorted({
            r['intervention_real_decision_index'] for r in states
            if r.get('intervention_real_decision_index') is not None}),
        intervention_time_scope_known=all(r.get('intervention_real_decision_index') is not None for r in states),
        probability_roundoff=dict(
            corrected_states=sum(r.get('probability_roundoff_correction',0)>0 for r in states),
            maximum_correction=max(r.get('probability_roundoff_correction',0) for r in states)),
        paired_repetitions=sum(r['repetitions'] for r in states),
        mean_net_utility=float(y.mean()), mean_task_utility=float(task.mean()),
        mean_additional_current_stage_computations=float(computations.mean()),
        mean_additional_current_stage_cost=float(costs.mean()),
        mean_all_future_cost_net_utility=float(np.mean([r['all_future_cost_net_utility'] for r in states])),
        fraction_paired_task_difference_exceeds_tail_bound=float(np.mean([
            r['task_difference_exceeds_tail_bound'] for r in states])),
        mean_net_utility_ci95=ci[1].tolist(),
        choice_matched_advantage=advantage, choice_matched_advantage_ci95=ci[0].tolist(),
        choice_matched_task_advantage=task_advantage,
        choice_matched_computation_cost_component=cost_advantage,
        choice_matched_advantage_leave_one_stream_out_range=(
            [min(leave_one_stream_out), max(leave_one_stream_out)]
            if leave_one_stream_out else None),
        choice_matched_advantage_tail_bound=float(np.mean(np.abs(p-reference)*tail)),
        ema_utility_spearman=correlation(x, y, rank=True),
        ema_task_utility_spearman=correlation(x, task, rank=True),
        states_with_any_task_difference_exceeding_tail_bound=sum(
            r['task_difference_exceeds_tail_bound'] > 0 for r in states),
        ema_utility_count_centered_pearson=correlation(residual_x, residual_y),
        ema_utility_rmse=float(np.sqrt(np.mean((x-y)**2))),
        mean_ema_prediction=float(x.mean()), ema_prediction_bias=float((x-y).mean()),
        ema_prediction_bias_ci95=ci[2].tolist(),
        zero_predictor_utility_rmse=float(np.sqrt(np.mean(y**2))),
        ema_rmse_improvement_over_zero=float(np.sqrt(np.mean(y**2))
                                            - np.sqrt(np.mean((x-y)**2))),
        ema_rmse_improvement_over_zero_ci95=ci[3].tolist(),
        fraction_utility_positive=float(np.mean(y>0)),
        mean_entry_probability=float(p.mean()), probability_sd=float(p.std()),
        per_count=per_count,
        interpretation='Single intervention on forced-count states. Positive choice-matched advantage means better allocation of the same expected CONTINUE-choice frequency; it does not establish equal total computation or a whole-episode policy improvement.',
        intervention='CONTINUE forces at least one more query, then follows the learned gate. Additional current-stage computation reports the resulting work, rather than assuming exactly one query.',
        truncation_note='Exceeding the task-tail bound establishes a nonzero paired realized reward difference under the bounded-reward assumption, not a precise expected benefit or statistical significance.',
        component_note='Net allocation advantage equals task-return advantage plus the computation-cost component. Net rank correlation can be driven by variable query cost when task returns are unchanged; the leave-one-stream-out range is sensitivity, not a confidence interval.',
        policy_note='Reward definition matches the training target, but evaluation removes training epsilon; prediction bias also reflects this continuation-policy shift.',
        uncertainty='Seed/stream cluster percentile bootstrap; conditional on this one trained checkpoint. Counts/repetitions within a stream are dependent.')


def allocation_report(rows, minimum=1, cap=30):
    if not rows:
        raise ValueError('empty allocation sample')
    identities = [(r['seed'], r['stream'], r['decision']) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate allocation decision')
    length = np.array([r['length'] for r in rows])
    decision = np.array([r['decision'] for r in rows], dtype=int)
    p, roundoff = checked_probabilities([r['first_optional_probability'] for r in rows])
    if np.any((length<minimum)|(length>cap)):
        raise ValueError('invalid allocation probability/length')
    groups = sorted({(r['seed'], r['stream']) for r in rows})
    indices = [np.array([i for i, r in enumerate(rows)
                        if (r['seed'], r['stream']) == group]) for group in groups]
    within_p = p.copy()
    for ix in indices:
        within_p[ix] -= within_p[ix].mean()
    stream_ids = np.empty(len(rows), dtype=int)
    for number, ix in enumerate(indices):
        stream_ids[ix] = number
    adjusted_p = remove_stream_and_decision_effects(p, stream_ids, decision)
    associations = {}
    for name in ('root_value', 'root_margin'):
        x = np.array([r[name] for r in rows], dtype=float)
        within_x = x.copy()
        for ix in indices:
            within_x[ix] -= within_x[ix].mean()
        associations[name] = dict(pearson_with_first_probability=correlation(x, p),
            within_stream_pearson_with_first_probability=correlation(within_x, within_p),
            stream_and_decision_adjusted_pearson_with_first_probability=correlation(
                remove_stream_and_decision_effects(x, stream_ids, decision), adjusted_p))
    expected_length,expected_cap = initial_gate_null(p,minimum,cap)
    continued=length>minimum
    excess=np.array([np.mean((length[ix]==cap)-expected_cap[ix]) for ix in indices])
    rng=np.random.default_rng(20261114)
    boot=excess[rng.integers(0,len(excess),(2000,len(excess)))].mean(axis=1)
    windows = []
    for start in range(int(decision.min()), int(decision.max())+1, 256):
        selected = (decision>=start)&(decision<start+256)
        if selected.any():
            windows.append(dict(
                first_decision=int(decision[selected].min()),
                last_decision=int(decision[selected].max()),
                decisions=int(selected.sum()), mean_length=float(length[selected].mean()),
                min_fraction=float(np.mean(length[selected]==minimum)),
                interior_fraction=float(np.mean((length[selected]>minimum)&(length[selected]<cap))),
                cap_fraction=float(np.mean(length[selected]==cap)),
                first_probability_mean=float(p[selected].mean())))
    # All trials here have the same optional count, eliminating the simple
    # explanation that probability changes only because query count changed.
    return dict(decisions=len(rows), independent_streams=len(groups),
        probability_roundoff=roundoff,
        observed_decision_range=[int(decision.min()),int(decision.max())],
        by_decision_window=windows,
        mean_length=float(length.mean()), min_fraction=float(np.mean(length==minimum)),
        interior_fraction=float(np.mean((length>minimum)&(length<cap))),
        cap_fraction=float(np.mean(length==cap)),
        first_probability_sd=float(p.std()),
        first_probability_within_stream_sd=float(np.sqrt(np.mean(within_p**2))),
        first_probability_after_stream_and_decision_adjustment_sd=float(np.sqrt(np.mean(adjusted_p**2))),
        first_probability_p05_p50_p95=np.quantile(p,[.05,.5,.95]).tolist(),
        first_probability_saturated_fraction=float(np.mean((p<.05)|(p>.95))),
        optional_sign_change_fraction=float(np.mean([r['sign_changes']>0 for r in rows])),
        initial_gate_constant_null=dict(
            expected_mean_length=float(expected_length.mean()),
            expected_cap_fraction=float(expected_cap.mean()),
            observed_cap_excess=float(excess.mean()),
            observed_cap_excess_cluster_ci95=np.quantile(boot,[.025,.975]).tolist(),
            observed_cap_given_first_continue=float(np.mean(length[continued]==cap)) if continued.any() else None,
            expected_cap_given_first_continue=float(expected_cap.mean()/p.mean()) if p.mean()>0 else None,
            observed_length_given_first_continue=float(np.mean(length[continued])) if continued.any() else None,
            expected_length_given_first_continue=float(minimum+(expected_length.mean()-minimum)/p.mean()) if p.mean()>0 else None,
            interpretation='Null holds each decision\'s first optional continuation probability fixed across subsequent queries. Excess length/capping diagnoses changes within planning, not their utility or cause.'),
        associations=associations,
        adjustment_note='Two-way descriptive residualization removes stream means and shared changes at each real-decision index. It does not control individual-stream drift, time since episode reset, or unmeasured scene features; residual associations do not establish useful or causal planning.',
        interpretation='Fixed checkpoint, no training epsilon. Decision indices count real actions since evaluation-stream initialization, not since the most recent episode reset. Results cover only the observed range. Associations are descriptive and serially dependent; they show allocation variation, not computation utility.')


def read_bound(path, sources):
    data = path.read_bytes()
    sources.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(data).hexdigest()))
    return json.loads(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--allocation-root', type=Path, action='append')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = []
    manifest = read_bound(args.campaign/'manifest.json', sources)
    reports = []
    for item in manifest['runs']:
        condition = item['condition']
        if condition == 'smoke':
            continue
        folder = args.campaign/f'evaluation_{condition}'
        report = dict(condition=condition,allocation_status='PENDING',utility_status='PENDING')
        outcome = args.campaign/f'training_outcome_{condition}.json'
        if outcome.exists():
            report['training_outcome'] = read_bound(outcome, sources)
        roots = [root for root in (args.allocation_root or [])
                 if (root/f'{condition}_summary.json').exists()]
        if len(roots)>1:
            raise ValueError('Multiple completed allocations for '+condition)
        root = roots[0] if roots else None
        allocation_path = (root/f'{condition}_allocation.json'
                           if root else folder/'allocation.json')
        completed_path = (root/f'{condition}_summary.json'
                          if root else folder/'summary.json')
        if (not args.allocation_root or root) and allocation_path.exists() and completed_path.exists():
            completed = read_bound(completed_path, sources)
            epsilon = completed.get('evaluation_epsilon', completed.get('evaluation_train_epsilon'))
            if epsilon != 0:
                raise ValueError('allocation must exclude training exploration')
            rows = read_bound(allocation_path, sources)
            report.update(allocation_status=completed['status'],allocation=allocation_report(rows),
                          checkpoint=completed['checkpoint'])
        if (folder/'summary.json').exists():
            summary = read_bound(folder/'summary.json', sources)
            if summary['status'] != 'COMPLETE':
                raise ValueError('summary exists but is not complete')
            pairs = read_bound(folder/'pairs.json', sources)
            arms = {}
            for pair in pairs:
                key = (pair['seed'], pair['count'], pair['repetition'])
                if key not in arms:
                    arms[key] = read_bound(folder/('arms_%s_%s_%s.json'%key), sources)
            report.update(utility_status='COMPLETE',utility=utility_report(paired_states(pairs,arms)),
                          utility_checkpoint=summary['checkpoint'])
        reports.append(report)
        report['same_allocation_and_utility_checkpoint'] = (
            report.get('checkpoint') is not None
            and report.get('checkpoint') == report.get('utility_checkpoint'))
    result = dict(reports=reports,sources=sources,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        complete=all(r['allocation_status']=='COMPLETE' and r['utility_status']=='COMPLETE'
                     and r['same_allocation_and_utility_checkpoint'] for r in reports))
    (args.output/'evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    table = []
    for report in reports:
        a = report.get('allocation', {})
        table.append(dict(condition=report['condition'],
            actor_env_step=report.get('checkpoint', {}).get('actor', {}).get('real_step'),
            allocation_status=report['allocation_status'],
            decisions=a.get('decisions'), independent_streams=a.get('independent_streams'),
            mean_imagination_steps=a.get('mean_length'),
            fraction_1=a.get('min_fraction'), fraction_2_to_29=a.get('interior_fraction'),
            fraction_30=a.get('cap_fraction'),
            first_continue_probability_sd=a.get('first_probability_sd'),
            within_decision_sign_change_fraction=a.get('optional_sign_change_fraction'),
            utility_status=report['utility_status']))
        table[-1]['training_outcome'] = report.get('training_outcome', {}).get('status', 'NOT_INDEPENDENTLY_CLASSIFIED')
    with (args.output/'allocation_table.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for report in reports:
        print(report['condition'], report['allocation_status'], report['utility_status'],
              report.get('utility',{}).get('choice_matched_advantage'))


if __name__ == '__main__':
    main()
