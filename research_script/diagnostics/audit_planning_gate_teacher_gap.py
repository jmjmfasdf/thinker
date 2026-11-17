"""Compare online VOC with its EMA-derived gate on recorded optional states.

This is an input-matched head diagnostic, not a rollout under a changed policy.
It cannot establish which estimate is correct or predict new planning lengths.
Completed allocation samples are verified against the registered checkpoints.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from compare_planning_allocation_milestones import load_sample


def sigmoid(value):
    value = np.asarray(value, dtype=float)
    return np.exp(-np.logaddexp(0., -value))


def summarize(rows, q_temperature):
    if not np.isfinite(q_temperature) or q_temperature <= 0:
        raise ValueError('invalid Q temperature')
    keys = [(r['seed'], r['stream'], r['decision']) for r in rows]
    if not rows or len(set(keys)) != len(keys):
        raise ValueError('empty or duplicated decisions')
    p = np.asarray([r['first_optional_probability'] for r in rows], dtype=float)
    delta = np.asarray([r['first_optional_delta'] for r in rows], dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(delta).all():
        raise ValueError('nonfinite probability or online delta')
    if np.any((p < -1e-6) | (p > 1+1e-6)):
        raise ValueError('invalid recorded probability')
    p = np.clip(p, 0, 1)  # Sum of two FP32 control probabilities can exceed 1.
    online_p = sigmoid(delta / q_temperature)
    online_positive, gate_positive = delta > 0, p > .5
    mismatch = online_positive != gate_positive
    confident = (np.abs(delta) > 1e-6) & (np.abs(p-.5) > 1e-5)
    cluster_rows = []
    for seed, stream in sorted({k[:2] for k in keys}):
        use = np.array([k[:2] == (seed, stream) for k in keys])
        cluster_rows.append(dict(seed=seed, stream=stream, decisions=int(use.sum()),
            sign_mismatch_fraction=float(mismatch[use].mean()),
            mean_online_minus_gate_probability=float((online_p-p)[use].mean()),
            mean_absolute_probability_gap=float(np.abs(online_p-p)[use].mean())))
    # Equal stream weights; registered samples have equal decision counts.
    if len({r['decisions'] for r in cluster_rows}) != 1:
        raise ValueError('unequal stream coverage')
    fields = ('sign_mismatch_fraction', 'mean_online_minus_gate_probability',
              'mean_absolute_probability_gap')
    values = np.array([[r[k] for k in fields] for r in cluster_rows])
    rng = np.random.default_rng(20261114)
    draws = values[rng.integers(0, len(values), (4000, len(values)))].mean(axis=1)
    intervals = np.quantile(draws, [.025, .975], axis=0)
    result = dict(decisions=len(rows), independent_streams=len(cluster_rows),
        gate_positive_count=int(gate_positive.sum()),
        online_positive_count=int(online_positive.sum()),
        online_positive_gate_nonpositive_count=int((online_positive & ~gate_positive).sum()),
        gate_positive_online_nonpositive_count=int((gate_positive & ~online_positive).sum()),
        sign_mismatch_count=int(mismatch.sum()),
        robust_sign_mismatch_count=int((mismatch & confident).sum()),
        near_boundary_count=int((~confident).sum()),
        mean_gate_probability=float(p.mean()),
        mean_online_probability_same_features=float(online_p.mean()))
    for j, key in enumerate(fields):
        result[key] = float(values[:, j].mean())
        result[key+'_ci95'] = intervals[:, j].tolist()
    return result, cluster_rows


def check_heads(checkpoint):
    import torch
    flags = checkpoint['flags']
    if not isinstance(flags, dict):
        flags = vars(flags)
    if checkpoint['voc_gate_policy_schema_version'] != 4:
        raise ValueError('this diagnostic is restricted to frozen schema4 pilots')
    if not flags['voc_gate_exact_projection'] or flags['voc_gate_epsilon_greedy_execution']:
        raise ValueError('expected exact soft projection')
    q_temp, policy_temp = flags['voc_gate_q_temperature'], flags['voc_gate_temperature']
    state = checkpoint['actor_net_state_dict']
    ema = checkpoint['voc_ema_gate_head_state_dict']
    for part in ('weight', 'bias'):
        expected = (policy_temp/q_temp)*(ema[part][0:1]-ema[part][1:2])
        if not torch.equal(state['voc_gate_head.'+part], expected):
            raise ValueError('checkpoint gate is not exactly its EMA projection')
    return flags


def check_entries(entries, checkpoint, reference_rows):
    """Independent CPU/float64 reconstruction from recorded actor features."""
    flags = check_heads(checkpoint)
    state, ema = checkpoint['actor_net_state_dict'], checkpoint['voc_ema_gate_head_state_dict']
    x = np.array([r['features'] for r in entries], dtype=float)
    def linear(weight, bias):
        return x @ weight.detach().cpu().double().numpy().T + bias.detach().cpu().double().numpy()
    online_q = linear(state['voc_head.weight'], state['voc_head.bias'])
    ema_q = linear(ema['weight'], ema['bias'])
    gate = linear(state['voc_gate_head.weight'], state['voc_gate_head.bias'])[:, 0]
    online_delta, ema_delta = online_q[:, 0]-online_q[:, 1], ema_q[:, 0]-ema_q[:, 1]
    recorded_p = np.array([r['p_continue'] for r in entries])
    recorded_delta = np.array([r['delta'] for r in entries])
    calculated_p = sigmoid(gate/flags['voc_gate_temperature'])
    errors = dict(online_delta=float(np.max(np.abs(online_delta-recorded_delta))),
        gate_probability=float(np.max(np.abs(calculated_p-recorded_p))),
        ema_probability=float(np.max(np.abs(sigmoid(ema_delta/flags['voc_gate_q_temperature'])-recorded_p))))
    if errors['online_delta'] > 1e-5 or max(errors['gate_probability'], errors['ema_probability']) > 1e-5:
        raise ValueError('recorded features do not reconstruct the VOC heads')
    for entry, row in zip(entries, reference_rows, strict=True):
        if entry['count'] != 1:
            raise ValueError('wrong optional query count')
        if abs(entry['delta']-row['first_optional_delta']) > 1e-6 or abs(
                entry['p_continue']-row['first_optional_probability']) > 1e-6:
            raise ValueError('utility entry is not the allocation first optional state')
    return dict(entries=len(entries), maximum_reconstruction_errors=errors,
        online_delta=online_delta.tolist(), ema_delta=ema_delta.tolist(),
        gate_probability=calculated_p.tolist(),
        online_probability_same_features=sigmoid(online_delta/flags['voc_gate_q_temperature']).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    import torch
    campaign = args.campaign.resolve()
    records, clusters, missing, reconstructions = [], [], [], []
    for milestone, label in ((500000, '500k'), (1000000, '1m')):
        for condition in ('control', 'burn8', 'gru10'):
            root = campaign / f'allocation_{label}_{condition}'
            marker = root/('final_binding_audit.json' if milestone == 1000000 else 'completion.json')
            if not marker.exists():
                if not args.allow_incomplete:
                    raise ValueError(f'incomplete registered sample: {label}/{condition}')
                missing.append(dict(milestone=milestone, condition=condition))
                continue
            verified = load_sample([root], condition, registered=True, final=milestone == 1000000)
            rows = json.loads((root/f'{condition}_allocation.json').read_text())
            checkpoint = torch.load(verified['checkpoint']['actor']['path'], map_location='cpu', weights_only=False)
            flags = check_heads(checkpoint)
            summary, per_stream = summarize(rows, flags['voc_gate_q_temperature'])
            records.append(dict(condition=condition, milestone=milestone,
                actor_step=checkpoint['real_step'], q_temperature=flags['voc_gate_q_temperature'],
                ema_update_rate=checkpoint['voc_gate_target_tau'], **summary,
                verified_sources=verified['sources'], checkpoint=verified['checkpoint']))
            clusters.extend(dict(condition=condition, milestone=milestone, **r) for r in per_stream)
            entry_root = campaign/(f'evaluation_{condition}' if milestone == 1000000
                                   else f'utility_500k_{condition}')
            entry_path = entry_root/'arms_8301_1_0.json'
            if entry_path.exists():
                raw = entry_path.read_bytes()
                arms = json.loads(raw)
                entries = arms['stop']['entries']
                if len(entries) != 8 or entries != arms['continue_']['entries']:
                    raise ValueError('STOP/CONTINUE entries differ')
                prefix = arms['stop']['prefix']
                if not 128 <= prefix < 1664 or arms['stop']['seed'] != 8301 or arms['continue_']['prefix'] != prefix:
                    raise ValueError('unexpected intervention time')
                reference = sorted((r for r in rows if r['seed']==8301 and r['decision']==prefix), key=lambda r:r['stream'])
                reconstructions.append(dict(condition=condition, milestone=milestone,
                    prefix=prefix, path=str(entry_path), sha256=hashlib.sha256(raw).hexdigest(),
                    **check_entries(entries, checkpoint, reference)))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in [('summary.csv', [{k:v for k,v in r.items() if k not in ('verified_sources','checkpoint')} for r in records]),
                       ('stream_summary.csv', clusters)]:
        with (args.output/name).open('w') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    source=args.output/'analysis_sources';source.mkdir()
    for name in (Path(__file__).name, 'compare_planning_allocation_milestones.py'):
        shutil.copy2(Path(__file__).parent/name, source/name)
    audit=dict(status='PARTIAL_CONDITION_COVERAGE' if missing else 'COMPLETE_INPUT_MATCHED_HEAD_DIAGNOSTIC',
        records=records, missing=missing, feature_reconstructions=reconstructions,
        production_changes='NONE', limitations=[
            'Recorded gate uses the EMA head; online probabilities are hypothetical at those same recorded features.',
            'Sign mismatch does not identify which estimate is correct, or prove either predicts useful computation.',
            'No alternate-policy environment rollout was performed; no claim about resulting length or reward.',
            'Only first optional states (after one mandatory query) are compared. Different models visit different scenes.',
            'The EMA updates only head weights, evaluated on current actor features. This is not an EMA of the complete actor.',
            'Intervals resample 16 seed/stream clusters, not independent decisions. One trained seed/participant/game.',
            'Partial utility arm files are used only to reconstruct the heads, never to estimate partial utility outcomes.'],
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=audit['status'], missing=missing,
        samples=[{k:r[k] for k in ('condition','milestone','sign_mismatch_fraction','mean_absolute_probability_gap')} for r in records],
        reconstructed_entries=sum(r['entries'] for r in reconstructions)), indent=2))


if __name__ == '__main__':
    main()
