"""Pool pre-update TD errors and reconcile held-out sums across resume.

These errors concern selected-action bootstrap targets. Held-out actor streams
are excluded from VOC Q regression, but still train the shared actor/value
representation. This is neither independent task-return accuracy nor accuracy
of Q(continue)-Q(stop).
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import time


def pooled_errors(rows, *, count_key, prefix):
    """Pool sums, not batch RMSEs; zero-support rows contribute nothing."""
    count, total, absolute, square = 0, 0., 0., 0.
    for row in rows:
        n = float(row[count_key])
        if not math.isfinite(n) or n < 0 or n != int(n):
            raise ValueError('invalid support')
        if n == 0:
            continue
        bias, mae, rmse = (float(row[prefix + suffix]) for suffix in ('bias', 'mae', 'rmse'))
        if not all(math.isfinite(v) for v in (bias, mae, rmse)):
            raise ValueError('nonfinite supported TD metric')
        if mae < 0 or rmse < 0 or abs(bias) > mae + 1e-6 or mae > rmse + 1e-6:
            raise ValueError('inconsistent error moments')
        count += int(n)
        total += n * bias
        absolute += n * mae
        square += n * rmse ** 2
    return dict(count=count, error_sum=total, absolute_error_sum=absolute,
        squared_error_sum=square, bias=total / count if count else None,
        mae=absolute / count if count else None,
        rmse=math.sqrt(square / count) if count else None)


def checkpoint_sums(saved):
    return dict(count=int(saved['voc_holdout_count']),
        error_sum=float(saved['voc_holdout_td_sum']),
        absolute_error_sum=float(saved['voc_holdout_td_abs_sum']),
        squared_error_sum=float(saved['voc_holdout_td_sq_sum']))


def reconcile(parent, final, logged):
    results = {}
    for key in ('count', 'error_sum', 'absolute_error_sum', 'squared_error_sum'):
        expected = final[key] - parent[key]
        observed = logged[key]
        # Logged batch metrics were reduced in FP32; checkpoint sums in FP64.
        scale = max(abs(expected), abs(observed), logged['absolute_error_sum']
                    if key == 'error_sum' else 0.)
        tolerance = 0 if key == 'count' else 2e-6 * scale + 1e-7
        if abs(expected - observed) > tolerance:
            raise ValueError('resumed held-out accounting mismatch: ' + key)
        results[key] = dict(checkpoint_delta=expected, reconstructed=observed,
            absolute_difference=abs(expected - observed), tolerance=tolerance)
    return results


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    campaign = args.campaign.resolve()
    manifest_bytes = (campaign / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    summaries, audits = [], []
    for item in manifest['runs']:
        condition = item['condition']
        if condition not in ('control', 'burn8', 'gru10'):
            continue
        completion_path = campaign / ('completion_' + condition + '_001/audit.json')
        completion_raw = completion_path.read_bytes()
        completion = json.loads(completion_raw)
        if completion['status'] != 'COMPLETED_BUDGET_CHECKPOINT_COUNTERS_AND_WANDB_PASS':
            raise ValueError('training completion has not been audited')
        bindings = [item['parent_checkpoint']['actor'], completion['checkpoint']['actor']]
        saved_sums = []
        for binding in bindings:
            raw = Path(binding['path']).read_bytes()
            if sha(raw) != binding['sha256']:
                raise ValueError('checkpoint changed')
            saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
            if saved['real_step'] != binding['real_step']:
                raise ValueError('checkpoint step mismatch')
            saved_sums.append(checkpoint_sums(saved))
            del saved, raw
        logs_path = Path(item['run_directory']) / 'logs.csv'
        raw = logs_path.read_bytes()
        if sha(raw) != completion['logs_sha256']:
            raise ValueError('completed training log changed')
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
        prior = bindings[0]['real_step']
        for row in rows:
            step = int(float(row['real_step']))
            if step <= prior:
                raise ValueError('nonmonotone resumed steps')
            prior = step
            h = float(row['actor/voc_holdout_count'])
            if h != sum(float(row['actor/voc_holdout_' + side + '_support']) for side in ('continue', 'stop')):
                raise ValueError('held-out action counts disagree')
            if float(row['actor/voc_optional_choice_count']) != h + sum(
                    float(row['actor/voc_' + side + '_support']) for side in ('continue', 'stop')):
                raise ValueError('optional rows do not partition into training/held-out')
            if float(row['actor/voc_optimizer_stepped']) != 1:
                raise ValueError('skipped VOC update')
        if prior != bindings[1]['real_step']:
            raise ValueError('last log step differs from final checkpoint')
        logged = pooled_errors(rows, count_key='actor/voc_holdout_count', prefix='actor/voc_holdout_td_')
        checks = reconcile(*saved_sums, logged)
        for first, last in [(250000, 500000), (500000, 750000), (750000, 850000), (850000, 1000000)]:
            selected = [r for r in rows if first < float(r['real_step']) <= last]
            if not selected:
                raise ValueError('empty fixed window')
            record = dict(condition=condition, first_step=first, last_step=last, updates=len(selected))
            for label, count_key, prefix in [
                ('online_all_optional', 'actor/voc_optional_choice_count', 'actor/voc_td_'),
                ('online_q_holdout', 'actor/voc_holdout_count', 'actor/voc_holdout_td_'),
                ('ema_q_holdout', 'actor/voc_holdout_count', 'actor/voc_gate_holdout_td_'),
            ]:
                pooled = pooled_errors(selected, count_key=count_key, prefix=prefix)
                record.update({label + '_' + k: pooled[k] for k in ('count', 'bias', 'mae', 'rmse')})
            summaries.append(record)
        audits.append(dict(condition=condition, checkpoint_bindings=bindings, checkpoint_reconciliation=checks,
            logs_path=str(logs_path), logs_sha256=sha(raw), completion_audit_path=str(completion_path),
            completion_audit_sha256=sha(completion_raw), history=completion['bc_loss_history']))
    if {r['condition'] for r in audits} != {'control', 'burn8', 'gru10'}:
        raise ValueError('missing condition')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'window_summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    frozen = args.output / 'analysis_sources'; frozen.mkdir()
    shutil.copy2(__file__, frozen / Path(__file__).name)
    result = dict(status='POOLED_TD_METRICS_AND_RESUMED_COUNTERS_PASS', checked_at=time.time(),
        audits=audits, windows=summaries, manifest_sha256=sha(manifest_bytes),
        source_sha256=sha(Path(__file__).read_bytes()),
        limitations=[
            'voc_td_rmse includes all optional rows, including held-out streams; it is not a training-only RMSE.',
            'Held-out streams are withheld from VOC Q regression, not from RL or shared-representation learning.',
            'Targets are bootstrapped and policy-dependent; this does not measure actual paired computation utility.',
            'Selected-action Q error does not determine continue-minus-stop error; their covariance is unavailable.',
            'Visited states and targets differ between conditions; this is not a matched-state accuracy comparison.',
            'All continuation updates reconcile with checkpoints; fixed windows omit tiny final-budget overshoots.',
            'Burn8 historical divergence remains documented even if a later window is stable.',
        ])
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps([r for r in summaries if r['first_step'] == 850000], indent=2))


if __name__ == '__main__':
    main()
