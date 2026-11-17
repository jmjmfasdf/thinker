"""Audit optional-action support and local contrast sensitivity from training logs.

The sensitivity is a reference expectation under the current soft policy with
configured epsilon, NOT an observed gradient, optimizer update, or Q accuracy.
No training settings are changed. Completed update-endpoint windows only.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np
import yaml


FIELDS = (
    'real_step', 'actor/voc_optional_choice_count',
    'actor/voc_continue_support', 'actor/voc_stop_support',
    'actor/voc_holdout_continue_support', 'actor/voc_holdout_stop_support',
    'actor/voc_gate_student_continue_probability',
    'actor/voc_gate_student_saturation_jacobian',
    'actor/voc_gate_low_saturation_count', 'actor/voc_gate_high_saturation_count',
    'actor/voc_gate_teacher_student_kl',
    'actor/voc_gate_projection_batch_start_error_norm',
    'actor/voc_gate_projection_post_error_norm',
    'actor/voc_optimizer_stepped', 'actor/icopro_nll',
    'gradient/voc_gru_candidate_norm', 'gradient/voc_gru_applied_norm',
    'gradient/voc_gru_gradient_scale', 'gradient/actor_gru_gradient_norm',
)


def contrast_moments(mean_p, mean_pq, epsilon):
    """For J_C=1-p, J_S=-p; sampling b=(1-epsilon)*p+epsilon/2.

    For Q_C=V+C+(1-p)D and Q_S=V+C-pD, these are derivatives
    with respect to D while p, C and V are held fixed (detached for GRU).
    E[J^2] depends on epsilon as well as sigmoid's p*(1-p).
    """
    if not (0 <= epsilon <= 1 and 0 <= mean_p <= 1):
        raise ValueError('invalid probability/epsilon')
    # The two means are independently reduced/logged in FP32. At p near 1,
    # roundoff can violate Jensen's inequality by a few float32 ULPs.
    if not (-1e-6 <= mean_pq <= mean_p*(1-mean_p)+1e-6):
        raise ValueError('inconsistent probability moments')
    return dict(
        reference_mean_j=epsilon*(.5-mean_p),
        reference_mean_abs_j=2*(1-epsilon)*mean_pq+epsilon/2,
        reference_mean_j_squared=(1-2*epsilon)*mean_pq+epsilon/2,
    )


def validate_row(row, expected_scale):
    if any(not np.isfinite(row[k]) for k in FIELDS):
        raise ValueError('nonfinite required log metric')
    counts = [row['actor/voc_'+suffix] for suffix in (
        'continue_support', 'stop_support', 'holdout_continue_support',
        'holdout_stop_support')]
    if any(x < 0 or x != round(x) for x in counts):
        raise ValueError('invalid action support count')
    if sum(counts) != row['actor/voc_optional_choice_count']:
        raise ValueError('train/holdout support does not partition optional rows')
    train = counts[0]+counts[1]
    low, high = [row['actor/voc_gate_'+x+'_saturation_count'] for x in ('low','high')]
    if min(low, high) < 0 or low+high > train:
        raise ValueError('gate saturation counts exceed training rows')
    if row['actor/voc_optimizer_stepped'] != 1:
        raise ValueError('skipped update: support cannot be treated as trained')
    candidate = row['gradient/voc_gru_candidate_norm']
    applied = row['gradient/voc_gru_applied_norm']
    if row['gradient/voc_gru_gradient_scale'] != expected_scale or not np.isclose(
            applied, candidate*expected_scale, rtol=1e-6, atol=1e-9):
        raise ValueError('configured and applied GRU multiplier disagree')
    if row['actor/voc_gate_projection_batch_start_error_norm'] > 1e-6 or row[
            'actor/voc_gate_projection_post_error_norm'] > 1e-6:
        raise ValueError('gate projection is not exact')
    contrast_moments(row['actor/voc_gate_student_continue_probability'],
                     row['actor/voc_gate_student_saturation_jacobian'], 0)


def summarize(rows, *, condition, first, last, epsilon):
    if not rows:
        raise ValueError('empty window')
    c = sum(r['actor/voc_continue_support'] for r in rows)
    s = sum(r['actor/voc_stop_support'] for r in rows)
    hc = sum(r['actor/voc_holdout_continue_support'] for r in rows)
    hs = sum(r['actor/voc_holdout_stop_support'] for r in rows)
    n = c+s
    if n == 0:
        raise ValueError('no trained optional choices')
    def average(field):
        return sum(r[field]*(r['actor/voc_continue_support']+r['actor/voc_stop_support'])
                   for r in rows)/n
    p = average('actor/voc_gate_student_continue_probability')
    j = average('actor/voc_gate_student_saturation_jacobian')
    ratios = [r['gradient/voc_gru_applied_norm']/r['gradient/actor_gru_gradient_norm']
              for r in rows if r['gradient/actor_gru_gradient_norm'] > 0]
    result = dict(condition=condition,first_step=first,last_step=last,
        updates=len(rows),train_continue_rows=int(c),train_stop_rows=int(s),
        train_continue_fraction=c/n,holdout_continue_rows=int(hc),holdout_stop_rows=int(hs),
        holdout_continue_fraction=hc/(hc+hs) if hc+hs else None,
        batches_without_continue=sum(r['actor/voc_continue_support']==0 for r in rows),
        batches_without_stop=sum(r['actor/voc_stop_support']==0 for r in rows),
        gate_p_below_point1_fraction=sum(r['actor/voc_gate_low_saturation_count'] for r in rows)/n,
        gate_p_above_point9_fraction=sum(r['actor/voc_gate_high_saturation_count'] for r in rows)/n,
        mean_soft_continue_probability=p,mean_sigmoid_jacobian=j,
        mean_teacher_gate_kl=average('actor/voc_gate_teacher_student_kl'),
        training_epsilon=epsilon,
        reference_expected_continue_fraction=(1-epsilon)*p+epsilon/2,
        voc_to_other_gru_preoptimizer_norm_ratio_median=float(np.median(ratios)) if ratios else None,
        bc_nll_median=float(np.median([r['actor/icopro_nll'] for r in rows])),
        bc_nll_max=max(r['actor/icopro_nll'] for r in rows),
        **contrast_moments(p,j,epsilon))
    return result


def write_csv(path, rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--first-step',type=int,default=250000)
    ap.add_argument('--last-step',type=int,default=850000)
    ap.add_argument('--window',type=int,default=50000)
    args=ap.parse_args()
    if args.window<=0 or args.last_step<=args.first_step or (args.last_step-args.first_step)%args.window:
        raise ValueError('invalid fixed interval')
    campaign=args.campaign.resolve()
    manifest_path=campaign/'manifest.json'
    manifest_raw=manifest_path.read_bytes();manifest=json.loads(manifest_raw)
    repo=Path(__file__).resolve().parents[2]
    code_checks=[]
    for filename,function in [('learn_actor.py','compute_dynamic_voc_loss'),
                               ('actor_net.py','compute_voc_gate_distribution')]:
        relative=Path('thinker/thinker')/filename
        paths=[repo/relative,campaign/'source'/relative]
        nodes=[]
        for path in paths:
            tree=ast.parse(path.read_text())
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
            nodes.append(ast.dump(node,include_attributes=False))
        if nodes[0]!=nodes[1]:
            raise ValueError('tested current helper differs from frozen training helper')
        code_checks.append(dict(function=function,current_path=str(paths[0]),
            frozen_path=str(paths[1]),ast_sha256=hashlib.sha256(nodes[0].encode()).hexdigest(),
            exact_current_frozen_match=True))
    summaries, selected, sources = [], [], []
    for item in manifest['runs']:
        condition=item['condition']
        if condition not in ('control','burn8','gru10'):continue
        config_path=Path(item['config']);config_raw=config_path.read_bytes()
        config=yaml.safe_load(config_raw)
        path=Path(item['run_directory'])/'logs.csv';raw=path.read_bytes()
        complete=[r for r in csv.DictReader(io.StringIO(raw.decode()))
                  if None not in r and None not in r.values()]
        if not complete or float(complete[-1]['real_step'])<args.last_step:
            raise ValueError('fixed observation window not complete')
        prior=item['parent_checkpoint']['actor']['real_step'];rows=[]
        for row in complete:
            step=int(float(row['real_step']))
            if step<=prior:raise ValueError('nonmonotone training steps')
            prior=step
            if step>args.last_step:break
            if step<=args.first_step:continue
            numeric={k:float(row[k]) for k in FIELDS}
            validate_row(numeric,config['voc_gru_gradient_scale'])
            rows.append(numeric)
            selected.append(dict(condition=condition,**numeric))
        for first in range(args.first_step,args.last_step,args.window):
            chunk=[r for r in rows if first<r['real_step']<=first+args.window]
            summaries.append(summarize(chunk,condition=condition,first=first,
                last=first+args.window,epsilon=config['voc_train_epsilon']))
        for first,last in [(args.first_step,args.last_step),(500000,args.last_step)]:
            if args.first_step<=first<last:
                summaries.append(summarize([r for r in rows if first<r['real_step']<=last],
                    condition=condition,first=first,last=last,epsilon=config['voc_train_epsilon']))
        sources.append(dict(condition=condition,path=str(path),read_bytes=len(raw),
            read_sha256=hashlib.sha256(raw).hexdigest(),latest_observed_step=int(float(complete[-1]['real_step'])),
            config_path=str(config_path),config_sha256=hashlib.sha256(config_raw).hexdigest(),
            config=config,parent_actor_step=item['parent_checkpoint']['actor']['real_step']))
    if {r['condition'] for r in summaries}!={'control','burn8','gru10'}:
        raise ValueError('missing comparison condition')
    report=dict(status='TRAINING_SUPPORT_AND_REFERENCE_CONTRAST_MOMENTS_PASS',updated_at=time.time(),
        first_step=args.first_step,last_step=args.last_step,window=args.window,
        sources=sources,manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        tested_current_and_frozen_training_helpers=code_checks,
        probability_moment_tolerance=1e-6,
        maximum_jensen_roundoff_excess=max(0.,max(
            r['actor/voc_gate_student_saturation_jacobian']-
            r['actor/voc_gate_student_continue_probability']*(1-r['actor/voc_gate_student_continue_probability'])
            for r in selected)),
        windows=summaries,production_changes='NONE',
        gradient_contract='Q centering probability is detached. GRU common offset is detached. '
            'J_C=1-p and J_S=-p refer to the D coordinate, not the raw head rows.',
        reference_moments='Under current-policy epsilon sampling: E[J]=epsilon*(.5-E[p]); '
            'E[abs(J)]=2*(1-epsilon)*E[p*(1-p)]+epsilon/2; '
            'E[J^2]=(1-2*epsilon)*E[p*(1-p)]+epsilon/2.',
        limitations=[
            'Counts represent trained optional query rows, not independent observations or paired counterfactuals.',
            'Pooled action support cannot establish coverage of both choices at every state, query count, or game phase.',
            'Mean probability and saturation use Q-training rows; held-out streams are counted separately.',
            'Reference moments assume sampling from the current epsilon-mixed policy. Actual actor data may come from earlier weights; the expected choice rate is not a calibration test.',
            'J is a local prediction derivative. The true gradient also depends on TD residual, Huber clipping, network Jacobian, gradient clipping and Adam moments.',
            'Excluding the common-offset derivative does not guarantee independence from common TD error: E[J] can be nonzero with epsilon mixing. This algebra is not evidence that it causes the observed allocation.',
            'Counts and mean probabilities include different visited scenes across conditions; no state-matched causal inference.',
            'Windows use update endpoints and learner-consumed real transitions, not physical collector actions.',
            'This audit does not demonstrate reward utility or justify further training changes by itself.'])
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/'selected_updates.csv',selected)
    write_csv(args.output/'window_summary.csv',summaries)
    snapshot=args.output/'analysis_sources';snapshot.mkdir()
    shutil.copy2(__file__,snapshot/Path(__file__).name)
    (snapshot/'manifest.json').write_bytes(manifest_raw)
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['selected_updates_sha256']=hashlib.sha256((args.output/'selected_updates.csv').read_bytes()).hexdigest()
    (args.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for row in summaries:
        if row['first_step']==500000 and row['last_step']==args.last_step:
            print(json.dumps(row))


if __name__=='__main__':
    main()
