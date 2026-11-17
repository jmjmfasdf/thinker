"""Audit training support and real-transition coverage without changing training.

Optional-query rows are not independent real decisions. Actor unroll length
counts augmented calls, so its directly sampled real-transition span varies.
Bootstrapped targets can represent returns beyond that span.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import yaml

from summarize_adaptive_planning_pilot import loss_health


FIELDS = ('real_step', 'search/real_transition_fraction',
    'actor/voc_optional_choice_count',
    'actor/voc_continue_support','actor/voc_stop_support',
    'actor/voc_holdout_continue_support','actor/voc_holdout_stop_support',
    'actor/voc_continue_td_mae','actor/voc_stop_td_mae',
    'actor/voc_td_rmse','actor/voc_delta_q','actor/voc_gate_delta_q',
    'actor/voc_task_target','actor/voc_think_target','actor/voc_effective_think_cost',
    'actor/voc_q_continue','actor/voc_q_stop',
    'actor/voc_raw_common_abs_mean','actor/voc_raw_contrast_abs_mean',
    'actor/icopro_nll','actor/icopro_loss','actor/icopro_scaled_loss','actor/icopro_count',
    'actor/icopro_numeric_h_encoder_absmax',
    'actor/icopro_numeric_all_phase_logits_absmax',
    'gradient/voc_gru_candidate_norm','gradient/voc_gru_applied_norm',
    'gradient/voc_gru_gradient_scale','gradient/actor_gru_gradient_norm')


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    repo = campaign.parents[3]
    manifest = json.loads((campaign/'manifest.json').read_text())
    selected, summaries, sources, health = [], [], [], {}
    for item in manifest['runs']:
        condition = item['condition']
        if condition not in ('control','burn8','gru10'):
            continue
        config = yaml.safe_load(Path(item['config']).read_text())
        path = repo/'logs/thinker'/item['xpid']/'logs.csv'
        raw = path.read_bytes()
        complete = [r for r in csv.DictReader(io.StringIO(raw.decode()))
                    if None not in r and None not in r.values()]
        prior = item['parent_checkpoint']['actor']['real_step']
        rows = []
        for row in complete:
            step = int(float(row['real_step']))
            if step <= prior:
                raise ValueError('training steps are not strictly increasing')
            if step > 500000:
                break
            values = {key:float(row[key]) for key in FIELDS}
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError('nonfinite training audit field')
            expected = config['actor_batch_size']*config['actor_unroll_len']*values['search/real_transition_fraction']
            discrepancy = abs((step-prior)-expected)
            if discrepancy > .001:
                raise ValueError('real-transition rate disagrees with learner step increments')
            # These two counters already use q_train_valid; holdout support
            # is separately counted and must not be subtracted a second time.
            train_c = values['actor/voc_continue_support']
            train_s = values['actor/voc_stop_support']
            holdout = (values['actor/voc_holdout_continue_support']
                       + values['actor/voc_holdout_stop_support'])
            if min(train_c,train_s,holdout)<0 or train_c+train_s+holdout != values['actor/voc_optional_choice_count']:
                raise ValueError('train/holdout support does not partition optional choices')
            candidate, applied = (values['gradient/voc_gru_candidate_norm'],
                                  values['gradient/voc_gru_applied_norm'])
            scale = values['gradient/voc_gru_gradient_scale']
            if not np.isclose(applied, candidate*scale, rtol=1e-6, atol=1e-9):
                raise ValueError('actual VOC gradient multiplier differs from log')
            if config['icopro_supervised_freq'] != 1 or values['actor/icopro_count'] != config['icopro_batch_size']*config['batch_length']:
                raise ValueError('this audit expects one fixed-target BC batch per update')
            if values['actor/icopro_loss'] <= 0:
                raise ValueError('cannot reconstruct BC scaling from a zero mean loss')
            bc_weight = values['actor/icopro_scaled_loss']/values['actor/icopro_loss']
            if not np.isclose(bc_weight, round(bc_weight), rtol=1e-6, atol=1e-5):
                raise ValueError('BC scaling differs from an integer real-policy row count')
            values.update(condition=condition,previous_real_step=prior,
                actual_step_increment=step-prior,real_step_reconstruction_error=discrepancy,
                real_transitions_per_stream=(step-prior)/config['actor_batch_size'],
                train_continue_count=train_c, train_stop_count=train_s,
                bc_real_policy_weight=round(bc_weight))
            rows.append(values)
            prior=step
        if not rows or max(int(float(r['real_step'])) for r in complete)<500000:
            raise ValueError('the fixed 250k–500k training window is not complete')
        health[condition] = loss_health([r for r in complete if float(r['real_step'])<=500000])
        selected.extend(rows)
        for first in range(250000,500000,50000):
            chunk=[r for r in rows if first<float(r['real_step'])<=first+50000]
            if not chunk:
                raise ValueError('empty training audit window')
            total_c=sum(r['train_continue_count'] for r in chunk)
            total_s=sum(r['train_stop_count'] for r in chunk)
            ratios=[r['gradient/voc_gru_applied_norm']/r['gradient/actor_gru_gradient_norm']
                    for r in chunk if r['gradient/actor_gru_gradient_norm']>0]
            real_span=np.array([r['real_transitions_per_stream'] for r in chunk])
            env_steps=sum(r['actual_step_increment'] for r in chunk)
            bc_targets=sum(r['actor/icopro_count'] for r in chunk)
            bc_weight=sum(r['bc_real_policy_weight'] for r in chunk)
            summaries.append(dict(condition=condition,first_step=first,last_step=first+50000,
                updates=len(chunk),train_continue_rows=total_c,train_stop_rows=total_s,
                actual_env_step_increments=env_steps,
                actor_updates_per_1000_env_steps=1000*len(chunk)/env_steps,
                bc_target_count=bc_targets,bc_targets_per_env_step=bc_targets/env_steps,
                bc_real_policy_weight=bc_weight,bc_weight_per_env_step=bc_weight/env_steps,
                train_continue_fraction=total_c/(total_c+total_s),
                real_transitions_per_stream_p05=float(np.quantile(real_span,.05)),
                real_transitions_per_stream_median=float(np.median(real_span)),
                real_transitions_per_stream_p95=float(np.quantile(real_span,.95)),
                voc_to_other_gru_norm_median=float(np.median(ratios)),
                continue_td_mae_weighted=float(sum(r['actor/voc_continue_td_mae']*r['actor/voc_continue_support'] for r in chunk)/sum(r['actor/voc_continue_support'] for r in chunk)),
                stop_td_mae_weighted=float(sum(r['actor/voc_stop_td_mae']*r['actor/voc_stop_support'] for r in chunk)/sum(r['actor/voc_stop_support'] for r in chunk))))
        sources.append(dict(condition=condition,path=str(path),read_bytes=len(raw),
            read_sha256=hashlib.sha256(raw).hexdigest(),config=config,
            parent_step=item['parent_checkpoint']['actor']['real_step']))
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/'selected_updates.csv',selected)
    write_csv(args.output/'window_summary.csv',summaries)
    report=dict(checked_at=time.time(),status='TRAINING_SUPPORT_AND_STEP_ACCOUNTING_PASS',
        sources=sources,windows=summaries,bc_loss_history=health,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        maximum_step_reconstruction_error=max(r['real_step_reconstruction_error'] for r in selected),
        scope='Complete post-resume updates ending by 500k; window assignment by update endpoint.',
        limitations=[
            'Support counts are optional-query training rows, not independent scene coverage or counterfactual pairs.',
            'Real-transition span is sampled batch coverage; bootstrap can carry reward beyond this span.',
            'TD MAE measures selected individual Q values, not the error of Q_continue minus Q_stop.',
            'VOC/other gradient ratio is a pre-optimizer norm ratio, not an Adam parameter update ratio.',
            'BC loss weight is reconstructed before gradient clipping/Adam. Equal aggregate weight does not imply equal updates, sampled targets, or parameter changes.',
            'These summaries do not establish the cause of allocation collapse.'])
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
