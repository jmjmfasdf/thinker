"""Audit logged pre-clip gradients; never change a training run.

Norm ratios describe gradient vectors, NOT contributions to Adam updates.
Historical per-gate/per-timestep gradients cannot be recovered from these logs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml


NAMES = {
    'voc_norm': 'gradient/voc_gru_applied_norm',
    'candidate_norm': 'gradient/voc_gru_candidate_norm',
    'actor_norm': 'gradient/actor_gru_gradient_norm',
    'cosine': 'gradient/voc_actor_gru_cosine',
    'update_norm': 'gradient/gru_update_norm',
    'stepped': 'gradient/actor_optimizer_stepped',
    'actor_total_norm_per_tb': 'actor/total_norm',
}
EXTRA = ['actor/learning_rate', 'search/mean_steps',
         'actor/voc_optional_choice_count', 'actor/voc_mandatory_excluded_count',
         'actor/voc_continue_support', 'actor/voc_stop_support',
         'actor/voc_q_loss', 'actor/total_loss', 'actor/icopro_scaled_loss',
         'actor/voc_gate_projection_post_error_norm',
         'actor/voc_gate_projection_batch_start_error_norm']
METRICS = ['voc_norm', 'actor_norm', 'ratio_pct', 'cosine',
           'voc_projection_on_actor_pct', 'merged_direction_angle_deg',
           'clip_coefficient', 'voc_postclip_norm', 'actor_postclip_norm',
           'update_norm', 'search/mean_steps']


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def summarize(part, label, start, end):
    row = dict(interval=label, start=start, end=end, rows=len(part),
               first_logged_step=int(part.real_step.min()),
               last_logged_step=int(part.real_step.max()),
               nonzero_voc_fraction=float((part.voc_norm > 0).mean()),
               optimizer_success_fraction=float(part.stepped.mean()),
               clipping_fraction=float((part.clip_coefficient < 1).mean()),
               negative_cosine_fraction=float(part.loc[part.voc_norm > 0, 'cosine'].lt(0).mean()))
    for field in METRICS:
        for q, tag in [(0.1, 'p10'), (0.5, 'median'), (0.9, 'p90')]:
            row[f'{field}_{tag}'] = float(part[field].quantile(q))
    return row


def checkpoint_audit(run, output):
    """Saved moments reflect all actor losses combined, not VOC in isolation.

    This exact architecture has only two state-dict buffers (norm_low/high).
    Validate the whole optimizer order/shapes before attaching moment names.
    """
    paths = sorted(run.glob('ckp_actor.tar_step_*'), key=lambda p: int(p.name.rsplit('_', 1)[1]))
    if (run/'ckp_actor.tar').exists():
        paths.append(run/'ckp_actor.tar')
    rows, meta, previous, snapshots = [], [], {}, []
    previous_step = None
    for path in paths:
        blob = path.read_bytes()
        ck = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
        state = ck['actor_net_state_dict']
        buffers = [n for n in state if not n.endswith(('weight', 'bias', 'weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'))]
        assert buffers == ['norm_low', 'norm_high'], buffers
        names = [n for n in state if n not in buffers and not n.startswith(('voc_head.', 'voc_gate_head.'))]
        opt = ck['actor_net_optimizer_state_dict']
        assert len(opt['param_groups']) == 1
        ids = opt['param_groups'][0]['params']
        assert len(names) == len(ids)
        moments = dict(zip(names, [opt['state'][i] for i in ids]))
        assert all(state[n].shape == moments[n]['exp_avg'].shape == moments[n]['exp_avg_sq'].shape for n in names)
        step = int(ck['real_step'])
        selected = {n: v for n, v in state.items() if n.startswith('tree_rep_encoder.')}
        snapshots.append(dict(real_step=step, parameters=selected,
                              optimizer_moments={n: moments[n] for n in selected}))
        for name, tensor in selected.items():
            for gate_id, gate in enumerate(['reset', 'update', 'candidate']):
                v = tensor.double().chunk(3, dim=0)[gate_id]
                m = moments[name]['exp_avg'].double().chunk(3, dim=0)[gate_id]
                s = moments[name]['exp_avg_sq'].double().chunk(3, dim=0)[gate_id]
                row = dict(real_step=step, previous_step=previous_step, parameter=name,
                           gate=gate, parameter_count=v.numel(), parameter_norm=float(v.norm()),
                           adam_moment_norm=float(m.norm()), adam_second_moment_norm=float(s.norm()),
                           adam_first_moment_nonzero_fraction=float((m != 0).double().mean()),
                           optimizer_updates=float(moments[name]['step']),
                           all_finite=bool(torch.isfinite(v).all() & torch.isfinite(m).all() & torch.isfinite(s).all()))
                if name in previous:
                    old = previous[name].double().chunk(3, dim=0)[gate_id]
                    row['delta_norm_from_previous_checkpoint'] = float((v-old).norm())
                    row['delta_relative_to_previous_norm'] = float((v-old).norm()/old.norm())
                rows.append(row)
        previous, previous_step = selected, step
        meta.append(dict(path=str(path), sha256=sha(blob), real_step=step,
                         gru_layers=len(selected)//4, gru_hidden_size=next(iter(selected.values())).shape[0]//3,
                         recurrent_flags={k: ck['flags'].get(k) for k in [
                             'real_state_rnn', 'x_rnn', 'h_rnn', 'dynamic_search_state_reset',
                             'voc_gru_connected', 'actor_unroll_len']}))
    pd.DataFrame(rows).to_csv(output/'checkpoint_gru_components.csv', index=False)
    torch.save(snapshots, output/'checkpoint_gru_moments.pt')
    return meta


def plot_trajectory(bins, output, last):
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x = (bins.start + bins.end).to_numpy()/2e6
    def band(ax, field, color, label):
        ax.plot(x, bins[f'{field}_median'], color=color, lw=2, label=label)
        ax.fill_between(x, bins[f'{field}_p10'], bins[f'{field}_p90'], color=color, alpha=.13)
    ax = axes[0, 0]
    band(ax, 'actor_norm', '#375a7f', 'Other actor losses → GRU')
    band(ax, 'voc_norm', '#ce642a', 'VOC loss → GRU')
    ax.set(yscale='log', ylabel='Gradient L2 norm (before clipping)', title='A. Gradient magnitudes')
    ax.legend(fontsize=9, loc='upper left')
    ax = axes[0, 1]
    band(ax, 'ratio_pct', '#773e94', 'VOC / other actor')
    ax.set(yscale='log', ylabel='100 × VOC norm / other actor norm (%)', title='B. Relative VOC gradient')
    ax = axes[1, 0]
    band(ax, 'voc_postclip_norm', '#ce642a', 'VOC component after shared clipping')
    ax.set(yscale='log', ylabel='VOC gradient L2 norm (after clipping)', title='C. Signal passed to Adam')
    ax = axes[1, 1]
    band(ax, 'cosine', '#247b72', 'Gradient cosine')
    ax.axhline(0, color='.4', ls='--', lw=1)
    ax.set(ylabel='Cosine similarity', ylim=(-1, 1), title='D. VOC versus other actor direction')
    for ax in axes.flat:
        ax.axvline(6, color='.5', ls=':', lw=1)
        ax.grid(alpha=.15)
        ax.set(xlim=(0, last/1e6))
    for ax in axes[1]:
        ax.set_xlabel('Training environment steps (millions)')
    fig.suptitle('Enduro sub001 · minimum 1 search · cost 0.0001 · temperature 0.01', fontsize=14)
    fig.text(.5, .925, '100k-step bins: median and 10–90% of learner updates; dotted line = 6M', ha='center', fontsize=10)
    fig.text(.5, .015, 'Gradient ratios are not fractions of Adam parameter updates. Shared clipping preserves the ratio.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .91))
    fig.savefig(output/'gru_gradient_trajectory.png', dpi=180)
    fig.savefig(output/'gru_gradient_trajectory.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = args.run/'logs.csv'
    with source.open('rb') as f:
        limit = source.stat().st_size
        blob = f.read(limit)
    # An actively appended final partial line is excluded explicitly.
    complete_end = blob.rfind(b'\n')+1
    complete = blob[:complete_end]
    wanted = {'real_step', 'step', *NAMES.values(), *EXTRA}
    frame = pd.read_csv(io.BytesIO(complete), comment='#', usecols=lambda c: c in wanted)
    frame = frame.rename(columns={v: k for k, v in NAMES.items()})
    required = ['real_step', 'step', *NAMES]
    assert frame[required].notna().all().all(), 'Incomplete required fields'
    assert np.isfinite(frame[required].to_numpy(dtype=float)).all(), 'Nonfinite required fields'
    assert (frame.real_step.diff().dropna() > 0).all(), 'Nonmonotonic or duplicate real steps'
    assert (frame.actor_norm > 0).all() and (frame.voc_norm >= 0).all()
    assert frame.cosine.abs().le(1.000001).all()
    cfg_blob = (args.run/'config_c.yaml').read_bytes()
    config = yaml.safe_load(cfg_blob)
    transition_tb = int(config['actor_unroll_len'])*int(config['actor_batch_size'])
    assert (frame.step.diff().dropna() == transition_tb).all(), 'Variable unroll sizes need explicit T*B'
    # consume_data_single receives the bootstrap observation as well, and uses
    # that original shape for BOTH clipping and CSV normalization. Its step
    # counter counts only transitions. T*B cancels except in the 1e-6 term.
    tb = (int(config['actor_unroll_len'])+1)*int(config['actor_batch_size'])
    # actor/total_norm is raw global norm divided by this T*B, unlike GRU norms.
    raw_total = frame.actor_total_norm_per_tb*tb
    frame['clip_coefficient'] = np.minimum(1., float(config['actor_grad_norm_clipping'])*tb/(raw_total+1e-6))
    ratio = frame.voc_norm/frame.actor_norm
    frame['ratio_pct'] = ratio*100
    frame['voc_projection_on_actor_pct'] = ratio*frame.cosine*100
    perpendicular = ratio*np.sqrt(np.maximum(0, 1-frame.cosine**2))
    frame['merged_direction_angle_deg'] = np.degrees(np.arctan2(perpendicular, 1+ratio*frame.cosine))
    frame['voc_postclip_norm'] = frame.voc_norm*frame.clip_coefficient
    frame['actor_postclip_norm'] = frame.actor_norm*frame.clip_coefficient
    frame.to_csv(args.output/'gradient_updates.csv', index=False)
    last = int(frame.real_step.max())
    boundaries = sorted(set([0, 250000, *range(1000000, last, 1000000), last]))
    summaries = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        part = frame.loc[(frame.real_step > a) & (frame.real_step <= b)]
        summaries.append(summarize(part, f'{a/1e6:g}–{b/1e6:g}M', a, b))
    pd.DataFrame(summaries).to_csv(args.output/'gradient_intervals.csv', index=False)
    bins = []
    for a in range(0, last, 100000):
        b = min(a+100000, last)
        part = frame.loc[(frame.real_step > a) & (frame.real_step <= b)]
        if len(part):
            bins.append(summarize(part, f'{a}–{b}', a, b))
    bins = pd.DataFrame(bins)
    bins.to_csv(args.output/'gradient_100k_bins.csv', index=False)
    plot_trajectory(bins, args.output, last)
    checkpoints = checkpoint_audit(args.run, args.output)
    qa = dict(rows=len(frame), first_real_step=int(frame.real_step.min()), last_real_step=last,
              zero_voc_steps=frame.loc[frame.voc_norm == 0, 'real_step'].astype(int).tolist(),
              candidate_equals_applied=bool((frame.candidate_norm == frame.voc_norm).all()),
              successful_optimizer_updates=int(frame.stepped.sum()),
              all_required_values_finite=True, no_duplicate_steps=True,
              clipping_fraction=float((frame.clip_coefficient < 1).mean()),
              full_run=summarize(frame, 'all', 0, last),
              after_6m=summarize(frame.loc[frame.real_step >= 6000000], '6M onward', 6000000, last),
              note='Cosine=0 on the initial zero-VOC update is a logging convention, not an observed direction.')
    (args.output/'qa.json').write_text(json.dumps(qa, indent=2, allow_nan=False))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), run=str(args.run),
                    log_path=str(source), log_snapshot_bytes=len(complete), partial_bytes_excluded=len(blob)-len(complete),
                    log_sha256=sha(complete), config_sha256=sha(cfg_blob),
                    script_sha256=sha(Path(__file__).read_bytes()), checkpoints=checkpoints,
                    definitions=dict(ratio='100 * ||VOC GRU gradient|| / ||other actor GRU gradient|| before clipping',
                                     aggregation='Unweighted learner-update quantiles within environment-step intervals',
                                     clip='min(1, clipping*T*B / (actor/total_norm*T*B+1e-6))',
                                     actual_update='Logged ||theta_after_Adam-theta_before_Adam|| for all GRU parameters'),
                    limits=['One Enduro sub001 run, not independent subjects/seeds.',
                            'Per-loss Adam contributions and historical per-gate/time gradients were not logged.',
                            'Checkpoint moment statistics combine all actor and injected VOC gradients.',
                            'Nonzero recurrent weight updates do not prove adequate long-range credit assignment.'])
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output/'config_snapshot.yaml').write_bytes(cfg_blob)
    print(pd.DataFrame(summaries)[['interval', 'rows', 'voc_norm_median', 'actor_norm_median',
          'ratio_pct_median', 'cosine_median', 'clip_coefficient_median', 'voc_postclip_norm_median',
          'update_norm_median']].to_string(index=False))
    print(json.dumps({k: v for k, v in qa.items() if k not in ['full_run', 'after_6m']}, indent=2))


if __name__ == '__main__':
    main()
