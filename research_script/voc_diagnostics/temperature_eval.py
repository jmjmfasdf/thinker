"""Frozen-checkpoint temperature interventions; no training or checkpoint edits.

Change the *loaded gate output*, not the saved temperature flag: exact-projected
gate weights already contain the training temperature. Keep epsilon, conditional
PROCEED/RESET policy, environment seeds and real-action random draws unchanged.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from reward_probe import load_checkpoints, util, keyed_seed, preserve_tokens, digest
from reward_learning_eval import BatchRuntime, categorical
from gru_ablation_eval import prescribed_budget, score_rollout as reference_rollout, score_comparison
from gru_information_probe import atomic_json
from eval_voc10m import score_metrics


def continue_probability(delta, temperature, epsilon):
    if not np.isfinite(temperature) or temperature <= 0 or not 0 <= epsilon <= 1:
        raise ValueError('Invalid temperature or epsilon')
    x = np.asarray(delta, dtype=np.float64) / temperature
    p = np.exp(-np.logaddexp(0., -x))
    return epsilon / 2 + (1 - epsilon) * p


class GateIntervention:
    def __init__(self, actor, ema, base_temperature, temperature):
        if not getattr(actor.flags, 'voc_gate_exact_projection', False):
            raise ValueError('This intervention requires exact EMA projection')
        if float(actor.flags.voc_gate_temperature) != 1.0:
            raise ValueError('Protocol expects unit policy temperature')
        self.base = float(base_temperature)
        self.temperature = float(temperature)
        if self.base <= 0 or self.temperature <= 0:
            raise ValueError('Nonpositive temperature')
        device = actor.voc_gate_head.weight.device
        self.weight = (ema['weight'][0] - ema['weight'][1]).to(device).unsqueeze(0)
        self.bias = (ema['bias'][0] - ema['bias'][1]).to(device).reshape(1)
        self.delta = None
        self.max_error = 0.
        self.handle = actor.voc_gate_head.register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        self.delta = torch.nn.functional.linear(inputs[0], self.weight, self.bias).reshape(-1)
        error = float((output.reshape(-1) * self.base - self.delta).detach().abs().max())
        self.max_error = max(self.max_error, error)
        if not torch.allclose(output.reshape(-1) * self.base, self.delta, atol=2e-5, rtol=2e-5):
            raise AssertionError('Saved gate is not the declared EMA projection')
        # Returning the identical tensor at baseline preserves reference execution.
        return output if self.temperature == self.base else output * (self.base / self.temperature)

    def close(self):
        self.handle.remove()


def distribution_summary(delta, probability):
    q = np.asarray(delta, dtype=float)
    p = np.asarray(probability, dtype=float)
    if not len(q):
        return {'count': 0}
    return dict(count=len(q), delta_mean=float(q.mean()), delta_sd=float(q.std()),
                delta_q05=float(np.quantile(q, .05)), delta_q95=float(np.quantile(q, .95)),
                positive_fraction=float((q > 0).mean()), probability_mean=float(p.mean()),
                probability_q05=float(np.quantile(p, .05)), probability_q95=float(np.quantile(p, .95)),
                probability_positive=float(p[q > 0].mean()) if np.any(q > 0) else None,
                probability_negative=float(p[q < 0].mean()) if np.any(q < 0) else None)


def rollout(flags, actor, model, *, temperature, seed, horizon, batch=8,
            mode='learned', mean=None, sample_path=None, progress=None):
    rt = BatchRuntime(flags, actor, model, seed, batch, 'cuda:0')
    intervention = GateIntervention(rt.actor, rt.ema, flags.voc_gate_q_temperature, temperature)
    before = digest(dict(actor=rt.actor.state_dict(), model=rt.model.state_dict()))
    raw = np.zeros(batch); last = np.zeros(batch); counts = np.zeros(batch, dtype=np.int64)
    current_counts = np.zeros(batch, dtype=np.int64); budgets = []; completed = [[] for _ in range(batch)]
    cap = int(flags.max_search_steps); real_index = 0; started = time.monotonic()
    epsilon = float(flags.voc_train_epsilon)
    root_delta = []; root_p = []; all_delta = []; all_p = []
    maximum_probability_error = 0.
    trace = hashlib.sha256()
    samples = sample_path.open('w') if sample_path else None
    torch.cuda.reset_peak_memory_stats()
    try:
        state, info = rt.env.reset(rt.model)
        env = util.init_env_out(state, info, rt.flags, rt.actor.dim_actions, rt.actor.tuple_action)
        core = rt.actor.initial_state(batch_size=batch, device=rt.device)
        with torch.no_grad():
            for call in range((horizon + 1) * (cap + 4)):
                phase = env.phase[0].cpu().numpy(); k = env.search_steps[0].cpu().numpy()
                out, core = rt.actor(env, core, compute_loss=False, greedy=False)
                logits = out.search_control_logits[-1]
                p = torch.softmax(logits, -1)[:, :2].sum(-1).cpu().numpy()
                delta = intervention.delta.cpu().numpy()
                legal = env.legal_control_mask[0].cpu().numpy().astype(bool)
                valid = (phase == util.SEARCH_PHASE) & legal[:, :2].any(1) & legal[:, util.STOP]
                expected = continue_probability(delta, temperature, epsilon)
                if valid.any():
                    error = float(np.max(np.abs(p[valid] - expected[valid])))
                    maximum_probability_error = max(maximum_probability_error, error)
                    if error > 2e-5:
                        raise AssertionError('Executed soft gate does not match temperature / epsilon')
                    all_delta.extend(delta[valid].tolist()); all_p.extend(p[valid].tolist())
                    root = valid & (k == 0)
                    root_delta.extend(delta[root].tolist()); root_p.extend(p[root].tolist())
                draws = np.array([[keyed_seed(seed+i, real_index, int(k[i]), int(phase[i])+100*j, 0)/2**32
                                   for i in range(batch)] for j in range(3)])
                for i in np.flatnonzero(phase == util.NEED_REAL_ACTION_PHASE):
                    draws[0, i] = keyed_seed(seed+i, real_index, 0, int(phase[i]), 0)/2**32
                primary = categorical(out.pri_param[-1].reshape(batch, -1), draws[0])
                control = categorical(logits, draws[1])
                if mode != 'learned':
                    conditional = categorical(logits[:, :2], draws[2])
                    for i in np.flatnonzero(phase == util.SEARCH_PHASE):
                        u = keyed_seed(seed+i, real_index, 0, 8701, 0)/2**32
                        target = prescribed_budget(mode, mean, cap, real_index, u)
                        control[i] = util.STOP if k[i] >= target else conditional[i]
                if samples and real_index % 64 == 0:
                    online = out.voc_q[0].cpu().numpy()
                    for i in np.flatnonzero(valid):
                        row = dict(real_action=real_index, stream=int(i), count=int(k[i]),
                                   ema_delta=float(delta[i]), online_delta=float(online[i, 0]-online[i, 1]),
                                   p_continue=float(p[i]), selected_control=int(control[i]),
                                   same_state_probabilities={str(t): float(continue_probability(delta[i], t, epsilon))
                                                             for t in (.05, .025, .01)})
                        samples.write(json.dumps(row, allow_nan=False)+'\n')
                action = (torch.as_tensor(primary, device=rt.device).reshape(out.action[0].shape),
                          torch.as_tensor(control, device=rt.device).reshape(out.action[1].shape))
                state, reward, done, truncated, info = rt.env.step(action, rt.model)
                env = preserve_tokens(env, util.create_env_out(action, state, reward, done, truncated, info, rt.flags), info)
                real = info['real_transition'].bool().cpu().numpy()
                if real.any() and not real.all():
                    raise AssertionError('Asynchronous real actions')
                think = info['think_reward'].cpu().numpy().reshape(-1)
                counts += (think == -1); current_counts += (think == -1)
                # Smoke fingerprints include actual accepted controls and observations.
                if horizon <= 128:
                    trace.update(digest(dict(action=action, reward=reward, done=done,
                                             accepted=info['accepted_control'], real=info['real_transition'])).encode())
                    if real.all():
                        trace.update(digest(state['real_states']).encode())
                if real.all():
                    ep = info['episode_return'].cpu().numpy().reshape(-1)
                    finished = info['real_done'].bool().cpu().numpy().reshape(-1)
                    raw += ep-last
                    for i in np.flatnonzero(finished):
                        completed[i].append(float(ep[i]))
                    last = ep.copy(); last[finished] = 0.
                    if mode != 'learned':
                        prescribed = [prescribed_budget(mode, mean, cap, real_index,
                                      keyed_seed(seed+i, real_index, 0, 8701, 0)/2**32) for i in range(batch)]
                        if not np.array_equal(current_counts, prescribed):
                            raise AssertionError('Random budget was not executed')
                    budgets.append(current_counts.copy()); current_counts[:] = 0; real_index += 1
                    if progress and (real_index % 1024 == 0 or real_index == horizon):
                        if samples: samples.flush()
                        progress(dict(completed_actions_per_stream=real_index, target_actions_per_stream=horizon,
                                      raw_reward_per_1000_actions=float(raw.mean()/real_index*1000),
                                      computations_per_action=float(counts.mean()/real_index),
                                      gpu_peak_allocated_gb=float(torch.cuda.max_memory_allocated()/2**30)))
                    if real_index == horizon:
                        break
            else:
                raise RuntimeError('Augmented-call watchdog')
        if digest(dict(actor=rt.actor.state_dict(), model=rt.model.state_dict())) != before:
            raise AssertionError('Evaluation changed checkpoint parameters or buffers')
        hist = np.bincount(np.asarray(budgets).ravel(), minlength=cap+1)
        return dict(mode=mode, temperature=float(temperature), seed=seed, real_actions_per_stream=real_index,
                    score_per_1000_actions=(raw/real_index*1000).tolist(), raw_window=raw.tolist(),
                    computations=counts.tolist(), mean_computations_per_action=float(counts.mean()/real_index),
                    prescribed_mean=mean, budget_histogram=hist.tolist(), completed_episode_returns=completed,
                    seconds=time.monotonic()-started, weights_sha256=before,
                    diagnostics=dict(root=distribution_summary(root_delta, root_p),
                                     all_search=distribution_summary(all_delta, all_p),
                                     max_gate_projection_error=intervention.max_error,
                                     max_probability_error=maximum_probability_error),
                    gpu_peak_allocated_gb=float(torch.cuda.max_memory_allocated()/2**30),
                    smoke_trace_sha256=trace.hexdigest() if horizon <= 128 else None,
                    reward_definition='raw undiscounted game reward; survives episodic-life boundaries; no compute-cost subtraction')
    finally:
        if samples: samples.close()
        intervention.close(); rt.close()


def smoke(flags, actor, model):
    args = dict(seed=58101, horizon=64, batch=8)
    baseline = rollout(flags, actor, model, temperature=.05, **args)
    reference = reference_rollout(flags, actor, model, device='cuda:0', **args)
    keys = ('raw_window', 'computations', 'budget_histogram', 'completed_episode_returns', 'weights_sha256')
    for key in keys:
        if baseline[key] != reference[key]:
            raise AssertionError('Baseline differs from existing evaluation: '+key)
    low = rollout(flags, actor, model, temperature=.01, **args)
    repeat = rollout(flags, actor, model, temperature=.01, **args)
    for key in (*keys, 'smoke_trace_sha256'):
        if low[key] != repeat[key]:
            raise AssertionError('Low-temperature evaluation not reproducible: '+key)
    return dict(status='PASS', baseline_matches_reference=True, low_temperature_repeat_exact=True,
                batch=8, horizon=64, max_gpu_allocated_gb=max(baseline['gpu_peak_allocated_gb'], low['gpu_peak_allocated_gb']),
                baseline_diagnostics=baseline['diagnostics'], low_temperature_diagnostics=low['diagnostics'])


def write_tables(output, results):
    rows = []
    for temperature, record in results.items():
        native = record['native']; random = record['random']
        comparison = score_comparison(native, random)
        metrics = score_metrics(native)
        rows.append(dict(temperature=float(temperature), **metrics,
                         cap_fraction=native['budget_histogram'][-1]/sum(native['budget_histogram']),
                         random_reward_per_1000_actions=float(np.mean(random['score_per_1000_actions'])),
                         random_compute= random['mean_computations_per_action'],
                         gain_vs_random=comparison['reward_difference_per_1000_actions'],
                         gain_vs_random_ci95_low=comparison['ci95'][0], gain_vs_random_ci95_high=comparison['ci95'][1],
                         **{'root_'+k: v for k, v in native['diagnostics']['root'].items()}))
    if not rows: return
    with (output/'temperature_scores.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    baseline = results.get('0.05')
    contrasts = {t: score_comparison(r['native'], baseline['native']) for t, r in results.items()} if baseline else {}
    atomic_json(output/'comparisons.json', dict(temperature_minus_005=contrasts,
                 scope='Same checkpoint and environment seeds; stream-bootstrap intervals, exploratory, one training seed; different temperatures need not use equal compute'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('SLURM GPU required')
    task = json.loads(args.task.read_text()); output = Path(task['output']); output.mkdir(parents=True, exist_ok=True)
    lock = (output/'evaluation.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output/'summary.json').exists():
        raise FileExistsError('Completed output already exists; no unvalidated automatic reuse')
    flags, actor, model, binding = load_checkpoints(argparse.Namespace(bundle=Path(task['bundle']), checkpoint=None, threads=1))
    if float(flags.voc_gate_q_temperature) != .05 or not flags.voc_eval_stochastic:
        raise ValueError('Unexpected baseline execution configuration')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    import wandb
    run = wandb.init(project='thinker', entity=os.getenv('WANDB_USER', 'jmme425'), mode='online',
                     group=args.campaign.name, job_type='temperature-evaluation',
                     name='temperature-'+task['game'], dir=str(output), config=task)
    if run.settings.mode != 'online': raise RuntimeError('Online W&B required')
    started = time.monotonic(); results = {}
    def progress(phase, temperature=None, **extra):
        status = dict(phase=phase, temperature=temperature, wandb_url=run.url,
                      job_id=os.environ['SLURM_JOB_ID'], updated_at=time.time(), **extra)
        atomic_json(output/'status.json', status)
        run.log({'progress/'+k: v for k, v in status.items() if isinstance(v, (int, float))})
    try:
        progress('SMOKE')
        check = smoke(flags, actor, model); atomic_json(output/'smoke.json', check)
        run.summary['smoke_pass'] = True
        for temperature in task['temperatures']:
            tag = str(temperature); folder = output/('tau_'+tag); folder.mkdir(exist_ok=False)
            def execute(phase, **kwargs):
                progress(phase, temperature)
                result = rollout(flags, actor, model, temperature=temperature,
                    progress=lambda m: progress(phase, temperature, **m), **kwargs)
                atomic_json(folder/(phase.lower()+'.json'), result)
                return result
            calibration = execute('CALIBRATION', seed=task['calibration_seed'], horizon=1024, batch=4)
            native = execute('NATIVE', seed=task['evaluation_seed'], horizon=task['horizon'], batch=task['batch'],
                             sample_path=folder/'control_samples.jsonl')
            random = execute('RANDOM', seed=task['evaluation_seed'], horizon=task['horizon'], batch=task['batch'],
                             mode='random', mean=calibration['mean_computations_per_action'])
            results[tag] = dict(native=native, random=random, calibration=calibration)
            write_tables(output, results)
            metrics = score_metrics(native)
            metrics.update(gain_vs_random=score_comparison(native, random)['reward_difference_per_1000_actions'])
            run.log({f'tau_{tag}/'+k: v for k, v in metrics.items() if v is not None})
            run.summary.update({f'tau_{tag}/'+k: v for k, v in metrics.items() if v is not None})
        atomic_json(output/'summary.json', dict(status='COMPLETE', task=task, checkpoint=binding,
            results=results, smoke=check, wandb_url=run.url, seconds=time.monotonic()-started,
            scope='Inference-only temperature changes. Same-state probabilities are mechanical transforms of unchanged VOC. On-policy state distributions differ. No human replay or training.'))
        progress('COMPLETE'); run.summary['phase'] = 'COMPLETE'; run.finish(exit_code=0)
    except BaseException as exc:
        progress('FAILED', error=f'{type(exc).__name__}: {exc}')
        run.summary['phase'] = 'FAILED'; run.finish(exit_code=1); raise


if __name__ == '__main__':
    main()
