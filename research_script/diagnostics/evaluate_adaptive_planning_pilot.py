"""Matched no-exploration budget traces and paired reward-based VOC checks.

No separate game-score evaluation is run. The primary utility target matches
the learner: discounted clipped task rewards minus remaining current-stage
computation cost. All-future-cost utility is saved only as a secondary target.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research_script/voc_diagnostics'))
from reward_learning_eval import BatchRuntime, categorical
from reward_probe import util, keyed_seed, preserve_tokens
from gru_reward_readout import arm_rollout, verify
from thinker.planning_protocol import validate_planning_training_checkpoint


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError('unsupported JSON value: '+type(value).__name__)
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=scalar)+'\n')
    temporary.replace(path)


def load_pair(actor_path, model_path):
    # Read immutable step checkpoints once, and bind the loaded bytes by hash.
    import io
    payloads = {name: path.read_bytes() for name, path in
                (('actor', actor_path), ('model', model_path))}
    loaded = {name: torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
              for name, data in payloads.items()}
    actor, model = loaded['actor'], loaded['model']
    validate_planning_training_checkpoint(actor)
    flags = argparse.Namespace(**actor['flags'])
    util.validate_voc_gate_policy_checkpoint(actor, flags=flags)
    for key in ('name', 'base_seed', 'xpid', 'model_sr_dynamics_norm', 'frame_stack_n',
                'voc_critic_parameterization', 'icopro_burn_in_steps', 'voc_gru_gradient_scale'):
        if actor['flags'].get(key) != model['flags'].get(key):
            raise ValueError('actor/model configuration mismatch: '+key)
    binding = {name: dict(path=str(path), sha256=hashlib.sha256(payloads[name]).hexdigest(),
                         real_step=int(loaded[name]['real_step']))
               for name, path in (('actor', actor_path), ('model', model_path))}
    return flags, actor, model, binding


def allocation(flags, actor, model, *, seed, batch, decisions, warmup, device):
    runtime = BatchRuntime(flags, actor, model, seed, batch, device)
    # BatchRuntime historically enables the training actor switch. Make the
    # evaluation surface explicit: stochastic gate, without training epsilon.
    runtime.actor.train_actor_enabled = False
    pending = [None]*batch
    rows = []
    steps = 0
    lengths = np.zeros(batch, dtype=int)
    schema = util.slice_dynamic_tree_reps(runtime.actor.num_actions)
    try:
        state, info = runtime.env.reset(runtime.model)
        env = util.init_env_out(state, info, runtime.flags,
                                runtime.actor.dim_actions, runtime.actor.tuple_action)
        core = runtime.actor.initial_state(batch_size=batch, device=runtime.device)
        with torch.no_grad():
            for _ in range((warmup+decisions+1)*(flags.max_search_steps+4)):
                phase = env.phase[0].cpu().numpy()
                counts = env.search_steps[0].cpu().numpy().astype(int)
                out, core = runtime.actor(env, core, compute_loss=False, greedy=False)
                probability = out.search_control_logits[-1].softmax(-1)[:, :2].sum(-1).cpu().numpy()
                q = out.voc_q[-1].cpu().numpy()
                tree = env.tree_reps[0].cpu().numpy()
                legal = env.legal_control_mask[0].cpu().numpy().astype(bool)
                for i in np.flatnonzero((phase == util.SEARCH_PHASE) & (counts == 0)):
                    prior = np.sort(tree[i, schema['root_policy']])
                    pending[i] = dict(seed=seed, stream=i, decision=steps,
                        root_margin=float(prior[-1]-prior[-2]),
                        root_value=float(tree[i, schema['root_v']][0]),
                        first_optional_probability=None, first_optional_delta=None,
                        last_sign=0, sign_changes=0, optional_choices=0)
                for i in np.flatnonzero((phase == util.SEARCH_PHASE) & legal[:, util.STOP] & legal[:, :2].any(-1)):
                    row = pending[i]
                    if row is None:
                        raise AssertionError('missing beginning of planning stage')
                    delta = float(q[i, 0]-q[i, 1])
                    if row['first_optional_probability'] is None:
                        row.update(first_optional_probability=float(probability[i]), first_optional_delta=delta)
                    sign = int(np.sign(probability[i]-.5))
                    row['sign_changes'] += int(row['last_sign'] != 0 and sign != 0 and row['last_sign'] != sign)
                    row['last_sign'] = sign
                    row['optional_choices'] += 1
                draws = np.array([[keyed_seed(seed+i, steps,
                    0 if phase[i] == util.NEED_REAL_ACTION_PHASE and j == 0 else int(counts[i]),
                    int(phase[i])+100*j)/2**32 for i in range(batch)] for j in range(2)])
                primary = categorical(out.pri_param[-1].reshape(batch, -1), draws[0])
                control = categorical(out.search_control_logits[-1], draws[1])
                action = (torch.as_tensor(primary, device=runtime.device).reshape(out.action[0].shape),
                          torch.as_tensor(control, device=runtime.device).reshape(out.action[1].shape))
                state, reward, done, truncated, info = runtime.env.step(action, runtime.model)
                env = preserve_tokens(env, util.create_env_out(action, state, reward, done, truncated,
                    info, runtime.flags), info)
                lengths += (info['think_reward'].cpu().numpy().reshape(-1) == -1)
                real = info['real_transition'].bool().cpu().numpy()
                if real.any() and not real.all():
                    raise AssertionError('asynchronous real barrier')
                if real.all():
                    for i, row in enumerate(pending):
                        if row is None:
                            raise AssertionError('missing planning row')
                        if steps >= warmup:
                            row['length'] = int(lengths[i])
                            rows.append(row)
                    lengths[:] = 0
                    pending = [None]*batch
                    steps += 1
                    if steps >= warmup+decisions:
                        break
            else:
                raise RuntimeError('allocation watchdog exceeded')
    finally:
        runtime.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--actor', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--decisions', type=int, default=512)
    ap.add_argument('--prefix', type=int, default=128)
    ap.add_argument('--seeds', type=int, nargs='+', default=[8301, 8401])
    ap.add_argument('--counts', type=int, nargs='+', default=[1, 3, 6, 12])
    ap.add_argument('--repetitions', type=int, default=2)
    ap.add_argument('--horizon', type=int, default=1536)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gpu = torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "CPU"
    print(json.dumps(dict(device=gpu)), flush=True)
    if args.device.startswith("cuda") and "2080 Ti" not in gpu:
        raise RuntimeError("Utility evaluation is restricted to RTX 2080 Ti")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    flags, actor, model, binding = load_pair(args.actor, args.model)
    if min(args.counts) < flags.min_search_steps or max(args.counts) >= flags.max_search_steps:
        raise ValueError('paired entries must permit both STOP and CONTINUE')
    evaluation_flags = copy.deepcopy(flags)
    evaluation_flags.voc_train_epsilon = 0.0
    evaluation_flags.voc_eval_stochastic = True
    budget_rows, pairs = [], []
    for seed in args.seeds:
        budget_rows.extend(allocation(evaluation_flags, actor, model, seed=seed,
            batch=args.batch, decisions=args.decisions, warmup=args.prefix, device=args.device))
        atomic_json(args.output/'allocation.json', budget_rows)
        for count in args.counts:
            for repetition in range(args.repetitions):
                kwargs = dict(seed=seed, repetition=repetition, prefix=args.prefix,
                    targets=np.full(args.batch, count), horizon=args.horizon, device=args.device)
                stop = arm_rollout(evaluation_flags, actor, model, arm='stop', **kwargs)
                cont = arm_rollout(evaluation_flags, actor, model, arm='continue', **kwargs)
                verify(stop, cont)
                if repetition == 0:
                    repeated = arm_rollout(evaluation_flags, actor, model, arm='stop', **kwargs)
                    verify(stop, repeated)
                    if stop['returns'] != repeated['returns']:
                        raise AssertionError('STOP repeat is not reproducible')
                atomic_json(args.output/f'arms_{seed}_{count}_{repetition}.json', dict(stop=stop, continue_=cont))
                ew = actor['voc_ema_gate_head_state_dict']['weight'].double().numpy()
                eb = actor['voc_ema_gate_head_state_dict']['bias'].double().numpy()
                for i, (a, b) in enumerate(zip(stop['returns'], cont['returns'])):
                    entry = stop['entries'][i]
                    q = np.asarray(entry['features'])@ew.T + eb
                    pairs.append(dict(seed=seed, stream=i, count=count, repetition=repetition,
                        entry_hash=entry['hash'], online_delta=entry['delta'], ema_delta=float(q[0]-q[1]),
                        delta_net=b['net_truncated']-a['net_truncated'],
                        delta_task=b['task_truncated']-a['task_truncated'],
                        delta_all_future_cost_net=b['all_future_cost_net_truncated']-a['all_future_cost_net_truncated'],
                        pair_task_tail_bound=stop['tail_bounds'][i]+cont['tail_bounds'][i]))
                atomic_json(args.output/'pairs.json', pairs)
                print(json.dumps(dict(seed=seed, count=count, repetition=repetition, pairs=len(pairs))), flush=True)
    lengths = np.array([r['length'] for r in budget_rows])
    # Average repetitions within a matched state before correlating the VOC.
    states = {}
    for row in pairs:
        states.setdefault((row['seed'], row['stream'], row['count']), []).append(row)
    x = np.array([rr[0]['ema_delta'] for rr in states.values()])
    y = np.array([np.mean([r['delta_net'] for r in rr]) for rr in states.values()])
    corr = float(spearmanr(x, y).statistic) if np.std(x)>0 and np.std(y)>0 else None
    if corr is not None and not np.isfinite(corr):
        corr = None
    atomic_json(args.output/'summary.json', dict(status='COMPLETE', checkpoint=binding,
        evaluation_train_epsilon=0., device=gpu, allocation_decisions=len(lengths),
        length_histogram=np.bincount(lengths, minlength=flags.max_search_steps+1).tolist(),
        min_fraction=float(np.mean(lengths==flags.min_search_steps)),
        interior_fraction=float(np.mean((lengths>flags.min_search_steps)&(lengths<flags.max_search_steps))),
        cap_fraction=float(np.mean(lengths==flags.max_search_steps)), mean_length=float(lengths.mean()),
        quartiles=np.quantile(lengths, [.25,.5,.75]).tolist(),
        pairs=len(pairs), paired_states=len(states), independent_streams=len(args.seeds)*args.batch,
        ema_utility_spearman=corr, mean_paired_utility=float(y.mean()),
        maximum_pair_task_tail_bound=max(r['pair_task_tail_bound'] for r in pairs),
        primary_target='discounted clipped task return minus remaining current-stage cost',
        paired_scope='forced-count entries, same state within STOP/CONTINUE pair; clustered states/repetitions',
        interpretation='single training seed pilot; distribution diversity alone does not prove useful computation',
        score_evaluation='DISABLED'))


if __name__ == '__main__':
    main()
