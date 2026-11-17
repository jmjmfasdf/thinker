"""Offline loss/Adam audit using frozen checkpoints and production learner code.

Collects short fresh EnvPool trajectories, not historical training minibatches.
No training job, checkpoint, score evaluation, or production source is changed.
Adam counterfactuals use independent parameter/moment copies only.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'thinker'))
sys.path.insert(0, str(ROOT / 'research_script/voc_diagnostics'))
from reward_learning_eval import BatchRuntime
from reward_probe import preserve_tokens, util
import thinker.learn_actor as la
from thinker.self_play import TrainActorOut


class NullWriter:
    def __init__(self, **kwargs): pass
    def log(self, *args, **kwargs): pass
    def close(self, *args, **kwargs): pass


def stack_rows(rows, batch_size):
    data = {}
    for field in TrainActorOut._fields:
        if field == 'id':
            data[field] = torch.arange(batch_size).unsqueeze(0)
            continue
        values = [row[field] for row in rows]
        data[field] = None if values[0] is None else torch.cat(values, dim=0)
    return TrainActorOut(**data)


def collect(rt, calls, warmup):
    state, info = rt.env.reset(rt.model)
    env = util.init_env_out(state, info, rt.flags, rt.actor.dim_actions, rt.actor.tuple_action)
    core = rt.actor.initial_state(rt.batch_size, device=rt.device)
    rows, initial = [], None
    with torch.no_grad():
        for i in range(warmup + calls + 1):
            out, core = rt.actor(env, core, compute_loss=False, greedy=False)
            state, reward, done, truncated, info = rt.env.step(out.action, rt.model)
            env = preserve_tokens(env, util.create_env_out(out.action, state, reward, done, truncated, info, rt.flags), info)
            if i >= warmup:
                row = {}
                for field in TrainActorOut._fields:
                    if field == 'id': continue
                    value = getattr(env if field in util.EnvOut._fields else out, field)
                    row[field] = None if value is None else value.detach().clone()
                rows.append(row)
                if initial is None:
                    initial = tuple(x.detach().clone() for x in core)
            if (i + 1) % 128 == 0:
                print('collected augmented calls', i + 1, flush=True)
    return stack_rows(rows, rt.batch_size), initial


def vector(grads, parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().double().flatten()
                      for g, p in zip(grads, parameters)])


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm())) if a.norm() * b.norm() > 0 else None


def copied_adam_step(parameters, optimizer, grads, scale):
    assert len(optimizer.param_groups) == 1
    group = {k: copy.deepcopy(v) for k, v in optimizer.param_groups[0].items() if k != 'params'}
    cloned = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    opt = torch.optim.Adam([dict(group, params=cloned)])
    for p, q, g in zip(parameters, cloned, grads):
        opt.state[q] = copy.deepcopy(optimizer.state[p])
        q.grad = g.detach().clone() * scale
    before = [q.detach().clone() for q in cloned]
    opt.step()
    return vector([q.detach() - old for q, old in zip(cloned, before)], cloned)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--actor', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--calls', type=int, default=201)
    ap.add_argument('--warmup', type=int, default=256)
    ap.add_argument('--seed', type=int, default=20261114)
    ap.add_argument('--without-bc', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    ab, mb = args.actor.read_bytes(), args.model.read_bytes()
    ac = torch.load(io.BytesIO(ab), map_location='cpu', weights_only=False)
    mc = torch.load(io.BytesIO(mb), map_location='cpu', weights_only=False)
    flags = argparse.Namespace(**ac['flags'])
    flags.voc_gate_policy_schema_version = ac['voc_gate_policy_schema_version']
    rt = BatchRuntime(flags, ac, mc, args.seed, args.batch_size, 'cpu')
    try:
        batch, initial = collect(rt, args.calls, args.warmup)
        print('collection complete; constructing isolated learner', flush=True)
        lf = copy.deepcopy(flags)
        lf.parallel_actor = False
        lf.parallel = False
        lf.ckp = False
        lf.use_wandb = False
        lf.env_n = lf.actor_batch_size = args.batch_size
        lf.icopro_device = 'cpu'
        lf.ckpdir = lf.savedir = str(args.output)
        lf.xpid = 'offline-only'
        if args.without_bc: lf.icopro_data_path = ''
        # Construct with the documented fresh-head precondition, then restore
        # checkpoint tensors/moments. No resume/save or scheduler step is run.
        rt.actor.requires_grad_(True)
        with torch.no_grad():
            rt.actor.voc_head.weight.zero_(); rt.actor.voc_head.bias.zero_()
            rt.actor.voc_gate_head.weight.zero_(); rt.actor.voc_gate_head.bias.zero_()
        with patch.object(la, 'FileWriter', NullWriter):
            learner = la.SActorLearner(None, {}, lf, actor_net=rt.actor, device=torch.device('cpu'))
        rt.actor.load_state_dict(ac['actor_net_state_dict'], strict=True)
        learner.optimizer.load_state_dict(ac['actor_net_optimizer_state_dict'])
        learner.voc_ema_gate_weight.copy_(ac['voc_ema_gate_head_state_dict']['weight'])
        learner.voc_ema_gate_bias.copy_(ac['voc_ema_gate_head_state_dict']['bias'])
        for name in ['voc_update_count', 'voc_gate_update_count', 'voc_ema_gate_update_count',
                     'voc_ema_gate_parent_update_count']:
            setattr(learner, name, ac[name])
        learner.real_step = int(ac['real_step'])
        learner.anneal_c = 1. - util.schedule_progress(lf, learner.real_step)
        learner.imitation_model_source = rt.model
        learner._assert_voc_gate_exact_projection_invariant()
        if not args.without_bc:
            learner._restore_loader_rng_state(ac['imitation_rng_state'])
        rt.actor.train(True)
        baselines, qrecords, traces = [], [], []
        orig_b, orig_q, orig_v = la.compute_baseline_loss, la.compute_dynamic_voc_loss, la.compute_v_trace
        def baseline(*a, **kw):
            baselines.append(kw)
            return orig_b(*a, **kw)
        def critic(**kw):
            result = orig_q(**kw)
            qrecords.append((kw, result))
            return result
        def trace(**kw):
            result = orig_v(**kw)
            traces.append((kw, result))
            return result
        with patch.object(la, 'compute_baseline_loss', baseline), patch.object(la, 'compute_dynamic_voc_loss', critic), patch.object(la, 'compute_v_trace', trace):
            losses, shifted = learner.compute_losses(batch, initial)
        print('production losses computed', flush=True)
        params = tuple(rt.actor.tree_rep_encoder.parameters())
        voc_grads = learner._voc_gru_adapter.pending
        voc = vector(voc_grads, params)
        def grad(loss):
            if not loss.requires_grad:
                return vector([None] * len(params), params)
            return vector(torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True), params)
        components = {'task_policy': losses['pg_loss'] / rt.actor.dim_actions}
        base = baselines[0]
        re_rows = (base['baseline'] - base['target_baseline'].detach()).square() * lf.baseline_cost
        for name, mask in [('WAIT', ~shifted.policy_valid.bool()),
                           ('SEARCH', shifted.policy_type == util.POLICY_SEARCH),
                           ('REAL', shifted.policy_type == util.POLICY_REAL)]:
            components['task_value_' + name] = (re_rows * mask).sum()
        for prefix in learner.rewards_ls[1:]:
            cost = getattr(lf, prefix + '_cost')
            if getattr(lf, prefix + '_cost_anneal'): cost *= learner.anneal_c
            components[prefix + '_policy'] = cost * losses[prefix + '_pg_loss'] / rt.actor.dim_actions
            components[prefix + '_value'] = cost * lf.baseline_cost * losses[prefix + '_baseline_loss']
        components['entropy_real'] = lf.entropy_cost * losses['entropy_loss'] / rt.actor.dim_actions
        components['entropy_search'] = lf.im_entropy_cost * losses['im_entropy_loss'] * rt.actor.dim_actions
        components['regularization'] = lf.reg_cost * losses['reg_loss']
        if 'icopro_scaled_loss' in losses: components['BC'] = losses['icopro_scaled_loss']
        torch.testing.assert_close(sum(components.values()), losses['total_loss'], rtol=2e-5, atol=1e-4, check_dtype=False)
        total = grad(losses['total_loss'])
        gradients = {name: grad(loss) for name, loss in components.items()}
        torch.testing.assert_close(sum(gradients.values()), total, rtol=3e-4, atol=2e-3)
        rows = [dict(component=k, loss=float(components[k].detach()), gru_norm=float(g.norm()),
                     projection_on_total=float(torch.dot(g, total) / total.square().sum()),
                     cosine_total=cosine(g, total), cosine_voc=cosine(g, voc)) for k, g in gradients.items()]
        rows.append(dict(component='VOC', loss=float(losses['_voc_total_loss'].detach()),
                         gru_norm=float(voc.norm()), projection_on_total=float(torch.dot(voc, total)/total.square().sum()),
                         cosine_total=cosine(voc, total), cosine_voc=1.))
        pd.DataFrame(rows).to_csv(args.output/'gradient_components.csv', index=False)
        pd.DataFrame({a: {b: cosine(x, y) for b, y in gradients.items()} for a, x in gradients.items()}).to_csv(args.output/'gradient_cosines.csv')
        # Full actor norm determines clipping, including all image encoders.
        losses['total_loss'].backward()
        opt_params = [p for group in learner.optimizer.param_groups for p in group['params']]
        base_norm_sq = sum(float(p.grad.double().square().sum()) for p in opt_params if p.grad is not None)
        base_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        torch.testing.assert_close(vector(base_grads, params), total, rtol=3e-4, atol=2e-3)
        threshold = lf.actor_grad_norm_clipping * batch.reward.shape[0] * args.batch_size
        adam_rows, updates = [], {}
        for multiplier in [0., 1., 10., 100.]:
            combined = total + multiplier * voc
            norm = max(0., base_norm_sq - float(total.square().sum()) + float(combined.square().sum())) ** .5
            clip = min(1., threshold / (norm + 1e-6))
            gs = [g + multiplier * (torch.zeros_like(p) if v is None else v) for g, v, p in zip(base_grads, voc_grads, params)]
            update = copied_adam_step(params, learner.optimizer, gs, clip)
            updates[multiplier] = update
            row = dict(voc_multiplier=multiplier, clip_coefficient=clip, actor_raw_norm=norm, gru_update_norm=float(update.norm()))
            if multiplier:
                row.update(relative_change_vs_no_voc=float((update-updates[0.]).norm()/updates[0.].norm()),
                           update_cosine_vs_no_voc=cosine(update, updates[0.]))
            adam_rows.append(row)
        pd.DataFrame(adam_rows).to_csv(args.output/'matched_adam.csv', index=False)
        # Selected-target residual is observable; true common two-action error
        # is not identifiable without paired counterfactual outcomes.
        kw, result = qrecords[0]
        valid = result.q_train_valid
        p = result.continue_probability.detach()
        b = result.behavior_continue_probability.detach()
        selected_c = result.gate_action == 0
        derivative = torch.where(selected_c, 1-p, -p)
        err = -result.td_error.detach()
        qdata = dict(p_current=p, p_behavior=b, continue_choice=selected_c,
                     delta=result.delta_q.detach(), baseline=kw['voc_state_value'].detach(),
                     target=result.target.detach(), td_error=result.td_error.detach(),
                     gradient_delta=err.clamp(-1,1)*derivative,
                     dQ_selected_dDelta=derivative, search_steps=batch.search_steps[:-1])
        pd.DataFrame({k: v[valid].cpu().numpy() for k,v in qdata.items()}).to_csv(args.output/'critic_rows.csv', index=False)
        metadata = dict(actor_path=str(args.actor), model_path=str(args.model),
            actor_sha256=hashlib.sha256(ab).hexdigest(), model_sha256=hashlib.sha256(mb).hexdigest(),
            actor_step=int(ac['real_step']), model_step=int(mc['real_step']), schema=ac['voc_gate_policy_schema_version'],
            seed=args.seed, batch_size=args.batch_size, unroll_transitions=args.calls, warmup_calls=args.warmup,
            real_actions_per_stream=shifted.real_transition.sum(0).tolist(), wait_fraction=float((~shifted.policy_valid.bool()).float().mean()),
            optional_train_rows=int(valid.sum()), bc_targets=float(losses.get('icopro_count', 0)),
            gru_voc_to_other_ratio=float(voc.norm()/total.norm()), actor_clip_threshold=threshold,
            maximum_current_behavior_p_gap=float((p-b)[valid].abs().max()),
            seconds=time.monotonic()-started,
            limitations=['Fresh short frozen-policy sample, not the original historical training batch.',
                         'Behavior/current weights identical, so this does not measure historical asynchronous policy lag.',
                         'One-step Adam counterfactuals do not establish training benefit of multiplying VOC gradients.',
                         'Only Enduro sub001 current min1/episode-memory lineage; no score evaluation.'],
            checks=['Weighted losses sum to production total.', 'Component gradients sum to production GRU gradient.',
                    'Checkpoint EMA gate projection invariant.', 'Full actor gradient used for clipping; stored Adam moments restored.'])
        (args.output/'manifest.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2), flush=True)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
        learner._voc_gru_adapter.close()
    finally:
        rt.close()


if __name__ == '__main__': main()
