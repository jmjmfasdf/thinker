"""Frozen-checkpoint gradient audit and held-stream counter decoding."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from reward_learning_eval import BatchRuntime, categorical
from reward_probe import load_checkpoints, util, keyed_seed, preserve_tokens, digest


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def gradient_audit(actor, env, core):
    """Use an actual forward graph, holding parameters and buffers unchanged."""
    before = digest(actor.state_dict())
    actor.requires_grad_(True)
    captured = {}
    hook = actor.policy.register_forward_pre_hook(
        lambda module, inputs: captured.update(final=inputs[0]))
    try:
        out, _ = actor(env, core, compute_loss=False, greedy=False)
        target = out.voc_q.detach() + torch.tensor([.25, -.5], device=out.voc_q.device)
        gru_params = tuple(actor.tree_rep_encoder.parameters())
        head_params = tuple(actor.voc_head.parameters())
        parameters = gru_params + head_params
        def gradients(loss):
            grads = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
            def stats(g):
                return dict(connected=sum(x is not None for x in g),
                            nonzero=sum(x is not None and bool(torch.any(x != 0)) for x in g),
                            l2=float(sum((x.detach().double().square().sum() for x in g if x is not None),
                                         torch.zeros((), device=target.device)).sqrt()))
            return dict(gru=stats(grads[:len(gru_params)]), head=stats(grads[len(gru_params):]))
        from thinker.learn_actor import compute_dynamic_voc_loss
        def training_loss(q):
            return compute_dynamic_voc_loss(voc_q=q,
                target_control_logits=out.search_control_logits,
                behavior_control_logits=out.search_control_logits.detach(),
                control_action=torch.full_like(env.phase, util.PROCEED),
                control_valid=out.control_valid,
                voc_target=out.baseline[..., 0].detach()+.5,
                mode='control', dueling_q=actor.flags.voc_dueling_q,
                voc_state_value=out.baseline[..., 0],
                expected_gate_loss=actor.flags.voc_expected_gate_loss,
                gate_policy_schema_version=actor.flags.voc_gate_policy_schema_version).q_loss
        actual = gradients(training_loss(out.voc_q))
        connected = gradients(training_loss(actor.voc_head(captured['final']).view_as(target)))
        policy_loss = -torch.log_softmax(actor.policy(captured['final']), -1)[:, 0].mean()
        policy = gradients(policy_loss)
        norms = [type(m).__name__ for m in actor.tree_rep_encoder.modules()
                 if isinstance(m, (torch.nn.LayerNorm, torch.nn.GroupNorm,
                                   torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
        if actual['gru']['connected'] != 0 or actual['head']['nonzero'] == 0:
            raise AssertionError('VOC detach contract failed')
        if connected['gru']['nonzero'] == 0 or policy['gru']['nonzero'] == 0:
            raise AssertionError('Gradient positive control failed')
        return dict(actual_voc_loss=actual, same_loss_without_detach=connected,
                    policy_loss=policy, tree_encoder_normalization_modules=norms,
                    status='PASS')
    finally:
        hook.remove()
        actor.requires_grad_(False)
        if digest(actor.state_dict()) != before:
            raise AssertionError('Gradient audit changed checkpoint')


def collect(flags, actor_ck, model_ck, *, seed, steps, batch_size, mode, device):
    rt = BatchRuntime(flags, actor_ck, model_ck, seed, batch_size, device)
    before = digest(dict(actor=rt.actor.state_dict(), model=rt.model.state_dict()))
    buf = {k: [] for k in ('gru', 'features', 'token', 'count', 'depth', 'stream', 'decision', 'delta')}
    captured = {}
    hook1 = rt.actor.tree_rep_encoder.register_forward_hook(
        lambda module, inputs, output: captured.update(gru=output[0].detach()))
    hook2 = rt.actor.voc_head.register_forward_pre_hook(
        lambda module, inputs: captured.update(features=inputs[0].detach()))
    started = time.monotonic()
    audit = None
    real_count = 0
    forced_caps = np.array([1, 4, 8, 16, 30])
    try:
        state, info = rt.env.reset(rt.model)
        env = util.init_env_out(state, info, rt.flags, rt.actor.dim_actions, rt.actor.tuple_action)
        core = rt.actor.initial_state(batch_size=batch_size, device=rt.device)
        schema = rt.actor.tree_rep_meaning
        while real_count < steps:
            if audit is None:
                audit = gradient_audit(rt.actor, env, core)
            with torch.no_grad():
                out, core = rt.actor(env, core, compute_loss=False, greedy=False)
            phase = env.phase[0].cpu().numpy()
            count = env.search_steps[0].cpu().numpy()
            token = env.tree_reps[0].cpu().numpy()
            depth = np.rint(np.expm1(token[:, schema['rollout_depth']].reshape(-1))).astype(int)
            valid = (phase == util.SEARCH_PHASE) & out.control_valid[0].cpu().numpy()
            decoded_count = np.rint(np.expm1(token[:, schema['search_steps']].reshape(-1))).astype(int)
            if not np.array_equal(decoded_count[valid], count[valid]):
                raise AssertionError('Token counter misaligned with SEARCH row')
            if valid.any():
                values = dict(gru=captured['gru'][0].cpu().numpy(),
                              features=captured['features'].cpu().numpy(), token=token,
                              count=count, depth=depth, stream=seed*100+np.arange(batch_size),
                              decision=np.full(batch_size, real_count),
                              delta=(out.voc_q[0, :, 0]-out.voc_q[0, :, 1]).cpu().numpy())
                for name, value in values.items():
                    buf[name].append(np.asarray(value[valid]).copy())
            draws = np.array([[keyed_seed(seed+i, real_count, int(count[i]), int(phase[i])+j*100, 0)/2**32
                               for i in range(batch_size)] for j in range(3)])
            primary = categorical(out.pri_param[-1].reshape(batch_size, -1), draws[0])
            control = categorical(out.search_control_logits[-1], draws[1])
            if mode == 'forced':
                caps = forced_caps[(real_count+np.arange(batch_size)) % len(forced_caps)]
                continuation = categorical(out.search_control_logits[-1, :, :2], draws[2])
                control[valid] = np.where(count[valid] < caps[valid], continuation[valid], util.STOP)
            action = (torch.as_tensor(primary, device=rt.device).reshape(out.action[0].shape),
                      torch.as_tensor(control, device=rt.device).reshape(out.action[1].shape))
            with torch.no_grad():
                state, reward, done, truncated, info = rt.env.step(action, rt.model)
            env = preserve_tokens(env, util.create_env_out(action, state, reward, done, truncated, info, rt.flags), info)
            real = info['real_transition'].bool().cpu().numpy()
            if real.any() and not real.all():
                raise AssertionError('Unsynchronized real transition')
            real_count += int(real.all())
        if digest(dict(actor=rt.actor.state_dict(), model=rt.model.state_dict())) != before:
            raise AssertionError('Collection modified checkpoint')
        return {k: np.concatenate(v) for k, v in buf.items()}, dict(
            gradient_audit=audit, seconds=time.monotonic()-started,
            real_actions=steps*batch_size, mode=mode, seed=seed,
            input_hash=before, status='PASS')
    finally:
        hook1.remove(); hook2.remove(); rt.close()


def ridge_oof(x, y, groups, *, alphas=(.01, 1., 100., 10000.), splits=4):
    """Nested group CV; normalization and alpha selection never see test groups."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    prediction = np.full_like(y, np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    selected = []
    outer = GroupKFold(n_splits=min(splits, len(np.unique(groups))))
    for fold, (train, test) in enumerate(outer.split(x, y, groups)):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError('Group leakage')
        scores = []
        inner = GroupKFold(n_splits=min(3, len(np.unique(groups[train]))))
        for alpha in alphas:
            losses = []
            for a, b in inner.split(x[train], y[train], groups[train]):
                scaler = StandardScaler().fit(x[train[a]])
                reg = Ridge(alpha=alpha).fit(scaler.transform(x[train[a]]), y[train[a]])
                # Some sklearn releases squeeze a single regression target.
                # Keep both inner scoring and outer assignment two-dimensional
                # to prevent accidental (N, N) broadcasting against y.
                pred = np.asarray(reg.predict(scaler.transform(x[train[b]]))).reshape(-1, y.shape[1])
                losses.append(float(np.mean((pred-y[train[b]])**2)))
            scores.append(np.mean(losses))
        alpha = float(alphas[int(np.argmin(scores))])
        scaler = StandardScaler().fit(x[train])
        reg = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y[train])
        prediction[test] = np.asarray(reg.predict(scaler.transform(x[test]))).reshape(-1, y.shape[1])
        fold_id[test] = fold
        selected.append(alpha)
    if not np.isfinite(prediction).all() or np.any(fold_id < 0):
        raise AssertionError('Incomplete out-of-fold prediction')
    return prediction, fold_id, selected


def decode_report(data):
    y = np.stack([data['count'], data['depth']], -1)
    groups = data['stream']
    features = dict(gru=data['gru'], full_voc=data['features'], explicit=np.log1p(y),
                    constant=np.zeros((len(y), 1)))
    result = dict(rows=len(y), streams=len(np.unique(groups)),
                  count_histogram=np.bincount(data['count'].astype(int)).tolist(),
                  depth_histogram=np.bincount(data['depth'].astype(int)).tolist(), readouts={})
    predictions = {}
    for name, x in features.items():
        # Explicit invertible-input positive control uses the same estimator.
        predlog, folds, alphas = ridge_oof(x, np.log1p(y), groups)
        pred = np.maximum(0, np.expm1(np.clip(predlog, -10, 10)))
        predictions[name] = pred
        scores = {}
        for j, field in enumerate(('count', 'depth')):
            def metric(mask):
                error = pred[mask, j]-y[mask, j]
                return dict(n=int(mask.sum()), mae=float(np.abs(error).mean()),
                            rmse=float(np.mean(error**2)**.5),
                            rounded_accuracy=float(np.mean(np.rint(pred[mask, j]) == y[mask, j])))
            scores[field] = metric(np.ones(len(y), dtype=bool))
            scores[field]['bins'] = {label: metric(mask) for label, mask in (
                ('0', y[:, j] == 0), ('1-3', (y[:, j] >= 1) & (y[:, j] <= 3)),
                ('4-9', (y[:, j] >= 4) & (y[:, j] <= 9)), ('10+', y[:, j] >= 10)) if mask.any()}
        result['readouts'][name] = dict(scores=scores, selected_alphas=alphas)
    return result, dict(**predictions, target=y, fold=folds, stream=groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[19201, 19241])
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--forced-steps', type=int, default=96)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--fit-only', action='store_true')
    args = p.parse_args(); args.checkpoint = None; args.threads = 1
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    flags, actor, model, binding = load_checkpoints(args)
    records = []
    if not args.fit_only:
        for mode, steps in [('policy', args.steps), ('forced', args.forced_steps)]:
            for seed in args.seeds:
                path = args.output/f'{mode}_{seed}.npz'
                if path.exists():
                    raise FileExistsError(path)
                data, info = collect(flags, actor, model, seed=seed, steps=steps,
                    batch_size=args.batch_size, mode=mode, device=args.device)
                np.savez_compressed(path, **data)
                atomic_json(path.with_suffix('.json'), info)
                records.append(info)
                print(json.dumps(dict(rows=len(data['count']), **info)), flush=True)
        atomic_json(args.output/'collection.json', dict(checkpoint=binding, records=records,
                    args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    import wandb
    run = wandb.init(project='thinker', entity=os.getenv('WANDB_USER','jmme425'),
        group='voc-gru-information-20260925', name=f'gru-probe-{flags.name}',
        job_type='frozen-representation-probe', config=dict(bundle=binding, seeds=args.seeds,
        method='nested group cross-validation ridge; frozen checkpoints'), dir=str(args.output), mode='online')
    report = dict(game=flags.name, checkpoint=binding, modes={}, wandb_url=run.url)
    try:
        for mode in ['policy', 'forced']:
            chunks = [np.load(args.output/f'{mode}_{seed}.npz') for seed in args.seeds]
            data = {k:np.concatenate([d[k] for d in chunks]) for k in chunks[0].files}
            result, predictions = decode_report(data)
            report['modes'][mode] = result
            np.savez_compressed(args.output/f'{mode}_decoding.npz', **predictions)
            for name, readout in result['readouts'].items():
                for field, score in readout['scores'].items():
                    run.log({f'{mode}/{name}/{field}/{k}':v for k,v in score.items() if k != 'bins'})
            print(json.dumps(dict(mode=mode, result=result)), flush=True)
        report['status'] = 'COMPLETE'
        atomic_json(args.output/'decoding_report.json', report)
        run.summary['status']='COMPLETE'
    finally:
        run.finish()


if __name__ == '__main__':
    main()
