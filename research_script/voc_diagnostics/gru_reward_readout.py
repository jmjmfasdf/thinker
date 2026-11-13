"""Direct metadata ablation with paired actual STOP/CONTINUE reward targets.

Each stream is intervened on at a preassigned search count. Streams are
independent; all pre-intervention features, recurrent states and tree snapshots
must match between arms even when other streams have already branched.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from gru_information_probe import atomic_json, ridge_oof
from reward_learning_eval import BatchRuntime, categorical, tail_bound
from reward_probe import load_checkpoints, ReturnAccumulator, util, keyed_seed, preserve_tokens, digest


def core_stream(core, i, batch):
    if torch.is_tensor(core):
        if core.shape[0] != batch:
            raise AssertionError(f'Unexpected core batch axis {core.shape}')
        return core[i:i+1]
    return tuple(core_stream(x, i, batch) for x in core)


def tree_stream(snapshot, i):
    return {k: v[i] for k, v in snapshot.items()}


def arm_rollout(flags, actor_ck, model_ck, *, seed, repetition, prefix,
                targets, horizon, arm, device):
    batch = len(targets)
    rt = BatchRuntime(flags, actor_ck, model_ck, seed, batch, device)
    before = digest(dict(actor=rt.actor.state_dict(), model=rt.model.state_dict()))
    captured = {}
    hooks = [rt.actor.tree_rep_encoder.register_forward_hook(
        lambda m, a, o: captured.update(gru=o[0].detach())),
        rt.actor.voc_head.register_forward_pre_hook(lambda m, a: captured.update(features=a[0].detach()))]
    entries = [None]*batch
    acc = [ReturnAccumulator(float(flags.discounting), float(flags.think_cost)) for _ in targets]
    real_count = 0
    started = time.monotonic()
    try:
        state, info = rt.env.reset(rt.model)
        env = util.init_env_out(state, info, rt.flags, rt.actor.dim_actions, rt.actor.tuple_action)
        core = rt.actor.initial_state(batch_size=batch, device=rt.device)
        schema = rt.actor.tree_rep_meaning
        with torch.no_grad():
            for call in range((prefix+horizon+1)*(int(flags.max_search_steps)+4)):
                phase = env.phase[0].cpu().numpy()
                counts = env.search_steps[0].cpu().numpy()
                is_entry = np.array([e is None for e in entries]) & (real_count == prefix) & (phase == 0) & (counts == targets)
                pre_core = core
                out, core = rt.actor(env, core, compute_loss=False, greedy=False)
                if is_entry.any():
                    snapshot = rt.planner.dynamic_trace_snapshot(include_xs=True, include_hs=True)
                    for i in np.flatnonzero(is_entry):
                        token = env.tree_reps[0, i].cpu().numpy()
                        k = int(np.rint(np.expm1(token[schema['search_steps']][0])))
                        depth = int(np.rint(np.expm1(token[schema['rollout_depth']][0])))
                        if k != targets[i] or not bool(out.control_valid[0, i]):
                            raise AssertionError('Invalid target entry')
                        selected_env = {key:getattr(env, key)[:, i:i+1] for key in env._fields
                                        if torch.is_tensor(getattr(env, key))}
                        identity = digest(dict(env=selected_env, core=core_stream(pre_core, i, batch),
                                               tree=tree_stream(snapshot, i)))
                        features = captured['features'][i].cpu().numpy()
                        entries[i] = dict(hash=identity, count=k, depth=depth, features=features.tolist(),
                            gru=captured['gru'][0, i].cpu().tolist(),
                            delta=float(out.voc_q[0, i, 0]-out.voc_q[0, i, 1]),
                            p_continue=float(torch.softmax(out.search_control_logits[0, i], -1)[:2].sum()))
                reps = [repetition if entries[i] is not None else 0 for i in range(batch)]
                draws = np.array([[keyed_seed(seed+i, real_count, int(counts[i]), int(phase[i])+j*100, reps[i])/2**32
                                   for i in range(batch)] for j in range(3)])
                # Real-action draws must not depend on how many computations
                # preceded that action. Identical real policies then sample
                # identically across STOP/CONTINUE arms; imagined action draws
                # remain keyed by their search index.
                for i in np.flatnonzero(phase == util.NEED_REAL_ACTION_PHASE):
                    draws[0, i] = keyed_seed(seed+i, real_count, 0, int(phase[i]), reps[i])/2**32
                primary = categorical(out.pri_param[-1].reshape(batch, -1), draws[0])
                control = categorical(out.search_control_logits[-1], draws[1])
                conditional = categorical(out.search_control_logits[-1, :, :2], draws[2])
                pending = np.array([e is None for e in entries]) & (real_count == prefix) & (phase == 0)
                if np.any(counts[pending] > targets[pending]):
                    raise AssertionError('Missed entry')
                control[pending] = conditional[pending]
                control[is_entry] = util.STOP if arm == 'stop' else conditional[is_entry]
                action = (torch.as_tensor(primary, device=rt.device).reshape(out.action[0].shape),
                          torch.as_tensor(control, device=rt.device).reshape(out.action[1].shape))
                state, reward, done, truncated, info = rt.env.step(action, rt.model)
                env = preserve_tokens(env, util.create_env_out(action, state, reward, done, truncated, info, rt.flags), info)
                real = info['real_transition'].bool().cpu().numpy()
                if real.any() and not real.all():
                    raise AssertionError('Real transitions not synchronized')
                re = bool(real.all())
                terminal = (done | truncated.bool()).cpu().numpy().reshape(-1)
                task = reward.cpu().numpy().reshape(-1)
                think = info['think_reward'].cpu().numpy().reshape(-1)
                stage = info['stage_end'].cpu().numpy().reshape(-1)
                for i in range(batch):
                    if entries[i] is not None and not acc[i].terminal:
                        acc[i].observe(reward=float(task[i]) if re else 0., think_reward=float(think[i]),
                                       real_transition=re, stage_end=bool(stage[i]), done=bool(terminal[i]))
                real_count += int(re)
                if all(e is not None for e in entries) and (real_count-prefix >= horizon or all(a.terminal for a in acc)):
                    break
            else:
                raise RuntimeError('Watchdog')
        if any(e is None for e in entries):
            raise AssertionError('Missing entry')
        if arm == 'stop' and any(a.stage_computations != 0 for a in acc):
            raise AssertionError('STOP charged')
        if arm == 'continue' and any(a.stage_computations < 1 for a in acc):
            raise AssertionError('CONTINUE not executed')
        if digest(dict(actor=rt.actor.state_dict(), model=rt.model.state_dict())) != before:
            raise AssertionError('Checkpoint changed')
        return dict(entries=entries, returns=[a.scores() for a in acc],
                    tail_bounds=[tail_bound(float(flags.discounting), a.real_steps, float(flags.reward_clip), a.terminal) for a in acc],
                    seconds=time.monotonic()-started, real_steps=real_count-prefix, arm=arm,
                    seed=seed, repetition=repetition, prefix=prefix, targets=targets.tolist(),
                    real_action_rng='seed/stream/real_index/repetition; independent of search count')
    finally:
        for hook in hooks: hook.remove()
        rt.close()


def verify(a, b):
    if len(a['entries']) != len(b['entries']): raise AssertionError('Batch mismatch')
    for i, (aa, bb) in enumerate(zip(a['entries'], b['entries'])):
        if aa != bb: raise AssertionError(f'Paired entry mismatch stream {i}')


def reward_report(records, bootstrap=2000):
    states = {}
    for row in records:
        states.setdefault((row['seed'], row['stream'], row['prefix'], row['entry']['count']), []).append(row)
    rr = list(states.values())
    y = np.array([np.mean([x['delta_return'] for x in rows]) for rows in rr])
    x = np.array([rows[0]['entry']['features'] for rows in rr])
    gru = np.array([rows[0]['entry']['gru'] for rows in rr])
    meta = np.array([[rows[0]['entry']['count'], rows[0]['entry']['depth']] for rows in rr], dtype=float)
    groups = np.array([rows[0]['seed']*1000+rows[0]['stream'] for rows in rr])
    preds = dict(checkpoint=np.array([rows[0]['entry']['delta'] for rows in rr]), zero=np.zeros(len(rr)))
    alpha = {}
    for name, features in dict(full=x, full_plus_metadata=np.c_[x, np.log1p(meta)],
                               gru=gru, gru_plus_metadata=np.c_[gru, np.log1p(meta)],
                               metadata=np.log1p(meta), constant=np.zeros((len(y), 1))).items():
        pred, folds, selected = ridge_oof(features, y, groups)
        preds[name] = pred[:, 0]; alpha[name] = selected
    rng = np.random.default_rng(72831)
    unique = np.unique(groups)
    bootstrap_indices = [np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(unique, len(unique), replace=True)]) for _ in range(bootstrap)]
    results = {}
    for name, pred in preds.items():
        correlation = spearmanr(y, pred).statistic if np.std(pred)>1e-12 and np.std(y)>1e-12 else None
        results[name] = dict(rmse=float(np.mean((pred-y)**2)**.5), mae=float(np.abs(pred-y).mean()),
                             spearman=float(correlation) if correlation is not None and np.isfinite(correlation) else None)
    comparisons = {}
    for base in ['full', 'gru']:
        extra = base+'_plus_metadata'
        def gain(idx):
            return float(np.mean((preds[base][idx]-y[idx])**2)**.5-np.mean((preds[extra][idx]-y[idx])**2)**.5)
        comparisons[extra] = dict(rmse_improvement=gain(np.arange(len(y))),
            ci95=np.quantile([gain(idx) for idx in bootstrap_indices], [.025,.975]).tolist())
    result = dict(states=len(y), pairs=len(records), streams=len(unique),
        counts={str(k):int(np.sum(meta[:,0]==k)) for k in np.unique(meta[:,0])},
        target_mean=float(y.mean()), target_sd=float(y.std()), readouts=results,
        direct_metadata_gain=comparisons, selected_alphas=alpha,
        maximum_pair_tail_bound=max(row['tail_bound'] for row in records),
        interpretation='Frozen-policy local compute benefit; forced-count coverage; single trained checkpoint per game. Conditional bootstrap of nested OOF errors; no end-to-end retraining claim.')
    return result, dict(**preds, target=y, group=groups, count=meta[:,0], depth=meta[:,1], fold=folds)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[22301,22381])
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--prefix',type=int,default=128)
    p.add_argument('--counts',type=int,nargs='+',default=[0,1,3,6]);p.add_argument('--repetitions',type=int,default=2)
    p.add_argument('--horizon',type=int,default=1536);p.add_argument('--device',default='cuda:0')
    p.add_argument('--fit-only',action='store_true')
    args=p.parse_args();args.checkpoint=None;args.threads=1
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=True)
    flags,actor,model,binding=load_checkpoints(args)
    if min(args.counts)<0 or max(args.counts)>=flags.max_search_steps: raise ValueError('Counts outside valid control support')
    records=[]
    if not args.fit_only:
      for count in args.counts:
        # Homogeneous counts keep pre-entry model microbatch shapes identical.
        # Mixed counts produced ~1e-7 floating differences when another stream
        # branched early; do not weaken the exact paired-entry requirement.
        targets=np.full(args.batch_size,count,dtype=int)
        for seed in args.seeds:
            for rep in range(args.repetitions):
                kwargs=dict(seed=seed,repetition=rep,prefix=args.prefix,targets=targets,horizon=args.horizon,device=args.device)
                arms=[]
                for name in ['stop','continue']:
                    path=args.output/f'arm_k{count}_{seed}_{rep}_{name}.json'
                    if path.exists(): raise FileExistsError(path)
                    arm=arm_rollout(flags,actor,model,arm=name,**kwargs)
                    atomic_json(path,arm);arms.append(arm)
                    print(json.dumps(dict(count=count,seed=seed,rep=rep,arm=name,seconds=arm['seconds'])),flush=True)
                verify(*arms)
                if rep==0:
                    repeat=arm_rollout(flags,actor,model,arm='stop',**kwargs)
                    verify(arms[0],repeat)
                    if arms[0]['returns']!=repeat['returns']:raise AssertionError('Repeated STOP return mismatch')
                    atomic_json(args.output/f'repeat_k{count}_{seed}.json',dict(status='PASS',seconds=repeat['seconds']))
                for i in range(args.batch_size):
                    a,b=arms
                    records.append(dict(seed=seed,stream=i,prefix=args.prefix,repetition=rep,entry=a['entries'][i],
                        delta_return=b['returns'][i]['net_truncated']-a['returns'][i]['net_truncated'],
                        tail_bound=a['tail_bounds'][i]+b['tail_bounds'][i]))
                atomic_json(args.output/'pairs.json',records)
        atomic_json(args.output/'manifest.json',dict(checkpoint=binding,status='PAIRED_PASS',
            args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    else:
        records=json.loads((args.output/'pairs.json').read_text())
        collection=json.loads((args.output/'manifest.json').read_text())
        expected_pairs=len(args.counts)*len(args.seeds)*args.batch_size*args.repetitions
        if len(records) != expected_pairs or collection['checkpoint'] != binding:
            raise AssertionError('Fit-only requires a complete matching collection')
    import wandb
    run=wandb.init(project='thinker',entity=os.getenv('WANDB_USER','jmme425'),mode='online',
        group='voc-gru-information-20260925',name=f'gru-reward-readout-{flags.name}',
        job_type='frozen-reward-readout',dir=str(args.output),config=dict(bundle=binding,
        protocol='paired actual returns; nested held-stream ridge; metadata ablation'))
    try:
        report,preds=reward_report(records)
        fit_sources=[Path(__file__).resolve(),Path(ridge_oof.__code__.co_filename).resolve()]
        report.update(game=flags.name,wandb_url=run.url,status='COMPLETE',
            fit_provenance=dict(fit_only=args.fit_only,
                pairs_sha256=hashlib.sha256((args.output/'pairs.json').read_bytes()).hexdigest(),
                source_files=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                              for path in fit_sources]))
        atomic_json(args.output/'reward_readout_report.json',report)
        np.savez_compressed(args.output/'reward_predictions.npz',**preds)
        for model_name,metrics in report['readouts'].items():
            run.log({f'{model_name}/{key}':value for key,value in metrics.items() if value is not None})
        run.summary['status']='COMPLETE'
        print(json.dumps(report),flush=True)
    finally:
        run.finish()


if __name__=='__main__':main()
