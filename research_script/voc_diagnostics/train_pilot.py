"""Bounded real RL+imitation pilot with immutable periodic reward evaluation.

Uses the production SelfPlayWorker, actor and model learners synchronously.
A --resume restores learner/optimizer/BC RNG state; environment, planner,
recurrent state and online replay restart explicitly (not bitwise continuation).
This smoke configuration is not the 5M scientific comparison.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import types

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'thinker'))
import numpy as np
import torch
import yaml
from thinker import util
from thinker.self_play import SelfPlayWorker
from thinker.gym_add import wrapper
from thinker.core import file_writer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def scalar_stats(stats):
    result = {}
    for key, value in stats.items():
        if torch.is_tensor(value) and value.numel() == 1:
            value = value.detach().item()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float, bool)):
            result[key] = value if np.isfinite(value) else None
    return result


def freeze_runtime(output):
    dest = output / 'source' / 'runtime'
    records = {}
    paths = list((ROOT/'thinker'/'thinker').rglob('*.py'))
    paths += list((ROOT/'thinker'/'thinker').rglob('*.pyx'))
    paths += list((ROOT/'thinker'/'thinker').rglob('*.so'))
    paths += list((ROOT/'thinker'/'config').rglob('*.yaml'))
    paths += [ROOT/'thinker'/'visual_behav.py']
    for path in paths:
        rel = path.relative_to(ROOT/'thinker')
        target = dest/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(path) != sha(target):
                raise ValueError(f'Runtime changed since pilot creation: {rel}')
        else:
            shutil.copyfile(path, target)
        records[str(rel)] = sha(target)
    for path in Path(__file__).parent.glob('*.py'):
        target = output/'source'/'diagnostics'/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(path) != sha(target):
            raise ValueError(f'Diagnostic changed since pilot creation: {path.name}')
        shutil.copyfile(path, target)
    atomic_json(output/'source'/'manifest.json', records)
    return dest


def save_bundle(worker, output, update, attempt):
    actor, model = worker.actor_learner, worker.env.model_learner
    actor.save_checkpoint()
    model.save_checkpoint()
    path = output/'snapshots'/f'update_{update:06d}'
    path.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, source in [('actor', actor.ckp_path), ('model', model.ckp_path)]:
        target = path/f'{name}.tar'
        shutil.copyfile(source, target)
        files[name] = dict(path=str(target), sha256=sha(target))
    config = path/'config.yaml'
    config.write_text(yaml.safe_dump(vars(worker.flags)))
    files['config'] = dict(path=str(config), sha256=sha(config))
    result = dict(files=files, update=update, attempt=attempt,
        actor_real_step=int(actor.real_step), model_real_step=int(model.real_step),
        actor_augmented_step=int(actor.step), model_sample_step=int(model.step),
        imitation_updates=int(actor.imitation_update_count),
        voc_updates=int(actor.voc_update_count), ema_updates=int(actor.voc_ema_gate_update_count),
        imitation_data_signature=actor.imitation_data_signature,
        model_updates_m=int(model._model_grad_step_count_m),
        model_updates_p=int(model._model_grad_step_count_p),
        runtime_manifest_sha256=sha(output/'source'/'manifest.json'))
    atomic_json(path/'bundle.json', result)
    return path/'bundle.json', result


def evaluate(bundle, output, runtime, args, run):
    dest = bundle.parent/'reward_evaluation'
    command = [sys.executable, str(Path(__file__).with_name('reward_probe.py')),
        '--bundle', str(bundle), '--output', str(dest), '--device', args.device,
        '--seeds', str(args.eval_seed), '--prefix-steps', str(args.eval_prefix),
        '--repetitions', str(args.eval_repetitions), '--horizon', str(args.eval_horizon),
        '--policy-epsilon']
    env = dict(os.environ, VOC_DIAGNOSTIC_RUNTIME=str(runtime))
    with (bundle.parent/'evaluation.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = [json.loads(line) for line in (dest/'pairs.jsonl').read_text().splitlines()]
    values = np.array([r['delta_net_truncated'] for r in rows])
    prediction = np.array([r['ema_delta_q'] for r in rows])
    metrics = dict(pairs=len(rows), delta_net_truncated_mean=float(values.mean()),
        delta_net_bootstrapped_mean=float(np.mean([r['delta_net_bootstrapped'] for r in rows])),
        ema_delta_mean=float(prediction.mean()), rmse_single_rollout=float(np.mean((values-prediction)**2)**.5),
        full_return_pair_fraction=float(np.mean([r['full_return_observed'] for r in rows])))
    run.log({f'evaluation/{k}':v for k,v in metrics.items()})
    atomic_json(dest/'summary.json', dict(**metrics, scope='bounded smoke; not critic acceptance evidence'))
    return metrics


def make_flags(args, output):
    config_path = ROOT/'logs'/'thinker'/f'dynamic-imitation-{args.game}-voc-sub1-leavesession1'/'config_c.yaml'
    raw = yaml.safe_load(config_path.read_text())
    for key in ('voc_gate_policy_schema_version', 'config'):
        raw.pop(key, None)
    raw.update(name='Enduro-v5' if args.game == 'enduro' else 'Pong-v5',
        xpid='train', savedir=str(output), ckp=args.resume,
        preload='', preload_actor='', voc_parent_checkpoint='',
        parallel=False, parallel_actor=False, env_n=4, actor_batch_size=4,
        self_play_n=1, actor_unroll_len=32, model_batch_size=4,
        model_unroll_len=4, model_return_n=5, model_warm_up_n=32,
        model_buffer_n=1024, total_steps=4096, checkpoint_interval=0,
        min_replay_ratio=1., max_replay_ratio=1., base_seed=args.seed,
        icopro_data_path=str(ROOT/'behavioral_data_block'), icopro_batch_size=2,
        icopro_device='cuda' if args.device.startswith('cuda') else 'cpu',
        gpu_self_play=1. if args.device.startswith('cuda') else 0.,
        use_wandb=True, voc_gate_exact_projection=(args.mode == 'projection'),
        voc_gate_param_align=False, voc_gate_epsilon_greedy_execution=False)
    flags = util.create_flags(['default_thinker.yaml', 'default_actor.yaml'],
        save_flags=True, post_fn=util.process_flags_actor, **raw)
    # Env reconstructs flags from defaults. All explicit actor keys are known
    # to default_thinker too only after its config has been composed here.
    return flags


def attach_logging(learner, run, output, kind):
    original = learner.plogger.log
    def log(stats, *args, **kwargs):
        original(stats, *args, **kwargs)
        row = scalar_stats(stats)
        row['pipeline/actor_or_model_real_step'] = int(learner.real_step)
        with (output/f'{kind}_metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        run.log({k:v for k,v in row.items() if v is not None})
    learner.plogger.log = log


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game', choices=['enduro','pong'], required=True)
    p.add_argument('--mode', choices=['control','projection'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=8, help='additional production actor unroll updates')
    p.add_argument('--eval-every', type=int, default=4)
    p.add_argument('--eval-horizon', type=int, default=64)
    p.add_argument('--eval-prefix', type=int, default=16)
    p.add_argument('--eval-repetitions', type=int, default=1)
    p.add_argument('--eval-seed', type=int, default=303)
    p.add_argument('--seed', type=int, default=701)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.updates,args.eval_every,args.eval_horizon,args.eval_repetitions) < 1:
        p.error('Budgets must be positive')
    output = args.output.resolve()
    state_path = output/'pilot_state.json'
    if args.resume:
        previous = json.loads(state_path.read_text())
        if previous['status'] != 'PASS':
            raise ValueError('Only completed pilot chunks can resume')
        for key in ('game','mode','seed','device','eval_every','eval_horizon','eval_prefix','eval_repetitions','eval_seed'):
            if previous['arguments'][key] != getattr(args, key):
                raise ValueError(f'Resume configuration mismatch: {key}')
        for name in ('actor', 'model'):
            if sha(output/'train'/f'ckp_{name}.tar') != previous['checkpoint']['files'][name]['sha256']:
                raise ValueError(f'Resume checkpoint differs from completed snapshot: {name}')
        start_update = previous['update']
        attempt = previous['attempt']+1
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        start_update, attempt = 0, 0
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    flags = make_flags(args, output)
    runtime = freeze_runtime(output)
    import wandb
    run = wandb.init(project=flags.project, entity=os.getenv('WANDB_ENTITY') or os.getenv('WANDB_USER') or None,
        name=f'voc-pipeline-{args.game}-{args.mode}-a{attempt}', group='voc-training-pipeline-20260924',
        tags=['voc','pipeline-smoke',args.mode,args.game], dir=str(output), mode='online',
        config=dict(training=vars(flags), pilot={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
    if run.settings.mode != 'online':
        raise RuntimeError('Training requires online W&B logging')
    # Record bounded environment settings without serializing credentials.
    gather = file_writer.gather_metadata
    def metadata():
        value = gather()
        value['env'] = {k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS') if k in os.environ}
        return value
    file_writer.gather_metadata = metadata
    def seeded_envpool(name, training_flags, env_n=1):
        import envpool
        env = envpool.make_gymnasium(name, num_envs=env_n, batch_size=env_n, num_threads=1,
            seed=args.seed + attempt*100000, gray_scale=training_flags.grayscale,
            episodic_life=True, stack_num=training_flags.frame_stack_n)
        return wrapper.EnvPoolWrap(env, env_n, frame_stack_n=training_flags.frame_stack_n)
    wrapper.create_envpool = seeded_envpool
    worker = None
    try:
        worker = SelfPlayWorker.__ray_metadata__.modified_class(
            {'actor_buffer':None,'actor_param_buffer':None,'model_param_buffer':None},
            None, rank=0, env_n=flags.env_n, flags=flags)
        actor, model = worker.actor_learner, worker.env.model_learner
        before = dict(actor_real_step=int(actor.real_step), model_real_step=int(model.real_step),
            voc_updates=int(actor.voc_update_count), imitation_updates=int(actor.imitation_update_count),
            model_updates=int(model._model_grad_step_count_p))
        if args.resume:
            for key in ('actor_real_step','model_real_step','voc_updates','imitation_updates'):
                if before[key] != previous['checkpoint'][key]:
                    raise AssertionError(f'Resume counter mismatch: {key}')
        attach_logging(actor, run, output, 'actor')
        attach_logging(model, run, output, 'model')
        initial = hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in worker.actor_net.state_dict().values())).hexdigest()
        atomic_json(output/f'attempt_{attempt}.json', dict(before=before, initial_actor_sha256=initial,
            wandb_url=run.url, wandb_id=run.id, environment_seed=args.seed+attempt*100000,
            resume_semantics='optimizer, schedules, learner counters and BC RNG restored; environment, recurrent state and online replay cold'))
        with torch.no_grad():
            state, info = worker.env.reset()
            env_out = worker.init_env_out(state, info)
            core = worker.actor_net.initial_state(batch_size=flags.env_n, device=worker.device)
            actor_out, core, env_out, info = worker.env_step(env_out, core)
        final = None
        for update in range(start_update+1, start_update+args.updates+1):
            initial_core = util.tuple_map(core, lambda x:x.detach())
            worker.write_actor_buffer(env_out, actor_out, 0)
            with torch.no_grad():
                for t in range(flags.actor_unroll_len):
                    actor_out, core, env_out, info = worker.env_step(env_out, core)
                    worker.write_actor_buffer(env_out, actor_out, t+1)
            worker.actor_net.train(True)
            actor.consume_data((worker.actor_local_buffer, initial_core))
            worker.actor_net.train(False)
            if args.mode == 'projection':
                actor._assert_voc_gate_exact_projection_invariant()
            run.log(dict(pipeline_update=update, **{f'pipeline/{k}':v for k,v in dict(
                actor_real_step=actor.real_step, model_real_step=model.real_step,
                voc_updates=actor.voc_update_count, imitation_updates=actor.imitation_update_count,
                model_updates=int(model._model_grad_step_count_p)).items()}))
            if update % args.eval_every == 0 or update == start_update+args.updates:
                bundle, final = save_bundle(worker, output, update, attempt)
                evaluate(bundle, output, runtime, args, run)
                print(json.dumps(dict(update=update, checkpoint=final)), flush=True)
        if final['voc_updates'] <= before['voc_updates'] or final['imitation_updates'] <= before['imitation_updates']:
            raise AssertionError('VOC and actual human imitation must both update')
        if final['model_updates_p'] <= before['model_updates'] or final['model_updates_m'] < 1:
            raise AssertionError('Both world model branches must update')
        atomic_json(state_path, dict(status='PASS', attempt=attempt, update=update, checkpoint=final,
            before=before, arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            wandb_url=run.url, wandb_id=run.id, scope='training pipeline smoke, not efficacy'))
        run.summary['pipeline_status'] = 'PASS'
        run.finish(exit_code=0)
    except BaseException:
        run.summary['pipeline_status'] = 'FAIL'
        run.finish(exit_code=1)
        raise
    finally:
        if worker is not None:
            if worker.actor_learner.bc_runner is not None:
                worker.actor_learner.bc_runner.close()
            worker.env.close()

if __name__ == '__main__':
    main()
