"""Fresh reward VOC + selective GRU learning with committed restart bundles.

One fold, one latest-VOC condition. A restart restores learner/optimizer/EMA,
schedule and RNG state, but explicitly restarts environment/planner/model replay.
Evaluation consumes immutable bundles on other SLURM GPUs.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import random
import shutil
import signal
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from train_pilot import (ROOT, util, torch, np, yaml, wrapper, file_writer,
    SelfPlayWorker, freeze_runtime, save_bundle, atomic_json, sha, scalar_stats)
from train_reward_learning import attach_logging
from voc_gru_gradient import SelectiveVOCGradient
from voc10m_metrics import TrainingMetrics

MILESTONES = (0,200_000,500_000,1_000_000,2_000_000,3_000_000,5_000_000,7_500_000,10_000_000)
GAMES = {'enduro':('Enduro-v5',0),'pong':('Pong-v5',1),'spaceinvaders':('SpaceInvaders-v5',2)}


def recipe(args):
    return dict(game=args.game,seed=args.seed,total_steps=args.env_steps,warmup=args.warmup,
        subject=1,train_sessions=[2,3,4],holdout_sessions=[1],env_n=8,
        gradient='selective_voc_to_tree_gru',precision='fp32',
        bc_schedule='inherited_per_actor_update',schedule_total_steps=100_000_000)


def make_flags(args, resume):
    config = ROOT/'logs/thinker'/f'dynamic-imitation-{args.game}-voc-sub1-leavesession1/config_c.yaml'
    raw = yaml.safe_load(config.read_text())
    for key in ('config','voc_gate_policy_schema_version'):
        raw.pop(key,None)
    raw.update(name=GAMES[args.game][0],xpid='train',savedir=str(args.output),ckp=resume,
        preload='',preload_actor='',voc_parent_checkpoint='',parallel=False,parallel_actor=False,
        env_n=8,actor_batch_size=8,self_play_n=1,actor_unroll_len=64,
        model_batch_size=8,model_unroll_len=10,model_return_n=5,
        model_warm_up_n=args.warmup,model_buffer_n=100_000,
        min_replay_ratio=4.,max_replay_ratio=4.,total_steps=args.env_steps,
        schedule_total_steps=100_000_000,checkpoint_interval=0,base_seed=args.seed,
        float16=False,model_float16=False,model_sr_dynamics_norm='group',
        icopro_subjects='1',icopro_game_id=GAMES[args.game][1],
        icopro_train_sessions='2,3,4',icopro_holdout_sessions='1',
        icopro_data_path=os.environ.get('VOC_REWARD_DATA_ROOT',str(ROOT/'behavioral_data_block')),
        icopro_batch_size=8,icopro_device='cuda',gpu_self_play=1.,use_wandb=True,
        voc_gate_exact_projection=True,voc_gate_param_align=False,
        voc_gate_epsilon_greedy_execution=False)
    return util.create_flags(['default_thinker.yaml','default_actor.yaml'],save_flags=True,
                             post_fn=util.process_flags_actor,**raw)


def restore_files(output, latest):
    binding = json.loads(Path(latest['bundle']).read_text())
    for name in ('actor','model','config','runner'):
        record = binding['files'][name]
        if sha(record['path']) != record['sha256']:
            raise ValueError('Corrupt committed checkpoint: '+name)
    for name in ('actor','model'):
        dest = output/'train'/f'ckp_{name}.tar'
        shutil.copyfile(binding['files'][name]['path'], str(dest)+'.restore')
        os.replace(str(dest)+'.restore',dest)
    # These local environment checkpoints do not serialize full EnvPool/planner
    # state. Never accidentally mix one with the declared cold restart.
    env_ckp = output/'train/ckp_env.npz'
    if env_ckp.exists():
        env_ckp.rename(output/f'ignored_env_checkpoint_{time.time_ns()}.npz')
    return binding, torch.load(binding['files']['runner']['path'],map_location='cpu',weights_only=False)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def set_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def queue_evaluation(args, bundle, steps):
    if not args.campaign or args.smoke:
        return
    tasks = args.campaign/'eval_tasks'
    tasks.mkdir(exist_ok=True)
    cell = args.output.name
    from voc10m_parallel_monitor import evaluation_kinds, evaluation_policy
    for kind in evaluation_kinds(cell,steps,evaluation_policy(args.campaign)):
        task = tasks/f'{cell}_{steps:08d}_{kind}.json'
        payload = dict(cell=cell,game=args.game,real_step=steps,kind=kind,bundle=str(bundle),
                       output=str(args.campaign/'evaluations'/task.stem))
        if task.exists():
            if json.loads(task.read_text()) != payload:
                raise ValueError('Evaluation task identity changed')
        else:
            atomic_json(task,payload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game',choices=list(GAMES),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--campaign',type=Path)
    p.add_argument('--seed',type=int,default=701)
    p.add_argument('--env-steps',type=int,default=10_000_000)
    p.add_argument('--warmup',type=int,default=1024)
    p.add_argument('--stop-at',type=int)
    p.add_argument('--snapshot-every',type=int,default=100_000)
    p.add_argument('--max-seconds',type=float,default=5.5*86400)
    p.add_argument('--smoke',action='store_true')
    args = p.parse_args(); args.output=args.output.resolve()
    if args.campaign: args.campaign=args.campaign.resolve()
    if any(n<=0 or n%8 for n in (args.env_steps,args.snapshot_every,args.warmup)):
        p.error('Budgets must be positive multiples of eight')
    stop_at = args.stop_at or args.env_steps
    if stop_at>args.env_steps or stop_at%8 or stop_at<=args.warmup:
        p.error('Invalid stop-at')
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('This runner requires a SLURM GPU allocation')
    args.output.mkdir(parents=True,exist_ok=True)
    lock = (args.output/'training.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    latest_path = args.output/'latest.json'
    resume = latest_path.exists()
    spec = recipe(args)
    spec_path = args.output/'recipe.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError('Resume recipe differs')
    if not spec_path.exists(): atomic_json(spec_path,spec)
    restored = binding = None
    if resume:
        binding,restored = restore_files(args.output,json.loads(latest_path.read_text()))
        if binding['actor_real_step']>=stop_at:
            print('Committed budget already reached',flush=True); return
    elif any((args.output/'train').glob('ckp_*.tar')):
        raise RuntimeError('Uncommitted training state: inspect before restart')
    flags=make_flags(args,resume); runtime=freeze_runtime(args.output)
    if binding and sha(args.output/'source/manifest.json')!=binding['runtime_manifest_sha256']:
        raise ValueError('Resume runtime hash differs')
    attempts=args.output/'attempts';attempts.mkdir(exist_ok=True)
    attempt=max([int(x.stem) for x in attempts.glob('*.json')]+[-1])+1
    steps=int(binding['actor_real_step']) if binding else 0
    update=int(binding['update']) if binding else 0
    initial_steps=steps
    counters=restored['counters'] if restored else dict(bc_targets_attempted=0,bc_targets_committed=0,
        actor_updates_committed=0,actor_updates_skipped=0)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    import wandb
    run=wandb.init(project='thinker',entity=os.getenv('WANDB_USER','jmme425'),mode='online',
        group=args.campaign.name if args.campaign else 'voc10m-smoke',
        name=f'voc10m-{args.game}-sub001-s{args.seed}-a{attempt}',
        job_type='smoke' if args.smoke else 'training',dir=str(args.output),
        config=dict(training=vars(flags),recipe=spec,attempt=attempt,resume=resume))
    if run.settings.mode!='online':raise RuntimeError('Online W&B required')
    run.define_metric('real_step')
    for prefix in ('actor','model','train','gradient','resources'):
        run.define_metric(prefix+'/*',step_metric='real_step')
    gather=file_writer.gather_metadata
    def metadata():
        v=gather();v['env']={k:os.environ[k] for k in ('SLURM_JOB_ID','CUDA_VISIBLE_DEVICES') if k in os.environ};return v
    file_writer.gather_metadata=metadata
    def seeded_env(name,f,env_n=1):
        import envpool
        env=envpool.make_gymnasium(name,num_envs=env_n,batch_size=env_n,num_threads=1,
            seed=args.seed+attempt*100000,gray_scale=f.grayscale,stack_num=f.frame_stack_n,episodic_life=True)
        return wrapper.EnvPoolWrap(env,env_n,frame_stack_n=f.frame_stack_n)
    wrapper.create_envpool=seeded_env
    worker=adapter=telemetry=None;started=time.monotonic();last_save=started
    interrupted=False;phase='INITIALIZING';last_metrics={};last_resources={};last_bundle=None
    def request_stop(signum,frame):
        nonlocal interrupted
        interrupted=True
    for signum in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT): signal.signal(signum,request_stop)
    def status(new_phase,**extra):
        nonlocal phase
        phase=new_phase
        atomic_json(args.output/'status.json',dict(phase=phase,game=args.game,env_steps=steps,
            total_env_steps=args.env_steps,attempt=attempt,update=update,job_id=os.getenv('SLURM_JOB_ID'),
            node=os.getenv('SLURMD_NODENAME'),wandb_url=run.url,updated_at=time.time(),
            seconds=time.monotonic()-started,actions_per_second=(steps-initial_steps)/max(time.monotonic()-started,1),
            metrics=last_metrics,resources=last_resources,**extra))
    def checkpoint():
        nonlocal last_save,last_bundle
        if last_bundle and json.loads(last_bundle.read_text())['update']==update:return last_bundle
        if worker.env.model_learner.real_step!=steps:
            raise AssertionError('Model/actor checkpoint real steps differ')
        telemetry.flush()
        bundle,b=save_bundle(worker,args.output,update,attempt)
        runner=bundle.parent/'runner.tar'
        torch.save(dict(rng=rng_state(),telemetry=telemetry.state(),counters=counters),runner)
        b['files']['runner']=dict(path=str(runner),sha256=sha(runner))
        b['voc_gru_connected']=True
        b['resume_semantics']='optimizers, schedules, EMA, BC and RNG restored; environment/planner/replay cold'
        atomic_json(bundle,b)
        atomic_json(latest_path,dict(bundle=str(bundle),env_steps=steps,update=update))
        last_save=time.monotonic();last_bundle=bundle
        return bundle
    try:
        status('INITIALIZING')
        worker=SelfPlayWorker.__ray_metadata__.modified_class(
            {'actor_buffer':None,'actor_param_buffer':None,'model_param_buffer':None},None,0,flags.env_n,flags)
        actor=worker.actor_learner;model=worker.env.model_learner
        if actor.real_step!=steps or model.real_step!=steps:raise AssertionError('Restored counters differ')
        adapter=SelectiveVOCGradient(actor,True)
        telemetry=TrainingMetrics(worker.actor_net,flags,args.output,restored['telemetry'] if restored else None)
        attach_logging(actor,run,args.output,'actor');attach_logging(model,run,args.output,'model')
        # Observe the actual BC batch used by production compute_losses.
        original_compute=actor.compute_losses
        latest_bc=0
        def capture_bc(*a,**kw):
            nonlocal latest_bc
            result=original_compute(*a,**kw)
            latest_bc=int(float(result[0].get('icopro_count',0)))
            return result
        actor.compute_losses=capture_bc
        atomic_json(attempts/f'{attempt:04d}.json',dict(start_steps=steps,job_id=os.getenv('SLURM_JOB_ID'),
            wandb_url=run.url,resume=resume,recipe=spec,
            resume_bundle=str(last_bundle or (json.loads(latest_path.read_text())['bundle'] if resume else ''))))
        if restored: set_rng(restored['rng'])
        torch.cuda.reset_peak_memory_stats()
        if steps==0:
            queue_evaluation(args,checkpoint(),steps)
        with torch.no_grad():
            state,info=worker.env.reset();env=worker.init_env_out(state,info)
            core=worker.actor_net.initial_state(batch_size=flags.env_n,device=worker.device)
            previous=env
            out,core,env,info=worker.env_step(env,core)
            if bool(info['real_transition'].any()):raise AssertionError('Uncounted initial real action')
            telemetry.observe(previous,out,info,real_step=steps,attempt=attempt)
        milestones=sorted({x for x in MILESTONES if steps<x<=stop_at}|{stop_at})
        status('TRAINING')
        while steps<stop_at and not interrupted and time.monotonic()-started<args.max_seconds:
            target=min(milestones)
            initial_core=util.tuple_map(core,lambda x:x.detach())
            worker.write_actor_buffer(env,out,0)
            with torch.no_grad():
                for t in range(flags.actor_unroll_len):
                    previous=env
                    out,core,env,info=worker.env_step(env,core)
                    telemetry.observe(previous,out,info,real_step=steps,attempt=attempt)
                    worker.write_actor_buffer(env,out,t+1)
                    steps+=int(info['real_transition'].sum())
                    if steps==target:break
                    if steps>target:raise AssertionError('Overshot action budget')
            if t+1<flags.actor_unroll_len:
                worker.actor_local_buffer=util.tuple_map(worker.actor_local_buffer,lambda v:torch.stack(v,dim=0))
            worker.actor_net.train(True)
            actor.consume_data((worker.actor_local_buffer,initial_core))
            worker.actor_net.train(False);update+=1
            if actor.real_step!=steps:raise AssertionError('Actor real-step mismatch')
            actor._assert_voc_gate_exact_projection_invariant()
            committed=bool(adapter.last_metrics['actor_optimizer_stepped'])
            counters['bc_targets_attempted']+=latest_bc
            counters['bc_targets_committed']+=latest_bc if committed else 0
            counters['actor_updates_committed' if committed else 'actor_updates_skipped']+=1
            if update%10==0 or steps==target:
                telemetry.flush()
                last_metrics={**telemetry.metrics(),**counters,
                    'actor_updates':update,'voc_updates':int(actor.voc_update_count),
                    'bc_updates':int(actor.imitation_update_count),
                    'model_updates':int(model._model_grad_step_count_p)}
                resources=dict(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    allocated_gib=torch.cuda.memory_allocated()/2**30,
                    device_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
                last_resources=resources
                run.log(dict(real_step=steps,**{'train/'+k:v for k,v in last_metrics.items() if v is not None},
                    **{'gradient/'+k:v for k,v in adapter.last_metrics.items()},
                    **{'resources/'+k:v for k,v in resources.items()}))
                with (args.output/'training_metrics.jsonl').open('a') as f:
                    f.write(json.dumps(dict(real_step=steps,attempt=attempt,**last_metrics,
                        gradient=adapter.last_metrics,resources=resources),allow_nan=False)+'\n')
                status('TRAINING')
                print(json.dumps(dict(real_step=steps,episode_return=last_metrics['mean_last_100_episode_return'],
                    computations=last_metrics['computations_per_action'],**resources)),flush=True)
            if steps==target:
                queue_evaluation(args,checkpoint(),steps)
                milestones.remove(target)
            elif (steps//args.snapshot_every > initial_steps//args.snapshot_every and
                  (last_bundle is None or steps-json.loads(last_bundle.read_text())['actor_real_step']>=args.snapshot_every)) or time.monotonic()-last_save>1800:
                checkpoint()
        checkpoint()
        if int(actor.imitation_update_count)==0 or int(model._model_grad_step_count_p)==0:
            raise AssertionError('No actual BC/model learning')
        final_phase='COMPLETE' if steps==args.env_steps else 'PAUSED'
        status(final_phase)
        run.summary['phase']=final_phase;run.finish(exit_code=0)
    except BaseException as exc:
        status('FAILED',error=f'{type(exc).__name__}: {exc}')
        run.summary['phase']='FAILED';run.finish(exit_code=1);raise
    finally:
        if telemetry:telemetry.close()
        if adapter:adapter.close()
        if worker:
            worker.actor_learner.close(successful=phase in ('COMPLETE','PAUSED'))
            worker.env.model_learner.close(successful=phase in ('COMPLETE','PAUSED'))
            worker.env.close()


if __name__=='__main__':main()
