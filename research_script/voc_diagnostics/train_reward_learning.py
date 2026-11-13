"""Fixed-environment-budget VOC learning experiment (fresh control/projection).

One subject, one training seed per game: an exploratory learning pilot, not a
confirmatory population comparison. Milestones are fixed before execution.
"""
from __future__ import annotations
import argparse, json, os, sys, time, subprocess, hashlib
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from train_pilot import (ROOT, util, torch, np, yaml, wrapper, file_writer, SelfPlayWorker,
                         freeze_runtime, save_bundle, atomic_json, scalar_stats)


def attach_logging(learner, run, output, kind):
    original=learner.plogger.log
    def log(stats,*args,**kwargs):
        original(stats,*args,**kwargs)
        row=scalar_stats(stats)
        row['real_step']=int(learner.real_step)
        with (output/f'{kind}_metrics.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')
        run.log({'real_step':row['real_step'],**{(k if k.startswith(kind+'/') else kind+'/'+k):v for k,v in row.items() if v is not None}})
    learner.plogger.log=log


def flags_for(args):
    config=ROOT/'logs/thinker'/f'dynamic-imitation-{args.game}-voc-sub1-leavesession1/config_c.yaml'
    raw=yaml.safe_load(config.read_text());raw.pop('voc_gate_policy_schema_version',None);raw.pop('config',None)
    raw.update(xpid='train',savedir=str(args.output),ckp=False,preload='',preload_actor='',voc_parent_checkpoint='',
        parallel=False,parallel_actor=False,env_n=8,actor_batch_size=8,self_play_n=1,actor_unroll_len=64,
        model_batch_size=8,model_unroll_len=10,model_return_n=5,model_warm_up_n=args.warmup,
        model_buffer_n=100_000,total_steps=args.env_steps,checkpoint_interval=0,
        min_replay_ratio=4.,max_replay_ratio=4.,base_seed=args.seed,
        icopro_data_path=os.environ.get('VOC_REWARD_DATA_ROOT',str(ROOT/'behavioral_data_block')),
        icopro_batch_size=8,icopro_device='cuda',gpu_self_play=1.,use_wandb=True,
        voc_gate_exact_projection=args.mode=='projection',voc_gate_param_align=False,
        voc_gate_epsilon_greedy_execution=False,model_sr_dynamics_norm='group')
    return util.create_flags(['default_thinker.yaml','default_actor.yaml'],save_flags=True,
        post_fn=util.process_flags_actor,**raw)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game',choices=['pong','enduro'],required=True)
    p.add_argument('--mode',choices=['control','projection'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--env-steps',type=int,default=100_000)
    p.add_argument('--seed',type=int,default=701);p.add_argument('--warmup',type=int,default=1024)
    p.add_argument('--milestones',type=int,nargs='+',default=[0,10000,25000,50000,100000])
    p.add_argument('--paired-milestones',type=int,nargs='+',default=[25000,100000])
    p.add_argument('--score-horizon',type=int,default=4096);p.add_argument('--voc-horizon',type=int,default=1536)
    p.add_argument('--eval-batch',type=int,default=8);p.add_argument('--eval-repetitions',type=int,default=3)
    p.add_argument('--eval-prefixes',type=int,nargs='+',default=[128,1024])
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args();args.output=args.output.resolve()
    if args.env_steps%8 or args.env_steps<=args.warmup or any(x%8 or x<0 or x>args.env_steps for x in args.milestones):
        p.error('Budgets/milestones must be multiples of 8 within the total budget, beyond model warmup')
    if not set(args.paired_milestones)<=set(args.milestones):p.error('Paired milestones must be score milestones')
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    flags=flags_for(args);runtime=freeze_runtime(args.output)
    import wandb
    run=wandb.init(project='thinker',entity=os.getenv('WANDB_USER','jmme425'),mode='online',
        name=f'voc-reward-{args.game}-{args.mode}-s{args.seed}',group='voc-reward-learning-20260924',
        tags=['voc','reward-learning-pilot',args.game,args.mode],dir=str(args.output),
        config=dict(training=vars(flags),protocol={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
    if run.settings.mode!='online':raise RuntimeError('Online W&B required')
    run.define_metric('train/env_steps')
    run.define_metric('learning/*',step_metric='train/env_steps')
    run.define_metric('real_step')
    run.define_metric('actor/*',step_metric='real_step')
    run.define_metric('model/*',step_metric='real_step')
    run.define_metric('eval/*',step_metric='eval/env_steps');run.define_metric('eval/env_steps')
    gather=file_writer.gather_metadata
    def metadata():
        v=gather();v['env']={k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','SLURM_JOB_ID','OMP_NUM_THREADS') if k in os.environ};return v
    file_writer.gather_metadata=metadata
    def seeded_env(name,f,env_n=1):
        import envpool
        raw=envpool.make_gymnasium(name,num_envs=env_n,batch_size=env_n,num_threads=1,seed=args.seed,
            gray_scale=f.grayscale,stack_num=f.frame_stack_n,episodic_life=True)
        return wrapper.EnvPoolWrap(raw,env_n,frame_stack_n=f.frame_stack_n)
    wrapper.create_envpool=seeded_env
    worker=None;started=time.monotonic();update=0;real_steps=0;completed=[];milestones=[];eval_seconds=0.
    protocol={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    def status(phase,**extra):
        elapsed=time.monotonic()-started
        payload=dict(phase=phase,env_steps=real_steps,actor_updates=update,total_env_steps=args.env_steps,
            elapsed_seconds=elapsed,evaluation_seconds=eval_seconds,
            train_env_steps_per_second=real_steps/max(elapsed-eval_seconds,1e-9),
            wandb_url=run.url,wandb_id=run.id,milestones=milestones,protocol=protocol,**extra)
        atomic_json(args.output/'status.json',payload)
    def evaluate_at(step):
        nonlocal eval_seconds
        bundle,binding=save_bundle(worker,args.output,update,0)
        status('EVALUATING',evaluation_step=step)
        command=[sys.executable,str(Path(__file__).with_name('reward_learning_eval.py')),
            '--bundle',str(bundle),'--output',str(bundle.parent/'evaluation'),'--batch-size',str(args.eval_batch),
            '--score-horizon',str(args.score_horizon),'--horizon',str(args.voc_horizon),
            '--repetitions',str(args.eval_repetitions),'--prefixes',*[str(x) for x in args.eval_prefixes],
            '--device',args.device]
        if step not in args.paired_milestones:command+=['--score-only']
        begin=time.monotonic()
        # Release unused training allocator blocks before the frozen evaluator
        # shares this GPU. Live training parameters and optimizer state remain.
        torch.cuda.empty_cache()
        with (bundle.parent/'evaluation.log').open('w') as log:
            subprocess.run(command,env=dict(os.environ,VOC_DIAGNOSTIC_RUNTIME=str(runtime)),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        eval_seconds+=time.monotonic()-begin
        result=json.loads((bundle.parent/'evaluation/summary.json').read_text())
        scalar={k:v for k,v in result.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
        scalar.update({'voc_'+k:v for k,v in result.get('voc',{}).items() if isinstance(v,(int,float))})
        run.log({'eval/env_steps':step,**{'eval/'+k:v for k,v in scalar.items()}})
        milestones.append(dict(env_steps=step,bundle=str(bundle),summary=str(bundle.parent/'evaluation/summary.json')))
        atomic_json(args.output/'milestones.json',milestones)
        status('TRAINING')
        subprocess.run([sys.executable,str(Path(__file__).with_name('report_reward_learning.py')),
            '--root',str(args.output.parent)],check=True)
    try:
        worker=SelfPlayWorker.__ray_metadata__.modified_class(
            {'actor_buffer':None,'actor_param_buffer':None,'model_param_buffer':None},None,0,flags.env_n,flags)
        actor,model=worker.actor_learner,worker.env.model_learner
        attach_logging(actor,run,args.output,'actor');attach_logging(model,run,args.output,'model')
        initial={name:hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in network.state_dict().values())).hexdigest()
            for name,network in [('actor',worker.actor_net),('model',worker.env.model_net)]}
        atomic_json(args.output/'initialization.json',dict(initial=initial,wandb_url=run.url,protocol=protocol))
        if 0 in args.milestones:evaluate_at(0)
        with torch.no_grad():
            state,info=worker.env.reset()
            env=worker.init_env_out(state,info);core=worker.actor_net.initial_state(batch_size=flags.env_n,device=worker.device)
            out,core,env,info=worker.env_step(env,core)
            if bool(info['real_transition'].any()):raise AssertionError('Uncounted initial real transition')
        next_milestone=min([s for s in args.milestones if s>0]+[args.env_steps])
        while real_steps<args.env_steps:
            initial_core=util.tuple_map(core,lambda x:x.detach())
            worker.write_actor_buffer(env,out,0)
            with torch.no_grad():
                for t in range(flags.actor_unroll_len):
                    out,core,env,info=worker.env_step(env,core)
                    worker.write_actor_buffer(env,out,t+1)
                    real_steps+=int(info['real_transition'].sum())
                    if bool((info['real_done'].bool() & info['real_transition'].bool()).any()):
                        mask=info['real_done'].bool() & info['real_transition'].bool()
                        for value in info['episode_return'][mask].reshape(-1).cpu().tolist():
                            completed.append(dict(env_steps=real_steps,return_raw=value))
                            with (args.output/'episodes.jsonl').open('a') as f:f.write(json.dumps(completed[-1])+'\n')
                    if real_steps==next_milestone:break
                    if real_steps>next_milestone:raise AssertionError('Overshot fixed environment budget')
            if t+1<flags.actor_unroll_len:
                worker.actor_local_buffer=util.tuple_map(worker.actor_local_buffer,lambda v:torch.stack(v,dim=0))
            worker.actor_net.train(True)
            actor.consume_data((worker.actor_local_buffer,initial_core));worker.actor_net.train(False);update+=1
            if int(actor.real_step)!=real_steps:raise AssertionError('Environment/learner count mismatch')
            if args.mode=='projection':actor._assert_voc_gate_exact_projection_invariant()
            metrics=dict(voc_updates=int(actor.voc_update_count),ema_updates=int(actor.voc_ema_gate_update_count),
                imitation_updates=int(actor.imitation_update_count),model_updates=int(model._model_grad_step_count_p),
                completed_episodes=len(completed))
            if completed:metrics['last_20_episode_return_mean']=float(np.mean([r['return_raw'] for r in completed[-20:]]))
            run.log({'train/env_steps':real_steps,**{'learning/'+k:v for k,v in metrics.items()}})
            status('TRAINING',metrics=metrics)
            if real_steps==next_milestone:
                evaluate_at(real_steps)
                next_milestone=min([s for s in args.milestones if s>real_steps]+[args.env_steps])
        if model.real_step!=real_steps or actor.imitation_update_count<1 or model._model_grad_step_count_p<1:
            raise AssertionError('Incomplete learner/model progress')
        run.summary['experiment_status']='COMPLETE';status('COMPLETE');run.finish(exit_code=0)
    except BaseException as error:
        status('FAILED',error=f'{type(error).__name__}: {error}')
        run.summary['experiment_status']='FAILED';run.finish(exit_code=1);raise
    finally:
        if worker is not None:
            worker.actor_learner.close(successful=real_steps==args.env_steps)
            worker.env.model_learner.plogger.close(successful=real_steps==args.env_steps)
            worker.env.close()
        subprocess.run([sys.executable,str(Path(__file__).with_name('report_reward_learning.py')),
            '--root',str(args.output.parent)],check=False)

if __name__=='__main__':main()
