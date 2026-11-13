"""Matched 100k continuation: detached VOC versus selective VOC -> tree GRU."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, subprocess, sys, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from train_pilot import (ROOT,util,torch,np,yaml,wrapper,file_writer,SelfPlayWorker,
                         freeze_runtime,save_bundle,atomic_json,sha)
from train_reward_learning import attach_logging
from reward_probe import load_checkpoints,digest
from voc_gru_gradient import SelectiveVOCGradient


def branch_weights(worker,actor_ck,model_ck,binding):
    a=worker.actor_learner;m=worker.env.model_learner
    assert a.actor_net is worker.actor_net and m.model_net is worker.env.model_net
    for opt in [a.optimizer,a.voc_optimizer,m.optimizer_p,m.optimizer_m]:
        assert not opt.state, 'Branch optimizers must start fresh'
    worker.actor_net.set_weights(actor_ck['actor_net_state_dict'],strict=True)
    worker.env.model_net.set_weights(model_ck['model_net_state_dict'])
    # Projection counters describe the lifetime of the inherited teacher/gate.
    # Zeroing them falsely labels a trained gate as zero-initialized. Preserve
    # those counters and report branch-local increments separately.
    a.voc_update_count=int(actor_ck['voc_update_count'])
    a.voc_gate_update_count=int(actor_ck['voc_gate_update_count'])
    a._load_voc_ema_gate_state(actor_ck['voc_ema_gate_head_state_dict'],
        int(actor_ck['voc_ema_gate_update_count']),parent_update_count=0)
    if actor_ck.get('action_prior_ema') is not None:
        a.action_prior_ema=actor_ck['action_prior_ema'].to(worker.device).clone()
    a._assert_voc_gate_exact_projection_invariant()
    assert a.real_step==m.real_step==0
    assert digest(worker.actor_net.state_dict())==digest(actor_ck['actor_net_state_dict'])
    assert digest(worker.env.model_net.state_dict())==digest(model_ck['model_net_state_dict'])
    if a.imitation_data_signature != actor_ck['imitation_data_signature']:
        raise AssertionError('Branch human training data identity differs')
    return dict(actor=digest(worker.actor_net.state_dict()),model=digest(worker.env.model_net.state_dict()),
        ema=digest(a._voc_ema_gate_state_dict()),parent=binding,
        policy='weights/EMA/action-prior inherited; optimizers, schedules, replay, environment and recurrent state reset',
        parent_env_steps=binding['actor_real_step'],additional_env_steps=0,
        parent_voc_updates=a.voc_update_count,parent_ema_updates=a.voc_ema_gate_update_count)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--mode',choices=['detached','gru'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--env-steps',type=int,default=100000)
    p.add_argument('--seed',type=int,default=90211)
    p.add_argument('--warmup',type=int,default=1024)
    p.add_argument('--milestones',type=int,nargs='+',default=[0,25000,50000,100000])
    p.add_argument('--no-evaluation',action='store_true')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args();args.bundle=args.bundle.resolve();args.output=args.output.resolve()
    args.checkpoint=None;args.threads=1
    if args.env_steps%8 or args.env_steps<=args.warmup or any(s%8 or s<0 or s>args.env_steps for s in args.milestones):
        p.error('Budgets/milestones must be multiples of 8; total exceeds warmup')
    if args.env_steps not in args.milestones:p.error('Final budget must be a milestone')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parent_flags,actor_ck,model_ck,binding=load_checkpoints(args)
    if not parent_flags.voc_gate_exact_projection:raise ValueError('Expected projected-gate parent')
    raw=vars(parent_flags).copy()
    raw.pop('config',None)
    raw.update(xpid='train',savedir=str(args.output),ckp=False,preload='',preload_actor='',voc_parent_checkpoint='',
        total_steps=args.env_steps,base_seed=args.seed,model_warm_up_n=args.warmup,
        icopro_data_path=os.getenv('VOC_REWARD_DATA_ROOT',parent_flags.icopro_data_path))
    flags=util.create_flags(['default_thinker.yaml','default_actor.yaml'],save_flags=True,
                            post_fn=util.process_flags_actor,**raw)
    flags.voc_gru_ablation_mode=args.mode
    flags.voc_gru_ablation_parent_bundle=str(args.bundle)
    flags.voc_gru_ablation_parent_sha256=sha(args.bundle)
    flags.voc_gru_ablation_parent_env_steps=int(binding['actor_real_step'])
    flags.voc_gru_ablation_gradient_method='feature-leaf VJP to tree GRU; unscaled merge before actor clipping'
    runtime=freeze_runtime(args.output)
    import wandb
    run=wandb.init(project='thinker',entity=os.getenv('WANDB_USER','jmme425'),mode='online',
        group='voc-gru-detach-ablation-20260925',name=f'{flags.name}-{args.mode}-s{args.seed}',
        job_type='continuation-smoke' if args.no_evaluation else 'matched-continuation',
        dir=str(args.output),config=dict(training=vars(flags),parent=binding,
            additional_env_steps=args.env_steps,reset_optimizer=True,mode=args.mode))
    if run.settings.mode!='online':raise RuntimeError('Online W&B required')
    run.define_metric('real_step');run.define_metric('actor/*',step_metric='real_step')
    run.define_metric('model/*',step_metric='real_step');run.define_metric('gradient/*',step_metric='real_step')
    run.define_metric('eval/env_steps');run.define_metric('eval/*',step_metric='eval/env_steps')
    gather=file_writer.gather_metadata
    def metadata():
        v=gather();v['env']={k:os.environ[k] for k in ['SLURM_JOB_ID','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS'] if k in os.environ};return v
    file_writer.gather_metadata=metadata
    def seeded_env(name,f,env_n=1):
        import envpool
        env=envpool.make_gymnasium(name,num_envs=env_n,batch_size=env_n,num_threads=1,seed=args.seed,
            gray_scale=f.grayscale,stack_num=f.frame_stack_n,episodic_life=True)
        return wrapper.EnvPoolWrap(env,env_n,frame_stack_n=f.frame_stack_n)
    wrapper.create_envpool=seeded_env
    worker=None;adapter=None;steps=0;update=0;started=time.monotonic();eval_seconds=0.;milestones=[]
    def status(phase,**extra):
        atomic_json(args.output/'status.json',dict(phase=phase,game=flags.name,mode=args.mode,
            env_steps=steps,total_env_steps=args.env_steps,parent_env_steps=binding['actor_real_step'],
            actor_updates=update,elapsed_seconds=time.monotonic()-started,evaluation_seconds=eval_seconds,
            train_env_steps_per_second=steps/max(time.monotonic()-started-eval_seconds,1),
            wandb_url=run.url,job_id=os.getenv('SLURM_JOB_ID'),milestones=milestones,**extra))
    def checkpoint():
        bundle,b=save_bundle(worker,args.output,update,0)
        b['branch_provenance']=dict(parent_bundle=str(args.bundle),parent_bundle_sha256=sha(args.bundle),
            parent_env_steps=binding['actor_real_step'],additional_env_steps=steps,mode=args.mode,
            gradient_method=flags.voc_gru_ablation_gradient_method,
            additional_voc_updates=int(worker.actor_learner.voc_update_count)-binding['voc_updates'],
            additional_ema_updates=int(worker.actor_learner.voc_ema_gate_update_count)-binding['ema_updates'])
        atomic_json(bundle,b)
        return bundle
    def evaluate_at():
        nonlocal eval_seconds
        bundle=checkpoint()
        milestones.append(dict(env_steps=steps,bundle=str(bundle)))
        atomic_json(args.output/'milestones.json',milestones)
        if args.no_evaluation:return
        status('EVALUATING',evaluation_step=steps)
        command=[sys.executable,str(Path(__file__).with_name('gru_ablation_eval.py')),
            '--bundle',str(bundle),'--output',str(bundle.parent/'evaluation'),'--device',args.device]
        if steps<args.env_steps:command+=['--score-only']
        torch.cuda.empty_cache();begin=time.monotonic()
        with (bundle.parent/'evaluation.log').open('w') as log:
            subprocess.run(command,env=dict(os.environ,VOC_DIAGNOSTIC_RUNTIME=str(runtime)),
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        eval_seconds+=time.monotonic()-begin
        result=json.loads((bundle.parent/'evaluation/summary.json').read_text())
        run.log({'eval/env_steps':steps,**{f'eval/{k}':v for k,v in result['metrics'].items() if v is not None}})
        milestones[-1]['evaluation']=str(bundle.parent/'evaluation/summary.json')
        atomic_json(args.output/'milestones.json',milestones)
    try:
        status('INITIALIZING')
        worker=SelfPlayWorker.__ray_metadata__.modified_class(
            {'actor_buffer':None,'actor_param_buffer':None,'model_param_buffer':None},None,0,flags.env_n,flags)
        actor=worker.actor_learner;model=worker.env.model_learner
        initialization=branch_weights(worker,actor_ck,model_ck,binding)
        atomic_json(args.output/'initialization.json',initialization)
        del actor_ck,model_ck
        adapter=SelectiveVOCGradient(actor,args.mode=='gru')
        attach_logging(actor,run,args.output,'actor');attach_logging(model,run,args.output,'model')
        if 0 in args.milestones:evaluate_at()
        with torch.no_grad():
            state,info=worker.env.reset();env=worker.init_env_out(state,info)
            core=worker.actor_net.initial_state(batch_size=flags.env_n,device=worker.device)
            out,core,env,info=worker.env_step(env,core)
            if bool(info['real_transition'].any()):raise AssertionError('Uncounted initial environment action')
        next_milestone=min(s for s in args.milestones if s>0)
        status('TRAINING')
        while steps<args.env_steps:
            initial_core=util.tuple_map(core,lambda x:x.detach())
            worker.write_actor_buffer(env,out,0)
            with torch.no_grad():
                for t in range(flags.actor_unroll_len):
                    out,core,env,info=worker.env_step(env,core)
                    worker.write_actor_buffer(env,out,t+1)
                    steps+=int(info['real_transition'].sum())
                    mask=info['real_done'].bool() & info['real_transition'].bool()
                    if mask.any():
                        with (args.output/'episodes.jsonl').open('a') as log:
                            for value in info['episode_return'][mask].reshape(-1).cpu().tolist():
                                log.write(json.dumps(dict(env_steps=steps,return_raw=value))+'\n')
                    if steps==next_milestone:break
                    if steps>next_milestone:raise AssertionError('Overshot environment budget')
            if t+1<flags.actor_unroll_len:
                worker.actor_local_buffer=util.tuple_map(worker.actor_local_buffer,lambda v:torch.stack(v,dim=0))
            worker.actor_net.train(True)
            actor.consume_data((worker.actor_local_buffer,initial_core))
            worker.actor_net.train(False);update+=1
            assert int(actor.real_step)==steps
            actor._assert_voc_gate_exact_projection_invariant()
            metrics=dict(adapter.last_metrics,voc_updates=int(actor.voc_update_count)-binding['voc_updates'],
                imitation_updates=int(actor.imitation_update_count),model_updates=int(model._model_grad_step_count_p))
            run.log({'real_step':steps,**{f'gradient/{k}':v for k,v in metrics.items()}})
            with (args.output/'gradient_metrics.jsonl').open('a') as log:
                log.write(json.dumps(dict(env_steps=steps,update=update,**metrics),allow_nan=False)+'\n')
            status('TRAINING',gradient=metrics)
            if update%20==0:print(json.dumps(dict(env_steps=steps,mode=args.mode,**metrics)),flush=True)
            if steps==next_milestone:
                evaluate_at()
                next_milestone=min([s for s in args.milestones if s>steps]+[args.env_steps])
                status('TRAINING')
        assert model.real_step==steps and actor.imitation_update_count>0 and model._model_grad_step_count_p>0
        run.summary['experiment_status']='COMPLETE';status('COMPLETE');run.finish(exit_code=0)
    except BaseException as exc:
        status('FAILED',error=f'{type(exc).__name__}: {exc}')
        run.summary['experiment_status']='FAILED';run.finish(exit_code=1);raise
    finally:
        if adapter is not None:adapter.close()
        if worker is not None:
            worker.actor_learner.close(successful=steps==args.env_steps)
            worker.env.model_learner.plogger.close(successful=steps==args.env_steps)
            worker.env.close()


if __name__=='__main__':main()
