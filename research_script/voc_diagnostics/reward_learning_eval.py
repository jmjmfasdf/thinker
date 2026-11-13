"""Batched, independent score and paired reward evaluation of learning VOC.

Continuation randomness is keyed by environment stream, real decision, search
step and phase. Fresh seeded EnvPool/planner reconstruction pairs both arms.
Neither observed human actions nor bootstrapped tails are efficacy targets.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from scipy.stats import spearmanr
from reward_probe import (ReturnAccumulator, keyed_seed, seed_policy, preserve_tokens,
                         load_checkpoints, digest, util, ActorNet, ModelNet,
                         cModelWrapper, wrapper, _VoCFeatureObserver, _voc_trace_diagnostics)


def categorical(logits, uniforms):
    probability=torch.softmax(logits.detach().double(),-1).cpu().numpy()
    if not np.isfinite(probability).all(): raise ValueError('Nonfinite action probability')
    return np.minimum((uniforms[:,None] > np.cumsum(probability,axis=-1)).sum(-1),probability.shape[-1]-1)


def tail_bound(gamma, horizon, clip, terminal=False):
    if terminal: return 0.
    if not 0 < gamma < 1 or clip <= 0: raise ValueError('Bound needs discounted bounded rewards')
    return gamma**horizon*clip/(1-gamma)


class BatchRuntime:
    def __init__(self, flags, actor_checkpoint, model_checkpoint, seed, batch_size, device):
        import envpool
        self.flags=copy.deepcopy(flags); f=self.flags
        f.train_actor=False;f.train_model=False;f.parallel=False;f.parallel_actor=False
        f.env_n=batch_size;f.ckp=False;f.use_wandb=False
        self.device=torch.device(device);self.batch_size=batch_size
        seed_policy(seed)
        raw=envpool.make_gymnasium(f.name,num_envs=batch_size,batch_size=batch_size,num_threads=1,
            seed=seed,gray_scale=f.grayscale,stack_num=f.frame_stack_n,episodic_life=True)
        base=wrapper.VectorWrap(wrapper.EnvPoolWrap(raw,batch_size,frame_stack_n=f.frame_stack_n),f)
        self.model=ModelNet(obs_space=base.get_wrapper_attr('single_observation_space'),
            action_space=base.get_wrapper_attr('single_action_space'),flags=f,frame_stack_n=f.frame_stack_n).to(self.device)
        self.model.set_weights(model_checkpoint['model_net_state_dict']);self.model.eval();self.model.requires_grad_(False)
        self.planner=cModelWrapper(env=base,env_n=batch_size,flags=f,model_net=self.model,device=self.device,timing=False)
        self.env=wrapper.PostWrapper(self.planner,f)
        self.actor=ActorNet(obs_space=self.env.observation_space,action_space=self.env.action_space,flags=f,
            tree_rep_meaning=util.get_tree_rep_meaning(int(raw.action_space.n),1,f)).to(self.device)
        self.actor.set_weights(actor_checkpoint['actor_net_state_dict'],strict=True)
        self.actor.eval();self.actor.requires_grad_(False);self.actor.train_actor_enabled=True
        self.ema=actor_checkpoint['voc_ema_gate_head_state_dict']
        if f.think_cost_anneal or self.actor.voc_gate_epsilon_greedy_execution:
            raise ValueError('Protocol requires constant cost and soft stochastic gate')
    def close(self): self.env.close()


def rollout(flags, actor_ck, model_ck, *, seed, prefix, repetition, horizon, arm,
            batch_size=8, device='cuda:0'):
    if arm not in ('policy','stop','continue'): raise ValueError(arm)
    rt=BatchRuntime(flags,actor_ck,model_ck,seed,batch_size,device)
    before=digest(dict(actor=rt.actor.state_dict(),model=rt.model.state_dict()))
    acc=[ReturnAccumulator(float(flags.discounting),float(flags.think_cost)) for _ in range(batch_size)]
    prefix_hash=hashlib.sha256();trace=hashlib.sha256();entry=None;real_count=0
    raw_window=np.zeros(batch_size);last_episode_return=np.zeros(batch_size)
    completed=[[] for _ in range(batch_size)];compute=np.zeros(batch_size,dtype=np.int64)
    stage_compute=np.zeros(batch_size,dtype=np.int64)
    started=time.monotonic()
    try:
        state,info=rt.env.reset(rt.model)
        env=util.init_env_out(state,info,rt.flags,rt.actor.dim_actions,rt.actor.tuple_action)
        core=rt.actor.initial_state(batch_size=batch_size,device=rt.device)
        with torch.no_grad(),_VoCFeatureObserver(rt.actor) as observer:
            for call in range((prefix+horizon+1)*(int(flags.max_search_steps)+4)):
                phase=env.phase[0].cpu().numpy();depth=env.search_steps[0].cpu().numpy()
                is_entry=entry is None and real_count==prefix
                if is_entry:
                    if np.any(phase!=0) or np.any(depth!=0): raise AssertionError('Not root boundary')
                    entry=dict(prefix_sha256=prefix_hash.hexdigest(),entry_sha256=digest(dict(env=env,core=core)),
                        tree_sha256=digest(rt.planner.dynamic_trace_snapshot(include_xs=True,include_hs=True)))
                observer.begin_row()
                out,core=rt.actor(env,core,compute_loss=False,greedy=False)
                features=observer.take(out.voc_q)
                if is_entry:
                    diag=_voc_trace_diagnostics(out,features,flags=rt.flags,ema_head_state=rt.ema,
                        effective_think_cost=float(flags.think_cost))
                    q=diag['ema_q'].reshape(batch_size,2);oq=diag['online_q'].reshape(batch_size,2)
                    entry.update(ema_delta=(q[:,0]-q[:,1]).cpu().tolist(),online_delta=(oq[:,0]-oq[:,1]).cpu().tolist(),
                        continue_probability=diag['soft_continue_probability'].reshape(-1).cpu().tolist())
                rep=repetition if real_count>=prefix else 0
                draws=np.array([[keyed_seed(seed+i,real_count,int(depth[i]),int(phase[i])+j*100,rep)/2**32
                    for i in range(batch_size)] for j in range(3)])
                primary=categorical(out.pri_param[-1].reshape(batch_size,-1),draws[0])
                control=categorical(out.search_control_logits[-1],draws[1])
                if is_entry and arm!='policy':
                    if not bool(out.control_valid[-1].all()): raise AssertionError('Invalid intervention')
                    control[:]=util.STOP if arm=='stop' else categorical(out.search_control_logits[-1,:,:2],draws[2])
                primary=torch.as_tensor(primary,device=rt.device).reshape(out.action[0].shape)
                control=torch.as_tensor(control,device=rt.device).reshape(out.action[1].shape)
                action=(primary,control)
                state,reward,done,truncated,info=rt.env.step(action,rt.model)
                new_env=preserve_tokens(env,util.create_env_out(action,state,reward,done,truncated,info,rt.flags),info)
                real=info['real_transition'].bool().cpu().numpy()
                if real.any() and not real.all(): raise AssertionError('Expected synchronous real transition')
                re=bool(real.all());terminal=(done|truncated.bool()).cpu().numpy().reshape(-1)
                think=info['think_reward'].cpu().numpy().reshape(-1)
                task=reward.cpu().numpy().reshape(-1)
                stage=info['stage_end'].cpu().numpy().reshape(-1)
                ep=info['episode_return'].cpu().numpy().reshape(-1)
                real_done=info['real_done'].cpu().numpy().reshape(-1)
                h=trace if entry is not None else prefix_hash
                h.update(digest(dict(action=action,reward=reward,done=done,truncated=truncated,
                    control=info['accepted_control'],real=info['real_transition'])).encode())
                if re: h.update(digest(state['real_states']).encode())
                if entry is not None:
                    compute+=(think==-1)
                    for i,a in enumerate(acc):
                        if not a.terminal:
                            a.observe(reward=float(task[i]) if re else 0.,think_reward=float(think[i]),
                                real_transition=re,stage_end=bool(stage[i]),done=bool(terminal[i]))
                    if re:
                        raw_window+=ep-last_episode_return
                        for i in np.flatnonzero(real_done): completed[i].append(float(ep[i]))
                if re:
                    last_episode_return=ep.copy();last_episode_return[real_done]=0.
                    real_count+=1
                env=new_env
                if entry is not None and (real_count-prefix>=horizon or (arm!='policy' and all(a.terminal for a in acc))):
                    break
            else: raise RuntimeError('Augmented-call watchdog')
        if entry is None: raise AssertionError('No intervention state')
        if arm=='stop' and any(a.stage_computations!=0 for a in acc): raise AssertionError('STOP charged')
        if arm=='continue' and any(a.stage_computations<1 for a in acc): raise AssertionError('CONTINUE not executed')
        if digest(dict(actor=rt.actor.state_dict(),model=rt.model.state_dict()))!=before:
            raise AssertionError('Evaluation changed parameters')
        return dict(entry=entry,returns=[a.scores() for a in acc],trace_sha256=trace.hexdigest(),
            seed=seed,prefix=prefix,repetition=repetition,horizon=horizon,arm=arm,batch_size=batch_size,
            real_steps_per_stream=real_count-prefix,raw_window=raw_window.tolist(),
            completed_episode_returns=completed,computations=compute.tolist(),seconds=time.monotonic()-started)
    finally: rt.close()


def verify_pair(a,b,repeat=None):
    for key in ('prefix_sha256','entry_sha256','tree_sha256'):
        if a['entry'][key]!=b['entry'][key]: raise AssertionError('Pair mismatch: '+key)
        if repeat is not None and a['entry'][key]!=repeat['entry'][key]: raise AssertionError('Repeat mismatch')
    if repeat is not None and (a['trace_sha256']!=repeat['trace_sha256'] or a['returns']!=repeat['returns']):
        raise AssertionError('STOP repeat differs')


def aggregate_pairs(rows, bootstrap=2000):
    # Repetitions average within state; uncertainty clusters by EnvPool stream,
    # retaining all prefixes and repetitions from that stream in each sample.
    states={}
    for r in rows: states.setdefault((r['seed'],r['stream'],r['prefix']),[]).append(r)
    values=[]
    for key,rr in states.items():
        values.append(dict(seed=key[0],stream=key[1],prefix=key[2],
            delta=float(np.mean([r['delta'] for r in rr])),pred=float(np.mean([r['ema_delta'] for r in rr])),
            p=float(np.mean([r['p_continue'] for r in rr])),cost=float(np.mean([r['continue_stage_cost'] for r in rr])),
            tail=float(max(r['tail_bound'] for r in rr)),repetitions=len(rr)))
    if not values: return dict(states=0,status='NO_PAIRS')
    def metrics(v):
        y=np.array([r['delta'] for r in v]);q=np.array([r['pred'] for r in v]);p=np.array([r['p'] for r in v]);c=np.array([r['cost'] for r in v])
        random_p=float(np.sum(p*c)/np.sum(c)) if c.sum()>0 else float(p.mean())
        return dict(mean_delta=float(y.mean()),rmse=float(np.mean((q-y)**2)**.5),
            zero_predictor_rmse=float(np.mean(y*y)**.5),rmse_improvement=float(np.mean(y*y)**.5-np.mean((q-y)**2)**.5),
            allocation_gain_vs_compute_matched_random=float(np.mean((p-random_p)*y)),
            random_continue_probability=random_p)
    result=metrics(values);rng=np.random.default_rng(621)
    groups=sorted({(r['seed'],r['stream']) for r in values}); samples=[]
    for _ in range(bootstrap):
        chosen=rng.integers(0,len(groups),len(groups))
        samples.append(metrics([r for j in chosen for r in values if (r['seed'],r['stream'])==groups[j]]))
    for key in ('mean_delta','rmse_improvement','allocation_gain_vs_compute_matched_random'):
        result[key+'_ci95']=np.quantile([s[key] for s in samples],[.025,.975]).tolist()
    y=np.array([r['delta'] for r in values]);q=np.array([r['pred'] for r in values])
    result['spearman']=float(spearmanr(q,y).statistic) if np.std(y)>1e-12 and np.std(q)>1e-12 else None
    result.update(states=len(values),pairs=len(rows),independent_environment_streams=len(groups),
        max_pair_tail_bound=max(r['tail'] for r in values),state_scores=values,
        scope='local one-step compute allocation under checkpoint continuation; one training seed; exploratory')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=9101)
    p.add_argument('--prefixes',type=int,nargs='+',default=[128,1024]);p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--horizon',type=int,default=1536);p.add_argument('--score-horizon',type=int,default=4096)
    p.add_argument('--device',default='cuda:0');p.add_argument('--score-only',action='store_true')
    args=p.parse_args();args.checkpoint=None;args.threads=1
    if min(args.batch_size,args.horizon,args.score_horizon,args.repetitions)<1 or min(args.prefixes)<0: p.error('invalid budget')
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    flags,actor,model,binding=load_checkpoints(args)
    score=rollout(flags,actor,model,seed=args.seed+10000,prefix=0,repetition=0,
        horizon=args.score_horizon,arm='policy',batch_size=args.batch_size,device=args.device)
    (args.output/'score.json').write_text(json.dumps(score))
    summary=dict(score_raw_per_1000=float(np.mean(score['raw_window'])/args.score_horizon*1000),
        score_raw_per_stream=score['raw_window'],score_real_steps=args.score_horizon*args.batch_size,
        score_compute_per_real_step=float(np.sum(score['computations'])/(args.score_horizon*args.batch_size)),
        complete_episodes=sum(len(r) for r in score['completed_episode_returns']),
        mean_complete_episode_return=None)
    completed=sum(score['completed_episode_returns'],[])
    if completed: summary['mean_complete_episode_return']=float(np.mean(completed))
    print(json.dumps(summary),flush=True)
    pairs=[]
    if not args.score_only:
        for prefix in args.prefixes:
            for rep in range(args.repetitions):
                arms=[rollout(flags,actor,model,seed=args.seed,prefix=prefix,repetition=rep,
                    horizon=args.horizon,arm=arm,batch_size=args.batch_size,device=args.device) for arm in ('stop','continue')]
                repeat=rollout(flags,actor,model,seed=args.seed,prefix=prefix,repetition=rep,
                    horizon=args.horizon,arm='stop',batch_size=args.batch_size,device=args.device) if rep==0 else None
                verify_pair(*arms,repeat)
                for i in range(args.batch_size):
                    s,c=(a['returns'][i] for a in arms)
                    pairs.append(dict(seed=args.seed,stream=i,prefix=prefix,repetition=rep,
                        ema_delta=arms[0]['entry']['ema_delta'][i],online_delta=arms[0]['entry']['online_delta'][i],
                        p_continue=arms[0]['entry']['continue_probability'][i],
                        delta=c['net_truncated']-s['net_truncated'],delta_task=c['task_truncated']-s['task_truncated'],
                        continue_stage_cost=c['current_stage_cost'],stop_return=s['net_truncated'],continue_return=c['net_truncated'],
                        tail_bound=sum(tail_bound(float(flags.discounting),a['real_steps'],float(flags.reward_clip),a['terminal']) for a in (s,c)),
                        paired_entry_sha256=arms[0]['entry']['entry_sha256']))
                (args.output/f'arms_p{prefix}_r{rep}.json').write_text(json.dumps(dict(stop=arms[0],continue_arm=arms[1],repeat_verified=repeat is not None)))
                (args.output/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
                print(f'prefix={prefix} repetition={rep} pairs={len(pairs)}',flush=True)
        summary['voc']=aggregate_pairs(pairs)
    summary.update(status='PASS',checkpoint=binding,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        policy_epsilon=float(flags.voc_train_epsilon),reward_clip=float(flags.reward_clip),gamma=float(flags.discounting),
        objective='future clipped task return minus current-stage cost; no learned bootstrap target')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__':main()
