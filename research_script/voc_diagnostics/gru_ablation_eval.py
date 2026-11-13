"""Held-out scores, calibrated budget baselines and actual paired VOC returns."""
from __future__ import annotations
import argparse,json,time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from reward_probe import load_checkpoints,util,keyed_seed,preserve_tokens,digest
from reward_learning_eval import BatchRuntime,categorical
from gru_reward_readout import arm_rollout,verify
from gru_information_probe import atomic_json


@lru_cache(maxsize=32)
def geometric_survival(mean,cap):
    if not 0<=mean<=cap:raise ValueError('Budget outside cap')
    if mean==0:return 0.
    if mean==cap:return 1.
    lo,hi=0.,1.
    for _ in range(60):
        mid=(lo+hi)/2
        if sum(mid**k for k in range(1,cap+1))<mean:lo=mid
        else:hi=mid
    return (lo+hi)/2


def prescribed_budget(mode,mean,cap,decision,uniform):
    if mode=='fixed':return min(cap,int(np.floor((decision+1)*mean+1e-10)-np.floor(decision*mean+1e-10)))
    r=geometric_survival(mean,cap)
    if r==0:return 0
    if r==1:return cap
    return min(cap,int(np.floor(np.log1p(-uniform)/np.log(r))))


def score_rollout(flags,actor_ck,model_ck,*,seed,horizon,batch=8,mode='learned',mean=None,device='cuda:0',gate_policy='native'):
    if gate_policy not in ('native','sign'):
        raise ValueError('Unknown evaluation gate policy')
    rt=BatchRuntime(flags,actor_ck,model_ck,seed,batch,device)
    gate_log_odds=[]
    def capture_gate(module,inputs,output):
        gate_log_odds[:] = [output.detach().reshape(-1)]
    hook=rt.actor.voc_gate_head.register_forward_hook(capture_gate) if gate_policy=='sign' else None
    before=digest(dict(actor=rt.actor.state_dict(),model=rt.model.state_dict()))
    raw=np.zeros(batch);last=np.zeros(batch);counts=np.zeros(batch,dtype=int)
    current_counts=np.zeros(batch,dtype=int);budgets=[];completed=[[] for _ in range(batch)]
    cap=int(flags.max_search_steps);real_index=0;started=time.monotonic()
    try:
        state,info=rt.env.reset(rt.model)
        env=util.init_env_out(state,info,rt.flags,rt.actor.dim_actions,rt.actor.tuple_action)
        core=rt.actor.initial_state(batch_size=batch,device=rt.device)
        with torch.no_grad():
            for call in range((horizon+1)*(cap+4)):
                phase=env.phase[0].cpu().numpy();k=env.search_steps[0].cpu().numpy()
                out,core=rt.actor(env,core,compute_loss=False,greedy=False)
                draws=np.array([[keyed_seed(seed+i,real_index,int(k[i]),int(phase[i])+100*j,0)/2**32
                                 for i in range(batch)] for j in range(3)])
                for i in np.flatnonzero(phase==util.NEED_REAL_ACTION_PHASE):
                    draws[0,i]=keyed_seed(seed+i,real_index,0,int(phase[i]),0)/2**32
                primary=categorical(out.pri_param[-1].reshape(batch,-1),draws[0])
                control=categorical(out.search_control_logits[-1],draws[1])
                if gate_policy=='sign' and mode=='learned':
                    conditional=categorical(out.search_control_logits[-1,:,:2],draws[2])
                    sign=gate_log_odds[0].cpu().numpy()>0
                    for i in np.flatnonzero(phase==util.SEARCH_PHASE):
                        legal=env.legal_control_mask[0,i].cpu().numpy().astype(bool)
                        may_continue=legal[:2].any()
                        must_continue=not legal[util.STOP]
                        control[i]=conditional[i] if may_continue and (sign[i] or must_continue) else util.STOP
                if mode!='learned':
                    conditional=categorical(out.search_control_logits[-1,:,:2],draws[2])
                    for i in np.flatnonzero(phase==0):
                        u=keyed_seed(seed+i,real_index,0,8701,0)/2**32
                        target=prescribed_budget(mode,mean,cap,real_index,u)
                        control[i]=util.STOP if k[i]>=target else conditional[i]
                action=(torch.as_tensor(primary,device=rt.device).reshape(out.action[0].shape),
                        torch.as_tensor(control,device=rt.device).reshape(out.action[1].shape))
                state,reward,done,truncated,info=rt.env.step(action,rt.model)
                env=preserve_tokens(env,util.create_env_out(action,state,reward,done,truncated,info,rt.flags),info)
                real=info['real_transition'].bool().cpu().numpy()
                if real.any() and not real.all():raise AssertionError('Asynchronous real actions')
                think=info['think_reward'].cpu().numpy().reshape(-1)
                counts+=(think==-1);current_counts+=(think==-1)
                if real.all():
                    ep=info['episode_return'].cpu().numpy().reshape(-1)
                    finished=info['real_done'].bool().cpu().numpy().reshape(-1)
                    raw+=ep-last
                    for i in np.flatnonzero(finished):completed[i].append(float(ep[i]))
                    last=ep.copy();last[finished]=0.
                    if mode!='learned':
                        expected=[prescribed_budget(mode,mean,cap,real_index,
                            keyed_seed(seed+i,real_index,0,8701,0)/2**32) for i in range(batch)]
                        if not np.array_equal(current_counts,expected):
                            raise AssertionError(f'Budget execution mismatch: {current_counts}, {expected}')
                    budgets.append(current_counts.copy());current_counts[:]=0;real_index+=1
                    if real_index==horizon:break
            else:raise RuntimeError('Score rollout watchdog')
        if digest(dict(actor=rt.actor.state_dict(),model=rt.model.state_dict()))!=before:
            raise AssertionError('Score evaluation changed checkpoint')
        return dict(mode=mode,gate_policy=gate_policy,seed=seed,real_actions_per_stream=real_index,
            score_per_1000_actions=(raw/real_index*1000).tolist(),raw_window=raw.tolist(),
            computations=counts.tolist(),mean_computations_per_action=float(counts.mean()/real_index),
            prescribed_mean=mean,budget_histogram=np.bincount(np.asarray(budgets).ravel(),minlength=cap+1).tolist(),
            completed_episode_returns=completed,seconds=time.monotonic()-started,weights_sha256=before,
            real_action_rng='seed/stream/real decision; independent of search count')
    finally:
        if hook is not None:hook.remove()
        rt.close()


def paired_summary(rows):
    grouped={}
    for r in rows:grouped.setdefault((r['seed'],r['stream'],r['prefix'],r['count']),[]).append(r)
    states=[dict(key=list(k),target=float(np.mean([r['delta'] for r in rr])),
        online=float(np.mean([r['online'] for r in rr])),ema=float(np.mean([r['ema'] for r in rr])),
        probability=float(np.mean([r['probability'] for r in rr])),
        cost=float(np.mean([r['cost'] for r in rr]))) for k,rr in grouped.items()]
    y=np.array([s['target'] for s in states]);online=np.array([s['online'] for s in states]);ema=np.array([s['ema'] for s in states])
    p=np.array([s['probability'] for s in states]);c=np.array([s['cost'] for s in states])
    groups=np.array([s['key'][0]*1000+s['key'][1] for s in states]);unique=np.unique(groups)
    def calc(ids):
        yy=y[ids];ee=ema[ids];oo=online[ids];pp=p[ids];cc=c[ids]
        null=float(np.mean(yy**2)**.5);er=float(np.mean((ee-yy)**2)**.5)
        constant_p=float(np.sum(pp*cc)/np.sum(cc))
        return dict(ema_rmse=er,online_rmse=float(np.mean((oo-yy)**2)**.5),
            zero_rmse=null,ema_rmse_improvement_vs_zero=null-er,
            allocation_gain_vs_cost_matched_random=float(np.mean((pp-constant_p)*yy)),mean_net_benefit=float(yy.mean()))
    result=calc(np.arange(len(y)));rng=np.random.default_rng(4982)
    draws=[calc(np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])) for _ in range(2000)]
    for key in ['ema_rmse_improvement_vs_zero','allocation_gain_vs_cost_matched_random','mean_net_benefit']:
        result[key+'_ci95']=np.quantile([d[key] for d in draws],[.025,.975]).tolist()
    rho=spearmanr(y,ema).statistic if np.std(ema)>1e-12 and np.std(y)>1e-12 else None
    result.update(ema_spearman=float(rho) if rho is not None and np.isfinite(rho) else None,
        states=len(states),pairs=len(rows),independent_streams=len(unique),state_records=states,
        max_pair_tail_bound=max(r['tail_bound'] for r in rows),
        task_changed_pairs=sum(abs(r['task_delta'])>1e-12 for r in rows))
    return result


def score_comparison(a,b):
    if a['seed']!=b['seed'] or a['real_actions_per_stream']!=b['real_actions_per_stream']:
        raise AssertionError('Score protocols differ')
    x=np.asarray(a['score_per_1000_actions'])-b['score_per_1000_actions']
    rng=np.random.default_rng(5913);means=x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
    return dict(reward_difference_per_1000_actions=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
        learned_compute=a['mean_computations_per_action'],baseline_compute=b['mean_computations_per_action'],
        compute_difference=b['mean_computations_per_action']-a['mean_computations_per_action'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--score-only',action='store_true')
    a=p.parse_args();a.checkpoint=None;a.threads=1
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    flags,actor,model,binding=load_checkpoints(a)
    atomic_json(a.output/'status.json',dict(phase='SCORE'))
    score=score_rollout(flags,actor,model,seed=34101,horizon=2048,device=a.device)
    atomic_json(a.output/'score_learned.json',score)
    metrics=dict(score_per_1000_actions=float(np.mean(score['score_per_1000_actions'])),
                 computations_per_action=score['mean_computations_per_action'])
    summary=dict(checkpoint=binding,score=score,metrics=metrics,status='PARTIAL')
    if not a.score_only:
        atomic_json(a.output/'status.json',dict(phase='CALIBRATE_BUDGET'))
        calibration=score_rollout(flags,actor,model,seed=34181,horizon=1024,batch=4,device=a.device)
        atomic_json(a.output/'budget_calibration.json',calibration)
        mean=calibration['mean_computations_per_action']
        comparisons={}
        for mode in ['fixed','random']:
            atomic_json(a.output/'status.json',dict(phase='SCORE_'+mode.upper()))
            baseline=score_rollout(flags,actor,model,seed=34101,horizon=2048,mode=mode,mean=mean,device=a.device)
            atomic_json(a.output/f'score_{mode}.json',baseline)
            comparisons[mode]=score_comparison(score,baseline)
            metrics[f'reward_gain_vs_{mode}']=comparisons[mode]['reward_difference_per_1000_actions']
            metrics[f'compute_difference_{mode}']=comparisons[mode]['compute_difference']
        summary['budget_comparisons']=comparisons
        rows=[];ema=actor['voc_ema_gate_head_state_dict']
        ew=ema['weight'].double().cpu().numpy();eb=ema['bias'].double().cpu().numpy()
        for prefix in [128,1024]:
          for k in [0,3]:
            for rep in range(2):
                atomic_json(a.output/'status.json',dict(phase='PAIRED',prefix=prefix,count=k,repetition=rep,pairs=len(rows)))
                kwargs=dict(seed=33101,repetition=rep,prefix=prefix,targets=np.full(8,k),horizon=1536,device=a.device)
                arms=[]
                for mode in ['stop','continue']:
                    v=arm_rollout(flags,actor,model,arm=mode,**kwargs);arms.append(v)
                    atomic_json(a.output/f'pair_p{prefix}_k{k}_r{rep}_{mode}.json',v)
                verify(*arms)
                if rep==0:
                    repeat=arm_rollout(flags,actor,model,arm='stop',**kwargs);verify(arms[0],repeat)
                    if repeat['returns']!=arms[0]['returns']:raise AssertionError('Repeat STOP differs')
                    atomic_json(a.output/f'repeat_p{prefix}_k{k}.json',dict(status='PASS'))
                for i,(entry,stop,cont) in enumerate(zip(arms[0]['entries'],arms[0]['returns'],arms[1]['returns'])):
                    q=np.asarray(entry['features'])@ew.T+eb
                    rows.append(dict(seed=33101,stream=i,prefix=prefix,count=k,repetition=rep,
                        entry_hash=entry['hash'],delta=cont['net_truncated']-stop['net_truncated'],
                        task_delta=cont['task_truncated']-stop['task_truncated'],
                        cost=cont['current_stage_cost']-stop['current_stage_cost'],
                        online=entry['delta'],ema=float(q[0]-q[1]),probability=entry['p_continue'],
                        tail_bound=arms[0]['tail_bounds'][i]+arms[1]['tail_bounds'][i]))
                atomic_json(a.output/'pairs.json',rows)
        voc=paired_summary(rows)
        assert voc['states']==32 and voc['pairs']==64 and voc['independent_streams']==8
        assert voc['max_pair_tail_bound']<4e-5
        summary['voc']=voc
        metrics.update({'voc_'+k:v for k,v in voc.items() if isinstance(v,(int,float)) or v is None})
    summary['status']='COMPLETE';atomic_json(a.output/'summary.json',summary)
    atomic_json(a.output/'status.json',dict(phase='COMPLETE'))
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
