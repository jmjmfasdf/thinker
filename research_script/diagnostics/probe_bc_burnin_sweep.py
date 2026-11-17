"""Compare burn-in lengths on identical human targets and planner tokens.

Uses identical observations, tree tokens, imagined choices, and other features.
This measures recurrent-context sensitivity, not a causal return improvement
or a full rerun of short-context planning (which could choose other queries).
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'thinker'))
sys.path.insert(0,str(ROOT/'research_script/voc_diagnostics'))
from reward_learning_eval import BatchRuntime
from thinker.bc_loader import FrameStackedBehavioralDataLoader
from thinker.dynamic_imitation import DynamicImitationRunner
from thinker import util


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261114)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    started=time.monotonic();ab=a.actor.read_bytes();mb=a.model.read_bytes()
    ac=torch.load(io.BytesIO(ab),map_location='cpu',weights_only=False)
    mc=torch.load(io.BytesIO(mb),map_location='cpu',weights_only=False)
    flags=argparse.Namespace(**ac['flags']);flags.voc_gate_policy_schema_version=ac['voc_gate_policy_schema_version']
    rt=BatchRuntime(flags,ac,mc,a.seed,a.batch_size,'cpu')
    hooks=[]
    try:
        loader=FrameStackedBehavioralDataLoader(base_path=flags.icopro_data_path,
            subjects=[1],game_id=flags.icopro_game_id,sessions=[1,2,3,4],num_actions=rt.actor.num_actions,
            scored_length=40,frame_stack_n=flags.frame_stack_n,target_size=(84,84),grayscale=False,normalize=False,seed=a.seed)
        batch=loader.get_sequence_batch(batch_size=a.batch_size,sequence_length=40)
        runner=DynamicImitationRunner(rt.actor,rt.model,rt.flags,device=rt.device,noop_action_index=0)
        events=[];current={};edge=0
        def begin(module,args,kwargs):
            nonlocal edge,current
            env=args[0]
            if events and bool(env.real_transition[0].all()): edge+=1
            current=dict(edge=edge,phase=env.phase[0].clone(),count=env.search_steps[0].clone())
        def token(module,args,kwargs):
            current.update(x=args[0][0].clone(),h=kwargs['state'][0].clone(),
                update=kwargs['update_mask'][0].clone(),reset=kwargs['reset_mask'][0].clone())
        def features(module,args): current['features']=args[0].clone()
        def end(module,args,output): events.append(current.copy())
        hooks=[rt.actor.register_forward_pre_hook(begin,with_kwargs=True),
            rt.actor.tree_rep_encoder.register_forward_pre_hook(token,with_kwargs=True),
            rt.actor.policy.register_forward_pre_hook(features),rt.actor.register_forward_hook(end)]
        with torch.no_grad(): result=runner.rollout(batch,tree_carry=True,training=True)
        for h in hooks:h.remove()
        hooks=[]
        assert max(e['edge'] for e in events)==40
        cell=rt.actor.tree_rep_encoder.cells[0]
        offset=rt.actor.dim_rep_actions+2+rt.actor.num_input_rewards
        size=cell.hidden_size
        ema=ac['voc_ema_gate_head_state_dict']
        w=(ema['weight'][0]-ema['weight'][1]);bias=ema['bias'][0]-ema['bias'][1]
        rows=[]
        # Reference carries from the beginning of the sampled 41-edge window;
        # it is not claimed to be the exact full-episode hidden state.
        assert rt.actor.voc_gate_execution_epsilon == .02
        with torch.no_grad():
            for target_start,burnin in [(s,b) for s in [17,25,33] for b in [0,1,4,8,16]]:
                start=target_start-burnin
                segment=[e for e in events if start<=e['edge']<target_start+4]
                carried=segment[0]['h'].clone();cold=torch.zeros_like(carried)
                for e in segment:
                    carried=carried.masked_fill(e['reset'][:,None],0)
                    cold=cold.masked_fill(e['reset'][:,None],0)
                    carried=torch.where(e['update'][:,None],cell(e['x'],carried),carried)
                    cold=torch.where(e['update'][:,None],cell(e['x'],cold),cold)
                    f=e['features'].clone()
                    phase_embedding=rt.actor.phase_embedding(e['phase'])
                    torch.testing.assert_close(carried+phase_embedding[:,offset:offset+size],f[:,offset:offset+size],rtol=2e-5,atol=2e-6)
                    other=f.clone();other[:,offset:offset+size]+=cold-carried
                    d=f@w+bias;dc=other@w+bias
                    pc=.01+.98*torch.sigmoid(d/flags.voc_gate_q_temperature)
                    pr=.01+.98*torch.sigmoid(dc/flags.voc_gate_q_temperature)
                    if e['edge']<target_start:continue
                    lp=torch.log_softmax(rt.actor.policy(f),dim=-1)
                    lc=torch.log_softmax(rt.actor.policy(other),dim=-1)
                    for stream in range(a.batch_size):
                        kind=('real' if int(e['phase'][stream])==util.NEED_REAL_ACTION_PHASE else
                              'first_optional' if int(e['phase'][stream])==util.SEARCH_PHASE and int(e['count'][stream])==1 else None)
                        if kind is None:continue
                        action=int(batch['actions_seq'][stream,e['edge']])
                        row=dict(target_start=target_start,burnin=burnin,chunk_start=start,edge=e['edge'],target_in_chunk=e['edge']-target_start+1,stream=stream,kind=kind,
                            h_difference=float((cold[stream]-carried[stream]).norm()),
                            h_relative_difference=float((cold[stream]-carried[stream]).norm()/carried[stream].norm().clamp_min(1e-9)),
                            ema_delta_carried=float(d[stream]),ema_delta_cold=float(dc[stream]),
                            delta_difference=float(dc[stream]-d[stream]),
                            sign_flip=bool((d[stream]>0)!=(dc[stream]>0)),
                            p_carried=float(pc[stream]),p_cold=float(pr[stream]),
                            probability_difference=float(pr[stream]-pc[stream]),
                            real_policy_kl=float((lp[stream].exp()*(lp[stream]-lc[stream])).sum()),
                            human_ll_carried=float(lp[stream,action]),human_ll_cold=float(lc[stream,action]),
                            human_ll_difference=float(lc[stream,action]-lp[stream,action]))
                        rows.append(row)
        df=pd.DataFrame(rows);df.to_csv(a.output/'memory_context_rows.csv',index=False)
        summary=df.groupby(['kind','burnin','target_in_chunk']).agg(n=('stream','size'),
            median_hidden_relative_difference=('h_relative_difference','median'),
            mean_absolute_delta_difference=('delta_difference',lambda x:x.abs().mean()),
            mean_absolute_probability_difference=('probability_difference',lambda x:x.abs().mean()),
            sign_flip_fraction=('sign_flip','mean'),mean_real_policy_kl=('real_policy_kl','mean'),
            mean_human_ll_difference=('human_ll_difference','mean'))
        summary.to_csv(a.output/'memory_context_summary.csv')
        metadata=dict(actor_sha256=hashlib.sha256(ab).hexdigest(),model_sha256=hashlib.sha256(mb).hexdigest(),
            actor_step=int(ac['real_step']),batch_size=a.batch_size,seed=a.seed,
            windows=3*a.batch_size,burnins=[0,1,4,8,16],full_sequence_edges=41,seconds=time.monotonic()-started,
            batch_metadata={k:np.asarray(v).tolist() for k,v in batch.items() if k not in ['obs_seq']},
            assertions='Carried hidden reconstruction matches exact production features at every call.',
            limitations=['Fixed-token recurrence isolation; other tree and image features held identical.',
                        'Reference starts zero at sampled window, not full episode; within-sequence measurements correlated.',
                        'No claim cold initialization causes worse reward or human accuracy.'])
        (a.output/'manifest.json').write_text(json.dumps(metadata,indent=2))
        print(summary.to_string(),flush=True)
    finally:
        for h in hooks:h.remove()
        rt.close()


if __name__=='__main__':main()
