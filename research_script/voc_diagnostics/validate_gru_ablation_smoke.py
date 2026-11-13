"""Audit branch equality and actual training, then exercise new budget controls."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from reward_probe import load_checkpoints
from gru_ablation_eval import score_rollout
from train_pilot import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();results={}
    for game in ['enduro','pong']:
        records=[]
        for mode in ['detached','gru']:
            d=a.root/f'{game}_{mode}'
            s=json.loads((d/'status.json').read_text());init=json.loads((d/'initialization.json').read_text())
            assert s['phase']=='COMPLETE' and s['env_steps']==512
            gradients=[json.loads(x) for x in (d/'gradient_metrics.jsonl').read_text().splitlines()]
            assert any(g['voc_gru_candidate_norm']>0 and g['actor_optimizer_stepped'] for g in gradients)
            assert all((g['voc_gru_applied_norm']==g['voc_gru_candidate_norm'] if mode=='gru' else g['voc_gru_applied_norm']==0) for g in gradients)
            assert gradients[-1]['model_updates']>0 and gradients[-1]['imitation_updates']>0
            records.append(init)
        for key in ['actor','model','ema','parent']:
            assert records[0][key]==records[1][key],(game,key)
        milestones=json.loads((a.root/f'{game}_gru/milestones.json').read_text())
        bundle=Path(milestones[-1]['bundle'])
        flags,actor,model,binding=load_checkpoints(argparse.Namespace(bundle=bundle,checkpoint=None,threads=1))
        scores={}
        for mode,mean in [('learned',None),('fixed',1.5),('random',1.5)]:
            scores[mode]=score_rollout(flags,actor,model,seed=81231,horizon=16,batch=2,
                                      mode=mode,mean=mean,device=a.device)
        assert scores['fixed']['mean_computations_per_action']==1.5
        results[game]=dict(status='PASS',initial_hashes_equal=True,rl_bc_model_updates=True,
            voc_gradient_routing=True,budget_execution=scores)
    atomic_json(a.root/'validation.json',dict(status='PASS',games=results))
    print(json.dumps({'status':'PASS','games':list(results)}),flush=True)


if __name__=='__main__':main()
