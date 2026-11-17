"""Bind successful training completion to finite, reloadable final checkpoints."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'thinker'))
from thinker import util
from thinker.planning_protocol import validate_planning_training_checkpoint


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--condition',required=True)
    args=ap.parse_args()
    torch.set_num_threads(1)
    c=args.campaign.resolve()
    item=next(r for r in json.loads((c/'manifest.json').read_text())['runs']
              if r['condition']==args.condition)
    job=json.loads((c/'submissions.json').read_text())[args.condition]['job_id']
    scheduler=subprocess.check_output(['sacct','-j',str(job),'--noheader','--parsable2',
        '--format=JobID,State,ExitCode,Elapsed,NodeList'],text=True)
    status=next(row.split('|') for row in scheduler.splitlines() if row.startswith(str(job)+'|'))
    if status[1:3]!=['COMPLETED','0:0']:
        raise RuntimeError('Training is not successfully complete: '+str(status))
    run=c.parents[3]/'logs/thinker'/item['xpid']
    loaded,binding={},{}
    for kind in ['actor','model']:
        path=max(run.glob(f'ckp_{kind}.tar_step_*'),key=lambda p:int(p.name.rsplit('_',1)[1]))
        raw=path.read_bytes()
        checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        step=int(checkpoint['real_step'])
        if step<item['training_steps'] or step!=int(path.name.rsplit('_',1)[1]):
            raise ValueError('Checkpoint has incorrect step or unfinished budget')
        state=checkpoint[f'{kind}_net_state_dict']
        for name,tensor in state.items():
            if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
                raise ValueError('Nonfinite/non-tensor state: '+kind+'/'+name)
        loaded[kind]=checkpoint
        binding[kind]=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),real_step=step,
            finite_state_tensors=len(state),state_elements=sum(v.numel() for v in state.values()))
    actor,model=loaded['actor'],loaded['model']
    flags=argparse.Namespace(**actor['flags'])
    validate_planning_training_checkpoint(actor)
    util.validate_voc_gate_policy_checkpoint(actor,flags=flags)
    for key in ['xpid','name','base_seed','voc_critic_parameterization','icopro_burn_in_steps',
                'icopro_sampling_context','voc_gru_gradient_scale','total_steps']:
        if actor['flags'].get(key)!=model['flags'].get(key):
            raise ValueError('Actor/model configurations differ: '+key)
    if not actor['imitation_data_signature'] or not actor['imitation_rng_state']:
        raise ValueError('Missing behavioral-data identity or sampling RNG state')
    result=dict(condition=args.condition,job_id=job,scheduler=status,verified_at=time.time(),
        status='TRAINING_CHECKPOINT_AUDIT_PASS',checkpoint=binding,
        planning_training_contract=actor['planning_training_contract'],
        imitation_data_signature=actor['imitation_data_signature'],
        imitation_update_count=actor['imitation_update_count'],voc_update_count=actor['voc_update_count'],
        requested_learner_env_steps=item['training_steps'],
        budget_note='Learner-consumed real transitions; asynchronous collection may exceed this counter.',
        efficacy_status='Unproven; fixed-policy allocation and paired utility are evaluated separately.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=c/f'checkpoint_completion_{args.condition}.json'
    with out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(condition=args.condition,status=result['status'],checkpoint=binding,
                         imitation_updates=result['imitation_update_count'])),flush=True)


if __name__=='__main__':
    main()
