"""Verify a completed 1M continuation against checkpoints, logs and W&B."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import torch
import wandb
import yaml

from thinker.planning_protocol import validate_planning_training_checkpoint
from summarize_adaptive_planning_pilot import loss_health


def finite_tensors(value):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():raise ValueError('nonfinite model/optimizer tensor')
        return 1
    if isinstance(value,dict):return sum(finite_tensors(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(finite_tensors(v) for v in value)
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite optimizer scalar')
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--condition',choices=['control','burn8','gru10'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();campaign=args.campaign.resolve()
    manifest=json.loads((campaign/'manifest.json').read_text())
    item=next(r for r in manifest['runs'] if r['condition']==args.condition)
    submission=json.loads((campaign/'submissions.json').read_text())[args.condition]
    run=Path(item['run_directory'])
    scheduler=subprocess.check_output(['sacct','-j',str(submission['job_id']),'-X','-n','-P',
        '--format=JobID,State,ExitCode,Elapsed,NodeList'],text=True)
    job,state,exit_code,elapsed,node=scheduler.strip().split('|')[:5]
    if state!='COMPLETED' or exit_code!='0:0' or not (run/'finish').exists():
        raise ValueError('training is not successfully complete')
    config=yaml.safe_load(Path(item['config']).read_bytes())
    checkpoints={};loaded={};tensor_counts={}
    for kind in ('actor','model'):
        eligible=[p for p in run.glob(f'ckp_{kind}.tar_step_*')
                  if int(p.name.rsplit('_',1)[1])>=item['training_steps']]
        if len(eligible)!=1:raise ValueError('expected exactly one final checkpoint per network')
        path=eligible[0];raw=path.read_bytes();saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        if not item['training_steps']<=saved['real_step']<item['training_steps']+2000:
            raise ValueError('final counter outside registered budget tolerance')
        if saved['real_step']!=int(path.name.rsplit('_',1)[1]):raise ValueError('filename counter mismatch')
        tensor_counts[kind]=finite_tensors({k:v for k,v in saved.items() if 'state_dict' in k})
        if tensor_counts[kind]==0:raise ValueError('missing saved network/optimizer tensors')
        checkpoints[kind]=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),real_step=int(saved['real_step']))
        loaded[kind]=saved
    actor=loaded['actor']
    contract=validate_planning_training_checkpoint(actor,config)
    if actor['voc_gate_policy_schema_version']!=4:raise ValueError('changed gate policy schema')
    fast_binding=campaign/f'fast_allocation_1m/{args.condition}_binding.json'
    if fast_binding.exists() and json.loads(fast_binding.read_text())['checkpoint']!=checkpoints:
        raise ValueError('early allocation evaluates different final checkpoint bytes')
    parent_path=Path(item['parent_checkpoint']['actor']['path']);parent_raw=parent_path.read_bytes()
    if hashlib.sha256(parent_raw).hexdigest()!=item['parent_checkpoint']['actor']['sha256']:
        raise ValueError('parent checkpoint bytes changed')
    parent=torch.load(io.BytesIO(parent_raw),map_location='cpu',weights_only=False)
    raw_logs=(run/'logs.csv').read_bytes()
    rows=list(csv.DictReader(io.StringIO(raw_logs.decode())))
    if not rows or any(None in r or None in r.values() for r in rows):raise ValueError('incomplete final log')
    if int(float(rows[-1]['real_step']))!=actor['real_step']:raise ValueError('last log differs from checkpoint')
    counter_checks=[]
    mapping={'imitation_update_count':'actor/optimizer_stepped',
             'voc_update_count':'actor/voc_optimizer_stepped',
             'voc_continue_count':'actor/voc_continue_support',
             'voc_stop_count':'actor/voc_stop_support'}
    for key,column in mapping.items():
        observed=sum(float(r[column]) for r in rows)
        delta=actor[key]-parent[key]
        if observed!=delta:raise ValueError('resume counter/log mismatch: '+key)
        counter_checks.append(dict(counter=key,parent=parent[key],final=actor[key],logged_delta=observed))
    if any(float(r['actor/icopro_count'])!=config['icopro_batch_size']*config['batch_length'] for r in rows):
        raise ValueError('BC target count changed')
    remote=wandb.Api(timeout=30).run('jmme425/thinker/'+item['xpid'])
    keys=['real_step','actor/icopro_nll','actor/total_loss','gradient/voc_gru_candidate_norm',
          'gradient/voc_gru_applied_norm','gradient/voc_gru_gradient_scale']
    metrics={k:remote.summary.get(k) for k in keys}
    if remote.state!='finished' or metrics['real_step']!=actor['real_step']:
        raise ValueError('remote W&B final state/step is not synchronized yet')
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in metrics.values()):
        raise ValueError('missing/nonfinite final remote metric')
    if metrics['gradient/voc_gru_gradient_scale']!=config['voc_gru_gradient_scale'] or not math.isclose(
        metrics['gradient/voc_gru_applied_norm'],metrics['gradient/voc_gru_candidate_norm']*config['voc_gru_gradient_scale'],rel_tol=1e-6,abs_tol=1e-9):
        raise ValueError('registered gradient multiplier differs from remote applied value')
    report=dict(status='COMPLETED_BUDGET_CHECKPOINT_COUNTERS_AND_WANDB_PASS',checked_at=time.time(),
        condition=args.condition,scheduler=scheduler,checkpoint=checkpoints,finite_saved_tensor_counts=tensor_counts,
        contract=contract,gate_schema=actor['voc_gate_policy_schema_version'],counter_checks=counter_checks,
        invocation_bc_targets=sum(float(r['actor/icopro_count']) for r in rows),
        bc_loss_history=loss_health(rows),logs_sha256=hashlib.sha256(raw_logs).hexdigest(),
        wandb=dict(state=remote.state,url=remote.url,metrics=metrics),
        evaluation_binding_checked=fast_binding.exists(),
        limitation='Completion and numerical validity do not establish useful planning. Historical loss instability is retained and must be considered in comparisons.')
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/Path(__file__).name)
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
