"""Verify a frozen continuation before or after training inside a SLURM job.

Used after a successful train process; the job exits successfully only when
checkpoint, optimizer, BC counters, numerical history and W&B agree.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import yaml

from thinker import util
from thinker.planning_protocol import validate_planning_training_checkpoint, planning_training_contract
from summarize_adaptive_planning_pilot import loss_health


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all(): raise ValueError('nonfinite tensor')
    elif isinstance(value,dict):
        for child in value.values(): finite(child)
    elif isinstance(value,(tuple,list)):
        for child in value: finite(child)
    elif isinstance(value,float) and not math.isfinite(value):
        raise ValueError('nonfinite scalar')


def optimizer_step(state):
    return max(float(v.get('step',0)) for v in state['state'].values())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--condition',required=True)
    parser.add_argument('--phase',choices=['smoke','full'],required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--require-gpu',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    campaign=args.campaign.resolve();manifest=json.loads((campaign/'manifest.json').read_text())
    items=manifest['smoke_runs'] if args.phase=='smoke' else manifest['runs']
    item=next(r for r in items if r['condition']==args.condition)
    config=Path(item['config']);folder=config.parent;run=Path(item['run_directory'])
    flags=util.create_setting(args=['--config',str(config)],save_flags=False)
    if (flags.xpid!=item['xpid'] or flags.total_steps!=item['training_steps']
            or not flags.ckp or Path(flags.ckpdir)!=run):
        raise ValueError('resolved resume destination/budget mismatch')
    parent={}
    for kind,info in item['parent_checkpoint'].items():
        if sha(info['path'])!=info['sha256'] or sha(info['immutable_copy'])!=info['sha256']:
            raise ValueError('parent checkpoint changed')
        parent[kind]=torch.load(info['immutable_copy'],map_location='cpu',weights_only=False)
    contract=validate_planning_training_checkpoint(parent['actor'],flags)
    util.validate_voc_resume_protocol(parent['actor'],flags)
    if contract!=planning_training_contract(flags) or contract['version']!=2:
        raise ValueError('normalized continuation contract differs')
    base=yaml.safe_load(Path(item['parent_config']).read_text())
    current=yaml.safe_load(config.read_text())
    changes={k:[base.get(k),current.get(k)] for k in base.keys()|current.keys() if base.get(k)!=current.get(k)}
    if set(changes)!={'ckp','total_steps','xpid'}:
        raise ValueError('unregistered algorithm/configuration change')
    if args.preflight:
        for kind,info in item['parent_checkpoint'].items():
            if sha(run/f'ckp_{kind}.tar')!=info['sha256']:
                raise ValueError('training would not start from registered250kparent')
        if args.require_gpu:
            required = {'rtx3090':'3090','titanrtx':'TITAN RTX'}[manifest['training_gpu']]
            if torch.cuda.device_count()!=2 or any(required not in torch.cuda.get_device_name(i) for i in range(2)):
                raise ValueError('training does not have two registered GPUs')
            if any(torch.cuda.get_device_properties(i).total_memory<23*1024**3 for i in range(2)):
                raise ValueError('training requires24GB-class GPUs')
            if args.phase=='full':
                smoke=next(r for r in manifest['smoke_runs'] if r['condition']==args.condition)
                result=json.loads((Path(smoke['config']).parent/'completion.json').read_text())
                if result['status']!='CONTINUATION_CHECKPOINT_OPTIMIZER_COUNTERS_WANDB_PASS':
                    raise ValueError('same-condition resume smoke has not passed')
        report=dict(status='CONTINUATION_PREFLIGHT_PASS',contract=contract,changes=changes,
                    total_steps=flags.total_steps,checked_at=time.time(),gpu_checked=args.require_gpu,
                    training_gpu=manifest['training_gpu'],
                    gpu_names=[torch.cuda.get_device_name(i) for i in range(2)] if args.require_gpu else [])
        output=folder/('gpu_preflight.json' if args.require_gpu else 'preflight.json')
    else:
        if not (run/'finish').exists():raise ValueError('finish marker missing')
        logs=list(csv.DictReader((run/'logs.csv').open()))
        if not logs or any(None in r or None in r.values() for r in logs):raise ValueError('incomplete logs')
        steps=[int(float(r['real_step'])) for r in logs]
        if steps[0]<=parent['actor']['real_step'] or any(b<=a for a,b in zip(steps,steps[1:])):
            raise ValueError('resume counters are not monotonic')
        checkpoints={};saved={};optimizers={}
        for kind in ('actor','model'):
            paths=[p for p in run.glob(f'ckp_{kind}.tar_step_*')
                   if flags.total_steps<=int(p.name.rsplit('_',1)[1])<flags.total_steps+2000]
            if len(paths)!=1:raise ValueError('missing/ambiguous final checkpoint')
            path=paths[0];value=torch.load(path,map_location='cpu',weights_only=False)
            if value['real_step']!=int(path.name.rsplit('_',1)[1]):raise ValueError('filename/counter mismatch')
            for key,v in value.items():
                if 'state_dict' in key:finite(v)
            net=kind+'_net_state_dict'
            changed=sum(not torch.equal(t,parent[kind][net][k]) for k,t in value[net].items())
            if not changed:raise ValueError('weights did not change')
            updates={}
            for key,state in value.items():
                if 'optimizer' not in key or not key.endswith('state_dict') or not state:continue
                if not state['state']:
                    if key!='voc_gate_optimizer_state_dict' or not flags.voc_gate_exact_projection:
                        raise ValueError('unexpected inactive optimizer')
                    continue
                before,after=optimizer_step(parent[kind][key]),optimizer_step(state)
                if after<=before:raise ValueError('optimizer was reset or did not advance')
                updates[key]=dict(parent=before,final=after)
            if not updates:raise ValueError('no active optimizer restored')
            checkpoints[kind]=dict(path=str(path),real_step=value['real_step'],sha256=sha(path),changed_tensors=changed)
            optimizers[kind]=updates;saved[kind]=value
        actor=saved['actor'];validate_planning_training_checkpoint(actor,flags)
        if actor['planning_training_contract']!=parent['actor']['planning_training_contract']:
            raise ValueError('training contract changed')
        if actor['imitation_data_signature']!=parent['actor']['imitation_data_signature'] or not actor['imitation_rng_state']:
            raise ValueError('BCdata/RNGchanged or missing')
        mapping=dict(imitation_update_count='actor/optimizer_stepped',voc_update_count='actor/voc_optimizer_stepped',
            voc_continue_count='actor/voc_continue_support',voc_stop_count='actor/voc_stop_support')
        counters={}
        for key,column in mapping.items():
            delta=actor[key]-parent['actor'][key];logged=sum(float(r[column]) for r in logs)
            if delta!=logged:raise ValueError('restored counter disagrees with logs: '+key)
            counters[key]=dict(parent=parent['actor'][key],final=actor[key],logged_delta=logged)
        if steps[-1]!=actor['real_step']:raise ValueError('last log/checkpoint mismatch')
        if any(float(r['actor/icopro_count'])!=flags.icopro_batch_size*flags.batch_length for r in logs):
            raise ValueError('scored BCtarget schedule changed')
        health=loss_health(logs)
        if health['bc_history_status']!='NO_LARGE_NLL_OBSERVED':
            (folder/'failed_health.json').write_text(json.dumps(health,indent=2)+'\n')
            raise ValueError('loss instability requires diagnosis before efficacy evaluation')
        import wandb
        remote=wandb.Api(timeout=30).run('jmme425/thinker/'+item['xpid'])
        if remote.state!='finished' or remote.summary.get('real_step')!=actor['real_step']:
            raise ValueError('W&Bfinalstep/state not synchronized')
        metrics={key:remote.summary.get(key) for key in ('real_step','actor/icopro_nll',
            'actor/total_loss','gradient/voc_gru_applied_norm','gradient/voc_gru_gradient_scale')}
        if any(not isinstance(v,(float,int)) or not math.isfinite(v) for v in metrics.values()):
            raise ValueError('W&Bfinalmetrics incomplete/nonfinite')
        if metrics['gradient/voc_gru_gradient_scale']!=1.:
            raise ValueError('gradient multiplier changed')
        collector=json.loads((run/'collector_metrics.json').read_text())
        report=dict(status='CONTINUATION_CHECKPOINT_OPTIMIZER_COUNTERS_WANDB_PASS',
            condition=args.condition,phase=args.phase,checked_at=time.time(),checkpoint=checkpoints,
            contract=contract,optimizers=optimizers,counters=counters,health=health,
            bc_targets=sum(float(r['actor/icopro_count']) for r in logs),
            actor_added_transitions=actor['real_step']-parent['actor']['real_step'],
            physical_collected_transitions=collector['physical_env_steps'],
            logs_sha256=sha(run/'logs.csv'),wandb=dict(url=remote.url,state=remote.state,metrics=metrics),
            scope='Resume smoke is excluded from efficacy; full endpoints are registered in advance.',
            restart_semantics=manifest['restart_semantics'])
        output=folder/'completion.json'
    report['script_sha256']=sha(__file__)
    with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
