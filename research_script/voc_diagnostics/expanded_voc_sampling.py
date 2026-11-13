"""Protocol and GPU shard runner for broader, unselected paired VOC sampling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

SEEDS = [33101, 43101, 53101, 63101]
PREFIX_BLOCKS = [[128, 512], [1024, 2048]]
COUNTS = [0, 3]


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sampling_shards():
    return [dict(shard=index*2+block, seed=seed, prefixes=prefixes,
                 counts=COUNTS, batch_size=8, repetitions=2, horizon=1536)
            for index, seed in enumerate(SEEDS)
            for block, prefixes in enumerate(PREFIX_BLOCKS)]


def protocol_hash(task):
    binding = json.loads(Path(task['bundle']).read_text())
    content = dict(task=task, checkpoint=binding)
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def read_arm(path, signature):
    if not path.exists():
        return None
    cached = json.loads(path.read_text())
    if cached['protocol_sha256'] != signature:
        raise ValueError('Cached arm belongs to another protocol')
    return cached['result']


def run_task(task):
    import fcntl
    import numpy as np
    import torch
    source = Path(task['source'])
    sys.path.insert(0, str(source/'thinker'))
    sys.path.insert(0, str(source/'research_script/voc_diagnostics'))
    from reward_probe import load_checkpoints
    from gru_reward_readout import arm_rollout, verify
    from gru_ablation_eval import paired_summary
    import wandb
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('SLURM GPU required')
    if os.getenv('SLURM_JOB_PARTITION') != 'rtx2080' or '2080 Ti' not in torch.cuda.get_device_name(0):
        raise RuntimeError('RTX 2080 Ti evaluation required')
    if os.getenv('SLURMD_NODENAME') == 'bml-compute02':
        raise RuntimeError('compute02 excluded')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path(task['output']);output.mkdir(parents=True, exist_ok=True)
    lock = (output/'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signature = protocol_hash(task)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['protocol_sha256'] != signature:
            raise ValueError('Cannot resume with changed protocol')
    else:
        manifest = dict(protocol_sha256=signature, task=task, wandb_id=uuid.uuid4().hex[:8],
                        created_at=time.time())
        atomic(manifest_path, manifest)
    if (output/'summary.json').exists():
        summary = json.loads((output/'summary.json').read_text())
        if summary['status'] == 'COMPLETE' and summary['protocol_sha256'] == signature:
            return
    run = wandb.init(project='thinker', entity='jmme425', mode='online',
        group='voc-utility-expanded-20260928', job_type='paired-utility-expanded',
        name=f"voc-expanded-t001-{task['game']}-4M-shard{task['shard']}",
        id=manifest['wandb_id'], resume='allow', dir=str(output), config=task)
    if run.settings.mode != 'online': raise RuntimeError('Online W&B required')
    flags, actor, model, binding = load_checkpoints(argparse.Namespace(
        bundle=Path(task['bundle']), checkpoint=None, threads=1))
    assert int(task['horizon']) == 1536 and task['batch_size'] == 8
    assert max(task['counts']) < flags.max_search_steps
    ema = actor['voc_ema_gate_head_state_dict']
    weight = ema['weight'].double().cpu().numpy();bias = ema['bias'].double().cpu().numpy()
    rows = [];started = time.monotonic()
    def progress(phase, **details):
        atomic(output/'status.json', dict(phase=phase, updated_at=time.time(),
            job_id=os.environ['SLURM_JOB_ID'], wandb_url=run.url, pairs=len(rows), **details))
    try:
        progress('STARTED')
        for prefix in task['prefixes']:
          for count in task['counts']:
            for rep in range(task['repetitions']):
                kwargs = dict(seed=task['seed'], repetition=rep, prefix=prefix,
                    targets=np.full(task['batch_size'], count), horizon=task['horizon'], device='cuda:0')
                stem = f'p{prefix}_k{count}_r{rep}'
                arms = []
                for mode in ['stop', 'continue']:
                    progress('PAIRED', prefix=prefix, count=count, repetition=rep, arm=mode)
                    path = output/(stem+'_'+mode+'.json')
                    result = read_arm(path, signature)
                    if result is None:
                        result = arm_rollout(flags, actor, model, arm=mode, **kwargs)
                        atomic(path, dict(protocol_sha256=signature, result=result))
                    arms.append(result)
                verify(*arms)
                if rep == 0:
                    check = output/(stem+'_repeat_verified.json')
                    if check.exists():
                        if json.loads(check.read_text())['protocol_sha256'] != signature:
                            raise ValueError('Verification signature mismatch')
                    else:
                        repeated = arm_rollout(flags, actor, model, arm='stop', **kwargs)
                        verify(arms[0], repeated)
                        if arms[0]['returns'] != repeated['returns']:
                            raise AssertionError('Repeated STOP return mismatch')
                        atomic(check, dict(protocol_sha256=signature, status='PASS',
                                           seconds=repeated['seconds']))
                for i, (entry, stop, cont) in enumerate(zip(arms[0]['entries'], arms[0]['returns'], arms[1]['returns'])):
                    q = np.asarray(entry['features']) @ weight.T + bias
                    rows.append(dict(seed=task['seed'], stream=i, prefix=prefix, count=count,
                        repetition=rep, entry_hash=entry['hash'],
                        delta=cont['net_truncated']-stop['net_truncated'],
                        task_delta=cont['task_truncated']-stop['task_truncated'],
                        cost=cont['current_stage_cost']-stop['current_stage_cost'],
                        online=entry['delta'], ema=float(q[0]-q[1]), probability=entry['p_continue'],
                        tail_bound=arms[0]['tail_bounds'][i]+arms[1]['tail_bounds'][i]))
                atomic(output/'pairs.json', rows)
                run.log(dict(pairs=len(rows), elapsed_seconds=time.monotonic()-started))
        expected = task['batch_size']*len(task['prefixes'])*len(task['counts'])*task['repetitions']
        assert len(rows) == expected
        voc = paired_summary(rows)
        metrics = {'voc/'+k:v for k,v in voc.items() if isinstance(v,(int,float)) or v is None}
        atomic(output/'summary.json', dict(status='COMPLETE', protocol_sha256=signature,
            task=task, checkpoint=binding, voc=voc, metrics=metrics, seconds=time.monotonic()-started,
            job_id=os.environ['SLURM_JOB_ID'], wandb_url=run.url,
            scope='Paired utility only; broad predetermined samples; no repeated whole-policy scoring'))
        run.log(metrics);run.summary.update(metrics)
        progress('COMPLETE');run.finish(exit_code=0)
    except BaseException as exc:
        progress('FAILED', error=f'{type(exc).__name__}: {exc}')
        run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args();run_task(json.loads(args.task.read_text()))
