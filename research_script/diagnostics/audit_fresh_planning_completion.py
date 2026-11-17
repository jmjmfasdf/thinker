"""Bind fresh pilot completion to logs, saved optimizers, evaluation and W&B."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import torch
import wandb
import yaml

from audit_planning_training_completion import finite_tensors
from summarize_adaptive_planning_pilot import loss_health
from thinker.planning_protocol import validate_planning_training_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    c = args.campaign.resolve()
    item = next(r for r in json.loads((c/'manifest.json').read_text())['runs']
                if r['condition'] == args.condition)
    run = c.parents[3]/'logs/thinker'/item['xpid']
    job = json.loads((c/'submissions.json').read_text())[args.condition]['job_id']
    scheduler = subprocess.check_output(['sacct', '-j', str(job), '-X', '-n', '-P',
        '--format=JobID,State,ExitCode,Elapsed,NodeList'], text=True).strip().split('|')
    if scheduler[1:3] != ['COMPLETED', '0:0'] or not (run/'finish').exists():
        raise ValueError('fresh training did not successfully finish')
    config = yaml.safe_load(Path(item['config']).read_bytes())
    summary = json.loads((c/f'evaluation_{args.condition}/summary.json').read_text())
    loaded, binding, tensors = {}, {}, {}
    for kind in ('actor', 'model'):
        path = Path(summary['checkpoint'][kind]['path'])
        raw = path.read_bytes()
        saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != summary['checkpoint'][kind]['sha256']:
            raise ValueError('evaluation checkpoint changed')
        step = saved['real_step']
        if not item['training_steps'] <= step < item['training_steps']+2000:
            raise ValueError('final budget mismatch')
        if step != int(path.name.rsplit('_', 1)[1]):
            raise ValueError('checkpoint filename/counter mismatch')
        tensors[kind] = finite_tensors({k:v for k,v in saved.items() if 'state_dict' in k})
        loaded[kind] = saved
        binding[kind] = dict(path=str(path), sha256=digest, real_step=step)
    actor = loaded['actor']
    contract = validate_planning_training_checkpoint(actor, config)
    for key in ('xpid', 'name', 'base_seed', 'total_steps', 'icopro_burn_in_steps',
                'icopro_sampling_context', 'voc_gru_gradient_scale', 'actor_h_encoder_norm'):
        if actor['flags'].get(key) != loaded['model']['flags'].get(key):
            raise ValueError('actor/model flag mismatch: '+key)
    raw_logs = (run/'logs.csv').read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw_logs.decode())))
    if not rows or any(None in r or None in r.values() for r in rows):
        raise ValueError('incomplete logs')
    if int(float(rows[-1]['real_step'])) != actor['real_step']:
        raise ValueError('final logs/checkpoint mismatch')
    mapping = dict(imitation_update_count='actor/optimizer_stepped',
                   voc_update_count='actor/voc_optimizer_stepped',
                   voc_continue_count='actor/voc_continue_support',
                   voc_stop_count='actor/voc_stop_support')
    counters = {}
    for key, column in mapping.items():
        observed = sum(float(row[column]) for row in rows)
        if observed != actor[key]:
            raise ValueError('fresh counter/log mismatch: '+key)
        counters[key] = observed
    if any(float(r['actor/icopro_count']) != config['icopro_batch_size']*config['batch_length'] for r in rows):
        raise ValueError('scored BC target count changed')
    remote = wandb.Api(timeout=30).run('jmme425/thinker/'+item['xpid'])
    keys = ['real_step', 'actor/icopro_nll', 'actor/total_loss',
            'gradient/voc_gru_candidate_norm', 'gradient/voc_gru_applied_norm',
            'gradient/voc_gru_gradient_scale']
    metrics = {k:remote.summary.get(k) for k in keys}
    if remote.state != 'finished' or metrics['real_step'] != actor['real_step']:
        raise ValueError('W&B has not finished at final step')
    for key in keys:
        if abs(float(metrics[key])-float(rows[-1][key])) > 1e-8:
            raise ValueError('W&B/local metric mismatch: '+key)
    result = dict(status='FRESH_BUDGET_LOGS_OPTIMIZERS_EVALUATION_AND_WANDB_PASS',
        checked_at=time.time(), condition=args.condition, scheduler=scheduler,
        checkpoint=binding, finite_saved_tensor_counts=tensors, contract=contract,
        counters=counters, bc_targets=sum(float(r['actor/icopro_count']) for r in rows),
        health=loss_health(rows), logs_sha256=hashlib.sha256(raw_logs).hexdigest(),
        wandb=dict(url=remote.url, state=remote.state, final_metrics=metrics),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='Training completeness and numerical stability do not establish planning utility.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
