"""Verify a bounded resume run before authorizing longer continuations."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'thinker'))
from thinker import util
from thinker.planning_protocol import validate_planning_training_checkpoint


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optimizer_step(state):
    return max(float(v.get('step', 0)) for v in state['state'].values())


def check_finite(value):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite checkpoint tensor')
    elif isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_finite(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = args.validation.resolve()
    output = directory/'completion_audit.json'
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((directory/'manifest.json').read_text())
    submission = json.loads((directory/'submission.json').read_text())
    job = submission['job_id']
    scheduler = subprocess.check_output(['sacct', '-j', str(job), '-X', '-n', '-P',
        '--format=JobID,State,ExitCode,Elapsed,NodeList'], text=True)
    row = next(r.split('|') for r in scheduler.splitlines() if r.startswith(str(job)+'|'))
    if row[1:3] != ['COMPLETED', '0:0']:
        raise RuntimeError('resume job did not finish successfully: '+str(row))
    run = Path(manifest['run_directory'])
    if not (run/'finish').exists():
        raise ValueError('training finish marker missing')
    with (run/'logs.csv').open() as stream:
        logs = list(csv.DictReader(stream))
    steps = [int(r['real_step']) for r in logs]
    if not steps or steps[0] <= manifest['baseline']['real_step'] or any(
            right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError('resume log counters did not continue monotonically')
    checkpoints, report = {}, {}
    for kind, binding in manifest['input_checkpoints'].items():
        parent_path = Path(binding['path'])
        if sha(parent_path) != binding['sha256']:
            raise ValueError('original parent checkpoint changed')
        parent = torch.load(parent_path, map_location='cpu', weights_only=False)
        path = max(run.glob(f'ckp_{kind}.tar_step_*'),
                   key=lambda p: int(p.name.rsplit('_', 1)[1]))
        raw = path.read_bytes()
        current = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
        if current['real_step'] < manifest['total_steps'] or current['real_step'] != int(
                path.name.rsplit('_', 1)[1]):
            raise ValueError('incomplete or incorrectly named checkpoint')
        state_key = f'{kind}_net_state_dict'
        state = current[state_key]
        check_finite(state)
        changed = sum(not torch.equal(v, parent[state_key][k]) for k, v in state.items())
        if not changed:
            raise ValueError('weights did not change during resumed training')
        optimizers = {}
        for key in current:
            if 'optimizer' in key and key.endswith('state_dict') and current[key]:
                check_finite(current[key])
                if not current[key]['state']:
                    if (key != 'voc_gate_optimizer_state_dict' or parent[key]['state']
                            or not current['flags']['voc_gate_exact_projection']):
                        raise ValueError('unexpected inactive optimizer: '+key)
                    optimizers[key] = dict(status='inactive_exact_gate_projection')
                    continue
                before, after = optimizer_step(parent[key]), optimizer_step(current[key])
                if after <= before:
                    raise ValueError('optimizer did not advance: '+key)
                optimizers[key] = dict(parent_step=before, resumed_step=after)
        report[kind] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
            parent_step=parent['real_step'], resumed_step=current['real_step'],
            finite_state_tensors=len(state), changed_state_tensors=changed, optimizers=optimizers)
        checkpoints[kind] = current
    actor = checkpoints['actor']
    flags = argparse.Namespace(**actor['flags'])
    validate_planning_training_checkpoint(actor)
    util.validate_voc_gate_policy_checkpoint(actor, flags=flags)
    for name in ('planning_training_contract', 'imitation_data_signature'):
        if actor[name] != manifest['baseline'][name]:
            raise ValueError('resume changed '+name)
    for name in ('imitation_update_count', 'voc_update_count'):
        if actor[name] - manifest['baseline'][name] != len(logs):
            raise ValueError('update counter differs from resumed log rows: '+name)
    if not actor['imitation_rng_state'] or steps[-1] != actor['real_step']:
        raise ValueError('missing BC RNG or inconsistent final log counter')
    if flags.schedule_total_steps != manifest['schedule_total_steps']:
        raise ValueError('schedule horizon changed')
    for key in ('actor/icopro_nll', 'actor/total_loss', 'actor/learning_rate'):
        if not all(torch.isfinite(torch.tensor(float(r[key]))) for r in logs):
            raise ValueError('nonfinite training metric: '+key)
    import wandb
    remote = wandb.Api(timeout=60).run('jmme425/thinker/'+manifest['xpid'])
    summary = dict(remote.summary)
    if remote.state != 'finished' or int(summary['real_step']) != actor['real_step']:
        raise ValueError('W&B has not finished at the final actor counter')
    for key in ('actor/icopro_nll', 'actor/total_loss'):
        if not torch.isfinite(torch.tensor(float(summary[key]))):
            raise ValueError('missing/nonfinite W&B metric: '+key)
    collector = json.loads((run/'collector_metrics.json').read_text())
    result = dict(status='RESUME_VALIDATION_PASS', verified_at=time.time(),
        job_id=job, scheduler=row, checkpoint=report, resumed_actor_updates=len(logs),
        first_logged_step=steps[0], final_logged_step=steps[-1],
        actor_additional_consumed_transitions=actor['real_step']-manifest['baseline']['real_step'],
        physical_collected_transitions=collector['physical_env_steps'],
        original_parent_hashes='UNCHANGED', training_contract=actor['planning_training_contract'],
        final_bc_updates=actor['imitation_update_count'], final_voc_updates=actor['voc_update_count'],
        final_learning_rate=float(logs[-1]['actor/learning_rate']),
        wandb=dict(state=remote.state, url=remote.url, summary_metric_count=len(summary),
                   metrics={k:summary[k] for k in ('real_step','actor/icopro_nll','actor/total_loss')}),
        restart_semantics=manifest['restart_semantics'],
        efficacy_status='Resume correctness only; this run is excluded from efficacy comparisons.',
        script_sha256=sha(Path(__file__)))
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
