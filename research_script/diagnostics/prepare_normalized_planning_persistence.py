"""Freeze an unchanged burn16 continuation from its audited 1M endpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import yaml


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent, out = args.parent.resolve(), args.output.resolve()
    parent_manifest = json.loads((parent / 'manifest.json').read_text())
    item = next(r for r in parent_manifest['runs'] if r['condition'] == 'burn16ln')
    audit_path = parent / 'burn16ln/completion.json'
    audit = json.loads(audit_path.read_text())
    if (audit['status'] != 'CONTINUATION_CHECKPOINT_OPTIMIZER_COUNTERS_WANDB_PASS'
            or audit['health']['bc_history_status'] != 'NO_LARGE_NLL_OBSERVED'
            or audit['wandb']['state'] != 'finished'):
        raise ValueError('parent training audit failed')
    config = yaml.safe_load(Path(item['config']).read_text())
    for key, value in {'ckp': True, 'total_steps': 1000000, 'actor_h_encoder_norm': 'layer',
                       'icopro_burn_in_steps': 16, 'voc_gru_gradient_scale': 1.0,
                       'voc_gate_q_temperature': .01, 'voc_train_epsilon': .1,
                       'min_search_steps': 1, 'max_search_steps': 30,
                       'think_cost': .0001, 'schedule_total_steps': 100000000,
                       'checkpoint_interval': 50000}.items():
        if config[key] != value:
            raise ValueError('unexpected parent setting: ' + key)
    for info in audit['checkpoint'].values():
        if sha(info['path']) != info['sha256'] or not 1000000 <= info['real_step'] < 1002000:
            raise ValueError('parent checkpoint content or milestone changed')
    out.mkdir(parents=True, exist_ok=False)
    for directory in ('source', 'evaluation_source_v2'):
        (out / directory).symlink_to((parent / directory).resolve(), target_is_directory=True)
    registered = []
    for name in ('train.slurm', 'allocation.py', 'summarize_adaptive_planning_pilot.py'):
        shutil.copy2(parent / name, out / name)
        registered.append(out / name)
    # The parent is already a continuation: ckp=True stays unchanged. All
    # algorithm, data, schedule, model and optimizer settings remain fixed.
    verify = (parent / 'verify.py').read_text()
    before = "if set(changes)!={'ckp','total_steps','xpid'}:"
    after = "if set(changes)!=({'total_steps','xpid'} | ({'ckp'} if not base.get('ckp') else set())):"
    if verify.count(before) != 1:
        raise ValueError('unexpected verifier source')
    (out / 'verify.py').write_text(verify.replace(before, after))
    evaluation = (parent / 'evaluate.slurm').read_text()
    if evaluation.count('500000|1000000') != 1 or evaluation.count('[ "$MILESTONE" = 1000000 ]') != 1:
        raise ValueError('unexpected evaluation launcher')
    (out / 'evaluate.slurm').write_text(evaluation.replace('500000|1000000', '1500000|2000000')
                                      .replace('[ "$MILESTONE" = 1000000 ]', '[ "$MILESTONE" = 2000000 ]'))
    shutil.copy2(audit_path, out / 'parent_completion.json')
    shutil.copy2(__file__, out / 'prepare.py')
    diagnostics = Path(__file__).resolve().parent
    shutil.copy2(diagnostics / 'finalize_normalized_planning_persistence.py', out / 'finalize.py')
    for name in ('finalize_normalized_planning_extension.py', 'summarize_adaptive_planning_evidence.py'):
        shutil.copy2(parent / 'completion_report_001' / name, out / name)
        registered.append(out / name)
    immutable = out / 'parent_checkpoints/burn16ln'
    immutable.mkdir(parents=True)
    binding = {}
    for kind, info in audit['checkpoint'].items():
        saved = immutable / f'ckp_{kind}.tar'
        shutil.copy2(info['path'], saved)
        binding[kind] = dict(info, immutable_copy=str(saved))
        registered.append(saved)
    runs, smokes = [], []
    for phase, target in [('smoke', 1020000), ('full', 2000000)]:
        folder = out / ('validation_burn16ln' if phase == 'smoke' else 'burn16ln')
        folder.mkdir()
        xpid = f'enduro-s001-all-norm-burn16ln-to2m-{phase}-s701-20261116'
        destination = Path(config['savedir']) / xpid
        destination.mkdir(exist_ok=False)
        current = dict(config, xpid=xpid, total_steps=target)
        config_path = folder / 'config.yaml'
        config_path.write_text(yaml.safe_dump(current, sort_keys=True))
        shutil.copy2(config_path, destination / 'config_c.yaml')
        for kind, info in binding.items():
            shutil.copy2(info['immutable_copy'], destination / f'ckp_{kind}.tar')
        entry = dict(condition='burn16ln', config=str(config_path), xpid=xpid,
                     run_directory=str(destination), training_steps=target,
                     parent_checkpoint=binding, parent_config=item['config'],
                     parent_completion_audit=str(audit_path),
                     actor_additional_budget=target-binding['actor']['real_step'],
                     wandb_url='https://wandb.ai/jmme425/thinker/runs/' + xpid)
        (smokes if phase == 'smoke' else runs).append(entry)
        registered.append(config_path)
    manifest = dict(status='REGISTERED_BURN16_1M_TO_2M_PERSISTENCE', created_at=time.time(),
                    runs=runs, smoke_runs=smokes, parent_campaign=str(parent),
                    milestones=[1500000, 2000000], utility_endpoint=2000000,
                    training_changes='NONE; only output identity and cumulative training budget change.',
                    training_gpu='titanrtx', training_resources='Two 24GB TITAN RTX GPUs, 16 CPUs, 62000M.',
                    allocation_protocol=parent_manifest['allocation_protocol'],
                    paired_protocol=parent_manifest['paired_protocol'],
                    evaluation_resources='RTX2080Ti excluding compute02', score_evaluation='DISABLED',
                    restart_semantics=parent_manifest['restart_semantics'],
                    source_sha256=sha(out / 'source/SOURCE.sha256'),
                    evaluation_source_sha256=sha(out / 'evaluation_source_v2/SOURCE.sha256'),
                    smoke='Separate 20k resume validation; main run starts from the original audited 1M parent.',
                    rationale='Burn16 has a broader 1M distribution and promising but inconclusive paired utility; earlier 0.5M capping and GRU10 collapse require persistence checks.',
                    decision='Report both 1.5M and 2M. No intermediate checkpoint selection, no training parameter change, no forced histogram shaping.',
                    limitations=['Candidate selected after inspecting 8/16 pilot results; this is exploratory.',
                                 'One training seed, Enduro sub001 all sessions.',
                                 'Environment, actor rollout memory and world-model replay restart cold.',
                                 'Matching expected CONTINUE frequency does not match total computation.'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report_launcher = (parent / 'completion_report_001/finalize_normalized_planning_extension.slurm').read_text()
    before = 'exec python -u "$PLAN/finalize_normalized_planning_extension.py" --campaign "$CAMPAIGN" --output "$PLAN/result"'
    after = 'exec python -u "$PLAN/finalize.py" --campaign "$CAMPAIGN" --output "$PLAN/result"'
    if report_launcher.count(before) != 1:
        raise ValueError('unexpected report launcher')
    (out / 'finalize.slurm').write_text(report_launcher.replace(before, after))
    registered.extend(out / name for name in ('verify.py', 'evaluate.slurm', 'prepare.py', 'finalize.py',
                      'finalize.slurm', 'parent_completion.json', 'manifest.json',
                      'source/SOURCE.sha256', 'evaluation_source_v2/SOURCE.sha256'))
    (out / 'INPUTS.sha256').write_text(''.join(sha(path) + '  ' + str(path) + '\n' for path in registered))
    print(json.dumps(dict(campaign=str(out), parent_step=binding['actor']['real_step'],
                          target=2000000, additional_budget=runs[0]['actor_additional_budget'])))


if __name__ == '__main__':
    main()
