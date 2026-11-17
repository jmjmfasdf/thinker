"""Resume the audited 1M checkpoint in its existing directory and W&B run.

The cancelled branch is excluded. Preserve the original completed artifacts
before appending logs and replacing rolling checkpoints in the original run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import yaml


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Unexpected template: ' + before)
    return text.replace(before, after)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--template', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    template, out = args.template.resolve(), args.output.resolve()
    previous = json.loads((template / 'manifest.json').read_text())
    parent = Path(previous['parent_campaign'])
    parent_registry = json.loads((parent / 'manifest.json').read_text())
    old_item = next(r for r in parent_registry['runs'] if r['condition'] == 'burn16ln')
    audit = json.loads((parent / 'burn16ln/completion.json').read_text())
    assert audit['status'] == 'CONTINUATION_CHECKPOINT_OPTIMIZER_COUNTERS_WANDB_PASS'
    run = Path(old_item['run_directory'])
    for kind, info in audit['checkpoint'].items():
        assert sha(info['path']) == info['sha256']
        assert info['real_step'] == 1000080
    state = subprocess.check_output(['sacct', '-j', '56025', '-n', '-X', '-o', 'State'], text=True)
    assert state.strip().startswith('CANCELLED'), state
    out.mkdir(parents=True, exist_ok=False)
    (out / 'burn16ln').mkdir()
    snapshot = out / 'original_1m_snapshot'
    snapshot.mkdir()
    originals = {}
    for path in sorted(run.iterdir()):
        if not path.is_file():
            raise ValueError('Inspect unexpected run entry before copying: ' + str(path))
        # Milestone files are immutable and retained in place. All files that
        # the resumed process may append/replace receive a separate copy.
        if '_step_' in path.name:
            continue
        shutil.copy2(path, snapshot / path.name)
        originals[path.name] = sha(path)
        assert sha(snapshot / path.name) == originals[path.name]
    for name in ('source', 'evaluation_source_v2'):
        (out / name).symlink_to((template / name).resolve(), target_is_directory=True)
    for name in ('allocation.py', 'evaluate.slurm', 'finalize.py', 'finalize.slurm',
                 'finalize_normalized_planning_extension.py',
                 'summarize_adaptive_planning_evidence.py', 'summarize_adaptive_planning_pilot.py'):
        shutil.copy2(template / name, out / name)
    shutil.copy2(parent / 'burn16ln/completion.json', out / 'parent_completion.json')
    shutil.copy2(__file__, out / 'prepare.py')
    config = yaml.safe_load(Path(old_item['config']).read_text())
    assert config['total_steps'] == 1000000 and config['ckp'] is True
    config['total_steps'] = 2000000
    cfg = out / 'burn16ln/config.yaml'
    cfg.write_text(yaml.safe_dump(config, sort_keys=True))
    checkpoint_folder = out / 'parent_checkpoints'
    checkpoint_folder.mkdir()
    binding = {}
    for kind, info in audit['checkpoint'].items():
        target = checkpoint_folder / f'ckp_{kind}.tar'
        shutil.copy2(info['path'], target)
        assert sha(target) == info['sha256']
        binding[kind] = dict(info, immutable_copy=str(target))
    item = dict(previous['runs'][0], config=str(cfg), xpid=old_item['xpid'],
                run_directory=str(run), parent_checkpoint=binding,
                wandb_url=audit['wandb']['url'])
    manifest = dict(previous, status='REGISTERED_SAME_DIRECTORY_AND_WANDB_RESUME',
                    created_at=time.time(), runs=[item],
                    smoke_runs=previous['smoke_runs'], training_gpu='titanrtx_or_rtx3090',
                    training_resources='Two TITAN RTX / RTX 3090 GPUs on one node; 16 CPUs; 62000M.',
                    training_changes='Only cumulative budget changes; same xpid, source, algorithm, data, optimizer and W&B ID.',
                    original_run_snapshot=str(snapshot), original_run_hashes=originals,
                    excluded_job=56025, excluded_branch=previous['runs'][0]['run_directory'],
                    discarded_branch_policy='Do not load its weights, optimizer, counters or W&B history.',
                    smoke='Previously passed resume check retained; no new smoke training requested.',
                    historical_log_policy='Existing CSV rows are retained. Completion counters cover only rows after the audited 1M endpoint.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    s = (template / 'verify.py').read_text()
    s = replace_once(s,
        "if set(changes)!=({'total_steps','xpid'} | ({'ckp'} if not base.get('ckp') else set())):",
        "if set(changes)!={'total_steps'}:")
    s = replace_once(s,
        "            required = {'rtx3090':'3090','titanrtx':'TITAN RTX'}[manifest['training_gpu']]\n"
        "            if torch.cuda.device_count()!=2 or any(required not in torch.cuda.get_device_name(i) for i in range(2)):",
        "            if torch.cuda.device_count()!=2 or any(\n"
        "                    not any(n in torch.cuda.get_device_name(i) for n in ('TITAN RTX','RTX 3090'))\n"
        "                    for i in range(2)):")
    s = replace_once(s,
        "    if args.preflight:\n        for kind,info in item['parent_checkpoint'].items():",
        "    if args.preflight:\n"
        "        for name in ('logs.csv','logs_model.csv'):\n"
        "            if sha(run/name)!=manifest['original_run_hashes'][name]:\n"
        "                raise ValueError('original history changed before resume')\n"
        "        for kind,info in item['parent_checkpoint'].items():")
    s = replace_once(s,
        "        logs=list(csv.DictReader((run/'logs.csv').open()))",
        "        snapshot=Path(manifest['original_run_snapshot'])\n"
        "        prior=(snapshot/'logs.csv').read_bytes()\n"
        "        with (run/'logs.csv').open('rb') as stream:\n"
        "            if stream.read(len(prior))!=prior:\n"
        "                raise ValueError('original training history was modified')\n"
        "        logs=list(csv.DictReader((run/'logs.csv').open()))\n"
        "        logs=[r for r in logs if float(r['real_step'])>parent['actor']['real_step']]")
    (out / 'verify.py').write_text(s)
    # --resume=allow already uses flags.xpid as the explicit W&B run ID.
    # No training source or logging code change is necessary.
    shutil.copy2(template / 'train.slurm', out / 'train.slurm')
    immutable = [p for p in out.rglob('*') if p.is_file() and not p.is_symlink()]
    immutable += [out / 'source/SOURCE.sha256', out / 'evaluation_source_v2/SOURCE.sha256']
    (out / 'INPUTS.sha256').write_text(''.join(sha(p) + '  ' + str(p) + '\n' for p in immutable))
    # Resume loads config_c.yaml, so update its budget after backing it up.
    temporary = run / 'config_c.yaml.resume_tmp'
    shutil.copy2(cfg, temporary)
    temporary.replace(run / 'config_c.yaml')
    for kind, info in binding.items():
        temporary = run / f'ckp_{kind}.tar.resume_tmp'
        shutil.copy2(info['immutable_copy'], temporary)
        temporary.replace(run / f'ckp_{kind}.tar')
    print(json.dumps(dict(campaign=str(out), run=str(run), wandb_url=item['wandb_url'],
                          resume_step=1000080, target=2000000)))


if __name__ == '__main__':
    main()
