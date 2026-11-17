"""Freeze a burn8 comparison using the exact source of the normalized burn16 trial.

Preparation never submits work. The launcher requires a stable completed
burn16 trial and reuses its GPU smoke, but starts burn8 training from scratch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


GUARD = r'''import csv, hashlib, json, math, statistics, sys, time
from pathlib import Path
import torch
campaign = Path(sys.argv[1])
manifest = json.loads((campaign/'manifest.json').read_text())
reference = manifest['matched_reference']
config = Path(reference['config'])
assert hashlib.sha256(config.read_bytes()).hexdigest() == reference['config_sha256']
folder = campaign.parents[3]/'logs/thinker'/reference['xpid']
with (folder/'logs.csv').open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) >= 20 and float(rows[-1]['real_step']) >= reference['training_steps']
keys = ['actor/icopro_nll', 'actor/icopro_numeric_h_encoder_pre_norm_absmax',
        'actor/icopro_numeric_h_encoder_absmax']
assert all(math.isfinite(float(row[key])) for row in rows[-20:] for key in keys)
median = statistics.median(float(row['actor/icopro_nll']) for row in rows[-20:])
assert median < 20, 'normalized burn16 failed the final numerical health screen'
all_nll = [float(row['actor/icopro_nll']) for row in rows]
assert all(math.isfinite(value) for value in all_nll), 'nonfinite loss in reference history'
peak_rolling = max(statistics.median(all_nll[i-20:i]) for i in range(20,len(all_nll)+1))
assert peak_rolling < 20, 'reference recovered from sustained large loss; requires diagnosis before comparison'
bindings = {}
for kind in ('actor', 'model'):
    paths = [p for p in folder.glob(f'ckp_{kind}.tar_step_*')
             if reference['training_steps'] <= int(p.name.rsplit('_',1)[1])
                < reference['training_steps']+2000]
    assert paths, 'missing immutable final checkpoint'
    path = min(paths, key=lambda p:int(p.name.rsplit('_',1)[1]))
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['real_step'] >= reference['training_steps']
    assert saved['flags']['actor_h_encoder_norm'] == 'layer'
    assert saved['flags']['icopro_burn_in_steps'] == 16
    assert all(torch.isfinite(v).all() for v in saved[kind+'_net_state_dict'].values())
    if kind == 'actor':
        assert saved['planning_training_contract']['version'] == 2
        assert saved['voc_gate_policy_schema_version'] == 4
        assert 'h_encoder.output_norm.weight' in saved['actor_net_state_dict']
    bindings[kind] = dict(path=str(path), real_step=saved['real_step'],
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(campaign/'reference_health_audit.json').write_text(json.dumps(dict(
    status='REFERENCE_NUMERICAL_SCREEN_PASS', checked_at=time.time(),
    last20_nll_median=median, maximum20_nll_median=peak_rolling, checkpoints=bindings,
    scope='A numerical screen only. No reference weights are loaded into the fresh comparison.'
), indent=2)+'\n')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference, output = args.reference.resolve(), args.output.resolve()
    previous = json.loads((reference/'manifest.json').read_text())
    item = next(r for r in previous['runs'] if r['condition'] == 'burn16ln')
    smoke = next(r for r in previous['runs'] if r['condition'] == 'smoke')
    base = yaml.safe_load(Path(item['config']).read_text())
    assert base['actor_h_encoder_norm'] == 'layer'
    assert base['icopro_burn_in_steps'] == base['icopro_sampling_context'] == 16
    output.mkdir(parents=True, exist_ok=False)
    for name in ('source', 'evaluation_source_v2'):
        shutil.copytree(reference/name, output/name)
        for line in (output/name/'SOURCE.sha256').read_text().splitlines():
            checksum, relative = line.split('  ', 1)
            assert sha(output/name/relative) == checksum
    condition = 'burn8ln'
    folder = output/condition
    folder.mkdir()
    config = dict(base, icopro_burn_in_steps=8,
        xpid=f'enduro-s001-all-adapt-burn8ln-hencnorm-s701-{output.name.split("_")[-1]}')
    differences = {k: [base.get(k), config.get(k)] for k in set(base)|set(config)
                   if base.get(k) != config.get(k)}
    assert set(differences) == {'icopro_burn_in_steps', 'xpid'}
    (folder/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=True))
    launch = (reference/'launch.slurm').read_text()
    original = 'exec python -u thinker/train.py --config "$CONFIG"'
    assert launch.count(original) == 1
    launch = launch.replace(original,
        'python "$CAMPAIGN/reference_health_guard.py" "$CAMPAIGN"\n'+original)
    (output/'launch.slurm').write_text(launch)
    (output/'reference_health_guard.py').write_text(GUARD)
    for name in ('evaluate_v2.slurm', 'evaluate_final.slurm'):
        shutil.copy2(reference/name, output/name)
    shutil.copy2(__file__, output/'prepare.py')
    run = dict(condition=condition, config=str(folder/'config.yaml'),
               xpid=config['xpid'], training_steps=config['total_steps'])
    manifest = dict(runs=[run, smoke], reference=str(reference),
        matched_reference=dict(item, config_sha256=sha(Path(item['config']))),
        changed_factors=differences, source_sha256=sha(output/'source/SOURCE.sha256'),
        training='Fresh initialization; shared reference smoke only. No checkpoint warm start.',
        comparison='Both8and16 use identical normalized architecture, source, seed and target sampling population.',
        requirements=['RTX3090 training with W&B; RTX2080Ti evaluation.',
            'Start only after reference training succeeds and passes numerical screen.',
            'Zero-exploration allocation and paired utility at the fixed250k endpoint.',
            'No separate full-game scoring. Distribution breadth alone is not evidence of useful planning.'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    files = ['launch.slurm', 'reference_health_guard.py', 'evaluate_v2.slurm',
             'evaluate_final.slurm', 'prepare.py', 'manifest.json',
             'source/SOURCE.sha256', 'evaluation_source_v2/SOURCE.sha256',
             condition+'/config.yaml']
    (output/'INPUTS.sha256').write_text(''.join(f'{sha(output/p)}  {p}\n' for p in files))
    print(json.dumps(dict(output=str(output), differences=differences), indent=2))


if __name__ == '__main__':
    main()
