"""Freeze one fresh burn16 output-normalization test and a separate GPU smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    parent, output = args.parent.resolve(), args.output.resolve()
    diagnosis = json.loads((parent/'burn16_encoder_diagnostic/result/evidence.json').read_text())
    if diagnosis['status'] != 'IDENTICAL_INPUT_ENCODER_COMPARISON_PASS':
        raise ValueError('missing same-input encoder diagnosis')
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    paths = [line.split('  ', 1)[1] for line in (parent/'source/SOURCE.sha256').read_text().splitlines()]
    paths.append('thinker/tests/test_actor_encoder_scale.py')
    records = []
    for relative in sorted(set(paths)):
        destination = source/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/relative, destination)
        records.append(f'{sha(destination)}  {relative}\n')
    (source/'SOURCE.sha256').write_text(''.join(records))
    launch = (parent/'launch.slurm').read_text().replace(
        'assert torch.cuda.device_count() == 2',
        'assert torch.cuda.device_count() == 2\nassert all("RTX 3090" in torch.cuda.get_device_name(i) for i in range(2))\nassert f.actor_h_encoder_norm == "layer"')
    launch = launch.replace('export WANDB_MODE=online', 'export DI72_NUMERIC_STOP_AT_LOGIT=1000\nexport WANDB_MODE=online')
    guard = '''
if Path(sys.argv[1]).parent.name != 'smoke':
    import csv, math
    campaign = Path(sys.argv[1]).parents[1]
    manifest = json.loads((campaign/'manifest.json').read_text())
    smoke = next(r for r in manifest['runs'] if r['condition'] == 'smoke')
    folder = campaign.parents[3]/'logs/thinker'/smoke['xpid']
    checkpoints = {}
    for kind in ('actor','model'):
        path = max(folder.glob(f'ckp_{kind}.tar_step_*'), key=lambda p:int(p.name.rsplit('_',1)[1]))
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['real_step'] >= smoke['training_steps']
        assert saved['flags']['actor_h_encoder_norm'] == 'layer'
        assert all(torch.isfinite(v).all() for v in saved[kind+'_net_state_dict'].values())
        checkpoints[kind] = str(path)
        if kind == 'actor':
            assert saved['planning_training_contract']['version'] == 2
            assert saved['voc_gate_policy_schema_version'] == 4
            assert 'h_encoder.output_norm.weight' in saved['actor_net_state_dict']
    with (folder/'logs.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    keys = ['actor/icopro_nll','actor/icopro_numeric_h_encoder_pre_norm_absmax','actor/icopro_numeric_h_encoder_absmax',
            'actor/icopro_numeric_x_encoder_absmax','actor/icopro_numeric_real_state_encoder_absmax',
            'actor/icopro_numeric_tree_encoder_absmax','actor/icopro_numeric_policy_absmax']
    assert rows and all(math.isfinite(float(row[key])) for row in rows for key in keys)
    assert max(float(row['actor/icopro_nll']) for row in rows) < 10
    assert max(float(row['actor/icopro_numeric_h_encoder_pre_norm_absmax']) for row in rows) > 0
    assert all(max(float(row[key]) for row in rows) > 0 for key in keys[3:])
    (campaign/'smoke_checkpoint_audit.json').write_text(json.dumps(dict(status='SMOKE_TRAINING_CHECKPOINT_AND_TELEMETRY_PASS',checkpoints=checkpoints,updates=len(rows)),indent=2)+'\\n')
'''
    launch = launch.replace("print(json.dumps(dict(preflight='passed'", guard+"print(json.dumps(dict(preflight='passed'")
    (output/'launch.slurm').write_text(launch)
    shutil.copy2(__file__, output/'prepare.py')
    base = yaml.safe_load((parent/'burn16/config.yaml').read_text())
    runs = []
    for condition in ('burn16ln', 'smoke'):
        folder = output/condition
        folder.mkdir()
        config = dict(base, actor_h_encoder_norm='layer',
                      xpid=f'enduro-s001-all-adapt-{condition}-hencnorm-s701-{output.name.split("_")[-1]}')
        if condition == 'smoke':
            config.update(total_steps=2048, checkpoint_interval=1024, model_warm_up_n=128)
        (folder/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=True))
        runs.append(dict(condition=condition, config=str(folder/'config.yaml'),
                         xpid=config['xpid'], training_steps=config['total_steps']))
    manifest = dict(runs=runs, parent=str(parent),
        changed_factor='LayerNorm after actor h_encoder output; raw actor/world-model/tree features otherwise retain their configured paths.',
        unchanged='Fresh seed701, Enduro sub001 sessions1-4, common+contrast critic, epsilon0.1, tau0.01, burn16, GRU multiplier1, scored4targets/sequence, min1/max30, cost0.0001.',
        diagnostic_source=str(parent/'burn16_encoder_diagnostic/result/evidence.json'),
        diagnostic_sha256=sha(parent/'burn16_encoder_diagnostic/result/evidence.json'),
        scope='One stability ablation after sustained encoder amplification. Finite stable training is necessary, but does not establish useful adaptive planning.',
        numerical_stop='Capture and fail if BC action logits reach magnitude1000; no clipping, loss replacement, or skipped updates.',
        source_sha256=sha(source/'SOURCE.sha256'))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    files = [output/'launch.slurm', output/'prepare.py', output/'manifest.json', source/'SOURCE.sha256']
    files += [Path(run['config']) for run in runs]
    (output/'INPUTS.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(output)}\n' for p in files))
    print(output)


if __name__ == '__main__':
    main()
