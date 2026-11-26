"""Prepare a fresh h+x output-LayerNorm replacement of an existing 18-run grid."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    previous, out = args.previous.resolve(), args.output.resolve()
    old = json.loads((previous/'manifest.json').read_text())
    assert len(old['runs']) == 18
    out.mkdir(parents=True, exist_ok=False)
    source = out/'source'
    shutil.copytree((previous/'source').resolve(), source)
    changed = ['thinker/thinker/actor_net.py', 'thinker/thinker/util.py',
               'thinker/thinker/planning_protocol.py', 'thinker/thinker/learn_actor.py',
               'thinker/thinker/numeric_diagnostics.py', 'thinker/tests/test_actor_encoder_scale.py']
    for relative in changed:
        shutil.copy2(root/relative, source/relative)
    records = [line.split('  ', 1)[1] for line in (source/'SOURCE.sha256').read_text().splitlines()]
    (source/'SOURCE.sha256').write_text(''.join(f'{sha(source/rel)}  {rel}\n' for rel in records))
    shutil.copy2(__file__, out/'prepare.py')
    (out/'slurm').mkdir()
    (out/'audit').mkdir()
    runs = []
    for record in old['runs']:
        config = yaml.safe_load(Path(record['config']).read_text())
        assert config['actor_h_encoder_norm'] == 'layer'
        assert config.get('actor_x_encoder_norm', 'none') == 'none'
        xpid = f"{record['game']}-s{record['subject']:03d}-all-burn16hxln-gru10-10m-s701-20261119"
        new_config = dict(config, xpid=xpid, actor_h_encoder_norm='layer', actor_x_encoder_norm='layer')
        assert {k for k in new_config if new_config[k] != config.get(k)} == {'xpid', 'actor_x_encoder_norm'}
        folder = out/f"{record['index']:02d}_{record['game']}_sub{record['subject']:03d}"
        folder.mkdir()
        cfg = folder/'config.yaml'
        cfg.write_text(yaml.safe_dump(new_config, sort_keys=True))
        run_directory = Path(config['savedir'])/xpid
        assert not run_directory.exists()
        updated = dict(record, xpid=xpid, config=str(cfg), run_directory=str(run_directory))
        for entry in updated['data_files']:
            path = Path(entry['path'])
            assert path.stat().st_size == entry['size']
            entry['sha256'] = sha(path)
        runs.append(updated)
    (out/'config_paths.txt').write_text(''.join(r['config']+'\n' for r in runs))
    with (out/'runs.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['index','game','subject','xpid','config','run_directory'], extrasaction='ignore')
        w.writeheader(); w.writerows(runs)
    launch = (previous/'launch.slurm').read_text()
    assert launch.startswith('#!/bin/bash\n')
    launch = launch.replace('#!/bin/bash\n',
                            '#!/bin/bash\n'
                            '# W&B service sockets belong to the submitting host, not this worker.\n'
                            'unset WANDB_SERVICE _WANDB_SERVICE\n', 1)
    launch = launch.replace('assert f.actor_h_encoder_norm == "layer"',
                            'assert f.actor_h_encoder_norm == f.actor_x_encoder_norm == "layer"\nassert planning_training_contract(f)["version"] == 3')
    (out/'launch.slurm').write_text(launch)
    verify = (previous/'verify.py').read_text()
    verify = verify.replace("assert f.actor_h_encoder_norm=='layer' and f.voc_gru_gradient_scale==10.",
                            "assert f.actor_h_encoder_norm==f.actor_x_encoder_norm=='layer' and f.voc_gru_gradient_scale==10.\n        assert planning_training_contract(f)['version']==3")
    verify = verify.replace("Frozen runtime unchanged;19 normalization/gradient/protocol tests passed previously.",
                            "See audit/unit_tests.log and audit/gpu_smoke.json for this revised runtime.")
    (out/'verify.py').write_text(verify)
    manifest = dict(old, created_at=time.time(), reference=str(previous), runs=runs,
                    superseded_job_ids=[56524,56525,56526],
                    source_sha256=sha(source/'SOURCE.sha256'),
                    source_changes=changed,
                    changed_factor='Add x encoder output LayerNorm; preserve existing h encoder output LayerNorm. CNN internal layers unchanged.',
                    conditions=old['conditions'].replace('h-encoder output LayerNorm', 'h- and x-encoder output LayerNorm'),
                    planning_contract_version=3,
                    telemetry='Record h/x features before and after LayerNorm plus learned affine magnitudes, losses, logits, and existing VOC/GRU diagnostics.',
                    wandb=dict(old['wandb'], group=out.name),
                    status='PREPARED_SUBMISSION_RECORDED_SEPARATELY')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    inputs=[out/'prepare.py',out/'manifest.json',out/'launch.slurm',out/'verify.py',
            out/'runs.csv',out/'config_paths.txt',source/'SOURCE.sha256']
    inputs += [Path(r['config']) for r in runs]
    (out/'INPUTS.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in inputs))
    print(json.dumps({'campaign':str(out),'runs':len(runs),'source':str(source)},indent=2))


if __name__ == '__main__':
    main()
