"""Freeze a fresh Enduro pilot and a worst-case-memory smoke; never submit implicitly."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'research_script/outputs/00/enduro_sub001_all_min1_cost0001_20261111'


LAUNCH = r'''#!/bin/bash
set -euo pipefail
CONFIG="$(realpath "${1:?config path}")"
RUN="$(dirname "$CONFIG")"
CAMPAIGN="$(dirname "$RUN")"
SOURCE="$CAMPAIGN/source"
JOBTMP="/scratch/$USER/ap${SLURM_JOB_ID}"
ARCHIVE=/work/jeongmin/conda-envs/dynamic-thinker-py310.tar.zst
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:?}"
test "${#DEVICES[@]}" -eq 2
mkdir -p "$JOBTMP/e" "$JOBTMP/tmp" "$JOBTMP/w/cache" "$JOBTMP/w/data"
cd "$SOURCE"
sha256sum --quiet -c SOURCE.sha256
cd "$CAMPAIGN"
sha256sum --quiet -c INPUTS.sha256
ACTUAL=$(sha256sum "$ARCHIVE" | awk '{print $1}')
test "$ACTUAL" = 0221345519333f783c413adca1240991fe2fcdc1a563bf312d4e57001c6b3714
tar --zstd -xf "$ARCHIVE" -C "$JOBTMP/e"
"$JOBTMP/e/bin/conda-unpack"
export PATH="$JOBTMP/e/bin:$PATH" PYTHONPATH="$SOURCE/thinker"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$JOBTMP/tmp" RAY_TMPDIR="$JOBTMP/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_USER=jmme425 WANDB_SILENT=true
export WANDB_RUN_GROUP="$(basename "$CAMPAIGN")" WANDB_DIR="$RUN"
export WANDB_CACHE_DIR="$JOBTMP/w/cache" WANDB_DATA_DIR="$JOBTMP/w/data"
export WANDB_MODE=online CUBLAS_WORKSPACE_CONFIG=:4096:8
unset WANDB_API_KEY WANDB_DISABLED
cd "$SOURCE"
python - "$CONFIG" <<'PY'
import sys, json, torch
from pathlib import Path
from thinker import util, cenv
from thinker.planning_protocol import planning_training_contract
f = util.create_setting(args=['--config', sys.argv[1]], save_flags=False)
assert not f.ckp and not any((f.preload, f.preload_actor, f.voc_parent_checkpoint))
assert f.min_search_steps == 1 and f.think_cost == .0001
assert f.icopro_train_sessions == '1,2,3,4' and f.icopro_holdout_sessions == ''
assert f.voc_gru_connected and f.dynamic_search_state_reset == 'episode'
assert f.voc_gate_q_temperature == .01 and f.voc_gate_exact_projection
assert f.use_wandb and f.icopro_batch_size == 16 and f.batch_length == 4
assert torch.cuda.device_count() == 2
assert str(Path(cenv.__file__).resolve()).startswith(str(Path.cwd()))
print(json.dumps(dict(preflight='passed', xpid=f.xpid, contract=planning_training_contract(f),
    gpus=[torch.cuda.get_device_name(i) for i in range(2)])), flush=True)
PY
exec python -u thinker/train.py --config "$CONFIG"
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=250_000)
    args = ap.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    paths = [line.split('  ', 1)[1] for line in (REFERENCE/'source/SOURCE.sha256').read_text().splitlines()]
    paths += ['thinker/thinker/planning_protocol.py', 'thinker/tests/test_planning_training_revision.py']
    records = []
    for relative in sorted(set(paths)):
        origin = ROOT/relative
        target = source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        records.append(f'{sha(target)}  {relative}\n')
    (source/'SOURCE.sha256').write_text(''.join(records))
    shutil.copy2(__file__, output/'prepare.py')
    (output/'launch.slurm').write_text(LAUNCH)
    reference = yaml.safe_load((REFERENCE/'config.yaml').read_text())
    # Shared target population even for the 1-step control. Training starts from
    # scratch; the existing 10M run remains an unmatched historical comparison.
    conditions = [
        ('control', 'policy_centered_v1', .02, 1, 1.),
        ('common', 'common_contrast_v1', .02, 1, 1.),
        ('explore', 'common_contrast_v1', .10, 1, 1.),
        ('burn8', 'common_contrast_v1', .10, 8, 1.),
        ('burn16', 'common_contrast_v1', .10, 16, 1.),
        ('gru10', 'common_contrast_v1', .10, 8, 10.),
        ('smoke', 'common_contrast_v1', .10, 16, 10.),
    ]
    runs = []
    for name, critic, epsilon, burn, scale in conditions:
        config = dict(reference)
        config.update(voc_critic_parameterization=critic, voc_train_epsilon=epsilon,
            icopro_burn_in_steps=burn, icopro_sampling_context=16,
            voc_gru_gradient_scale=scale, total_steps=args.steps,
            checkpoint_interval=50_000,
            xpid=f'enduro-s001-all-adapt-{name}-s701-{output.name.split("_")[-1]}')
        if name == 'smoke':
            config.update(total_steps=2048, checkpoint_interval=1024, model_warm_up_n=128)
        run = output/name
        run.mkdir()
        (run/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=True))
        runs.append(dict(condition=name, config=str(run/'config.yaml'),
                         xpid=config['xpid'], training_steps=config['total_steps']))
    (output/'manifest.json').write_text(json.dumps(dict(
        status='PREPARED', source_sha256=sha(source/'SOURCE.sha256'),
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        runs=runs, reference=str(REFERENCE),
        matched_sampling_context=16, scored_targets_per_update=64,
        burnin_gradient='retained (same as legacy)',
        primary_target='discounted task return minus computation cost',
        evaluation='matched stochastic soft gate without training epsilon; paired utility only',
        success='useful state-dependent budgets, not exploration-induced histogram spreading',
        pilot_scope='one game/subject/seed; staged ablations, not a factorial interaction test',
    ), indent=2)+'\n')
    inputs = [output/'launch.slurm', output/'prepare.py', source/'SOURCE.sha256']
    inputs += [output/item[0]/'config.yaml' for item in conditions]
    (output/'INPUTS.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(output)}\n' for p in inputs))
    print(output)


if __name__ == '__main__':
    main()
