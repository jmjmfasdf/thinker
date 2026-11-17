"""Prepare matched normalized burn-in8/16 persistence trials; never submit here."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import yaml


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


TRAIN = r'''#!/bin/bash
set -euo pipefail
CAMPAIGN="${1:?campaign}"
CONDITION="${2:?condition}"
PHASE="${3:?smoke or full}"
case "$CONDITION" in burn8ln|burn16ln) ;; *) exit 2 ;; esac
case "$PHASE" in smoke) RUN="$CAMPAIGN/validation_$CONDITION" ;; full) RUN="$CAMPAIGN/$CONDITION" ;; *) exit 2 ;; esac
SOURCE="$CAMPAIGN/source"
JOBTMP="/scratch/$USER/ne${SLURM_JOB_ID}"
mkdir -p "$JOBTMP/e" "$JOBTMP/t" "$JOBTMP/w/cache" "$JOBTMP/w/data"
cd "$CAMPAIGN"
sha256sum --quiet -c INPUTS.sha256
cd "$SOURCE"
sha256sum --quiet -c SOURCE.sha256
ARCHIVE=/work/jeongmin/conda-envs/dynamic-thinker-py310.tar.zst
ACTUAL=$(sha256sum "$ARCHIVE" | awk '{print $1}')
test "$ACTUAL" = 0221345519333f783c413adca1240991fe2fcdc1a563bf312d4e57001c6b3714
tar --zstd -xf "$ARCHIVE" -C "$JOBTMP/e"
"$JOBTMP/e/bin/conda-unpack"
export PATH="$JOBTMP/e/bin:$PATH" PYTHONPATH="$SOURCE/thinker"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$JOBTMP/t" RAY_TMPDIR="$JOBTMP/t"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_USER=jmme425 WANDB_SILENT=true
export WANDB_RUN_GROUP="$(basename "$CAMPAIGN")" WANDB_DIR="$RUN"
export WANDB_CACHE_DIR="$JOBTMP/w/cache" WANDB_DATA_DIR="$JOBTMP/w/data"
export WANDB_MODE=online CUBLAS_WORKSPACE_CONFIG=:4096:8
unset WANDB_API_KEY WANDB_DISABLED
python "$CAMPAIGN/verify.py" --campaign "$CAMPAIGN" --condition "$CONDITION" --phase "$PHASE" --preflight --require-gpu
python -u thinker/train.py --config "$RUN/config.yaml"
python "$CAMPAIGN/verify.py" --campaign "$CAMPAIGN" --condition "$CONDITION" --phase "$PHASE"
'''


EVALUATE = r'''#!/bin/bash
set -euo pipefail
CAMPAIGN="${1:?campaign}"
CONDITION="${2:?condition}"
MODE="${3:?allocation or utility}"
MILESTONE="${4:?registered milestone}"
case "$CONDITION" in burn8ln|burn16ln) ;; *) exit 2 ;; esac
case "$MILESTONE" in 500000|1000000) ;; *) exit 2 ;; esac
JOBTMP="/scratch/$USER/nv${SLURM_JOB_ID}"
mkdir -p "$JOBTMP/e" "$JOBTMP/t"
cd "$CAMPAIGN"
sha256sum --quiet -c INPUTS.sha256
cd "$CAMPAIGN/source"
sha256sum --quiet -c SOURCE.sha256
cd "$CAMPAIGN/evaluation_source_v2"
sha256sum --quiet -c SOURCE.sha256
ARCHIVE=/work/jeongmin/conda-envs/dynamic-thinker-py310.tar.zst
ACTUAL=$(sha256sum "$ARCHIVE" | awk '{print $1}')
test "$ACTUAL" = 0221345519333f783c413adca1240991fe2fcdc1a563bf312d4e57001c6b3714
tar --zstd -xf "$ARCHIVE" -C "$JOBTMP/e"
"$JOBTMP/e/bin/conda-unpack"
export PATH="$JOBTMP/e/bin:$PATH" PYTHONPATH="$CAMPAIGN/source/thinker"
export VOC_DIAGNOSTIC_RUNTIME="$CAMPAIGN/source/thinker"
export TMPDIR="$JOBTMP/t" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
python - "$CAMPAIGN" "$CONDITION" <<'CHECK'
from pathlib import Path
import json,sys
c=Path(sys.argv[1]);condition=sys.argv[2]
audit=json.loads((c/condition/'completion.json').read_text())
assert audit['status']=='CONTINUATION_CHECKPOINT_OPTIMIZER_COUNTERS_WANDB_PASS'
assert audit['health']['bc_history_status']=='NO_LARGE_NLL_OBSERVED'
CHECK
if [ "$MODE" = allocation ]; then
  exec python -u "$CAMPAIGN/allocation.py" --campaign "$CAMPAIGN" --output "$CAMPAIGN/allocation_${MILESTONE}_$CONDITION" --conditions "$CONDITION" --milestone "$MILESTONE" --batch 8 --decisions 1536 --seeds 8301 8401
elif [ "$MODE" = utility ] && [ "$MILESTONE" = 1000000 ]; then
  mapfile -t CHECKPOINTS < <(python - "$CAMPAIGN" "$CONDITION" <<'CHECK'
from pathlib import Path
import json,sys
c=Path(sys.argv[1]);condition=sys.argv[2]
audit=json.loads((c/condition/'completion.json').read_text())
for kind in ('actor','model'):print(audit['checkpoint'][kind]['path'])
CHECK
)
  test "${#CHECKPOINTS[@]}" -eq 2
  exec python -u "$CAMPAIGN/evaluation_source_v2/research_script/diagnostics/evaluate_adaptive_planning_pilot.py" --actor "${CHECKPOINTS[0]}" --model "${CHECKPOINTS[1]}" --output "$CAMPAIGN/evaluation_$CONDITION" --decisions 1536 --batch 8 --seeds 8301 8401 --prefix 128 --counts 1 3 6 12 --repetitions 2 --horizon 1536
else
  exit 2
fi
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training-gpu',choices=['rtx3090','titanrtx'],default='rtx3090')
    p.add_argument('--tag',default='20261115')
    args=p.parse_args();base=args.base.resolve();out=args.output.resolve()
    parents=dict(burn16ln=base/'adaptive_planning_encoder_stability_20261114c',
                 burn8ln=base/'adaptive_planning_encoder_burn8_matched_20261114c')
    hashes={sha(c/'source/SOURCE.sha256') for c in parents.values()}
    if len(hashes)!=1:raise ValueError('parent training sources differ')
    for parent in parents.values():
        audit=json.loads((parent/'completion_audit_001/audit.json').read_text())
        if audit['status']!='FRESH_BUDGET_LOGS_OPTIMIZERS_EVALUATION_AND_WANDB_PASS' or audit['health']['bc_history_status']!='NO_LARGE_NLL_OBSERVED':
            raise ValueError('parent did not complete numerical/W&B screen')
    out.mkdir(parents=True,exist_ok=False)
    (out/'source').symlink_to(parents['burn16ln']/'source',target_is_directory=True)
    # New evaluation snapshot contains only the verified mandatory-RNG repair.
    src=parents['burn16ln']/'evaluation_source_v2'
    shutil.copytree(src,out/'evaluation_source_v2')
    evaldir=out/'evaluation_source_v2'
    relative='research_script/voc_diagnostics/gru_reward_readout.py'
    old=(evaldir/relative).read_text();before='control[pending] = conditional[pending]'
    after=('force_pending = pending & env.legal_control_mask[0, :, util.STOP].cpu().numpy().astype(bool)\n'
           '                control[force_pending] = conditional[force_pending]')
    if old.count(before)!=1:raise ValueError('unexpected parent evaluator')
    (evaldir/relative).write_text(old.replace(before,after))
    paths=[line.split('  ',1)[1] for line in (src/'SOURCE.sha256').read_text().splitlines()]
    changed=[path for path in paths if sha(src/path)!=sha(evaldir/path)]
    if changed!=[relative]:raise ValueError('unregistered evaluator change')
    (evaldir/'SOURCE.sha256').write_text(''.join(sha(evaldir/path)+'  '+path+'\n' for path in paths))
    diagnostics=Path(__file__).resolve().parent
    shutil.copy2(diagnostics/'verify_planning_continuation.py',out/'verify.py')
    shutil.copy2(diagnostics/'summarize_adaptive_planning_pilot.py',out/'summarize_adaptive_planning_pilot.py')
    shutil.copy2(base/'adaptive_planning_persistence_15m_20261114/allocation_midpoint.py',out/'allocation.py')
    shutil.copy2(__file__,out/'prepare.py')
    (out/'train.slurm').write_text(TRAIN)
    (out/'evaluate.slurm').write_text(EVALUATE)
    runs=[];smokes=[];registered_files=[]
    for condition,parent in parents.items():
        manifest=json.loads((parent/'manifest.json').read_text())
        item=next(r for r in manifest['runs'] if r['condition']==condition)
        base_config=yaml.safe_load(Path(item['config']).read_text())
        audit=json.loads((parent/'completion_audit_001/audit.json').read_text())
        parent_ck={}
        immutable=out/'parent_checkpoints'/condition;immutable.mkdir(parents=True)
        for kind,info in audit['checkpoint'].items():
            if sha(info['path'])!=info['sha256']:raise ValueError('parent bytes changed')
            saved=immutable/f'ckp_{kind}.tar';shutil.copy2(info['path'],saved)
            parent_ck[kind]=dict(info,immutable_copy=str(saved))
            registered_files.append(saved)
        for phase,target in [('smoke',270000),('full',1000000)]:
            folder=out/(f'validation_{condition}' if phase=='smoke' else condition);folder.mkdir()
            xpid=f'enduro-s001-all-norm-{condition}-{phase}-s701-{args.tag}'
            config=dict(base_config,ckp=True,total_steps=target,xpid=xpid)
            assert config['actor_h_encoder_norm']=='layer' and config['voc_gru_gradient_scale']==1
            assert config['checkpoint_interval']==50000 and config['schedule_total_steps']==100000000
            config_path=folder/'config.yaml';config_path.write_text(yaml.safe_dump(config,sort_keys=True))
            destination=Path(config['savedir'])/xpid;destination.mkdir(exist_ok=False)
            for kind,info in parent_ck.items():shutil.copy2(info['immutable_copy'],destination/f'ckp_{kind}.tar')
            (destination/'config_c.yaml').write_text(yaml.safe_dump(config,sort_keys=True))
            entry=dict(condition=condition,config=str(config_path),xpid=xpid,run_directory=str(destination),
                training_steps=target,parent_checkpoint=parent_ck,parent_config=item['config'],
                parent_completion_audit=str(parent/'completion_audit_001/audit.json'),
                actor_additional_budget=target-parent_ck['actor']['real_step'],
                wandb_url='https://wandb.ai/jmme425/thinker/runs/'+xpid)
            (smokes if phase=='smoke' else runs).append(entry)
            registered_files.append(config_path)
    report=dict(status='REGISTERED_NORMALIZED_BURNIN_PERSISTENCE',created_at=time.time(),runs=runs,smoke_runs=smokes,
        source_sha256=next(iter(hashes)),training_changes='NONE relative to each normalized250kparent',
        comparison='Same normalized architecture, Enduro sub001 all sessions, seed701, burn-in8 versus16, GRU multiplier1.',
        milestones=[500000,1000000],utility_endpoint=1000000,
        paired_protocol=dict(seeds=[8301,8401],batch=8,prefix=128,counts=[1,3,6,12],repetitions=2,horizon=1536,
            target='Discounted clipped task reward minus remaining current-stage cost',pairs=128,states=64,independent_streams=16),
        allocation_protocol=dict(seeds=[8301,8401],batch=8,decisions_per_stream=1536,first_decision=128,epsilon=0),
        evaluation_change=dict(protocol='mandatory_draw_preservation_v2',changed_files=changed,
            proof=str(base/'adaptive_planning_extension_1m_20261114/selected_utility_repair_001/probe/completion.json')),
        smoke='Independent bounded resume20k validation. Full training restarts from original250kparent; smoke weights are excluded.',
        restart_semantics='Cold environment, actor rollout recurrent state, and model replay; restored network/optimizer/counters/BC RNG. Physical interactions reported separately from learner steps.',
        decision='Report both500kand1M; no selection by histogram or utility. Numerical failure triggers diagnosis, not clipping or selective row removal.',
        interpretation='Exploratory single-seed follow-up; different update/BCtarget totals arise from variable imagination length and are reported. Broader distributions do not prove useful computation.',
        training_gpu=args.training_gpu,training_resources=f'Two{args.training_gpu}GPUs per run',
        hardware_note='Both comparison conditions use the same GPUfamily; parent250kcheckpoints were trained onRTX3090. GPUarithmetic and cold-restart effects cannot be isolated from continuation learning.',
        evaluation_resources='RTX2080Ti excludingcompute02',score_evaluation='DISABLED')
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    registered_files.extend(out/x for x in ['prepare.py','verify.py','summarize_adaptive_planning_pilot.py','allocation.py',
        'train.slurm','evaluate.slurm','manifest.json','source/SOURCE.sha256','evaluation_source_v2/SOURCE.sha256'])
    (out/'INPUTS.sha256').write_text(''.join(sha(f)+'  '+str(f)+'\n' for f in registered_files))
    print(json.dumps(dict(output=str(out),full_runs=[r['xpid'] for r in runs],smoke_runs=[r['xpid'] for r in smokes]),indent=2))


if __name__=='__main__':main()
