"""Freeze a fresh tau=.01 three-game 10M campaign from the verified .05 runtime.

Prepares reviewable configs/scripts only; submission is explicit and recorded in
jobs.json. No checkpoint weights are copied into the new learners.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import yaml


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--submit-dir',type=Path,required=True)
    args=parser.parse_args();parent=args.parent.resolve();c=args.output.resolve()
    if c.exists():raise FileExistsError(c)
    c.mkdir(parents=True);args.submit_dir.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).resolve().parent
    shutil.copytree(parent/'source',c/'source')
    changed=['voc10m_parallel_monitor.py','voc10m.slurm','eval_voc10m.py',
             'test_voc10m_evaluation_policy.py','test_voc10m_paired_only.py',
             'validate_voc_tau001_smoke.py']
    for name in changed:shutil.copy2(here/name,c/'source/research_script/voc_diagnostics'/name)
    for name in ['configs','cells','smoke_configs','smoke/cells','eval_tasks','evaluations']:
        (c/name).mkdir(parents=True,exist_ok=True)
    (c/'smoke/source').symlink_to('../source',target_is_directory=True)
    cells=[];diffs={}
    for game in ('pong','enduro','spaceinvaders'):
        raw=yaml.safe_load((parent/f'configs/{game}.yaml').read_text())
        config={**raw,'voc_gate_q_temperature':.01,'savedir':str(c/'cells'),
                'xpid':f'v10t001-20260926-{game}_s701'}
        assert config['ckp'] is False and not config['preload'] and not config['preload_actor']
        assert config['total_steps']==10_000_000 and config['base_seed']==701
        path=c/f'configs/{game}.yaml';path.write_text(yaml.safe_dump(config,sort_keys=True))
        diffs[game]={key:{'old':raw.get(key),'new':value} for key,value in config.items() if value!=raw.get(key)}
        assert set(diffs[game])=={'voc_gate_q_temperature','savedir','xpid'}
        cells.append(dict(name=config['xpid'],game=game,node=None,gpus=2,config=str(path),
            allowed_training_nodes=['bml-compute07','bml-compute08','bml-compute10']))
        smoke={**config,'savedir':str(c/'smoke/cells'),'xpid':'smoke-t001-'+game,
               'total_steps':6000,'model_warm_up_n':1024,'checkpoint_interval':1000}
        (c/f'smoke_configs/{game}.yaml').write_text(yaml.safe_dump(smoke,sort_keys=True))
    source=c/'source'
    files=sorted(p for p in source.rglob('*') if p.is_file() and p.name!='SOURCE.sha256' and '__pycache__' not in p.parts)
    (source/'SOURCE.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(source)}\n' for p in files))
    manifest=dict(created_at=time.time(),parent_campaign=str(parent),
        parent_source_manifest_sha256=sha(parent/'source/SOURCE.sha256'),
        source_sha256=sha(source/'SOURCE.sha256'),changed_source_files=changed,cells=cells,
        subject=1,train_sessions=[2,3,4],holdout_sessions=[1],seed=701,
        target_actor_env_steps=10_000_000,model_and_actor_fresh=True,
        evaluation_milestones=[2_000_000,4_000_000,6_000_000,8_000_000,10_000_000],
        temperature=.01,training_condition='exact EMA gate projection + selective VOC to GRU; fresh tau .01',
        configuration='env16/actor16x201/model32x30, FP32, GroupNorm; unchanged from .05 comparator',
        budget_rule='first complete actor unroll at or beyond 10M; exact actor/model/collector counts retained',
        evaluation_gpu='NVIDIA GeForce RTX 2080 Ti',resume_policy='no automatic restart on failure',
        config_sha256={x['game']:sha(Path(x['config'])) for x in cells})
    (c/'campaign.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (c/'config_diff_from_tau005.json').write_text(json.dumps(diffs,indent=2)+'\n')
    (c/'evaluation_policy.json').write_text(json.dumps(dict(disabled_kinds=['episodes','score','score_sign','score_zero'],
        benefit_include_scores=False,
        evaluation_partition='rtx2080',evaluation_max_jobs=4,
        evaluation_script=str(source/'research_script/voc_diagnostics/voc10m.slurm')),indent=2)+'\n')
    main_script=(here/'voc10m_parallel.slurm').read_text()
    main_script=main_script.replace('CONFIG="${2:?configuration}"','CONFIG="${2:?configuration}"\ncase "$SLURMD_NODENAME" in bml-compute07|bml-compute08|bml-compute10) ;; *) exit 2;; esac')
    (args.submit_dir/'train.slurm').write_text(main_script)
    smoke_script=main_script.replace('CONFIG="${2:?configuration}"','CONFIG=""')
    smoke_script=smoke_script.replace('export WANDB_RUN_GROUP="$(basename "$CAMPAIGN")"','export WANDB_RUN_GROUP="$(basename "$CAMPAIGN")-smoke"')
    smoke_script=smoke_script.replace('exec python thinker/train.py --config "$CONFIG"',
        'for GAME in pong enduro spaceinvaders; do\n'
        '    python thinker/train.py --config "$CAMPAIGN/smoke_configs/$GAME.yaml"\n'
        'done\n'
        'python research_script/voc_diagnostics/validate_voc_tau001_smoke.py "$CAMPAIGN"')
    (args.submit_dir/'preflight.slurm').write_text(smoke_script)
    shutil.copy2(source/'research_script/voc_diagnostics/voc10m.slurm',args.submit_dir/'evaluate.slurm')
    (c/'protocol.md').write_text('''# Fresh tau 0.01 training and utility evaluation

Sub001, seed701, sessions2–4 training/session1 held out; Pong, Enduro and Space
Invaders each train fresh actor/world model for10M actor-consumed environment
actions. Only Q-to-gate temperature changes from0.05 to0.01; policy temperature1,
epsilon0.02, think cost0.0005, BC/RL geometry and learning-rate horizon stay fixed.
The original0.05 training continues independently. No trained weights are loaded.

Each training job requests two GPUs on one of compute07/08/10. All evaluations
request one RTX2080Ti in partitionrtx2080, with a device-name check at startup.
Native/sign/zero and paired benefit tasks are published at2,4,6,8,10M only.
Recovery snapshots remain every100k so asynchronous actor/model pairs can be
validated within the existing+100k publication window. The inherited benefit
protocol uses64 paired returns below5M and256 at/above5M; horizon1536, gamma0.99,
clipped rewards minus current-stage computation cost. Raw policy reward and
achieved compute differences are reported separately. No long episodes jobs.

All training and evaluation runs log to W&B. A three-game6000-step full-geometry
preflight must pass, followed by a smoke benefit evaluation for each game on an
RTX2080Ti, before any10M training starts. Smoke data are separate. Failed jobs
are visible and never automatically restarted. Numerical checkpoints and source
are immutable inputs to evaluation. Initialization/return variation across GPU
models is not treated as a temperature effect by itself.
''')
    print(json.dumps({'campaign':str(c),'manifest':manifest,'submission_directory':str(args.submit_dir)},indent=2))


if __name__=='__main__':main()
