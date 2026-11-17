"""One registered gradient-scale ablation using the frozen normalized runtime.

Fresh training, never a rewritten checkpoint or a change to an existing run.
The pilot is a screen, not proof of useful or stable adaptive computation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    ref=args.reference.resolve();out=args.output.resolve()
    baseline=yaml.safe_load((ref/'burn16ln/config.yaml').read_text())
    assert baseline['total_steps']==250000 and not baseline['ckp']
    assert baseline['voc_gru_gradient_scale']==1 and baseline['icopro_burn_in_steps']==16
    assert baseline['actor_h_encoder_norm']=='layer' and baseline['voc_train_epsilon']==.1
    out.mkdir(parents=True,exist_ok=False)
    (out/'source').symlink_to((ref/'source').resolve(),target_is_directory=True)
    (out/'evaluation_source_v2').symlink_to((ref/'evaluation_source_v2').resolve(),target_is_directory=True)
    shutil.copy2(__file__,out/'prepare.py')
    runs=[]
    for condition in ('burn16ln_gru10','smoke'):
        folder=out/condition;folder.mkdir()
        config=dict(baseline,voc_gru_gradient_scale=10.,
            xpid=f'enduro-s001-all-burn16ln-gru10-{condition}-s701-20261117')
        if condition=='smoke':
            config.update(total_steps=2048,checkpoint_interval=1024,model_warm_up_n=128)
        else:
            changed={k for k in baseline if baseline[k]!=config[k]}
            assert changed=={'xpid','voc_gru_gradient_scale'},changed
        p=folder/'config.yaml';p.write_text(yaml.safe_dump(config,sort_keys=True))
        runs.append(dict(condition=condition,config=str(p),xpid=config['xpid'],
            run_directory=str(Path(config['savedir'])/config['xpid']),training_steps=config['total_steps']))
    launch=(ref/'launch.slurm').read_text()
    launch=launch.replace('assert all("RTX 3090" in torch.cuda.get_device_name(i) for i in range(2))',
        'assert all(any(t in torch.cuda.get_device_name(i) for t in ("RTX 3090", "TITAN RTX")) for i in range(2))')
    launch=launch.replace('assert f.actor_h_encoder_norm == "layer"',
        'assert f.actor_h_encoder_norm == "layer"\nassert f.voc_gru_gradient_scale == 10.')
    launch=launch.replace("assert saved['planning_training_contract']['version'] == 2",
        "assert saved['planning_training_contract']['version'] == 2\n            assert saved['planning_training_contract']['gru_gradient_scale'] == 10.")
    launch=launch.replace("assert max(float(row['actor/icopro_nll']) for row in rows) < 10",
        "assert max(float(row['actor/icopro_nll']) for row in rows) < 10\n"
        "    assert all(float(r['gradient/voc_gru_gradient_scale']) == 10. for r in rows)\n"
        "    assert all(math.isclose(float(r['gradient/voc_gru_applied_norm']),10.*float(r['gradient/voc_gru_candidate_norm']),rel_tol=1e-6,abs_tol=1e-8) for r in rows)")
    (out/'launch.slurm').write_text(launch)
    manifest=dict(status='REGISTERED_NOT_SUBMITTED',created_at=time.time(),runs=runs,reference=str(ref),
        source_sha256=sha(out/'source/SOURCE.sha256'),
        evaluation_source_sha256=sha(out/'evaluation_source_v2/SOURCE.sha256'),
        changed_factor='VOC contrast gradient into shared GRU, multiplier1 to10. Isolated VOC-Q regression and forward policy unchanged by multiplier.',
        controls='Fresh seed701, Enduro sub001 all sessions, burn16, output LayerNorm, common+contrast critic, tau0.01, epsilon0.1, cost0.0001, min1/max30.',
        sampling='16 environments; identical human BC sampler/context/targets and loss weights to frozen reference.',
        checkpoint_milestones=[50000,100000,150000,200000,250000],
        evaluation=dict(gpu='RTX2080Ti',epsilon=0.,seeds=[8301,8401],batch=8,
            decisions=1536,prefix=128,counts=[1,3,6,12],repetitions=2,horizon=1536),
        score_evaluation='DISABLED',
        rationale='Normalized scale1 run remains nonstationary across1M/1.5M/2M. Median applied VOC/other actor GRU gradient norm ratio over1M–2M is0.2718%; this motivates a test, not a causal diagnosis.',
        reference_evidence=str(ref/'evidence_full_001/evidence.json'),
        progression='Screen the fixed250k endpoint and every saved milestone. Extend only after numerical stability and useful state-dependent control are assessed; no selection of a flattering checkpoint.',
        success_requirements=['Stable finite training, gradients and saved state.',
            'Evaluate1/30 and interior fractions with epsilon0, including time within episode.',
            'Compare state-dependent stopping to constant-initial-probability and matched-choice-frequency controls.',
            'Paired real-environment return utility must support the allocation; histogram spreading alone is insufficient.',
            'Persistence beyond a single checkpoint and independent training-seed replication remain necessary before goal completion.'],
        limits=['Single exploratory condition and seed. Historical scale10 without normalized encoder was not sufficient.',
                'Raw gradient norm ratio is not the marginal Adam parameter update.',
                'Existing scale1 reference is a prior run, not a concurrently randomized experiment.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files=[out/'prepare.py',out/'launch.slurm',out/'manifest.json',out/'source/SOURCE.sha256']
    files.extend(Path(r['config']) for r in runs)
    (out/'INPUTS.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in files))
    print(out)


if __name__=='__main__':main()
