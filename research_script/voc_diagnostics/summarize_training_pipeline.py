"""Audit local pilot bundles and summarize reward probes without efficacy claims."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_rows(path): return [json.loads(x) for x in path.read_text().splitlines()]
def write_csv(path, rows):
    if not rows: return
    with path.open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args(); root=args.root.resolve()
    training=[]; evaluation=[]; rewards=[]; pending=[]
    for game in ('pong','enduro'):
        for mode in ('control','projection'):
            directory=root/f'{game}_{mode}'
            state_path=directory/'pilot_state.json'
            if not state_path.exists():
                pending.append(f'{game}_{mode}'); continue
            state=json.loads(state_path.read_text()); ck=state['checkpoint']
            assert state['status']=='PASS'
            for name,record in ck['files'].items():
                assert sha(record['path'])==record['sha256'], (directory,name)
            training.append(dict(game=game,mode=mode,update=state['update'],attempt=state['attempt'],
                actor_real_step=ck['actor_real_step'],model_real_step=ck['model_real_step'],
                voc_updates=ck['voc_updates'],ema_updates=ck['ema_updates'],imitation_updates=ck['imitation_updates'],
                model_updates_m=ck['model_updates_m'],model_updates_p=ck['model_updates_p'],wandb_url=state['wandb_url']))
            for bundle in sorted((directory/'snapshots').glob('*/bundle.json')):
                binding=json.loads(bundle.read_text())
                for record in binding['files'].values():
                    assert sha(record['path'])==record['sha256']
                flags=yaml.safe_load(Path(binding['files']['config']['path']).read_text())
                dest=bundle.parent/'reward_evaluation'
                manifest=json.loads((dest/'manifest.json').read_text())
                assert manifest['status']=='PAIRED_ONLINE_PASS'
                rows=read_rows(dest/'pairs.jsonl')
                gaps=[]; kls=[]; raw_kls=[]
                for row in rows:
                    entry=row['arms'][0]['entry']
                    teacher=1/(1+np.exp(-entry['ema_delta_q']/flags['voc_gate_q_temperature']))
                    student=entry['learned_continue_probability']
                    p0=np.clip(teacher,1e-12,1-1e-12); q0=np.clip(student,1e-12,1-1e-12)
                    raw_kls.append(float(p0*np.log(p0/q0)+(1-p0)*np.log((1-p0)/(1-q0))))
                    epsilon=row['arms'][0]['policy_epsilon']
                    teacher=(1-epsilon)*teacher+epsilon*.5
                    p0=np.clip(teacher,1e-12,1-1e-12)
                    kls.append(float(p0*np.log(p0/q0)+(1-p0)*np.log((1-p0)/(1-q0))))
                    gaps.append(abs(teacher-student))
                    for other in row['arms'][1:]:
                        for k in ('prefix_sha256','entry_sha256','tree_sha256'):
                            assert entry[k]==other['entry'][k]
                    assert row['arms'][0]['trajectory']==row['arms'][2]['trajectory']
                    assert row['arms'][0]['returns']==row['arms'][2]['returns']
                evaluation.append(dict(game=game,mode=mode,update=binding['update'],
                    actor_real_step=binding['actor_real_step'],pairs=len(rows),
                    raw_teacher_to_executed_gate_kl=float(np.mean(raw_kls)),
                    epsilon_adjusted_teacher_gate_kl=float(np.mean(kls)),
                    epsilon_adjusted_max_probability_gap=float(max(gaps)),
                    mean_delta_net_truncated=float(np.mean([r['delta_net_truncated'] for r in rows])),
                    horizon=manifest['arguments']['horizon']))
    for game in ('pong','enduro'):
        path=root/f't3_{game}'; manifest=json.loads((path/'manifest.json').read_text()); rows=read_rows(path/'pairs.jsonl')
        groups={}
        for r in rows: groups.setdefault((r['seed'],r['prefix_steps']),[]).append(r)
        for (seed,prefix), repeated in groups.items():
            rewards.append(dict(game=game,seed=seed,prefix_steps=prefix,repetitions=len(repeated),
                ema_delta_q=float(np.mean([r['ema_delta_q'] for r in repeated])),
                delta_task_truncated=float(np.mean([r['arms'][1]['returns']['task_truncated']-r['arms'][0]['returns']['task_truncated'] for r in repeated])),
                delta_net_truncated=float(np.mean([r['delta_net_truncated'] for r in repeated])),
                delta_net_bootstrapped=float(np.mean([r['delta_net_bootstrapped'] for r in repeated])),
                full_return_pairs=sum(r['full_return_observed'] for r in repeated),horizon=manifest['arguments']['horizon']))
    write_csv(root/'training_summary.csv',training)
    write_csv(root/'periodic_evaluation.csv',evaluation)
    write_csv(root/'t3_state_scores.csv',rewards)
    report=['# Reward-based VOC training pipeline audit','',
        'This is an implementation smoke test. It does not establish reward improvement or justify changing the critic target.','',
        '| Game | Gate | Actor real steps | Model real steps | VOC/EMA updates | Human imitation updates | SR/VP updates |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in training:
        report.append(f"| {r['game']} | {r['mode']} | {r['actor_real_step']} | {r['model_real_step']} | {r['voc_updates']}/{r['ema_updates']} | {r['imitation_updates']} | {r['model_updates_m']}/{r['model_updates_p']} |")
    report += ['', '## Existing 5M checkpoint reward probes','',
        'One subject/leave-session checkpoint per game, four online states, two paired continuations per state; 256 real-step horizon. Evaluation uses sampled primary/conditional actions and a stochastic gate, with epsilon=0. Training-target calibration should use the matching epsilon (the periodic pilot evaluation uses `--policy-epsilon`).','',
        '| Game | Paired rollouts | Mean truncated task gain | Mean truncated task gain − current-stage cost | Full-return pairs |',
        '|---|---:|---:|---:|---:|']
    for game in ('pong','enduro'):
        rr=[r for r in rewards if r['game']==game]
        report.append(f"| {game} | {sum(r['repetitions'] for r in rr)} | {np.mean([r['delta_task_truncated'] for r in rr]):.6f} | {np.mean([r['delta_net_truncated'] for r in rr]):.6f} | {sum(r['full_return_pairs'] for r in rr)} |")
    report += ['',
        'Enduro has a large positive gain in one continuation and a negative gain in another. With four states and no completed returns, these averages are not evidence of calibrated VOC. Pong shows no task reward gain within the sampled horizon. The unobserved reward tail and rollout variation preclude concluding that true VOC is zero.', '',
        '## Checks and limits','',
        '- Every evaluation bundle is hash-bound; STOP replay is reproduced exactly and branch entry fingerprints agree.',
        '- Periodic evaluation is a separate process, so it does not consume training RNG or mutate live weights.',
        '- Exact projection preserves the soft gate temperatures, stochastic execution and existing current-stage cost objective.',
        '- A resumed chunk restores learner optimizers/counters/BC RNG; online environment, recurrent state and replay restart. It is not a bitwise trajectory continuation.',
        '- Small batches, unrolls and replay warm-up are shared smoke settings. These runs are not learning-curve or equal-environment-budget comparisons.',
        '- Original 5M checkpoints and running cluster jobs were not modified.', '',
        f"Incomplete requested pilot cells: {', '.join(pending) if pending else 'none'}."]
    (root/'pipeline_report.md').write_text('\n'.join(report)+'\n')
    (root/'qa.json').write_text(json.dumps(dict(status='PASS' if not pending else 'INCOMPLETE',pending=pending,
        pilots=len(training),evaluations=len(evaluation),original_online_states=len(rewards)),indent=2))
    print(json.dumps(dict(pilots=len(training),evaluations=len(evaluation),pending=pending)))

if __name__=='__main__': main()
