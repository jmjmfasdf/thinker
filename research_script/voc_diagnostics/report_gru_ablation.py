"""Regenerate partial/final experiment tables; never label missing cells complete."""
import argparse,csv,hashlib,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from train_pilot import atomic_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


class ArtifactsPending(RuntimeError):
    """A completed worker's output is not yet visible to the report reader."""


def required_read(path):
    value=read(path)
    if value is None:
        raise ArtifactsPending(f'Missing final artifact: {path}')
    return value


def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def validate_final_cell(d):
    status=required_read(d/'status.json');milestones=required_read(d/'milestones.json')
    assert status['phase']=='COMPLETE' and status['env_steps']==100000
    bundle=required_read(Path(milestones[-1]['bundle']))
    assert bundle['actor_real_step']==bundle['model_real_step']==100000
    assert bundle['branch_provenance']['additional_env_steps']==100000
    for file in bundle['files'].values():
        assert hashlib.sha256(Path(file['path']).read_bytes()).hexdigest()==file['sha256']
    lines=[json.loads(x) for x in (d/'gradient_metrics.jsonl').read_text().splitlines()]
    mode=status['mode']
    assert any(x['voc_gru_candidate_norm']>0 for x in lines)
    assert all(x['voc_gru_applied_norm']==(x['voc_gru_candidate_norm'] if mode=='gru' else 0) for x in lines)
    assert lines[-1]['model_updates']>0 and lines[-1]['imitation_updates']>0
    e=Path(milestones[-1].get('evaluation',str(Path(milestones[-1]['bundle']).parent/'evaluation/summary.json')))
    summary=required_read(e);folder=e.parent
    assert summary['status']=='COMPLETE'
    assert summary['voc']['states']==32 and summary['voc']['pairs']==64 and summary['voc']['independent_streams']==8
    rows=required_read(folder/'pairs.json')
    for row in rows:
        prefix,k,rep,i=row['prefix'],row['count'],row['repetition'],row['stream']
        a,b=[required_read(folder/f'pair_p{prefix}_k{k}_r{rep}_{arm}.json') for arm in ['stop','continue']]
        assert a['entries']==b['entries']
        assert a['entries'][i]['hash']==row['entry_hash']
        assert row['delta']==b['returns'][i]['net_truncated']-a['returns'][i]['net_truncated']
        assert row['tail_bound']<4e-5
        assert required_read(folder/f'repeat_p{prefix}_k{k}.json')['status']=='PASS'
    return dict(status='PASS',additional_env_steps=100000,pairs=64,states=32)


def build(root):
    rows=[];scores=[];voc=[];gradients=[];evidence={};initial={};final={}
    for game in ['enduro','pong']:
      for mode in ['detached','gru']:
        key=f'{game}_{mode}';d=root/'cells'/key;s=read(d/'status.json') or {}
        init=read(d/'initialization.json')
        if init:initial[key]=init
        g=[]
        if (d/'gradient_metrics.jsonl').exists():
            # A writer may be appending its final line; ignore only a partial
            # trailing line. Finished outputs are checked strictly below.
            for line in (d/'gradient_metrics.jsonl').read_text().splitlines():
                try:g.append(json.loads(line))
                except json.JSONDecodeError:pass
        gradients.extend(dict(game=game,mode=mode,**v) for v in g)
        rows.append(dict(game=game,mode=mode,phase=s.get('phase','NOT_STARTED'),
            additional_env_steps=s.get('env_steps',0),budget=100000,job=s.get('job_id'),
            actor_updates=s.get('actor_updates',0),wandb=s.get('wandb_url'),error=s.get('error'),
            model_updates=g[-1].get('model_updates',0) if g else 0,
            imitation_updates=g[-1].get('imitation_updates',0) if g else 0,
            voc_updates=g[-1].get('voc_updates',0) if g else 0,
            voc_gru_applied_norm=g[-1]['voc_gru_applied_norm'] if g else None,
            training_sps=s.get('train_env_steps_per_second')))
        for point in read(d/'milestones.json') or []:
            ep=Path(point.get('evaluation',str(Path(point['bundle']).parent/'evaluation/summary.json')))
            result=read(ep)
            if not result or result.get('status')!='COMPLETE':
                # Score collection finishes before the longer paired VOC test.
                score=read(ep.parent/'score_learned.json')
                if score and score['real_actions_per_stream']==2048:
                    scores.append(dict(game=game,mode=mode,additional_env_steps=point['env_steps'],
                        score_per_1000_actions=float(np.mean(score['score_per_1000_actions'])),
                        computations_per_action=score['mean_computations_per_action'],
                        evaluation_status='SCORE_COMPLETE_VOC_PENDING'))
                continue
            scores.append(dict(game=game,mode=mode,additional_env_steps=point['env_steps'],
                score_per_1000_actions=result['metrics']['score_per_1000_actions'],
                computations_per_action=result['metrics']['computations_per_action'],evaluation_status='COMPLETE'))
            if 'voc' in result:
                final[key]=result
                voc.append(dict(game=game,mode=mode,**{k:v for k,v in result['voc'].items()
                    if isinstance(v,(int,float)) or v is None}))
        if s.get('phase')=='COMPLETE':
            try:
                evidence[key]=validate_final_cell(d)
            except (ArtifactsPending,FileNotFoundError) as exc:
                # Status and outputs are separate files on shared storage.
                # Retry missing artifacts; never hide failed numeric assertions.
                rows[-1]['phase']='AWAITING_FINAL_ARTIFACTS'
                rows[-1]['error']=str(exc)
    init_comparison={}
    for game in ['enduro','pong']:
        a,b=initial.get(game+'_detached'),initial.get(game+'_gru')
        if a and b:
            init_comparison[game]={k:a[k]==b[k] for k in ['actor','model','ema','parent']}
            assert all(init_comparison[game].values()),'Unequal branch initialization'
    comparisons=[]
    for game in ['enduro','pong']:
        a,b=final.get(game+'_gru'),final.get(game+'_detached')
        if a and b:
            delta=np.asarray(a['score']['score_per_1000_actions'])-b['score']['score_per_1000_actions']
            rng=np.random.default_rng(9311)
            ci=np.quantile(delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1),[.025,.975])
            comparisons.append(dict(game=game,contrast='gru_minus_detached',
                reward_difference_per_1000_actions=float(delta.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),
                gru_compute=a['score']['mean_computations_per_action'],detached_compute=b['score']['mean_computations_per_action']))
    failed=any(r['phase']=='FAILED' for r in rows)
    status='COMPLETE' if len(evidence)==4 else ('FAILED_OR_PARTIAL' if failed else 'RUNNING_OR_PARTIAL')
    csv_write(root/'training_status.csv',rows);csv_write(root/'score_curve.csv',scores)
    csv_write(root/'gradient_curve.csv',gradients);csv_write(root/'voc_scores.csv',voc)
    csv_write(root/'reward_comparisons.csv',comparisons)
    lines=['# VOC -> tree GRU continuation experiment','',f'Status: {status}',
        '', 'One continuation seed per game; same 100k parent, plus 100k actions in each arm. Training-seed uncertainty is not estimated.',
        '', '| Game | Condition | Phase | Additional actions | Actor / model / BC updates | W&B |',
        '|---|---|---|---:|---:|---|']
    for r in rows:lines.append(f"| {r['game']} | {r['mode']} | {r['phase']} | {r['additional_env_steps']:,}/100,000 | {r['actor_updates']} / {r['model_updates']} / {r['imitation_updates']} | {r['wandb'] or 'NA'} |")
    lines+=['','Equal real-action budgets do not imply equal actor/VOC optimizer updates: the number of imagined transitions can change with the policy. Actual actor, model and BC updates are reported above.']
    lines+=['','## Predetermined score checkpoints','',
        '| Game | Condition | Additional actions | Raw score / 1000 actions | Computations / action |','|---|---|---:|---:|---:|']
    for r in scores:lines.append(f"| {r['game']} | {r['mode']} | {r['additional_env_steps']} | {r['score_per_1000_actions']:.5f} | {r['computations_per_action']:.5f} |")
    lines+=['','## Final VOC prediction','',
        '| Game | Condition | EMA RMSE | Zero RMSE | EMA rank correlation | Local allocation gain |','|---|---|---:|---:|---:|---:|']
    for r in voc:lines.append(f"| {r['game']} | {r['mode']} | {r['ema_rmse']:.6g} | {r['zero_rmse']:.6g} | {r['ema_spearman']} | {r['allocation_gain_vs_cost_matched_random']:.6g} |")
    lines+=['','VOC targets are evaluated under each trained continuation policy; raw RMSE differences across policies also reflect different visited states and targets. Paired comparisons hold STOP/CONTINUE entries exactly equal within each checkpoint. Full-policy fixed/random baselines are calibrated on separate streams; achieved computation mismatch remains visible in each evaluation summary.','',
        '## GRU condition minus detached control','',
        '| Game | Reward difference / 1000 actions | 95% stream interval |','|---|---:|---|']
    for r in comparisons:lines.append(f"| {r['game']} | {r['reward_difference_per_1000_actions']:.6g} | [{r['ci95_low']:.6g}, {r['ci95_high']:.6g}] |")
    lines+=['','Initialization equality: '+json.dumps(init_comparison),
        '', 'No efficacy conclusion is available while any cell or final evaluation is incomplete. Gate calibration and nonzero gradients alone are not evidence that planning improves reward.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    result=dict(status=status,timestamp=datetime.now(timezone.utc).isoformat(),cells=rows,
                initialization=init_comparison,final_validation=evidence)
    atomic_json(root/'progress.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=45)
    a=p.parse_args();jobs_absent=0
    while True:
        try:
            r=build(a.root);print(json.dumps(r),flush=True)
        except Exception as exc:
            atomic_json(a.root/'report_error.json',dict(error=repr(exc),timestamp=datetime.now(timezone.utc).isoformat()))
            raise
        if not a.watch or r['status']=='COMPLETE':break
        submission=read(a.root/'submission.json') or {}
        ids=list(submission.get('science_jobs',{}).values())
        if ids:
            q=subprocess.run(['squeue','-j',','.join(map(str,ids)),'-h'],capture_output=True,text=True)
            if q.returncode==0 and not q.stdout.strip():
                jobs_absent+=1
                if jobs_absent<3:
                    time.sleep(min(a.interval,60))
                    continue
                # A killed worker may be unable to write FAILED. Do not leave a
                # stale RUNNING report after every scheduler job has exited.
                r=build(a.root)
                if r['status']!='COMPLETE':
                    r['status']='JOBS_EXITED_WITH_INCOMPLETE_RESULTS'
                    atomic_json(a.root/'progress.json',r)
                    report=a.root/'report.md'
                    report.write_text(report.read_text().replace(
                        'Status: RUNNING_OR_PARTIAL', 'Status: JOBS_EXITED_WITH_INCOMPLETE_RESULTS'))
                break
            if q.returncode==0 and q.stdout.strip():jobs_absent=0
        time.sleep(min(a.interval,60))


if __name__=='__main__':main()
