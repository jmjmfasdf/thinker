"""Read-only interim audit of the parallel campaign and completed evaluations.

No training or inference is performed, and no campaign inputs are modified.
Bootstrap intervals are exploratory, stream-clustered, and uncorrected.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def markdown_table(frame):
    def fmt(value):
        if isinstance(value,float):return f'{value:.6g}'
        return str(value).replace('|','\\|')
    rows=['| '+' | '.join(frame.columns)+' |',
          '| '+' | '.join(['---']*len(frame.columns))+' |']
    rows += ['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def compare(a, b):
    for key in ('seed', 'real_actions_per_stream', 'weights_sha256'):
        if a[key] != b[key]:
            raise ValueError('Comparison mismatch: '+key)
    delta = np.asarray(a['score_per_1000_actions']) - b['score_per_1000_actions']
    rng = np.random.default_rng(5913)
    means = delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1)
    ci = np.quantile(means,[.025,.975])
    return dict(delta_reward_per_1000_actions=float(delta.mean()),
                ci95_low=float(ci[0]),ci95_high=float(ci[1]),streams=len(delta))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();c=args.campaign.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    stamp=dt.datetime.now().astimezone().isoformat()
    manifest=read(c/'campaign.json');jobs=read(c/'jobs.json')
    diagnostics={};train_rows=[];episode_curves={}
    wanted=['real_step','_time','rmean_episode_return','search/mean_steps',
        'search/forced_stop_count','actor/voc_gate_projection_post_error_norm',
        'gradient/voc_gru_applied_norm','gradient/gru_update_norm',
        'actor/amp_skip_count','voc/amp_skip_count',
        'actor/nonfinite_gradient_parameter_count','voc/nonfinite_gradient_parameter_count',
        'training/bc_targets_committed','actor/voc_delta_q','actor/voc_td_rmse',
        'actor/voc_holdout_cumulative_td_rmse']
    for cell in manifest['cells']:
        game=cell['game'];run=c/'cells'/cell['name']
        d=pd.read_csv(run/'logs.csv',usecols=lambda k:k in wanted)
        last=d.iloc[-1];recent=d[d.real_step>=last.real_step-500000]
        hour=d[d._time>=last._time-3600]
        rate=(hour.real_step.iloc[-1]-hour.real_step.iloc[0])/(hour._time.iloc[-1]-hour._time.iloc[0])
        collector=read(run/'collector_metrics.json')
        episodes=pd.read_json(run/'episodes.jsonl',lines=True)
        episodes['rolling100']=episodes.return_raw.rolling(100,min_periods=100).mean()
        episodes['game']=game;episode_curves[game]=episodes
        control=pd.read_json(run/'control_samples.jsonl',lines=True)
        root=control[(control.real_step>=last.real_step-500000)&(control['count']==0)&control.valid]
        di=dict(training_rows=len(d),recent_root_rows=len(root),
            root_p_continue_quantiles=root.p_continue.quantile([.05,.5,.95]).to_dict(),
            actor_forced_stop_sum=float(d['search/forced_stop_count'].sum()),
            collector_forced_stop_count=collector['forced_stop_count'],
            collector_cap_decisions=collector['count_30_decisions'])
        for key in wanted[4:]:
            if key in d:
                di[key]=dict(latest=float(last[key]),recent_median=float(recent[key].median()),
                    all_max=float(d[key].max()),nonfinite_rows=int((~np.isfinite(d[key])).sum()))
        diagnostics[game]=di
        train_rows.append(dict(game=game,job=jobs['training'][cell['name']]['job_id'],node=cell['node'],
            state=jobs['training'][cell['name']]['state'],actor_env_steps=int(last.real_step),
            progress_percent=float(last.real_step/100000),physical_env_steps=collector['physical_env_steps'],
            recent_actions_per_second=float(rate),completed_episodes=collector['episodes_completed'],
            recent100_episode_return=float(episodes.return_raw.tail(100).mean()),
            legacy_rmean_episode_return=float(last.rmean_episode_return),
            bc_targets_committed=int(last['training/bc_targets_committed'])))
    rows=[];comparisons=[];voc_rows=[];results={};scope=[]
    paths=sorted((c/'evaluations').glob('*/summary.json'))
    for path in paths:
        s=read(path)
        if s.get('status')!='COMPLETE':continue
        t=s['task'];game=t['game'];step=t['real_step'];kind=t['kind']
        entry=dict(game=game,milestone=step,kind=kind,**s['metrics'],
            actor_checkpoint_step=s['checkpoint']['actor_real_step'],
            model_checkpoint_step=s['checkpoint']['model_real_step'],source=str(path))
        rows.append(entry);results[game,step,kind]=(s,path.parent)
        scope.append(str(path))
        if kind=='benefit':
            learned=read(path.parent/'score_learned.json')
            for baseline in ('fixed','random'):
                reference=read(path.parent/f'score_{baseline}.json')
                comparisons.append(dict(game=game,milestone=step,baseline=baseline,
                    learned_compute=learned['mean_computations_per_action'],
                    baseline_compute=reference['mean_computations_per_action'],
                    learned_reward=np.mean(learned['score_per_1000_actions']),
                    baseline_reward=np.mean(reference['score_per_1000_actions']),
                    same_gpu_job=True,**compare(learned,reference)))
            voc_rows.append(dict(game=game,milestone=step,
                **{k:v for k,v in s['voc'].items() if k!='state_records'}))
    for (game,step,kind),(summary,directory) in results.items():
        if kind!='score':continue
        learned=read(directory/'score.json')
        for reference_kind in ('score_zero','score_sign'):
            ref=results.get((game,step,reference_kind))
            if ref is None:continue
            reference=read(ref[1]/'score.json')
            comparisons.append(dict(game=game,milestone=step,baseline=reference_kind,
                learned_compute=learned['mean_computations_per_action'],
                baseline_compute=reference['mean_computations_per_action'],
                learned_reward=np.mean(learned['score_per_1000_actions']),
                baseline_reward=np.mean(reference['score_per_1000_actions']),
                same_gpu_job=False,**compare(learned,reference)))
    training=pd.DataFrame(train_rows);scores=pd.DataFrame(rows);comparison=pd.DataFrame(comparisons)
    training.to_csv(out/'training_progress.csv',index=False)
    scores.to_csv(out/'evaluation_scores.csv',index=False)
    comparison.to_csv(out/'reward_comparisons.csv',index=False)
    pd.DataFrame(voc_rows).to_csv(out/'voc_benefit.csv',index=False)
    pd.concat(episode_curves.values()).to_csv(out/'episode_curves.csv',index=False)
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    qa=dict(checked_at=stamp,completed_evaluations=len(rows),source_summaries=scope,
        fixed_random_protocol='Within each benefit job: same checkpoint, seed, horizon, GPU; calibrated approximate mean compute',
        zero_sign_protocol='Separate jobs; same checkpoint, seed, horizon; GPU type may differ',
        inference='One training seed, eight evaluation streams; interim uncorrected intervals, no checkpoint selection',
        paired_support='32 states from eight independent streams, two replications each in current benefit results',
        forced_stop_logging='Collector counter remains zero despite cap hits. Use actor forced-stop counts; frozen running source left untouched.',
        numeric_zero_intervals='Allocation CI endpoints of order 1e-18 are floating-point zero, not directional evidence.',
        curve_scope='Training returns use the last100 completed games. Evaluation windows grow from4096 to16384 actions/stream at3M; long episodes jobs use50000.')
    (out/'qa.json').write_text(json.dumps(qa,indent=2))
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    styles={'score':('VOC native','#176c9e'),'score_zero':('Zero search','#333333'),
            'score_sign':('Sign gate','#e68623')}
    for col,game in enumerate(('pong','enduro','spaceinvaders')):
        ep=episode_curves[game]
        axes[0,col].plot(ep.real_step/1e6,ep.rolling100,color='#176c9e',lw=1.7)
        axes[0,col].set_title({'pong':'Pong','enduro':'Enduro','spaceinvaders':'Space Invaders'}[game])
        axes[0,col].set_ylabel('Training return\nlast100 completed games')
        for kind,(label,color) in styles.items():
            data=scores[(scores.game==game)&(scores.kind==kind)].sort_values('milestone')
            axes[1,col].plot(data.milestone/1e6,data.raw_reward_per_1000_actions,'o-',label=label,color=color,lw=1.5)
        axes[1,col].set_ylabel('Evaluation raw reward\nper1,000 environment actions')
        for ax in axes[:,col]:
            ax.set_xlabel('Training environment actions (millions)');ax.grid(alpha=.2)
    axes[1,0].legend(fontsize=8)
    fig.suptitle('Interim reward learning and search interventions — one training seed\nEvaluation horizons differ across milestones; comparisons within a milestone share horizon',fontsize=11)
    fig.savefig(out/'progress.png',dpi=170);fig.savefig(out/'progress.pdf');plt.close(fig)
    lines=[f'# VOC10M interim review — {stamp}', '',
        'Task learning is clear in Pong. Useful computation allocation is not yet established across games.', '',
        '## Training progress', '',markdown_table(training), '',
        '## Latest matched-compute comparisons', '',
        'Raw reward gain per1,000 environment actions; native minus baseline. Uncorrected95% stream-bootstrap intervals.', '']
    for game in ('pong','enduro','spaceinvaders'):
        matched=comparison[(comparison.game==game)&comparison.baseline.isin(['fixed','random'])]
        last=matched[matched.milestone==matched.milestone.max()]
        lines.append(markdown_table(last[['game','milestone','baseline','learned_reward','baseline_reward','delta_reward_per_1000_actions','ci95_low','ci95_high']]));lines.append('')
    lines += ['## Interpretation', '',
        '- Pong:3M native long evaluation averages+12.26 over35 completed games, after approximately-21 at1M and-20 at2M. Late learning is real.',
        '- Pong3M beats fixed allocation in this exploratory comparison but does not beat matched random allocation. Zero-search point estimate is higher than native in the matched-horizon score job.',
        '- Enduro1M offers a tentative paired-VOC signal, but only2/64 paired returns change task reward and the RMSE improvement interval crosses zero.',
        '- Space Invaders1M has4/64 paired reward changes and no clear VOC prediction or allocation gain. Later checkpoints are needed.',
        '- Pong3M has0/64 paired task-return changes. EMA-VOC RMSE is0.02349 versus0.00093 for predicting zero. This is a local probe of sampled states, not proof that search has no effect everywhere.',
        '- EMA projection errors remain zero and selective GRU gradients are nonzero; these establish the training connection, not useful planning.',
        '- Root gate probabilities vary in the recent single-stream training sample. Training weights also change within this window, so this is descriptive only.',
        '- Collector forced_stop_count is undercounted; actor search/forced_stop_count and exact budget histograms remain available. No training job or frozen source was changed.', '',
        'Keep the registered training condition through10M. Next decisive tests are Pong5M and Enduro/Space Invaders3M matched-budget and paired-return evaluations. No new hyperparameter selection from these interim outcomes.', '',
        'See qa.json for sampling, horizon, counter and interval limitations.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(out)
    print(training.to_json(orient='records'))
    print(comparison[comparison.baseline.isin(['fixed','random','score_zero'])].to_json(orient='records'))


if __name__=='__main__':main()
