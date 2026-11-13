"""Exploratory planning/bout associations using existing held-out matched pairs.

This does not change training or select checkpoints from the observed effects.
All six measures and all three outcomes belong to one 54-cell test family.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import itertools
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import sha, write_json
from common.paths import OUTPUTS, GAMES, tag
from common.outcomes import risk_set
from common.matching import corrected_pair_root

METRICS = {
    'M': 'Initial support for ongoing action',
    'log_budget': 'Planning count: log(1 + B)',
    'log_depth': 'Search depth: log(1 + depth)',
    'ema_initial': 'Initial VOC (EMA)',
    'U_all': 'Absolute value updating / B',
    'incumbent_down_mean': 'Ongoing-action devaluation / B',
}
POPS = ('overall', 'noop_to_active', 'active_to_noop')
BASECOLS = ['run_label','real_step','real_step_id','episode_id','session','checkpoint','held_out',
            'incumbent_action','human_action','scored','M','additional_computations','max_depth',
            'ema_initial','U_all','bout_age','n_actions','first_control_native_event',
            'last_control_native_event','real_action_native_event']


def verified(path, expected, sources):
    digest = sha(path)
    if digest != expected:
        raise ValueError('Source hash mismatch: ' + str(path))
    sources[str(path.resolve())] = digest


def load_subject(subject, game, sources):
    label = tag(subject, game)
    parts = []
    for cp in range(1, 5):
        root = OUTPUTS/'00/derived'/label/f'leave{cp:02d}'
        manifest = json.loads((root/'manifest.json').read_text())
        assert manifest['status'] == 'COMPLETED'
        path = root/'decisions.csv.gz'
        verified(path, manifest['artifacts'][path.name], sources)
        z = pd.read_csv(path, usecols=BASECOLS)
        z = z.loc[z.session.eq(cp)].copy()
        assert z.checkpoint.eq(cp).all() and z.held_out.eq(True).all()
        assert (z.first_control_native_event <= z.last_control_native_event).all()
        assert (z.last_control_native_event < z.real_action_native_event).all()
        parts.append(z)
    d = pd.concat(parts, ignore_index=True)
    assert not d.real_step_id.duplicated().any()
    root = OUTPUTS/'02/value_update_candidates_20261111'/label
    manifest = json.loads((root/'manifest.json').read_text())
    path = root/'candidate_decisions.csv.gz'
    verified(path, manifest['artifacts'][path.name], sources)
    c = pd.read_csv(path, usecols=['real_step_id','human_action','additional_computations',
                                 'U_all','incumbent_down_mean'])
    d = d.merge(c, on='real_step_id', validate='one_to_one', suffixes=('', '_candidate'),
                how='left', indicator=True)
    assert d._merge.eq('both').all()
    for col in ('human_action','additional_computations','U_all'):
        np.testing.assert_allclose(d[col], d[col+'_candidate'], equal_nan=True)
    assert d.loc[d.additional_computations.eq(0), 'U_all'].eq(0).all()
    d['log_budget'] = np.log1p(d.additional_computations)
    d['log_depth'] = np.log1p(d.max_depth)
    root = corrected_pair_root(OUTPUTS,label)
    manifest = json.loads((root/'manifest.json').read_text())
    path = root/'tables/matched_pairs.csv'
    verified(path, manifest['artifacts']['tables/matched_pairs.csv'], sources)
    pairs = pd.read_csv(path, usecols=['run_label','population','switch_id','persist_id'])
    return d, pairs


def contrasts(d, pairs, subject, game):
    runs = []
    coverage = []
    by_id = d.set_index('real_step_id')
    for pop in POPS:
        risk = risk_set(d, pop)
        outcome = risk.set_index('real_step_id').outcome
        for run, p in pairs.loc[pairs.population.eq(pop)].groupby('run_label'):
            sw, stay = by_id.loc[p.switch_id], by_id.loc[p.persist_id]
            assert sw.run_label.eq(run).all() and stay.run_label.eq(run).all()
            assert outcome.loc[p.switch_id].eq(1).all() and outcome.loc[p.persist_id].eq(0).all()
            assert np.array_equal(sw.incumbent_action, stay.incumbent_action)
            population = risk.loc[risk.run_label.eq(run)]
            coverage.append(dict(subject=subject, game=game, population=pop, run_label=run,
                                 n_risk=len(population), n_transitions=int(population.outcome.sum()),
                                 n_pairs=len(p), unique_controls=p.persist_id.nunique()))
            for metric in METRICS:
                a, b = sw[metric].to_numpy(float), stay[metric].to_numpy(float)
                finite = np.isfinite(a) & np.isfinite(b)
                sd = population[metric].std(ddof=0)
                status = 'ok' if finite.sum() >= 20 and np.isfinite(sd) else 'insufficient'
                if status == 'ok':
                    delta = float((a[finite]-b[finite]).mean())
                    # A constant observable supplies no descriptive difference.
                    value = delta/sd if sd > 1e-10 else 0.
                else:
                    delta = value = np.nan
                runs.append(dict(subject=subject, game=game, population=pop, run_label=run,
                                 metric=metric, n_pairs=int(finite.sum()), status=status,
                                 value=value, raw_difference=delta, scale=sd,
                                 switch_mean=float(a[finite].mean()) if finite.any() else np.nan,
                                 maintain_mean=float(b[finite].mean()) if finite.any() else np.nan))
    return pd.DataFrame(runs), pd.DataFrame(coverage)


def group_inference(subjects):
    rows, vectors = [], []
    for game, pop, metric in itertools.product(GAMES, POPS, METRICS):
        z = subjects.loc[subjects.game.eq(game)&subjects.population.eq(pop)&subjects.metric.eq(metric)]
        v = z.set_index('subject').value.reindex(range(1,7)).to_numpy(float)
        good = np.isfinite(v)
        measurable = bool(z.informative_runs.gt(0).any())
        rows.append(dict(game=game, population=pop, metric=metric, n_subjects=int(good.sum()),
                         mean=float(v[good].mean()) if good.any() else np.nan,
                         sem=float(v[good].std(ddof=1)/np.sqrt(good.sum())) if good.sum()>1 else np.nan,
                         positive_subjects=int((v[good]>0).sum()),negative_subjects=int((v[good]<0).sum()),
                         status=('ok' if measurable else 'no_recorded_variation') if good.all() else 'incomplete',
                         p=np.nan,p_max_t=np.nan))
        vectors.append(v if good.all() and measurable else np.zeros(6))
    matrix = np.column_stack(vectors)
    signs = np.array(list(itertools.product([-1,1], repeat=6)))
    perm = signs[:,:,None]*matrix[None,:,:]
    null = np.abs(perm.mean(1))/np.maximum(perm.std(1,ddof=1)/np.sqrt(6),1e-12)
    observed = np.abs(matrix.mean(0))/np.maximum(matrix.std(0,ddof=1)/np.sqrt(6),1e-12)
    for i, row in enumerate(rows):
        if row['status']=='ok':
            row['p'] = float(np.mean(null[:,i]>=observed[i]-1e-12))
            row['p_max_t'] = float(np.mean(null.max(1)>=observed[i]-1e-12))
    return pd.DataFrame(rows)


def plot(group, output):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig, axes = plt.subplots(1,3,figsize=(13,5.4),sharey=True)
    limit = max(.1, group['mean'].abs().max())
    for ax, game in zip(axes,GAMES):
        z = group.loc[group.game.eq(game)].set_index(['metric','population'])
        arr = np.array([[z.loc[(m,p),'mean'] for p in POPS[1:]] for m in METRICS])
        im = ax.imshow(arr,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        ax.set(xticks=[0,1],xticklabels=['NOOP → active','Active → NOOP'],
               yticks=range(len(METRICS)),yticklabels=list(METRICS.values()),title=GAMES[game])
        for i,m in enumerate(METRICS):
            for j,p in enumerate(POPS[1:]):
                row=z.loc[(m,p)]
                count=int(row.positive_subjects if row['mean']>0 else row.negative_subjects)
                text=f'{row["mean"]:+.3f}\n{count}/6 same direction'
                if row.status == 'no_recorded_variation':
                    text='No recorded variation\nAssociation not estimable'
                if row.p_max_t < .05: text += ' *'
                ax.text(j,i,text,ha='center',va='center',fontsize=8,
                        color='white' if abs(row['mean']) > limit*.65 else '#222222')
    fig.suptitle('Planning measures at human action-bout transitions',fontsize=14,y=.99)
    fig.text(.5,.92,'Matched transition minus maintenance · equal-run / equal-subject means',ha='center')
    fig.subplots_adjust(left=.245,right=.89,bottom=.19,top=.84,wspace=.12)
    cb=fig.colorbar(im,cax=fig.add_axes([.915,.22,.014,.57]))
    cb.set_label('Within-run SD units')
    fig.text(.5,.06,'Positive: higher at transitions. Negative: higher during maintenance. * Family-corrected p < .05.',
             ha='center',fontsize=9)
    fig.text(.5,.02,'Existing 10M leave-session models · 6 participants · exploratory associations, not causal planning effects',
             ha='center',fontsize=9,color='#555555')
    fig.savefig(output/'planning_bout_associations.png',dpi=180)
    fig.savefig(output/'planning_bout_associations.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output/'protocol.json',dict(
        scope='EXPLORATORY: existing 10M tau .01 VOC-GRU connected leave-session models. Not the ongoing minimum-one/cost-.0001/episode-memory Enduro run.',
        metrics=METRICS,outcomes=POPS,
        sample='Only session == checkpoint leave-session; each human decision used once.',
        matching='Reuse existing matches within run, incumbent action, bout-age quintile, run-progress quintile, and prior reward; nearest visual descriptor with replacement.',
        features='All recorded before current human action is executed. U uses repeated valid root-Q revisions / actual B, and is zero at B=0. Ongoing devaluation uses negative signed revisions of the incumbent branch / max(B,1).',
        estimand='Transition minus maintenance, divided by SD of the corresponding run/risk population. Equal pairs within run, equal runs within subject, equal six subjects.',
        minimum_pairs=20,minimum_runs=3,
        inference='All 64 synchronized subject sign flips, two-sided max-|t| over 6 metrics x 3 games x 3 outcomes = 54 cells. Requires symmetric subject-effect null; n=6 limits resolution.',
        active_bout='For active→NOOP, active→other-active is continued active behavior. Overall outcome instead tests exact-action bout termination.',
        limits='Descriptive matching does not eliminate all confounding; no predictive or causal claim. Models vary across subjects and held-out folds.'))
    sources={};runparts=[];coverparts=[];audits=[]
    for game,subject in itertools.product(GAMES,range(1,7)):
        d,p=load_subject(subject,game,sources)
        r,c=contrasts(d,p,subject,game)
        runparts.append(r);coverparts.append(c)
        audits.append(dict(subject=subject,game=game,n_decisions=len(d),
                           n_scored=int(d.scored.eq(1).sum()),n_runs=d.run_label.nunique(),
                           unique_ids=True,heldout_only=True,pre_action_features=True))
        pd.concat(runparts).to_csv(args.output/'run_contrasts.csv',index=False)
        print(f'Completed {tag(subject,game)}: {len(d):,} held-out decisions',flush=True)
    runs=pd.concat(runparts,ignore_index=True)
    subjects=runs.groupby(['subject','game','population','metric'],as_index=False).agg(
        value=('value','mean'),n_runs=('value','count'),raw_difference=('raw_difference','mean'),
        switch_mean=('switch_mean','mean'),maintain_mean=('maintain_mean','mean'),
        informative_runs=('scale',lambda x:int(x.gt(1e-10).sum())))
    subjects.loc[subjects.n_runs.lt(3),'value']=np.nan
    subjects.to_csv(args.output/'subject_contrasts.csv',index=False)
    group=group_inference(subjects)
    group.to_csv(args.output/'group_results.csv',index=False)
    pd.concat(coverparts).to_csv(args.output/'coverage.csv',index=False)
    pd.DataFrame(audits).to_csv(args.output/'cohort.csv',index=False)
    plot(group,args.output)
    write_json(args.output/'manifest.json',dict(status='COMPLETED',sources=sources,
        total_decisions=sum(x['n_decisions'] for x in audits),group_cells=len(group),
        complete_group_cells=int(group.status.eq('ok').sum()),
        corrected_cells=int(group.p_max_t.lt(.05).sum()),
        qa=dict(source_hashes_verified=True,unique_heldout_decisions=True,
                pair_outcomes_and_incumbents_verified=True,features_precede_human_execution=True)))
    print(group[['game','population','metric','mean','positive_subjects','negative_subjects','p_max_t']].to_string(index=False))


if __name__=='__main__':
    main()
