"""Matched, within-bout trajectories preceding human action transitions.

Use only held-out session replays. Preserve one cohort across all lags within
each horizon. Distinguish a strictly preceding change from a boundary jump.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import itertools
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.io import sha,write_json
from common.paths import OUTPUTS,GAMES,tag
from common.outcomes import risk_set
from common.matching import corrected_pair_root

METRICS={
    'M':'Initial support for ongoing action',
    'ema_initial':'Initial VOC',
    'log_budget':'Planning count: log(1 + B)',
    'U_all':'Absolute value updating / B',
    'incumbent_down_mean':'Ongoing-action devaluation / B',
    'alternatives_up_mean':'Alternative-action updating / B',
}
POPS=('noop_to_active','active_to_noop')
HORIZONS=(5,10)
ENDPOINTS=('preboundary_change','boundary_jump')
BASECOLS=['run_label','real_step','real_step_id','episode_id','session','checkpoint','held_out',
          'incumbent_action','human_action','scored','M','additional_computations','ema_initial',
          'U_all','bout_age','n_actions','prior_done','decision_time','first_control_native_event',
          'last_control_native_event','real_action_native_event']


def verify(path,expected,sources):
    digest=sha(path)
    if digest!=expected:raise ValueError('Source changed: '+str(path))
    sources[str(path.resolve())]=digest


def load_unit(subject,game,sources):
    label=tag(subject,game);parts=[]
    for cp in range(1,5):
        root=OUTPUTS/'00/derived'/label/f'leave{cp:02d}'
        meta=json.loads((root/'manifest.json').read_text())
        path=root/'decisions.csv.gz';verify(path,meta['artifacts'][path.name],sources)
        z=pd.read_csv(path,usecols=BASECOLS)
        z=z.loc[z.session.eq(cp)].copy()
        assert z.checkpoint.eq(cp).all() and z.held_out.eq(True).all()
        assert (z.first_control_native_event<=z.last_control_native_event).all()
        assert (z.last_control_native_event<z.real_action_native_event).all()
        parts.append(z)
    d=pd.concat(parts,ignore_index=True).sort_values(['run_label','real_step']).reset_index(drop=True)
    assert not d.real_step_id.duplicated().any()
    root=OUTPUTS/'02/value_update_candidates_20261111'/label
    meta=json.loads((root/'manifest.json').read_text());path=root/'candidate_decisions.csv.gz'
    verify(path,meta['artifacts'][path.name],sources)
    c=pd.read_csv(path,usecols=['real_step_id','human_action','additional_computations','U_all',
                               'incumbent_down_mean','alternatives_up_mean'])
    d=d.merge(c,on='real_step_id',validate='one_to_one',how='left',suffixes=('','_candidate'),indicator=True)
    assert d._merge.eq('both').all()
    for col in ('human_action','additional_computations','U_all'):
        np.testing.assert_allclose(d[col],d[col+'_candidate'],equal_nan=True)
    d['log_budget']=np.log1p(d.additional_computations)
    start=(d.run_label.ne(d.run_label.shift())|d.episode_id.ne(d.episode_id.shift())|
           d.real_step.diff().ne(1)|d.prior_done.eq(1))
    d['segment']=start.cumsum()
    continuing=(~start)&d.scored.eq(1)
    assert d.loc[continuing,'incumbent_action'].eq(d.human_action.shift()[continuing]).all()
    root=corrected_pair_root(OUTPUTS,label);meta=json.loads((root/'manifest.json').read_text())
    path=root/'tables/matched_pairs.csv';verify(path,meta['artifacts']['tables/matched_pairs.csv'],sources)
    pairs=pd.read_csv(path,usecols=['run_label','population','switch_id','persist_id'])
    return d,pairs


def history_eligibility(d,horizon):
    """All past actions AND all incumbent labels refer to this same action.

    The anchor's human action is deliberately not used for eligibility. It may
    switch at lag zero. Past windows never cross episodes, gaps or bouts.
    """
    ix=np.arange(len(d));past=ix[:,None]+np.arange(-horizon,1)[None,:]
    safe=np.maximum(past,0)
    incumbent=d.incumbent_action.to_numpy(int)
    human=d.human_action.to_numpy(int)
    segment=d.segment.to_numpy()
    valid=(past[:,0]>=0)&(incumbent>=0)&d.scored.eq(1).to_numpy()
    valid&=(segment[safe]==segment[:,None]).all(1)
    valid&=(incumbent[safe]==incumbent[:,None]).all(1)
    valid&=(human[safe[:,:-1]]==incumbent[:,None]).all(1)
    valid&=d.scored.to_numpy()[safe].astype(bool).all(1)
    return valid,past


def check_history_contract():
    # At index 3 an action switches: its two preceding decisions are valid;
    # the following decision crosses the action boundary and is not eligible.
    d=pd.DataFrame(dict(incumbent_action=[0,0,0,0,1,1,1],
                        human_action=[0,0,0,1,1,1,1],scored=1,
                        segment=[1,1,1,1,1,1,2]))
    valid,_=history_eligibility(d,2)
    assert valid.tolist()==[False,False,True,True,False,False,False]
    # Same time trends in cases and controls must yield no difference in change.
    t=np.arange(6,dtype=float);case=t+2;control=t
    delta=case-control
    assert np.isclose(delta[-3:-1].mean()-delta[:3].mean(),0)
    # A difference appearing only at lag zero is not a preceding change.
    delta=np.array([0,0,0,0,0,1.])
    assert delta[-3:-1].mean()-delta[:3].mean()==0 and delta[-1]-delta[-2]==1


def unit_analysis(d,pairs,subject,game):
    location=pd.Series(np.arange(len(d)),index=d.real_step_id)
    metrics=d[list(METRICS)].to_numpy(float)
    rows=[];contrasts=[];coverage=[]
    for horizon in HORIZONS:
        valid,indices=history_eligibility(d,horizon)
        lags=np.arange(-horizon,1)
        for pop in POPS:
            risk=risk_set(d,pop);outcome=risk.set_index('real_step_id').outcome
            for run,p in pairs.loc[pairs.population.eq(pop)].groupby('run_label'):
                ci=location.loc[p.switch_id].to_numpy();mi=location.loc[p.persist_id].to_numpy()
                assert outcome.loc[p.switch_id].eq(1).all() and outcome.loc[p.persist_id].eq(0).all()
                assert d.iloc[ci].run_label.eq(run).all() and d.iloc[mi].run_label.eq(run).all()
                both=valid[ci]&valid[mi];cix=ci[both];mix=mi[both]
                coverage.append(dict(subject=subject,game=game,population=pop,horizon=horizon,run_label=run,
                                     original_pairs=len(p),case_history_available=int(valid[ci].sum()),
                                     control_history_available=int(valid[mi].sum()),retained_pairs=int(both.sum()),
                                     unique_controls=len(np.unique(mix))))
                if not len(cix):continue
                a=metrics[indices[cix]];b=metrics[indices[mix]]
                rr=risk.loc[risk.run_label.eq(run)]
                for j,metric in enumerate(METRICS):
                    good=np.isfinite(a[:,:,j]).all(1)&np.isfinite(b[:,:,j]).all(1)
                    n=int(good.sum())
                    if n<20:continue
                    scale=float(rr[metric].std(ddof=0));center=float(rr[metric].mean())
                    if not np.isfinite(scale):continue
                    informative=scale>1e-10
                    aa=a[good,:,j];bb=b[good,:,j]
                    case=(aa.mean(0)-center)/scale if informative else np.zeros(len(lags))
                    maintain=(bb.mean(0)-center)/scale if informative else np.zeros(len(lags))
                    difference=case-maintain
                    early_c=case[:3].mean();early_m=maintain[:3].mean()
                    centered=difference-difference[:3].mean()
                    for k,lag in enumerate(lags):
                        rows.append(dict(subject=subject,game=game,population=pop,horizon=horizon,
                                         run_label=run,metric=metric,lag=int(lag),n_pairs=n,
                                         informative=informative,case=case[k],maintain=maintain[k],
                                         difference=difference[k],centered_difference=centered[k],
                                         case_change=case[k]-early_c,maintain_change=maintain[k]-early_m,
                                         raw_case=aa[:,k].mean(),raw_maintain=bb[:,k].mean()))
                    for endpoint,ci,mi in [
                        ('preboundary_change',case[-3:-1].mean()-early_c,maintain[-3:-1].mean()-early_m),
                        ('boundary_jump',case[-1]-case[-2],maintain[-1]-maintain[-2])]:
                        contrasts.append(dict(subject=subject,game=game,population=pop,horizon=horizon,
                                              run_label=run,metric=metric,endpoint=endpoint,n_pairs=n,
                                              informative=informative,case_change=ci,maintain_change=mi,value=ci-mi))
    return pd.DataFrame(rows),pd.DataFrame(contrasts),pd.DataFrame(coverage)


def infer(subjects):
    records=[];vectors=[]
    keys=['game','population','horizon','metric','endpoint']
    for key in itertools.product(GAMES,POPS,HORIZONS,METRICS,ENDPOINTS):
        z=subjects
        for name,value in zip(keys,key):z=z.loc[z[name].eq(value)]
        v=z.set_index('subject').value.reindex(range(1,7)).to_numpy(float)
        good=np.isfinite(v);informative=z.informative_runs.gt(0).any()
        status='incomplete_subjects' if not good.all() else ('ok' if informative else 'no_recorded_variation')
        records.append(dict(zip(keys,key),n_subjects=int(good.sum()),mean=float(v[good].mean()) if good.any() else np.nan,
                            sem=float(v[good].std(ddof=1)/np.sqrt(good.sum())) if good.sum()>1 else np.nan,
                            positive_subjects=int((v[good]>0).sum()),negative_subjects=int((v[good]<0).sum()),
                            status=status,p=np.nan,p_max_t=np.nan))
        vectors.append(v if status=='ok' else np.zeros(6))
    matrix=np.column_stack(vectors)
    signs=np.array(list(itertools.product([-1,1],repeat=6)))
    perm=signs[:,:,None]*matrix[None,:,:]
    null=np.abs(perm.mean(1))/np.maximum(perm.std(1,ddof=1)/np.sqrt(6),1e-12)
    obs=np.abs(matrix.mean(0))/np.maximum(matrix.std(0,ddof=1)/np.sqrt(6),1e-12)
    for i,row in enumerate(records):
        if row['status']=='ok':
            row['p']=float((null[:,i]>=obs[i]-1e-12).mean())
            row['p_max_t']=float((null.max(1)>=obs[i]-1e-12).mean())
    return pd.DataFrame(records)


def figures(subject_curves,output,metrics=None,prefix='pretransition'):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors={'noop_to_active':'#bd6337','active_to_noop':'#237da5'}
    labels={'noop_to_active':'NOOP → active','active_to_noop':'Active → NOOP'}
    metrics=list(METRICS)[:4] if metrics is None else metrics
    for horizon in HORIZONS:
        for adjusted in (True,False):
            measure='centered_difference' if adjusted else 'difference'
            z=subject_curves.loc[subject_curves.horizon.eq(horizon)]
            agg=z.groupby(['game','population','metric','lag'])[measure].agg(['mean','sem','count']).reset_index()
            fig,axes=plt.subplots(len(metrics),3,figsize=(12.6,max(8,2.1*len(metrics)+2)),sharex=True,sharey='row')
            for j,game in enumerate(GAMES):
                for i,metric in enumerate(metrics):
                    ax=axes[i,j]
                    participant_counts=[]
                    for pop in POPS:
                        r=agg.loc[agg.game.eq(game)&agg.population.eq(pop)&agg.metric.eq(metric)].sort_values('lag')
                        # Fixed participant membership at every lag; report
                        # incomplete groups descriptively, without a group test.
                        assert r['count'].nunique()<=1
                        n=int(r['count'].iloc[0]) if len(r) else 0
                        participant_counts.append(n)
                        if len(r)!=horizon+1 or n<2:continue
                        x=r.lag.to_numpy();y=r['mean'].to_numpy();se=r['sem'].to_numpy()
                        ax.plot(x,y,color=colors[pop],lw=1.8,marker='o',ms=3,label=labels[pop])
                        ax.fill_between(x,y-se,y+se,color=colors[pop],alpha=.15)
                    ax.axhline(0,color='#777777',lw=.7)
                    ax.axvline(0,color='#777777',ls=':',lw=1)
                    ax.grid(axis='y',alpha=.15)
                    if i==0:
                        ax.set_title(GAMES[game]+'\n'+f'n = {participant_counts[0]} / {participant_counts[1]} participants',fontsize=11)
                    if j==0:ax.set_ylabel(METRICS[metric]+'\n(SD units)',fontsize=9)
                    if i==len(metrics)-1:ax.set_xlabel('Decision steps relative to transition')
                    ax.set_xticks(np.arange(-horizon,1,1 if horizon==5 else 2))
            title='Change in the transition–maintenance difference' if adjusted else 'Transition minus matched maintenance (no baseline subtraction)'
            fig.suptitle(title,fontsize=14,y=.99)
            handles,legend=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.53,.965),ncol=2,frameon=False)
            subtitle='Each curve subtracts its own first-three-step baseline. Shading: participant SEM.' if adjusted else 'Within-run standardization; equal runs, then equal participants. Shading: participant SEM.'
            fig.text(.53,.895 if len(metrics)<=2 else .915,subtitle,ha='center',fontsize=9)
            fig.text(.53,.015,'Same ongoing action throughout the preceding window · n shown as NOOP→active / active→NOOP · lag 0 is pre-action',ha='center',fontsize=9)
            fig.tight_layout(rect=[0,.04,1,.86 if len(metrics)<=2 else .90])
            name=f'{prefix}_{"change" if adjusted else "levels"}_h{horizon}_coverage'
            fig.savefig(output/(name+'.png'),dpi=170);fig.savefig(output/(name+'.pdf'));plt.close(fig)


def component_figures(subject_curves,output,horizon=5):
    """Show both groups: a flat contrast does not imply flat model features."""
    metrics=list(METRICS)[:4]
    for population in POPS:
        z=subject_curves.loc[subject_curves.horizon.eq(horizon)&subject_curves.population.eq(population)]
        fig,axes=plt.subplots(len(metrics),3,figsize=(12.6,10),sharex=True,sharey='row')
        for j,game in enumerate(GAMES):
            for i,metric in enumerate(metrics):
                ax=axes[i,j];r=z.loc[z.game.eq(game)&z.metric.eq(metric)]
                r=r.loc[r.case.notna()&r.maintain.notna()]
                count=r.groupby('lag').subject.nunique()
                assert count.nunique()<=1
                n=int(count.iloc[0]) if len(count) else 0
                for key,label,color in [('case','Human transition','#b75c39'),('maintain','Matched maintenance','#347aa2')]:
                    a=r.groupby('lag')[key].agg(['mean','sem']).reset_index()
                    if n<2:continue
                    x=a.lag.to_numpy();y=a['mean'].to_numpy();se=a['sem'].to_numpy()
                    ax.plot(x,y,color=color,lw=1.7,marker='o',ms=3,label=label)
                    ax.fill_between(x,y-se,y+se,color=color,alpha=.14)
                if i==0:ax.set_title(GAMES[game]+f' · n={n}')
                if j==0:ax.set_ylabel(METRICS[metric]+'\n(within-run SD)',fontsize=9)
                if i==len(metrics)-1:ax.set_xlabel('Decision steps before transition')
                ax.axvline(0,color='#888888',ls=':',lw=.8);ax.grid(axis='y',alpha=.15)
                ax.set_xticks(range(-horizon,1))
        name='NOOP → active' if population=='noop_to_active' else 'Active → NOOP'
        fig.suptitle(name+': transition and maintenance shown separately',y=.99,fontsize=14)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.52,.965),frameon=False)
        fig.text(.52,.925,'Standardized within each run; equal runs, then participants. No subtraction between groups or time points.',ha='center',fontsize=9)
        fig.text(.52,.015,'Same eligible histories as the contrast plots; lag 0 is before the human action. Shading: participant SEM.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.04,1,.91])
        for ext in ('png','pdf'):fig.savefig(output/f'pretransition_components_{population}_h{horizon}.{ext}',dpi=170)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    check_history_contract()
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output/'protocol.json',dict(
        scope='Exploratory existing 10M tau .01 leave-session replays; not the ongoing Enduro run.',
        metrics=METRICS,populations=POPS,primary_horizon=5,sensitivity_horizon=10,
        time_axis='Replay decision steps; no assumption of uniform source timestamps or conversion to seconds.',
        matching='Reuse previous within-run incumbent/age-quintile/time-quintile/reward strata and nearest visual matching; retain only pairs with complete eligible histories.',
        eligibility='All lags from -H to 0 scored, consecutive within episode/run; same incumbent at every lag; all human actions at negative lags equal anchor incumbent. Anchor outcome is not used for eligibility.',
        active_bout='Active-to-other-active at lag zero remains a non-termination of the binary active bout, as in the previous risk set; the past window must nevertheless maintain one exact action.',
        cohort='Same pairs at all lags per metric/horizon; require >=20 pairs per run and >=3 runs per participant. Six participants required for group inference.',
        endpoints={'preboundary_change':'Mean(-2,-1) minus mean(-H,-H+1,-H+2), cases minus matched controls; excludes lag 0.',
                   'boundary_jump':'Lag 0 minus lag -1, cases minus matched controls.'},
        scale='Within-run SD of the full corresponding transition risk set; equal pair/run/subject weighting.',
        inference='Two-sided exact synchronized subject sign flips over all 64 combinations, max-|t| correction over 6 metrics x 3 games x 2 directions x 2 horizons x 2 endpoints = 144 cells. Constant measures not testable.',
        limits='Conditioned on sufficiently long bouts and matching support; not all transitions. Descriptive temporal associations, not prospective predictive validation or causal planning effects. U=0 includes no recorded revisions.'))
    sources={};curves=[];effects=[];cover=[];cohort=[]
    for game,subject in itertools.product(GAMES,range(1,7)):
        d,p=load_unit(subject,game,sources)
        c,e,v=unit_analysis(d,p,subject,game)
        curves.append(c);effects.append(e);cover.append(v)
        cohort.append(dict(subject=subject,game=game,decisions=len(d),runs=d.run_label.nunique()))
        pd.concat(effects).to_csv(args.output/'run_effects.csv',index=False)
        print(f'Completed {tag(subject,game)}; eligible pairs h5={v.loc[v.horizon.eq(5),"retained_pairs"].sum():,}',flush=True)
    run_curves=pd.concat(curves,ignore_index=True);run_effects=pd.concat(effects,ignore_index=True)
    run_curves.to_csv(args.output/'run_trajectories.csv.gz',index=False)
    pd.concat(cover).to_csv(args.output/'coverage.csv',index=False)
    pd.DataFrame(cohort).to_csv(args.output/'cohort.csv',index=False)
    keys=['subject','game','population','horizon','metric']
    values=['case','maintain','difference','centered_difference','case_change','maintain_change','raw_case','raw_maintain']
    sc=run_curves.groupby(keys+['lag'],as_index=False).agg(**{m:(m,'mean') for m in values},n_runs=('difference','size'))
    sc.loc[sc.n_runs.lt(3),values]=np.nan
    sc.to_csv(args.output/'subject_trajectories.csv',index=False)
    se=run_effects.groupby(keys+['endpoint'],as_index=False).agg(value=('value','mean'),n_runs=('value','size'),
        case_change=('case_change','mean'),maintain_change=('maintain_change','mean'),informative_runs=('informative','sum'))
    se.loc[se.n_runs.lt(3),'value']=np.nan
    se.to_csv(args.output/'subject_effects.csv',index=False)
    group=infer(se);group.to_csv(args.output/'group_results.csv',index=False)
    figures(sc,args.output)
    component_figures(sc,args.output)
    write_json(args.output/'manifest.json',dict(status='COMPLETED',sources=sources,
        total_decisions=sum(c['decisions'] for c in cohort),test_cells=len(group),
        statuses=group.status.value_counts().to_dict(),corrected_significant=int(group.p_max_t.lt(.05).sum()),
        qa=dict(source_hashes_verified=True,heldout_only=True,no_duplicate_decisions=True,
                pre_action_alignment=True,paired_outcomes_verified=True,history_boundary_self_checks=True,
                fixed_pair_cohort_across_lags=True),
        artifacts={p.name:sha(p) for p in args.output.iterdir() if p.is_file()}))
    print(group.loc[group.horizon.eq(5),['game','population','metric','endpoint','mean','positive_subjects','negative_subjects','p_max_t','status']].to_string(index=False))


if __name__=='__main__':main()
