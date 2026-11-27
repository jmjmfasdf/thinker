"""Participant-preserving group summaries from canonical persisted results.

No capture, behavioral source, checkpoint, or training-log reads. Model/behavior
IDs and predictions are never averaged indiscriminately. Heatmaps are recomputed
on one common game/metric grid for each participant and only then averaged.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from common.result_store import _verify

RAW_TABLES={'observations','bouts','decisions','choices','decision_scores','decision_allocation','baseline_comparison_decisions',
    'voc_choices','voc_transitions','voc_decisions','voc_endpoints','voc_readout_predictions',
    'voc_temporal_records','voc_window_features',
    'choice_observations','task_state_observations','primary_predictions','primary_windows','primary_directional_features',
    'risk','risk_set','risk_observations','matched_pairs','heatmap_points',
    'action_support_windows','action_support_predictions'}
EXAMPLE_TABLES={'primary_examples','primary_example_trajectories','primary_example_human',
    'bout_examples','bout_example_trajectories'}
POPS=('overall','noop_to_active','active_to_other','active_to_noop')
LABELS={'overall':'Any action switch','noop_to_active':'No input → active',
        'active_to_other':'Active → any other action','active_to_noop':'Active → no input'}


def _read_unit(root):
    root=Path(root)
    meta=json.loads((root/'logs/run.json').read_text())
    _verify(root,meta,required=('analyze',))
    tables={}
    for p in sorted((root/'tables').glob('*.csv*')):
        key=p.name.removesuffix('.gz').removesuffix('.csv')
        if key in tables:raise ValueError(f'Ambiguous table encoding for {root}/{key}')
        tables[key]=pd.read_csv(p)
    return tables,meta


def _participant_means(d,keys,values):
    """Runs (when present) are averaged before participants, independently per cell."""
    run='run' if 'run' in d else 'run_label' if 'run_label' in d else None
    if run and run not in keys:
        d=d.groupby(['subject',run,*keys],dropna=False)[values].mean().reset_index()
    return d.groupby(['subject',*keys],dropna=False)[values].mean().reset_index()


def _summary(d,keys,value):
    sub=_participant_means(d,keys,[value])
    out=sub.groupby(keys,dropna=False)[value].agg(mean='mean',sem='sem',n_subjects='count').reset_index()
    # Descriptive between-participant uncertainty; no n=1 confidence interval.
    out['ci_low']=out['mean']-1.96*out['sem'];out['ci_high']=out['mean']+1.96*out['sem']
    return out


def _group_040608_summaries(number,tables):
    """Compact explicit participant summaries; never average IDs or readout rows."""
    def add(name,frame,keys,value):
        if frame.empty:return
        p=_participant_means(frame,keys,[value])
        tables['participant_'+name]=p
        tables['group_'+name]=_summary(p,keys,value)
    if number=='04':
        d=tables.get('voc_utility_regression',pd.DataFrame())
        if len(d):
            metrics=[c for c in ['mean_task_reward_difference','mean_computation_cost','utility_mean','utility_voc_slope'] if c in d]
            long=d.melt(id_vars=['subject','protocol','nominal_step'],value_vars=metrics,var_name='metric',value_name='value')
            add('paired_utility',long,['protocol','nominal_step','metric'],'value')
        tables.pop('state_outcomes',None)
    elif number=='06':
        d=tables.get('policy_revision_run_means',pd.DataFrame())
        if len(d):
            metrics=['same_phase_policy_js','stop_input_policy_js','total_policy_js']
            long=d.melt(id_vars=['subject','run','budget'],value_vars=[c for c in metrics if c in d],var_name='metric',value_name='value')
            add('policy_revision',long,['budget','metric'],'value')
        d=tables.get('actor_run_scores',pd.DataFrame())
        if len(d):add('actor_readout',d,['model'],'cross_entropy')
        for key in ['actor_predictions','next_choice_pairs','policy_revision']:tables.pop(key,None)
    elif number=='08':
        d=tables.get('temporal_records',pd.DataFrame());rows=[]
        if len(d):
            for (subject,run),p in d.groupby(['subject','run']):
                valid=p[['lag1_U_V','U_V']].dropna()
                ok=len(valid)>=3 and valid.lag1_U_V.std()>1e-10 and valid.U_V.std()>1e-10
                rows.append(dict(subject=subject,run=run,metric='lagged_U_V_correlation',
                    value=float(valid.lag1_U_V.corr(valid.U_V)) if ok else np.nan,
                    n_valid=len(valid),status='ok' if ok else 'insufficient_variation'))
            corr=pd.DataFrame(rows);tables['run_temporal_continuity']=corr
            add('temporal_continuity',corr,['metric'],'value')
            carry=d.groupby(['subject','run','carry']).budget.mean().reset_index()
            tables['run_carry_allocation']=carry
            add('carry_allocation',carry,['carry'],'budget')
        d=tables.get('run_scores',pd.DataFrame())
        if len(d):add('temporal_readout',d,['population','model'],'log_score')
        for key in ['temporal_records','predictions']:tables.pop(key,None)


def _render_group_040608(number,tables,title,save):
    """Participant lines and their equally weighted mean from saved summaries."""
    import matplotlib.pyplot as plt
    def curve(ax,part,group,x,value):
        for subject,d in part.groupby('subject'):
            d=d.sort_values(x);ax.plot(d[x],d[value],'-',color='#8395a6',alpha=.4,lw=.9)
        group=group.sort_values(x)
        ax.plot(group[x],group['mean'],'o-',color='#174e73',lw=2,label='Equal participant mean')
        ax.axhline(0,color='gray',lw=.6)
    def nested(name,value,order,populations=None):
        p=tables.get('participant_'+name,pd.DataFrame());g=tables.get('group_'+name,pd.DataFrame())
        if p.empty:return
        facets=sorted(p[populations].unique()) if populations else [None]
        fig,axs=plt.subplots(1,len(facets),squeeze=False,figsize=(5*len(facets),4))
        for ax,facet in zip(axs.ravel(),facets):
            sp=p.loc[p[populations].eq(facet)] if populations else p
            sg=g.loc[g[populations].eq(facet)] if populations else g
            models=[m for m in order if m in set(sp.model)]
            for subject,d in sp.groupby('subject'):
                z=d.set_index('model')[value].reindex(models)
                ax.plot(range(len(models)),z,'o-',alpha=.5,lw=1,label=f'sub{int(subject):03d}')
            z=sg.set_index('model')['mean'].reindex(models)
            ax.plot(range(len(models)),z,'o-',color='black',lw=2.2,label='Equal participant mean')
            ax.set(xticks=range(len(models)),xticklabels=[m.split('_',1)[0] for m in models],
                   ylabel=value.replace('_',' '),title=str(facet) if facet else title)
            ax.legend(fontsize=7)
        fig.suptitle(title+' · held-session readouts; equal runs within participant')
        save(fig,name+'_group')
    if number=='04':
        p=tables.get('participant_paired_utility',pd.DataFrame());g=tables.get('group_paired_utility',pd.DataFrame())
        if p.empty:return
        metrics=['mean_task_reward_difference','utility_mean','utility_voc_slope']
        labels=['Paired reward difference','Net utility after cost','Utility / one-SD predicted VOC']
        fig,axs=plt.subplots(1,3,figsize=(13,4))
        for ax,metric,label in zip(axs,metrics,labels):
            for color,(protocol,sp) in zip(plt.cm.tab10.colors,p.loc[p.metric.eq(metric)].groupby('protocol')):
                sg=g.loc[g.metric.eq(metric)&g.protocol.eq(protocol)].sort_values('nominal_step')
                for _,sub in sp.groupby('subject'):
                    sub=sub.sort_values('nominal_step');ax.plot(sub.nominal_step/1e6,sub.value,color=color,alpha=.3,lw=.8)
                ax.plot(sg.nominal_step/1e6,sg['mean'],'o-',color=color,label=str(protocol))
            ax.axhline(0,color='gray',lw=.6);ax.set(xlabel='Training steps (millions)',ylabel=label);ax.legend(fontsize=8)
        fig.suptitle(title+' · explicit paired input; equal participant means')
        save(fig,'paired_utility_group')
    elif number=='06':
        p=tables.get('participant_policy_revision',pd.DataFrame());g=tables.get('group_policy_revision',pd.DataFrame())
        if len(p):
            fig,axs=plt.subplots(1,2,figsize=(10,4))
            for ax,metric,label in zip(axs,['same_phase_policy_js','stop_input_policy_js'],
                    ['Computation endpoint change','Separate STOP-input/output change']):
                curve(ax,p.loc[p.metric.eq(metric)],g.loc[g.metric.eq(metric)],'budget','value')
                ax.set(xlabel='Actual computation count',ylabel='Policy JS divergence',title=label)
            fig.suptitle(title+' · equal runs, then equal participants; thin lines are participants')
            save(fig,'policy_revision_group')
        nested('actor_readout','cross_entropy',['R0_prior','R1_quantity','R2_search'])
    elif number=='08':
        p=tables.get('participant_temporal_continuity',pd.DataFrame());g=tables.get('group_temporal_continuity',pd.DataFrame())
        carry=tables.get('participant_carry_allocation',pd.DataFrame());cg=tables.get('group_carry_allocation',pd.DataFrame())
        if len(p) or len(carry):
            fig,axs=plt.subplots(1,2,figsize=(10,4))
            if len(p):
                for subject,sub in p.groupby('subject'):
                    axs[0].plot([subject],sub.value,'o',label=f'sub{int(subject):03d}')
                mean=float(g.loc[g.metric.eq('lagged_U_V_correlation'),'mean'].iloc[0])
                axs[0].axhline(mean,color='black',lw=1.5,label='Equal participant mean')
                axs[0].axhline(0,color='gray',lw=.6)
                axs[0].set(xticks=sorted(p.subject.unique()),xlabel='Participant',ylabel='Mean run correlation',title='Previous versus current U_V')
                axs[0].legend(fontsize=7)
            if len(carry):
                curve(axs[1],carry,cg,'carry','budget')
                axs[1].set(xticks=[0,1],xlabel='Recorded incoming tree carry',ylabel='Mean computation count',title='Carry and allocation')
            fig.suptitle(title+' · descriptive continuity; equal runs then equal participants')
            save(fig,'continuity_group')
        nested('temporal_readout','log_score',['T0_current','T1_previous_evaluation','T2_bout_history','T3_arrived_mismatch'],'population')

def _common_heatmaps(points,requested):
    from panels.mu_heatmaps import GRID,kernel_surface,mean_surfaces
    d=points.copy()
    if 'M_z' not in d:raise ValueError('Group M–U heatmaps require explicit within-run M_z')
    run='run' if 'run' in d else 'run_label'
    if run not in d:raise ValueError('Heatmap points need run identity for per-run normalization')
    for target,source in [('A_V','V_total'),('A_Q','Q_total')]:
        if target not in d and source in d:d[target]=d[source]
    budget='budget' if 'budget' in d else 'additional_computations'
    for kind in ['V','Q']:
        if f'U_{kind}' not in d and f'A_{kind}' in d and budget in d:d[f'U_{kind}']=d[f'A_{kind}']/d[budget].clip(lower=1)
    if 'switch' not in d:d['switch']=d.human_action.ne(d.incumbent_action).astype(int)
    surfaces=[];axes=[];coverage=[]
    minimum=min(4,len(requested))
    for metric in ['A_V','U_V','A_Q','U_Q']:
        if metric not in d:continue
        normalized=metric+'_relative'
        if normalized in d:r=d.assign(U_relative=d[normalized],U_all=d[metric]).copy()
        else:
            mean=d.groupby(['subject',run])[metric].transform('mean')
            relative=np.divide(d[metric],mean,out=np.zeros(len(d)),where=mean.gt(0))
            r=d.assign(U_relative=relative,U_all=d[metric]).copy()
        r=r.loc[np.isfinite(r.M_z)&np.isfinite(r.U_relative)&r.incumbent_action.ge(0)&r.human_action.ge(0)]
        if not len(r):continue
        xlim=np.quantile(r.M_z,[.01,.99]);ymax=float(r.U_relative.quantile(.99))
        if not np.isfinite(xlim).all() or np.ptp(xlim)<=0:xlim=np.array([-1.,1.])
        if not np.isfinite(ymax) or ymax<=0:
            positive=r.loc[r.U_relative.gt(0),'U_relative'];ymax=float(positive.quantile(.99)) if len(positive) else 1.
        ylim=np.array([0.,ymax]);x=np.linspace(*xlim,GRID);y=np.linspace(*ylim,GRID);xx,yy=np.meshgrid(x,y)
        axes.append(dict(metric=metric,x_min=xlim[0],x_max=xlim[1],y_min=ylim[0],y_max=ylim[1],
                         grid_size=GRID,required_subjects=minimum))
        for population in POPS:
            by_subject=[];local_subject=[];positive_subject=[]
            for subject in sorted(r.subject.unique()):
                z=r.loc[r.subject.eq(subject)].copy()
                if population=='noop_to_active':z=z.loc[z.incumbent_action.eq(0)].copy()
                elif population.startswith('active'):z=z.loc[z.incumbent_action.ne(0)].copy()
                z['outcome']=z.human_action.eq(0).astype(int) if population=='active_to_noop' else z['switch']
                prob,local,positive=kernel_surface(z,xlim,ylim)
                by_subject.append(prob);local_subject.append(local);positive_subject.append(positive)
                coverage.append(dict(subject=int(subject),metric=metric,population=population,n=len(z),
                    fraction_zero_update=float(z.U_all.eq(0).mean()),supported_pixels=int(np.isfinite(prob).sum())))
                surfaces.append(pd.DataFrame(dict(subject=int(subject),owner=f'sub{int(subject):03d}',metric=metric,
                    population=population,x=xx.ravel(),y=yy.ravel(),probability=prob.ravel(),
                    support=local.ravel(),positive_support=positive.ravel(),n_subjects=np.isfinite(prob).ravel().astype(int))))
            average,support=mean_surfaces(np.stack(by_subject),minimum=minimum)
            surfaces.append(pd.DataFrame(dict(subject=0,owner='group',metric=metric,population=population,
                x=xx.ravel(),y=yy.ravel(),probability=average.ravel(),support=support.ravel(),
                positive_support=np.sum(positive_subject,axis=0).ravel(),n_subjects=support.ravel())))
    return {'heatmap_surfaces':pd.concat(surfaces,ignore_index=True) if surfaces else pd.DataFrame(),
            'heatmap_axes':pd.DataFrame(axes),'heatmap_coverage':pd.DataFrame(coverage)}


def aggregate(number,args,units):
    """units: [(subject, game, canonical_result_root), ...] for exactly one game."""
    number=f'{int(number):02d}'
    if not units:raise ValueError('No completed participant results to aggregate')
    games={str(g) for _,g,_ in units};subjects=[int(s) for s,_,_ in units]
    if len(games)!=1 or len(subjects)!=len(set(subjects)):
        raise ValueError('Each group contains one game and one result per participant')
    game=next(iter(games));collected={};points=[];inputs=[]
    for subject,_,root in units:
        tables,meta=_read_unit(root)
        inputs.append(dict(subject=int(subject),game=game,result_root=str(root),
            analysis_signature=meta.get('stages',{}).get('analyze',{}).get('signature'),
            source_tables=len(tables)))
        for key,d in tables.items():
            if len(d) and 'subject' in d and not d.subject.eq(int(subject)).all():
                raise ValueError(f'Subject contamination in {root}/{key}')
            if number=='09' and key.startswith('action_support_') and len(d) and 'game' in d and not d.game.eq(game).all():
                raise ValueError(f'Action-support game contamination in {root}/{key}')
            d=d.copy();d['subject']=int(subject);d['game']=game
            if key=='heatmap_points':points.append(d)
            if key in RAW_TABLES or key in EXAMPLE_TABLES:continue
            # Existing per-person grids must never be concatenated and called group maps.
            if key.startswith('heatmap_'):continue
            collected.setdefault(key,[]).append(d)
    tables={key:pd.concat(parts,ignore_index=True) for key,parts in collected.items()}
    tables['group_inputs']=pd.DataFrame(inputs)
    if number=='02' and len(tables.get('baseline_comparison_subject_scores',[])):
        from analyses.validation_allocation import comparison_accuracy_tests, comparison_episode_tests
        # Recompute on paired participants, never concatenate/average individual p-values.
        tables['baseline_comparison_accuracy_tests']=comparison_accuracy_tests(tables)
        # The displayed group figure uses separate episode-level panels per person.
        tables.update(comparison_episode_tests(tables))
    requested=sorted(set(int(x) for x in getattr(args,'subjects',subjects)))
    missing=sorted(set(requested)-set(subjects))
    if number=='05' and len(tables.get('decision_updates',[])):
        # Subject histograms originally have different bin edges. Rebin their
        # recorded values once on shared axes before making any group curve.
        d=tables['decision_updates'];histograms=[]
        for metric in ['A_V','U_V','D_V','A_Q','U_Q']:
            finite=d.loc[np.isfinite(d[metric]),metric]
            if not len(finite):continue
            lo,hi=np.quantile(finite,[0,.995]);hi=max(hi,lo+1e-9);edges=np.linspace(lo,hi,41)
            for (subject,run),r in d.groupby(['subject','run']):
                count,_=np.histogram(r[metric],edges)
                for left,right,n in zip(edges[:-1],edges[1:],count):
                    histograms.append(dict(subject=subject,run=run,game=game,metric=metric,left=left,
                        right=right,n=int(n),probability=n/len(r)))
        tables['update_distributions']=pd.DataFrame(histograms)
    if number=='07':
        if not points:raise ValueError('Analysis07 group requires saved heatmap_points from participant analyses')
        tables.update(_common_heatmaps(pd.concat(points,ignore_index=True),requested))
        if 'run_condition_means' in tables:
            d=tables['run_condition_means']
            numeric=[c for c in d.select_dtypes(include=[np.number]).columns if c not in {'subject','session','switch','game','run','n','n_decisions'}]
            tables['subject_condition_means']=_participant_means(d,['switch'],numeric)
    if number=='09' and len(tables.get('matched_curve_summary',[])):
        d=tables['matched_curve_summary'];keys=[c for c in ['metric','condition','lag'] if c in d]
        if len(keys)==3 and 'gain' in d:tables['group_matched_curve_summary']=_summary(d,keys,'gain')
    if number=='09':
        from . import action_support
        tables.update(action_support.aggregate(tables,requested))
    if number=='10' and len(tables.get('switch_curves',[])):
        d=tables['switch_curves']
        required={'subject','held','metric','lag','mean','shifted'}
        if not required.issubset(d):raise ValueError('Control switch curves lack '+str(sorted(required-set(d))))
        scalar=d.loc[d.metric.isin(['K','U','D','policy_change'])]
        if len(scalar):
            participant=scalar.groupby(['subject','metric','lag'],dropna=False)[['mean','shifted']].mean().reset_index()
            group=participant.groupby(['metric','lag'])[['mean','shifted']].mean().reset_index()
            group['n_subjects']=participant.groupby(['metric','lag']).subject.nunique().to_numpy()
            tables['control_switch_scalar_participant']=participant
            tables['control_switch_scalar_group']=group
    if number in ('04','06','08'):_group_040608_summaries(number,tables)
    qa=dict(status='COMPLETED',number=number,game=game,subjects=subjects,requested_subjects=requested,
        missing_subjects=missing,n_subjects=len(subjects),
        aggregation='Equal run within participant, then equal participant. Individual statistics retained without averaging IDs, p-values or prediction rows.',
        heatmaps='Common pooled1–99% axes; original COSYNE Gaussian kernel per participant; mean of supported participant probabilities; minimum min(4,requested subjects).',
        population_inference='Descriptive participant comparison; few participants do not establish a population mechanism',
        input_scope='Persisted canonical tables only; no raw replay/checkpoint access')
    if number=='02' and 'baseline_comparison_accuracy_tests' in tables:
        from analyses.validation_allocation import COMPARISON_INFERENCE, EPISODE_COMPARISON_INFERENCE
        qa['accuracy_inference']=COMPARISON_INFERENCE
        qa['accuracy_figure_inference']=EPISODE_COMPARISON_INFERENCE
        qa['accuracy_figure_layout']='One panel per participant; equal episode/run points, shared 0–1 accuracy scale'
    if number=='09':
        qa['action_support'] = dict(primary='Conditional switch destination, post minus pre',
            aggregation='Equal runs within session, equal sessions within participant, then equal participants',
            uncertainty='Individual 30-second block bootstrap; group descriptive mean +/- 1.96 SEM',
            model_training='Thinker trained on all four sessions; exploratory, not model-held-out validation')
    return tables,qa,{}


def _render_heatmaps(root,tables,title,save):
    import matplotlib.pyplot as plt
    d=tables.get('heatmap_surfaces',pd.DataFrame())
    if not len(d):return
    axes=tables['heatmap_axes'].set_index('metric')
    for metric,md in d.groupby('metric',sort=False):
        axis=axes.loc[metric];extent=[axis.x_min,axis.x_max,axis.y_min,axis.y_max]
        limits={}
        for pop,p in md.loc[md.owner.eq('group')].groupby('population'):
            finite=p.probability.dropna();limits[pop]=max(float(finite.quantile(.97)) if len(finite) else 1.,.01)
        for owner,od in md.groupby('owner',sort=False):
            fig,axs=plt.subplots(1,4,figsize=(15,3.8),sharex=True,sharey=True)
            for ax,pop in zip(axs,POPS):
                p=od.loc[od.population.eq(pop)]
                image=p.pivot(index='y',columns='x',values='probability').sort_index().sort_index(axis=1).to_numpy()
                im=ax.imshow(image,origin='lower',extent=extent,aspect='auto',interpolation='bilinear',
                    vmin=0,vmax=limits.get(pop,1.),cmap='viridis')
                ax.set(title=LABELS[pop],xlabel='Initial M (within-run z)')
                if not np.isfinite(image).any():ax.text(.5,.5,'Insufficient shared support',ha='center',transform=ax.transAxes)
                fig.colorbar(im,ax=ax,shrink=.75,label='P(transition)')
            axs[0].set_ylabel(metric+' / run mean')
            fig.suptitle(f'{title} · {owner} · {metric}\nCommon axes; equal participant mean; group requires {int(axis.required_subjects)} supported participants',fontsize=12)
            save(fig,f'MU_heatmap_{metric}_{owner}')


def render(number,root,tables,title,arrays=None):
    """All figures can be redrawn solely from the persisted group tables."""
    number=f'{int(number):02d}'
    if number=='01':
        from analyses.human_behavior import render as draw
        return draw(root,tables,title)
    if number=='05':
        from analyses.revision_behavior import render as draw
        draw(number,root,tables,title)
        return [str(p) for p in (Path(root)/'figures').glob('*') if p.suffix in ('.png','.pdf')]
    if number in ('02','03'):
        from analyses.validation_allocation import render as draw
        return draw(number,root,tables,title,arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(root)/'figures';out.mkdir(parents=True,exist_ok=True);written=[]
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:
            p=out/f'{name}.{ext}';fig.savefig(p,dpi=180,bbox_inches='tight');written.append(str(p))
        plt.close(fig)
    if number=='09':
        from . import action_support
        written.extend(action_support.render(root,tables))
    if number in ('04','06','08'):
        _render_group_040608(number,tables,title,save)
        return written
    if number=='07':
        _render_heatmaps(root,tables,title,save)
        d=tables.get('subject_condition_means',pd.DataFrame())
        if len(d):
            metrics=[c for c in ['M_z','A_V','U_V','A_Q','U_Q','V_total','Q_total','V_relative','Q_relative','budget'] if c in d]
            for metric in metrics:
                z=d.pivot(index='subject',columns='switch',values=metric).reindex(columns=[0,1]).dropna()
                if not len(z):continue
                fig,ax=plt.subplots(figsize=(5,4));bp=ax.boxplot([z[0],z[1]],patch_artist=True,showfliers=False)
                for box,color in zip(bp['boxes'],['#2780a3','#d68b40']):box.set_facecolor(color);box.set_alpha(.25)
                for sub,row in z.iterrows():ax.plot([1,2],row,'o-',lw=.9,label=f'sub{int(sub):03d}')
                ax.set(xticks=[1,2],xticklabels=['Persist','Switch'],ylabel=metric,title=title+' · equal-participant values');ax.legend(fontsize=8)
                save(fig,'persist_switch_'+metric)
    if number=='09' and len(tables.get('group_matched_curve_summary',[])):
        d=tables['group_matched_curve_summary'];metrics=list(d.metric.unique())
        fig,axes=plt.subplots(2,(len(metrics)+1)//2,figsize=(12,7),squeeze=False)
        for ax,metric in zip(axes.flat,metrics):
            for condition,r in d.loc[d.metric.eq(metric)].groupby('condition'):
                r=r.sort_values('lag');ax.plot(r.lag,r['mean'],label=condition)
                ax.fill_between(r.lag,r.ci_low,r.ci_high,alpha=.15)
            ax.axvline(0,color='gray',ls='--');ax.set(title=metric,xlabel='Seconds from actual / matched switch',ylabel='Participant mean change');ax.legend()
        for ax in axes.flat[len(metrics):]:ax.axis('off')
        fig.suptitle(title+' · matched switch/persist; equal participants');save(fig,'matched_switch_persist_group')
    # Retained methods expose meaningful curves under their own coordinate names.
    # Only explicit curve/trajectory tables are eligible; no arbitrary numeric-column plots.
    for name,d in tables.items():
        if not len(d) or name.startswith(('heatmap','group_')) or name in {'matched_curve_summary'}:continue
        if not any(k in name for k in ['curve','trajectory','switch_locked']):continue
        x=next((c for c in ['lag','relative_time','seconds','search_steps','choice_index','computation','fraction','budget','bin'] if c in d),None)
        if x is None or 'subject' not in d:continue
        value=next((c for c in ['gain','value','mean','probability','accuracy','log_score','V_relative','budget'] if c in d and c!=x),None)
        if value is None:continue
        keys=[c for c in ['metric','condition','target','population','source','representation'] if c in d]
        if len(keys)>3:keys=keys[:3]
        grouped=d.groupby(keys,dropna=False) if keys else [((),d)]
        series=list(grouped)
        if len(series)>20:continue
        fig,ax=plt.subplots(figsize=(8,4))
        for labels,r in series:
            z=_summary(r,[x],value);ax.plot(z[x],z['mean'],label=' / '.join(map(str,labels if isinstance(labels,tuple) else (labels,))))
            ax.fill_between(z[x],z.ci_low,z.ci_high,alpha=.1)
        ax.set(xlabel=x,ylabel=value,title=title+' · '+name.replace('_',' '))
        if keys:ax.legend(fontsize=7,ncol=2)
        save(fig,name+'_group')
    return written
