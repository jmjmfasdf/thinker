"""Within-imagination revision (05) and computation-to-human behavior (07).

V is the common SEARCH task critic. Q revisions require visits at both ends.
Session-held-out readouts do not make the all-session Thinker itself held out.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.research_inputs import enrich_decisions
from common.inference import fit_pooled
from common.prediction import baseline, probabilities, scores
from common.outcomes import risk_set
from panels.mu_heatmaps import kernel_surface, GRID
from .replay_themes import paired_box, savefig

METRICS = ['M_z','A_V','U_V','D_V','A_Q','U_Q','V_relative','Q_relative','budget']


def computation_values(d, prepared):
    from .replay_themes import PackedTrace, verify_packs
    episodes={q['metadata']['episode_id']:q for q in prepared['metadata']['episodes']}
    offsets=np.asarray(prepared['qa']['phase_offsets'])
    records=[]
    for episode,part in d.groupby('episode',sort=False):
        rec=episodes[episode];t=PackedTrace(Path(prepared['metadata']['capture'])/rec['path'])
        verify_packs(t,rec)
        value=t.read('actor_baseline')[:,0].astype(float)-offsets[t.read('phase_pre').astype(int)]
        search=t.read('search_computation').astype(bool)
        depth=t.read('current_node_depth_post');reset=t.read('actor_reset')
        pre=t.read('tree_reps_pre');post=t.read('tree_reps_post')
        field=lambda x,k:x[:,slice(*t.tree_schema[k])]
        q0,q1=field(pre,'root_qs_mean'),field(post,'root_qs_mean')
        n0,n1=field(pre,'root_ns'),field(post,'root_ns')
        for row in part.itertuples():
            computations=np.flatnonzero(search[row.root_event+1:row.real_event])+row.root_event+1
            if len(computations)!=row.budget:raise ValueError('Budget/event mismatch')
            endpoints=np.r_[computations[0],computations+1] if len(computations) else np.array([row.root_event+1])
            v=value[endpoints];dv=np.r_[0,np.diff(v)]
            np.testing.assert_allclose(np.abs(dv).sum(),row.A_V,atol=1e-7,rtol=1e-5)
            valid=(n0[computations]>0)&(n1[computations]>0)
            dq=np.r_[0,np.where(valid,abs(q1[computations]-q0[computations]),0).sum(1)]
            np.testing.assert_allclose(dq.sum(),row.A_Q,atol=1e-6,rtol=1e-5)
            for k,ev in enumerate(endpoints):
                records.append(dict(run=row.run,episode=episode,session=row.session,decision=row.decision,
                    time=row.time,budget=row.budget,k=k,value=v[k],delta_V=dv[k],absolute_delta_V=abs(dv[k]),
                    absolute_delta_Q=dq[k],depth=int(depth[computations[k-1]]) if k else 0,
                    reset=bool(reset[computations[k-1]]) if k else False))
    return pd.DataFrame(records)


def revision(d,prepared):
    e=computation_values(d,prepared)
    tables={'decision_updates':d[['run','session','episode','decision','time','budget','A_V','U_V','D_V','A_Q','U_Q','Q_supported','valid_update_count','first_abs_dV']],
            'computation_endpoints':e}
    tables['update_by_budget']=d.groupby(['run','budget'])[METRICS[1:-1]].mean().reset_index()
    tables['update_by_computation']=e.groupby(['run','k'])[['absolute_delta_V','delta_V','absolute_delta_Q','value']].mean().reset_index()
    tables['coverage']=d.groupby('run').agg(decisions=('budget','size'),Q_supported=('Q_supported','mean'),
        budget_one=('budget',lambda x:x.eq(1).mean()),mean_budget=('budget','mean'),
        zero_budget=('budget',lambda x:x.eq(0).sum())).reset_index()
    distributions=[]
    for metric in ('A_V','U_V','D_V','A_Q','U_Q'):
        v=d[metric];lo,hi=np.quantile(v,[0,.995]);hi=max(hi,lo+1e-9);edges=np.linspace(lo,hi,41)
        for run,r in d.groupby('run'):
            count,_=np.histogram(r[metric],edges)
            for a,b,n in zip(edges[:-1],edges[1:],count):distributions.append(dict(run=run,metric=metric,left=a,right=b,n=n,probability=n/len(r)))
    tables['update_distributions']=pd.DataFrame(distributions)
    return tables,dict(definition='A_V=sum|ΔV|; U_V=A_V/B; D_V=Vfinal−Vinitial; B=0 gives zero',
        Q='Only visited-to-visited revisions; no-support zeros are identified by Q_supported',
        phase='Common SEARCH task critic; first imagination included; no cross-observation differences',
        distribution_tail='Histogram excludes top0.5% for display only; decision table retains all rows'),{}


def locked(d):
    records=[]
    for ep,r in d.groupby('episode'):
        r=r.sort_values('decision').set_index('decision');centers=r.index[r['switch'].eq(1)]
        for lag in range(-10,11):
            pos=r.index.get_indexer(centers+lag);origin=r.index.get_indexer(centers)
            good=(pos>=0)&(pos-origin==lag)
            if good.any():
                s=r.iloc[pos[good]]
                records.append(dict(run=r.run.iloc[0],episode=ep,lag=lag,n=len(s),**s[METRICS].mean().to_dict()))
    return pd.DataFrame(records)


def surfaces(d):
    parts=[]
    for estimator in ('A_V','U_V','A_Q','U_Q'):
        mean=d.groupby('run')[estimator].transform('mean')
        z=d.assign(U_relative=np.divide(d[estimator],mean,out=np.zeros(len(d)),where=mean>0),U_all=d[estimator])
        xlim=z.M_z.quantile([.01,.99]).to_numpy();ylim=np.array([0,max(z.U_relative.quantile(.99),1e-6)])
        if not np.isfinite(xlim).all() or np.ptp(xlim)<=0:xlim=np.array([-1.,1.])
        x,y=np.meshgrid(np.linspace(*xlim,GRID),np.linspace(*ylim,GRID))
        for population in ('overall','noop_to_active','active_to_other','active_to_noop'):
            r=z if population=='overall' else z.loc[z.incumbent_action.eq(0) if population=='noop_to_active' else z.incumbent_action.ne(0)]
            r=r.assign(outcome=r.human_action.eq(0).astype(int) if population=='active_to_noop' else r['switch'])
            p,n,positive=kernel_surface(r,xlim,ylim)
            parts.append(pd.DataFrame(dict(estimator=estimator,population=population,x=x.ravel(),y=y.ravel(),
                probability=p.ravel(),local_n=n.ravel(),positive_update_n=positive.ravel())))
    return pd.concat(parts,ignore_index=True)


def heldout_prediction(d,temporal=False):
    """Same regularized readouts, now outer folds over all recorded sessions."""
    base=baseline(d)
    b1=base+['M','root_value','root_entropy','initial_q_margin']+[c for c in d if c.startswith('log_policy_')]
    b2=b1+['log_budget','searched','carried']
    b3=b2+['U_V','D_V','A_Q','Q_supported']+[c for c in d if c.startswith(('root_qs_mean_terminal_','root_ns_terminal_'))]
    definitions={'B0_behavior_visual_history':base,'B1_initial_support':b1,'B2_quantity':b2,
                 'B3_evaluation_revision':b3,'B4_VOC':b3+['ema_initial']}
    if temporal:
        current=definitions['B4_VOC']
        definitions={'T0_current':current,'T1_previous_evaluation':current+['lag1_M','lag1_U_V','lag1_D_V','lag1_ema_initial'],
            'T2_bout_history':current+['lag1_M','lag1_U_V','lag1_D_V','lag1_ema_initial','past_bout_A_V','lag1_log_budget']}
        if d.arrived_error_available.any():
            definitions['T3_arrived_mismatch']=definitions['T2_bout_history']+['arrived_reward_error','arrived_error_available']
    metrics=[];qa=[];predictions=[]
    for population in ('exact_action','overall','noop_to_active','active_to_noop'):
        if population=='exact_action':
            part=d.loc[d.scored.eq(1)&d.incumbent_action.ge(0)].copy();part['outcome']=part.human_action
            nact=int(d.n_actions.iloc[0])
        else:part=risk_set(d,population);nact=2
        part=part.reset_index(drop=True);truth=part.outcome.to_numpy(int)
        for session in sorted(part.session.unique()):
            train=part.session.ne(session).to_numpy();test=~train;result={}
            for name,columns in definitions.items():
                p,audit=probabilities(part,columns,truth,train,test,actions=nact)
                qa.append(dict(held_session=session,population=population,model=name,**audit))
                if p is not None:result[name]=p
            # Compare nested models only where every model fitted successfully.
            if len(result)!=len(definitions):continue
            out=part.loc[test,['run','episode','decision','session','outcome']].reset_index(drop=True)
            out['population']=population;out['held_session']=session
            for name,p in result.items():
                out['log_score_'+name]=np.log(p[np.arange(len(p)),out.outcome.to_numpy(int)].clip(1e-12))
                for run,ix in out.groupby('run').groups.items():
                    metrics.append(dict(run=run,held_session=session,population=population,model=name,**scores(out.loc[ix,'outcome'],p[ix])))
            predictions.append(out)
    return dict(prediction_run_scores=pd.DataFrame(metrics),prediction_fit_qa=pd.DataFrame(qa),
        heldout_predictions=pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame())


def behavior(d,prepared):
    tables={'run_condition_means':d.groupby(['run','session','switch'])[METRICS].mean().reset_index(),
        'switch_locked_episode_means':locked(d),'heatmap_surfaces':surfaces(d),
        'heatmap_points':d[['run','M_z','A_V','U_V','A_Q','U_Q','incumbent_action','human_action','switch']],
        'equal_Q_support_means':d.loc[d.Q_supported].groupby(['run','switch'])[METRICS].mean().reset_index()}
    z=enrich_decisions(d,prepared['metadata'])
    z['K']=4*z.U_V*expit(z.M)*expit(-z.M)
    from .behavior import boundary
    tables.update({f'boundary_{k}':v for k,v in boundary(z,prepared['choices']).items()})
    tables.update(heldout_prediction(z))
    effects=[]
    controls=['log_budget','log_age','root_value','prior_reward','carry']+[c for c in d if c.startswith('previous_action_')]
    for metric in ('A_V','U_V','D_V','A_Q','U_Q'):
        for r in fit_pooled(d,'switch',['M',metric],controls,True):effects.append(dict(estimator=metric,**r))
    tables['budget_adjusted_associations']=pd.DataFrame(effects)
    return tables,dict(phase='Same-phase within-imagination updates; current human switch defined against preceding replay action',
        prediction='Session held out for readout/scaling only; Thinker trained on all human sessions',
        populations='Risk sets use incumbent action only; active→NOOP includes other active transitions as non-terminations',
        heatmaps='COSYNE padded Gaussian kernel, 4% bandwidth, at least12 local decisions; A and U shown separately',
        inference='Descriptive within participant; run-level/HAC uncertainty is not population inference'),{}


def analyze(number,args,subject,game,prepared,workdir):
    d=prepared['decisions'].copy()
    result=revision(d,prepared) if str(number)=='05' else behavior(d,prepared)
    # The canonical dispatcher also appends VoC fits. Give it an explicit
    # base status so unsupported appended fits can downgrade it truthfully.
    failed=sum(int(table.status.ne('ok').sum()) for name,table in result[0].items()
               if name.endswith('fit_qa') and 'status' in table)
    result[1].update(status='PARTIAL_FITS' if failed else 'COMPLETED',failed_readout_fits=failed)
    for name,table in list(result[0].items()):
        table=table.copy();table['subject']=subject;table['game']=game;result[0][name]=table
    return result


def _equal_subject_curve(frame, keys, metrics):
    frame=frame.copy()
    if 'subject' not in frame:frame['subject']=1
    first=frame.groupby(['subject',*keys],dropna=False)[metrics].mean().reset_index()
    return first.groupby(keys,dropna=False)[metrics].mean().reset_index()


def _paired_units(frame):
    frame=frame.copy()
    if 'subject' not in frame or frame.subject.nunique()==1:return frame
    metrics=[c for c in METRICS if c in frame]
    # A group box contains participants, each represented by their mean of runs.
    run=frame.groupby(['subject','run','switch'])[metrics].mean().reset_index()
    subject=run.groupby(['subject','switch'])[metrics].mean().reset_index()
    subject['run']=subject.subject.map(lambda s:f'sub{int(s):03d}')
    return subject


def render(number,root,tables,title):
    root=Path(root);figs=root/'figures';figs.mkdir(parents=True,exist_ok=True)
    if str(number)=='05':
        z=tables['update_distributions'];fig,axes=plt.subplots(1,5,figsize=(16,3.4))
        for ax,metric in zip(axes,('A_V','U_V','D_V','A_Q','U_Q')):
            s=_equal_subject_curve(z.loc[z.metric.eq(metric)],['left','right'],['probability'])
            ax.bar((s.left+s.right)/2,s.probability,width=s.right-s.left,color='#287fa8');ax.set(title=metric,ylabel='Run-mean fraction')
        fig.suptitle(title);savefig(fig,figs,'value_revision_distributions')
        fig,axes=plt.subplots(1,3,figsize=(11,3.5))
        for ax,metric in zip(axes,('absolute_delta_V','delta_V','absolute_delta_Q')):
            s=_equal_subject_curve(tables['update_by_computation'],['k'],[metric]).set_index('k')[metric];s=s.loc[s.index>0]
            ax.plot(s.index,s,'o-');ax.set(title=metric,xlabel='Computation number (surviving decisions)')
        fig.suptitle(title);savefig(fig,figs,'within_imagination_updates')
        fig,axes=plt.subplots(1,2,figsize=(9,3.6))
        for ax,metrics in zip(axes,(('A_V','U_V'),('A_Q','U_Q'))):
            for metric in metrics:
                s=_equal_subject_curve(tables['update_by_budget'],['budget'],[metric]).set_index('budget')[metric];ax.plot(s.index,s,label=metric)
            ax.legend();ax.set(xlabel='Actual computation count',ylabel='Run-mean revision')
        fig.suptitle(title);savefig(fig,figs,'revision_by_budget')
    else:
        means=_paired_units(tables['run_condition_means'])
        for kind in ('V','Q'):
            fig,axes=plt.subplots(1,3,figsize=(11,3.6))
            for ax,metric in zip(axes,('M_z','A_'+kind,'U_'+kind)):paired_box(ax,means,metric,metric)
            fig.suptitle(title);savefig(fig,figs,'persist_switch_'+kind)
        for estimator,z in tables['heatmap_surfaces'].groupby('estimator'):
            fig,axes=plt.subplots(1,4,figsize=(14,3.5),sharex=True,sharey=True)
            for ax,pop in zip(axes,('overall','noop_to_active','active_to_other','active_to_noop')):
                s=z.loc[z.population.eq(pop)];p=s.pivot(index='y',columns='x',values='probability')
                im=ax.imshow(p,origin='lower',extent=[p.columns.min(),p.columns.max(),p.index.min(),p.index.max()],aspect='auto',vmin=0,vmax=1,cmap='viridis')
                ax.set(title=pop,xlabel='M (within-run z)');fig.colorbar(im,ax=ax,shrink=.8)
            axes[0].set_ylabel(estimator+' / run mean');fig.suptitle(title);savefig(fig,figs,'MU_heatmap_'+estimator)
        fig,axes=plt.subplots(1,4,figsize=(12,3.4));z=tables['switch_locked_episode_means']
        for ax,metric in zip(axes,('M_z','U_V','D_V','budget')):
            if 'subject' not in z:z=z.assign(subject=1)
            r=z.assign(total=z[metric]*z.n).groupby(['subject','run','lag'])[['total','n']].sum()
            r=(r.total/r.n).rename(metric).reset_index()
            v=_equal_subject_curve(r,['lag'],[metric]).set_index('lag')[metric]
            ax.plot(v.index,v);ax.axvline(0,color='gray',ls='--');ax.set(title=metric,xlabel='Decisions from switch')
        fig.suptitle(title);savefig(fig,figs,'human_switch_aligned')
        z=tables['prediction_run_scores']
        if not z.empty:
            fig,ax=plt.subplots(figsize=(9,4))
            for pop,r in z.groupby('population'):
                s=_equal_subject_curve(r,['model'],['log_score']).set_index('model').log_score;ax.plot(s.index,s,'o-',label=pop)
            ax.set(ylabel='Held-session human log score',title=title);ax.tick_params(axis='x',rotation=20);ax.legend();savefig(fig,figs,'held_session_human_prediction')
