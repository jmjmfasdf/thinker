"""Additive 03/05–10 VoC relationships; never estimate causal reward utility."""
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd

from common.voc_dynamics import KEYS, unique
from common.inference import fit_pooled, PRIMARY_LAG
from common.prediction import baseline, probabilities, scores

THEMES = ('03','05','06','07','08','09','10')
LOCAL_CONTROLS = ['log_steps','depth','M','root_value','log_age','prior_reward','carry']
VOC_FEATURES = ['initial_voc','endpoint_voc_change']
REVISION_FEATURES = ['U_V','D_V','A_Q','Q_supported']
POLICY_FEATURES = ['policy_js']
LABELS = {'delta_v':'Signed task-value revision', 'abs_delta_v':'Absolute task-value revision',
    'abs_delta_q':'Supported tree-Q revision', 'policy_js_step':'Policy JS distance per computation',
    'endpoint_voc_delta':'Common-readout VoC change', 'next_continue':'Next free CONTINUE probability',
    'budget':'Actual computation count', 'computation_cost':'Configured computation cost',
    'probability':'Recorded gate probability', 'continued':'Observed CONTINUE frequency',
    'initial_voc':'Initial EMA VoC', 'voc_ema':'Pre-choice EMA VoC', 'voc_online':'Pre-choice online VoC',
    'human_logp_change':'Change in human-action log probability'}


def empty(): return pd.DataFrame(columns=['status'])


def combine(frames):
    frames = [f for f in frames if len(f)]
    return pd.concat(frames,ignore_index=True) if frames else empty()


def join_decisions(left, right, columns):
    keys = ['episode','decision']
    unique(left,keys); unique(right,keys)
    result = left.merge(right[keys+columns],on=keys,how='left',validate='one_to_one',sort=False)
    return result


def curves(frame, x, targets, fixed_probability=False):
    """Within-run ranks, equal-run summaries later; no outcome-based binning."""
    rows=[]
    for (subject,game,run), part in frame.groupby(['subject','game','run'],sort=False):
        for target in targets:
            r=part.loc[np.isfinite(part[x]) & np.isfinite(part[target])].copy()
            if r.empty: continue
            r['bin'] = np.minimum((r[x]*10).astype(int),9) if fixed_probability else np.minimum((r[x].rank(pct=True)*10).astype(int),9)
            for b,a in r.groupby('bin'):
                rows.append(dict(subject=subject,game=game,run=run,x_metric=x,metric=target,bin=int(b),
                    x=float(a[x].mean()),value=float(a[target].mean()),n=len(a),n_decisions=a.decision.nunique(),
                    binning='probability_decile' if fixed_probability else 'within_run_predictor_rank'))
    return pd.DataFrame(rows) if rows else empty()


def effects(frame, target, predictor, controls, family, binary=False):
    z=frame.assign(run_label=frame.episode,real_step=frame.decision)
    rows=fit_pooled(z,target,[predictor],controls,binary)
    return pd.DataFrame([dict(family=family,target=target,predictor=predictor,**r) for r in rows])


def prediction_blocks(d):
    base=baseline(d)+['M','root_value','V_initial','root_entropy','initial_q_margin']
    base += [c for c in d if c.startswith('log_policy_')]
    quantity=base+['log_budget','searched','carried']
    return dict(reference=quantity,full=quantity+VOC_FEATURES+REVISION_FEATURES+POLICY_FEATURES,
        no_voc=quantity+REVISION_FEATURES+POLICY_FEATURES,
        no_revision=quantity+VOC_FEATURES+POLICY_FEATURES,
        no_policy=quantity+VOC_FEATURES+REVISION_FEATURES)


def readouts(d, blocks, targets=('exact_action','switch')):
    """Matched held-session fits; no target-dependent feature is permitted."""
    forbidden={'human_logp_change','human_probability','human_action','switch','outcome'}
    if any(forbidden.intersection(cols) for cols in blocks.values()):
        raise ValueError('Label-dependent prediction input')
    metrics=[]; audits=[]; predictions=[]; gains=[]
    for target in targets:
        z=d.loc[d.incumbent_action.ge(0)].reset_index(drop=True)
        truth=z.human_action.to_numpy(int) if target=='exact_action' else z.switch.to_numpy(int)
        actions=int(z.n_actions.iloc[0]) if target=='exact_action' and len(z) else 2
        for held in sorted(z.session.unique()):
            train=z.session.ne(held).to_numpy(); test=~train; fitted={}
            for name,columns in blocks.items():
                p,audit=probabilities(z,columns,truth,train,test,actions=actions)
                audits.append(dict(target=target,held_session=held,model=name,**audit))
                if p is not None: fitted[name]=p
            if len(fitted)!=len(blocks): continue
            out=z.loc[test,['run','episode','decision','session']].reset_index(drop=True)
            out['target']=target; out['truth']=truth[test]
            for name,p in fitted.items():
                out['log_score_'+name]=np.log(p[np.arange(len(p)),truth[test]].clip(1e-12))
                for run,ids in out.groupby('run').groups.items():
                    metrics.append(dict(run=run,target=target,held_session=held,model=name,**scores(truth[test][ids],p[ids])))
            for name in blocks:
                if name=='full': continue
                delta=out['log_score_full']-out['log_score_'+name]
                for run,ids in out.groupby('run').groups.items():
                    gains.append(dict(run=run,target=target,held_session=held,reference=name,value=float(delta.iloc[ids].mean()),n=len(ids)))
            predictions.append(out)
    return dict(voc_readout_scores=pd.DataFrame(metrics),voc_readout_fit_qa=pd.DataFrame(audits),
        voc_readout_predictions=combine(predictions),voc_readout_gains=pd.DataFrame(gains))


def trailing_features(d, width=.5):
    """Strictly past summaries; raw-bout boundaries supplied by audited _history."""
    z=d.sort_values(['episode','decision']).copy().reset_index(drop=True)
    from common.sampling import sequence_index
    boundary=(z.episode.ne(z.episode.shift()) | sequence_index(z).diff().ne(1) |
              z.time.diff().le(0) | z.time.diff().gt(.25))
    if 'prior_raw_bout_id' in z: boundary |= z.prior_raw_bout_id.ne(z.prior_raw_bout_id.shift())
    z['_voc_segment']=boundary.cumsum()
    for c in ('initial_voc','endpoint_voc_change','U_V'):
        z['lag1_'+c]=z.groupby('_voc_segment')[c].shift(1)
    z['past_voc_mean']=np.nan;z['past_voc_slope']=np.nan;z['past_revision_mean']=np.nan
    for _,part in z.groupby('_voc_segment'):
        times=part.time.to_numpy(); v=part.initial_voc.to_numpy(); rev=part.U_V.to_numpy()
        for j,idx in enumerate(part.index):
            start=np.searchsorted(times,times[j]-width,side='left')
            if j-start<2:continue
            # Piecewise constant until the next observed time; current value excluded.
            weights=np.diff(times[start:j+1]); x=times[start:j]; y=v[start:j]
            if not np.isfinite(y).all(): continue
            z.loc[idx,'past_voc_mean']=np.average(y,weights=weights)
            x=x-x.mean(); den=x@x
            z.loc[idx,'past_voc_slope']=float(x@y/den) if den>0 else np.nan
            z.loc[idx,'past_revision_mean']=np.average(rev[start:j],weights=weights)
    return z


def window_features(d, windows, voc_decisions):
    """Unfitted physical-time window summaries; no future endpoint is read."""
    from common.sampling import sequence_index
    z=join_decisions(d,voc_decisions,VOC_FEATURES)
    out=[]
    for w in windows.itertuples():
        part=z.iloc[int(w.start):int(w.end)+1]
        times=part.time.to_numpy(); v=part.initial_voc.to_numpy(); change=part.endpoint_voc_change.to_numpy()
        valid=(len(part)>=2 and part.episode.nunique()==1 and np.all(np.diff(sequence_index(part))==1)
            and np.all((np.diff(times)>0)&(np.diff(times)<=.25))
            and np.isfinite(v).all() and np.isfinite(change).all())
        vals=[np.nan]*3
        if valid:
            dt=np.diff(times); x=times-times.mean()
            vals=[float(np.average(v[:-1],weights=dt)),float(x@v/(x@x)),float(np.average(change[:-1],weights=dt))]
        out.append(dict(window_id=w.window_id,voc_window_mean=vals[0],voc_window_slope=vals[1],
            voc_window_revision=vals[2],voc_window_supported=valid))
    return pd.DataFrame(out)


def pattern_relationships(existing, transitions):
    """Candidate state is a descriptive stratifier, never a causal moderator."""
    assignment=existing.get('candidate_assignments',empty()); native=existing.get('decisions',empty())
    if assignment.empty or native.empty:return {'voc_pattern_effects':empty()}
    # Use one prespecified content view and width; IDs stay within held folds.
    a=assignment.loc[assignment.view.eq('evaluation') & np.isclose(assignment.width,.5)].copy()
    if a.empty:return {'voc_pattern_effects':empty()}
    unique(a,['held','row']); mapping=native[['row','episode','decision']]
    unique(mapping,['row'])
    a=a[['held','row','state']].merge(mapping,on='row',validate='many_to_one')
    merged=a.merge(transitions,on=['episode','decision'],how='inner',validate='one_to_many')
    if not merged.session.eq(merged.held).all(): raise ValueError('Pattern assignment is not held-session')
    result=[]; support=[]
    for (held,state),g in merged.groupby(['held','state']):
        support.append(dict(held=held,state=state,n=len(g),n_decisions=g.decision.nunique(),
            n_next_free=int(g.next_voluntary.sum())))
        for target,predictor,part,binary in [('abs_delta_v','voc_ema',g,False),
                ('next_continue','abs_delta_v',g.loc[g.next_voluntary],True)]:
            controls=LOCAL_CONTROLS+(['voc_ema'] if predictor!='voc_ema' else [])
            e=effects(part,target,predictor,controls,'pattern_conditional',binary)
            e['held']=held;e['state']=state;result.append(e)
    return dict(voc_pattern_effects=combine(result),voc_pattern_support=pd.DataFrame(support))


def extend(number,args,prepared,existing,tables):
    """New tables are prefixed voc_; legacy numeric tables are never modified."""
    c=tables['voc_choices']; tr=tables['voc_transitions']; d=tables['voc_decisions']
    subject=int(d.subject.iloc[0]);game=str(d.game.iloc[0])
    free=c.loc[c.voluntary]; transitions=tr.loc[tr.voluntary] if len(tr) else tr
    result={}; assoc=[]; binned=[]
    if number=='03':
        result.update(voc_choices=c,voc_decisions=d)
        for source in ('voc_ema','voc_online'):
            binned.append(curves(free,source,['probability','continued','q_continue_ema','q_stop_ema']))
        binned += [curves(free,'probability',['continued'],True),curves(d,'initial_voc',['budget','computation_cost'])]
        diag=[]
        for run,r in free.groupby('run'):
            use=r.loc[r.gate_identity_applicable]
            diag.append(dict(run=run,n=len(r),gate_identity_n=len(use),
                gate_identity_max_error=float((use.probability-use.gate_expected).abs().max()) if len(use) else np.nan,
                online_ema_mean_absolute_gap=float(r.online_ema_gap.abs().mean()),
                online_ema_sign_disagreement=float((np.sign(r.voc_online)!=np.sign(r.voc_ema)).mean()),
                brier=float(((r.probability-r.continued)**2).mean())))
        result['voc_gate_diagnostics']=pd.DataFrame(diag)
        result['voc_budget_cost']=d.groupby(['subject','game','run','budget','termination']).agg(
            n=('decision','size'),cost=('computation_cost','mean')).reset_index()
        distributions=[]
        for source,metrics in [(free,['q_continue_ema','q_stop_ema','voc_ema','online_ema_gap']),
                               (d,['budget','computation_cost'])]:
            for run,r in source.groupby('run'):
                for metric in metrics:
                    for q,value in zip(np.linspace(0,1,101),np.quantile(r[metric],np.linspace(0,1,101))):
                        distributions.append(dict(run=run,metric=metric,quantile=q,value=value,n=len(r)))
        result['voc_distributions']=pd.DataFrame(distributions)
    if number in ('05','06'):
        result.update(voc_transitions=tr,voc_decisions=d)
        if number=='05':
            targets=['delta_v','abs_delta_v','abs_delta_q','policy_js_step']
            for condition,part in [('free',transitions),('forced_minimum',tr.loc[~tr.voluntary.astype(bool)])]:
                for source in ('voc_ema','voc_online'):
                    curve=curves(part,source,targets);curve['condition']=condition;binned.append(curve)
                    for target in targets:
                        assoc.append(effects(part,target,source,LOCAL_CONTROLS+['value_pre'],'pre_voc_next_revision_'+condition))
        else:
            next_free=tr.loc[tr.next_voluntary].copy()
            for predictor in ('delta_v','abs_delta_v','policy_js_step'):
                binned.append(curves(next_free,predictor,['next_voc','next_continue']))
                assoc.append(effects(next_free,'next_continue',predictor,LOCAL_CONTROLS+['voc_ema'],'revision_next_choice',True))
                assoc.append(effects(next_free,'next_voc',predictor,LOCAL_CONTROLS+['voc_ema'],'revision_next_voc'))
            # Endpoint change remains observable at the cap, but is not a next choice.
            binned.append(curves(tr,'abs_delta_v',['endpoint_voc_delta']))
            result['voc_stopping_by_k']=free.groupby(['subject','game','run','k']).agg(
                continuation=('continued','mean'),gate=('probability','mean'),n=('event','size')).reset_index()
            result['voc_human_fit']=d.groupby(['subject','game','run','termination']).agg(
                value=('human_logp_change','mean'),n=('decision','size')).reset_index()
            chosen=[]
            for _,part in d.groupby(['termination','budget'],sort=True):
                part=part.sort_values(['run','episode','decision']);chosen.append(part.iloc[len(part)//2])
            # Up to six examples spanning budgets; selection never examines revisions/fit.
            chosen=pd.DataFrame(chosen)
            if len(chosen)>6:chosen=chosen.iloc[np.linspace(0,len(chosen)-1,6).astype(int)]
            result['voc_examples']=chosen
            selected=chosen[KEYS]
            result['voc_example_endpoints']=tables['voc_endpoints'].merge(selected,on=KEYS,validate='many_to_one')
            result['voc_example_choices']=c.merge(selected,on=KEYS,validate='many_to_one')
        ep=tables['voc_endpoints']
        result['voc_computation_curves']=ep.groupby(['subject','game','run','k']).agg(
            value=('value','mean'),endpoint_voc=('endpoint_voc','mean'),policy_js_step=('policy_js_step','mean'),
            n=('event','size')).reset_index()
    if number in ('07','08'):
        from common.research_inputs import enrich_decisions
        enriched=enrich_decisions(prepared['decisions'],prepared['metadata'])
        columns=VOC_FEATURES+POLICY_FEATURES+['human_logp_change']
        enriched=join_decisions(enriched,d,columns)
        blocks=prediction_blocks(enriched)
        if number=='08':
            from .policy_continuity import _history
            enriched=trailing_features(_history(enriched,prepared['metadata']),getattr(args,'window',.5))
            past=['lag1_initial_voc','lag1_endpoint_voc_change','past_voc_mean','past_voc_slope']
            revisions=['lag1_U_V','past_revision_mean']
            current=blocks['full']
            blocks=dict(reference=current,full=current+past+revisions,
                        no_voc=current+revisions,no_revision=current+past)
            # Identical complete-history support in every model, including reference.
            enriched=enriched.dropna(subset=past+revisions).reset_index(drop=True)
            result['voc_temporal_records']=enriched[KEYS+['session','time']+past+revisions+VOC_FEATURES]
            binned.append(curves(enriched,'lag1_initial_voc',['initial_voc','U_V']))
        result.update(readouts(enriched,blocks))
        result['voc_human_fit']=d.groupby(['subject','game','run','termination']).agg(
            value=('human_logp_change','mean'),n=('decision','size')).reset_index()
    if number=='09':
        # Fold-specific comparisons are fitted in the existing temporal backend.
        result['voc_window_features']=existing.get('voc_window_features',empty())
        comparisons=existing.get('primary_comparisons',empty())
        if len(comparisons):result['voc_window_comparisons']=comparisons.loc[comparisons.model.str.startswith('voc_')].copy()
        fits=existing.get('primary_fit_status',empty())
        if len(fits):result['voc_window_fit_qa']=fits.loc[fits.model.str.startswith('voc_')].copy()
    if number=='10':
        result.update(pattern_relationships(existing,tr))
        # Preserve the existing early-only allocation readout and report its output.
        predictions=existing.get('budget_predictions',empty())
        if len(predictions):
            native=existing['decisions'][['row','run','session']]
            p=predictions.merge(native,on='row',validate='many_to_one')
            p['value']=(p.target-p.baseline)**2-(p.target-p.prediction)**2
            result['voc_early_budget_gains']=p.groupby(['run','session','held','model']).agg(value=('value','mean'),n=('row','size')).reset_index()
    if assoc:result['voc_relationship_effects']=combine(assoc)
    if binned:result['voc_relationship_curves']=combine(binned)
    for frame in result.values():frame['subject']=subject;frame['game']=game
    failures={name:frame.status.value_counts().to_dict() for name,frame in result.items()
              if ('fit_qa' in name or 'effects' in name) and 'status' in frame}
    qa=dict(status='COMPLETED',fit_status_counts=failures,primary='EMA',sensitivity='online',
        n_decisions=len(d),n_free_choices=len(free),n_computations=len(tr),
        n_supported_q=int(tr.q_support_actions.gt(0).sum()),n_next_free=int(tr.next_voluntary.sum()),
        scope='Execution integrity / internal revision / behavioral fit; no realized reward utility',
        conditioning='Observed CONTINUE only for next revision; later k conditions on survival',
        cache_source='Reproducible from native capture and hash-bound original checkpoint')
    if any(any(k!='ok' and n for k,n in counts.items()) for counts in failures.values()):qa['status']='PARTIAL_FITS'
    if number in ('07','08') and result['voc_readout_gains'].empty:qa['status']='INSUFFICIENT_SUPPORT'
    if number=='09' and result.get('voc_window_comparisons',empty()).empty:qa['status']='INSUFFICIENT_SUPPORT'
    if number=='10' and result.get('voc_pattern_support',empty()).empty:qa['status']='PARTIAL_FITS'
    return result,qa


@contextmanager
def render_style():
    """Compatibility entry for the shared research-wide export policy."""
    from common.figure_style import paper_style
    with paper_style():
        yield


def equal_runs(frame, keys, values):
    """Never pool choices/decisions or align candidate states across participants."""
    z=frame.groupby(['subject','run',*keys],dropna=False)[values].mean().reset_index()
    return z.groupby(['subject',*keys],dropna=False)[values].mean().reset_index()


def render(root,tables):
    """All plots consume saved tables only; no extraction or model fit."""
    import matplotlib.pyplot as plt
    out=Path(root)/'figures';out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext))
        plt.close(fig)
    with render_style():
        frame=tables.get('voc_distributions',empty())
        if len(frame):
            for metric,part in frame.groupby('metric'):
                p=equal_runs(part,['quantile'],['value']);g=p.groupby('quantile').value.mean()
                fig,ax=plt.subplots(figsize=(10,7))
                for _,r in p.groupby('subject'):ax.plot(r['quantile'],r.value,alpha=.4)
                ax.plot(g.index,g.values,color='black',lw=3)
                ax.set(xlabel='Within-run quantile (equal run weighting)',ylabel=LABELS.get(metric,metric.replace('_',' ')))
                save(fig,'voc_distribution_'+metric)
        frame=tables.get('voc_computation_curves',empty())
        if len(frame):
            for metric in ('value','endpoint_voc','policy_js_step'):
                p=equal_runs(frame,['k'],[metric]);g=p.groupby('k')[metric].mean()
                support=frame.groupby('k').n.sum()
                fig,axs=plt.subplots(2,1,figsize=(11,10),sharex=True,gridspec_kw={'height_ratios':[3,1]})
                for _,r in p.groupby('subject'):axs[0].plot(r.k,r[metric],alpha=.4)
                axs[0].plot(g.index,g.values,color='black',lw=3)
                axs[0].set_ylabel(LABELS.get(metric,metric.replace('_',' ')))
                axs[1].plot(support.index,support.values,'o-');axs[1].set(xlabel='k (conditional on reaching k)',ylabel='N')
                save(fig,'voc_computation_'+metric)
        frame=tables.get('voc_human_fit',empty())
        if len(frame):
            p=equal_runs(frame,['termination'],['value']);order=sorted(p.termination.unique())
            fig,ax=plt.subplots(figsize=(11,7))
            for subject,r in p.groupby('subject'):
                ax.plot(range(len(order)),r.set_index('termination').value.reindex(order),'o-',label=f'sub{int(subject):03d}')
            ax.axhline(0,color='gray');ax.set(xticks=range(len(order)),xticklabels=order,ylabel=LABELS['human_logp_change']);ax.legend()
            save(fig,'voc_human_fit_change')
        frame=tables.get('voc_stopping_by_k',empty())
        if len(frame):
            p=equal_runs(frame,['k'],['continuation','gate']);g=p.groupby('k')[['continuation','gate']].mean()
            fig,ax=plt.subplots(figsize=(11,7))
            ax.plot(g.index,1-g.continuation,'o-',label='Observed free STOP')
            ax.plot(g.index,1-g.gate,'--',label='Gate P(STOP)');ax.set(xlabel='k (cap is not a STOP)',ylabel='Conditional STOP probability');ax.legend()
            save(fig,'voc_stopping_hazard')
        frame=tables.get('voc_window_comparisons',empty())
        if len(frame):
            use=frame.loc[frame.representation.eq('heads') & frame.sampling.eq('nonoverlap') & frame.target.eq('human_switch')]
            if len(use):
                fig,ax=plt.subplots(figsize=(12,max(6,.7*len(use))))
                labels=[f'sub{int(r.subject):03d}: {r.reference}' for r in use.itertuples()]
                ax.hlines(np.arange(len(use)),use.ci_low,use.ci_high);ax.scatter(use.gain,np.arange(len(use)))
                ax.axvline(0,color='gray');ax.set(yticks=np.arange(len(use)),yticklabels=labels,xlabel='Reduction in prospective switch log loss')
                save(fig,'voc_prospective_window_gains')
        frame=tables.get('voc_pattern_effects',empty())
        if len(frame):
            use=frame.loc[frame.hac_lag.eq(PRIMARY_LAG)&frame.status.eq('ok')]
            for (subject,held,target),part in use.groupby(['subject','held','target']):
                fig,ax=plt.subplots(figsize=(11,max(6,.8*len(part))))
                ax.errorbar(part['mean'],np.arange(len(part)),xerr=1.96*part.se,fmt='o',capsize=4)
                ax.axvline(0,color='gray');ax.set(yticks=np.arange(len(part)),yticklabels=[f'Pattern {int(s)}' for s in part.state],
                    xlabel='Within-fold descriptive association (95% HAC interval)')
                save(fig,f'voc_pattern_sub{int(subject):03d}_held{int(held)}_{target}')
        frame=tables.get('voc_relationship_curves',empty())
        if len(frame):
            frame=frame.copy()
            frame['condition']=frame['condition'].fillna('observed') if 'condition' in frame else 'observed'
            for (x,metric,condition),part in frame.groupby(['x_metric','metric','condition']):
                p=equal_runs(part,['bin'],['x','value'])
                g=p.groupby('bin')[['x','value']].mean()
                fig,ax=plt.subplots(figsize=(10,7))
                for _,q in p.groupby('subject'):ax.plot(q.x,q.value,alpha=.35,lw=1.5)
                ax.plot(g.x,g.value,'o-',color='#215d83',lw=3,label=condition.replace('_',' '));ax.legend()
                ax.set(xlabel=LABELS.get(x,x.replace('_',' ')),ylabel=LABELS.get(metric,metric.replace('_',' ')))
                save(fig,'voc_relation_'+x+'_'+metric+'_'+condition)
        frame=tables.get('voc_relationship_effects',empty())
        if len(frame):
            for (family,predictor),part in frame.loc[(frame.hac_lag==PRIMARY_LAG)&frame.status.eq('ok')].groupby(['family','predictor']):
                labels=[f"sub{int(r.subject):03d}: {r.target}" for r in part.itertuples()]
                fig,ax=plt.subplots(figsize=(13,max(6,.65*len(part))))
                ax.errorbar(part['mean'],np.arange(len(part)),xerr=1.96*part.se,fmt='o',capsize=4)
                ax.axvline(0,color='gray',lw=1);ax.set(yticks=np.arange(len(part)),yticklabels=labels,
                    xlabel='Standardized association (95% HAC interval)')
                save(fig,'voc_effect_'+family+'_'+predictor)
        for name,category in [('voc_readout_gains','reference'),('voc_early_budget_gains','model')]:
            frame=tables.get(name,empty())
            if frame.empty:continue
            for target,part in frame.groupby('target') if 'target' in frame else [('budget',frame)]:
                p=equal_runs(part,[category],['value']);order=sorted(p[category].unique())
                fig,ax=plt.subplots(figsize=(12,7))
                for subject,g in p.groupby('subject'):
                    ax.plot(range(len(order)),g.set_index(category).value.reindex(order),'o-',label=f'sub{int(subject):03d}')
                ax.axhline(0,color='gray',lw=1);ax.set(xticks=range(len(order)),xticklabels=order,
                    ylabel='Full minus comparator log score' if name=='voc_readout_gains' else 'Reduction in budget prediction MSE')
                ax.tick_params(axis='x',rotation=20);ax.legend()
                save(fig,name+'_'+str(target))
        ep=tables.get('voc_example_endpoints',empty());choice=tables.get('voc_example_choices',empty())
        if len(ep):
            for idx,(_,g) in enumerate(ep.groupby(KEYS)):
                c=choice.loc[choice.episode.eq(g.episode.iloc[0])&choice.decision.eq(g.decision.iloc[0])&choice.subject.eq(g.subject.iloc[0])]
                fig,axs=plt.subplots(6,1,figsize=(13,25),sharex=True)
                axs[0].plot(c.k,c.q_continue_ema,'o-',label='Q(C)');axs[0].plot(c.k,c.q_stop_ema,'o-',label='Q(S)');axs[0].legend();axs[0].set_ylabel('Native EMA Q')
                axs[1].plot(g.k,g.endpoint_voc,'o-',label='Common readout');axs[1].scatter(c.k,c.voc_ema,marker='x',label='Native pre-choice');axs[1].axhline(0,color='gray');axs[1].set_ylabel('VoC');axs[1].legend()
                axs[2].plot(g.k,g.value,'o-');axs[2].set_ylabel('Task value')
                for column in [col for col in g if col.startswith('root_q_')]:
                    if g[column].notna().any():axs[3].plot(g.k,g[column],label='Action '+column.rsplit('_',1)[1])
                axs[3].set_ylabel('Visited root Q')
                if axs[3].lines:axs[3].legend(ncol=3)
                axs[4].plot(g.k,g.policy_js_step,'o-');axs[4].set_ylabel('Policy JS distance per step')
                axs[5].plot(c.k,c.probability,'o-',label='Gate P(C)');axs[5].scatter(c.k,c.continued,marker='x',s=100,label='Actual choice')
                forced=c.loc[~c.voluntary.astype(bool)]
                if len(forced):axs[5].scatter(forced.k,forced.continued,marker='s',s=160,facecolors='none',edgecolors='black',label='Forced minimum')
                reset=c.loc[c.accepted_control.eq(1)]
                if len(reset):axs[5].scatter(reset.k,reset.continued,marker='v',s=100,label='RESET + CONTINUE')
                if g.termination.iloc[0]=='cap':axs[5].axvline(g.k.max(),ls=':',color='black',label='Cap (no STOP)')
                axs[5].set(xlabel='Imagination index k',ylabel='CONTINUE',ylim=(-.1,1.1));axs[5].legend(ncol=2)
                save(fig,f'voc_trajectory_{idx:02d}_{g.termination.iloc[0]}')
