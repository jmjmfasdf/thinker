"""Canonical 02 model validation and 03 voluntary computation allocation.

Consumes prepared native replay tables. Factual training logs are optional;
missing logs are reported and never replaced by new gameplay evaluation.
The native all-session model is not a held-out behavioral predictor.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from common.inference import fit_pooled

REPO=Path(__file__).resolve().parents[2]
CONFIG_KEYS=('min_search_steps','max_search_steps','voc_gate_q_temperature','voc_gru_connected',
             'voc_gate_exact_projection','voc_gate_epsilon_greedy_execution','voc_gate_execution_epsilon',
             'voc_gru_gradient_scale','actor_layer_norm','actor_h_layer_norm','actor_x_layer_norm',
             'actor_h_encoder_norm','actor_x_encoder_norm','dynamic_search_state_reset',
             'bc_burn_in','bc_burnin','icopro_burn_in_steps','discounting','reward_clip','think_cost','env_n','seed')
TRAIN_KEYS=('rmean_episode_return','rmean_len','mean_budget','max_budget','search/mean_steps',
    'search/forced_stop_rate','search/learned_stop_rate','search/mean_continue_probability',
    'actor/total_norm','actor/pg_loss','actor/baseline_loss','actor/mean_abs_v',
    'actor/bc_loss','actor/bc_accuracy','actor/voc_q_loss','actor/voc_gate_teacher_student_kl',
    'actor/voc_gate_teacher_student_gap_abs','actor/voc_gate_q_temperature',
    'actor/voc_continue_probability_delta_positive','actor/voc_continue_probability_delta_negative',
    'actor/voc_acceptance_continue_probability_delta_positive','actor/voc_acceptance_continue_probability_delta_negative',
    'actor/voc_acceptance_sampled_continue_given_delta_positive_rate',
    'actor/voc_acceptance_sampled_stop_given_delta_negative_rate',
    'actor/voc_gru_grad_ratio','actor/voc_gru_gradient_ratio','actor/learning_rate',
    'model/rs_loss','model/done_loss','model/img_loss','model/fea_loss','model/vs_loss',
    'model/policy_loss','model/total_loss_m','model/total_loss_p')


def _tables(prepared):
    d=prepared['decisions'].copy()
    e=prepared.get('choices',pd.DataFrame()).copy()
    aliases={'budget':'additional_computations','human_probability':'actor_human_probability',
             'model_correct':'actor_correct','run':'run_label','episode':'run_label',
             'decision':'real_step','age':'age_seconds','carry':'carried'}
    for key,source in aliases.items():
        if key not in d and source in d:d[key]=d[source]
    for col,value in [('log_budget',np.log1p(d.budget) if 'budget' in d else np.nan),
                      ('log_age',np.log1p(d.age) if 'age' in d else np.nan)]:
        if col not in d:d[col]=value
    if 'run_label' not in d:d['run_label']=d.episode
    if 'real_step' not in d:d['real_step']=d.decision
    for key,source in {'continued':'continue','probability':'voc_soft_continue_probability',
        'run':'run_label','episode':'run_label','decision':'real_step','search_steps':'choice_index'}.items():
        if key not in e and source in e:e[key]=e[source]
    return d,e


def _metadata(prepared):
    m=prepared.get('metadata',{}) or {}
    cp=m.get('checkpoint',m.get('semantic',{}).get('checkpoint',{}))
    if not isinstance(cp,dict):cp={'path':str(cp)}
    cfg=dict(prepared.get('config',{}))
    cfg.update(m.get('config',{}))
    qa=m.get('qa',{}) if isinstance(m.get('qa',{}),dict) else {}
    cfg.update(qa.get('config',{}))
    path=prepared.get('cache')
    if path:
        for filename in ['qa.json','logs/run.json']:
            p=Path(path)/filename
            if p.exists():
                stored=json.loads(p.read_text());cfg.update(stored.get('config',stored.get('qa',{}).get('config',{})))
                if not cp:cp=stored.get('checkpoint',{})
                break
    if isinstance(cp,dict) and cp.get('config'):
        expected=cp['config'].get('sha256');name=cp['config'].get('name')
        checkpoint_folder=qa.get('checkpoint')
        candidates=[]
        if checkpoint_folder and name:candidates.append(Path(checkpoint_folder)/name)
        logical=cp.get('logical_xpid')
        if logical:candidates += [REPO/'logs/thinker'/logical/n for n in [name,'config_c.yaml'] if n]
        for candidate in candidates:
            if not candidate.is_file():continue
            import hashlib,yaml
            if expected and hashlib.sha256(candidate.read_bytes()).hexdigest()!=expected:continue
            original=yaml.safe_load(candidate.read_text())
            cfg.update({k:original[k] for k in CONFIG_KEYS if k in original})
            break
    return m,cp,cfg


def _training(args,cp,subject,game):
    supplied=getattr(args,'training_root',None)
    base=Path(supplied) if supplied else REPO/'logs/thinker'
    logical=cp.get('logical_xpid') if isinstance(cp,dict) else None
    # Never choose a checkpoint by modification time or a partial name match.
    if (base/'logs.csv').is_file():folder=base
    elif logical:folder=base/logical
    else:return {},{'status':'missing_checkpoint_training_identity'}
    if not folder.is_dir():return {},{'status':'training_logs_unavailable','path':str(folder)}
    frames=[];audits=[]
    for learner,filename in [('actor','logs.csv'),('world_model','logs_model.csv')]:
        path=folder/filename
        if not path.exists():audits.append(dict(learner=learner,status='missing',path=str(path)));continue
        header=pd.read_csv(path,nrows=0).columns.tolist()
        cols=[c for c in ['real_step',*TRAIN_KEYS] if c in header]
        # Retain logged BC and gradient diagnostics regardless of naming generation.
        cols+= [c for c in header if (c.startswith(('actor/','model/')) and any(x in c for x in ('bc_','gru_grad','gru_','gradient_ratio'))) and c not in cols]
        if 'real_step' not in cols:audits.append(dict(learner=learner,status='missing_real_step',path=str(path)));continue
        data=pd.read_csv(path,usecols=cols).apply(pd.to_numeric,errors='coerce')
        data=data.dropna(subset=['real_step']).drop_duplicates('real_step',keep='last').sort_values('real_step')
        long=data.melt(id_vars='real_step',var_name='metric',value_name='value').dropna(subset=['value'])
        long['learner']=learner;frames.append(long)
        audits.append(dict(learner=learner,status='available',path=str(path),rows=len(data),
                           first_real_step=float(data.real_step.min()) if len(data) else np.nan,
                           last_real_step=float(data.real_step.max()) if len(data) else np.nan))
    status=[]
    for learner,filename in [('actor','meta.json'),('world_model','meta_model.json')]:
        p=folder/filename
        if p.exists():
            # Whitelist; never persist environment variables or credentials from meta.
            raw=json.loads(p.read_text())
            status.append(dict(learner=learner,successful=raw.get('successful'),date_start=raw.get('date_start'),date_end=raw.get('date_end')))
    result={'training_source_audit':pd.DataFrame(audits),'training_completion':pd.DataFrame(status)}
    if frames:result['training_curves']=pd.concat(frames,ignore_index=True)
    return result,{'status':'available' if frames else 'missing_numeric_training_logs','path':str(folder),
                  'logged_return_only':'Existing training-policy returns; no new game score evaluation'}


def validation(args,subject,game,prepared):
    d,_=_tables(prepared);meta,cp,cfg=_metadata(prepared)
    if 'human_probability' not in d or 'model_correct' not in d:
        pcols=sorted([c for c in d if c.startswith('actor_action_probability_') and c.rsplit('_',1)[-1].isdigit()],key=lambda x:int(x.rsplit('_',1)[1]))
        if not pcols:raise ValueError('Model validation requires recorded REAL probabilities or verified human_probability/model_correct')
        p=d[pcols].to_numpy(float);human=d.human_action.to_numpy(int)
        if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Invalid recorded REAL policy')
        d['human_probability']=p[np.arange(len(d)),human];d['model_correct']=(p.argmax(1)==human).astype(int)
    p=d.human_probability.to_numpy(float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid recorded human-action probabilities')
    d['log_score']=np.log(np.maximum(p,1e-12));d['negative_log_likelihood']=-d.log_score
    rows=[]
    for run,r in d.groupby('run'):
        for population,mask in [('all',np.ones(len(r),bool)),('persist',r['switch'].eq(0)),('switch',r['switch'].eq(1))]:
            z=r.loc[mask]
            rows.append(dict(run=run,session=int(r.session.iloc[0]),population=population,n=len(z),
                accuracy=z.model_correct.mean(),log_score=z.log_score.mean(),
                human_probability=z.human_probability.mean(),zero_probability_count=int(z.human_probability.eq(0).sum())))
    run_scores=pd.DataFrame(rows)
    by_action=d.groupby(['run','session','human_action']).agg(n=('model_correct','size'),
        accuracy=('model_correct','mean'),log_score=('log_score','mean'),human_probability=('human_probability','mean')).reset_index()
    columns=[x for x in ['run','episode','session','decision','source_frame','time','human_action',
        'incumbent_action','switch','human_probability','model_correct','log_score'] if x in d]
    checkpoint={k:cfg.get(k) for k in CONFIG_KEYS if k in cfg}
    checkpoint.update(logical_xpid=cp.get('logical_xpid'),checkpoint_requested_step=cp.get('requested_step'),actor_sha256=cp.get('actor',{}).get('sha256'),
        model_sha256=cp.get('model',{}).get('sha256'),training_scope='all sessions; training/replay overlap',
        evaluated_decisions=len(d),sessions=d.session.nunique(),source='recorded final REAL policy; no second softmax')
    tables={'decision_scores':d[columns],'run_scores':run_scores,'action_scores':by_action,
            'checkpoint_conditions':pd.DataFrame([checkpoint])}
    training,trainingqa=_training(args,cp,subject,game);tables.update(training)
    qa={'status':'completed','model_generalization':'not held out from Thinker training',
        'training_diagnostics':trainingqa,'final_REAL_decisions':len(d),
        'aggregation':'equal runs within subject then equal subjects',
        'checkpoint_step':'Use checkpoint binding if present; never infer learning progress from a single replay',
        'no_new_gameplay_evaluation':True}
    return tables,qa,{}


def allocation(args,subject,game,prepared):
    d,e=_tables(prepared);meta,cp,cfg=_metadata(prepared)
    required=['voluntary','probability','continued','voc_ema','voc_online','search_steps','depth','run','episode','decision','session']
    missing=[c for c in required if c not in e]
    if missing:raise ValueError('Allocation requires prechoice fields: '+','.join(missing))
    if 'cap_reached' not in e:e['cap_reached']=False
    v=e.loc[e.voluntary.eq(True)].copy()
    maximum=int(cfg.get('max_search_steps',d.budget.max()))
    minimum=int(cfg.get('min_search_steps',d.budget.min()))
    budget=[];cal=[];signals=[];voccurves=[];statecurves=[];hazard=[];depth=[];run_diag=[]
    for run,r in d.groupby('run'):
        counts=r.budget.value_counts()
        for k in range(maximum+1):budget.append(dict(run=run,budget=k,n=int(counts.get(k,0)),probability=counts.get(k,0)/len(r)))
        for metric in [c for c in ['ema_initial','root_value','M','root_entropy'] if c in r]:
            finite=r.loc[np.isfinite(r[metric])]
            if not len(finite):continue
            z=finite.assign(bin=np.minimum((finite[metric].rank(pct=True)*10).astype(int),9))
            for b,a in z.groupby('bin'):statecurves.append(dict(run=run,metric=metric,bin=b,state=a[metric].mean(),budget=a.budget.mean(),n=len(a)))
    for run,r in v.groupby('run'):
        z=r.assign(bin=np.minimum((r.probability*10).astype(int),9))
        for b,a in z.groupby('bin'):cal.append(dict(run=run,bin=b,probability=a.probability.mean(),actual=a.continued.mean(),n=len(a)))
        for k,a in r.groupby('search_steps'):hazard.append(dict(run=run,search_steps=k,probability=a.probability.mean(),actual=a.continued.mean(),n=len(a)))
        for k,a in r.groupby('depth'):depth.append(dict(run=run,depth=k,probability=a.probability.mean(),actual=a.continued.mean(),n=len(a)))
        for source in ['ema','online']:
            x=r['voc_'+source]
            for sign,mask in [('positive',x>0),('negative',x<0),('tie',x==0)]:
                a=r.loc[mask];signals.append(dict(run=run,source=source,sign=sign,n=len(a),probability=a.probability.mean(),actual=a.continued.mean()))
            z=r.assign(voc_bin=np.minimum((x.rank(pct=True)*10).fillna(-1).astype(int),9))
            for b,a in z.loc[z.voc_bin.ge(0)].groupby('voc_bin'):voccurves.append(dict(run=run,source=source,bin=b,voc=a['voc_'+source].mean(),probability=a.probability.mean(),actual=a.continued.mean(),n=len(a)))
        row=dict(run=run,n=len(r),brier=float(np.square(r.probability-r.continued).mean()),
            positive_count=int(r.voc_ema.gt(0).sum()),negative_count=int(r.voc_ema.lt(0).sum()),
            gate_contrast=float(r.loc[r.voc_ema.gt(0),'probability'].mean()-r.loc[r.voc_ema.lt(0),'probability'].mean()))
        for source in ['ema','online']:
            finite=np.isfinite(r['voc_'+source])&np.isfinite(r.continued)
            row[source+'_continue_auroc']=float(roc_auc_score(r.loc[finite,'continued'],r.loc[finite,'voc_'+source])) if r.loc[finite,'continued'].nunique()==2 else np.nan
        temp=cfg.get('voc_gate_q_temperature')
        if temp is not None and float(temp)>0:
            teacher=expit(r.voc_ema.to_numpy()/float(temp));a=r.probability.clip(1e-10,1-1e-10).to_numpy();b=teacher.clip(1e-10,1-1e-10)
            teachercontrast=float(np.mean(teacher[r.voc_ema.gt(0)])-np.mean(teacher[r.voc_ema.lt(0)])) if row['positive_count'] and row['negative_count'] else np.nan
            row.update(teacher_gate_KL=float(np.mean(b*np.log(b/a)+(1-b)*np.log((1-b)/(1-a)))),
                teacher_probability_mae=float(np.mean(abs(r.probability-teacher))),teacher_contrast=teachercontrast,
                contrast_retention=row['gate_contrast']/teachercontrast if np.isfinite(teachercontrast) and teachercontrast>1e-12 else np.nan)
        run_diag.append(row)
    frames=dict(budget_distribution=pd.DataFrame(budget),gate_calibration=pd.DataFrame(cal),
        sign_conditioned_continuation=pd.DataFrame(signals),voc_continuation=pd.DataFrame(voccurves),
        state_budget=pd.DataFrame(statecurves),continuation_by_search_step=pd.DataFrame(hazard),
        continuation_by_depth=pd.DataFrame(depth),run_gate_diagnostics=pd.DataFrame(run_diag))
    # Preserve all observed prechoice signals for later table-only diagnostic redraws.
    frames['choice_observations']=e
    cols=[c for c in ['run','episode','session','decision','budget','cap_reached','ema_initial',
        'online_initial','root_value','root_entropy','M','age','carry'] if c in d]
    frames['decision_allocation']=d[cols]
    v['run_label']=v.episode;v['real_step']=v.decision;v['log_search_steps']=np.log1p(v.search_steps)
    effects=[]
    for source in ['ema','online']:
        needed=['log_search_steps','depth','M','root_value','log_age','prior_reward']
        if all(c in v for c in needed):
            for row in fit_pooled(v,'continued',['voc_'+source],needed,True):effects.append(dict(family='voluntary_continuation',source=source,**row))
    controls=['M','root_value','log_age','prior_reward','carry']
    if all(c in d for c in controls+['ema_initial']):
        for row in fit_pooled(d,'log_budget',['ema_initial'],controls,False):effects.append(dict(family='budget',source='ema',**row))
    frames['adjusted_associations']=pd.DataFrame(effects)
    summary=dict(decisions=len(d),mean_budget=float(d.budget.mean()),fraction_budget0=float(d.budget.eq(0).mean()),
        fraction_budget1=float(d.budget.eq(1).mean()),fraction_minimum=float(d.budget.eq(minimum).mean()),
        fraction_cap=float(d.cap_reached.mean()) if 'cap_reached' in d else np.nan,
        min_forced_choices=int((~e.voluntary.astype(bool)).sum()),voluntary_choices=len(v),
        cap_reaching_choices_retained=int(v.cap_reached.sum()),minimum=minimum,maximum=maximum,
        temperature=cfg.get('voc_gate_q_temperature',np.nan))
    frames['allocation_summary']=pd.DataFrame([summary])
    qa=dict(status='completed',**summary,
        initiation='Mandatory below min_search_steps; exclude forced choices from voluntary evidence',
        stopping='Cap-reaching sampled CONTINUE retained; no fictitious STOP at the cap',
        event_timing='VoC and gate probability measured before the same control choice',
        inference='Descriptive human-trajectory replay associations; not causal environment reward benefit',
        regression='Session fixed effects, temporal HAC, real decision grouping without crossing episodes',
        aggregation='Equal runs within subject then equal subjects',
        temperature_available='voc_gate_q_temperature' in cfg)
    state_tables,state_qa=task_state(d,meta)
    frames.update(state_tables);qa['task_state']=state_qa
    return frames,qa,{}


def analyze(number,args,subject,game,prepared,workdir=None):
    number=f'{int(number):02d}'
    fn={'02':validation,'03':allocation}.get(number)
    if fn is None:raise ValueError('validation_allocation supports only analyses 02 and 03')
    tables,qa,arrays=fn(args,subject,game,prepared)
    for key,frame in tables.items():
        frame=frame.copy();frame['subject']=subject;frame['game']=game;tables[key]=frame
    return tables,qa,arrays


def _mean(d,keys,values):
    # Tables may contain several runs per participant. Equal participants, not rows.
    per=d.groupby(['subject',*keys],dropna=False)[values].mean().reset_index()
    return per.groupby(keys,dropna=False)[values].mean().reset_index() if keys else per[values].mean().to_frame().T


def render(number,root,tables,title,arrays=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);figures=root/'figures';figures.mkdir(parents=True,exist_ok=True);out=[]
    def save(fig,name):
        fig.suptitle(title,fontsize=13);fig.tight_layout()
        for ext in ('png','pdf'):
            p=figures/f'{name}.{ext}';fig.savefig(p,dpi=180,bbox_inches='tight');out.append(str(p))
        plt.close(fig)
    if int(number)==2:
        d=tables.get('run_scores',pd.DataFrame())
        if len(d):
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for ax,metric,label in [(axes[0],'accuracy','Final REAL human-action accuracy'),(axes[1],'log_score','Human action log likelihood')]:
                z=_mean(d,['population'],[metric]).set_index('population').reindex(['all','persist','switch'])
                ax.bar(range(3),z[metric],color=['#5b7182','#2780a3','#d68b40']);ax.set(xticks=range(3),xticklabels=['All','Persist','Switch'],ylabel=label)
            save(fig,'human_action_validation')
        d=tables.get('training_curves',pd.DataFrame())
        if len(d):
            families={'world_model_learning':['model/policy_loss','model/vs_loss','model/rs_loss','model/img_loss'],
                'actor_learning':['actor/bc_loss','actor/pg_loss','actor/baseline_loss','actor/voc_q_loss'],
                'training_allocation':['mean_budget','search/mean_steps','search/forced_stop_rate','search/learned_stop_rate'],
                'recorded_training_return':['rmean_episode_return']}
            for name,metrics in families.items():
                metrics=[m for m in metrics if d.metric.eq(m).any()]
                if not metrics:continue
                fig,axes=plt.subplots(1,len(metrics),figsize=(max(6,3.7*len(metrics)),3.5),squeeze=False)
                for ax,metric in zip(axes.flat,metrics):
                    part=d.loc[d.metric.eq(metric)].copy();part['bin']=(part.real_step//50000).astype(int)
                    for sub,r in part.groupby('subject'):
                        z=r.groupby('bin')[['real_step','value']].mean();ax.plot(z.real_step/1e6,z.value,label=f'sub{int(sub):03d}')
                    ax.set(title=metric,xlabel='Training environment actions (M)');ax.legend(fontsize=7,frameon=False)
                save(fig,name)
    elif int(number)==3:
        d=tables.get('budget_distribution',pd.DataFrame())
        if len(d):
            z=_mean(d,['budget'],['probability']);fig,ax=plt.subplots(figsize=(8,4));ax.bar(z.budget,z.probability,color='#2780a3')
            ax.set(xlabel='Imagination computations',ylabel='Fraction of decisions (equal run/subject weight)');save(fig,'planning_budget_histogram')
        cal=tables.get('gate_calibration',pd.DataFrame());sign=tables.get('sign_conditioned_continuation',pd.DataFrame())
        if len(cal) and len(sign):
            fig,axes=plt.subplots(1,3,figsize=(12,4));z=_mean(cal,['bin'],['probability','actual'])
            axes[0].plot(z.probability,z.actual,'o-');axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Recorded continue probability',ylabel='Actual continuation',title='Voluntary execution')
            for ax,source in zip(axes[1:],['ema','online']):
                z=_mean(sign.loc[sign.source.eq(source)],['sign'],['probability','actual']).set_index('sign').reindex(['negative','positive'])
                ax.bar(np.arange(2)-.16,z.probability,.3,label='Gate probability',color='#2780a3');ax.bar(np.arange(2)+.16,z.actual,.3,label='Actual continue',color='#d68b40')
                ax.set(xticks=[0,1],xticklabels=['VoC < 0','VoC > 0'],ylim=(0,1),title=source.upper());ax.legend(fontsize=8)
            save(fig,'voc_and_actual_continuation')
        d=tables.get('state_budget',pd.DataFrame())
        if len(d):
            metrics=list(d.metric.unique());fig,axes=plt.subplots(1,len(metrics),figsize=(3.6*len(metrics),3.5),squeeze=False)
            for ax,metric in zip(axes.flat,metrics):
                z=_mean(d[d.metric.eq(metric)],['bin'],['budget']);ax.plot((z.bin+.5)/10,z.budget,'o-');ax.set(xlabel=metric+' percentile within run',ylabel='Mean imagination count')
            save(fig,'state_dependent_budget')
        for key,x,label in [('continuation_by_search_step','search_steps','Completed imagination computations'),('continuation_by_depth','depth','Current search-tree depth')]:
            d=tables.get(key,pd.DataFrame())
            if not len(d):continue
            z=_mean(d,[x],['probability','actual']);fig,ax=plt.subplots(figsize=(7,4));ax.plot(z[x],z.probability,'o-',label='Gate probability');ax.plot(z[x],z.actual,'x--',label='Actual continuation');ax.set(xlabel=label,ylabel='Voluntary continue probability',ylim=(0,1));ax.legend();save(fig,key)
        d=tables.get('voc_continuation',pd.DataFrame())
        if len(d):
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for ax,source in zip(axes,['ema','online']):
                z=_mean(d[d.source.eq(source)],['bin'],['voc','probability','actual']);ax.plot(z.voc,z.probability,'o-',label='Gate probability');ax.plot(z.voc,z.actual,'x--',label='Actual continuation');ax.axvline(0,color='gray',lw=.7);ax.set(xlabel=source.upper()+' VoC',ylabel='Continue probability',ylim=(0,1));ax.legend()
            save(fig,'voc_continuation_curve')
        d=tables.get('task_state_curves',pd.DataFrame())
        if len(d):
            metrics=list(d.metric.unique())
            for target,label in [('budget','Mean imagination count'),('voc','Initial EMA VoC')]:
                fig,axes=plt.subplots(2,3,figsize=(12,7),squeeze=False)
                for ax,metric in zip(axes.flat,metrics):
                    z=_mean(d.loc[d.metric.eq(metric)],['bin'],[target]);ax.plot((z.bin+.5)/10,z[target],'o-')
                    ax.set(xlabel=metric+' percentile within run',ylabel=label)
                for ax in list(axes.flat)[len(metrics):]:ax.set_visible(False)
                save(fig,'task_state_'+target)
        d=tables.get('observational_gramian',pd.DataFrame())
        if len(d):
            fig,axes=plt.subplots(1,3,figsize=(10,3.6))
            for ax,metric in zip(axes,['trace','largest_eigenvalue','effective_rank']):
                per_run=d.groupby(['subject','run'])[metric].mean().reset_index()
                per_subject=per_run.groupby('subject')[metric].mean()
                ax.scatter(np.arange(len(per_subject)),per_subject,color='#2780a3');ax.set(xticks=np.arange(len(per_subject)),xticklabels=[f'sub{int(x):03d}' for x in per_subject.index],ylabel=metric,title='Observed action geometry')
            save(fig,'observational_ram_geometry')
    else:raise ValueError('Only 02/03 can be rendered by validation_allocation')
    return out


def _screen_descriptors(images):
    """Unfitted screen summaries; neither object labels nor latent goals."""
    n=len(images);coarse=np.empty((n,49));contrast=np.empty(n);edges=np.empty(n)
    for start in range(0,n,256):
        frame=images[start:start+256].astype(np.float32)
        if frame.ndim==4:
            if frame.shape[-1] in (1,3,4):frame=frame.mean(axis=-1)
            elif frame.shape[1] in (1,3,4):frame=frame.mean(axis=1)
            else:raise ValueError('Unknown source screen channel layout')
        if frame.ndim!=3 or min(frame.shape[1:])<7:raise ValueError('Unsupported screen dimensions')
        frame/=255.
        ys=np.array_split(np.arange(frame.shape[1]),7);xs=np.array_split(np.arange(frame.shape[2]),7)
        stop=start+len(frame)
        coarse[start:stop]=np.column_stack([frame[:,y][:,:,x].mean((1,2)) for y in ys for x in xs])
        contrast[start:stop]=frame.std((1,2))
        edges[start:stop]=.5*(np.abs(np.diff(frame,axis=1)).mean((1,2))+np.abs(np.diff(frame,axis=2)).mean((1,2)))
    return coarse,contrast,edges


def task_state(decisions,metadata):
    """Native-source screen/RAM descriptors and descriptive action geometry.

    RAM change uses consecutive replay decisions within one source segment.
    Screen change compares immediately consecutive raw frames. No future
    observation or predicted image enters these state/budget association fits.
    """
    from common.research_inputs import sha
    eps={e['metadata']['episode_id']:e['metadata'] for e in metadata.get('episodes',[])}
    if not eps:return {},{'status':'missing_native_source_bindings','inference':'No task-state values inferred from legacy tables'}
    d=decisions.copy().reset_index(drop=True)
    needed=['episode','decision','source_frame','human_action','time','run','session']
    if any(k not in d for k in needed):return {},{'status':'missing_native_alignment_columns'}
    metrics=['visual_change','screen_contrast','screen_edge_density','ram_entropy','ram_change','ram_variance']
    for col in metrics:d[col]=np.nan
    missing=set(d.episode)-set(eps)
    if missing:raise ValueError('Unbound task-state episodes: '+str(sorted(missing)))
    source_episodes={}
    for episode in d.episode.unique():
        q=eps[episode];source_episodes.setdefault(q['source_path'],[]).append(episode)
    audit=[];grams=[]
    for source,episodes in source_episodes.items():
        path=Path(source);expected={eps[e]['source_sha256'] for e in episodes}
        if len(expected)!=1 or not path.is_file() or sha(path)!=next(iter(expected)):
            raise ValueError('Task-state source identity mismatch: '+source)
        with np.load(path,allow_pickle=False) as raw:
            action=raw['action'];action=action.argmax(1) if action.ndim==2 else action.reshape(-1)
            times=raw['time'].reshape(-1).astype(float)
            first=raw['is_first'].reshape(-1).astype(bool);terminal=raw['is_terminal'].reshape(-1).astype(bool)
            dt=np.diff(times);positive=dt[dt>0]
            gap=max(.25,10*float(np.median(positive))) if len(positive) else .25
            boundary=first.copy();boundary[0]=True;boundary[1:]|=terminal[:-1]|(dt<=0)|(dt>gap)
            segment=np.cumsum(boundary)
            ram=raw['RAM'].copy() if 'RAM' in raw else None
            if ram is not None and (ram.ndim!=2 or len(ram)!=len(times) or not np.issubdtype(ram.dtype,np.integer) or ram.min()<0 or ram.max()>255):
                raise ValueError('RAM must contain aligned byte-valued vectors')
            selected=np.unique(d.loc[d.episode.isin(episodes),'source_frame'].to_numpy(int))
            if len(selected) and (selected.min()<0 or selected.max()>=len(times)):raise ValueError('Task-state source index outside source')
            screen_frames=np.unique(np.r_[selected,np.maximum(selected-1,0)])
            visual=contrast=edges=None
            if 'image' in raw:
                images=raw['image']
                if len(images)!=len(times):raise ValueError('Screen/source-time length mismatch')
                visual,contrast,edges=_screen_descriptors(images[screen_frames]);del images
        source_count=0
        for episode in episodes:
            part=d.loc[d.episode.eq(episode)].sort_values('decision');ix=part.index
            f=part.source_frame.to_numpy(int);source_count+=len(f)
            if not np.array_equal(action[f],part.human_action.to_numpy(int)):raise ValueError('Task-state action alignment mismatch')
            np.testing.assert_allclose(times[f],part.time,atol=1e-8,rtol=0)
            ok=(np.diff(part.decision.to_numpy(int))==1)&(np.diff(f)>0)&(segment[f[1:]]==segment[f[:-1]])
            if visual is not None:
                here=np.searchsorted(screen_frames,f);prior=np.searchsorted(screen_frames,np.maximum(f-1,0))
                change=np.linalg.norm(visual[here]-visual[prior],axis=1);change[boundary[f]]=np.nan
                d.loc[ix,'visual_change']=change;d.loc[ix,'screen_contrast']=contrast[here];d.loc[ix,'screen_edge_density']=edges[here]
            if ram is not None:
                r=ram[f].astype(np.uint8);norm=r.astype(float)/255.
                entropy=[]
                for row in r:
                    p=np.bincount(row,minlength=256)/len(row);p=p[p>0];entropy.append(-(p*np.log2(p)).sum())
                d.loc[ix,'ram_entropy']=entropy;d.loc[ix,'ram_variance']=norm.var(1)
                d.loc[ix,'ram_change']=np.r_[np.nan,np.where(ok,(r[1:]!=r[:-1]).mean(1),np.nan)]
                delta=norm[1:]-norm[:-1];previous_action=part.human_action.to_numpy(int)[:-1]
                columns=[delta[ok&(previous_action==a)].mean(0) for a in np.unique(previous_action[ok])]
                if columns:
                    eigen=np.linalg.svd(np.stack(columns,axis=1),compute_uv=False)**2
                    positive=eigen[eigen>1e-12];p=positive/positive.sum() if len(positive) else positive
                    grams.append(dict(run=part.run.iloc[0],episode=episode,session=int(part.session.iloc[0]),n_transitions=int(ok.sum()),observed_actions=len(columns),
                        trace=float(eigen.sum()),largest_eigenvalue=float(eigen.max()),effective_rank=float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0.,
                        interpretation='Observed mean RAM delta by preceding action; descriptive geometry, not causal controllability'))
        audit.append(dict(source=source,source_sha256=next(iter(expected)),decisions=source_count,episodes=len(episodes),
            ram_status='available' if ram is not None else 'missing',screen_status='available' if visual is not None else 'missing',ram_bytes=ram.shape[1] if ram is not None else 0))
    effects=[];curves=[]
    controls=['M','root_value','log_age','prior_reward','carry']
    for metric in metrics:
        for target in ['log_budget','ema_initial']:
            if all(k in d for k in [target,*controls]):
                for row in fit_pooled(d,target,[metric],controls,False):effects.append(dict(family='observed_environment_state',target=target,**row))
        for run,r in d.groupby('run'):
            z=r.loc[np.isfinite(r[metric])].copy()
            if not len(z):continue
            z['bin']=np.minimum((z[metric].rank(pct=True)*10).astype(int),9)
            for b,a in z.groupby('bin'):curves.append(dict(run=run,metric=metric,bin=b,n=len(a),state=a[metric].mean(),budget=a.budget.mean(),voc=a.ema_initial.mean()))
    identity=[c for c in ['run','episode','session','decision','source_frame','time','human_action','budget','ema_initial'] if c in d]
    tables=dict(task_state_observations=d[identity+metrics],task_state_sources=pd.DataFrame(audit),task_state_effects=pd.DataFrame(effects),
                observational_gramian=pd.DataFrame(grams),task_state_curves=pd.DataFrame(curves))
    qa=dict(status='complete' if all(q['ram_status']==q['screen_status']=='available' for q in audit) else 'partial_source_support',
        source_count=len(audit),source_sha256_verified=True,metrics=metrics,fit='Session fixed effects and decision-level temporal HAC',
        visual_change='Euclidean fixed 7x7 grayscale change versus immediately preceding raw frame; no segment crossing',
        ram_change='Fraction of differing RAM bytes versus preceding consecutive replay decision; no episode/segment crossing',
        screen_descriptors='Spatial contrast and mean adjacent-pixel absolute difference; not object-level complexity',
        gramian='Mean observed RAM delta by preceding action; descriptive geometry, not causal controllability',
        heldout_prediction='Budget prediction with held-session fitting is in analysis10; these associations are descriptive')
    return tables,qa
