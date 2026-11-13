"""Themes 02--05 and 12: separate human boundaries, model control and prediction."""
import numpy as np
import pandas as pd
from scipy.special import logit
from common.matching import nearest_visual
from common import native
from common.inference import fit_pooled
from common.prediction import baseline, blocks, probabilities, scores
from common.outcomes import risk_set

POPULATIONS=("overall","noop_to_active","active_to_noop")

def association(frame,target,predictors,controls,binary,**meta):
    return [dict(meta,**r) for r in fit_pooled(frame,target,predictors,controls,binary)]

def boundary(d,e):
    records=[]; matched=[]
    controls=baseline(d)+["root_value","root_entropy","log_budget","searched","carried"]
    for population in POPULATIONS:
        part=risk_set(d,population)
        # U and K are highly related; test them in separate predeclared models.
        for family,predictors in [("M_U",["M","U_all"]),("K",["K"])]:
            records+=association(part,"outcome",predictors,controls,True,population=population,family=family)
        # Matched descriptive contrasts. Fit matching scaling within run; never match on M/U/K.
        for run,r in part.groupby("run_label"):
            r=r.copy(); r["age_bin"]=pd.qcut(r.log_age.rank(method="first"),5,labels=False,duplicates="drop")
            r["time_bin"]=np.floor(r.run_progress*5).clip(0,4)
            pairs=[]
            for _,s in r.groupby(["incumbent_action","age_bin","time_bin","prior_reward"],dropna=False):
                sw=s.loc[s.outcome.eq(1)];stay=s.loc[s.outcome.eq(0)]
                if not len(sw) or not len(stay):continue
                # Fixed visual descriptor nearest neighbour, matching with replacement.
                cols=[c for c in s if c.startswith("visual_") and c!='visual_change']
                dist,ix=nearest_visual(sw[cols],stay[cols],s[cols])
                st=stay.iloc[ix[:,0]]
                for metric in ["M","U_all","K"]:
                    sd=r[metric].std(ddof=0)
                    if not sd>1e-10: continue
                    delta=(sw[metric].to_numpy()-st[metric].to_numpy())/sd
                    pairs.append(dict(metric=metric,sum=float(np.nansum(delta)),n=int(np.isfinite(delta).sum())))
            for metric in ["M","U_all","K"]:
                z=[x for x in pairs if x['metric']==metric];n=sum(x['n'] for x in z)
                matched.append(dict(run_label=run,population=population,term=metric,n_pairs=n,
                                    value=sum(x['sum'] for x in z)/n if n else np.nan))
    return {"effects":pd.DataFrame(records),"matched_run_effects":pd.DataFrame(matched)}

def control(d,e):
    records=[];summary=[];curves=[]
    v=e.loc[e.voluntary.eq(True)].copy()
    for population,part in [("start",v.loc[v.choice_index.eq(0)]),("continue",v.loc[v.choice_index.gt(0)])]:
        for source in ["ema","online"]:
            records+=association(part,"continue",[f"voc_{source}"],
                ["log_search_steps","depth","margin","root_value","log_age","prior_reward"],True,
                family="prechoice_VOC",population=population,source=source)
            for run,r in part.groupby("run_label"):
                x=r[f"voc_{source}"];p=r.voc_soft_continue_probability;y=r["continue"]
                good=np.isfinite(x)&np.isfinite(p)
                rr=dict(run_label=run,population=population,source=source,n=int(good.sum()),
                    p_continue_voc_positive=float(p[good&x.gt(0)].mean()),
                    p_continue_voc_negative=float(p[good&x.lt(0)].mean()),
                    observed_continue_voc_positive=float(y[good&x.gt(0)].mean()),
                    observed_continue_voc_negative=float(y[good&x.lt(0)].mean()),
                    brier=float(((p[good]-y[good])**2).mean()))
                summary.append(rr)
            q=part.loc[part[f"voc_{source}"].notna()].copy()
            if len(q):
                q["bin"]=pd.qcut(q[f"voc_{source}"],10,duplicates="drop")
                for b,r in q.groupby("bin",observed=True):
                    curves.append(dict(population=population,source=source,voc=r[f"voc_{source}"].mean(),
                        predicted_continue=r.voc_soft_continue_probability.mean(),actual_continue=r["continue"].mean(),n=len(r)))
    records+=association(d.loc[d.scored.eq(1)],"log_budget",["ema_initial"],
                         ["M","root_value","log_age","prior_reward","carried"],False,
                         family="decision_budget",population="all",source="ema")
    budget=d.groupby(["run_label","additional_computations"]).size().rename('n').reset_index()
    return {"effects":pd.DataFrame(records),"gate_run_scores":pd.DataFrame(summary),
            "gate_curves":pd.DataFrame(curves),"budget_distribution":budget}

def search_pairs(e):
    z=e.sort_values(native.KEYS+['choice_index']).copy()
    g=z.groupby(native.KEYS,sort=False)
    z['next_continue']=g['continue'].shift(-1)
    z['next_voluntary']=g.voluntary.shift(-1)
    z['next_proceed']=g.accepted_control.shift(-1).eq(0).astype(float)
    # Current post-computation evidence is available before the next choice.
    z=z.loc[z.has_next_choice & z.next_voluntary.eq(True) & z.search_computation.eq(1)]
    return z

def search_process(d,e):
    records=[];z=search_pairs(e)
    for outcome,binary in [('next_continue',True),('next_proceed',True)]:
        records+=association(z,outcome,['return_update'],
            ['voc_ema','margin','root_value','log_search_steps','depth','return_gain'],binary,
            family='next_search_choice',population=outcome)
    return {'effects':pd.DataFrame(records)}

def actor_readout(d):
    a=int(d.n_actions.iloc[0]); cp=int(d.checkpoint.iloc[0])
    d=d.loc[d.actor_action_probability_valid.eq(1)&d.scored.eq(1)].reset_index(drop=True)
    target=d[[f'actor_action_probability_{j}' for j in range(a)]].to_numpy(float)
    train=d.session.ne(cp).to_numpy();test=~train
    block=blocks(d); definitions={'R0_prior':block['B1_initial_support'],
        'R1_quantity':block['B2_quantity'],
        'R2_search':block['B3_search_content']+[f'root_ns_terminal_{j}' for j in range(a)]}
    metrics=[];qa=[]
    for name,columns in definitions.items():
        p,audit=probabilities(d,columns,target,train,test,soft=True)
        qa.append(dict(model=name,checkpoint=cp,**audit))
        if p is None:continue
        selected=d.loc[test].copy();selected['cross_entropy']=-(target[test]*np.log(p.clip(1e-12))).sum(1)
        for run,r in selected.groupby('run_label'):
            metrics.append(dict(run_label=run,checkpoint=cp,model=name,n=len(r),cross_entropy=r.cross_entropy.mean()))
    return {'actor_run_scores':pd.DataFrame(metrics),'actor_fit_qa':pd.DataFrame(qa)}

def human_prediction(d,temporal=False):
    cp=int(d.checkpoint.iloc[0]);a=int(d.n_actions.iloc[0]);metrics=[];qa=[];predictions=[]
    defs=blocks(d)
    if temporal:
        base=defs['B4_VOC']
        defs={'T0_current':base,'T1_previous_evaluation':base+['lag1_M','lag1_U_all','lag1_K','lag1_ema_initial'],
              'T2_bout_history':base+['lag1_M','lag1_U_all','lag1_K','lag1_ema_initial','past_bout_U','lag1_log_budget'],
              'T3_arrived_mismatch':base+['lag1_M','lag1_U_all','lag1_K','lag1_ema_initial','past_bout_U','lag1_log_budget',
                                         'arrived_reward_error','arrived_error_available']}
    for population in ['exact_action',*POPULATIONS]:
        if population=='exact_action':
            part=d.loc[d.scored.eq(1)&d.incumbent_action.ge(0)].copy();part['outcome']=part.human_action
            n_actions=a
        else:part=risk_set(d,population);n_actions=2
        part=part.reset_index(drop=True);train=part.session.ne(cp).to_numpy();test=~train
        truth=part.outcome.to_numpy(int);result={}
        for name,cols in defs.items():
            p,audit=probabilities(part,cols,truth,train,test,actions=n_actions)
            qa.append(dict(checkpoint=cp,population=population,model=name,**audit))
            if p is not None:result[name]=p
        # Comparisons have identical held-out decisions and all prespecified models.
        if len(result)!=len(defs):continue
        scored=part.loc[test,native.KEYS+['session','checkpoint','outcome']].copy().reset_index(drop=True)
        scored['population']=population
        for name,p in result.items():
            scored['log_score_'+name]=np.log(p[np.arange(len(p)),scored.outcome.to_numpy(int)].clip(1e-12))
            for run,ix in scored.groupby('run_label').groups.items():
                metrics.append(dict(run_label=run,checkpoint=cp,population=population,model=name,
                                    **scores(scored.loc[ix,'outcome'],p[ix])))
        predictions.append(scored)
    return {'run_scores':pd.DataFrame(metrics),'fit_qa':pd.DataFrame(qa),
            'predictions':pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame()}

def temporal_associations(d,e):
    d=d.copy()
    d['change_x_U']=d.visual_change*d.U_all
    records=[]
    for population in POPULATIONS:
        part=risk_set(d,population)
        records+=association(part,'outcome',['lag1_U_all','past_bout_U'],
            baseline(d)+['M','U_all','K','root_value','log_budget','ema_initial'],True,
            family='past_evidence_given_current',population=population)
        records+=association(part,'outcome',['change_x_U'],
            baseline(d)+['M','U_all','visual_change','root_value','log_budget','ema_initial'],True,
            family='environment_change_interaction',population=population)
    records+=association(d.loc[d.scored.eq(1)],'log_budget',['carried'],
        ['M','root_value','log_age','prior_reward','visual_change','ema_initial'],False,
        family='recorded_tree_carry',population='budget')
    records+=association(d.loc[d.scored.eq(1)],'log_budget',['arrived_reward_error'],
        ['M','root_value','log_age','prior_reward','visual_change','ema_initial','carried'],False,
        family='arrived_model_mismatch',population='budget')
    carry=d.groupby('run_label').agg(decisions=('real_step','size'),incoming_carry=('carried','mean'),
        useful_carry=('useful_carry','mean')).reset_index()
    return {'effects':pd.DataFrame(records),'recorded_carry':carry}

def complexity(d,e):
    records=[]
    for outcome in ['log_budget','ema_initial']:
        records+=association(d.loc[d.scored.eq(1)],outcome,['visual_change'],
            ['M','root_value','log_age','prior_reward','carried'],False,
            family='observed_visual_change',population=outcome)
    return {'effects':pd.DataFrame(records)}
