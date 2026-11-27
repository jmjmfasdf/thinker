"""Current-capture policy linkage (06) and across-observation continuity (08).

No old-number output paths are read. All predictions hold out whole human
sessions of the readout; Thinker itself trained on all sessions. Same-phase
frozen readouts are separated from the actual REAL output after STOP input.
Forecast mismatch is optional and never inferred from human replay alone.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import softmax

from common.io import sha
from common.prediction import probabilities, scores, baseline
from common.inference import fit_pooled
from common.outcomes import risk_set
from common.research_inputs import enrich_decisions

POPULATIONS = ('overall', 'noop_to_active', 'active_to_noop')
KEYS = ['episode', 'decision']


def _association(d, target, predictors, controls, binary, **labels):
    return [dict(labels, **r) for r in fit_pooled(d, target, predictors, controls, binary)]


def _blocks(d):
    prior = baseline(d) + ['M', 'root_value', 'root_entropy', 'initial_q_margin']
    prior += [c for c in d if c.startswith('log_policy_')]
    quantity = prior + ['log_budget', 'searched', 'carried']
    if 'max_depth' in d: quantity.append('max_depth')
    content = quantity + ['A_V', 'U_V', 'D_V', 'A_Q', 'U_Q', 'Q_supported', 'valid_update_count']
    content += [c for c in d if c.startswith(('root_qs_mean_terminal_', 'root_ns_terminal_'))]
    if 'terminal_return_gain' in d and d.terminal_return_gain.notna().any(): content.append('terminal_return_gain')
    return {'R0_prior':prior, 'R1_quantity':quantity, 'R2_search':content}


def _actor_readout(d):
    target_cols = sorted((c for c in d if c.startswith('actor_action_probability_') and c.rsplit('_',1)[1].isdigit()), key=lambda c:int(c.rsplit('_',1)[1]))
    d = d.loc[d.scored.eq(1) & d.actor_action_probability_valid.eq(1)].reset_index(drop=True)
    target = d[target_cols].to_numpy(float)
    if (not np.isfinite(target).all() or (target<0).any()
            or not np.allclose(target.sum(1),1,atol=1e-5)):
        raise ValueError('Recorded REAL policy is invalid')
    rows, audits, predictions = [], [], []
    definitions = _blocks(d)
    for held in sorted(d.session.unique()):
        train, test = d.session.ne(held).to_numpy(), d.session.eq(held).to_numpy()
        fitted = {}
        for name, columns in definitions.items():
            probability, audit = probabilities(d, columns, target, train, test, soft=True)
            audits.append(dict(model=name, heldout_session=int(held), **audit))
            if probability is not None: fitted[name] = probability
        # All nested comparisons retain the same held-out observations/folds.
        if len(fitted) != len(definitions): continue
        selected = d.loc[test, ['run','episode','decision','session','time']].reset_index(drop=True)
        for name, probability in fitted.items():
            selected['cross_entropy_'+name] = -(target[test]*np.log(probability.clip(1e-12))).sum(1)
            for run, part in selected.groupby('run'):
                rows.append(dict(run_label=run, heldout_session=int(held), model=name, n=len(part),
                                 cross_entropy=float(part['cross_entropy_'+name].mean())))
        predictions.append(selected)
    return {'actor_run_scores':pd.DataFrame(rows), 'actor_fit_qa':pd.DataFrame(audits),
            'actor_predictions':pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame()}


def _captured_actor_head(metadata, num_actions):
    """Load the matching head and its pickled buffer from the owning capture."""
    import yaml
    from common.native_replay import ActorHeads, frozen_import
    from common.voc_dynamics import checkpoint_for_capture
    cp, actor, config = checkpoint_for_capture(metadata)
    cfg = yaml.safe_load(config.read_text())
    if cfg['critic_enc_type'] != 0 or cfg['last_layer_n'] != 0 or cfg['sep_actor_critic']:
        raise ValueError('Same-phase policy analysis requires the captured linear actor head')
    # Checkpoints include a serialized return-stat buffer even though this
    # analysis reads only linear weights. A reused VoC cache has not imported it.
    frozen_import(Path(actor).parents[4], 'buffer')
    return ActorHeads(dict(checkpoint=str(cp), config=cfg, num_actions=int(num_actions)))


def _rich_policy(args, subject, game, d, e, metadata, qa):
    """Verified numeric trace reads; no latent images or new model rollout."""
    from common.native_replay import Trace, verify_chunks, decision_edges, js_distance
    head = _captured_actor_head(metadata, d.n_actions.iloc[0])
    d, e = d.copy(), e.copy()
    quality = []
    episodes = {x['metadata']['episode_id']:x for x in metadata['episodes']}
    for episode, part in d.groupby('episode',sort=False):
        ep = episodes[episode]
        t = Trace(Path(metadata['capture'])/ep['path'])
        flags = ['initial_root','real_transition','search_computation','current_node_depth_post',
                 'actor_reset','phase_pre','search_steps_pre','pre_observation_source_index']
        verify_chunks(t, flags)
        f = {name:t.scalar(name) for name in flags}
        lookup = {int(row.decision):row for row in part.itertuples()}
        edges = [edge for edge in decision_edges(f['initial_root'],f['real_transition'],f['search_computation'],
                 f['current_node_depth_post'],f['actor_reset']) if edge[0] in lookup]
        selected = np.unique(np.concatenate([np.r_[left+1,right,comp,comp+1] for _,left,right,comp,_ in edges])).astype(int)
        fields = ['voc_features','actor_baseline','actor_action_probability','tree_reps_pre','tree_reps_post']
        verify_chunks(t, fields, selected)
        z = t.read('voc_features',selected).astype(float)
        if not t.valid('voc_features')[selected].all(): raise ValueError('Missing frozen actor features')
        position = {int(ev):i for i,ev in enumerate(selected)}
        baseline = head.linear(z,'baseline')
        recorded = t.read('actor_baseline',selected)
        error = float(np.max(np.abs(baseline-recorded)))
        if error>2e-4 or np.any(np.abs(recorded[:,0])>=head.clip-1e-5):
            raise ValueError('Frozen actor baseline cannot reproduce saved computation endpoints')
        maximum_policy_error=0.
        for j,left,right,comp,_ in edges:
            row=lookup[j]; ix=row.Index
            if left!=row.root_event or right!=row.real_event or len(comp)!=row.budget:
                raise ValueError('Native decision key/budget differs from scalar preparation')
            if len(comp) and not np.all(np.diff(comp)==1):
                raise ValueError('Noncontiguous computations need a separate timing audit')
            endpoints=np.r_[comp[0],comp+1].astype(int) if len(comp) else np.array([left+1])
            if not np.array_equal(f['search_steps_pre'][endpoints],np.arange(len(comp)+1)):
                raise ValueError('Computation endpoint order differs from actual count')
            if not np.all(f['pre_observation_source_index'][endpoints]==row.source_frame):
                raise ValueError('Computation endpoints cross real observations')
            features=z[[position[int(k)] for k in endpoints]]
            common=head.common(features,f['phase_pre'][endpoints],target=1)
            policy=softmax(head.linear(common,'policy'),axis=1)
            real_feature=z[position[right]:position[right]+1]
            actual=softmax(head.linear(real_feature,'policy'),axis=1)
            saved=t.read('actor_action_probability',[right])
            policy_error=float(np.max(np.abs(actual-saved)));maximum_policy_error=max(maximum_policy_error,policy_error)
            if policy_error>2e-5: raise ValueError('Frozen REAL policy differs from saved probabilities')
            d.loc[ix,'same_phase_policy_js']=float(js_distance(policy[:1],policy[-1:])[0])
            d.loc[ix,'stop_input_policy_js']=float(js_distance(policy[-1:],actual)[0])
            d.loc[ix,'total_policy_js']=float(js_distance(policy[:1],actual)[0])
            d.loc[ix,'same_phase_incumbent_change']=float(policy[-1,int(row.incumbent_action)]-policy[0,int(row.incumbent_action)])
            d.loc[ix,'max_depth']=float(np.max(f['current_node_depth_post'][comp])) if len(comp) else 0.
            d.loc[ix,'first_control_native_event']=left+1
        choices=e.loc[e.episode.eq(episode)].sort_values(['decision','choice_index'])
        ev=choices.event.to_numpy(int)
        verify_chunks(t,['tree_reps_pre','tree_reps_post'],ev)
        pre=t.tree('rollout_return',ev,'pre').reshape(-1)
        post=t.tree('rollout_return',ev,'post').reshape(-1)
        e.loc[choices.index,'return_update']=post-pre
        e.loc[choices.index,'return_gain']=pre-choices.root_value.to_numpy(float)
        e.loc[choices.index,'search_computation']=f['search_computation'][ev]
        # Cap reaching has no subsequent voluntary STOP choice. Its final
        # control record is pre-computation, so read the actual final endpoint.
        final_events=np.array([comp[-1]+1 if len(comp) else left+1 for _,left,_,comp,_ in edges],int)
        terminal=t.tree('rollout_return',final_events,'pre').reshape(-1)
        for offset,(j,_,_,_,_) in enumerate(edges):
            ix=lookup[j].Index
            d.loc[ix,'terminal_return_gain']=float(terminal[offset]-d.loc[ix,'root_value'])
        last=choices.groupby('decision',sort=False).tail(1)
        for qrow in last.itertuples():
            ix=lookup[int(qrow.decision)].Index
            d.loc[ix,'ema_change']=float(qrow.voc_ema-d.loc[ix,'ema_initial'])
        quality.append(dict(episode=episode, decisions=len(part), baseline_max_error=error,
                            actual_policy_max_error=maximum_policy_error))
    e['log_search_steps']=np.log1p(e.choice_index)
    e['run_label']=e.episode; e['real_step']=e.decision
    e['margin']=e.M
    return d,e,quality


def _next_choice(e):
    z=e.sort_values(['episode','decision','choice_index']).copy()
    grouped=z.groupby(KEYS,sort=False)
    nxt=grouped.choice_index.shift(-1)
    z['next_continue']=grouped.continued.shift(-1)
    z['next_voluntary']=grouped.voluntary.shift(-1)
    z['next_proceed']=grouped.accepted_control.shift(-1).eq(0).astype(int)
    valid=(z.continued.eq(1)&nxt.eq(z.choice_index+1)&z.next_voluntary.eq(True)&z.search_computation.eq(1))
    return z.loc[valid].copy()


def _history(d, metadata):
    """Raw bout IDs catch switch-and-return between sparse replay observations."""
    d=d.sort_values(['episode','decision']).copy().reset_index(drop=True)
    episodes={x['metadata']['episode_id']:x['metadata'] for x in metadata['episodes']}
    sources={}
    for episode,part in d.groupby('episode',sort=False):
        meta=episodes[episode];source=Path(meta['source_path'])
        if str(source) not in sources:
            if sha(source)!=meta['source_sha256']: raise ValueError('Raw behavioral source changed before temporal analysis')
            with np.load(source,allow_pickle=False) as raw:
                action=raw['action'];action=action.argmax(1) if action.ndim==2 else action
                times=raw['time'].astype(float);boundary=raw['is_first'].astype(bool).copy();boundary[0]=True
                boundary[1:]|=raw['is_terminal'][:-1].astype(bool)|(np.diff(times)<=0)
                onset=boundary.copy();onset[1:]|=action[1:]!=action[:-1]
                sources[str(source)]=(onset.cumsum(),action,times)
        bouts,action,times=sources[str(source)];frames=part.source_frame.to_numpy(int)
        if not np.array_equal(action[frames],part.human_action): raise ValueError('Temporal source action mismatch')
        np.testing.assert_allclose(times[frames],part.time,atol=1e-8,rtol=0)
        # Use the last raw action strictly BEFORE the target action. Resetting
        # on action[frames] would put the current switch outcome into a predictor.
        d.loc[part.index,'prior_raw_bout_id']=bouts[np.maximum(frames-1,0)]
    from common.sampling import sequence_index
    start=d.episode.ne(d.episode.shift())|sequence_index(d).diff().ne(1)|d.time.diff().le(0)|d.time.diff().gt(.25)
    d['segment']=start.cumsum()
    for name in ['M','A_V','U_V','D_V','ema_initial','log_budget']:
        d['lag1_'+name]=d.groupby('segment',sort=False)[name].shift(1)
    bout=(start|d.prior_raw_bout_id.ne(d.prior_raw_bout_id.shift())).cumsum()
    d['past_bout_A_V']=d.groupby(bout,sort=False).A_V.cumsum()-d.A_V
    d.loc[start,'past_bout_A_V']=np.nan
    d['first_control_native_event']=d.root_event+1
    d['change_x_U_V']=d.visual_change*d.U_V
    d['run_label']=d.episode;d['real_step']=d.decision
    return d


def _attach_forecasts(d, path, metadata):
    d=d.copy();d['arrived_reward_error']=np.nan;d['arrived_error_available']=0.
    if path is None:
        return d,dict(status='BLOCKED_INPUT',reason='No verified forecast-outcome input; arrived mismatch omitted',required_argument='--forecast-input CSV')
    path=Path(path);manifest_path=path.parent/'manifest.json'
    if not path.is_file() or not manifest_path.is_file():
        return d,dict(status='BLOCKED_INPUT',reason='Forecast table or manifest is missing',required=[str(path),str(manifest_path)])
    m=json.loads(manifest_path.read_text())
    if m.get('capture_manifest_sha256')!=metadata['manifest_sha256']:
        raise ValueError('Forecast input belongs to a different or undeclared native capture')
    expected=m.get('artifacts',{}).get(path.name)
    if isinstance(expected,dict):expected=expected.get('sha256')
    if not expected or expected!=sha(path):raise ValueError('Forecast input hash is absent or changed')
    f=pd.read_csv(path)
    need=['run','episode','forecast_available_time','forecast_available_event','outcome_available_time',
          'outcome_available_event','state_comparison_primary','reward_prediction_error']
    if not set(need).issubset(f):
        return d,dict(status='BLOCKED_INPUT',reason='Forecast table lacks availability/path-match fields',required=sorted(set(need)-set(f)))
    f=f.loc[f.state_comparison_primary.eq(1)].copy()
    numeric=need[2:]
    f=f.loc[np.isfinite(f[numeric].to_numpy(float)).all(1)]
    if (f.outcome_available_time.lt(f.forecast_available_time).any()
            or f.outcome_available_event.lt(f.forecast_available_event).any()):
        raise ValueError('Forecast outcome precedes the forecast computation')
    keys=set(zip(d.run,d.episode))
    if not set(zip(f.run,f.episode)).issubset(keys):
        raise ValueError('Forecast input contains run/episode identities absent from this analysis')
    groups={key:p for key,p in f.groupby(['run','episode'])}
    incorporated=0
    for key,part in d.groupby(['run','episode'],sort=False):
        if key not in groups:continue
        part=part.sort_values('decision');q=groups[key]
        times=part.time.to_numpy(float);events=part.first_control_native_event.to_numpy(int)
        if np.any(np.diff(times)<0) or np.any(np.diff(events)<=0):raise ValueError('Nonmonotonic decision availability')
        ix=np.maximum(np.searchsorted(times,q.outcome_available_time.to_numpy(),side='left'),
                      np.searchsorted(events,q.outcome_available_event.to_numpy(),side='right'))
        good=ix<len(part);ix=ix[good];q=q.loc[good]
        sums=np.zeros(len(part));counts=np.zeros(len(part))
        np.add.at(sums,ix,np.abs(q.reward_prediction_error.to_numpy()));np.add.at(counts,ix,1)
        arrived=pd.Series(np.divide(sums,counts,out=np.full(len(part),np.nan),where=counts>0)).ffill().to_numpy()
        d.loc[part.index,'arrived_reward_error']=arrived
        d.loc[part.index,'arrived_error_available']=np.isfinite(arrived).astype(int)
        incorporated+=len(q)
    return d,dict(status='COMPLETED' if incorporated else 'BLOCKED_INPUT',source=str(path),sha256=sha(path),
                  matched_forecasts=len(f),incorporated_forecasts=incorporated,
                  available_decisions=int(d.arrived_error_available.sum()),
                  availability='outcome time <= observation time AND outcome event < first control event; reset per episode')


def _temporal_prediction(d, mismatch):
    current=_blocks(d)['R2_search']+['ema_initial']
    prior=current+['lag1_M','lag1_U_V','lag1_A_V','lag1_D_V','lag1_ema_initial']
    history=prior+['past_bout_A_V','lag1_log_budget']
    definitions={'T0_current':current,'T1_previous_evaluation':prior,'T2_bout_history':history}
    if mismatch:definitions['T3_arrived_mismatch']=history+['arrived_reward_error','arrived_error_available']
    rows,qa,predictions=[],[],[]
    for population in ('exact_action',*POPULATIONS):
        if population=='exact_action':
            part=d.loc[d.scored.eq(1)&d.incumbent_action.ge(0)].copy();part['outcome']=part.human_action
            n_actions=int(d.n_actions.iloc[0])
        else:part=risk_set(d,population);n_actions=2
        part=part.reset_index(drop=True)
        for held in sorted(part.session.unique()):
            train=part.session.ne(held).to_numpy();test=~train
            fitted={}
            for model,columns in definitions.items():
                probability,audit=probabilities(part,columns,part.outcome.to_numpy(int),train,test,actions=n_actions)
                qa.append(dict(population=population,heldout_session=int(held),model=model,**audit))
                if probability is not None:fitted[model]=probability
            if len(fitted)!=len(definitions):continue
            selected=part.loc[test,['run','episode','decision','session','time','outcome']].reset_index(drop=True)
            selected['population']=population
            for model,p in fitted.items():
                selected['log_score_'+model]=np.log(p[np.arange(len(p)),selected.outcome.to_numpy(int)].clip(1e-12))
                for run,ix in selected.groupby('run').groups.items():
                    rows.append(dict(run_label=run,heldout_session=int(held),population=population,model=model,
                                     **scores(selected.loc[ix,'outcome'],p[ix])))
            predictions.append(selected)
    return {'run_scores':pd.DataFrame(rows),'fit_qa':pd.DataFrame(qa),
            'predictions':pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame()}


def analyze(number,args,subject,game,prepared,workdir):
    number=f'{int(number):02d}'
    if number not in ('06','08'):raise ValueError('Policy/continuity analysis must be 06 or 08')
    d=enrich_decisions(prepared['decisions'],prepared['metadata'])
    e=prepared['choices'].copy()
    d['run_label']=d.episode;d['real_step']=d.decision
    qa=dict(status='COMPLETED',subject=int(subject),game=game,readout_holdout='whole human session',
            network_training_scope='all sessions; readout holdout is not unseen-network-session prediction',
            causal_claim=False)
    if number=='06':
        d,e,quality=_rich_policy(args,subject,game,d,e,prepared['metadata'],prepared['qa'])
        tables=_actor_readout(d)
        next_choice=_next_choice(e)
        effects=[]
        for target in ['next_continue','next_proceed']:
            effects+=_association(next_choice,target,['return_update'],
                ['voc_ema','margin','root_value','log_search_steps','depth','return_gain'],True,
                family='completed_evidence_next_choice',population=target)
        tables['next_choice_effects']=pd.DataFrame(effects)
        columns=['run','episode','session','decision','time','budget','A_V','U_V','D_V',
                 'same_phase_policy_js','stop_input_policy_js','total_policy_js','same_phase_incumbent_change']
        tables['policy_revision']=d[columns]
        tables['policy_revision_run_means']=d.groupby(['run','session','budget'])[columns[6:]].mean().reset_index()
        tables['next_choice_pairs']=next_choice[['run','episode','session','decision','event','return_update',
            'return_gain','voc_ema','choice_index','next_continue','next_proceed']]
        tables['reconstruction_qa']=pd.DataFrame(quality)
        qa.update(decisions=len(d),next_choice_pairs=len(next_choice),
                  policy_readout='same frozen REAL-phase policy at first/final computation endpoints; actual post-STOP REAL policy is separate',
                  endpoint_intervention=False)
    else:
        d=_history(d,prepared['metadata'])
        d,forecast_qa=_attach_forecasts(d,getattr(args,'forecast_input',None),prepared['metadata'])
        effects=[]
        for population in POPULATIONS:
            part=risk_set(d,population)
            effects+=_association(part,'outcome',['lag1_U_V','past_bout_A_V'],
                baseline(d)+['M','U_V','A_V','D_V','root_value','log_budget','ema_initial'],True,
                family='past_evaluation_given_current',population=population)
            effects+=_association(part,'outcome',['change_x_U_V'],
                baseline(d)+['M','U_V','visual_change','root_value','log_budget','ema_initial'],True,
                family='environment_change_interaction',population=population)
        controls=['M','root_value','log_age','prior_reward','visual_change','ema_initial']
        effects+=_association(d,'log_budget',['carried'],controls,False,family='recorded_tree_carry',population='budget')
        if forecast_qa['status']=='COMPLETED':
            effects+=_association(d,'log_budget',['arrived_reward_error'],controls+['carried'],False,
                                  family='arrived_model_mismatch',population='budget')
        tables=_temporal_prediction(d,forecast_qa['status']=='COMPLETED')
        tables['effects']=pd.DataFrame(effects)
        tables['recorded_carry']=d.groupby(['run','session']).agg(decisions=('decision','size'),
            incoming_carry=('carried','mean')).reset_index()
        columns=['run','episode','session','decision','time','budget','carry','prior_raw_bout_id','segment',
                 'M','A_V','U_V','D_V','ema_initial','lag1_M','lag1_A_V','lag1_U_V','lag1_D_V',
                 'lag1_ema_initial','lag1_log_budget','past_bout_A_V','arrived_reward_error','arrived_error_available']
        tables['temporal_records']=d[columns]
        qa.update(decisions=len(d),segments=int(d.segment.nunique()),forecast_input=forecast_qa,
                  complete_previous_evaluation_rows=int(d.lag1_U_V.notna().sum()),
                  carry_interpretation='recorded incoming tree carry; not proof of action-sequence reuse')
    fit_tables=[t for name,t in tables.items() if name.endswith('fit_qa')]
    failures=sum(int(t.status.ne('ok').sum()) for t in fit_tables if 'status' in t)
    qa['failed_readout_fits']=failures
    if failures:qa['status']='PARTIAL_FITS'
    for table in tables.values():table['subject']=int(subject);table['game']=game
    return tables,qa,{}


def render(number,root,tables,title):
    """Render only persisted numeric results; never reopen source captures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);folder=root/'figures';folder.mkdir(parents=True,exist_ok=True)
    made=[]
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:
            path=folder/(name+'.'+ext);fig.savefig(path,dpi=160,bbox_inches='tight');made.append(str(path))
        plt.close(fig)
    if int(number)==6:
        revision=tables.get('policy_revision_run_means',pd.DataFrame())
        if len(revision):
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for run,p in revision.groupby('run'):
                axes[0].plot(p.budget,p.same_phase_policy_js,alpha=.4)
                axes[1].plot(p.budget,p.stop_input_policy_js,alpha=.4)
            for ax,label in zip(axes,['Same-phase computation endpoint change','Separate STOP-input/output change']):
                ax.set(xlabel='Actual computation count',ylabel='Policy JS divergence',title=label)
            fig.suptitle(title);save(fig,'policy_revision')
        score=tables.get('actor_run_scores',pd.DataFrame())
        if len(score):
            fig,ax=plt.subplots(figsize=(6,4))
            p=score.pivot_table(index='run_label',columns='model',values='cross_entropy')
            order=['R0_prior','R1_quantity','R2_search']
            for _,row in p.reindex(columns=order).iterrows():ax.plot(range(3),row,'o-',alpha=.4)
            ax.set(xticks=range(3),xticklabels=['Prior/context','+ quantity','+ search content'],ylabel='Held-out policy cross entropy',title=title)
            save(fig,'actor_readout')
    else:
        records=tables.get('temporal_records',pd.DataFrame())
        if len(records):
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for run,p in records.groupby('run'):
                # Descriptive summaries retain raw units and coverage in tables.
                valid=p.dropna(subset=['lag1_U_V','U_V'])
                if len(valid):
                    bins=pd.qcut(valid.lag1_U_V.rank(method='first'),10,labels=False,duplicates='drop')
                    avg=valid.groupby(bins)[['lag1_U_V','U_V']].mean()
                    axes[0].plot(avg.lag1_U_V,avg.U_V,'o-',alpha=.4)
                q=p.groupby('carry').budget.mean()
                axes[1].plot(q.index,q.values,'o-',alpha=.4)
            axes[0].set(xlabel='Previous-observation U_V',ylabel='Current-observation U_V',title='Evaluation continuity')
            axes[1].set(xlabel='Recorded incoming carry',ylabel='Mean computation count',xticks=[0,1],title='Carry and allocation')
            fig.suptitle(title);save(fig,'continuity')
        scores=tables.get('run_scores',pd.DataFrame())
        if len(scores):
            populations=list(scores.population.unique());fig,axes=plt.subplots(1,len(populations),figsize=(4*len(populations),4),squeeze=False)
            for ax,population in zip(axes.ravel(),populations):
                selected=scores.loc[scores.population.eq(population)]
                p=selected.pivot_table(index='run_label',columns='model',values='log_score')
                order=sorted(p.columns)
                for _,row in p[order].iterrows():ax.plot(range(len(order)),row,'o-',alpha=.4)
                ax.set(xticks=range(len(order)),xticklabels=[x.split('_',1)[0] for x in order],ylabel='Held-out human log score',title=population)
            fig.suptitle(title);save(fig,'temporal_prediction')
    return made
