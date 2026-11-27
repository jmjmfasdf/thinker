"""Action-identified prospective test for theme 09 (frozen, observational).

The literal pre/post forecast never fits a direction to human outcomes. A
separate session-held-out readout tests incremental information using one
signed coefficient shared by all candidate actions. Neither is a STOP branch
experiment or an evaluation on data unseen by the trained Thinker.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from common.io import signature
from common.native import prepare_matrix
from common.native_replay import sha
from common.paths import REPO
from common.voc_dynamics import KEYS, unique

VERSION = 'action-support-v1'
PREFIX = 'action_support_'
CONTEXT = ['log_age', 'log_age_sq', 'recent10', 'recent30', 'prior_reward',
           'prior_done', 'root_value', 'root_entropy', 'initial_q_margin', 'visual_change']
MEAN_FIELDS = ['budget', 'carry', 'initial_voc', 'voc_A_V', 'policy_js']
SCORE_KEYS = ['population', 'model', 'reference']


def empty(columns=('status',)):
    return pd.DataFrame(columns=list(columns))


def protocol(args, sample_mode):
    result = dict(version=VERSION, sample_mode=sample_mode,
        window=float(getattr(args, 'window', .5)), horizon=float(getattr(args, 'horizon', .5)),
        max_gap=float(getattr(args, 'max_gap', .25)), block_seconds=30.,
        penalty=float(getattr(args, 'penalty', .1)), minimum_train=int(getattr(args, 'minimum_train', 200)),
        minimum_test=20, bootstrap=int(getattr(args, 'bootstrap', 1000)), seed=int(getattr(args, 'seed', 701)),
        primary='destination: post minus pre conditional log score at nonoverlapping landmarks',
        window_rule='left-held time integral on [t-window,t); constant raw action through t',
        outcome='incumbent or first raw switch destination in (t,t+horizon]',
        selection='past eligibility and timestamps, never future action identity',
        readout='matched REAL-phase linear head; not a counterfactual STOP policy',
        model_training='all four sessions; exploratory frozen-model analysis',
        aggregation='equal run within session, equal session within subject; equal subjects per game')
    if sample_mode not in ('all', '15hz'):
        raise ValueError('Unknown action-support sample mode')
    if any(not np.isfinite(result[k]) or result[k] <= 0 for k in ('window','horizon','max_gap','penalty')):
        raise ValueError('Action-support durations and penalty must be finite and positive')
    if result['bootstrap'] < 20 or result['minimum_train'] < 1:
        raise ValueError('Invalid bootstrap/training support settings')
    result['protocol_id'] = signature(result)
    return result


def logit_columns(frame, phase):
    prefix = 'policy_logit_' + phase + '_'
    columns = sorted((c for c in frame if c.startswith(prefix)), key=lambda c: int(c[len(prefix):]))
    if len(columns) < 2 or columns != [prefix+str(i) for i in range(len(columns))]:
        raise ValueError('Missing/invalid action-logit schema; regenerate the native VoC cache')
    return columns


def raw_behavior(times, actions, first, terminal, max_gap=.25):
    """Segment original time, preserving reset, terminal and clock boundaries."""
    t = np.asarray(times, float).reshape(-1)
    a = np.asarray(actions).reshape(-1)
    first = np.asarray(first, bool).reshape(-1)
    terminal = np.asarray(terminal, bool).reshape(-1)
    if not len(t) or not (len(t) == len(a) == len(first) == len(terminal)):
        raise ValueError('Invalid raw behavior lengths')
    if not np.isfinite(t).all() or not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError('Nonfinite time or noninteger human action')
    a = a.astype(int)
    boundary = first.copy(); boundary[0] = True
    boundary[1:] |= terminal[:-1] | (np.diff(t) <= 0) | (np.diff(t) > max_gap)
    segment = np.cumsum(boundary)-1
    starts = np.flatnonzero(boundary); ends = np.r_[starts[1:]-1, len(t)-1]
    bout_start = boundary | np.r_[False, a[1:] != a[:-1]]
    bouts = np.flatnonzero(bout_start)
    bout = np.cumsum(bout_start)-1
    next_frame = np.r_[bouts[1:], -1][bout]
    valid = (next_frame >= 0) & (next_frame <= ends[segment])
    next_frame = np.where(valid, next_frame, -1)
    return dict(time=t, action=a, segment=segment, segment_end=ends[segment],
                bout=bout, bout_start=bouts[bout], next_frame=next_frame)


def load_raw(metadata, max_gap):
    sources = {}
    for ep in metadata['episodes']:
        meta = ep['metadata']; run = meta['source_id']
        if run in sources:
            if sources[run]['sha'] != meta['source_sha256']:
                raise ValueError('Conflicting raw source bindings')
            continue
        path = Path(meta['source_path'])
        if not path.exists(): path = REPO/'behavioral_data_block'/run
        if sha(path) != meta['source_sha256']: raise ValueError('Changed raw behavioral source: '+str(path))
        with np.load(path, allow_pickle=False) as raw:
            a = raw['action']; a = a.argmax(1) if a.ndim == 2 else a
            source = raw_behavior(raw['time'], a, raw['is_first'], raw['is_terminal'], max_gap)
        source['sha'] = meta['source_sha256']; sources[run] = source
    return sources


def _weighted(values, weights):
    # Do not interpolate invalid source measurements. Training-only imputation
    # handles missing nuisance controls; missing policy endpoints fail closed.
    x = np.asarray(values, float)
    return np.sum(x * weights.reshape((-1,)+(1,)*(x.ndim-1)), axis=0)


def make_windows(decisions, voc, metadata, raw, spec):
    """Separate input construction from labels; never reuse legacy head windows."""
    unique(decisions, KEYS); unique(voc, KEYS)
    pre_cols, post_cols = logit_columns(voc, 'pre'), logit_columns(voc, 'post')
    if len(pre_cols) != len(post_cols): raise ValueError('Pre/post action counts differ')
    actions = len(pre_cols)
    extra = pre_cols+post_cols+['initial_voc','voc_A_V','policy_js']
    d = decisions.drop(columns=[c for c in extra if c in decisions]).merge(
        voc[KEYS+extra], on=KEYS, how='left', validate='one_to_one', indicator=True)
    if not d['_merge'].eq('both').all(): raise ValueError('Missing native decision readouts')
    d = d.drop(columns='_merge')
    if d.empty or d.subject.nunique()!=1 or d.game.nunique()!=1:
        raise ValueError('Action-support extraction requires one nonempty participant/game unit')
    if 'n_actions' in d and not d.n_actions.eq(actions).all():
        raise ValueError('Policy readout and native action space disagree')
    if not np.isfinite(d[pre_cols+post_cols].to_numpy(float)).all():
        raise ValueError('Nonfinite policy logits')
    controls = [c for c in CONTEXT if c in d] + sorted(c for c in d if c.startswith(('visual_', 'previous_action_')))
    controls = list(dict.fromkeys(controls))
    episodes = {e['metadata']['episode_id']: e['metadata'] for e in metadata['episodes']}
    rows, exclusions = [], []
    for episode, part in d.groupby('episode', sort=False):
        part = part.sort_values('source_frame').reset_index(drop=True)
        meta = episodes[episode]; source = raw[meta['source_id']]
        if not part.run.eq(meta['source_id']).all(): raise ValueError('Episode/run binding mismatch')
        frames = part.source_frame.to_numpy(int); times = part.time.to_numpy(float)
        if np.any(np.diff(frames) <= 0) or np.any(frames < 0) or np.any(frames >= len(source['time'])):
            raise ValueError('Nonunique/out-of-range source frames')
        np.testing.assert_array_equal(source['action'][frames], part.human_action.to_numpy())
        np.testing.assert_allclose(source['time'][frames], times, atol=1e-8, rtol=0)
        if np.any(source['action'][frames] < 0) or np.any(source['action'][frames] >= actions):
            raise ValueError('Human action outside checkpoint action space')
        episode_first = int(meta['source_first_frame'])
        episode_last = episode_first+int(meta['source_logical_observation_count'])-1
        if frames[0] < episode_first or frames[-1] > episode_last:
            raise ValueError('Decision outside capture episode')
        pre, post = part[pre_cols].to_numpy(float), part[post_cols].to_numpy(float)
        # Remove the arbitrary common logit offset before integrating.
        pre -= pre[:, :1]; post -= post[:, :1]
        if spec['sample_mode'] == '15hz':
            if 'sample_index' not in part: raise ValueError('15Hz selection lacks sequence identity')
            sequence = part.sample_index.to_numpy(int)
        else:
            sequence = frames
        next_allowed = -np.inf
        counts = dict(candidates=len(part), short_past=0, past_gap=0, action_changed=0,
                      future_censored=0, overlapping=0, retained=0)
        # A replay episode is one monotone source segment; fail rather than
        # silently running searchsorted across a hidden reset or clock change.
        if np.any(np.diff(times) <= 0): raise ValueError('Nonmonotone replay episode clock')
        for end, t in enumerate(times):
            left = t-spec['window']; stop = t+spec['horizon']; f = frames[end]
            start = int(np.searchsorted(times, left+1e-10, side='right')-1)
            if start < 0 or start >= end:
                counts['short_past'] += 1; continue
            span = slice(start, end+1)
            if (np.any(np.diff(sequence[span]) != 1) or np.any(np.diff(times[span]) > spec['max_gap'])
                    or source['segment'][frames[start]] != source['segment'][f]):
                counts['past_gap'] += 1; continue
            if source['bout'][frames[start]] != source['bout'][f]:
                counts['action_changed'] += 1; continue
            # No future visualization/readout is required; raw observation
            # coverage must nevertheless extend through the fixed horizon.
            last = min(episode_last, int(source['segment_end'][f]))
            if last >= len(source['time']) or source['time'][last]+1e-10 < stop:
                counts['future_censored'] += 1; continue
            if left <= next_allowed+1e-10:
                counts['overlapping'] += 1; continue
            incumbent = int(source['action'][f])
            ix = np.arange(start, end)
            weights = np.maximum(0., np.minimum(times[ix+1], t)-np.maximum(times[ix], left))
            if not np.isclose(weights.sum(), spec['window'], rtol=0, atol=1e-8):
                raise ValueError('Incomplete past-window integration')
            weights /= spec['window']
            active = weights > 0; ix, weights = ix[active], weights[active]
            before = _weighted(pre[ix], weights); after = _weighted(post[ix], weights)
            before -= before[incumbent]; after -= after[incumbent]
            change = after-before
            row = {k: part.iloc[end][k] for k in KEYS}
            row.update(window_id=len(rows), session=int(part.session.iloc[end]), time=float(t),
                       start_time=float(left), future_time=float(stop), source_frame=int(f),
                       start_source_frame=int(frames[start]), incumbent=incumbent, n_actions=actions,
                       n_past_decisions=len(ix), nonoverlap=True, sample_mode=spec['sample_mode'],
                       protocol_id=spec['protocol_id'])
            for a in range(actions):
                row[f'pre_{a}'] = float(before[a]); row[f'delta_{a}'] = float(change[a])
                row[f'context_incumbent_{a}'] = float(a == incumbent)
            for c in controls:
                row['context_'+c] = float(part[c].iloc[end])
            # Bout age is defined by the original stream, not by the selected
            # visualization rows (particularly important for the 15Hz view).
            age=float(t-source['time'][source['bout_start'][f]])
            row['context_log_age']=float(np.log1p(age))
            row['context_log_age_sq']=row['context_log_age']**2
            for c in MEAN_FIELDS:
                row['context_mean_'+c] = float(_weighted(part[c].to_numpy(float)[ix], weights))
            row['context_mean_log_budget'] = float(_weighted(np.log1p(part.budget.to_numpy(float)[ix]),weights))
            row['context_mean_searched'] = float(_weighted((part.budget.to_numpy()[ix]>0).astype(float),weights))
            if 'visual_change' in part:
                row['context_mean_visual_change'] = float(_weighted(part.visual_change.to_numpy(float)[ix],weights))
            for c in ('prior_reward', 'root_entropy'):
                if c in part: row['context_mean_'+c] = float(_weighted(part[c].to_numpy(float)[ix],weights))
            for a in range(actions):
                row[f'context_initial_support_{a}'] = float(pre[end,a]-pre[end,incumbent])
            lp=before-logsumexp(before)
            row['context_pre_entropy']=float(-np.sum(np.exp(lp)*lp))
            row['context_pre_margin']=float(np.sort(before)[-1]-np.sort(before)[-2])
            # This is the only label-dependent section. It cannot affect the
            # already-fixed feature vectors or the time-only landmark schedule.
            next_frame = int(source['next_frame'][f])
            switched = next_frame >= 0 and source['time'][next_frame] <= stop+1e-10
            outcome = int(source['action'][next_frame]) if switched else incumbent
            if not 0 <= outcome < actions: raise ValueError('Future action outside checkpoint action space')
            row.update(outcome=outcome, switched=bool(switched),
                       first_switch_frame=next_frame if switched else -1,
                       first_switch_time=float(source['time'][next_frame]) if switched else np.nan)
            rows.append(row); counts['retained'] += 1; next_allowed = stop
        exclusions.append(dict(subject=int(part.subject.iloc[0]),game=part.game.iloc[0],
            episode=episode,run=meta['source_id'],protocol_id=spec['protocol_id'],**counts))
    return pd.DataFrame(rows) if rows else empty(), pd.DataFrame(exclusions)


def log_probabilities(logits):
    x = np.asarray(logits, float)
    if x.ndim != 2 or not np.isfinite(x).all(): raise ValueError('Invalid forecast logits')
    return x-logsumexp(x,axis=1,keepdims=True)


def forecasts(windows):
    a = int(windows.n_actions.iloc[0])
    before = windows[[f'pre_{j}' for j in range(a)]].to_numpy(float)
    change = windows[[f'delta_{j}' for j in range(a)]].to_numpy(float)
    return before, change, {name:log_probabilities(value) for name,value in
        [('pre',before),('post',before+change),('reverse',before-change)]}


def prediction_table(windows, logp, model, held=-1):
    keys = ['subject','game','run','episode','session','window_id','time','incumbent','outcome',
            'switched','n_actions','protocol_id','sample_mode']
    out = windows[keys].copy().reset_index(drop=True)
    n, a = logp.shape
    if n != len(out) or not np.isfinite(logp).all(): raise ValueError('Invalid prediction support')
    y, inc = out.outcome.to_numpy(int), out.incumbent.to_numpy(int)
    masked = logp.copy(); masked[np.arange(n),inc] = -np.inf
    conditional = masked-logsumexp(masked,axis=1,keepdims=True)
    out['model'] = model; out['held_session'] = held
    out['log_score'] = logp[np.arange(n),y]
    best=np.isclose(logp,np.max(logp,axis=1,keepdims=True),rtol=0,atol=1e-12)
    alternative_best=np.isclose(masked,np.max(masked,axis=1,keepdims=True),rtol=0,atol=1e-12)
    # Fractional ties make accuracy invariant to arbitrary action ID numbering.
    out['accuracy'] = best[np.arange(n),y]/best.sum(axis=1)
    out['predicted_action'] = np.argmax(logp,axis=1)
    out['destination_log_score'] = np.where(out.switched,conditional[np.arange(n),y],np.nan)
    out['destination_accuracy'] = np.where(out.switched,alternative_best[np.arange(n),y]/alternative_best.sum(axis=1),np.nan)
    out['predicted_destination'] = np.argmax(masked,axis=1)
    # Outcome-only audit: conditional destination accuracy can improve even
    # when every alternative lost ground to the incumbent. Do not confuse
    # better ranking among alternatives with positive support over staying.
    delta=windows[[f'delta_{j}' for j in range(a)]].to_numpy(float)
    out['destination_relative_update']=np.where(out.switched,delta[np.arange(n),y],np.nan)
    for j in range(a):
        out[f'probability_{j}'] = np.exp(logp[:,j])
        out[f'destination_vote_{j}'] = alternative_best[:,j]/alternative_best.sum(axis=1)
    return out


def _objective(flat, x, before, magnitude, direction, y, penalty, signed):
    """Shared candidate slopes; context effects may differ across actions.

    All coefficients, including intercepts, have fixed L2 regularization so a
    class absent from a training fold still receives finite probability.
    """
    n, a = before.shape; size = x.shape[1]*a
    beta = flat[:size].reshape(x.shape[1],a)
    logits = before+x@beta+flat[size]*magnitude
    if signed: logits += flat[size+1]*direction
    lp = logits-logsumexp(logits,axis=1,keepdims=True)
    residual = np.exp(lp); residual[np.arange(n),y] -= 1
    loss = -lp[np.arange(n),y].mean()+.5*penalty*np.dot(flat,flat)
    gradients = [(x.T@residual/n).ravel(),np.array([(residual*magnitude).sum()/n])]
    if signed: gradients.append(np.array([(residual*direction).sum()/n]))
    return loss,np.concatenate(gradients)+penalty*flat


def heldout_readouts(w, before, change, spec):
    predictions, audits = [], []
    columns = sorted(c for c in w if c.startswith('context_'))
    y = w.outcome.to_numpy(int)
    for held in sorted(w.session.unique()):
        train = w.session.to_numpy()!=held; test = ~train
        shared = dict(held_session=int(held),n_train=int(train.sum()),n_test=int(test.sum()),
                      training_sessions=json.dumps(sorted(map(int,w.loc[train,'session'].unique()))),
                      train_classes=json.dumps(sorted(map(int,np.unique(y[train])))))
        reason = None
        if train.sum()<spec['minimum_train'] or test.sum()<spec['minimum_test']: reason='insufficient_rows'
        elif len(np.unique(y[train]))<2: reason='single_training_class'
        if reason:
            audits.extend(dict(shared,model=model,status=reason) for model in ('unsigned','signed')); continue
        x, names = prepare_matrix(w, columns, train)
        x = np.column_stack([np.ones(len(w)),x])
        # One positive scale for the entire action vector preserves candidate
        # meaning and the common sign; no per-action centering or rotation.
        scale = max(float(np.sqrt(np.mean(change[train]**2))),1e-8)
        direction = change/scale; magnitude = np.abs(direction)
        fold_predictions = []; fold_ok = True
        for model,signed in [('unsigned',False),('signed',True)]:
            size=x.shape[1]*before.shape[1]
            result = minimize(_objective,np.zeros(size+1+int(signed)),
                args=(x[train],before[train],magnitude[train],direction[train],y[train],spec['penalty'],signed),
                jac=True,method='L-BFGS-B',options={'maxiter':1500,'ftol':1e-12,'gtol':1e-7})
            gradient=float(np.abs(result.jac).max())
            ok=bool(result.success and np.isfinite(result.fun) and gradient<=1e-4)
            audit=dict(shared,model=model,status='ok' if ok else 'not_converged',
                iterations=int(result.nit),gradient_max=gradient,message=str(result.message),
                context_features=json.dumps(names),direction_scale=scale,
                unsigned_coefficient=float(result.x[size]/scale),
                direction_coefficient=float(result.x[size+1]/scale) if signed else np.nan)
            audits.append(audit)
            if not ok: fold_ok=False; continue
            logits=before[test]+x[test]@result.x[:size].reshape(x.shape[1],before.shape[1])+result.x[size]*magnitude[test]
            if signed: logits += result.x[size+1]*direction[test]
            fold_predictions.append(prediction_table(w.loc[test],log_probabilities(logits),model,int(held)))
        # Comparisons use exactly the same held rows; never score only the
        # successful member of a failed comparator pair.
        if fold_ok: predictions.extend(fold_predictions)
    return (pd.concat(predictions,ignore_index=True) if predictions else empty(),
            pd.DataFrame(audits) if audits else empty())


def score_tables(predictions, spec):
    from .evaluation_accumulation import block_interval, balanced_mean
    run_rows, summaries, matrices = [], [], []
    identity=['subject','game','protocol_id','sample_mode']
    for population in ('all','destination'):
        selected=predictions.loc[predictions.switched].copy() if population=='destination' else predictions.copy()
        score='destination_log_score' if population=='destination' else 'log_score'
        accuracy='destination_accuracy' if population=='destination' else 'accuracy'
        for model,reference in [('post','pre'),('reverse','pre'),('signed','unsigned')]:
            left=selected.loc[selected.model.eq(model)]; right=selected.loc[selected.model.eq(reference)]
            if left.empty or right.empty: continue
            pair=left.merge(right[['window_id',score,accuracy]],on='window_id',how='inner',validate='one_to_one',suffixes=('','_reference'))
            if len(pair)!=len(left) or len(pair)!=len(right): raise ValueError('Unmatched forecast comparison')
            pair['gain']=pair[score]-pair[score+'_reference']
            pair['accuracy_gain']=pair[accuracy]-pair[accuracy+'_reference']
            base={k:pair[k].iloc[0] for k in identity}; base.update(population=population,model=model,reference=reference)
            for (session,run),r in pair.groupby(['session','run']):
                run_rows.append(dict(base,session=int(session),run=run,n=len(r),gain=float(r.gain.mean()),
                    accuracy_gain=float(r.accuracy_gain.mean()),log_score=float(r[score].mean()),
                    reference_log_score=float(r[score+'_reference'].mean()),accuracy=float(r[accuracy].mean()),
                    reference_accuracy=float(r[accuracy+'_reference'].mean()),
                    destination_relative_update=float(r.destination_relative_update.mean()) if population=='destination' else np.nan,
                    destination_reinforced_fraction=float(r.destination_relative_update.gt(0).mean()) if population=='destination' else np.nan))
            ci=block_interval(pair,pair.gain.to_numpy(),spec['bootstrap'],spec['seed'])
            supported=pair.session.nunique()>=2 and ci['blocks']>=2 and len(pair)>=20
            if not supported: ci['ci_low']=np.nan; ci['ci_high']=np.nan
            summaries.append(dict(base,**ci,n=len(pair),sessions=pair.session.nunique(),runs=pair.run.nunique(),
                accuracy_gain=balanced_mean(pair,'accuracy_gain'),
                destination_relative_update=balanced_mean(pair,'destination_relative_update') if population=='destination' else np.nan,
                destination_reinforced_fraction=balanced_mean(pair.assign(reinforced=pair.destination_relative_update.gt(0).astype(float)),
                    'reinforced') if population=='destination' else np.nan,
                status='ok' if supported else 'insufficient_support'))
        if population=='destination':
            for model in ('pre','post'):
                part=selected.loc[selected.model.eq(model)]
                if part.empty: continue
                a=int(part.n_actions.iloc[0]); base={k:part[k].iloc[0] for k in identity}
                for (session,run),r in part.groupby(['session','run']):
                    for actual,g in r.groupby('outcome'):
                        counts=g[[f'destination_vote_{j}' for j in range(a)]].sum().to_numpy(float)
                        for predicted,count in enumerate(counts):
                            matrices.append(dict(base,session=int(session),run=run,model=model,
                                actual=int(actual),predicted=predicted,count=float(count),n=len(g),
                                probability=float(count/len(g)),n_actions=a))
    return dict(action_support_run_scores=pd.DataFrame(run_rows) if run_rows else empty(),
        action_support_scores=pd.DataFrame(summaries) if summaries else empty(),
        action_support_destination_counts=pd.DataFrame(matrices) if matrices else empty())


def analyze(args, prepared):
    """All inputs originate in the bound capture/raw source, not another theme."""
    from common.research_inputs import enrich_decisions
    spec=protocol(args,prepared.get('sample_mode','all'))
    decisions=enrich_decisions(prepared['decisions'],prepared['metadata'])
    raw=load_raw(prepared['metadata'],spec['max_gap'])
    w,coverage=make_windows(decisions,prepared['voc_tables']['voc_decisions'],prepared['metadata'],raw,spec)
    tables=dict(action_support_windows=w,action_support_coverage=coverage,
        action_support_protocol=pd.DataFrame([spec]))
    identity={k:decisions[k].iloc[0] for k in ('subject','game')}
    def stamp():
        for frame in tables.values():
            for key,value in dict(identity,protocol_id=spec['protocol_id'],sample_mode=spec['sample_mode']).items():
                frame[key]=value
    if w.empty:
        tables.update({PREFIX+k:empty() for k in ('predictions','fit_qa','run_scores','scores','destination_counts')})
        stamp()
        return tables,dict(status='INSUFFICIENT_SUPPORT',protocol=spec,windows=0)
    before,change,literal=forecasts(w)
    parts=[prediction_table(w,lp,name) for name,lp in literal.items()]
    fitted,audit=heldout_readouts(w,before,change,spec)
    if len(fitted): parts.append(fitted)
    predictions=pd.concat(parts,ignore_index=True)
    tables.update(action_support_predictions=predictions,action_support_fit_qa=audit,
                  **score_tables(predictions,spec))
    stamp()
    scores=tables['action_support_scores']
    primary=scores.loc[scores.population.eq('destination') & scores.model.eq('post')] if len(scores) else empty()
    primary_supported=bool(len(primary) and primary.status.eq('ok').all())
    fits_ok=bool(len(audit) and audit.status.eq('ok').all() and w.session.nunique()==4)
    status='COMPLETED' if primary_supported and fits_ok else 'PARTIAL_FITS' if primary_supported else 'INSUFFICIENT_SUPPORT'
    return tables,dict(status=status,protocol=spec,windows=len(w),switch_windows=int(w.switched.sum()),
        primary_supported=primary_supported,fit_status_counts=audit.status.value_counts().to_dict() if len(audit) else {},
        interpretation='Destination gain is conditional on a future switch; full-risk gain is separate. '
            'Conditional gain alone can reflect better alternative ranking without strengthening it over staying; '
            'destination_relative_update and destination_reinforced_fraction audit that distinction. '
            'A negative fitted direction coefficient is contrary to the hypothesis even if prediction improves. '
            'Endpoint revision is not independent path information or causal computation utility.')


def aggregate(tables, expected_subjects):
    """Saved-table-only, protocol-checked summaries. Never pool action IDs across games."""
    p=tables.get('action_support_protocol',empty())
    if p.empty or set(p.subject.astype(int)) != set(map(int,expected_subjects)):
        raise ValueError('Action-support group requires the new protocol for every requested subject; rerun 09')
    if p.protocol_id.nunique()!=1 or p.sample_mode.nunique()!=1 or p.game.nunique()!=1:
        raise ValueError('Mixed action-support protocols, sample modes or games')
    if p.subject.duplicated().any(): raise ValueError('Duplicate participant protocol')
    action_counts=set()
    for name,frame in tables.items():
        if not name.startswith(PREFIX) or frame.empty: continue
        if ('protocol_id' in frame and not frame.protocol_id.eq(p.protocol_id.iloc[0]).all()
                or 'sample_mode' in frame and not frame.sample_mode.eq(p.sample_mode.iloc[0]).all()):
            raise ValueError('Mixed protocol/sample values in '+name)
        if 'n_actions' in frame: action_counts.update(frame.n_actions.dropna().astype(int))
    if len(action_counts)>1: raise ValueError('Mixed native action spaces within a group game')
    result={}; scores=tables.get('action_support_run_scores',empty())
    if len(scores):
        keys=['subject','game','protocol_id','sample_mode']+SCORE_KEYS
        values=['gain','accuracy_gain','log_score','reference_log_score','accuracy','reference_accuracy']
        values += [c for c in ('destination_relative_update','destination_reinforced_fraction') if c in scores]
        session=scores.groupby(keys+['session'],dropna=False)[values].mean().reset_index()
        participant=session.groupby(keys,dropna=False)[values].mean().reset_index()
        result['action_support_participant_scores']=participant
        records=[]
        for cell,g in participant.groupby(['game','protocol_id','sample_mode']+SCORE_KEYS,dropna=False):
            row=dict(zip(['game','protocol_id','sample_mode']+SCORE_KEYS,cell))
            for value in values: row[value]=float(g[value].mean())
            sem=float(g.gain.sem()) if len(g)>1 else np.nan
            row.update(n_subjects=len(g),n_requested=len(expected_subjects),ci_low=row['gain']-1.96*sem,
                ci_high=row['gain']+1.96*sem,positive_subjects=int(g.gain.gt(0).sum()),
                status='descriptive' if len(g)==len(expected_subjects) else 'partial_subject_support',
                interval='descriptive mean +/- 1.96 SEM; not population inference')
            records.append(row)
        result['action_support_group_scores']=pd.DataFrame(records)
    counts=tables.get('action_support_destination_counts',empty())
    if len(counts):
        keys=['subject','game','model','actual','predicted','n_actions']
        session=counts.groupby(keys+['session']).probability.mean().reset_index()
        participant=session.groupby(keys).probability.mean().reset_index()
        group=participant.groupby(['game','model','actual','predicted','n_actions']).probability.agg(
            probability='mean',n_subjects='count').reset_index()
        result['action_support_participant_destinations']=participant
        result['action_support_group_destinations']=group
    return result


def render(root, tables):
    """Three figure families, from persisted tables only; no source/model access."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from common.figure_style import paper_style
    out=Path(root)/'figures'; out.mkdir(parents=True,exist_ok=True); written=[]
    def save(fig,name):
        for ext in ('png','pdf'):
            path=out/(PREFIX+name+'.'+ext); fig.savefig(path); written.append(str(path))
        plt.close(fig)
    def forest(ax, data, group, population):
        p=data.loc[data.population.eq(population)].sort_values('subject')
        for i,row in enumerate(p.itertuples()):
            ax.plot(row.gain,i,'o',color='#28799d')
            if np.isfinite(row.ci_low) and np.isfinite(row.ci_high): ax.plot([row.ci_low,row.ci_high],[i,i],color='#28799d')
        labels=[f'sub{int(s):03d}' for s in p.subject]
        g=group.loc[group.population.eq(population)] if len(group) else empty()
        if len(g):
            row=g.iloc[0]; i=len(labels); labels.append(f'Group (n={int(row.n_subjects)}/{int(row.n_requested)})')
            ax.plot(row.gain,i,'D',color='black')
            if np.isfinite(row.ci_low) and np.isfinite(row.ci_high): ax.plot([row.ci_low,row.ci_high],[i,i],color='black')
        ax.set(yticks=np.arange(len(labels)),yticklabels=labels,
               xlabel='Log-score gain: '+('switch destination' if population=='destination' else 'all future outcomes'))
        ax.axvline(0,color='gray',ls='--'); ax.invert_yaxis()
    with paper_style():
        scores=tables.get('action_support_scores',empty()); group=tables.get('action_support_group_scores',empty())
        if len(scores):
            direct=scores.loc[scores.model.eq('post')]; gg=group.loc[group.model.eq('post')] if len(group) else empty()
            if len(direct):
                fig,axs=plt.subplots(1,2,figsize=(16,6))
                for ax,pop in zip(axs,('all','destination')): forest(ax,direct,gg,pop)
                save(fig,'forecast_gain')
            increment=scores.loc[scores.model.eq('signed')]; fits=tables.get('action_support_fit_qa',empty())
            if len(increment):
                fig,axs=plt.subplots(1,3,figsize=(23,6))
                gg=group.loc[group.model.eq('signed')] if len(group) else empty()
                for ax,pop in zip(axs[:2],('all','destination')): forest(ax,increment,gg,pop)
                f=fits.loc[fits.model.eq('signed') & fits.status.eq('ok')] if len(fits) else empty()
                if len(f):
                    for subject,g in f.groupby('subject'):
                        axs[2].plot(g.held_session,g.direction_coefficient,'o-',label=f'sub{int(subject):03d}')
                    axs[2].legend()
                axs[2].axhline(0,color='gray',ls='--')
                axs[2].set(xlabel='Held session',ylabel='Shared signed coefficient',xticks=[1,2,3,4])
                save(fig,'direction_increment')
        counts=tables.get('action_support_destination_counts',empty())
        matrix=tables.get('action_support_group_destinations',empty())
        if matrix.empty and len(counts):
            keys=['model','actual','predicted','n_actions']
            session=counts.groupby(keys+['session']).probability.mean().reset_index()
            matrix=session.groupby(keys).probability.mean().reset_index()
        if len(matrix):
            fig,axs=plt.subplots(1,2,figsize=(16,7)); a=int(matrix.n_actions.iloc[0])
            for ax,model in zip(axs,('pre','post')):
                p=matrix.loc[matrix.model.eq(model)]
                grid=p.pivot(index='actual',columns='predicted',values='probability').reindex(index=range(a),columns=range(a))
                im=ax.imshow(grid.to_numpy(),vmin=0,vmax=1,cmap='Blues')
                ax.set(xlabel=('Before' if model=='pre' else 'After')+' computation: predicted destination',
                    ylabel='Observed first switch destination',xticks=range(a),yticks=range(a))
                fig.colorbar(im,ax=ax,label='Conditional fraction')
            save(fig,'destination_matrix')
    return written
