"""Only new comparisons: process, Actor soft readouts and pre-control STOP.

Actor hyperparameters and weights never use human targets. Locked human
evaluation is performed after the held-out Actor probability has been saved.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from core import risk


PROCESS = ('process_signed_mean', 'process_abs_mean', 'process_abs_total',
           'process_valid_events', 'process_available', 'process_first_mean',
           'process_first_events', 'process_first_available')
EVIDENCE = ('prior_rollout_advantage', 'prior_first_gain', 'prior_repeat_delta')


class SoftReadout:
    """Global state coefficients by action + alternative-specific content.

The same content coefficient evaluates each action's own Q/prior/visits,
matching the legacy action-long readout. Global covariates have action-specific
coefficients. Penalty and imputation are learned within training folds only.
"""
    def fit(self, x, alt, target, weights, penalty, config):
        if not np.isfinite(target).all() or np.any(target<0) or not np.allclose(target.sum(1), 1):
            raise ValueError('Invalid soft Actor targets')
        n, k = target.shape
        xa = np.column_stack((np.ones(n), x))
        p, f = xa.shape[1], alt.shape[2]
        reg = np.full((p, k), 1/(penalty*n)); reg[0] = 1e-6
        def fun(v):
            beta, gamma = v[:p*k].reshape(p, k), v[p*k:]
            scores = xa@beta + np.einsum('nkf,f->nk', alt, gamma)
            logs = scores-logsumexp(scores, axis=1, keepdims=True)
            residual = (np.exp(logs)-target)*weights[:, None]
            loss = -np.sum(weights[:, None]*target*logs) + .5*np.sum(reg*beta**2)+.5*np.sum(gamma**2)/(penalty*n)
            grad = np.r_[(xa.T@residual+reg*beta).ravel(),
                         np.einsum('nk,nkf->f', residual, alt)+gamma/(penalty*n)]
            return loss, grad
        fit = minimize(fun, np.zeros(p*k+f), jac=True, method='L-BFGS-B',
            options=dict(maxiter=config['max_iter'], ftol=config['tolerance']*.01,
                         gtol=config['tolerance'], maxls=40))
        if not fit.success:
            raise ValueError(f'Actor readout nonconvergence: {fit.message}, iterations={fit.nit}')
        self.beta, self.gamma = fit.x[:p*k].reshape(p, k), fit.x[p*k:]
        self.iterations = int(fit.nit)
        return self

    def predict(self, x, alt):
        scores = np.column_stack((np.ones(len(x)), x))@self.beta + np.einsum('nkf,f->nk', alt, self.gamma)
        return np.exp(scores-logsumexp(scores, axis=1, keepdims=True))


def alternative_raw(d, actions, level):
    matrices, names = [], []
    definitions = [('Root prior', 'root_p', 'log'), ('Initial root Q', 'root_q', ''),
                   ('Initial visits', 'root_n', 'log1p')]
    if level == 'R2':
        definitions += [('Terminal Q', 'terminal_q', ''), ('Mean rollout gain', 'rollout_mean', ''),
            ('Maximum rollout gain', 'rollout_max', ''), ('Terminal visits', 'terminal_n', 'log1p'),
            ('Accepted computations', 'event_visits', 'log1p'), ('Evaluated branch', 'evaluated', ''),
            ('First evaluation gain', 'first_return', ''), ('Endpoint re-evaluation', 'revision', ''),
            ('Process signed change', 'process_signed', ''), ('Process absolute change', 'process_absolute', ''),
            ('Process valid updates', 'process_count', 'log1p')]
    for label, stem, transform in definitions:
        x = d[[f'{stem}_{a}' for a in range(actions)]].to_numpy(float)
        if transform=='log': x = np.log(np.clip(x, 1e-8, 1))
        if transform=='log1p': x = np.log1p(np.maximum(x, 0))
        matrices.append(x); names.append(label)
    return np.stack(matrices, axis=2), names


def alternative_design(raw, train, test, names):
    finite = np.isfinite(raw)
    denom = finite[train].sum((0, 1))
    mean = np.divide(np.where(finite[train], raw[train], 0).sum((0, 1)), denom,
                     out=np.zeros(raw.shape[2]), where=denom>0)
    x = np.concatenate((np.where(finite, raw, mean), (~finite).astype(float)), axis=2)
    center, sd = x[train].mean((0, 1)), x[train].std((0, 1))
    keep = sd>1e-10
    x = (x[:, :, keep]-center[keep])/sd[keep]
    labels = names+[n+' unavailable' for n in names]
    return x[train], x[test], [v for v, k in zip(labels, keep) if k]


def actor_fit(ctx, d, pixels, level):
    M = ctx.M
    out = ctx.root/'fits'/level
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        ctx.verify_seal(out)
        return pd.read_csv(out/'predictions.csv.gz'), json.loads((out/'qa.json').read_text())
    blocks = M.feature_blocks(d, ctx.actions)
    keys = ['H', 'V', 'VL', 'P'] + (['B'] if level!='R0' else [])
    spec = M.make_spec(level, blocks, keys,
        extra=PROCESS if level=='R2' else (),
        remove=[f'root_{kind}_{a}' for kind in ('p', 'q', 'n') for a in range(ctx.actions)])
    if set(spec.columns) & {'human_action', 'switch'}:
        raise ValueError('Human outcome in Actor predictors')
    target = d[[f'actor_p_{a}' for a in range(ctx.actions)]].to_numpy(float)
    target /= target.sum(1, keepdims=True)
    eligible = d.scored.to_numpy(bool) & (d.incumbent>=0).to_numpy()
    train_data = d.drop(columns=['human_action', 'switch']).copy()
    train_data['row_id'] = np.arange(len(d)); train_data.attrs['num_actions'] = ctx.actions
    design = M.Design(train_data, pixels, ctx.config, ctx.root/'actor_transforms')
    raw, names = alternative_raw(train_data, ctx.actions, level)
    predictions, weights, folds, failures = [], [], [], []
    for label, otr, ote in M.outer_folds(train_data):
        path, qpath = out/(label+'.csv.gz'), out/(label+'.json')
        if path.exists() and qpath.exists():
            predictions.append(pd.read_csv(path)); q = json.loads(qpath.read_text())
            folds.append(q); weights.extend(q['weights']); continue
        train, test = otr[eligible[otr]], ote[eligible[ote]]
        print(f'Actor {level}: {label}, train={len(train)}, test={len(test)}', flush=True)
        try:
            scores = {c: [] for c in ctx.config['c_grid']}
            tuning_failures = []
            for it, iv in M.inner_folds(train_data, otr, ctx.config['embargo_seconds']):
                it, iv = it[eligible[it]], iv[eligible[iv]]
                if not len(it) or not len(iv): raise ValueError('Empty Actor inner fold')
                xt, xv, _ = design.prepare(it, iv, spec)
                at, av, _ = alternative_design(raw, it, iv, names)
                for penalty in scores:
                    try:
                        model = SoftReadout().fit(xt, at, target[it], M.run_weights(train_data, it), penalty, ctx.config)
                        p = model.predict(xv, av)
                        scores[penalty].append(float(M.run_weights(train_data, iv) @ np.sum(target[iv]*np.log(np.clip(p, 1e-15, 1)), 1)))
                    except ValueError as err:
                        scores[penalty].append(np.nan); tuning_failures.append(str(err))
            valid = {c: np.mean(v) for c, v in scores.items() if v and np.isfinite(v).all()}
            if not valid: raise ValueError('No Actor regularization valid in all inner folds')
            best = max(sorted(valid), key=lambda c: valid[c])
            xt, xv, xqa = design.prepare(train, test, spec)
            at, av, labels = alternative_design(raw, train, test, names)
            model = SoftReadout().fit(xt, at, target[train], M.run_weights(train_data, train), best, ctx.config)
            p = model.predict(xv, av)
            pred = train_data.loc[test, ['row_id', 'decision_id', 'run', 'session', 'episode', 'time', 'incumbent']].copy()
            pred['actor_logp'] = np.sum(target[test]*np.log(np.clip(p, 1e-15, 1)), 1)
            pred['actor_brier'] = np.sum((p-target[test])**2, 1)
            for a in range(ctx.actions): pred[f'p_{a}'] = p[:, a]
            # Lock predictions BEFORE any human outcome is accessed for evaluation.
            pred.to_csv(path, index=False)
            w = [dict(run=label, model=level, feature=n, weight=float(v)) for n, v in zip(labels, model.gamma)]
            q = dict(fold=label, C=best, inner_actor_scores=valid, tuning_failures=tuning_failures,
                training_runs=sorted(train_data.loc[train, 'run'].unique()),
                train_rows=len(train), test_rows=len(test), iterations=model.iterations, weights=w,
                human_targets_used_for_fit_or_tuning=False, **xqa)
            ctx.C.write_json(qpath, q)
            folds.append(q); weights.extend(w); predictions.append(pred)
        except ValueError as err:
            failures.append(dict(fold=label, reason=str(err)))
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=[
        'row_id','decision_id','run','session','episode','time','incumbent','actor_logp','actor_brier',
        'human_action','switch','human_logp','switch_probability','human_brier','incumbent_margin']+
        [f'p_{a}' for a in range(ctx.actions)])
    if len(pred):
        labels = d.set_index('decision_id').loc[pred.decision_id]
        p = pred[[f'p_{a}' for a in range(ctx.actions)]].to_numpy()
        human = labels.human_action.to_numpy(int); inc = labels.incumbent.to_numpy(int)
        pred['human_action'] = human; pred['switch'] = (human!=inc).astype(int)
        pred['human_logp'] = np.log(np.clip(p[np.arange(len(p)), human], 1e-15, 1))
        pred['switch_probability'] = 1-p[np.arange(len(p)), inc]
        pred['human_brier'] = np.sum((p-np.eye(ctx.actions)[human])**2, 1)
        pred['incumbent_margin'] = np.log(np.clip(p[np.arange(len(p)), inc], 1e-15, 1))-np.max(
            np.where(np.arange(ctx.actions)[None, :]!=inc[:, None], np.log(np.clip(p, 1e-15, 1)), -np.inf), 1)
    pred.to_csv(out/'predictions.csv.gz', index=False)
    pd.DataFrame(weights, columns=['run', 'model', 'feature', 'weight']).to_csv(out/'weights.csv', index=False)
    qa = dict(complete=not failures and len(pred)==int(eligible.sum()), failures=failures, folds=folds,
              rows=len(pred), expected_rows=int(eligible.sum()), target='terminal REAL-head distribution',
              tuning_target='Actor cross entropy only', validation='locked human labels; never refit',
              cv='within-heldout-session nested leave-one-block-out, 8-second inner embargo')
    ctx.C.write_json(out/'qa.json', qa)
    ctx.seal(out, complete=qa['complete'])
    return pred, qa


def fit_all(ctx):
    out = ctx.root/'fits'
    if (out/'manifest.json').exists():
        ctx.verify_seal(out); return
    out.mkdir(exist_ok=True)
    d = ctx.decisions(); pixels = ctx.pixels()
    specs, blocks = ctx.A.specs(d, ctx.actions)
    sig = ctx.C.digest(dict(v1=ctx.manifest['signature'], sources=ctx.source_hashes,
                            derived=ctx.C.sha(ctx.root/'derived/manifest.json'), config=ctx.config))
    engine = ctx.M.Engine(ctx.engine_args(), d, pixels, dict(signature=sig, num_actions=ctx.actions))
    effects, runs, qa = [], [], {}
    def add(bp, bq, ep, eq, name, family):
        summary, by_run, joined = ctx.M.compare(bp, ep, bq, eq, name, family)
        summary.update(subject=ctx.args.subject, game=ctx.args.game)
        effects.append(summary); runs.append(by_run.assign(contrast=name, family=family))
        joined.to_csv(out/(name+'.csv.gz'), index=False)
    process = ctx.M.Spec('endpoint_plus_process', specs['compute'].columns+PROCESS)
    for outcome in ('switch', 'destination'):
        bp, bq = ctx.prediction('compute', outcome)
        ep, eq = engine.fit(process, outcome)
        add(bp, bq, ep, eq, 'process_'+outcome, 'process')
        qa['process_'+outcome] = eq
    # This comparison did not exist in V1. It remains a descriptive secondary
    # destination-history decomposition, never a replacement for H2 or H3.
    bp, bq = ctx.prediction('compute', 'destination')
    ep, eq = engine.fit(specs['full_history'], 'destination')
    add(bp, bq, ep, eq, 'history_destination', 'descriptive_history')
    qa['history_destination'] = eq
    events = pd.read_csv(ctx.root/'derived/events.csv.gz')
    missing = [c for c in d if c not in events and c!='decision_id']
    events = events.merge(d[['decision_id']+missing], on='decision_id', validate='many_to_one')
    event_engine = ctx.M.Engine(ctx.engine_args(), d, pixels,
        dict(signature=sig+'-controls', num_actions=ctx.actions), events=events)
    base = ctx.M.make_spec('stop_precontrol', blocks, ['H', 'V', 'VL', 'P'],
        extra=('event_index', 'depth_pre', 'node_entropy'))
    ev = tuple(v for name in EVIDENCE for v in (name, name+'_available'))
    content = ctx.M.Spec('stop_evidence', base.columns+ev)
    interaction = ctx.M.Spec('stop_margin_interaction', content.columns+tuple('margin_x_'+name for name in EVIDENCE))
    control = {}
    for spec in (base, content, interaction):
        p, q = event_engine.fit(spec, 'stop')
        if not len(p):
            for col in ('event','p_1'):
                if col not in p:p[col]=pd.Series(dtype=float)
        p.to_csv(out/(spec.name+'.csv.gz'), index=False)
        control[spec.name] = (p, q); qa[spec.name] = q
    add(*control[base.name], *control[content.name], 'stop_evidence', 'control')
    add(*control[content.name], *control[interaction.name], 'stop_moderation', 'control')
    # Fold-specific M cutpoints for diagnostic curves: no test-fold quantiles.
    cuts = []
    for label, tr, te in ctx.M.outer_folds(events):
        vals = events.loc[tr, 'root_margin'].dropna()
        cuts.append(dict(run=label, low=vals.quantile(.33), high=vals.quantile(.67)))
    pd.DataFrame(cuts).to_csv(out/'margin_cutpoints.csv', index=False)
    actors = {name: actor_fit(ctx, d, pixels, name) for name in ('R0', 'R1', 'R2')}
    for first, last in (('R0', 'R1'), ('R1', 'R2'), ('R0', 'R2')):
        for target in ('actor', 'human'):
            bp, bq = actors[first]; ep, eq = actors[last]
            joined = bp[['decision_id', 'run', target+'_logp']].merge(
                ep[['decision_id', target+'_logp']], on='decision_id', suffixes=('_base', '_extended'), validate='one_to_one')
            joined['delta_ll'] = joined[target+'_logp_extended']-joined[target+'_logp_base']
            by_run = joined.groupby('run', as_index=False).agg(delta_ll=('delta_ll', 'mean'), rows=('decision_id', 'size'))
            valid = bq['complete'] and eq['complete'] and len(joined)==len(bp)==len(ep)
            contrast = f'{target}_{first}_{last}'
            effects.append(dict(subject=ctx.args.subject, game=ctx.args.game, contrast=contrast,
                family='actor', outcome=target+'_exact_policy', delta_ll=by_run.delta_ll.mean() if valid else np.nan,
                estimable=valid, runs=len(by_run), rows=len(joined), reason='' if valid else 'Incomplete Actor folds'))
            runs.append(by_run.assign(contrast=contrast, family='actor'))
            joined.to_csv(out/(contrast+'.csv.gz'), index=False)
    pd.DataFrame(effects).to_csv(out/'effects.csv', index=False)
    pd.concat(runs, ignore_index=True).to_csv(out/'run_effects.csv', index=False)
    ctx.C.write_json(out/'qa.json', qa)
    ctx.seal(out, complete=all(x['estimable'] for x in effects),
        primary_unchanged=True, posthoc_reanalysis=True)
