"""Panel-level analyses on the held-out dynamic replay, never old model results.

Fig2 A--C -> 01, Fig2 D--E and Fig3 C -> 02, Fig3 A--B -> 04.
The original fixed-29-event cohort becomes a variable-budget cohort. Absent
events are missing, not zero updates; invalid *observed* revisions contribute
zero to U_all, as in the shared feature contract.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.special import expit
from common.matching import nearest_visual
from common.paths import REPO, tag
from common.io import sha
from common.native import KEYS, read_columns
from common.outcomes import risk_set
from common.prediction import probabilities, scores
from . import estimators as est

POPS = ('overall', 'noop_to_active', 'active_to_other', 'active_to_noop')


def population(d, pop):
    if pop == 'active_to_other':
        r = risk_set(d, 'overall')
        return r.loc[r.incumbent_action.ne(0)].copy()
    return risk_set(d, pop)


def timing(d, cache, subject, game):
    """Use factual source time, but preserve the delivered replay decision grid."""
    d = d.copy()
    sources = {}
    for cp in range(1, 5):
        p = Path(cache)/f'checkpoint_leave{cp:02d}'/tag(subject, game)
        for name, item in json.loads((p/'manifest.json').read_text())['source_runs'].items():
            sources[item['run_label']] = (name, item['source_sha256'])
    audits = []
    for run, r in d.groupby('run_label'):
        name, expected = sources[run]; path = REPO/'behavioral_data_block'/name
        if sha(path) != expected:
            raise ValueError('Behavioral source hash mismatch: '+str(path))
        with np.load(path, allow_pickle=False) as z:
            ix = r.current_frame.to_numpy(int)
            action = z['action'].argmax(1) if z['action'].ndim == 2 else z['action']
            np.testing.assert_array_equal(action[ix], r.human_action)
            d.loc[r.index, 'time_seconds'] = z['time'][ix]
        audits.append(dict(run_label=run, source=str(path), sha256=expected,
                           replay_decisions=len(r), repeated_source_frames=int(r.current_frame.duplicated().sum()),
                           scope='delivered replay grid; no new downsampling; repeated source timestamps retained'))
    d['age_seconds'] = np.nan
    for _, r in d.sort_values(['run_label', 'real_step']).groupby(['run_label', 'segment']):
        t = r.time_seconds.to_numpy(); a = r.human_action.to_numpy()
        starts = np.r_[0, np.flatnonzero(a[1:] != a[:-1])+1]
        bout_start = np.maximum.accumulate(np.where(np.isin(np.arange(len(r)), starts), np.arange(len(r)), 0))
        age = np.r_[np.nan, t[1:]-t[bout_start[:-1]]]
        d.loc[r.index, 'age_seconds'] = age
    return d, pd.DataFrame(audits)


def bouts_from_decisions(d):
    """Censor at real episode/gap boundaries, not at the next run's first action."""
    rows = []
    d = d.sort_values(['run_label', 'real_step'])
    for (_, _), r in d.groupby(['run_label', 'segment'], sort=False):
        a = r.human_action.to_numpy(int); t = r.time_seconds.to_numpy(float)
        if len(a) < 2: continue
        if (np.diff(t) < 0).any(): raise ValueError('Decreasing source timestamps')
        if ((np.diff(t) == 0)&(a[1:] != a[:-1])).any():
            raise ValueError('Different actions assigned to one source timestamp')
        starts = np.r_[0, np.flatnonzero(a[1:] != a[:-1])+1]
        ends = np.r_[starts[1:], len(a)]
        for start, end in zip(starts, ends):
            censored = end == len(a)
            duration = t[-1]-t[start] if censored else t[end]-t[start]
            destination = -1 if censored else a[end]
            rows.append(dict(run_label=r.run_label.iloc[0], action=a[start],
                active=int(a[start] != 0), duration_seconds=duration,
                n_decisions=end-start, censored=censored,
                route='censored' if censored else ('to_noop' if destination == 0 else 'to_active'),
                destination=destination))
    return pd.DataFrame(rows)


def competing_risks(bouts, grid):
    """Aalen--Johansen survival and cumulative incidences, censoring respected."""
    survival = 1.; noop = active = 0.; states = [(0., survival, noop, active)]
    for time, r in bouts.groupby('duration_seconds', sort=True):
        at_risk = int(bouts.duration_seconds.ge(time).sum())
        dn = int(r.route.eq('to_noop').sum()); da = int(r.route.eq('to_active').sum())
        if at_risk:
            noop += survival*dn/at_risk; active += survival*da/at_risk
            survival *= 1-(dn+da)/at_risk
        states.append((float(time), survival, noop, active))
    st = np.array(states)
    ix = np.searchsorted(st[:, 0], grid, side='right')-1
    out = pd.DataFrame(dict(seconds=grid, survival=st[ix, 1], to_noop=st[ix, 2], to_active=st[ix, 3]))
    out['at_risk'] = [int(bouts.duration_seconds.ge(t).sum()) for t in grid]
    out.loc[out.at_risk.lt(5), ['survival', 'to_noop', 'to_active']] = np.nan
    return out


def behavior_panels(d):
    bouts = bouts_from_decisions(d); curves = []
    for (run, active), r in bouts.groupby(['run_label', 'active']):
        c = competing_risks(r, np.linspace(0, 5, 101))
        curves.append(c.assign(run_label=run, active=active))
    duration = []; prediction = []; qa = []
    age_edges = np.r_[0, np.geomspace(.01, 10, 16), np.inf]
    for pop in POPS:
        r = population(d, pop).dropna(subset=['age_seconds'])
        r['age_bin'] = np.searchsorted(age_edges, r.age_seconds, side='right')-1
        r['log_duration'] = np.log(r.age_seconds.clip(lower=.001))
        for (run, b), z in r.groupby(['run_label', 'age_bin']):
            duration.append(dict(run_label=run, population=pop, bin=b, age_seconds=z.age_seconds.mean(),
                                 lower_seconds=age_edges[b], upper_seconds=age_edges[b+1],
                                 probability=z.outcome.mean(), n=len(z)))
        acts = [c for c in r if c.startswith('previous_action_')]
        for c in acts: r[c+'_duration'] = r[c]*r.log_duration
        models = {'previous_action': acts,
                  'action_and_duration': acts+['log_duration']+[c+'_duration' for c in acts]}
        for cp in sorted(r.session.unique()):
            train = r.session.ne(cp).to_numpy(); test = ~train
            for model, cols in models.items():
                if not cols or not (r.loc[train, cols].std().fillna(0) > 1e-10).any():
                    rate = r.loc[train, 'outcome'].mean()
                    p = np.tile([1-rate, rate], (int(test.sum()), 1))
                    audit = dict(status='ok', estimator='training-only intercept', n_train=int(train.sum()), n_test=int(test.sum()))
                else:
                    p, audit = probabilities(r, cols, r.outcome.to_numpy(), train, test, actions=2)
                qa.append(dict(population=pop, session=cp, model=model, **audit))
                if p is None: continue
                selected = r.loc[test].reset_index(drop=True)
                for run, ix in selected.groupby('run_label').groups.items():
                    prediction.append(dict(run_label=run, population=pop, model=model,
                                           **scores(selected.loc[ix, 'outcome'], p[ix])))
    return dict(bouts=bouts, persistence_curves=pd.concat(curves, ignore_index=True),
                duration_curves=pd.DataFrame(duration), duration_prediction=pd.DataFrame(prediction),
                duration_fit_qa=pd.DataFrame(qa))


def scaled(d):
    d = d.copy()
    for metric in ['M', 'K']:
        g = d.groupby('run_label')[metric]
        d[metric+'_z'] = (d[metric]-g.transform('mean'))/g.transform('std').replace(0, np.nan)
    mu = d.groupby('run_label').U_all.transform('mean')
    d['U_relative'] = d.U_all/mu.replace(0, np.nan)
    d['K_percentile'] = d.groupby('run_label').K.rank(method='average', pct=True)
    return d


def matched_pairs(d):
    """Same strata as primary theme 02; retain identities and both levels."""
    pairs = []; means = []
    for pop in POPS:
        part = population(d, pop)
        for run, r in part.groupby('run_label'):
            r = r.copy()
            r['age_bin'] = pd.qcut(r.log_age.rank(method='first'), 5, labels=False, duplicates='drop')
            r['time_bin'] = np.floor(r.run_progress*5).clip(0, 4)
            visual = [c for c in r if c.startswith('visual_') and c != 'visual_change']
            for _, z in r.groupby(['incumbent_action', 'age_bin', 'time_bin', 'prior_reward'], dropna=False):
                sw = z.loc[z.outcome.eq(1)]; stay = z.loc[z.outcome.eq(0)]
                if not len(sw) or not len(stay): continue
                distance, ix = nearest_visual(sw[visual],stay[visual],z[visual])
                st = stay.iloc[ix[:, 0]]
                p = pd.DataFrame(dict(run_label=run, population=pop,
                    switch_id=sw.real_step_id.to_numpy(), persist_id=st.real_step_id.to_numpy(),
                    visual_distance=distance[:, 0]))
                for metric in ['M_z', 'U_relative', 'K_z', 'U_all']:
                    p[metric+'_switch'] = sw[metric].to_numpy(); p[metric+'_persist'] = st[metric].to_numpy()
                pairs.append(p)
    pairs = pd.concat(pairs, ignore_index=True)
    for (run, pop), p in pairs.groupby(['run_label', 'population']):
        for metric in ['M_z', 'U_relative', 'K_z', 'U_all']:
            means.append(dict(run_label=run, population=pop, metric=metric, n_pairs=len(p),
                persist=p[metric+'_persist'].mean(), switch=p[metric+'_switch'].mean(),
                difference=(p[metric+'_switch']-p[metric+'_persist']).mean()))
    return pairs, pd.DataFrame(means)


def mu_surface(d):
    """Equal-run descriptive kernel surfaces, no observed decisions fabricated."""
    rows = []; xedges = np.linspace(-3, 3, 41); yedges = np.linspace(0, 4, 41)
    for pop in POPS:
        r = population(d, pop)
        for run, z in r.groupby('run_label'):
            z = z.loc[np.isfinite(z.M_z)&np.isfinite(z.U_relative)]
            n, _, _ = np.histogram2d(z.M_z, z.U_relative, bins=[xedges, yedges])
            sy, _, _ = np.histogram2d(z.M_z, z.U_relative, bins=[xedges, yedges], weights=z.outcome)
            den = gaussian_filter(n, 1.3, mode='constant'); num = gaussian_filter(sy, 1.3, mode='constant')
            prob = np.divide(num, den, out=np.full_like(num, np.nan), where=den >= 5)
            ii, jj = np.indices(n.shape)
            rows.append(pd.DataFrame(dict(run_label=run, population=pop, xbin=ii.ravel(), ybin=jj.ravel(),
                M_z=((xedges[:-1]+xedges[1:])/2)[ii.ravel()],
                U_relative=((yedges[:-1]+yedges[1:])/2)[jj.ravel()], probability=prob.ravel(),
                n=n.ravel(), smoothed_count=den.ravel())))
    return pd.concat(rows, ignore_index=True)


def k_curves(d):
    """Original natural cubic K-percentile curve with context and run controls."""
    curves = []; audits = []
    grid = np.linspace(.05, .95, 91); knots = np.array([0., .25, .5, .75, 1.])
    controls = [c for c in d if c.startswith(('visual_', 'previous_action_')) and c != 'visual_change']
    for pop in ('noop_to_active', 'active_to_noop'):
        r = population(d, pop).dropna(subset=['K_percentile', 'log_age', 'prior_reward', *controls])
        if len(r) < 200 or r.outcome.value_counts().min() < 20 or r.outcome.nunique() < 2:
            audits.append(dict(population=pop, status='insufficient', n=len(r))); continue
        runs, run = np.unique(r.run_label, return_inverse=True); nr = np.bincount(run)
        weights = 1/(len(runs)*nr[run]); q = r.K_percentile.to_numpy()
        aknots = np.unique(np.quantile(r.log_age, [0, .25, .5, .75, 1]))
        age = est.basis(r.log_age.to_numpy(), aknots) if len(aknots) >= 3 else r[['log_age']].to_numpy()
        nuisance = np.column_stack([r[controls].to_numpy(), age, r.prior_reward])
        sd = nuisance.std(0); sd[sd < 1e-12] = 1.
        nuisance = (nuisance-nuisance.mean(0))/sd
        kb = est.basis(q, knots)
        x = np.column_stack([np.ones(len(r)), kb, nuisance, np.eye(len(runs))[run][:, 1:]])
        try:
            beta, info = est.fit_model(x, r.outcome.to_numpy(), weights)
            eta = beta[0]+x[:, 1+kb.shape[1]:]@beta[1+kb.shape[1]:]
            delta = est.basis(grid, knots)@beta[1:1+kb.shape[1]]
            probability = np.array([weights@expit(eta+b) for b in delta])
            lo, hi = np.quantile(q, [.01, .99])
            for t, p in zip(grid, probability):
                near = np.abs(q-t) <= .05; counts = np.bincount(run[near], minlength=len(runs))
                ok = lo <= t <= hi and near.sum() >= 200 and (counts >= 20).sum() >= 3
                curves.append(dict(population=pop, percentile=t*100, probability=p if ok else np.nan,
                                   supported=ok, n_local=int(near.sum()), n_runs=int((counts >= 20).sum())))
            audits.append(dict(population=pop, status='ok', n=len(r), **info))
        except (AssertionError, ValueError, np.linalg.LinAlgError) as error:
            audits.append(dict(population=pop, status='unestimable', n=len(r), error=str(error)))
    return pd.DataFrame(curves), pd.DataFrame(audits)


def native_search(d, cache, subject, game):
    """Same SEARCH head at root before choices, including voluntary STOP queries.

    REAL (policy_type=2) is never an endpoint. Fewer than two root SEARCH
    choices, including B=0, has no measurable same-head endpoint difference.
    """
    n = int(d.n_actions.iloc[0]); endpoints = []; updates = []; audits = []
    probs = [f'actor_action_probability_{a}' for a in range(n)]
    for cp in range(1, 5):
        folder = Path(cache)/f'checkpoint_leave{cp:02d}'/tag(subject, game)
        selected = d.loc[d.session.eq(cp)].copy()
        prior = selected[[f'model_root_p_{a}' for a in range(n)]].to_numpy().copy()
        inc = selected.incumbent_action.to_numpy(int)
        prior[np.arange(len(selected)), np.maximum(inc, 0)] = -np.inf
        selected['initial_alternative'] = prior.argmax(1)
        pe = pd.read_csv(folder/'persistence_events.csv.gz')
        pe = pe.merge(selected[KEYS+['incumbent_action', 'human_action', 'U_all', 'additional_computations', 'initial_alternative']],
                      on=KEYS, validate='many_to_one', how='inner')
        pe['update'] = pe.root_q_revision.abs().where(pe.value_revision_valid.eq(1)&np.isfinite(pe.root_q_revision), 0.)
        pe['branch_type'] = np.where(pe.branch.eq(-1), 'reset',
                                     np.where(pe.branch.eq(pe.incumbent_action), 'ongoing',
                                     np.where(pe.branch.eq(pe.initial_alternative), 'strongest_alternative', 'other_actions')))
        updates.append(pe)
        cols = KEYS+['event_order', 'policy_type', 'policy_type_valid', 'is_control_choice',
            'current_node_depth_pre', 'current_node_depth_pre_valid', 'actor_action_probability_valid']+probs
        ev = read_columns(folder/'native_events.csv.gz', cols)
        ev = ev.merge(selected[KEYS], on=KEYS, validate='many_to_one', how='inner')
        ev = ev.loc[ev.policy_type_valid.eq(1)&ev.policy_type.eq(1)&ev.is_control_choice.eq(1)&
                    ev.current_node_depth_pre_valid.eq(1)&ev.current_node_depth_pre.eq(0)&
                    ev.actor_action_probability_valid.eq(1)].sort_values(KEYS+['event_order'])
        if len(ev):
            pp = ev[probs].to_numpy()
            if not np.isfinite(pp).all() or (pp < 0).any() or not np.allclose(pp.sum(1), 1, atol=1e-5):
                raise ValueError('Invalid native SEARCH-head probabilities')
            first = ev.groupby(KEYS, sort=False).head(1).set_index(KEYS)
            last = ev.groupby(KEYS, sort=False).tail(1).set_index(KEYS)
            count = ev.groupby(KEYS, sort=False).size()
            zz = selected.set_index(KEYS).join(count.rename('root_choices'))
            zz = zz.loc[zz.root_choices.ge(2)&zz.incumbent_action.ge(0)&zz.scored.eq(1)].copy()
            if len(zz):
                prior = zz[[f'model_root_p_{a}' for a in range(n)]].to_numpy().copy()
                inc = zz.incumbent_action.to_numpy(int); ix = np.arange(len(zz))
                prior[ix, inc] = -np.inf; alt = prior.argmax(1)
                p0 = first.reindex(zz.index)[probs].to_numpy(); p1 = last.reindex(zz.index)[probs].to_numpy()
                den0 = p0[ix, inc]+p0[ix, alt]; den1 = p1[ix, inc]+p1[ix, alt]
                valid = (den0 > 1e-12)&(den1 > 1e-12)
                b = p0[ix, inc]/np.maximum(den0, 1e-12); a = p1[ix, inc]/np.maximum(den1, 1e-12)
                zz['first_preference'] = np.where(valid, b, np.nan)
                zz['last_preference'] = np.where(valid, a, np.nan)
                zz['endpoint_change'] = np.where(valid, np.abs(a-b), np.nan)
                zz['fixed_alternative'] = alt
                endpoints.append(zz.reset_index())
        audits.append(dict(checkpoint=cp, decisions=len(selected), root_search_choices=len(ev),
            zero_budget=int(selected.additional_computations.eq(0).sum()),
            source_sha256=sha(folder/'native_events.csv.gz')))
    ep = pd.concat(endpoints, ignore_index=True) if endpoints else pd.DataFrame(columns=[*d.columns, 'endpoint_change', 'root_choices'])
    return ep, pd.concat(updates, ignore_index=True), pd.DataFrame(audits)


def update_composition(d, events, pairs):
    """Paired search-index/branch contrast, conditioning on observed support.

    Within an observed index, other branches contribute zero to that branch's
    component. A stopped search has *no* subsequent event and is not padded.
    """
    source = events[['real_step_id', 'status2_position', 'branch_type', 'update', 'value_revision_valid']]
    if source.duplicated(['real_step_id', 'status2_position']).any(): raise ValueError('Duplicate imagination index')
    matched = pairs.loc[pairs.population.eq('overall'), ['run_label', 'switch_id', 'persist_id']].reset_index(drop=True)
    matched['pair'] = np.arange(len(matched))
    a = matched.merge(source, left_on='switch_id', right_on='real_step_id').drop(columns='real_step_id')
    b = source.rename(columns={'real_step_id': 'persist_id', 'update': 'persist_update',
                               'branch_type': 'persist_branch', 'value_revision_valid': 'persist_valid'})
    a = a.merge(b, on=['persist_id', 'status2_position'], validate='many_to_one')
    means = d.groupby('run_label').U_all.mean(); rows = []
    for branch in ['ongoing', 'strongest_alternative', 'other_actions', 'reset']:
        a['delta'] = (a['update']*a.branch_type.eq(branch)-a.persist_update*a.persist_branch.eq(branch))/a.run_label.map(means).replace(0, np.nan)
        z = a.groupby(['run_label', 'status2_position']).agg(value=('delta', 'mean'), n_pairs=('pair', 'size')).reset_index()
        z['branch'] = branch; rows.append(z)
    coverage = events.groupby(['run_label', 'status2_position']).agg(n_events=('update', 'size'), n_valid=('value_revision_valid', 'sum')).reset_index()
    return pd.concat(rows, ignore_index=True), coverage


def endpoint_sensitivity(ep):
    """COSYNE's unrestricted local raw-U slope with equal-run context controls."""
    cols = [c for c in ep if c.startswith(('visual_', 'previous_action_')) and c != 'visual_change']+['log_age', 'prior_reward']
    ep = ep.replace([np.inf, -np.inf], np.nan).dropna(subset=['endpoint_change', 'M', 'U_all', *cols])
    audit = []; curves = []; run_scores = []
    if not len(ep): return pd.DataFrame(), pd.DataFrame([dict(status='insufficient_root_choices', n=0)]), pd.DataFrame()
    nuisance = ep[cols].to_numpy(); sd = nuisance.std(0); sd[sd < 1e-12] = 1
    nuisance = (nuisance-nuisance.mean(0))/sd
    ep = ep.reset_index(drop=True); blocks = []
    for run, ix in ep.groupby('run_label').groups.items():
        r = ep.loc[ix]
        if len(r) < 2: continue
        x = np.column_stack([r.U_all, r.M, nuisance[ix]])
        blocks.append((run, est.moments(r.M.to_numpy(), x, r[['endpoint_change']].to_numpy()), len(r)))
    if not blocks: return pd.DataFrame(), pd.DataFrame([dict(status='insufficient_root_choices', n=len(ep))]), pd.DataFrame()
    gs, bs, mass, neff = [np.stack([b[1][i] for b in blocks]) for i in range(4)]
    ix = list(range(gs.shape[-1])); results = [est.solve(g, b, ix) for g, b in zip(gs, bs)]
    per = np.array([v[0][:, 0] for v in results]); fraction = np.array([v[1] for v in results]); rank = np.array([v[2] for v in results])
    support = (neff >= np.maximum(100, 5*rank))&(fraction >= 1e-5)&np.isfinite(per)
    pooled = est.solve(gs.sum(0), bs.sum(0), ix)[0][:, 0]
    valid = (support.sum(0) >= 3)&np.isfinite(pooled)
    w = 4*expit(est.GRID)*expit(-est.GRID); gains = []
    for test, (run, _, n) in enumerate(blocks):
        train = np.arange(len(blocks)) != test
        learned = est.solve(gs[train].sum(0), bs[train].sum(0), ix)[0][:, 0]
        ok = (support[train].sum(0) >= 3)&np.isfinite(learned)
        gain = est.gain(learned, w, mass[train].sum(0), ok)
        flat = est.gain(learned, np.ones(len(w)), mass[train].sum(0), ok)
        use = ok&support[test]; mse = mse_flat = np.nan
        if use.sum() >= 5 and np.isfinite(gain) and np.isfinite(flat):
            weights = mass[test, use]/mass[test, use].sum()
            mse = weights@((per[test, use]-gain*w[use])**2)
            mse_flat = weights@((per[test, use]-flat)**2); gains.append(gain)
        run_scores.append(dict(run_label=run, n=n, gain=gain, mse_shape=mse, mse_flat=mse_flat, supported_grid=int(use.sum())))
    gain = np.mean(gains) if gains else np.nan
    for j, m in enumerate(est.GRID):
        curves.append(dict(M=m, observed=pooled[j] if valid[j] else np.nan,
            theory=gain*w[j] if valid[j] else np.nan, n_supported_runs=int(support[:, j].sum()), supported=valid[j]))
    audit.append(dict(status='ok' if valid.any() else 'insufficient_local_support', n=len(ep),
                      n_runs=len(blocks), n_supported_grid=int(valid.sum()), minimum_root_choices=2))
    return pd.DataFrame(curves), pd.DataFrame(audit), pd.DataFrame(run_scores)


def event_locked(d):
    """Descriptive transition-aligned trajectories; no crossing episode boundaries."""
    records = []
    for run, r in d.groupby('run_label'):
        r = r.sort_values('real_step').reset_index(drop=True)
        for pop in ('noop_to_active', 'active_to_noop'):
            events = population(r, pop); anchors = events.index[events.outcome.eq(1)].to_numpy()
            for lag in range(-8, 9):
                ix = anchors+lag; keep = (ix >= 0)&(ix < len(r)); origin = anchors[keep]; ix = ix[keep]
                keep = r.segment.to_numpy()[ix] == r.segment.to_numpy()[origin]
                ix = ix[keep]; origin = origin[keep]
                for metric in ['M_z', 'U_relative', 'K_z', 'additional_computations', 'ema_initial']:
                    records.append(dict(run_label=run, population=pop, lag=lag, metric=metric,
                        value=r[metric].iloc[ix].mean(), n=len(ix),
                        mean_seconds=float(np.mean(r.time_seconds.to_numpy()[ix]-r.time_seconds.to_numpy()[origin])) if len(ix) else np.nan))
    return pd.DataFrame(records)


def allocation_curves(d, e):
    curves = []; budget = []; state = []
    v = e.loc[e.voluntary.eq(True)&np.isfinite(e.voc_ema)&np.isfinite(e.voc_soft_continue_probability)].copy()
    for run, r in v.groupby('run_label'):
        for pop, z in [('start', r.loc[r.choice_index.eq(0)]), ('continue', r.loc[r.choice_index.gt(0)])]:
            if not len(z): continue
            z = z.copy(); z['bin'] = np.minimum((z.voc_soft_continue_probability*10).astype(int), 9)
            for b, t in z.groupby('bin'):
                curves.append(dict(run_label=run, population=pop, bin=b, predicted=t.voc_soft_continue_probability.mean(),
                    actual=t['continue'].mean(), voc=t.voc_ema.mean(), n=len(t)))
    maximum = int(d.additional_computations.max())
    for run, r in d.groupby('run_label'):
        counts = r.additional_computations.value_counts()
        for b in range(maximum+1): budget.append(dict(run_label=run, budget=b, probability=counts.get(b, 0)/len(r)))
        for metric in ['root_value', 'visual_change', 'M']:
            z = r.loc[np.isfinite(r[metric])].copy()
            z['bin'] = np.minimum((z[metric].rank(method='average', pct=True)*10).astype(int), 9)
            for b, t in z.groupby('bin'):
                state.append(dict(run_label=run, metric=metric, bin=b, state=t[metric].mean(),
                                  budget=t.additional_computations.mean(), voc=t.ema_initial.mean(), n=len(t)))
    return pd.DataFrame(curves), pd.DataFrame(budget), pd.DataFrame(state)
