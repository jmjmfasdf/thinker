"""Build auditable panel CSVs; rendering consumes these without any fitting."""
from __future__ import annotations

import json
import shutil
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from core import (FIGURES, CAUSES, CAUSE_LABELS, BUDGET_EDGES, BUDGET_LABELS,
                  MARGIN_EDGES, AGE_EDGES, PRIMARY, risk)


class Panels:
    def __init__(self, ctx):
        self.ctx = ctx
        self.root = ctx.root/'panels'
        self.root.mkdir(exist_ok=True)

    def save(self, fig, panel, data, title, kind='line', xlabel='', ylabel='', test='', note='', **kwargs):
        data = data.copy()
        if 'unit' not in data and 'run' in data: data['unit'] = data.run
        if 'unit' not in data: data['unit'] = self.ctx.tag
        if 'n' not in data: data['n'] = 0
        if 'y' in data and kind not in ('effect', 'paired', 'weights', 'primary'):
            data.loc[data.n < 20, 'y'] = np.nan
        data['subject'] = self.ctx.args.subject
        data['game'] = self.ctx.args.game
        name = fig+'_'+panel
        data.to_csv(self.root/(name+'.csv'), index=False)
        meta = dict(figure=fig, panel=panel, title=title, kind=kind, xlabel=xlabel, ylabel=ylabel,
            test_id=test, note=note, minimum_observations_per_bin=20,
            unit='source run (subject figure); subject (group figure)',
            missing='Unavailable evidence remains NaN; not numerical zero',
            exclusions='native score_mask; missing incumbent; finite panel variables; N<20 bins masked',
            units=dict(x=xlabel, y=ylabel), raw_table=name+'.csv', **kwargs)
        self.ctx.C.write_json(self.root/(name+'.json'), meta)

    def pending(self, fig, panel, title, reason):
        self.save(fig, panel, pd.DataFrame({'reason':[reason]}), title, 'pending', note=reason)


def bins(data, feature, edges, outcome, series='All', extra=()):
    q = data.copy()
    q['_bin'] = pd.cut(q[feature], edges, labels=False, include_lowest=True)
    group = ['run', '_bin']+list(extra)
    a = q.groupby(group, observed=True).agg(y=(outcome, 'mean'), n=(outcome, 'count'),
        x=(feature, 'mean')).reset_index()
    a['series'] = series
    return a


def unit_scores(ctx, names, outcome='switch'):
    frames = []
    for name, label in names:
        p, qa = ctx.prediction(name, outcome)
        z = p.groupby('run', as_index=False).agg(y=('logp', 'mean'), n=('logp', 'size'))
        z['series'] = label; z['x'] = label
        z['complete'] = qa['complete']
        if not qa['complete']: z['y'] = np.nan
        frames.append(z)
    return pd.concat(frames, ignore_index=True)


def bout_table(d):
    records = []
    for ep, rows in d.groupby('episode', sort=False):
        rows = rows.sort_values('time')
        action = rows.human_action.to_numpy(int); times = rows.time.to_numpy()
        starts = np.r_[0, np.flatnonzero(action[1:]!=action[:-1])+1]
        ends = np.r_[starts[1:], len(rows)]
        for st, en in zip(starts, ends):
            records.append(dict(run=rows.run.iloc[0], episode=ep, action=action[st],
                start=times[st], duration=(times[en] if en<len(times) else times[-1]+1/15)-times[st],
                event=int(en<len(times)), left_unknown=int(st==0)))
    return pd.DataFrame(records)


def km_curves(bouts):
    # Timestamp subtraction produces tiny float differences for tied durations.
    # Group ties ONCE; tolerance-based counting at each raw unique float would
    # apply the same failures repeatedly and artificially collapse survival.
    bouts = bouts.copy()
    bouts['duration'] = bouts.duration.round(7)
    results = []
    for run, q in bouts.groupby('run'):
        survival = 1.
        for age in np.sort(q.duration.unique()):
            risk_n = int((q.duration>=age).sum())
            events = int(((q.duration==age)&(q.event==1)).sum())
            survival *= 1-events/risk_n
            results.append(dict(run=run, x=age, y=survival, n=risk_n, events=events, series='Observed-onset bouts'))
    # Common elapsed-time grid, right-continuous Kaplan-Meier step function.
    frames = []
    for run, q in pd.DataFrame(results).groupby('run') if results else []:
        times = np.array([0, .125, .25, .5, 1, 2, 4, 8, 16, 32, 64])
        src = np.searchsorted(q.x, times, side='right')-1
        for x, j in zip(times, src):
            n = int((bouts[bouts.run==run].duration>=x).sum())
            frames.append(dict(run=run, x=x, y=q.iloc[j].y if j>=0 else 1., n=n, series='Survival'))
    return pd.DataFrame(frames, columns=['run', 'x', 'y', 'n', 'series'])


def figure02(ctx, store, d):
    q = risk(d)
    q['mb'] = pd.cut(q.root_margin, MARGIN_EDGES, labels=False, include_lowest=True)
    q['bb'] = pd.cut(q.budget, BUDGET_EDGES, labels=False, include_lowest=True)
    q['one'] = 1.
    actor = q[[f'actor_p_{a}' for a in range(ctx.actions)]].to_numpy()
    incumbent = q.incumbent.to_numpy(int)
    q['actor_margin'] = np.log(np.clip(actor[np.arange(len(q)), incumbent], 1e-15, 1))-np.max(
        np.where(np.arange(ctx.actions)[None, :]!=incumbent[:, None], np.log(np.clip(actor, 1e-15, 1)), -np.inf), 1)
    for panel, subset, variable, title in (
        ('A', q, 'one', 'Observed decisions'),
        ('B', q[q.incumbent==0], 'switch', 'NOOP incumbent: switch probability'),
        ('C', q[q.incumbent!=0], 'switch', 'Active incumbent: any switch'),
        ('D', q[q.incumbent==0], 'actor_margin', 'NOOP: terminal REAL-head support'),
        ('E', q[q.incumbent!=0], 'actor_margin', 'Active: terminal REAL-head support')):
        a = subset.groupby(['run', 'mb', 'bb']).agg(y=(variable, 'mean'), n=(variable, 'count')).reset_index()
        if panel=='A': a['y'] = a.n/a.run.map(q.run.value_counts())*100
        a = a.rename(columns={'mb':'x', 'bb':'row'})
        store.save('02-F1', panel, a, title, 'geometry', 'Initial root margin (log probability)',
            'Additional computations', note='Selected budget; descriptive, not causal. Cells N<20 masked.',
            color_label='Decision frequency (%)' if panel=='A' else ('Switch probability' if panel in 'BC' else 'Terminal incumbent margin'),
            x_edges=MARGIN_EDGES.tolist(), row_labels=BUDGET_LABELS,
            outside_margin_support=int(q.mb.isna().sum()))
    inc_obs = q.incumbent_change.notna(); alt_obs = q.alternative_change.notna()
    q['availability'] = np.select([q.budget==0, inc_obs&~alt_obs, ~inc_obs&alt_obs, inc_obs&alt_obs],
        ['B = 0', 'Incumbent only', 'Alternatives only', 'Both'], default='No valid evidence')
    categories=['B = 0','Incumbent only','Alternatives only','Both','No valid evidence']
    a = q.groupby(['run', 'availability']).size().reindex(
        pd.MultiIndex.from_product([q.run.unique(),categories],names=['run','availability']),fill_value=0).rename('category_count').reset_index()
    a['n']=a.run.map(q.run.value_counts())
    a['y'] = a.category_count/a.n; a['x'] = a.availability; a['series'] = 'Availability'
    store.save('02-F1', 'F', a, 'Computation / evidence coverage', 'bar', ylabel='Fraction of decisions')
    # Transition-aligned trajectories: no samples before the incumbent bout onset.
    transitions = []
    for ep, rows in q.groupby('episode', sort=False):
        times = rows.time.to_numpy()
        for event in rows[rows.switch==1].itertuples():
            lo = max(event.time-4, event.time-event.bout_age)
            ix = np.flatnonzero((times>=lo-1e-8)&(times<=event.time+1e-8))
            part = rows.iloc[ix].copy()
            part['x'] = np.round((part.time-event.time)*15)/15
            part['bout_id'] = event.decision_id
            part['cause'] = CAUSES[0] if event.incumbent==0 else CAUSES[1] if event.human_action==0 else CAUSES[2]
            transitions.append(part[['run', 'x', 'bout_id', 'cause', 'root_margin', 'process_first_mean', 'process_signed_mean', 'budget']])
    t = pd.concat(transitions, ignore_index=True)
    for col, cause in enumerate(CAUSES):
        for row, specs in enumerate(( [('root_margin', 'Root margin')],
                                     [('process_first_mean', 'First evaluation'), ('process_signed_mean', 'Re-evaluation')],
                                     [('budget', 'Computations')] )):
            tables = []
            for variable, label in specs:
                z = t[t.cause==cause].groupby(['run', 'x']).agg(y=(variable, 'mean'), n=(variable, 'count'), bouts=('bout_id', 'nunique')).reset_index()
                z['series'] = label; tables.append(z)
            store.save('02-F2', chr(65+row*3+col), pd.concat(tables), CAUSE_LABELS[col],
                xlabel='Time before switch (s)', ylabel=('Root margin', 'Evidence (Q units)', 'Computations')[row],
                note='Switch-ending bouts only; descriptive. Never crosses incumbent bout onset.')
    v1runs = pd.concat([ctx.table(m, 'run_effects.csv').assign(module=m) for m in ('01', '02', '03', '05')])
    new = pd.read_csv(ctx.root/'fits/run_effects.csv')
    for panel, outcome in [('A', 'switch'), ('B', 'destination')]:
        frames = []
        names = [(f'amount_{outcome}', 'Quantity', v1runs), (f'content_given_amount_{outcome}', 'Endpoint content', v1runs),
                 (f'process_{outcome}', 'Process after endpoint', new),
                 ('all_earlier_computation_increment' if outcome=='switch' else 'history_destination', 'Earlier computation', v1runs if outcome=='switch' else new)]
        for contrast, label, source in names:
            z = source[source.contrast==contrast].copy(); z['series'] = label
            z['y'] = z.delta_ll; z['x'] = label; z['row'] = z.run; z['n'] = z.rows
            frames.append(z)
        store.save('02-F3', panel, pd.concat(frames), 'Switch gain' if outcome=='switch' else 'Destination gain', 'tile',
            ylabel='Source run', color_label='Held-out ΔLL (nats)', test='process (new); existing contrasts retained')
    primary = pd.read_csv(ctx.root/'provenance/primary_subject.csv')
    primary_group_path = ctx.args.v1_root/'08_group_report/primary_max_t.csv'
    primary_group = pd.read_csv(primary_group_path) if primary_group_path.exists() else pd.DataFrame()
    frames = []
    for h, name in PRIMARY.items():
        z = v1runs[v1runs.contrast==name].copy(); z['series'] = h; z['x'] = h; z['y'] = z.delta_ll; z['n'] = z.rows
        frames.append(z)
    store.save('02-F3', 'C', pd.concat(frames), 'Unchanged primary hypotheses', 'primary', ylabel='Held-out ΔLL (nats)',
        test='V1 primary 6-comparison max-|t|', primary_subject=primary.to_dict('records'),
        primary_group=primary_group[primary_group.game==ctx.args.game].to_dict('records') if len(primary_group) else [])
    actor_rows = []
    for name, label in [('R0', 'Pre-search readout'), ('R2', 'Search-informed readout')]:
        p = pd.read_csv(ctx.root/f'fits/{name}/predictions.csv.gz')
        actor_rows.append(p[['decision_id', 'incumbent_margin']].assign(series=label).rename(columns={'incumbent_margin':'value'}))
    actor_rows.append(q[['decision_id', 'actor_margin']].assign(series='Observed terminal Actor').rename(columns={'actor_margin':'value'}))
    data = pd.concat(actor_rows).merge(q[['decision_id','run','incumbent','switch']], on='decision_id')
    for panel, flag in [('A', True), ('B', False)]:
        z = data[(data.incumbent==0)==flag].groupby(['run','switch','series']).agg(y=('value','mean'), n=('value','count')).reset_index()
        z['x'] = z.switch.map({0:'Stay',1:'Switch'})
        store.save('02-F4', panel, z, 'NOOP incumbent' if flag else 'Active incumbent', 'paired',
            ylabel='Terminal incumbent margin (log probability)', note='Descriptive outcome stratification; no pre-search REAL-head output is assumed.')
    cause_scores = ctx.table('02','competing_transition_scores.csv')
    for i, cause in enumerate(CAUSES):
        z = q[q.incumbent==0].copy() if i==0 else q[q.incumbent!=0].copy()
        z['cause_event'] = (z.human_action!=0) if i==0 else (z.human_action==0) if i==1 else ((z.human_action!=0)&(z.human_action!=z.incumbent))
        b = bins(z, 'bout_age', AGE_EDGES, 'cause_event')
        store.save('02-F5', chr(65+i), b, CAUSE_LABELS[i], xlabel='Elapsed bout age (s)', ylabel='Cause-specific one-step hazard',
            note='All at-risk rows, including competing transitions as non-events.')
        b = cause_scores[cause_scores.cause==cause].copy(); b['y'] = b.delta_ll; b['x'] = 'Current computation'; b['n'] = b.rows
        store.save('02-F5', chr(68+i), b, 'Current computation: held-out gain', 'effect', ylabel='ΔLL (nats)', test='V1 competing transitions')
    for row, noop in enumerate((True, False)):
        z = q[(q.incumbent==0)==noop].copy()
        z['uncertainty'] = z.root_entropy/np.log(ctx.actions)
        for col, variable in enumerate(('begin','budget','max_depth')):
            v = z if col==0 else z[z.budget>0]
            b = pd.concat([bins(v, 'root_margin', MARGIN_EDGES, variable, 'Root margin').assign(axis='margin'),
                           bins(v, 'uncertainty', np.linspace(0,1,9), variable, 'Root uncertainty').assign(axis='uncertainty')])
            store.save('02-F6', chr(65+row*3+col), b, ('NOOP' if noop else 'Active')+' · '+('Initiation','Positive budget','Maximum depth')[col],
                'dual', ylabel=('P(additional computation)','Computations | B > 0','Depth | B > 0')[col],
                note='Initial-state conditional descriptions, not causal effects.',
                budget_depth_equal_fraction=float((v.budget==v.max_depth).mean()))
    bouts = bout_table(d)
    for row, noop in enumerate((True,False)):
        b = bouts[(bouts.action==0)==noop]
        label = 'NOOP' if noop else 'Active'
        store.save('02-F7', chr(65+row*4), km_curves(b[b.left_unknown==0]), label+' bout survival',
            xlabel='Elapsed bout age (s)', ylabel='Survival probability', note='Known onset only; episode-end censoring retained.', excluded_left_boundary=int(b.left_unknown.sum()))
        z = q[(q.incumbent==0)==noop]
        for col, variable, title in [(1,'history_cumulative_budget','Past computations'),(2,'history_cumulative','Past signed evidence')]:
            store.save('02-F7',chr(65+row*4+col), bins(z,'bout_age',AGE_EDGES,variable), label+' · '+title,
                xlabel='Elapsed bout age (s)',ylabel='Cumulative count' if col==1 else 'Cumulative evidence (Q units)',
                note='History strictly precedes current decision; left-boundary indicator retained.')
        baseline,_ = ctx.prediction('compute'); full,_ = ctx.prediction('full_history')
        j = baseline[['decision_id','logp']].merge(full[['decision_id','logp']],on='decision_id',suffixes=('_base','_full')).merge(z[['decision_id','run','bout_age']],on='decision_id')
        j['gain'] = j.logp_full-j.logp_base
        store.save('02-F7', chr(68+row*4), bins(j,'bout_age',AGE_EDGES,'gain'),label+' · history prediction gain',
            xlabel='Elapsed bout age (s)',ylabel='OOF ΔLL (nats)', test='H3 descriptive state decomposition')
    sub = q[q.process_available==1].copy()
    sub['revision'] = sub.evidence_signed
    revision_edges=np.array([-1e3,-10,-2,-.5,-.1,-.02,-.005,0,.005,.02,.1,.5,2,10,1e3])
    for i, noop in enumerate((True,False)):
        z=sub[(sub.incumbent==0)==noop]
        if len(revision_edges)<2:
            store.pending('02-S1',chr(65+i),'Revision geometry','Re-evaluation support is constant / absent');continue
        z=z.copy();z['row']=pd.cut(z.revision,revision_edges,labels=False,include_lowest=True)
        b=z.groupby(['run','mb','row']).agg(y=('switch','mean'),n=('switch','count')).reset_index().rename(columns={'mb':'x'})
        store.save('02-S1',chr(65+i),b,
            'NOOP: valid re-evaluation' if noop else 'Active: valid re-evaluation', 'revision_geometry',xlabel='Initial root margin',ylabel='Endpoint net evidence (Q units)',
            x_edges=MARGIN_EDGES.tolist(), y_edges=revision_edges.tolist(), color_label='Switch probability',
            note='Valid repeated-update subset; endpoint net evidence, not mean process ΔQ.',fraction_of_all=float(len(z)/len(q)))


def figure03(ctx, store, d, events):
    q = risk(d); e = events[events.decision_id.isin(q.decision_id)].copy()
    z=q.copy(); z['uncertainty']=z.root_entropy/np.log(ctx.actions)
    a=pd.concat([bins(z,'root_margin',MARGIN_EDGES,'begin','Root margin').assign(axis='margin'),
                 bins(z,'uncertainty',np.linspace(0,1,9),'begin','Root uncertainty').assign(axis='uncertainty')])
    store.save('03-F1','A',a,'Starting additional computation','dual',ylabel='P(B > 0)')
    z['bb']=pd.cut(z.budget,BUDGET_EDGES,labels=False)
    b=z.groupby(['run','bb']).size().rename('n').reset_index(); b['y']=b.n/b.run.map(z.run.value_counts())
    b['x']=b.bb.map(dict(enumerate(BUDGET_LABELS))); b['series']='Budget distribution'
    b['positive_conditional_fraction']=np.where(b.bb>0,b.n/b.run.map(z[z.budget>0].run.value_counts()),np.nan)
    store.save('03-F1','B',b,'Root-only and positive budgets','bar',ylabel='Fraction of all decisions',
        note='B=0 retained as its own category; conditional positive-budget fractions also saved.')
    hazard=e.groupby(['run','event_index']).agg(y=('stop_choice','mean'),n=('stop_choice','size'),
        stops=('stop_choice','sum'), forced=('forced_terminal','sum'), resets=('control',lambda x:int((x==1).sum()))).reset_index().rename(columns={'event_index':'x'})
    hazard['series']='Voluntary STOP'
    surv=[]
    for run, rows in hazard.groupby('run'):
        p=1.
        for r in rows.sort_values('x').itertuples():
            surv.append(dict(run=run,x=r.x,y=p,n=r.n,series='Still computing'))
            p*=1-r.y
    store.save('03-F1','C',pd.DataFrame(surv),'Computation survival',xlabel='Completed computations j',ylabel='Probability still computing',
        note='Forced termination is censoring after accepted final computation.')
    store.save('03-F1','D',hazard,'Voluntary STOP; forced caps and RESET separate',xlabel='Completed computations j',ylabel='STOP hazard',
        note='Includes the actual last STOP and j=0. Forced CONTINUE is not relabeled STOP.',markers='forced, resets')
    weights=pd.read_csv(ctx.root/'fits/R2/weights.csv')
    weights['x']=weights.feature;weights['y']=weights.weight;weights['n']=1;weights['series']='R2'
    store.save('03-F2','A',weights,'Action-specific content readout','weights',ylabel='Standardized descriptive readout weight',
        note='Training-standardized weights; collinearity precludes causal attribution.')
    scores=[]; exact=[]
    for name in ('R0','R1','R2'):
        p=pd.read_csv(ctx.root/f'fits/{name}/predictions.csv.gz')
        for variable,target in [('actor_logp',scores),('human_logp',exact)]:
            b=p.groupby('run').agg(y=(variable,'mean'),n=(variable,'size')).reset_index();b['x']=name;b['series']='Readout'
            if variable=='actor_logp':b['y']=-b.y
            target.append(b)
    store.save('03-F2','B',pd.concat(scores),'Predicting the held-out Actor policy','paired',ylabel='Soft-policy cross entropy (nats; lower is better)',test='Actor family')
    store.save('03-F3','A',pd.concat(exact),'Locked readouts evaluated on human actions','paired',ylabel='Human exact-action log probability (nats)',test='Actor family')
    for panel,target in [('B','switch'),('C','destination')]:
        tables=[]
        for name in ('R0','R1','R2'):
            p=pd.read_csv(ctx.root/f'fits/{name}/predictions.csv.gz')
            if target=='switch':
                v=p[['run','switch_probability','switch']].rename(columns={'switch_probability':'prob','switch':'outcome'})
            else:
                p=p[p.switch==1].copy();arrays=[]
                for a in range(ctx.actions):
                    z=p[p.incumbent!=a].copy();z['prob']=z[f'p_{a}']/z.switch_probability.clip(lower=1e-15);z['outcome']=(z.human_action==a).astype(int)
                    arrays.append(z[['run','prob','outcome']])
                v=pd.concat(arrays)
            b=bins(v,'prob',np.linspace(0,1,11),'outcome',name);tables.append(b)
        store.save('03-F3',panel,pd.concat(tables),'Locked '+target+' calibration','calibration',xlabel='Predicted probability',ylabel='Observed frequency',
            note='Diagnostic only; no human refitting. Conditional alternatives pooled for destination.')
    gains=pd.read_csv(ctx.root/'fits/run_effects.csv');gains=gains[gains.contrast.str.startswith('human_R')].copy()
    gains['x']=gains.contrast.str.replace('human_','',regex=False).str.replace('_',' → ',regex=False);gains['y']=gains.delta_ll;gains['n']=gains.rows
    store.save('03-F3','D',gains,'Locked exact-action improvement','effect',ylabel='ΔLL (nats)',test='Actor 12-comparison max-|t|')
    qc=[]
    for label,mask in [('First evaluation',e.first_count>0),('Repeated update',e.repeat_count>0),('RESET',e.control==1)]:
        z=e[mask].groupby('run').agg(y=('q_update_error','mean'),n=('q_update_error','count')).reset_index();z['x']=label;z['series']='Q arithmetic residual';qc.append(z)
    store.save('03-S1','A',pd.concat(qc),'Native visit-count update arithmetic','effect',ylabel='Observed Q − incremental-mean prediction',
        note='QC only; one-count increments. First evaluations, repeated visits and accepted RESET separate.')
    qc=e.groupby(['run','control']).agg(y=('q_update_error',lambda s:np.nanmax(abs(s)) if s.notna().any() else np.nan),n=('q_update_error','count')).reset_index()
    qc['x']=qc.control.map({0:'CONTINUE',1:'RESET',2:'STOP'});qc['series']='Maximum absolute discrepancy'
    store.save('03-S1','B',qc,'Update QC by accepted control','effect',ylabel='Maximum absolute arithmetic residual',note='STOP has no Q update; unavailable is not zero.')


def evidence_bins(e, predictor, value='residual'):
    finite=e[predictor].dropna()
    if finite.nunique()<2:return pd.DataFrame(columns=['run','x','y','n','series'])
    # Identical real-unit edges across subjects/games. Never average different
    # subjects' quantile-bin indices as if they were the same evidence value.
    edges=np.array([-1e6,-10,-2,-.5,-.1,-.02,-.005,0,.005,.02,.1,.5,2,10,1e6])
    return bins(e,predictor,edges,value)


def figure04(ctx,store,d,e):
    eligible=risk(d)
    q=e[e.decision_id.isin(eligible.decision_id)&(e.event_index>0)].copy()
    p=pd.read_csv(ctx.root/'fits/stop_precontrol.csv.gz')
    q=q.merge(p[['decision_id','event','p_1','target']],on=['decision_id','event'],validate='one_to_one')
    q['residual']=q.target-q.p_1
    store.save('04-F1','A',evidence_bins(q,'prior_rollout_advantage'),'Last rollout → next voluntary STOP',
        xlabel='Previous rollout advantage (return units)',ylabel='Observed − baseline STOP probability',note='Baseline-adjusted OOF residual; pre-control evidence only.')
    rows=[]
    for name,label in [('prior_first_gain','First evaluation'),('prior_repeat_delta','Repeated Q update')]:
        b=evidence_bins(q,name);b['series']=label;rows.append(b)
    store.save('04-F1','B',pd.concat(rows),'First versus repeated evidence',xlabel='Previous evidence (Q units)',ylabel='OOF STOP residual')
    store.save('04-F1','C',evidence_bins(q[q.control!=2],'prior_rollout_advantage','future_depth'),
        'Subsequent depth among accepted CONTINUE / RESET',xlabel='Previous rollout advantage',ylabel='Subsequent maximum depth',
        note='Selected subset descriptive outcome, never a STOP predictor.')
    effects=pd.read_csv(ctx.root/'fits/run_effects.csv')
    a=effects[effects.contrast=='stop_evidence'].copy();a['x']='Evidence block';a['y']=a.delta_ll;a['n']=a.rows
    store.save('04-F1','D',a,'Evidence improves held-out STOP prediction?','effect',ylabel='ΔLL (nats)',test='Control 4-comparison max-|t|')
    cuts=pd.read_csv(ctx.root/'fits/margin_cutpoints.csv')
    q=q.merge(cuts,on='run');q['margin_group']=np.select([q.root_margin<=q.low,q.root_margin>=q.high],['Low initial margin','High initial margin'],default='Middle')
    for i,noop in enumerate((True,False)):
        for j,(var,label) in enumerate([('prior_rollout_advantage','Rollout advantage'),('prior_first_gain','First evaluation gain'),('prior_repeat_delta','Repeated Q update')]):
            rows=[]
            for band in ('Low initial margin','High initial margin'):
                b=evidence_bins(q[((q.incumbent==0)==noop)&(q.margin_group==band)],var);b['series']=band;rows.append(b)
            store.save('04-F1b',chr(65+i*3+j),pd.concat(rows),('NOOP' if noop else 'Active')+' · '+label,
                xlabel=label+' (Q units)',ylabel='OOF STOP residual',test='Control: margin × evidence block',
                note='Margin cutpoints from each outer training fold; do not compare separate curve significance.')
    # Supplement: each measure must exist in BOTH halves of a B>=2 window.
    comp=e[e.computation.astype(bool)&e.decision_id.isin(eligible.decision_id)].copy()
    info=eligible.set_index('decision_id')
    comp['B']=comp.decision_id.map(info.budget);comp=comp[comp.B>=2]
    comp['half']=np.where(comp.event_index < comp.B//2,'Early','Late')
    for i,noop in enumerate((True,False)):
        for j,(var,label) in enumerate([('repeat_abs','Mean absolute repeated ΔQ'),('depth_post','Reached depth')]):
            selected=comp[(comp.incumbent==0)==noop]
            z=selected.groupby(['decision_id','run','half'])[var].mean().unstack('half').dropna(subset=['Early','Late']).reset_index()
            b=z.melt(id_vars=['decision_id','run'],value_vars=['Early','Late'],var_name='x',value_name='value').groupby(['run','x']).agg(y=('value','mean'),n=('value','size')).reset_index()
            b['series']=label
            store.save('04-F2',chr(65+i*2+j),b,('NOOP' if noop else 'Active')+' · '+label,'paired',ylabel='Q units' if j==0 else 'Depth',
                note='Supplement only; selected long-search subset. Depth growth can be mechanical.',included_decisions=len(z),all_decisions=len(eligible),fraction_of_all=len(z)/len(eligible))
    steps=[('baseline','Baseline'),('amount','+ Quantity'),('compute','+ Endpoint'),('actor','+ Actor')]
    for panel,outcome in [('A','switch'),('B','destination')]:
        store.save('04-F3',panel,unit_scores(ctx,steps,outcome),'Human '+outcome+': nested models','paired',ylabel='Held-out log probability (nats)',test='V1 comparisons unchanged')
    store.save('04-F3','C',unit_scores(ctx,[('compute','Current computation'),('full_history','+ Earlier history')]),
        'Separate extension: earlier computation history','paired',ylabel='Held-out switch log probability (nats)',test='Unchanged H3',note='Parallel extension, not after terminal Actor.')
    exact=ctx.table('02','factorized_exact_action.csv');exact['x']=exact.model.map(dict(steps));exact['y']=exact.logp_exact;exact['n']=exact.run.map(eligible.run.value_counts());exact['series']='Factorized exact action'
    store.save('04-F3','D',exact,'Coherent stay × conditional-destination likelihood','paired',ylabel='Exact-action log probability (nats)',test='V1 descriptive exact-action')


def new_figures(ctx,store,d):
    runs=ctx.table('01','run_effects.csv')
    timeline=pd.DataFrame(dict(x=[0,1,2,3], observation=['x(t)','x(t+1)','x(t+2)','x(t+3)'],action=['a(t)','a(t+1)','a(t+2)','a(t+3)'],n=[1]*4))
    store.save('N1','A',timeline,'Separate real time and imagination order','timeline',note='All search events belong to their real decision; no invented millisecond spacing.')
    for panel in ('B','D'):
        rows=[]
        for kind in ('visual','model'):
            for lag,(lo,hi) in enumerate(ctx.C.LAGS):
                z=runs[runs.contrast==f'{kind}_lag{lag}'].copy();z['series']='Visual history' if kind=='visual' else 'Model history';z['x']=(lo+hi)/2;z['low']=lo;z['high']=hi;z['y']=z.delta_ll;z['n']=z.rows;z['row']=z.series
                z['coverage']=float((d[f'lag{lag}_coverage']>0).mean());rows.append(z)
        store.save('N1',panel,pd.concat(rows),'Lag-band removal: '+('gain and coverage' if panel=='B' else 'real-time profile'),
            'tile' if panel=='B' else 'line',xlabel='Past lag (s)',ylabel='Held-out ΔLL (nats)',color_label='ΔLL (nats)',test='V1 information-timescales family',note='Not a precise human reaction latency.')
    frames=[]
    for name,label in [('past_model_before_lagged_visual','Before past-visual control'),('past_model_after_lagged_visual','After past-visual control')]:
        z=runs[runs.contrast==name].copy();z['x']=label;z['y']=z.delta_ll;z['n']=z.rows;z['series']='Earlier model information';frames.append(z)
    store.save('N1','C',pd.concat(frames),'Past model information after stimulus-history control','paired',ylabel='ΔLL (nats)',test='V1 information-timescales family')
    examples=json.loads((ctx.root/'derived/path_examples.json').read_text())
    store.save('N2','A',pd.DataFrame(examples),'Imagined and realized paths','paths',note='First valid matching/diverging examples, not effect-selected.')
    source=ctx.table('05','horizon_alignment.csv')
    rows=[]
    for variable,label in [('persistence_match','Versus persistence'),('root_match','Versus root policy')]:
        z=source[source['mode']=='all_paths'].copy();z['y']=z.action_match-z[variable];z['x']=z.horizon;z['series']=label;rows.append(z)
    store.save('N2','B',pd.concat(rows),'Action agreement beyond baselines','line',xlabel='Forecast horizon (decisions; /15 for seconds)',ylabel='Agreement improvement',
        note='All paths; NOT conditioned on target action matching. V1 target weights reused.')
    z=source[source['mode']=='all_paths'].copy();z['x']=z.horizon;z['y']=z.targets;z['series']='Unique targets';z['coverage']=z.targets/z.run.map(risk(d).run.value_counts())
    store.save('N2','C',z,'Forecast coverage by horizon',xlabel='Forecast horizon (decisions)',ylabel='Unique eligible targets',note='Coverage denominator: scored decisions per source run.')
    rows=[]
    for variable,label in [('pixel_error','Imagined state'),('persistence_pixel_error','Stationary-screen baseline')]:
        z=source[source['mode']=='prefix_matched'].copy();z['x']=z.horizon;z['y']=z[variable];z['series']=label;rows.append(z)
    store.save('N2','D',pd.concat(rows),'State error after matched action prefixes',xlabel='Forecast horizon (decisions)',ylabel='Pixel MSE (images scaled 0–1)',
        note='Prefix-matched subset; path divergence is not state prediction error.')
    pred,_=ctx.prediction('compute')
    current=risk(d)
    matched=current[current.mismatch_count>0].merge(pred[['decision_id','p_1','target']],on='decision_id')
    matched['residual']=matched.target-matched.p_1
    bands=[]
    for session,session_rows in current.groupby('session'):
        for run in session_rows.run.unique():
            training=session_rows[(session_rows.run!=run)&(session_rows.mismatch_count>0)]
            cut=training.mismatch_pixel.median()
            test=matched[matched.run==run].copy()
            test['series']=np.where(test.mismatch_pixel<=cut,'Low arrived mismatch','High arrived mismatch')
            bands.append(test)
    tables=[]
    if bands:
        for label,z in pd.concat(bands).groupby('series'):
            b=evidence_bins(z,'evidence_signed');b['series']=label;tables.append(b)
    source=pd.concat(tables) if tables else pd.DataFrame(columns=['run','x','y','n','series'])
    store.save('N2','E',source,'Arrived mismatch × current evidence',xlabel='Current endpoint net evidence (Q units)',ylabel='OOF human-switch residual',
        test='V1 continuity block tests retained; curves descriptive',note='Only already observed, action-prefix-matched mismatch; training-fold mismatch split. No new causal interaction test.')
    carry=ctx.table('05','carry_descriptive.csv');carry['x']=carry.carried.map({0:'No carry',1:'Tree carry'});carry['y']=carry.budget;carry['n']=carry['count'];carry['series']='Mean budget'
    store.save('N2','F',carry,'Structural tree carry: descriptive','carry',ylabel='Additional computations',note='Side-by-side budget and human switch probability; no claim of human plan reuse.')
    validation=ctx.table('06','tracker_validation.json')
    frames=ctx.table('06','annotation_frames.csv')
    store.save('N3','A',frames.head(3),'Object-tracking candidates: unvalidated','objects',note='Preselected annotation frames, not selected for detector success.')
    reason=validation.get('reason','Independent validation gate pending')
    store.pending('N3','B','Independent 200-frame validation',reason)
    store.pending('N3','C','Urgency / computation / human switching','Validation pending: no urgency estimates generated')
    store.pending('N3','D','Urgency × evidence / budget','Validation pending: no interaction estimates generated')


def build(ctx):
    if (ctx.root/'panels/manifest.json').exists():
        ctx.verify_seal(ctx.root/'panels');return
    ctx.verify_seal(ctx.root/'fits')
    d=ctx.decisions();events=pd.read_csv(ctx.root/'derived/events.csv.gz')
    store=Panels(ctx)
    figure02(ctx,store,d);figure03(ctx,store,d,events);figure04(ctx,store,d,events);new_figures(ctx,store,d)
    ctx.seal(store.root, figures=list(FIGURES), stage='numeric panel preparation',
        inference='Read-only V1 tests and separately identified V2 families; no frame-level CI')
