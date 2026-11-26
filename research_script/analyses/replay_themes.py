"""Themes 02/03 from completed native human-trajectory visualization captures.

All-session-trained checkpoints: descriptive associations, not held-out human
prediction or causal computation utility. Reuses the audited event timing and
same-head cumulative V/Q definitions; no latent-activation filtering is used.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import html
import hashlib
import inspect
import json
import os
import shutil
import time

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import pandas as pd
from scipy.special import softmax, expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .value_updates import trace, updates
from common.io import signature
from common.inference import fit_pooled
from panels.mu_heatmaps import kernel_surface, mean_surfaces

RESEARCH = Path(__file__).resolve().parents[1]
REPO = RESEARCH.parent
GAMES = {'enduro': 0, 'pong': 1, 'spaceinvaders': 2}
LABELS = {'enduro': 'Enduro', 'pong': 'Pong', 'spaceinvaders': 'Space Invaders'}
COLORS = ('#287fa8', '#db8734')
VERSION = 'native-replay-themes-v1'
PROTOCOL = dict(
    M='log prior(incumbent human action) - max log prior(other actions), before search',
    V_total='sum_k abs(V_k-V_(k-1)), same actor task critic, including first imagination',
    Q_total='sum_(k,a) abs(Q_after-Q_before) where root-action visits are positive at both endpoints',
    phase='Subtract the direct linear phase-embedding offset to use the common SEARCH readout; verify reconstruction and no clipping',
    normalization='M: within-run population z score; V/Q totals: divide by all-decision run mean, before splitting behavior',
    switch='Current human action differs from previous replay human action within the same episode',
    coverage='Keep all scored decisions with a valid preceding real transition and source time gap <= 0.25 s; retain repeated frames',
    forced_controls='Exclude controls before min_search_steps from voluntary-choice analyses. A cap-reaching sampled CONTINUE is retained; no artificial STOP is added at the cap.',
    aggregation='Equal run weights within subject, then equal subject weights. Heatmaps pool within subject and average supported subject surfaces.',
    heatmap='COSYNE padded Gaussian histogram estimator; bandwidth 4% of axis range, >=12 local decisions. Group requires all requested subjects at a pixel.',
    limitations=['All human sessions overlap Thinker training; no held-out generalization claim',
        'Two participants are not sufficient for population-level inference; run-level boxes are descriptive',
        'V/Q totals can grow with computation count; controlled regressions are sensitivity analyses',
        'Q=0 without an observed revision means missing revision support, not no value evaluation',
        'Human replay allocation and value changes are not actual-environment reward benefit'])


class PackedTrace(trace.Trace):
    """Load each small numeric pack once; never load unused images/latents."""
    def __init__(self, path):
        super().__init__(path)
        self.packs = {}

    def read(self, name, rows=None):
        spec = self.specs[name]
        key = spec['storage_group']
        if key not in self.packs:
            self.packs[key] = np.asarray(self.z['packs/'+key][:])
        arr = self.packs[key][:, spec['storage_index'], ...]
        return arr if rows is None else arr[np.asarray(rows, int)]


FIELDS = ('initial_root', 'real_transition', 'search_computation', 'current_node_depth_pre',
    'current_node_depth_post', 'actor_reset', 'accepted_primary_action', 'human_action',
    'score_mask', 'done', 'pre_observation_source_index', 'search_steps_pre', 'phase_pre',
    'root_carried', 'forced_stop', 'control_valid', 'accepted_control', 'reward',
    'voc_ema_available', 'voc_soft_continue_probability', 'voc_ema_q', 'voc_online_q',
    'actor_baseline', 'voc_features', 'actor_action_probability', 'tree_reps_pre', 'tree_reps_post')


def verify_packs(t, record):
    m = t.manifest
    assert m['state'] == 'complete' and m['event_count'] == record['event_count']
    assert m['files_sha256'] == record['files_sha256'] and m['metadata'] == record['metadata']
    assert hashlib.sha256(json.dumps(m['files'],sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest() == m['files_sha256']
    wanted = set(FIELDS) | {f+'_valid' for f in FIELDS if f+'_valid' in t.specs}
    groups = {t.specs[f]['storage_group'] for f in wanted}
    checked = 0
    for item in m['files']:
        rel = Path(item['path'])
        if len(rel.parts) == 1 or (rel.parts[0] == 'packs' and rel.parts[1] in groups):
            p = t.path/rel
            assert p.stat().st_size == item['size'] and trace.sha(p) == item['sha256'], str(p)
            checked += 1
    return checked


def find_checkpoint(plan):
    cp = plan['semantic']['checkpoint']
    candidates = list((REPO/'logs/visualization/campaigns').glob(
        '*/checkpoints/'+cp['logical_xpid']+'/milestones/10000000'))
    for folder in candidates:
        actor = folder/cp['actor']['name']
        config = folder/cp['config']['name']
        if actor.is_file() and config.is_file() and trace.sha(actor) == cp['actor']['sha256'] and trace.sha(config) == cp['config']['sha256']:
            return folder, actor, config
    raise FileNotFoundError('No checkpoint matching the capture hashes: '+cp['logical_xpid'])


def normalized(d):
    d = d.copy()
    g = d.groupby('run').M
    sd = g.transform(lambda x: x.std(ddof=0))
    d['M_z'] = (d.M-g.transform('mean'))/sd.replace(0, np.nan)
    for kind in ('V', 'Q'):
        mean = d.groupby('run')[kind+'_total'].transform('mean')
        d[kind+'_relative'] = np.divide(d[kind+'_total'], mean, out=np.zeros(len(d)), where=mean.gt(0))
        d['log_'+kind] = np.log1p(d[kind+'_total'])
        d['K_'+kind] = 4*d[kind+'_total']*expit(d.M)*expit(-d.M)
    d['log_budget'] = np.log1p(d.budget)
    d['log_age'] = np.log1p(d.age)
    d['run_label'] = d.episode  # HAC products must never cross episode boundaries.
    d['real_step'] = d.decision
    for a in sorted(d.incumbent_action.unique()):
        d[f'previous_action_{a}'] = d.incumbent_action.eq(a).astype(float)
    return d


def extract_unit(input_root, cache_root, subject, game):
    import torch
    import yaml
    torch.set_num_threads(1)
    capture = Path(input_root)/f'sub{subject:03d}'/game
    out = Path(cache_root)/f'sub{subject:03d}'/game
    out.mkdir(parents=True, exist_ok=True)
    complete = trace.read_json(capture/'COMPLETED.json')
    manifest = trace.read_json(capture/'manifest.json')
    unsigned=dict(manifest);claimed=unsigned.pop('manifest_sha256')
    digest=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
    assert complete['status'] == 'COMPLETED' and digest(unsigned) == claimed == complete['manifest_sha256']
    assert digest(manifest['episodes']) == manifest['episodes_sha256']
    assert digest(manifest['semantic']) == manifest['semantic_sha256'] == complete['semantic_sha256']
    plan = trace.read_json(capture/'capture_plan.json')
    folder, actor, config = find_checkpoint(plan)
    cfg = yaml.safe_load(config.read_text())
    assert cfg['critic_enc_type'] == 0 and not cfg['sep_actor_critic'] and cfg['last_layer_n'] == 0
    assert cfg['model_enc_type'] == 0
    ident = dict(version=VERSION, subject=subject, game=game, input=str(capture.resolve()),
        manifest_sha256=complete['manifest_sha256'], checkpoint=plan['semantic']['checkpoint'],
        source_sha256=signature(dict(code='\n'.join(inspect.getsource(f) for f in
            [PackedTrace, verify_packs, find_checkpoint, normalized, extract_unit]), fields=FIELDS,
            value_reader=trace.sha(Path(__file__).with_name('value_updates.py')),
            trace_reader=trace.sha(RESEARCH/'common/native_replay.py'))))
    token = signature(ident)
    marker = out/'manifest.json'
    if marker.exists():
        old = trace.read_json(marker)
        if old.get('signature') == token and old.get('status') == 'COMPLETED' and all(
                (out/f).is_file() and trace.sha(out/f) == h for f,h in old['artifacts'].items()):
            trace.log(f'sub{subject:03d} {game}: reuse verified numeric cache')
            return str(out)
    trace.write_json(marker, dict(ident, signature=token, status='RUNNING'))
    state = torch.load(actor, map_location='cpu', weights_only=False)['actor_net_state_dict']
    w = state['baseline.weight'][0].double().numpy()
    bias = state['baseline.bias'][0].double().item()
    embedding = state['phase_embedding.weight'].double().numpy()
    offsets = (embedding-embedding[0]) @ w
    clip = cfg['reward_clip']/(1-cfg['discounting']) if cfg['reward_clip'] > 0 else np.inf
    minimum, maximum = cfg['min_search_steps'], cfg['max_search_steps']
    rows, eventframes, audits, sources = [], [], [], {}
    for number, ep in enumerate(manifest['episodes'], 1):
        path = capture/ep['path']
        t = PackedTrace(path)
        checked_files = verify_packs(t, ep)
        meta = t.meta
        assert meta['subject'] == subject and meta['game'] == GAMES[game]
        for key, label in [('actor','checkpoint_actor_sha256'),('config','checkpoint_config_sha256'),('model','checkpoint_model_sha256')]:
            assert meta[label] == plan['semantic']['checkpoint'][key]['sha256']
        source = Path(meta['source_path'])
        if str(source) not in sources:
            sources[str(source)] = trace.sha(source)
        assert sources[str(source)] == meta['source_sha256']
        with np.load(source, allow_pickle=False) as raw:
            actions = raw['action'].argmax(1) if raw['action'].ndim == 2 else raw['action'].copy()
            times = raw['time'].astype(float)
        f = {k:t.read(k) for k in FIELDS if k not in ['voc_features']}
        field = lambda arr, name: arr[:, slice(*t.tree_schema[name])]
        pre, post = f['tree_reps_pre'].astype(float), f['tree_reps_post'].astype(float)
        assert np.isfinite(pre).all() and np.isfinite(post).all()
        q0, q1 = field(pre,'root_qs_mean'), field(post,'root_qs_mean')
        n0, n1 = field(pre,'root_ns'), field(post,'root_ns')
        q_update = updates(q0,q1,n0,n1)
        assert ((q1[f['search_computation']] != q0[f['search_computation']]).sum(1) <= 1).all()
        phase = f['phase_pre'].astype(int)
        value = f['actor_baseline'][:,0].astype(float)
        corrected = value-offsets[phase]
        p = f['actor_action_probability'].astype(float)
        ema = f['voc_ema_q'][:,0].astype(float)-f['voc_ema_q'][:,1].astype(float)
        online = f['voc_online_q'][:,0].astype(float)-f['voc_online_q'][:,1].astype(float)
        qc = dict(episode=meta['episode_id'], events=t.n, kept=0, unscored=0, time_gap=0,
                  checked_numeric_files=checked_files, manifest_sha256=trace.sha(path/'manifest.json'))
        used=[]; previous=-1; bout_start=None
        row_start=len(rows); decision_map=np.full(t.n,-1,dtype=np.int64)
        for j,left,right,comps,_ in trace.decision_edges(f['initial_root'],f['real_transition'],
                f['search_computation'], f['current_node_depth_post'], f['actor_reset']):
            src=int(f['pre_observation_source_index'][right]); human=int(f['human_action'][right])
            assert human == int(actions[src])
            now=float(times[src])
            if bout_start is None: bout_start=now
            age=max(now-bout_start,0.)
            if human != previous: bout_start=now
            previous=human
            if not f['score_mask'][right]: qc['unscored']+=1; continue
            assert f['real_transition'][left] and not f['done'][left]
            inc=int(f['human_action'][left]); prev_src=int(f['pre_observation_source_index'][left])
            assert inc==actions[prev_src]
            dt=float(times[src]-times[prev_src]); assert dt>=0
            if dt==0: assert inc==human
            if dt>.25: qc['time_gap']+=1; continue
            B=len(comps); assert B==f['search_steps_pre'][right] and minimum<=B<=maximum
            controls=np.arange(left+1,right)
            controls=controls[(phase[controls]==0)&f['control_valid'][controls]]
            assert len(controls) and np.isin(f['accepted_control'][controls],[0,1,2]).all()
            assert np.sum(f['accepted_control'][controls]!=2)==B
            capped=bool(f['forced_stop'][comps[-1]]) if B else False
            assert np.sum(f['accepted_control'][controls]==2)==int(not capped)
            first=controls[0]
            prior=softmax(field(post,'root_policy')[left]); lp=np.log(np.clip(prior,1e-10,1))
            M=float(lp[inc]-np.max(np.delete(lp,inc)))
            assert t.valid('actor_action_probability')[right] and np.isfinite(p[right]).all()
            assert (p[right]>=0).all() and np.isclose(p[right].sum(),1,atol=1e-5)
            if B:
                assert (np.diff(comps)==1).all()
                endpoints=np.r_[comps[0],comps+1]
            else:endpoints=np.array([first])
            assert t.valid('actor_baseline')[endpoints].all()
            assert (f['pre_observation_source_index'][endpoints]==src).all()
            assert np.array_equal(f['search_steps_pre'][endpoints],np.arange(B+1))
            assert (phase[comps]==0).all()
            assert (np.abs(value[endpoints])<clip-1e-5).all() and (np.abs(corrected[endpoints])<clip-1e-5).all()
            delta=np.diff(corrected[endpoints]); used.extend(endpoints)
            valid_count=0;branch=-1
            for event in comps:
                if f['current_node_depth_pre'][event]==0:branch=int(f['accepted_primary_action'][event])
                assert 0<=branch<len(prior)
                valid_count+=int(n0[event,branch]>0 and n1[event,branch]>0)
                assert np.allclose(np.delete(q1[event]-q0[event],branch),0,atol=2e-6,rtol=0)
            base=dict(subject=subject,game=game,session=meta['session'],run=meta['source_id'],
                episode=meta['episode_id'],trace=str(path.resolve()),decision=j,root_event=left,
                real_event=right,source_frame=src,time=now,source_dt=dt,age=age,
                human_action=human,incumbent_action=inc,switch=int(human!=inc),M=M,budget=B,
                V_total=float(np.abs(delta).sum()),Q_total=float(q_update[comps].sum()),
                valid_update_count=valid_count,Q_supported=valid_count>0,
                V_initial=float(corrected[endpoints[0]]),V_final=float(corrected[endpoints[-1]]),
                first_abs_dV=float(abs(delta[0])) if B else 0.,
                root_value=float(field(post,'root_v')[left,0]),
                root_entropy=float(-(prior*np.log(prior.clip(1e-12))).sum()),
                carry=int(f['root_carried'][left]),cap_reached=int(capped),
                ema_initial=float(ema[first]),online_initial=float(online[first]),
                prior_reward=float(f['reward'][left].reshape(-1)[0]),
                model_switch=float(1-p[right,inc]),human_probability=float(p[right,human]),
                model_correct=int(p[right].argmax()==human),scored=1)
            rows.append(base);qc['kept']+=1
            assert t.valid('voc_ema_q')[controls].all() and t.valid('voc_online_q')[controls].all()
            assert t.valid('voc_soft_continue_probability')[controls].all() and f['voc_ema_available'][controls].all()
            probs=f['voc_soft_continue_probability'][controls].astype(float)
            assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
            voluntary=f['search_steps_pre'][controls]>=minimum
            assert (f['accepted_control'][controls[~voluntary]]!=2).all()
            decision_map[controls]=len(rows)-1
        chosen=np.flatnonzero(decision_map>=0)
        if len(chosen):
            meta_frame=pd.DataFrame(rows[row_start:])
            ef=meta_frame.iloc[decision_map[chosen]-row_start][['subject','game','session','run','episode',
                'decision','M','root_value','age','prior_reward']].reset_index(drop=True)
            ef['log_age']=np.log1p(ef.pop('age'))
            for key,array in dict(event=chosen,choice_index=f['search_steps_pre'][chosen],
                    search_steps=f['search_steps_pre'][chosen],depth=f['current_node_depth_pre'][chosen],
                    accepted_control=f['accepted_control'][chosen],continued=(f['accepted_control'][chosen]!=2).astype(int),
                    voluntary=f['search_steps_pre'][chosen]>=minimum,cap_reached=f['forced_stop'][chosen],
                    probability=f['voc_soft_continue_probability'][chosen],voc_ema=ema[chosen],voc_online=online[chosen]).items():
                ef[key]=array
            eventframes.append(ef)
        used=np.unique(used)
        if len(used):
            sample=np.unique(np.r_[used[np.linspace(0,len(used)-1,min(16,len(used))).astype(int)],used[phase[used]!=0]])
            assert t.valid('voc_features')[sample].all()
            z=t.read('voc_features',sample).astype(float)
            error=float(np.abs(z@w+bias-value[sample]).max())
            assert error<2e-5
            np.testing.assert_allclose((z-embedding[phase[sample]]+embedding[0])@w+bias,corrected[sample],atol=2e-5,rtol=0)
            qc.update(value_reconstruction_rows=len(sample),value_reconstruction_max_error=error)
        audits.append(qc)
        trace.log(f'sub{subject:03d} {game}: episode {number}/{len(manifest["episodes"])}; {len(rows):,} decisions')
    d=normalized(pd.DataFrame(rows));e=pd.concat(eventframes,ignore_index=True)
    assert d.session.nunique()==4 and not d.duplicated(['episode','real_event']).any()
    assert np.isfinite(d[['V_total','Q_total']]).all().all() and d[['V_total','Q_total']].ge(0).all().all()
    assert d.loc[~d.Q_supported,'Q_total'].eq(0).all()
    np.testing.assert_allclose(d.loc[d.budget.eq(1),'V_total'],d.loc[d.budget.eq(1),'first_abs_dV'])
    d.to_csv(out/'decisions.csv.gz',index=False);e.to_csv(out/'choices.csv.gz',index=False)
    qa=dict(decisions=len(d),choices=len(e),voluntary_choices=int(e.voluntary.sum()),runs=d.run.nunique(),
        episodes=audits,config={k:cfg.get(k) for k in ['min_search_steps','max_search_steps','voc_gate_q_temperature',
        'voc_gate_exact_projection','voc_gate_epsilon_greedy_execution','voc_gate_execution_epsilon']},
        phase_offsets=offsets,checkpoint=str(folder),sources=sources,
        verification='Hashes of all packs used in the analysis and behavioral sources; unused image/latent payloads not rehashed')
    trace.write_json(out/'qa.json',qa)
    artifacts={f:trace.sha(out/f) for f in ['decisions.csv.gz','choices.csv.gz','qa.json']}
    trace.write_json(marker,dict(ident,signature=token,status='COMPLETED',qa=qa,artifacts=artifacts))
    return str(out)


def savefig(fig, root, name):
    root.mkdir(parents=True,exist_ok=True)
    fig.savefig(root/(name+'.png'),dpi=160,bbox_inches='tight')
    fig.savefig(root/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def paired_box(ax, means, metric, title):
    p=means.pivot(index='run',columns='switch',values=metric).reindex(columns=[0,1]).dropna()
    if len(p):
        bx=ax.boxplot([p[0],p[1]],positions=[0,1],widths=.42,patch_artist=True,showfliers=False)
        for box,color in zip(bx['boxes'],COLORS):box.set_facecolor(color);box.set_alpha(.28)
        for _,row in p.iterrows():ax.plot([0,1],row,color='#8a9298',alpha=.5,lw=.7,marker='.',ms=4)
    ax.set(xticks=[0,1],xticklabels=['Persist','Switch'],title=title)
    ax.spines[['top','right']].set_visible(False)


def unit02(d, out):
    tables=out/'tables';figs=out/'figures';tables.mkdir(parents=True,exist_ok=True)
    metrics=['M_z','V_total','Q_total','V_relative','Q_relative','budget','K_V','K_Q']
    means=d.groupby(['run','session','switch'])[metrics].mean().reset_index()
    means.to_csv(tables/'run_condition_means.csv',index=False)
    summaries=[]
    for metric in metrics:
        p=means.pivot(index='run',columns='switch',values=metric).reindex(columns=[0,1]).dropna()
        summaries.append(dict(metric=metric,runs=len(p),persist=p[0].mean(),switch=p[1].mean(),difference=(p[1]-p[0]).mean()))
    pd.DataFrame(summaries).to_csv(tables/'summary.csv',index=False)
    coverage=[]
    for label,mask in [('all',np.ones(len(d),bool)),('Persist',d['switch'].eq(0)),('Switch',d['switch'].eq(1))]:
        r=d.loc[mask];coverage.append(dict(condition=label,decisions=len(r),V_computable=int(np.isfinite(r.V_total).sum()),
            Q_any_revision=int(r.Q_supported.sum()),Q_revision_fraction=r.Q_supported.mean(),
            mean_budget=r.budget.mean(),fraction_budget1=r.budget.eq(1).mean(),fraction_cap=r.cap_reached.mean()))
    pd.DataFrame(coverage).to_csv(tables/'coverage.csv',index=False)
    title=f"{LABELS[d.game.iloc[0]]} · sub{int(d.subject.iloc[0]):03d} · 10M"
    for kind in ['V','Q']:
        fig,axes=plt.subplots(1,3,figsize=(11,3.8))
        for ax,metric,label in zip(axes,['M_z',kind+'_relative',kind+'_total'],
                ['Pre-search M (run z)',kind+' total / run mean',kind+' cumulative absolute update']):
            paired_box(ax,means,metric,label)
        fig.suptitle(title);fig.text(.5,-.02,'Each connected pair is one run; descriptive within-participant comparison.',ha='center',fontsize=9)
        savefig(fig,figs,'persist_switch_'+kind)
    sensitivity=[]
    supported=d.loc[d.Q_supported]
    supported.groupby(['run','switch'])[['V_total','Q_total','V_relative','Q_relative']].mean().to_csv(tables/'equal_Q_support_means.csv')
    controls=['log_budget','log_age','root_value','prior_reward','carry']+[c for c in d if c.startswith('previous_action_')]
    for kind in ['V','Q']:
        for r in fit_pooled(d,'switch',['M','log_'+kind],controls,True):sensitivity.append(dict(value_estimator=kind,**r))
    pd.DataFrame(sensitivity).to_csv(tables/'budget_adjusted_associations.csv',index=False)
    switch_locked(d,out)


def switch_locked(d,out):
    """Do not bridge excluded decisions; pool episodes by event count per run."""
    tables=out/'tables';figs=out/'figures'
    locked=[]
    for episode,r in d.groupby('episode'):
        r=r.sort_values('decision').set_index('decision')
        for lag in range(-10,11):
            centers=r.index[r['switch'].eq(1)];ids=centers+lag
            pos=r.index.get_indexer(ids);origin=r.index.get_indexer(centers)
            good=(pos>=0)&(pos-origin==lag)
            ids=ids[good]
            if len(ids):
                row=r.loc[ids,['M_z','V_relative','Q_relative','budget']].mean().to_dict()
                locked.append(dict(run=r.run.iloc[0],episode=episode,lag=lag,n=len(ids),**row))
    locked=pd.DataFrame(locked)
    locked.to_csv(tables/'switch_locked_episode_means.csv',index=False)
    fig,axes=plt.subplots(1,4,figsize=(12,3))
    for ax,col in zip(axes,['M_z','V_relative','Q_relative','budget']):
        weighted=locked.assign(total=locked[col]*locked.n).groupby(['run','lag'])[['total','n']].sum()
        run=(weighted.total/weighted.n).unstack(0)
        ax.plot(run.index,run.mean(axis=1),color=COLORS[0]);ax.axvline(0,color='gray',ls='--',lw=.8)
        ax.set(title=col,xlabel='Real decisions from human switch')
    savefig(fig,figs,'switch_locked')


def unit03(d,e,out,cfg):
    tables=out/'tables';figs=out/'figures';tables.mkdir(parents=True,exist_ok=True)
    v=e.loc[e.voluntary].copy()
    budget=[];cal=[];signals=[];voccurves=[];statecurves=[];hazard=[]
    for run,r in d.groupby('run'):
        counts=r.budget.value_counts()
        for k in range(cfg['max_search_steps']+1):budget.append(dict(run=run,budget=k,n=int(counts.get(k,0)),probability=counts.get(k,0)/len(r)))
        for metric in ['ema_initial','root_value','M']:
            z=r.assign(bin=np.minimum((r[metric].rank(pct=True)*10).astype(int),9))
            for b,a in z.groupby('bin'):statecurves.append(dict(run=run,metric=metric,bin=b,state=a[metric].mean(),budget=a.budget.mean(),n=len(a)))
    for run,r in v.groupby('run'):
        r=r.assign(bin=np.minimum((r.probability*10).astype(int),9))
        for b,a in r.groupby('bin'):cal.append(dict(run=run,bin=b,probability=a.probability.mean(),actual=a.continued.mean(),n=len(a)))
        for k,a in r.groupby('search_steps'):hazard.append(dict(run=run,search_steps=k,probability=a.probability.mean(),actual=a.continued.mean(),n=len(a)))
        for source in ['ema','online']:
            x=r['voc_'+source]
            for sign,mask in [('positive',x>0),('negative',x<0)]:
                a=r.loc[mask];signals.append(dict(run=run,source=source,sign=sign,n=len(a),probability=a.probability.mean(),actual=a.continued.mean()))
            z=r.assign(voc_bin=np.minimum((x.rank(pct=True)*10).astype(int),9))
            for b,a in z.groupby('voc_bin'):voccurves.append(dict(run=run,source=source,bin=b,voc=a['voc_'+source].mean(),probability=a.probability.mean(),actual=a.continued.mean(),n=len(a)))
    frames=dict(budget_distribution=pd.DataFrame(budget),gate_calibration=pd.DataFrame(cal),
        sign_conditioned_continuation=pd.DataFrame(signals),voc_continuation=pd.DataFrame(voccurves),
        state_budget=pd.DataFrame(statecurves),continuation_by_search_step=pd.DataFrame(hazard))
    for name,frame in frames.items():frame.to_csv(tables/(name+'.csv'),index=False)
    p=v.probability.to_numpy();teacher=expit(v.voc_ema.to_numpy()/cfg['voc_gate_q_temperature'])
    a=p.clip(1e-10,1-1e-10);b=teacher.clip(1e-10,1-1e-10)
    diag=dict(decisions=len(d),mean_budget=d.budget.mean(),fraction_budget1=d.budget.eq(1).mean(),
        fraction_cap=d.cap_reached.mean(),min_forced_choices=int((~e.voluntary).sum()),voluntary_choices=len(v),
        cap_reaching_choices_retained=int(v.cap_reached.sum()),brier=float(np.mean((p-v.continued)**2)),
        teacher_gate_KL=float(np.mean(b*np.log(b/a)+(1-b)*np.log((1-b)/(1-a)))),
        teacher_probability_mae=float(np.mean(abs(p-teacher))),
        initiation='Structurally forced by minimum budget; no voluntary start/stop comparison',
        raw_environment_reward_benefit='Not measured by human replay')
    trace.write_json(out/'summary.json',diag)
    title=f"{LABELS[d.game.iloc[0]]} · sub{int(d.subject.iloc[0]):03d} · 10M"
    fig,ax=plt.subplots(figsize=(7,3.6));z=frames['budget_distribution'].groupby('budget').probability.mean()
    ax.bar(z.index,z,width=.85,color=COLORS[0]);ax.set(xlabel='Imagination computations',ylabel='Fraction of decisions',title=title)
    savefig(fig,figs,'planning_budget_histogram')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    z=frames['gate_calibration'].groupby('bin')[['probability','actual']].mean()
    axes[0].plot(z.probability,z.actual,'o-',label='Observed');axes[0].plot([0,1],[0,1],'--',color='gray',label='Calibrated')
    axes[0].set(xlabel='Recorded continue probability',ylabel='Actual continuation',xlim=(0,1),ylim=(0,1),title='Voluntary execution')
    for ax,source in zip(axes[1:],['ema','online']):
        z=frames['sign_conditioned_continuation'].query('source == @source').groupby('sign')[['probability','actual']].mean().reindex(['negative','positive'])
        ax.bar(np.arange(2)-.16,z.probability,.3,label='Gate probability',color=COLORS[0]);ax.bar(np.arange(2)+.16,z.actual,.3,label='Actual continue',color=COLORS[1])
        ax.set(xticks=[0,1],xticklabels=['VOC < 0','VOC > 0'],ylim=(0,1),title=source.upper()+' self-evaluation')
    axes[1].legend(fontsize=8);fig.suptitle(title);savefig(fig,figs,'voc_and_actual_continuation')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,metric in zip(axes,['ema_initial','root_value','M']):
        z=frames['state_budget'].query('metric == @metric').groupby('bin').budget.mean()
        ax.plot((z.index+.5)/10,z,'o-');ax.set(xlabel=metric+' percentile within run',ylabel='Mean imagination count')
    fig.suptitle(title);savefig(fig,figs,'state_dependent_budget')
    v['run_label']=v.episode;v['real_step']=v.decision;v['log_search_steps']=np.log1p(v.search_steps)
    effects=[]
    for source in ['ema','online']:
        for r in fit_pooled(v,'continued',['voc_'+source],['log_search_steps','depth','M','root_value','log_age','prior_reward'],True):
            effects.append(dict(family='voluntary_continuation',source=source,**r))
    for r in fit_pooled(d,'log_budget',['ema_initial'],['M','root_value','log_age','prior_reward','carry'],False):
        effects.append(dict(family='budget',source='ema',**r))
    pd.DataFrame(effects).to_csv(tables/'adjusted_associations.csv',index=False)


def heatmaps(data, stage, subjects, games):
    pops=[('overall','Any action switch'),('noop_to_active','No input → active'),
          ('active_to_other','Active → any other action'),('active_to_noop','Active → no input')]
    for game in games:
        frames=[data[(s,game)] for s in subjects]
        for kind in ['V','Q']:
            x=np.concatenate([d.M_z.dropna().to_numpy() for d in frames]);y=np.concatenate([d[kind+'_relative'].to_numpy() for d in frames])
            xlim=np.quantile(x,[.01,.99]);ylim=np.array([0.,max(float(np.quantile(y,.99)),1e-6)])
            maps={s:[] for s in subjects};counts={s:[] for s in subjects}
            for pop,label in pops:
                for s,d in zip(subjects,frames):
                    r=d.copy()
                    if pop=='noop_to_active':r=r.loc[r.incumbent_action.eq(0)]
                    elif pop.startswith('active'):r=r.loc[r.incumbent_action.ne(0)]
                    r=r.assign(outcome=r.human_action.eq(0).astype(int) if pop=='active_to_noop' else r['switch'],
                               U_relative=r[kind+'_relative'],U_all=r[kind+'_total'])
                    prob,local,_=kernel_surface(r,xlim,ylim)
                    maps[s].append(prob);counts[s].append(local)
            grouped=[];supports=[]
            for j in range(len(pops)):
                mean,support=mean_surfaces(np.stack([maps[s][j] for s in subjects]),minimum=len(subjects))
                grouped.append(mean);supports.append(support)
            for owner,surfaces in [(f'sub{s:03d}',maps[s]) for s in subjects]+[('group',grouped)]:
                root=stage/'02'/owner/game;figs=root/'figures';tables=root/'tables';tables.mkdir(parents=True,exist_ok=True)
                fig,axes=plt.subplots(1,len(pops),figsize=(14,3.7),sharex=True,sharey=True)
                for j,(ax,(pop,label),surf) in enumerate(zip(axes,pops,surfaces)):
                    supported=np.concatenate([a[np.isfinite(a)] for a in [maps[s][j] for s in subjects]])
                    vmax=max(float(np.quantile(supported,.97)) if len(supported) else 1.,.01)
                    im=ax.imshow(surf,origin='lower',extent=[*xlim,*ylim],aspect='auto',vmin=0,vmax=vmax,cmap='viridis')
                    ax.set(title=label,xlabel='Pre-search M (run z)');fig.colorbar(im,ax=ax,shrink=.78,label='P(transition)')
                axes[0].set_ylabel(kind+' total / run mean')
                fig.suptitle(LABELS[game]+' · '+owner+' · '+('equal-participant surface' if owner=='group' else 'within-participant surface'))
                savefig(fig,figs,'MU_heatmap_'+kind)
                arrays={p[0]:a for p,a in zip(pops,surfaces)}
                arrays.update({p[0]+'_support':a for p,a in zip(pops,supports if owner=='group' else counts[int(owner[3:])])})
                np.savez_compressed(tables/('MU_heatmap_'+kind+'.npz'),xlim=xlim,ylim=ylim,**arrays)


def group_plots(data,stage,themes,subjects,games):
    for game in games:
        for theme in themes:
            root=stage/theme/'group'/game;tables=root/'tables';figs=root/'figures';tables.mkdir(parents=True,exist_ok=True)
            if theme=='02':
                pieces=[]
                for s in subjects:
                    f=pd.read_csv(stage/theme/f'sub{s:03d}'/game/'tables/run_condition_means.csv')
                    pieces.append(f.assign(subject=s))
                run=pd.concat(pieces);means=run.groupby(['subject','switch']).mean(numeric_only=True).reset_index()
                means.to_csv(tables/'subject_condition_means.csv',index=False)
                for kind in ['V','Q']:
                    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
                    for ax,metric in zip(axes,['M_z',kind+'_relative',kind+'_total']):
                        for s,r in means.groupby('subject'):
                            rr=r.set_index('switch').reindex([0,1]);ax.plot([0,1],rr[metric],'o-',label=f'sub{s:03d}')
                        avg=means.groupby('switch')[metric].mean().reindex([0,1]);ax.plot([0,1],avg,'D--',color='black',label='Equal-subject mean')
                        ax.set(xticks=[0,1],xticklabels=['Persist','Switch'],title=metric)
                    axes[0].legend(fontsize=8);fig.suptitle(LABELS[game]+' · two-participant descriptive comparison')
                    savefig(fig,figs,'persist_switch_'+kind)
            else:
                for name in ['budget_distribution','gate_calibration','sign_conditioned_continuation','state_budget','voc_continuation','continuation_by_search_step']:
                    parts=[pd.read_csv(stage/theme/f'sub{s:03d}'/game/'tables'/(name+'.csv')).assign(subject=s) for s in subjects]
                    pd.concat(parts).to_csv(tables/(name+'.csv'),index=False)
                b=pd.read_csv(tables/'budget_distribution.csv').groupby(['subject','budget']).probability.mean().reset_index()
                fig,ax=plt.subplots(figsize=(7,3.8))
                for s,r in b.groupby('subject'):ax.plot(r.budget,r.probability,'o-',ms=3,lw=1,label=f'sub{s:03d}')
                z=b.groupby('budget').probability.mean();ax.bar(z.index,z,width=.85,alpha=.2,color='gray',label='Equal-subject mean')
                ax.set(xlabel='Imagination computations',ylabel='Fraction of decisions',title=LABELS[game]);ax.legend();savefig(fig,figs,'planning_budget_histogram')
                sign=pd.read_csv(tables/'sign_conditioned_continuation.csv').groupby(['subject','source','sign'])[['probability','actual']].mean().reset_index()
                sign.to_csv(tables/'subject_sign_conditioned_continuation.csv',index=False)
                fig,axes=plt.subplots(1,2,figsize=(9,3.8))
                for ax,source in zip(axes,['ema','online']):
                    for s in subjects:
                        r=sign.loc[(sign.subject==s)&(sign.source==source)].set_index('sign').reindex(['negative','positive'])
                        ax.plot([0,1],r.probability,'o-',label=f'sub{s:03d}: probability')
                        ax.plot([0,1],r.actual,'x--',label=f'sub{s:03d}: actual')
                    ax.set(xticks=[0,1],xticklabels=['VOC < 0','VOC > 0'],ylim=(0,1),title=source.upper())
                axes[0].legend(fontsize=8);fig.suptitle(LABELS[game]);savefig(fig,figs,'voc_and_actual_continuation')


def compare_previous(data, outputs):
    checks=[]
    for game in ['enduro','pong']:
        p=outputs/'02/sub001_hxln_10m_total_vq_updates_20261123'/game/'decisions.csv.gz'
        if (1,game) not in data or not p.is_file():continue
        old=pd.read_csv(p);new=data[(1,game)]
        match=old.merge(new,on=['episode','real_event'],suffixes=('_old','_new'),validate='one_to_one')
        assert len(match)==len(old)==len(new)
        for metric in ['V_total','Q_total','V_relative','Q_relative','budget','switch','valid_update_count']:
            np.testing.assert_allclose(match[metric+'_old'],match[metric+'_new'],atol=1e-10,rtol=1e-10)
        checks.append(dict(game=game,decisions=len(match),status='MATCHED_PREVIOUS_VERIFIED_TOTALS'))
    return checks


def cohort_tables(root, themes, subjects, games):
    """Convenient cohort-level tables retain explicit participant/game labels."""
    for theme in themes:
        if theme=='02':
            for name in ['summary','coverage','budget_adjusted_associations']:
                pieces=[]
                for s in subjects:
                    for game in games:
                        p=root/theme/f'sub{s:03d}'/game/'tables'/(name+'.csv')
                        pieces.append(pd.read_csv(p).assign(subject=s,game=game))
                pd.concat(pieces,ignore_index=True).to_csv(root/theme/(name+'.csv'),index=False)
        else:
            rows=[];signals=[]
            for s in subjects:
                for game in games:
                    p=root/theme/f'sub{s:03d}'/game
                    rows.append(dict(subject=s,game=game,**trace.read_json(p/'summary.json')))
                    z=pd.read_csv(p/'tables/sign_conditioned_continuation.csv')
                    signals.append(z.groupby(['source','sign'])[['probability','actual']].mean().reset_index().assign(subject=s,game=game))
            pd.DataFrame(rows).to_csv(root/theme/'summary.csv',index=False)
            pd.concat(signals,ignore_index=True).to_csv(root/theme/'sign_conditioned_continuation.csv',index=False)


def index_page(root,theme):
    pictures=sorted(root.glob('**/figures/*.png'))
    body=['<!doctype html><html><meta charset="utf-8"><title>Replay theme '+theme+'</title>',
        '<style>body{font:16px sans-serif;max-width:1200px;margin:40px auto}img{width:100%}h2{margin-top:50px}</style>',
        '<h1>Theme '+theme+' · sub001/sub002 · Enduro, Pong, Space Invaders</h1>',
        '<p>10M human-trajectory replay. All-session-trained models. Descriptive results; no causal utility or population-level inference.</p>']
    for p in pictures:
        rel=p.relative_to(root).as_posix();body.append('<h2>'+html.escape(rel)+'</h2><a href="'+rel+'"><img loading="lazy" src="'+rel+'"></a>')
    (root/'index.html').write_text('\n'.join(body)+'</html>')


def main(default_theme='02'):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=REPO/'logs/visualization/hxln_10m')
    parser.add_argument('--outputs-root',type=Path,default=RESEARCH/'outputs')
    parser.add_argument('--subjects',type=int,nargs='+',default=[1,2])
    parser.add_argument('--games',choices=list(GAMES),nargs='+',default=list(GAMES))
    parser.add_argument('--themes',choices=['02','03'],nargs='+',default=[default_theme])
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--replace',action='store_true',help='Replace only selected numbered output directories after successful analysis and verification')
    parser.add_argument('--extract-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.workers<=4:parser.error('Use one to four local workers')
    outputs=args.outputs_root.resolve();cache=outputs/'00/replay_features'
    for theme in args.themes:
        if (outputs/theme).exists() and not args.replace and not args.extract_only:parser.error('Existing outputs require explicit --replace')
    stage=outputs/'.native_replay_build';stage.mkdir(parents=True,exist_ok=True)
    state=dict(status='EXTRACTING',started_at=time.time(),subjects=args.subjects,games=args.games,themes=args.themes,completed=[])
    trace.write_json(stage/'progress.json',state)
    jobs={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for subject in args.subjects:
            for game in args.games:jobs[pool.submit(extract_unit,args.input_root,cache,subject,game)]=(subject,game)
        for future in as_completed(jobs):
            pair=jobs[future];future.result();state['completed'].append(pair);trace.write_json(stage/'progress.json',state)
    if args.extract_only:return
    data={(s,g):pd.read_csv(cache/f'sub{s:03d}'/g/'decisions.csv.gz') for s in args.subjects for g in args.games}
    regression=compare_previous(data,outputs)
    state.update(status='ANALYZING',prior_result_checks=regression);trace.write_json(stage/'progress.json',state)
    for (subject,game),d in data.items():
        trace.log(f'Analyzing sub{subject:03d} {game}')
        if '02' in args.themes:unit02(d,stage/'02'/f'sub{subject:03d}'/game)
        if '03' in args.themes:
            e=pd.read_csv(cache/f'sub{subject:03d}'/game/'choices.csv.gz')
            cfg=trace.read_json(cache/f'sub{subject:03d}'/game/'qa.json')['config']
            unit03(d,e,stage/'03'/f'sub{subject:03d}'/game,cfg)
    group_plots(data,stage,args.themes,args.subjects,args.games)
    if '02' in args.themes:heatmaps(data,stage,args.subjects,args.games)
    cohort_tables(stage,args.themes,args.subjects,args.games)
    for theme in args.themes:
        root=stage/theme;trace.write_json(root/'protocol.json',PROTOCOL)
        index_page(root,theme)
        artifacts={str(p.relative_to(root)):trace.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
        assert artifacts and len(list(root.glob('sub*/game*')))==0
        trace.write_json(root/'manifest.json',dict(status='COMPLETED',version=VERSION,subjects=args.subjects,games=args.games,
            source=str(args.input_root.resolve()),numeric_cache=str(cache),prior_result_checks=regression,
            analysis_code_sha256={str(p.relative_to(RESEARCH)):trace.sha(p) for p in [Path(__file__),RESEARCH/'common/inference.py',RESEARCH/'panels/mu_heatmaps.py']},
            decisions={f'sub{s:03d}/{g}':len(d) for (s,g),d in data.items()},artifacts=artifacts,completed_at=time.time()))
        target=outputs/theme
        assert target.parent==outputs and not target.is_symlink()
        if target.exists():shutil.rmtree(target)
        root.rename(target)
    shutil.rmtree(stage)
    trace.log('COMPLETED: published '+', '.join(str(outputs/t) for t in args.themes))


if __name__=='__main__':
    main()
