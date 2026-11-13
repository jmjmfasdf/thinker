"""Descriptive stopping audit and reanalysis of existing paired utility trials.

Never trains, replays, or launches evaluations. Human replay and environment
interventions have separate sampling populations and are never pooled.
"""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs'
CACHE = OUTPUTS / '00_dynamic_cache_10m_20261110'
CANDIDATES = OUTPUTS / '02/value_update_candidates_20261111'
CAMPAIGN = OUTPUTS / 'voc10m_tau001_20260926'
GAMES = {0: 'enduro', 1: 'pong', 2: 'spaceinvaders'}
KEY = ['seed', 'stream', 'prefix', 'count']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def describe_native(arg):
    subject, game = arg
    tag = f'sub{subject:03d}_game{game}'
    source = CANDIDATES / tag / 'candidate_decisions.csv.gz'
    d = pd.read_csv(source, usecols=['real_step_id', 'session', 'run_label',
        'scored', 'incumbent_action', 'additional_computations', 'valid_update_count'])
    d = d[d.scored.eq(1) & d.incumbent_action.ge(0)].copy()
    cols = ['real_step_id', 'additional_computations', 'forced_stop',
        'stop_choice_censored', 'voluntary_stop_observed',
        'voc_ema_voc_initial', 'voc_ema_voc_initial_valid',
        'voc_ema_voc_terminal', 'voc_ema_voc_terminal_valid',
        'voc_soft_continue_probability_initial',
        'voc_soft_continue_probability_initial_valid',
        'voc_soft_continue_probability_terminal']
    parts = []
    sources = {str(source): sha(source)}
    for cp in range(1, 5):
        p = CACHE / f'checkpoint_leave{cp:02d}' / tag / 'native_decisions.csv.gz'
        z = pd.read_csv(p, usecols=cols)
        z = d.loc[d.session.eq(cp)].merge(z, on='real_step_id', how='left',
            validate='one_to_one', suffixes=('', '_native'), indicator=True)
        assert z._merge.eq('both').all()
        assert z.additional_computations.eq(z.additional_computations_native).all()
        assert z.voc_ema_voc_initial_valid.eq(1).all()
        assert z.voc_soft_continue_probability_initial_valid.eq(1).all()
        parts.append(z)
        sources[str(p)] = sha(p)
    d = pd.concat(parts, ignore_index=True)
    assert not d.real_step_id.duplicated().any()
    rows = []
    for run, z in d.groupby('run_label'):
        b = z.additional_computations
        q = z.voc_ema_voc_initial
        p = z.voc_soft_continue_probability_initial
        start = b.gt(0)
        voluntary = z.voluntary_stop_observed.eq(1) & z.forced_stop.eq(0) & z.stop_choice_censored.eq(0)
        terminal = z.loc[voluntary & z.voc_ema_voc_terminal_valid.eq(1), 'voc_ema_voc_terminal']
        row = dict(subject=subject, game=game, run_label=run, n=len(z),
            zero_fraction=b.eq(0).mean(), one_fraction=b.eq(1).mean(),
            mean_budget=b.mean(), start_fraction=start.mean(),
            mean_budget_given_start=b[start].mean(),
            initial_negative_fraction=q.lt(0).mean(), initial_voc_mean=q.mean(),
            initial_voc_median=q.median(),
            initial_continue_probability=p.mean(),
            initial_sigmoid_max_abs_error=float(np.max(np.abs(p-expit(q/.01)))),
            no_search_negative_voc_fraction=q[~start].lt(0).mean(),
            no_search_positive_voc_fraction=q[~start].gt(0).mean(),
            no_search_near_cost_negative_fraction=q[~start].between(-.0005, 0, inclusive='left').mean(),
            voluntary_stop_fraction=voluntary.mean(),
            stop_negative_fraction=terminal.lt(0).mean(),
            stop_near_cost_negative_fraction=terminal.between(-.0005, 0, inclusive='left').mean(),
            stop_below_minus_tau_fraction=terminal.lt(-.01).mean(),
            stop_voc_median=terminal.median(),
            revised_given_start=z.loc[start, 'valid_update_count'].gt(0).mean())
        for tau in [.01, .025, .05]:
            row[f'fixed_state_start_p_tau{tau}'] = expit(q/tau).mean()
        for bias in [.0005, .005, .01]:
            row[f'fixed_state_start_p_bias{bias}'] = expit((q+bias)/.01).mean()
        for sign, mask in [('positive', q.gt(0)), ('negative', q.lt(0))]:
            row[f'start_given_{sign}'] = start[mask].mean()
            row[f'expected_start_given_{sign}'] = p[mask].mean()
        rows.append(row)
    return rows, sources


def bootstrap_means(frame, values, iterations=3000):
    """Resample entire environment streams, retaining prefixes and repetitions.

    Input outcomes have already averaged continuation repetitions per state.
    Intervals are exploratory, conditional on this trained checkpoint, and not
    corrected across endpoints or stages.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if not len(frame):
        return np.full((3, values.shape[1]), np.nan)
    labels = pd.factorize(pd.MultiIndex.from_frame(frame[['seed','stream']]))[0]
    groups = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    means = values.mean(axis=0)
    if len(groups) < 2:
        return np.vstack([means, np.full_like(means, np.nan), np.full_like(means, np.nan)])
    rng = np.random.default_rng(20261111)
    draws = np.array([values[np.concatenate([groups[i] for i in
        rng.integers(len(groups), size=len(groups))])].mean(axis=0) for _ in range(iterations)])
    return np.vstack([means, np.quantile(draws, [.025,.975], axis=0)])


def paired_trials(dest):
    registry_path = CAMPAIGN / 'jobs.json'
    registry = json.loads(registry_path.read_text())['evaluation']
    sources = {str(registry_path): sha(registry_path)}
    state_rows, summary_rows, sensitivity_rows, checks = [], [], [], []
    for game_id, game in GAMES.items():
        for step in [2_000_000,4_000_000,6_000_000,8_000_000,10_000_000]:
            key = f'v10t001-20260926-{game}_s701_{step:08d}_benefit'
            record = registry[key]
            directory = Path(record.get('output') or CAMPAIGN / 'evaluations' / key)
            sp, pp = directory / 'summary.json', directory / 'pairs.json'
            if record['state'] != 'COMPLETED' or not sp.exists():
                checks.append(dict(game=game, step=step, status='MISSING', path=str(directory)))
                continue
            summary = json.loads(sp.read_text())
            assert summary['status'] == 'COMPLETE'
            raw = pd.DataFrame(json.loads(pp.read_text()))
            assert not raw.duplicated(KEY+['repetition']).any()
            assert np.allclose(raw.delta, raw.task_delta-raw.cost, atol=1e-10, rtol=0)
            for _, z in raw.groupby(KEY):
                assert z.entry_hash.nunique() == 1
                assert np.ptp(z.ema) < 1e-10 and np.ptp(z.probability) < 1e-10
            states = raw.groupby(KEY, as_index=False).agg(
                voc=('ema','mean'), probability=('probability','mean'),
                utility=('delta','mean'), reward_delta=('task_delta','mean'),
                cost=('cost','mean'), repeats=('delta','size'),
                tail_bound=('tail_bound','max'),
                positive_repeats=('delta',lambda x:int((x>1e-10).sum())))
            states['game'], states['step'] = game, step
            assert len(states) == summary['voc']['states']
            assert np.isclose(states.utility.mean(), summary['voc']['mean_net_benefit'])
            sigmoid_error = float(np.max(np.abs(states.probability-(.01+.98*expit(states.voc/.01)))))
            match = False
            if step == 10_000_000:
                manifest = CACHE / f'checkpoint_leave01/sub001_game{game_id}/manifest.json'
                hashes = json.loads(manifest.read_text())['captures'][0]['checkpoint_hashes']
                files = summary['checkpoint']['files']
                match = all(files[name]['sha256'] == hashes[f'checkpoint_{name}_sha256']
                    for name in ['actor','model'])
                assert match
                sources[str(manifest)] = sha(manifest)
            checks.append(dict(game=game, step=step, status='COMPLETE',
                checkpoint_matches_current_cache=match if step==10_000_000 else None,
                gate_probability_formula_max_error=sigmoid_error,
                states=len(states), pairs=len(raw), path=str(directory)))
            state_rows.append(states)
            for stage, stage_mask in [('all',np.ones(len(states),dtype=bool)),
                                      ('start_k0',states['count'].eq(0)),
                                      ('after_k3',states['count'].eq(3))]:
                for sign, mask in [('all',np.ones(len(states),dtype=bool)),
                                   ('negative',states.voc.lt(0)),('positive',states.voc.gt(0))]:
                    z = states.loc[stage_mask & mask]
                    if not len(z):
                        continue
                    bs = bootstrap_means(z, z[['utility','reward_delta','cost']])
                    row = dict(game=game, step=step, stage=stage, voc_sign=sign,
                        states=len(z), streams=len(z[['seed','stream']].drop_duplicates()),
                        positive_mean_utility_states=int(z.utility.gt(1e-10).sum()),
                        positive_beyond_tail_states=int(z.utility.gt(z.tail_bound).sum()),
                        negative_mean_utility_states=int(z.utility.lt(-1e-10).sum()),
                        mean_predicted_voc=z.voc.mean(), mean_continue_probability=z.probability.mean(),
                        max_tail_bound=z.tail_bound.max())
                    for j,name in enumerate(['utility','reward_delta','cost']):
                        row.update({name+'_mean':bs[0,j],name+'_ci_low':bs[1,j],name+'_ci_high':bs[2,j]})
                    summary_rows.append(row)
                z = states.loc[stage_mask]
                # Only the probability at the paired entry is changed. Subsequent
                # actions/continuation policies remain the original experiment's.
                for label,tau,bias in [('tau025',.025,0),('tau05',.05,0),
                    ('bias0005',.01,.0005),('bias005',.01,.005),('bias01',.01,.01)]:
                    newp = .01+.98*expit((z.voc+bias)/tau)
                    gain = (newp-z.probability)*z.utility
                    bs = bootstrap_means(z,gain)
                    sensitivity_rows.append(dict(game=game,step=step,stage=stage,change=label,
                        states=len(z),entry_probability_delta=(newp-z.probability).mean(),
                        local_gain=bs[0,0],ci_low=bs[1,0],ci_high=bs[2,0]))
            sources[str(sp)], sources[str(pp)] = sha(sp), sha(pp)
    pd.concat(state_rows,ignore_index=True).to_csv(dest/'paired_state_outcomes.csv',index=False)
    pd.DataFrame(summary_rows).to_csv(dest/'paired_sign_summary.csv',index=False)
    pd.DataFrame(sensitivity_rows).to_csv(dest/'paired_local_sensitivity.csv',index=False)
    return sources, checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUTS/'10/voc_stopping_audit_20261111')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source_hash=sha(__file__)
    shutil.copyfile(__file__,args.output/f'source_{source_hash[:12]}.py')
    protocol=dict(native='Held-out session only; same scored decision identities as value update analysis; equal run then subject means.',
        paired='All completed standard sub001 tau=.01 milestones. Repetitions averaged within state, bootstrap by seed and environment stream.',
        interpretation='Exploratory descriptive audit. No causal conclusion from human replay; paired utility is discounted task reward minus current-stage cost, not all future cost. One checkpoint seed per game. No multiple-testing-adjusted significance claims.',
        sensitivity='Frozen-state probability sensitivity, and local paired entry reweighting with original future policy. Neither is a retrained policy or complete rollout of a modified gate. A +0.0005 bias is not equivalent to removing learned computation cost.',
        source_sha256=source_hash)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows,sources=[],{}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rr,ss in pool.map(describe_native,[(s,g) for s in range(1,7) for g in range(3)]):
            rows.extend(rr);sources.update(ss)
            print(f'native complete: {rr[0]["subject"]} {GAMES[rr[0]["game"]]}',flush=True)
    runs=pd.DataFrame(rows)
    cols=[c for c in runs if c not in ['subject','game','run_label','n']]
    subjects=runs.groupby(['subject','game'],as_index=False)[cols].mean()
    group=subjects.groupby('game',as_index=False)[cols].mean()
    for name,data in [('native_runs',runs),('native_subjects',subjects),('native_group',group)]:
        data.to_csv(args.output/(name+'.csv'),index=False)
    ss,checks=paired_trials(args.output);sources.update(ss)
    qa=dict(status='PASSED',native_decisions=int(runs.n.sum()),native_runs=len(runs),
        native_subject_games=len(subjects),native_max_sigmoid_error=float(runs.initial_sigmoid_max_abs_error.max()),
        paired_checks=checks)
    assert len(subjects)==18
    assert qa['native_max_sigmoid_error'] < .01
    assert len(checks)==15 and all(c['status']=='COMPLETE' for c in checks)
    assert all(c['gate_probability_formula_max_error'] < .01 for c in checks)
    (args.output/'qa.json').write_text(json.dumps(qa,indent=2))
    (args.output/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',
        source_sha256=source_hash,sources=sources,
        outputs={p.name:sha(p) for p in args.output.iterdir() if p.is_file()}),indent=2))
    print('COMPLETED',args.output,flush=True)


if __name__=='__main__':
    main()
