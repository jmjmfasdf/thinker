"""Paired STOP/CONTINUE utility from explicitly supplied, verified evaluations.

This adapter never creates environment rollouts or score evaluations. Supply
--utility-input DIRECTORY (utility_states.csv plus manifest.json) or the CSV
itself. Required CSV columns: game, protocol, nominal_step, seed, stream,
predicted_voc, utility, task_reward_difference, computation_cost. Subject must
be in the table or explicitly declared by manifest.subject/manifest.subjects.
The manifest must retain nonempty utility_sources with hashed path/pairs_path
records from the original paired evaluation. This matches the previous pilot
provenance contract; missing material becomes BLOCKED_INPUT.

analyze(args,subject,game,prepared,workdir) -> (tables,qa,arrays)
render(root,tables,title) uses only completed numeric tables, never raw inputs.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats

from common.io import sha

GAMES = {'enduro': 0, 'pong': 1, 'spaceinvaders': 2}
REQUIRED = ['game', 'protocol', 'nominal_step', 'seed', 'stream', 'predicted_voc',
            'utility', 'task_reward_difference', 'computation_cost']


def _blocked(reason, **details):
    return {}, dict(status='BLOCKED_INPUT', reason=reason,
                    environment_evaluation_launched=False, score_evaluation_launched=False,
                    **details), {}


def _game(value):
    value = str(value).strip().lower().replace(' ', '').replace('_', '').replace('-', '')
    if value in ('0', '1', '2'):
        return {v:k for k,v in GAMES.items()}[int(value)]
    if value not in GAMES:
        raise ValueError('Unknown paired-evaluation game: ' + value)
    return value


def _subject(value):
    text = str(value).lower().replace('subject', '').replace('sub', '').replace('-', '').replace('_', '')
    return int(text)


def _resolve(base, name):
    path = Path(name)
    return path if path.is_absolute() else base/path


def _regression(d):
    """Original slope and cluster bootstrap, preserving seed×stream dependence."""
    rng = np.random.default_rng(20261111)
    rows = []
    for (game, protocol, step), whole in d.groupby(['game', 'protocol', 'nominal_step'], sort=True):
        part = whole.dropna(subset=['predicted_voc', 'utility']).copy()
        outcomes = dict(n_outcome_records=len(whole), utility_mean=float(whole.utility.mean()),
                        mean_task_reward_difference=float(whole.task_reward_difference.mean()),
                        mean_computation_cost=float(whole.computation_cost.mean()),
                        positive_utility_fraction=float((whole.utility > 0).mean()))
        if not len(part):
            rows.append(dict(game=game, protocol=protocol, nominal_step=step, status='no_finite_pairs',
                             n_records=0, n_streams=0, utility_fit_subset_mean=np.nan,
                             utility_voc_slope=np.nan, slope_ci_low=np.nan, slope_ci_high=np.nan,
                             **outcomes))
            continue
        part['cluster'] = part.seed.astype(str)+'_'+part.stream.astype(str)
        x, y = part.predicted_voc.to_numpy(float), part.utility.to_numpy(float)
        sd, mean = x.std(), x.mean()
        slope = float(np.linalg.lstsq(np.column_stack([np.ones(len(x)), (x-mean)/sd]), y, rcond=None)[0][1]) if sd > 1e-10 else np.nan
        clusters = [p.index.to_numpy() for _,p in part.groupby('cluster')]
        estimates = []
        if sd > 1e-10 and len(clusters) > 1:
            for _ in range(1000):
                sample = part.loc[np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])]
                xx = (sample.predicted_voc.to_numpy(float)-mean)/sd
                if xx.std() > 1e-10:
                    estimates.append(np.linalg.lstsq(np.column_stack([np.ones(len(xx)), xx]), sample.utility, rcond=None)[0][1])
        ci = np.quantile(estimates, [.025, .975]) if estimates else [np.nan, np.nan]
        rows.append(dict(game=game, protocol=protocol, nominal_step=step, n_records=len(part),
                         n_states=int(part.state_id.nunique()) if 'state_id' in part else len(part),
                         n_streams=len(clusters), utility_fit_subset_mean=float(y.mean()),
                         utility_voc_slope=slope, slope_ci_low=float(ci[0]), slope_ci_high=float(ci[1]),
                         spearman=float(stats.spearmanr(x,y).statistic) if sd > 1e-10 and y.std() > 1e-10 else np.nan,
                         status='ok' if sd > 1e-10 and len(clusters)>1 else 'constant_VOC' if sd <= 1e-10 else 'one_stream_no_interval',
                         **outcomes))
    return pd.DataFrame(rows)


def analyze(args, subject, game, prepared, workdir):
    """Read the explicitly selected paired evaluation; no capture fallback."""
    supplied = getattr(args, 'utility_input', None)
    if supplied is None:
        return _blocked('Paired environment STOP/CONTINUE outcomes are required; human replay cannot supply utility.',
                        required_argument='--utility-input DIRECTORY_OR_utility_states.csv')
    supplied = Path(supplied)
    csv = supplied/'utility_states.csv' if supplied.is_dir() else supplied
    manifest_path = csv.parent/'manifest.json'
    if not csv.is_file() or not manifest_path.is_file():
        return _blocked('Paired utility table or provenance manifest is missing', required=[str(csv),str(manifest_path)])
    manifest = json.loads(manifest_path.read_text())
    sources = manifest.get('utility_sources', [])
    if not sources:
        return _blocked('Paired evaluation source provenance is absent', required='manifest.utility_sources with paired source hashes')
    verified = []
    for rec in sources:
        for pk,hk in [('path','sha256'), ('pairs_path','pairs_sha256')]:
            if not rec.get(pk) or not rec.get(hk):
                return _blocked('Incomplete paired source provenance', required=[pk,hk])
            source = _resolve(csv.parent, rec[pk])
            if not source.is_file():
                return _blocked('An original paired-evaluation source is unavailable', required=str(source))
            if sha(source) != rec[hk]:
                raise ValueError('Paired evaluation source hash mismatch: ' + str(source))
            verified.append(dict(path=str(source), sha256=rec[hk]))
    for field in ['artifacts','source_hashes']:
        expected = manifest.get(field, {}).get(csv.name)
        if isinstance(expected, dict):
            expected = expected.get('sha256')
        if expected and sha(csv) != expected:
            raise ValueError('Paired utility table hash differs from manifest')
    d = pd.read_csv(csv)
    missing = sorted(set(REQUIRED)-set(d))
    if missing:
        return _blocked('Paired utility table lacks required fields', required_columns=missing)
    if 'subject' in d:
        source_subjects = sorted(set(d.subject.map(_subject)))
        d = d.loc[d.subject.map(_subject).eq(int(subject))].copy()
    else:
        subjects = manifest.get('subjects', [manifest['subject']] if 'subject' in manifest else [])
        subjects = [_subject(s) for s in subjects]
        if len(subjects) != 1:
            return _blocked('Utility rows lack unambiguous participant identity', required='subject column or manifest.subject')
        if subjects[0] != int(subject):
            return _blocked('Paired evaluation belongs to another participant', available_subjects=subjects)
        source_subjects = subjects
        d['subject'] = int(subject)
    d['game'] = d.game.map(_game)
    game = _game(game)
    d = d.loc[d.game.eq(game)].copy().reset_index(drop=True)
    if getattr(args,'sample_mode','all') == '15hz':
        from common.sampling import KEYS
        if not set(KEYS).issubset(d) or manifest.get('capture_manifest_sha256') != prepared['metadata']['manifest_sha256']:
            return _blocked('15-Hz utility selection requires episode/real_event keys and capture_manifest_sha256 matching the current capture; independent paired trials cannot be resampled as human replay')
        keys = pd.MultiIndex.from_frame(prepared['decisions'][KEYS])
        d = d.loc[pd.MultiIndex.from_frame(d[KEYS]).isin(keys)].copy().reset_index(drop=True)
    if d.empty:
        return _blocked('No paired outcomes for the requested participant/game', subject=int(subject), game=game)
    numeric = ['utility','task_reward_difference','computation_cost','nominal_step']
    if not np.isfinite(d[numeric].to_numpy(float)).all() or (d.computation_cost < 0).any():
        raise ValueError('Nonfinite paired outcomes or negative computation cost')
    if not np.allclose(d.utility, d.task_reward_difference-d.computation_cost, atol=1e-10, rtol=1e-7):
        raise ValueError('Utility decomposition mismatch')
    if np.isinf(d.predicted_voc.to_numpy(float)).any():
        raise ValueError('Infinite prechoice VOC')
    summary = _regression(d)
    tables = {'state_outcomes':d, 'voc_utility_regression':summary}
    scores = csv.parent/'utility_scores.csv'
    score_scope = 'not supplied'
    if scores.is_file():
        old = pd.read_csv(scores)
        if 'game' in old:
            old = old.loc[old.game.map(_game).eq(game)].copy()
            if 'subject' in old:
                old = old.loc[old.subject.map(_subject).eq(int(subject))]
                score_scope = 'explicit participant column'
            elif source_subjects != [int(subject)]:
                old = old.iloc[:0]
                score_scope = 'omitted: participant identity absent from multi-participant source'
            else:
                score_scope = 'single participant in original paired source'
            tables['previous_pilot_scores'] = old
        else:
            score_scope = 'omitted: game identity absent'
    qa = dict(status='COMPLETED', scientific_scope='explicit paired-evaluation input; not automatically the current native capture checkpoint',
              input=str(csv), subject=int(subject), game=game,
              source_hashes={csv.name:sha(csv),manifest_path.name:sha(manifest_path)}, verified_paired_sources=verified,
              rows=len(d), rows_missing_predicted_VOC=int(d.predicted_voc.isna().sum()),
              outcome_summary_scope='all valid paired outcomes, including rows with missing predicted VOC',
              previous_score_scope=score_scope,
              nominal_steps=sorted(map(float,d.nominal_step.unique())), protocols=sorted(d.protocol.astype(str).unique()),
              protocol_metadata={k:manifest[k] for k in ['scope','protocol','horizon','discounting','checkpoint','checkpoints','temperature','training_condition'] if k in manifest},
              current_checkpoint_binding='not established by this adapter; inspect explicit paired provenance',
              independent_resampling='seed x environment stream, 1000 bootstrap draws; paired observations retained together',
              state_count_definition='unique state_id' if 'state_id' in d else 'input rows; no state identifier supplied',
              environment_evaluation_launched=False, score_evaluation_launched=False,
              limitations=['Finite-horizon outcomes need not equal the long-horizon critic target.',
                           'No claim of six-subject coverage from a single-participant pilot.',
                           'Human replay and old paired pilots do not establish utility of a different trained checkpoint.'])
    if scores.is_file():
        qa['source_hashes'][scores.name]=sha(scores)
    return tables, qa, {}


def render(root, tables, title):
    """Render from the persisted numeric tables only; no raw evaluation access."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = tables.get('voc_utility_regression', pd.DataFrame())
    if summary.empty:
        return []
    root = Path(root)
    figures = root/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    for protocol,part in summary.groupby('protocol'):
        part=part.sort_values('nominal_step')
        step=part.nominal_step.to_numpy(float)/1e6
        for ax,column in zip(axes,['mean_task_reward_difference','utility_mean','utility_voc_slope']):
            if column in part:
                ax.plot(step,part[column],'o-',label=str(protocol))
        if {'slope_ci_low','slope_ci_high'}.issubset(part):
            axes[2].fill_between(step,part.slope_ci_low.to_numpy(float),part.slope_ci_high.to_numpy(float),alpha=.15)
    for ax,ylabel in zip(axes,['Paired reward difference','Net utility after computation cost','Utility / one-SD predicted VOC']):
        ax.axhline(0,color='gray',lw=.7)
        ax.set(xlabel='Training steps (millions)',ylabel=ylabel)
        ax.legend(frameon=False)
    fig.suptitle(title+' · paired evaluation input')
    fig.tight_layout()
    paths=[]
    for ext in ['png','pdf']:
        path=figures/('paired_utility.'+ext)
        fig.savefig(path,dpi=160,bbox_inches='tight');paths.append(str(path))
    plt.close(fig)
    return paths
