"""Additional native evidence, with pre-control lagging and actual visit counts."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from core import HERE, module, primary_lock


def event_metrics(events, decisions, actions):
    e = events.sort_values(['decision_id', 'event']).reset_index(drop=True).copy()
    info = decisions.set_index('decision_id')
    root = e.decision_id.map(info.root_value).to_numpy()
    before_n = e[[f'root_n_before_{a}' for a in range(actions)]].to_numpy(float)
    after_n = e[[f'root_n_after_{a}' for a in range(actions)]].to_numpy(float)
    before_q = e[[f'root_q_before_{a}' for a in range(actions)]].to_numpy(float)
    after_q = e[[f'root_q_after_{a}' for a in range(actions)]].to_numpy(float)
    changed = (after_n > before_n + 1e-7) & e.computation.to_numpy(bool)[:, None]
    first = changed & (before_n <= 1e-7)
    repeat = changed & (before_n > 1e-7)
    dq = np.where(repeat, after_q-before_q, np.nan)
    gain = np.where(first, after_q-root[:, None], np.nan)
    def mean(matrix):
        n = np.isfinite(matrix).sum(1)
        return np.divide(np.nansum(matrix, 1), n, out=np.full(len(matrix), np.nan), where=n>0)
    e['first_gain'] = mean(gain)
    e['repeat_delta'] = mean(dq)
    e['repeat_abs'] = mean(abs(dq))
    e['repeat_count'] = repeat.sum(1)
    e['first_count'] = first.sum(1)
    e['rollout_advantage'] = np.where(e.computation, e.rollout_return-root, np.nan)
    # This is a QC residual, not an assumed update equation used as evidence.
    expected = np.divide(before_n*np.nan_to_num(before_q) + e.rollout_return.to_numpy()[:, None],
                         after_n, out=np.full_like(after_n, np.nan), where=after_n>0)
    e['q_update_error'] = mean(np.where(changed & (abs(after_n-before_n-1)<1e-4), after_q-expected, np.nan))
    # Every control gets information from the previous accepted computation only.
    # No forward filling across decision, RESET, episode or source boundaries.
    for name in ('rollout_advantage', 'first_gain', 'repeat_delta', 'repeat_abs'):
        e['prior_'+name] = e.groupby('decision_id', sort=False)[name].shift(1)
        e['prior_'+name+'_available'] = e['prior_'+name].notna().astype(int)
        e['margin_x_prior_'+name] = e.root_margin*e['prior_'+name]
    extra = []
    for decision, rows in e.groupby('decision_id', sort=False):
        ix = rows.index.to_numpy()
        computations = rows[rows.computation.astype(bool)]
        r = dict(decision_id=decision, process_first_mean=rows.first_gain.mean(),
            process_signed_mean=rows.repeat_delta.mean(), process_abs_mean=rows.repeat_abs.mean(),
            process_abs_total=rows.repeat_abs.sum(min_count=1),
            process_valid_events=int(rows.repeat_delta.notna().sum()),
            process_first_events=int(rows.first_gain.notna().sum()),
            process_available=int(rows.repeat_delta.notna().any()),
            process_first_available=int(rows.first_gain.notna().any()))
        paths = []; current = []
        branch = {}
        for q in computations.itertuples():
            if len(current) != int(q.depth_pre):
                raise ValueError('Native path depth mismatch')
            path = [] if q.actor_reset and q.depth_post == 0 else current+[int(q.accepted_action)]
            if len(path) != int(q.depth_post):
                raise ValueError('Native path post-depth mismatch')
            branch[q.Index] = path[0] if path else -1
            paths.append(path)
            current = [] if q.actor_reset else path
        for a in range(actions):
            last = rows.iloc[-1]
            r[f'terminal_q_{a}'] = last[f'root_q_after_{a}']
            r[f'terminal_n_{a}'] = last[f'root_n_after_{a}']
            selected = [k for k, v in branch.items() if v == a]
            returns = e.loc[selected, 'rollout_advantage']
            r[f'rollout_mean_{a}'] = returns.mean()
            r[f'rollout_max_{a}'] = returns.max()
            r[f'event_visits_{a}'] = len(selected)
            finite = dq[ix, a][np.isfinite(dq[ix, a])]
            r[f'process_signed_{a}'] = finite.mean() if len(finite) else np.nan
            r[f'process_absolute_{a}'] = abs(finite).mean() if len(finite) else np.nan
            r[f'process_count_{a}'] = len(finite)
        extra.append(r)
    e['future_depth'] = e.groupby('decision_id').depth_post.transform(lambda s: s[::-1].cummax()[::-1])
    # It is an outcome used for selected-CONTINUE descriptive plots only.
    return e, pd.DataFrame(extra)


def derive(ctx):
    out = ctx.root/'derived'
    if (out/'manifest.json').exists():
        ctx.verify_seal(out)
        primary_lock(ctx)
        return
    out.mkdir(exist_ok=True)
    primary_lock(ctx)
    d = ctx.decisions(False)
    e = ctx.verified(ctx.cache/'search_events.csv.gz')
    if ctx.inputs[str(ctx.cache/'search_events.csv.gz')] != ctx.manifest['artifacts']['search_events.csv.gz']:
        raise ValueError('Changed V1 event cache')
    index = ctx.args.capture_index or ctx.manifest['inputs']['index']
    if ctx.C.sha(index) != ctx.manifest['inputs']['index_sha256']:
        raise ValueError('Capture index differs from V1')
    io = module('v2_native_io', HERE.parents[1]/'thinker/thinker/behavior_trace.py')
    e['rollout_return'] = np.nan
    examples = []
    forecasts = ctx.verified(ctx.cache/'forecasts.csv.gz')
    chosen = {}
    eligible = forecasts[(forecasts.target_id>=0) & (forecasts.depth>=2)]
    for label, mask in [('match', eligible.prefix_match), ('divergence', ~eligible.prefix_match)]:
        rows = eligible[mask].sort_values(['run', 'origin_time', 'event'])
        if len(rows): chosen[label] = rows.iloc[0]
    image_arrays = {}
    native_hashes = {}
    for episode, rows in d.groupby('capture_episode', sort=False):
        print('native supplement:', episode, flush=True)
        native = io.NativeVideoStat(episode)  # Full original integrity validation.
        native_hashes[episode] = ctx.C.digest(native.manifest)
        selected = e.episode.isin(rows.episode.unique())
        values = native.read_field_rows('tree_reps_post', e.loc[selected, 'event'].to_numpy(int))
        sl = slice(*native.tree_rep_schema['rollout_return'])
        e.loc[selected, 'rollout_return'] = values[:, sl][:, 0]
        for label, f in chosen.items():
            if f.episode not in set(rows.episode): continue
            path = json.loads(f.path)
            origin = rows.set_index('decision_id').loc[int(f.origin_id)]
            local = rows[rows.local.between(origin.local, origin.local+len(path))]
            source = forecasts[(forecasts.origin_id==f.origin_id) & (forecasts.event<=f.event)]
            # Use only actual events of this exact prefix; never other reset paths.
            source = source[source.path.map(lambda p: path[:len(json.loads(p))] == json.loads(p))]
            source = source.drop_duplicates('depth', keep='last').sort_values('depth')
            image_arrays[label+'_real'] = native.read_field_rows('real_image', local.pre_image_event.to_numpy(int))
            image_arrays[label+'_predicted'] = native.read_field_rows('imagined_image', source.event.to_numpy(int))
            examples.append(dict(label=label, run=f.run, episode=f.episode, origin_time=f.origin_time,
                path=path, real_actions=local.human_action.astype(int).tolist(),
                prediction_depths=source.depth.astype(int).tolist(), selection='chronological first depth>=2; not effect-selected'))
    e, extra = event_metrics(e, d, ctx.actions)
    e.to_csv(out/'events.csv.gz', index=False)
    extra.to_csv(out/'decision_extra.csv.gz', index=False)
    np.savez_compressed(out/'path_images.npz', **image_arrays)
    ctx.C.write_json(out/'path_examples.json', examples)
    ctx.C.write_json(out/'qa.json', dict(decisions=len(d), events=len(e),
        budget0=int((d.budget==0).sum()), first_events=int(e.first_gain.notna().sum()),
        repeat_events=int(e.repeat_delta.notna().sum()), forced=int(e.forced_terminal.sum()),
        reset=int((e.control==1).sum()), stop=int(e.stop_choice.sum()),
        evidence_units='return / Q units; root evaluation excluded from budget',
        control_availability='previous accepted computation only; no terminal budget',
        native_manifest_hashes=native_hashes))
    ctx.seal(out)
