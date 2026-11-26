"""Latest completed captures and verified reusable scalar preparation (theme 00).

Preparation is deliberately independent of hypotheses and statistical fitting.
Old leave-session inputs are accepted only by the explicit comparison runner.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import numpy as np
import pandas as pd
from .paths import REPO, OUTPUTS
from .result_store import ResultStore
from .io import signature

DEFAULT_INPUT = REPO/'logs/visualization/hxln_10m'
GAMES = ('pong', 'enduro', 'spaceinvaders')
VERSION = 'native-preparation-v2'

class BlockedInput(FileNotFoundError):
    pass


def sha(path):
    from .native_replay import sha as f
    return f(path)


def read(path):
    return json.loads(Path(path).read_text())


def capture_metadata(input_root, subject, game):
    root = Path(input_root)/f'sub{subject:03d}'/game
    if not (root/'COMPLETED.json').is_file():
        raise BlockedInput(f'No completed visualization: {root}')
    m, done = read(root/'manifest.json'), read(root/'COMPLETED.json')
    from .native_replay import digest
    unsigned = dict(m); claim = unsigned.pop('manifest_sha256')
    if done['status'] != 'COMPLETED' or digest(unsigned) != claim or done['manifest_sha256'] != claim:
        raise ValueError('Capture completion or manifest integrity failed')
    if digest(m['episodes']) != m['episodes_sha256'] or digest(m['semantic']) != m['semantic_sha256']:
        raise ValueError('Capture semantic/episode digest mismatch')
    plan = read(root/'capture_plan.json')
    if plan['semantic_sha256'] != m['semantic_sha256']:
        raise ValueError('Capture plan differs from completed capture')
    return dict(capture=str(root.resolve()), manifest_sha256=claim,
                checkpoint=plan['semantic']['checkpoint'], episodes=m['episodes'],
                semantic_sha256=m['semantic_sha256'], dataset='latest_native_10m')


def aliases(d):
    d = d.copy()
    b = d.budget.to_numpy(float)
    if np.any(b < 0): raise ValueError('Negative computation count')
    d['A_V'] = d.V_total
    d['U_V'] = np.divide(d.V_total, b, out=np.zeros(len(d)), where=b > 0)
    d['D_V'] = np.where(b > 0, d.V_final-d.V_initial, 0.)
    d['A_Q'] = d.Q_total
    d['U_Q'] = np.divide(d.Q_total, b, out=np.zeros(len(d)), where=b > 0)
    if not d.loc[b == 0, ['A_V','A_Q']].eq(0).all().all():
        raise ValueError('Nonzero within-imagination update at zero budget')
    if 'first_abs_dV' in d and np.any(b == 1):
        np.testing.assert_allclose(d.loc[b == 1,'A_V'], d.loc[b == 1,'first_abs_dV'])
    if not d.loc[~d.Q_supported.astype(bool), 'A_Q'].eq(0).all():
        raise ValueError('Unobserved Q revisions must have explicit zero total and false coverage')
    return d


def scalar_cache_candidates(args, subject, game):
    suffix = Path(f'sub{subject:03d}')/game
    paths = [Path(args.outputs_root)/'00/replay_features'/suffix,
             OUTPUTS/'00/replay_features'/suffix,
             REPO/'legacy/research_script/pre_reorganization/outputs/00/replay_features'/suffix]
    if getattr(args, 'scalar_cache', None): paths.insert(0, Path(args.scalar_cache)/suffix)
    return list(dict.fromkeys(paths))


def verified_scalar_cache(args, subject, game, metadata):
    for path in scalar_cache_candidates(args, subject, game):
        marker = path/'manifest.json'
        if not marker.is_file(): continue
        old = read(marker)
        if (old.get('version') != 'native-replay-themes-v1' or old.get('status') != 'COMPLETED'
            or old.get('subject') != subject or old.get('game') != game
            or old.get('manifest_sha256') != metadata['manifest_sha256']
            or old.get('checkpoint') != metadata['checkpoint']): continue
        if not all((path/f).is_file() and sha(path/f) == h for f,h in old['artifacts'].items()):
            raise ValueError(f'Changed scalar cache artifact: {path}')
        # This is an explicit import of the audited v1 same-phase implementation,
        # not a claim that its old code signature equals the refactored source.
        old['import_note'] = 'Verified v1 artifacts; same-phase V/Q semantics retained; new aliases separately checked'
        return path, old
    return None, None


def validate_source_bindings(metadata, qa):
    capture = Path(metadata['capture'])
    expected = {q['episode']:q['manifest_sha256'] for q in qa.get('episodes', [])}
    for ep in metadata['episodes']:
        p = capture/ep['path']/'manifest.json'
        record = read(p)
        if record['metadata'] != ep['metadata'] or record['files_sha256'] != ep['files_sha256']:
            raise ValueError(f'Changed episode metadata: {p}')
        if record['metadata']['episode_id'] in expected and sha(p) != expected[record['metadata']['episode_id']]:
            raise ValueError(f'Changed episode manifest: {p}')
    for p, digest in qa.get('sources', {}).items():
        if sha(p) != digest: raise ValueError(f'Changed human behavioral source: {p}')


def prepare_native(args, subject, game):
    meta = capture_metadata(args.input_root, subject, game)
    root = Path(args.outputs_root)/'00'/f'sub{subject:03d}'/game
    identity = dict(dataset=meta['dataset'], subject=subject, game=game, manifest_sha256=meta['manifest_sha256'])
    sig = signature(dict(version=VERSION, code=sha(__file__), manifest=meta['manifest_sha256']))
    with ResultStore(root, identity, resume=True) as store:
        preparation = None
        with store.stage('prepare', sig) as stage:
            if not stage.reused:
                cached, old = verified_scalar_cache(args, subject, game, meta)
                with tempfile.TemporaryDirectory(prefix='research-prepare-') as temporary:
                    if cached is None:
                        from analyses.replay_themes import extract_unit
                        cached = Path(extract_unit(args.input_root, Path(temporary), subject, game))
                        old = read(cached/'manifest.json')
                    validate_source_bindings(meta, old['qa'])
                    d = aliases(pd.read_csv(cached/'decisions.csv.gz'))
                    e = pd.read_csv(cached/'choices.csv.gz')
                    stage.write_table('decisions', d)
                    stage.write_table('choices', e)
                    preparation = dict(metadata=meta, qa=old['qa'],
                        imported_signature=old['signature'], import_note=old.get('import_note'))
        if preparation is not None: store.meta['preparation'] = preparation
        # 00 contains preparation only; STAGE_COMPLETE is a successful 00 result.
        store.publish(required_stages=('prepare',))
    saved = read(root/'logs/run.json')['preparation']
    validate_source_bindings(meta, saved['qa'])
    return dict(decisions=pd.read_csv(root/'tables/decisions.csv.gz'),
                choices=pd.read_csv(root/'tables/choices.csv.gz'), metadata=meta,
                qa=saved['qa'], config=saved['qa']['config'], cache=root)


def enrich_decisions(d, metadata):
    """Pre-choice visual/history controls and recorded terminal policy/tree.

    No fit, temporal interpolation or cross-episode lag is performed here.
    Columns read after search are explicitly labelled terminal and never put
    into the pre-search reference block. recent10/recent30 are legacy names for
    counts of strictly preceding raw switches in0.5/1.5-second windows, not ten
    or thirty replay steps. History resets at first/terminal/clock/gap boundaries.
    visual_change is the immediately preceding raw-frame difference within the
    same source segment; it is missing at segment onsets. Unvisited Q is missing.
    """
    from analyses.replay_themes import PackedTrace, verify_packs
    from scipy.special import softmax
    d = d.copy().reset_index(drop=True)
    raw_cache = {}
    episodes = {q['metadata']['episode_id']:q for q in metadata['episodes']}
    for episode, part in d.groupby('episode', sort=False):
        ep = episodes[episode]
        t = PackedTrace(Path(metadata['capture'])/ep['path'])
        verify_packs(t, ep)
        ix = part.index
        left = part.root_event.to_numpy(int); right = part.real_event.to_numpy(int)
        post = t.read('tree_reps_post').astype(float)
        pre = t.read('tree_reps_pre').astype(float)
        field = lambda x,k: x[:,slice(*t.tree_schema[k])]
        prior = softmax(field(post[left], 'root_policy'), axis=1)
        actual = t.read('actor_action_probability',right)
        for a in range(prior.shape[1]):
            d.loc[ix,f'log_policy_{a}'] = np.log(prior[:,a].clip(1e-12))
            d.loc[ix,f'actor_action_probability_{a}'] = actual[:,a]
            for key in ('root_qs_mean','root_ns'):
                initial = field(post[left],key)[:,a]
                terminal_q = field(pre[right],key)[:,a]
                if key == 'root_qs_mean':
                    initial = np.where(field(post[left],'root_ns')[:,a] > 0, initial, np.nan)
                    terminal_q = np.where(field(pre[right],'root_ns')[:,a] > 0, terminal_q, np.nan)
                d.loc[ix,f'{key}_initial_{a}'] = initial
                d.loc[ix,f'{key}_terminal_{a}'] = terminal_q
        d.loc[ix,'n_actions'] = prior.shape[1]
        visits = field(post[left],'root_ns') > 0
        initial_q = np.where(visits, field(post[left],'root_qs_mean'), -np.inf)
        top = np.sort(initial_q, axis=1)[:,-2:]
        margin = np.full(len(ix), np.nan)
        supported = visits.sum(1) >= 2
        margin[supported] = top[supported,1] - top[supported,0]
        d.loc[ix,'initial_q_margin'] = margin
        raw_path = Path(t.meta['source_path'])
        if str(raw_path) not in raw_cache:
            if sha(raw_path) != t.meta['source_sha256']: raise ValueError('Changed raw source')
            with np.load(raw_path, allow_pickle=False) as raw:
                action = raw['action'].argmax(1) if raw['action'].ndim == 2 else raw['action'].copy()
                times = raw['time'].copy().astype(float).reshape(-1)
                img = raw['image'].copy()
                if img.ndim == 4 and img.shape[-1] in (1,3,4): img = img.mean(axis=-1)
                elif img.ndim == 4: img = img.mean(axis=1)
                # Fixed 7x7 nuisance; neither learned nor fitted on held-out sessions.
                ys = np.array_split(np.arange(img.shape[-2]),7); xs = np.array_split(np.arange(img.shape[-1]),7)
                visual = np.column_stack([img[:,y][:,:,x].mean(axis=(1,2)) for y in ys for x in xs])/255.
                terminal = raw['is_terminal'].reshape(-1).astype(bool)
                first = raw['is_first'].reshape(-1).astype(bool)
                dt = np.diff(times); positive = dt[dt > 0]
                gap = max(.25, 10*float(np.median(positive))) if len(positive) else .25
                boundary = first.copy(); boundary[0] = True
                boundary[1:] |= terminal[:-1] | (dt <= 0) | (dt > gap)
                segments = np.cumsum(boundary)
                starts = np.maximum.accumulate(np.where(boundary,np.arange(len(times)),0))
                raw_switch = np.r_[False,action[1:] != action[:-1]]
                raw_switch[boundary] = False
                cumulative = np.r_[0,np.cumsum(raw_switch.astype(int))]
                progress = np.cumsum(np.r_[0,np.where(boundary[1:],0,dt)])
                progress /= max(float(progress[-1]),1e-9)
                raw_cache[str(raw_path)] = action,times,visual,terminal,boundary,starts,cumulative,progress
        actions,times,visual,terminal,boundary,starts,cumulative,progress = raw_cache[str(raw_path)]
        frames = part.source_frame.to_numpy(int)
        if not np.array_equal(actions[frames],part.human_action): raise ValueError('Human action alignment mismatch')
        np.testing.assert_allclose(times[frames],part.time,atol=1e-8,rtol=0)
        for k in range(49): d.loc[ix,f'visual_{k}'] = visual[frames,k]
        earlier = np.maximum(frames-1,0)
        d.loc[ix,'visual_change'] = np.linalg.norm(visual[frames]-visual[earlier],axis=1)
        d.loc[ix[boundary[frames]],'visual_change'] = np.nan
        # Search only within a strictly increasing clock segment and exclude
        # the current observation from both history windows.
        for seconds, column in ((.5,'recent10'),(1.5,'recent30')):
            counts=[]
            for f in frames:
                origin = int(starts[f])
                start = origin + np.searchsorted(times[origin:f],times[f]-seconds,side='left')
                counts.append(int(cumulative[f]-cumulative[start]))
            d.loc[ix,column] = counts
        d.loc[ix,'prior_done'] = terminal[earlier] | boundary[frames]
        d.loc[ix,'run_progress'] = progress[frames]
    d = d.copy()  # Consolidate columns before adding derived aliases.
    d['log_age_sq'] = d.log_age**2
    d['run_progress_sq'] = d.run_progress**2
    d['actor_action_probability_valid'] = 1
    d['searched'] = d.budget.gt(0).astype(int)
    d['carried'] = d.carry
    d['additional_computations'] = d.budget
    d['U_all'] = d.U_V
    from scipy.special import expit
    d['K'] = 4*d.U_V*expit(d.M)*expit(-d.M)
    d['signed_update_sum'] = d.D_V
    d['terminal_return_gain'] = np.nan  # No unrecorded return estimate is invented.
    return d
