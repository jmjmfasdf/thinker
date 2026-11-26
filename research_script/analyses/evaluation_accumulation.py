#!/usr/bin/env python3
"""Prospective directional evaluation during a constant human action.

Primary question: does continuation of an EARLY evaluation-change direction
predict a subsequent human switch or REAL-policy change, beyond evaluation
level, movement magnitude, screen/root state, action/history and budget?

No training, game-score evaluation, clustering, or inferred human goal labels.
The primary representation is the first-imagination SRN/VPN output profile;
pooled activation changes are a separately reported sensitivity analysis.
All results are conditional associations. Thinker trained on all four sessions.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

RESEARCH = Path(__file__).resolve().parents[1]
REPO = RESEARCH.parent
VERSION = 'directional-evaluation-v1'
GAME_IDS = {'enduro': 0, 'pong': 1, 'spaceinvaders': 2}
ARRAYS = ('root', 'heads', 'activation', 'actor_p', 'prior')


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def write_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.with_suffix(p.suffix + '.partial').write_text(json.dumps(obj, indent=2,
        ensure_ascii=False, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')
    os.replace(p.with_suffix(p.suffix + '.partial'), p)


def read_json(p):
    return json.loads(Path(p).read_text())


def reader():
    from common import native_replay
    return native_replay


def validate_capture(capture):
    m = read_json(capture/'manifest.json'); c = read_json(capture/'COMPLETED.json')
    unsigned = dict(m); claimed = unsigned.pop('manifest_sha256')
    if c['status'] != 'COMPLETED' or digest(unsigned) != claimed or claimed != c['manifest_sha256']:
        raise ValueError('Capture is incomplete or its manifest was changed')
    if digest(m['episodes']) != m['episodes_sha256'] or digest(m['semantic']) != m['semantic_sha256']:
        raise ValueError('Capture semantic/episode digest mismatch')
    return m


def verify_selected(t, fields, events=None):
    """Verify metadata and the stored chunks actually read, not unused images."""
    groups = {t.specs[f]['storage_group'] for f in fields}
    selected = {}
    for g in groups:
        chunk = t.z['packs/'+g].chunks[0]
        selected[g] = None if events is None else set((np.asarray(events)//chunk).tolist())
    for item in t.manifest['files']:
        rel = item['path']; parts = rel.split('/')
        needed = rel in ('.zmetadata', '.zattrs', '.zgroup')
        if len(parts) >= 3 and parts[0] == 'packs' and parts[1] in groups:
            needed = parts[-1].startswith('.') or selected[parts[1]] is None or int(parts[-1].split('.')[0]) in selected[parts[1]]
        if needed:
            p = t.path/rel
            if p.stat().st_size != item['size'] or sha(p) != item['sha256']:
                raise ValueError('Changed native input: '+str(p))


def pool(h):
    return np.asarray(h).reshape(-1, 128, 2, 3, 2, 3).mean(axis=(3, 5)).reshape(-1, 512)


def head_profile(native):
    # Policy is that of the imagined next state, NOT a root-action Q advantage.
    return np.column_stack([native[:, 0], expit(native[:, 1]), native[:, 2],
                            np.sqrt(softmax(native[:, 3:], axis=1))]).astype('float32')


def trim_smoke(d, n):
    ids = []
    for _, r in d.groupby('episode', sort=False):
        start = max(0, (len(r)-n)//2); ids.extend(r.index[start:start+n])
    return np.asarray(ids, int)


def add_raw_behavior(d, manifest):
    """Catch even switch-and-return events between two replay observations."""
    d = d.copy()
    sources = {e['metadata']['source_id']: e['metadata'] for e in manifest['episodes']}
    for run, rows in d.groupby('run', sort=False):
        meta = sources[run]; p = Path(meta['source_path'])
        if not p.exists(): p = REPO/'behavioral_data_block'/run
        if sha(p) != meta['source_sha256']: raise ValueError('Behavioral source changed')
        with np.load(p, allow_pickle=False) as raw:
            a = raw['action']; a = a.argmax(1) if a.ndim == 2 else a
            ts = raw['time'].astype(float)
        f = rows.source_frame.to_numpy(int)
        np.testing.assert_array_equal(a[f], rows.human_action.to_numpy())
        np.testing.assert_allclose(ts[f], rows.time.to_numpy(), rtol=0, atol=1e-8)
        boundaries = np.r_[0, np.flatnonzero(a[1:] != a[:-1])+1]
        bout = np.searchsorted(boundaries, f, side='right')-1
        d.loc[rows.index, 'raw_bout'] = bout
        d.loc[rows.index, 'age'] = ts[f]-ts[boundaries[bout]]
        next_ix = np.minimum(bout+1, len(boundaries)-1)
        nxt = ts[boundaries[next_ix]].copy(); nxt[bout+1 == len(boundaries)] = np.inf
        d.loc[rows.index, 'next_human_switch_time'] = nxt
    return d


def reusable_features(candidates, manifest, capture):
    expected = {e['metadata']['episode_id']: e['metadata'] for e in manifest['episodes']}
    needed = ['decisions.csv.gz', 'baseline_raw.npy', 'head_native.npy', 'actor_p.npy',
              'root_prior.npy', 'sr_root.npy', 'sr_first.npy', 'vp_root.npy', 'vp_first.npy']
    for candidate in candidates:
        p = Path(candidate)
        if not (p/'extract.done.json').exists() or not (p/'audit.json').exists(): continue
        audit = read_json(p/'audit.json'); marker = read_json(p/'extract.done.json')
        if marker.get('status') != 'COMPLETE': continue
        if set(e['episode'] for e in audit['episodes']) != set(expected): continue
        hashes = set(audit['checkpoint_hashes'].values())
        cp = manifest['semantic']['checkpoint']
        if not {cp['actor']['sha256'], cp['model']['sha256'], cp['config']['sha256']}.issubset(hashes): continue
        # An earlier capture may use the same checkpoint and data, but another
        # random gate seed. Require the actual immutable episode manifests.
        matches = True
        for ep in audit['episodes']:
            actual = next(e for e in manifest['episodes'] if e['metadata']['episode_id'] == ep['episode'])
            if ep['manifest_sha256'] != sha(capture/actual['path']/'manifest.json'): matches = False
        if not matches: continue
        for name in needed:
            if name not in marker['artifacts'] or sha(p/name) != marker['artifacts'][name]:
                raise ValueError('Prior extraction artifact changed: '+str(p/name))
        return p
    return None


def extract(args, subject, game, out, manifest):
    capture = args.input_root/f'sub{subject:03d}'/game
    candidates = [p/game if (p/game).is_dir() else p for p in args.reuse_from]
    reuse = reusable_features(candidates, manifest, capture)
    if reuse is not None:
        log(f'sub{subject:03d} {game}: reuse hash-verified first-imagination extraction')
        full = pd.read_csv(reuse/'decisions.csv.gz')
        if 'done_logit_valid' not in full or not full.done_logit_valid.eq(1).all():
            raise ValueError('Reused extraction lacks valid depth1 termination heads; use native extraction')
        ids = trim_smoke(full, args.smoke_decisions) if args.smoke else np.arange(len(full))
        d = full.iloc[ids].copy().reset_index(drop=True)
        arr = {'root': np.load(reuse/'baseline_raw.npy', mmap_mode='r')[ids].astype('float32'),
               'heads': head_profile(np.load(reuse/'head_native.npy', mmap_mode='r')[ids]),
               'actor_p': np.load(reuse/'actor_p.npy', mmap_mode='r')[ids],
               'prior': np.load(reuse/'root_prior.npy', mmap_mode='r')[ids]}
        parts = []
        for prefix in ['sr', 'vp']:
            first = np.load(reuse/(prefix+'_first.npy'), mmap_mode='r')
            root = np.load(reuse/(prefix+'_root.npy'), mmap_mode='r')
            pieces = [pool(first[ix])-pool(root[ix]) for ix in np.array_split(ids, max(1, len(ids)//256))]
            parts.append(np.concatenate(pieces))
        arr['activation'] = np.column_stack(parts).astype('float32')
        coverage = {'source': str(reuse), 'available_decisions': len(full), 'kept': len(d)}
    else:
        tr = reader(); frames, arrays, audits = [], {k: [] for k in ARRAYS}, []
        for ep in manifest['episodes']:
            path = capture/ep['path']; t = tr.Trace(path); meta = t.meta
            if (t.manifest['state'] != 'complete' or t.manifest['event_count'] != ep['event_count']
                    or t.manifest['metadata'] != ep['metadata']
                    or digest(t.manifest['files']) != t.manifest['files_sha256']
                    or t.manifest['files_sha256'] != ep['files_sha256']):
                raise ValueError('Episode manifest changed')
            for key in ['actor', 'model', 'config']:
                if meta['checkpoint_'+key+'_sha256'] != manifest['semantic']['checkpoint'][key]['sha256']:
                    raise ValueError('Checkpoint binding mismatch')
            if meta['subject'] != subject or meta['game'] != GAME_IDS[game]: raise ValueError('Subject/game mismatch')
            names = ['initial_root', 'real_transition', 'search_computation', 'current_node_depth_post',
                'actor_reset', 'score_mask', 'human_action', 'pre_observation_source_index',
                'root_carried', 'accepted_primary_action', 'model_done_target_depth',
                'model_done_logit', 'reward', 'tree_reps_pre', 'tree_reps_post',
                'actor_real_logits_pre_override', 'actor_action_probability']
            names += [n+'_valid' for n in names if n+'_valid' in t.specs]
            verify_selected(t, names)
            f = {n: t.scalar(n) for n in names}
            source = Path(meta['source_path'])
            if not source.exists(): source = REPO/'behavioral_data_block'/meta['source_id']
            if sha(source) != meta['source_sha256']: raise ValueError('Behavior source mismatch')
            with np.load(source, allow_pickle=False) as raw: ts = raw['time'].astype(float)
            rows = []; skipped = 0
            for j, left, right, comps, prefix in tr.decision_edges(f['initial_root'], f['real_transition'],
                    f['search_computation'], f['current_node_depth_post'], f['actor_reset']):
                if not f['score_mask'][right]: continue
                if not len(comps) or not prefix or prefix[0] != comps[0]: skipped += 1; continue
                first = int(comps[0])
                if f['current_node_depth_post'][first] != 1: raise ValueError('First new computation is not depth1')
                valid = all(t.valid(n)[[left, first]].all() for n in ['im_vectors', 'im_vp_vectors'])
                valid &= t.valid('model_done_logit')[first] and f['model_done_target_depth'][first] == 1
                valid &= t.valid('actor_real_logits_pre_override')[right] and t.valid('real_image')[left]
                if not valid: skipped += 1; continue
                src = int(f['pre_observation_source_index'][right])
                rows.append(dict(session=meta['session'], run=meta['source_id'], episode=meta['episode_id'],
                    decision=j, source_frame=src, time=float(ts[src]), human_action=int(f['human_action'][right]),
                    root_event=int(left), first_event=first, real_event=int(right), trace=str(path),
                    budget=len(comps), carry=int(f['root_carried'][left]),
                    first_action=int(f['accepted_primary_action'][first]),
                    reward_prior=float(f['reward'][left].reshape(-1)[0])))
            frame = pd.DataFrame(rows); available = len(frame)
            if frame.empty: continue
            if args.smoke: frame = frame.iloc[trim_smoke(frame, args.smoke_decisions)].reset_index(drop=True)
            chosen = np.unique(frame[['root_event', 'first_event', 'real_event']].to_numpy().ravel())
            verify_selected(t, ['im_vectors','im_vp_vectors','real_image'], chosen)
            chunks = {k: [] for k in ARRAYS}
            for start in range(0, len(frame), 128):
                rr = frame.iloc[start:start+128]; roots = rr.root_event.to_numpy(); firsts = rr.first_event.to_numpy(); reals = rr.real_event.to_numpy()
                root_lat, delta = [], []
                for field in ['im_vectors', 'im_vp_vectors']:
                    x0 = pool(t.read(field, roots)); x1 = pool(t.read(field, firsts))
                    root_lat.append(x0); delta.append(x1-x0)
                image = t.read('real_image', roots).astype('float32')/255
                screen = image.reshape(-1, 3, 7, 12, 7, 12).mean(axis=(3, 5)).reshape(-1, 147)
                chunks['root'].append(np.column_stack([screen]+root_lat))
                chunks['activation'].append(np.column_stack(delta))
                native = np.column_stack([t.tree('cur_r', firsts), f['model_done_logit'][firsts],
                                          t.tree('cur_v', firsts), t.tree('cur_policy', firsts)])
                chunks['heads'].append(head_profile(native))
                p = softmax(t.read('actor_real_logits_pre_override', reals).reshape(len(rr), -1), axis=1)
                np.testing.assert_allclose(p, t.read('actor_action_probability', reals).reshape(len(rr), -1), atol=1e-4, rtol=1e-4)
                chunks['actor_p'].append(p); chunks['prior'].append(softmax(t.tree('root_policy', firsts, 'pre'), axis=1))
                frame.loc[rr.index, 'root_value'] = t.tree('root_v', firsts, 'pre').ravel()
            frames.append(frame)
            for key in ARRAYS: arrays[key].append(np.concatenate(chunks[key]))
            audits.append(dict(episode=meta['episode_id'], available=available, invalid=skipped, kept=len(frame)))
            log(f'sub{subject:03d} {game}: extracted episode {len(audits)}/{len(manifest["episodes"])}')
        if not frames: raise ValueError('No valid first-imagination observations')
        d = pd.concat(frames, ignore_index=True); arr = {k: np.concatenate(v) for k,v in arrays.items()}
        coverage = {'source': str(capture), 'episodes': audits, 'kept': len(d)}
    d['subject'] = subject; d['game'] = game
    d = add_raw_behavior(d, manifest)
    if d.duplicated(['episode','decision']).any(): raise ValueError('Duplicate replay decisions')
    if sorted(d.session.unique()) != [1,2,3,4]: raise ValueError('Require all four analysis sessions')
    for key, a in arr.items():
        if len(a) != len(d) or not np.isfinite(a).all(): raise ValueError('Invalid feature array '+key)
        np.save(out/(key+'.npy'), a.astype('float32'))
    d.to_csv(out/'decisions.csv.gz', index=False); write_json(out/'coverage.json', coverage)
    return d, arr


def js_distance(p, q):
    p = np.clip(p, 1e-12, 1); q = np.clip(q, 1e-12, 1); m = (p+q)/2
    return np.sqrt(np.maximum(0, .5*np.sum(p*np.log(p/m)+q*np.log(q/m), axis=-1)))


def make_windows(d, actor_p, width=.5, horizon=.5, max_gap=.25):
    records = []; counts = {'candidates': 0, 'past_gap_or_short': 0, 'action_changed': 0, 'future_censored': 0, 'kept': 0}
    for ep, rows in d.groupby('episode', sort=False):
        rows = rows.sort_values('decision'); ix = rows.index.to_numpy(); ts = rows.time.to_numpy()
        dec = rows.decision.to_numpy(); bouts = rows.raw_bout.to_numpy()
        next_allowed = -np.inf
        for end in range(len(rows)):
            counts['candidates'] += 1
            start = np.searchsorted(ts, ts[end]-width, side='left')
            middle = np.searchsorted(ts, ts[end]-width/2, side='right')-1
            if middle-start+1 < 3 or end-middle < 3 or ts[end]-ts[start] < width-.10:
                counts['past_gap_or_short'] += 1; continue
            future = np.searchsorted(ts, ts[end]+horizon, side='left')
            if future >= len(rows) or ts[future]-(ts[end]+horizon) > max_gap:
                counts['future_censored'] += 1; continue
            sl = slice(start, future+1)
            if np.any(np.diff(dec[sl]) != 1) or np.any(np.diff(ts[sl]) <= 0) or np.any(np.diff(ts[sl]) > max_gap):
                counts['past_gap_or_short'] += 1; continue
            if bouts[start] != bouts[end]: counts['action_changed'] += 1; continue
            # Disjoint input AND future spans; chosen using timestamps alone.
            disjoint = ts[start] > next_allowed
            if disjoint: next_allowed = ts[future]
            j = ix[end]; incumbent = int(d.loc[j,'human_action'])
            switch = float(d.loc[j,'next_human_switch_time'] <= ts[end]+horizon)
            p0, p1 = actor_p[j], actor_p[ix[future]]
            logodds = lambda p: np.log(np.clip(1-p[incumbent],1e-8,1))-np.log(np.clip(p[incumbent],1e-8,1))
            records.append(dict(window_id=len(records), episode=ep, run=rows.run.iloc[0], session=int(rows.session.iloc[0]),
                start=int(ix[start]), middle=int(ix[middle]), end=int(j), future=int(ix[future]),
                start_time=ts[start], time=ts[end], future_time=ts[future], incumbent=incumbent,
                human_switch=switch, policy_js=float(js_distance(p0,p1)), policy_away=float(logodds(p1)-logodds(p0)),
                nonoverlap=bool(disjoint), first_action_constant=bool(rows.first_action.iloc[start:end+1].nunique()==1)))
    w = pd.DataFrame(records); counts['kept'] = len(w)
    return w, counts


def geometry(z, middle, permutation=None):
    """Freeze direction in early half; evaluate subsequent progress along it.

Projection is NET late progress (the signed increments telescope); coherence
divides it by late path length. Neither is an estimated neural integrator.
"""
    steps = np.diff(z, axis=0)
    if permutation is not None: steps = steps[permutation]
    early, late = steps[:middle], steps[middle:]
    e = early.sum(axis=0); norm = np.linalg.norm(e)
    early_path = np.linalg.norm(early,axis=1).sum(); late_path = np.linalg.norm(late,axis=1).sum()
    u = e/max(norm,1e-12); proj = late@u
    directed = np.array([proj.sum(), proj.sum()/max(late_path,1e-12),
        np.mean(proj>0) if len(proj) else 0., norm/max(early_path,1e-12), float(norm>1e-8)])
    # Amplitude controls must not already encode the angle being tested.
    # The three net endpoint lengths determine the signed projection exactly
    # by the cosine rule; those belong in a separate endpoint sensitivity.
    en = np.linalg.norm(early, axis=1); ln = np.linalg.norm(late, axis=1)
    magnitude = np.array([early_path, late_path, np.sqrt(np.mean(en**2)),
                          np.sqrt(np.mean(ln**2)), en.max(), ln.max()])
    return directed, magnitude


def fit_coordinates(x, train, rank, seed):
    scale = StandardScaler().fit(x[train]); y = scale.transform(x)
    rank = min(rank, len(train)-1, x.shape[1]); rng = np.random.default_rng(seed)
    sampled = rng.choice(train,min(4096,len(train)),replace=False)
    pca = PCA(n_components=rank, svd_solver='randomized', random_state=seed).fit(y[sampled])
    return pca.transform(y), {'scale': scale, 'pca': pca, 'sampled': sampled}


def fold_features(d, a, w, held, representation, args):
    train_dec = np.flatnonzero(d.session.to_numpy()!=held)
    root, root_fit = fit_coordinates(a['root'], train_dec, args.root_rank, args.seed+held)
    nact = a['actor_p'].shape[1]; onehot = np.eye(nact)
    nuisance = np.column_stack([root, a['prior'], onehot[d.first_action.to_numpy(int)],
        d[['root_value','reward_prior','carry']].to_numpy()])
    ns = StandardScaler().fit(nuisance[train_dec]); nuisance = ns.transform(nuisance)
    if representation=='heads':
        ev = StandardScaler().fit(a['heads'][train_dec]).transform(a['heads'])
    else: ev, _ = fit_coordinates(a['activation'],train_dec,args.activation_rank,args.seed+held+101)
    # Remove predictable pre-search state/first-branch content using training
    # sessions only. Residuals are not proof of a causal imagination effect.
    nuisance_fit = Ridge(alpha=args.penalty*len(train_dec)).fit(nuisance[train_dec],ev[train_dec])
    z = ev-nuisance_fit.predict(nuisance)
    z = StandardScaler().fit(z[train_dec]).transform(z)
    root_history = np.column_stack([root,a['prior'],a['actor_p'],
        onehot[d.first_action.to_numpy(int)],d[['root_value','reward_prior','carry']].to_numpy()])
    baseline, levels, magnitudes, endpoints, directions, nulls = [],[],[],[],[],[]
    rng = np.random.default_rng(args.seed+held)
    for row in w.itertuples():
        ix = np.arange(row.start,row.end+1); mid = row.middle-row.start
        r = root_history[ix]; zz = z[ix]
        meta = d.loc[row.end]
        structure = [np.log1p(meta.age),np.log1p(meta.budget),float(meta.carry),meta.reward_prior,
                     d.loc[ix,'budget'].mean(),d.loc[ix,'carry'].mean(),float(row.first_action_constant),
                     len(ix),float(row.time-row.start_time)]
        first_frequency = onehot[d.loc[ix,'first_action'].to_numpy(int)].mean(axis=0)
        baseline.append(np.r_[r[-1],r.mean(axis=0),r[-1]-r[0],r.std(axis=0),
            structure,onehot[row.incumbent],first_frequency])
        levels.append(np.r_[zz[-1],zz.mean(axis=0),zz[0]])
        dr, mag = geometry(zz,mid); magnitudes.append(mag); directions.append(dr)
        endpoints.append([np.linalg.norm(zz[-1]-zz[0]),np.linalg.norm(zz[mid]-zz[0]),
                          np.linalg.norm(zz[-1]-zz[mid]),np.mean(np.var(zz,axis=0))])
        perms=[]
        for _ in range(args.permutations):
            # Same increments/magnitude multiset and same endpoints; remove
            # their temporal ordering. Keep the observed nuisance/levels fixed.
            perm=rng.permutation(len(ix)-1); perms.append(geometry(zz,mid,perm)[0])
        nulls.append(perms)
    b=np.asarray(baseline); level=np.column_stack([b,levels]); magnitudes=np.asarray(magnitudes)
    # Give the amplitude-only comparator nonlinear terms too. A linear-only
    # comparator could mistake an amplitude interaction for directional evidence.
    nonlinear=np.column_stack([np.log1p(magnitudes)]+[
        (magnitudes[:,i]*magnitudes[:,j])[:,None] for i in range(magnitudes.shape[1]) for j in range(i,magnitudes.shape[1])])
    directions=np.asarray(directions)
    mag=np.column_stack([level,magnitudes,nonlinear,directions[:,4]])
    endpoint_mag=np.column_stack([mag,endpoints])
    designs={'baseline':b,'level':level,'magnitude':mag,
             'directional':np.column_stack([mag,directions[:,0]]),
             'directional_profile':np.column_stack([mag,directions]),
             'endpoint_magnitude':endpoint_mag,
             'endpoint_directional':np.column_stack([endpoint_mag,directions[:,0]])}
    for j in range(args.permutations):designs[f'order_null_{j:02d}']=np.column_stack([mag,np.asarray(nulls)[:,j,0]])
    # Out-of-session residual quality is an explicit diagnostic, not assumed.
    test_dec=np.flatnonzero(d.session.to_numpy()==held)
    mse=np.mean((ev[test_dec]-nuisance_fit.predict(nuisance[test_dec]))**2)
    denom=np.mean((ev[test_dec]-ev[train_dec].mean(axis=0))**2)
    qa={'held_session':int(held),'representation':representation,'residual_nuisance_test_r2':float(1-mse/max(denom,1e-12)),
        'training_decisions':len(train_dec),'test_decisions':len(test_dec),'training_sessions':sorted(map(int,d.loc[train_dec,'session'].unique())),
        'root_basis_training_sample':len(root_fit['sampled'])}
    return designs, np.asarray(directions), z, qa


def predict_model(x, y, train, test, binary, penalty, minimum):
    if len(train)<minimum or len(test)<20: return None,'insufficient_rows',None
    if binary and min(np.sum(y[train]==0),np.sum(y[train]==1))<20: return None,'insufficient_training_classes',None
    scaler=StandardScaler().fit(x[train]); xt=scaler.transform(x[train]); xv=scaler.transform(x[test])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        if binary:
            model=LogisticRegression(C=1/(penalty*len(train)),solver='lbfgs',max_iter=1500,tol=1e-7).fit(xt,y[train])
            pred=model.predict_proba(xv)[:,1]
        else:
            model=Ridge(alpha=penalty*len(train)).fit(xt,y[train]); pred=model.predict(xv)
    if any(issubclass(c.category,ConvergenceWarning) for c in caught): return None,'not_converged',None
    if not np.isfinite(pred).all(): return None,'nonfinite_prediction',None
    coef=model.coef_.reshape(-1)
    return pred,'ok',coef


def balanced_mean(d, value):
    return float(d.groupby(['session','run'])[value].mean().groupby('session').mean().mean())


def block_interval(rows, values, repeats, seed):
    d=rows[['session','run','time']].copy(); d['gain']=values
    d['block']=np.floor(d.time/30).astype(int)
    agg=d.groupby(['session','run','block']).gain.agg(['sum','count']).reset_index()
    rng=np.random.default_rng(seed); sessions={}
    for (session,_),r in agg.groupby(['session','run'],sort=False):
        sums=r['sum'].to_numpy();counts=r['count'].to_numpy()
        ii=rng.integers(0,len(r),(repeats,len(r)))
        estimates=sums[ii].sum(axis=1)/counts[ii].sum(axis=1)
        sessions.setdefault(session,[]).append(estimates)
    estimates=np.mean([np.mean(v,axis=0) for v in sessions.values()],axis=0)
    low,high=np.quantile(estimates,[.025,.975]) if len(estimates) else [np.nan,np.nan]
    return dict(gain=balanced_mean(d,'gain'),ci_low=float(low),ci_high=float(high),blocks=len(agg))


def fit_readouts(d,a,w,args,out):
    predictions=[]; statuses=[]; foldqa=[]; directional_tables=[]; display=None
    for representation in args.representations:
        for held in sorted(w.session.unique()):
            log(f'{out.name}: {representation}, analysis session {held}')
            designs,directions,z,qa=fold_features(d,a,w,held,representation,args);foldqa.append(qa)
            select=w.session.to_numpy()==held
            q=w.loc[select,['window_id','session','run','episode','time','start','middle','end','future','nonoverlap']].copy()
            q['representation']=representation
            for j,name in enumerate(['progress','coherence','positive_fraction','early_coherence','direction_defined']):q[name]=directions[select,j]
            directional_tables.append(q)
            if display is None and representation==args.representations[0]:display=(z,directions,int(held))
            for sampling in ['nonoverlap','all','nonoverlap_fixed_branch']:
                allowed=w.nonoverlap.to_numpy(bool) if sampling!='all' else np.ones(len(w),bool)
                if sampling=='nonoverlap_fixed_branch':allowed=allowed&w.first_action_constant.to_numpy(bool)
                train=np.flatnonzero(~select&allowed);test=np.flatnonzero(select&allowed)
                for target in ['human_switch','policy_js','policy_away']:
                    y=w[target].to_numpy();binary=target=='human_switch'
                    for name,x in designs.items():
                        pred,status,coef=predict_model(x,y,train,test,binary,args.penalty,args.minimum_train)
                        statuses.append(dict(representation=representation,session=int(held),sampling=sampling,target=target,model=name,
                            status=status,n_train=len(train),n_test=len(test),n_events=int(y[test].sum()) if binary else None,
                            standardized_progress_coefficient=float(coef[-1]) if coef is not None and name in ['directional','endpoint_directional'] else None))
                        if pred is None:continue
                        r=w.iloc[test][['window_id','session','run','episode','time']].copy()
                        r['representation']=representation;r['sampling']=sampling;r['target']=target;r['model']=name;r['truth']=y[test];r['prediction']=pred
                        r['loss']=-(y[test]*np.log(np.clip(pred,1e-10,1-1e-10))+(1-y[test])*np.log(np.clip(1-pred,1e-10,1))) if binary else (pred-y[test])**2
                        predictions.append(r)
    stat=pd.DataFrame(statuses);stat.to_csv(out/'fit_status.csv',index=False)
    write_json(out/'fold_qa.json',foldqa)
    ds=pd.concat(directional_tables,ignore_index=True);ds.to_csv(out/'heldout_directional_features.csv.gz',index=False)
    if not predictions:return pd.DataFrame(),pd.DataFrame(),display
    p=pd.concat(predictions,ignore_index=True);p.to_csv(out/'heldout_predictions.csv.gz',index=False)
    comparisons=[]
    for (rep,sampling,target),part in p.groupby(['representation','sampling','target']):
        wide=part.pivot(index='window_id',columns='model',values='loss')
        contrasts=[('level','baseline'),('magnitude','level'),('directional','magnitude'),
                   ('directional_profile','directional'),('endpoint_directional','endpoint_magnitude')]
        contrasts += [('directional',c) for c in wide if c.startswith('order_null_')]
        for model,reference in contrasts:
            if model not in wide or reference not in wide:continue
            ok=wide[[model,reference]].dropna();rows=w.loc[w.window_id.isin(ok.index)].set_index('window_id').loc[ok.index]
            delta=(ok[reference]-ok[model]).to_numpy()
            interval=block_interval(rows,delta,args.bootstrap,args.seed)
            per=rows[['session','run']].copy();per['gain']=delta
            session=per.groupby(['session','run']).gain.mean().groupby('session').mean()
            coef_model='endpoint_directional' if model=='endpoint_directional' else 'directional'
            coefs=stat.loc[stat.representation.eq(rep)&stat.sampling.eq(sampling)&stat.target.eq(target)&stat.model.eq(coef_model)&stat.status.eq('ok'),'standardized_progress_coefficient']
            comparisons.append(dict(representation=rep,sampling=sampling,target=target,model=model,reference=reference,n=len(rows),
                sessions=rows.session.nunique(),positive_sessions=int(session.gt(0).sum()),
                positive_progress_coefficients=int(coefs.gt(0).sum()),progress_coefficient_folds=len(coefs),**interval))
    result=pd.DataFrame(comparisons);result.to_csv(out/'comparisons.csv',index=False)
    return p,result,display


def render(d,a,w,p,comparisons,display,out,args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=out/'figures';figs.mkdir(exist_ok=True)
    title=f'{d.game.iloc[0]} · sub{int(d.subject.iloc[0]):03d}'
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(figs/(name+'.'+ext),dpi=160,bbox_inches='tight')
        plt.close(fig)
    # Select examples by past coherence only, within one held analysis session.
    z,directions,held=display
    eligible=np.flatnonzero(w.session.eq(held).to_numpy() & w.nonoverlap.to_numpy(bool) & (directions[:,4]>0))
    chosen=[]
    if len(eligible):
        ordered=eligible[np.argsort(directions[eligible,1])]
        chosen=[ordered[int(q*(len(ordered)-1))] for q in [.1,.5,.9]]
    examples=[]
    for label,i in zip(['low','middle','high'],chosen):
        row=w.iloc[i];idx=np.arange(int(row.start),int(row.future)+1);tt=d.loc[idx,'time'].to_numpy()-row.time
        anchor=z[int(row.middle)]-z[int(row.start)];anchor/=max(np.linalg.norm(anchor),1e-12)
        projection=(z[idx]-z[int(row.start)])@anchor
        fig,axes=plt.subplots(4,1,figsize=(9,8),sharex=True)
        axes[0].plot(tt,projection,'o-',ms=3,label='Evaluation projection onto early direction')
        axes[0].axvline(d.loc[int(row.middle),'time']-row.time,color='grey',ls=':',label='Direction fixed here')
        axes[0].set_ylabel('Projection (a.u.)');axes[0].legend(fontsize=8)
        for j in range(a['actor_p'].shape[1]):axes[1].plot(tt,a['actor_p'][idx,j],label=str(j),lw=1)
        axes[1].set_ylabel('REAL policy');axes[1].legend(title='Action',ncol=6,fontsize=7)
        source=REPO/'behavioral_data_block'/str(row.run)
        if not source.exists():source=Path(reader().Trace(Path(d.loc[int(row.end),'trace'])).meta['source_path'])
        with np.load(source,allow_pickle=False) as raw:
            action=raw['action'];action=action.argmax(1) if action.ndim==2 else action
            rawtime=raw['time'].astype(float)-row.time
        shown=(rawtime>=tt[0])&(rawtime<=tt[-1])
        axes[2].step(rawtime[shown],action[shown],where='post',label='Human action (original sampling)')
        axes[2].set_ylabel('Human action')
        axes[3].step(tt,d.loc[idx,'budget'],where='post');axes[3].set(ylabel='Computations',xlabel='Seconds relative to prediction time')
        for ax in axes:ax.axvline(0,color='black',ls='--');ax.axvspan(0,tt[-1],color='#eee',alpha=.5);ax.spines[['top','right']].set_visible(False)
        fig.suptitle(title+f' · past-coherence {label} example · shaded = future\nDescriptive only; example selected without future outcome')
        save(fig,'timeline_'+label)
        examples.append(dict(window_id=int(row.window_id),selection=label,session=int(row.session),run=row.run,
            start_time=float(row.start_time),prediction_time=float(row.time),coherence=float(directions[i,1])))
    pd.DataFrame(examples).to_csv(out/'timeline_examples.csv',index=False)
    primary=comparisons[(comparisons.representation==args.representations[0]) & comparisons.sampling.eq('nonoverlap')] if len(comparisons) else comparisons
    body='<html><meta charset="utf-8"><title>Directional evaluation</title><body style="font-family:sans-serif;max-width:1100px;margin:auto">'
    body+=f'<h1>{html.escape(title)}</h1><p>'+('SOFTWARE CHECK — not scientific evidence. ' if args.smoke else 'Exploratory prospective analysis. ')
    body+='Early direction is fixed before late progress; every outcome occurs after the complete input window. No goal labels or causal claims.</p>'
    body+='<p>Human switching and model-policy changes are separate outcomes. Analysis sessions are held out from readout fitting; all sessions trained Thinker.</p>'
    body+='<p>Primary: heads, disjoint spans, adding the single signed-progress variable to magnitude controls. Prediction gain alone does not establish that MORE positive progress predicts MORE switching: inspect coefficient signs too. The five-feature direction profile is a diagnostic. Policy JS is secondary; signed policy movement away from the incumbent is diagnostic. Activation and overlapping windows are sensitivity analyses.</p>'
    body+='<p>Block intervals condition on one participant and fitted models; they are not population tests and are not multiplicity adjusted.</p>'
    body+='<p>The primary magnitude comparator controls early/late step lengths, RMS and maximum increments. Net endpoint lengths already determine progress by the cosine rule: endpoint_directional is a conservative sensitivity, not evidence of information independent of those lengths. Neither analysis proves that Thinker stores an accumulator.</p>'
    if len(primary):body+=primary.to_html(index=False,float_format=lambda x:f'{x:.5g}')
    else:body+='<p>Insufficient support for predictive fits; inspect fit_status.csv.</p>'
    for x in sorted(figs.glob('*.png')):body+=f'<img style="width:100%" src="figures/{x.name}">'
    body+='<p><a href="comparisons.csv">All comparisons</a> · <a href="fit_status.csv">Fit/support status</a> · <a href="protocol.json">Protocol and limitations</a></p></body></html>'
    (out/'index.html').write_text(body)


def protocol(args):
    return dict(version=VERSION,window_seconds=args.window,horizon_seconds=args.horizon,penalty=args.penalty,
        root_rank=args.root_rank,activation_rank=args.activation_rank,permutations=args.permutations,
        minimum_train=args.minimum_train,bootstrap=args.bootstrap,seed=args.seed,representations=args.representations,
        smoke=args.smoke,smoke_decisions=args.smoke_decisions if args.smoke else None,
        primary='heads / nonoverlap / human_switch / adding one signed-progress predictor to magnitude; inspect expected positive coefficient separately',
        secondary='policy_js; directional versus within-window increment-order nulls',
        diagnostic='policy_away; five-variable directional_profile; endpoint_magnitude comparator; activation representation; overlapping windows; windows with a fixed first imagined action',
        direction='unit vector from early first to early last evaluation; never oriented using future actions or policy',
        projection='sum of subsequent increments onto early direction; this telescopes to net late progress',
        coherence='signed late progress / late path length; early flat trajectories carry an explicit undefined-direction flag',
        input='first new imagination at depth1; all decisions have equal opportunity; tree carry may contain older computations',
        heads='SR reward and done probability; VPN next-state value and sqrt action probabilities; not root-action utilities',
        activation='pooled (first imagined activation - root activation), SRN and VPN; training-session PCA',
        nuisance='screen/root features and history, current/past REAL policy, prior, human action/age, first branch history, carry and budget; nonlinear early/late step-length, RMS and maximum-increment controls',
        endpoint_caveat='net early, late and total endpoint lengths mathematically determine progress via the cosine rule; endpoint comparison is a parameterization sensitivity, not independent directional information',
        normalization='every scaler, PCA and nuisance regression fits only outer training sessions; no hyperparameter selection',
        temporal='constant raw human bout; disjoint input+future spans primary; exclude missing/gapped/censored observations',
        null='permute increment order, preserving increments, endpoints and total path; observed nuisance/magnitude features unchanged',
        aggregation='equal runs within session, equal sessions; 30s within-run block intervals conditional on fitted models',
        inference='exploratory; all Thinker sessions used in network training; no population, human-goal or causal planning inference')


def self_test():
    # Mechanistic distinction: same total distance, late reversal versus continuation.
    monotone=np.column_stack([np.arange(7.),np.zeros(7)])
    reverse=np.column_stack([[0,1,2,3,2,1,0],np.zeros(7)])
    a,_=geometry(monotone,3);b,_=geometry(reverse,3)
    assert a[1]>0.999 and b[1]<-.999
    np.testing.assert_allclose(geometry(monotone,3)[1],geometry(reverse,3)[1])
    assert geometry(np.ones((7,2)),3)[0][-1]==0
    rot=np.array([[0.,-1.],[1.,0.]])
    np.testing.assert_allclose(geometry(monotone@rot,3)[0],a)
    # Input windows precede outcomes; raw within-gap switch-and-return disqualifies a window.
    n=100;ts=np.arange(n)/15
    d=pd.DataFrame(dict(episode=['ep']*n,run=['run']*n,session=[1]*n,decision=np.arange(n),time=ts,
        human_action=np.zeros(n,int),first_action=np.zeros(n,int),raw_bout=np.zeros(n,int),next_human_switch_time=np.full(n,np.inf)))
    pp=np.tile([.8,.2],(n,1));w,_=make_windows(d,pp)
    assert len(w)>0 and (w.future>w.end).all()
    selected=w[w.nonoverlap]
    assert np.all(selected.start_time.to_numpy()[1:]>selected.future_time.to_numpy()[:-1])
    d.loc[40:,'raw_bout']=2
    ww,_=make_windows(d,pp)
    assert not ((ww.start<40)&(ww.end>=40)).any()
    d.loc[60:,'decision']+=1
    ww,_=make_windows(d,pp)
    assert not ((ww.start<60)&(ww.future>=60)).any()
    # Changing a held-out value cannot influence the fitted training transform.
    rng=np.random.default_rng(4);x=rng.normal(size=(200,5));ids=np.arange(150)
    trajectory=np.cumsum(rng.normal(size=(9,4)),axis=0)
    early=trajectory[4]-trajectory[0];late=trajectory[-1]-trajectory[4]
    cosine_projection=(np.linalg.norm(early+late)**2-np.linalg.norm(early)**2-np.linalg.norm(late)**2)/(2*np.linalg.norm(early))
    np.testing.assert_allclose(geometry(trajectory,4)[0][0],cosine_projection)
    shuffled=np.vstack([trajectory[0],trajectory[0]+np.cumsum(np.diff(trajectory,axis=0)[rng.permutation(8)],axis=0)])
    np.testing.assert_allclose(shuffled[-1],trajectory[-1])
    np.testing.assert_allclose(np.linalg.norm(np.diff(shuffled,axis=0),axis=1).sum(),np.linalg.norm(np.diff(trajectory,axis=0),axis=1).sum())
    z,f=fit_coordinates(x,ids,3,1);xx=x.copy();xx[150:]+=100
    zz,ff=fit_coordinates(xx,ids,3,1)
    np.testing.assert_allclose(f['pca'].components_,ff['pca'].components_)
    np.testing.assert_allclose(z[:150],zz[:150])
    print('PASS: matched-amplitude progress/reversal, cosine identity, permutation invariants, flat direction, rotation, prospective windows, raw-bout gaps, disjoint spans, held-out transforms')

