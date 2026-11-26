#!/usr/bin/env python3
"""Two-clock planning analysis: computation, valuation, human switching and candidate motifs.

Read-only native replay analysis. No training, environment rollout or score evaluation.
ACE is output-relevant geometry, not a goal label. Actor readout reconstruction is
conditional on recorded features, not a forced-budget intervention. All inference
uses session-held-out fits; Thinker itself was trained on these human sessions.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings
sys.dont_write_bytecode = True
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.special import expit, softmax, logsumexp
RESEARCH = Path(__file__).resolve().parent
DEFAULT_INPUT = RESEARCH.parent / 'logs/visualization/campaigns/visualization_hxln_completed_10m_20261124'
VERSION = 'control-patterns-v3.1'
GAME_IDS = {'enduro': 0, 'pong': 1, 'spaceinvaders': 2}
STAGES = ('audit', 'extract', 'representation', 'temporal', 'motifs', 'validate', 'render')
REFERENCE = 'https://arxiv.org/html/2602.22600v3'

def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()

def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(clean_json(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text())


def read_table(path):
    # Raw switch times can equal replay observation times exactly. A changed
    # final float bit can otherwise move a sample to the wrong side of t=0.
    return pd.read_csv(path, float_precision='round_trip')

def file_stamp(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]

def save_table(path, rows, columns=None):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False)

def chunks(indices, size=256):
    for start in range(0, len(indices), size):
        yield indices[start:start + size]

def frozen_import(root, module):
    source = (root / "runtime_source/thinker").resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    result = importlib.import_module("thinker." + module)
    if not Path(result.__file__).resolve().is_relative_to(source):
        raise RuntimeError("A different Thinker package was imported; use a fresh Python process")
    return result

class Trace:
    """Read-only packed Zarr reader; hash validation is owned by audit stage."""
    def __init__(self, path):
        import zarr
        self.path = Path(path)
        self.manifest = read_json(self.path / "manifest.json")
        self.specs = {f["name"]: f for f in self.manifest["field_schema"]}
        self.tree_schema = self.manifest["tree_rep_schema"]
        self.n = self.manifest["event_count"]
        self.meta = self.manifest["metadata"]
        try:
            self.z = zarr.open_consolidated(str(path), mode="r")
        except (KeyError, ValueError):
            self.z = zarr.open_group(str(path), mode="r")
        self.cache = {}

    def read(self, name, rows=None):
        s = self.specs[name]
        a = self.z["packs/" + s["storage_group"]]
        if rows is None:
            return np.asarray(a[:, s["storage_index"], ...])
        selection = (np.asarray(rows, int), s["storage_index"]) + (slice(None),) * (a.ndim - 2)
        return np.asarray(a.oindex[selection])

    def scalar(self, name):
        if name not in self.cache:
            self.cache[name] = self.read(name)
        return self.cache[name]

    def valid(self, name):
        key = name + "_valid"
        return self.scalar(key).astype(bool) if key in self.specs else np.ones(self.n, bool)

    def tree(self, name, rows, phase="post"):
        a, b, step = self.tree_schema[name]
        return self.read("tree_reps_" + phase, rows)[:, slice(a, b, step)]

def decision_edges(initial, real, computation, depth, reset):
    """Pair a_t's output with x_t, never the REAL row's post-state x_(t+1)."""
    roots = np.flatnonzero(initial)
    if len(roots) != 1:
        raise ValueError("Expected exactly one initial root")
    rights = np.flatnonzero(real)
    lefts = np.r_[roots[0], rights[:-1]]
    for j, (left, right) in enumerate(zip(lefts, rights)):
        events = np.flatnonzero(computation[left + 1:right]) + left + 1
        if not len(events):
            yield j, left, right, events, []
            continue
        prefix = []
        for event in events:
            if reset[event] or depth[event] != len(prefix) + 1:
                break
            prefix.append(int(event))
        yield j, int(left), int(right), events, prefix

def game_inputs(root, game, subject=1):
    capture = root / 'captures' / f'{game}-sub{subject:03d}-all'
    if not (capture/'COMPLETED.json').is_file():
        raise ValueError(f'Missing completed capture: {capture}')
    pattern = f'{game}-s{subject:03d}-*/milestones/10000000'
    checkpoints = sorted({p.resolve() for p in (root/'checkpoints').glob(pattern)})
    # The completed collection reuses sub001 Pong/Enduro from the first
    # campaign. Its immutable checkpoints remain in that sibling campaign.
    if not checkpoints:
        checkpoints = sorted({p.resolve() for p in root.parent.glob('*/checkpoints/'+pattern)})
    if len(checkpoints) != 1:
        raise ValueError(f'Expected one sub{subject:03d}/{game} checkpoint, found {len(checkpoints)}')
    return capture, checkpoints[0]

def pool_latents(a):
    a = a.reshape(len(a), 128, 2, 3, 2, 3).mean(axis=(3, 5))
    return a.reshape(len(a), -1)

class FrozenHeads:
    def __init__(self, args, out):
        import torch
        import gymnasium as gym
        audit = read_json(out / "audit.json")
        cp = Path(audit["checkpoint"])
        weights = torch.load(next(cp.glob("ckp_model*")), map_location="cpu", weights_only=False)["model_net_state_dict"]
        OutputNet = frozen_import(args.input_root, "model_net").OutputNet
        self.nact = audit["num_actions"]
        self.nets = {}
        for net in ("sr", "vp"):
            head = OutputNet(gym.spaces.Discrete(self.nact), (128, 6, 6), 1, 0, 0, False,
                audit["config"]["model_size_nn"], predict_v_pi=net == "vp",
                predict_r=net == "sr", predict_done=net == "sr", ordinal=False)
            prefix = net + "_net.out."
            head.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}, strict=True)
            self.nets[net] = head.eval().requires_grad_(False)

    def tensor(self, net, h, contrast=False):
        import torch
        o = self.nets[net](h.reshape(-1, 128, 6, 6))
        if net == "sr":
            return torch.cat([o.rs, o.done_logits[:, None]], dim=1)
        logits = o.policy[:, 0]
        if contrast:
            logits = logits - logits.mean(1, keepdim=True)
        return torch.cat([o.vs, logits], dim=1)

    def predict(self, net, h, batch=64, contrast=False):
        import torch
        values = []
        with torch.no_grad():
            for ix in chunks(np.arange(len(h)), batch):
                x = torch.as_tensor(np.asarray(h[ix]).copy(), dtype=torch.float32)
                values.append(self.tensor(net, x, contrast).numpy())
        return np.concatenate(values)

    def jacobian(self, net, h, scale, batch=16):
        import torch
        result = []
        for ix in chunks(np.arange(len(h)), batch):
            x = torch.as_tensor(np.asarray(h[ix]).copy(), dtype=torch.float32).requires_grad_(True)
            y = self.tensor(net, x, contrast=True)
            # Output heads have no batch-dependent operations in evaluation mode.
            gradients = []
            for k in range(y.shape[1]):
                if scale[k] <= 0:
                    gradients.append(torch.zeros_like(x))
                else:
                    gradients.append(torch.autograd.grad(y[:, k].sum(), x, retain_graph=True)[0] / float(scale[k]))
            g = torch.stack(gradients, 1).detach().numpy()
            if not np.isfinite(g).all():
                raise ValueError("Nonfinite original-head Jacobian")
            result.append(g)
        return np.concatenate(result).reshape(-1, h.shape[1])

def sample_runs(d, indices, count, seed):
    """Equal-run stratified sample without replacement; reproducible within folds."""
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g.index, int) for _, g in d.loc[indices].groupby("run", sort=True)]
    groups = [rng.permutation(g) for g in groups]
    selected = []
    while groups and len(selected) < count:
        next_groups = []
        for group in groups:
            if len(selected) == count:
                break
            selected.append(group[0])
            if len(group) > 1:
                next_groups.append(group[1:])
        groups = next_groups
    return np.asarray(selected, int)

def balanced_weights(d, indices):
    runs = d.loc[indices, "run"]
    return (1 / runs.map(runs.value_counts()) / runs.nunique()).to_numpy(float)

def activation_factor(h, rows, weights, chunk=256):
    """Exact activation range, never pre-truncate low-variance directions."""
    dim = h.shape[1]
    mean = np.zeros(dim)
    for ids in chunks(np.arange(len(rows)), chunk):
        mean += weights[ids] @ np.asarray(h[rows[ids]], dtype=np.float64)
    if len(rows) <= dim:
        x = (np.asarray(h[rows], dtype=np.float64) - mean) * np.sqrt(weights[:, None])
        # L L^T = covariance directly. Avoid a needless wide-matrix LAPACK
        # SVD (gesdd crashes on this host for 64 x 4608); ACE's interaction SVD
        # below removes zero-sensitivity/rank-deficient directions itself.
        factor = x.T.copy()
    else:
        cov = np.zeros((dim, dim), np.float64)
        for ids in chunks(np.arange(len(rows)), chunk):
            x = (np.asarray(h[rows[ids]], dtype=np.float64) - mean) * np.sqrt(weights[ids, None])
            cov += x.T @ x
        w, u = linalg.eigh(cov, check_finite=False)
        keep = w > max(w[-1] * 1e-12, 1e-14)
        factor = u[:, keep] * np.sqrt(w[keep])
    return mean, factor

def ace_basis(factor, jacobian, energy=.99, max_rank=128):
    if factor.shape[1] == 0 or not np.any(jacobian):
        return np.empty((factor.shape[0], 0)), np.array([]), 0, 0.
    interaction = factor.T @ jacobian.T / np.sqrt(max(len(jacobian), 1))
    # This is activation-rank by Jacobian-row-count, not N by NK.
    u, s, _ = linalg.svd(interaction, full_matrices=False, check_finite=False, lapack_driver='gesvd')
    weights = s ** 2
    if not len(weights) or weights.sum() <= 1e-24:
        return np.empty((factor.shape[0], 0)), weights, 0, 0.
    r_energy = int(np.searchsorted(np.cumsum(weights) / weights.sum(), energy) + 1)
    rank = min(max_rank, factor.shape[1], int((s > s[0] * 1e-10).sum()))
    q, _ = linalg.qr(factor @ u[:, :rank], mode="economic")
    return q, weights, r_energy, float(weights[:min(r_energy, rank)].sum() / weights.sum())

def intervene(h, q, mean, mode, centered=True):
    origin = mean if centered else np.zeros_like(mean)
    delta = np.asarray(h) - origin
    projected = (delta @ q) @ q.T
    return origin + (projected if mode == "keep" else delta - projected)

def subspace_overlap(a, b):
    return float(np.linalg.norm(a.T @ b, "fro") ** 2 / min(a.shape[1], b.shape[1])) if a.shape[1] and b.shape[1] else np.nan

def js_distance(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    divergence = .5 * np.sum(p * np.log((p + 1e-15)/(m + 1e-15)) +
                              q * np.log((q + 1e-15)/(m + 1e-15)), axis=-1)
    return np.sqrt(np.maximum(divergence, 0))

def tree_content(raw, schema):
    """Evaluation content only. Unvisited Q is missing, never evidence for value zero.

    Visit masks/counts, action identities, depth and reset metadata are returned
    separately as controls. NaNs are imputed from TRAINING data only downstream.
    These are actor INPUT tokens, not saved GRU hidden states or complete trees.
    """
    def get(name):
        a, b, step = schema[name]
        return raw[:, slice(a, b, step)]
    root_v, cur_v = get('root_v'), get('cur_v')
    values, names = [], []
    def add(name, x):
        values.append(x)
        names.extend([name] if x.shape[1] == 1 else [f'{name}_{j}' for j in range(x.shape[1])])
    for where, value in [('root', root_v), ('cur', cur_v)]:
        seen = get(where + '_ns') > 0
        for kind in ('mean', 'max'):
            add(where + '_q_' + kind + '_adv',
                np.where(seen, get(where + '_qs_' + kind) - value, np.nan))
    add('cur_reward', get('cur_r'))
    add('cur_done', get('cur_d'))
    add('cur_minus_root_value', cur_v - root_v)
    add('trail_reward', get('root_trail_r'))
    add('rollout_minus_root_value', get('rollout_return') - root_v)
    add('max_rollout_minus_root_value', get('max_rollout_return') - root_v)
    add('policy_change', softmax(get('cur_policy'), axis=1) - softmax(get('root_policy'), axis=1))
    controls = np.column_stack([get(k) for k in (
        'root_action', 'cur_action', 'root_ns', 'cur_ns', 'tree_reset',
        'depth_discount', 'search_steps', 'rollout_depth', 'search_start')] +
        [(get(k) > 0).astype(float) for k in ('root_ns', 'cur_ns')])
    return np.column_stack(values).astype('float32'), controls.astype('float32'), names

class ActionTimeline:
    """Exact integrals of piecewise-constant logged human actions."""
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as raw:
            self.t=raw['time'].astype(float)
            self.a=(raw['action'].argmax(1) if raw['action'].ndim==2 else raw['action']).astype(int)
            self.nact=raw['action'].shape[1] if raw['action'].ndim==2 else int(self.a.max()+1)
            first=raw['is_first'].astype(bool);terminal=raw['is_terminal'].astype(bool)
        if not np.all(np.isfinite(self.t)) or np.any(np.diff(self.t)<0):raise ValueError('Nonmonotonic behavioral timestamps')
        dt=np.diff(self.t);one=np.eye(self.nact)[self.a]
        self.one=one;self.integral=np.vstack([np.zeros(self.nact),np.cumsum(one[:-1]*dt[:,None],axis=0)])
        change=np.r_[False,self.a[1:]!=self.a[:-1]]
        count=np.zeros((len(self.a),self.nact*self.nact))
        ix=np.flatnonzero(change);count[ix,self.a[ix-1]*self.nact+self.a[ix]]=1
        self.transitions=np.vstack([np.zeros(self.nact*self.nact),np.cumsum(count,axis=0)])
        bad=first|terminal|np.r_[False,(dt>.25)|(dt<=0)]
        self.bad=np.r_[0,np.cumsum(bad)]

    def area(self,x):
        x=np.asarray(x);j=np.searchsorted(self.t,x,side='right')-1
        if np.any(j<0) or np.any(x>self.t[-1]+1e-7):raise ValueError('Action integral outside observed time')
        j=np.clip(j,0,len(self.t)-1)
        return self.integral[j]+(x-self.t[j])[...,None]*self.one[j]

    def counts(self,start,end):
        a=np.searchsorted(self.t,start,side='right');b=np.searchsorted(self.t,end,side='right')
        return self.transitions[b]-self.transitions[a]

    def valid(self,start,end):
        a=np.searchsorted(self.t,start,side='right');b=np.searchsorted(self.t,end,side='right')
        return (start>=self.t[0])&(end<=self.t[-1])&((self.bad[b]-self.bad[a])==0)

    def profile(self,start,end):
        mid=(start+end)/2;width=end-start
        early=(self.area(mid)-self.area(start))/(width/2)
        late=(self.area(end)-self.area(mid))/(width/2)
        rate=self.counts(start,end)/width
        j=np.searchsorted(self.t,end,side='right')-1
        return np.r_[early,late,late-early,rate,self.one[j],np.log1p(rate.sum())]

def sa_indices(times, events, grid, max_age=.12):
    """Previous observation only; never interpolate across the real switch."""
    targets = np.asarray(events)[:, None] + np.asarray(grid)[None, :]
    ix = np.searchsorted(times, targets, side='right')-1
    safe = np.maximum(ix, 0)
    valid = ((ix >= 0) & (targets <= times[-1]) &
             (targets-times[safe] <= max_age))
    return np.where(valid, ix, -1)

def sa_sample(values, indices):
    result = values[np.maximum(indices, 0)].astype(np.float32)
    result[indices < 0] = np.nan
    return result

def sa_mean(values, axis=0):
    count = np.isfinite(values).sum(axis=axis)
    return np.divide(np.nansum(values, axis=axis), count,
                     out=np.full(count.shape, np.nan, dtype=float), where=count > 0)

# Public Trace/decision_edges/sha/write_json above remain compatible with 13/02.

def digest(obj):
    return hashlib.sha256(json.dumps(clean_json(obj), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_chunks(t, names, events=None):
    """Authenticate exactly the native packs read, including validity masks."""
    names = list(names) + [n+'_valid' for n in names if n+'_valid' in t.specs]
    groups = {t.specs[n]['storage_group'] for n in names}
    selected = {g: None if events is None else set((np.asarray(events)//t.z['packs/'+g].chunks[0]).tolist()) for g in groups}
    for f in t.manifest['files']:
        rel = f['path']; parts = rel.split('/')
        needed = rel in ('.zmetadata', '.zattrs', '.zgroup')
        if len(parts) >= 3 and parts[0] == 'packs' and parts[1] in groups:
            needed = parts[-1].startswith('.') or selected[parts[1]] is None or int(parts[-1].split('.')[0]) in selected[parts[1]]
        if needed:
            p = t.path/rel
            if p.stat().st_size != f['size'] or sha(p) != f['sha256']:
                raise ValueError('Changed native chunk: '+str(p))


def audit_game(args, game, out):
    import yaml
    capture, cp = game_inputs(args.input_root, game, args.subject)
    cm = read_json(capture/'manifest.json'); completed = read_json(capture/'COMPLETED.json')
    unsigned = dict(cm); claim = unsigned.pop('manifest_sha256')
    if digest(unsigned) != claim or completed['manifest_sha256'] != claim or completed['status'] != 'COMPLETED':
        raise ValueError('Invalid completed capture')
    if digest(cm['episodes']) != cm['episodes_sha256']: raise ValueError('Episode index changed')
    cfg = yaml.safe_load((cp/'config_c.yaml').read_text())
    for key, value in [('critic_enc_type', 0), ('model_enc_type', 0), ('last_layer_n', 0), ('sep_actor_critic', False)]:
        if cfg.get(key) != value: raise ValueError('Unsupported reconstruction: '+key)
    if cfg.get('model_ordinal') or cfg.get('ordinal'): raise ValueError('Ordinal head needs a different adapter')
    files = [cp/'config_c.yaml', cp/'bundle.json', next(cp.glob('ckp_actor*')), next(cp.glob('ckp_model*'))]
    hashes = {str(p): sha(p) for p in files}; episodes = []; sources = {}; stamps = {}
    for rec in cm['episodes']:
        p = capture/rec['path']; m = read_json(p/'manifest.json'); meta = m['metadata']
        if m['state'] != 'complete' or m['metadata'] != rec['metadata'] or m['event_count'] != rec['event_count']:
            raise ValueError('Episode metadata mismatch')
        if digest(m['files']) != m['files_sha256'] or m['files_sha256'] != rec['files_sha256']:
            raise ValueError('Episode chunk index mismatch')
        for key, f in [('actor',files[2]),('model',files[3]),('config',files[0])]:
            if meta['checkpoint_'+key+'_sha256'] != hashes[str(f)]: raise ValueError('Capture/checkpoint mismatch')
        raw = Path(meta['source_path'])
        if not raw.exists(): raw = RESEARCH.parent/'behavioral_data_block'/meta['source_id']
        if str(raw) not in sources: sources[str(raw)] = sha(raw)
        if sources[str(raw)] != meta['source_sha256']: raise ValueError('Changed human source')
        if meta['subject'] != args.subject or meta['game'] != GAME_IDS[game]:
            raise ValueError('Capture subject/game does not match requested analysis')
        episodes.append(dict(path=str(p), source=str(raw), episode=meta['episode_id'], session=meta['session'],
                             run=meta['source_id'], manifest_sha256=sha(p/'manifest.json')))
        for f in p.rglob('*'):
            if f.is_file(): stamps[str(f)] = file_stamp(f)
    for p in [*files, *map(Path,sources), capture/'manifest.json']:
        stamps[str(p)] = file_stamp(p)
    write_json(out/'audit.json', dict(subject=args.subject, game=game, episodes=episodes, sources=sources, checkpoint=str(cp), config=cfg,
        num_actions=m['num_actions'], checkpoint_hashes=hashes, input_stamps=stamps,
        chunk_verification='All chunks read are verified by extraction/representation; smoke reads selected chunks only',
        limitations=['one participant per game','Thinker trained on all four sessions','observational human-forced replay']))


class ActorHeads:
    """Exact frozen linear readouts; no recurrent or environment counterfactual."""
    def __init__(self, audit):
        import torch
        cp = Path(audit['checkpoint']); self.cfg = audit['config']
        c = torch.load(next(cp.glob('ckp_actor*')), map_location='cpu', weights_only=False)
        self.state = {k:v.detach().double().numpy() for k,v in c['actor_net_state_dict'].items()}
        self.phase = self.state['phase_embedding.weight']
        self.ema = {k:v.detach().double().numpy() for k,v in c['voc_ema_gate_head_state_dict'].items()}
        self.nact = audit['num_actions']; self.width = self.phase.shape[1]
        nr = 1 + int(self.cfg['im_cost'] > 0) + int(self.cfg['cur_cost'] > 0)
        self.learned_start = self.nact+2+nr
        self.gru_slice = slice(self.learned_start, self.learned_start+self.cfg['dynamic_search_hidden_dim'])
        self.clip = self.cfg['reward_clip']/(1-self.cfg['discounting']) if self.cfg['reward_clip'] > 0 else np.inf

    def linear(self, x, name):
        return x@self.state[name+'.weight'].T+self.state[name+'.bias']

    def common(self, features, phases, target=0):
        return np.asarray(features, float)-self.phase[np.asarray(phases,int)]+self.phase[target]

    def outputs(self, search_features):
        x = np.asarray(search_features,float)
        value = self.linear(x,'baseline')[:,0]
        real_x = x-self.phase[0]+self.phase[1]
        logits = self.linear(real_x,'policy')
        raw = self.linear(x,'voc_head'); ema = x@self.ema['weight'].T+self.ema['bias']
        return value, softmax(logits,axis=1), raw[:,0]-raw[:,1], ema[:,0]-ema[:,1]

    def weights(self, target):
        if target == 'value': return self.state['baseline.weight'][:1]
        if target == 'policy':
            w=self.state['policy.weight']; return w-w.mean(0)
        if target == 'voc': return (self.ema['weight'][0]-self.ema['weight'][1])[None]
        raise KeyError(target)

    def check(self, t, rows, features):
        phase=t.scalar('phase_pre')[rows].astype(int); checks={}
        actual=self.linear(features,'baseline'); ref=t.read('actor_baseline',rows)
        if np.any(np.abs(ref[:,0]) >= self.clip-1e-5): raise ValueError('Saturated task value cannot use linear phase correction')
        checks['baseline']=float(np.max(np.abs(actual-np.asarray(ref))))
        for name,key,w,b in [('online','voc_online_raw_q',self.state['voc_head.weight'],self.state['voc_head.bias']),
                              ('ema','voc_ema_raw_q',self.ema['weight'],self.ema['bias'])]:
            raw=features@w.T+b; recorded=t.read(key,rows)
            checks[name+'_raw']=float(np.max(np.abs(raw-recorded)))
            probability=t.scalar('voc_soft_continue_probability')[rows]
            sv=t.scalar('voc_state_value')[rows]
            mode=t.scalar('voc_reconstruction_mode')[rows]
            reconstructed=raw.copy()
            gate=np.column_stack([probability,1-probability]); centered=raw-(gate*raw).sum(1,keepdims=True)
            # Constants are owned by the frozen trace schema, resolved below.
            native=frozen_import(self.cfg_root,'dynamic_imitation')
            for enum,common in [(native.VOC_Q_RECONSTRUCTION_DUELING_CENTERED,False),
                                (native.VOC_Q_RECONSTRUCTION_DUELING_COMMON,True)]:
                take=mode==enum
                reconstructed[take]=sv[take,None]+centered[take]+(raw[take].mean(1,keepdims=True) if common else 0)
            checks[name+'_q']=float(np.max(np.abs(reconstructed-t.read('voc_'+name+'_q',rows))))
        valid=t.valid('actor_real_logits_pre_override')[rows]
        if valid.any():
            checks['real_logits']=float(np.max(np.abs(self.linear(features[valid],'policy')-t.read('actor_real_logits_pre_override',rows[valid]))))
        if max(checks.values())>2e-4: raise ValueError('Frozen actor reproduction failed: '+str(checks))
        return checks


def value_summary(v):
    dv=np.diff(v); k=len(dv); total=float(np.abs(dv).sum()); net=float(dv.sum())
    return dict(K=k,A=total,D=net,U=total/k if k else 0., coherence=abs(net)/total if total>0 else np.nan,
                first_dV=float(dv[0]) if k else 0., V_initial=float(v[0]),V_final=float(v[-1]))


def timeline_segments(d):
    bad=d.episode.ne(d.episode.shift())|d.decision.diff().ne(1)|d.time.diff().le(0)|d.time.diff().gt(.25)
    d=d.copy();d['segment']=bad.cumsum();return d


def extract_game(args, game, out):
    a=read_json(out/'audit.json'); head=ActorHeads(a);head.cfg_root=args.input_root
    records=[]; events=[]; actor=[]; policies=[]; actual_policies=[]; controls=[]; images=[]; tree=[]; quality=[]
    for ep in a['episodes']:
        t=Trace(ep['path'])
        names=['initial_root','real_transition','search_computation','current_node_depth_post','current_node_depth_pre','actor_reset',
               'score_mask','human_action','pre_observation_source_index','phase_pre','search_steps_pre',
               'control_valid','accepted_control','forced_stop','root_carried','accepted_primary_action','done']
        verify_chunks(t,names);f={n:t.scalar(n) for n in names}
        tl=ActionTimeline(ep['source'])
        edges=[e for e in decision_edges(f['initial_root'],f['real_transition'],f['search_computation'],
               f['current_node_depth_post'],f['actor_reset']) if f['score_mask'][e[2]]]
        available=len(edges)
        if args.smoke and available>args.smoke_decisions:
            start=(available-args.smoke_decisions)//2;edges=edges[start:start+args.smoke_decisions]
        selected=np.unique(np.concatenate([np.r_[left,right,comp,comp+1,left+1] for _,left,right,comp,_ in edges]))
        fields=['voc_features','actor_baseline','voc_online_raw_q','voc_ema_raw_q','voc_online_q','voc_ema_q',
                'voc_state_value','voc_reconstruction_mode','voc_soft_continue_probability','actor_real_logits_pre_override',
                'actor_action_probability','tree_reps_pre','tree_reps_post']
        verify_chunks(t,fields,selected)
        verify_chunks(t,['real_image'],np.array([e[1] for e in edges]))
        all_features=t.read('voc_features',selected).astype(float); position={int(x):i for i,x in enumerate(selected)}
        usable=selected[t.valid('voc_features')[selected] & t.valid('actor_baseline')[selected]]
        ix=np.array([position[int(x)] for x in usable]); qc=head.check(t,usable,all_features[ix])
        pre=t.read('tree_reps_pre');post=t.read('tree_reps_post');verify_chunks(t,['tree_reps_pre','tree_reps_post'])
        content,_,_=tree_content(post,t.tree_schema)
        pre_content,_,_=tree_content(pre,t.tree_schema)
        def tf(raw,key):
            x,y,s=t.tree_schema[key];return raw[:,slice(x,y,s)]
        q0,q1=tf(pre,'root_qs_mean'),tf(post,'root_qs_mean');n0,n1=tf(pre,'root_ns'),tf(post,'root_ns')
        for j,left,right,comp,prefix in edges:
            src=int(f['pre_observation_source_index'][right]); when=float(tl.t[src]);human=int(f['human_action'][right]);K=len(comp)
            if human != tl.a[src]: raise ValueError('Human frame/action mismatch')
            if K and not np.all(np.diff(comp)==1): raise ValueError('Noncontiguous computations need timing audit')
            ar=np.r_[comp[0],comp+1].astype(int) if K else np.array([left+1])
            if ar[-1]>right or f['search_computation'][ar[-1]:right].any():
                raise ValueError('Unaccounted computation before REAL')
            if not t.valid('voc_features')[ar].all(): raise ValueError('Missing actor feature')
            np.testing.assert_array_equal(f['search_steps_pre'][ar],np.arange(K+1))
            np.testing.assert_array_equal(f['pre_observation_source_index'][ar],np.full(K+1,src))
            z=all_features[[position[int(x)] for x in ar]];zs=head.common(z,f['phase_pre'][ar])
            v,p,vo,ve=head.outputs(zs)
            real_feature=all_features[position[right]:position[right]+1]
            real_p=softmax(head.linear(real_feature,'policy'),axis=1)[0]
            np.testing.assert_allclose(real_p,t.read('actor_action_probability',[right]).ravel(),atol=2e-5,rtol=0)
            actual_policies.append(real_p)
            if np.any(np.abs(v)>=head.clip-1e-5): raise ValueError('Common-phase task value saturated')
            row=len(records);begin=len(events);model_rows=np.r_[left,comp].astype(int)
            delta=np.r_[0,np.diff(v)]; reference=p[0]
            for k,(ev,mev) in enumerate(zip(ar,model_rows)):
                voluntary=bool(f['control_valid'][ev] and not f['forced_stop'][ev] and k>=a['config']['min_search_steps'])
                events.append(dict(row=row,k=k,actor_event=int(ev),model_event=int(mev),trace=ep['path'],
                    session=ep['session'],run=ep['run'],episode=ep['episode'],time=when,
                    value=v[k],delta_V=delta[k],voc_online=vo[k],voc_ema=ve[k],
                    recorded_value=float(t.read('actor_baseline',[ev])[0,0]),
                    gate_probability=float(t.scalar('voc_soft_continue_probability')[ev]),
                    continue_choice=int(f['accepted_control'][ev] in (0,1)) if f['control_valid'][ev] else np.nan,
                    voluntary=voluntary,phase=int(f['phase_pre'][ev]),
                    policy_change=float(js_distance(reference[None],p[k:k+1])[0]),
                    first_branch=int(f['accepted_primary_action'][comp[0]]) if K else -1,
                    node_depth=int(f['current_node_depth_post'][mev]), reset=bool(f['actor_reset'][mev])))
                actor.append(zs[k].astype('float32')); policies.append(p[k].astype('float32'))
            previous=int(tl.a[src-1]) if src else -1
            switches=np.flatnonzero(tl.a[1:]!=tl.a[:-1])+1
            preceding=switches[switches<=src]; age=when-tl.t[preceding[-1]] if len(preceding) else when-tl.t[0]
            strictly_prior=switches[switches<src]
            age_prior=when-tl.t[strictly_prior[-1]] if len(strictly_prior) else when-tl.t[0]
            tree_seq=np.vstack([pre_content[ar[0]],content[comp]]) if K else pre_content[ar[:1]]
            # Mean and endpoint changes do not concatenate distinct future branches into a value trajectory.
            finite=np.isfinite(tree_seq); cnt=finite.sum(0)
            avg=np.divide(np.nansum(tree_seq,axis=0),cnt,out=np.full(tree_seq.shape[1],np.nan),where=cnt>0)
            tree.append(np.r_[tree_seq[0],tree_seq[min(1,K)],tree_seq[-1],avg])
            branch=-1;validq=[];qtotal=0.
            for ce in comp:
                if f['current_node_depth_pre'][ce]==0:branch=int(f['accepted_primary_action'][ce])
                if not 0<=branch<head.nact:raise ValueError('Missing evaluated root branch')
                valid=bool(n0[ce,branch]>0 and n1[ce,branch]>0);validq.append(valid)
                if valid:qtotal+=abs(float(q1[ce,branch]-q0[ce,branch]))
            validq=np.asarray(validq,bool)
            summary=value_summary(v)
            records.append(dict(row=row,game=game,subject=args.subject,session=ep['session'],run=ep['run'],episode=ep['episode'],
                source=ep['source'],trace=ep['path'],decision=j,root_event=left,real_event=right,
                source_frame=src,time=when,human_action=human,previous_action=previous,age=age,age_prior=age_prior,
                raw_switch=int(previous>=0 and human!=previous),start=begin,end=len(events),
                first_action=int(f['accepted_primary_action'][comp[0]]) if K else -1,
                forced_stop=bool(f['forced_stop'][ar[-1]]),carry=int(f['root_carried'][left]),
                voc_initial=ve[0],voc_first=ve[min(1,K)],voc_final=ve[-1],voc_mean=float(ve.mean()),
                voc_online_initial=vo[0],policy_change=float(js_distance(p[:1],p[-1:])[0]),
                stop_policy_change=float(js_distance(p[-1:],real_p[None])[0]),
                total_policy_change=float(js_distance(p[:1],real_p[None])[0]),
                stop_value_change=float(head.linear(head.common(real_feature,[f['phase_pre'][right]]),'baseline')[0,0]-v[-1]),
                stop_learned_change=float(np.linalg.norm(head.common(real_feature,[f['phase_pre'][right]])[0,head.learned_start:]-zs[-1,head.learned_start:])),
                policy_first_change=float(js_distance(p[:1],p[min(1,K):min(1,K)+1])[0]),
                Q_total=qtotal,Q_comparisons=int(validq.sum()),Q_coverage=float(validq.mean()) if K else 0.,
                **summary))
            image=t.read('real_image',[left])[0];images.append(image)
            controls.append(image.astype(float).reshape(3,7,12,7,12).mean(axis=(2,4)).reshape(-1)/255.)
        quality.append(dict(episode=ep['episode'],available=available,selected=len(edges),max_errors=qc))
        log(f'{game}: {len(records):,} decisions; {len(events):,} computation endpoints')
    d=timeline_segments(pd.DataFrame(records));e=pd.DataFrame(events)
    d.to_csv(out/'decisions.csv.gz',index=False);e.to_csv(out/'computations.csv.gz',index=False)
    for name,values in [('actor_features',actor),('policies',policies),('actual_policy',actual_policies),('screen',controls),('tree_summary',tree),('screens',images)]:
        np.save(out/(name+'.npy'),np.asarray(values,dtype='uint8' if name=='screens' else 'float32'))
    write_json(out/'extraction.json',dict(n=len(d),endpoints=len(e),quality=quality,task_channel=0,
        actor_width=head.width,learned_start=head.learned_start,gru_slice=[head.gru_slice.start,head.gru_slice.stop],
        U='sum(abs(diff(common_SEARCH_task_value)))/K; K=0 -> 0',
        policy='common REAL phase at computation endpoints; actual REAL output stored separately; STOP input changes are a separate effect',
        Q='Only visited-to-visited root Q revisions; missing comparisons remain explicit',
        no_branch_path_claim=True,smoke=args.smoke))

def load_latents(e, ids, field, verify=False):
    ids=np.asarray(ids,int); result=np.empty((len(ids),4608),np.float32)
    selected=e.iloc[ids].copy();selected['_order']=np.arange(len(ids))
    for path,g in selected.groupby('trace',sort=False):
        t=Trace(path);ev=g.model_event.to_numpy(int)
        if verify: verify_chunks(t,[field],ev)
        if not t.valid(field)[ev].all(): raise ValueError('Missing '+field+' activation')
        for ii in chunks(np.arange(len(g)),128):
            result[g.iloc[ii]._order.to_numpy(int)]=t.read(field,ev[ii]).reshape(len(ii),-1)
    return result


def fit_key(sessions):
    return 'train_'+'_'.join(map(str,sorted(sessions)))


def fit_sets(sessions):
    return [tuple(x) for k in (2,3) for x in itertools.combinations(sorted(sessions),k)]


def finite_scale(y):
    s=float(np.sqrt(np.var(y,axis=0).sum()))
    return s if s>1e-8 else 0.


def summarize_coordinates(z,d):
    """Fixed-stage and whole-computation summaries. No padding of short searches."""
    initial=z[d.start.to_numpy(int)];first=z[np.minimum(d.start.to_numpy(int)+1,d.end.to_numpy(int)-1)]
    final=z[d.end.to_numpy(int)-1]
    means=np.vstack([z[r.start:r.end].mean(0) for r in d.itertuples()])
    return np.column_stack([initial,first-initial,final-initial,means-initial]).astype('float32')


def local_head_reproduction_values(values, net, config):
    """Match saved-node reward semantics without changing raw ACE targets."""
    result = np.array(values, copy=True)
    if net == 'sr' and config['reward_clip'] > 0:
        # ModelNet._prepare_out clips SRN rewards before cenv stores cur_r.
        clip = config['reward_clip']
        result[:, 0] = np.clip(result[:, 0], -clip, clip)
    return result


def representation_game(args,game,out):
    d=read_table(out/'decisions.csv.gz');e=read_table(out/'computations.csv.gz');a=read_json(out/'audit.json')
    actor=np.load(out/'actor_features.npy',mmap_mode='r');ah=ActorHeads(a);nh=FrozenHeads(args,out)
    fields={'sr':'im_vectors','vp':'im_vp_vectors'};sessions=sorted(d.session.unique());summary=[]
    # Authenticate the full projected endpoint support, not only fit samples.
    for path,g in e.groupby('trace',sort=False):
        verify_chunks(Trace(path),list(fields.values()),g.model_event.to_numpy(int))
    for train_sessions in fit_sets(sessions):
        folder=out/'representation'/fit_key(train_sessions);folder.mkdir(parents=True,exist_ok=True)
        if resume_unit(folder,args):
            summary.extend(read_json(folder/'summary.json'));continue
        train=d.index[d.session.isin(train_sessions)].to_numpy()
        # Equal-run, equal-decision fitting; one uniformly chosen k per decision.
        rng=np.random.default_rng(args.seed);picked=sample_runs(d,train,args.basis_samples,args.seed)
        endpoint=np.array([rng.integers(d.loc[i,'start'],d.loc[i,'end']) for i in picked],int)
        weights=balanced_weights(d,picked);fold_summary=[];bank={};fixed={};coordinates={}
        held=d.index[~d.session.isin(train_sessions)].to_numpy()
        val_dec=sample_runs(d,held,min(args.jacobian_samples,len(held)),args.seed+9)
        val_ep=np.minimum(d.loc[val_dec,'start'].to_numpy(int)+1,d.loc[val_dec,'end'].to_numpy(int)-1)
        for net in ('sr','vp','actor'):
            log(f'{game} {fit_key(train_sessions)}: {net} functional spaces')
            x=load_latents(e,endpoint,fields[net],True) if net!='actor' else np.asarray(actor[endpoint,ah.learned_start:])
            mean,factor=activation_factor(x,np.arange(len(x)),weights,args.chunk_size)
            if net!='actor':
                y=nh.predict(net,x,contrast=True)
                n=min(args.jacobian_samples,len(x));rawgrad=nh.jacobian(net,x[:n],np.ones(y.shape[1])).reshape(n,y.shape[1],-1)
                targets={'sr_reward_terminal':[0,1]} if net=='sr' else {'vp_value':[0],'vp_policy':list(range(1,y.shape[1]))}
                xv=load_latents(e,val_ep,fields[net],True);yv=nh.predict(net,xv,contrast=True)
                # Check directly recorded local heads only where the evaluated node is retained.
                errors=[]
                for path,g in e.iloc[val_ep].assign(_order=np.arange(len(val_ep))).groupby('trace'):
                    t=Trace(path);ev=g.model_event.to_numpy(int);idx=g._order.to_numpy(int)
                    ok=(g.k.to_numpy()>0)&(~t.scalar('actor_reset')[ev].astype(bool))&(t.tree('cur_d',ev).ravel()==0)
                    if net=='sr':
                        ok &= t.valid('model_done_logit')[ev] & (t.scalar('model_done_target_depth')[ev]==t.scalar('current_node_depth_post')[ev])
                        ref=np.column_stack([t.tree('cur_r',ev),t.scalar('model_done_logit')[ev]])
                    else:
                        logits=t.tree('cur_policy',ev);ref=np.column_stack([t.tree('cur_v',ev),logits-logits.mean(1,keepdims=True)])
                    if ok.any():
                        reproduced=local_head_reproduction_values(yv[idx[ok]],net,a['config'])
                        error=np.max(np.abs(reproduced-ref[ok]),axis=0);tol=np.r_[.002,np.full(len(error)-1,.02)]
                        if np.any(error>tol):raise ValueError('Local head reproduction failed: '+str(error))
                        errors.append(error.tolist())
            else:
                full=np.asarray(actor[endpoint]);v,p,vo,ve=ah.outputs(full)
                logits=ah.linear(full-ah.phase[0]+ah.phase[1],'policy');logits-=logits.mean(1,keepdims=True)
                y=np.column_stack([v,logits,ve]);targets={'actor_value':[0],'actor_policy':list(range(1,1+ah.nact)),
                                                        'actor_voc':[ah.nact+1],'actor_joint':list(range(ah.nact+2))}
                w=np.vstack([ah.weights('value'),ah.weights('policy'),ah.weights('voc')])[:,ah.learned_start:]
                rawgrad=w[None];xv=np.asarray(actor[val_ep,ah.learned_start:]);yv=None;errors=[]
            for label,cols in targets.items():
                scale=np.ones(len(cols))
                if label=='sr_reward_terminal':
                    scale=np.array([finite_scale(y[:,[c]]) for c in cols])*np.sqrt(2)
                elif label=='actor_joint':
                    for block in ([0],list(range(1,1+ah.nact)),[ah.nact+1]):scale[block]=finite_scale(y[:,block])*np.sqrt(3)
                else:scale[:]=finite_scale(y[:,cols])
                grad_tensor=np.divide(rawgrad[:,cols],scale[None,:,None],out=np.zeros_like(rawgrad[:,cols]),where=scale[None,:,None]>0)
                grad=grad_tensor.reshape(-1,x.shape[1])
                ordered,spectrum,needed,retained=ace_basis(factor,grad,args.energy,args.max_rank)
                rank=min(needed,ordered.shape[1]);basis=ordered[:,:rank]
                # Sign convention fixed on training data only; never on switch labels.
                for col in range(rank):
                    if basis[np.argmax(abs(basis[:,col])),col]<0:basis[:,col]*=-1
                trainz=(x-mean)@basis
                sd=np.sqrt(np.average((trainz-np.average(trainz,axis=0,weights=weights))**2,axis=0,weights=weights));sd=np.maximum(sd,1e-8)
                np.savez_compressed(folder/(label+'_basis.npz'),mean=mean,basis=basis,sd=sd,spectrum=spectrum,
                                    train_decisions=picked,train_endpoints=endpoint,output_scale=scale)
                validation={}
                if len(xv) and rank:
                    keep=intervene(xv,basis,mean,'keep');remove=intervene(xv,basis,mean,'remove')
                    if net!='actor':
                        pred=nh.predict(net,xv,contrast=True)[:,cols]
                        pk=nh.predict(net,keep,contrast=True)[:,cols];pr=nh.predict(net,remove,contrast=True)[:,cols]
                    else:
                        pred=xv@w[cols].T;pk=keep@w[cols].T;pr=remove@w[cols].T
                    validation=dict(keep_rmse=float(np.sqrt(np.mean((pk-pred)**2))),remove_rmse=float(np.sqrt(np.mean((pr-pred)**2))),
                                    output_sd=float(np.std(pred)),scope='local head only; other actor blocks fixed')
                # Stability against independent equal-run decision sample, same covariance convention.
                stability=np.nan
                if len(grad_tensor)>2:
                    b2,_,r2,_=ace_basis(factor,grad_tensor[::2].reshape(-1,x.shape[1]),args.energy,args.max_rank)
                    stability=subspace_overlap(basis,b2[:,:min(r2,b2.shape[1])])
                fold_summary.append(dict(fit=fit_key(train_sessions),target=label,rank=rank,required_rank=needed,
                    retained_energy=retained,first_energy=float(spectrum[0]/spectrum.sum()) if spectrum.sum() else 0.,
                    sensitivity_subsample_overlap=stability,training_decisions=len(picked),equal_decision_stage_sampling=True,
                    local_head_errors=errors,**validation))
                coordinates[label]=np.lib.format.open_memmap(folder/(label+'_events.npy'),mode='w+',dtype='float32',shape=(len(e),rank))
                bank[label]=(net,mean,basis,sd)
            del x,factor
        # Stream all computation endpoints, preserving RESET/branch metadata separately.
        for net in ('sr','vp','actor'):
            labels=[k for k,v in bank.items() if v[0]==net]
            for ids in chunks(np.arange(len(e)),args.chunk_size):
                x=load_latents(e,ids,fields[net]) if net!='actor' else np.asarray(actor[ids,ah.learned_start:])
                for label in labels:
                    _,mean,basis,sd=bank[label];coordinates[label][ids]=((x-mean)@basis/sd).astype('float32')
        for label,z in coordinates.items():
            z.flush();np.save(folder/(label+'_decisions.npy'),summarize_coordinates(z,d))
            i=d.start.to_numpy(int);j=np.minimum(i+1,d.end.to_numpy(int)-1)
            np.save(folder/(label+'_first.npy'),np.column_stack([z[i],z[j]-z[i]]))
        # Actual GRU slice is available in the fused feature vector; save without claiming a full recurrent intervention.
        gru=np.asarray(actor[:,ah.gru_slice]);np.save(folder/'gru_decisions.npy',summarize_coordinates(gru,d))
        write_json(folder/'summary.json',fold_summary);complete_unit(folder);summary.extend(fold_summary)
    save_table(out/'representation_summary.csv',summary)


DIRECT=['V_initial','first_dV','A','D','U','coherence','voc_initial','voc_first','voc_final','voc_mean']
FIRST=['V_initial','first_dV','voc_initial','voc_first']


def window_features(x,d,w):
    """Only observations at/before endpoint; no centered smoothing or future padding."""
    x=np.asarray(x,float);answer=np.full((len(w),6*x.shape[1]),np.nan,np.float32)
    for i,r in enumerate(w.itertuples()):
        ids=np.arange(r.left,r.row+1);v=x[ids];t=d.time.to_numpy()[ids];dt=t-t.mean()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            mean=np.nanmean(v,axis=0);sd=np.nanstd(v,axis=0)
            valid=np.isfinite(v);n=valid.sum(0)
            local_t=np.divide((valid*t[:,None]).sum(0),n,out=np.zeros(x.shape[1]),where=n>0)
            centered=t[:,None]-local_t;den=(valid*centered**2).sum(0)
            slope=np.divide(np.nansum(centered*(v-mean),axis=0),den,out=np.zeros(x.shape[1]),where=den>1e-12)
            path=np.nansum(np.abs(np.diff(v,axis=0)),axis=0)
        answer[i]=np.r_[v[-1],mean,sd,slope,v[-1]-v[0],path]
    return answer


def temporal_game(args,game,out):
    d=read_table(out/'decisions.csv.gz');policy=np.load(out/'actual_policy.npy');timelines={p:ActionTimeline(p) for p in d.source.unique()}
    rows=[];action=[];raw_events=[]
    for segment,g in d.groupby('segment',sort=False):
        ids=g.index.to_numpy();times=g.time.to_numpy();tl=timelines[g.source.iloc[0]]
        changes=np.flatnonzero(tl.a[1:]!=tl.a[:-1])+1
        chosen=changes[(tl.t[changes]>=times[0])&(tl.t[changes]<=times[-1])]
        for ix in chosen:
            raw_events.append(dict(segment=segment,session=g.session.iloc[0],run=g.run.iloc[0],time=tl.t[ix],
                source_frame=ix,action_from=tl.a[ix-1],action_to=tl.a[ix]))
        for width in args.windows_seconds:
            for pos,idx in enumerate(ids):
                start=times[pos]-width
                if start<times[0] or not tl.valid(start,times[pos]):continue
                left=np.searchsorted(times,start,side='right')-1
                if left<0 or start-times[left]>.12 or pos-left<2:continue
                r=dict(row=int(idx),left=int(ids[left]),width=width,segment=segment,session=g.session.iloc[0],
                       run=g.run.iloc[0],episode=g.episode.iloc[0],time=times[pos],human_action=int(d.loc[idx,'human_action']))
                for horizon in args.horizons_seconds:
                    end=times[pos]+horizon;key=f'{horizon:g}'
                    r['switch_'+key]=float(tl.counts(times[pos],end).sum()>0) if end<=times[-1] and tl.valid(times[pos],end) else np.nan
                    dest=np.searchsorted(times,end,side='right')-1
                    r['policy_'+key]=(float(js_distance(policy[idx],policy[int(ids[dest])]))
                        if end<=times[-1] and end-times[dest]<=.12 and np.isfinite(r['switch_'+key]) else np.nan)
                rows.append(r);action.append(tl.profile(start,times[pos]))
    w=pd.DataFrame(rows);w.to_csv(out/'windows.csv.gz',index=False);pd.DataFrame(raw_events).to_csv(out/'switches.csv.gz',index=False)
    np.save(out/'action_windows.npy',np.asarray(action,np.float32))
    for name,features in [('direct',d[DIRECT].to_numpy()),('first_direct',d[FIRST].to_numpy()),
                          ('budget',np.log1p(d[['K']].to_numpy())),('tree',np.load(out/'tree_summary.npy'))]:
        np.save(out/(name+'_windows.npy'),window_features(features,d,w))
    # Fold-specific representations; pair fits are used for completely nested K selection.
    for folder in sorted((out/'representation').glob('train_*')):
        pieces=[];first=[]
        for label in ('sr_reward_terminal','vp_value','vp_policy','actor_joint'):
            pieces.append(window_features(np.load(folder/(label+'_decisions.npy')),d,w))
            first.append(window_features(np.load(folder/(label+'_first.npy')),d,w))
        np.save(folder/'ace_windows.npy',np.column_stack(pieces));np.save(folder/'first_ace_windows.npy',np.column_stack(first))
    write_json(out/'temporal.json',dict(windows=len(w),raw_switches=len(raw_events),widths=args.windows_seconds,
        horizons=args.horizons_seconds,primary_width=.5,primary_horizon=.25,max_age=.12,all_switches=True,
        first_only_sensitivity='fixed k=1 features; K=0 carries zero change and explicit recorded K',
        overlapping_windows='Only nonoverlapping training windows; block uncertainty on held-out rows'))

class BlockProjection:
    """Train-only imputation/scaling, equal block energy, bounded PCA."""
    def fit(self, blocks, ids, seed=701):
        from sklearn.decomposition import PCA
        self.stats=[];pieces=[]
        for block in blocks:
            b=np.asarray(block[ids],float)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning);mu=np.nanmean(b,axis=0)
            mu=np.nan_to_num(mu);b=np.where(np.isfinite(b),b,mu)
            sd=b.std(0);keep=sd>1e-8
            self.stats.append((mu,np.maximum(sd,1e-8),keep))
            if keep.any():pieces.append((b[:,keep]-mu[keep])/sd[keep]/np.sqrt(keep.sum()))
        self.pca=None;self.rank=0;self.retained=0.
        if not pieces:return self
        x=np.column_stack(pieces);rank=min(16,x.shape[1],len(ids)-1)
        if rank<=0:return self
        self.pca=PCA(n_components=rank,svd_solver='randomized',random_state=seed).fit(x)
        self.rank=min(rank,int(np.searchsorted(np.cumsum(self.pca.explained_variance_ratio_),.95)+1))
        self.retained=float(self.pca.explained_variance_ratio_[:self.rank].sum())
        return self

    def transform(self,blocks):
        pieces=[]
        for block,(mu,sd,keep) in zip(blocks,self.stats):
            b=np.where(np.isfinite(block),block,mu)
            if keep.any():pieces.append((b[:,keep]-mu[keep])/sd[keep]/np.sqrt(keep.sum()))
        if self.pca is None:return np.zeros((len(blocks[0]),1))
        return self.pca.transform(np.column_stack(pieces))[:,:self.rank]

    def save(self,path):
        arrays={}
        for i,(mu,sd,keep) in enumerate(self.stats):arrays.update({f'mean_{i}':mu,f'sd_{i}':sd,f'keep_{i}':keep})
        if self.pca is not None:arrays.update(pca_mean=self.pca.mean_,pca_components=self.pca.components_[:self.rank])
        np.savez_compressed(path,**arrays)


def nonoverlap(w,ids,width):
    result=[]
    for _,g in w.loc[ids].groupby('segment',sort=False):
        last=-np.inf
        for r in g.sort_values('time').itertuples():
            if r.time-last>=width-1e-9:result.append(r.Index);last=r.time
    return np.asarray(result,int)


def feature_blocks(out,fit,view,first=False):
    folder=out/'representation'/fit_key(fit)
    actions=np.load(out/'action_windows.npy',mmap_mode='r')
    if view=='action_only':return [actions]
    direct=np.load(out/('first_direct_windows.npy' if first else 'direct_windows.npy'),mmap_mode='r')
    if view=='outputs_only':return [direct]
    blocks=[direct,np.load(folder/('first_ace_windows.npy' if first else 'ace_windows.npy'),mmap_mode='r')]
    if not first:blocks.append(np.load(out/'tree_windows.npy',mmap_mode='r'))
    if view.startswith('joint'):blocks.append(actions)
    if view.endswith('_budget'):blocks.append(np.load(out/'budget_windows.npy',mmap_mode='r'))
    return blocks


def mixture_fit(x,k,args):
    from sklearn.mixture import GaussianMixture
    if len(x)<max(k*3,3):raise ValueError('Too few windows for mixture')
    with warnings.catch_warnings():
        from sklearn.exceptions import ConvergenceWarning
        warnings.simplefilter('error',ConvergenceWarning)
        model=GaussianMixture(n_components=k,covariance_type='diag',reg_covar=1e-5,
            n_init=1 if args.smoke else 5,max_iter=300,random_state=args.seed).fit(x)
    return model


def one_se_choice(tab, folds):
    eligible=tab[tab.status.eq('ok')].groupby('k').score.agg(['mean','std','count'])
    eligible=eligible[eligible['count'].eq(folds)]
    if eligible.empty:raise ValueError('No candidate converged in every inner fold')
    best=eligible['mean'].idxmax();se=eligible.loc[best,'std']/np.sqrt(eligible.loc[best,'count'])
    return int(eligible.index[eligible['mean']>=eligible.loc[best,'mean']-se].min())


def select_mixture(args,out,w,train_sessions,view,width,first=False):
    """Each inner fold re-fits ACE, scaling and PCA without its validation session."""
    scores=[]
    for held in train_sessions:
        inner=tuple(s for s in train_sessions if s!=held)
        train=nonoverlap(w,w.index[w.session.isin(inner)&w.width.eq(width)].to_numpy(),width)
        valid=nonoverlap(w,w.index[w.session.eq(held)&w.width.eq(width)].to_numpy(),width)
        if len(train)<4 or len(valid)<2:continue
        blocks=feature_blocks(out,inner,view,first);proj=BlockProjection().fit(blocks,train,args.seed)
        xt=proj.transform([b[train] for b in blocks]);xv=proj.transform([b[valid] for b in blocks])
        for k in range(1,(2 if args.smoke else 8)+1):
            if len(train)<k*3:continue
            try:
                gm=mixture_fit(xt,k,args);score=float(gm.score(xv));status='ok'
            except (ValueError,RuntimeError,Warning) as exc:score=np.nan;status=type(exc).__name__
            scores.append(dict(held=held,k=k,score=score,status=status))
    tab=pd.DataFrame(scores)
    if tab.empty:raise ValueError('No inner validation support')
    return one_se_choice(tab,len(train_sessions)),tab


def motifs_game(args,game,out):
    w=read_table(out/'windows.csv.gz');rows=[];sessions=sorted(w.session.unique())
    views=['evaluation','evaluation_budget','joint','joint_budget','action_only','outputs_only']
    for held in sessions:
        fit=tuple(s for s in sessions if s!=held)
        for width in args.windows_seconds:
            train=nonoverlap(w,w.index[w.session.isin(fit)&w.width.eq(width)].to_numpy(),width)
            test=w.index[w.session.eq(held)&w.width.eq(width)].to_numpy()
            for view in views:
                folder=out/'motifs'/f'held_{held}'/f'w{width:g}'/view;folder.mkdir(parents=True,exist_ok=True)
                if resume_unit(folder,args):continue
                k,scores=select_mixture(args,out,w,fit,view,width)
                blocks=feature_blocks(out,fit,view);prep=BlockProjection().fit(blocks,train,args.seed)
                xt=prep.transform([b[train] for b in blocks]);xv=prep.transform([b[test] for b in blocks])
                gm=mixture_fit(xt,k,args);prob=gm.predict_proba(xv);state=prob.argmax(1)
                frame=w.loc[test,['row','width','segment','session','run','episode','time','human_action']].copy()
                frame['window']=test;frame['state']=state;frame['confidence']=prob.max(1)
                for c in range(k):frame[f'p{c}']=prob[:,c]
                frame.to_csv(folder/'assignments.csv.gz',index=False);np.save(folder/'test_embedding.npy',xv);prep.save(folder/'projection.npz')
                np.savez_compressed(folder/'mixture.npz',weights=gm.weights_,means=gm.means_,covariances=gm.covariances_)
                save_table(folder/'selection.csv',scores)
                write_json(folder/'fit.json',dict(game=game,held=held,train_sessions=fit,width=width,view=view,k=k,
                    train_windows=len(train),test_windows=len(test),rank=prep.rank,pca_retained=prep.retained,
                    heldout_density=float(gm.score(xv)),cluster_labels='fold-specific; never semantically aligned by outcome',
                    smoothing=False,selection='nested session CV, one standard error, smallest K including 1'))
                # Fixed first-computation sensitivity, same selection protocol and no post-first content.
                if view in ('evaluation','joint') and width==.5:
                    k1,s1=select_mixture(args,out,w,fit,view,width,True);b1=feature_blocks(out,fit,view,True)
                    pp=BlockProjection().fit(b1,train,args.seed)
                    gg=mixture_fit(pp.transform([b[train] for b in b1]),k1,args)
                    prob1=gg.predict_proba(pp.transform([b[test] for b in b1]));sensitivity=frame[['window','row','time']].copy()
                    sensitivity['state']=prob1.argmax(1)
                    for c in range(k1):sensitivity[f'p{c}']=prob1[:,c]
                    sensitivity.to_csv(folder/'first_only_assignments.csv.gz',index=False)
                    save_table(folder/'first_only_selection.csv',s1);pp.save(folder/'first_only_projection.npz')
                    np.savez_compressed(folder/'first_only_mixture.npz',weights=gg.weights_,means=gg.means_,covariances=gg.covariances_)
                complete_unit(folder);log(f'{game} held {held}, {width:g}s {view}: K={k}')
    for folder in sorted((out/'motifs').glob('held_*/w*/*')):
        rows.append(read_json(folder/'fit.json'))
    save_table(out/'motif_summary.csv',rows)


def block_interval(frame,values,reps,seed,seconds=10):
    """Paired mean within run, then equal-run average; resample blocks within run."""
    f=frame[['run','time']].copy();f['value']=values;f=f[np.isfinite(f.value)]
    if f.empty:return dict(mean=np.nan,low=np.nan,high=np.nan,runs=0,blocks=0)
    f['block']=np.floor(f.time/seconds).astype(int)
    groups=[]
    for _,g in f.groupby('run'):
        tab=g.groupby('block').value.agg(['sum','count']);groups.append(tab.to_numpy())
    rng=np.random.default_rng(seed);res=[]
    for _ in range(reps):
        means=[]
        for b in groups:
            v=b[rng.integers(0,len(b),len(b))];means.append(v[:,0].sum()/v[:,1].sum())
        res.append(np.mean(means))
    return dict(mean=float(f.groupby('run').value.mean().mean()),low=float(np.quantile(res,.025)),
                high=float(np.quantile(res,.975)),runs=len(groups),blocks=sum(map(len,groups)))


def readout(blocks,y,train,test,seed,classification=True):
    from sklearn.linear_model import LogisticRegression,Ridge
    from sklearn.preprocessing import StandardScaler
    # Preserve the baseline columns when adding a feature family: a joint PCA
    # could displace them and confound the incremental comparison.
    selected=np.r_[train,test];local_train=np.arange(len(train));local_test=np.arange(len(train),len(selected));pieces=[]
    for block in blocks:
        prep=BlockProjection().fit([block],train,seed);pieces.append(prep.transform([block[selected]]))
    x=np.column_stack(pieces);scaler=StandardScaler().fit(x[local_train]);x=scaler.transform(x)
    if classification:
        if len(np.unique(y[train]))<2:return np.full(len(test),(y[train].sum()+.5)/(len(train)+1))
        model=LogisticRegression(C=1.,max_iter=1000,solver='liblinear',random_state=seed).fit(x[local_train],y[train])
        return model.predict_proba(x[local_test])[:,1]
    model=Ridge(alpha=10.,solver='lsqr').fit(x[local_train],y[train]);return model.predict(x[local_test])


def loss(y,p,classification=True):
    if not classification:return (y-p)**2
    p=np.clip(p,1e-7,1-1e-7);return -(y*np.log(p)+(1-y)*np.log1p(-p))


def raw_pair_changes(frame,d,timelines):
    """A same-button endpoint pair may contain an unobserved switch-return."""
    good=np.r_[False,(np.diff(frame.time)<=.12)&(frame.segment.to_numpy()[1:]==frame.segment.to_numpy()[:-1])]
    changed=np.zeros(len(frame),bool)
    sources=d.iloc[frame.row.to_numpy(int)].source.to_numpy()
    for source in np.unique(sources):
        ii=np.flatnonzero(good&(sources==source))
        if len(ii):
            if source not in timelines:timelines[source]=ActionTimeline(source)
            tl=timelines[source]
            changed[ii]=tl.counts(frame.time.to_numpy()[ii-1],frame.time.to_numpy()[ii]).sum(axis=1)>0
    return good,changed


def validate_game(args,game,out):
    d=read_table(out/'decisions.csv.gz');w=read_table(out/'windows.csv.gz');p=np.load(out/'policies.npy')
    nact=p.shape[1];di=w.row.to_numpy(int);sessions=sorted(w.session.unique())
    root=np.load(out/'screen.npy');actual=np.load(out/'actual_policy.npy')
    history=np.column_stack([np.eye(nact)[d.human_action],np.log1p(d.age),d.raw_switch,d.carry,
                              np.eye(nact+1)[d.first_action+1]])
    # The joint candidate receives these exact action-window summaries. Put
    # them in the baseline too, so extra action history is not called an
    # imagination contribution.
    base=[root[di],history[di],actual[di],np.load(out/'action_windows.npy',mmap_mode='r')]
    # Independent visual/root control (no ACE target fitting).
    e=read_table(out/'computations.csv.gz');roots=d.start.to_numpy(int)
    rootlat=[]
    for net,field in [('sr','im_vectors'),('vp','im_vp_vectors')]:
        dest=out/(net+'_root_controls.npy')
        # Stage-level resume authenticates completed output. An unmarked
        # partially written memmap is never accepted merely because it exists.
        arr=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(len(d),512))
        for ids in chunks(np.arange(len(d)),args.chunk_size):arr[ids]=pool_latents(load_latents(e,roots[ids],field,True))
        arr.flush()
        rootlat.append(np.load(dest,mmap_mode='r')[di])
    base.extend(rootlat)
    predrows=[];comparison=[];continuity=[];profiles=[];examples=[];budget_predictions=[];timelines={};motif_preds={}
    for held in sessions:
        fit=tuple(s for s in sessions if s!=held)
        content=feature_blocks(out,fit,'evaluation');budget=np.load(out/'budget_windows.npy')
        # Allocation prediction uses only fixed early information, never the
        # realized length, whole-search summaries or final actor policy.
        root_base=[root,np.column_stack([np.eye(nact+1)[d.previous_action.to_numpy(int)+1],np.log1p(d.age_prior),d.carry])]
        root_base += [np.load(out/(net+'_root_controls.npy'),mmap_mode='r') for net in ('sr','vp')]
        train_d=d.index[d.session.isin(fit)].to_numpy();test_d=d.index[d.session.eq(held)].to_numpy()
        early=[d[['V_initial','first_dV','voc_initial','voc_first']].to_numpy()]
        initial=[d[['V_initial','voc_initial']].to_numpy()]
        fdir=out/'representation'/fit_key(fit)
        for label in ('sr_reward_terminal','vp_value','vp_policy','actor_joint'):
            x=np.load(fdir/(label+'_first.npy'));early.append(x);initial.append(x[:,:x.shape[1]//2])
        target=np.log1p(np.maximum(d.K.to_numpy()-1,0))
        bp=readout(root_base,target,train_d,test_d,args.seed,False)
        for name,features in [('initial',initial),('first_computation',early)]:
            pred=readout(root_base+features,target,train_d,test_d,args.seed,False)
            gain=loss(target[test_d],bp,False)-loss(target[test_d],pred,False)
            comparison.append(dict(held=held,width=0,horizon=0,target='additional_budget',model=name,parent='scene_history',
                block_seconds=10,n=len(test_d),**block_interval(d.loc[test_d],gain,args.bootstrap,args.seed)))
            for i,idx in enumerate(test_d):budget_predictions.append(dict(row=int(idx),held=held,model=name,
                target=target[idx],prediction=pred[i],baseline=bp[i]))
        for width in args.windows_seconds:
            ids=w.index[w.width.eq(width)].to_numpy()
            for horizon in args.horizons_seconds:
                for target,classification in [('switch',True),('policy',False)]:
                    y=w[f'{target}_{horizon:g}'].to_numpy();valid=ids[np.isfinite(y[ids])]
                    train=nonoverlap(w,valid[w.loc[valid,'session'].isin(fit)],width)
                    test=valid[w.loc[valid,'session'].eq(held)]
                    if len(train)<8 or not len(test):continue
                    blocks_by_name={'baseline':base,'budget':base+[budget],
                        'value':base+[budget,content[0]],'ace':base+[budget,*content],
                        'first':base+feature_blocks(out,fit,'evaluation',True)}
                    predictions={name:readout(b,y,train,test,args.seed,classification) for name,b in blocks_by_name.items()}
                    for name in ('budget','value','ace','first'):
                        parent={'budget':'baseline','value':'budget','ace':'value','first':'baseline'}[name]
                        gain=loss(y[test],predictions[parent],classification)-loss(y[test],predictions[name],classification)
                        for sec in (5,10,20):
                            ci=block_interval(w.loc[test],gain,args.bootstrap,args.seed,sec)
                            comparison.append(dict(held=held,width=width,horizon=horizon,target=target,model=name,parent=parent,block_seconds=sec,n=len(test),**ci))
                    for j,ix in enumerate(test):predrows.append(dict(window=int(ix),held=held,width=width,horizon=horizon,target=target,y=y[ix],
                        **{k:float(v[j]) for k,v in predictions.items()}))
            for view in ('evaluation','evaluation_budget','joint','joint_budget','action_only','outputs_only'):
                folder=out/'motifs'/f'held_{held}'/f'w{width:g}'/view
                f=read_table(folder/'assignments.csv.gz');k=read_json(folder/'fit.json')['k'];idx=f.row.to_numpy(int)
                attached=d.iloc[idx].reset_index(drop=True)
                good,humanchange=raw_pair_changes(f,d,timelines)
                change=np.r_[False,f.state.to_numpy()[1:]!=f.state.to_numpy()[:-1]]
                # Shift the complete state sequence within each continuous segment, preserving autocorrelation.
                rng=np.random.default_rng(args.seed);null_same=[];null_cross=[]
                for _ in range(args.null_repeats):
                    shifted=f.state.to_numpy().copy()
                    for _,g in f.groupby('segment'):
                        ii=g.index.to_numpy();shifted[ii]=np.roll(shifted[ii],int(rng.integers(max(1,len(ii)//5),max(2,4*len(ii)//5))))
                    nc=np.r_[False,np.diff(shifted)!=0]
                    if (good&~humanchange).any():null_same.append(nc[good&~humanchange].mean())
                    if (good&humanchange).any():null_cross.append((~nc[good&humanchange]).mean())
                continuity.append(dict(held=held,width=width,view=view,k=k,same_action_pairs=int((good&~humanchange).sum()),
                    cross_action_pairs=int((good&humanchange).sum()),
                    state_change_same_action=float(change[good&~humanchange].mean()) if (good&~humanchange).any() else np.nan,
                    state_persistence_cross_action=float((~change[good&humanchange]).mean()) if (good&humanchange).any() else np.nan,
                    shifted_change_same_action=np.mean(null_same) if null_same else np.nan,
                    shifted_persistence_cross_action=np.mean(null_cross) if null_cross else np.nan))
                if (folder/'first_only_assignments.csv.gz').exists():
                    first_only=read_table(folder/'first_only_assignments.csv.gz')
                    np.testing.assert_array_equal(first_only.window,f.window)
                    fc=np.r_[False,np.diff(first_only.state)!=0]
                    continuity.append(dict(held=held,width=width,view=view+'_first_only',
                        k=len([c for c in first_only if c.startswith('p') and c[1:].isdigit()]),
                        same_action_pairs=int((good&~humanchange).sum()),cross_action_pairs=int((good&humanchange).sum()),
                        state_change_same_action=float(fc[good&~humanchange].mean()) if (good&~humanchange).any() else np.nan,
                        state_persistence_cross_action=float((~fc[good&humanchange]).mean()) if (good&humanchange).any() else np.nan))
                for state in range(k):
                    take=f.state.eq(state).to_numpy()
                    for metric in ['K',*DIRECT,'policy_change']:
                        profiles.append(dict(held=held,width=width,view=view,state=state,metric=metric,n=int(take.sum()),
                            mean=float(attached.loc[take,metric].mean()),sd=float(attached.loc[take,metric].std())))
                    for action in range(nact):profiles.append(dict(held=held,width=width,view=view,state=state,metric=f'action_{action}',
                        n=int(take.sum()),mean=float((attached.loc[take,'human_action']==action).mean()),sd=np.nan))
                    # Representative by distance to fitted center, plus seeded random held-out example.
                    candidates=np.flatnonzero(take)
                    if len(candidates):
                        coords=np.load(folder/'test_embedding.npy');gm=np.load(folder/'mixture.npz')
                        dist=np.sum((coords[candidates]-gm['means'][state])**2/gm['covariances'][state],axis=1)
                        for kind,j in [('central',candidates[np.argmin(dist)]),('random',rng.choice(candidates))]:
                            examples.append(dict(held=held,width=width,view=view,state=state,kind=kind,row=int(f.row.iloc[j]),window=int(f.window.iloc[j])))
                # Conditional per-K continuity, so length-driven state separation remains visible.
                if not view.endswith('_budget'):
                    for kval in sorted(attached.K.unique()):
                        mask=good&attached.K.eq(kval).to_numpy()&np.r_[False,attached.K.to_numpy()[:-1]==kval]
                        continuity.append(dict(held=held,width=width,view=view,k=k,budget_stratum=int(kval),
                            same_action_pairs=int((mask&~humanchange).sum()),cross_action_pairs=int((mask&humanchange).sum()),
                            state_change_same_action=float(change[mask&~humanchange].mean()) if (mask&~humanchange).any() else np.nan,
                            state_persistence_cross_action=float((~change[mask&humanchange]).mean()) if (mask&humanchange).any() else np.nan))
                # Test candidate membership beyond baseline, using training responsibilities from frozen outer fit.
                if width==.5:
                    blocks=feature_blocks(out,fit,view);train=nonoverlap(w,w.index[w.session.isin(fit)&w.width.eq(width)].to_numpy(),width)
                    prep=BlockProjection().fit(blocks,train,args.seed)
                    gm=mixture_fit(prep.transform([b[train] for b in blocks]),k,args)
                    selected=np.unique(np.r_[train,f.window.to_numpy(int)]);resp=np.zeros((len(w),k))
                    resp[selected]=gm.predict_proba(prep.transform([b[selected] for b in blocks]))
                    y=w['switch_0.25'].to_numpy();tr=train[np.isfinite(y[train])];te=f.window.to_numpy(int);te=te[np.isfinite(y[te])]
                    if len(tr)>8 and len(te):
                        bp=readout(base,y,tr,te,args.seed);mp=readout(base+[resp],y,tr,te,args.seed)
                        motif_preds[(held,view)]=(te,mp)
                        gain=loss(y[te],bp)-loss(y[te],mp)
                        comparison.append(dict(held=held,width=width,horizon=.25,target='switch',model='motif_'+view,parent='baseline',
                            block_seconds=10,n=len(te),**block_interval(w.loc[te],gain,args.bootstrap,args.seed)))
    y=w['switch_0.25'].to_numpy()
    for held in sessions:
        for view in ('evaluation','joint','evaluation_budget','joint_budget'):
            for reference in ('action_only','outputs_only'):
                if (held,view) not in motif_preds or (held,reference) not in motif_preds:continue
                te,mp=motif_preds[held,view];tr,bp=motif_preds[held,reference]
                np.testing.assert_array_equal(te,tr)
                gain=loss(y[te],bp)-loss(y[te],mp)
                for sec in (5,10,20):
                    comparison.append(dict(held=held,width=.5,horizon=.25,target='switch',model='motif_'+view+'_over_'+reference,
                        parent='motif_'+reference,block_seconds=sec,n=len(te),**block_interval(w.loc[te],gain,args.bootstrap,args.seed,sec)))
    save_table(out/'predictions.csv.gz',predrows);save_table(out/'budget_predictions.csv.gz',budget_predictions);save_table(out/'comparisons.csv',comparison)
    save_table(out/'continuity.csv',continuity);save_table(out/'profiles.csv',profiles);save_table(out/'examples.csv',examples)
    # Actual voluntary continuation is a descriptive calibration check, not a causal utility test.
    ev=e.loc[e.voluntary.astype(bool)].copy();ev.to_csv(out/'voluntary_choices.csv.gz',index=False)
    write_json(out/'validation.json',dict(outer_folds=4,heldout='analysis fits only; model trained on all sessions',
        predictions=len(predrows),conditional_associations_only=True,uncertainty='within-run time block bootstrap; equal-run mean',
        continuity_null='circularly shifted state sequence per segment, preserves overlap-induced temporal dependence'))

def render_game(args,game,out):
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
    d=read_table(out/'decisions.csv.gz');e=read_table(out/'computations.csv.gz');w=read_table(out/'windows.csv.gz')
    screens=np.load(out/'screens.npy',mmap_mode='r');policy=np.load(out/'policies.npy');nact=policy.shape[1]
    figures=[];figdir=out/'figures';figdir.mkdir(exist_ok=True)
    def save(fig,name):
        fig.suptitle(f'sub{args.subject:03d} · {game}',fontsize=12)
        fig.tight_layout();fig.savefig(figdir/(name+'.png'),dpi=160);fig.savefig(figdir/(name+'.pdf'));plt.close(fig);figures.append(name)
    palette=plt.get_cmap('tab10')
    examples=read_table(out/'examples.csv');refs=examples[(examples.held==4)&examples.width.eq(.5)&examples.view.isin(['evaluation','joint'])]
    # Examples determined on the held-out session by centrality/random rules, never outcome amplitude.
    for ex in refs.itertuples():
        r=d.loc[ex.row];ids=d.index[d.segment.eq(r.segment)&d.time.between(r.time-1,r.time+1)].to_numpy()
        if not len(ids):continue
        time=d.loc[ids,'time'].to_numpy()-r.time;f=out/'motifs'/'held_4'/'w0.5'/ex.view
        ass=read_table(f/'assignments.csv.gz');ass=ass[ass.row.isin(ids)]
        fig,ax=plt.subplots(7,1,figsize=(10,10),gridspec_kw={'height_ratios':[1.5,1,1,1,1,1,1]})
        chosen=ids[np.linspace(0,len(ids)-1,min(6,len(ids))).astype(int)]
        montage=np.concatenate([screens[i].transpose(1,2,0) for i in chosen],axis=1);ax[0].imshow(montage);ax[0].axis('off')
        ax[0].set_title(f'{game} held-out session 4: {ex.view}, C{ex.state+1}, {ex.kind}; t={r.time:.2f}s')
        ax[1].step(time,d.loc[ids,'human_action'],where='post');ax[1].set_ylabel('Human action')
        ax[2].step(time,d.loc[ids,'K'],where='post');ax[2].set_ylabel('Computations')
        ax[3].plot(time,d.loc[ids,'D'],label='signed net');ax[3].plot(time,d.loc[ids,'U'],label='absolute / K');ax[3].legend();ax[3].set_ylabel('Value update')
        ax[4].plot(time,d.loc[ids,'voc_initial'],label='initial');ax[4].plot(time,d.loc[ids,'voc_final'],label='final');ax[4].legend();ax[4].set_ylabel('EMA VoC')
        pp=np.load(out/'actual_policy.npy',mmap_mode='r')[ids]
        ax[5].imshow(pp.T,aspect='auto',origin='lower',extent=[time[0],time[-1],-.5,nact-.5],vmin=0,vmax=1);ax[5].set_ylabel('Real policy')
        for c in [x for x in ass if x.startswith('p') and x[1:].isdigit()]:ax[6].plot(ass.time-r.time,ass[c],label='C'+str(int(c[1:])+1))
        ax[6].legend(ncol=4);ax[6].set_ylabel('Candidate P');ax[6].set_xlabel('Seconds from selected window endpoint')
        for a in ax[1:]:a.axvline(0,color='gray',ls=':',lw=.7)
        save(fig,f'01_timeline_{ex.view}_c{ex.state+1}_{ex.kind}')
    fig,ax=plt.subplots(2,3,figsize=(12,7));counts=[]
    for k,g in e.groupby('k'):
        if k>30:continue
        counts.append(dict(k=k,n=len(g)))
    for col,metric in enumerate(['value','voc_ema','policy_change']):
        for session,g in e.groupby('session'):
            mean=g.groupby('k')[metric].mean();ax[0,col].plot(mean.index,mean,label=f'Session {session}')
        ax[0,col].set(xlabel='Completed computations k',ylabel=metric)
    ax[0,0].legend();choice=e[e.voluntary.astype(bool)]
    for col,metric in enumerate(['voc_ema','delta_V','gate_probability']):
        if len(choice):
            b=pd.qcut(choice[metric],min(10,choice[metric].nunique()),duplicates='drop')
            grouped=choice.groupby(b,observed=True).agg(x=(metric,'mean'),y=('continue_choice','mean'),n=('row','size'))
            ax[1,col].plot(grouped.x,grouped.y,'o-');ax[1,col].set(xlabel=metric,ylabel='Observed voluntary continue')
    save(fig,'02_computation_and_stopping');save_table(out/'computation_support.csv',counts)
    # All-switch alignment. Fold-specific axes are displayed separately, not pooled by sign.
    switches=read_table(out/'switches.csv.gz');grid=np.arange(-.5,.5001,.05);aligned=[];coverage=[]
    metrics=['K','U','D','policy_change','vp_value_ACE1','actor_voc_ACE1','actor_policy_ACE1']
    fig,axes=plt.subplots(4,len(metrics),figsize=(19,10),sharex=True)
    for si,held in enumerate(sorted(d.session.unique())):
        fit=tuple(s for s in sorted(d.session.unique()) if s!=held);folder=out/'representation'/fit_key(fit)
        vals=d[['K','U','D','policy_change']].to_numpy().astype(float)
        zcols=[]
        for label in ['vp_value','actor_voc','actor_policy']:
            z=np.load(folder/(label+'_events.npy'),mmap_mode='r');first=np.minimum(d.start.to_numpy(int)+1,d.end.to_numpy(int)-1)
            zcols.append(z[first,0] if z.shape[1] else np.full(len(d),np.nan))
        vals=np.column_stack([vals,*zcols]);samples=[];shifts=[];rng=np.random.default_rng(args.seed)
        for segment,g in d[d.session.eq(held)].groupby('segment'):
            ev=switches[switches.segment.eq(segment)];times=g.time.to_numpy();ids=g.index.to_numpy()
            if ev.empty or times[-1]<=times[0]:continue
            ix=sa_indices(times,ev.time.to_numpy(),grid);samples.append(sa_sample(vals[ids],ix))
            shifted=times[0]+np.mod(ev.time.to_numpy()-times[0]+rng.uniform(.2,.8)*(times[-1]-times[0]),times[-1]-times[0])
            shifts.append(sa_sample(vals[ids],sa_indices(times,shifted,grid)))
            coverage.append(dict(held=held,segment=segment,raw_switches=len(ev),complete=int((ix>=0).all(1).sum()),
                lag_counts=(ix>=0).sum(0).tolist()))
        if not samples:continue
        observed=np.concatenate(samples);null=np.concatenate(shifts)
        np.savez_compressed(out/f'switch_aligned_held{held}.npz',grid=grid,observed=observed,shifted=null,metrics=metrics)
        for j,metric in enumerate(metrics):
            mean=sa_mean(observed[:,:,j],axis=0);ref=sa_mean(null[:,:,j],axis=0)
            axes[si,j].plot(grid,mean,label='human switch');axes[si,j].plot(grid,ref,ls='--',label='time-shifted')
            axes[si,j].axvline(0,color='gray',ls=':');axes[si,j].set_title(metric if si==0 else '')
            if j==0:axes[si,j].set_ylabel(f'Held session {held}')
            if si==3:axes[si,j].set_xlabel('Seconds')
            for lag,y,refy,n in zip(grid,mean,ref,np.isfinite(observed[:,:,j]).sum(0)):
                aligned.append(dict(held=held,metric=metric,lag=lag,mean=y,shifted=refy,n=n))
    axes[0,0].legend(fontsize=7);save(fig,'03_all_switches');save_table(out/'switch_curves.csv',aligned);write_json(out/'switch_coverage.json',coverage)
    # Transition directions retained, so cancellation in pooled signed axes is visible.
    fig,axes=plt.subplots(1,2,figsize=(12,5));held=4;fit=(1,2,3);folder=out/'representation'/fit_key(fit)
    for ax,label in zip(axes,['vp_value','actor_policy']):
        z=np.load(folder/(label+'_events.npy'),mmap_mode='r');z=z[:,0] if z.shape[1] else np.full(len(e),np.nan)
        z=z[np.minimum(d.start.to_numpy(int)+1,d.end.to_numpy(int)-1)];traces={}
        for seg,g in d[d.session.eq(4)].groupby('segment'):
            ev=switches[switches.segment.eq(seg)]
            for transition,ee in ev.groupby(['action_from','action_to']):
                xx=sa_sample(z[g.index],sa_indices(g.time.to_numpy(),ee.time.to_numpy(),grid));traces.setdefault(transition,[]).append(xx)
        for key,arrays in sorted(traces.items()):ax.plot(grid,sa_mean(np.concatenate(arrays),axis=0),label=f'{key[0]}>{key[1]}')
        ax.axvline(0,color='gray',ls=':');ax.set(title=label+' ACE1, held session 4',xlabel='Seconds');ax.legend(fontsize=6,ncol=3)
    save(fig,'04_transition_directions')
    # Candidate profiles: scale for display only using training sessions, never per-candidate normalization.
    profiles=read_table(out/'profiles.csv');fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,view in zip(axes,['evaluation','joint']):
        f=profiles[profiles.held.eq(4)&profiles.width.eq(.5)&profiles.view.eq(view)]
        names=['K','V_initial','D','U','voc_initial','policy_change'];tab=f[f.metric.isin(names)].pivot(index='state',columns='metric',values='mean').reindex(columns=names)
        tr=d[d.session.ne(4)];den=tr[names].std().replace(0,1);val=(tab-tr[names].mean())/den
        im=ax.imshow(val,aspect='auto',cmap='coolwarm',vmin=-2,vmax=2);ax.set_xticks(range(len(names)),names,rotation=45,ha='right')
        ax.set_yticks(range(len(tab)),['C'+str(i+1) for i in tab.index]);ax.set_title(view+' (held session 4)');fig.colorbar(im,ax=ax,label='Training SD')
    save(fig,'05_candidate_profiles')
    comp=read_table(out/'comparisons.csv');f=comp[(comp.width==.5)&(comp.horizon==.25)&(comp.block_seconds==10)&comp.target.eq('switch')]
    fig,ax=plt.subplots(figsize=(12,5));names=list(f.model.unique())
    for held,g in f.groupby('held'):
        x=np.array([names.index(s) for s in g.model])+(held-2.5)*.12
        ax.errorbar(x,g['mean'],yerr=[np.maximum(0,g['mean']-g.low),np.maximum(0,g.high-g['mean'])],fmt='o',label=f'Session {held}',capsize=2)
    ax.axhline(0,color='gray');ax.set_xticks(range(len(names)),names,rotation=35,ha='right');ax.set_ylabel('Held-out log-loss gain over listed parent');ax.legend();save(fig,'06_heldout_comparisons')
    # Explicitly show both desired motifs, chosen without maximizing physiological/value effects.
    fig,axes=plt.subplots(2,3,figsize=(13,6));ass=read_table(out/'motifs/held_4/w0.5/evaluation/assignments.csv.gz')
    good,actionchange=raw_pair_changes(ass,d,{})
    statechange=np.r_[False,np.diff(ass.state)!=0]
    different_action=np.r_[False,np.diff(ass.human_action)!=0]
    for row,(name,mask) in enumerate([('Same action, different candidate',good&statechange&~actionchange),('Different action, same candidate',good&~statechange&different_action)]):
        candidates=np.flatnonzero(mask)
        if not len(candidates):
            for ax in axes[row]:ax.text(.5,.5,'No supported example',ha='center');ax.set_axis_off()
            continue
        j=candidates[len(candidates)//2];center=ass.iloc[j];g=ass[ass.segment.eq(center.segment)&ass.time.between(center.time-.5,center.time+.5)]
        ids=g.row.to_numpy(int);tt=g.time-center.time
        axes[row,0].step(tt,g.human_action,where='post');axes[row,0].set_title(name);axes[row,0].set_ylabel('Human action')
        axes[row,1].step(tt,g.state+1,where='post');axes[row,1].set_ylabel('Candidate')
        axes[row,2].plot(tt,d.loc[ids,'U']);axes[row,2].set_ylabel('Value update / K')
    save(fig,'07_within_across_action_examples')
    caution=('SMOKE TEST: software verification only. ' if args.smoke else '')+'All associations are observational; candidate labels are not goals. Fold-specific axes/states are not aligned using held-out outcomes. Model training used all four sessions.'
    body='<h1>'+html.escape(f'sub{args.subject:03d} / {game}')+' — computation and control-pattern candidates</h1><p>'+html.escape(caution)+'</p>'
    body+='<p>Read computation_support.csv and switch_coverage.json for support. Late k conditions select surviving searches. Q comparisons exclude unvisited branches.</p>'
    for name in figures:body+=f'<h2>{html.escape(name)}</h2><a href="figures/{name}.pdf"><img style="max-width:100%" src="figures/{name}.png"></a>'
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8">'+body)


def complete_unit(folder):
    files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='unit.done.json'}
    write_json(folder/'unit.done.json',dict(version=VERSION,artifacts=files))


def resume_unit(folder,args):
    marker=folder/'unit.done.json'
    if not marker.exists():return False
    if not args.resume:raise RuntimeError('Completed unit exists; use --resume')
    old=read_json(marker)
    if old['version']!=VERSION or not all((folder/k).is_file() and sha(folder/k)==v for k,v in old['artifacts'].items()):
        raise RuntimeError('Completed unit changed: '+str(folder))
    return True


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=DEFAULT_INPUT)
    p.add_argument('--output-root',type=Path,default=RESEARCH/'outputs/14')
    p.add_argument('--subjects',type=int,nargs='+',default=[1,2])
    p.add_argument('--games',nargs='+',choices=list(GAME_IDS),default=['pong','enduro','spaceinvaders'])
    p.add_argument('--workers',type=int,default=1,help='Independent subject/game processes; fits never pool participants')
    p.add_argument('--stage',choices=['all',*STAGES],default='all')
    p.add_argument('--resume',action='store_true');p.add_argument('--self-test',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-decisions',type=int,default=96)
    p.add_argument('--windows-seconds',type=float,nargs='+',default=[.25,.5,1.])
    p.add_argument('--horizons-seconds',type=float,nargs='+',default=[.25,.5,1.])
    p.add_argument('--basis-samples',type=int,default=2048,help='Equal-run decisions, one uniformly chosen computation endpoint each')
    p.add_argument('--jacobian-samples',type=int,default=512)
    p.add_argument('--energy',type=float,default=.99);p.add_argument('--max-rank',type=int,default=128)
    p.add_argument('--chunk-size',type=int,default=256);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=701);p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--null-repeats',type=int,default=100)
    return p


def verify_stage(folder,stage,signature):
    marker=read_json(folder/(stage+'.done.json'))
    if marker['signature']!=signature:raise ValueError('Stage signature mismatch')
    for k,v in marker['artifacts'].items():
        if not (folder/k).is_file() or sha(folder/k)!=v:raise ValueError('Changed completed result: '+k)
    if stage=='audit':
        for path,stamp in read_json(folder/'audit.json')['input_stamps'].items():
            if not Path(path).exists() or file_stamp(path)!=stamp:raise ValueError('Input changed since audit: '+path)


def process_pair(args, subject, game, root, signature):
    """Isolate each participant/game fit and its log in one worker process."""
    import contextlib
    import fcntl
    import traceback
    import torch
    from threadpoolctl import threadpool_limits
    local=argparse.Namespace(**vars(args));local.subject=subject
    folder=root/f'sub{subject:03d}'/game;folder.mkdir(parents=True,exist_ok=True)
    label=f'sub{subject:03d}_{game}';logs=root/'logs';logs.mkdir(exist_ok=True)
    state=dict(subject=subject,game=game,pid=os.getpid(),signature=signature,output=str(folder))
    stage='startup'
    with (logs/(label+'.log')).open('a',buffering=1) as stream, \
         contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream), \
         (folder/'.lock').open('a+') as lock,threadpool_limits(1,user_api='blas'):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(args.threads)
        try:
            frozen_import(args.input_root,'behavior_trace')
            actions=dict(audit=audit_game,extract=extract_game,representation=representation_game,temporal=temporal_game,
                         motifs=motifs_game,validate=validate_game,render=render_game)
            for stage in STAGES if args.stage=='all' else [args.stage]:
                write_json(logs/(label+'.status.json'),dict(state,status='RUNNING',stage=stage,updated=time.time()))
                for previous in STAGES[:STAGES.index(stage)]:verify_stage(folder,previous,signature)
                marker=folder/(stage+'.done.json')
                if marker.exists():
                    if not args.resume:raise ValueError('Existing stage; use --resume')
                    verify_stage(folder,stage,signature);log(f'{label}: verified {stage}');continue
                start=time.monotonic();log(f'{label}: {stage} -> {folder}')
                actions[stage](local,game,folder)
                owned=set()
                for previous in STAGES[:STAGES.index(stage)]:owned.update(read_json(folder/(previous+'.done.json'))['artifacts'])
                # Include partial files rebuilt by this stage, excluding worker locks.
                files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*')
                       if p.is_file() and str(p.relative_to(folder)) not in owned
                       and not p.name.endswith('.done.json') and p.name!='.lock'}
                write_json(marker,dict(signature=signature,artifacts=files,elapsed_seconds=time.monotonic()-start,status='COMPLETE'))
                log(f'{label}: {stage} completed in {time.monotonic()-start:.1f}s')
            state.update(status='COMPLETE' if (folder/'render.done.json').exists() else 'STAGE_COMPLETE',stage=stage)
        except Exception as exc:
            traceback.print_exc();state.update(status='FAILED',stage=stage,error=repr(exc))
        write_json(logs/(label+'.status.json'),dict(state,updated=time.time()))
    return state


def main(argv=None):
    args=parser().parse_args(argv)
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(args.threads)
    if args.self_test:
        with threadpool_limits(1,user_api='blas'):return 0 if self_test() else 1
    for name in ('threads','workers','chunk_size','basis_samples','jacobian_samples','max_rank','smoke_decisions','bootstrap','null_repeats'):
        if getattr(args,name)<1:raise ValueError(name+' must be positive')
    args.subjects=sorted(set(args.subjects));args.games=list(dict.fromkeys(args.games))
    if not args.subjects or any(s<1 for s in args.subjects):raise ValueError('Subjects must be positive integers')
    if not 0<args.energy<=1:raise ValueError('energy must lie in (0,1]')
    for name in ('windows_seconds','horizons_seconds'):
        if not all(np.isfinite(x) and x>0 for x in getattr(args,name)):raise ValueError(name+' invalid')
        setattr(args,name,sorted(set(getattr(args,name))))
    if .5 not in args.windows_seconds or .25 not in args.horizons_seconds:raise ValueError('Primary .5s window/.25s horizon required')
    if args.smoke:
        args.basis_samples=min(args.basis_samples,64);args.jacobian_samples=min(args.jacobian_samples,16)
        args.max_rank=min(args.max_rank,8);args.bootstrap=min(args.bootstrap,20);args.null_repeats=min(args.null_repeats,3)
    args.input_root=args.input_root.expanduser().resolve();args.output_root=args.output_root.expanduser().resolve()
    if args.output_root.is_relative_to(args.input_root):raise ValueError('Output may not modify captures')
    settings={k:clean_json(v) for k,v in vars(args).items() if k not in ('stage','resume','self_test','output_root','workers')}
    pairs=list(itertools.product(args.subjects,args.games));inputs={}
    for subject,game in pairs:
        capture,cp=game_inputs(args.input_root,game,subject)
        inputs[f'sub{subject:03d}/{game}']={str(p):sha(p) for p in [capture/'manifest.json',cp/'config_c.yaml',cp/'bundle.json']}
    runtime=args.input_root/'runtime_source/thinker';sources={str(p):sha(p) for p in sorted(runtime.rglob('*.py'))}
    frozen_import(args.input_root,'behavior_trace')
    identity=dict(version=VERSION,script_sha256=sha(__file__),settings=settings,inputs=inputs,runtime=sources)
    signature=digest(identity);root=args.output_root/(('control_patterns_smoke_' if args.smoke else 'control_patterns_')+signature[:12])
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    import fcntl
    with open(root/'.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=root/'analysis_source.py'
        if source.exists() and sha(source)!=identity['script_sha256']:raise ValueError('Frozen source changed')
        if not source.exists():source.write_bytes(Path(__file__).read_bytes())
        states={f'sub{s:03d}/{g}':dict(subject=s,game=g,status='QUEUED') for s,g in pairs}
        def progress(status):
            write_json(root/'run.json',dict(identity,signature=signature,status=status,pid=os.getpid(),
                       workers=args.workers,output=root,pairs=states,updated=time.time()))
        progress('RUNNING');log(f'Output: {root}; {len(pairs)} subject/game pairs, {args.workers} workers')
        # A complete report index is available immediately; only finished reports are linked.
        def index():
            links=[]
            for s,g in pairs:
                rel=f'sub{s:03d}/{g}';state=states[rel]['status']
                if (root/rel/'report.html').is_file():links.append(f'<li><a href="{rel}/report.html">{rel}</a> — {state}</li>')
                else:links.append(f'<li>{rel} — {state}</li>')
            (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Control-pattern analysis</h1>'+('<p>SMOKE TEST ONLY</p>' if args.smoke else '')+'<ul>'+''.join(links)+'</ul>')
        index()
        try:
            if args.workers==1:
                for subject,game in pairs:
                    result=process_pair(args,subject,game,root,signature)
                    states[f'sub{subject:03d}/{game}']=result;progress('RUNNING');index()
                    log(f'sub{subject:03d}/{game}: {result["status"]}')
            else:
                import multiprocessing
                from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
                with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                    futures={pool.submit(process_pair,args,s,g,root,signature):(s,g) for s,g in pairs}
                    pending=set(futures)
                    while pending:
                        done,pending=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                        for future in done:
                            subject,game=futures[future]
                            try:result=future.result()
                            except Exception as exc:result=dict(subject=subject,game=game,status='FAILED',error=repr(exc))
                            states[f'sub{subject:03d}/{game}']=result
                            log(f'sub{subject:03d}/{game}: {result["status"]}')
                        for future in pending:
                            subject,game=futures[future];path=root/'logs'/f'sub{subject:03d}_{game}.status.json'
                            if path.exists():states[f'sub{subject:03d}/{game}']=read_json(path)
                        progress('RUNNING');index()
            failed=[v for v in states.values() if v['status']=='FAILED']
            complete=all((root/f'sub{s:03d}'/g/'render.done.json').exists() for s,g in pairs)
            status='FAILED' if failed else ('COMPLETE' if complete else 'STAGE_COMPLETE')
            progress(status);index()
            if complete and not failed:
                write_json(root/'COMPLETED.json',dict(signature=signature,subjects=args.subjects,games=args.games,smoke=args.smoke))
            if failed:write_json(root/'FAILED.json',dict(signature=signature,pairs=failed))
            elif (root/'FAILED.json').exists():(root/'FAILED.json').unlink()
            log(f'{status}: {root}')
            return 1 if failed else 0
        except BaseException as exc:
            progress('INTERRUPTED');raise


def self_test():
    import unittest
    import tempfile
    from types import SimpleNamespace
    class Contracts(unittest.TestCase):
        def test_checkpoint_resolution_separates_participants(self):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'campaign'
                for subject in (1,2):
                    capture=root/'captures'/f'pong-sub{subject:03d}-all'
                    capture.mkdir(parents=True);(capture/'COMPLETED.json').write_text('{}')
                    cp=root/'checkpoints'/f'pong-s{subject:03d}-test'/'milestones/10000000';cp.mkdir(parents=True)
                capture,cp=game_inputs(root,'pong',2)
                self.assertEqual(capture.name,'pong-sub002-all')
                self.assertIn('pong-s002-test',str(cp));self.assertNotIn('s001',str(cp))
        def test_checkpoint_reuse_from_sibling_campaign(self):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'current';capture=root/'captures'/'spaceinvaders-sub002-all'
                capture.mkdir(parents=True);(capture/'COMPLETED.json').write_text('{}')
                cp=Path(tmp)/'previous/checkpoints/spaceinvaders-s002-test/milestones/10000000';cp.mkdir(parents=True)
                self.assertEqual(game_inputs(root,'spaceinvaders',2)[1],cp)
                with self.assertRaises(ValueError):game_inputs(root,'spaceinvaders',1)
        def test_zero_and_one_computation_values(self):
            self.assertEqual(value_summary([2])['U'],0)
            self.assertTrue(np.isnan(value_summary([2])['coherence']))
            self.assertEqual(value_summary([2,1])['A'],1)
            self.assertEqual(value_summary([2,1])['D'],-1)
            self.assertEqual(value_summary([2,1])['U'],1)
        def test_net_vs_absolute_updates(self):
            r=value_summary([1,3,2]);self.assertEqual(r['A'],3);self.assertEqual(r['D'],1)
            self.assertAlmostEqual(r['coherence'],1/3)
        def test_decision_uses_pre_real_state(self):
            edges=list(decision_edges(np.array([1,0,0,0,0],bool),np.array([0,0,1,0,1],bool),
                np.array([0,1,0,1,0],bool),np.array([0,1,0,1,0]),np.zeros(5,bool)))
            self.assertEqual([(x[1],x[2]) for x in edges],[(0,2),(2,4)])
            self.assertEqual(edges[1][3].tolist(),[3])
        def test_reset_breaks_future_path(self):
            edges=list(decision_edges(np.array([1,0,0,0],bool),np.array([0,0,0,1],bool),
                np.array([0,1,1,0],bool),np.array([0,1,0,0]),np.array([0,0,1,0],bool)))
            self.assertEqual(edges[0][4],[1]);self.assertEqual(len(edges[0][3]),2)
        def test_actor_phase_reconstruction(self):
            h=object.__new__(ActorHeads);h.phase=np.array([[0,0],[3,4],[1,2]])
            x=np.array([[1,2],[4,6]])
            np.testing.assert_allclose(h.common(x,[0,1]),[[1,2],[1,2]])
        def test_exact_voc_difference_invariant_to_dueling_center(self):
            raw=np.array([[2.,1.],[-3,2]])
            for prob in (0.,.5,1.):
                q=5+raw-(raw*np.array([prob,1-prob])).sum(1,keepdims=True)+raw.mean(1,keepdims=True)
                np.testing.assert_allclose(q[:,0]-q[:,1],raw[:,0]-raw[:,1])
        def test_no_future_event_interpolation(self):
            ix=sa_indices(np.array([0,.1,.2]),np.array([.15]),np.array([0,.1]),max_age=.12)
            np.testing.assert_array_equal(ix,[[1,-1]])
        def test_action_timeline_catches_switch_return(self):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'a.npz';np.savez(p,time=[0,.05,.1,.15,.2],action=np.eye(2)[[0,1,0,0,0]],
                    is_first=[1,0,0,0,0],is_terminal=[0,0,0,0,0]);tl=ActionTimeline(p)
                self.assertEqual(tl.counts(0,.15).sum(),2)
                np.testing.assert_allclose(tl.area(.15),[.1,.05])
        def test_same_action_endpoints_do_not_hide_raw_switch_return(self):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'a.npz';np.savez(p,time=[0,.03,.06,.1],action=np.eye(2)[[0,1,0,0]],
                    is_first=[1,0,0,0],is_terminal=[0,0,0,0])
                d=pd.DataFrame(dict(source=[str(p),str(p)]))
                f=pd.DataFrame(dict(row=[0,1],time=[0,.1],segment=[1,1]))
                valid,changed=raw_pair_changes(f,d,{})
                np.testing.assert_array_equal(valid,[False,True]);np.testing.assert_array_equal(changed,[False,True])
        def test_gap_segmentation(self):
            d=pd.DataFrame(dict(episode=['a']*4,decision=[0,1,2,4],time=[0,.1,.5,.6]))
            np.testing.assert_array_equal(timeline_segments(d).segment,[1,1,2,3])
        def test_window_never_uses_future(self):
            d=pd.DataFrame(dict(time=np.arange(10)*.1));w=pd.DataFrame(dict(left=[0],row=[4]))
            x=np.arange(10.)[:,None];a=window_features(x,d,w);x[5:]=1e9
            np.testing.assert_array_equal(a,window_features(x,d,w))
        def test_fit_does_not_use_heldout_scaling(self):
            rng=np.random.default_rng(7);x=rng.normal(size=(40,6));tr=np.arange(25)
            p=BlockProjection().fit([x],tr);a=p.transform([x])[:25];x[25:]=1e10
            q=BlockProjection().fit([x],tr);np.testing.assert_allclose(a,q.transform([x])[:25])
        def test_missing_blocks_and_constant_features(self):
            x=np.full((30,3),np.nan);p=BlockProjection().fit([x],np.arange(20))
            self.assertEqual(p.rank,0);self.assertTrue(np.isfinite(p.transform([x])).all())
        def test_equal_decision_sampling(self):
            d=pd.DataFrame(dict(run=['a']*5+['b']*100))
            ids=sample_runs(d,d.index.to_numpy(),10,701)
            self.assertEqual((d.loc[ids,'run']=='a').sum(),5)
        def test_zero_sensitivity_not_fabricated(self):
            q,s,r,e=ace_basis(np.eye(3),np.zeros((2,3)));self.assertEqual(q.shape,(3,0));self.assertEqual(r,0)
        def test_scalar_linear_target_rank_one(self):
            q,s,r,e=ace_basis(np.eye(4),np.array([[1,2,3,4.]]));self.assertEqual(r,1)
            np.testing.assert_allclose(np.linalg.norm(q.T@q-np.eye(1)),0,atol=1e-12)
        def test_covariance_factor_exact_without_wide_svd(self):
            x=np.random.default_rng(2).normal(size=(32,100));weights=np.ones(32)/32
            mean,factor=activation_factor(x,np.arange(32),weights)
            np.testing.assert_allclose(factor@factor.T,(x-mean).T@(x-mean)/32,atol=1e-12)
        def test_heldout_labels_do_not_fit_readout(self):
            rng=np.random.default_rng(4);x=rng.normal(size=(60,8));y=(x[:,0]>0).astype(float)
            tr=np.arange(40);te=np.arange(40,60);a=readout([x],y,tr,te,701)
            y[te]=1-y[te];np.testing.assert_allclose(a,readout([x],y,tr,te,701))
        def test_one_se_prefers_single_component_when_supported(self):
            t=pd.DataFrame([dict(k=k,score=s,status='ok') for k,ss in [(1,[1.,1.,1.]),(2,[.5,1.1,1.7])] for s in ss])
            self.assertEqual(one_se_choice(t,3),1)
        def test_one_se_rejects_missing_validation_fold(self):
            t=pd.DataFrame([dict(k=1,score=1.,status='ok'),dict(k=2,score=100.,status='ok')])
            with self.assertRaises(ValueError):one_se_choice(t,3)
        def test_coordinate_summary_handles_zero_and_one(self):
            d=pd.DataFrame(dict(start=[0,1],end=[1,3]));z=np.array([[4.],[2.],[3.]])
            s=summarize_coordinates(z,d);np.testing.assert_allclose(s[0],[4,0,0,0]);np.testing.assert_allclose(s[1],[2,1,1,.5])
        def test_nonoverlap_is_segment_specific(self):
            w=pd.DataFrame(dict(segment=[1]*5+[2]*5,time=list(np.arange(5)*.1)*2))
            self.assertEqual(nonoverlap(w,w.index.to_numpy(),.25).tolist(),[0,3,5,8])
        def test_block_interval_not_event_independence(self):
            f=pd.DataFrame(dict(run=['a']*20,time=np.arange(20)/2));v=np.arange(20.)
            a=block_interval(f,v,50,701,5);b=block_interval(pd.concat([f,f]),np.r_[v,v],50,701,5)
            for k in ('mean','low','high','blocks'):self.assertAlmostEqual(a[k],b[k])
        def test_single_component_is_allowed(self):
            x=np.random.default_rng(1).normal(size=(50,2));m=mixture_fit(x,1,SimpleNamespace(smoke=True,seed=701))
            np.testing.assert_allclose(m.predict_proba(x),1)
        def test_partial_unit_resume_checksum(self):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);(p/'x').write_text('a');complete_unit(p)
                self.assertTrue(resume_unit(p,SimpleNamespace(resume=True)))
                (p/'x').write_text('b')
                with self.assertRaises(RuntimeError):resume_unit(p,SimpleNamespace(resume=True))
        def test_session_sets_exclude_outer_test(self):
            sets=fit_sets([1,2,3,4]);self.assertEqual(len(sets),10)
            for held in [1,2,3,4]:
                outer=tuple(s for s in [1,2,3,4] if s!=held)
                self.assertIn(outer,sets)
                for inner in itertools.combinations(outer,2):self.assertIn(inner,sets);self.assertNotIn(held,inner)
        def test_probability_distance_identity(self):
            p=np.array([[.2,.8]]);self.assertEqual(js_distance(p,p)[0],0)
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)).wasSuccessful()


if __name__=='__main__':
    raise SystemExit(main())
