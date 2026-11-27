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
RESEARCH = Path(__file__).resolve().parents[1]
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
    root=Path(root)
    source=(root/'runtime_source/thinker').resolve()
    if not source.is_dir():
        # Prefer the campaign physically owning these captures. If a collection
        # spans campaigns, require identical Thinker Python code before reuse.
        captures=list(root.glob('sub*/*/COMPLETED.json'))
        owners={p.parent.resolve().parents[1] for p in captures}
        candidates=[p/'runtime_source/thinker' for p in owners if (p/'runtime_source/thinker').is_dir()]
        if not candidates:
            candidates=list((RESEARCH.parent/'logs/visualization/campaigns').glob('*/runtime_source/thinker'))
        if not candidates:raise FileNotFoundError('No frozen capture runtime')
        fingerprints=[]
        for candidate in candidates:
            fingerprints.append({str(p.relative_to(candidate)):sha(p) for p in (candidate/'thinker').rglob('*.py')})
        # runtime evaluation entrypoints can differ; inference modules must agree.
        needed=('thinker/behavior_trace.py','thinker/model_net.py','thinker/actor_net.py',
                'thinker/util.py','thinker/dynamic_imitation.py')
        selected=[{k:v for k,v in f.items() if k in needed or k.startswith('thinker/core/')} for f in fingerprints]
        if any(f!=selected[0] for f in selected[1:]):
            raise ValueError('Multiple incompatible frozen runtimes; provide a specific campaign root')
        source=sorted(candidates)[0].resolve()
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    result=importlib.import_module('thinker.'+module)
    if not Path(result.__file__).resolve().is_relative_to(source):
        raise RuntimeError('A different Thinker package was imported; use a fresh Python process')
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
    root=Path(root)
    capture = root/f'sub{subject:03d}'/game
    if not (capture/'COMPLETED.json').is_file():
        capture = root / 'captures' / f'{game}-sub{subject:03d}-all'
    if not (capture/'COMPLETED.json').is_file():
        raise ValueError(f'Missing completed capture: {capture}')
    pattern = f'{game}-s{subject:03d}-*/milestones/10000000'
    checkpoints = sorted({p.resolve() for p in (root/'checkpoints').glob(pattern)})
    # The completed collection reuses sub001 Pong/Enduro from the first
    # campaign. Its immutable checkpoints remain in that sibling campaign.
    if not checkpoints:
        checkpoints = sorted({p.resolve() for p in root.parent.glob('*/checkpoints/'+pattern)})
    if not checkpoints:
        checkpoints=sorted({p.resolve() for p in (RESEARCH.parent/'logs/visualization/campaigns').glob('*/checkpoints/'+pattern)})
    plan=read_json(capture/'capture_plan.json')['semantic']['checkpoint']
    matching=[]
    for folder in checkpoints:
        if all((folder/plan[k]['name']).is_file() and sha(folder/plan[k]['name'])==plan[k]['sha256']
               for k in ('actor','model','config')):matching.append(folder)
    if not matching:
        raise FileNotFoundError(f'No checkpoint matching completed capture: sub{subject:03d}/{game}')
    return capture,matching[0]


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
        stamps[str(p/'manifest.json')]=file_stamp(p/'manifest.json')
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
    from common.sampling import sequence_index
    bad=d.episode.ne(d.episode.shift())|sequence_index(d).diff().ne(1)|d.time.diff().le(0)|d.time.diff().gt(.25)
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


