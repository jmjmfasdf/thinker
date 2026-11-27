"""Audited, additive two-clock VoC tables. No recurrent/environment intervention.

Native choices and common-phase endpoint readouts are deliberately distinct.
The private cache is reproducible; it is never an externally required input.
"""
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
from scipy.special import softmax, expit

from .io import signature
from .result_store import ResultStore
from .native_replay import Trace, sha, verify_chunks, decision_edges, js_distance

VERSION = 'voc-dynamics-v2-action-logits'
KEYS = ['subject', 'game', 'run', 'episode', 'decision']
FLAGS = ['initial_root', 'real_transition', 'search_computation', 'current_node_depth_post',
         'current_node_depth_pre', 'actor_reset', 'phase_pre', 'search_steps_pre',
         'pre_observation_source_index', 'control_valid', 'accepted_control', 'accepted_primary_action', 'forced_stop']
VALUES = ['voc_features', 'actor_baseline', 'actor_action_probability',
          'voc_online_raw_q', 'voc_ema_raw_q', 'voc_online_q', 'voc_ema_q',
          'voc_state_value', 'voc_soft_continue_probability', 'voc_reconstruction_mode',
          'voc_effective_think_cost', 'voc_ema_available', 'tree_reps_pre', 'tree_reps_post']


def unique(frame, keys):
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate VoC keys: ' + ', '.join(keys))


def summary(value, policy, endpoint_voc, human, costs):
    """Pure endpoint summary; label-dependent fit metrics are outcomes only."""
    dv = np.diff(value); n = len(dv)
    if len(policy) != n+1 or len(endpoint_voc) != n+1 or len(costs) != n:
        raise ValueError('Endpoint/count mismatch')
    if not all(np.isfinite(x).all() for x in (value, policy, endpoint_voc, costs)):
        raise ValueError('Nonfinite endpoint data')
    if np.any(np.asarray(costs) < 0): raise ValueError('Negative effective cost')
    return dict(voc_budget=n, computation_cost=float(np.sum(costs)),
        voc_A_V=float(np.abs(dv).sum()), voc_U_V=float(np.abs(dv).mean()) if n else 0.,
        voc_D_V=float(value[-1]-value[0]), endpoint_voc_initial=float(endpoint_voc[0]),
        endpoint_voc_final=float(endpoint_voc[-1]), endpoint_voc_change=float(endpoint_voc[-1]-endpoint_voc[0]),
        policy_js=float(js_distance(policy[:1], policy[-1:])[0]),
        human_logp_change=float(np.log(policy[-1,human].clip(1e-12))-np.log(policy[0,human].clip(1e-12))),
        first_abs_delta_v=float(abs(dv[0])) if n else 0.,
        first_endpoint_voc=float(endpoint_voc[min(1,n)]))


def supported_q_change(pre, post, visits_pre, visits_post):
    support = (visits_pre > 0) & (visits_post > 0)
    changed = np.abs(post-pre)
    # Valid zero revisions are distinct from no visited-to-visited support.
    return (float(changed[support].sum()) if support.any() else np.nan,
            int(support.sum()))


class Readout:
    def __init__(self, actor, cfg):
        import torch
        import sys
        from .native_replay import frozen_import
        torch.set_num_threads(1)
        if cfg['critic_enc_type'] != 0 or cfg['last_layer_n'] != 0 or cfg['sep_actor_critic']:
            raise ValueError('VoC dynamics requires captured linear actor heads')
        # The original checkpoint also pickles its return-stat buffer. Resolve
        # that class from its frozen capture runtime, not today's training code.
        runtime = Path(actor).parents[4]
        buffer_source = runtime/'runtime_source/thinker/thinker/buffer.py'
        loaded = sys.modules.get('thinker.buffer')
        if loaded is None:
            frozen_import(runtime,'buffer')
        elif sha(loaded.__file__) != sha(buffer_source):
            raise ValueError('Loaded checkpoint buffer code differs from frozen runtime')
        checkpoint = torch.load(actor, map_location='cpu', weights_only=False)
        self.state = {k:v.detach().double().numpy() for k,v in checkpoint['actor_net_state_dict'].items()}
        self.ema = {k:v.detach().double().numpy() for k,v in checkpoint['voc_ema_gate_head_state_dict'].items()}
        self.phase = self.state['phase_embedding.weight']
        gamma = checkpoint.get('planning_meta_state', {}).get('gamma', cfg['discounting'])
        self.planning_values = (dict(discounting=gamma,
            think_cost=checkpoint['planning_meta_state']['think_cost'])
            if 'planning_meta_state' in checkpoint else {})
        self.clip = cfg['reward_clip']/(1-gamma) if cfg['reward_clip'] > 0 else np.inf

    def linear(self, x, name):
        return x @ self.state[name+'.weight'].T + self.state[name+'.bias']

    def evaluate(self, trace, rows, *, with_policy_logits=False):
        """Keep the four-value API; optionally append unclipped REAL-head logits."""
        z = trace.read('voc_features', rows).astype(float)
        phase = trace.read('phase_pre', rows).astype(int)
        baseline = self.linear(z, 'baseline')
        saved = trace.read('actor_baseline', rows)
        error = float(np.max(np.abs(baseline-saved)))
        if error > 2e-4 or np.any(np.abs(saved[:,0]) >= self.clip-1e-5):
            raise ValueError('Unreproducible or clipped task baseline')
        common = z-self.phase[phase]+self.phase[0]
        value = self.linear(common, 'baseline')[:,0]
        if np.any(np.abs(value) >= self.clip-1e-5): raise ValueError('Clipped common task readout')
        policy_logits = self.linear(common-self.phase[0]+self.phase[1], 'policy')
        if not np.isfinite(policy_logits).all(): raise ValueError('Nonfinite common policy logits')
        policy = softmax(policy_logits, axis=1)
        voc = (common @ self.ema['weight'].T + self.ema['bias'])
        probability = trace.read('voc_soft_continue_probability', rows).astype(float)
        state_value = trace.read('voc_state_value', rows).astype(float)
        mode = trace.read('voc_reconstruction_mode', rows)
        if not np.isin(mode, [0,1,2]).all(): raise ValueError('Unknown Q reconstruction schema')
        if not (np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()):
            raise ValueError('Invalid native gate probability')
        for kind, raw in [('online', self.linear(z, 'voc_head')),
                          ('ema', z @ self.ema['weight'].T + self.ema['bias'])]:
            recorded = trace.read('voc_'+kind+'_raw_q', rows)
            error = max(error, float(np.max(np.abs(raw-recorded))))
            center = probability*raw[:,0] + (1-probability)*raw[:,1]
            q = raw.copy()
            for code in (1,2):
                take = mode == code
                q[take] = state_value[take,None] + raw[take]-center[take,None]
                if code == 2: q[take] += raw[take].mean(1,keepdims=True)
            error = max(error, float(np.max(np.abs(q-trace.read('voc_'+kind+'_q', rows)))))
        real = phase == 1
        if real.any():
            actual = softmax(self.linear(z[real], 'policy'), axis=1)
            saved_policy = trace.read('actor_action_probability', rows[real])
            if not trace.valid('actor_action_probability')[rows[real]].all() or not np.isfinite(saved_policy).all():
                raise ValueError('Missing valid native REAL policy')
            error = max(error, float(np.max(np.abs(actual-saved_policy))))
        if not np.isfinite(error) or error > 2e-4: raise ValueError('Frozen Q/policy reproduction failed')
        result = value, policy, voc[:,0]-voc[:,1], error
        return (*result, policy_logits) if with_policy_logits else result


def extract(prepared, actor, cfg, chunk_size=256):
    """Stream features in bounded chunks; retain only numerical endpoint tables."""
    d = prepared['decisions'].copy(); unique(d, KEYS)
    metadata = prepared['metadata']; head = Readout(actor, cfg)
    cfg = dict(cfg, **getattr(head, 'planning_values', {}))
    choices, transitions, decisions, endpoints, quality = [], [], [], [], []
    episodes = {e['metadata']['episode_id']:e for e in metadata['episodes']}
    for episode, part in d.groupby('episode', sort=False):
        ep = episodes[episode]; t = Trace(Path(metadata['capture'])/ep['path'])
        if t.meta != ep['metadata'] or t.manifest['files_sha256'] != ep['files_sha256']:
            raise ValueError('Changed episode binding')
        files_digest = hashlib.sha256(json.dumps(t.manifest['files'],sort_keys=True,
            separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
        if files_digest != ep['files_sha256']: raise ValueError('Changed file-hash manifest')
        verify_chunks(t, FLAGS)
        flags = {k:t.scalar(k) for k in FLAGS}
        lookup = {int(r.decision):r for r in part.itertuples()}
        edges = [q for q in decision_edges(*(flags[k] for k in
            ('initial_root','real_transition','search_computation','current_node_depth_post','actor_reset')))
            if q[0] in lookup]
        if len(edges) != len(part): raise ValueError('Missing decision edges')
        selected = np.unique(np.concatenate([np.arange(left+1,right+1) for _,left,right,_,_ in edges]))
        verify_chunks(t, VALUES, selected)
        for name in VALUES:
            # Action probabilities are only defined at REAL outputs; checked there.
            if name == 'actor_action_probability': continue
            if not t.valid(name)[selected].all(): raise ValueError('Missing valid field: '+name)
        if not t.read('voc_ema_available', selected).all(): raise ValueError('EMA unavailable')
        value_parts, policy_parts, voc_parts, errors, logit_parts = [], [], [], [], []
        for begin in range(0,len(selected),chunk_size):
            v,p,c,error,logits = head.evaluate(t,selected[begin:begin+chunk_size],with_policy_logits=True)
            value_parts.append(v);policy_parts.append(p);voc_parts.append(c);errors.append(error)
            logit_parts.append(logits)
        values = np.concatenate(value_parts); policies = np.concatenate(policy_parts); vocs = np.concatenate(voc_parts)
        policy_logits = np.concatenate(logit_parts)
        np.testing.assert_allclose(softmax(policy_logits,axis=1),policies,atol=1e-12,rtol=1e-12)
        pos = {int(e):i for i,e in enumerate(selected)}
        small = {name:t.read(name,selected) for name in VALUES if name not in
                 ('voc_features','actor_baseline','actor_action_probability')}
        if not all(np.isfinite(a).all() for a in small.values()): raise ValueError('Nonfinite trace values')
        def tree(which, name, indices):
            return small['tree_reps_'+which][indices, slice(*t.tree_schema[name])]
        for j,left,right,comp,_ in edges:
            row = lookup[j]; n = len(comp)
            if left != row.root_event or right != row.real_event or n != row.budget:
                raise ValueError('Scalar/native decision mismatch')
            if n and not np.all(np.diff(comp)==1): raise ValueError('Noncontiguous computation')
            end = np.r_[comp[0],comp+1].astype(int) if n else np.array([left+1])
            if not np.array_equal(flags['search_steps_pre'][end],np.arange(n+1)):
                raise ValueError('Wrong computation endpoint count')
            if not np.all(flags['pre_observation_source_index'][end] == row.source_frame):
                raise ValueError('Endpoint crosses observation')
            ids = np.array([pos[int(e)] for e in end]); vv,pp,cc = values[ids],policies[ids],vocs[ids]
            base = {k:getattr(row,k) for k in KEYS}
            base.update(session=row.session,time=row.time,M=row.M,root_value=row.root_value,
                        log_age=np.log1p(row.age),prior_reward=row.prior_reward,carry=row.carry)
            controls = np.arange(left+1,right)
            controls = controls[(flags['phase_pre'][controls]==0)&flags['control_valid'][controls].astype(bool)]
            if not len(controls): raise ValueError('No control for scored decision')
            if not np.isin(flags['accepted_control'][controls],[0,1,2]).all(): raise ValueError('Invalid control')
            capped = bool(n and flags['forced_stop'][comp[-1]])
            if np.sum(flags['accepted_control'][controls]!=2)!=n or np.sum(flags['accepted_control'][controls]==2)!=int(not capped):
                raise ValueError('Unexpected STOP/CONTINUE count')
            cost = small['voc_effective_think_cost'][[pos[int(e)] for e in comp]]
            aggregate = summary(vv,pp,cc,int(row.human_action),cost)
            np.testing.assert_allclose([aggregate['voc_A_V'],vv[0],vv[-1]],
                                       [row.V_total,row.V_initial,row.V_final],atol=2e-4,rtol=1e-4)
            last = pos[int(controls[-1])]; first = pos[int(controls[0])]
            ema = small['voc_ema_q']; online = small['voc_online_q']
            decision = dict(base, **aggregate, initial_voc=float(ema[first,0]-ema[first,1]),
                last_choice_voc=float(ema[last,0]-ema[last,1]),
                initial_online_voc=float(online[first,0]-online[first,1]),
                termination='cap' if capped else 'voluntary_stop',
                human_action=int(row.human_action),switch=int(row.switch),budget=n)
            # Only two A-wide vectors per decision, not another vector at every
            # imagination event. No human/future label enters these readouts.
            for action in range(policy_logits.shape[1]):
                decision[f'policy_logit_pre_{action}'] = float(policy_logits[ids[0],action])
                decision[f'policy_logit_post_{action}'] = float(policy_logits[ids[-1],action])
            decisions.append(decision)
            for k,event in enumerate(end):
                endpoint = dict(base,k=k,event=int(event),value=vv[k],endpoint_voc=cc[k],
                    policy_js_step=float(js_distance(pp[k-1:k],pp[k:k+1])[0]) if k else 0.,
                    budget=n,termination=decision['termination'])
                qi=tree('pre','root_qs_mean',[pos[int(event)]])[0]
                visits=tree('pre','root_ns',[pos[int(event)]])[0]
                for action in range(len(qi)):
                    endpoint[f'root_q_{action}']=float(qi[action]) if visits[action]>0 else np.nan
                    endpoint[f'root_visits_{action}']=float(np.expm1(visits[action]))
                endpoints.append(endpoint)
            control_set = set(map(int,controls)); comp_index = {int(e):k for k,e in enumerate(comp)}
            branch = -1
            q_total = 0.; q_updates = 0
            for event in controls:
                i = pos[int(event)]; k = int(flags['search_steps_pre'][event]); control = int(flags['accepted_control'][event])
                prob = float(small['voc_soft_continue_probability'][i]); vema = float(ema[i,0]-ema[i,1])
                vo = float(online[i,0]-online[i,1]); free = k >= cfg['min_search_steps']
                if not free and control == 2: raise ValueError('STOP below minimum')
                choice = dict(base,event=int(event),k=k,log_steps=np.log1p(k),depth=int(flags['current_node_depth_pre'][event]),
                    q_continue_ema=ema[i,0],q_stop_ema=ema[i,1],q_continue_online=online[i,0],q_stop_online=online[i,1],
                    voc_ema=vema,voc_online=vo,online_ema_gap=vo-vema,probability=prob,
                    gate_expected=float(expit(vema/cfg['voc_gate_q_temperature'])),
                    gate_identity_applicable=bool(cfg.get('voc_gate_exact_projection',False)),
                    state_value=float(small['voc_state_value'][i]),reconstruction_mode=int(small['voc_reconstruction_mode'][i]),
                    accepted_control=control,continued=int(control!=2),voluntary=free,cap_reached=bool(flags['forced_stop'][event]))
                choices.append(choice)
                if event not in comp_index: continue
                ck = comp_index[int(event)]; delta = float(vv[ck+1]-vv[ck])
                if flags['current_node_depth_pre'][event] == 0:
                    branch = int(flags['accepted_primary_action'][event])
                if not 0 <= branch < pp.shape[1]: raise ValueError('Missing evaluated root branch')
                aq, qs = supported_q_change(tree('pre','root_qs_mean',[i])[0,branch:branch+1],
                    tree('post','root_qs_mean',[i])[0,branch:branch+1],
                    tree('pre','root_ns',[i])[0,branch:branch+1],tree('post','root_ns',[i])[0,branch:branch+1])
                if qs: q_total += aq; q_updates += 1
                nxt = int(event)+1; has_next = nxt in control_set
                next_free = has_next and flags['search_steps_pre'][nxt] >= cfg['min_search_steps']
                ni = pos[nxt]
                tr = dict(choice,delta_v=delta,abs_delta_v=abs(delta),abs_delta_q=aq,q_support_actions=qs,
                    policy_js_step=float(js_distance(pp[ck:ck+1],pp[ck+1:ck+2])[0]),
                    value_pre=vv[ck],value_post=vv[ck+1],endpoint_voc_post=cc[ck+1],
                    endpoint_voc_delta=cc[ck+1]-cc[ck],effective_cost=float(cost[ck]),
                    next_choice_available=has_next,next_voluntary=bool(next_free),
                    next_voc=float(ema[ni,0]-ema[ni,1]) if has_next else np.nan,
                    next_continue=int(flags['accepted_control'][nxt]!=2) if next_free else np.nan)
                transitions.append(tr)
            decision.update(voc_A_Q=q_total,voc_Q_supported_computations=q_updates)
            if hasattr(row,'Q_total'):
                np.testing.assert_allclose(q_total,row.Q_total,atol=2e-5,rtol=1e-4)
                if q_updates != row.valid_update_count: raise ValueError('Q update coverage differs from scalar preparation')
        quality.append(dict(episode=episode,decisions=len(part),readout_max_error=max(errors)))
    transition_columns = list(choices[0]) + ['delta_v','abs_delta_v','abs_delta_q','q_support_actions',
        'policy_js_step','value_pre','value_post','endpoint_voc_post','endpoint_voc_delta','effective_cost',
        'next_choice_available','next_voluntary','next_voc','next_continue']
    tables = dict(voc_choices=pd.DataFrame(choices),voc_transitions=pd.DataFrame(transitions,columns=transition_columns),
                  voc_decisions=pd.DataFrame(decisions),voc_endpoints=pd.DataFrame(endpoints))
    for name, frame in tables.items():
        if len(frame): unique(frame,KEYS+([] if name=='voc_decisions' else ['k']))
    return tables, dict(version=VERSION,episodes=quality,config={k:cfg[k] for k in
        ('min_search_steps','max_search_steps','think_cost','discounting','reward_clip','voc_gate_q_temperature')},
        scope='observational frozen readouts; no counterfactual reward utility')


def checkpoint_for_capture(metadata, campaigns=None):
    """Prefer the campaign owning this capture over duplicate checkpoint archives.

    A matching actor hash alone does not locate its frozen runtime. In particular,
    checkpoint-only archival campaigns need not contain runtime_source at all.
    """
    from .paths import REPO
    from analyses.replay_themes import find_checkpoint
    campaigns = Path(campaigns) if campaigns is not None else REPO/'logs/visualization/campaigns'
    capture = Path(metadata['capture']).resolve()
    owners = []
    for manifest in sorted(campaigns.glob('*/manifest.json')):
        record = json.loads(manifest.read_text())
        if not isinstance(record, dict):
            continue
        if any(isinstance(run, dict) and run.get('capture') and
               Path(run['capture']).resolve() == capture for run in record.get('runs', [])):
            owners.append(manifest.parent)
    if len(owners) > 1:
        raise ValueError('Multiple campaigns claim this capture: '+str(capture))
    if owners:
        cp = metadata['checkpoint']
        folder = owners[0]/'checkpoints'/cp['logical_xpid']/'milestones'/str(cp['requested_step'])
        actor, config = (folder/cp[k]['name'] for k in ('actor', 'config'))
        for kind, path in (('actor', actor), ('config', config)):
            if not path.is_file() or sha(path) != cp[kind]['sha256']:
                raise ValueError('Capture-owned checkpoint hash mismatch: '+str(path))
    else:
        # Preserve supported legacy callers, but never search arbitrary runtimes.
        folder, actor, config = find_checkpoint({'semantic': {'checkpoint': metadata['checkpoint']}})
    buffer = Path(actor).parents[4]/'runtime_source/thinker/thinker/buffer.py'
    if not buffer.is_file():
        raise FileNotFoundError('Capture checkpoint has no frozen buffer runtime: '+str(buffer))
    return folder, actor, config


def prepare(args, prepared):
    """Private, shared numeric cache; canonical outputs/00 are not overwritten."""
    import yaml
    meta = prepared['metadata']
    folder,actor,config = checkpoint_for_capture(meta)
    cfg = yaml.safe_load(config.read_text())
    # Changes to any source pack stat invalidate reuse, then extract verifies SHA.
    stamps = []
    for ep in meta['episodes']:
        p = Path(meta['capture'])/ep['path']
        manifest = json.loads((p/'manifest.json').read_text())
        specs = {f['name']:f for f in manifest['field_schema']}
        names = FLAGS+VALUES+[n+'_valid' for n in FLAGS+VALUES if n+'_valid' in specs]
        groups = {specs[n]['storage_group'] for n in names}
        stamps.append((str(p/'manifest.json'),sha(p/'manifest.json')))
        for file in manifest['files']:
            relative=Path(file['path'])
            if len(relative.parts)>1 and (relative.parts[0]!='packs' or relative.parts[1] not in groups):continue
            stat = (p/file['path']).stat()
            stamps.append((str(p/file['path']),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns))
    subject = int(prepared['decisions'].subject.iloc[0]); game = str(prepared['decisions'].game.iloc[0])
    from .paths import output_base
    root = output_base(args)/'.voc_cache'/'00'/f'sub{subject:03d}'/game
    token = signature(dict(version=VERSION,code=sha(__file__),native=sha(Path(__file__).with_name('native_replay.py')),
        manifest=meta['manifest_sha256'],sources=stamps,
        decisions=sha(prepared.get('decisions_path',Path(prepared['cache'])/'tables/decisions.csv.gz')),
        sample_mode=prepared.get('sample_mode','all'),
        frozen_buffer=sha(Path(actor).parents[4]/'runtime_source/thinker/thinker/buffer.py')))
    with ResultStore(root,dict(subject=subject,game=game,capture=meta['manifest_sha256']),resume=True) as store:
        qa = None
        with store.stage('prepare',token) as stage:
            if not stage.reused:
                tables,qa = extract(prepared,actor,cfg,getattr(args,'chunk_size',256))
                for name, frame in tables.items(): stage.write_table(name,frame)
        if qa is not None: store.meta['voc_dynamics'] = qa
        store.publish(required_stages=('prepare',))
    tables = {p.name.removesuffix('.csv.gz'):pd.read_csv(p,float_precision='round_trip') for p in (root/'tables').glob('*.csv.gz')}
    return tables,json.loads((root/'logs/run.json').read_text())['voc_dynamics']
