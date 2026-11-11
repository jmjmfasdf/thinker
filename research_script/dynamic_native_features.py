"""Additive native-cache features; no model fitting and no behavioral NPZ input.

Scalar event values retain explicit validity. Search control includes STOP even
when budget is zero. Real time and within-decision event order are separate.
Forecast outcomes live in a separate table: they are NOT origin-time predictors.
"""
import json
import numpy as np
import pandas as pd


VERSION = 1
TREE_METRICS = (
    'tree_visited_node_count', 'tree_reachable_node_count',
    'tree_expanded_node_count', 'tree_leaf_node_count', 'tree_max_reachable_depth',
)
CARRY = ('root_carried', 'useful_carry', 'carried_descendant_visit_count',
         'carried_descendant_expanded_count')
VOC = ('voc_online_q', 'voc_ema_q', 'voc_soft_continue_probability',
       'voc_state_value', 'voc_effective_think_cost')
TREE_COMPONENTS = ('root_qs_max', 'root_qs_mean', 'root_ns', 'root_policy',
    'root_v', 'root_r', 'root_trail_r', 'root_d', 'cur_r', 'cur_v', 'cur_d',
    'cur_policy', 'cur_qs_mean', 'cur_qs_max', 'cur_ns', 'rollout_return',
    'max_rollout_return', 'rollout_depth', 'depth_discount')
FLAGS = ('initial_root', 'real_transition', 'search_computation', 'search_expansion',
    'search_revisit', 'actor_reset', 'forced_stop', 'stop', 'control_valid',
    'accepted_control', 'accepted_primary_action', 'phase_pre', 'phase_post',
    'policy_type', 'current_node_depth_pre', 'current_node_depth_post',
    'current_node_has_real_state_post', 'score_mask', 'human_action', 'done',
    'truncated', 'source_frame_index', 'pre_observation_source_index',
    'post_observation_source_index', 'decision_time', 'search_steps_pre',
    'search_steps_post', 'tree_token_valid')


def read_optional(native, name, width=1):
    """Missing/invalid is NaN, never an observed zero; validity is independent."""
    if not native.has_field(name):
        return np.full((native.event_count, width), np.nan), np.zeros(native.event_count, bool)
    value = np.asarray(native.field(name), dtype=float).reshape(native.event_count, -1)
    if value.shape[1] != width:
        raise ValueError(f'Unexpected native width for {name}: {value.shape[1]} != {width}')
    valid = np.asarray(native.explicit_valid(name), bool).reshape(-1).copy()
    # -inf is a legitimate masked logit; retain it rather than inventing support.
    valid &= ~np.isnan(value).any(axis=1) if 'logits' in name else np.isfinite(value).all(axis=1)
    return np.where(valid[:, None], value, np.nan), valid


def probability(logits):
    logits = np.asarray(logits, float)
    maximum = np.max(logits, axis=-1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        exp = np.exp(logits - maximum)
        return exp / exp.sum(axis=-1, keepdims=True)


def slope(values, order):
    valid = np.isfinite(values)
    if valid.sum() < 2 or np.ptp(order[valid]) == 0:
        return np.nan
    x = order[valid].astype(float); y = values[valid]
    x -= x.mean()
    return float(np.dot(x, y-y.mean()) / np.dot(x, x))


class NativeFeatures:
    def __init__(self, native, anchors, pre, post, run, episode, window_offset, real_offset):
        self.native, self.anchors = native, np.asarray(anchors, int)
        self.pre, self.post = pre, post
        self.run, self.episode = run, episode
        self.window_offset, self.real_offset = window_offset, real_offset
        self.values, self.valid = {}, {}
        specs = {name: 1 for name in FLAGS + CARRY + VOC}
        specs.update(voc_online_q=2, voc_ema_q=2, actor_control_logits=3,
                     actor_action_probability=native.num_actions)
        specs.update({name+'_'+suffix: 1 for name in TREE_METRICS for suffix in ('pre','post','delta')})
        for name, width in specs.items():
            self.values[name], self.valid[name] = read_optional(native, name, width)
        if native.has_field('reward'):
            width=int(np.prod(native.field('reward').shape[1:]))
            self.values['reward'],self.valid['reward']=read_optional(native,'reward',width)
        times = self.scalar('decision_time')[self.anchors[1:]]
        good = np.isfinite(times)
        if good.any() and (not good.all() or np.any(np.diff(times) <= 0)):
            raise ValueError('Decision timestamps are partial or nonmonotonic within episode')
        self.times = times
        self.records, self.forecasts, self.path_rows = [], [], []

    def scalar(self, name):
        return self.values[name][:, 0]

    def mask(self, name):
        return self.scalar(name) == 1

    def key(self, local):
        step = self.real_offset + local + 1
        return dict(run_label=self.run, real_step=step,
                    real_step_id=f'{self.run}_real{step:06d}',
                    episode_id=self.episode, window_index=self.window_offset+local)

    def decision(self, local, computations, paths, depths):
        left, right = self.anchors[local:local+2]
        stage = np.arange(left+1, right)
        controls = stage[(self.scalar('phase_pre')[stage] == 0) & self.mask('control_valid')[stage]]
        if not len(controls):
            raise ValueError('Native decision has no observed pre-action search control')
        first, last = int(controls[0]), int(controls[-1])
        stops = controls[self.mask('stop')[controls] & (self.scalar('accepted_control')[controls] == 2)]
        forced = controls[self.mask('forced_stop')[controls]]
        if len(stops)+len(forced) != 1 or (len(stops) and stops[-1] != last) or (len(forced) and forced[-1] != last):
            raise ValueError('Decision must terminate in one voluntary STOP or forced computation')
        time = self.times[local]
        if np.isfinite(time) and not np.allclose(self.scalar('decision_time')[stage], time, atol=1e-7):
            raise ValueError('Search event human-time alignment differs from its real decision')
        row = dict(self.key(local), decision_time=time, decision_time_valid=int(np.isfinite(time)),
            origin_time=time, available_time=time, first_control_native_event=first,
            last_control_native_event=last, real_action_native_event=int(right),
            stop_native_event=int(stops[0]) if len(stops) else -1,
            additional_computations=len(computations), control_event_count=len(controls),
            stop=int(bool(len(stops))), forced_stop=int(bool(len(forced))),
            voluntary_stop_observed=int(bool(len(stops))), stop_choice_censored=int(bool(len(forced))),
            search_expansion_count=int(self.mask('search_expansion')[computations].sum()),
            search_revisit_count=int(self.mask('search_revisit')[computations].sum()),
            reset_count=int((self.scalar('accepted_control')[computations] == 1).sum()),
            accepted_control=self.scalar('accepted_control')[last],
            max_path_depth=int(max(depths, default=0)),
            evaluated_path_count=len({tuple(p[:d]) for p,d in zip(paths,depths) if d}),
            carry_native_event=int(left), human_action=self.scalar('human_action')[right])
        # Carry from the incoming root, NOT the new root after a_t was executed.
        for name in CARRY:
            row[name] = self.scalar(name)[left]
            row[name+'_valid'] = int(self.valid[name][left])
        for name in VOC + ('actor_control_logits',):
            values = self.values[name]
            labels = ('continue','stop') if name.endswith('_q') else ('proceed','reset','stop') if name == 'actor_control_logits' else ('',)
            for col,label in enumerate(labels):
                stem = name+'_'+label if label else name
                for suffix,index in [('initial',first),('terminal',last),('at_stop',int(stops[0]) if len(stops) else -1)]:
                    row[stem+'_'+suffix] = values[index,col] if index >= 0 else np.nan
                    row[stem+'_'+suffix+'_valid'] = int(index >= 0 and self.valid[name][index])
                if name != 'actor_control_logits':
                    observed = values[controls,col]
                    row[stem+'_mean'] = float(observed[np.isfinite(observed)].mean()) if np.isfinite(observed).any() else np.nan
                    row[stem+'_slope'] = slope(observed,np.arange(len(controls)))
                    row[stem+'_valid_events'] = int(np.isfinite(observed).sum())
            if name.endswith('_q'):
                diff = values[controls,0]-values[controls,1]
                stem = name.replace('_q','')+'_voc'
                row.update({stem+'_initial':diff[0], stem+'_terminal':diff[-1],
                    stem+'_mean':float(np.mean(diff[np.isfinite(diff)])) if np.isfinite(diff).any() else np.nan,
                    stem+'_slope':slope(diff,np.arange(len(controls))),
                    stem+'_valid_events':int(np.isfinite(diff).sum()),
                    stem+'_initial_valid':int(np.isfinite(diff[0])),
                    stem+'_terminal_valid':int(np.isfinite(diff[-1]))})
        for name in TREE_METRICS:
            a,b = self.scalar(name+'_pre')[first], self.scalar(name+'_post')[last]
            row.update({name+'_pre':a,name+'_post':b,name+'_delta':b-a,
                        name+'_valid':int(np.isfinite(a) and np.isfinite(b))})
        for suffix,index in [('pre',first),('post',last)]:
            row['search_steps_'+suffix] = self.scalar('search_steps_'+suffix)[index]
        row['search_steps_delta'] = row['search_steps_post']-row['search_steps_pre']
        for name in TREE_COMPONENTS:
            if name not in self.pre or name not in self.post:
                continue
            for suffix,tree,index in [('initial',self.pre,first),('terminal',self.post,last)]:
                vector = np.asarray(tree[name][index]).reshape(-1)
                for a,value in enumerate(vector):
                    row[f'{name}_{suffix}'+(f'_{a}' if len(vector)>1 else '')] = float(value)
        probs = self.values['actor_action_probability'][right]
        row['actor_action_probability_valid'] = int(self.valid['actor_action_probability'][right])
        if row['actor_action_probability_valid'] and (np.any(probs < 0) or not np.isclose(probs.sum(),1,atol=1e-5)):
            raise ValueError('Recorded action probability is not a probability distribution')
        for a,value in enumerate(probs):
            row[f'actor_action_probability_{a}'] = value  # already probabilities, NO softmax
        for j,(event,path,depth) in enumerate(zip(computations,paths,depths)):
            path = path[:depth].astype(int).tolist()
            self.path_rows.append(dict(window_index=row['window_index'], event_index=j,
                                       native_event=int(event), path=path))
            self.forecasts.append(self.forecast(local,int(event),j,path))
        self.records.append(row)
        return row

    def forecast(self, local, event, event_index, path):
        depth = len(path); total = len(self.times)
        a_target, x_target = local+depth-1, local+depth
        action_ok = depth > 0 and a_target < total
        state_ok = depth > 0 and x_target <= total
        actions = self.scalar('human_action')[self.anchors[1:]]
        prefix = action_ok and np.array_equal(path[:-1], actions[local:a_target])
        full = prefix and path[-1] == actions[a_target]
        # x_T has a captured frame, but no next decision timestamp. Do not invent one.
        state_time = self.times[x_target] if state_ok and x_target < total else np.nan
        return dict(self.key(local), native_event=event,event_index=event_index,depth=depth,
            path=json.dumps(path,separators=(',',':')), origin_time=self.times[local],
            forecast_available_time=self.times[local],forecast_available_event=event,
            action_target_local=a_target if action_ok else -1,
            action_target_time=self.times[a_target] if action_ok else np.nan,
            state_target_local=x_target if state_ok else -1,
            state_target_native_event=int(self.anchors[x_target]) if state_ok else -1,
            state_target_time=state_time, outcome_available_time=state_time,
            outcome_available_event=int(self.anchors[x_target]) if state_ok else -1,
            target_has_next_decision=int(state_ok and x_target < total),
            action_target_valid=int(action_ok),state_target_valid=int(state_ok),
            prefix_before_match=float(prefix) if action_ok else np.nan,
            prefix_full_match=float(full) if action_ok else np.nan,
            target_action=actions[a_target] if action_ok else np.nan,
            action_match=float(path[-1] == actions[a_target]) if action_ok else np.nan,
            state_comparison_primary=int(full), predicted_reward=self.post['cur_r'][event,0] if 'cur_r' in self.post else np.nan,
            predicted_value=self.post['cur_v'][event,0] if 'cur_v' in self.post else np.nan,
            realized_root_reward=self.post['root_r'][self.anchors[x_target],0] if state_ok and 'root_r' in self.post else np.nan,
            realized_root_value=self.post['root_v'][self.anchors[x_target],0] if state_ok and 'root_v' in self.post else np.nan)

    def event_table(self):
        count = self.native.event_count
        local = np.searchsorted(self.anchors[1:],np.arange(count),side='left')
        if local.max() >= len(self.records):
            raise ValueError('Unassigned native events after the last complete transition')
        keys = pd.DataFrame(self.records)[['run_label','real_step','real_step_id','episode_id','window_index']]
        out = {k:keys[k].to_numpy()[local] for k in keys}
        out.update(native_event=np.arange(count),event_order=np.arange(count)-self.anchors[local]-1,
                   origin_time=self.times[local],pre_action_available_time=self.times[local],
                   available_time=self.times[local].copy())
        # The post-state on a real transition is x_(t+1), not a predictor for a_t.
        real=self.mask('real_transition')
        next_times=np.r_[self.times[1:],np.nan]
        out['available_time'][real]=next_times[local[real]]
        out['post_state_available_time']=out['available_time'].copy()
        out['available_event']=np.arange(count)
        for name,values in self.values.items():
            out[name+'_valid'] = self.valid[name].astype(np.int8)
            labels = ('continue','stop') if name in ('voc_online_q','voc_ema_q') else ('proceed','reset','stop') if name == 'actor_control_logits' else tuple(map(str,range(values.shape[1])))
            for j,label in enumerate(labels):
                out[name+(('_'+label) if values.shape[1]>1 else '')] = values[:,j]
        out['is_control_choice'] = ((self.scalar('phase_pre') == 0) & self.mask('control_valid') & ~self.mask('initial_root') & ~self.mask('real_transition')).astype(np.int8)
        for mode,tree in [('pre',self.pre),('post',self.post)]:
            for name in TREE_COMPONENTS:
                if name not in tree:
                    continue
                values = np.asarray(tree[name]).reshape(count,-1)
                for a in range(values.shape[1]):
                    out[f'{name}_{mode}'+(f'_{a}' if values.shape[1]>1 else '')] = values[:,a]
                if name in ('root_policy','cur_policy'):
                    for a,v in enumerate(probability(values).T):out[f'{name}_probability_{mode}_{a}'] = v
        for name in ('voc_online','voc_ema'):
            out[name+'_voc'] = self.values[name+'_q'][:,0]-self.values[name+'_q'][:,1]
        return pd.DataFrame(out)


def pack_paths(rows):
    """CSR/ragged paths: no arbitrary maximum depth and no object/pickle arrays."""
    lengths = np.array([len(r['path']) for r in rows],dtype=np.int32)
    return dict(path_offsets=np.r_[0,np.cumsum(lengths,dtype=np.int64)],
        path_actions=np.array([a for r in rows for a in r['path']],dtype=np.int16),
        path_lengths=lengths,**{k:np.array([r[k] for r in rows],dtype=np.int64)
            for k in ('window_index','event_index','native_event')})


def static_descriptor(images):
    """24D static subset of the established 40D image descriptor, no future motion."""
    x = np.asarray(images,np.float32)/255
    if x.ndim != 4 or x.shape[1] != 3:raise ValueError('Expected NCHW RGB images')
    gray = np.einsum('nchw,c->nhw',x,np.array([.2989,.5870,.1140],np.float32))
    h,w=gray.shape[1:];h=h//4*4;w=w//4*4
    pool=gray[:,:h,:w].reshape(len(x),4,h//4,4,w//4).mean((2,4)).reshape(len(x),16)
    return np.column_stack((pool,x.mean((2,3)),x.std((2,3)),
        abs(np.diff(gray,axis=2)).mean((1,2)),abs(np.diff(gray,axis=1)).mean((1,2))))


def xs_display(values):
    """Recorder-compatible RGB of latest stacked frame; retain raw XS separately in Zarr."""
    x=np.asarray(values,np.float32)
    if x.ndim!=4 or x.shape[1]<3:raise ValueError('predicted_xs must be stacked CHW RGB')
    x=x[:,-3:]
    for i in range(len(x)):
        if np.isfinite(x[i]).all() and x[i].max()<=1.0:x[i]*=255
    return np.rint(np.clip(x,0,255))


def prediction_summaries(features, include_images=False, include_latents=False, batch_size=128):
    """Heavy reads are opt-in, bounded, and never duplicate raw images/latents.

Latents: fixed channel mean/std features, not a full-data-fitted PCA. Actual
prediction errors are outcomes with their target observation availability.
"""
    if batch_size<1:raise ValueError('batch_size must be positive')
    native=features.native
    columns=list(features.forecast(0,int(features.anchors[0]),0,[])) if len(features.times) else []
    forecasts=pd.DataFrame(features.forecasts,columns=columns)
    summaries=[]
    sources=[]
    if include_images:sources+=['imagined_image','predicted_xs']
    if include_latents:sources+=['im_vectors']
    validity={name:np.asarray(native.explicit_valid(name),bool)
              for name in sources+(['real_image'] if include_images else [])
              if native.has_field(name)}
    for name in sources:
        for metric in ('mse','baseline_mse','descriptor_rmse') if name!='im_vectors' else ('mse','baseline_mse','cosine_distance'):
            forecasts[name+'_'+metric]=np.nan
        forecasts[name+'_comparison_valid']=0
        forecasts[name+'_source_valid']=0
    needed_anchors=features.anchors[:len(features.records)+1]
    rows=np.unique(np.r_[needed_anchors,np.array([r['native_event'] for r in features.forecasts],dtype=int)])
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size]
        records={int(e):dict(native_event=int(e),episode_id=features.episode,run_label=features.run,
                             capture_episode_path=str(native.path)) for e in batch}
        for event,r in records.items():
            local=int(np.searchsorted(features.anchors[1:],event,side='left'))
            target=local+1 if features.mask('real_transition')[event] else local
            r.update(window_index=features.window_offset+local,origin_time=features.times[local],
                     available_time=features.times[target] if target<len(features.times) else np.nan,
                     available_event=event)
        for name in (['real_image'] if include_images else [])+sources:
            valid = validity[name][batch] if name in validity else np.zeros(len(batch),bool)
            values=native.read_field_rows(name,batch) if native.has_field(name) else None
            for j,event in enumerate(batch):
                r=records[int(event)];good=bool(valid[j] and np.isfinite(values[j]).all()) if values is not None else False
                r[name+'_valid']=int(good)
                channels=int(np.prod(values.shape[1:-2])) if name=='im_vectors' and values is not None and values.ndim>=4 else 0
                if name=='im_vectors':
                    for c in range(channels):
                        r[f'im_vectors_mean_{c}']=np.nan;r[f'im_vectors_std_{c}']=np.nan
                else:
                    for k in range(24):r[f'{name}_descriptor_{k}']=np.nan
                if not good:continue
                v=values[j].astype(np.float32)
                if name=='im_vectors':
                    v=v.reshape(-1,*v.shape[-2:])
                    for c,(mean,std) in enumerate(zip(v.mean((1,2)),v.std((1,2)))):
                        r[f'im_vectors_mean_{c}']=float(mean);r[f'im_vectors_std_{c}']=float(std)
                else:
                    image=xs_display(v[None])[0] if name=='predicted_xs' else v
                    for k,value in enumerate(static_descriptor(image[None])[0]):r[f'{name}_descriptor_{k}']=float(value)
        summaries.extend(records.values())
    # Pairwise comparison is additionally conditioned on action-prefix agreement
    # downstream via state_comparison_primary; divergent paths stay in the table.
    for start in range(0,len(forecasts),batch_size):
        ids=forecasts.index[start:start+batch_size]
        if not len(ids):continue
        block=forecasts.loc[ids]
        target_ok=block.state_target_valid.to_numpy(bool)
        origin=features.anchors[np.searchsorted(features.anchors[1:],block.native_event.to_numpy(int),side='left')]
        target=np.where(target_ok,block.state_target_native_event.to_numpy(int),origin)
        for name in sources:
            target_name='im_vectors' if name=='im_vectors' else 'real_image'
            if not native.has_field(name) or not native.has_field(target_name):continue
            events=block.native_event.to_numpy(int)
            pred=native.read_field_rows(name,events).astype(np.float32)
            actual=native.read_field_rows(target_name,target).astype(np.float32)
            past=native.read_field_rows(target_name,origin).astype(np.float32)
            source_valid=validity[name][events] & np.isfinite(pred).reshape(len(ids),-1).all(1)
            good=target_ok & source_valid & validity[target_name][target] & np.isfinite(actual).reshape(len(ids),-1).all(1)
            baseline_good=good & validity[target_name][origin] & np.isfinite(past).reshape(len(ids),-1).all(1)
            if name=='im_vectors':
                # A carried prediction is not an independently observed state.
                good &= features.mask('current_node_has_real_state_post')[target]
                baseline_good &= good & features.mask('current_node_has_real_state_post')[origin]
            else:
                pred=xs_display(pred)/255 if name=='predicted_xs' else pred/255
                actual/=255;past/=255
            a,b,c=(v.reshape(len(ids),-1) for v in (pred,actual,past))
            if a.shape!=b.shape:raise ValueError(f'Prediction/target shape mismatch for {name}')
            forecasts.loc[ids,name+'_source_valid']=source_valid.astype(int)
            forecasts.loc[ids,name+'_comparison_valid']=good.astype(int)
            forecasts.loc[ids,name+'_mse']=np.where(good,np.mean((a-b)**2,axis=1),np.nan)
            forecasts.loc[ids,name+'_baseline_mse']=np.where(baseline_good,np.mean((c-b)**2,axis=1),np.nan)
            if name=='im_vectors':
                denom=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
                cosine=1-np.divide((a*b).sum(1),denom,out=np.full(len(ids),np.nan),where=denom>0)
                forecasts.loc[ids,name+'_cosine_distance']=np.where(good,cosine,np.nan)
            else:
                distance=np.sqrt(np.mean((static_descriptor(pred*255)-static_descriptor(actual*255))**2,axis=1))
                forecasts.loc[ids,name+'_descriptor_rmse']=np.where(good,distance,np.nan)
    targets=forecasts.state_target_native_event.to_numpy(int)
    if 'reward' in features.values:
        forecasts['observed_reward']=np.where(targets>=0,features.values['reward'][np.maximum(targets,0),0],np.nan)
    else:
        forecasts['observed_reward']=np.nan
    forecasts['reward_prediction_error']=forecasts.observed_reward-forecasts.predicted_reward
    forecasts['value_discrepancy']=forecasts.realized_root_value-forecasts.predicted_value
    return forecasts,pd.DataFrame(summaries)


def schema_contract(options):
    return dict(version=VERSION,options=options,
        control_codes={'0':'PROCEED','1':'RESET','2':'STOP'},voc_q_order=['CONTINUE','STOP'],
        tree_visits='Decoded expm1(log1p(count)); no budget normalization in native tables',
        carry_alignment='Incoming anchor x_t before a_t; not post-action carry',
        tree_alignment='Initial pre-first-control; terminal post-last-control, before real action',
        voc_alignment='Actor-visible pre-control Q, including final voluntary STOP and B=0',
        voc_slope_units='Q per observed control-choice index; no conversion to human seconds',
        stop_counterfactual='Model Q(CONTINUE) at observed STOP; not an observed causal counterfactual',
        paths='CSR path_offsets/path_actions; every computation, including empty root RESET paths',
        forecast_alignment='depth d action -> a[t+d-1]; state -> x[t+d]; same episode only',
        outcome_warning='Forecast mismatch columns are target-time outcomes, NEVER origin-time predictors. Use outcome_available_time/event and prefix_full_match.',
        terminal_state_time='Captured x_T retained, but timestamp NaN without a next decision; do not fabricate 15-Hz time',
        actor_probability='Recorded actor_action_probability, no softmax; distinct from actor_policy logits',
        latent_summary='Fixed channel mean/std, no globally fitted PCA. Fit PCA only on training folds. Latent distance is model-state discrepancy, not ground-truth latent error',
        policy_summary='Native pre/post logits plus probability columns; not SEARCH-to-REAL same-head revision')
