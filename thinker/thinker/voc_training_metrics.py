"""Counter-backed online diagnostics; no effect on the training objective."""
from collections import deque
import json
import time
import numpy as np
import torch
from thinker import util


class TrainingMetrics:
    def __init__(self, actor, flags, output, state=None):
        self.flags, self.output = flags, output
        self.schema = util.slice_dynamic_tree_reps(int(actor.num_actions))
        self.cap = int(flags.max_search_steps)
        self.current = np.zeros(flags.env_n, dtype=np.int64)
        self.last_episode = np.zeros(flags.env_n)
        self.returns = deque(maxlen=100)
        self.hist = np.zeros(self.cap + 1, dtype=np.int64)
        self.steps = self.think = self.episodes = self.controls = self.forced = 0
        self.stops = self.resets = 0
        self.raw_reward = 0.
        self.best = None
        self.gate_logits = None
        self.hook = actor.voc_gate_head.register_forward_hook(self._gate)
        self.rows = []
        if state:
            for k in ('steps','think','episodes','controls','forced','stops','resets','raw_reward','best'):
                setattr(self, k, state[k])
            self.hist = np.asarray(state['hist'], dtype=np.int64)
            self.returns.extend(state['returns'])
        # Cold resume starts a new episode/planning stage. Old partial stages
        # remain in think, but are not falsely counted as completed decisions.

    def _gate(self, module, inputs, output):
        self.gate_logits = output.detach()

    def observe(self, previous, out, info, *, real_step, attempt):
        def array(x):
            return x.detach().cpu().numpy().reshape(-1)
        real = array(info['real_transition']).astype(bool)
        if real.any() and not real.all():
            raise AssertionError('Expected synchronous real-action barrier')
        imagined = array(info['think_reward']) == -1
        accepted = array(info['accepted_control'])
        valid = array(out.control_valid).astype(bool)
        forced = array(previous.forced_stop).astype(bool)
        phase = array(previous.phase)
        self.think += int(imagined.sum())
        self.current += imagined
        self.controls += int(valid.sum())
        self.stops += int(((accepted == util.STOP) & valid).sum())
        self.resets += int(((accepted == util.RESET) & valid).sum())
        self.forced += int(((phase == util.SEARCH_PHASE) & forced).sum())
        # Deterministic stream-0 subsample, including every query within the
        # sampled decision. Other streams remain in all aggregate counters.
        decision = real_step // self.flags.env_n
        if decision % 128 == 0 and phase[0] == util.SEARCH_PHASE:
            tree = previous.tree_reps[0, 0].detach().float().cpu().numpy()
            logits = out.search_control_logits[0, 0].detach().float()
            p = logits.softmax(-1).cpu().numpy()
            q = out.voc_q[0, 0].detach().float().cpu().numpy()
            prior = tree[self.schema['root_policy']]
            ordered = np.sort(prior)
            self.rows.append(dict(attempt=attempt,real_step=real_step,stream=0,decision=decision,
                count=int(array(previous.search_steps)[0]),
                depth=int(round(float(np.expm1(tree[self.schema['rollout_depth']][0])))),
                online_delta_q=float(q[0]-q[1]),
                ema_delta_q=float(self.gate_logits.reshape(-1)[0].item()*self.flags.voc_gate_q_temperature),
                p_continue=float(p[0]+p[1]),control=int(accepted[0]),valid=bool(valid[0]),
                forced=bool(forced[0]),root_policy_margin=float(ordered[-1]-ordered[-2]),
                root_value_token=float(tree[self.schema['root_v']][0])))
        if real.all():
            self.steps += int(real.sum())
            if np.any(self.current > self.cap):
                raise AssertionError('Imagination budget exceeded cap')
            self.hist += np.bincount(self.current, minlength=self.cap+1)
            self.current[:] = 0
            ep = array(info['episode_return'])
            done = array(info['real_done']).astype(bool)
            self.raw_reward += float((ep-self.last_episode).sum())
            self.last_episode = ep.copy()
            self.last_episode[done] = 0
            for value in ep[done]:
                value = float(value)
                self.returns.append(value)
                self.episodes += 1
                self.best = value if self.best is None else max(self.best, value)
                with (self.output/'episodes.jsonl').open('a') as f:
                    f.write(json.dumps(dict(attempt=attempt,real_step=real_step+int(real.sum()),return_raw=value))+'\n')

    def flush(self):
        if self.rows:
            with (self.output/'control_samples.jsonl').open('a') as f:
                for row in self.rows:
                    f.write(json.dumps(row, allow_nan=False)+'\n')
            self.rows.clear()

    def publish(self, terminal=False):
        self.flush()
        payload = dict(physical_env_steps=self.steps, updated_at=time.time(),
                       terminal=terminal, **self.metrics())
        path = self.output / 'collector_metrics.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, allow_nan=False))
        tmp.replace(path)

    def state(self):
        return {**{k:getattr(self,k) for k in ('steps','think','episodes','controls','forced','stops','resets','raw_reward','best')},
                'hist':self.hist.tolist(),'returns':list(self.returns)}

    def metrics(self):
        n = int(self.hist.sum())
        result = dict(episodes_completed=self.episodes,raw_reward_sum=self.raw_reward,
            mean_last_100_episode_return=float(np.mean(self.returns)) if self.returns else None,
            max_completed_episode_return=self.best,
            raw_reward_per_1000_actions=1000*self.raw_reward/max(self.steps,1),
            imagined_transitions=self.think,computations_per_action=self.think/max(self.steps,1),
            zero_search_fraction=float(self.hist[0]/n) if n else None,
            cap_search_fraction=float(self.hist[-1]/n) if n else None,
            learned_stop_fraction=self.stops/max(self.controls,1),
            reset_fraction=self.resets/max(self.controls,1),forced_stop_count=self.forced)
        for k, value in enumerate(self.hist):
            result[f'count_{k}_decisions'] = int(value)
        return result

    def close(self):
        self.flush()
        self.hook.remove()
