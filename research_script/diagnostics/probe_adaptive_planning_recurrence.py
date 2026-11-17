"""Isolate tree-GRU readout sensitivity on unchanged allocation trajectories.

After the first optional choice, compare actual features with (1) the same
tree hidden state held fixed and (2) the first tree token repeatedly processed
by a private GRU state. Other actor features remain their observed values.
Neither alternative controls the environment, so these are representation
diagnostics, not counterfactual policies or estimates of computation utility.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


FEATURE_GROUPS = {
    'freeze_progress': ('search_steps', 'rollout_depth', 'depth_discount'),
    'freeze_visits': ('root_ns', 'cur_ns'),
    'freeze_values': ('root_r', 'root_v', 'root_qs_mean', 'root_qs_max',
                      'root_trail_r', 'rollout_return', 'max_rollout_return',
                      'cur_r', 'cur_v', 'cur_qs_mean', 'cur_qs_max'),
}


def exact_overlap(rows, reference):
    lookup = {(r['seed'], r['stream'], r['decision']): r for r in reference}
    for row in rows:
        if lookup.get((row['seed'], row['stream'], row['decision'])) != row:
            raise ValueError('read-only observer changed allocation or lacks reference')
    return len(rows)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['count']].append(row)
    result = []
    for count, group in sorted(groups.items()):
        entry = dict(count=count, observed_decisions=len(group))
        names = ('actual', 'frozen_hidden', 'repeated_token', 'frozen_other', *FEATURE_GROUPS)
        for name in (name for name in names if name in group[0]):
            entry[name+'_mean_probability'] = float(np.mean([r[name] for r in group]))
            entry[name+'_change_from_first'] = float(np.mean([
                r[name]-r['first_probability'] for r in group]))
            entry[name+'_mean_absolute_gap_from_actual'] = float(np.mean([
                abs(r[name]-r['actual']) for r in group]))
            if name+'_log_odds' in group[0]:
                entry[name+'_mean_log_odds'] = float(np.mean([r[name+'_log_odds'] for r in group]))
                entry[name+'_log_odds_change_from_first'] = float(np.mean([
                    r[name+'_log_odds']-r['first_log_odds'] for r in group]))
        result.append(entry)
    return result


def observe(evaluation, flags, actor, model, *, seed, decisions, warmup, feature_freezes=False):
    original = evaluation.BatchRuntime
    torch, util = evaluation.torch, evaluation.util
    rows = []

    class ObservedRuntime(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.actor.last_layer_n != 0 or len(self.actor.tree_rep_encoder.cells) != 1:
                raise ValueError('this probe requires direct features and a single tree GRU')
            if not self.actor.voc_dedicated_gate or self.actor.voc_gate_temperature != 1.:
                raise ValueError('this probe requires the documented scalar soft gate')
            self.decision = -1
            self.env_input = self.token = self.features = None
            self.first_token = self.anchor = self.repeated = self.first_p = None
            self.first_features = self.first_log_odds = None
            self.feature_hidden = {}
            self.schema = util.slice_dynamic_tree_reps(self.actor.num_actions)
            self.anchor_decision = np.full(self.batch_size, -1, dtype=int)
            self.offset = self.actor.dim_rep_actions+2+self.actor.num_input_rewards
            self.hidden_size = self.actor.tree_rep_encoder.cells[0].hidden_size
            self.hooks = [
                self.actor.register_forward_pre_hook(self.before_actor),
                self.actor.tree_rep_encoder.register_forward_pre_hook(self.before_tree, with_kwargs=True),
                self.actor.policy.register_forward_pre_hook(self.before_policy),
                self.actor.register_forward_hook(self.after_actor),
            ]

        def before_actor(self, module, args):
            self.env_input = args[0]
            if self.decision < 0:
                self.decision = 0
            elif self.env_input.real_transition[0].all():
                self.decision += 1

        def before_tree(self, module, args, kwargs):
            self.token = dict(x=args[0][0].detach(), h=kwargs['state'][0].detach(),
                update=kwargs['update_mask'][0], reset=kwargs['reset_mask'][0])

        def before_policy(self, module, args):
            self.features = args[0].detach()

        def after_actor(self, module, args, output):
            if self.decision < warmup:
                return
            env = self.env_input
            phase = env.phase[0]
            counts = env.search_steps[0]
            legal = env.legal_control_mask[0].bool()
            optional = (phase == util.SEARCH_PHASE) & legal[:, util.STOP] & legal[:, :2].any(-1)
            if not optional.any():
                return
            x, prior = self.token['x'], self.token['h']
            update, reset = self.token['update'], self.token['reset']
            cell = self.actor.tree_rep_encoder.cells[0]
            base = torch.where((reset & update)[:, None], torch.zeros_like(prior), prior)
            actual_h = torch.where(update[:, None], cell(x, base), base)
            sl = slice(self.offset, self.offset+self.hidden_size)
            phase_embedding = self.actor.phase_embedding(phase)[:, sl]
            torch.testing.assert_close(actual_h+phase_embedding, self.features[:, sl],
                                       rtol=2e-5, atol=2e-6)
            if self.anchor is None:
                self.first_token = torch.zeros_like(x)
                self.anchor = torch.zeros_like(actual_h)
                self.repeated = torch.zeros_like(actual_h)
                self.first_p = torch.zeros_like(counts, dtype=torch.float32)
                self.first_log_odds = torch.zeros_like(self.first_p)
                self.first_features = torch.zeros_like(self.features)
                if feature_freezes:
                    self.feature_hidden = {name: torch.zeros_like(actual_h) for name in FEATURE_GROUPS}
            actual_p = output[0].search_control_logits[-1].softmax(-1)[:, :2].sum(-1)
            raw_actual = self.actor.voc_gate_head(self.features).flatten()
            reconstructed = raw_actual.sigmoid()
            torch.testing.assert_close(reconstructed[optional], actual_p[optional], rtol=1e-5, atol=3e-7)
            first = optional & (counts == flags.min_search_steps)
            later = optional & ~first
            for i in first.nonzero().flatten().tolist():
                self.anchor_decision[i] = self.decision
            if any(self.anchor_decision[i] != self.decision for i in later.nonzero().flatten().tolist()):
                raise ValueError('missing first optional state in this decision')
            if reset[later].any() or not update[later].all():
                raise ValueError('unexpected reset or skipped tree token during optional continuation')
            self.first_token[first] = x[first]
            self.anchor[first] = actual_h[first]
            self.repeated[first] = actual_h[first]
            self.first_p[first] = actual_p[first]
            self.first_log_odds[first] = raw_actual[first]
            self.first_features[first] = self.features[first]
            for hidden in self.feature_hidden.values():
                hidden[first] = actual_h[first]
            if later.any():
                advanced = cell(self.first_token, self.repeated)
                self.repeated[later] = advanced[later]
                for name, hidden in self.feature_hidden.items():
                    token = x.clone()
                    for key in FEATURE_GROUPS[name]:
                        token[:, self.schema[key]] = self.first_token[:, self.schema[key]]
                    advanced = cell(token, hidden)
                    hidden[later] = advanced[later]
            alternatives, raw_alternatives = {}, {}
            variants = dict(frozen_hidden=self.anchor, repeated_token=self.repeated, **self.feature_hidden)
            for name, hidden in variants.items():
                features = self.features.clone()
                features[:, sl] += hidden-actual_h
                raw_alternatives[name] = self.actor.voc_gate_head(features).flatten()
                alternatives[name] = raw_alternatives[name].sigmoid()
                torch.testing.assert_close(alternatives[name][first], actual_p[first], rtol=1e-5, atol=3e-7)
            if feature_freezes:
                features = self.first_features.clone()
                features[:, sl] = self.features[:, sl]
                raw_alternatives['frozen_other'] = self.actor.voc_gate_head(features).flatten()
                alternatives['frozen_other'] = raw_alternatives['frozen_other'].sigmoid()
                # With a linear gate and no final MLP, component log-odds
                # changes must sum to the actual change exactly up to roundoff.
                combined = (raw_alternatives['frozen_hidden']
                            + raw_alternatives['frozen_other']-self.first_log_odds)
                torch.testing.assert_close(combined[optional], raw_actual[optional],
                                           rtol=2e-5, atol=2e-5)
            for i in optional.nonzero().flatten().tolist():
                rows.append(dict(seed=seed, stream=i, decision=self.decision, count=int(counts[i]),
                    actual=float(actual_p[i]), first_probability=float(self.first_p[i]),
                    actual_log_odds=float(raw_actual[i]), first_log_odds=float(self.first_log_odds[i]),
                    **{name: float(value[i]) for name, value in alternatives.items()},
                    **{name+'_log_odds': float(value[i]) for name, value in raw_alternatives.items()}))

        def close(self):
            for hook in self.hooks:
                hook.remove()
            super().close()

    evaluation.BatchRuntime = ObservedRuntime
    try:
        with torch.no_grad():
            allocations = evaluation.allocation(flags, actor, model, seed=seed, batch=8,
                decisions=decisions, warmup=warmup, device='cuda:0')
    finally:
        evaluation.BatchRuntime = original
    return allocations, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditions', nargs='+', default=['control', 'gru10', 'burn8'])
    parser.add_argument('--seed', type=int, default=8301)
    parser.add_argument('--warmup', type=int, default=128)
    parser.add_argument('--decisions', type=int, default=1536)
    parser.add_argument('--feature-freezes', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if '2080 Ti' not in torch.cuda.get_device_name(0):
        raise RuntimeError('evaluation requires RTX 2080 Ti')
    campaign = args.campaign.resolve()
    source = campaign/'evaluation_source_v2/research_script/diagnostics/evaluate_adaptive_planning_pilot.py'
    spec = importlib.util.spec_from_file_location('frozen_recurrence_evaluator', source)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    args.output.mkdir(parents=True, exist_ok=False)
    for condition in args.conditions:
        start = time.monotonic()
        audit = json.loads((campaign/f'checkpoint_completion_{condition}.json').read_text())
        paths = [Path(audit['checkpoint'][kind]['path']) for kind in ('actor', 'model')]
        flags, actor, model, binding = evaluation.load_pair(*paths)
        flags = copy.deepcopy(flags)
        flags.voc_train_epsilon, flags.voc_eval_stochastic = 0., True
        allocations, rows = observe(evaluation, flags, actor, model, seed=args.seed,
            decisions=args.decisions, warmup=args.warmup, feature_freezes=args.feature_freezes)
        reference_path = campaign/('allocation_250k_burn8' if condition == 'burn8'
                                   else 'allocation_250k_long')/f'{condition}_allocation.json'
        reference = reference_path.read_bytes()
        overlap = exact_overlap(allocations, json.loads(reference))
        groups = defaultdict(list)
        for row in rows:
            groups[(row['seed'], row['stream'], row['decision'])].append(row['count'])
        for allocation in allocations:
            key = (allocation['seed'], allocation['stream'], allocation['decision'])
            expected = list(range(flags.min_search_steps, min(allocation['length']+1, flags.max_search_steps)))
            if groups.pop(key, []) != expected:
                raise ValueError('optional query coverage differs from realized allocation')
        if groups:
            raise ValueError('query records outside allocation range')
        with gzip.open(args.output/(condition+'_queries.json.gz'), 'wt') as stream:
            json.dump(rows, stream, allow_nan=False)
        result = dict(status='COMPLETE', condition=condition, checkpoint=binding,
            decisions=len(allocations), queries=len(rows), seed=args.seed, streams=8,
            observed_decision_range=[args.warmup, args.warmup+args.decisions-1],
            by_count=summarize(rows), seconds=time.monotonic()-start,
            reference_overlap=dict(path=str(reference_path), matched_rows=overlap,
                sha256=hashlib.sha256(reference).hexdigest()),
            validation='EXACT_ALLOCATION_OVERLAP_AND_PRODUCTION_READOUT_RECONSTRUCTION_PASS',
            feature_freezes=FEATURE_GROUPS if args.feature_freezes else {},
            interpretation=__doc__,
            limitations=['Alternatives hold other actor features at observed values.',
                'Repeated-token sensitivity includes recurrence even with unchanged input; it does not identify a harmful attractor.',
                'Freezing selected fields can make token combinations inconsistent with a valid search tree; this tests readout sensitivity, not a realizable planning policy.',
                'Risk sets select surviving decisions. No utility or causal policy effect is measured.'],
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        evaluation.atomic_json(args.output/(condition+'_summary.json'), result)
        print(json.dumps({key: result[key] for key in ('condition', 'decisions', 'queries', 'seconds', 'validation')}), flush=True)
    evaluation.atomic_json(args.output/'completion.json', dict(status='COMPLETE',
        conditions=args.conditions, decisions_per_condition=args.decisions*8, seed=args.seed))


if __name__ == '__main__':
    main()
