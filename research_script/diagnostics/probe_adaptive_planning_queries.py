"""Observe every optional search choice without changing the frozen evaluator.

The observer runs the campaign's original allocation loop and keyed action
draws. Overlapping allocation records must match existing records exactly.
This describes within-stage control; it does not estimate computation utility.
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


def query_report(queries, allocations, minimum=1, cap=30):
    grouped = defaultdict(list)
    for row in queries:
        grouped[(row['seed'], row['stream'], row['decision'])].append(row)
    stages, risks = [], defaultdict(list)
    keys = set()
    for stage in allocations:
        key = (stage['seed'], stage['stream'], stage['decision'])
        if key in keys:
            raise ValueError('duplicate allocation decision')
        keys.add(key)
        rows = sorted(grouped.pop(key, []), key=lambda r: r['count'])
        length = stage['length']
        if not minimum <= length <= cap:
            raise ValueError('invalid realized length')
        expected = list(range(minimum, min(length+1, cap)))
        if [r['count'] for r in rows] != expected:
            raise ValueError('missing or duplicate optional query')
        if stage['optional_choices'] != len(rows):
            raise ValueError('optional query count differs from allocation')
        p = np.asarray([r['probability'] for r in rows], dtype=float)
        if not np.isfinite(p).all() or np.any((p < -2*np.finfo(np.float32).eps)
                                             | (p > 1+2*np.finfo(np.float32).eps)):
            raise ValueError('invalid probability')
        if p[0] != stage['first_optional_probability']:
            raise ValueError('first probability differs from allocation')
        p = p.clip(0, 1)
        chosen = np.asarray([r['continue'] for r in rows], dtype=bool)
        # The cap is forced. Its STOP is excluded from optional-choice rows.
        expected_chosen = np.ones(len(rows), dtype=bool)
        if length < cap:
            expected_chosen[-1] = False
        if not np.array_equal(chosen, expected_chosen):
            raise ValueError('chosen controls disagree with realized length')
        signs = np.sign(p-.5)
        nonzero = signs[signs != 0]
        flips = int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))
        # Saved allocation uses the immediately preceding sign, including zero.
        exact_flips = int(np.count_nonzero((signs[1:] != signs[:-1])
                                          & (signs[1:] != 0) & (signs[:-1] != 0)))
        if stage['sign_changes'] != exact_flips:
            raise ValueError('sign changes differ from allocation')
        stages.append(dict(seed=key[0], stream=key[1], decision=key[2], length=length,
            optional_choices=len(rows), first_probability=float(p[0]),
            last_probability=float(p[-1]), probability_range=float(np.ptp(p)),
            last_minus_first=float(p[-1]-p[0]), sign_flips_ignoring_exact_ties=flips,
            real_episode=rows[0]['real_episode'], episode_age=rows[0]['episode_age']))
        for row, probability in zip(rows, p):
            risks[row['count']].append((probability, row['continue'],
                                       probability-p[0], row['online_delta']))
    if grouped:
        raise ValueError('query rows outside recorded allocation decisions')
    if not stages:
        raise ValueError('no completed stages')
    by_count = []
    for count, rows in sorted(risks.items()):
        values = np.asarray(rows, dtype=float)
        by_count.append(dict(count=int(count), at_risk_decisions=len(rows),
            observed_continue_fraction=float(values[:, 1].mean()),
            predicted_continue_mean=float(values[:, 0].mean()),
            mean_probability_change_from_first=float(values[:, 2].mean()),
            mean_online_delta=float(values[:, 3].mean())))
    by_length = []
    for name, select in (
        ('minimum', lambda r: r['length'] == minimum),
        ('interior', lambda r: minimum < r['length'] < cap),
        ('cap', lambda r: r['length'] == cap)):
        rows = [r for r in stages if select(r)]
        by_length.append(dict(group=name, decisions=len(rows),
            fraction=len(rows)/len(stages),
            **{key: float(np.mean([r[key] for r in rows])) if rows else None
               for key in ('first_probability', 'last_probability', 'probability_range',
                           'last_minus_first')}))
    return dict(decisions=len(stages), optional_queries=len(queries), by_count=by_count,
        independent_streams=len({(r['seed'], r['stream']) for r in stages}),
        by_realized_length=by_length,
        interpretation='Observed risk sets select decisions that survived previous queries; '
            'length groups condition on outcomes. Within-stage changes describe execution, '
            'not causal scene effects or useful computation. Cap STOP is forced and excluded.'), stages


def exact_overlap(rows, reference):
    saved = {(r['seed'], r['stream'], r['decision']): r for r in reference}
    matched = 0
    for row in rows:
        key = (row['seed'], row['stream'], row['decision'])
        if key in saved:
            if row != saved[key]:
                raise ValueError('observer changed frozen allocation: '+str(key))
            matched += 1
    if not matched:
        raise ValueError('no overlap for observer validation')
    return matched


def observe(evaluation, flags, actor, model, *, seed, batch, decisions, warmup):
    original_runtime = evaluation.BatchRuntime
    torch, util = evaluation.torch, evaluation.util
    queries = []

    class ObservedEnv:
        def __init__(self, env, recorder):
            self.env, self.recorder = env, recorder

        def __getattr__(self, key):
            return getattr(self.env, key)

        def step(self, action, model):
            result = self.env.step(action, model)
            self.recorder.after_step(action, result)
            return result

    class ObservedRuntime(original_runtime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.decision = 0
            self.episode = np.zeros(batch, dtype=int)
            self.age = np.zeros(batch, dtype=int)
            self.observed = []
            self.env = ObservedEnv(self.env, self)
            self.hook = self.actor.register_forward_hook(self.after_actor)

        def after_actor(self, module, args, output):
            env, (out, _) = args[0], output
            self.observed = []
            if self.decision < warmup:
                return
            phase = env.phase[0].cpu().numpy()
            legal = env.legal_control_mask[0].cpu().numpy().astype(bool)
            selected = np.flatnonzero((phase == util.SEARCH_PHASE)
                & legal[:, util.STOP] & legal[:, :2].any(-1))
            if not len(selected):
                return
            counts = env.search_steps[0].cpu().numpy().astype(int)
            probability = out.search_control_logits[-1].softmax(-1)[:, :2].sum(-1).cpu().numpy()
            q = out.voc_q[-1].cpu().numpy()
            tree = util.decode_dynamic_tree_reps(env.tree_reps,
                self.actor.num_actions, enc_type=getattr(flags, 'model_enc_type', 0),
                f_type=getattr(flags, 'model_enc_f_type', 0))
            values = {key: tree[key].cpu().numpy() for key in
                      ('root_qs_mean', 'root_ns', 'rollout_return', 'rollout_depth')}
            for i in selected:
                self.observed.append(dict(seed=seed, stream=int(i), decision=self.decision,
                    count=int(counts[i]), probability=float(probability[i]),
                    online_delta=float(q[i, 0]-q[i, 1]),
                    root_q_mean=values['root_qs_mean'][i].tolist(),
                    root_visits=values['root_ns'][i].tolist(),
                    rollout_return=float(values['rollout_return'][i, 0]),
                    depth=int(np.rint(np.expm1(values['rollout_depth'][i, 0]))),
                    real_episode=int(self.episode[i]), episode_age=int(self.age[i])))

        def after_step(self, action, result):
            _, _, _, _, info = result
            control = action[1].cpu().numpy().reshape(-1)
            accepted = info['accepted_control'].cpu().numpy().reshape(-1)
            for row in self.observed:
                i = row['stream']
                if int(accepted[i]) != int(control[i]):
                    raise ValueError('optional control was not accepted')
                row.update(control=int(control[i]), continue_=bool(control[i] != util.STOP))
                row['continue'] = row.pop('continue_')
                queries.append(row)
            real = info['real_transition'].bool().cpu().numpy().reshape(-1)
            if real.any() and not real.all():
                raise ValueError('asynchronous real barrier')
            if real.all():
                done = info['real_done'].bool().cpu().numpy().reshape(-1)
                self.episode += done
                self.age = np.where(done, 0, self.age+1)
                self.decision += 1

        def close(self):
            self.hook.remove()
            super().close()

    evaluation.BatchRuntime = ObservedRuntime
    try:
        with torch.no_grad():
            rows = evaluation.allocation(flags, actor, model, seed=seed, batch=batch,
                decisions=decisions, warmup=warmup, device='cuda:0')
    finally:
        evaluation.BatchRuntime = original_runtime
    return rows, queries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--conditions', nargs='+', default=['control', 'gru10', 'burn8'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[8301])
    ap.add_argument('--decisions', type=int, default=1536)
    args = ap.parse_args()
    import torch
    if '2080 Ti' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Use RTX 2080 Ti for evaluation')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    campaign = args.campaign.resolve()
    source = campaign/'evaluation_source_v2/research_script/diagnostics/evaluate_adaptive_planning_pilot.py'
    spec = importlib.util.spec_from_file_location('bound_query_evaluation', source)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    manifest = json.loads((campaign/'manifest.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    for condition in args.conditions:
        started = time.monotonic()
        item = next(r for r in manifest['runs'] if r['condition'] == condition)
        run = campaign.parents[3]/'logs/thinker'/item['xpid']
        paths = [min((p for p in run.glob(f'ckp_{kind}.tar_step_*')
                      if 250000 <= int(p.name.rsplit('_', 1)[1]) < 252000),
                     key=lambda p: int(p.name.rsplit('_', 1)[1])) for kind in ('actor', 'model')]
        flags, actor, model, binding = evaluation.load_pair(*paths)
        flags = copy.deepcopy(flags)
        flags.voc_train_epsilon, flags.voc_eval_stochastic = 0., True
        allocations, queries = [], []
        for seed in args.seeds:
            rows, observed = observe(evaluation, flags, actor, model,
                seed=seed, batch=8, decisions=args.decisions, warmup=128)
            allocations.extend(rows)
            queries.extend(observed)
        references = []
        for path in (campaign/'allocation_250k_long'/f'{condition}_allocation.json',
                     campaign/f'evaluation_{condition}'/'allocation.json'):
            if path.exists():
                data = path.read_bytes()
                references.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                    matched_rows=exact_overlap(allocations, json.loads(data))))
        if not references:
            raise ValueError('no reference allocation for noninterference check')
        report, stages = query_report(queries, allocations, flags.min_search_steps, flags.max_search_steps)
        report.update(status='COMPLETE', condition=condition, checkpoint=binding,
            evaluation_epsilon=0., seconds=time.monotonic()-started,
            reference_overlap=references, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            observed_decision_range=[128, 128+args.decisions-1],
            episode_age_definition='Real decisions since environment real_done; life-loss pseudo-terminals are not counted as real episodes.')
        with gzip.open(args.output/(condition+'_queries.json.gz'), 'wt') as stream:
            json.dump(queries, stream, allow_nan=False)
        evaluation.atomic_json(args.output/(condition+'_stages.json'), stages)
        evaluation.atomic_json(args.output/(condition+'_summary.json'), report)
        print(json.dumps(dict(condition=condition, decisions=report['decisions'],
            queries=len(queries), seconds=report['seconds'], validation='EXACT_OVERLAP_PASS')), flush=True)
    evaluation.atomic_json(args.output/'completion.json', dict(status='COMPLETE',
        conditions=args.conditions, seeds=args.seeds, decisions_per_condition=8*args.decisions*len(args.seeds),
        scope='Within-stage allocation diagnostics only; no separate game scoring.'))


if __name__ == '__main__':
    main()
