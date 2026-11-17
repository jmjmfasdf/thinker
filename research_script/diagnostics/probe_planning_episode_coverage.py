"""Extend fixed-policy allocation coverage and record actual episode boundaries.

No task scores are saved. The complete previously sampled prefix must replay
exactly, so later-scene coverage cannot silently change the original result.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import types

import numpy as np


class EpisodeClock:
    def __init__(self, batch):
        self.batch = batch
        self.reset()

    def reset(self):
        self.decision = 0
        self.episode = np.zeros(self.batch, dtype=int)
        self.age = np.zeros(self.batch, dtype=int)
        self.records = {}

    def step(self, real, terminal, truncated):
        real = np.asarray(real, dtype=bool).reshape(self.batch)
        if not real.any():
            return
        if not real.all():
            raise ValueError('episode coverage expects the synchronous real barrier')
        ended = (np.asarray(terminal, dtype=bool).reshape(self.batch)
                 | np.asarray(truncated, dtype=bool).reshape(self.batch))
        for stream in range(self.batch):
            self.records[(stream, self.decision)] = dict(
                episode=int(self.episode[stream]), episode_step=int(self.age[stream]),
                episode_ended=bool(ended[stream]))
        self.episode += ended
        self.age = np.where(ended, 0, self.age+1)
        self.decision += 1


def allocation_with_episode_clock(original, flags, actor, model, **kwargs):
    runtime_class = original.__globals__['BatchRuntime']
    clock = EpisodeClock(kwargs['batch'])

    class ObservedEnvironment:
        def __init__(self, env):
            self.wrapped = env

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def reset(self, *args, **kw):
            clock.reset()
            return self.wrapped.reset(*args, **kw)

        def step(self, *args, **kw):
            result = self.wrapped.step(*args, **kw)
            _, _, _, truncated, info = result
            clock.step(info['real_transition'].detach().cpu().numpy(),
                       info['real_done'].detach().cpu().numpy(),
                       truncated.detach().cpu().numpy())
            return result

    class ObservedRuntime(runtime_class):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.env = ObservedEnvironment(self.env)

    wrapped = types.FunctionType(original.__code__,
        dict(original.__globals__, BatchRuntime=ObservedRuntime), original.__name__,
        original.__defaults__, original.__closure__)
    wrapped.__kwdefaults__ = original.__kwdefaults__
    rows = wrapped(flags, actor, model, **kwargs)
    if clock.decision != kwargs['warmup']+kwargs['decisions']:
        raise ValueError('episode clock differs from allocation real-decision clock')
    return [dict(row, **clock.records[(row['stream'], row['decision'])]) for row in rows]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', default='gru10')
    parser.add_argument('--decisions', type=int, default=8192)
    args = parser.parse_args()
    import torch
    campaign = args.campaign.resolve()
    reference = json.loads(args.reference.read_text())
    if args.decisions < 1536:
        raise ValueError('extended coverage must include the original1536 decisions')
    if reference['status'] != 'COMPLETE' or reference['evaluation_epsilon'] != 0:
        raise ValueError('a complete epsilon-zero reference is required')
    evaluation = module(campaign/'evaluation_source_v2/research_script/diagnostics/evaluate_adaptive_planning_pilot.py', 'bound_episode_evaluation')
    midpoint = module(campaign/'allocation_midpoint.py', 'bound_episode_summary')
    original_rows = json.loads((args.reference.parent/f'{args.condition}_allocation.json').read_text())
    if len(original_rows) != 24576 or reference['decisions'] != 24576:
        raise ValueError('unexpected reference coverage')
    args.output.mkdir(parents=True, exist_ok=False)
    gpu = torch.cuda.get_device_name(0)
    if '2080 Ti' not in gpu:
        raise ValueError('allocation evaluation requires RTX2080Ti')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    flags, actor, model, binding = evaluation.load_pair(
        Path(reference['checkpoint']['actor']['path']), Path(reference['checkpoint']['model']['path']))
    if binding != reference['checkpoint']:
        raise ValueError('checkpoint differs from completed reference')
    flags = copy.deepcopy(flags)
    flags.voc_train_epsilon, flags.voc_eval_stochastic = 0., True
    extra_keys = {'episode', 'episode_step', 'episode_ended'}

    def original_fields(rows):
        return [{k:v for k,v in row.items() if k not in extra_keys} for row in rows]

    smoke = allocation_with_episode_clock(evaluation.allocation, flags, actor, model,
        seed=8301, batch=8, decisions=16, warmup=128, device='cuda:0')
    expected = [r for r in original_rows if r['seed']==8301 and r['decision']<144]
    if original_fields(smoke) != expected:
        raise ValueError('episode instrumentation changed the reference prefix')
    evaluation.atomic_json(args.output/'smoke.json', dict(status='EXACT_PREFIX_WITH_EPISODE_CLOCK_PASS',
        reference_rows=len(expected), actual_episode_boundaries=sum(r['episode_ended'] for r in smoke)))
    print('exact-prefix instrumentation smoke passed', flush=True)
    rows, checks = [], []
    for seed in (8301, 8401):
        sample = allocation_with_episode_clock(evaluation.allocation, flags, actor, model,
            seed=seed, batch=8, decisions=args.decisions, warmup=128, device='cuda:0')
        overlap = [r for r in sample if r['decision']<1664]
        expected = [r for r in original_rows if r['seed']==seed]
        if original_fields(overlap) != expected:
            raise ValueError('extended rollout changed the original1536-decision prefix')
        checks.append(dict(seed=seed, exact_overlap_rows=len(overlap)))
        rows.extend(sample)
        evaluation.atomic_json(args.output/f'{args.condition}_allocation.json', rows)
        print(json.dumps(dict(seed=seed, decisions=len(rows), exact_overlap_rows=len(overlap))), flush=True)
    windows = []
    for first in range(128,128+args.decisions,1024):
        last = min(first+1023,127+args.decisions)
        sample = [r for r in rows if first<=r['decision']<=last]
        lengths = np.array([r['length'] for r in sample])
        windows.append(dict(first_decision=first,last_decision=last,decisions=len(sample),
            mean_length=float(lengths.mean()),min_fraction=float((lengths==1).mean()),
            interior_fraction=float(((lengths>1)&(lengths<30)).mean()),cap_fraction=float((lengths==30).mean()),
            episode_boundaries=sum(r['episode_ended'] for r in sample)))
    summary = dict(condition=args.condition,status='COMPLETE',checkpoint=binding,device=gpu,
        training_epsilon=float(actor['flags']['voc_train_epsilon']),evaluation_epsilon=0.,
        scope='Extended episode coverage; whole fixed window reported. No score evaluation or checkpoint selection.',
        observed_decision_range=[128,127+args.decisions],exact_prefix_checks=checks,
        actual_episode_boundaries=sum(r['episode_ended'] for r in rows),
        maximum_episode_age=max(r['episode_step'] for r in rows),by_decision_window=windows,
        **midpoint.summarize(rows,flags.min_search_steps,flags.max_search_steps))
    evaluation.atomic_json(args.output/f'{args.condition}_summary.json',summary)
    evaluation.atomic_json(args.output/'completion.json',dict(status='COMPLETE',conditions=[args.condition],
        decisions_per_condition=len(rows),scope=summary['scope']))
    print(json.dumps({k:summary[k] for k in ['mean_length','min_fraction','interior_fraction','cap_fraction','actual_episode_boundaries']}),flush=True)


if __name__ == '__main__':
    main()
