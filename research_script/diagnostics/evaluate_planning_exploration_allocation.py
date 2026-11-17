"""Diagnose the allocation effect of training exploration at one frozen checkpoint.

The ordinary epsilon-zero efficacy results stay unchanged. This diagnostic
uses the training epsilon throughout the prefix and sampled episode segment,
so its resulting state visitation and recurrent history may also differ.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import types

import torch


def allocation_with_epsilon(original, flags, actor, model, *, epsilon, **kwargs):
    if not 0 <= epsilon <= 1:
        raise ValueError('invalid exploration probability')
    runtime_class = original.__globals__['BatchRuntime']
    checks = dict(optional_rows=0, maximum_probability_error=0., epsilon=epsilon)

    class CheckedRuntime(runtime_class):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            if self.actor.voc_gate_temperature != 1 or self.actor.voc_gate_epsilon_greedy_execution:
                raise ValueError('diagnostic requires the ordinary stochastic soft gate')
            self.gate_log_odds = None
            self.handles = [
                self.actor.register_forward_pre_hook(self.before_actor),
                self.actor.voc_gate_head.register_forward_hook(self.capture_gate),
                self.actor.register_forward_hook(self.after_actor),
            ]

        def before_actor(self, module, args):
            # The bound allocation routine disables training exploration.
            # Override only its gate policy switch, before each forward call.
            self.actor.train_actor_enabled = True
            self.actor.voc_train_epsilon = epsilon
            self.current_input = args[0]
            self.gate_log_odds = None

        def capture_gate(self, module, args, output):
            self.gate_log_odds = output.detach().reshape(-1)

        def after_actor(self, module, args, output):
            env = self.current_input
            legal = env.legal_control_mask[0].bool()
            optional = (env.phase[0] == 0) & legal[:, 2] & legal[:, :2].any(-1)
            if optional.any():
                if self.gate_log_odds is None:
                    raise ValueError('missing raw gate readout')
                expected = (1-epsilon)*self.gate_log_odds.sigmoid()+epsilon/2
                actual = output[0].search_control_logits[-1].softmax(-1)[:, :2].sum(-1)
                torch.testing.assert_close(actual[optional], expected[optional], rtol=1e-5, atol=3e-7)
                checks['maximum_probability_error'] = max(checks['maximum_probability_error'],
                    float((actual[optional]-expected[optional]).abs().max()))
                checks['optional_rows'] += int(optional.sum())

        def close(self):
            for handle in self.handles:
                handle.remove()
            super().close()

    local_globals = dict(original.__globals__, BatchRuntime=CheckedRuntime)
    wrapped = types.FunctionType(original.__code__, local_globals,
                                 original.__name__, original.__defaults__, original.__closure__)
    wrapped.__kwdefaults__ = original.__kwdefaults__
    rows = wrapped(flags, actor, model, **kwargs)
    if not checks['optional_rows']:
        raise ValueError('diagnostic did not exercise optional gate decisions')
    return rows, checks


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', default='gru10')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    reference = json.loads(args.reference.read_text())
    source = campaign/'evaluation_source_v2/research_script/diagnostics/evaluate_adaptive_planning_pilot.py'
    evaluation = load_module(source, 'bound_exploration_evaluation')
    midpoint = load_module(campaign/'allocation_midpoint.py', 'bound_allocation_summary')
    args.output.mkdir(parents=True, exist_ok=False)
    gpu = torch.cuda.get_device_name(0)
    if '2080 Ti' not in gpu:
        raise ValueError('diagnostic requires RTX2080Ti')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    flags, actor, model, binding = evaluation.load_pair(
        Path(reference['checkpoint']['actor']['path']), Path(reference['checkpoint']['model']['path']))
    if binding != reference['checkpoint'] or reference['evaluation_epsilon'] != 0:
        raise ValueError('not the exact completed epsilon-zero reference checkpoint')
    if reference['decisions'] != 24576 or reference['independent_streams'] != 16:
        raise ValueError('unexpected reference evaluation coverage')
    training_epsilon = float(flags.voc_train_epsilon)
    if training_epsilon != .1:
        raise ValueError('this planned comparison requires training epsilon0.1')
    flags = copy.deepcopy(flags)
    flags.voc_train_epsilon, flags.voc_eval_stochastic = 0., True
    common = dict(batch=8, warmup=128, device='cuda:0')
    original_rows = json.loads((args.reference.parent/f'{args.condition}_allocation.json').read_text())
    expected = [r for r in original_rows if r['seed'] == 8301 and r['decision'] < 144]
    smoke_rows, zero_check = allocation_with_epsilon(evaluation.allocation, flags, actor, model,
        epsilon=0., seed=8301, decisions=16, **common)
    if smoke_rows != expected:
        raise ValueError('epsilon-zero wrapper changed reference decisions or readouts')
    _, train_check = allocation_with_epsilon(evaluation.allocation, flags, actor, model,
        epsilon=training_epsilon, seed=8301, decisions=16, **common)
    evaluation.atomic_json(args.output/'smoke.json', dict(status='EXACT_REFERENCE_AND_GATE_FORMULA_PASS',
        exact_reference_rows=len(expected), epsilon_zero=zero_check, training_epsilon=train_check))
    print('smoke passed: exact epsilon-zero reference and both gate formulas', flush=True)
    rows, checks = [], []
    for seed in (8301, 8401):
        sample, check = allocation_with_epsilon(evaluation.allocation, flags, actor, model,
            epsilon=training_epsilon, seed=seed, decisions=1536, **common)
        rows.extend(sample)
        checks.append(dict(seed=seed, **check))
        evaluation.atomic_json(args.output/f'{args.condition}_allocation.json', rows)
        print(json.dumps(dict(seed=seed, decisions=len(rows))), flush=True)
    summary = dict(condition=args.condition, status='COMPLETE', checkpoint=binding, device=gpu,
        training_epsilon=training_epsilon, evaluation_epsilon=training_epsilon,
        evaluation_role='Training-exploration occupancy diagnostic; excluded from epsilon-zero efficacy comparisons.',
        prefix_epsilon=training_epsilon, formula_checks=checks,
        reference=str(args.reference.resolve()), reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        **midpoint.summarize(rows, flags.min_search_steps, flags.max_search_steps))
    evaluation.atomic_json(args.output/f'{args.condition}_summary.json', summary)
    evaluation.atomic_json(args.output/'completion.json', dict(status='COMPLETE',
        decisions_per_condition=len(rows), conditions=[args.condition], epsilon=training_epsilon,
        scope='Allocation only. Changed visitation is part of the policy intervention; no paired utility or game scoring.'))
    print(json.dumps({k:summary[k] for k in ('mean_length','min_fraction','interior_fraction','cap_fraction')}), flush=True)


if __name__ == '__main__':
    main()
