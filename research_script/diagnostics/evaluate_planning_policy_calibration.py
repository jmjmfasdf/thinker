"""Measure VOC under its training continuation policy at fixed evaluation states.

The entire prefix, including the intervention entry, uses zero exploration.
Only AFTER the entry forward pass does the actor use checkpoint training
epsilon. STOP/CONTINUE still share the exact state and prediction. This is
critic calibration under a different continuation policy; the original
zero-exploration efficacy evaluation remains separate and unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def calibration_arm(original, flags, actor, model, *, future_epsilon=None, **kwargs):
    """Wrap the frozen rollout locally; never change checkpoint parameters."""
    runtime_class = original.__globals__['BatchRuntime']
    epsilon = float(actor['flags']['voc_train_epsilon'] if future_epsilon is None else future_epsilon)
    if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError('invalid future exploration probability')
    if float(flags.voc_train_epsilon) != 0:
        raise ValueError('the intervention prefix must have zero exploration')
    targets = np.asarray(kwargs['targets'])
    if len(set(targets.tolist())) != 1:
        raise ValueError('calibration requires homogeneous intervention counts')
    prefix = kwargs['prefix']
    checks = dict(switched=False, before_entry_optional_rows=0, after_entry_optional_rows=0,
                  maximum_probability_error=0., future_epsilon=epsilon,
                  prefix_epsilon=0., entry_epsilon=0.)

    class CalibrationRuntime(runtime_class):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            if self.actor.voc_gate_temperature != 1 or self.actor.voc_gate_epsilon_greedy_execution:
                raise ValueError('calibration requires the scalar soft stochastic gate')
            if not self.actor.train_actor_enabled or self.actor.voc_train_epsilon != 0:
                raise ValueError('unexpected initial execution policy')
            self.real_index = -1
            self.gate_log_odds = None
            self.calibration_hooks = [
                self.actor.register_forward_pre_hook(self.before_actor),
                self.actor.voc_gate_head.register_forward_hook(self.capture_gate),
                self.actor.register_forward_hook(self.after_actor),
            ]

        def before_actor(self, module, args):
            env = args[0]
            if self.real_index < 0:
                self.real_index = 0
            elif bool(env.real_transition[0].all()):
                self.real_index += 1
            self.current_input = env
            self.gate_log_odds = None

        def capture_gate(self, module, args, output):
            self.gate_log_odds = output.detach().reshape(-1)

        def after_actor(self, module, args, output):
            import torch
            env = self.current_input
            phase, counts = env.phase[0], env.search_steps[0]
            legal = env.legal_control_mask[0].bool()
            optional = (phase == 0) & legal[:, 2] & legal[:, :2].any(-1)
            if optional.any():
                if self.gate_log_odds is None:
                    raise ValueError('missing raw gate output')
                active_epsilon = epsilon if checks['switched'] else 0.
                expected = (1-active_epsilon)*self.gate_log_odds.sigmoid()+active_epsilon/2
                actual = output[0].search_control_logits[-1].softmax(-1)[:, :2].sum(-1)
                torch.testing.assert_close(actual[optional], expected[optional], rtol=1e-5, atol=3e-7)
                error = float((actual[optional]-expected[optional]).abs().max())
                checks['maximum_probability_error'] = max(checks['maximum_probability_error'], error)
                key = 'after_entry_optional_rows' if checks['switched'] else 'before_entry_optional_rows'
                checks[key] += int(optional.sum())
            at_entry = ((phase == 0) & (counts == int(targets[0])))
            if not checks['switched'] and self.real_index == prefix and at_entry.any():
                if not at_entry.all() or not optional.all():
                    raise ValueError('entry must be simultaneous and optional for the whole batch')
                # The output and recurrent-state update just computed still
                # used epsilon0. Only subsequent actor calls use exploration.
                self.actor.voc_train_epsilon = epsilon
                checks['switched'] = True
            if not checks['switched'] and self.real_index > prefix:
                raise ValueError('missed calibration intervention')

        def close(self):
            for handle in self.calibration_hooks:
                handle.remove()
            super().close()

    original.__globals__['BatchRuntime'] = CalibrationRuntime
    try:
        result = original(flags, actor, model, **kwargs)
    finally:
        original.__globals__['BatchRuntime'] = runtime_class
    if not checks['switched']:
        raise ValueError('continuation policy was never activated')
    result['continuation_policy_check'] = checks
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args, remaining = parser.parse_known_args()
    campaign = args.campaign.resolve()
    source = campaign/'evaluation_source_v2/research_script/diagnostics/evaluate_adaptive_planning_pilot.py'
    spec = importlib.util.spec_from_file_location('calibration_evaluation', source)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    original = evaluation.arm_rollout
    torch = evaluation.torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if '2080 Ti' not in torch.cuda.get_device_name(0):
        raise ValueError('utility evaluation requires RTX2080Ti')
    output = Path(remaining[remaining.index('--output')+1])
    if args.smoke:
        actor_path = Path(remaining[remaining.index('--actor')+1])
        model_path = Path(remaining[remaining.index('--model')+1])
        flags, actor, model, binding = evaluation.load_pair(actor_path, model_path)
        flags = copy.deepcopy(flags)
        flags.voc_train_epsilon, flags.voc_eval_stochastic = 0., True
        kw = dict(seed=8301, repetition=0, prefix=4, targets=np.array([1, 1]), horizon=32, device='cuda:0')
        reports = []
        for arm in ('stop', 'continue'):
            reference = original(flags, actor, model, arm=arm, **kw)
            zero = calibration_arm(original, flags, actor, model, future_epsilon=0., arm=arm, **kw)
            matched = calibration_arm(original, flags, actor, model, arm=arm, **kw)
            evaluation.verify(reference, zero)
            evaluation.verify(reference, matched)
            if reference['returns'] != zero['returns'] or reference['tail_bounds'] != zero['tail_bounds']:
                raise ValueError('zero-epsilon wrapper changed returns')
            if matched['continuation_policy_check']['after_entry_optional_rows'] == 0:
                raise ValueError('smoke did not exercise the changed continuation policy')
            reports.append(dict(arm=arm, check=matched['continuation_policy_check']))
        output.mkdir(parents=True, exist_ok=False)
        evaluation.atomic_json(output/'smoke.json', dict(status='FIXED_ENTRY_AND_CONTINUATION_POLICY_SMOKE_PASS',
            checkpoint=binding, checks=reports, zero_epsilon_returns='EXACT_MATCH',
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        print(json.dumps(reports), flush=True)
    else:
        used_epsilon = set()
        def wrapped(flags, actor, model, **kwargs):
            result = calibration_arm(original, flags, actor, model, **kwargs)
            used_epsilon.add(result['continuation_policy_check']['future_epsilon'])
            return result
        evaluation.arm_rollout = wrapped
        sys.argv = [str(source), *remaining]
        evaluation.main()
        if len(used_epsilon) != 1:
            raise ValueError('inconsistent continuation policy across arms')
        summary = json.loads((output/'summary.json').read_text())
        summary.update(evaluation_train_epsilon=None, allocation_evaluation_epsilon=0.,
            prefix_epsilon=0., entry_epsilon=0., continuation_train_epsilon=next(iter(used_epsilon)),
            evaluation_role='Training-policy critic calibration; separate from zero-exploration efficacy.')
        evaluation.atomic_json(output/'summary.json', summary)
        evaluation.atomic_json(output/'continuation_policy.json', dict(status='COMPLETE',
            prefix_epsilon=0., entry_epsilon=0., future_epsilon=next(iter(used_epsilon)),
            role='Training-policy critic calibration. Not the zero-exploration efficacy result.',
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
