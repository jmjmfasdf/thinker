"""Compare pre/post-failure actor encoders on identical human-replay latents.

The healthy checkpoint controls the replay; the final encoder is only a
read-only observer. This localizes representation amplification, not which
training loss caused it. No full-game reward evaluation is performed.
"""
import argparse
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--actor', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--comparison-actor', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    source = args.campaign.resolve()/'evaluation_source_v2/research_script/diagnostics/evaluate_adaptive_planning_pilot.py'
    spec = importlib.util.spec_from_file_location('encoder_evaluation', source)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    torch = evaluation.torch
    from thinker.bc_loader import FrameStackedBehavioralDataLoader
    from thinker.dynamic_imitation import DynamicImitationRunner
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if '2080 Ti' not in torch.cuda.get_device_name(0):
        raise ValueError('diagnostic requires RTX2080Ti')
    flags, actor, model, binding = evaluation.load_pair(args.actor, args.model)
    raw = args.comparison_actor.read_bytes()
    comparison = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    for key in ('xpid', 'name', 'icopro_burn_in_steps', 'voc_critic_parameterization'):
        if comparison['flags'][key] != actor['flags'][key]:
            raise ValueError('comparison must use the same training lineage')
    flags = copy.deepcopy(flags)
    flags.voc_train_epsilon, flags.voc_eval_stochastic = 0., True
    runtime = evaluation.BatchRuntime(flags, actor, model, 9011, 4, 'cuda:0')
    hooks, rows = [], []
    started = time.monotonic()
    try:
        changed = copy.deepcopy(runtime.actor.h_encoder)
        changed.load_state_dict({k.removeprefix('h_encoder.'): v for k, v in comparison['actor_net_state_dict'].items()
                                 if k.startswith('h_encoder.')}, strict=True)
        changed.eval().requires_grad_(False)
        def observe(module, inputs, output):
            alternative = changed(inputs[0])
            values = [inputs[0], output, alternative]
            if not all(torch.isfinite(x).all() for x in values):
                raise ValueError('nonfinite encoder input/output')
            record = dict(call=len(rows), batch=int(inputs[0].shape[0]))
            for name, value in zip(('input', 'before', 'after'), values):
                record[name+'_absmax'] = float(value.abs().max())
                record[name+'_rms'] = float(value.double().square().mean().sqrt())
            record['rms_ratio'] = record['after_rms']/max(record['before_rms'], 1e-12)
            rows.append(record)
        hooks.append(runtime.actor.h_encoder.register_forward_hook(observe))
        loader = FrameStackedBehavioralDataLoader(base_path=flags.icopro_data_path,
            subjects=[1], game_id=flags.icopro_game_id, sessions=[1,2,3,4],
            num_actions=runtime.actor.num_actions, scored_length=4,
            burn_in_steps=16, sampling_context=16, frame_stack_n=flags.frame_stack_n,
            target_size=(84,84), grayscale=False, normalize=False, seed=9011)
        batch = loader.get_sequence_batch(batch_size=4, sequence_length=4)
        runner = DynamicImitationRunner(runtime.actor, runtime.model, flags, device=runtime.device,
                                        noop_action_index=0)
        before_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
            for v in runtime.actor.state_dict().values())).hexdigest()
        with torch.no_grad():
            result = runner.rollout(batch, tree_carry=True, training=False)
        after_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
            for v in runtime.actor.state_dict().values())).hexdigest()
        if before_hash != after_hash or result.count != 16:
            raise ValueError('observer changed weights or scored-target count')
        import numpy as np
        report = dict(status='IDENTICAL_INPUT_ENCODER_COMPARISON_PASS', checkpoint=binding,
            comparison_actor=dict(path=str(args.comparison_actor.resolve()), real_step=comparison['real_step'],
                                  sha256=hashlib.sha256(raw).hexdigest()),
            encoder_calls=len(rows), scored_targets=result.count,
            rms_ratio_median=float(np.median([r['rms_ratio'] for r in rows])),
            maximum_old_output=max(r['before_absmax'] for r in rows),
            maximum_new_output=max(r['after_absmax'] for r in rows),
            maximum_input=max(r['input_absmax'] for r in rows),
            seconds=time.monotonic()-started, rows=rows,
            scope=__doc__, source_files=[str(v) for v in batch['source_file']],
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        args.output.mkdir(parents=True, exist_ok=False)
        evaluation.atomic_json(args.output/'evidence.json', report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('rows','source_files','checkpoint')}), flush=True)
    finally:
        for handle in hooks:
            handle.remove()
        runtime.close()


if __name__ == '__main__':
    main()
