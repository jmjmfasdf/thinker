"""Short CPU-only frozen-weight recurrent Jacobian probe on real planner tokens.

This is NOT replay of original training batches or a game-score evaluation.
Fixed input tokens isolate the direct hidden-state recurrence path. Their
dependence on earlier actions or weights is intentionally not differentiated.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'thinker'))
sys.path.insert(0, str(ROOT/'research_script/voc_diagnostics'))
from reward_learning_eval import BatchRuntime
from reward_probe import util, preserve_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calls', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261113)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ab, mb = a.actor.read_bytes(), a.model.read_bytes()
    ac = torch.load(io.BytesIO(ab), map_location='cpu', weights_only=False)
    mc = torch.load(io.BytesIO(mb), map_location='cpu', weights_only=False)
    flags = argparse.Namespace(**ac['flags'])
    flags.voc_gate_policy_schema_version = ac['voc_gate_policy_schema_version']
    rt = BatchRuntime(flags, ac, mc, a.seed, a.batch_size, 'cpu')
    assert len(rt.actor.tree_rep_encoder.cells) == 1 and flags.last_layer_n == 0
    cell = rt.actor.tree_rep_encoder.cells[0]
    events = []
    def capture(module, args, kwargs):
        events.append(dict(token=args[0][0].clone(), hidden=kwargs['state'][0].clone(),
                           update=kwargs['update_mask'][0].clone(), reset=kwargs['reset_mask'][0].clone()))
    hook = rt.actor.tree_rep_encoder.register_forward_pre_hook(capture, with_kwargs=True)
    start = time.monotonic()
    real_steps = np.zeros(a.batch_size, dtype=int)
    try:
        state, info = rt.env.reset(rt.model)
        env = util.init_env_out(state, info, rt.flags, rt.actor.dim_actions, rt.actor.tuple_action)
        core = rt.actor.initial_state(batch_size=a.batch_size, device=rt.device)
        with torch.no_grad():
            for _ in range(a.calls):
                out, core = rt.actor(env, core, compute_loss=False, greedy=False)
                state, reward, done, truncated, info = rt.env.step(out.action, rt.model)
                real_steps += info['real_transition'].cpu().numpy().astype(int)
                env = preserve_tokens(env, util.create_env_out(out.action, state, reward, done, truncated, info, rt.flags), info)
        hook.remove()
        torch.save(events, a.output/'tokens_and_states.pt')
        offset = rt.actor.dim_rep_actions+2+rt.actor.num_input_rewards
        hidden_size = cell.hidden_size
        weights = rt.actor.voc_head.weight.detach()[:, offset:offset+hidden_size]
        directions = {'delta_Q': weights[0]-weights[1], 'Q_continue': weights[0], 'Q_stop': weights[1]}
        # A one-layer GRU uses PyTorch's reset/update/candidate order.
        # Save tensors rather than touching the trained network/optimizer.
        gate_rows, temporal = [], []
        for stream in range(a.batch_size):
            valid = [e for e in events if bool(e['update'][stream])]
            for i, event in enumerate(valid):
                x = event['token'][stream:stream+1]
                h = event['hidden'][stream:stream+1]
                if bool(event['reset'][stream]):
                    h = torch.zeros_like(h)
                gi, gh = F.linear(x, cell.weight_ih, cell.bias_ih), F.linear(h, cell.weight_hh, cell.bias_hh)
                ir, iz, inn = gi.chunk(3, dim=-1)
                hr, hz, hn = gh.chunk(3, dim=-1)
                r, z = torch.sigmoid(ir+hr), torch.sigmoid(iz+hz)
                n = torch.tanh(inn+r*hn)
                independent = (1-z)*n+z*h
                torch.testing.assert_close(independent, cell(x, h), rtol=1e-5, atol=1e-6)
                gate_rows.append(dict(stream=stream, token=i,
                    reset_saturated_fraction=float(((r < .01) | (r > .99)).float().mean()),
                    update_saturated_fraction=float(((z < .01) | (z > .99)).float().mean()),
                    update_near_zero_fraction=float((z < .01).float().mean()),
                    update_near_one_fraction=float((z > .99).float().mean()),
                    candidate_saturated_fraction=float((n.abs() > .99).float().mean()),
                    update_gate_mean=float(z.mean())))
            # Common endpoints for all lags, excluding true episode boundaries.
            for end in range(128, len(valid)+1, 16):
                for lag in [1, 4, 8, 16, 30, 64, 128]:
                    segment = valid[end-lag:end]
                    if any(bool(e['reset'][stream]) for e in segment):
                        continue
                    h0 = segment[0]['hidden'][stream:stream+1].clone().requires_grad_(True)
                    h = h0
                    for e in segment:
                        h = cell(e['token'][stream:stream+1], h)
                    # Endpoints must match collection, validating skipped WAITs.
                    if end < len(valid) and not bool(valid[end]['reset'][stream]):
                        torch.testing.assert_close(h, valid[end]['hidden'][stream:stream+1], rtol=1e-5, atol=1e-6)
                    for name, direction in directions.items():
                        grad, = torch.autograd.grad((h*direction).sum(), h0, retain_graph=True)
                        temporal.append(dict(stream=stream, endpoint=end, lag=lag, readout=name,
                            gradient_norm=float(grad.double().norm()),
                            direct_readout_norm=float(direction.double().norm()),
                            relative_gradient_norm=float(grad.double().norm()/direction.double().norm())))
        gates, temp = pd.DataFrame(gate_rows), pd.DataFrame(temporal)
        gates.to_csv(a.output/'gate_activations.csv', index=False)
        temp.to_csv(a.output/'temporal_gradients.csv', index=False)
        summary = temp.groupby(['readout', 'lag']).relative_gradient_norm.agg(['count', 'median', 'min', 'max'])
        summary.to_csv(a.output/'temporal_summary.csv')
        metadata = dict(actor_path=str(a.actor), model_path=str(a.model),
            actor_sha256=hashlib.sha256(ab).hexdigest(), model_sha256=hashlib.sha256(mb).hexdigest(),
            actor_real_step=int(ac['real_step']), model_real_step=int(mc['real_step']),
            calls=a.calls, seed=a.seed, batch_size=a.batch_size, real_actions_per_stream=real_steps.tolist(),
            elapsed_seconds=time.monotonic()-start, valid_tokens=len(gates),
            gate_means=gates.drop(columns=['stream', 'token']).mean().to_dict(),
            definition='||d readout(h_end)/d h_start|| / ||d readout(h_end)/d h_end||; lag counts valid GRU tokens, not real actions.',
            interpretation='Local direct-recurrence sensitivity on short frozen-checkpoint trajectories, not historical training gradients; online VOC readout.',
            assertions='Independent gate arithmetic and recurrent endpoint reconstruction match production GRUCell.',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (a.output/'manifest.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2), flush=True)
        print(summary.to_string(), flush=True)
    finally:
        hook.remove()
        rt.close()


if __name__ == '__main__':
    main()
