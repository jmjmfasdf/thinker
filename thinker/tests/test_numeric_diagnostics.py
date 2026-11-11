from types import SimpleNamespace

import pytest
import torch

from thinker.numeric_diagnostics import ImitationNumericTrace, save_actor_failure


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.))

    def forward(self, env, state):
        return SimpleNamespace(pri_param=env.hs * self.weight), state


def setup(tmp_path):
    actor = Actor()
    learner = SimpleNamespace(actor_net=actor, bc_model_net=Actor(),
                              optimizer=torch.optim.SGD(actor.parameters(), lr=.1),
                              flags=SimpleNamespace(ckpdir=str(tmp_path)), real_step=42)
    env = SimpleNamespace(hs=torch.tensor([[[1e5, 1.]]]), phase=torch.tensor([[1]]),
                          search_steps=torch.tensor([[29]]))
    return learner, env


def test_trace_preserves_output_rng_and_gradient_and_captures_large_batch(tmp_path):
    learner, env = setup(tmp_path)
    before_rng = torch.get_rng_state().clone()
    with ImitationNumericTrace(learner, {"source_index": 123}) as trace:
        out, _ = learner.actor_net(env, ())
        result = SimpleNamespace(loss=out.pri_param.sum(), logits=out.pri_param,
                                 targets=torch.tensor([0]), normalized_ce=torch.tensor(1.),
                                 margin_loss=torch.tensor(0.), pvp_loss=torch.tensor(0.))
        trace.finish(result)
    result.loss.backward()
    assert learner.actor_net.weight.grad.item() == pytest.approx(100001.)
    torch.testing.assert_close(torch.get_rng_state(), before_rng)
    assert not learner.actor_net._forward_hooks
    saved = torch.load(tmp_path / 'numeric_diagnostics/bc-large-42.pt', weights_only=False)
    assert saved['batch']['source_index'] == 123
    assert saved['maxima']['actor.logits'] == 2e5
    torch.testing.assert_close(saved['actor_state_dict']['weight'], torch.tensor(2.))


def test_diagnostic_stop_records_state_and_removes_hooks(tmp_path, monkeypatch):
    learner, env = setup(tmp_path)
    monkeypatch.setenv('DI72_NUMERIC_STOP_AT_LOGIT', '10000')
    with pytest.raises(FloatingPointError, match='diagnostic stop'):
        with ImitationNumericTrace(learner, {}) as trace:
            out, _ = learner.actor_net(env, ())
            trace.finish(SimpleNamespace(loss=out.pri_param.sum(), logits=out.pri_param,
                                         targets=torch.tensor([0]), normalized_ce=torch.tensor(1.),
                                         margin_loss=torch.tensor(0.), pvp_loss=torch.tensor(0.)))
    assert (tmp_path / 'numeric_diagnostics/bc-failure-42.pt').is_file()
    assert not learner.actor_net._forward_hooks
    assert learner.actor_net.weight.grad is None


def test_online_failure_keeps_named_loss_components(tmp_path):
    learner, _ = setup(tmp_path)
    save_actor_failure(learner, (torch.tensor([1.]), ()), ValueError('bad total'),
                       losses={'reg_loss': torch.tensor(float('nan')), 'ce_loss': torch.tensor(2.)})
    saved = torch.load(tmp_path / 'numeric_diagnostics/actor-failure-42.pt', weights_only=False)
    assert torch.isnan(saved['losses']['reg_loss'])
    assert saved['losses']['ce_loss'].item() == 2.
