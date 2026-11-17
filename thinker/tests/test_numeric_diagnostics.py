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


def test_stop_below_default_capture_threshold_saves_exact_actor_input(tmp_path, monkeypatch):
    learner, env = setup(tmp_path)
    env.hs = torch.tensor([[[600., 1.]]])
    monkeypatch.setenv('DI72_NUMERIC_STOP_AT_LOGIT', '1000')
    state = (torch.tensor([[.25]]),)
    with pytest.raises(FloatingPointError, match='diagnostic stop'):
        with ImitationNumericTrace(learner, {}) as trace:
            out, _ = learner.actor_net(env, state)
            trace.finish(SimpleNamespace(loss=out.pri_param.sum(), logits=out.pri_param,
                targets=torch.tensor([0]), normalized_ce=torch.tensor(1.),
                margin_loss=torch.tensor(0.), pvp_loss=torch.tensor(0.)))
    saved = torch.load(tmp_path/'numeric_diagnostics/bc-failure-42.pt', weights_only=False)
    assert saved['trigger_event'] is not None
    torch.testing.assert_close(saved['trigger_event']['env_out'].hs, env.hs)
    torch.testing.assert_close(saved['trigger_event']['actor_state'][0], state[0])
    assert saved['maxima']['actor.logits'] == 1200.


def test_other_visual_paths_and_final_mlp_are_observed_without_changing_gradients(tmp_path):
    class BranchActor(Actor):
        def __init__(self):
            super().__init__()
            self.x_encoder_pre = torch.nn.Linear(2,2,bias=False)
            self.real_state_encoder = torch.nn.Linear(2,2,bias=False)
            self.final_mlp = torch.nn.Linear(2,2,bias=False)

        def forward(self, env, state):
            value = self.final_mlp(self.x_encoder_pre(env.hs)+self.real_state_encoder(env.hs))
            return SimpleNamespace(pri_param=value*self.weight), state

    learner, env = setup(tmp_path)
    learner.actor_net = BranchActor()
    learner.optimizer = torch.optim.SGD(learner.actor_net.parameters(),lr=.1)
    env.hs = torch.tensor([[[1.,2.]]])
    actor=learner.actor_net
    expected,_=actor(env,())
    expected.pri_param.sum().backward()
    grads={name:p.grad.clone() for name,p in actor.named_parameters()}
    actor.zero_grad(set_to_none=True)
    rng=torch.get_rng_state().clone()
    with ImitationNumericTrace(learner,{}) as trace:
        actual,_=actor(env,())
    actual.pri_param.sum().backward()
    torch.testing.assert_close(actual.pri_param,expected.pri_param)
    torch.testing.assert_close(torch.get_rng_state(),rng)
    for name,p in actor.named_parameters():
        torch.testing.assert_close(p.grad,grads[name])
    for name in ('x_encoder_pre','real_state_encoder','final_mlp'):
        assert trace.maxima['actor.'+name]>0
        assert not getattr(actor,name)._forward_hooks
