from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thinker.cenv import cModelWrapper
from thinker.planning_tree import PlanningTreeContext, TorchNode
from test_dynamic_cenv import FakeVectorEnv, FakeModel, TerminalFakeModel, _flags


@pytest.mark.parametrize("reset_mode", [0, 1])
@pytest.mark.parametrize("carry", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
def test_torch_tree_matches_native(reset_mode, carry, terminal):
    wrappers, models = [], []
    for backend in ("cython", "torch"):
        flags = _flags(cap=7, max_depth=4, tree_carry=carry)
        flags.reset_mode = reset_mode
        flags.planning_backend = backend
        model = (TerminalFakeModel if terminal else FakeModel)(5)
        wrappers.append(cModelWrapper(FakeVectorEnv(env_n=3), 3, flags, model))
        models.append(model)
    try:
        states = [w.reset(m) for w, m in zip(wrappers, models)]
        rng = np.random.RandomState(12)
        for _ in range(70):
            np.testing.assert_allclose(states[0][0]["tree_reps"], states[1][0]["tree_reps"], atol=3e-5, rtol=2e-5)
            legal = states[0][-1]["legal_control_mask"].numpy()
            controls = np.array([rng.choice(np.flatnonzero(row)) if row.any() else 0 for row in legal])
            primary = rng.randint(0, 5, size=3)
            states = [w.step((primary, controls), m) for w, m in zip(wrappers, models)]
            for key, value in states[0][-1].items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, states[1][-1][key], atol=3e-5, rtol=2e-5)
    finally:
        for w in wrappers:
            w.close()


def test_tree_gamma_gradient_matches_finite_difference_and_revalues():
    gamma = torch.tensor(.91, dtype=torch.float64, requires_grad=True)
    ctx = PlanningTreeContext(gamma)
    root = TorchNode(ctx, None, 0, 0, 2)
    root.expand(0, 2, False, [])
    a = TorchNode(ctx, root, 0, 0, 2)
    TorchNode(ctx, root, 1, 0, 2)
    root.visit()
    a.expand(1, 3, False, [])
    b = TorchNode(ctx, a, 0, 0, 2)
    TorchNode(ctx, a, 1, 0, 2)
    a.visit()
    b.expand(4, 5, False, [])
    b.visit()
    def objective():
        return root.statistics(True).square().sum()
    analytic = torch.autograd.grad(objective(), gamma)[0]
    eps = 1e-5
    ctx.set_gamma(gamma.detach() + eps)
    plus = objective()
    ctx.set_gamma(gamma.detach() - eps)
    minus = objective()
    torch.testing.assert_close(analytic, (plus-minus)/(2*eps), rtol=1e-7, atol=1e-6)
    assert analytic.abs() > 0


def test_functional_adam_matches_live_without_mutation():
    import copy
    from thinker.planning_meta import functional_adam
    p = torch.nn.Parameter(torch.tensor([1., -2., .3], dtype=torch.float64))
    opt = torch.optim.Adam([p], lr=.003, eps=1e-8)
    p.grad = torch.tensor([.1, -.3, .2], dtype=p.dtype)
    opt.step()
    before, state = p.detach().clone(), copy.deepcopy(opt.state_dict())
    phi = torch.tensor(.7, dtype=p.dtype, requires_grad=True)
    g = phi * torch.tensor([.6, -.1, .3], dtype=p.dtype)
    fast = functional_adam({"p": p.detach().clone()}, {"p": g}, opt, {id(p): "p"})["p"]
    assert torch.equal(before, p)
    assert torch.equal(state["state"][0]["exp_avg"], opt.state[p]["exp_avg"])
    p.grad = g.detach()
    opt.step()
    torch.testing.assert_close(fast, p, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(torch.autograd.grad(fast.sum(), phi)[0])


def test_vtrace_keeps_only_meta_discount_path():
    from thinker.core.vtrace import compute_v_trace
    gamma = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
    values = torch.tensor([[.2], [.4]], dtype=gamma.dtype, requires_grad=True)
    def target(discount, differentiable):
        return compute_v_trace(torch.zeros_like(values), discount.expand_as(values),
            torch.ones_like(values), values, values[-1], -1,
            differentiable_discount=differentiable).vs.sum()
    value = target(gamma, True)
    derivative = torch.autograd.grad(value, (gamma, values), allow_unused=True)
    assert derivative[1] is None
    eps = 1e-5
    numerical = (target(gamma.detach()+eps, False)-target(gamma.detach()-eps, False))/(2*eps)
    torch.testing.assert_close(derivative[0], numerical, rtol=1e-8, atol=1e-8)
    assert not target(gamma, False).requires_grad


def test_raw_value_clipping_has_gamma_derivative():
    gamma = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
    context = PlanningTreeContext(gamma, reward_clip=1.)
    root = TorchNode(context, None, 0, 0, 2)
    root.expand(0., 5., False, [], raw_value=100.)
    value = root.statistics(True)[4]  # A=2: action, reward, terminal, value
    torch.testing.assert_close(value, 1/(1-gamma))
    torch.testing.assert_close(torch.autograd.grad(value, gamma)[0], 1/(1-gamma).square())


def test_paired_score_gradient_matches_exact_enumeration():
    from thinker.planning_meta import paired_bc_surrogate
    phi = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    p = phi.sigmoid()
    costs = (3*phi, 1+phi.square())
    expected = torch.autograd.grad((1-p)*costs[0]+p*costs[1], phi, retain_graph=True)[0]
    estimate = 0.
    for a in (0, 1):
        for b in (0, 1):
            results = [SimpleNamespace(
                meta_nll=torch.stack((phi*0, costs[x])).view(1, 2),
                score_mask=torch.tensor([[False, True]]),
                # A burn-in choice must get credit for its scored suffix.
                meta_search_log_probs=((0, (p if x else 1-p).log().view(1)),),
            ) for x in (a, b)]
            objective, _, _ = paired_bc_surrogate(results)
            gradient = torch.autograd.grad(objective, phi, retain_graph=True)[0]
            estimate += ((p if a else 1-p)*(p if b else 1-p)).detach()*gradient
    torch.testing.assert_close(estimate, expected, rtol=1e-12, atol=1e-12)


def test_higher_order_gru_matches_standard_cell():
    from thinker.core.rnn import MaskedGRU
    torch.manual_seed(72)
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        reference = MaskedGRU(5, 7).to(device).double()
        meta = MaskedGRU(5, 7, higher_order=True).to(device).double()
        meta.load_state_dict(reference.state_dict())
        x = torch.randn(3, 2, 5, device=device, dtype=torch.float64, requires_grad=True)
        actual = meta(x)[0]
        torch.testing.assert_close(actual, reference(x)[0], rtol=1e-12, atol=1e-12)
        first = torch.autograd.grad(actual.square().sum(), x, create_graph=True)[0]
        assert torch.isfinite(torch.autograd.grad(first.sum(), x)[0]).all()


def test_revalue_preserves_emitted_reset_token():
    flags = _flags(cap=7, max_depth=4)
    flags.planning_backend = "torch"
    model = FakeModel(5)
    wrapper = cModelWrapper(FakeVectorEnv(env_n=1), 1, flags, model)
    try:
        wrapper.reset(model)
        wrapper.step((np.array([1]), np.array([0])), model)
        emitted = wrapper.step((np.array([2]), np.array([1])), model)[0]["tree_reps"]
        wrapper.set_planning_gamma(flags.discounting)
        torch.testing.assert_close(emitted, wrapper.planning_current_state(), rtol=0, atol=0)
        from thinker.main import Env
        flags.planning_meta_enabled = True
        local_model = SimpleNamespace(flags=SimpleNamespace())
        env = SimpleNamespace(flags=flags, model_net=local_model, env=SimpleNamespace(env=wrapper))
        updated = Env.apply_planning_parameters(env, .8, .002, 4)
        assert not torch.equal(emitted, updated)
        assert flags.discounting == .8 and flags.think_cost == .002
        assert env.planning_parameter_version == 4
    finally:
        wrapper.close()


def test_recorder_binds_learned_values_but_preserves_other_identity(tmp_path):
    from tests.test_visual_behav_recorder import _checkpoint_bundle
    from thinker.planning_meta import PlanningMetaParameters
    import visual_behav as recorder
    import copy
    root, config = _checkpoint_bundle(tmp_path)
    bundle = recorder.resolve_checkpoint_bundle(root, stable_polls=1, stable_interval_seconds=0)
    config.update(planning_meta_enabled=True, discounting=.99, think_cost=.0001)
    parameters = PlanningMetaParameters(.98, .0002)
    gamma, cost = (float(v.detach()) for v in parameters.values())
    actor = dict(real_step=5000000, flags=dict(config, discounting=gamma, think_cost=cost),
        actor_net_state_dict=dict(planning_gamma=torch.tensor(gamma), planning_cost=torch.tensor(cost),
                                 planning_version=torch.tensor(3)),
        planning_meta_state=dict(schema=1, gamma=gamma, think_cost=cost, updates=3,
                                 parameters=parameters.state_dict()))
    model = dict(real_step=5000000, flags=dict(config), model_net_state_dict=dict(weight=torch.ones(1)),
                 planning_parameters=dict(gamma=.99, think_cost=.0001, version=2))
    result = recorder._validate_loaded_checkpoint_payloads(actor, model, bundle,
        config_xpid=config['xpid'], config_mapping=config)
    assert result['actor_real_step'] == 5000000
    bad = copy.deepcopy(model)
    bad['flags']['max_depth'] += 1
    with pytest.raises(recorder.VisualBehaviorError, match='flags differ'):
        recorder._validate_loaded_checkpoint_payloads(actor, bad, bundle,
            config_xpid=config['xpid'], config_mapping=config)


def test_configuration_and_meta_schedule_fail_closed():
    from thinker import util
    from thinker.planning_meta import configure_meta, PlanningMetaLearner
    import copy
    from pathlib import Path
    config = str(Path(__file__).resolve().parents[1]/'config/planning_meta.yaml')
    flags = util.create_setting(args=['--config', config], save_flags=False)
    assert flags.planning_backend == 'torch' and flags.batch_length == 8
    for key, value in [('float16', True), ('voc_gru_connected', False),
                       ('icopro_holdout_sessions', '4'), ('planning_meta_samples', 1),
                       ('autotune', True), ('noise_enable', True)]:
        bad = copy.copy(flags)
        setattr(bad, key, value)
        with pytest.raises(ValueError):
            configure_meta(bad)
    meta = PlanningMetaLearner.__new__(PlanningMetaLearner)
    meta.learner = SimpleNamespace(flags=flags, real_step=flags.planning_meta_warmup_steps-1)
    meta.actor_updates = 31
    assert not meta.due()
    meta.learner.real_step += 1
    assert meta.due()
    meta.actor_updates += 1
    assert not meta.due()


def test_real_learner_meta_step(tmp_path, monkeypatch):
    from tests.test_dynamic_voc_learner import _exact_projection_learner_flags, _rollout, _NullWriter
    from tests.test_dynamic_imitation_actor import _FrozenFakeModel
    from thinker.learn_actor import SActorLearner
    from thinker.dynamic_imitation import DynamicImitationRunner
    from thinker.planning_meta import configure_meta
    from gymnasium import spaces
    from collections import namedtuple
    import thinker.planning_meta as meta_module

    flags = _exact_projection_learner_flags(tmp_path, "meta-integration")
    flags.planning_meta_enabled = True
    flags.planning_meta_every = 1
    flags.planning_meta_warmup_steps = 0
    flags.planning_meta_batch_size = 1
    flags.voc_gru_connected = True
    flags.voc_critic_parameterization = "common_contrast_v1"
    flags.voc_gru_gradient_scale = 10.
    flags.batch_length = 8
    flags.icopro_burn_in_steps = 16
    flags.icopro_data_path = "synthetic-test-only"
    flags.icopro_holdout_sessions = ""
    flags.icopro_train_sessions = "1,2,3,4"
    configure_meta(flags)
    (tmp_path/flags.xpid).mkdir()
    actor, rollout, core = _rollout(flags)
    import copy
    resume_actor = copy.deepcopy(actor)
    monkeypatch.setattr("thinker.learn_actor.FileWriter", lambda **kw: _NullWriter())
    monkeypatch.setattr(SActorLearner, "_init_imitation_components", lambda self: None)
    learner = SActorLearner(None, {}, flags, actor, torch.device("cpu"))
    model = _FrozenFakeModel(5)
    model.flags = flags
    actor.online_real_state_space = model.observation_space
    learner.bc_model_net = model
    learner.bc_runner = DynamicImitationRunner(actor, model, flags, device="cpu")
    batch = dict(obs_seq=np.zeros((1, 25, 4, 8, 8), np.uint8),
                 actions_seq=np.zeros((1, 24), np.int64),
                 initial_prev_action=np.zeros(1, np.int64),
                 score_mask=np.array([False]*16+[True]*8))
    learner.bc_loader = SimpleNamespace(rng=np.random.default_rng(10),
                                       get_sequence_batch=lambda **kw: batch)
    def real_bc():
        learner._planning_inner_batch = batch
        learner._planning_inner_rng = (torch.get_rng_state(), None)
        return learner.bc_runner.rollout(batch, tree_carry=True, training=True)
    learner._maybe_compute_imitation = real_bc
    fast_snapshots = []
    functional_step = meta_module.functional_adam
    def capture_fast(*args, **kwargs):
        fast = functional_step(*args, **kwargs)
        fast_snapshots.append({k: v.detach().clone() for k, v in fast.items()})
        return fast
    monkeypatch.setattr(meta_module, "functional_adam", capture_fast)
    MetaOut = namedtuple("MetaOut", rollout._fields+("planning_version",))
    rollout = MetaOut(*rollout, torch.zeros_like(rollout.done, dtype=torch.long))
    before = learner.planning_meta.parameters.raw.detach().clone()
    learner.consume_data((rollout, core))
    assert learner.planning_meta.updates == 1
    assert int(actor.planning_version) == 1
    assert torch.isfinite(learner.planning_meta.parameters.raw).all()
    assert not torch.equal(before, learner.planning_meta.parameters.raw)
    assert model.anchor.grad is None
    raw = learner.planning_meta.parameters.raw
    assert all(p is not raw for opt in (learner.optimizer, learner.voc_optimizer)
               for group in opt.param_groups for p in group['params'])
    assert learner.planning_meta.metrics["gamma_gradient"] != 0
    assert learner.planning_meta.metrics["lambda_gradient"] != 0
    for name, p in actor.named_parameters():
        if not name.startswith("voc_gate_head."):
            torch.testing.assert_close(p, fast_snapshots[-1][name], rtol=2e-5, atol=2e-7, msg=name)

    learner.save_checkpoint()
    resume_flags = copy.deepcopy(flags)
    resume_flags.ckp = True
    resumed_learner = SActorLearner(None, {}, resume_flags, resume_actor, torch.device("cpu"))
    assert resumed_learner.planning_meta.updates == 1
    torch.testing.assert_close(resumed_learner.planning_meta.parameters.raw,
                               learner.planning_meta.parameters.raw, rtol=0, atol=0)

    # Persist and restore all meta state, then take the same next update.
    import copy
    state = copy.deepcopy(learner.planning_meta.state_dict())
    checkpoint = dict(planning_meta_state=state, actor_net_state_dict=actor.state_dict())
    restored_flags = copy.deepcopy(flags)
    restored_flags.discounting, restored_flags.think_cost = .99, .0001
    meta_module.restore_checkpoint_planning(restored_flags, checkpoint)
    assert restored_flags.discounting == flags.discounting
    replica_actor = SimpleNamespace(planning_gamma=actor.planning_gamma.clone(),
        planning_cost=actor.planning_cost.clone(), planning_version=actor.planning_version.clone())
    replica = SimpleNamespace(flags=restored_flags, actor_net=replica_actor, device=torch.device("cpu"))
    restored = meta_module.PlanningMetaLearner(replica, state)
    assert restored.rng.integers(2**30) == learner.planning_meta.rng.integers(2**30)
    for control in (learner.planning_meta, restored):
        control.pending = torch.tensor([.12, -.2])
        control.commit(True, True)
    torch.testing.assert_close(restored.parameters.raw, learner.planning_meta.parameters.raw, rtol=0, atol=0)
    corrupted = copy.deepcopy(checkpoint)
    corrupted["actor_net_state_dict"]["planning_gamma"] = torch.tensor(.8)
    with pytest.raises(ValueError, match="actor buffers"):
        meta_module.restore_checkpoint_planning(restored_flags, corrupted)
