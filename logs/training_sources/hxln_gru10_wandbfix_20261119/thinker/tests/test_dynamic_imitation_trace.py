from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from thinker import util
from thinker import actor_net as actor_net_module
from thinker.actor_net import ActorNet, ActorOut
from thinker.dataset_env import LazyBehaviorEpisodeVectorEnv
from thinker.dynamic_imitation import (
    DYNAMIC_TRACE_EVENT_FIELDS,
    DYNAMIC_TRACE_EVENT_SCHEMA_VERSION,
    DynamicImitationRunner,
    VOC_Q_RECONSTRUCTION_DUELING_CENTERED,
    VOC_Q_RECONSTRUCTION_DUELING_COMMON,
    _trace_predicted_xs,
    _trace_unit_probability,
    _voc_trace_diagnostics,
    dynamic_trace_event_fields,
)


class _CountingProvider:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.calls = []

    def __getitem__(self, index):
        self.calls.append(int(index))
        return self.values[index]


def test_lazy_episode_reads_only_reset_and_teacher_forced_real_edges():
    frames = np.arange(3 * 4, dtype=np.uint8).reshape(3, 1, 2, 2)
    provider = _CountingProvider(frames)
    env = LazyBehaviorEpisodeVectorEnv(
        provider,
        actions_seq=np.asarray([1, 2]),
        rewards_seq=np.asarray([0.5, 1.5], dtype=np.float32),
        initial_prev_action=np.asarray([0]),
        score_mask=np.asarray([False, True]),
        num_actions=3,
        observation_space=spaces.Box(0, 255, shape=(1, 2, 2), dtype=np.uint8),
    )

    np.testing.assert_array_equal(env.reset()[0], frames[None, 0])
    assert provider.calls == [0]
    env.quick_save(slot_id=7)
    with pytest.raises(ValueError, match="Teacher-forced"):
        env.step(np.asarray([0]))
    assert provider.calls == [0]
    np.testing.assert_array_equal(env.step(np.asarray([1]))[0], frames[None, 1])
    env.quick_load(slot_id=7)
    np.testing.assert_array_equal(env.step(np.asarray([1]))[0], frames[None, 1])
    np.testing.assert_array_equal(env.step(np.asarray([2]))[0], frames[None, 2])
    assert provider.calls == [0, 1, 1, 2]
    assert not env.has_more()[0]
    with pytest.raises(RuntimeError, match="exhausted"):
        env.step(np.asarray([2]))


class _TraceModel(torch.nn.Module):
    hidden_shape = (5, 1, 1)

    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.observation_space = spaces.Box(
            0, 255, shape=(1, 2, 2), dtype=np.uint8
        )
        self.sr_net = SimpleNamespace(hidden_shape=(2, 1, 1))
        self.vp_net = SimpleNamespace(hidden_shape=(3, 1, 1))

    def initial_state(self, batch_size, device=None):
        return {"core": torch.zeros(batch_size, 1, device=device)}

    @staticmethod
    def _latent(batch_size, device, offset=0.0):
        hs = torch.arange(5, dtype=torch.float32, device=device).view(
            1, 1, 5, 1, 1
        )
        hs = hs.expand(1, batch_size, -1, -1, -1) + float(offset)
        xs = torch.zeros(1, batch_size, 1, 2, 2, device=device) + float(offset)
        return xs, hs

    def forward(self, env_state, done, actions, state, **_kwargs):
        del done, actions
        batch_size = env_state.shape[0]
        device = env_state.device
        xs, hs = self._latent(batch_size, device)
        return SimpleNamespace(
            vs=torch.zeros(1, batch_size, 1, device=device),
            policy=torch.zeros(1, batch_size, 1, self.num_actions, device=device),
            xs=xs,
            hs=hs,
            state={"core": torch.zeros(batch_size, 1, device=device)},
        )

    def forward_single(self, state, action, **_kwargs):
        batch_size = action.shape[0]
        device = action.device
        offset = action.float().reshape(batch_size)[0].item() + 1.0
        xs, hs = self._latent(batch_size, device, offset)
        return SimpleNamespace(
            rs=torch.zeros(1, batch_size, 1, device=device),
            vs=torch.zeros(1, batch_size, 1, device=device),
            policy=torch.zeros(1, batch_size, 1, self.num_actions, device=device),
            xs=xs,
            hs=hs,
            dones=torch.zeros(1, batch_size, 1, dtype=torch.bool, device=device),
            state={"core": state["core"]},
        )


class _ProceedResetStopActor(torch.nn.Module):
    def __init__(self, num_actions, reward_n, control_mode="schedule"):
        super().__init__()
        self.num_actions = num_actions
        self.dim_actions = 1
        self.tuple_action = False
        self.discrete_action = True
        self.reward_n = reward_n
        self.control_mode = control_mode
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.voc_eval_stochastic = True
        self.voc_gate_epsilon_greedy_execution = True
        self.runtime_history = []

    def initial_state(self, batch_size, device=None):
        del batch_size, device
        return ()

    def forward(self, env_out, state, compute_loss=False, greedy=False):
        del compute_loss, greedy
        self.runtime_history.append(
            (
                self.voc_eval_stochastic,
                self.voc_gate_epsilon_greedy_execution,
            )
        )
        phase = env_out.phase[-1]
        search_steps = env_out.search_steps[-1]
        batch_size = phase.shape[0]
        logits = torch.arange(
            self.num_actions, dtype=torch.float32, device=phase.device
        ).view(1, 1, self.num_actions).expand(batch_size, 1, -1)
        primary = torch.ones(batch_size, dtype=torch.long, device=phase.device)
        if self.control_mode == "stop":
            control = torch.full_like(phase, util.STOP)
        elif self.control_mode == "proceed":
            control = torch.full_like(phase, util.PROCEED)
        elif self.control_mode == "schedule":
            control = torch.where(
                search_steps == 0,
                torch.full_like(phase, util.PROCEED),
                torch.where(
                    search_steps == 1,
                    torch.full_like(phase, util.RESET),
                    torch.full_like(phase, util.STOP),
                ),
            )
        else:
            raise AssertionError(f"unsupported test control mode {self.control_mode}")
        search = phase == util.SEARCH_PHASE
        real = phase == util.NEED_REAL_ACTION_PHASE
        policy_type = torch.where(
            search,
            torch.full_like(phase, util.POLICY_SEARCH),
            torch.where(
                real,
                torch.full_like(phase, util.POLICY_REAL),
                torch.full_like(phase, util.POLICY_NONE),
            ),
        )
        policy_valid = policy_type != util.POLICY_NONE
        return ActorOut(
            pri=primary.unsqueeze(0),
            pri_param=logits.unsqueeze(0),
            reset=control.unsqueeze(0),
            reset_logits=torch.zeros(1, batch_size, 2, device=phase.device),
            action=(primary, control),
            action_prob=torch.ones(1, batch_size, device=phase.device),
            c_action_log_prob=torch.zeros(1, batch_size, device=phase.device),
            baseline=torch.zeros(
                1, batch_size, self.reward_n, device=phase.device
            ),
            baseline_enc=None,
            entropy_loss=torch.zeros(1, batch_size, device=phase.device),
            reg_loss=torch.zeros(1, batch_size, device=phase.device),
            misc={},
            search_control=control.unsqueeze(0),
            search_control_logits=torch.zeros(
                1, batch_size, 3, device=phase.device
            ),
            primary_valid=policy_valid.unsqueeze(0),
            control_valid=search.unsqueeze(0),
            policy_valid=policy_valid.unsqueeze(0),
            policy_type=policy_type.unsqueeze(0),
            voc_q=None,
            voc_features=None,
        ), state


def _trace_flags(*, max_search_steps=4, max_depth=4):
    return util.create_setting(
        args=[
            "--dynamic_search", "true",
            "--max_search_steps", str(max_search_steps),
            "--max_depth", str(max_depth),
            "--see_h", "true",
            "--see_x", "true",
            "--model_done_loss_cost", "0",
            "--parallel", "false",
            "--float16", "false",
        ],
        save_flags=False,
    )


def _trace_batch(edge_count=2, num_actions=3):
    frames = np.arange((edge_count + 1) * 4, dtype=np.uint8).reshape(
        edge_count + 1, 1, 2, 2
    )
    provider = _CountingProvider(frames)
    actions = (
        np.arange(edge_count, dtype=np.int64) + num_actions - 1
    ) % num_actions
    rewards = np.arange(1, edge_count + 1, dtype=np.float32)
    done = np.zeros(edge_count, dtype=np.bool_)
    done[-1] = True
    score_mask = np.ones(edge_count, dtype=np.bool_)
    score_mask[0] = False
    batch = {
        "observation_provider": provider,
        "observation_space": spaces.Box(
            0, 255, shape=(1, 2, 2), dtype=np.uint8
        ),
        "actions_seq": actions,
        "rewards_seq": rewards,
        "done_seq": done,
        "truncated_seq": np.zeros(edge_count, dtype=np.bool_),
        "initial_prev_action": np.asarray([min(1, num_actions - 1)], dtype=np.int64),
        "score_mask": score_mask,
        "observation_source_index": np.arange(
            10, 10 * (edge_count + 2), 10, dtype=np.int64
        )[: edge_count + 1],
    }
    return batch, provider


class _ListSink:
    def __init__(self, images="none"):
        self.events = []
        self.close_calls = 0
        self.images = images

    def write_event(self, event):
        self.events.append(event)

    def close(self):
        self.close_calls += 1


def _make_trace_runner(
    *, num_actions=3, max_search_steps=4, max_depth=4, control_mode="schedule"
):
    flags = _trace_flags(
        max_search_steps=max_search_steps, max_depth=max_depth
    )
    model = _TraceModel(num_actions)
    actor = _ProceedResetStopActor(
        num_actions,
        len(util.get_reward_names(flags)),
        control_mode=control_mode,
    )
    return DynamicImitationRunner(actor, model, flags, device="cpu")


def test_stream_rollout_emits_one_fixed_numeric_root_and_exact_events():
    runner = _make_trace_runner()
    batch, provider = _trace_batch()
    sink = _ListSink()

    summary = runner.stream_rollout_episode(batch, event_sink=sink)

    assert summary == {
        "edge_count": 2,
        "scored_edge_count": 1,
        "augmented_steps": 8,
        "native_event_count": 9,
        "initial_root_count": 1,
        "search_computation_count": 4,
        "search_expansion_count": 4,
        "search_revisit_count": 0,
        "reset_control_count": 2,
        "stop_count": 2,
        "forced_stop_count": 0,
        "real_transition_count": 2,
        "real_action_stored_count": 0,
        "wait_count": 0,
    }
    assert sink.close_calls == 0
    assert provider.calls == [0, 1, 2]
    assert [bool(event["initial_root"]) for event in sink.events] == [
        True, False, False, False, False, False, False, False, False
    ]
    first_schema = {
        key: (np.asarray(value).dtype.str, np.asarray(value).shape)
        for key, value in sink.events[0].items()
    }
    assert set(first_schema) == set(DYNAMIC_TRACE_EVENT_FIELDS)
    for event in sink.events:
        assert int(event["trace_schema_version"]) == DYNAMIC_TRACE_EVENT_SCHEMA_VERSION
        assert all(value is not None for value in event.values())
        assert not any(isinstance(value, (str, bytes)) for value in event.values())
        assert {
            key: (np.asarray(value).dtype.str, np.asarray(value).shape)
            for key, value in event.items()
        } == first_schema
    assert [bool(event["search_computation"]) for event in sink.events].count(True) == 4
    assert [bool(event["actor_reset"]) for event in sink.events].count(True) == 2
    assert [bool(event["real_transition"]) for event in sink.events].count(True) == 2
    assert not any(bool(event["forced_root_reset"]) for event in sink.events)
    assert sink.events[1]["tree_reachable_node_count_delta"] == 3
    assert sink.events[2]["tree_reachable_node_count_delta"] == 3
    assert sink.events[0]["tree_max_reachable_depth_post"] == 1
    assert sink.events[2]["tree_max_reachable_depth_post"] == 3
    assert sink.events[0]["im_vectors"].shape == (1, 2, 1, 1)
    assert sink.events[0]["im_vp_vectors"].shape == (1, 3, 1, 1)
    assert sink.events[0]["predicted_xs"].shape == (1, 2, 2)
    assert sink.events[0]["predicted_xs"].dtype == np.float32
    assert all(bool(event["predicted_xs_valid"]) for event in sink.events)
    np.testing.assert_array_equal(
        sink.events[0]["predicted_xs"], np.zeros((1, 2, 2), dtype=np.float32)
    )
    assert all(
        np.asarray(event["step_times"]).shape == (1,)
        and np.asarray(event["step_times"]).dtype == np.float32
        and not bool(event["step_times_valid"])
        for event in sink.events
    )
    assert sink.events[4]["post_observation_source_index"] == 20
    runner.close()


def test_writer_reader_round_trip_preserves_dynamic_status_semantics(tmp_path):
    pytest.importorskip("zarr")
    pytest.importorskip("numcodecs")
    from thinker.behavior_trace import EpisodeZarrWriter, open_video_stat

    runner = _make_trace_runner()
    batch, _provider = _trace_batch()
    partial = tmp_path / "episode.partial"
    final = tmp_path / "episode.zarr"
    writer = EpisodeZarrWriter(
        partial,
        metadata={"test": True},
        num_actions=3,
        tree_rep_schema={"all": [0, 44]},
        chunk_events=2,
        images="none",
    )
    summary = runner.stream_rollout_episode(batch, event_sink=writer)
    writer.finalize(final)
    opened = open_video_stat(final)

    assert summary["native_event_count"] == 9
    np.testing.assert_array_equal(
        opened["status"], [0, 2, 2, 1, 0, 2, 2, 1, 0]
    )
    np.testing.assert_array_equal(opened["human_action"][4], [0, 0, 1])
    np.testing.assert_array_equal(opened["thinker_action"][4], [0, 1, 0])
    assert opened.native.field("initial_root")[:].sum() == 1
    assert opened.native.field("pre_observation_source_index")[0] == 10
    assert opened.native.field("post_observation_source_index")[4] == 20
    runner.close()


def test_greedy_stream_temporarily_forces_deterministic_voc_runtime():
    runner = _make_trace_runner()
    batch, _provider = _trace_batch()
    sink = _ListSink()

    runner.stream_rollout_episode(
        batch, event_sink=sink, policy_mode="greedy"
    )

    assert set(runner.actor_net.runtime_history) == {(False, False)}
    assert runner.actor_net.voc_eval_stochastic is True
    assert runner.actor_net.voc_gate_epsilon_greedy_execution is True
    runner.close()


def test_sampled_stream_uses_training_sampling_epsilon_and_human_real_action(
    monkeypatch,
):
    flags = _trace_flags(max_search_steps=2, max_depth=4)
    for name, value in {
        "see_h": False,
        "see_x": False,
        "see_real_state": False,
        "dynamic_voc_mode": "control",
        "dynamic_factorized_control": True,
        "voc_dedicated_gate": True,
        "voc_eval_stochastic": True,
        "voc_gate_epsilon_greedy_execution": False,
        "voc_gate_temperature": 1.0,
        "voc_train_epsilon": 0.02,
        "train_actor": True,
        "voc_gate_policy_schema_version": 3,
        "voc_dueling_q": True,
    }.items():
        setattr(flags, name, value)

    num_actions = 3
    model = _TraceModel(num_actions)
    bootstrap_actor = _ProceedResetStopActor(
        num_actions,
        len(util.get_reward_names(flags)),
        control_mode="stop",
    )
    runner = DynamicImitationRunner(
        bootstrap_actor, model, flags, device="cpu"
    )
    batch, _provider = _trace_batch(edge_count=2, num_actions=num_actions)
    # Make the human target differ from the actor's known primary proposal on
    # every real-policy row, so accepted actions prove teacher forcing rather
    # than passing accidentally because proposal == target.
    batch["actions_seq"] = np.full(2, 2, dtype=np.int64)

    planner = runner._make_or_update_planner(batch, tree_carry=True)
    actor = ActorNet(
        obs_space=planner.observation_space,
        action_space=planner.action_space,
        flags=flags,
        tree_rep_meaning=util.get_tree_rep_meaning(
            num_actions, len(model.sr_net.hidden_shape[:1]), flags
        ),
    )
    with torch.no_grad():
        # Keep every primary sample at action zero with a representable large
        # margin while retaining the categorical sampled code path.
        for head in (actor.policy, actor.im_policy):
            head.weight.zero_()
            head.bias.copy_(torch.tensor([0.0, -1000.0, -1000.0]))
        actor.reset.weight.zero_()
        actor.reset.bias.zero_()
        actor.voc_gate_head.weight.zero_()
        actor.voc_gate_head.bias.fill_(2.0)
    runner.actor_net = actor

    original_sample = actor_net_module.sample
    sample_calls = []

    def observe_sample(logits, greedy, dim=-1):
        sample_calls.append((logits.detach().clone(), bool(greedy), int(dim)))
        return original_sample(logits, greedy=greedy, dim=dim)

    monkeypatch.setattr(actor_net_module, "sample", observe_sample)
    torch.manual_seed(123)
    sink = _ListSink()
    summary = runner.stream_rollout_episode(
        batch, event_sink=sink, policy_mode="sampled"
    )

    # Every ActorNet forward calls the binary gate, conditional P/R bout and
    # routed primary samplers in that order.  sampled mode must leave all three
    # categorical decisions non-greedy.
    assert len(sample_calls) >= 3
    assert len(sample_calls) % 3 == 0
    assert all(not greedy for _logits, greedy, _dim in sample_calls)
    assert all(
        [call[0].shape[-1] for call in sample_calls[offset : offset + 3]]
        == [2, 2, num_actions]
        for offset in range(0, len(sample_calls), 3)
    )

    # With dedicated gate log-odds 2, temperature 1 and training epsilon .02,
    # the exact executed CONTINUE probability is the soft probability mixed
    # with .02 uniform binary exploration.
    root_gate_probability = torch.softmax(sample_calls[0][0], dim=-1)[..., 0]
    expected_continue = (1.0 - 0.02) * torch.sigmoid(torch.tensor(2.0)) + 0.01
    torch.testing.assert_close(
        root_gate_probability,
        torch.full_like(root_gate_probability, expected_continue),
    )

    real_events = [event for event in sink.events if bool(event["real_transition"])]
    assert summary["real_transition_count"] == len(real_events) == 2
    assert [int(event["primary_proposal"]) for event in real_events] == [0, 0]
    assert [int(event["human_action"]) for event in real_events] == [2, 2]
    assert [int(event["accepted_primary_action"]) for event in real_events] == [
        2,
        2,
    ]
    assert all(bool(event["human_action_valid"]) for event in real_events)
    assert all(
        bool(event["accepted_primary_action_valid"]) for event in real_events
    )
    runner.close()


@pytest.mark.parametrize(
    "images", ["none", "real", "imagined", "all"]
)
def test_stream_image_profile_keeps_native_prediction_in_every_mode(
    images,
):
    runner = _make_trace_runner()
    batch, _provider = _trace_batch()
    sink = _ListSink(images=images)

    runner.stream_rollout_episode(batch, event_sink=sink)

    assert "predicted_xs" in sink.events[0]
    assert sink.events[0]["predicted_xs"].shape == (1, 2, 2)
    assert sink.events[0]["predicted_xs"].dtype == np.float32
    assert bool(sink.events[0]["predicted_xs_valid"])
    assert "pre_xs" not in sink.events[0]
    assert "post_xs" not in sink.events[0]
    assert set(sink.events[0]) == set(dynamic_trace_event_fields(images))
    assert "pre_real_states" not in sink.events[0]
    assert "post_real_states" not in sink.events[0]
    runner.close()


def test_predicted_xs_missing_row_keeps_raw_template_shape_and_invalidity():
    pre = SimpleNamespace(
        xs=torch.arange(24, dtype=torch.float32).reshape(1, 1, 2, 3, 4)
    )
    missing = SimpleNamespace(xs=None)

    initial, initial_valid = _trace_predicted_xs(
        pre, missing, initial_root=True
    )
    post, post_valid = _trace_predicted_xs(pre, missing, initial_root=False)

    assert initial.shape == post.shape == (2, 3, 4)
    assert initial.dtype == post.dtype == np.float32
    assert bool(initial_valid)
    assert not bool(post_valid)
    np.testing.assert_array_equal(
        initial, np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    )
    np.testing.assert_array_equal(post, np.zeros((2, 3, 4), dtype=np.float32))


def test_voc_trace_reconstructs_schema3_and_common_mode_without_second_forward():
    flags = _trace_flags()
    flags.voc_dueling_q = True
    flags.voc_gate_policy_schema_version = 3
    flags.think_cost_anneal = False
    flags.think_cost = 0.5
    rewards = util.get_reward_names(flags)
    baseline = torch.zeros(1, 1, len(rewards))
    baseline[..., rewards.index("re")] = 10.0
    baseline[..., rewards.index("think")] = 2.0
    actor_out = SimpleNamespace(
        voc_q=torch.tensor([[[2.0, 4.0]]]),
        baseline=baseline,
        search_control_logits=torch.zeros(1, 1, 3),
        reset_logits=None,
        misc={},
    )
    features = torch.tensor([[1.0, 2.0]])
    ema = {
        "weight": torch.eye(2),
        "bias": torch.zeros(2),
    }

    schema3 = _voc_trace_diagnostics(
        actor_out,
        features,
        flags=flags,
        ema_head_state=ema,
        effective_think_cost=None,
    )
    assert schema3["soft_probability_source"] == 3
    assert schema3["common_mode_q"] is False
    assert schema3["soft_continue_probability"].item() == pytest.approx(2 / 3)
    np.testing.assert_allclose(
        schema3["online_q"].numpy(), [[10 + 1 / 3, 12 + 1 / 3]], rtol=1e-6
    )

    flags.voc_gate_policy_schema_version = 13
    schema13 = _voc_trace_diagnostics(
        actor_out,
        features,
        flags=flags,
        ema_head_state=ema,
        effective_think_cost=None,
    )
    assert schema13["common_mode_q"] is True
    np.testing.assert_allclose(
        schema13["online_q"].numpy(), schema3["online_q"].numpy() + 3.0
    )
    assert VOC_Q_RECONSTRUCTION_DUELING_CENTERED == 1
    assert VOC_Q_RECONSTRUCTION_DUELING_COMMON == 2


@pytest.mark.parametrize("source", ["execution", "soft", "reset"])
@pytest.mark.parametrize("values", [
    [-0.00019307808543089777, -8.552160263061523, -19.208982467651367],
    [-0.008423720486462116, -4.7809038162231445, -23.94445037841797],
    [-0.0004619234532583505, -7.68019437789917, -18.306337356567383],
])
def test_voc_trace_real_capture_probability_roundoff(values, source):
    """Logits from failed Enduro/Pong/SpaceInvaders 10M captures."""
    logits = torch.tensor([[values]], dtype=torch.float32)
    before = logits.clone()
    # Demonstrate the original failure on the CPU recorder runtime.
    assert (torch.softmax(logits, -1)[..., :2].sum(-1) > 1).all()
    flags = _trace_flags()
    flags.voc_dueling_q = True
    flags.voc_gate_policy_schema_version = 4
    actor_out = SimpleNamespace(
        voc_q=torch.tensor([[[2.0, 4.0]]]),
        baseline=torch.zeros(1, 1, len(util.get_reward_names(flags))),
        search_control_logits=logits if source != "reset" else None,
        reset_logits=logits,
        misc={"voc_gate_soft_control_logits": logits} if source == "soft" else {},
    )
    result = _voc_trace_diagnostics(
        actor_out, torch.tensor([[1.0, 2.0]]), flags=flags,
        ema_head_state=None, effective_think_cost=0.5,
    )
    probability = result["soft_continue_probability"]
    assert probability.dtype == torch.float32
    assert ((probability >= 0) & (probability <= 1)).all()
    assert torch.isfinite(result["online_q"]).all()
    assert result["soft_probability_source"] == (2 if source == "soft" else 3)
    torch.testing.assert_close(
        probability, torch.softmax(logits.double(), -1)[..., :2].sum(-1)[-1].float(),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(logits, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_trace_probability_accepts_only_roundoff_without_mutating_input(dtype):
    eps = torch.finfo(dtype).eps
    values = torch.tensor([[-eps, 0.0, 0.4, 1.0, 1.0 + eps]], dtype=dtype)
    before = values.clone()
    result = _trace_unit_probability(values)
    torch.testing.assert_close(result, torch.tensor([0.0, 0.0, 0.4, 1.0, 1.0]))
    torch.testing.assert_close(values, before, rtol=0, atol=0)
    for value in (-8 * eps, 1 + 8 * eps):
        with pytest.raises(RuntimeError, match="probability is invalid"):
            _trace_unit_probability(torch.tensor([[value]], dtype=dtype))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.01, 1.01])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_trace_probability_rejects_nonfinite_and_material_errors(value, dtype):
    with pytest.raises(RuntimeError, match="probability is invalid"):
        _trace_unit_probability(torch.tensor([[value]], dtype=dtype))


@pytest.mark.parametrize("edge_count", [1, 3])
def test_immediate_stop_has_zero_search_rows_and_exact_real_offsets(
    tmp_path, edge_count
):
    pytest.importorskip("zarr")
    pytest.importorskip("numcodecs")
    from thinker.behavior_trace import EpisodeZarrWriter, open_video_stat

    runner = _make_trace_runner(control_mode="stop")
    batch, provider = _trace_batch(edge_count=edge_count)
    partial = tmp_path / f"stop-{edge_count}.partial"
    final = tmp_path / f"stop-{edge_count}.zarr"
    writer = EpisodeZarrWriter(
        partial,
        metadata={"case": "immediate-stop", "edge_count": edge_count},
        num_actions=3,
        tree_rep_schema={"all": [0, 44]},
        chunk_events=2,
        images="none",
    )

    summary = runner.stream_rollout_episode(batch, event_sink=writer)
    writer.finalize(final)
    opened = open_video_stat(final)

    assert summary["edge_count"] == edge_count
    assert summary["search_computation_count"] == 0
    assert summary["stop_count"] == edge_count
    assert summary["real_transition_count"] == edge_count
    assert summary["native_event_count"] == 1 + 2 * edge_count
    assert opened.native.field("search_computation")[:].sum() == 0
    np.testing.assert_array_equal(
        opened["status"], np.zeros(edge_count + 1, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        opened.source_event_index,
        np.arange(0, 2 * edge_count + 1, 2, dtype=np.int64),
    )
    assert provider.calls == list(range(edge_count + 1))
    runner.close()


def test_budget_cap_30_keeps_final_forced_computation_as_status2(tmp_path):
    pytest.importorskip("zarr")
    pytest.importorskip("numcodecs")
    from thinker.behavior_trace import EpisodeZarrWriter, open_video_stat

    runner = _make_trace_runner(
        max_search_steps=30, max_depth=64, control_mode="proceed"
    )
    batch, _provider = _trace_batch(edge_count=1)
    sink = _ListSink()
    summary = runner.stream_rollout_episode(batch, event_sink=sink)

    assert summary["search_computation_count"] == 30
    assert summary["forced_stop_count"] == 1
    assert summary["stop_count"] == 0
    assert summary["real_transition_count"] == 1
    assert summary["native_event_count"] == 32
    search_indices = [
        index
        for index, event in enumerate(sink.events)
        if bool(event["search_computation"])
    ]
    assert search_indices == list(range(1, 31))
    assert bool(sink.events[30]["forced_stop"])
    assert not any(bool(event["forced_stop"]) for event in sink.events[:30])

    partial = tmp_path / "cap30.partial"
    final = tmp_path / "cap30.zarr"
    writer = EpisodeZarrWriter(
        partial,
        metadata={"case": "cap-30"},
        num_actions=3,
        tree_rep_schema={"all": [0, 44]},
        chunk_events=8,
        images="none",
    )
    # Replaying into the strict writer also checks that the cap row retains
    # the exact same schema as all earlier expansion rows.
    runner.stream_rollout_episode(batch, event_sink=writer)
    writer.finalize(final)
    opened = open_video_stat(final)
    np.testing.assert_array_equal(
        opened["status"],
        np.asarray([0] + [2] * 30 + [0], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        opened.source_event_index, np.arange(32, dtype=np.int64)
    )
    assert int(np.count_nonzero(opened["status"] == 2)) == 30
    assert bool(opened.native.field("forced_stop")[30])
    runner.close()


@pytest.mark.parametrize("num_actions", [9, 6])
@pytest.mark.parametrize(
    ("max_search_steps", "max_depth"), [(1, 2), (30, 30)]
)
def test_dynamic_tree_rep_width_is_action_only_while_topology_grows(
    num_actions, max_search_steps, max_depth
):
    runner = _make_trace_runner(
        num_actions=num_actions,
        max_search_steps=max_search_steps,
        max_depth=max_depth,
        control_mode="schedule",
    )
    batch, _provider = _trace_batch(edge_count=1, num_actions=num_actions)
    sink = _ListSink()

    runner.stream_rollout_episode(batch, event_sink=sink)

    expected_width = 10 * num_actions + 14
    assert {event["tree_reps_raw"].shape for event in sink.events} == {
        (expected_width,)
    }
    assert {event["tree_reps_pre"].shape for event in sink.events} == {
        (expected_width,)
    }
    assert {event["tree_reps_post"].shape for event in sink.events} == {
        (expected_width,)
    }
    initial_nodes = int(sink.events[0]["tree_reachable_node_count_post"])
    reachable_counts = [
        int(event["tree_reachable_node_count_post"]) for event in sink.events
    ]
    assert initial_nodes == num_actions + 1
    assert max(reachable_counts) > initial_nodes
    assert any(
        int(event["tree_reachable_node_count_delta"]) == num_actions
        for event in sink.events
    )
    runner.close()
