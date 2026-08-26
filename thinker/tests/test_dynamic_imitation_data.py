from pathlib import Path

import numpy as np
import pytest

from thinker.bc_loader import FrameStackedBehavioralDataLoader
from thinker.dataset_env import BehaviorSequenceVectorEnv


def _archive(
    path: Path,
    *,
    length: int = 40,
    episode_starts=(0,),
    episode_ends=None,
    times=True,
    time_values=None,
    game_actions: int = 9,
    action_values=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    images = np.empty((length, 6, 7, 3), dtype=np.uint8)
    for index in range(length):
        images[index].fill(index)
    if action_values is None:
        action_ids = np.arange(length, dtype=np.int64) % game_actions
        actions = np.eye(game_actions, dtype=np.float32)[action_ids]
    else:
        actions = np.asarray(action_values)
        if actions.shape[0] != length:
            raise ValueError("action_values first dimension must equal length")
    rewards = np.arange(length, dtype=np.float32)
    first = np.zeros(length, dtype=np.bool_)
    first[np.asarray(episode_starts, dtype=np.int64)] = True
    terminal = np.zeros(length, dtype=np.bool_)
    if episode_ends is not None:
        terminal[np.asarray(episode_ends, dtype=np.int64)] = True
    payload = {
        "image": images,
        "action": actions,
        "reward": rewards,
        "is_first": first,
        "is_terminal": terminal,
    }
    if times:
        payload["time"] = (
            np.arange(length, dtype=np.float64) * 0.05
            if time_values is None
            else np.asarray(time_values, dtype=np.float64)
        )
    np.savez_compressed(path, **payload)
    return path


def _modern_path(root, session, block=1, game=0):
    return (
        root
        / "sub-001"
        / f"ses-{session:02d}"
        / f"sub001-ses{session:02d}-block{block}-game{game}.npz"
    )


def test_session_split_discovery_and_selected_action_prior(tmp_path):
    for session in (1, 2, 3, 4):
        _archive(_modern_path(tmp_path, session), length=20, times=False)
    _archive(_modern_path(tmp_path, 1, block=2, game=1), length=20, game_actions=6)

    train = FrameStackedBehavioralDataLoader(
        tmp_path,
        subjects=(1,),
        game_id=0,
        num_actions=9,
        split="train",
        scored_length=4,
    )
    holdout = FrameStackedBehavioralDataLoader(
        tmp_path,
        subjects=(1,),
        game_id=0,
        num_actions=9,
        split="holdout",
        scored_length=4,
    )

    assert {record.session for record in train.file_records} == {1, 2, 3}
    assert {record.session for record in holdout.file_records} == {4}
    assert len(train.data_files) == 3
    assert np.isclose(train.action_distribution.sum(), 1.0)
    expected = np.bincount(np.arange(19) % 9, minlength=9).astype(float)
    np.testing.assert_allclose(train.action_distribution, expected / expected.sum())


def test_generic_five_action_archive_uses_runtime_action_count(tmp_path):
    game_id = 17
    path = _modern_path(tmp_path, 1, game=game_id)
    scalar_actions = (np.arange(24) % 5).reshape(-1, 1)
    _archive(
        path,
        length=24,
        times=False,
        game_actions=5,
        action_values=scalar_actions,
    )

    loader = FrameStackedBehavioralDataLoader(
        tmp_path,
        subjects=(1,),
        sessions=(1,),
        split=None,
        game_id=game_id,
        num_actions=5,
        frame_stack_n=1,
        target_size=(6, 7),
    )
    batch = loader.get_sequence_batch(2, replace=False)

    assert loader.num_actions == 5
    assert loader.action_distribution.shape == (5,)
    assert batch["actions_seq"].shape == (2, 5)
    assert np.all((batch["actions_seq"] >= 0) & (batch["actions_seq"] < 5))


@pytest.mark.parametrize("num_actions", [0, -1, 5.5, "5", True])
def test_loader_requires_a_positive_integer_runtime_action_count(
    tmp_path, num_actions
):
    with pytest.raises(ValueError, match="num_actions must be a positive integer"):
        FrameStackedBehavioralDataLoader(tmp_path, num_actions=num_actions)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("width", "one-hot width 4 does not match num_actions=5"),
        ("all_zero", "strict one-hot rows"),
        ("multi_hot", "strict one-hot rows"),
        ("non_finite", "must all be finite"),
    ],
)
def test_loader_rejects_malformed_one_hot_actions(tmp_path, failure, message):
    length = 20
    action_ids = np.arange(length) % 5
    actions = np.eye(5, dtype=np.float32)[action_ids]
    if failure == "width":
        actions = actions[:, :4]
    elif failure == "all_zero":
        actions[0] = 0
    elif failure == "multi_hot":
        actions[0, :2] = 1
    elif failure == "non_finite":
        actions[0, 0] = np.nan
    path = _modern_path(tmp_path, 1, game=17)
    _archive(path, length=length, times=False, action_values=actions)

    with pytest.raises(ValueError, match=message):
        FrameStackedBehavioralDataLoader(
            tmp_path,
            subjects=(1,),
            sessions=(1,),
            split=None,
            game_id=17,
            num_actions=5,
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("fractional", "must be integer-valued"),
        ("negative", r"outside \[0, 4\]"),
        ("too_large", r"outside \[0, 4\]"),
        ("non_finite", "must all be finite"),
    ],
)
def test_loader_rejects_invalid_scalar_actions(tmp_path, failure, message):
    length = 20
    actions = (np.arange(length) % 5).astype(np.float64)
    if failure == "fractional":
        actions[0] = 1.5
    elif failure == "negative":
        actions[0] = -1
    elif failure == "too_large":
        actions[0] = 5
    elif failure == "non_finite":
        actions[0] = np.inf
    path = _modern_path(tmp_path, 1, game=17)
    _archive(path, length=length, times=False, action_values=actions)

    with pytest.raises(ValueError, match=message):
        FrameStackedBehavioralDataLoader(
            tmp_path,
            subjects=(1,),
            sessions=(1,),
            split=None,
            game_id=17,
            num_actions=5,
        )


def test_legacy_layout_is_supported(tmp_path):
    legacy = (
        tmp_path
        / "sub_1"
        / "game_0"
        / "day_2"
        / "block_3"
        / "chunk.npz"
    )
    _archive(legacy, length=20, times=False)
    loader = FrameStackedBehavioralDataLoader(
        tmp_path,
        subjects=(1,),
        sessions=(2,),
        split=None,
        game_id=0,
        num_actions=9,
    )
    assert loader.data_files == [str(legacy.resolve())]
    assert loader.file_records[0].block == 3


def test_causal_resampling_alignment_and_exact_sequence_schema(tmp_path):
    path = _archive(_modern_path(tmp_path, 1), length=50, episode_ends=(49,))
    loader = FrameStackedBehavioralDataLoader(
        tmp_path,
        subjects=(1,),
        sessions=(1,),
        split=None,
        game_id=0,
        num_actions=9,
        scored_length=4,
        frame_stack_n=2,
        target_size=(6, 7),
        grayscale=True,
        decision_hz=10.0,
        seed=7,
    )
    batch = loader.get_sequence_batch(3, replace=False)

    assert batch["obs_seq"].shape == (3, 6, 2, 6, 7)
    assert batch["actions_seq"].shape == (3, 5)
    assert batch["rewards_seq"].shape == (3, 5)
    assert batch["done_seq"].shape == (3, 5)
    assert batch["truncated_seq"].shape == (3, 5)
    assert batch["initial_prev_action"].shape == (3,)
    np.testing.assert_array_equal(
        batch["score_mask"], np.array([False, True, True, True, True])
    )
    assert set(batch["source_file"]) == {str(path.resolve())}
    assert batch["decision_times"].shape == (3, 6)
    assert batch["observation_source_index"].shape == (3, 6)

    for row in range(3):
        source = batch["observation_source_index"][row]
        # Raw frames are at 20 Hz; causal sampling onto 10 Hz selects 0,2,4,...
        np.testing.assert_array_equal(np.diff(source), np.full(5, 2))
        assert np.all(source * 0.05 <= batch["decision_times"][row] + 1e-12)
        np.testing.assert_array_equal(batch["actions_seq"][row], source[:-1] % 9)
        burn = int(batch["window_start"][row])
        assert batch["initial_prev_action"][row] == (2 * (burn - 1)) % 9
        # The newest channel of each stack is its causally selected raw frame.
        np.testing.assert_array_equal(
            batch["obs_seq"][row, :, -1, 0, 0], source.astype(np.uint8)
        )
        # The older stack channel is the preceding *decision-time* frame
        # (two raw capture rows back), not the adjacent raw capture frame.
        np.testing.assert_array_equal(
            batch["obs_seq"][row, :, 0, 0, 0], (source - 2).astype(np.uint8)
        )
        expected_reward = np.array(
            [np.arange(lo, hi, dtype=np.float32).sum() for lo, hi in zip(source[:-1], source[1:])]
        )
        np.testing.assert_allclose(batch["rewards_seq"][row], expected_reward)


def test_windows_never_cross_episode_or_timestamp_gap(tmp_path):
    starts = (0, 20)
    ends = (19, 39)
    times = np.r_[np.arange(20) * 0.05, 3.0 + np.arange(20) * 0.05]
    _archive(
        _modern_path(tmp_path, 1),
        length=40,
        episode_starts=starts,
        episode_ends=ends,
        time_values=times,
    )
    loader = FrameStackedBehavioralDataLoader(
        tmp_path,
        subjects=(1,),
        sessions=(1,),
        split=None,
        game_id=0,
        num_actions=9,
        scored_length=4,
        frame_stack_n=1,
        target_size=(6, 7),
        decision_hz=10.0,
    )

    seen_episodes = set()
    for batch in loader.iter_batches(3, shuffle=False, stride=1):
        for episode, source in zip(
            batch["episode_index"], batch["observation_source_index"]
        ):
            seen_episodes.add(int(episode))
            assert np.all(source < 20) or np.all(source >= 20)
    assert seen_episodes == {0, 1}


def test_terminal_marker_is_attached_to_the_edge_ending_at_terminal_observation(
    tmp_path,
):
    _archive(
        _modern_path(tmp_path, 1),
        length=12,
        episode_ends=(11,),
        # 0.55 s is off the default 15-Hz decision grid, so the loader must
        # append the genuine terminal observation as the last endpoint.
        times=True,
    )
    loader = FrameStackedBehavioralDataLoader(
        tmp_path,
        subjects=(1,),
        sessions=(1,),
        split=None,
        game_id=0,
        num_actions=9,
        scored_length=4,
        frame_stack_n=1,
        target_size=(6, 7),
    )

    batches = list(loader.iter_batches(16, shuffle=False, stride=1))
    starts = np.concatenate([batch["window_start"] for batch in batches])
    final_index = int(np.argmax(starts))
    done = np.concatenate([batch["done_seq"] for batch in batches], axis=0)
    truncated = np.concatenate(
        [batch["truncated_seq"] for batch in batches], axis=0
    )

    np.testing.assert_array_equal(done[final_index], [False, False, False, False, True])
    assert not truncated[final_index].any()
    assert sum(int(episode.done.sum()) for episode in loader._episodes) == 1


def test_exhaustive_stride_has_auditable_coverage(tmp_path):
    _archive(_modern_path(tmp_path, 4), length=15, times=False)
    loader = FrameStackedBehavioralDataLoader(
        tmp_path,
        subjects=(1,),
        sessions=(4,),
        split=None,
        game_id=0,
        num_actions=9,
        scored_length=4,
        frame_stack_n=1,
        target_size=(6, 7),
    )
    batches = list(loader.iter_batches(2, shuffle=False, drop_last=False))
    assert [len(batch["actions_seq"]) for batch in batches] == [2, 1]
    np.testing.assert_array_equal(
        np.concatenate([batch["window_start"] for batch in batches]), [1, 5, 9]
    )
    coverage = loader.evaluation_coverage()
    assert coverage["n_windows"] == 3
    assert coverage["unique_scored_targets"] == 12
    assert coverage["eligible_scored_targets"] == 12
    assert coverage["coverage_fraction"] == 1.0


def test_sampling_is_deterministic_for_equal_seed(tmp_path):
    _archive(_modern_path(tmp_path, 1), length=30, times=False)
    kwargs = dict(
        base_path=tmp_path,
        subjects=(1,),
        sessions=(1,),
        split=None,
        game_id=0,
        num_actions=9,
        seed=19,
    )
    first = FrameStackedBehavioralDataLoader(**kwargs).get_sequence_batch(4)
    second = FrameStackedBehavioralDataLoader(**kwargs).get_sequence_batch(4)
    np.testing.assert_array_equal(first["window_start"], second["window_start"])
    np.testing.assert_array_equal(
        first["observation_source_index"], second["observation_source_index"]
    )


def test_sequence_env_teacher_forces_edges_and_never_wraps():
    obs = np.arange(2 * 6, dtype=np.uint8).reshape(2, 6, 1, 1, 1)
    actions = np.array([[1, 2, 3, 4, 0], [2, 3, 4, 0, 1]], dtype=np.int64)
    rewards = np.arange(10, dtype=np.float32).reshape(2, 5)
    env = BehaviorSequenceVectorEnv(
        obs,
        actions_seq=actions,
        rewards_seq=rewards,
        initial_prev_action=np.array([4, 3]),
        score_mask=np.array([False, True, True, True, True]),
        num_actions=5,
    )
    reset_obs, info = env.reset()
    np.testing.assert_array_equal(reset_obs, obs[:, 0])
    np.testing.assert_array_equal(info["initial_prev_action"], [4, 3])
    np.testing.assert_array_equal(env.current_human_action(), [1, 2])

    with pytest.raises(ValueError, match="Teacher-forced action mismatch"):
        env.step(np.array([0, 2]))
    for edge in range(5):
        next_obs, reward, done, truncated, edge_info = env.step(actions[:, edge])
        np.testing.assert_array_equal(next_obs, obs[:, edge + 1])
        np.testing.assert_array_equal(reward, rewards[:, edge])
        np.testing.assert_array_equal(done, [False, False])
        np.testing.assert_array_equal(truncated, [False, False])
        np.testing.assert_array_equal(edge_info["score_mask"], [edge > 0, edge > 0])
    np.testing.assert_array_equal(env.has_more(), [False, False])
    with pytest.raises(RuntimeError, match="exhausted"):
        env.step(actions[:, 0])
    # Exhaustion never silently returns obs[:, 0]; an explicit reset is needed.
    np.testing.assert_array_equal(env.reset()[0], obs[:, 0])


def test_loader_fails_fast_when_no_file_matches(tmp_path):
    with pytest.raises(FileNotFoundError, match="No behavioral NPZ files matched"):
        FrameStackedBehavioralDataLoader(
            tmp_path,
            subjects=(1,),
            sessions=(4,),
            split=None,
            game_id=0,
            num_actions=9,
        )
