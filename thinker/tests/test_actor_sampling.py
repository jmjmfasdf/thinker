import pytest
import torch

from thinker.actor_net import sample


@pytest.mark.parametrize(
    "logits",
    [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, -1.0],
        [1.7, 0.2, -0.4, -1.3],
    ],
    ids=["uniform", "nonuniform-three", "nonuniform-four"],
)
def test_gumbel_max_empirical_distribution_matches_softmax(logits):
    sample_n = 300_000
    logits = torch.tensor(logits, dtype=torch.float32)
    batch_logits = logits.expand(sample_n, -1)

    torch.manual_seed(8128)
    actions = sample(batch_logits, greedy=False)
    empirical = torch.bincount(
        actions, minlength=logits.numel()
    ).float() / sample_n

    torch.testing.assert_close(
        empirical,
        torch.softmax(logits, dim=-1),
        rtol=0.0,
        atol=0.004,
    )


def test_categorical_sampling_is_seed_deterministic_and_supports_arbitrary_dim():
    logits = torch.tensor(
        [
            [[0.2, -0.1, 0.7, 0.3], [1.0, 0.0, -0.4, 0.2], [-0.5, 0.8, 0.1, 0.4]],
            [[-0.2, 0.4, 0.3, 0.9], [0.6, -0.7, 0.2, 0.0], [0.1, 0.5, -0.3, 0.8]],
        ],
        dtype=torch.float32,
    )

    torch.manual_seed(91)
    first = sample(logits, greedy=False, dim=1)
    torch.manual_seed(91)
    second = sample(logits, greedy=False, dim=1)

    assert first.shape == (2, 4)
    assert torch.equal(first, second)
    assert torch.all((first >= 0) & (first < logits.shape[1]))


def test_greedy_sampling_preserves_argmax_and_tie_behavior_without_rng_use():
    logits = torch.tensor(
        [[1.0, 1.0, -2.0], [-3.0, 0.5, 0.5]], dtype=torch.float32
    )
    torch.manual_seed(17)
    state_before = torch.get_rng_state().clone()

    actions = sample(logits, greedy=True)

    assert actions.tolist() == [0, 1]
    assert torch.equal(torch.get_rng_state(), state_before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_categorical_sampling_stays_valid_for_supported_floating_dtypes(dtype):
    logits = torch.tensor(
        [[1.0e4, -1.0e4, -1.0e4], [-1.0e4, 1.0e4, -1.0e4]],
        dtype=dtype,
    ).expand(1024, -1, -1)

    torch.manual_seed(5)
    actions = sample(logits, greedy=False)

    assert actions.shape == (1024, 2)
    assert torch.equal(actions[:, 0], torch.zeros(1024, dtype=torch.long))
    assert torch.equal(actions[:, 1], torch.ones(1024, dtype=torch.long))


def test_categorical_sampling_rejects_non_floating_logits():
    with pytest.raises(TypeError, match="must be floating point"):
        sample(torch.tensor([[1, 2, 3]]), greedy=False)
