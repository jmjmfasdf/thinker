import math

import torch

from thinker import util
from thinker.core.vtrace import compute_v_trace


def test_wait_identity_transitions_connect_stored_action_to_barrier_reward():
    # t=0 stores an early real action, t=1 is WAIT, and t=2 is the barrier
    # release.  The WAIT row has rho=1 and discount=1, so reward at t=2 must
    # still reach the action chosen at t=0.
    out = compute_v_trace(
        log_rhos=torch.tensor([[math.log(0.5)], [0.0], [0.0]]),
        discounts=torch.tensor([[1.0], [1.0], [0.97]]),
        rewards=torch.tensor([[0.0], [0.0], [1.0]]),
        values=torch.zeros(3, 1),
        bootstrap_value=torch.zeros(1),
        return_norm_type=-1,
    )

    torch.testing.assert_close(out.vs[:, 0], torch.tensor([0.5, 1.0, 1.0]))
    assert out.pg_advantages[0, 0] > 0


def test_return_normalization_excludes_scheduler_only_rows():
    rewards = torch.tensor([[1.0], [3.0], [1000.0]])
    mask = torch.tensor([[True], [True], [False]])
    common = dict(
        log_rhos=torch.zeros(3, 1),
        discounts=torch.zeros(3, 1),
        rewards=rewards,
        values=torch.zeros(3, 1),
        bootstrap_value=torch.zeros(1),
        norm_mask=mask,
    )

    percentile_buffer = util.FifoBuffer(16, device=torch.device("cpu"))
    compute_v_trace(
        **common,
        return_norm_type=0,
        norm_stat=(None, None, None, percentile_buffer),
    )
    assert percentile_buffer.num_elements == 2
    torch.testing.assert_close(
        percentile_buffer.buffer[:2], torch.tensor([1.0, 3.0])
    )

    standardized = compute_v_trace(
        **common, return_norm_type=2, norm_stat=None
    )
    torch.testing.assert_close(
        standardized.pg_advantages[:2, 0], torch.tensor([-1.0, 1.0])
    )


def test_empty_stage_local_normalization_mask_is_neutral():
    buffer = util.FifoBuffer(16, device=torch.device("cpu"))
    out = compute_v_trace(
        log_rhos=torch.zeros(2, 1),
        discounts=torch.ones(2, 1),
        rewards=torch.zeros(2, 1),
        values=torch.zeros(2, 1),
        bootstrap_value=torch.zeros(1),
        return_norm_type=0,
        norm_stat=(None, None, None, buffer),
        norm_mask=torch.zeros(2, 1, dtype=torch.bool),
    )
    assert buffer.num_elements == 0
    assert torch.isfinite(out.pg_advantages).all()
    assert out.norm_stat[2].item() == 1.0
