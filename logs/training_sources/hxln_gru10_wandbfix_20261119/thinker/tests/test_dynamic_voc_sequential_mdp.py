"""Sequential sparse-return tests for the state-conditioned VoC gate.

These tests deliberately avoid a per-state lookup table.  Easy roots, hard
roots, and post-compute states all pass through the same feature map and the
same value, dueling-advantage, and gate heads.  The only task reward appears
several transitions after the gate decisions, and reaches those decisions via
the production V-trace -> VoC-target path.
"""

from dataclasses import dataclass

import torch
from torch import nn

from thinker import util
from thinker.actor_net import compute_voc_gate_distribution
from thinker.core.vtrace import compute_v_trace
from thinker.learn_actor import compute_dynamic_voc_loss, compute_dynamic_voc_target


_COMPUTATION_COST = 0.2
_TRAJECTORY_LENGTH = 5


@dataclass
class _SparseTrajectory:
    voc_target: torch.Tensor
    state_return: torch.Tensor
    post_valid: torch.Tensor
    task_rewards: torch.Tensor


def _sparse_trajectory(
    difficulty: torch.Tensor,
    root_control: torch.Tensor,
    post_control: torch.Tensor,
) -> _SparseTrajectory:
    """Build a two-decision MDP whose reward is terminal and delayed.

    Difficulty zero is easy: acting immediately yields return 1 and thinking
    cannot improve it.  Difficulty one is hard: acting immediately yields zero,
    while one PROCEED/RESET reveals the useful action and yields terminal return
    2.  A second computation never improves either state and only pays cost.
    """

    batch_n = difficulty.numel()
    device = difficulty.device
    dtype = torch.float32
    post_valid = root_control != util.STOP
    root_continue = post_valid
    post_continue = post_valid & (post_control != util.STOP)

    task_rewards = torch.zeros(
        (_TRAJECTORY_LENGTH, batch_n), device=device, dtype=dtype
    )
    easy = difficulty == 0
    hard_improved = (difficulty == 1) & root_continue
    task_rewards[-1] = easy.to(dtype) + 2.0 * hard_improved.to(dtype)

    think_rewards = torch.zeros_like(task_rewards)
    think_rewards[0] = -root_continue.to(dtype)
    think_rewards[1] = -post_continue.to(dtype)

    discounts = torch.ones_like(task_rewards)
    discounts[-1] = 0.0
    zeros = torch.zeros_like(task_rewards)
    bootstrap = torch.zeros(batch_n, device=device, dtype=dtype)
    log_rhos = torch.zeros_like(task_rewards)

    # No future value is hand-injected here.  With on-policy log-ratios, lambda
    # one, zero baselines, and a terminal bootstrap, V-trace recursively carries
    # the sole terminal reward (and computation costs) back to both gate calls.
    task_trace = compute_v_trace(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=task_rewards,
        values=zeros,
        bootstrap_value=bootstrap,
        return_norm_type=-1,
        lamb=1.0,
    )
    think_trace = compute_v_trace(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=think_rewards,
        values=zeros,
        bootstrap_value=bootstrap,
        return_norm_type=-1,
        lamb=1.0,
    )
    target = compute_dynamic_voc_target(
        task_rewards=task_rewards,
        think_rewards=think_rewards,
        task_discounts=discounts,
        think_discounts=discounts,
        task_vs=task_trace.vs,
        think_vs=think_trace.vs,
        task_bootstrap_value=bootstrap,
        think_bootstrap_value=bootstrap,
        think_cost=_COMPUTATION_COST,
    )
    return _SparseTrajectory(
        voc_target=target.net[:2],
        state_return=(
            task_trace.vs + _COMPUTATION_COST * think_trace.vs
        )[:2],
        post_valid=post_valid,
        task_rewards=task_rewards,
    )


class _SharedFeatureVoC(nn.Module):
    """One shared conditioned model, rather than one parameter row per state."""

    def __init__(self, *, erase_conditioning: bool):
        super().__init__()
        self.erase_conditioning = erase_conditioning
        # [constant, difficulty, depth, difficulty * depth] is the smallest
        # shared feature basis that can express useful compute at hard depth 0
        # but not at easy depth 0 or hard depth 1.
        self.value_head = nn.Linear(4, 1)
        self.q_advantage_head = nn.Linear(4, 2)
        self.gate_head = nn.Linear(4, 3)
        for head in (self.value_head, self.q_advantage_head, self.gate_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def features(
        self, difficulty: torch.Tensor, depth: torch.Tensor
    ) -> torch.Tensor:
        difficulty = difficulty.to(dtype=torch.float32)
        depth = depth.to(dtype=torch.float32)
        if self.erase_conditioning:
            difficulty = torch.zeros_like(difficulty)
            depth = torch.zeros_like(depth)
        return torch.stack(
            (
                torch.ones_like(difficulty),
                difficulty,
                depth,
                difficulty * depth,
            ),
            dim=-1,
        )


@dataclass
class _LearnedSnapshot:
    q: torch.Tensor
    continue_probability: torch.Tensor
    raw_logits: torch.Tensor
    root_support: torch.Tensor
    post_support: torch.Tensor
    root_continue_count: int
    post_decision_count: int
    post_after_proceed_count: int
    post_after_reset_count: int


def _sample_control(joint_logits: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(joint_logits.detach().softmax(dim=-1), 1).squeeze(-1)


def _evaluate_states(model: _SharedFeatureVoC) -> tuple[torch.Tensor, ...]:
    # easy root, hard root, and the same hard state after one computation
    difficulty = torch.tensor([0.0, 1.0, 1.0])
    depth = torch.tensor([0.0, 0.0, 1.0])
    feature = model.features(difficulty, depth)
    raw_logits = model.gate_head(feature)
    distribution = compute_voc_gate_distribution(
        raw_logits, temperature=1.0, epsilon=0.0
    )
    gate_probability = torch.stack(
        (distribution.continue_prob, distribution.stop_prob), dim=-1
    )
    raw_advantage = model.q_advantage_head(feature)
    state_value = model.value_head(feature)
    q_values = state_value + raw_advantage - torch.sum(
        gate_probability.detach() * raw_advantage, dim=-1, keepdim=True
    )
    return q_values, distribution.continue_prob, raw_logits


def _train_shared_sparse_mdp(*, erase_conditioning: bool) -> _LearnedSnapshot:
    torch.manual_seed(20260827)
    model = _SharedFeatureVoC(erase_conditioning=erase_conditioning)
    q_optimizer = torch.optim.Adam(model.q_advantage_head.parameters(), lr=0.04)
    policy_value_optimizer = torch.optim.Adam(
        list(model.gate_head.parameters()) + list(model.value_head.parameters()),
        lr=0.025,
    )

    root_support = torch.zeros(3, dtype=torch.long)
    post_support = torch.zeros(3, dtype=torch.long)
    root_continue_count = 0
    post_decision_count = 0
    post_after_proceed_count = 0
    post_after_reset_count = 0

    for _ in range(360):
        batch_n = 192
        # Every update contains exactly as many easy as hard states.  Shuffling
        # prevents batch position or training time from serving as difficulty.
        difficulty = torch.cat(
            (torch.zeros(batch_n // 2), torch.ones(batch_n // 2))
        )
        difficulty = difficulty[torch.randperm(batch_n)]

        root_feature = model.features(difficulty, torch.zeros_like(difficulty))
        root_raw_logits = model.gate_head(root_feature)
        root_distribution = compute_voc_gate_distribution(
            root_raw_logits, temperature=1.0, epsilon=0.02
        )
        root_control = _sample_control(root_distribution.joint_logits)
        root_continue = root_control != util.STOP

        post_feature = model.features(difficulty, torch.ones_like(difficulty))
        post_raw_logits = model.gate_head(post_feature)
        post_distribution = compute_voc_gate_distribution(
            post_raw_logits, temperature=1.0, epsilon=0.02
        )
        post_control = _sample_control(post_distribution.joint_logits)
        post_control = torch.where(
            root_continue,
            post_control,
            torch.full_like(post_control, util.STOP),
        )

        trajectory = _sparse_trajectory(
            difficulty, root_control, post_control
        )
        # Gate rows themselves remain reward-free; learning must use the delayed
        # terminal return rather than an immediate synthetic label.
        assert torch.count_nonzero(trajectory.task_rewards[:-1]) == 0

        feature = torch.stack((root_feature, post_feature))
        raw_logits = torch.stack(
            (root_distribution.joint_logits, post_distribution.joint_logits)
        )
        control = torch.stack((root_control, post_control))
        valid = torch.stack(
            (torch.ones_like(root_continue), trajectory.post_valid)
        )
        state_value = model.value_head(feature).squeeze(-1)
        raw_advantage = model.q_advantage_head(feature.detach())
        result = compute_dynamic_voc_loss(
            voc_q=raw_advantage,
            target_control_logits=raw_logits,
            behavior_control_logits=raw_logits.detach(),
            control_action=control,
            control_valid=valid,
            voc_target=trajectory.voc_target,
            mode="control",
            dueling_q=True,
            voc_state_value=state_value.detach(),
            expected_gate_loss=True,
        )
        valid_n = valid.sum().to(dtype=torch.float32)
        value_loss = torch.sum(
            (state_value - trajectory.state_return.detach()).square()
            * valid.to(dtype=torch.float32)
        ) / valid_n

        q_optimizer.zero_grad()
        (result.q_loss / valid_n).backward()
        q_optimizer.step()

        policy_value_optimizer.zero_grad()
        (result.gate_pg_loss / valid_n + value_loss).backward()
        policy_value_optimizer.step()

        for action in (util.PROCEED, util.RESET, util.STOP):
            root_support[action] += (root_control == action).sum()
            post_support[action] += (
                trajectory.post_valid & (post_control == action)
            ).sum()
        continued_n = int(root_continue.sum())
        root_continue_count += continued_n
        post_decision_count += int(trajectory.post_valid.sum())
        post_after_proceed_count += int((root_control == util.PROCEED).sum())
        post_after_reset_count += int((root_control == util.RESET).sum())

    with torch.no_grad():
        q_values, continue_probability, raw_logits = _evaluate_states(model)
    return _LearnedSnapshot(
        q=q_values,
        continue_probability=continue_probability,
        raw_logits=raw_logits,
        root_support=root_support,
        post_support=post_support,
        root_continue_count=root_continue_count,
        post_decision_count=post_decision_count,
        post_after_proceed_count=post_after_proceed_count,
        post_after_reset_count=post_after_reset_count,
    )


def test_sparse_terminal_return_is_recursively_credited_to_both_gate_calls():
    difficulty = torch.tensor([0, 0, 1, 1, 1])
    root_control = torch.tensor(
        [util.STOP, util.PROCEED, util.STOP, util.RESET, util.PROCEED]
    )
    post_control = torch.tensor(
        [util.STOP, util.STOP, util.STOP, util.STOP, util.RESET]
    )
    trajectory = _sparse_trajectory(difficulty, root_control, post_control)

    assert torch.count_nonzero(trajectory.task_rewards[:-1]) == 0
    torch.testing.assert_close(
        trajectory.voc_target[0],
        torch.tensor([1.0, 0.8, 0.0, 1.8, 1.6]),
    )
    # Invalid post rows are ignored by the learner.  The valid rows show that
    # STOP keeps the terminal return while another computation subtracts cost.
    torch.testing.assert_close(
        trajectory.voc_target[1, trajectory.post_valid],
        torch.tensor([1.0, 2.0, 1.8]),
    )


def test_shared_features_learn_easy_hard_and_post_compute_gate_decisions():
    learned = _train_shared_sparse_mdp(erase_conditioning=False)
    easy_root, hard_root, hard_post = range(3)

    assert learned.q[easy_root, 1] > learned.q[easy_root, 0]
    assert learned.q[hard_root, 0] > learned.q[hard_root, 1]
    assert learned.q[hard_post, 1] > learned.q[hard_post, 0]
    assert learned.continue_probability[easy_root] < 0.2
    assert learned.continue_probability[hard_root] > 0.8
    assert learned.continue_probability[hard_post] < 0.2

    # The policy remains a finite soft distribution, not a greedy Q cutoff.
    assert torch.all(
        (learned.continue_probability > 0.0)
        & (learned.continue_probability < 1.0)
    )
    # Expected gate credit is a common CONTINUE shift and leaves the unchanged
    # conditional PROCEED/RESET bout exactly neutral.
    torch.testing.assert_close(
        learned.raw_logits[:, util.PROCEED],
        learned.raw_logits[:, util.RESET],
        atol=1e-6,
        rtol=0.0,
    )

    assert torch.all(learned.root_support > 0)
    assert torch.all(learned.post_support > 0)
    assert learned.root_continue_count == learned.post_decision_count
    assert learned.post_after_proceed_count > 0
    assert learned.post_after_reset_count > 0
    assert (
        learned.post_after_proceed_count + learned.post_after_reset_count
        == learned.post_decision_count
    )


def test_erasing_difficulty_and_depth_cannot_fake_state_conditioned_voc():
    erased = _train_shared_sparse_mdp(erase_conditioning=True)

    # All three probes become the identical feature vector.  A time-global gate
    # must therefore make the same prediction and cannot satisfy the required
    # easy STOP / hard CONTINUE / post-compute STOP pattern simultaneously.
    torch.testing.assert_close(
        erased.continue_probability,
        erased.continue_probability[0].expand_as(erased.continue_probability),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        erased.q,
        erased.q[0].expand_as(erased.q),
        atol=0.0,
        rtol=0.0,
    )
    passes_all_three = bool(
        (erased.continue_probability[0] < 0.2)
        and (erased.continue_probability[1] > 0.8)
        and (erased.continue_probability[2] < 0.2)
    )
    assert not passes_all_three

