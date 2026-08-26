"""Dynamic Thinker imitation rollouts with human real-action execution.

The runner in this module intentionally uses the ordinary Dynamic ``cenv``.
It changes only the primary action submitted on ``POLICY_REAL`` rows; SEARCH
primary actions and the PROCEED/RESET/STOP control are always produced by the
actor.  A batch is expected to follow this edge-aligned contract::

    obs_seq             [B, L + 2, C, H, W]
    actions_seq         [B, L + 1]
    initial_prev_action [B]
    score_mask          [B, L + 1]  # first column is the burn-in edge

``BehaviorSequenceVectorEnv`` supplies the recorded transition for each edge.
The first edge is executed but not scored, so its human action conditions both
the first scored root and any subtree carried to that root.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import thinker.util as util


def _last(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value[-1] if value.ndim >= 2 else value


def _as_bool_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=torch.bool)


def _as_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    out = np.asarray(value)
    return out.astype(dtype, copy=False) if dtype is not None else out


def _observation_numpy(value: Any) -> np.ndarray:
    """Detach observations without changing their runtime dtype or scale."""
    obs = _as_numpy(value)
    if np.issubdtype(obs.dtype, np.floating) and not np.all(np.isfinite(obs)):
        raise ValueError("behavior observations contain non-finite values")
    return obs


def validate_behavior_batch(batch: Mapping[str, Any]) -> Tuple[int, int]:
    """Validate and return ``(batch_size, edge_count)``.

    The validation is strict because an off-by-one here silently changes the
    supervised label from ``obs[t] -> action[t]`` to the next action.
    """
    required = ("obs_seq", "actions_seq", "initial_prev_action", "score_mask")
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError("behavior batch is missing: %s" % ", ".join(missing))

    obs = batch["obs_seq"]
    actions = batch["actions_seq"]
    initial_prev = batch["initial_prev_action"]
    score_mask = batch["score_mask"]
    if len(obs.shape) < 3:
        raise ValueError("obs_seq must have shape [B,L+2,...]")
    if len(actions.shape) != 2:
        raise ValueError("actions_seq must have shape [B,L+1]")
    batch_size, edge_count = actions.shape
    if obs.shape[0] != batch_size or obs.shape[1] != edge_count + 1:
        raise ValueError(
            "obs_seq/actions_seq mismatch: expected obs second dimension "
            f"{edge_count + 1}, got {tuple(obs.shape[:2])}"
        )
    if tuple(initial_prev.shape) != (batch_size,):
        raise ValueError(
            f"initial_prev_action must have shape {(batch_size,)}, "
            f"got {tuple(initial_prev.shape)}"
        )
    if tuple(score_mask.shape) == (edge_count,):
        pass
    elif tuple(score_mask.shape) != (batch_size, edge_count):
        raise ValueError(
            "score_mask must have shape [L+1] or [B,L+1], got "
            f"{tuple(score_mask.shape)}"
        )
    score_np = _as_numpy(score_mask, np.bool_)
    if score_np.ndim == 1:
        score_np = np.broadcast_to(score_np, (batch_size, edge_count))
    if np.any(score_np[:, 0]):
        raise ValueError("the first (burn-in) transition must not be scored")
    if not np.all(score_np[:, 1:]):
        raise ValueError("all transitions after burn-in must be scored")
    for name in ("rewards_seq", "done_seq", "truncated_seq"):
        if name in batch and batch[name] is not None:
            if tuple(batch[name].shape) != (batch_size, edge_count):
                raise ValueError(
                    f"{name} must have shape {(batch_size, edge_count)}, "
                    f"got {tuple(batch[name].shape)}"
                )
    return int(batch_size), int(edge_count)


def dqfd_margin_per_row(
    logits: torch.Tensor, targets: torch.Tensor, margin: float
) -> torch.Tensor:
    """Return the DQfD large-margin classification loss for every row."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N,A]")
    targets = targets.long().reshape(-1)
    if targets.shape[0] != logits.shape[0]:
        raise ValueError("target count must match logits")
    margin_matrix = torch.full_like(logits, float(margin))
    margin_matrix.scatter_(1, targets[:, None], 0.0)
    selected = logits.gather(1, targets[:, None]).squeeze(1)
    return torch.max(logits + margin_matrix, dim=1).values - selected


def pvp_per_row(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the legacy PVP positive/negative value penalty per example."""
    targets = targets.long().reshape(-1)
    predicted = logits.detach().argmax(dim=-1)
    human_value = logits.gather(1, targets[:, None]).squeeze(1)
    actor_value = logits.gather(1, predicted[:, None]).squeeze(1)
    differs = (predicted != targets).to(logits.dtype)
    return (human_value - 1.0).square() + differs * (actor_value + 1.0).square()


def compute_imitation_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ce_coef: float = 1.0,
    margin: float = 1.0,
    margin_coef: float = 1.0,
    pvp_coef: float = 0.0,
    overall_coef: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute mean supervised losses without any search-control target."""
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("at least one scored [N,A] logit row is required")
    targets = targets.long().reshape(-1)
    raw_nll_rows = F.cross_entropy(logits, targets, reduction="none")
    normalizer = math.log(max(int(logits.shape[-1]), 2))
    normalized_ce_rows = raw_nll_rows / normalizer
    margin_rows = dqfd_margin_per_row(logits, targets, margin)
    pvp_rows = pvp_per_row(logits, targets)
    normalized_ce = normalized_ce_rows.mean()
    margin_loss = margin_rows.mean()
    pvp_loss = pvp_rows.mean()
    loss = float(overall_coef) * (
        float(ce_coef) * normalized_ce
        + float(margin_coef) * margin_loss
        + float(pvp_coef) * pvp_loss
    )
    return {
        "loss": loss,
        "nll_rows": raw_nll_rows,
        "nll": raw_nll_rows.mean(),
        "normalized_ce": normalized_ce,
        "margin_loss": margin_loss,
        "pvp_loss": pvp_loss,
    }


def compute_masked_imitation_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    score_mask: torch.Tensor,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    """Apply the objective only to scored ``[B,E]`` real-policy rows."""
    if logits.ndim != 3 or tuple(targets.shape) != tuple(logits.shape[:2]):
        raise ValueError("logits/targets must have shapes [B,E,A] and [B,E]")
    score_mask = torch.as_tensor(
        score_mask, device=logits.device, dtype=torch.bool
    )
    if score_mask.ndim == 1:
        score_mask = score_mask.unsqueeze(0).expand(logits.shape[0], -1)
    if tuple(score_mask.shape) != tuple(targets.shape):
        raise ValueError("score_mask must broadcast to [B,E]")
    return compute_imitation_objective(
        logits[score_mask], targets[score_mask], **kwargs
    )


def detached_imitation_logit_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """Return stable, detached tail/range diagnostics for scored BC rows.

    Diagnostics run on a CPU float64 copy so large-but-finite actor logits do
    not overflow while computing RMS or cross entropy.  This function is
    deliberately outside the autograd graph and therefore cannot alter the
    imitation objective or its gradients.
    """

    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("at least one scored [N,A] logit row is required")
    targets = targets.detach().reshape(-1).long()
    if targets.shape[0] != logits.shape[0]:
        raise ValueError("target count must match logits")
    if torch.any(targets < 0) or torch.any(targets >= logits.shape[-1]):
        raise ValueError("target action lies outside the scored logit width")

    with torch.no_grad():
        scored_logits = logits.detach().to(device="cpu", dtype=torch.float64)
        scored_targets = targets.to(device="cpu")
        row_index = torch.arange(scored_logits.shape[0])
        target_logits = scored_logits[row_index, scored_targets]
        row_nll = torch.logsumexp(scored_logits, dim=-1) - target_logits

        if scored_logits.shape[-1] > 1:
            other_logits = scored_logits.clone()
            other_logits[row_index, scored_targets] = -torch.inf
            best_other = other_logits.max(dim=-1).values
            target_gap = target_logits - best_other
        else:
            # A one-action categorical policy has no competing logit.
            target_gap = torch.zeros_like(target_logits)

        abs_logits = scored_logits.abs()
        logits_absmax = abs_logits.max()
        # Scaling first keeps RMS finite for values whose square would
        # overflow even in their original floating dtype.
        scale = torch.where(
            logits_absmax == 0,
            torch.ones_like(logits_absmax),
            logits_absmax,
        )
        logits_rms = scale * torch.sqrt(
            torch.mean((scored_logits / scale).square())
        )

        return {
            "nll_max": float(row_nll.max()),
            "nll_p99": float(torch.quantile(row_nll, 0.99)),
            "target_vs_best_other_logit_gap_max": float(target_gap.max()),
            "target_vs_best_other_logit_gap_p99": float(
                torch.quantile(target_gap, 0.99)
            ),
            "scored_logits_absmax": float(logits_absmax),
            "scored_logits_rms": float(logits_rms),
        }


def scale_imitation_for_online_rows(
    mean_loss: torch.Tensor, real_policy_mask: torch.Tensor
) -> torch.Tensor:
    """Convert a mean BC objective to Thinker's summed online convention."""
    return mean_loss * real_policy_mask.to(mean_loss.dtype).sum()


def imitation_checkpoint_state(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Read optional imitation state from old or new actor checkpoints."""
    return {
        "update_count": int(checkpoint.get("imitation_update_count", 0)),
        "schedule_step": int(checkpoint.get("imitation_schedule_step", 0)),
        "rng_state": checkpoint.get("imitation_rng_state"),
        "data_signature": checkpoint.get("imitation_data_signature"),
        "action_prior_ema": checkpoint.get("action_prior_ema"),
    }


@dataclass(frozen=True)
class ExecutionDecision:
    """Actor proposal and the separate action submitted to ``cenv``."""

    primary_proposal: torch.Tensor
    primary_argmax: torch.Tensor
    search_control_proposal: torch.Tensor
    execution_primary: torch.Tensor
    execution_control: torch.Tensor
    real_policy_mask: torch.Tensor


class HumanActionExecutionAdapter:
    """Teacher-force only real-policy rows and preserve accepted WAIT tokens."""

    def __init__(self, initial_prev_action: torch.Tensor):
        initial_prev_action = torch.as_tensor(initial_prev_action).long().reshape(-1)
        self.effective_primary = initial_prev_action.detach().clone()
        self.effective_control = torch.zeros_like(self.effective_primary)

    @staticmethod
    def _replace_rows(
        execution: torch.Tensor,
        mask: torch.Tensor,
        human_action: torch.Tensor,
    ) -> torch.Tensor:
        execution = execution.detach().clone()
        human_action = human_action.to(execution.device, dtype=execution.dtype)
        if execution.ndim == 1:
            execution[mask] = human_action[mask]
        else:
            if human_action.ndim == 1:
                human_action = human_action.unsqueeze(-1)
            execution[mask] = human_action[mask]
        return execution

    def prepare(
        self, actor_out: Any, human_action: torch.Tensor
    ) -> ExecutionDecision:
        """Clone an execution action without changing any ``ActorOut`` tensor."""
        primary, control = actor_out.action
        primary_proposal = primary.detach().clone()
        control_proposal = control.detach().clone()
        policy_type = _last(actor_out.policy_type).to(primary.device)
        policy_valid = _last(actor_out.policy_valid).to(primary.device).bool()
        real_mask = policy_valid & (policy_type == util.POLICY_REAL)
        execution_primary = self._replace_rows(
            primary_proposal, real_mask, human_action
        )
        pri_logits = _last(actor_out.pri_param)
        if pri_logits.ndim == 3 and pri_logits.shape[-2] == 1:
            pri_logits = pri_logits[:, 0]
        primary_argmax = pri_logits.detach().argmax(dim=-1)
        return ExecutionDecision(
            primary_proposal=primary_proposal,
            primary_argmax=primary_argmax,
            search_control_proposal=control_proposal,
            execution_primary=execution_primary,
            execution_control=control_proposal.detach().clone(),
            real_policy_mask=real_mask,
        )

    def observe(self, info: Mapping[str, Any]) -> Dict[str, Any]:
        """Update accepted/executed caches and add effective tokens to info."""
        out = dict(info)
        device = self.effective_primary.device
        accepted = info.get("accepted_primary_action")
        if accepted is not None:
            accepted = torch.as_tensor(accepted, device=device).long()
            mask = accepted >= 0
            self.effective_primary[mask] = accepted[mask]
        executed = info.get("executed_primary_action")
        real_transition = info.get("real_transition")
        if executed is not None and real_transition is not None:
            executed = torch.as_tensor(executed, device=device).long()
            mask = torch.as_tensor(real_transition, device=device).bool()
            self.effective_primary[mask] = executed[mask]
        accepted_control = info.get("accepted_control")
        if accepted_control is not None:
            accepted_control = torch.as_tensor(
                accepted_control, device=device
            ).long()
            mask = accepted_control >= 0
            self.effective_control[mask] = accepted_control[mask]
        out["effective_primary_action"] = self.effective_primary.detach().clone()
        out["effective_search_control"] = self.effective_control.detach().clone()
        return out


@dataclass
class DynamicImitationResult:
    """Differentiable loss plus detached rollout diagnostics."""

    loss: torch.Tensor
    normalized_ce: torch.Tensor
    margin_loss: torch.Tensor
    pvp_loss: torch.Tensor
    nll_sum: torch.Tensor
    count: int
    accuracy: float
    sampled_accuracy: float
    logits: torch.Tensor
    targets: torch.Tensor
    all_logits: torch.Tensor
    all_proposal: torch.Tensor
    all_argmax: torch.Tensor
    all_executed: torch.Tensor
    score_mask: torch.Tensor
    per_stage_nll: torch.Tensor
    root_carried: torch.Tensor
    carried_descendant_visit_count: torch.Tensor
    carried_descendant_expanded_count: torch.Tensor
    useful_carry: torch.Tensor
    proposal: torch.Tensor
    argmax: torch.Tensor
    executed: torch.Tensor
    burnin_proposal: torch.Tensor
    burnin_executed: torch.Tensor
    augmented_steps: int

    def detached_metrics(self) -> Dict[str, float]:
        count = max(self.count, 1)
        metrics = {
            "loss": float(self.loss.detach().cpu()),
            "nll": float(self.nll_sum.detach().cpu()) / count,
            "normalized_ce": float(self.normalized_ce.detach().cpu()),
            "margin_loss": float(self.margin_loss.detach().cpu()),
            "pvp_loss": float(self.pvp_loss.detach().cpu()),
            "accuracy": self.accuracy,
            "sampled_accuracy": self.sampled_accuracy,
            "root_carried_rate": float(
                self.root_carried.float().mean().detach().cpu()
            ),
            "useful_carry_rate": float(
                self.useful_carry.float().mean().detach().cpu()
            ),
            "carried_descendant_visit_count_mean": float(
                self.carried_descendant_visit_count.float().mean().detach().cpu()
            ),
            "carried_descendant_expanded_count_mean": float(
                self.carried_descendant_expanded_count.float().mean().detach().cpu()
            ),
            "count": float(self.count),
            "augmented_steps": float(self.augmented_steps),
        }
        metrics.update(
            detached_imitation_logit_metrics(self.logits, self.targets)
        )
        return metrics


def _move_env_out(env_out: Any, device: torch.device) -> Any:
    return util.tuple_map(
        env_out,
        lambda value: value.to(device) if torch.is_tensor(value) else value,
    )


def _scatter_stage(
    existing: Optional[torch.Tensor],
    indices: torch.Tensor,
    values: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if existing is None:
        existing = values.new_zeros((batch_size,) + tuple(values.shape[1:]))
    return existing.index_copy(0, indices, values)


class DynamicImitationRunner:
    """Run a behavioral sequence through Dynamic search and score real heads.

    ``model_net`` is treated as a frozen world model.  A runner may be reused;
    its underlying behavioral environment is updated and ``cenv.reset`` starts
    a new tree forest on every call.
    """

    def __init__(
        self,
        actor_net: torch.nn.Module,
        model_net: torch.nn.Module,
        flags: Any,
        device: Optional[torch.device] = None,
    ):
        self.actor_net = actor_net
        self.model_net = model_net
        self.flags = flags
        self.device = torch.device(device) if device is not None else next(
            actor_net.parameters()
        ).device
        try:
            self.model_device = next(model_net.parameters()).device
        except StopIteration:
            self.model_device = self.device
        self._planner = None
        self._behavior_env = None
        self._planner_key = None
        self._validate_flags()
        self._validate_component_contracts()
        self.model_net.eval()
        for parameter in self.model_net.parameters():
            parameter.requires_grad_(False)

    def _validate_flags(self) -> None:
        if not util.dynamic_search_enabled(self.flags):
            raise ValueError("dynamic imitation requires dynamic_search=true")
        if not bool(getattr(self.flags, "sep_im_head", False)):
            raise ValueError("dynamic imitation requires sep_im_head=true")
        max_search_steps = int(getattr(self.flags, "max_search_steps", -1))
        if max_search_steps <= 0:
            raise ValueError(
                "dynamic imitation requires a positive max_search_steps watchdog"
            )
        if not bool(getattr(self.actor_net, "discrete_action", True)):
            raise ValueError("dynamic imitation currently requires discrete actions")

    def _validate_component_contracts(self) -> None:
        from gymnasium import spaces

        actor_action_n = int(self.actor_net.num_actions)
        model_action_n = int(getattr(self.model_net, "num_actions", actor_action_n))
        if actor_action_n != model_action_n:
            raise ValueError(
                "Actor and behavioral ModelNet action counts disagree: "
                f"actor={actor_action_n}, model={model_action_n}"
            )
        primary_space = getattr(self.actor_net, "pri_action_space", None)
        if primary_space is not None:
            if not isinstance(primary_space, spaces.Discrete):
                raise TypeError("dynamic imitation requires a Discrete action space")
            if (
                int(getattr(primary_space, "start", 0)) != 0
                or int(primary_space.n) != actor_action_n
            ):
                raise ValueError(
                    "Actor primary action-space metadata disagree with its head"
                )
        model_stack_n = getattr(self.model_net, "frame_stack_n", None)
        if model_stack_n is not None and int(model_stack_n) != int(
            self.flags.frame_stack_n
        ):
            raise ValueError(
                "behavioral ModelNet frame-stack metadata disagree with flags"
            )
        actor_obs = getattr(self.actor_net, "online_real_state_space", None)
        model_obs = getattr(self.model_net, "observation_space", None)
        if actor_obs is not None and model_obs is not None:
            if (
                tuple(actor_obs.shape) != tuple(model_obs.shape)
                or np.dtype(actor_obs.dtype) != np.dtype(model_obs.dtype)
                or not np.array_equal(actor_obs.low, model_obs.low)
                or not np.array_equal(actor_obs.high, model_obs.high)
            ):
                raise ValueError(
                    "Actor online observation contract and behavioral ModelNet "
                    "contract disagree"
                )

    def _make_or_update_planner(
        self, batch: Mapping[str, Any], tree_carry: bool
    ) -> Any:
        # Import lazily: unit tests for the loss/adapter do not require a built
        # Cython extension, and the data module can be used independently.
        from gymnasium import spaces
        from thinker.cenv import cModelWrapper
        from thinker.dataset_env import BehaviorSequenceVectorEnv

        obs = _observation_numpy(batch["obs_seq"])
        actions = _as_numpy(batch["actions_seq"], np.int64)
        sequence_kwargs = {
            "obs_seq": obs,
            "actions_seq": actions,
            "rewards_seq": _as_numpy(
                batch.get("rewards_seq", np.zeros_like(actions)), np.float32
            ),
            "done_seq": _as_numpy(
                batch.get("done_seq", np.zeros_like(actions)), np.bool_
            ),
            "truncated_seq": _as_numpy(
                batch.get("truncated_seq", np.zeros_like(actions)), np.bool_
            ),
            "initial_prev_action": _as_numpy(
                batch["initial_prev_action"], np.int64
            ),
            "score_mask": _as_numpy(batch["score_mask"], np.bool_),
        }
        model_observation_space = getattr(
            self.model_net, "observation_space", None
        )
        batch_size, edge_count = actions.shape
        key = (
            batch_size,
            edge_count,
            bool(tree_carry),
            str(self.model_device),
            tuple(obs.shape[2:]),
            np.dtype(obs.dtype).str,
            int(self.actor_net.num_actions),
            int(self.flags.frame_stack_n),
        )
        if self._planner is None or key != self._planner_key:
            self.close()
            constructor_kwargs = {
                **sequence_kwargs,
                "num_actions": int(self.actor_net.num_actions),
            }
            if model_observation_space is not None:
                constructor_kwargs["observation_space"] = model_observation_space
            self._behavior_env = BehaviorSequenceVectorEnv(**constructor_kwargs)
            replay_primary = self._behavior_env.action_space[0]
            if (
                not isinstance(replay_primary, spaces.Discrete)
                or int(getattr(replay_primary, "start", 0)) != 0
                or int(replay_primary.n) != int(self.actor_net.num_actions)
            ):
                raise ValueError(
                    "behavior replay and Actor action-space contracts disagree"
                )
            planner_flags = copy.copy(self.flags)
            planner_flags.tree_carry = bool(tree_carry)
            self._planner = cModelWrapper(
                env=self._behavior_env,
                env_n=batch_size,
                flags=planner_flags,
                model_net=self.model_net,
                device=self.model_device,
                timing=False,
            )
            planner_primary = self._planner.action_space[0][0]
            planner_control = self._planner.action_space[1][0]
            if (
                not isinstance(planner_primary, spaces.Discrete)
                or int(getattr(planner_primary, "start", 0)) != 0
                or int(planner_primary.n) != int(self.actor_net.num_actions)
            ):
                raise ValueError("cenv and Actor primary action counts disagree")
            if (
                not isinstance(planner_control, spaces.Discrete)
                or int(getattr(planner_control, "start", 0)) != 0
                or int(planner_control.n) != 3
            ):
                raise ValueError(
                    "dynamic cenv must expose zero-based PROCEED/RESET/STOP controls"
                )
            self._planner_key = key
        else:
            self._behavior_env.update_sequences(**sequence_kwargs)
        return self._planner

    def rollout(
        self,
        batch: Mapping[str, Any],
        tree_carry: bool = True,
        training: bool = True,
    ) -> DynamicImitationResult:
        batch_size, edge_count = validate_behavior_batch(batch)
        actions = torch.as_tensor(
            batch["actions_seq"], device=self.device, dtype=torch.long
        )
        initial_prev = torch.as_tensor(
            batch["initial_prev_action"], device=self.device, dtype=torch.long
        )
        score_mask = torch.as_tensor(
            batch["score_mask"], device=self.device, dtype=torch.bool
        )
        if score_mask.ndim == 1:
            score_mask = score_mask.unsqueeze(0).expand(batch_size, -1)

        planner = self._make_or_update_planner(batch, tree_carry)
        states, info = planner.reset(
            self.model_net,
            initial_action=_as_numpy(batch["initial_prev_action"], np.int64),
        )
        env_out = util.init_env_out(
            states,
            info,
            self.flags,
            dim_actions=self.actor_net.dim_actions,
            tuple_action=self.actor_net.tuple_action,
        )
        env_out = _move_env_out(env_out, self.device)
        actor_state = self.actor_net.initial_state(
            batch_size=batch_size, device=self.device
        )
        adapter = HumanActionExecutionAdapter(initial_prev.to(self.device))

        stage_logits: Sequence[Optional[torch.Tensor]] = [None] * edge_count
        stage_proposals: Sequence[Optional[torch.Tensor]] = [None] * edge_count
        stage_argmax: Sequence[Optional[torch.Tensor]] = [None] * edge_count
        stage_root_carried: Sequence[Optional[torch.Tensor]] = [None] * edge_count
        stage_carried_descendant_visit_count: Sequence[
            Optional[torch.Tensor]
        ] = [None] * edge_count
        stage_carried_descendant_expanded_count: Sequence[
            Optional[torch.Tensor]
        ] = [None] * edge_count
        stage_useful_carry: Sequence[Optional[torch.Tensor]] = [None] * edge_count
        filled = torch.zeros(
            (edge_count, batch_size), dtype=torch.bool, device=self.device
        )
        root_carried_for_current = torch.zeros(
            batch_size, dtype=torch.bool, device=self.device
        )
        carried_descendant_visit_count_for_current = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        )
        carried_descendant_expanded_count_for_current = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        )
        useful_carry_for_current = torch.zeros(
            batch_size, dtype=torch.bool, device=self.device
        )
        # ``cModelWrapper`` is normally wrapped by PostWrapper/Main.Env.  The
        # direct behavioral runner mirrors their stage-return bookkeeping so
        # util.create_env_out receives a complete reward-channel contract.
        episode_accumulators = {
            "im": torch.zeros(
                batch_size, dtype=torch.float32, device=self.model_device
            ),
            "cur": torch.zeros(
                batch_size, dtype=torch.float32, device=self.model_device
            ),
            "think": torch.zeros(
                batch_size, dtype=torch.float32, device=self.model_device
            ),
        }
        cursor = 0
        max_search_steps = int(self.flags.max_search_steps)
        watchdog = edge_count * (max_search_steps + 3)
        augmented_steps = 0

        while cursor < edge_count:
            if augmented_steps >= watchdog:
                raise RuntimeError(
                    "dynamic imitation watchdog exceeded "
                    f"({augmented_steps}/{watchdog}); phase/barrier did not advance"
                )
            actor_out, actor_state = self.actor_net(
                env_out,
                actor_state,
                compute_loss=False,
                greedy=not training,
            )
            decision = adapter.prepare(actor_out, actions[:, cursor])
            real_indices = torch.nonzero(
                decision.real_policy_mask, as_tuple=False
            ).flatten()
            if real_indices.numel():
                logits = _last(actor_out.pri_param)
                if logits.ndim == 3 and logits.shape[-2] == 1:
                    logits = logits[:, 0]
                stage_logits[cursor] = _scatter_stage(
                    stage_logits[cursor],
                    real_indices,
                    logits.index_select(0, real_indices),
                    batch_size,
                )
                stage_proposals[cursor] = _scatter_stage(
                    stage_proposals[cursor],
                    real_indices,
                    decision.primary_proposal.index_select(0, real_indices),
                    batch_size,
                )
                stage_argmax[cursor] = _scatter_stage(
                    stage_argmax[cursor],
                    real_indices,
                    decision.primary_argmax.index_select(0, real_indices),
                    batch_size,
                )
                stage_root_carried[cursor] = _scatter_stage(
                    stage_root_carried[cursor],
                    real_indices,
                    root_carried_for_current.index_select(0, real_indices),
                    batch_size,
                )
                stage_carried_descendant_visit_count[cursor] = _scatter_stage(
                    stage_carried_descendant_visit_count[cursor],
                    real_indices,
                    carried_descendant_visit_count_for_current.index_select(
                        0, real_indices
                    ),
                    batch_size,
                )
                stage_carried_descendant_expanded_count[cursor] = _scatter_stage(
                    stage_carried_descendant_expanded_count[cursor],
                    real_indices,
                    carried_descendant_expanded_count_for_current.index_select(
                        0, real_indices
                    ),
                    batch_size,
                )
                stage_useful_carry[cursor] = _scatter_stage(
                    stage_useful_carry[cursor],
                    real_indices,
                    useful_carry_for_current.index_select(0, real_indices),
                    batch_size,
                )
                filled[cursor, real_indices] = True

            with torch.no_grad():
                states, reward, done, truncated, raw_info = planner.step(
                    (
                        decision.execution_primary,
                        decision.execution_control,
                    ),
                    self.model_net,
                )
            info = adapter.observe(raw_info)
            real_done = torch.as_tensor(
                info.get("real_done", done),
                device=self.model_device,
                dtype=torch.bool,
            ).reshape(-1)
            for prefix in ("im", "cur", "think"):
                value = info.get(prefix + "_reward")
                if value is not None:
                    value = torch.as_tensor(
                        value, device=self.model_device, dtype=torch.float32
                    )
                    while value.ndim > 1:
                        value = value[..., 0]
                    finite = torch.isfinite(value)
                    episode_accumulators[prefix][finite] += value[finite]
                info[prefix + "_episode_return"] = (
                    episode_accumulators[prefix].clone()
                )
                episode_accumulators[prefix][real_done] = 0.0
            stage_end = info.get("stage_end")
            if stage_end is not None:
                stage_end_mask = torch.as_tensor(
                    stage_end, device=self.model_device, dtype=torch.bool
                )
                episode_accumulators["im"][stage_end_mask] = 0.0
                episode_accumulators["think"][stage_end_mask] = 0.0
            next_env_out = util.create_env_out(
                (
                    decision.execution_primary,
                    decision.execution_control,
                ),
                states,
                reward,
                done,
                truncated,
                info,
                flags=self.flags,
            )
            env_out = _move_env_out(next_env_out, self.device)
            augmented_steps += 1

            real_transition = _as_bool_tensor(
                info.get("real_transition", False), device=self.device
            ).reshape(-1)
            if torch.any(real_transition):
                if real_transition.shape != (batch_size,) or not torch.all(
                    real_transition
                ):
                    raise RuntimeError(
                        "behavior sequence cursor may advance only on a "
                        "full-batch real_transition"
                    )
                if not torch.all(filled[cursor]):
                    missing = torch.nonzero(
                        ~filled[cursor], as_tuple=False
                    ).flatten().tolist()
                    raise RuntimeError(
                        f"real transition {cursor} lacks POLICY_REAL logits "
                        f"for batch rows {missing}"
                    )
                executed = torch.as_tensor(
                    info.get("executed_primary_action"),
                    device=self.device,
                    dtype=torch.long,
                )
                if not torch.equal(executed, actions[:, cursor]):
                    raise RuntimeError(
                        "cenv executed a non-human real action during imitation"
                    )
                root_carried_for_current = _as_bool_tensor(
                    info.get(
                        "root_carried",
                        torch.zeros(batch_size, dtype=torch.bool),
                    ),
                    device=self.device,
                ).reshape(-1)
                carried_descendant_visit_count_for_current = torch.as_tensor(
                    info.get(
                        "carried_descendant_visit_count",
                        torch.zeros(batch_size, dtype=torch.long),
                    ),
                    device=self.device,
                    dtype=torch.long,
                ).reshape(-1)
                carried_descendant_expanded_count_for_current = torch.as_tensor(
                    info.get(
                        "carried_descendant_expanded_count",
                        torch.zeros(batch_size, dtype=torch.long),
                    ),
                    device=self.device,
                    dtype=torch.long,
                ).reshape(-1)
                useful_carry_for_current = _as_bool_tensor(
                    info.get(
                        "useful_carry",
                        root_carried_for_current
                        & (carried_descendant_visit_count_for_current > 0),
                    ),
                    device=self.device,
                ).reshape(-1)
                for name, value in (
                    (
                        "carried_descendant_visit_count",
                        carried_descendant_visit_count_for_current,
                    ),
                    (
                        "carried_descendant_expanded_count",
                        carried_descendant_expanded_count_for_current,
                    ),
                ):
                    if value.shape != (batch_size,) or torch.any(value < 0):
                        raise RuntimeError(
                            f"{name} must be a nonnegative batch vector"
                        )
                    if torch.any(value[~root_carried_for_current] != 0):
                        raise RuntimeError(
                            f"{name} was nonzero while root_carried=false"
                        )
                if torch.any(
                    carried_descendant_expanded_count_for_current
                    > carried_descendant_visit_count_for_current
                ) or torch.any(
                    (carried_descendant_expanded_count_for_current > 0)
                    != (carried_descendant_visit_count_for_current > 0)
                ):
                    raise RuntimeError(
                        "carried descendant expanded/visit counts violate "
                        "the cenv tree contract"
                    )
                expected_useful_carry = root_carried_for_current & (
                    carried_descendant_visit_count_for_current > 0
                )
                if not torch.equal(
                    useful_carry_for_current, expected_useful_carry
                ):
                    raise RuntimeError(
                        "useful_carry must equal root_carried and "
                        "carried_descendant_visit_count > 0"
                    )
                cursor += 1

        logits_all = torch.stack(tuple(stage_logits), dim=1)
        proposals_all = torch.stack(tuple(stage_proposals), dim=1).long()
        argmax_all = torch.stack(tuple(stage_argmax), dim=1).long()
        root_carried_all = torch.stack(tuple(stage_root_carried), dim=1).bool()
        carried_descendant_visit_count_all = torch.stack(
            tuple(stage_carried_descendant_visit_count), dim=1
        ).long()
        carried_descendant_expanded_count_all = torch.stack(
            tuple(stage_carried_descendant_expanded_count), dim=1
        ).long()
        useful_carry_all = torch.stack(tuple(stage_useful_carry), dim=1).bool()
        scored_logits = logits_all[score_mask]
        scored_targets = actions[score_mask]
        objective = compute_masked_imitation_objective(
            logits_all,
            actions,
            score_mask,
            ce_coef=float(getattr(self.flags, "icopro_action_diff_coef", 1.0)),
            margin=float(getattr(self.flags, "icopro_margin", 1.0)),
            margin_coef=float(getattr(self.flags, "icopro_margin_coef", 1.0)),
            pvp_coef=float(getattr(self.flags, "icopro_pvp_coef", 0.0)),
            overall_coef=float(getattr(self.flags, "icopro_coef", 1.0)),
        )
        per_stage_nll_all = F.cross_entropy(
            logits_all.flatten(0, 1), actions.flatten(), reduction="none"
        ).view(batch_size, edge_count)
        # The canonical mask is identical across rows; retain only scored
        # columns to make evaluator output [B,L].
        scored_columns = torch.all(score_mask, dim=0)
        per_stage_nll = per_stage_nll_all[:, scored_columns]
        scored_proposals = proposals_all[score_mask]
        scored_argmax = argmax_all[score_mask]
        scored_root_carried = root_carried_all[:, scored_columns]
        scored_carried_descendant_visit_count = (
            carried_descendant_visit_count_all[:, scored_columns]
        )
        scored_carried_descendant_expanded_count = (
            carried_descendant_expanded_count_all[:, scored_columns]
        )
        scored_useful_carry = useful_carry_all[:, scored_columns]
        accuracy = float(
            (scored_argmax == scored_targets).float().mean().detach().cpu()
        )
        sampled_accuracy = float(
            (scored_proposals == scored_targets).float().mean().detach().cpu()
        )
        return DynamicImitationResult(
            loss=objective["loss"],
            normalized_ce=objective["normalized_ce"],
            margin_loss=objective["margin_loss"],
            pvp_loss=objective["pvp_loss"],
            nll_sum=objective["nll_rows"].sum().detach(),
            count=int(scored_targets.numel()),
            accuracy=accuracy,
            sampled_accuracy=sampled_accuracy,
            logits=scored_logits,
            targets=scored_targets.detach(),
            all_logits=logits_all,
            all_proposal=proposals_all.detach(),
            all_argmax=argmax_all.detach(),
            all_executed=actions.detach(),
            score_mask=score_mask.detach(),
            per_stage_nll=per_stage_nll.detach(),
            root_carried=scored_root_carried.detach(),
            carried_descendant_visit_count=(
                scored_carried_descendant_visit_count.detach()
            ),
            carried_descendant_expanded_count=(
                scored_carried_descendant_expanded_count.detach()
            ),
            useful_carry=scored_useful_carry.detach(),
            proposal=proposals_all[:, scored_columns].detach(),
            argmax=argmax_all[:, scored_columns].detach(),
            executed=actions[:, scored_columns].detach(),
            burnin_proposal=proposals_all[:, 0].detach(),
            burnin_executed=actions[:, 0].detach(),
            augmented_steps=augmented_steps,
        )

    def close(self) -> None:
        if self._planner is not None and hasattr(self._planner, "close"):
            self._planner.close()
        self._planner = None
        self._behavior_env = None
        self._planner_key = None
