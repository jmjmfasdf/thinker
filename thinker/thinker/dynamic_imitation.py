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


# Exact recorder-only operation codes emitted by ``cenv``.  They are kept
# separate from the public ``step_status`` clock, which intentionally merges
# several search operations for actor-learning semantics.
TRACE_REAL_TRANSITION = 1
TRACE_SEARCH_REVISIT = 2
TRACE_SEARCH_TERMINAL = 3
TRACE_SEARCH_EXPANSION = 4
TRACE_SEARCH_RESET = 5
TRACE_SEARCH_STOP = 6
TRACE_REAL_ACTION_STORED = 7
TRACE_WAIT = 8


def _trace_scalar(value: Any, index: int = 0) -> Any:
    if torch.is_tensor(value):
        value = value.detach().reshape(-1)
        if value.numel() <= index:
            raise IndexError("trace field is shorter than its batch index")
        return value[index].item()
    value = np.asarray(value).reshape(-1)
    if value.size <= index:
        raise IndexError("trace field is shorter than its batch index")
    return value[index].item()


def dynamic_trace_predicates(
    trace_snapshot: Mapping[str, Any],
    observed_info: Mapping[str, Any],
    *,
    batch_index: int = 0,
) -> Dict[str, Any]:
    """Classify one native Dynamic transition without legacy frame aliases.

    In particular, RESET is a property of a search computation, not a second
    row, and STOP is a policy-state transition with no tree mutation.  Invalid
    combinations fail closed so a recorder cannot silently relabel rows.
    """

    status = int(_trace_scalar(trace_snapshot["internal_status"], batch_index))
    accepted_control = int(
        _trace_scalar(trace_snapshot["accepted_control"], batch_index)
    )
    real_transition = bool(
        _trace_scalar(observed_info.get("real_transition", False), batch_index)
    )
    tree_token_valid = bool(
        _trace_scalar(observed_info.get("tree_token_valid", False), batch_index)
    )
    stage_end = bool(
        _trace_scalar(observed_info.get("stage_end", False), batch_index)
    )
    stored_action = bool(
        _trace_scalar(observed_info.get("stored_action_mask", False), batch_index)
    )
    is_search = status in (
        TRACE_SEARCH_REVISIT,
        TRACE_SEARCH_TERMINAL,
        TRACE_SEARCH_EXPANSION,
        TRACE_SEARCH_RESET,
    )
    is_reset_control = is_search and accepted_control == util.RESET

    if status == TRACE_REAL_TRANSITION:
        if not real_transition:
            raise RuntimeError("internal real transition lacks real_transition flag")
        kind = "real_transition"
    elif status == TRACE_SEARCH_REVISIT:
        kind = "search_revisit"
    elif status == TRACE_SEARCH_TERMINAL:
        kind = "search_terminal"
    elif status == TRACE_SEARCH_EXPANSION:
        kind = "search_expansion"
    elif status == TRACE_SEARCH_RESET:
        kind = "search_reset"
    elif status == TRACE_SEARCH_STOP:
        if not stage_end or accepted_control != util.STOP:
            raise RuntimeError("internal STOP lacks its stage/control flags")
        kind = "search_stop"
    elif status == TRACE_REAL_ACTION_STORED:
        if not stored_action:
            raise RuntimeError("stored-action status lacks stored_action_mask")
        kind = "real_action_stored"
    elif status == TRACE_WAIT:
        kind = "wait"
    else:
        raise RuntimeError(f"unsupported dynamic internal status {status}")

    if is_search and not tree_token_valid:
        raise RuntimeError("search computation lacks a valid tree token")
    if real_transition and status != TRACE_REAL_TRANSITION:
        raise RuntimeError("real_transition flag disagrees with internal status")

    return {
        "kind": kind,
        "internal_status": status,
        "is_search_computation": is_search,
        "is_search_expansion": status in (
            TRACE_SEARCH_EXPANSION, TRACE_SEARCH_TERMINAL
        ),
        "is_search_revisit": status == TRACE_SEARCH_REVISIT,
        "is_reset_control": is_reset_control,
        "is_stop": status == TRACE_SEARCH_STOP,
        "is_real_transition": status == TRACE_REAL_TRANSITION,
        "is_real_action_stored": status == TRACE_REAL_ACTION_STORED,
        "is_wait": status == TRACE_WAIT,
        "tree_token_valid": tree_token_valid,
        "stage_end": stage_end,
    }


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


def validate_stream_behavior_episode(
    batch: Mapping[str, Any],
) -> Tuple[int, int]:
    """Validate a complete batch-one episode without reading its observations."""

    if "observation_provider" not in batch:
        batch_size, edge_count = validate_behavior_batch(batch)
        if batch_size != 1:
            raise ValueError("stream_rollout_episode requires batch_size=1")
        return batch_size, edge_count

    required = ("actions_seq", "initial_prev_action", "score_mask")
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError("behavior episode is missing: %s" % ", ".join(missing))
    actions = _as_numpy(batch["actions_seq"], np.int64)
    if actions.ndim == 1:
        edge_count = int(actions.shape[0])
    elif actions.ndim == 2 and actions.shape[0] == 1:
        edge_count = int(actions.shape[1])
    else:
        raise ValueError("lazy actions_seq must have shape [E] or [1,E]")
    if edge_count < 1:
        raise ValueError("lazy actions_seq must contain at least one edge")
    previous = _as_numpy(batch["initial_prev_action"], np.int64).reshape(-1)
    if previous.shape != (1,):
        raise ValueError("lazy initial_prev_action must be scalar or shape [1]")
    score_mask = _as_numpy(batch["score_mask"], np.bool_)
    if score_mask.ndim == 2 and score_mask.shape[0] == 1:
        score_mask = score_mask[0]
    if score_mask.shape != (edge_count,):
        raise ValueError("lazy score_mask must have shape [E] or [1,E]")
    if score_mask[0] or not np.all(score_mask[1:]):
        raise ValueError("score_mask must be [False, True, ..., True]")
    for name in ("rewards_seq", "done_seq", "truncated_seq"):
        if batch.get(name) is None:
            continue
        value = _as_numpy(batch[name])
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.shape != (edge_count,):
            raise ValueError(f"lazy {name} must have shape [E] or [1,E]")
    provider_shape = getattr(batch["observation_provider"], "shape", None)
    if provider_shape is not None and int(provider_shape[0]) != edge_count + 1:
        raise ValueError("observation provider must contain exactly E+1 frames")
    return 1, edge_count


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
    # Inactive terms are diagnostics, not zero-weight autograd branches:
    # IEEE arithmetic makes 0 * inf/nan non-finite.  Float64 diagnostics
    # also keep the square of every finite FP32 logit representable.
    margin_active = float(overall_coef) != 0.0 and float(margin_coef) != 0.0
    pvp_active = float(overall_coef) != 0.0 and float(pvp_coef) != 0.0
    diagnostic_logits = logits.detach().to(dtype=torch.float64)
    margin_rows = dqfd_margin_per_row(
        logits if margin_active else diagnostic_logits, targets, margin
    )
    pvp_rows = pvp_per_row(
        logits if pvp_active else diagnostic_logits, targets
    )
    normalized_ce = normalized_ce_rows.mean()
    margin_loss = margin_rows.mean()
    pvp_loss = pvp_rows.mean()
    # An empty sum supplies a differentiable zero without reading any
    # potentially non-finite logit when every objective term is disabled.
    loss = logits.reshape(-1)[:0].sum()
    if float(overall_coef) != 0.0:
        for coefficient, term in (
            (ce_coef, normalized_ce),
            (margin_coef, margin_loss),
            (pvp_coef, pvp_loss),
        ):
            if float(coefficient) != 0.0:
                loss = loss + float(coefficient) * term
        loss = float(overall_coef) * loss
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


def resolve_noop_action_index(
    action_meanings: Optional[Sequence[str]], num_actions: int
) -> Optional[int]:
    """Resolve a unique runtime-declared NOOP action without assuming index 0.

    ``None`` means that the runtime did not declare action semantics.  A
    declared action table is validated exactly against the live policy width;
    malformed or ambiguous metadata must not silently produce a NOOP metric.
    An action table with no ``NOOP`` entry is valid and represents an
    environment with no supported NOOP-frequency diagnostic.
    """

    if isinstance(num_actions, (bool, np.bool_)) or not isinstance(
        num_actions, (int, np.integer)
    ):
        raise TypeError("num_actions must be an integer")
    num_actions = int(num_actions)
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")
    if action_meanings is None:
        return None
    if isinstance(action_meanings, (str, bytes)):
        raise TypeError("action_meanings must be a sequence of names")
    meanings = tuple(action_meanings)
    if len(meanings) != num_actions:
        raise ValueError(
            "runtime action meanings must match the policy width: "
            f"{len(meanings)} != {num_actions}"
        )
    if any(not isinstance(name, str) or not name.strip() for name in meanings):
        raise ValueError("runtime action meanings must be non-empty strings")
    matches = [
        index
        for index, name in enumerate(meanings)
        if name.strip().upper() == "NOOP"
    ]
    if len(matches) > 1:
        raise ValueError("runtime action meanings contain multiple NOOP actions")
    return matches[0] if matches else None


def detached_behavioral_action_metrics(
    targets: torch.Tensor,
    argmax_actions: torch.Tensor,
    sampled_actions: torch.Tensor,
    *,
    num_actions: int,
    noop_action_index: Optional[int],
) -> Dict[str, float]:
    """Count-based BC accuracy and NOOP diagnostics for scored human rows.

    Every ratio is accompanied by its numerator and denominator so multiple
    learner rows can be aggregated without averaging batch means.  Unsupported
    or empty NOOP diagnostics deliberately return finite zero ratios with a
    zero ``noop_support_count``.
    """

    if isinstance(num_actions, (bool, np.bool_)) or not isinstance(
        num_actions, (int, np.integer)
    ):
        raise TypeError("num_actions must be an integer")
    num_actions = int(num_actions)
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")
    if noop_action_index is not None:
        if isinstance(noop_action_index, (bool, np.bool_)) or not isinstance(
            noop_action_index, (int, np.integer)
        ):
            raise TypeError("noop_action_index must be an integer or None")
        noop_action_index = int(noop_action_index)
        if noop_action_index < 0 or noop_action_index >= num_actions:
            raise ValueError(
                "noop_action_index lies outside the runtime action space"
            )

    with torch.no_grad():
        vectors = []
        for name, value in (
            ("targets", targets),
            ("argmax_actions", argmax_actions),
            ("sampled_actions", sampled_actions),
        ):
            vector = torch.as_tensor(value).detach().reshape(-1).to("cpu")
            if vector.dtype == torch.bool or torch.is_floating_point(vector):
                raise TypeError(f"{name} must use an integer dtype")
            vector = vector.long()
            if torch.any(vector < 0) or torch.any(vector >= num_actions):
                raise ValueError(
                    f"{name} contains an action outside [0,{num_actions - 1}]"
                )
            vectors.append(vector)
        scored_targets, scored_argmax, scored_samples = vectors
        if not (
            scored_targets.numel()
            == scored_argmax.numel()
            == scored_samples.numel()
        ):
            raise ValueError(
                "targets, argmax_actions and sampled_actions must have the "
                "same number of scored rows"
            )

        support_count = int(scored_targets.numel())
        argmax_correct_count = int(
            torch.count_nonzero(scored_argmax == scored_targets).item()
        )
        sampled_correct_count = int(
            torch.count_nonzero(scored_samples == scored_targets).item()
        )

        def finite_rate(numerator: int, denominator: int) -> float:
            return float(numerator) / denominator if denominator else 0.0

        noop_supported = noop_action_index is not None
        noop_support_count = support_count if noop_supported else 0
        if noop_supported:
            target_noop_count = int(
                torch.count_nonzero(scored_targets == noop_action_index).item()
            )
            argmax_noop_count = int(
                torch.count_nonzero(scored_argmax == noop_action_index).item()
            )
            sampled_noop_count = int(
                torch.count_nonzero(scored_samples == noop_action_index).item()
            )
        else:
            target_noop_count = 0
            argmax_noop_count = 0
            sampled_noop_count = 0

    return {
        "behavioral_argmax_accuracy": finite_rate(
            argmax_correct_count, support_count
        ),
        "behavioral_sampled_accuracy": finite_rate(
            sampled_correct_count, support_count
        ),
        "behavioral_argmax_correct_count": float(argmax_correct_count),
        "behavioral_sampled_correct_count": float(sampled_correct_count),
        "behavioral_support_count": float(support_count),
        "noop_action_index": float(
            noop_action_index if noop_action_index is not None else -1
        ),
        "noop_supported": float(noop_supported),
        "noop_support_count": float(noop_support_count),
        "target_noop_count": float(target_noop_count),
        "target_noop_frequency": finite_rate(
            target_noop_count, noop_support_count
        ),
        "argmax_noop_count": float(argmax_noop_count),
        "argmax_noop_frequency": finite_rate(
            argmax_noop_count, noop_support_count
        ),
        "sampled_noop_count": float(sampled_noop_count),
        "sampled_noop_frequency": finite_rate(
            sampled_noop_count, noop_support_count
        ),
    }


def empty_behavioral_action_metrics(
    *, num_actions: int, noop_action_index: Optional[int]
) -> Dict[str, float]:
    """Return the finite zero-support form used on unscored learner rows."""

    empty = torch.empty(0, dtype=torch.long)
    return detached_behavioral_action_metrics(
        empty,
        empty,
        empty,
        num_actions=num_actions,
        noop_action_index=noop_action_index,
    )


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


class _ModelDoneLogitObserver:
    """Record the existing done head before RESET; never run a second forward."""

    def __init__(self, model_net):
        network = getattr(model_net, "sr_net" if getattr(model_net, "dual_net", False) else "vp_net", None)
        self.module = getattr(network, "out", None)
        self.handle = None
        self.value = None
        self.active = False

    def __enter__(self):
        if isinstance(self.module, torch.nn.Module):
            self.handle = self.module.register_forward_hook(self._observe)
        return self

    def _observe(self, _module, _inputs, output):
        value = getattr(output, "done_logits", None)
        if not self.active or value is None:
            return
        if self.value is not None or not torch.is_tensor(value) or value.numel() != 1:
            raise RuntimeError("Expected one singleton-batch done prediction per trace row")
        self.value = value.detach().clone()

    def begin_step(self):
        self.value = None
        self.active = True

    def take(self, *, model_expansion):
        value = self.value if model_expansion else None
        self.value = None
        self.active = False
        return value

    def __exit__(self, *_args):
        self.active = False
        self.value = None
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class _VoCFeatureObserver:
    """Capture the existing VoC-head input without a second actor forward."""

    def __init__(self, actor_net: torch.nn.Module):
        candidates = (
            getattr(actor_net, "voc_head", None),
            getattr(getattr(actor_net, "critic", None), "voc_head", None),
        )
        self.module = next(
            (module for module in candidates if isinstance(module, torch.nn.Linear)),
            None,
        )
        self.handle = None
        self.value: Optional[torch.Tensor] = None

    def __enter__(self):
        if self.module is not None:
            self.handle = self.module.register_forward_pre_hook(self._observe)
        return self

    def _observe(self, _module, inputs) -> None:
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
            raise RuntimeError("VoC head received an unsupported input contract")
        if self.value is not None:
            raise RuntimeError("VoC head ran more than once for one trace row")
        self.value = inputs[0].detach()

    def begin_row(self) -> None:
        self.value = None

    def take(self, voc_q: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if voc_q is None:
            if self.value is not None:
                raise RuntimeError("VoC feature was observed without ActorOut.voc_q")
            return None
        if self.module is None or self.value is None:
            raise RuntimeError(
                "ActorOut.voc_q is present but its voc_head input was not observed"
            )
        value = self.value
        self.value = None
        return value

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _actor_trace_diagnostics(actor_out: Any) -> Dict[str, Any]:
    """Expose native ActorOut tensors needed by storage-side projections."""

    fields = (
        "pri",
        "pri_param",
        "reset",
        "reset_logits",
        "action_prob",
        "baseline",
        "search_control",
        "search_control_logits",
        "primary_valid",
        "control_valid",
        "policy_valid",
        "policy_type",
        "voc_q",
    )
    diagnostics = {name: getattr(actor_out, name, None) for name in fields}
    misc = getattr(actor_out, "misc", None)
    if isinstance(misc, Mapping):
        diagnostics["misc"] = {
            name: misc[name]
            for name in (
                "tree_rep_enc",
                "voc_gate_soft_control_logits",
                "voc_gate_soft_continue_probability",
                "voc_gate_execution_continue_probability",
                "voc_gate_entropy",
            )
            if name in misc
        }
    else:
        diagnostics["misc"] = {}
    return diagnostics


def _validate_ema_head_state(
    state: Optional[Mapping[str, Any]], feature_width: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if state is None:
        return None
    if not isinstance(state, Mapping) or set(state) != {"weight", "bias"}:
        raise ValueError("VoC EMA state must contain exactly weight and bias")
    weight = torch.as_tensor(state["weight"]).detach()
    bias = torch.as_tensor(state["bias"]).detach()
    if tuple(weight.shape) != (2, feature_width) or tuple(bias.shape) != (2,):
        raise ValueError(
            "VoC EMA head shape disagrees with captured voc_head features"
        )
    if not torch.all(torch.isfinite(weight)) or not torch.all(torch.isfinite(bias)):
        raise ValueError("VoC EMA head contains non-finite values")
    return weight, bias


def _reconstruct_voc_q(
    raw_head: torch.Tensor,
    state_value: torch.Tensor,
    continue_probability: torch.Tensor,
    *,
    dueling_q: bool,
    common_mode_q: bool,
) -> torch.Tensor:
    if raw_head.shape[-1] != 2:
        raise ValueError("VoC raw head must end in [CONTINUE, STOP]")
    if not dueling_q:
        return raw_head
    if tuple(state_value.shape) != tuple(raw_head.shape[:-1]):
        raise ValueError("VoC state value shape disagrees with raw head")
    if tuple(continue_probability.shape) != tuple(raw_head.shape[:-1]):
        raise ValueError("VoC soft gate probability shape disagrees with raw head")
    gate_probability = torch.stack(
        (continue_probability, 1.0 - continue_probability), dim=-1
    ).to(device=raw_head.device, dtype=raw_head.dtype)
    centered = raw_head - torch.sum(
        gate_probability * raw_head, dim=-1, keepdim=True
    )
    reconstructed = state_value.to(raw_head).unsqueeze(-1) + centered
    if common_mode_q:
        reconstructed = reconstructed + raw_head.mean(dim=-1, keepdim=True)
    return reconstructed


def _trace_unit_probability(probability: torch.Tensor) -> torch.Tensor:
    """Validate a recording-only probability, allowing only roundoff at 0/1.

    Keep the trace/storage float32 contract.  Validate before that cast so a
    genuinely invalid float64 value cannot be rounded into the valid range.
    This detached calculation never changes ActorOut or action selection.
    """
    probability = _last(probability).detach()
    if not torch.is_floating_point(probability):
        raise RuntimeError("VoC soft gate probability must be floating point")
    tolerance = 4.0 * min(
        torch.finfo(probability.dtype).eps, torch.finfo(torch.float32).eps
    )
    precise = probability.double()
    if torch.any(~torch.isfinite(precise)) or torch.any(
        (precise < -tolerance) | (precise > 1.0 + tolerance)
    ):
        raise RuntimeError(
            "VoC soft gate probability is invalid: "
            f"values={precise.cpu().tolist()}, tolerance={tolerance}"
        )
    return precise.clamp(0.0, 1.0).float()


def _voc_trace_diagnostics(
    actor_out: Any,
    captured_features: Optional[torch.Tensor],
    *,
    flags: Any,
    ema_head_state: Optional[Mapping[str, Any]],
    effective_think_cost: Optional[float],
) -> Dict[str, Any]:
    raw_q = getattr(actor_out, "voc_q", None)
    if raw_q is None:
        if captured_features is not None:
            raise RuntimeError("captured VoC features without ActorOut.voc_q")
        return {"enabled": False}
    if captured_features is None:
        raise RuntimeError("active VoC trace lacks captured voc_head features")
    schema = getattr(flags, "voc_gate_policy_schema_version", None)
    if (
        isinstance(schema, (bool, np.bool_))
        or not isinstance(schema, (int, np.integer))
        or not 1 <= int(schema) <= util.VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "active VoC trace requires a strict supported gate schema in [1, 13]"
        )
    schema = int(schema)
    dueling_q = bool(getattr(flags, "voc_dueling_q", False))
    common_mode_q = schema >= util.VOC_GATE_POLICY_COMMON_MODE_Q_SCHEMA_VERSION

    raw_q = _last(raw_q).detach().float()
    features = captured_features.detach().float().reshape(
        raw_q.shape[0], -1
    )
    baseline = _last(getattr(actor_out, "baseline", None))
    if baseline is None:
        raise RuntimeError("active VoC trace lacks ActorOut.baseline")
    reward_names = util.get_reward_names(flags)
    if "re" not in reward_names or "think" not in reward_names:
        raise RuntimeError("active VoC trace lacks task/think baseline channels")
    if effective_think_cost is None:
        if bool(getattr(flags, "think_cost_anneal", False)):
            raise RuntimeError(
                "annealed VoC trace requires its exact effective_think_cost"
            )
        effective_think_cost = float(getattr(flags, "think_cost", 0.0))
    effective_think_cost = float(effective_think_cost)
    if not np.isfinite(effective_think_cost) or effective_think_cost < 0.0:
        raise ValueError("effective_think_cost must be finite and nonnegative")
    baseline = baseline.detach().float()
    state_value = (
        baseline[..., reward_names.index("re")]
        + effective_think_cost * baseline[..., reward_names.index("think")]
    )

    misc = getattr(actor_out, "misc", None)
    continue_probability = None
    probability_source = 0
    if isinstance(misc, Mapping):
        continue_probability = misc.get("voc_gate_soft_continue_probability")
        if continue_probability is not None:
            probability_source = 1
    if continue_probability is None:
        soft_logits = (
            misc.get("voc_gate_soft_control_logits")
            if isinstance(misc, Mapping)
            else None
        )
        if soft_logits is not None:
            # Reconstruct telemetry in float64: summing two float32 softmax
            # probabilities near one can exceed one by a float32 ULP.
            soft_logits = _last(soft_logits).detach().double()
            if soft_logits.shape[-1] != 3:
                raise RuntimeError("VoC soft control logits must end in 3 controls")
            continue_probability = torch.softmax(soft_logits, dim=-1)[..., :2].sum(
                dim=-1
            )
            probability_source = 2
    if continue_probability is None:
        soft_logits = getattr(actor_out, "search_control_logits", None)
        if soft_logits is None:
            soft_logits = getattr(actor_out, "reset_logits", None)
        if soft_logits is None:
            raise RuntimeError("active VoC trace lacks control logits")
        soft_logits = _last(soft_logits).detach().double()
        if soft_logits.shape[-1] != 3:
            raise RuntimeError("VoC control logits must end in 3 controls")
        continue_probability = torch.softmax(soft_logits, dim=-1)[..., :2].sum(
            dim=-1
        )
        probability_source = 3
    continue_probability = _trace_unit_probability(continue_probability)
    online_q = _reconstruct_voc_q(
        raw_q,
        state_value,
        continue_probability,
        dueling_q=dueling_q,
        common_mode_q=common_mode_q,
    )

    ema_state = _validate_ema_head_state(ema_head_state, features.shape[-1])
    ema_raw_q = None
    ema_q = None
    if ema_state is not None:
        weight, bias = ema_state
        ema_raw_q = F.linear(
            features,
            weight.to(device=features.device, dtype=features.dtype),
            bias.to(device=features.device, dtype=features.dtype),
        )
        ema_q = _reconstruct_voc_q(
            ema_raw_q,
            state_value,
            continue_probability,
            dueling_q=dueling_q,
            common_mode_q=common_mode_q,
        )
    return {
        "enabled": True,
        "schema_version": schema,
        "dueling_q": dueling_q,
        "common_mode_q": common_mode_q,
        # Numeric provenance: 1=explicit probability, 2=misc soft logits,
        # 3=ActorOut execution control logits.
        "soft_probability_source": probability_source,
        "features": features,
        "effective_think_cost": effective_think_cost,
        "state_value": state_value,
        "soft_continue_probability": continue_probability,
        "online_raw_q": raw_q,
        "online_q": online_q,
        "ema_available": ema_q is not None,
        "ema_raw_q": ema_raw_q,
        "ema_q": ema_q,
    }


DYNAMIC_TRACE_EVENT_SCHEMA_VERSION = 3
VOC_Q_RECONSTRUCTION_RAW = 0
VOC_Q_RECONSTRUCTION_DUELING_CENTERED = 1
VOC_Q_RECONSTRUCTION_DUELING_COMMON = 2

# Public, storage-facing native row contract.  The values are deliberately
# flat and numeric: EpisodeZarrWriter freezes this key/dtype/shape schema on
# the initial-root row and rejects every later mismatch.  Optional values use
# an explicit ``*_valid`` bit and a numeric sentinel, never None.
DYNAMIC_TRACE_EVENT_FIELDS = (
    "trace_schema_version",
    "initial_root",
    "augmented_index",
    "real_cursor",
    "cursor_before",
    "cursor_after",
    "edge_count",
    "score_mask",
    "phase_pre",
    "phase_post",
    "internal_status",
    "internal_status_valid",
    "accepted_control",
    "accepted_control_valid",
    "accepted_primary_action",
    "accepted_primary_action_valid",
    "executed_primary_action",
    "executed_primary_action_valid",
    "human_action",
    "human_action_valid",
    "thinker_action",
    "thinker_action_valid",
    "primary_proposal",
    "primary_argmax",
    "control_proposal",
    "execution_primary",
    "execution_control",
    "execution_decision_valid",
    "policy_type",
    "policy_valid",
    "primary_valid",
    "control_valid",
    "search_computation",
    "search_expansion",
    "search_revisit",
    "actor_reset",
    "stop",
    "real_transition",
    "real_action_stored",
    "wait",
    "tree_token_valid",
    "stage_end",
    "search_state_reset",
    "forced_root_reset",
    "forced_stop",
    "barrier_released",
    "tree_reps_raw",
    "tree_reps_raw_valid",
    "tree_reps_pre",
    "tree_reps_post",
    "actor_policy",
    "actor_policy_valid",
    "actor_real_logits_pre_override",
    "actor_real_logits_pre_override_valid",
    "model_done_logit",
    "model_done_logit_valid",
    "model_done_target_depth",
    "model_done_target_action",
    "predicted_xs",
    "predicted_xs_valid",
    "step_times",
    "step_times_valid",
    "im_vectors",
    "im_vectors_valid",
    "im_vp_vectors",
    "im_vp_vectors_valid",
    "actor_control_logits",
    "actor_control_logits_valid",
    "actor_action_probability",
    "actor_action_probability_valid",
    "actor_baseline",
    "actor_baseline_valid",
    "reward",
    "reward_valid",
    "cur_rewards",
    "cur_rewards_valid",
    "env_return",
    "env_return_valid",
    "done",
    "truncated",
    "pre_observation_source_index",
    "pre_observation_source_index_valid",
    "post_observation_source_index",
    "post_observation_source_index_valid",
    "tree_reachable_node_count_pre",
    "tree_reachable_node_count_post",
    "tree_reachable_node_count_delta",
    "tree_expanded_node_count_pre",
    "tree_expanded_node_count_post",
    "tree_expanded_node_count_delta",
    "tree_visited_node_count_pre",
    "tree_visited_node_count_post",
    "tree_visited_node_count_delta",
    "tree_leaf_node_count_pre",
    "tree_leaf_node_count_post",
    "tree_leaf_node_count_delta",
    "tree_max_reachable_depth_pre",
    "tree_max_reachable_depth_post",
    "tree_max_reachable_depth_delta",
    "current_node_depth_pre",
    "current_node_depth_post",
    "current_node_action_pre",
    "current_node_action_post",
    "current_node_done_pre",
    "current_node_done_post",
    "current_node_expanded_pre",
    "current_node_expanded_post",
    "current_node_is_root_pre",
    "current_node_is_root_post",
    "current_node_has_real_state_pre",
    "current_node_has_real_state_post",
    "search_steps_pre",
    "search_steps_post",
    "rollout_depth_pre",
    "rollout_depth_post",
    "root_carried",
    "carried_descendant_visit_count",
    "carried_descendant_expanded_count",
    "useful_carry",
    "raw_info_real_transition",
    "raw_info_tree_token_valid",
    "raw_info_search_state_reset",
    "raw_info_stage_end",
    "raw_info_forced_stop",
    "raw_info_stored_action_mask",
    "raw_info_barrier_released",
    "raw_info_accepted_primary_action",
    "raw_info_accepted_control",
    "raw_info_executed_primary_action",
    "raw_info_search_steps",
    "raw_info_think_reward",
    "observed_info_effective_primary_action",
    "observed_info_effective_search_control",
    "observed_info_im_episode_return",
    "observed_info_cur_episode_return",
    "observed_info_think_episode_return",
    "voc_enabled",
    "voc_schema_version",
    "voc_reconstruction_mode",
    "voc_soft_probability_source",
    "voc_features",
    "voc_features_valid",
    "voc_effective_think_cost",
    "voc_effective_think_cost_valid",
    "voc_state_value",
    "voc_state_value_valid",
    "voc_soft_continue_probability",
    "voc_soft_continue_probability_valid",
    "voc_online_raw_q",
    "voc_online_raw_q_valid",
    "voc_online_q",
    "voc_online_q_valid",
    "voc_ema_available",
    "voc_ema_raw_q",
    "voc_ema_raw_q_valid",
    "voc_ema_q",
    "voc_ema_q_valid",
)
def dynamic_trace_event_fields(image_mode: str = "none") -> Tuple[str, ...]:
    """Return the single exact native runner schema.

    ``image_mode`` is validated for the sink contract, but display images are
    projections of always-present native fields and never change this schema.
    """

    if image_mode not in ("none", "real", "imagined", "all"):
        raise ValueError("image_mode must be none, real, imagined, or all")
    return DYNAMIC_TRACE_EVENT_FIELDS


def _trace_batch_value(
    value: Any,
    *,
    dtype: Any,
    time_major: bool = False,
    name: str,
) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"required trace value {name!r} is unavailable")
    if time_major:
        value = _last(value)
    array = _as_numpy(value)
    if array.ndim == 0:
        return np.asarray(array, dtype=dtype)
    if array.shape[0] != 1:
        raise RuntimeError(
            f"stream trace {name!r} must have a singleton batch axis"
        )
    return np.asarray(array[0], dtype=dtype)


def _trace_optional_array(
    value: Any,
    *,
    dtype: Any,
    time_major: bool,
    name: str,
    missing_shape: Tuple[int, ...] = (1,),
) -> Tuple[np.ndarray, np.bool_]:
    if value is None:
        return np.zeros(missing_shape, dtype=dtype), np.bool_(False)
    return (
        _trace_batch_value(
            value, dtype=dtype, time_major=time_major, name=name
        ),
        np.bool_(True),
    )


def _trace_predicted_xs(
    pre_env_out: Any,
    post_env_out: Any,
    *,
    initial_root: bool,
) -> Tuple[np.ndarray, np.bool_]:
    """Return the actor-visible current-node prediction for one native row.

    Dynamic cenv constructs ``EnvOut.xs`` directly from the current tree node.
    The initial-root row describes the pre-state; every transition row describes
    the post-state.  If a checkpoint does not expose ``xs``, use a stable
    numeric sentinel.  A transiently missing post value retains the pre value's
    raw shape while marking the row invalid, so strict streaming sinks never
    receive shape-changing placeholder data.
    """

    pre_xs = getattr(pre_env_out, "xs", None)
    post_xs = getattr(post_env_out, "xs", None)
    selected = pre_xs if initial_root else post_xs
    if selected is not None:
        return (
            _trace_batch_value(
                selected,
                dtype=np.float32,
                time_major=True,
                name="current-node predicted xs",
            ),
            np.bool_(True),
        )

    template = pre_xs if pre_xs is not None else post_xs
    if template is None:
        return np.zeros((1,), dtype=np.float32), np.bool_(False)
    template_value = _trace_batch_value(
        template,
        dtype=np.float32,
        time_major=True,
        name="current-node predicted xs template",
    )
    return np.zeros_like(template_value, dtype=np.float32), np.bool_(False)


def _trace_scalar_array(
    mapping: Mapping[str, Any], name: str, dtype: Any, default: Any
) -> np.generic:
    value = mapping.get(name)
    if value is None:
        return np.asarray(default, dtype=dtype)[()]
    return np.asarray(_trace_scalar(value), dtype=dtype)[()]


def _trace_node_latent(
    snapshot: Mapping[str, Any], name: str
) -> Tuple[np.ndarray, np.bool_]:
    values = snapshot.get(name)
    if values is None or len(values) != 1 or values[0] is None:
        return np.zeros((1,), dtype=np.float32), np.bool_(False)
    value = _as_numpy(values[0], np.float32)
    if value.ndim == 0:
        value = value.reshape(1)
    return np.asarray(value, dtype=np.float32), np.bool_(True)


def _initial_trace_predicates() -> Dict[str, Any]:
    return {
        "internal_status": -1,
        "is_search_computation": False,
        "is_search_expansion": False,
        "is_search_revisit": False,
        "is_reset_control": False,
        "is_stop": False,
        "is_real_transition": False,
        "is_real_action_stored": False,
        "is_wait": False,
        "tree_token_valid": False,
        "stage_end": False,
    }


def build_dynamic_trace_event(
    *,
    initial_root: bool,
    augmented_index: int,
    cursor_before: int,
    cursor_after: int,
    edge_count: int,
    score_mask: bool,
    human_action: int,
    pre_env_out: Any,
    post_env_out: Any,
    actor_out: Any,
    decision: ExecutionDecision,
    reward: Any,
    done: Any,
    truncated: Any,
    raw_info: Mapping[str, Any],
    observed_info: Mapping[str, Any],
    pre_cenv: Mapping[str, Any],
    post_cenv: Mapping[str, Any],
    predicates: Mapping[str, Any],
    voc: Mapping[str, Any],
    pre_observation_source_index: Optional[int],
    post_observation_source_index: Optional[int],
    sr_hidden_shape: Optional[Tuple[int, ...]],
    vp_hidden_shape: Optional[Tuple[int, ...]],
    image_mode: str = "none",
    model_done_logit: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Return one canonical, flat, numeric native replay row.

    ``initial_root`` is a non-transition row.  It reuses the first actor
    forward for diagnostics, while all accepted/executed action validity bits
    stay false.  Transition rows describe the exact cenv call just completed.
    """

    phase_pre = _trace_scalar_array(pre_cenv, "phase", np.int8, -1)
    phase_post = _trace_scalar_array(post_cenv, "phase", np.int8, -1)
    accepted_control = _trace_scalar_array(
        post_cenv, "accepted_control", np.int8, -1
    )
    accepted_primary = _trace_scalar_array(
        post_cenv, "accepted_primary_action", np.int16, -1
    )
    executed_primary = _trace_scalar_array(
        post_cenv, "executed_primary_action", np.int16, -1
    )
    is_search = bool(predicates["is_search_computation"])
    is_real = bool(predicates["is_real_transition"])
    status = int(predicates["internal_status"])

    actor_policy = _trace_batch_value(
        getattr(actor_out, "pri_param", None),
        dtype=np.float32,
        time_major=True,
        name="ActorOut.pri_param",
    )
    if actor_policy.ndim == 2 and actor_policy.shape[0] == 1:
        actor_policy = actor_policy[0]
    if actor_policy.ndim != 1:
        raise RuntimeError("streaming actor policy must be one discrete logit vector")
    actor_control, actor_control_valid = _trace_optional_array(
        getattr(actor_out, "search_control_logits", None),
        dtype=np.float32,
        time_major=True,
        name="ActorOut.search_control_logits",
        missing_shape=(3,),
    )
    if actor_control.ndim == 2 and actor_control.shape[0] == 1:
        actor_control = actor_control[0]
    if actor_control.shape != (3,):
        raise RuntimeError("streaming control policy must contain three logits")
    actor_action_probability, actor_action_probability_valid = (
        _trace_optional_array(
            getattr(actor_out, "action_prob", None),
            dtype=np.float32,
            time_major=True,
            name="ActorOut.action_prob",
        )
    )
    actor_action_probability = actor_action_probability.reshape(-1)
    actor_baseline, actor_baseline_valid = _trace_optional_array(
        getattr(actor_out, "baseline", None),
        dtype=np.float32,
        time_major=True,
        name="ActorOut.baseline",
    )
    actor_baseline = actor_baseline.reshape(-1)

    tree_pre = _trace_batch_value(
        pre_env_out.tree_reps,
        dtype=np.float32,
        time_major=True,
        name="pre tree_reps",
    ).reshape(-1)
    tree_post = _trace_batch_value(
        post_env_out.tree_reps,
        dtype=np.float32,
        time_major=True,
        name="post tree_reps",
    ).reshape(-1)
    if tree_pre.shape != tree_post.shape:
        raise RuntimeError("pre/post Dynamic tree representations disagree")

    predicted_xs, predicted_xs_valid = _trace_predicted_xs(
        pre_env_out,
        post_env_out,
        initial_root=initial_root,
    )

    if image_mode not in ("none", "real", "imagined", "all"):
        raise ValueError("image_mode must be none, real, imagined, or all")

    node_hs_post, node_hs_post_valid = _trace_node_latent(
        post_cenv, "current_node_hs"
    )
    sr_shape = tuple(sr_hidden_shape or ())
    vp_shape = tuple(vp_hidden_shape or ())
    if sr_shape and vp_shape and sr_shape[1:] != vp_shape[1:]:
        raise RuntimeError("SR/VP hidden spatial shapes disagree")
    expected_combined_shape = None
    if sr_shape and vp_shape:
        expected_combined_shape = (sr_shape[0] + vp_shape[0],) + sr_shape[1:]
    elif sr_shape:
        expected_combined_shape = sr_shape
    elif vp_shape:
        expected_combined_shape = vp_shape
    if node_hs_post_valid and expected_combined_shape is not None:
        if tuple(node_hs_post.shape) != expected_combined_shape:
            raise RuntimeError(
                "current-node hidden state disagrees with ModelNet hidden shapes"
            )
    if sr_shape:
        im_vectors = (
            node_hs_post[: sr_shape[0]][None, ...]
            if node_hs_post_valid
            else np.zeros((1,) + sr_shape, dtype=np.float32)
        )
        im_vectors_valid = np.bool_(node_hs_post_valid)
    else:
        im_vectors = np.zeros((1, 1), dtype=np.float32)
        im_vectors_valid = np.bool_(False)
    if vp_shape:
        vp_start = sr_shape[0] if sr_shape else 0
        im_vp_vectors = (
            node_hs_post[vp_start:][None, ...]
            if node_hs_post_valid
            else np.zeros((1,) + vp_shape, dtype=np.float32)
        )
        im_vp_vectors_valid = np.bool_(node_hs_post_valid)
    else:
        im_vp_vectors = np.zeros((1, 1), dtype=np.float32)
        im_vp_vectors_valid = np.bool_(False)

    reward_value, reward_valid = _trace_optional_array(
        reward,
        dtype=np.float32,
        time_major=False,
        name="reward",
    )
    # The cenv return is the scalar real-environment reward, whereas EnvOut
    # carries the fixed reward-channel vector.  Keep those schemas distinct.
    reward_value = np.asarray(
        [reward_value.reshape(-1)[0]], dtype=np.float32
    )
    cur_rewards, cur_rewards_valid = _trace_optional_array(
        post_env_out.reward,
        dtype=np.float32,
        time_major=True,
        name="post EnvOut.reward",
    )
    cur_rewards = cur_rewards.reshape(-1)
    env_return = np.float32(reward_value.reshape(-1)[0] if is_real else 0.0)

    primary_proposal = int(
        _trace_batch_value(
            decision.primary_proposal,
            dtype=np.int64,
            name="primary proposal",
        ).reshape(-1)[0]
    )
    primary_argmax = int(
        _trace_batch_value(
            decision.primary_argmax,
            dtype=np.int64,
            name="primary argmax",
        ).reshape(-1)[0]
    )
    control_proposal = int(
        _trace_batch_value(
            decision.search_control_proposal,
            dtype=np.int64,
            name="control proposal",
        ).reshape(-1)[0]
    )
    execution_primary = int(
        _trace_batch_value(
            decision.execution_primary,
            dtype=np.int64,
            name="execution primary",
        ).reshape(-1)[0]
    )
    execution_control = int(
        _trace_batch_value(
            decision.execution_control,
            dtype=np.int64,
            name="execution control",
        ).reshape(-1)[0]
    )

    policy_type = int(
        _trace_batch_value(
            getattr(actor_out, "policy_type", None),
            dtype=np.int64,
            time_major=True,
            name="ActorOut.policy_type",
        ).reshape(-1)[0]
    )
    policy_valid = bool(
        _trace_batch_value(
            getattr(actor_out, "policy_valid", None),
            dtype=np.bool_,
            time_major=True,
            name="ActorOut.policy_valid",
        ).reshape(-1)[0]
    )
    primary_valid = bool(
        _trace_batch_value(
            getattr(actor_out, "primary_valid", None),
            dtype=np.bool_,
            time_major=True,
            name="ActorOut.primary_valid",
        ).reshape(-1)[0]
    )
    control_valid = bool(
        _trace_batch_value(
            getattr(actor_out, "control_valid", None),
            dtype=np.bool_,
            time_major=True,
            name="ActorOut.control_valid",
        ).reshape(-1)[0]
    )

    # An explicit REAL-only alias; actor_policy remains completely unchanged.
    real_logits_valid = not initial_root and policy_valid and policy_type == util.POLICY_REAL
    model_done_valid = model_done_logit is not None
    if model_done_valid and (initial_root or not is_search or status != TRACE_SEARCH_EXPANSION):
        raise RuntimeError("Model done logit must belong to a new model expansion")
    # After RESET the current node may be root; the evaluated target is instead
    # the accepted child of the pre-step current node.
    model_done_depth = (
        int(_trace_scalar_array(pre_cenv, "current_node_depth", np.int16, -1)) + 1
        if model_done_valid else -1
    )
    if model_done_valid and (model_done_depth < 1 or accepted_primary < 0):
        raise RuntimeError("Model done logit lacks evaluated target provenance")

    voc_enabled = bool(voc.get("enabled", False))
    if voc_enabled:
        voc_features = _trace_batch_value(
            voc["features"], dtype=np.float32, name="VoC features"
        ).reshape(-1)
        voc_online_raw_q = _trace_batch_value(
            voc["online_raw_q"], dtype=np.float32, name="online raw VoC Q"
        ).reshape(2)
        voc_online_q = _trace_batch_value(
            voc["online_q"], dtype=np.float32, name="online VoC Q"
        ).reshape(2)
        voc_ema_available = bool(voc["ema_available"])
        if voc_ema_available:
            voc_ema_raw_q = _trace_batch_value(
                voc["ema_raw_q"], dtype=np.float32, name="EMA raw VoC Q"
            ).reshape(2)
            voc_ema_q = _trace_batch_value(
                voc["ema_q"], dtype=np.float32, name="EMA VoC Q"
            ).reshape(2)
        else:
            voc_ema_raw_q = np.zeros((2,), dtype=np.float32)
            voc_ema_q = np.zeros((2,), dtype=np.float32)
        state_value = float(
            _trace_batch_value(
                voc["state_value"], dtype=np.float32, name="VoC state value"
            ).reshape(-1)[0]
        )
        soft_probability = float(
            _trace_batch_value(
                voc["soft_continue_probability"],
                dtype=np.float32,
                name="VoC soft continue probability",
            ).reshape(-1)[0]
        )
        reconstruction_mode = (
            VOC_Q_RECONSTRUCTION_DUELING_COMMON
            if voc["common_mode_q"]
            else VOC_Q_RECONSTRUCTION_DUELING_CENTERED
            if voc["dueling_q"]
            else VOC_Q_RECONSTRUCTION_RAW
        )
    else:
        voc_features = np.zeros((1,), dtype=np.float32)
        voc_online_raw_q = np.zeros((2,), dtype=np.float32)
        voc_online_q = np.zeros((2,), dtype=np.float32)
        voc_ema_raw_q = np.zeros((2,), dtype=np.float32)
        voc_ema_q = np.zeros((2,), dtype=np.float32)
        voc_ema_available = False
        state_value = 0.0
        soft_probability = 0.0
        reconstruction_mode = VOC_Q_RECONSTRUCTION_RAW

    event: Dict[str, Any] = {
        "trace_schema_version": np.int16(DYNAMIC_TRACE_EVENT_SCHEMA_VERSION),
        "initial_root": np.bool_(initial_root),
        "augmented_index": np.int64(augmented_index),
        "real_cursor": np.int64(cursor_before),
        "cursor_before": np.int64(cursor_before),
        "cursor_after": np.int64(cursor_after),
        "edge_count": np.int64(edge_count),
        "score_mask": np.bool_(score_mask),
        "phase_pre": np.int8(phase_pre),
        "phase_post": np.int8(phase_post),
        "internal_status": np.int8(status),
        "internal_status_valid": np.bool_(not initial_root),
        "accepted_control": np.int8(accepted_control if not initial_root else -1),
        "accepted_control_valid": np.bool_(not initial_root and accepted_control >= 0),
        "accepted_primary_action": np.int16(
            accepted_primary if not initial_root else -1
        ),
        "accepted_primary_action_valid": np.bool_(
            not initial_root and accepted_primary >= 0
        ),
        "executed_primary_action": np.int16(
            executed_primary if not initial_root else -1
        ),
        "executed_primary_action_valid": np.bool_(
            not initial_root and executed_primary >= 0
        ),
        "human_action": np.int16(human_action if is_real else -1),
        "human_action_valid": np.bool_(is_real),
        "thinker_action": np.int16(primary_proposal if is_real else -1),
        "thinker_action_valid": np.bool_(is_real),
        "primary_proposal": np.int16(primary_proposal),
        "primary_argmax": np.int16(primary_argmax),
        "control_proposal": np.int8(control_proposal),
        "execution_primary": np.int16(execution_primary),
        "execution_control": np.int8(execution_control),
        "execution_decision_valid": np.bool_(not initial_root),
        "policy_type": np.int8(policy_type),
        "policy_valid": np.bool_(policy_valid),
        "primary_valid": np.bool_(primary_valid),
        "control_valid": np.bool_(control_valid),
        "search_computation": np.bool_(is_search),
        "search_expansion": np.bool_(predicates["is_search_expansion"]),
        "search_revisit": np.bool_(predicates["is_search_revisit"]),
        "actor_reset": np.bool_(predicates["is_reset_control"]),
        "stop": np.bool_(predicates["is_stop"]),
        "real_transition": np.bool_(is_real),
        "real_action_stored": np.bool_(predicates["is_real_action_stored"]),
        "wait": np.bool_(predicates["is_wait"]),
        "tree_token_valid": np.bool_(predicates["tree_token_valid"]),
        "stage_end": np.bool_(predicates["stage_end"]),
        "search_state_reset": np.bool_(
            _trace_scalar_array(post_cenv, "search_state_reset", np.bool_, False)
        ),
        "forced_root_reset": np.bool_(
            _trace_scalar_array(post_cenv, "forced_root_reset", np.bool_, False)
        ),
        "forced_stop": np.bool_(
            _trace_scalar_array(post_cenv, "forced_stop", np.bool_, False)
        ),
        "barrier_released": np.bool_(
            _trace_scalar_array(post_cenv, "barrier_released", np.bool_, False)
        ),
        "tree_reps_raw": tree_pre if initial_root else tree_post,
        "tree_reps_raw_valid": np.bool_(True),
        "tree_reps_pre": tree_pre,
        "tree_reps_post": tree_post,
        "actor_policy": np.asarray(actor_policy, dtype=np.float32),
        "actor_policy_valid": np.bool_(True),
        "actor_real_logits_pre_override": (
            actor_policy.copy() if real_logits_valid else np.zeros_like(actor_policy)
        ),
        "actor_real_logits_pre_override_valid": np.bool_(real_logits_valid),
        "model_done_logit": np.float32(
            _as_numpy(model_done_logit).reshape(-1)[0] if model_done_valid else 0.0
        ),
        "model_done_logit_valid": np.bool_(model_done_valid),
        # Both target fields share model_done_logit_valid; this is not real done.
        "model_done_target_depth": np.int16(model_done_depth),
        "model_done_target_action": np.int16(accepted_primary if model_done_valid else -1),
        # This is the raw actor-visible ModelNet prediction for cenv's current
        # node.  It is native replay state, not an optional display image.
        "predicted_xs": np.asarray(predicted_xs, dtype=np.float32),
        "predicted_xs_valid": predicted_xs_valid,
        # Current Dynamic EnvOut has no wall-clock timing channel.
        "step_times": np.zeros((1,), dtype=np.float32),
        "step_times_valid": np.bool_(False),
        # cenv's encoded `hs` is exactly ModelNet's channel-wise SR+VP output.
        # Split it using the declared network shapes without another forward.
        "im_vectors": np.asarray(im_vectors, dtype=np.float32),
        "im_vectors_valid": im_vectors_valid,
        "im_vp_vectors": np.asarray(im_vp_vectors, dtype=np.float32),
        "im_vp_vectors_valid": im_vp_vectors_valid,
        "actor_control_logits": np.asarray(actor_control, dtype=np.float32),
        "actor_control_logits_valid": actor_control_valid,
        "actor_action_probability": np.asarray(
            actor_action_probability, dtype=np.float32
        ),
        "actor_action_probability_valid": actor_action_probability_valid,
        "actor_baseline": np.asarray(actor_baseline, dtype=np.float32),
        "actor_baseline_valid": actor_baseline_valid,
        "reward": np.asarray(reward_value, dtype=np.float32),
        "reward_valid": reward_valid,
        "cur_rewards": np.asarray(cur_rewards, dtype=np.float32),
        "cur_rewards_valid": cur_rewards_valid,
        "env_return": env_return,
        "env_return_valid": np.bool_(True),
        "done": np.bool_(_trace_scalar(done)),
        "truncated": np.bool_(_trace_scalar(truncated)),
        # Real frames stay in the immutable behavioral source.  Native rows
        # reference them instead of repeating a full image for every search
        # event.  Decoded imagined states are likewise left to an explicit
        # image-profile sink rather than silently bloating the default trace.
        "pre_observation_source_index": np.int64(
            -1
            if pre_observation_source_index is None
            else pre_observation_source_index
        ),
        "pre_observation_source_index_valid": np.bool_(
            pre_observation_source_index is not None
        ),
        "post_observation_source_index": np.int64(
            -1
            if post_observation_source_index is None
            else post_observation_source_index
        ),
        "post_observation_source_index_valid": np.bool_(
            post_observation_source_index is not None
        ),
        "root_carried": np.bool_(
            _trace_scalar_array(post_cenv, "root_carried", np.bool_, False)
        ),
        "carried_descendant_visit_count": np.int32(
            _trace_scalar_array(
                observed_info,
                "carried_descendant_visit_count",
                np.int32,
                0,
            )
        ),
        "carried_descendant_expanded_count": np.int32(
            _trace_scalar_array(
                observed_info,
                "carried_descendant_expanded_count",
                np.int32,
                0,
            )
        ),
        "useful_carry": np.bool_(
            _trace_scalar_array(observed_info, "useful_carry", np.bool_, False)
        ),
        "raw_info_real_transition": np.bool_(
            _trace_scalar_array(raw_info, "real_transition", np.bool_, False)
        ),
        "raw_info_tree_token_valid": np.bool_(
            _trace_scalar_array(raw_info, "tree_token_valid", np.bool_, False)
        ),
        "raw_info_search_state_reset": np.bool_(
            _trace_scalar_array(raw_info, "search_state_reset", np.bool_, False)
        ),
        "raw_info_stage_end": np.bool_(
            _trace_scalar_array(raw_info, "stage_end", np.bool_, False)
        ),
        "raw_info_forced_stop": np.bool_(
            _trace_scalar_array(raw_info, "forced_stop", np.bool_, False)
        ),
        "raw_info_stored_action_mask": np.bool_(
            _trace_scalar_array(raw_info, "stored_action_mask", np.bool_, False)
        ),
        "raw_info_barrier_released": np.bool_(
            _trace_scalar_array(raw_info, "barrier_released", np.bool_, False)
        ),
        "raw_info_accepted_primary_action": np.int16(
            _trace_scalar_array(raw_info, "accepted_primary_action", np.int16, -1)
        ),
        "raw_info_accepted_control": np.int8(
            _trace_scalar_array(raw_info, "accepted_control", np.int8, -1)
        ),
        "raw_info_executed_primary_action": np.int16(
            _trace_scalar_array(raw_info, "executed_primary_action", np.int16, -1)
        ),
        "raw_info_search_steps": np.int32(
            _trace_scalar_array(raw_info, "search_steps", np.int32, 0)
        ),
        "raw_info_think_reward": np.float32(
            _trace_scalar_array(raw_info, "think_reward", np.float32, 0.0)
        ),
        "observed_info_effective_primary_action": np.int16(
            _trace_scalar_array(
                observed_info, "effective_primary_action", np.int16, -1
            )
        ),
        "observed_info_effective_search_control": np.int8(
            _trace_scalar_array(
                observed_info, "effective_search_control", np.int8, -1
            )
        ),
        "observed_info_im_episode_return": np.float32(
            _trace_scalar_array(
                observed_info, "im_episode_return", np.float32, 0.0
            )
        ),
        "observed_info_cur_episode_return": np.float32(
            _trace_scalar_array(
                observed_info, "cur_episode_return", np.float32, 0.0
            )
        ),
        "observed_info_think_episode_return": np.float32(
            _trace_scalar_array(
                observed_info, "think_episode_return", np.float32, 0.0
            )
        ),
        "voc_enabled": np.bool_(voc_enabled),
        "voc_schema_version": np.int16(voc.get("schema_version", 0)),
        "voc_reconstruction_mode": np.int8(reconstruction_mode),
        "voc_soft_probability_source": np.int8(
            voc.get("soft_probability_source", 0)
        ),
        "voc_features": np.asarray(voc_features, dtype=np.float32),
        "voc_features_valid": np.bool_(voc_enabled),
        "voc_effective_think_cost": np.float32(
            voc.get("effective_think_cost", 0.0)
        ),
        "voc_effective_think_cost_valid": np.bool_(voc_enabled),
        "voc_state_value": np.float32(state_value),
        "voc_state_value_valid": np.bool_(voc_enabled),
        "voc_soft_continue_probability": np.float32(soft_probability),
        "voc_soft_continue_probability_valid": np.bool_(voc_enabled),
        "voc_online_raw_q": np.asarray(voc_online_raw_q, dtype=np.float32),
        "voc_online_raw_q_valid": np.bool_(voc_enabled),
        "voc_online_q": np.asarray(voc_online_q, dtype=np.float32),
        "voc_online_q_valid": np.bool_(voc_enabled),
        "voc_ema_available": np.bool_(voc_ema_available),
        "voc_ema_raw_q": np.asarray(voc_ema_raw_q, dtype=np.float32),
        "voc_ema_raw_q_valid": np.bool_(voc_ema_available),
        "voc_ema_q": np.asarray(voc_ema_q, dtype=np.float32),
        "voc_ema_q_valid": np.bool_(voc_ema_available),
    }

    for metric in (
        "tree_reachable_node_count",
        "tree_expanded_node_count",
        "tree_visited_node_count",
        "tree_leaf_node_count",
        "tree_max_reachable_depth",
    ):
        before = int(_trace_scalar_array(pre_cenv, metric, np.int32, 0))
        after = int(_trace_scalar_array(post_cenv, metric, np.int32, 0))
        event[metric + "_pre"] = np.int32(before)
        event[metric + "_post"] = np.int32(after)
        event[metric + "_delta"] = np.int32(after - before)
    for metric in ("current_node_depth", "current_node_action"):
        event[metric + "_pre"] = np.int32(
            _trace_scalar_array(pre_cenv, metric, np.int32, -1)
        )
        event[metric + "_post"] = np.int32(
            _trace_scalar_array(post_cenv, metric, np.int32, -1)
        )
    for metric in (
        "current_node_done",
        "current_node_expanded",
        "current_node_is_root",
        "current_node_has_real_state",
    ):
        event[metric + "_pre"] = np.bool_(
            _trace_scalar_array(pre_cenv, metric, np.bool_, False)
        )
        event[metric + "_post"] = np.bool_(
            _trace_scalar_array(post_cenv, metric, np.bool_, False)
        )
    for metric in ("search_steps", "rollout_depth"):
        event[metric + "_pre"] = np.int32(
            _trace_scalar_array(pre_cenv, metric, np.int32, 0)
        )
        event[metric + "_post"] = np.int32(
            _trace_scalar_array(post_cenv, metric, np.int32, 0)
        )

    actual = set(event)
    expected = set(dynamic_trace_event_fields(image_mode))
    if actual != expected:
        raise AssertionError(
            "canonical dynamic trace schema drift: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if any(value is None or isinstance(value, (str, bytes)) for value in event.values()):
        raise AssertionError("canonical dynamic trace rows must be numeric and non-null")
    return event


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
    noop_action_index: Optional[int] = None

    def detached_metrics(self) -> Dict[str, float]:
        count = max(self.count, 1)
        action_metrics = detached_behavioral_action_metrics(
            self.targets,
            self.argmax,
            self.proposal,
            num_actions=int(self.logits.shape[-1]),
            noop_action_index=self.noop_action_index,
        )
        metrics = {
            "loss": float(self.loss.detach().cpu()),
            "nll": float(self.nll_sum.detach().cpu()) / count,
            "normalized_ce": float(self.normalized_ce.detach().cpu()),
            "margin_loss": float(self.margin_loss.detach().cpu()),
            "pvp_loss": float(self.pvp_loss.detach().cpu()),
            # Keep the historical names while defining them explicitly as
            # scored-human-target argmax/sample accuracies.
            "accuracy": action_metrics["behavioral_argmax_accuracy"],
            "sampled_accuracy": action_metrics[
                "behavioral_sampled_accuracy"
            ],
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
        metrics.update(action_metrics)
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
        noop_action_index: Optional[int] = None,
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
        self.noop_action_index = noop_action_index
        if self.noop_action_index is not None:
            self.noop_action_index = int(self.noop_action_index)
            if not 0 <= self.noop_action_index < int(self.actor_net.num_actions):
                raise ValueError(
                    "noop_action_index lies outside the Actor action space"
                )
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
        from thinker.dataset_env import (
            BehaviorSequenceVectorEnv,
            LazyBehaviorEpisodeVectorEnv,
        )

        if "observation_provider" in batch:
            # A long episode owns its provider/cursor.  Recreate the lightweight
            # wrapper instead of retaining or materializing any prior episode.
            self.close()
            actions = _as_numpy(batch["actions_seq"], np.int64)
            if actions.ndim == 2:
                actions = actions[0]
            model_observation_space = getattr(
                self.model_net, "observation_space", None
            )
            observation_space = batch.get(
                "observation_space", model_observation_space
            )
            self._behavior_env = LazyBehaviorEpisodeVectorEnv(
                batch["observation_provider"],
                actions_seq=actions,
                rewards_seq=batch.get("rewards_seq"),
                done_seq=batch.get("done_seq"),
                truncated_seq=batch.get("truncated_seq"),
                initial_prev_action=batch["initial_prev_action"],
                score_mask=batch["score_mask"],
                num_actions=int(self.actor_net.num_actions),
                observation_space=observation_space,
            )
            replay_primary = self._behavior_env.action_space[0]
            if (
                not isinstance(replay_primary, spaces.Discrete)
                or int(getattr(replay_primary, "start", 0)) != 0
                or int(replay_primary.n) != int(self.actor_net.num_actions)
            ):
                raise ValueError(
                    "lazy behavior replay and Actor action spaces disagree"
                )
            planner_flags = copy.copy(self.flags)
            planner_flags.tree_carry = bool(tree_carry)
            self._planner = cModelWrapper(
                env=self._behavior_env,
                env_n=1,
                flags=planner_flags,
                model_net=self.model_net,
                device=self.model_device,
                timing=False,
            )
            self._planner_key = (
                "lazy_episode",
                id(batch["observation_provider"]),
                len(actions),
                bool(tree_carry),
            )
            return self._planner

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

    def _run_rollout(
        self,
        batch: Mapping[str, Any],
        tree_carry: bool = True,
        training: bool = True,
        *,
        event_sink: Optional[Any] = None,
        collect_result: bool = True,
        voc_feature_observer: Optional[_VoCFeatureObserver] = None,
        model_done_observer: Optional[_ModelDoneLogitObserver] = None,
        voc_ema_head_state_dict: Optional[Mapping[str, Any]] = None,
        effective_think_cost: Optional[float] = None,
    ) -> Any:
        if collect_result:
            batch_size, edge_count = validate_behavior_batch(batch)
        else:
            batch_size, edge_count = validate_stream_behavior_episode(batch)
        image_mode = "none"
        if event_sink is not None:
            image_mode = str(
                getattr(
                    event_sink,
                    "image_mode",
                    getattr(event_sink, "images", "none"),
                )
            ).lower()
            dynamic_trace_event_fields(image_mode)
        actions = torch.as_tensor(
            batch["actions_seq"], device=self.device, dtype=torch.long
        )
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        initial_prev = torch.as_tensor(
            batch["initial_prev_action"], device=self.device, dtype=torch.long
        ).reshape(-1)
        score_mask = torch.as_tensor(
            batch["score_mask"], device=self.device, dtype=torch.bool
        )
        if score_mask.ndim == 1:
            score_mask = score_mask.unsqueeze(0).expand(batch_size, -1)
        observation_source_index = batch.get("observation_source_index")
        if observation_source_index is not None:
            observation_source_index = _as_numpy(
                observation_source_index, np.int64
            )
            if (
                observation_source_index.ndim == 2
                and observation_source_index.shape[0] == 1
            ):
                observation_source_index = observation_source_index[0]
            if observation_source_index.shape not in (
                (edge_count + 1,), (batch_size, edge_count + 1)
            ):
                raise ValueError(
                    "observation_source_index must have shape [E+1] or [B,E+1]"
                )
        sr_net = getattr(self.model_net, "sr_net", None)
        vp_net = getattr(self.model_net, "vp_net", None)
        sr_hidden_shape = (
            tuple(int(size) for size in sr_net.hidden_shape)
            if sr_net is not None and getattr(sr_net, "hidden_shape", None) is not None
            else None
        )
        vp_hidden_shape = (
            tuple(int(size) for size in vp_net.hidden_shape)
            if vp_net is not None and getattr(vp_net, "hidden_shape", None) is not None
            else None
        )

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

        if collect_result:
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
        else:
            # The streaming path must stay O(1) in episode length.  One flag is
            # enough because batch=1 barriers complete each supplied edge.
            stage_logits = stage_proposals = stage_argmax = None
            stage_root_carried = None
            stage_carried_descendant_visit_count = None
            stage_carried_descendant_expanded_count = None
            stage_useful_carry = None
            filled = None
        stream_policy_real_seen = False
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
        stream_counts = {
            "native_event_count": 0,
            "initial_root_count": 0,
            "search_computation_count": 0,
            "search_expansion_count": 0,
            "search_revisit_count": 0,
            "reset_control_count": 0,
            "stop_count": 0,
            "forced_stop_count": 0,
            "real_transition_count": 0,
            "real_action_stored_count": 0,
            "wait_count": 0,
        }

        while cursor < edge_count:
            if augmented_steps >= watchdog:
                raise RuntimeError(
                    "dynamic imitation watchdog exceeded "
                    f"({augmented_steps}/{watchdog}); phase/barrier did not advance"
                )
            pre_env_out = env_out if event_sink is not None else None
            pre_trace = (
                planner.dynamic_trace_snapshot(include_xs=False, include_hs=True)
                if event_sink is not None else None
            )
            if voc_feature_observer is not None:
                voc_feature_observer.begin_row()
            if collect_result:
                actor_out, actor_state = self.actor_net(
                    env_out,
                    actor_state,
                    compute_loss=False,
                    greedy=not training,
                )
            else:
                with torch.no_grad():
                    actor_out, actor_state = self.actor_net(
                        env_out,
                        actor_state,
                        compute_loss=False,
                        greedy=not training,
                    )
            captured_voc_features = (
                voc_feature_observer.take(getattr(actor_out, "voc_q", None))
                if voc_feature_observer is not None
                else None
            )
            decision = adapter.prepare(actor_out, actions[:, cursor])
            voc_diagnostics = (
                _voc_trace_diagnostics(
                    actor_out,
                    captured_voc_features,
                    flags=self.flags,
                    ema_head_state=voc_ema_head_state_dict,
                    effective_think_cost=effective_think_cost,
                )
                if event_sink is not None
                else None
            )
            if event_sink is not None and augmented_steps == 0:
                initial_observed_info = dict(info)
                initial_observed_info.update(
                    {
                        "effective_primary_action": (
                            adapter.effective_primary.detach().clone()
                        ),
                        "effective_search_control": (
                            adapter.effective_control.detach().clone()
                        ),
                        "im_episode_return": episode_accumulators["im"].clone(),
                        "cur_episode_return": episode_accumulators["cur"].clone(),
                        "think_episode_return": episode_accumulators[
                            "think"
                        ].clone(),
                    }
                )
                initial_event = build_dynamic_trace_event(
                    initial_root=True,
                    augmented_index=-1,
                    cursor_before=cursor,
                    cursor_after=cursor,
                    edge_count=edge_count,
                    score_mask=bool(score_mask[0, cursor].item()),
                    human_action=int(actions[0, cursor].item()),
                    pre_env_out=pre_env_out,
                    post_env_out=pre_env_out,
                    actor_out=actor_out,
                    decision=decision,
                    reward=_last(pre_env_out.reward),
                    done=_last(pre_env_out.done),
                    truncated=_last(pre_env_out.truncated_done),
                    raw_info=info,
                    observed_info=initial_observed_info,
                    pre_cenv=pre_trace,
                    post_cenv=pre_trace,
                    predicates=_initial_trace_predicates(),
                    voc=voc_diagnostics,
                    pre_observation_source_index=(
                        None
                        if observation_source_index is None
                        else int(observation_source_index[0])
                    ),
                    post_observation_source_index=(
                        None
                        if observation_source_index is None
                        else int(observation_source_index[0])
                    ),
                    sr_hidden_shape=sr_hidden_shape,
                    vp_hidden_shape=vp_hidden_shape,
                    image_mode=image_mode,
                )
                event_sink.write_event(initial_event)
                stream_counts["native_event_count"] += 1
                stream_counts["initial_root_count"] += 1
            real_indices = torch.nonzero(
                decision.real_policy_mask, as_tuple=False
            ).flatten()
            if real_indices.numel() and collect_result:
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
            elif real_indices.numel():
                if real_indices.tolist() != [0]:
                    raise RuntimeError(
                        "streaming POLICY_REAL mask must select its sole batch row"
                    )
                stream_policy_real_seen = True

            if model_done_observer is not None:
                model_done_observer.begin_step()
            with torch.no_grad():
                states, reward, done, truncated, raw_info = planner.step(
                    (
                        decision.execution_primary,
                        decision.execution_control,
                    ),
                    self.model_net,
                )
            post_trace = (
                planner.dynamic_trace_snapshot(include_xs=False, include_hs=True)
                if event_sink is not None else None
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
            augmented_steps += 1

            real_transition = _as_bool_tensor(
                info.get("real_transition", False), device=self.device
            ).reshape(-1)
            if event_sink is not None:
                predicates = dynamic_trace_predicates(post_trace, info)
                captured_model_done = (
                    model_done_observer.take(
                        model_expansion=int(predicates["internal_status"]) == TRACE_SEARCH_EXPANSION
                    ) if model_done_observer is not None else None
                )
                cursor_after = cursor + int(predicates["is_real_transition"])
                event = build_dynamic_trace_event(
                    initial_root=False,
                    augmented_index=augmented_steps - 1,
                    cursor_before=cursor,
                    cursor_after=cursor_after,
                    edge_count=edge_count,
                    score_mask=bool(score_mask[0, cursor].item()),
                    human_action=int(actions[0, cursor].item()),
                    pre_env_out=pre_env_out,
                    post_env_out=next_env_out,
                    actor_out=actor_out,
                    decision=decision,
                    reward=reward,
                    done=done,
                    truncated=truncated,
                    raw_info=raw_info,
                    observed_info=info,
                    pre_cenv=pre_trace,
                    post_cenv=post_trace,
                    predicates=predicates,
                    voc=voc_diagnostics,
                    model_done_logit=captured_model_done,
                    pre_observation_source_index=(
                        None
                        if observation_source_index is None
                        else int(observation_source_index[cursor])
                    ),
                    post_observation_source_index=(
                        None
                        if observation_source_index is None
                        else int(observation_source_index[cursor_after])
                    ),
                    sr_hidden_shape=sr_hidden_shape,
                    vp_hidden_shape=vp_hidden_shape,
                    image_mode=image_mode,
                )
                event_sink.write_event(event)
                stream_counts["native_event_count"] += 1
                stream_counts["search_computation_count"] += int(
                    predicates["is_search_computation"]
                )
                stream_counts["search_expansion_count"] += int(
                    predicates["is_search_expansion"]
                )
                stream_counts["search_revisit_count"] += int(
                    predicates["is_search_revisit"]
                )
                stream_counts["reset_control_count"] += int(
                    predicates["is_reset_control"]
                )
                stream_counts["stop_count"] += int(predicates["is_stop"])
                stream_counts["forced_stop_count"] += int(
                    bool(_trace_scalar(info.get("forced_stop", False)))
                )
                stream_counts["real_transition_count"] += int(
                    predicates["is_real_transition"]
                )
                stream_counts["real_action_stored_count"] += int(
                    predicates["is_real_action_stored"]
                )
                stream_counts["wait_count"] += int(predicates["is_wait"])

            env_out = _move_env_out(next_env_out, self.device)
            if torch.any(real_transition):
                if real_transition.shape != (batch_size,) or not torch.all(
                    real_transition
                ):
                    raise RuntimeError(
                        "behavior sequence cursor may advance only on a "
                        "full-batch real_transition"
                    )
                if collect_result and not torch.all(filled[cursor]):
                    missing = torch.nonzero(
                        ~filled[cursor], as_tuple=False
                    ).flatten().tolist()
                    raise RuntimeError(
                        f"real transition {cursor} lacks POLICY_REAL logits "
                        f"for batch rows {missing}"
                    )
                if not collect_result and not stream_policy_real_seen:
                    raise RuntimeError(
                        f"real transition {cursor} lacks its POLICY_REAL row"
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
                stream_policy_real_seen = False

        if not collect_result:
            if cursor != edge_count:
                raise RuntimeError("streaming replay ended before all real edges")
            return {
                "edge_count": edge_count,
                "scored_edge_count": int(score_mask.sum().item()),
                "augmented_steps": augmented_steps,
                **stream_counts,
            }

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
            noop_action_index=self.noop_action_index,
        )

    def rollout(
        self,
        batch: Mapping[str, Any],
        tree_carry: bool = True,
        training: bool = True,
    ) -> DynamicImitationResult:
        """Run the existing differentiable batched imitation rollout.

        This remains the training API.  It intentionally does not install a
        recorder, inspect the Cython tree, or alter actor train/eval state.
        """

        return self._run_rollout(
            batch,
            tree_carry=tree_carry,
            training=training,
            event_sink=None,
            collect_result=True,
        )

    def stream_rollout_episode(
        self,
        batch: Mapping[str, Any],
        *,
        event_sink: Any,
        tree_carry: bool = True,
        policy_mode: str = "checkpoint",
        voc_ema_head_state_dict: Optional[Mapping[str, Any]] = None,
        effective_think_cost: Optional[float] = None,
    ) -> Dict[str, int]:
        """Replay one complete batch-one episode into ``event_sink``.

        ``event_sink`` owns storage lifecycle and must implement only
        ``write_event(event: Mapping[str, Any])``.  This method never calls
        ``close``, ``finalize`` or ``abort``.  With an
        ``observation_provider`` batch, frames are read lazily at real edges;
        no result-sized tensor/list is accumulated in this runner.
        ``checkpoint`` preserves checkpoint evaluation semantics (including
        stochastic VoC evaluation when configured); ``greedy`` is fully
        deterministic, and ``sampled`` uses training-style action sampling.
        A sink may declare ``image_mode`` (or ``images``) as one of
        none/real/imagined/all.  Every profile retains raw ``predicted_xs``;
        imagined/all may derive a display image from it without another model
        forward.  Real frames remain source-index references.
        """

        if policy_mode not in ("checkpoint", "greedy", "sampled"):
            raise ValueError(
                "policy_mode must be 'checkpoint', 'greedy', or 'sampled'"
            )
        if event_sink is None or not callable(
            getattr(event_sink, "write_event", None)
        ):
            raise TypeError("event_sink must implement write_event(event)")
        validate_stream_behavior_episode(batch)

        actor_was_training = self.actor_net.training
        runtime_overrides = []
        if policy_mode == "greedy":
            # ActorNet's historical `greedy=True` still samples the VoC gate
            # when voc_eval_stochastic=true.  Also, the epsilon-execution
            # branch deliberately samples an exact 1/2 tie.  A trace-requested
            # greedy mode temporarily selects argmax on the underlying soft
            # gate and every primary/bout head, then restores checkpoint state.
            for module in self.actor_net.modules():
                for name, value in (
                    ("voc_eval_stochastic", False),
                    ("voc_gate_epsilon_greedy_execution", False),
                ):
                    if hasattr(module, name):
                        runtime_overrides.append(
                            (module, name, getattr(module, name))
                        )
                        setattr(module, name, value)
        self.actor_net.eval()
        try:
            with _VoCFeatureObserver(self.actor_net) as observer, \
                    _ModelDoneLogitObserver(self.model_net) as done_observer:
                summary = self._run_rollout(
                    batch,
                    tree_carry=tree_carry,
                    training=policy_mode == "sampled",
                    event_sink=event_sink,
                    collect_result=False,
                    voc_feature_observer=observer,
                    model_done_observer=done_observer,
                    voc_ema_head_state_dict=voc_ema_head_state_dict,
                    effective_think_cost=effective_think_cost,
                )
        finally:
            for module, name, value in reversed(runtime_overrides):
                setattr(module, name, value)
            self.actor_net.train(actor_was_training)
        return summary

    def close(self) -> None:
        if self._planner is not None and hasattr(self._planner, "close"):
            self._planner.close()
        self._planner = None
        self._behavior_env = None
        self._planner_key = None
