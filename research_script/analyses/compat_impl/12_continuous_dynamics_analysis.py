#!/usr/bin/env python3
"""Cross-decision continuous-control analyses computed only from the shared 00 cache.

A temporal origin, B persistence hierarchy, C plan reuse, D plan revision,
E switch destination, F within-bout accumulation, G urgency interactions.

Examples
--------
python research_script/12_continuous_dynamics_analysis.py --subject 1 --game 1 \
    --shared-cache-root research_script/outputs/heldout_5m_20260921/00_dynamic_cache
python research_script/12_continuous_dynamics_analysis.py --game 1 --aggregate-subjects 1 2 3 4 5 6
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp

from shared_analysis_cache import (
    DEFAULT_SHARED_CACHE_ROOT,
    KEYS,
    cache_paths,
    validate_manifest,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE04_PATH = SCRIPT_DIR / "04_search_evidence_geometry_pathway_v2.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "12_continuous_dynamics"
SCHEMA_VERSION = 1
ANALYSIS_VERSION = "2026-09-23-continuous-dynamics-cache-only-v1"
# Thinker training config "discounting".
GAMMA = 0.99
EPS = 1e-12
MIN_TRAIN_ROWS = 400
CLOGIT_MAXITER = 1000
ANALYSES = ("A", "B", "C", "D", "E", "F", "G")
RISK_SETS = ("exact_switch", "noop_to_active", "active_to_noop", "active_to_active")
DEFAULT_LAGS = (1, 2, 3, 4, 6, 8, 12, 16)
DEFAULT_HORIZONS = (1, 2, 3, 4, 6, 8, 12, 16)
HISTORY_LAGS = (1, 2, 3)
EWMA_HALFLIFE = 4.0
CARRY_LIMIT = 30
N_VISUAL = 40
ACCUMULATED = (
    "abs_revision_sum", "signed_revision_sum", "inc_revision_step",
    "alt_revision_step", "searched",
)

INK = "#33363A"
GRID = "#D9DDE3"
MUTED = "#8A8F96"
BLUE = "#2F6DB5"
ORANGE = "#CF7B35"
PURPLE = "#7B5EA7"
SERIES_COLORS = (BLUE, ORANGE, PURPLE)

_T0 = time.time()


def log(message: str) -> None:
    print(f"[12 +{time.time() - _T0:8.1f}s] {message}", flush=True)


def _load_base04():
    spec = importlib.util.spec_from_file_location("base04_v2_for_12", BASE04_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {BASE04_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if int(module.STAGE3_MIN_TRAIN_ROWS) != MIN_TRAIN_ROWS:
        raise RuntimeError("04 Stage-3 minimum training rows changed; update analysis 12")
    return module


B04 = _load_base04()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Cache loading and cache-only feature construction
# ---------------------------------------------------------------------------

def lag_index(segment: np.ndarray, k: int) -> np.ndarray:
    """Row index k decisions earlier (k>0) or later (k<0) in the same segment, else -1."""
    n = len(segment)
    source = np.arange(n) - int(k)
    inside = np.flatnonzero((source >= 0) & (source < n))
    result = np.full(n, -1, dtype=np.int64)
    same = segment[source[inside]] == segment[inside]
    result[inside[same]] = source[inside[same]]
    return result


def take_lag(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    output = np.full(array.shape, np.nan, dtype=float)
    valid = index >= 0
    output[valid] = array[index[valid]]
    return output


def load_subject_inputs(
    cache_root: Path, subject: int, game: int
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict[str, object]]:
    paths = cache_paths(cache_root, subject, game)
    manifest = validate_manifest(paths, subject=subject, game=game)
    n_actions = int(manifest.get("num_actions", manifest.get("n_actions", 6)))
    B04.N_ACTIONS = n_actions
    windows = pd.read_csv(paths.model_control_windows, low_memory=False)
    alignment = pd.read_csv(paths.visual_alignment, usecols=KEYS + ["actor_row"])
    windows = windows.merge(alignment, on=KEYS, how="left", validate="one_to_one")
    if windows["actor_row"].isna().any():
        raise ValueError("Visual alignment does not cover every real step")
    with np.load(paths.visual_controls, allow_pickle=False) as archive:
        descriptor = np.asarray(archive["descriptor"], dtype=float)
    if descriptor.shape[1] != N_VISUAL:
        raise ValueError(f"Expected {N_VISUAL} visual dimensions, found {descriptor.shape}")
    windows = windows.sort_values(["run_label", "real_step"], kind="mergesort").reset_index(drop=True)
    visual = descriptor[windows["actor_row"].to_numpy(dtype=int)]
    events = pd.read_csv(
        paths.persistence_events,
        usecols=KEYS + ["root_q_revision", "value_revision_valid"],
        low_memory=False,
    )
    inputs = {
        name: {"path": str(Path(path).resolve()), "sha256": sha256_path(Path(path))}
        for name, path in (
            ("manifest", paths.manifest),
            ("model_control_windows", paths.model_control_windows),
            ("persistence_events", paths.persistence_events),
            ("visual_controls", paths.visual_controls),
            ("visual_alignment", paths.visual_alignment),
        )
    }
    meta = {"n_actions": n_actions, "cache_manifest": manifest, "inputs": inputs}
    return windows, events, visual, meta


def assign_segments(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    key = frame["run_label"].astype(str) + "|" + frame["episode_id"].astype(str)
    segment = pd.factorize(key)[0].astype(np.int64)
    if int(np.sum(np.diff(segment) != 0)) != int(segment.max()):
        raise ValueError("Run/episode segments are not contiguous blocks")
    step = frame["real_step"].to_numpy(dtype=np.int64)
    same = np.r_[False, segment[1:] == segment[:-1]]
    if np.any(same & (np.r_[0, np.diff(step)] != 1)):
        raise ValueError("real_step is not contiguous within a run/episode segment")
    position = frame.groupby(segment, sort=False).cumcount().to_numpy(dtype=np.int64)
    return segment, position


def _grouped_ewm_excluding_current(values: pd.Series, segment: np.ndarray) -> np.ndarray:
    shifted = values.groupby(segment, sort=False).shift(1)
    smoothed = shifted.groupby(segment, sort=False).transform(
        lambda series: series.ewm(halflife=EWMA_HALFLIFE, adjust=True).mean()
    )
    return smoothed.fillna(0.0).to_numpy(dtype=float)


def build_features(
    windows: pd.DataFrame, events: pd.DataFrame, visual: np.ndarray, n_actions: int
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    frame = windows
    n = len(frame)
    rows = np.arange(n)
    segment, position = assign_segments(frame)
    run_labels = frame["run_label"].astype(str).to_numpy()
    num = lambda column: pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    action_block = lambda prefix: np.column_stack([num(f"{prefix}{a}") for a in range(n_actions)])

    prior = action_block("model_root_p_")
    prior = prior / np.clip(prior.sum(axis=1, keepdims=True), EPS, None)
    log_prior = np.log(np.clip(prior, EPS, None))
    actor_terminal = action_block("actor_terminal_p_")
    actor_terminal = actor_terminal / np.clip(actor_terminal.sum(axis=1, keepdims=True), EPS, None)
    log_actor_terminal = np.log(np.clip(actor_terminal, EPS, None))
    q_final = action_block("root_q_final_")
    visited = action_block("root_n_final_") > 0
    value = num("root_value")
    # Unvisited root Q is a 0 placeholder in the cache, never a value estimate.
    q_advantage = np.where(visited, q_final - value[:, None], 0.0)
    incumbent = pd.to_numeric(frame["incumbent_action"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    human = pd.to_numeric(frame["human_action"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    before = pd.to_numeric(frame["action_before_incumbent"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    incumbent_valid = (incumbent >= 0) & (incumbent < n_actions)
    incumbent_clip = np.clip(incumbent, 0, n_actions - 1)

    F = pd.DataFrame(index=frame.index)
    bout_age = num("bout_age")
    F["log_bout_age"] = np.log1p(np.clip(bout_age, 0, None))
    F["log_bout_age_sq"] = F["log_bout_age"] ** 2
    F["log_noop_run_prev"] = np.log1p(np.clip(num("noop_run_length_prev"), 0, None))
    switched = pd.Series(((human != incumbent) & incumbent_valid).astype(float))
    previous_switch = switched.groupby(segment, sort=False).shift(1)
    for width in (10, 30):
        F[f"recent_switches_{width}"] = (
            previous_switch.groupby(segment, sort=False)
            .transform(lambda series, w=width: series.rolling(w, min_periods=1).sum())
            .fillna(0.0).to_numpy()
        )
    F["run_progress"] = num("run_progress")
    F["run_progress_sq"] = F["run_progress"] ** 2
    F["prior_reward"] = num("prior_reward")
    F["prior_done"] = num("prior_done")
    active = pd.Series(np.where(incumbent >= 1, incumbent, np.nan))
    pre_bout_active = active.groupby(segment, sort=False).ffill().to_numpy()
    action_dummies: List[str] = []
    for a in range(1, n_actions):
        for name, source in (("incumbent_is", incumbent), ("previous_incumbent_is", before),
                             ("pre_bout_active_is", pre_bout_active)):
            F[f"{name}_{a}"] = (source == a).astype(float)
            action_dummies.append(f"{name}_{a}")
    session = pd.to_numeric(frame["session"], errors="coerce").fillna(-1).astype(int).to_numpy()
    for session_id in sorted(np.unique(session))[1:]:
        F[f"session_is_{session_id}"] = (session == session_id).astype(float)
    for index in range(N_VISUAL):
        F[f"visual_{index:02d}"] = visual[:, index]

    F["M"] = B04.transition_margin(prior, incumbent)
    F["M_comp"] = num("strongest_competitor_margin")
    F["logp_incumbent"] = np.where(incumbent_valid, log_prior[rows, incumbent_clip], np.nan)
    F["logp_noop"] = log_prior[:, 0]

    valid_events = events.loc[pd.to_numeric(events["value_revision_valid"], errors="coerce").eq(1)].copy()
    valid_events["revision"] = pd.to_numeric(valid_events["root_q_revision"], errors="coerce")
    valid_events["abs_revision"] = valid_events["revision"].abs()
    revision = valid_events.groupby(KEYS, sort=False).agg(
        abs_revision_mean=("abs_revision", "mean"), abs_revision_sum=("abs_revision", "sum"),
        signed_revision_mean=("revision", "mean"), signed_revision_sum=("revision", "sum"),
    ).reset_index()
    revision = frame[KEYS].merge(revision, on=KEYS, how="left", validate="one_to_one")
    F["U_log"] = np.log(np.clip(revision["abs_revision_mean"].to_numpy(dtype=float), EPS, None))
    F["revision_available"] = revision["abs_revision_mean"].notna().astype(float).to_numpy()
    F.loc[F["revision_available"] == 0, "U_log"] = np.nan
    F["Q_signed"] = revision["signed_revision_mean"].to_numpy(dtype=float)
    budget = num("n_status2")
    F["log_budget"] = np.log1p(np.clip(budget, 0, None))
    F["searched"] = (budget > 0).astype(float)
    F["search_depth"] = num("mean_bout_terminal_depth")
    F["deep_event_fraction"] = num("deep_event_fraction")
    F["branch_diversity"] = num("root_branch_allocation_diversity")
    inc_visited = visited[rows, incumbent_clip] & incumbent_valid
    F["inc_visited"] = inc_visited.astype(float)
    F["inc_Qadv"] = np.where(inc_visited, q_advantage[rows, incumbent_clip], np.nan)
    alternative = visited.copy()
    alternative[rows[incumbent_valid], incumbent[incumbent_valid]] = False
    F["alt_visited_any"] = alternative.any(axis=1).astype(float)
    F["alt_best_Qadv"] = np.where(
        alternative.any(axis=1), np.max(np.where(alternative, q_advantage, -np.inf), axis=1), np.nan
    )
    F["value_gap"] = F["alt_best_Qadv"] - F["inc_Qadv"]
    F["inc_revision"] = num("mean_incumbent_root_q_revision")
    F["alt_revision"] = num("mean_alternative_root_q_revision")

    previous = lag_index(segment, 1)
    value_previous = take_lag(value, previous)
    done = F["prior_done"].to_numpy(dtype=float)
    reward = F["prior_reward"].to_numpy(dtype=float)
    F["td"] = reward + GAMMA * (1.0 - done) * value - value_previous
    F["abs_td"] = np.abs(F["td"])
    human_clip = np.clip(human, 0, n_actions - 1)
    taken_visited = visited[rows, human_clip] & (human >= 0)
    q_taken = np.where(taken_visited, q_final[rows, human_clip], np.nan)
    F["td_plan"] = reward + GAMMA * (1.0 - done) * value - take_lag(q_taken, previous)
    F["td_plan_available"] = np.isfinite(F["td_plan"]).astype(float)
    F["abs_td_plan"] = np.abs(F["td_plan"])
    F["vis_change"] = np.linalg.norm(visual - take_lag(visual, previous), axis=1)
    F["vis_accel"] = F["vis_change"] - take_lag(F["vis_change"].to_numpy(), previous)
    F["abs_dV"] = np.abs(value - value_previous)

    for k in HISTORY_LAGS:
        source = lag_index(segment, k)
        ok = (source >= 0) & incumbent_valid
        safe = np.where(ok, source, 0)
        for column in ("M", "U_log", "revision_available", "Q_signed", "log_budget"):
            F[f"{column}_lag{k}"] = take_lag(F[column].to_numpy(), source)
        prev_visited = visited[safe, incumbent_clip] & ok
        F[f"prev{k}_inc_visited"] = prev_visited.astype(float)
        F[f"prev{k}_inc_Qadv"] = np.where(prev_visited, q_advantage[safe, incumbent_clip], np.nan)
        prev_alt = visited[safe].copy() & ok[:, None]
        prev_alt[rows[incumbent_valid], incumbent[incumbent_valid]] = False
        F[f"prev{k}_alt_best_Qadv"] = np.where(
            prev_alt.any(axis=1), np.max(np.where(prev_alt, q_advantage[safe], -np.inf), axis=1), np.nan
        )
        F[f"prev{k}_value_gap"] = F[f"prev{k}_alt_best_Qadv"] - F[f"prev{k}_inc_Qadv"]

    # Dynamic Thinker visits ~one root branch per decision, so incumbent-vs-alternative
    # comparisons exist only across decisions: carry each action's latest evaluation forward.
    carried_q = np.full((n, n_actions), np.nan)
    carried_age = np.full((n, n_actions), np.nan)
    for a in range(n_actions):
        q = pd.Series(np.where(visited[:, a], q_advantage[:, a], np.nan))
        when = pd.Series(np.where(visited[:, a], position, np.nan).astype(float))
        q = q.groupby(segment, sort=False).shift(1).groupby(segment, sort=False).ffill(limit=CARRY_LIMIT - 1)
        when = when.groupby(segment, sort=False).shift(1).groupby(segment, sort=False).ffill(limit=CARRY_LIMIT - 1)
        carried_q[:, a] = q.to_numpy()
        carried_age[:, a] = position - when.to_numpy()
    carried_inc = np.where(incumbent_valid, carried_q[rows, incumbent_clip], np.nan)
    carried_alt = carried_q.copy()
    carried_alt[rows[incumbent_valid], incumbent[incumbent_valid]] = np.nan
    has_alt = np.isfinite(carried_alt).any(axis=1)
    F["carried_inc_Qadv"] = carried_inc
    F["carried_inc_age"] = np.where(incumbent_valid, carried_age[rows, incumbent_clip], np.nan)
    F["carried_inc_available"] = np.isfinite(carried_inc).astype(float)
    F["carried_alt_best_Qadv"] = np.where(has_alt, np.nanmax(np.where(np.isfinite(carried_alt), carried_alt, -np.inf), axis=1), np.nan)
    F["carried_alt_available"] = has_alt.astype(float)
    F["carried_gap"] = F["carried_alt_best_Qadv"] - F["carried_inc_Qadv"]

    step_values = pd.DataFrame({
        "abs_revision_sum": revision["abs_revision_sum"].fillna(0.0).to_numpy(),
        "signed_revision_sum": revision["signed_revision_sum"].fillna(0.0).to_numpy(),
        "inc_revision_step": F["inc_revision"].fillna(0.0).to_numpy(),
        "alt_revision_step": F["alt_revision"].fillna(0.0).to_numpy(),
        "searched": F["searched"].to_numpy(),
    })
    new_bout = np.r_[True, (segment[1:] != segment[:-1]) | (incumbent[1:] != incumbent[:-1])]
    bout_id = np.cumsum(new_bout)
    prior_in_bout = pd.Series(bout_id).groupby(bout_id).cumcount().to_numpy(dtype=float)
    accumulated = {}
    for column in ACCUMULATED:
        series = step_values[column]
        cumulative = series.groupby(bout_id).cumsum().to_numpy() - series.to_numpy()
        accumulated[f"ewma_{column}"] = _grouped_ewm_excluding_current(series, segment)
        accumulated[f"bout_cum_{column}"] = cumulative
        accumulated[f"bout_mean_{column}"] = np.divide(
            cumulative, prior_in_bout, out=np.zeros(n), where=prior_in_bout > 0
        )
    F = pd.concat([F, pd.DataFrame(accumulated, index=F.index)], axis=1)

    z = lambda column: _zscore(F[column].to_numpy(dtype=float))
    # Interaction components use feature-only full-data centring; the outcome never enters.
    components = {name: np.nan_to_num(z(name)) for name in ("M_comp", "U_log", "vis_change", "abs_dV", "log_bout_age", "abs_td", "abs_td_plan", "M")}
    interactions = {
        f"{left}_x_{right}": components[left] * components[right]
        for left, right in (("M_comp", "vis_change"), ("U_log", "vis_change"), ("M_comp", "abs_dV"),
                            ("U_log", "abs_dV"), ("M_comp", "log_bout_age"), ("U_log", "log_bout_age"),
                            ("abs_td", "U_log"), ("abs_td", "M"), ("abs_td_plan", "U_log"))
    }
    F = pd.concat([F, pd.DataFrame(interactions, index=F.index)], axis=1)

    arrays = {
        "segment": segment, "position": position, "run_label": run_labels, "session": session,
        "incumbent": incumbent,
        "human": human, "before": before, "log_prior": log_prior, "log_actor_terminal": log_actor_terminal,
        "visited": visited, "q_advantage": q_advantage, "visual": visual, "bout_id": bout_id,
    }
    return F.astype(np.float32), arrays


def _zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    output = np.full(array.shape, np.nan)
    if finite.sum() < 2:
        return output
    scale = float(np.std(array[finite]))
    output[finite] = (array[finite] - float(np.mean(array[finite]))) / (scale if scale > EPS else 1.0)
    return output


def feature_groups(F: pd.DataFrame, n_actions: int) -> Dict[str, List[str]]:
    dummies = [c for c in F.columns if c.split("_")[-1].isdigit() and c.rsplit("_", 1)[0] in
               ("incumbent_is", "previous_incumbent_is", "pre_bout_active_is")]
    return {
        "history": ["log_bout_age", "log_bout_age_sq", "log_noop_run_prev", "recent_switches_10",
                    "recent_switches_30", "run_progress", "run_progress_sq", "prior_reward",
                    "prior_done"] + dummies,
        "checkpoint": [c for c in F.columns if c.startswith("session_is_")],
        "visual": [f"visual_{i:02d}" for i in range(N_VISUAL)],
        "root": ["M", "M_comp", "logp_incumbent", "logp_noop"],
        "search": ["U_log", "revision_available", "Q_signed", "log_budget", "searched",
                   "search_depth", "deep_event_fraction", "branch_diversity", "inc_Qadv",
                   "inc_visited", "alt_best_Qadv", "alt_visited_any", "value_gap",
                   "inc_revision", "alt_revision"],
        "previous_evaluation": [
            f"{column}_lag{k}" for k in HISTORY_LAGS
            for column in ("M", "U_log", "revision_available", "Q_signed", "log_budget")
        ] + [f"prev{k}_{column}" for k in HISTORY_LAGS
             for column in ("inc_visited", "inc_Qadv", "alt_best_Qadv", "value_gap")]
          + ["carried_inc_Qadv", "carried_inc_age", "carried_inc_available",
             "carried_alt_best_Qadv", "carried_alt_available", "carried_gap"],
        "surprise": ["td", "abs_td", "td_plan", "abs_td_plan", "td_plan_available"],
        "surprise_interactions": ["abs_td_x_U_log", "abs_td_x_M", "abs_td_plan_x_U_log"],
        "ewma": [f"ewma_{c}" for c in ACCUMULATED],
        "bout": [f"bout_cum_{c}" for c in ACCUMULATED] + [f"bout_mean_{c}" for c in ACCUMULATED],
        "urgency": ["vis_change", "vis_accel", "abs_dV"],
        "urgency_interactions": ["M_comp_x_vis_change", "U_log_x_vis_change", "M_comp_x_abs_dV",
                                 "U_log_x_abs_dV", "M_comp_x_log_bout_age", "U_log_x_log_bout_age"],
    }


def risk_outcome(incumbent: np.ndarray, human: np.ndarray, n_actions: int, name: str) -> Tuple[np.ndarray, np.ndarray]:
    valid_human = (human >= 0) & (human < n_actions)
    active = (incumbent >= 1) & (incumbent < n_actions)
    if name == "exact_switch":
        return (incumbent >= 0) & valid_human, (human != incumbent).astype(int)
    if name == "noop_to_active":
        return (incumbent == 0) & valid_human, (human != 0).astype(int)
    if name == "active_to_noop":
        return active & valid_human & ((human == 0) | (human == incumbent)), (human == 0).astype(int)
    if name == "active_to_active":
        return active & valid_human & (human != 0), (human != incumbent).astype(int)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Held-out model machinery
# ---------------------------------------------------------------------------

def nested_binary_loro(
    F: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    *,
    analysis: str,
    risk_name: str,
    stages: Mapping[str, Sequence[str]],
    increments: Sequence[Tuple[str, str]],
    hard: Sequence[str],
    n_actions: int,
    min_position: int = 1,
    report: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[List[dict], List[dict], dict]:
    risk, outcome = risk_outcome(arrays["incumbent"], arrays["human"], n_actions, risk_name)
    finite = np.isfinite(F[list(hard)].to_numpy(dtype=float)).all(axis=1)
    complete = risk & finite & (arrays["position"] >= min_position)
    rows = np.flatnonzero(complete)
    sub = F.iloc[rows]
    y = outcome[rows]
    runs = arrays["fold"][rows]
    folds: List[dict] = []
    coefficients: List[dict] = []
    for held in pd.unique(runs):
        test = runs == held
        train = ~test
        losses: Dict[str, float] = {}
        reasons: Dict[str, str] = {}
        for stage, columns in stages.items():
            columns = list(columns)
            fit = B04._fit_binary_l2(sub.loc[train, columns].to_numpy(dtype=float), y[train])
            if fit is None:
                losses[stage] = np.nan
                reasons[stage] = (f"train rows={int(train.sum())} < {MIN_TRAIN_ROWS}"
                                  if train.sum() < MIN_TRAIN_ROWS else "training fold has one outcome class")
                continue
            probability = B04._predict_binary(fit, sub.loc[test, columns].to_numpy(dtype=float))
            losses[stage] = float(np.mean(B04.binary_log_loss(y[test], probability)))
            for feature in (report or {}).get(stage, ()):
                if feature not in columns:
                    continue
                coefficients.append({
                    "analysis": analysis, "risk_set": risk_name, "stage": stage, "feature": feature,
                    "fold": held, "beta": float(fit["model"].coef_[0, columns.index(feature)]),
                    "effect_id": f"{analysis}::{risk_name}::{stage}::{feature}",
                })
        for start, end in increments:
            folds.append({
                "analysis": analysis, "risk_set": risk_name, "increment": f"{start}_to_{end}",
                "subset": "all", "effect_id": f"{analysis}::{risk_name}::{start}_to_{end}",
                "fold": held, "beta": losses[start] - losses[end],
                "start_log_loss": losses[start], "end_log_loss": losses[end],
                "n_test": int(test.sum()), "test_events": int(y[test].sum()),
                "n_train": int(train.sum()), "train_events": int(y[train].sum()),
                "not_estimable_reason": "; ".join(f"{s}: {r}" for s, r in reasons.items() if s in (start, end)),
            })
    qa = {"risk_rows": int(risk.sum()), "complete_rows": int(complete.sum()),
          "events": int(y.sum()), "runs": int(len(pd.unique(runs))),
          "stages": {stage: list(columns) for stage, columns in stages.items()}}
    return folds, coefficients, qa


def fit_conditional_logit(x: np.ndarray, y: np.ndarray, mask: np.ndarray, l2: float = 1.0) -> Optional[dict]:
    """Shared-coefficient softmax over alternatives; L2 on standardized features (C=1 scale)."""
    n, n_alt, p = x.shape
    if n < MIN_TRAIN_ROWS:
        return None
    flat = x[mask]
    mean = np.nanmean(flat, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.nanstd(flat, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    z = _standardize_alternatives(x, mask, mean, scale)
    chosen = z[np.arange(n), y]

    def objective(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        utility = np.where(mask, z @ beta, -np.inf)
        normalizer = logsumexp(utility, axis=1)
        probability = np.exp(utility - normalizer[:, None])
        nll = float(np.sum(normalizer - utility[np.arange(n), y]))
        gradient = np.einsum("nap,na->p", z, probability) - chosen.sum(axis=0)
        return nll + 0.5 * l2 * float(beta @ beta), gradient + l2 * beta

    result = minimize(objective, np.zeros(p), jac=True, method="L-BFGS-B",
                      options={"maxiter": CLOGIT_MAXITER})
    if not result.success:
        return {"converged": False, "message": str(result.message)}
    return {"converged": True, "beta": result.x, "mean": mean, "scale": scale}


def _standardize_alternatives(x, mask, mean, scale) -> np.ndarray:
    z = (x - mean) / scale
    z = np.where(np.isfinite(z), z, 0.0)
    return np.where(mask[:, :, None], z, 0.0)


def conditional_logit_log_loss(fit: Mapping[str, object], x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = _standardize_alternatives(x, mask, fit["mean"], fit["scale"])
    utility = np.where(mask, z @ fit["beta"], -np.inf)
    return logsumexp(utility, axis=1) - utility[np.arange(len(y)), y]


def nested_clogit_loro(
    blocks: Mapping[str, Tuple[np.ndarray, Sequence[str]]],
    *,
    y: np.ndarray,
    mask: np.ndarray,
    runs: np.ndarray,
    analysis: str,
    label: str,
    stages: Mapping[str, Sequence[str]],
    increments: Sequence[Tuple[str, str]],
    subsets: Optional[Mapping[str, np.ndarray]] = None,
    report: Optional[Mapping[str, Sequence[str]]] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> Tuple[List[dict], List[dict]]:
    subsets = {"all": np.ones(len(y), dtype=bool), **(subsets or {})}
    folds: List[dict] = []
    coefficients: List[dict] = []
    stage_x = {stage: np.concatenate([blocks[b][0] for b in names], axis=2) for stage, names in stages.items()}
    stage_names = {stage: [f for b in names for f in blocks[b][1]] for stage, names in stages.items()}
    for held in pd.unique(runs):
        test = runs == held
        train = ~test
        losses: Dict[str, Optional[np.ndarray]] = {}
        reasons: Dict[str, str] = {}
        for stage, x in stage_x.items():
            fit = fit_conditional_logit(x[train], y[train], mask[train])
            if fit is None or not fit["converged"]:
                losses[stage] = None
                reasons[stage] = "too few training rows" if fit is None else f"not converged: {fit['message']}"
                continue
            losses[stage] = conditional_logit_log_loss(fit, x[test], y[test], mask[test])
            for feature in (report or {}).get(stage, ()):
                coefficients.append({
                    "analysis": analysis, "risk_set": label, "stage": stage, "feature": feature,
                    "fold": held, "beta": float(fit["beta"][stage_names[stage].index(feature)]),
                    "effect_id": f"{analysis}::{label}::{stage}::{feature}",
                })
        for subset, members in subsets.items():
            local = members[test]
            for start, end in increments:
                if losses[start] is None or losses[end] is None or local.sum() == 0:
                    beta, start_loss, end_loss = np.nan, np.nan, np.nan
                else:
                    start_loss = float(np.mean(losses[start][local]))
                    end_loss = float(np.mean(losses[end][local]))
                    beta = start_loss - end_loss
                folds.append({
                    "analysis": analysis, "risk_set": label, "increment": f"{start}_to_{end}",
                    "subset": subset,
                    "effect_id": f"{analysis}::{label}::{start}_to_{end}" + ("" if subset == "all" else f"::{subset}"),
                    "fold": held, "beta": beta, "start_log_loss": start_loss, "end_log_loss": end_loss,
                    "n_test": int(local.sum()), "n_train": int(train.sum()),
                    "not_estimable_reason": "; ".join(f"{s}: {r}" for s, r in reasons.items() if s in (start, end)),
                    **(extra or {}),
                })
    return folds, coefficients


def loro_ols_coefficients(
    F: pd.DataFrame, rows: np.ndarray, runs: np.ndarray, outcome: str, predictors: Sequence[str],
    *, analysis: str, label: str,
) -> List[dict]:
    y_all = F[outcome].to_numpy(dtype=float)[rows]
    x_all = F[list(predictors)].to_numpy(dtype=float)[rows]
    records = []
    for held in pd.unique(runs):
        train = runs != held
        x = x_all[train]
        y = y_all[train]
        if len(y) < MIN_TRAIN_ROWS:
            continue
        xs = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > EPS, x.std(axis=0), 1.0)
        ys = (y - y.mean()) / (y.std() if y.std() > EPS else 1.0)
        design = np.column_stack([np.ones(len(ys)), xs])
        beta = np.linalg.lstsq(design, ys, rcond=None)[0][1:]
        for feature, value in zip(predictors, beta):
            records.append({"analysis": analysis, "risk_set": label, "stage": "OLS", "feature": feature,
                            "fold": held, "beta": float(value),
                            "effect_id": f"{analysis}::{label}::OLS::{feature}"})
    return records


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def analysis_A(F, arrays, groups, n_actions, lags) -> Tuple[List[dict], List[dict], dict]:
    """Lag profile: stimulus or evaluation at t-k beyond the full current decision state."""
    segment = arrays["segment"]
    visual = arrays["visual"]
    extra = {}
    stim: Dict[int, List[str]] = {}
    evaluation: Dict[int, List[str]] = {}
    own_lag = [c for c in ("M", "U_log", "revision_available", "Q_signed", "log_budget", "inc_Qadv", "value_gap")
               if np.isfinite(F[c].to_numpy(dtype=float)).any()]
    for k in lags:
        source = lag_index(segment, k)
        lagged_visual = take_lag(visual, source)
        stim[k] = [f"visual_lag{k}_{i:02d}" for i in range(N_VISUAL)] + [f"prior_reward_lag{k}", f"vis_change_lag{k}"]
        for i in range(N_VISUAL):
            extra[stim[k][i]] = lagged_visual[:, i]
        extra[f"prior_reward_lag{k}"] = take_lag(F["prior_reward"].to_numpy(), source)
        extra[f"vis_change_lag{k}"] = take_lag(F["vis_change"].to_numpy(), source)
        evaluation[k] = [f"{c}_Alag{k}" for c in own_lag]
        for column in own_lag:
            extra[f"{column}_Alag{k}"] = take_lag(F[column].to_numpy(), source)
    wide = pd.concat([F, pd.DataFrame(extra, index=F.index).astype(np.float32)], axis=1)
    base = groups["history"] + groups["visual"] + groups["root"] + groups["search"]
    stages = {"base": base}
    increments = []
    for k in lags:
        stages[f"stim_lag{k}"] = base + stim[k]
        stages[f"eval_lag{k}"] = base + evaluation[k]
        increments += [("base", f"stim_lag{k}"), ("base", f"eval_lag{k}")]
    hard = groups["history"] + groups["visual"] + ["M"] + [c for k in lags for c in stim[k]] + [f"M_Alag{k}" for k in lags]
    folds, coefficients, qa = [], [], {}
    for risk in RISK_SETS:
        log(f"A temporal origin: {risk}")
        f, c, q = nested_binary_loro(wide, arrays, analysis="A_temporal_origin", risk_name=risk,
                                     stages=stages, increments=increments, hard=hard,
                                     n_actions=n_actions, min_position=max(lags) + 1)
        folds += f
        coefficients += c
        qa[risk] = {k: v for k, v in q.items() if k != "stages"}
    for row in folds:
        end = row["increment"].split("_to_")[1]
        row["block"], row["lag"] = end.rsplit("_lag", 1)[0], int(end.rsplit("_lag", 1)[1])
    return folds, coefficients, qa


def analysis_B(F, arrays, groups, n_actions) -> Tuple[List[dict], List[dict], dict]:
    """Autocorrelation -> stimulus -> root -> current search -> previous-decision evaluation."""
    stages = {"H0_autocorrelation": groups["history"]}
    stages["H1_stimulus"] = stages["H0_autocorrelation"] + groups["visual"]
    stages["H2_root"] = stages["H1_stimulus"] + groups["root"]
    stages["H3_search"] = stages["H2_root"] + groups["search"]
    stages["H4_previous_evaluation"] = stages["H3_search"] + groups["previous_evaluation"]
    names = list(stages)
    increments = list(zip(names[:-1], names[1:]))
    report = {
        "H3_search": ["M", "M_comp", "U_log", "Q_signed", "inc_revision", "alt_revision", "inc_Qadv", "alt_best_Qadv", "value_gap"],
        "H4_previous_evaluation": [c for c in groups["previous_evaluation"]
                                   if c.startswith(("prev1_", "carried_")) or c.endswith("_lag1")],
    }
    hard = groups["history"] + groups["visual"] + ["M", "M_lag1", "M_lag2", "M_lag3"]
    return _run_risk_sets(F, arrays, "B_persistence_hierarchy", stages, increments, hard, n_actions,
                          min_position=max(HISTORY_LAGS) + 1, report=report)


def analysis_D(F, arrays, groups, n_actions) -> Tuple[List[dict], List[dict], dict]:
    """Value surprise -> later search revision (OLS) and surprise x revision -> switch (LORO)."""
    base = groups["history"] + groups["visual"] + groups["root"] + groups["search"]
    stages = {"D0_current": base, "D1_surprise": base + groups["surprise"]}
    stages["D2_surprise_x_revision"] = stages["D1_surprise"] + groups["surprise_interactions"]
    increments = [("D0_current", "D1_surprise"), ("D1_surprise", "D2_surprise_x_revision")]
    report = {"D2_surprise_x_revision": groups["surprise"] + groups["surprise_interactions"] + ["U_log", "M"]}
    hard = groups["history"] + groups["visual"] + ["M", "abs_td"]
    folds, coefficients, qa = _run_risk_sets(F, arrays, "D_plan_revision", stages, increments, hard,
                                             n_actions, min_position=2, report=report)
    predictors = ["abs_td", "td", "abs_td_plan_filled", "td_plan_available", "U_log_lag1_filled",
                  "revision_available_lag1", "log_budget", "M_comp", "log_bout_age", "run_progress",
                  "prior_reward", "vis_change"]
    work = F.copy()
    work["abs_td_plan_filled"] = work["abs_td_plan"].fillna(0.0)
    work["U_log_lag1_filled"] = work["U_log_lag1"].fillna(work["U_log_lag1"].median())
    rows = np.flatnonzero(
        (F["revision_available"].to_numpy() == 1)
        & np.isfinite(work[predictors + ["U_log"]].to_numpy(dtype=float)).all(axis=1)
        & (arrays["position"] >= 2)
    )
    coefficients += loro_ols_coefficients(work, rows, arrays["fold"][rows], "U_log", predictors,
                                          analysis="D_plan_revision", label="search_revision_U")
    qa["search_revision_U"] = {"rows": int(len(rows)), "outcome": "U_log at decisions with valid revision"}
    return folds, coefficients, qa


def analysis_F(F, arrays, groups, n_actions) -> Tuple[List[dict], List[dict], dict]:
    """Within-bout cumulative evidence beyond an unbounded decaying history."""
    base = groups["history"] + groups["visual"] + groups["root"] + groups["search"]
    stages = {"F0_current": base, "F1_decaying_history": base + groups["ewma"]}
    stages["F2_bout_accumulation"] = stages["F1_decaying_history"] + groups["bout"]
    increments = [("F0_current", "F1_decaying_history"), ("F1_decaying_history", "F2_bout_accumulation")]
    report = {"F2_bout_accumulation": groups["bout"]}
    hard = groups["history"] + groups["visual"] + ["M"]
    return _run_risk_sets(F, arrays, "F_bout_accumulation", stages, increments, hard, n_actions,
                          min_position=1, report=report)


def analysis_G(F, arrays, groups, n_actions) -> Tuple[List[dict], List[dict], dict]:
    """Environmental change rate and elapsed bout time as moderators of M and U."""
    base = groups["history"] + groups["visual"] + groups["root"] + groups["search"] + groups["urgency"]
    stages = {"G0_main": base, "G1_interactions": base + groups["urgency_interactions"]}
    report = {"G1_interactions": groups["urgency"] + groups["urgency_interactions"] + ["M_comp", "U_log"]}
    hard = groups["history"] + groups["visual"] + ["M", "vis_change", "abs_dV"]
    return _run_risk_sets(F, arrays, "G_urgency", stages, [("G0_main", "G1_interactions")], hard,
                          n_actions, min_position=2, report=report)


def _run_risk_sets(F, arrays, analysis, stages, increments, hard, n_actions, *, min_position, report):
    folds, coefficients, qa = [], [], {}
    for risk in RISK_SETS:
        log(f"{analysis}: {risk}")
        f, c, q = nested_binary_loro(F, arrays, analysis=analysis, risk_name=risk, stages=stages,
                                     increments=increments, hard=hard, n_actions=n_actions,
                                     min_position=min_position, report=report)
        folds += f
        coefficients += c
        qa[risk] = q
    return folds, coefficients, qa


def analysis_C(F, arrays, n_actions, horizons) -> Tuple[List[dict], List[dict], dict]:
    """Search evaluation at t -> human action at t+k beyond persistence, current root, and current search."""
    segment = arrays["segment"]
    human = arrays["human"]
    incumbent = arrays["incumbent"]
    log_prior = arrays["log_prior"]
    visited = arrays["visited"].astype(float)
    q_advantage = arrays["q_advantage"]
    abs_td = np.nan_to_num(F["abs_td"].to_numpy(dtype=float))
    cumulative_td = np.cumsum(abs_td)
    depth = F["search_depth"].to_numpy(dtype=float)
    plan_source = arrays["visited"].any(axis=1) & (human >= 0)
    best_branch = np.argmax(np.where(arrays["visited"], q_advantage, -np.inf), axis=1)
    actions = np.arange(n_actions)[None, :]
    folds, coefficients, qa = [], [], {}
    for k in horizons:
        target = lag_index(segment, -k)
        rows = np.flatnonzero(plan_source & (target >= 0))
        j = target[rows]
        keep = (human[j] >= 0) & (human[j] < n_actions) & (incumbent[j] >= 0)
        rows, j = rows[keep], j[keep]
        divergence = ((cumulative_td[j] - cumulative_td[rows]) / k)[:, None]
        blocks = {
            "controls": (np.stack([
                log_prior[j], (actions == incumbent[j][:, None]).astype(float),
                log_prior[rows], (actions == human[rows][:, None]).astype(float),
            ], axis=2), ["logp_target", "stay_target", "logp_plan_time", "chosen_at_plan_time"]),
            "current_search": (np.stack([visited[j], q_advantage[j]], axis=2),
                               ["visited_target", "Qadv_target"]),
            "plan": (np.stack([visited[rows], q_advantage[rows]], axis=2), ["plan_visited", "plan_Qadv"]),
            "plan_x_divergence": (np.stack([visited[rows] * divergence, q_advantage[rows] * divergence], axis=2),
                                  ["plan_visited_x_divergence", "plan_Qadv_x_divergence"]),
        }
        stages = {
            "C0_persistence_root": ["controls"],
            "C1_current_search": ["controls", "current_search"],
            "C2_previous_plan": ["controls", "current_search", "plan"],
            "C3_plan_x_divergence": ["controls", "current_search", "plan", "plan_x_divergence"],
        }
        increments = [("C0_persistence_root", "C1_current_search"),
                      ("C1_current_search", "C2_previous_plan"),
                      ("C2_previous_plan", "C3_plan_x_divergence")]
        label = f"horizon{k}"
        log(f"C plan reuse: k={k}, rows={len(rows)}")
        f, c = nested_clogit_loro(
            blocks, y=human[j], mask=np.ones((len(rows), n_actions), dtype=bool),
            runs=arrays["fold"][rows], analysis="C_plan_reuse", label=label, stages=stages,
            increments=increments,
            subsets={"shallow_plan": depth[rows] <= 1.0, "deep_plan": depth[rows] > 1.0,
                     "plan_proposes_switch": best_branch[rows] != incumbent[rows]},
            report={"C2_previous_plan": ["plan_visited", "plan_Qadv", "Qadv_target"],
                    "C3_plan_x_divergence": ["plan_Qadv", "plan_Qadv_x_divergence", "plan_visited_x_divergence"]},
            extra={"horizon": int(k)},
        )
        folds += f
        coefficients += c
        qa[label] = {"rows": int(len(rows)), "deep_plan_rows": int(np.sum(depth[rows] > 1.0))}
    return folds, coefficients, qa


def analysis_E(F, arrays, n_actions) -> Tuple[List[dict], List[dict], dict]:
    """Which action is chosen at a switch: pre-search prior -> search evidence -> post-search readout."""
    human = arrays["human"]
    incumbent = arrays["incumbent"]
    rows = np.flatnonzero((incumbent >= 0) & (human >= 0) & (human < n_actions) & (human != incumbent))
    actions = np.arange(n_actions)[None, :]
    mask = actions != incumbent[rows][:, None]
    blocks = {
        "prior": (np.stack([arrays["log_prior"][rows], np.broadcast_to(actions == 0, mask.shape).astype(float),
                            (actions == arrays["before"][rows][:, None]).astype(float)], axis=2),
                  ["logp_prior", "is_noop", "is_action_before_incumbent"]),
        "search": (np.stack([arrays["visited"][rows].astype(float), arrays["q_advantage"][rows]], axis=2),
                   ["visited", "Qadv"]),
        "readout": (arrays["log_actor_terminal"][rows][:, :, None], ["logp_actor_terminal"]),
    }
    stages = {"E0_prior": ["prior"], "E1_search": ["prior", "search"], "E2_readout": ["prior", "search", "readout"]}
    log(f"E destination: rows={len(rows)}")
    folds, coefficients = nested_clogit_loro(
        blocks, y=human[rows], mask=mask, runs=arrays["fold"][rows], analysis="E_destination",
        label="switch_destination", stages=stages,
        increments=[("E0_prior", "E1_search"), ("E1_search", "E2_readout")],
        subsets={"from_noop": incumbent[rows] == 0, "from_active": incumbent[rows] >= 1},
        report={"E1_search": ["logp_prior", "visited", "Qadv"]},
    )
    return folds, coefficients, {"switch_rows": int(len(rows))}


# ---------------------------------------------------------------------------
# Inference and summaries
# ---------------------------------------------------------------------------

def family_of(effect_id: str) -> str:
    return "::".join(effect_id.split("::")[:2])


def subject_increment_inference(folds: pd.DataFrame) -> pd.DataFrame:
    """Run-level synchronized sign flip within each analysis x risk-set family."""
    rows = []
    for family, part in folds.groupby(folds["effect_id"].map(family_of), sort=False):
        matrix = part.pivot_table(index="fold", columns="effect_id", values="beta", aggfunc="mean")
        ids = list(dict.fromkeys(part["effect_id"]))
        matrix = matrix.reindex(columns=ids)
        usable = [c for c in ids if matrix[c].notna().sum() >= 2]
        complete = matrix[usable].dropna(axis=0)
        if len(usable) and len(complete) >= 2 and len(complete) <= 20:
            inference = B04.synchronized_sign_flip(complete.to_numpy(dtype=float))
            inference.insert(0, "effect_id", usable)
            inference["family"] = family
            inference["not_estimable_reason"] = ""
            rows.append(inference)
        missing = [c for c in ids if c not in usable]
        if missing:
            rows.append(pd.DataFrame({"effect_id": missing, "family": family,
                                      "not_estimable_reason": "fewer than two estimable runs"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Mean over leave-one-run-out fits with the delete-one-run jackknife SE."""
    if coefficients.empty:
        return coefficients
    records = []
    for effect_id, part in coefficients.groupby("effect_id", sort=False):
        beta = part["beta"].to_numpy(dtype=float)
        n = len(beta)
        mean = float(np.mean(beta))
        se = float(np.sqrt((n - 1) / n * np.sum((beta - mean) ** 2))) if n > 1 else np.nan
        first = part.iloc[0]
        records.append({"effect_id": effect_id, "analysis": first["analysis"], "risk_set": first["risk_set"],
                        "stage": first["stage"], "feature": first["feature"], "estimate": mean,
                        "jackknife_se": se, "z": mean / se if se and se > EPS else np.nan, "n_fits": n})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False,
    })


def mean_ci(values: np.ndarray) -> Tuple[float, float, int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, 0
    if len(x) == 1:
        return float(x[0]), np.nan, 1
    half = float(stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / math.sqrt(len(x)))
    return float(np.mean(x)), half, len(x)


def _unit_values(frame: pd.DataFrame, unit: str) -> pd.DataFrame:
    return frame.groupby(["effect_id", unit], sort=False)["beta"].mean().reset_index()


def plot_profile(frame: pd.DataFrame, *, x_column: str, series_column: str, series: Sequence[Tuple[str, str]],
                 panel_column: str, panels: Sequence[str], unit: str, xlabel: str, title: str, path: Path) -> None:
    data = frame.loc[frame[panel_column].isin(panels)]
    if data.empty:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.2), sharey=False, squeeze=False)
    for axis, panel in zip(axes[0], panels):
        part = data.loc[data[panel_column] == panel]
        xs = sorted(part[x_column].dropna().unique())
        for color, (key, label) in zip(SERIES_COLORS, series):
            means, halves = [], []
            for x in xs:
                values = part.loc[(part[x_column] == x) & (part[series_column] == key)]
                values = values.groupby(unit)["beta"].mean().to_numpy() * 1e3
                mean, half, _ = mean_ci(values)
                means.append(mean)
                halves.append(half)
            positions = np.arange(len(xs))
            means = np.asarray(means)
            halves = np.nan_to_num(np.asarray(halves))
            axis.fill_between(positions, means - halves, means + halves, color=color, alpha=0.15, linewidth=0)
            axis.plot(positions, means, color=color, linewidth=2, marker="o", markersize=4, label=label)
        axis.axhline(0, color=MUTED, linewidth=1)
        axis.set_xticks(np.arange(len(xs)), [str(int(x)) for x in xs])
        axis.set_xlabel(xlabel)
        axis.set_title(panel.replace("_", " "), fontsize=10)
    axes[0][0].set_ylabel("Held-out log-loss reduction\n(nats × 10⁻³ per decision)")
    axes[0][-1].legend(loc="best", fontsize=8.5)
    fig.suptitle(title, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_forest(frame: pd.DataFrame, *, value_column: str, unit: Optional[str], labels: Sequence[str],
                xlabel: str, title: str, path: Path, scale: float = 1.0) -> None:
    ids = [i for i in labels if i in set(frame["effect_id"])]
    if not ids:
        return
    fig, axis = plt.subplots(figsize=(7.2, 0.28 * len(ids) + 1.2))
    for position, effect_id in enumerate(ids[::-1]):
        part = frame.loc[frame["effect_id"] == effect_id]
        values = (part.groupby(unit)[value_column].mean() if unit else part[value_column]).to_numpy(dtype=float) * scale
        mean, half, n = mean_ci(values)
        if not np.isfinite(mean):
            continue
        axis.scatter(values, np.full(len(values), position), s=10, color=MUTED, alpha=0.5, linewidths=0, zorder=2)
        axis.errorbar(mean, position, xerr=np.nan_to_num(half), fmt="o", color=BLUE, markersize=5,
                      elinewidth=1.6, capsize=0, zorder=3)
    axis.axvline(0, color=MUTED, linewidth=1)
    axis.set_yticks(np.arange(len(ids)), [i.split("::", 1)[1] for i in ids[::-1]], fontsize=7.5)
    axis.set_xlabel(xlabel)
    axis.set_title(title, fontsize=11, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_figures(folds: pd.DataFrame, coefficients: pd.DataFrame, figure_dir: Path, *, unit: str, suffix: str) -> None:
    configure_style()
    a = folds.loc[folds["analysis"] == "A_temporal_origin"].copy()
    if not a.empty:
        plot_profile(a, x_column="lag", series_column="block", series=(("stim", "Stimulus at t−k"), ("eval", "Evaluation at t−k")),
                     panel_column="risk_set", panels=[r for r in RISK_SETS if r in set(a["risk_set"])], unit=unit,
                     xlabel="Lag k (decisions)", title=f"A · Temporal origin beyond the current decision state {suffix}",
                     path=figure_dir / "fig_A_temporal_origin.png")
    c = folds.loc[folds["analysis"] == "C_plan_reuse"].copy()
    if not c.empty:
        c["panel"] = c["increment"].map({"C0_persistence_root_to_C1_current_search": "new search at t+k",
                                         "C1_current_search_to_C2_previous_plan": "previous plan at t",
                                         "C2_previous_plan_to_C3_plan_x_divergence": "plan × surprise"})
        plot_profile(c, x_column="horizon", series_column="subset",
                     series=(("all", "All plans"), ("deep_plan", "Deep plans"), ("shallow_plan", "Shallow plans")),
                     panel_column="panel", panels=["new search at t+k", "previous plan at t", "plan × surprise"],
                     unit=unit, xlabel="Horizon k (decisions after the plan)",
                     title=f"C · Does the evaluation at t explain the human action at t+k? {suffix}",
                     path=figure_dir / "fig_C_plan_reuse.png")
    for analysis in ("B_persistence_hierarchy", "D_plan_revision", "E_destination",
                     "F_bout_accumulation", "G_urgency"):
        part = folds.loc[folds["analysis"] == analysis]
        if part.empty:
            continue
        plot_forest(part, value_column="beta", unit=unit, labels=list(dict.fromkeys(part["effect_id"])),
                    xlabel="Held-out log-loss reduction (nats × 10⁻³ per decision)", scale=1e3,
                    title=f"{analysis.replace('_', ' ')} · nested held-out increments {suffix}",
                    path=figure_dir / f"fig_increments_{analysis.split('_', 1)[0]}.png")
    if not coefficients.empty:
        selected = coefficients.loc[
            coefficients["feature"].str.contains("_x_|prev1_|_lag1|plan_Qadv|abs_td|inc_revision|alt_revision|Qadv$", regex=True)
        ]
        plot_forest(selected, value_column="beta", unit=unit, labels=list(dict.fromkeys(selected["effect_id"])),
                    xlabel="Standardized coefficient", title=f"Selected coefficients {suffix}",
                    path=figure_dir / "fig_coefficients.png")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def fold_suffix(args: argparse.Namespace) -> str:
    return "" if args.fold_unit == "run" else "_session_folds"


def _prepare_output(directory: Path, overwrite: bool) -> None:
    if (directory / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"Completed analysis exists: {directory}; pass --overwrite to replace it")
    directory.mkdir(parents=True, exist_ok=True)


def run_subject(args: argparse.Namespace) -> Path:
    directory = Path(args.output_root) / f"sub{args.subject:03d}_game{args.game}{fold_suffix(args)}"
    _prepare_output(directory, args.overwrite)
    windows, events, visual, meta = load_subject_inputs(Path(args.shared_cache_root), args.subject, args.game)
    n_actions = int(meta["n_actions"])
    F, arrays = build_features(windows, events, visual, n_actions)
    groups = feature_groups(F, n_actions)
    dropped = sorted({c for cols in groups.values() for c in cols if not np.isfinite(F[c].to_numpy(dtype=float)).any()})
    groups = {name: [c for c in cols if c not in dropped] for name, cols in groups.items()}
    if args.fold_unit == "run":
        # Each session is replayed by its own leave-session-out checkpoint, and the searched
        # branch is nearly fixed per checkpoint: absorb checkpoint identity in the controls.
        groups["history"] = groups["history"] + groups["checkpoint"]
        arrays["fold"] = arrays["run_label"]
    else:
        arrays["fold"] = np.array([f"session{value:02d}" for value in arrays["session"]])
    if dropped:
        log(f"dropping features with no finite value in this subject/game: {dropped}")
    log(f"features: rows={len(F)}, columns={F.shape[1]}, actions={n_actions}, "
        f"segments={int(arrays['segment'].max()) + 1}, runs={len(pd.unique(arrays['run_label']))}")
    folds, coefficients, qa = [], [], {}
    selected = set(args.analyses)
    runners = {
        "A": lambda: analysis_A(F, arrays, groups, n_actions, args.lags),
        "B": lambda: analysis_B(F, arrays, groups, n_actions),
        "C": lambda: analysis_C(F, arrays, n_actions, args.horizons),
        "D": lambda: analysis_D(F, arrays, groups, n_actions),
        "E": lambda: analysis_E(F, arrays, n_actions),
        "F": lambda: analysis_F(F, arrays, groups, n_actions),
        "G": lambda: analysis_G(F, arrays, groups, n_actions),
    }
    for key in ANALYSES:
        if key not in selected:
            continue
        f, c, q = runners[key]()
        folds += f
        coefficients += c
        qa[key] = q
    folds_frame = pd.DataFrame(folds)
    coefficient_frame = pd.DataFrame(coefficients)
    for frame in (folds_frame, coefficient_frame):
        if not frame.empty:
            frame.insert(0, "game", args.game)
            frame.insert(0, "subject", args.subject)
    tables = directory / "tables"
    write_table(folds_frame, tables / "increment_folds.csv")
    write_table(subject_increment_inference(folds_frame) if not folds_frame.empty else pd.DataFrame(),
                tables / "increment_subject_inference.csv")
    write_table(coefficient_frame, tables / "coefficient_folds.csv")
    write_table(summarize_coefficients(coefficient_frame), tables / "coefficient_summary.csv")
    render_figures(folds_frame, coefficient_frame, directory / "figures", unit="fold",
                   suffix=f"· sub{args.subject:03d}, game {args.game} ({args.fold_unit} folds)")
    atomic_json(directory / "qa" / "qa.json", qa)
    atomic_json(directory / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "analysis_version": ANALYSIS_VERSION,
        "subject": args.subject, "game": args.game, "analyses": sorted(selected),
        "fold_unit": args.fold_unit, "lags": list(args.lags), "horizons": list(args.horizons), "history_lags": list(HISTORY_LAGS),
        "gamma": GAMMA, "ewma_halflife": EWMA_HALFLIFE, "min_train_rows": MIN_TRAIN_ROWS,
        "script_sha256": sha256_path(Path(__file__)), "base04_sha256": sha256_path(BASE04_PATH),
        "inputs": meta["inputs"], "n_actions": n_actions, "rows": int(len(F)),
        "dropped_all_missing_features": dropped, "carry_limit": CARRY_LIMIT,
        "cache_only": True,
        "sign_convention": "increment beta = start log loss - end log loss; positive = added block helps",
        "notes": [
            "Unvisited root-Q placeholders (0) are masked; Qadv = root_q_final - root_value for visited actions.",
            "Plan reuse is value-level: depth>=2 imagined actions are not cached, so no action-sequence matching.",
            "Search-derived NaNs are train-fold median-imputed with explicit availability indicators (as in 04 Stage 3).",
            "Interaction components are z-scored on full-data features only; the outcome never enters.",
            "Tree is not carried across decisions in these caches (root_n_initial ~ 0), so reuse is behavioural/value-level.",
            "Search visits ~one root branch per decision and that branch is nearly fixed per session checkpoint; "
            "run folds therefore control session identity, session folds hold out whole checkpoints.",
            "carried_* features hold each action's latest evaluation for up to carry_limit decisions (excluding the current one).",
        ],
    })
    log(f"completed {directory}")
    return directory


def run_group(args: argparse.Namespace) -> Path:
    directory = Path(args.output_root) / f"group_game{args.game}{fold_suffix(args)}"
    _prepare_output(directory, args.overwrite)
    folds, coefficients, sources = [], [], {}
    for subject in args.aggregate_subjects:
        subject_dir = Path(args.output_root) / f"sub{subject:03d}_game{args.game}{fold_suffix(args)}"
        manifest_path = subject_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing completed subject analysis: {subject_dir}")
        sources[str(subject)] = {"manifest_sha256": sha256_path(manifest_path)}
        folds.append(pd.read_csv(subject_dir / "tables" / "increment_folds.csv"))
        coefficients.append(pd.read_csv(subject_dir / "tables" / "coefficient_folds.csv"))
    folds_frame = pd.concat(folds, ignore_index=True)
    coefficient_frame = pd.concat(coefficients, ignore_index=True)
    increment_inference = []
    for family, part in folds_frame.groupby(folds_frame["effect_id"].map(family_of), sort=False):
        ids = list(dict.fromkeys(part["effect_id"]))
        inference, _ = B04.exact_unit_inference(part, ids, "subject")
        inference["family"] = family
        increment_inference.append(inference)
    coefficient_units = coefficient_frame.groupby(["subject", "effect_id"], sort=False)["beta"].mean().reset_index()
    coefficient_inference = []
    for family, part in coefficient_units.groupby(coefficient_units["effect_id"].map(lambda e: "::".join(e.split("::")[:3])), sort=False):
        ids = list(dict.fromkeys(part["effect_id"]))
        inference, _ = B04.exact_unit_inference(part, ids, "subject")
        inference["family"] = family
        coefficient_inference.append(inference)
    tables = directory / "tables"
    write_table(_unit_values(folds_frame, "subject"), tables / "increment_subject_means.csv")
    write_table(pd.concat(increment_inference, ignore_index=True), tables / "increment_group_inference.csv")
    write_table(coefficient_units, tables / "coefficient_subject_means.csv")
    write_table(pd.concat(coefficient_inference, ignore_index=True), tables / "coefficient_group_inference.csv")
    render_figures(folds_frame, coefficient_frame, directory / "figures", unit="subject",
                   suffix=f"· game {args.game} (subjects, {args.fold_unit} folds)")
    atomic_json(directory / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "analysis_version": ANALYSIS_VERSION, "game": args.game,
        "subjects": list(args.aggregate_subjects), "sources": sources,
        "inference": "exact synchronized sign flip across subjects within analysis x risk-set families",
        "script_sha256": sha256_path(Path(__file__)),
    })
    log(f"completed {directory}")
    return directory


def run_self_test() -> Dict[str, object]:
    rng = np.random.default_rng(12)
    segment = np.repeat([0, 1, 2], [5, 3, 4])
    back = lag_index(segment, 2)
    ahead = lag_index(segment, -2)
    assert back.tolist() == [-1, -1, 0, 1, 2, -1, -1, 5, -1, -1, 8, 9]
    assert ahead.tolist() == [2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1]
    n, n_alt, p = 3000, 5, 3
    x = rng.normal(size=(n, n_alt, p))
    truth = np.array([1.0, -0.5, 0.0])
    utility = x @ truth
    mask = np.ones((n, n_alt), dtype=bool)
    mask[:, 0] = rng.random(n) < 0.5
    utility = np.where(mask, utility, -np.inf)
    probability = np.exp(utility - logsumexp(utility, axis=1, keepdims=True))
    y = np.array([rng.choice(n_alt, p=row) for row in probability])
    fit = fit_conditional_logit(x, y, mask)
    assert fit and fit["converged"]
    recovered = fit["beta"] / fit["scale"]
    assert np.allclose(recovered, truth, atol=0.12), recovered
    assert np.all(np.isfinite(conditional_logit_log_loss(fit, x, y, mask)))
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    bout = np.array([1, 1, 1, 2, 2])
    cumulative = series.groupby(bout).cumsum().to_numpy() - series.to_numpy()
    assert cumulative.tolist() == [0.0, 1.0, 3.0, 0.0, 4.0]
    return {"lag_index": "ok", "conditional_logit_recovery": recovered.round(3).tolist(),
            "bout_cumulative_excludes_current": "ok"}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", type=int)
    parser.add_argument("--game", type=int)
    parser.add_argument("--aggregate-subjects", type=int, nargs="+")
    parser.add_argument("--shared-cache-root", type=Path, default=DEFAULT_SHARED_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--analyses", type=lambda s: [x.strip().upper() for x in s.split(",") if x.strip()],
                        default=list(ANALYSES), help="Comma list from A,B,C,D,E,F,G")
    parser.add_argument("--lags", type=lambda s: [int(x) for x in s.split(",")], default=list(DEFAULT_LAGS))
    parser.add_argument("--horizons", type=lambda s: [int(x) for x in s.split(",")], default=list(DEFAULT_HORIZONS))
    parser.add_argument("--fold-unit", choices=("run", "session"), default="run",
                        help="run: LORO with checkpoint(session) controls; session: leave-one-checkpoint-out")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.self_test:
        print(json.dumps(run_self_test(), indent=2))
        return
    if args.game is None:
        raise SystemExit("--game is required")
    unknown = sorted(set(args.analyses) - set(ANALYSES))
    if unknown:
        raise SystemExit(f"Unknown analyses: {unknown}")
    if args.aggregate_subjects:
        run_group(args)
    elif args.subject is not None:
        run_subject(args)
    else:
        raise SystemExit("Pass --subject or --aggregate-subjects")


if __name__ == "__main__":
    main()
