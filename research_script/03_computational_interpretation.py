#!/usr/bin/env python3
"""
03_computational_interpretation.py

Section 3: Computational Interpretation.

Default input follows Section 2:
  test/sub001/ses-04/*.npy

Default output:
  research_script/outputs/03_computational_interpretation/

The script builds a real-step table from filtered thinker traces and tests
whether computational uncertainty variables predict NOOP withholding.

Main outputs
------------
results/
  real_step_computational_metrics.csv
  uncertainty_noop_bins.csv
  logistic_model_comparison.csv
  logistic_coefficients.csv
  cv_auc_scores.csv
  species_interactions.csv
  ar_residual_uncertainty.csv
  lagged_temporal_precedence.csv
  onset_trigger_models.csv
  commitment_trigger_models.csv
  commitment_trigger_coefficients.csv
  bout_length_regression.csv
  summary.txt

figures/
  fig_3_1_uncertainty_coupling.png
  fig_3_2_model_comparison.png
  fig_3_3_temporal_ordering.png

Notes
-----
The default trace set currently contains one subject and one game. In that
case, true subject/game random intercepts are not identifiable, so AIC/BIC
model comparison uses logistic GLMs with available file/subject/game fixed
intercepts. If more subjects/games are supplied, the same code automatically
adds those intercepts. An optional variational Bayes mixed logistic fit is
available via --run-mixed-logit for full-model robustness checks.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import font_manager
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "test" / "sub001" / "ses-04"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs" / "03_computational_interpretation"

EPS = 1e-12
NOOP_ACTION = 0
FRAGMENT_STATUSES = (1, 2)

GAME_LABELS = {
    1: "pong",
    2: "spaceinvaders",
}
GAME_TITLES = {
    1: "Pong",
    2: "Space Invaders",
}
TARGET_SPECS = {
    "human": "is_human_noop",
    "thinker": "is_thinker_noop",
}

CORE_FEATURES = [
    "entropy_actor",
    "neg_q_gap",
    "search_jsd_actor_root",
    "rollout_spread",
    "neg_actor_policy_gap",
    "branch_entropy",
    "tree_width",
]

TEMPORAL_FEATURES = [
    "entropy_actor",
    "neg_q_gap",
    "search_jsd_actor_root",
    "rollout_spread",
]

FEATURE_LABELS = {
    "entropy_actor": "Actor entropy",
    "entropy_root_policy": "Root policy entropy",
    "entropy_cur_policy": "Current policy entropy",
    "neg_q_gap": "Low Q gap",
    "q_gap": "Q gap",
    "rollout_spread": "Rollout spread",
    "search_jsd_actor_root": "Actor-root JSD",
    "search_jsd_actor_cur": "Actor-current JSD",
    "search_jsd_root_cur": "Root-current JSD",
    "neg_actor_policy_gap": "Low actor margin",
    "branch_entropy": "Branch entropy",
    "tree_width": "Tree width",
    "root_value": "Root value",
    "prev_reward_5": "Previous 5-step reward",
    "bout_age": "Bout age",
}

MODEL_SPECS = {
    "null": [],
    "actor_entropy": ["entropy_actor"],
    "q_conflict": ["neg_q_gap"],
    "search_disagreement": ["search_jsd_actor_root"],
    "rollout_spread": ["rollout_spread"],
    "actor_margin": ["neg_actor_policy_gap"],
    "branch_structure": ["branch_entropy", "tree_width"],
    "full_uncertainty": CORE_FEATURES,
}

PALETTE = {
    "human": "#1f77b4",
    "thinker": "#ff7f0e",
    "neutral": "#4b5563",
    "accent": "#d62728",
    "light": "#9ca3af",
}


# Font setup mirrors the other analysis scripts while remaining optional.
_FONT_CANDIDATES = ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]
_FONT_PATH_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
]
for _font_path in _FONT_PATH_CANDIDATES:
    if _font_path.exists():
        font_manager.fontManager.addfont(str(_font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_font_path)).get_name()
        break
else:
    _available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next(
        (font for font in _FONT_CANDIDATES if font in _available_fonts),
        "DejaVu Sans",
    )
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class FileMeta:
    subject: int
    session: int
    block: int
    game: int
    chunk: int
    path: Path


def parse_file_meta(path: Path) -> FileMeta:
    match = re.match(r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)_(\d+)\.npy$", path.name)
    if match is None:
        raise ValueError(f"Unexpected file name format: {path.name}")
    return FileMeta(
        subject=int(match.group(1)),
        session=int(match.group(2)),
        block=int(match.group(3)),
        game=int(match.group(4)),
        chunk=int(match.group(5)),
        path=path,
    )


def load_npy_dict(path: Path) -> Dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        item = obj.item()
        if isinstance(item, dict):
            return item
    if hasattr(obj, "files"):
        return {key: obj[key] for key in obj.files}
    raise ValueError(f"Cannot parse file as dict-like npy/npz: {path}")


def to_action_ids(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    return arr.astype(int).reshape(-1) if arr.ndim == 1 else np.argmax(arr, axis=1).astype(int)


def ensure_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.reshape(arr.shape[0], -1)


def softmax_rows(x: np.ndarray) -> np.ndarray:
    arr = ensure_2d(x)
    arr = arr - np.nanmax(arr, axis=1, keepdims=True)
    ex = np.exp(np.clip(arr, -60.0, 60.0))
    denom = np.nansum(ex, axis=1, keepdims=True)
    return ex / np.where(denom <= EPS, 1.0, denom)


def normalize_policy_rows(x: np.ndarray) -> np.ndarray:
    """
    Convert nonnegative policy-like rows to probabilities.
    Falls back to softmax for rows that cannot be normalized directly.
    """
    arr = ensure_2d(x)
    clipped = np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    sums = clipped.sum(axis=1, keepdims=True)
    direct = clipped / np.where(sums <= EPS, 1.0, sums)
    fallback = softmax_rows(arr)
    return np.where(sums > EPS, direct, fallback)


def entropy_rows(p: np.ndarray) -> np.ndarray:
    prob = normalize_policy_rows(p)
    return -np.sum(prob * np.log(prob + EPS), axis=1)


def top2_gap_rows(x: np.ndarray) -> np.ndarray:
    arr = ensure_2d(x)
    if arr.shape[1] < 2:
        return np.zeros(arr.shape[0], dtype=float)
    part = np.partition(arr, -2, axis=1)[:, -2:]
    return part[:, 1] - part[:, 0]


def chosen_values(x: np.ndarray, actions: np.ndarray) -> np.ndarray:
    arr = ensure_2d(x)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    n = min(len(arr), len(actions))
    out = np.full(n, np.nan, dtype=float)
    rows = np.arange(n)
    valid = (actions[:n] >= 0) & (actions[:n] < arr.shape[1])
    out[valid] = arr[rows[valid], actions[:n][valid]]
    return out


def js_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pp = normalize_policy_rows(p)
    qq = normalize_policy_rows(q)
    n = min(len(pp), len(qq))
    pp = pp[:n]
    qq = qq[:n]
    m = 0.5 * (pp + qq)
    kl_pm = np.sum(pp * (np.log(pp + EPS) - np.log(m + EPS)), axis=1)
    kl_qm = np.sum(qq * (np.log(qq + EPS) - np.log(m + EPS)), axis=1)
    return 0.5 * (kl_pm + kl_qm)


def prev_index_of_status(status_indices: np.ndarray, ref_indices: np.ndarray) -> np.ndarray:
    if len(status_indices) == 0:
        return np.full(len(ref_indices), -1, dtype=int)
    pos = np.searchsorted(status_indices, ref_indices) - 1
    return np.where(pos >= 0, status_indices[pos], -1).astype(int)


def gather_input_files(input_dir: Path, game_id: int | None = None) -> List[FileMeta]:
    metas: List[FileMeta] = []
    for path in sorted(input_dir.rglob("*.npy")):
        try:
            meta = parse_file_meta(path)
        except ValueError:
            continue
        if game_id is not None and meta.game != game_id:
            continue
        metas.append(meta)
    return metas


def build_real_step_table(data: Dict[str, np.ndarray], meta: FileMeta) -> pd.DataFrame:
    tree = data["tree_reps"]
    status = np.asarray(data["status"]).reshape(-1)
    human_action = to_action_ids(np.asarray(data["human_action"]))
    thinker_action = to_action_ids(np.asarray(data["thinker_action"]))
    actor_logits = np.asarray(data["actor_policy"]).reshape(len(status), -1)
    env_return = np.asarray(data["env_return"], dtype=float).reshape(-1)

    cur_qs = ensure_2d(np.asarray(tree["cur_qs_mean"], dtype=float))
    root_qs = ensure_2d(np.asarray(tree["root_qs_mean"], dtype=float))
    root_qs_max = ensure_2d(np.asarray(tree.get("root_qs_max", tree["root_qs_mean"]), dtype=float))
    cur_v = np.asarray(tree["cur_v"], dtype=float).reshape(-1)
    root_v = np.asarray(tree["root_v"], dtype=float).reshape(-1)
    root_policy_raw = ensure_2d(np.asarray(tree["root_policy"], dtype=float))
    cur_policy_raw = ensure_2d(np.asarray(tree["cur_policy"], dtype=float))
    root_ns = ensure_2d(np.asarray(tree.get("root_ns", np.zeros_like(root_policy_raw)), dtype=float))
    rollout_return = np.asarray(tree["rollout_return"], dtype=float).reshape(-1)
    max_rollout_return = np.asarray(tree["max_rollout_return"], dtype=float).reshape(-1)

    t = min(
        len(status),
        len(human_action),
        len(thinker_action),
        len(actor_logits),
        len(env_return),
        len(cur_qs),
        len(root_qs),
        len(root_qs_max),
        len(cur_v),
        len(root_v),
        len(root_policy_raw),
        len(cur_policy_raw),
        len(root_ns),
        len(rollout_return),
        len(max_rollout_return),
    )
    if t == 0:
        return pd.DataFrame()

    status = status[:t]
    human_action = human_action[:t]
    thinker_action = thinker_action[:t]
    actor_logits = actor_logits[:t]
    env_return = env_return[:t]
    cur_qs = cur_qs[:t]
    root_qs = root_qs[:t]
    root_qs_max = root_qs_max[:t]
    cur_v = cur_v[:t]
    root_v = root_v[:t]
    root_policy_raw = root_policy_raw[:t]
    cur_policy_raw = cur_policy_raw[:t]
    root_ns = root_ns[:t]
    rollout_return = rollout_return[:t]
    max_rollout_return = max_rollout_return[:t]

    actor_probs = softmax_rows(actor_logits)
    root_policy = normalize_policy_rows(root_policy_raw)
    cur_policy = normalize_policy_rows(cur_policy_raw)
    root_ns_prob = normalize_policy_rows(root_ns)

    entropy_actor = entropy_rows(actor_probs)
    entropy_root_policy = entropy_rows(root_policy)
    entropy_cur_policy = entropy_rows(cur_policy)
    actor_policy_gap = top2_gap_rows(actor_probs)
    root_policy_gap = top2_gap_rows(root_policy)
    cur_policy_gap = top2_gap_rows(cur_policy)
    q_gap = top2_gap_rows(root_qs)
    cur_q_gap = top2_gap_rows(cur_qs)
    q_gap_max = top2_gap_rows(root_qs_max)
    rollout_spread = np.abs(max_rollout_return - rollout_return)
    search_jsd_actor_root = js_divergence_rows(actor_probs, root_policy)
    search_jsd_actor_cur = js_divergence_rows(actor_probs, cur_policy)
    search_jsd_root_cur = js_divergence_rows(root_policy, cur_policy)
    branch_entropy = entropy_rows(root_ns_prob)
    tree_width = np.sum(root_ns > EPS, axis=1).astype(float)
    actor_prob_human_all = chosen_values(actor_probs, human_action)
    actor_prob_thinker_all = chosen_values(actor_probs, thinker_action)
    root_prob_human_all = chosen_values(root_policy, human_action)
    root_prob_thinker_all = chosen_values(root_policy, thinker_action)
    root_q_human_all = chosen_values(root_qs, human_action)
    root_q_thinker_all = chosen_values(root_qs, thinker_action)

    real_idx = np.flatnonzero(status == 0)
    if real_idx.size == 0:
        return pd.DataFrame()

    status2_idx = np.flatnonzero(status == 2)
    prev_s2_all = prev_index_of_status(status2_idx, real_idx)
    prev_s2_all = np.where(prev_s2_all >= 0, prev_s2_all, np.maximum(real_idx - 1, 0)).astype(int)

    rows: List[Dict[str, object]] = []
    episode_in_file = 0
    episode_step = 0

    for real_pos, idx_global in enumerate(real_idx):
        prev_real_global = int(real_idx[real_pos - 1]) if real_pos > 0 else -1
        between = np.arange(prev_real_global + 1, int(idx_global), dtype=int)
        if real_pos > 0 and between.size > 0 and np.isin(status[between], [1, 3]).any():
            episode_in_file += 1
            episode_step = 0

        prev_s2 = int(prev_s2_all[real_pos])
        if prev_s2 < 0 or prev_s2 >= t:
            prev_s2 = max(0, min(int(idx_global) - 1, t - 1))

        status_counts = {
            f"n_status_{code}_since_prev_real": int(np.sum(status[between] == code))
            if between.size > 0
            else 0
            for code in [1, 2, 3]
        }

        h_action = int(human_action[idx_global])
        th_action = int(thinker_action[idx_global])
        row = {
            "file": str(meta.path),
            "source_file_name": meta.path.name,
            "subject": meta.subject,
            "session": meta.session,
            "block": meta.block,
            "game": meta.game,
            "game_name": GAME_TITLES.get(meta.game, f"Game {meta.game}"),
            "chunk": meta.chunk,
            "episode_in_file": episode_in_file,
            "episode_step": episode_step,
            "real_pos": real_pos,
            "global_idx": int(idx_global),
            "prev_s2_idx": prev_s2,
            "human_action": h_action,
            "thinker_action": th_action,
            "is_human_noop": int(h_action == NOOP_ACTION),
            "is_thinker_noop": int(th_action == NOOP_ACTION),
            "env_return": float(env_return[idx_global]),
            "entropy_actor": float(entropy_actor[idx_global]),
            "entropy_root_policy": float(entropy_root_policy[idx_global]),
            "entropy_cur_policy": float(entropy_cur_policy[idx_global]),
            "actor_policy_gap": float(actor_policy_gap[idx_global]),
            "neg_actor_policy_gap": float(-actor_policy_gap[idx_global]),
            "root_policy_gap": float(root_policy_gap[idx_global]),
            "cur_policy_gap": float(cur_policy_gap[idx_global]),
            "q_gap": float(q_gap[idx_global]),
            "neg_q_gap": float(-q_gap[idx_global]),
            "cur_q_gap": float(cur_q_gap[idx_global]),
            "q_gap_max": float(q_gap_max[idx_global]),
            "rollout_spread": float(rollout_spread[idx_global]),
            "search_jsd_actor_root": float(search_jsd_actor_root[idx_global]),
            "search_jsd_actor_cur": float(search_jsd_actor_cur[idx_global]),
            "search_jsd_root_cur": float(search_jsd_root_cur[idx_global]),
            "branch_entropy": float(branch_entropy[idx_global]),
            "tree_width": float(tree_width[idx_global]),
            "root_value": float(root_v[idx_global]),
            "cur_value": float(cur_v[idx_global]),
            "root_cur_value_delta": float(root_v[idx_global] - cur_v[idx_global]),
            "actor_prob_human_action": float(actor_prob_human_all[idx_global]),
            "actor_prob_thinker_action": float(actor_prob_thinker_all[idx_global]),
            "root_prob_human_action": float(root_prob_human_all[idx_global]),
            "root_prob_thinker_action": float(root_prob_thinker_all[idx_global]),
            "root_q_human_action": float(root_q_human_all[idx_global]),
            "root_q_thinker_action": float(root_q_thinker_all[idx_global]),
            "entropy_actor_prev_s2": float(entropy_actor[prev_s2]),
            "q_gap_prev_s2": float(q_gap[prev_s2]),
            "neg_q_gap_prev_s2": float(-q_gap[prev_s2]),
            "rollout_spread_prev_s2": float(rollout_spread[prev_s2]),
            "search_jsd_actor_root_prev_s2": float(search_jsd_actor_root[prev_s2]),
            "branch_entropy_prev_s2": float(branch_entropy[prev_s2]),
            "tree_width_prev_s2": float(tree_width[prev_s2]),
        }
        row.update(status_counts)
        rows.append(row)
        episode_step += 1

    df = pd.DataFrame(rows).sort_values(["file", "real_pos"]).reset_index(drop=True)
    return add_transition_features(df)


def add_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["file", "real_pos"]).reset_index(drop=True).copy()
    group = out.groupby("file", sort=False)

    out["prev_env_return"] = group["env_return"].shift(1)
    out["prev_reward_5"] = group["env_return"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    out["prev_reward_10"] = group["env_return"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).sum()
    )

    for target_name, target_col in TARGET_SPECS.items():
        out[f"{target_name}_noop_lag1"] = group[target_col].shift(1)
        out[f"{target_name}_noop_lag2"] = group[target_col].shift(2)
        out[f"{target_name}_noop_lag3"] = group[target_col].shift(3)
        out[f"{target_name}_noop_lag4"] = group[target_col].shift(4)
        out[f"{target_name}_noop_lag5"] = group[target_col].shift(5)
        out[f"{target_name}_noop_onset"] = (
            (out[target_col] == 1) & (out[f"{target_name}_noop_lag1"] == 0)
        ).astype(int)
        out[f"{target_name}_commit"] = (
            (out[target_col] == 0) & (out[f"{target_name}_noop_lag1"] == 1)
        ).astype(int)

    return out


def load_real_step_dataset(input_dir: Path, game_id: int | None = None) -> pd.DataFrame:
    metas = gather_input_files(input_dir, game_id=game_id)
    if not metas:
        raise FileNotFoundError(f"No matching .npy files found under {input_dir}")

    frames = []
    for meta in metas:
        print(f"[load] {meta.path}")
        frame = build_real_step_table(load_npy_dict(meta.path), meta)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No real-step rows produced from input files.")

    df = pd.concat(frames, ignore_index=True).sort_values(["file", "real_pos"]).reset_index(drop=True)
    return add_lagged_metric_columns(df, TEMPORAL_FEATURES, max_lag=5)


def add_lagged_metric_columns(
    df: pd.DataFrame,
    features: Sequence[str],
    max_lag: int,
) -> pd.DataFrame:
    out = df.sort_values(["file", "real_pos"]).reset_index(drop=True).copy()
    group = out.groupby("file", sort=False)
    for feat in features:
        if feat not in out.columns:
            continue
        for lag in range(1, max_lag + 1):
            out[f"{feat}_lag{lag}"] = group[feat].shift(lag)
    return out


def finite_model_frame(
    df: pd.DataFrame,
    target_col: str,
    features: Sequence[str],
    extra_cols: Sequence[str] = (),
    max_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    cols = [target_col, "file", "subject", "game", *features, *extra_cols]
    cols = [col for col in cols if col in df.columns]
    sub = df[cols].dropna().copy()
    numeric_cols = [target_col, *features, *[c for c in extra_cols if c in sub.columns]]
    numeric_cols = [col for col in numeric_cols if col in sub.columns]
    if numeric_cols:
        sub = sub[np.isfinite(sub[numeric_cols].to_numpy(dtype=float)).all(axis=1)].copy()
    sub[target_col] = sub[target_col].astype(int)
    if max_rows is not None and len(sub) > max_rows:
        sub = sub.sample(max_rows, random_state=random_state).sort_index().copy()
    return sub.reset_index(drop=True)


def add_standardized_columns(df: pd.DataFrame, features: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    out = df.copy()
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for feat in features:
        mu = float(out[feat].mean())
        sd = float(out[feat].std(ddof=0))
        if not np.isfinite(sd) or sd < EPS:
            sd = 1.0
        out[f"z_{feat}"] = (out[feat] - mu) / sd
        means[feat] = mu
        stds[feat] = sd
    return out, means, stds


def design_matrix(
    df: pd.DataFrame,
    features: Sequence[str],
    include_fixed_effects: bool,
    interaction_pairs: Sequence[Tuple[str, str]] = (),
) -> pd.DataFrame:
    x = pd.DataFrame({"Intercept": np.ones(len(df), dtype=float)}, index=df.index)
    for feat in features:
        z_col = f"z_{feat}" if f"z_{feat}" in df.columns else feat
        if z_col in df.columns:
            x[z_col] = df[z_col].astype(float)

    for a, b in interaction_pairs:
        a_col = f"z_{a}" if f"z_{a}" in df.columns else a
        b_col = f"z_{b}" if f"z_{b}" in df.columns else b
        if a_col in df.columns and b_col in df.columns:
            x[f"{a_col}:{b_col}"] = df[a_col].astype(float) * df[b_col].astype(float)

    if include_fixed_effects:
        for col in ["subject", "game"]:
            if col in df.columns and df[col].nunique(dropna=True) > 1:
                dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=float)
                x = pd.concat([x, dummies], axis=1)
        if "file" in df.columns and df["file"].nunique(dropna=True) > 1:
            dummies = pd.get_dummies(df["file"].astype(str), prefix="file", drop_first=True, dtype=float)
            x = pd.concat([x, dummies], axis=1)
    return x.astype(float)


def fit_glm_binomial(
    df: pd.DataFrame,
    target_col: str,
    features: Sequence[str],
    include_fixed_effects: bool = True,
    interaction_pairs: Sequence[Tuple[str, str]] = (),
) -> Tuple[object | None, pd.DataFrame, str | None]:
    if df.empty or df[target_col].nunique() < 2:
        return None, pd.DataFrame(), "target has fewer than two classes"

    x = design_matrix(
        df=df,
        features=features,
        include_fixed_effects=include_fixed_effects,
        interaction_pairs=interaction_pairs,
    )
    y = df[target_col].astype(int).to_numpy()

    try:
        model = sm.GLM(y, x, family=sm.families.Binomial())
        if "file" in df.columns and df["file"].nunique() >= 2:
            result = model.fit(
                maxiter=200,
                disp=0,
                cov_type="cluster",
                cov_kwds={"groups": df["file"].astype(str).to_numpy()},
            )
        else:
            result = model.fit(maxiter=200, disp=0)
    except Exception as exc:
        return None, x, str(exc)
    return result, x, None


def stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def run_logistic_model_comparison(
    df: pd.DataFrame,
    target_names: Sequence[str],
    max_rows: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []

    for target_name in target_names:
        target_col = TARGET_SPECS[target_name]
        null_llf = np.nan

        for model_name, features in MODEL_SPECS.items():
            sub = finite_model_frame(
                df,
                target_col=target_col,
                features=features,
                max_rows=max_rows,
                random_state=random_state,
            )
            sub, means, stds = add_standardized_columns(sub, features)
            result, x, error = fit_glm_binomial(sub, target_col, features)

            row = {
                "target": target_name,
                "model": model_name,
                "features": ",".join(features),
                "n": len(sub),
                "events": int(sub[target_col].sum()) if len(sub) else 0,
                "event_rate": float(sub[target_col].mean()) if len(sub) else np.nan,
                "n_predictors": len(features),
                "aic": np.nan,
                "bic": np.nan,
                "llf": np.nan,
                "mcfadden_r2": np.nan,
                "error": error or "",
            }

            if result is not None:
                if model_name == "null":
                    null_llf = float(result.llf)
                llf = float(result.llf)
                row.update(
                    {
                        "aic": float(result.aic),
                        "bic": float(getattr(result, "bic_llf", result.bic)),
                        "llf": llf,
                        "mcfadden_r2": float(1.0 - llf / null_llf)
                        if np.isfinite(null_llf) and abs(null_llf) > EPS
                        else np.nan,
                    }
                )

                for term in result.params.index:
                    if term == "Intercept" or term.startswith(("file_", "subject_", "game_")):
                        continue
                    raw_feature = term[2:] if term.startswith("z_") else term
                    coef_rows.append(
                        {
                            "target": target_name,
                            "model": model_name,
                            "term": term,
                            "feature": raw_feature,
                            "label": FEATURE_LABELS.get(raw_feature, raw_feature),
                            "coef": float(result.params[term]),
                            "se": float(result.bse[term]) if term in result.bse.index else np.nan,
                            "z": float(result.tvalues[term]) if term in result.tvalues.index else np.nan,
                            "p": float(result.pvalues[term]) if term in result.pvalues.index else np.nan,
                            "sig": stars(float(result.pvalues[term])) if term in result.pvalues.index else "n/a",
                            "feature_mean": means.get(raw_feature, np.nan),
                            "feature_sd": stds.get(raw_feature, np.nan),
                            "n": len(sub),
                            "event_rate": row["event_rate"],
                        }
                    )
            model_rows.append(row)

    model_df = pd.DataFrame(model_rows)
    if not model_df.empty:
        model_df["delta_aic"] = model_df.groupby("target")["aic"].transform(lambda s: s - s.min())
        model_df["delta_bic"] = model_df.groupby("target")["bic"].transform(lambda s: s - s.min())
    return model_df, pd.DataFrame(coef_rows)


def build_splits(groups: np.ndarray, n_samples: int, max_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if unique_groups.size >= 2:
        splitter = GroupKFold(n_splits=min(max_splits, unique_groups.size))
        dummy_x = np.zeros((n_samples, 1))
        return list(splitter.split(dummy_x, groups=groups))
    n_splits = min(max_splits, max(2, min(5, n_samples)))
    splitter = KFold(n_splits=n_splits, shuffle=False)
    return list(splitter.split(np.zeros((n_samples, 1))))


def cross_validated_auc(
    df: pd.DataFrame,
    target_names: Sequence[str],
    max_rows: int,
    max_splits: int,
    random_state: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(random_state)

    for target_name in target_names:
        target_col = TARGET_SPECS[target_name]
        for model_name, features in MODEL_SPECS.items():
            sub = finite_model_frame(
                df,
                target_col=target_col,
                features=features,
                max_rows=max_rows,
                random_state=random_state,
            )
            if sub.empty or sub[target_col].nunique() < 2:
                rows.append(
                    {
                        "target": target_name,
                        "model": model_name,
                        "n": len(sub),
                        "mean_auc": np.nan,
                        "sem_auc": np.nan,
                        "mean_ap": np.nan,
                        "mean_brier": np.nan,
                        "n_folds": 0,
                        "error": "target has fewer than two classes",
                    }
                )
                continue

            y = sub[target_col].astype(int).to_numpy()
            groups = sub["file"].astype(str).to_numpy()
            splits = build_splits(groups, len(sub), max_splits=max_splits)
            fold_auc: List[float] = []
            fold_ap: List[float] = []
            fold_brier: List[float] = []

            for train_idx, test_idx in splits:
                y_train = y[train_idx]
                y_test = y[test_idx]
                if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                    continue
                if not features:
                    pred = np.full(len(test_idx), float(y_train.mean()))
                else:
                    x_train = sub.iloc[train_idx][features].to_numpy(dtype=float)
                    x_test = sub.iloc[test_idx][features].to_numpy(dtype=float)
                    scaler = StandardScaler()
                    x_train = scaler.fit_transform(x_train)
                    x_test = scaler.transform(x_test)
                    model = LogisticRegression(max_iter=1000, random_state=int(rng.integers(0, 1_000_000)))
                    model.fit(x_train, y_train)
                    pred = model.predict_proba(x_test)[:, 1]

                fold_auc.append(float(roc_auc_score(y_test, pred)))
                fold_ap.append(float(average_precision_score(y_test, pred)))
                fold_brier.append(float(brier_score_loss(y_test, pred)))

            rows.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "features": ",".join(features),
                    "n": len(sub),
                    "event_rate": float(y.mean()),
                    "mean_auc": float(np.mean(fold_auc)) if fold_auc else np.nan,
                    "sem_auc": float(scipy_stats.sem(fold_auc)) if len(fold_auc) > 1 else 0.0,
                    "mean_ap": float(np.mean(fold_ap)) if fold_ap else np.nan,
                    "mean_brier": float(np.mean(fold_brier)) if fold_brier else np.nan,
                    "n_folds": len(fold_auc),
                    "error": "" if fold_auc else "no valid CV folds",
                }
            )
    return pd.DataFrame(rows)


def make_uncertainty_bins(
    df: pd.DataFrame,
    target_names: Sequence[str],
    features: Sequence[str],
    n_bins: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for target_name in target_names:
        target_col = TARGET_SPECS[target_name]
        for feat in features:
            if feat not in df.columns:
                continue
            sub = df[[target_col, feat, "file", "subject", "game"]].dropna().copy()
            sub = sub[np.isfinite(sub[[target_col, feat]].to_numpy(dtype=float)).all(axis=1)]
            if sub.empty or sub[feat].nunique() < 2:
                continue
            try:
                sub["bin"] = pd.qcut(sub[feat], q=n_bins, labels=False, duplicates="drop")
            except ValueError:
                continue

            grouped = (
                sub.groupby(["bin", "file"], observed=True)
                .agg(
                    target_rate=(target_col, "mean"),
                    feature_mean=(feat, "mean"),
                    n_steps=(target_col, "size"),
                )
                .reset_index()
            )
            for bin_id, g in grouped.groupby("bin", observed=True):
                all_bin = sub[sub["bin"] == bin_id]
                rows.append(
                    {
                        "target": target_name,
                        "feature": feat,
                        "label": FEATURE_LABELS.get(feat, feat),
                        "bin": int(bin_id),
                        "bin_feature_mean": float(all_bin[feat].mean()),
                        "bin_feature_min": float(all_bin[feat].min()),
                        "bin_feature_max": float(all_bin[feat].max()),
                        "mean_noop_rate": float(g["target_rate"].mean()),
                        "sem_noop_rate": float(g["target_rate"].sem())
                        if len(g) > 1
                        else 0.0,
                        "n_files": int(g["file"].nunique()),
                        "n_steps": int(g["n_steps"].sum()),
                    }
                )
    return pd.DataFrame(rows)


def run_species_interactions(
    df: pd.DataFrame,
    features: Sequence[str],
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    long_parts = []
    for target_name, target_col in TARGET_SPECS.items():
        cols = ["file", "subject", "game", target_col, *features]
        part = df[[col for col in cols if col in df.columns]].copy()
        part = part.rename(columns={target_col: "noop"})
        part["species"] = target_name
        long_parts.append(part)
    long_df = pd.concat(long_parts, ignore_index=True)
    long_df["species_thinker"] = (long_df["species"] == "thinker").astype(int)

    for feat in features:
        sub = finite_model_frame(
            long_df,
            target_col="noop",
            features=[feat, "species_thinker"],
            extra_cols=[],
            max_rows=max_rows,
            random_state=random_state,
        )
        if sub.empty:
            continue
        sub, _, _ = add_standardized_columns(sub, [feat])
        sub["species_thinker"] = sub["species_thinker"].astype(float)
        sub["z_feature_x_species_thinker"] = sub[f"z_{feat}"] * sub["species_thinker"]

        x = pd.DataFrame(
            {
                "Intercept": 1.0,
                f"z_{feat}": sub[f"z_{feat}"].astype(float),
                "species_thinker": sub["species_thinker"].astype(float),
                f"z_{feat}:species_thinker": sub["z_feature_x_species_thinker"].astype(float),
            }
        )
        if sub["file"].nunique() > 1:
            dummies = pd.get_dummies(sub["file"].astype(str), prefix="file", drop_first=True, dtype=float)
            x = pd.concat([x, dummies], axis=1)

        y = sub["noop"].astype(int).to_numpy()
        try:
            model = sm.GLM(y, x.astype(float), family=sm.families.Binomial())
            if sub["file"].nunique() >= 2:
                result = model.fit(
                    maxiter=200,
                    disp=0,
                    cov_type="cluster",
                    cov_kwds={"groups": sub["file"].astype(str).to_numpy()},
                )
            else:
                result = model.fit(maxiter=200, disp=0)
            terms = [f"z_{feat}", "species_thinker", f"z_{feat}:species_thinker"]
            for term in terms:
                rows.append(
                    {
                        "feature": feat,
                        "label": FEATURE_LABELS.get(feat, feat),
                        "term": term,
                        "coef": float(result.params.get(term, np.nan)),
                        "se": float(result.bse.get(term, np.nan)),
                        "z": float(result.tvalues.get(term, np.nan)),
                        "p": float(result.pvalues.get(term, np.nan)),
                        "sig": stars(float(result.pvalues.get(term, np.nan))),
                        "n": len(sub),
                        "event_rate": float(sub["noop"].mean()),
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "feature": feat,
                    "label": FEATURE_LABELS.get(feat, feat),
                    "term": "error",
                    "coef": np.nan,
                    "se": np.nan,
                    "z": np.nan,
                    "p": np.nan,
                    "sig": "n/a",
                    "n": len(sub),
                    "event_rate": float(sub["noop"].mean()) if len(sub) else np.nan,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def extract_noop_bouts(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    target_col = TARGET_SPECS[target_name]
    rows: List[Dict[str, object]] = []
    for file_id, g0 in df.groupby("file", sort=False):
        g = g0.sort_values("real_pos").reset_index(drop=True)
        seq = g[target_col].to_numpy(dtype=int)
        noop_pos = np.flatnonzero(seq == 1)
        if noop_pos.size == 0:
            continue
        cuts = np.where(np.diff(noop_pos) != 1)[0] + 1
        for bout_idx, seg in enumerate(np.split(noop_pos, cuts)):
            if seg.size == 0:
                continue
            start_pos = int(seg[0])
            end_pos = int(seg[-1])
            pre_pos = start_pos - 1
            commit_pos = end_pos + 1
            if pre_pos < 0 or commit_pos >= len(g):
                continue
            if int(g.loc[commit_pos, target_col]) == 1:
                continue
            rows.append(
                {
                    "target": target_name,
                    "event_id": f"{Path(file_id).stem}::{target_name}::b{bout_idx:04d}",
                    "file": file_id,
                    "source_file_name": Path(file_id).name,
                    "subject": int(g.loc[start_pos, "subject"]),
                    "game": int(g.loc[start_pos, "game"]),
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "pre_pos": pre_pos,
                    "commit_pos": commit_pos,
                    "length_real_steps": end_pos - start_pos + 1,
                    "entropy_at_pre": float(g.loc[pre_pos, "entropy_actor"]),
                    "entropy_at_onset": float(g.loc[start_pos, "entropy_actor"]),
                    "entropy_at_commit": float(g.loc[commit_pos, "entropy_actor"]),
                    "neg_q_gap_at_pre": float(g.loc[pre_pos, "neg_q_gap"]),
                    "neg_q_gap_at_onset": float(g.loc[start_pos, "neg_q_gap"]),
                    "neg_q_gap_at_commit": float(g.loc[commit_pos, "neg_q_gap"]),
                    "rollout_spread_at_onset": float(g.loc[start_pos, "rollout_spread"]),
                    "search_jsd_at_onset": float(g.loc[start_pos, "search_jsd_actor_root"]),
                    "tree_width_at_onset": float(g.loc[start_pos, "tree_width"]),
                    "prev_reward_5_at_onset": float(g.loc[start_pos, "prev_reward_5"]),
                    "entropy_change_pre_to_commit": float(
                        g.loc[commit_pos, "entropy_actor"] - g.loc[pre_pos, "entropy_actor"]
                    ),
                    "neg_q_gap_change_pre_to_commit": float(
                        g.loc[commit_pos, "neg_q_gap"] - g.loc[pre_pos, "neg_q_gap"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_bout_length_regression(bouts: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    predictors = [
        "entropy_at_onset",
        "neg_q_gap_at_onset",
        "rollout_spread_at_onset",
        "search_jsd_at_onset",
        "tree_width_at_onset",
        "prev_reward_5_at_onset",
    ]
    for target_name, g0 in bouts.groupby("target", sort=False):
        g = g0.dropna(subset=["length_real_steps", *predictors]).copy()
        if len(g) < 10:
            rows.append({"target": target_name, "term": "error", "error": "fewer than 10 bouts"})
            continue
        g["log_length"] = np.log1p(g["length_real_steps"].astype(float))
        g, _, _ = add_standardized_columns(g, predictors)
        x = design_matrix(g, predictors, include_fixed_effects=True)
        y = g["log_length"].to_numpy(dtype=float)
        try:
            result = sm.OLS(y, x).fit()
            for term in result.params.index:
                if term == "Intercept" or term.startswith(("file_", "subject_", "game_")):
                    continue
                feature = term[2:] if term.startswith("z_") else term
                rows.append(
                    {
                        "target": target_name,
                        "term": term,
                        "feature": feature,
                        "label": FEATURE_LABELS.get(feature, feature),
                        "coef": float(result.params[term]),
                        "se": float(result.bse[term]),
                        "t": float(result.tvalues[term]),
                        "p": float(result.pvalues[term]),
                        "sig": stars(float(result.pvalues[term])),
                        "n_bouts": len(g),
                        "r2": float(result.rsquared),
                        "aic": float(result.aic),
                    }
                )
        except Exception as exc:
            rows.append({"target": target_name, "term": "error", "error": str(exc), "n_bouts": len(g)})
    return pd.DataFrame(rows)


def run_ar_residual_tests(
    df: pd.DataFrame,
    target_names: Sequence[str],
    features: Sequence[str],
    max_lag: int,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for target_name in target_names:
        target_col = TARGET_SPECS[target_name]
        lag_cols_all = [f"{target_name}_noop_lag{k}" for k in range(1, max_lag + 1)]
        for feature in features:
            for k in range(1, max_lag + 1):
                lag_cols = lag_cols_all[:k]
                sub = finite_model_frame(
                    df,
                    target_col=target_col,
                    features=[feature, *lag_cols],
                    max_rows=max_rows,
                    random_state=random_state,
                )
                if sub.empty:
                    continue
                sub, _, _ = add_standardized_columns(sub, [feature, *lag_cols])
                result, _, error = fit_glm_binomial(sub, target_col, [feature, *lag_cols])
                if result is None:
                    rows.append(
                        {
                            "target": target_name,
                            "feature": feature,
                            "lag_order": k,
                            "error": error,
                            "n": len(sub),
                        }
                    )
                    continue
                term = f"z_{feature}"
                rows.append(
                    {
                        "target": target_name,
                        "feature": feature,
                        "label": FEATURE_LABELS.get(feature, feature),
                        "lag_order": k,
                        "coef": float(result.params.get(term, np.nan)),
                        "se": float(result.bse.get(term, np.nan)),
                        "z": float(result.tvalues.get(term, np.nan)),
                        "p": float(result.pvalues.get(term, np.nan)),
                        "sig": stars(float(result.pvalues.get(term, np.nan))),
                        "aic": float(result.aic),
                        "bic": float(getattr(result, "bic_llf", result.bic)),
                        "n": len(sub),
                        "event_rate": float(sub[target_col].mean()),
                        "error": "",
                    }
                )
    return pd.DataFrame(rows)


def run_lagged_precedence_tests(
    df: pd.DataFrame,
    target_names: Sequence[str],
    features: Sequence[str],
    max_lag: int,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for target_name in target_names:
        target_col = TARGET_SPECS[target_name]
        target_lags = [f"{target_name}_noop_lag{k}" for k in range(1, max_lag + 1)]
        for feature in features:
            feature_lags = [f"{feature}_lag{k}" for k in range(1, max_lag + 1)]
            for k in range(1, max_lag + 1):
                ar_cols = target_lags[:k]
                lag_cols = feature_lags[:k]
                sub = finite_model_frame(
                    df,
                    target_col=target_col,
                    features=[*ar_cols, *lag_cols],
                    max_rows=max_rows,
                    random_state=random_state,
                )
                if sub.empty:
                    continue
                sub, _, _ = add_standardized_columns(sub, [*ar_cols, *lag_cols])
                ar_result, _, ar_error = fit_glm_binomial(sub, target_col, ar_cols)
                full_result, _, full_error = fit_glm_binomial(sub, target_col, [*ar_cols, *lag_cols])
                if ar_result is None or full_result is None:
                    rows.append(
                        {
                            "target": target_name,
                            "feature": feature,
                            "lag_order": k,
                            "error": ar_error or full_error,
                            "n": len(sub),
                        }
                    )
                    continue
                lr_stat = max(0.0, 2.0 * (float(full_result.llf) - float(ar_result.llf)))
                lr_df = max(1, int(full_result.df_model - ar_result.df_model))
                lr_p = float(scipy_stats.chi2.sf(lr_stat, lr_df))
                first_lag_term = f"z_{feature}_lag1"
                rows.append(
                    {
                        "target": target_name,
                        "feature": feature,
                        "label": FEATURE_LABELS.get(feature, feature),
                        "lag_order": k,
                        "lr_stat": lr_stat,
                        "lr_df": lr_df,
                        "lr_p": lr_p,
                        "lr_sig": stars(lr_p),
                        "delta_aic_ar_minus_full": float(ar_result.aic - full_result.aic),
                        "delta_bic_ar_minus_full": float(
                            getattr(ar_result, "bic_llf", ar_result.bic)
                            - getattr(full_result, "bic_llf", full_result.bic)
                        ),
                        "lag1_coef": float(full_result.params.get(first_lag_term, np.nan)),
                        "lag1_p": float(full_result.pvalues.get(first_lag_term, np.nan)),
                        "n": len(sub),
                        "event_rate": float(sub[target_col].mean()),
                        "error": "",
                    }
                )
    return pd.DataFrame(rows)


def run_onset_trigger_models(
    df: pd.DataFrame,
    target_names: Sequence[str],
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    features = [
        "entropy_actor",
        "neg_q_gap",
        "rollout_spread",
        "search_jsd_actor_root",
        "tree_width",
        "root_value",
        "prev_reward_5",
    ]
    rows: List[Dict[str, object]] = []
    for target_name in target_names:
        target_col = TARGET_SPECS[target_name]
        lag1_col = f"{target_name}_noop_lag1"
        onset_col = f"{target_name}_noop_onset"
        candidates = df[df[lag1_col] == 0].copy()
        candidates = candidates.rename(columns={onset_col: "onset_target"})

        for model_name, model_features in {
            "entropy_only": ["entropy_actor"],
            "margin_only": ["neg_q_gap"],
            "reward_context": ["prev_reward_5", "root_value"],
            "search_context": ["search_jsd_actor_root", "rollout_spread", "tree_width"],
            "full_onset": features,
        }.items():
            sub = finite_model_frame(
                candidates,
                target_col="onset_target",
                features=model_features,
                max_rows=max_rows,
                random_state=random_state,
            )
            sub, _, _ = add_standardized_columns(sub, model_features)
            result, _, error = fit_glm_binomial(sub, "onset_target", model_features)
            row = {
                "target": target_name,
                "model": model_name,
                "features": ",".join(model_features),
                "n": len(sub),
                "events": int(sub["onset_target"].sum()) if len(sub) else 0,
                "event_rate": float(sub["onset_target"].mean()) if len(sub) else np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "llf": np.nan,
                "error": error or "",
            }
            if result is not None:
                row.update(
                    {
                        "aic": float(result.aic),
                        "bic": float(getattr(result, "bic_llf", result.bic)),
                        "llf": float(result.llf),
                    }
                )
            rows.append(row)

        # Save single-feature coefficients from the full onset model.
        sub = finite_model_frame(
            candidates,
            target_col="onset_target",
            features=features,
            max_rows=max_rows,
            random_state=random_state,
        )
        sub, _, _ = add_standardized_columns(sub, features)
        result, _, _ = fit_glm_binomial(sub, "onset_target", features)
        if result is not None:
            for feat in features:
                term = f"z_{feat}"
                rows.append(
                    {
                        "target": target_name,
                        "model": "full_onset_coef",
                        "feature": feat,
                        "label": FEATURE_LABELS.get(feat, feat),
                        "coef": float(result.params.get(term, np.nan)),
                        "se": float(result.bse.get(term, np.nan)),
                        "z": float(result.tvalues.get(term, np.nan)),
                        "p": float(result.pvalues.get(term, np.nan)),
                        "sig": stars(float(result.pvalues.get(term, np.nan))),
                        "n": len(sub),
                        "event_rate": float(sub["onset_target"].mean()) if len(sub) else np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty and "aic" in out.columns:
        mask = out["model"].ne("full_onset_coef")
        out.loc[mask, "delta_aic"] = out[mask].groupby("target")["aic"].transform(lambda s: s - s.min())
    return out


def build_commitment_transition_rows(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    target_col = TARGET_SPECS[target_name]
    rows: List[Dict[str, object]] = []
    for file_id, g0 in df.groupby("file", sort=False):
        g = g0.sort_values("real_pos").reset_index(drop=True)
        seq = g[target_col].to_numpy(dtype=int)
        noop_pos = np.flatnonzero(seq == 1)
        if noop_pos.size == 0:
            continue
        cuts = np.where(np.diff(noop_pos) != 1)[0] + 1
        for seg in np.split(noop_pos, cuts):
            if seg.size == 0:
                continue
            start = int(seg[0])
            end = int(seg[-1])
            commit = end + 1
            if commit >= len(g):
                continue
            # Rows after the first NOOP ask: continue withholding or commit now?
            for pos in range(start + 1, commit + 1):
                rows.append(
                    {
                        "target": target_name,
                        "file": file_id,
                        "subject": int(g.loc[pos, "subject"]),
                        "game": int(g.loc[pos, "game"]),
                        "pos": pos,
                        "commit_now": int(pos == commit),
                        "bout_age": int(pos - start),
                        "entropy_actor": float(g.loc[pos, "entropy_actor"]),
                        "neg_q_gap": float(g.loc[pos, "neg_q_gap"]),
                        "rollout_spread": float(g.loc[pos, "rollout_spread"]),
                        "search_jsd_actor_root": float(g.loc[pos, "search_jsd_actor_root"]),
                        "tree_width": float(g.loc[pos, "tree_width"]),
                    }
                )
    return pd.DataFrame(rows)


def run_commitment_trigger_models(
    df: pd.DataFrame,
    target_names: Sequence[str],
    max_rows: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []
    model_specs = {
        "duration_only": ["bout_age"],
        "uncertainty_only": ["entropy_actor", "neg_q_gap", "rollout_spread", "search_jsd_actor_root"],
        "duration_plus_uncertainty": [
            "bout_age",
            "entropy_actor",
            "neg_q_gap",
            "rollout_spread",
            "search_jsd_actor_root",
        ],
        "entropy_duration_interaction": [
            "bout_age",
            "entropy_actor",
            "neg_q_gap",
            "rollout_spread",
            "search_jsd_actor_root",
        ],
    }

    for target_name in target_names:
        trans = build_commitment_transition_rows(df, target_name)
        if trans.empty:
            model_rows.append({"target": target_name, "model": "error", "error": "no commitment rows"})
            continue

        for model_name, features in model_specs.items():
            sub = finite_model_frame(
                trans,
                target_col="commit_now",
                features=features,
                max_rows=max_rows,
                random_state=random_state,
            )
            sub, _, _ = add_standardized_columns(sub, features)
            interactions: List[Tuple[str, str]] = []
            if model_name == "entropy_duration_interaction":
                interactions = [("entropy_actor", "bout_age")]
            result, _, error = fit_glm_binomial(
                sub,
                "commit_now",
                features,
                interaction_pairs=interactions,
            )
            row = {
                "target": target_name,
                "model": model_name,
                "features": ",".join(features),
                "n": len(sub),
                "events": int(sub["commit_now"].sum()) if len(sub) else 0,
                "event_rate": float(sub["commit_now"].mean()) if len(sub) else np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "llf": np.nan,
                "error": error or "",
            }
            if result is not None:
                row.update(
                    {
                        "aic": float(result.aic),
                        "bic": float(getattr(result, "bic_llf", result.bic)),
                        "llf": float(result.llf),
                    }
                )
                for term in result.params.index:
                    if term == "Intercept" or term.startswith(("file_", "subject_", "game_")):
                        continue
                    feature = term.replace("z_", "")
                    coef_rows.append(
                        {
                            "target": target_name,
                            "model": model_name,
                            "term": term,
                            "feature": feature,
                            "label": FEATURE_LABELS.get(feature, feature),
                            "coef": float(result.params[term]),
                            "se": float(result.bse[term]),
                            "z": float(result.tvalues[term]),
                            "p": float(result.pvalues[term]),
                            "sig": stars(float(result.pvalues[term])),
                            "n": len(sub),
                            "event_rate": row["event_rate"],
                        }
                    )
            model_rows.append(row)

    model_df = pd.DataFrame(model_rows)
    if not model_df.empty and "aic" in model_df.columns:
        model_df["delta_aic"] = model_df.groupby("target")["aic"].transform(lambda s: s - s.min())
    return model_df, pd.DataFrame(coef_rows)


def run_optional_mixed_logit(
    df: pd.DataFrame,
    target_names: Sequence[str],
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception as exc:
        return pd.DataFrame([{"error": f"BinomialBayesMixedGLM unavailable: {exc}"}])

    rows: List[Dict[str, object]] = []
    features = CORE_FEATURES
    for target_name in target_names:
        target_col = TARGET_SPECS[target_name]
        sub = finite_model_frame(
            df,
            target_col=target_col,
            features=features,
            max_rows=max_rows,
            random_state=random_state,
        )
        if sub.empty or sub[target_col].nunique() < 2:
            rows.append({"target": target_name, "term": "error", "error": "target has fewer than two classes"})
            continue
        sub, _, _ = add_standardized_columns(sub, features)
        sub = sub.rename(columns={target_col: "noop"})
        fixed_terms = " + ".join(f"z_{feat}" for feat in features)
        vc_formulas = {}
        if sub["subject"].nunique() > 1:
            vc_formulas["subject"] = "0 + C(subject)"
        if sub["game"].nunique() > 1:
            vc_formulas["game"] = "0 + C(game)"
        if sub["file"].nunique() > 1:
            vc_formulas["file"] = "0 + C(file)"
        if not vc_formulas:
            rows.append({"target": target_name, "term": "error", "error": "no grouping levels for random effects"})
            continue

        try:
            model = BinomialBayesMixedGLM.from_formula(f"noop ~ {fixed_terms}", vc_formulas, sub)
            result = model.fit_vb()
            names = list(result.model.exog_names)
            params = np.asarray(result.params[: len(names)], dtype=float)
            for term, coef in zip(names, params):
                if term == "Intercept":
                    continue
                feature = term[2:] if term.startswith("z_") else term
                rows.append(
                    {
                        "target": target_name,
                        "term": term,
                        "feature": feature,
                        "label": FEATURE_LABELS.get(feature, feature),
                        "posterior_mean_coef": float(coef),
                        "n": len(sub),
                        "event_rate": float(sub["noop"].mean()),
                    }
                )
        except Exception as exc:
            rows.append({"target": target_name, "term": "error", "error": str(exc), "n": len(sub)})
    return pd.DataFrame(rows)


def plot_uncertainty_coupling(bin_df: pd.DataFrame, out_path: Path) -> None:
    features = [feat for feat in CORE_FEATURES if feat in set(bin_df.get("feature", []))]
    if not features:
        return
    n_cols = min(4, len(features))
    n_rows = math.ceil(len(features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, feat in zip(axes_flat, features):
        sub = bin_df[bin_df["feature"] == feat].copy()
        for target_name in TARGET_SPECS:
            g = sub[sub["target"] == target_name].sort_values("bin")
            if g.empty:
                continue
            ax.errorbar(
                g["bin_feature_mean"],
                g["mean_noop_rate"],
                yerr=g["sem_noop_rate"],
                marker="o",
                lw=2,
                capsize=3,
                color=PALETTE.get(target_name, PALETTE["neutral"]),
                label=target_name,
            )
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=10)
        ax.set_xlabel("Feature bin mean")
        ax.set_ylabel("NOOP rate")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes_flat[len(features) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper right")
    fig.suptitle("Fig 3-1: Uncertainty-NOOP coupling", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.96, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_model_comparison(model_df: pd.DataFrame, auc_df: pd.DataFrame, out_path: Path) -> None:
    if model_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # Panel A: delta AIC by target/model.
    ax = axes[0]
    valid = model_df[model_df["model"] != "null"].dropna(subset=["delta_aic"]).copy()
    if not valid.empty:
        order = (
            valid.groupby("model")["delta_aic"]
            .mean()
            .sort_values()
            .index.tolist()
        )
        x = np.arange(len(order), dtype=float)
        width = 0.36
        for i, target_name in enumerate(TARGET_SPECS):
            vals = [
                valid[(valid["target"] == target_name) & (valid["model"] == model)]["delta_aic"].mean()
                for model in order
            ]
            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width=width,
                label=target_name,
                color=PALETTE.get(target_name, PALETTE["neutral"]),
                alpha=0.85,
            )
        ax.set_xticks(x, order, rotation=35, ha="right")
        ax.set_ylabel("Delta AIC (lower is better)")
        ax.set_title("Fig 3-2A: Logistic model comparison")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    # Panel B: cross-validated AUC.
    ax = axes[1]
    valid_auc = auc_df[auc_df["model"] != "null"].dropna(subset=["mean_auc"]).copy()
    if not valid_auc.empty:
        order = (
            valid_auc.groupby("model")["mean_auc"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        x = np.arange(len(order), dtype=float)
        width = 0.36
        for i, target_name in enumerate(TARGET_SPECS):
            vals = [
                valid_auc[(valid_auc["target"] == target_name) & (valid_auc["model"] == model)]["mean_auc"].mean()
                for model in order
            ]
            errs = [
                valid_auc[(valid_auc["target"] == target_name) & (valid_auc["model"] == model)]["sem_auc"].mean()
                for model in order
            ]
            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width=width,
                yerr=errs,
                capsize=3,
                label=target_name,
                color=PALETTE.get(target_name, PALETTE["neutral"]),
                alpha=0.85,
            )
        ax.axhline(0.5, color="#6b7280", lw=1, ls="--")
        ax.set_xticks(x, order, rotation=35, ha="right")
        ax.set_ylim(0.45, min(1.0, max(0.75, valid_auc["mean_auc"].max() + 0.08)))
        ax.set_ylabel("Cross-validated AUC")
        ax.set_title("Fig 3-2B: NOOP prediction accuracy")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_temporal_ordering(
    ar_df: pd.DataFrame,
    lagged_df: pd.DataFrame,
    commit_coef_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    for target_name in TARGET_SPECS:
        for feat, ls in [("entropy_actor", "-"), ("neg_q_gap", "--")]:
            sub = ar_df[(ar_df["target"] == target_name) & (ar_df["feature"] == feat)].sort_values("lag_order")
            if sub.empty:
                continue
            ax.errorbar(
                sub["lag_order"],
                sub["coef"],
                yerr=1.96 * sub["se"],
                marker="o",
                lw=2,
                ls=ls,
                capsize=3,
                color=PALETTE.get(target_name, PALETTE["neutral"]),
                label=f"{target_name}: {FEATURE_LABELS.get(feat, feat)}",
            )
    ax.axhline(0, color="#6b7280", lw=1, ls="--")
    ax.set_xlabel("AR lag order controlled")
    ax.set_ylabel("Current uncertainty coef")
    ax.set_title("Fig 3-3A: AR residual uncertainty")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    if not lagged_df.empty:
        plot_lag = lagged_df[lagged_df["feature"].isin(["entropy_actor", "neg_q_gap"])].copy()
        for target_name in TARGET_SPECS:
            sub = plot_lag[plot_lag["target"] == target_name]
            if sub.empty:
                continue
            by_lag = sub.groupby("lag_order")["delta_aic_ar_minus_full"].mean().reset_index()
            ax.plot(
                by_lag["lag_order"],
                by_lag["delta_aic_ar_minus_full"],
                marker="o",
                lw=2,
                color=PALETTE.get(target_name, PALETTE["neutral"]),
                label=target_name,
            )
    ax.axhline(0, color="#6b7280", lw=1, ls="--")
    ax.set_xlabel("Lag order")
    ax.set_ylabel("AIC improvement over AR-only")
    ax.set_title("Fig 3-3B: Lagged temporal precedence")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    if {"model", "term"}.issubset(commit_coef_df.columns):
        sub = commit_coef_df[
            (commit_coef_df["model"] == "entropy_duration_interaction")
            & (~commit_coef_df["term"].str.startswith("file_", na=False))
        ].copy()
    else:
        sub = pd.DataFrame()
    if not sub.empty:
        keep_terms = ["z_entropy_actor", "z_bout_age", "z_entropy_actor:z_bout_age", "z_neg_q_gap"]
        sub = sub[sub["term"].isin(keep_terms)]
        labels = sub["term"].drop_duplicates().tolist()
        x = np.arange(len(labels), dtype=float)
        width = 0.36
        for i, target_name in enumerate(TARGET_SPECS):
            g = sub[sub["target"] == target_name].set_index("term")
            vals = [g.loc[term, "coef"] if term in g.index else np.nan for term in labels]
            errs = [1.96 * g.loc[term, "se"] if term in g.index else np.nan for term in labels]
            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width=width,
                yerr=errs,
                capsize=3,
                color=PALETTE.get(target_name, PALETTE["neutral"]),
                alpha=0.85,
                label=target_name,
            )
        ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.axhline(0, color="#6b7280", lw=1, ls="--")
    ax.set_ylabel("Commit-now logit coef")
    ax.set_title("Fig 3-3C: Commitment trigger")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_summary(
    out_path: Path,
    df: pd.DataFrame,
    model_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    auc_df: pd.DataFrame,
    ar_df: pd.DataFrame,
    lagged_df: pd.DataFrame,
    onset_df: pd.DataFrame,
    commit_df: pd.DataFrame,
    bout_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("Section 3 computational interpretation summary")
    lines.append("")
    lines.append(f"Real steps: {len(df):,}")
    lines.append(f"Files: {df['file'].nunique():,}")
    lines.append(f"Subjects: {df['subject'].nunique():,}")
    lines.append(f"Games: {', '.join(df['game_name'].drop_duplicates().astype(str))}")
    lines.append("")

    for target_name, target_col in TARGET_SPECS.items():
        lines.append(f"[{target_name}]")
        lines.append(f"- NOOP rate: {df[target_col].mean():.3f}")

        target_models = model_df[(model_df["target"] == target_name) & model_df["aic"].notna()]
        if not target_models.empty:
            best = target_models.sort_values("aic").iloc[0]
            lines.append(
                f"- Best AIC model: {best['model']} "
                f"(delta AIC=0, AIC={best['aic']:.1f}, n={int(best['n'])})"
            )
        target_auc = auc_df[(auc_df["target"] == target_name) & auc_df["mean_auc"].notna()]
        if not target_auc.empty:
            best_auc = target_auc.sort_values("mean_auc", ascending=False).iloc[0]
            lines.append(
                f"- Best CV AUC: {best_auc['model']} "
                f"(AUC={best_auc['mean_auc']:.3f}, folds={int(best_auc['n_folds'])})"
            )
        full_coef = coef_df[(coef_df["target"] == target_name) & (coef_df["model"] == "full_uncertainty")]
        if not full_coef.empty:
            top = full_coef.reindex(full_coef["coef"].abs().sort_values(ascending=False).index).head(3)
            pretty = ", ".join(
                f"{row.label}={row.coef:+.3f} ({row.sig})"
                for row in top.itertuples()
            )
            lines.append(f"- Strongest full-model standardized coefficients: {pretty}")

        target_ar = ar_df[(ar_df["target"] == target_name) & (ar_df["feature"].isin(["entropy_actor", "neg_q_gap"]))]
        if not target_ar.empty:
            ar_pretty = ", ".join(
                f"{FEATURE_LABELS.get(row.feature, row.feature)} k{int(row.lag_order)}={row.coef:+.3f}"
                for row in target_ar[target_ar["lag_order"] == target_ar["lag_order"].max()].itertuples()
            )
            lines.append(f"- AR-controlled uncertainty at max lag: {ar_pretty}")

        target_lag = lagged_df[(lagged_df["target"] == target_name) & lagged_df["delta_aic_ar_minus_full"].notna()]
        if not target_lag.empty:
            best_lag = target_lag.sort_values("delta_aic_ar_minus_full", ascending=False).iloc[0]
            lines.append(
                f"- Strongest lagged improvement: {best_lag['label']} "
                f"k={int(best_lag['lag_order'])}, AIC gain={best_lag['delta_aic_ar_minus_full']:.1f}"
            )

        target_onset = onset_df[
            (onset_df["target"] == target_name)
            & (onset_df["model"] != "full_onset_coef")
            & onset_df["aic"].notna()
        ]
        if not target_onset.empty:
            best_onset = target_onset.sort_values("aic").iloc[0]
            lines.append(f"- Best onset trigger model: {best_onset['model']} (AIC={best_onset['aic']:.1f})")

        target_commit = commit_df[(commit_df["target"] == target_name) & commit_df["aic"].notna()]
        if not target_commit.empty:
            best_commit = target_commit.sort_values("aic").iloc[0]
            lines.append(f"- Best commitment trigger model: {best_commit['model']} (AIC={best_commit['aic']:.1f})")

        target_bouts = bout_df[bout_df["target"] == target_name] if not bout_df.empty else pd.DataFrame()
        if not target_bouts.empty:
            lines.append(
                f"- Complete NOOP bouts: {len(target_bouts):,}; "
                f"mean length={target_bouts['length_real_steps'].mean():.2f}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n")


def ensure_output_dirs(out_dir: Path) -> Tuple[Path, Path]:
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, res_dir


def run_analysis(args: argparse.Namespace) -> None:
    fig_dir, res_dir = ensure_output_dirs(args.out_dir)

    df = load_real_step_dataset(args.input_dir, game_id=args.game_id)
    target_names = [name for name in args.targets if name in TARGET_SPECS]
    if not target_names:
        raise ValueError(f"No valid targets requested. Choose from {sorted(TARGET_SPECS)}")

    print(f"[data] real steps={len(df):,} files={df['file'].nunique()} targets={target_names}")
    df.to_csv(res_dir / "real_step_computational_metrics.csv", index=False)

    bin_df = make_uncertainty_bins(df, target_names, CORE_FEATURES, n_bins=args.n_bins)
    bin_df.to_csv(res_dir / "uncertainty_noop_bins.csv", index=False)

    model_df, coef_df = run_logistic_model_comparison(
        df,
        target_names=target_names,
        max_rows=args.max_glm_rows,
        random_state=args.random_state,
    )
    model_df.to_csv(res_dir / "logistic_model_comparison.csv", index=False)
    coef_df.to_csv(res_dir / "logistic_coefficients.csv", index=False)

    auc_df = cross_validated_auc(
        df,
        target_names=target_names,
        max_rows=args.max_cv_rows,
        max_splits=args.max_splits,
        random_state=args.random_state,
    )
    auc_df.to_csv(res_dir / "cv_auc_scores.csv", index=False)

    species_df = run_species_interactions(
        df,
        features=CORE_FEATURES,
        max_rows=args.max_glm_rows,
        random_state=args.random_state,
    )
    species_df.to_csv(res_dir / "species_interactions.csv", index=False)

    ar_df = run_ar_residual_tests(
        df,
        target_names=target_names,
        features=["entropy_actor", "neg_q_gap"],
        max_lag=args.max_lag,
        max_rows=args.max_glm_rows,
        random_state=args.random_state,
    )
    ar_df.to_csv(res_dir / "ar_residual_uncertainty.csv", index=False)

    lagged_df = run_lagged_precedence_tests(
        df,
        target_names=target_names,
        features=["entropy_actor", "neg_q_gap", "search_jsd_actor_root", "rollout_spread"],
        max_lag=args.max_lag,
        max_rows=args.max_glm_rows,
        random_state=args.random_state,
    )
    lagged_df.to_csv(res_dir / "lagged_temporal_precedence.csv", index=False)

    onset_df = run_onset_trigger_models(
        df,
        target_names=target_names,
        max_rows=args.max_glm_rows,
        random_state=args.random_state,
    )
    onset_df.to_csv(res_dir / "onset_trigger_models.csv", index=False)

    commit_df, commit_coef_df = run_commitment_trigger_models(
        df,
        target_names=target_names,
        max_rows=args.max_glm_rows,
        random_state=args.random_state,
    )
    commit_df.to_csv(res_dir / "commitment_trigger_models.csv", index=False)
    commit_coef_df.to_csv(res_dir / "commitment_trigger_coefficients.csv", index=False)

    all_bouts = []
    for target_name in target_names:
        bouts = extract_noop_bouts(df, target_name)
        if not bouts.empty:
            all_bouts.append(bouts)
    bout_df = pd.concat(all_bouts, ignore_index=True) if all_bouts else pd.DataFrame()
    bout_df.to_csv(res_dir / "noop_bouts.csv", index=False)

    bout_reg_df = run_bout_length_regression(bout_df) if not bout_df.empty else pd.DataFrame()
    bout_reg_df.to_csv(res_dir / "bout_length_regression.csv", index=False)

    if args.run_mixed_logit:
        mixed_df = run_optional_mixed_logit(
            df,
            target_names=target_names,
            max_rows=args.max_mixed_rows,
            random_state=args.random_state,
        )
        mixed_df.to_csv(res_dir / "mixed_logit_vb_coefficients.csv", index=False)

    plot_uncertainty_coupling(bin_df, fig_dir / "fig_3_1_uncertainty_coupling.png")
    plot_model_comparison(model_df, auc_df, fig_dir / "fig_3_2_model_comparison.png")
    plot_temporal_ordering(
        ar_df,
        lagged_df,
        commit_coef_df,
        fig_dir / "fig_3_3_temporal_ordering.png",
    )

    write_summary(
        res_dir / "summary.txt",
        df=df,
        model_df=model_df,
        coef_df=coef_df,
        auc_df=auc_df,
        ar_df=ar_df,
        lagged_df=lagged_df,
        onset_df=onset_df,
        commit_df=commit_df,
        bout_df=bout_df,
    )

    print(f"[done] Section 3 analysis written to {args.out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Section 3: computational interpretation of NOOP withholding")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing filtered thinker trace .npy files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--game-id",
        type=int,
        default=None,
        help="Optional game filter, e.g. 2 for Space Invaders.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(TARGET_SPECS),
        default=["human", "thinker"],
        help="NOOP target(s) to analyze.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Number of quantile bins for uncertainty coupling plots.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=5,
        help="Maximum NOOP/uncertainty lag order for temporal tests.",
    )
    parser.add_argument(
        "--max-glm-rows",
        type=int,
        default=75_000,
        help="Maximum rows sampled for statsmodels GLM fits.",
    )
    parser.add_argument(
        "--max-cv-rows",
        type=int,
        default=100_000,
        help="Maximum rows sampled for cross-validated AUC fits.",
    )
    parser.add_argument(
        "--max-mixed-rows",
        type=int,
        default=20_000,
        help="Maximum rows sampled for optional mixed logistic VB fits.",
    )
    parser.add_argument(
        "--max-splits",
        type=int,
        default=5,
        help="Maximum GroupKFold/KFold splits for cross-validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for row sampling and CV models.",
    )
    parser.add_argument(
        "--run-mixed-logit",
        action="store_true",
        help="Also run optional BinomialBayesMixedGLM full-model robustness fits.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
