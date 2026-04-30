#!/usr/bin/env python3
"""
Analysis 3 — Game State Feature × NOOP

Research plan connection: Section 3-1 (uncertainty-NOOP coupling),
behavioral + RAM-based game state version.

For each game, extract game-state features from RAM, then test whether
NOOP probability varies systematically with those features.

Games
-----
Pong         (game_1): ball_x, ball_y, player_y, paddle_dist (|ball_y - player_y|)
SpaceInvaders (game_2): enemy_count, threat_level (enemy lowest y), player_x

RAM addresses (confirmed empirically)
--------------------------------------
Pong
  49  player paddle Y
  50  CPU paddle Y
  54  ball X
 121  ball Y

SpaceInvaders
  17  enemy count  (0–36, starts at 36, decreases by 1 per kill, resets on new wave)
  82  player X     (responds to LEFT / RIGHT actions)
  16  threat level / enemy descent proxy

Analysis pipeline
-----------------
1. Load all blocks (6 subjects × 2 games × 11 blocks).
2. Compute step-level features from RAM + NOOP labels from actions.
3. Bin-level NOOP rate: 4 quantile bins per feature, subject-mean ± SE.
4. Mixed-effects logistic regression: NOOP ~ features + (1|subject)
   using statsmodels MixedLM on a linear probability model (LPM) for
   interpretability.  Logistic MEM via statsmodels BinomialBayesMixedGLM
   as robustness check.
5. Cross-subject sign test: for each bin boundary, how many subjects show
   the same direction of NOOP increase?

Outputs (all in outputs/game_state_noop/)
-----------------------------------------
  figures/
    fig_pong_binned.png
    fig_si_binned.png
    fig_pong_regression.png
    fig_si_regression.png
  results/
    pong_regression.csv
    si_regression.csv
    sign_test.csv
    summary.txt
"""
from __future__ import annotations

import glob
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_ROOT = ROOT / "behavioral_data_block_old"
OUT_DIR   = Path(__file__).parent / "outputs" / "06_game_state_noop"
FIG_DIR   = OUT_DIR / "figures"
RES_DIR   = OUT_DIR / "results"
for d in [FIG_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"sub_{i}" for i in range(1, 7)]
NOOP_IDX = 0
N_BINS   = 4          # quantile bins for binned analysis

# ──────────────────────────────────────────────────────────────────────────────
# RAM addresses
# ──────────────────────────────────────────────────────────────────────────────
PONG_RAM = {
    # Verified against atari-representation-learning RAM annotations
    # and confirmed empirically (action-response analysis)
    "player_y":   51,   # strong response to UP(2)/DOWN(3): ±4 per frame
    "cpu_y":      50,   # opponent paddle Y (CPU-controlled)
    "ball_x":     49,   # ball horizontal position (range 0–205)
    "ball_y":     54,   # ball vertical position (range 0–207)
    "cpu_score":  13,   # CPU (opponent) score; corr=1.0 with cumulative reward=-1
    "player_score": 14, # Player score; corr=1.0 with cumulative reward=+1
}
# Player paddle X is fixed in Pong (addr 46 = 188, confirmed constant across all frames;
# addr 45 = 64 = cpu paddle X). Verified: ball_x = 205 at reward=-1 (right wall).
PLAYER_X_PONG = 188
SI_RAM = {
    # Verified against atari-representation-learning RAM annotations
    "enemy_count":  17,   # invaders remaining (0–36, confirmed: drops by 1 per kill)
    "player_x":     28,   # confirmed: RIGHT→+0.47, LEFT→-0.46 per frame (range 35–117)
    "enemies_y":    16,   # threat/progress proxy: rises as invaders descend
    "num_lives":    73,   # lives remaining (0–3)
}

# Enemies Y baseline for the addr 16 proxy.
# Later waves can restart above 0 as the formation begins closer to earth.
ENEMIES_Y_BASELINE = 0

# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_ram(path: str) -> np.ndarray:
    """Parse RAM.txt → float32 array shape (T, 128)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            _, val_str = line.split(":", 1)
            vals = [int(x) for x in val_str.rstrip(",").split(",")]
            rows.append(vals)
    return np.array(rows, dtype=np.float32)


def load_block(block_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ram, actions_idx, rewards, is_first) trimmed to common length."""
    ram_path  = glob.glob(block_dir + "/RAM.txt")[0]
    npz_path  = glob.glob(block_dir + "/*.npz")[0]

    ram = load_ram(ram_path)

    d       = np.load(npz_path, allow_pickle=True)
    actions = np.argmax(d["action"], axis=1).astype(np.int32)
    rewards = d["reward"].astype(np.float32)
    is_first = d["is_first"].astype(bool)

    T = min(len(ram), len(actions))
    return ram[:T], actions[:T], rewards[:T], is_first[:T]


def load_all_blocks(game_id: int) -> pd.DataFrame:
    """
    Load all blocks for all subjects for one game.
    Returns tidy DataFrame with columns:
      subject, game, block, step, noop, + game-specific feature columns
    """
    records: List[Dict] = []

    for sub in SUBJECTS:
        blocks = sorted(glob.glob(
            str(DATA_ROOT / sub / f"game_{game_id}" / "*" / "block_*")
        ))
        for blk_dir in blocks:
            try:
                ram, actions, rewards, is_first = load_block(blk_dir)
            except (IndexError, FileNotFoundError):
                continue

            T     = len(actions)
            noop  = (actions == NOOP_IDX).astype(np.float32)
            sub_id = int(sub.split("_")[1])

            if game_id == 1:  # Pong
                player_y     = ram[:, PONG_RAM["player_y"]]
                cpu_y        = ram[:, PONG_RAM["cpu_y"]]
                ball_x       = ram[:, PONG_RAM["ball_x"]]
                ball_y       = ram[:, PONG_RAM["ball_y"]]
                player_score = ram[:, PONG_RAM["player_score"]]
                cpu_score    = ram[:, PONG_RAM["cpu_score"]]
                score_diff   = (player_score - cpu_score).astype(np.float32)

                # Euclidean distance from ball to player paddle
                # player_x is fixed at PLAYER_X_PONG (addr 46 = 188, constant)
                paddle_dist = np.sqrt(
                    (ball_x - PLAYER_X_PONG) ** 2 + (ball_y - player_y) ** 2
                ).astype(np.float32)
                # in_play: ball is active (ball_x > 0 and ball_y > 0)
                in_play = ((ball_x > 0) & (ball_y > 0)).astype(np.float32)

                for t in range(T):
                    records.append({
                        "subject":      sub_id,
                        "game":         "Pong",
                        "block_dir":    blk_dir,
                        "step":         t,
                        "noop":         float(noop[t]),
                        "player_y":     float(player_y[t]),
                        "cpu_y":        float(cpu_y[t]),
                        "ball_x":       float(ball_x[t]),
                        "ball_y":       float(ball_y[t]),
                        "paddle_dist":  float(paddle_dist[t]),
                        "score_diff":   float(score_diff[t]),
                        "in_play":      float(in_play[t]),
                        "reward":       float(rewards[t]),
                        "is_first":     bool(is_first[t]),
                    })

            else:  # SpaceInvaders
                enemy_count = ram[:, SI_RAM["enemy_count"]]
                player_x    = ram[:, SI_RAM["player_x"]]
                # enemies_y: higher = enemies have descended further = more threat
                enemies_y   = (ram[:, SI_RAM["enemies_y"]]
                               - ENEMIES_Y_BASELINE).clip(0)
                num_lives   = ram[:, SI_RAM["num_lives"]]

                for t in range(T):
                    records.append({
                        "subject":     sub_id,
                        "game":        "SpaceInvaders",
                        "block_dir":   blk_dir,
                        "step":        t,
                        "noop":        float(noop[t]),
                        "enemy_count": float(enemy_count[t]),
                        "player_x":    float(player_x[t]),
                        "enemies_y":   float(enemies_y[t]),
                        "num_lives":   float(num_lives[t]),
                        "reward":      float(rewards[t]),
                        "is_first":    bool(is_first[t]),
                    })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Binned NOOP rate analysis
# ──────────────────────────────────────────────────────────────────────────────

def binned_noop_rate(
    df: pd.DataFrame,
    feature: str,
    n_bins: int = N_BINS,
    filter_fn=None,
) -> pd.DataFrame:
    """
    For each quantile bin of `feature`, compute per-subject NOOP rate,
    then aggregate mean ± SE across subjects.

    Returns DataFrame: bin_label, bin_mid, sub_mean[1..6], mean, se, n_subs
    """
    if filter_fn is not None:
        df = df[filter_fn(df)].copy()

    df = df.dropna(subset=[feature])
    df = df[np.isfinite(df[feature])].copy()

    # Quantile bins using all data
    try:
        df["bin"] = pd.qcut(df[feature], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    bin_edges = df.groupby("bin")[feature].agg(["min", "max", "mean"])

    rows = []
    for bin_id in sorted(df["bin"].dropna().unique()):
        sub_df = df[df["bin"] == bin_id]
        sub_rates = sub_df.groupby("subject")["noop"].mean()
        row = {
            "bin_id":   int(bin_id),
            "bin_min":  bin_edges.loc[bin_id, "min"],
            "bin_max":  bin_edges.loc[bin_id, "max"],
            "bin_mid":  bin_edges.loc[bin_id, "mean"],
            "n_steps":  len(sub_df),
            "n_subs":   len(sub_rates),
            "mean":     float(sub_rates.mean()),
            "se":       float(sub_rates.sem()) if len(sub_rates) > 1 else 0.0,
        }
        for sub_id, rate in sub_rates.items():
            row[f"sub_{sub_id}"] = float(rate)
        rows.append(row)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Mixed-effects logistic regression (linear probability model)
# ──────────────────────────────────────────────────────────────────────────────

def run_mixed_effects_lpm(
    df: pd.DataFrame,
    features: List[str],
    filter_fn=None,
    n_sample: int = 50_000,
    random_state: int = 42,
) -> Dict:
    """
    Linear probability model: NOOP ~ z(features) + (1|subject)
    using statsmodels MixedLM.
    Features are z-scored for interpretability.

    ⚠ n=6 subjects limitation: random intercepts per subject are included,
    but SE/p-values are step-level (N≈50k), not subject-level.
    statsmodels MixedLM does not apply Kenward-Roger df correction, so
    p-values are anti-conservative. Interpret effect sizes (% of baseline)
    rather than p-values when n_subjects is small.

    Returns dict with coef, se, z, p for each feature + model fit info.
    """
    if filter_fn is not None:
        df = df[filter_fn(df)].copy()

    df = df.dropna(subset=features + ["noop", "subject"]).copy()
    df = df[np.isfinite(df[features + ["noop"]].values).all(axis=1)]

    # Baseline NOOP rate computed before sampling (full filtered data)
    baseline_noop = float(df["noop"].mean())
    n_subjects    = int(df["subject"].nunique())
    print(f"    baseline NOOP rate: {baseline_noop:.3f}  n_subjects={n_subjects}")

    if len(df) > n_sample:
        df = df.sample(n_sample, random_state=random_state)

    # Z-score features
    feat_means, feat_stds = {}, {}
    for f in features:
        mu, sd = df[f].mean(), df[f].std()
        feat_means[f] = mu
        feat_stds[f]  = sd if sd > 1e-9 else 1.0
        df[f"z_{f}"]  = (df[f] - mu) / feat_stds[f]

    z_cols = [f"z_{f}" for f in features]
    formula_rhs = " + ".join(z_cols)

    try:
        model = MixedLM.from_formula(
            f"noop ~ {formula_rhs}",
            groups="subject",
            data=df,
        )
        result = model.fit(reml=False, method="lbfgs")

        rows = []
        for name in result.params.index:
            rows.append({
                "term":    name,
                "coef":    result.params[name],
                "se":      result.bse[name],
                "z":       result.tvalues[name],
                "p":       result.pvalues[name],
                "sig":     stars(result.pvalues[name]),
            })
        return {
            "table":        pd.DataFrame(rows),
            "aic":          result.aic,
            "llf":          result.llf,
            "n":            len(df),
            "n_subjects":   n_subjects,
            "baseline_noop": baseline_noop,
            "feat_means":   feat_means,
            "feat_stds":    feat_stds,
        }
    except Exception as e:
        return {"error": str(e)}


def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ──────────────────────────────────────────────────────────────────────────────
# Cross-subject sign test
# ──────────────────────────────────────────────────────────────────────────────

def cross_subject_sign_test(
    bin_df: pd.DataFrame,
    feature_label: str,
    expected_direction: str = "positive",  # "positive" or "negative"
) -> Dict:
    """
    Test whether subjects agree on the direction of NOOP change
    across feature bins (low bin → high bin).

    For each subject: compute slope of NOOP_rate ~ bin_id (linear regression).
    Sign test: how many subjects have slope in expected direction?
    Exact binomial test (H0: p=0.5).
    """
    sub_cols = [c for c in bin_df.columns if c.startswith("sub_")]
    slopes = []
    for col in sub_cols:
        rates = bin_df[col].dropna().values
        bins  = bin_df.loc[bin_df[col].notna(), "bin_id"].values
        if len(rates) >= 2:
            slope, _, _, _, _ = stats.linregress(bins, rates)
            slopes.append(slope)

    slopes = np.array(slopes)
    n = len(slopes)
    if n == 0:
        return {}

    if expected_direction == "positive":
        n_agree = int((slopes > 0).sum())
    else:
        n_agree = int((slopes < 0).sum())

    binom_p = stats.binomtest(n_agree, n, p=0.5, alternative="greater").pvalue

    return {
        "feature":            feature_label,
        "expected_direction": expected_direction,
        "n_subjects":         n,
        "n_agree":            n_agree,
        "slopes":             slopes.tolist(),
        "mean_slope":         float(slopes.mean()),
        "binom_p":            float(binom_p),
        "sig":                stars(binom_p),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bar":    "#4472C4",
    "sub":    "#AAAAAA",
    "sig":    "#C00000",
}

def plot_binned_noop(
    bin_dfs: Dict[str, pd.DataFrame],
    feature_labels: Dict[str, str],
    game_title: str,
    sign_results: Dict[str, Dict],
    out_path: Path,
) -> None:
    """
    Multi-panel figure: one panel per feature.
    Each panel: bar = subject-mean NOOP rate per bin,
                thin lines = individual subjects,
                annotation = sign-test result.
    """
    features = list(bin_dfs.keys())
    n_feat   = len(features)
    fig, axes = plt.subplots(1, n_feat, figsize=(4.5 * n_feat, 4.2), sharey=False)
    if n_feat == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        bdf = bin_dfs[feat]
        if bdf.empty:
            ax.set_title(f"{feature_labels.get(feat, feat)}\n(no data)")
            continue

        x     = np.arange(len(bdf))
        means = bdf["mean"].values
        ses   = bdf["se"].values

        ax.bar(x, means, color=PALETTE["bar"], alpha=0.8, zorder=2)
        ax.errorbar(x, means, yerr=ses, fmt="none", color="black",
                    capsize=4, lw=1.5, zorder=3)

        # Individual subject lines
        sub_cols = [c for c in bdf.columns if c.startswith("sub_")]
        for col in sub_cols:
            if col in bdf.columns:
                ax.plot(x, bdf[col].values, "-o", color=PALETTE["sub"],
                        alpha=0.45, lw=1.0, ms=3.5, zorder=1)

        # X labels: bin ranges
        bin_labels = [
            f"{row.bin_min:.0f}–{row.bin_max:.0f}"
            for _, row in bdf.iterrows()
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=8, rotation=30, ha="right")
        ax.set_xlabel(feature_labels.get(feat, feat), fontsize=10)
        ax.set_ylabel("NOOP rate" if ax is axes[0] else "", fontsize=10)
        ax.set_title(f"{feature_labels.get(feat, feat)}", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        # Sign-test annotation
        sr = sign_results.get(feat, {})
        if sr:
            txt = (f"n_agree={sr['n_agree']}/{sr['n_subjects']}\n"
                   f"p={sr['binom_p']:.3f} {sr['sig']}")
            ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{game_title}: Game State Feature × NOOP Rate", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_regression_coefs(
    reg_result: Dict,
    feature_labels: Dict[str, str],
    game_title: str,
    out_path: Path,
) -> None:
    """
    Forest plot: effect size expressed as % change in NOOP rate
    relative to baseline (β / baseline_noop × 100), with 95% CI.
    """
    if "error" in reg_result:
        print(f"  [regression error] {reg_result['error']}")
        return

    baseline_noop = reg_result.get("baseline_noop", 1.0)
    n_subjects    = reg_result.get("n_subjects", "?")

    table = reg_result["table"].copy()
    # Keep only feature terms (exclude Intercept and variance components)
    table = table[table["term"].str.startswith("z_")].copy()
    if table.empty:
        return

    table["display"] = table["term"].str.replace("z_", "", regex=False).map(
        lambda x: feature_labels.get(x, x)
    )

    # Convert to % change relative to baseline NOOP rate
    scale = 100.0 / baseline_noop
    table["rel_coef"] = table["coef"] * scale          # % change per 1 SD
    table["rel_ci95"] = 1.96 * table["se"] * scale     # half-width of 95% CI

    n_feat = len(table)
    fig, ax = plt.subplots(figsize=(6.5, 1.8 + 0.9 * n_feat))
    y = np.arange(n_feat)

    for i, (_, row) in enumerate(table.iterrows()):
        color = PALETTE["sig"] if row["p"] < 0.05 else PALETTE["sub"]
        ax.errorbar(row["rel_coef"], i, xerr=row["rel_ci95"],
                    fmt="o", color=color, ms=7, capsize=5, lw=2)
        # annotation: show both relative and absolute β, plus significance
        txt = f"{row['rel_coef']:+.1f}%  (β={row['coef']:.4f}) {row['sig']}"
        offset = row["rel_ci95"] + 0.3
        ax.text(row["rel_coef"] + offset, i, txt, fontsize=8,
                va="center", ha="left")

    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(table["display"].values, fontsize=10)
    ax.set_xlabel(
        f"Change in NOOP rate (% of baseline={baseline_noop:.2f})"
        f" per 1 SD of feature  [95% CI]",
        fontsize=9,
    )
    ax.set_title(
        f"{game_title}: Mixed-Effects LPM\n"
        f"NOOP ~ features + (1|subject)  "
        f"n_steps={reg_result['n']:,}  n_sub={n_subjects}",
        fontsize=10, pad=14,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.6, n_feat - 0.4)
    # add right margin for annotations
    x_max = (table["rel_coef"] + table["rel_ci95"]).max()
    ax.set_xlim(right=x_max * 1.55 if x_max > 0 else x_max * 0.5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# NOOP frame image sampler — visual sanity check
# ──────────────────────────────────────────────────────────────────────────────

def _load_block_images(block_dir: str) -> np.ndarray:
    """Load image array (T, 84, 84, 3) from a block's npz file."""
    npz_path = glob.glob(block_dir + "/*.npz")[0]
    d = np.load(npz_path, allow_pickle=True)
    return d["image"]  # uint8


def sample_and_plot_noop_images(
    df: pd.DataFrame,
    reg_result: Dict,
    features: Dict[str, str],   # feat_name → display_label
    filter_fn,
    game_title: str,
    out_path: Path,
    n_samples: int = 3,
    random_state: int = 42,
) -> None:
    """
    For each feature, select frames from the extreme quartile where the feature
    most drives NOOP (direction = sign of β), filter to NOOP actions, then
    sample n_samples frames and render their RGB images.

    Layout: (n_features rows) × (n_samples cols) = e.g. 3×3 grid.
    To minimise I/O, samples are drawn from the block with the most candidates
    for that feature (one npz load per feature).
    """
    if filter_fn is not None:
        df = df[filter_fn(df)].copy()
    noop_df = df[df["noop"] == 1.0].copy()

    table = reg_result.get("table")
    if table is None:
        print("  [image sample] No regression table — skipping")
        return

    beta_signs: Dict[str, float] = {}
    for _, row in table.iterrows():
        if row["term"].startswith("z_"):
            f = row["term"][2:]
            beta_signs[f] = float(np.sign(row["coef"]))

    feat_list = list(features.keys())
    n_feat    = len(feat_list)

    # ── collect sampled images ──────────────────────────────────────────────
    feat_samples: Dict[str, List] = {}  # feat → [(img, feat_val, sub_id), ...]

    for feat in feat_list:
        sign = beta_signs.get(feat, 1.0)

        # Extreme quartile where feature MOST drives NOOP change
        if sign >= 0:
            # positive β: high feature → high NOOP
            thresh = noop_df[feat].quantile(0.75)
            cands  = noop_df[noop_df[feat] >= thresh].copy()
            direction = "high"
        else:
            # negative β: low feature → high NOOP
            thresh = noop_df[feat].quantile(0.25)
            cands  = noop_df[noop_df[feat] <= thresh].copy()
            direction = "low"

        if len(cands) < n_samples:
            cands = noop_df.dropna(subset=[feat]).copy()

        # Draw from the block with the most candidates (one npz load)
        top_block = cands["block_dir"].value_counts().index[0]
        block_cands = cands[cands["block_dir"] == top_block]

        sampled = block_cands.sample(
            min(n_samples, len(block_cands)),
            random_state=random_state,
            replace=False,
        )

        images = _load_block_images(top_block)
        imgs: List = []
        for _, row in sampled.iterrows():
            step = int(row["step"])
            if step < len(images):
                imgs.append((
                    images[step].copy(),
                    float(row[feat]),
                    int(row["subject"]),
                    direction,
                ))
        feat_samples[feat] = imgs
        print(f"    {feat}: sampled {len(imgs)} NOOP frames "
              f"({direction} quartile, block {Path(top_block).name})")

    # ── plot grid ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_feat, n_samples,
        figsize=(3.0 * n_samples, 3.2 * n_feat),
    )
    # Ensure 2-D axes array
    if n_feat == 1:
        axes = axes[np.newaxis, :]
    if n_samples == 1:
        axes = axes[:, np.newaxis]

    for row_i, feat in enumerate(feat_list):
        imgs      = feat_samples.get(feat, [])
        sign      = beta_signs.get(feat, 1.0)
        direction = "high" if sign >= 0 else "low"
        β_val     = table.loc[table["term"] == f"z_{feat}", "coef"].values
        β_str     = f"β={β_val[0]:+.4f}" if len(β_val) else ""

        for col_i in range(n_samples):
            ax = axes[row_i, col_i]
            if col_i < len(imgs):
                img, feat_val, sub_id, _ = imgs[col_i]
                ax.imshow(img, interpolation="nearest")
                ax.set_title(
                    f"{feat}={feat_val:.0f}  sub{sub_id}",
                    fontsize=7, pad=3,
                )
            else:
                ax.set_facecolor("#DDDDDD")
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9)
            ax.axis("off")

        # Row label on the left of the first column
        axes[row_i, 0].text(
            -0.12, 0.5,
            f"{features[feat]}\n({direction} quartile)\n{β_str}",
            transform=axes[row_i, 0].transAxes,
            fontsize=7.5, rotation=0, va="center", ha="right",
            color="#222222",
        )

    fig.suptitle(
        f"{game_title}: NOOP frames at extreme feature quartile\n"
        f"(quartile where feature most increases NOOP probability)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.subplots_adjust(left=0.22)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Method 2: Subject-level aggregated OLS (conservative, n=6)
# ──────────────────────────────────────────────────────────────────────────────

def run_episode_level_regression(
    df: pd.DataFrame,
    features: List[str],
    filter_fn=None,
) -> Tuple[Dict, float, int, int]:
    """
    Aggregate to episode-level means, then within-subject demean both x and y
    (subtract each subject's mean) before running OLS.

    This removes between-subject confounds (play-style differences) so the
    slope estimates within-subject effects — consistent with Mixed LPM β.

    x-axis in plot: feature value relative to subject mean (demeaned units)
    y-axis in plot: NOOP rate relative to subject mean (demeaned %)

    Returns (per-feature results dict, baseline_noop, n_subjects, n_episodes).
    """
    if filter_fn is not None:
        df = df[filter_fn(df)].copy()
    df = df.dropna(subset=features + ["noop", "subject"]).copy()
    df = df[np.isfinite(df[features + ["noop"]].values).all(axis=1)].copy()

    # Assign episode IDs within each (subject, block_dir)
    df["_ep_id"] = (
        df.groupby(["subject", "block_dir"])["is_first"]
        .transform(lambda s: s.cumsum())
    )

    ep_df = (
        df.groupby(["subject", "block_dir", "_ep_id"])[features + ["noop"]]
        .mean()
        .reset_index()
    )
    baseline_noop = float(ep_df["noop"].mean())
    n_subjects    = int(ep_df["subject"].nunique())
    n_episodes    = len(ep_df)
    print(f"    episode-level (demeaned) n={n_episodes} episodes  n_subjects={n_subjects}  baseline NOOP={baseline_noop:.3f}")

    # Within-subject demeaning
    for col in features + ["noop"]:
        sub_means = ep_df.groupby("subject")[col].transform("mean")
        ep_df[f"_dm_{col}"] = ep_df[col] - sub_means

    results = {}
    for f in features:
        x = ep_df[f"_dm_{f}"].values.astype(float)
        y = ep_df["_dm_noop"].values.astype(float)
        slope, intercept, r, p, se = stats.linregress(x, y)
        results[f] = {
            "slope":     float(slope),
            "intercept": float(intercept),
            "r":         float(r),
            "p":         float(p),
            "se":        float(se),
            "ep_x":      x,
            "ep_y":      y,
            "sub_ids":   ep_df["subject"].values,
        }
    return results, baseline_noop, n_subjects, n_episodes


def _ols_ci_band(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_pred: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_pred, ci_lower, ci_upper) for OLS fit at x_pred."""
    n = len(x_data)
    slope, intercept, *_ = stats.linregress(x_data, y_data)
    y_fit  = intercept + slope * x_data
    mse    = np.sum((y_data - y_fit) ** 2) / max(n - 2, 1)
    x_mean = x_data.mean()
    ss_x   = np.sum((x_data - x_mean) ** 2) or 1.0
    y_pred = intercept + slope * x_pred
    se_fit = np.sqrt(mse * (1 / n + (x_pred - x_mean) ** 2 / ss_x))
    t_crit = stats.t.ppf(1 - alpha / 2, df=max(n - 2, 1))
    return y_pred, y_pred - t_crit * se_fit, y_pred + t_crit * se_fit


def plot_episode_regression(
    reg_results: Dict,
    feature_labels: Dict[str, str],
    game_title: str,
    baseline_noop: float,
    n_subjects: int,
    n_episodes: int,
    out_path: Path,
) -> None:
    """
    One panel per feature: episode-level scatter (colour = subject) +
    OLS regression line + 95% CI band.
    y-axis: NOOP rate (%).  Annotation: β, r, p (df = n_episodes - 2).
    """
    features  = list(reg_results.keys())
    n_feat    = len(features)
    sub_colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, n_feat, figsize=(4.5 * n_feat, 4.2))
    if n_feat == 1:
        axes = [axes]

    # Build subject → color mapping (consistent across panels)
    all_subs = sorted(set(
        sid for res in reg_results.values() for sid in res["sub_ids"]
    ))
    sub_color_map = {s: sub_colors[i % len(sub_colors)] for i, s in enumerate(all_subs)}

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=sub_color_map[s], markersize=7,
                   label=f"sub{int(s)}")
        for s in all_subs
    ]

    for ax, feat in zip(axes, features):
        res = reg_results[feat]
        x   = res["ep_x"]
        y   = res["ep_y"] * 100  # → %

        x_line = np.linspace(x.min(), x.max(), 200)
        y_line, ci_lo, ci_hi = _ols_ci_band(x, res["ep_y"], x_line)
        y_line *= 100; ci_lo *= 100; ci_hi *= 100

        ax.fill_between(x_line, ci_lo, ci_hi, alpha=0.15, color=PALETTE["bar"])
        ax.plot(x_line, y_line, color=PALETTE["bar"], lw=2)

        for xi, yi, sid in zip(x, y, res["sub_ids"]):
            ax.scatter(xi, yi, color=sub_color_map[sid], s=30, alpha=0.7,
                       zorder=4, edgecolors="none")

        color = PALETTE["sig"] if res["p"] < 0.05 else PALETTE["sub"]
        txt = (f"β={res['slope']:.4f}\n"
               f"r={res['r']:.2f}\n"
               f"p={res['p']:.3f} {stars(res['p'])}")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=1)
        ax.axvline(0, color="grey", lw=0.8, ls="--", zorder=1)
        ax.set_xlabel(f"Δ {feature_labels.get(feat, feat)}\n(episode − subject mean)", fontsize=9)
        ax.set_ylabel("Δ NOOP rate\n(episode − subject mean)" if ax is axes[0] else "", fontsize=9)
        ax.set_title(feature_labels.get(feat, feat), fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].legend(handles=legend_handles, title="Subject",
                    fontsize=8, title_fontsize=8,
                    bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    fig.suptitle(
        f"{game_title}: Within-subject Episode OLS (demeaned)  "
        f"(n={n_episodes} episodes, {n_subjects} subjects, df={n_episodes - 2})\n"
        f"baseline NOOP = {baseline_noop:.3f}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Method 3: Bayesian mixed-effects LPM (bambi)
# ──────────────────────────────────────────────────────────────────────────────

def run_bayesian_lpm(
    df: pd.DataFrame,
    features: List[str],
    filter_fn=None,
    n_sample: int = 10_000,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    random_state: int = 42,
) -> Dict:
    """
    Bayesian LPM: noop ~ z(features) + (1|subject)  via bambi (gaussian family).
    Returns dict with arviz InferenceData + metadata.
    """
    import bambi as bmb  # local import — optional dependency

    if filter_fn is not None:
        df = df[filter_fn(df)].copy()
    df = df.dropna(subset=features + ["noop", "subject"]).copy()
    df = df[np.isfinite(df[features + ["noop"]].values).all(axis=1)]

    baseline_noop = float(df["noop"].mean())
    n_subjects    = int(df["subject"].nunique())
    print(f"    Bayesian baseline NOOP={baseline_noop:.3f}  n_subjects={n_subjects}")

    if len(df) > n_sample:
        df = df.sample(n_sample, random_state=random_state)

    for f in features:
        mu, sd = df[f].mean(), df[f].std()
        df[f"z_{f}"] = (df[f] - mu) / (sd if sd > 1e-9 else 1.0)

    z_cols  = [f"z_{f}" for f in features]
    formula = "noop ~ " + " + ".join(z_cols) + " + (1|subject)"

    try:
        model = bmb.Model(formula, df, family="gaussian")
        idata = model.fit(
            draws=draws, tune=tune, chains=chains,
            random_seed=random_state, progressbar=False,
            target_accept=0.9,
        )
        return {
            "idata":         idata,
            "baseline_noop": baseline_noop,
            "n_subjects":    n_subjects,
            "n":             len(df),
            "features":      features,
        }
    except Exception as e:
        return {"error": str(e)}


def plot_bayesian_coefs(
    bayes_result: Dict,
    feature_labels: Dict[str, str],
    game_title: str,
    out_path: Path,
) -> None:
    """
    Forest plot: Bayesian posterior % change in NOOP relative to baseline.
    Shows posterior mean + 95% credible interval (equal-tailed percentiles).
    Red = CI excludes zero; grey = CI includes zero.
    """
    if "error" in bayes_result:
        print(f"  [Bayesian error] {bayes_result['error']}")
        return

    idata         = bayes_result["idata"]
    features      = bayes_result["features"]
    baseline_noop = bayes_result["baseline_noop"]
    scale         = 100.0 / baseline_noop
    posterior     = idata.posterior

    rows = []
    for f in features:
        z_name = f"z_{f}"
        if z_name not in posterior:
            print(f"  [Bayesian] '{z_name}' not found in posterior; skipping")
            continue
        samples = posterior[z_name].values.flatten().astype(float)
        mean    = float(samples.mean()) * scale
        lo, hi  = np.percentile(samples, [2.5, 97.5])
        lo *= scale; hi *= scale
        excludes_zero = (lo > 0) or (hi < 0)
        rows.append({
            "display":       feature_labels.get(f, f),
            "mean":          mean,
            "lo":            lo,
            "hi":            hi,
            "excludes_zero": excludes_zero,
        })

    if not rows:
        print("  [Bayesian] No rows to plot")
        return

    n_feat = len(rows)
    fig, ax = plt.subplots(figsize=(6.5, 1.8 + 0.9 * n_feat))
    y = np.arange(n_feat)

    for i, row in enumerate(rows):
        color  = PALETTE["sig"] if row["excludes_zero"] else PALETTE["sub"]
        lo_err = row["mean"] - row["lo"]
        hi_err = row["hi"]  - row["mean"]
        ax.errorbar(row["mean"], i, xerr=[[lo_err], [hi_err]],
                    fmt="o", color=color, ms=7, capsize=5, lw=2)
        sig_lbl = "*" if row["excludes_zero"] else "ns"
        txt = (f"{row['mean']:+.1f}%  "
               f"[95% CI: {row['lo']:.1f}, {row['hi']:.1f}] {sig_lbl}")
        ax.text(row["hi"] + 0.3, i, txt, fontsize=8, va="center", ha="left")

    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r["display"] for r in rows], fontsize=10)
    ax.set_xlabel(
        f"Change in NOOP rate (% of baseline={baseline_noop:.2f})"
        f" per 1 SD of feature  [95% posterior CI]",
        fontsize=9,
    )
    ax.set_title(
        f"{game_title}: Bayesian Mixed-Effects LPM (bambi)\n"
        f"NOOP ~ features + (1|subject)  "
        f"n_steps={bayes_result['n']:,}  n_sub={bayes_result['n_subjects']}",
        fontsize=10, pad=14,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.6, n_feat - 0.4)
    hi_vals = [r["hi"] for r in rows]
    ax.set_xlim(right=max(hi_vals) * 2.2 if max(hi_vals) > 0 else 5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def analyze_pong(df: pd.DataFrame) -> None:
    print("\n── Pong ──")

    # Only use frames where ball is in play (ball_x > 0 and ball_y > 0)
    in_play_filter = lambda d: (d["ball_x"] > 0) & (d["ball_y"] > 0)

    features = {
        "paddle_dist": "Paddle distance (Euclidean, ball↔player)",
        "score_diff":  "Score diff (player − CPU)",
    }

    # Expected directions:
    #   paddle_dist: large dist = ball far from paddle → NOOP ↑ (positive)
    #   score_diff:  winning (positive) → may relax → NOOP ↑ (positive)
    #                losing (negative) → more urgent → NOOP ↓
    expected_dirs = {
        "paddle_dist": "positive",   # farther ball → more NOOP
        "score_diff":  "positive",   # winning → more relaxed → more NOOP
    }

    bin_dfs     = {}
    sign_results = {}
    for feat, label in features.items():
        bdf = binned_noop_rate(df, feat, n_bins=N_BINS, filter_fn=in_play_filter)
        bin_dfs[feat]      = bdf
        sign_results[feat] = cross_subject_sign_test(
            bdf, label, expected_direction=expected_dirs[feat]
        )
        if sign_results[feat]:
            sr = sign_results[feat]
            print(f"  {label}: n_agree={sr['n_agree']}/{sr['n_subjects']} "
                  f"(mean slope={sr['mean_slope']:.4f}) binom_p={sr['binom_p']:.3f} {sr['sig']}")

    # Regression (replaces binned figure)
    print("  Running mixed-effects LPM for Pong...")
    reg = run_mixed_effects_lpm(
        df, list(features.keys()),
        filter_fn=in_play_filter,
    )
    if "table" in reg:
        print(reg["table"].to_string(index=False))
        reg["table"].to_csv(RES_DIR / "pong_regression.csv", index=False)

    plot_regression_coefs(reg, features, "Pong", FIG_DIR / "fig_pong_regression.png")

    # NOOP frame image sanity check
    print("  Sampling NOOP frame images for Pong...")
    sample_and_plot_noop_images(
        df, reg, features, in_play_filter,
        "Pong", FIG_DIR / "fig_pong_noop_samples.png",
    )

    # Episode-level OLS
    print("  Running episode-level OLS for Pong...")
    ep_reg, ep_baseline, ep_n_sub, ep_n_ep = run_episode_level_regression(
        df, list(features.keys()), filter_fn=in_play_filter,
    )
    plot_episode_regression(
        ep_reg, features, "Pong", ep_baseline, ep_n_sub, ep_n_ep,
        FIG_DIR / "fig_pong_subject_regression.png",
    )

    # Bayesian mixed-effects LPM
    print("  Running Bayesian LPM for Pong (this may take a minute)...")
    bayes = run_bayesian_lpm(df, list(features.keys()), filter_fn=in_play_filter)
    plot_bayesian_coefs(bayes, features, "Pong", FIG_DIR / "fig_pong_bayesian.png")

    # Sign test summary
    sign_df = pd.DataFrame([v for v in sign_results.values() if v])
    if not sign_df.empty:
        sign_df.to_csv(RES_DIR / "pong_sign_test.csv", index=False)

    return bin_dfs, sign_results, reg


def plot_playerx_noop_distribution(df: pd.DataFrame, filter_fn, out_path) -> None:
    """Per-subject histogram of player_x positions where action == NOOP.

    x-axis : player_x value (raw RAM units, range ~35–117)
    y-axis : number of NOOP frames at that position
    Each subject plotted as a separate colour; legend identifies subjects.
    """
    import matplotlib.pyplot as plt

    fdf = df[filter_fn(df)].copy() if filter_fn is not None else df.copy()
    noop_df = fdf[fdf["noop"] == 1.0]

    subjects = sorted(noop_df["subject"].unique())
    colors   = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8, 5))

    from scipy.stats import gaussian_kde

    x_min = noop_df["player_x"].min() - 5
    x_max = noop_df["player_x"].max() + 5
    x_grid = np.linspace(x_min, x_max, 500)

    for i, subj in enumerate(subjects):
        sub_x = noop_df.loc[noop_df["subject"] == subj, "player_x"].values
        if len(sub_x) < 2:
            continue
        kde = gaussian_kde(sub_x, bw_method="scott")
        density = kde(x_grid)
        ax.plot(x_grid, density, color=colors[i % len(colors)],
                lw=2, label=f"sub{subj}")
        ax.fill_between(x_grid, density, alpha=0.15, color=colors[i % len(colors)])

    ax.axvline(noop_df["player_x"].median(), color="black", lw=1.2,
               linestyle="--", label="overall median")
    ax.set_xlabel("Player X position (RAM units)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Space Invaders: Player X distribution during NOOP\n(per subject)", fontsize=13)
    ax.legend(title="Subject", fontsize=9, title_fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def analyze_space_invaders(df: pd.DataFrame) -> None:
    print("\n── SpaceInvaders ──")

    # Exclude frames where enemy_count == 0 (wave cleared / inter-wave reset)
    in_wave_filter = lambda d: d["enemy_count"] > 0

    features = {
        "enemy_count": "Enemy count (remaining)",
        "enemies_y":   "Enemies Y descent (threat)",
        "player_x":    "Player X position",
        "num_lives":   "Lives remaining",
    }

    # Hypothesised directions
    #   enemy_count: more remaining → uncertain which to target → NOOP ↑ (positive)
    #   enemies_y:   deeper descent → more urgent → NOOP ↓ (negative)
    #   player_x:    exploratory; no strong prior
    #   num_lives:   more lives → relaxed → NOOP ↑ (positive)
    expected_dirs = {
        "enemy_count": "positive",
        "enemies_y":   "negative",
        "player_x":    "positive",
        "num_lives":   "positive",
    }

    bin_dfs      = {}
    sign_results = {}
    for feat, label in features.items():
        bdf = binned_noop_rate(df, feat, n_bins=N_BINS, filter_fn=in_wave_filter)
        bin_dfs[feat]      = bdf
        sign_results[feat] = cross_subject_sign_test(
            bdf, label, expected_direction=expected_dirs[feat]
        )
        if sign_results[feat]:
            sr = sign_results[feat]
            print(f"  {label}: n_agree={sr['n_agree']}/{sr['n_subjects']} "
                  f"(mean slope={sr['mean_slope']:.4f}) binom_p={sr['binom_p']:.3f} {sr['sig']}")

    # Regression (replaces binned figure)
    print("  Running mixed-effects LPM for SpaceInvaders...")
    reg = run_mixed_effects_lpm(
        df, list(features.keys()),
        filter_fn=in_wave_filter,
    )
    if "table" in reg:
        print(reg["table"].to_string(index=False))
        reg["table"].to_csv(RES_DIR / "si_regression.csv", index=False)

    plot_regression_coefs(reg, features, "Space Invaders",
                          FIG_DIR / "fig_si_regression.png")

    # NOOP frame image sanity check
    print("  Sampling NOOP frame images for SpaceInvaders...")
    sample_and_plot_noop_images(
        df, reg, features, in_wave_filter,
        "Space Invaders", FIG_DIR / "fig_si_noop_samples.png",
    )

    # Episode-level OLS
    print("  Running episode-level OLS for SpaceInvaders...")
    ep_reg, ep_baseline, ep_n_sub, ep_n_ep = run_episode_level_regression(
        df, list(features.keys()), filter_fn=in_wave_filter,
    )
    plot_episode_regression(
        ep_reg, features, "Space Invaders", ep_baseline, ep_n_sub, ep_n_ep,
        FIG_DIR / "fig_si_subject_regression.png",
    )

    # Bayesian mixed-effects LPM
    print("  Running Bayesian LPM for SpaceInvaders (this may take a minute)...")
    bayes = run_bayesian_lpm(df, list(features.keys()), filter_fn=in_wave_filter)
    plot_bayesian_coefs(bayes, features, "Space Invaders",
                        FIG_DIR / "fig_si_bayesian.png")

    # Player-X positional bias: NOOP count distribution per subject
    print("  Plotting player_x NOOP distribution per subject...")
    plot_playerx_noop_distribution(df, in_wave_filter, FIG_DIR / "fig_si_playerx_noop_dist.png")

    sign_df = pd.DataFrame([v for v in sign_results.values() if v])
    if not sign_df.empty:
        sign_df.to_csv(RES_DIR / "si_sign_test.csv", index=False)

    return bin_dfs, sign_results, reg


def write_summary(pong_reg, si_reg, out_path: Path) -> None:
    lines = ["=" * 60, "Analysis 3: Game State Feature × NOOP — Summary", "=" * 60]

    for game, reg in [("Pong", pong_reg), ("SpaceInvaders", si_reg)]:
        lines.append(f"\n{game} Mixed-Effects LPM (n={reg.get('n','?'):,})")
        if "table" in reg:
            for _, row in reg["table"].iterrows():
                if row["term"] == "Intercept":
                    continue
                feat = row["term"].replace("z_", "")
                lines.append(
                    f"  {feat:<20s}  β={row['coef']:+.4f}  "
                    f"SE={row['se']:.4f}  z={row['z']:+.2f}  p={row['p']:.4f}  {row['sig']}"
                )
        elif "error" in reg:
            lines.append(f"  [error] {reg['error']}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n  Summary written → {out_path.name}")


def main() -> None:
    print("Loading Pong data (game_1)...")
    pong_df = load_all_blocks(game_id=1)
    print(f"  Loaded {len(pong_df):,} steps from {pong_df['subject'].nunique()} subjects")

    print("Loading SpaceInvaders data (game_2)...")
    si_df = load_all_blocks(game_id=2)
    print(f"  Loaded {len(si_df):,} steps from {si_df['subject'].nunique()} subjects")

    # Save raw feature distributions for inspection
    pong_df.groupby("subject")[["ball_x","ball_y","paddle_dist","noop"]].describe().to_csv(
        RES_DIR / "pong_feature_stats.csv"
    )
    si_df.groupby("subject")[["enemy_count","enemies_y","player_x","num_lives","noop"]].describe().to_csv(
        RES_DIR / "si_feature_stats.csv"
    )

    _, _, pong_reg = analyze_pong(pong_df)
    _, _, si_reg   = analyze_space_invaders(si_df)

    write_summary(pong_reg, si_reg, RES_DIR / "summary.txt")
    print("\nDone. All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
