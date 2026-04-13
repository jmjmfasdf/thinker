"""
1-5: NOOP Bout Survival Analysis
- KM survival curve per subject (6 subplots)
- Pong vs SpaceInvaders as separate lines
- Exponential baseline (random omission null hypothesis)
- Data: behavioral_data_block_old (human only)
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from pathlib import Path

BASE_DIR = Path("/home/jmme425/thinker/behavioral_data_block_old")
OUT_DIR = Path("/home/jmme425/thinker/research_script/outputs/behavioral_analysis/figures")

# game_1=Pong, game_2=SpaceInvaders
GAME_MAP = {1: "Pong", 2: "SpaceInvaders"}
COLORS = {"Pong": "#1f77b4", "SpaceInvaders": "#ff7f0e"}


# ── 1. Bout extraction ────────────────────────────────────────────────────────

def extract_bouts(action_seq, is_first, is_terminal):
    """
    Returns list of (length, censored).
    censored=True: bout cut off by episode boundary.
    """
    bouts = []
    in_bout = False
    bout_len = 0

    for t in range(len(action_seq)):
        # episode boundary: close any open bout as censored
        if is_first[t] and t > 0 and in_bout:
            bouts.append((bout_len, True))
            in_bout = False
            bout_len = 0

        if action_seq[t] == 0:  # NOOP
            if not in_bout:
                in_bout = True
                bout_len = 1
            else:
                bout_len += 1
        else:
            if in_bout:
                bouts.append((bout_len, False))  # natural termination
            in_bout = False
            bout_len = 0

        if is_terminal[t] and in_bout:
            bouts.append((bout_len, True))  # censored at episode end
            in_bout = False
            bout_len = 0

    return bouts


def load_bouts_for_subject_game(sub_id, game_id):
    pattern = str(BASE_DIR / f"sub_{sub_id}" / f"game_{game_id}" / "day_*" / "block_*" / "*.npz")
    bouts = []
    for f in sorted(glob.glob(pattern)):
        d = np.load(f, allow_pickle=True)
        action = np.argmax(d['action'], axis=1)
        bouts.extend(extract_bouts(action, d['is_first'], d['is_terminal']))
    return bouts  # list of (length, censored)


# ── 2. KM estimator ───────────────────────────────────────────────────────────

def kaplan_meier(bouts, t_max=None):
    """
    Standard KM estimator with right-censoring.
    Returns (times, survival_prob).
    """
    lengths = np.array([b[0] for b in bouts])
    events  = np.array([not b[1] for b in bouts], dtype=bool)  # True = event occurred

    if t_max is None:
        t_max = int(np.percentile(lengths, 99))  # trim extreme tail for display

    times = np.arange(1, t_max + 1)
    S = np.ones(len(times))
    s = 1.0

    for i, t in enumerate(times):
        n_at_risk = np.sum(lengths >= t)
        n_events  = np.sum((lengths == t) & events)
        if n_at_risk > 0:
            s *= (1 - n_events / n_at_risk)
        S[i] = s

    return times, S


def exponential_baseline(bouts, times):
    """S(t) = exp(-t / mean_length) using only uncensored bouts for mean."""
    uncensored = [b[0] for b in bouts if not b[1]]
    if not uncensored:
        uncensored = [b[0] for b in bouts]
    mu = np.mean(uncensored)
    return np.exp(-times / mu)


# ── 3. Figure ─────────────────────────────────────────────────────────────────

def plot_survival(t_max_pong=150, t_max_si=80):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, sub in enumerate(range(1, 7)):
        ax = axes[idx]

        for game_id, game_name in GAME_MAP.items():
            bouts = load_bouts_for_subject_game(sub, game_id)
            if not bouts:
                continue

            t_max = t_max_pong if game_name == "Pong" else t_max_si
            times, S = kaplan_meier(bouts, t_max=t_max)
            S_exp = exponential_baseline(bouts, times)

            color = COLORS[game_name]
            ax.plot(times, S, color=color, lw=2, label=game_name)
            ax.plot(times, S_exp, color=color, lw=1.2, ls='--', alpha=0.5)

            n_bouts = len(bouts)
            n_censored = sum(b[1] for b in bouts)
            pct_censored = 100 * n_censored / n_bouts if n_bouts > 0 else 0

            # annotate censoring rate in legend
            ax.plot([], [], color=color, lw=1.2, ls='--', alpha=0.5,
                    label=f'{game_name} exp. baseline')

        ax.set_title(f'S{sub}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Bout length (steps)', fontsize=9)
        ax.set_ylabel('P(surviving)', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)

    # shared legend note
    fig.text(0.5, 0.01,
             'Solid: KM estimate  |  Dashed: Exponential baseline (random omission null)',
             ha='center', fontsize=9, color='gray')

    fig.suptitle('1-5: NOOP Bout Survival Function by Subject\n'
                 'Pong vs SpaceInvaders — deviation from exponential = planned delay',
                 fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = OUT_DIR / "fig_1-5_survival_by_subject.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


# ── 4. Summary stats ──────────────────────────────────────────────────────────

def print_summary():
    print(f"{'Sub':>4} {'Game':>14} {'N_bouts':>8} {'Censored%':>10} "
          f"{'Mean_len':>9} {'Median_len':>11}")
    print("-" * 60)
    for sub in range(1, 7):
        for game_id, game_name in GAME_MAP.items():
            bouts = load_bouts_for_subject_game(sub, game_id)
            if not bouts:
                continue
            lengths = np.array([b[0] for b in bouts])
            n_cens  = sum(b[1] for b in bouts)
            print(f"{sub:>4} {game_name:>14} {len(bouts):>8} "
                  f"{100*n_cens/len(bouts):>9.1f}% "
                  f"{np.mean(lengths):>9.1f} {np.median(lengths):>11.1f}")


if __name__ == "__main__":
    print_summary()
    plot_survival()
