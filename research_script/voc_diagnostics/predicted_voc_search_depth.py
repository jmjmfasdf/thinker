"""Relate recorded pre-search EMA VOC to subsequently executed search depth.

Read-only analysis of training logs, not a reward-utility evaluation. No model
inference, checkpoint changes, or job submissions. The primary window is fixed
at 9--10M collector environment actions; 8--10M and 9.5--10M are sensitivity
windows. One complete search stage is the primary observational unit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr


GAMES = [("pong", "Pong", "#0072B2"), ("enduro", "Enduro", "#D55E00"),
         ("spaceinvaders", "Space Invaders", "#009E73")]


def rho(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def interval(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return np.quantile(a, [.025, .975]).tolist() if len(a) else [None, None]


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct(path, game):
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line]
    groups = defaultdict(list)
    identities = set()
    for r in records:
        identity = (r["attempt"], r["decision"], r["stream"], r["count"])
        assert identity not in identities, "Duplicate event identity"
        identities.add(identity)
        groups[identity[:3]].append(r)
    complete, stages, excluded = [], {}, Counter()
    for key, seq in groups.items():
        assert [r["count"] for r in seq] == list(range(len(seq))), "Missing search events"
        assert all(r["valid"] and not r["forced"] for r in seq)
        assert not any(r["control"] == 2 for r in seq[:-1])
        if seq[-1]["control"] != 2:
            # The cap can end a stage without a separately logged STOP.
            # Do not impute an unobserved final depth.
            excluded["no_observed_stop"] += 1
            continue
        budget = sum(r["control"] in [0, 1] for r in seq)
        assert budget == seq[-1]["count"]
        assert all(0 <= r["depth"] <= r["count"] for r in seq)
        first = seq[0]
        row = dict(game=game, attempt=key[0], decision=key[1], stream=key[2],
                   real_step=first["real_step"], initial_voc=first["ema_delta_q"],
                   initial_online_voc=first["online_delta_q"],
                   initial_continue_probability=first["p_continue"],
                   total_computations=budget, max_depth=max(r["depth"] for r in seq),
                   reset_count=sum(r["control"] == 1 for r in seq),
                   started=int(budget > 0))
        complete.append(row)
        stages[key] = seq
    probability_error = max(abs(r["p_continue"] - (.01 + .98 * expit(r["ema_delta_q"]/.05)))
                            for r in records)
    assert probability_error < 1e-6
    audit = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                 events=len(records), stages=len(groups), complete_stages=len(complete),
                 excluded=dict(excluded), gate_probability_max_error=probability_error)
    return complete, stages, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--refresh", action="store_true", help="Refresh this diagnostic's existing output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.refresh)
    summary, signed, binned, robustness, local, primary, sources = [], [], [], [], [], [], []
    plot_data = {}
    for gi, (game, label, color) in enumerate(GAMES):
        path = args.campaign / "cells" / f"v10p-20260925-{game}_s701" / "control_samples.jsonl"
        all_rows, stages, audit = reconstruct(path, game)
        sources.append(audit)
        for start in [8_000_000, 9_000_000, 9_500_000]:
            selected = [r for r in all_rows if start <= r["real_step"] < 10_000_000]
            q = np.array([r["initial_voc"] for r in selected])
            d = np.array([r["max_depth"] for r in selected])
            n = np.array([r["total_computations"] for r in selected])
            active = n > 0
            robustness.append(dict(game=game, start=start, end=10_000_000,
                n=len(q), n_started=int(active.sum()), rho_depth=rho(q, d),
                rho_computations=rho(q, n), rho_depth_started=rho(q[active], d[active])))
        rows = sorted([r for r in all_rows if 9_000_000 <= r["real_step"] < 10_000_000],
                      key=lambda r: r["real_step"])
        primary.extend(rows)
        q = np.array([r["initial_voc"] for r in rows])
        d = np.array([r["max_depth"] for r in rows])
        active = np.array([r["started"] for r in rows], dtype=bool)
        # Non-overlapping time blocks retain local dependence. These intervals
        # describe this one training trajectory, not training-seed variability.
        blocks = [np.arange(i, min(i+16, len(rows))) for i in range(0, len(rows), 16)]
        rng = np.random.default_rng(29092026 + gi)
        draws = [np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
                 for _ in range(args.bootstrap)]
        ci_all = interval([rho(q[ix], d[ix]) for ix in draws])
        ci_started = interval([rho(q[ix][active[ix]], d[ix][active[ix]]) for ix in draws])
        summary.append(dict(game=game, n=len(rows), n_started=int(active.sum()),
            rho_depth=rho(q, d), rho_depth_ci_low=ci_all[0], rho_depth_ci_high=ci_all[1],
            rho_depth_started=rho(q[active], d[active]),
            rho_depth_started_ci_low=ci_started[0], rho_depth_started_ci_high=ci_started[1],
            observed_resets=sum(r["reset_count"] for r in rows),
            depth_equals_computation_count=all(r["max_depth"] == r["total_computations"] for r in rows)))
        for sign, mask in [("negative", q < -1e-6), ("positive", q > 1e-6)]:
            signed.append(dict(game=game, voc_sign=sign, n=int(mask.sum()),
                mean_voc=float(q[mask].mean()), mean_depth=float(d[mask].mean()),
                search_start_fraction=float(active[mask].mean()),
                n_started=int((active & mask).sum()),
                mean_depth_if_started=float(d[active & mask].mean())))
        for population, keep in [("all", np.ones(len(rows), dtype=bool)), ("started", active)]:
            membership = np.full(len(rows), -1, dtype=int)
            ordered = np.flatnonzero(keep)[np.argsort(q[keep], kind="stable")]
            for k, ids in enumerate(np.array_split(ordered, 5)):
                membership[ids] = k
            data = []
            for k in range(5):
                ids = np.flatnonzero(membership == k)
                ci = interval([d[ix[membership[ix] == k]].mean()
                               if np.any(membership[ix] == k) else np.nan for ix in draws])
                point = dict(game=game, population=population, quintile=k+1, n=len(ids),
                    mean_voc=float(q[ids].mean()), min_voc=float(q[ids].min()), max_voc=float(q[ids].max()),
                    mean_depth=float(d[ids].mean()), ci_low=ci[0], ci_high=ci[1])
                binned.append(point)
                data.append(point)
            plot_data[game, population] = data
        # Match the current calculation count so accumulated search is never
        # used as the outcome of a VOC estimate computed afterwards.
        for count in range(4):
            entries = []
            for r in rows:
                seq = stages[r["attempt"], r["decision"], r["stream"]]
                if len(seq) <= count:
                    continue
                event = seq[count]
                entries.append((event["ema_delta_q"], r["total_computations"]-count,
                                max(s["depth"] for s in seq[count:])-event["depth"],
                                event["control"] in [0, 1]))
            e = np.asarray(entries, dtype=float)
            local.append(dict(game=game, current_count=count, n=len(e),
                rho_voc_future_computations=rho(e[:, 0], e[:, 1]),
                rho_voc_future_depth_gain=rho(e[:, 0], e[:, 2]),
                mean_remaining_computations=float(e[:, 1].mean())))
    for name, data in [("summary", summary), ("voc_sign", signed), ("depth_curves", binned),
                       ("window_sensitivity", robustness), ("matched_count", local), ("decisions", primary)]:
        csv_write(args.output / f"{name}.csv", data)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7), sharey="row")
    fig.subplots_adjust(left=.075, right=.985, top=.84, bottom=.15, hspace=.48, wspace=.23)
    for col, (game, label, color) in enumerate(GAMES):
        for row, population in enumerate(["all", "started"]):
            ax = axes[row, col]
            data = plot_data[game, population]
            x = np.array([r["mean_voc"] for r in data])
            y = np.array([r["mean_depth"] for r in data])
            lo = np.array([r["ci_low"] for r in data])
            hi = np.array([r["ci_high"] for r in data])
            ax.fill_between(x, lo, hi, color=color, alpha=.15)
            ax.plot(x, y, "o-", color=color, lw=1.8, markersize=5)
            ax.axvline(0, color="#777777", ls=":", lw=.8)
            ax.grid(axis="y", alpha=.15)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xlabel("Predicted VoC before search (EMA)")
            rr = summary[col]
            n = rr["n"] if row == 0 else rr["n_started"]
            corr = rr["rho_depth"] if row == 0 else rr["rho_depth_started"]
            ax.set_title(f"{label}\nn = {n} | Spearman rho = {corr:.2f}", fontsize=10)
    for row, population in enumerate(["all", "started"]):
        upper = max(r["ci_high"] for r in binned if r["population"] == population) * 1.08
        axes[row, 0].set_ylim(0, upper)
        assert all(r["ci_high"] < upper for r in binned if r["population"] == population)
    axes[0, 0].set_ylabel("Maximum search depth\nAll decisions, including zero search")
    axes[1, 0].set_ylabel("Maximum search depth\nDecisions that started search")
    fig.suptitle("Does predicted VoC track executed planning depth?", x=.075, ha="left", y=.975,
                 fontsize=16, fontweight="bold")
    fig.text(.075, .91, "Temperature 0.05 | Training steps 9–10M | One sub001-adapted agent per game", fontsize=11)
    fig.text(.075, .065, "Points: within-game VoC quintiles; bands: 95% time-block bootstrap intervals (16 logged decisions per block).", fontsize=9)
    fig.text(.075, .035, "Stream 0, every 128 decisions. No RESET observed in this window: depth equals computation count.", fontsize=9)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(args.output / f"predicted_voc_search_depth.{ext}", dpi=180, facecolor="white")
    plt.close(fig)
    manifest = dict(temperature=.05, epsilon=.02, sources=sources,
        predictor="EMA Q(CONTINUE)-Q(STOP), recorded before search",
        outcome="Maximum observed rollout depth before the recorded STOP; complete stages only",
        primary_window=[9_000_000, 10_000_000], source_kind="Training rollout, not frozen-checkpoint evaluation or human replay",
        uncertainty=f"{args.bootstrap} resamples of 16-consecutive-observation time blocks; conditional on one training trajectory",
        scope="Descriptive model compute allocation, not realized utility, human behavior, or a causal intervention",
        conditional_analysis="Conditioning on search having started is descriptive, not a causal adjustment",
        gate_formula="p_continue = epsilon/2 + (1-epsilon)*sigmoid(EMA_VoC/temperature)",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(summary=summary, sign=signed, matched_count=local), indent=2))


if __name__ == "__main__":
    main()
