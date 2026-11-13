"""Read-only multiview analysis of the sub001 tau=.01 connected-GRU pilot.

Uses validated snapshots and paired evaluations; never launches evaluation jobs.
Inference is conditional on one trained seed, with environment-stream bootstrap
for interventions and time-block bootstrap for sampled native training decisions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1] / "outputs"
GAMES = [("pong", "Pong", "#0072B2"), ("enduro", "Enduro", "#D55E00"),
         ("spaceinvaders", "Space Invaders", "#009E73")]
KEY = ["seed", "stream", "prefix", "count"]
OUTCOMES = ["task_reward_difference", "computation_cost", "utility"]
TOL = 1e-12


def source(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def cluster_draws(labels, iterations, rng):
    groups = [np.flatnonzero(labels == k) for k in np.unique(labels)]
    assert len(groups) >= 2
    return [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            for _ in range(iterations)]


def codes(frame, columns):
    return pd.factorize(pd.MultiIndex.from_frame(frame[columns]))[0]


def interval(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return np.quantile(values, [.025, .975]) if len(values) else [np.nan, np.nan]


def save(fig, output, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output / f"{name}.{ext}", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def regression(frame, draws, metadata):
    """State-averaged continuous outcomes on VOC, with optional design controls."""
    x = frame.predicted_voc.to_numpy() / .01
    y = frame[OUTCOMES].to_numpy()
    clusters = codes(frame, ["seed", "stream"])
    strata = frame["count"].astype(str) + "_" + frame["prefix"].astype(str)
    covariates = pd.get_dummies(strata, drop_first=True, dtype=float).to_numpy()
    rows, plot = [], {}
    for adjusted in [False, True]:
        baseline = (np.column_stack([np.ones(len(x)), covariates]) if adjusted
                    else np.ones((len(x), 1)))
        design = np.column_stack([baseline, x])
        fit = np.linalg.lstsq(design, y, rcond=None)[0]
        samples = np.array([np.linalg.lstsq(design[i], y[i], rcond=None)[0] for i in draws])
        assert np.allclose(fit[:, 2], fit[:, 0] - fit[:, 1], atol=1e-10)
        predicted, null = np.empty_like(y), np.empty_like(y)
        left_out_slopes = []
        for c in np.unique(clusters):
            test, train = clusters == c, clusters != c
            coef = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
            predicted[test] = design[test] @ coef
            null[test] = baseline[test] @ np.linalg.lstsq(baseline[train], y[train], rcond=None)[0]
            left_out_slopes.append(coef[-1])
        loo = np.array(left_out_slopes)
        for j, outcome in enumerate(OUTCOMES):
            constant = np.ptp(y[:, j]) < TOL
            ci = interval(samples[:, -1, j])
            sse = float(np.sum((predicted[:, j] - y[:, j]) ** 2))
            sse_null = float(np.sum((null[:, j] - y[:, j]) ** 2))
            rows.append(dict(**metadata, outcome=outcome,
                adjustment="count_x_prefix_fixed_effects" if adjusted else "none",
                states=len(frame), streams=len(np.unique(clusters)), constant_outcome=constant,
                slope_per_0p01_voc=float(fit[-1, j]), slope_ci_low=ci[0], slope_ci_high=ci[1],
                spearman_rho=(float(spearmanr(x, y[:, j]).statistic) if not constant else np.nan),
                loo_stream_rmse=np.sqrt(sse / len(frame)),
                loo_stream_null_rmse=np.sqrt(sse_null / len(frame)),
                loo_stream_skill_vs_null=(1 - sse / sse_null if sse_null > TOL else np.nan),
                loo_slope_min=loo[:, j].min(), loo_slope_max=loo[:, j].max(),
                mean_predicted_voc=frame.predicted_voc.mean(),
                outcome_mean=y[:, j].mean(),
                raw_voc_rmse=(np.sqrt(np.mean((frame.predicted_voc.to_numpy() - y[:, j]) ** 2))
                              if outcome == "utility" else np.nan)))
        if not adjusted:
            grid = np.linspace(x.min(), x.max(), 120)
            grid_design = np.column_stack([np.ones(len(grid)), grid])
            band = np.quantile(samples[:, :, 0] @ grid_design.T, [.025, .975], axis=0)
            plot = dict(x=frame.predicted_voc.to_numpy(), y=y[:, 0], count=frame["count"].to_numpy(),
                        grid=grid * .01, fit=grid_design @ fit[:, 0], low=band[0], high=band[1])
    return rows, plot


def read_interventions(input_dir, output, iterations, rng):
    scores_path, states_path = input_dir / "utility_scores.csv", input_dir / "utility_states.csv"
    scores, old_states = pd.read_csv(scores_path), pd.read_csv(states_path)
    original_manifest = json.loads((input_dir / "manifest.json").read_text())
    expected_hashes = {x["pairs_path"]: x["pairs_sha256"] for x in original_manifest["utility_sources"]}
    sources = [source(scores_path), source(states_path), source(input_dir / "manifest.json")]
    states, decompositions, regressions, plots, checks = [], [], [], {}, []
    for score in scores.itertuples(index=False):
        metadata = dict(game=score.game, protocol=score.protocol, nominal_step=score.nominal_step)
        path = Path(score.source).with_name("pairs.json")
        provenance = source(path)
        assert provenance["sha256"] == expected_hashes[str(path)], "Snapshot changed: " + str(path)
        sources.append(provenance)
        raw = pd.DataFrame(json.loads(path.read_text()))
        assert not raw.duplicated(KEY + ["repetition"]).any()
        assert np.allclose(raw.delta, raw.task_delta - raw.cost, atol=1e-12, rtol=0)
        local = []
        for key, group in raw.groupby(KEY, sort=True):
            assert group.entry_hash.nunique() == 1
            assert np.ptp(group.ema) < TOL and np.ptp(group.probability) < TOL
            changed = np.abs(group.task_delta.to_numpy()) > TOL
            local.append(dict(**metadata, **dict(zip(KEY, key)), repetitions=len(group),
                predicted_voc=group.ema.mean(), continue_probability=group.probability.mean(),
                task_reward_difference=group.task_delta.mean(), computation_cost=group.cost.mean(),
                utility=group.delta.mean(), changed_repeats=int(changed.sum()), any_reward_change=bool(changed.any()),
                positive_repeats=int((group.task_delta > TOL).sum()),
                negative_repeats=int((group.task_delta < -TOL).sum()),
                mixed_sign_repeats=bool((group.task_delta > TOL).any() and (group.task_delta < -TOL).any())))
        frame = pd.DataFrame(local)
        old = old_states[(old_states.game == score.game) & (old_states.protocol == score.protocol)
                         & (old_states.nominal_step == score.nominal_step)].sort_values(KEY)
        assert frame[KEY].to_numpy().tolist() == old[KEY].to_numpy().tolist()
        assert np.allclose(frame[OUTCOMES + ["predicted_voc"]], old[OUTCOMES + ["predicted_voc"]], atol=1e-12)
        states.extend(local)
        clusters = codes(frame, ["seed", "stream"])
        draws = cluster_draws(clusters, iterations, rng)
        y = frame[OUTCOMES].to_numpy()
        means = np.array([y[i].mean(axis=0) for i in draws])
        row = dict(**metadata, states=len(frame), pairs=len(raw), streams=len(np.unique(clusters)),
            reward_changed_states=int(frame.any_reward_change.sum()),
            reward_changed_pairs=int(frame.changed_repeats.sum()),
            positive_mean_reward_states=int((frame.task_reward_difference > TOL).sum()),
            negative_mean_reward_states=int((frame.task_reward_difference < -TOL).sum()),
            mixed_sign_repeat_states=int(frame.mixed_sign_repeats.sum()),
            changed_streams=int(len(np.unique(clusters[frame.any_reward_change]))),
            allocation_gain=score.allocation_gain,
            allocation_ci_low=score.allocation_ci_low, allocation_ci_high=score.allocation_ci_high,
            max_pair_tail_bound=score.max_pair_tail_bound)
        for j, name in enumerate(OUTCOMES):
            ci = interval(means[:, j])
            row.update({name + "_mean": y[:, j].mean(), name + "_ci_low": ci[0], name + "_ci_high": ci[1]})
        assert np.isclose(row["utility_mean"], score.mean_net_utility, atol=1e-12)
        decompositions.append(row)
        result, plot = regression(frame, draws, metadata)
        regressions.extend(result)
        plots[(score.game, score.protocol, score.nominal_step)] = plot
        checks.append(dict(**metadata, duplicate_pairs=0, consistent_entry_hashes=True,
                           repeats_have_identical_predictors=True, agrees_with_previous_snapshot=True))
    pd.DataFrame(states).to_csv(output / "state_outcomes.csv", index=False)
    decomposition, reg = pd.DataFrame(decompositions), pd.DataFrame(regressions)
    decomposition.to_csv(output / "utility_decomposition.csv", index=False)
    reg.to_csv(output / "voc_outcome_regression.csv", index=False)
    return decomposition, reg, plots, sources, checks


def native_stages(path, output, iterations, rng):
    data = pd.read_csv(path)
    assert not data.duplicated(["game", "segment", "attempt", "decision", "stream"]).any()
    assert np.array_equal(data.started.to_numpy(), (data.computations > 0).astype(int))
    assert np.array_equal(data.capped.to_numpy(), (data.computations == 30).astype(int))
    assert data.computations.between(0, 30).all()
    rows = []
    for game, _, _ in GAMES:
        for end in range(1_000_000, 8_000_001, 1_000_000):
            begin = end - 1_000_000
            d = data[(data.game == game) & (data.global_step <= end)
                     & ((data.global_step > begin) | ((begin == 0) & (data.global_step == 0)))].copy()
            d["time_block"] = ((d.global_step - begin) // 32768).astype(int)
            draws = cluster_draws(codes(d, ["segment", "time_block"]), iterations, rng)
            def stats(n):
                active = n[n > 0]
                return [np.mean(n > 0), np.mean(active) if len(active) else np.nan,
                        np.mean(n), np.mean(n == 30)]
            n = d.computations.to_numpy()
            value = stats(n)
            boot = np.array([stats(n[i]) for i in draws])
            assert np.isclose(value[0] * value[1], value[2])
            row = dict(game=game, window_start=begin, window_end=end, decisions=len(d),
                started_decisions=int(d.started.sum()), cap_decisions=int(d.capped.sum()),
                started_median=float(np.median(n[n > 0])),
                cap_fraction_among_started=float(np.mean(n[n > 0] == 30)))
            for j, name in enumerate(["start_probability", "mean_count_if_started", "mean_count", "cap_fraction"]):
                ci = interval(boot[:, j])
                row.update({name: value[j], name + "_ci_low": ci[0], name + "_ci_high": ci[1]})
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output / "native_start_and_persistence.csv", index=False)
    return result, [source(path), source(path.parent / "manifest.json")]


def figures(decomposition, plots, stages, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.1), layout="constrained")
    lines = [("task_reward_difference", "Reward difference", "#0072B2", "o", "-"),
             ("computation_cost", "Computation cost", "#D55E00", "s", "--"),
             ("utility", "Net utility", "#202020", "^", ":")]
    for ax, (game, title, _) in zip(axes, GAMES):
        d = decomposition[(decomposition.game == game) & (decomposition.protocol == "standard")].sort_values("nominal_step")
        for name, label, color, marker, style in lines:
            x = d.nominal_step.to_numpy() / 1e6
            ax.plot(x, d[name + "_mean"], label=label, color=color, marker=marker, linestyle=style)
            if name == "utility":
                ax.fill_between(x, d[name + "_ci_low"], d[name + "_ci_high"], alpha=.12, color=color)
        ax.axhline(0, color="#aaa", lw=.7)
        ax.set(title=title, xlabel="Training steps (M)", xticks=[2, 4, 6, 8])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.grid(alpha=.12)
    axes[0].set_ylabel("Discounted clipped-reward units")
    axes[1].legend(fontsize=9, loc="upper center")
    fig.suptitle("Reward gain and computation cost | sub001, temperature 0.01\n"
                 "Net utility = reward difference - cost; shading: 95% stream-bootstrap interval; separate y-scales", fontsize=11)
    save(fig, output, "01_utility_decomposition")

    fig, axes = plt.subplots(2, 3, figsize=(12.3, 7.1), layout="constrained")
    for row, (protocol, step, label) in enumerate([("standard", 8_000_000, "8M standard"),
                                                  ("expanded_4M", 4_000_000, "4M expanded")]):
        for col, (game, title, color) in enumerate(GAMES):
            ax = axes[row, col]
            p = plots[(game, protocol, step)]
            for count, marker in [(0, "o"), (3, "^")]:
                keep = p["count"] == count
                ax.scatter(p["x"][keep], p["y"][keep], s=22, color=color, alpha=.5, marker=marker,
                           linewidths=0, label=f"Prior computations: {count}")
            ax.plot(p["grid"], p["fit"], color="#222", lw=1.5)
            ax.fill_between(p["grid"], p["low"], p["high"], color="#222", alpha=.12)
            ax.axhline(0, color="#aaa", lw=.6)
            ax.axvline(0, color="#aaa", lw=.6)
            ax.set(title=f"{title} | {label}", xlabel="Predicted EMA VOC")
            if col == 0:
                ax.set_ylabel("Observed reward difference")
            if np.ptp(p["y"]) < TOL:
                ax.set_ylim(-.01, .01)
                ax.text(.03, .93, "No reward differences observed", transform=ax.transAxes, va="top", fontsize=9)
            ax.grid(alpha=.12)
    axes[1, 1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Does predicted VOC anticipate reward gain?\n"
                 "Each point is a distinct state averaged over repeats; line: continuous OLS; band: 95% stream bootstrap", fontsize=11)
    save(fig, output, "02_voc_reward_regression")

    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.3), layout="constrained")
    for ax, name, title, ylabel in zip(axes,
            ["start_probability", "mean_count_if_started", "mean_count"],
            ["Start imagination", "Computations once started", "Computations per decision"],
            ["Fraction of decisions", "Mean count | count > 0", "Mean count, including zero"]):
        for game, label, color in GAMES:
            d = stages[stages.game == game]
            x = d.window_end.to_numpy() / 1e6
            ax.plot(x, d[name], "o-", label=label, color=color, ms=4)
            ax.fill_between(x, d[name + "_ci_low"], d[name + "_ci_high"], color=color, alpha=.1)
        ax.set(title=title, ylabel=ylabel, xlabel="Training steps (M)", xticks=[1, 2, 4, 6, 8])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.12)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].legend(fontsize=9)
    fig.suptitle("Starting and persistence are distinct | native training decisions\n"
                 "Preceding 1M-step windows; 30-computation cap included; shading: 95% time-block bootstrap", fontsize=11)
    save(fig, output, "03_start_and_persistence")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "voc_utility_execution_sub001_tau001_20260930")
    parser.add_argument("--decisions", type=Path, default=ROOT / "voc_state_value_compute_sub001_tau001_20260930/decisions_all_to_8M.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "voc_multiview_sub001_tau001_20260930")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(930701)
    decomposition, reg, plots, sources, checks = read_interventions(args.input, args.output, args.bootstrap, rng)
    stages, stage_sources = native_stages(args.decisions, args.output, args.bootstrap, rng)
    figures(decomposition, plots, stages, args.output)
    qa = dict(paired_sources=checks, utility_equals_reward_minus_cost=True,
              regression_slopes_respect_decomposition=True, native_stage_accounting=True,
              capped_decisions_included=True, total_distinct_evaluation_states=int(decomposition.states.sum()),
              total_paired_repeats=int(decomposition.pairs.sum()), total_native_decisions=int(stages.decisions.sum()))
    (args.output / "qa.json").write_text(json.dumps(qa, indent=2))
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), subject=1, training_seed=701,
        temperature=.01, voc_gru_connected=True, milestones=[2_000_000, 4_000_000, 6_000_000, 8_000_000],
        intervention="STOP now vs force one computation then native policy; not fixed-k search",
        reward="Clipped +/-1 environment reward, discount=.99, horizon<=1536 real actions, no value bootstrap",
        cost="Additional computation cost in current search stage only; not all subsequent-stage costs",
        zero_tolerance=TOL, expanded_4M="Separate sampling protocol; not pooled with standard evaluations",
        regression="OLS on EMA VOC/0.01; unadjusted and count-by-prefix cell fixed effects; equal state weights",
        prediction="Leave-one-environment-stream-out vs same controls without VOC; skill=1-SSEmodel/SSEnull",
        intervals=f"{args.bootstrap} cluster bootstrap draws; no multiplicity correction; exploratory",
        native="Complete sampled training decisions, resume-offset corrected; count30 cap included",
        native_uncertainty="32768-action blocks, separate segments, preceding non-overlapping 1M windows",
        limits=["One training seed and one subject; intervals are not between-model/participant uncertainty",
                "Sparse reward changes; held-out prediction and regression may be dominated by a few streams",
                "Near-zero horizon reward changes do not rule out later or distributional action effects",
                "VOC training target and finite-horizon utility differ; regression is association, not exact critic calibration",
                "Training windows contain evolving policies; they do not establish a causal training effect"],
        sources=sources + stage_sources, script=source(Path(__file__).resolve()))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("OUTPUT", args.output)
    print("8M decomposition")
    print(decomposition[decomposition.nominal_step == 8_000_000].to_string(index=False))
    print("8M and expanded4M adjusted regression")
    print(reg[((reg.nominal_step == 8_000_000) | (reg.protocol == "expanded_4M"))
              & (reg.adjustment != "none")][["game", "protocol", "outcome", "slope_per_0p01_voc",
                  "slope_ci_low", "slope_ci_high", "loo_stream_skill_vs_null"]].to_string(index=False))
    print("Native final window")
    print(stages[stages.window_end == 8_000_000].to_string(index=False))


if __name__ == "__main__":
    main()
