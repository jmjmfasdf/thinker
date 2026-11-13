"""Relate pre-search predicted VoC to executed depth in all dynamic caches.

CPU-only, read-only cache analysis. The primary population contains each
checkpoint's held-out human sessions. Training-session replicas are summarized
separately. Runs, then checkpoints, then subjects receive equal weights.
This measures model computation allocation, not realized utility or human effects.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

ROOT = Path(__file__).resolve().parents[2]
GAMES = {0: ("Enduro", "#D55E00"), 1: ("Pong", "#0072B2"),
         2: ("Space Invaders", "#009E73")}
COLS = ["run_label", "episode_id", "window_index", "real_step",
        "first_control_native_event", "last_control_native_event",
        "real_action_native_event", "additional_computations", "max_path_depth",
        "stop", "forced_stop", "voluntary_stop_observed", "stop_choice_censored",
        "reset_count", "root_carried", "root_carried_valid",
        "voc_ema_voc_initial", "voc_ema_voc_initial_valid",
        "voc_online_voc_initial", "voc_online_voc_initial_valid",
        "voc_soft_continue_probability_initial",
        "voc_soft_continue_probability_initial_valid"]
RHO_METRICS = ["rho_voc_depth", "rho_voc_budget", "rho_voc_start",
               "rho_voc_gate", "rho_gate_depth", "rho_online_voc_depth"]
MEAN_METRICS = ["mean_voc", "mean_depth", "mean_budget", "start_fraction",
                "mean_gate", "forced_fraction", "censored_fraction",
                "reset_fraction", "carry_fraction", "depth_budget_differ_fraction"]
CURVE_METRICS = ["mean_voc", "mean_depth", "mean_budget", "start_fraction", "mean_gate"]
METRICS = RHO_METRICS + MEAN_METRICS


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def milestone_artifact(milestone, artifact):
    """Read both published milestone layouts without loading model weights."""
    items = milestone["artifacts"]
    if artifact in items:
        return items[artifact]
    matches = [(name, value) for name, value in items.items()
               if name.startswith(f"ckp_{artifact}.tar_step_")]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous milestone artifact: {artifact}")
    name, value = matches[0]
    return dict(value, filename=name, step=int(name.rsplit("_", 1)[1]))


def measure(d, correlations=True):
    if d.empty:
        raise ValueError("Empty population")
    v, depth, budget, gate = [d[c].to_numpy() for c in [
        "voc_ema_voc_initial", "max_path_depth", "additional_computations",
        "voc_soft_continue_probability_initial"]]
    out = dict(n=len(d), n_started=int((budget > 0).sum()),
               n_unique_voc=d.voc_ema_voc_initial.nunique(),
               mean_voc=float(v.mean()), min_voc=float(v.min()), max_voc=float(v.max()),
               mean_depth=float(depth.mean()), mean_budget=float(budget.mean()),
               start_fraction=float((budget > 0).mean()), mean_gate=float(np.nanmean(gate)),
               forced_fraction=float(d.forced_stop.eq(1).mean()),
               censored_fraction=float(d.stop_choice_censored.eq(1).mean()),
               reset_fraction=float(d.reset_count.gt(0).mean()),
               carry_fraction=float(d.root_carried.where(d.root_carried_valid.eq(1)).mean()),
               depth_budget_differ_fraction=float((depth != budget).mean()))
    if correlations:
        out.update(rho_voc_depth=rho(v, depth), rho_voc_budget=rho(v, budget),
                   rho_voc_start=rho(v, budget > 0), rho_voc_gate=rho(v, gate),
                   rho_gate_depth=rho(gate, depth),
                   rho_online_voc_depth=rho(d.voc_online_voc_initial, depth))
    return out


def check_prefix(cache, d):
    """Check cached initial VoC against the first native control choice."""
    e = pd.read_csv(cache / "native_events.csv.gz", nrows=256, usecols=[
        "run_label", "episode_id", "window_index", "native_event",
        "is_control_choice", "voc_ema_voc", "voc_ema_q_valid",
        "search_steps_pre", "search_steps_post", "search_computation",
        "current_node_depth_post"])
    choices = e[e.is_control_choice.eq(1)]
    first = choices.groupby(["run_label", "episode_id", "window_index"], sort=False).first().reset_index()
    keys = ["run_label", "episode_id", "window_index"]
    merged = d.merge(first, on=keys, validate="one_to_one")
    valid = merged.voc_ema_voc_initial_valid.eq(1) & merged.voc_ema_q_valid.eq(1)
    if not np.allclose(merged.loc[valid, "voc_ema_voc_initial"], merged.loc[valid, "voc_ema_voc"], atol=1e-7):
        raise ValueError(f"Initial VoC is not the first control-choice VoC: {cache}")
    if not merged.search_steps_pre.eq(0).all():
        raise ValueError(f"Initial VoC follows computation: {cache}")
    count = 0
    for r in merged.itertuples(index=False):
        ep = e[e.episode_id.eq(r.episode_id)]
        if r.last_control_native_event > ep.native_event.max():
            continue
        stage = ep[ep.native_event.between(r.first_control_native_event, r.last_control_native_event)]
        search = stage[stage.search_computation.eq(1)]
        depth = int(search.current_node_depth_post.max()) if len(search) else 0
        if len(search) != r.additional_computations or depth != r.max_path_depth:
            raise ValueError(f"Depth/budget disagrees with native prefix: {cache}")
        count += 1
    return dict(initial_choices_checked=len(merged), completed_depths_checked=count)


def analyze_cache(entry, root, inventory):
    cache = root / entry.relative_path
    bpath = cache / "checkpoint_binding.json"
    b = json.loads(bpath.read_text())
    if b["status"] != "COMPLETED":
        raise ValueError(f"Incomplete binding: {cache}")
    if any(int(b[k]) != int(getattr(entry, k)) for k in ["subject", "game", "leave_session"]):
        raise ValueError(f"Identity mismatch: {cache}")
    if b["session_roles"][str(b["leave_session"])] != "held_out":
        raise ValueError(f"Incorrect held-out assignment: {cache}")
    config_path = Path(b["checkpoint"]) / "config_c.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    cap = int(cfg["max_search_steps"])
    milestone_path = Path(b["checkpoint"]) / "milestone_5000000.json"
    milestone = json.loads(milestone_path.read_text())
    artifacts = {kind: milestone_artifact(milestone, kind) for kind in ["actor", "model"]}
    manifest = json.loads((cache / "manifest.json").read_text())
    hashes = manifest["captures"][0]["checkpoint_hashes"]
    for artifact in ["actor", "model"]:
        if hashes[f"checkpoint_{artifact}_sha256"] != artifacts[artifact]["sha256"]:
            raise ValueError(f"Milestone/capture checkpoint mismatch: {cache}")
    path = cache / "native_decisions.csv.gz"
    file_hash = sha256(path)
    if file_hash != inventory[entry.relative_path]["file_sha256"][path.name]:
        raise ValueError(f"Cache differs from immutable inventory: {path}")
    d = pd.read_csv(path, usecols=COLS)
    if len(d) != int(entry.real_steps) or d.duplicated(["run_label", "real_step"]).any():
        raise ValueError(f"Decision count or uniqueness failed: {cache}")
    prefix = check_prefix(cache, d)
    timing_ok = (d.first_control_native_event <= d.last_control_native_event) & (
        d.last_control_native_event < d.real_action_native_event)
    basic = d.voc_ema_voc_initial_valid.eq(1) & np.isfinite(d.voc_ema_voc_initial)
    basic &= np.isfinite(d.max_path_depth) & np.isfinite(d.additional_computations) & timing_ok
    if not ((d.max_path_depth >= 0) & (d.max_path_depth <= d.additional_computations)
            & (d.additional_computations <= cap)).all():
        raise ValueError(f"Depth/budget out of range: {cache}")
    d.loc[~d.voc_soft_continue_probability_initial_valid.eq(1), "voc_soft_continue_probability_initial"] = np.nan
    d.loc[~d.voc_online_voc_initial_valid.eq(1), "voc_online_voc_initial"] = np.nan
    gate = d.voc_soft_continue_probability_initial.dropna()
    if not gate.between(0, 1).all():
        raise ValueError(f"Gate probability out of range: {cache}")
    qa = dict(cache=str(cache), subject=b["subject"], game=b["game"],
              leave_session=b["leave_session"], rows=len(d), excluded_invalid=int((~basic).sum()),
              forced_stop=int(d.forced_stop.eq(1).sum()), censored=int(d.stop_choice_censored.eq(1).sum()),
              no_observed_stop=int(d.stop.ne(1).sum()), resets=int(d.reset_count.sum()),
              depth_budget_differ=int(d.max_path_depth.ne(d.additional_computations).sum()),
              numeric_file_sha256=file_hash, binding_sha256=sha256(bpath),
              config_sha256=sha256(config_path), cache_manifest_sha256=sha256(cache / "manifest.json"),
              actor_step=artifacts["actor"]["step"],
              model_step=artifacts["model"]["step"],
              flags={k: cfg.get(k) for k in ["voc_gate_q_temperature", "voc_gate_exact_projection",
                  "voc_gate_epsilon_greedy_execution", "voc_eval_stochastic", "max_search_steps", "tree_carry"]},
              prefix_validation=prefix)
    rows, curves, signs = [], [], []
    for run, raw in d.groupby("run_label", sort=True):
        match = re.search(r"_ses(\d+)_", run)
        if not match:
            raise ValueError(run)
        session = int(match[1])
        key = dict(game=b["game"], subject=b["subject"], leave_session=b["leave_session"],
                   run=run, session=session, split=b["session_roles"][str(session)])
        x = raw.loc[basic.loc[raw.index]]
        scopes = {"all": np.ones(len(x), bool), "started": x.additional_computations.gt(0),
                  "positive_depth": x.max_path_depth.gt(0),
                  "uncensored": x.stop_choice_censored.eq(0) & x.forced_stop.eq(0) & x.stop.eq(1),
                  "no_reset": x.reset_count.eq(0),
                  "no_carry": x.root_carried_valid.eq(1) & x.root_carried.eq(0)}
        for scope, mask in scopes.items():
            part = x.loc[mask]
            if part.empty:
                continue
            rows.append(dict(**key, scope=scope, n_input=len(raw), **measure(part)))
            if scope not in ["all", "started"]:
                continue
            # Mid-ranks preserve ties: equal VoC never goes into different bins.
            pct = (part.voc_ema_voc_initial.rank(method="average") - .5) / len(part)
            bins = np.minimum((pct * 5).astype(int), 4) + 1
            for q, group in part.groupby(bins):
                curves.append(dict(**key, scope=scope, quintile=int(q), **measure(group, False)))
            for sign, signmask in [("negative", part.voc_ema_voc_initial < -1e-6),
                                   ("near_zero", part.voc_ema_voc_initial.abs() <= 1e-6),
                                   ("positive", part.voc_ema_voc_initial > 1e-6)]:
                group = part.loc[signmask]
                if len(group):
                    signs.append(dict(**key, scope=scope, sign=sign, **measure(group, False)))
    return rows, curves, signs, qa


def aggregate(runs, extra, metrics):
    keys = ["game", "subject", "leave_session", "split", "scope"] + extra
    expanded = pd.concat([runs, runs.assign(split="all_sessions")], ignore_index=True)
    cache = expanded.groupby(keys, as_index=False)[metrics].mean()
    counts = expanded.groupby(keys, as_index=False).agg(n=("n", "sum"), n_runs=("run", "nunique"))
    cache = cache.merge(counts, on=keys, validate="one_to_one")
    skeys = [k for k in keys if k != "leave_session"]
    subject = cache.groupby(skeys, as_index=False)[metrics].mean()
    counts = cache.groupby(skeys, as_index=False).agg(n=("n", "sum"), n_checkpoints=("leave_session", "nunique"))
    subject = subject.merge(counts, on=skeys, validate="one_to_one")
    summary = []
    gkeys = ["game", "split", "scope"] + extra
    for identity, group in subject.groupby(gkeys):
        row = dict(zip(gkeys, identity))
        row.update(n=int(group.n.sum()), n_subjects=len(group))
        for metric in metrics:
            z = group[metric].dropna()
            row.update({metric: z.mean(), metric + "_sd": z.std(ddof=1),
                        metric + "_min": z.min(), metric + "_max": z.max(),
                        metric + "_n_subjects": len(z)})
            if metric.startswith("rho_"):
                row[metric + "_positive_subjects"] = int(z.gt(0).sum())
        summary.append(row)
    return cache, subject, pd.DataFrame(summary)


def plot(subject_curves, summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 3, figsize=(11.7, 9.2), sharex=True, sharey="row")
    fig.subplots_adjust(left=.092, right=.98, top=.865, bottom=.18, hspace=.38, wspace=.15)
    panels = [("all", "mean_depth", "Maximum search depth\nAll decisions"),
              ("started", "mean_depth", "Maximum search depth\nAfter any computation"),
              ("all", "start_fraction", "Decisions starting computation (%)")]
    for col, game in enumerate([0, 1, 2]):
        label, color = GAMES[game]
        for row, (scope, metric, ylabel) in enumerate(panels):
            ax = axes[row, col]
            z = subject_curves[(subject_curves.game == game) & (subject_curves.scope == scope)
                               & (subject_curves.split == "held_out")]
            scale = 100 if metric == "start_fraction" else 1
            for subject, one in z.groupby("subject"):
                one = one.sort_values("quintile")
                ax.plot(one.quintile, one[metric] * scale, color=color, alpha=.24, lw=1)
            mean = z.groupby("quintile", as_index=False)[metric].mean()
            ax.plot(mean.quintile, mean[metric] * scale, "o-", color=color, lw=2.8, ms=5)
            ax.grid(axis="y", alpha=.15)
            ax.set_ylim(bottom=0)
            ax.set_xticks(range(1, 6), ["Q1\nLow", "Q2", "Q3", "Q4", "Q5\nHigh"])
            s = summary[(summary.game == game) & (summary.scope == scope) & (summary.split == "held_out")].iloc[0]
            corr = s.rho_voc_start if row == 2 else s.rho_voc_depth
            ax.text(.04, .96, f"Mean within-run rho = {corr:.3f}", transform=ax.transAxes,
                    va="top", fontsize=9, bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=2))
            if row == 0:
                ax.set_title(f"{label} | {int(s.n):,} decisions", fontsize=11, pad=12)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("Pre-search predicted VoC quintile")
    for row, (scope, metric, _) in enumerate(panels):
        values = subject_curves[(subject_curves.split == "held_out") & (subject_curves.scope == scope)][metric]
        scale = 100 if metric == "start_fraction" else 1
        axes[row, 0].set_ylim(0, float(values.max()) * scale * 1.2)
    fig.suptitle("Predicted VoC and executed planning in human gameplay replay", x=.092, ha="left",
                 y=.978, fontsize=15, fontweight="bold")
    fig.text(.092, .93, "Held-out sessions | 6 participants × 3 games × 4 checkpoints | Frozen ~5M models, temperature 0.05", fontsize=10)
    fig.text(.092, .065, "Thin lines: subject-adapted models. Thick lines: mean of six subjects; runs and checkpoints weighted equally.", fontsize=9)
    fig.text(.092, .04, "VoC bins are defined within each run (separately for each row's population). Depth is the maximum explored in this decision.", fontsize=9)
    fig.text(.092, .015, "Model computation allocation only; these plots do not test realized utility or explain human action switching.", fontsize=9)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output / f"predicted_voc_search_depth.{ext}", dpi=180, facecolor="white")
    plt.close(fig)


def report(output, summary, curves, qa, runs):
    lines = ["# Predicted VoC versus executed planning: full dynamic cache", "",
             "## Data and method", "",
             f"- Read {len(qa)} frozen human-replay caches; {sum(q['rows'] for q in qa):,} decision rows including checkpoint replicas.",
             f"- Primary held-out set: {int(runs[(runs.split == 'held_out') & (runs.scope == 'all')].n.sum()):,} decisions, "
             f"{runs[runs.split == 'held_out'].run.nunique()} unique source runs. Each human source run occurs under exactly one held-out checkpoint.",
             "- Predictor: initial EMA Q(CONTINUE) minus Q(STOP), before any new computation. Online VoC is a sensitivity analysis.",
             "- Outcomes: maximum path depth explored during this decision, computation count (including RESET), and whether any computation started.",
             "- All decisions include immediate STOP (depth zero). Started-only includes RESET-only decisions; positive-depth sensitivity is also saved.",
             "- Compute Spearman correlations within each run; take arithmetic means over runs within checkpoint, checkpoints within subject, and six subjects. No decision-level p values.",
             "- Quintile curves use within-run mid-ranks, separately for all/started populations; tied VoC values are kept together. Sparse quintiles retain their actual support.",
             "- All-session and training-session summaries are separate descriptive checks. Repeated checkpoints are never independent human observations.",
             "- Observed capped depth is retained because the outcome is executed computation; uncensored-only sensitivity is separate.",
             "- No model inference, retraining, reward-utility comparison, or behavioral prediction test was performed.", "",
             "## Primary results", "",
             "Values below are means of subject-level mean within-run Spearman rho; brackets are ranges across six subjects, not confidence intervals.", "",
             "| Game | All decisions: VoC vs depth | Started: VoC vs depth | Positive subjects (all / started) | VoC vs start | VoC vs gate probability |",
             "|---|---:|---:|---:|---:|---:|"]
    for game, (label, _) in GAMES.items():
        s = summary[(summary.game == game) & (summary.split == "held_out")].set_index("scope")
        a, b = s.loc["all"], s.loc["started"]
        fmt = lambda r: f"{r.rho_voc_depth:.4f} [{r.rho_voc_depth_min:.4f}, {r.rho_voc_depth_max:.4f}]"
        lines.append(f"| {label} | {fmt(a)} | {fmt(b)} | {int(a.rho_voc_depth_positive_subjects)}/6 / {int(b.rho_voc_depth_positive_subjects)}/6 | {a.rho_voc_start:.4f} | {a.rho_voc_gate:.4f} |")
    lines += ["", "## Low versus high initial-VoC quintiles (held out)", "",
              "Within-run quintiles use relative VoC, so low/high is not equivalent to negative/positive VoC.", "",
              "| Game | All: depth Q1 → Q5 | Started: depth Q1 → Q5 | Starting computation Q1 → Q5 |",
              "|---|---:|---:|---:|"]
    for game, (label, _) in GAMES.items():
        c = curves[(curves.game == game) & (curves.split == "held_out")].set_index(["scope", "quintile"])
        a, b = c.loc[("all", 1)], c.loc[("all", 5)]
        x, y = c.loc[("started", 1)], c.loc[("started", 5)]
        lines.append(f"| {label} | {a.mean_depth:.3f} → {b.mean_depth:.3f} | {x.mean_depth:.3f} → {y.mean_depth:.3f} | {100*a.start_fraction:.2f}% → {100*b.start_fraction:.2f}% |")
    lines += ["", "## Validation and interpretation", "",
              f"- Excluded invalid predictor/timing rows: {sum(q['excluded_invalid'] for q in qa)}.",
              f"- Forced STOP / censored / no observed STOP rows (all cache replicas): {sum(q['forced_stop'] for q in qa)} / {sum(q['censored'] for q in qa)} / {sum(q['no_observed_stop'] for q in qa)}.",
              f"- Total RESET events: {sum(q['resets'] for q in qa):,}; decisions with depth differing from computation count: {sum(q['depth_budget_differ'] for q in qa):,}.",
              "- Every numeric decision table was SHA256-checked against the unified-cache inventory. Capture checkpoint hashes were matched to the published ~5M milestone.",
              "- In each cache, a native-event prefix independently verifies initial VoC at the first control choice with zero preceding computation, and depth/budget for completed prefix decisions.",
              "- These are older learned-gate checkpoints (exact projection disabled). They differ from the recent ~10M exact-projection training logs, despite sharing temperature 0.05. Pooling or attributing differences solely to evaluation versus training would be inappropriate.",
              "- An association here shows that the model allocates computation in relation to predicted VoC during human replay. It does not establish that humans use VoC, that computation improves performance, or that depth is causally controlled by VoC.",
              "- This pass tests pre-search VoC versus subsequent executed depth. It does not reanalyze current VoC at each internal step versus future remaining depth; that requires the full native-event tables.",
              "- Conditioning on having started search is descriptive, not a causal adjustment; no-carry and no-reset subsets are also descriptive.", "",
              "## Files", "",
              "`run_scores.csv`, `checkpoint_scores.csv`, `subject_scores.csv`, `summary.csv`: correlations, counts, and means for primary and sensitivity populations.",
              "`*_curves.csv`: quintiles and actual support. `*_sign.csv`: negative/near-zero/positive VoC descriptive subsets (unequal support).",
              "`manifest.json`: provenance, exact flags, checkpoint steps and validation. `predicted_voc_search_depth.pdf/png/svg`: primary held-out figure.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, default=ROOT / "research_script/outputs/00_dynamic_cache")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--resume", action="store_true", help="Reuse this script's completed per-cache summaries")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    partial = args.output / "cache_summaries"
    partial.mkdir(exist_ok=args.resume)
    script_hash = sha256(Path(__file__))
    index = pd.read_csv(args.cache_root / "cache_index.csv")
    inv = json.loads((args.cache_root / "manifest.json").read_text())
    inventory = {str(Path(c["cache"]).relative_to(Path(c["cache"]).parents[1])): c for c in inv["caches"]}
    collected, curves, signs, qa, errors = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        for row in index.itertuples(index=False):
            saved = partial / (row.relative_path.replace("/", "_") + ".json")
            if args.resume and saved.exists():
                data = json.loads(saved.read_text())
                if data["script_sha256"] != script_hash or data["source_sha256"] != inventory[row.relative_path]["file_sha256"]["native_decisions.csv.gz"]:
                    raise ValueError(f"Saved summary provenance differs: {saved}")
                r, c, s, q = data["result"]
                collected.extend(r); curves.extend(c); signs.extend(s); qa.append(q)
                print(f"Reused {len(qa)}/{len(index)}: {row.relative_path}", flush=True)
            else:
                pending[pool.submit(analyze_cache, row, args.cache_root, inventory)] = row.relative_path
        for future in as_completed(pending):
            try:
                r, c, s, q = future.result()
                saved = partial / (pending[future].replace("/", "_") + ".json")
                saved.write_text(json.dumps(dict(script_sha256=script_hash,
                    source_sha256=q["numeric_file_sha256"], result=[r, c, s, q])) + "\n")
                collected.extend(r); curves.extend(c); signs.extend(s); qa.append(q)
                print(f"Completed {len(qa)}/{len(index)}: {pending[future]}", flush=True)
            except Exception as e:
                errors.append(dict(cache=pending[future], error=repr(e)))
                print(f"FAILED {pending[future]}: {e!r}", flush=True)
    if errors:
        (args.output / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
        raise RuntimeError(f"{len(errors)} caches failed; no complete result claimed")
    if (args.output / "errors.json").exists():
        (args.output / "errors.json").unlink()
    runs = pd.DataFrame(collected)
    held = runs[(runs.split == "held_out") & (runs.scope == "all")]
    if held.run.duplicated().any() or len(held) != inv["unique_source_runs"]:
        raise ValueError("Held-out source runs are repeated or missing")
    if len(qa) != 72 or held.subject.nunique() != 6 or held.game.nunique() != 3:
        raise ValueError("Incomplete expected full-cache coverage")
    for name, frame, extra, metrics in [
            ("scores", runs, [], METRICS),
            ("curves", pd.DataFrame(curves), ["quintile"], CURVE_METRICS),
            ("sign", pd.DataFrame(signs), ["sign"], CURVE_METRICS)]:
        cache, subject, summary = aggregate(frame, extra, metrics)
        for level, table in [("run", frame), ("checkpoint", cache), ("subject", subject), ("summary", summary)]:
            file = "summary.csv" if name == "scores" and level == "summary" else f"{level}_{name}.csv"
            table.sort_values([c for c in ["game", "subject", "leave_session", "split", "scope", "run"] + extra if c in table]).to_csv(args.output / file, index=False)
        if name == "scores":
            result = summary
        elif name == "curves":
            subject_curves, mean_curves = subject, summary
    plot(subject_curves, result, args.output)
    report(args.output, result, mean_curves, qa, runs)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), status="COMPLETED",
                    source_kind="Frozen-checkpoint human gameplay replay, not training or reward-utility evaluation",
                    cache_root=str(args.cache_root.resolve()), input_manifest_sha256=sha256(args.cache_root / "manifest.json"),
                    script_sha256=sha256(Path(__file__)), primary_split="held_out",
                    weighting="Equal run -> checkpoint -> subject; six subjects are group units",
                    within_search_event_analysis=False, caches=sorted(qa, key=lambda r: (r["game"], r["subject"], r["leave_session"])))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(result[(result.split == "held_out") & result.scope.isin(["all", "started"])][
        ["game", "scope", "n", "rho_voc_depth", "rho_voc_budget", "rho_voc_start", "rho_voc_gate"]].to_string(index=False))


if __name__ == "__main__":
    main()
