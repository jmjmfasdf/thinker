#!/usr/bin/env python3
"""Human exact-action durations and switches in Atari and Mario (no model replay).

Primary estimand: timestamp duration of a constant exact action, with both onset
and following switch observed within a continuous recording segment. Group
distributions average subject-normalized densities, giving each subject equal
weight. Mario's primary actions are the original BK2 button sets; canonical NPZ
actions are also extracted to quantify information lost in the 9-action mapping.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
GAMES = ["Enduro", "Pong", "Space Invaders", "Super Mario Bros."]
SLUG = dict(zip(GAMES, ["enduro", "pong", "space_invaders", "super_mario"]))
ATARI = {
    "Enduro": ["NOOP", "FIRE", "RIGHT", "LEFT", "DOWN", "DOWNRIGHT", "DOWNLEFT", "RIGHTFIRE", "LEFTFIRE"],
    "Pong": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
    "Space Invaders": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
}
BUTTONS = ["A", "Right", "Left", "Down", "B", "Select", "Start", "Up"]
VOCAB = [[], ["Right"], ["Right", "B"], ["A", "Right"], ["A", "Right", "B"], ["Left"], ["A"], ["B"], ["Left", "B"]]
VOCAB_SETS = [set(x) for x in VOCAB]
DROP = {"Up", "Down", "Select", "Start"}
COLORS = ["#566573", "#2e86c1", "#d98d25"]
LOG_EDGES = np.linspace(np.log10(.003), np.log10(180), 91)
LOG_CENTERS = (LOG_EDGES[:-1] + LOG_EDGES[1:]) / 2
BIN_WIDTH = np.diff(LOG_EDGES)[0]
SMOOTH_DEX = .10


def raw_name(code):
    return "+".join(b for i, b in enumerate(BUTTONS) if int(code) & (1 << i)) or "NOOP"


def canon_index(selected):
    for j, item in enumerate(VOCAB_SETS):
        if selected == item:
            return j
    selected = selected - DROP
    return min(range(len(VOCAB)), key=lambda j: len(selected ^ VOCAB_SETS[j]))


def load_bk2(path, n):
    with zipfile.ZipFile(path) as z:
        payload = z.read("Input Log.txt")
    order = BUTTONS
    patterns = []
    active = False
    for line in payload.decode("utf-8").splitlines():
        line = line.strip()
        if line == "[Input]":
            active = True
        elif line == "[/Input]":
            break
        elif active and line.startswith("P1 "):
            order = [v.strip().split(" ", 1)[1] for v in line.split("|") if v.strip()]
        elif active and line.startswith("|"):
            patterns.append(line.split("|")[2])
    # Converter skips the movie's initial state, then stops at done.
    selected = patterns[1:1+n]
    if len(selected) != n or set(order) != set(BUTTONS):
        raise ValueError(f"BK2 alignment/header mismatch: {path}")
    lookup = {}
    for p in set(selected):
        if len(p) != len(order):
            raise ValueError(f"Unexpected controller pattern: {p!r}")
        keys = {b for c, b in zip(p, order) if c != "."}
        code = sum(1 << BUTTONS.index(b) for b in keys)
        lookup[p] = (code, canon_index(keys), keys not in VOCAB_SETS)
    raw = np.fromiter((lookup[p][0] for p in selected), dtype=np.int16, count=n)
    mapped = np.fromiter((lookup[p][1] for p in selected), dtype=np.int16, count=n)
    nonvocab = sum(lookup[p][2] for p in selected)
    return raw, mapped, nonvocab, len(patterns), hashlib.sha256(payload).hexdigest()


def extract(a, t, boundary, meta, representation):
    onset = boundary.copy()
    onset[1:] |= a[1:] != a[:-1]
    starts = np.flatnonzero(onset)
    ends = np.r_[starts[1:], len(a)]
    right = np.ones(len(starts), dtype=bool)
    valid_end = ends < len(a)
    right[valid_end] = boundary[ends[valid_end]]
    left = boundary[starts]
    # Exact durations for observed switches; lower bounds for censored endings.
    stop_index = np.where(right, ends - 1, ends)
    duration = t[stop_index] - t[starts]
    following = np.full(len(starts), -1, dtype=int)
    following[~right] = a[ends[~right]]
    previous = np.full(len(starts), -1, dtype=int)
    previous[~left] = a[starts[~left] - 1]
    out = pd.DataFrame(dict(start=starts, end=ends, action=a[starts],
        previous_action=previous, next_action=following,
        n_samples=ends-starts, duration_s=duration,
        left_censored=left, right_censored=right,
        segment=np.cumsum(boundary)[starts], representation=representation))
    if int(out.n_samples.sum()) != len(a) or (duration < 0).any():
        raise ValueError("Bout partition failed")
    for key, value in meta.items():
        out[key] = value
    return out


def process_source(task):
    path = Path(task["path"])
    with np.load(path, allow_pickle=False) as z:
        onehot = z["action"]
        times = z["time"].reshape(-1).astype(float)
        first = z["is_first"].reshape(-1).astype(bool)
        terminal = z["is_terminal"].reshape(-1).astype(bool)
    if onehot.ndim != 2 or not np.isfinite(onehot).all():
        raise ValueError(f"Invalid actions: {path}")
    if not ((onehot == 0) | (onehot == 1)).all() or not (onehot.sum(axis=1) == 1).all():
        raise ValueError(f"Actions are not one-hot: {path}")
    n = len(times)
    if not (n == len(onehot) == len(first) == len(terminal)) or not np.isfinite(times).all():
        raise ValueError(f"Invalid timestamps: {path}")
    a = onehot.argmax(axis=1).astype(np.int16)
    dt = np.diff(times)
    positive = dt[dt > 0]
    gap_limit = max(.25, 10 * float(np.median(positive)))
    boundary = first.copy()
    boundary[0] = True
    boundary[1:] |= terminal[:-1] | (dt <= 0) | (dt > gap_limit)
    eligible = ~boundary[1:]
    meta = {k: task[k] for k in ["file_id", "game", "subject", "session", "stage"]}
    audit = {**task, "n_frames": n, "n_first": int(first.sum()), "n_terminal": int(terminal.sum()),
        "n_segments": int(boundary.sum()), "dt_median": float(np.median(positive)),
        "dt_min": float(positive.min()), "dt_max": float(positive.max()),
        "n_clock_reversals": int((dt <= 0).sum()), "n_large_gaps": int((dt > gap_limit).sum()),
        "observed_seconds": float(dt[eligible].sum()),
        "switches_canonical": int(((a[1:] != a[:-1]) & eligible).sum())}
    h = hashlib.sha256()
    for arr in [onehot, times, first, terminal]:
        h.update(str((arr.shape, arr.dtype)).encode()); h.update(arr.tobytes())
    audit["behavior_fields_sha256"] = h.hexdigest()
    if task["game"] == GAMES[-1]:
        raw, mapped, nonvocab, n_patterns, digest = load_bk2(task["bk2_path"], n)
        mismatch = int((mapped != a).sum())
        if mismatch:
            raise ValueError(f"NPZ/BK2 alignment mismatch ({mismatch} frames): {path}")
        audit.update(nonvocab_frames=nonvocab, bk2_frames=n_patterns, bk2_input_sha256=digest,
            mapped_mismatch_frames=mismatch, switches_exact=int(((raw[1:] != raw[:-1]) & eligible).sum()),
            raw_noop_frames=int((raw == 0).sum()), canonical_noop_frames=int((a == 0).sum()))
        bouts = pd.concat([extract(raw, times, boundary, meta, "exact"),
            extract(a, times, boundary, meta, "canonical9")], ignore_index=True)
    else:
        if onehot.shape[1] != len(ATARI[task["game"]]):
            raise ValueError(f"Atari action count mismatch: {path}")
        audit["switches_exact"] = audit["switches_canonical"]
        bouts = extract(a, times, boundary, meta, "exact")
    return bouts, audit


def tasks_for_data():
    tasks = []
    for p in sorted((ROOT / "behavioral_data_block").glob("sub-*/ses-*/*.npz")):
        m = re.search(r"sub(\d+)-ses(\d+)-block(\d+)-game(\d+)\.npz$", p.name)
        if not m or int(m[4]) > 2:
            continue
        tasks.append(dict(path=str(p), game=GAMES[int(m[4])], subject=int(m[1]),
            session=int(m[2]), stage="", bk2_path=""))
    manifest = pd.read_csv(ROOT / "behavioral_data_mario/manifest_canonical.tsv", sep="\t")
    if manifest.output_path.duplicated().any():
        raise ValueError("Duplicate Mario manifest paths")
    for r in manifest.to_dict("records"):
        tasks.append(dict(path=r["output_path"], game=GAMES[-1],
            subject=int(r["sub"].split("-")[-1]), session=int(r["ses"].split("-")[-1]),
            stage=f"w{r['world']}l{r['level']}", bk2_path=r["bk2_path"]))
    for i, task in enumerate(tasks):
        task["file_id"] = i
    return tasks


def label(game, action, representation="exact"):
    if game != GAMES[-1]:
        return ATARI[game][int(action)]
    return raw_name(action) if representation == "exact" else "+".join(VOCAB[int(action)]) or "NOOP"


def complete(b):
    return b.loc[~b.left_censored & ~b.right_censored & (b.duration_s > 0)]


def density(values, smooth=True, samples=False):
    values = np.asarray(values)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return np.full(len(LOG_CENTERS), np.nan)
    hist = np.histogram(np.log10(values), LOG_EDGES)[0] / len(values) / BIN_WIDTH
    return gaussian_filter1d(hist, SMOOTH_DEX / BIN_WIDTH) if smooth else hist


def group_density(df, column="duration_s", smooth=True):
    curves = [density(d[column], smooth=smooth) for _, d in df.groupby("subject")]
    return np.mean(curves, axis=0), np.asarray(curves)


def style_axis(ax, title="", xlabel=True):
    ax.set_xscale("log")
    ax.set_xlim(.005, 100)
    ax.set_xticks([.01, .1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.18)
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    if xlabel:
        ax.set_xlabel("Unchanged action duration (s)")
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def summarize(b, audit, out):
    c = complete(b)
    rows = []
    for keys, d in c.groupby(["game", "representation", "subject", "action"]):
        game, rep, sub, act = keys
        q = d.duration_s.quantile([.1, .25, .5, .75, .9, .95])
        rows.append(dict(game=game, representation=rep, subject=sub, action=act,
            action_name=label(game, act, rep), n_bouts=len(d), median_s=q.loc[.5],
            q10_s=q.loc[.1], q25_s=q.loc[.25], q75_s=q.loc[.75], q90_s=q.loc[.9], q95_s=q.loc[.95],
            mean_s=d.duration_s.mean(), fraction_over_1s=(d.duration_s > 1).mean(),
            fraction_over_2s=(d.duration_s > 2).mean()))
    s = pd.DataFrame(rows)
    s.to_csv(out / "action_subject_summary.csv", index=False)
    group = s.groupby(["game", "representation", "action", "action_name"]).agg(
        n_subjects=("subject", "nunique"), n_bouts=("n_bouts", "sum"),
        mean_subject_median_s=("median_s", "mean"), min_subject_median_s=("median_s", "min"),
        max_subject_median_s=("median_s", "max"), fraction_over_1s=("fraction_over_1s", "mean"),
        fraction_over_2s=("fraction_over_2s", "mean")).reset_index()
    group.to_csv(out / "action_group_summary.csv", index=False)
    coverage = audit.groupby(["game", "subject"]).agg(n_files=("file_id", "size"),
        n_frames=("n_frames", "sum"), seconds=("observed_seconds", "sum"),
        exact_switches=("switches_exact", "sum"), canonical_switches=("switches_canonical", "sum"),
        dt_median=("dt_median", "median"), n_segments=("n_segments", "sum")).reset_index()
    coverage["switches_per_second"] = coverage.exact_switches / coverage.seconds
    coverage.to_csv(out / "coverage_by_subject.csv", index=False)
    inventory = []
    for game, d in b[b.representation.eq("exact")].groupby("game"):
        a = audit[audit.game.eq(game)]
        inventory.append(dict(game=game, n_subjects=d.subject.nunique(), n_files=len(a),
            n_frames=int(a.n_frames.sum()), n_actions=d.action.nunique(),
            hours=float(a.observed_seconds.sum()/3600), n_bouts=len(d),
            complete_bouts=len(complete(d)), excluded_censored_fraction=1-len(complete(d))/len(d),
            median_dt_file=float(a.dt_median.median()), n_large_gaps=int(a.n_large_gaps.sum()),
            n_clock_reversals=int(a.n_clock_reversals.sum())))
    pd.DataFrame(inventory).to_csv(out / "coverage_by_game.csv", index=False)
    return c


def figures_and_modes(b, audit, out):
    c = summarize(b, audit, out)
    exact = c[c.representation.eq("exact")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for game, ax in zip(GAMES, axes.flat):
        d = exact[exact.game.eq(game)]
        for i, (name, subset) in enumerate([("All exact actions", d), ("NOOP", d[d.action.eq(0)]), ("Non-NOOP actions", d[d.action.ne(0)])]):
            y, _ = group_density(subset)
            ax.plot(10**LOG_CENTERS, y, label=name, color=COLORS[i], lw=2)
        style_axis(ax, f"{game}  |  {d.subject.nunique()} subjects, {len(d):,} bouts")
        ax.set_ylabel("Density per log10(second)")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Human action persistence: time until the next exact-action switch", fontsize=16, weight="bold")
    fig.supxlabel("Equal subject weight; observed onset + observed switch; smoothing = 0.10 log10(s). Mario: original button sets.", fontsize=10)
    save(fig, out, "01_four_game_overview")
    modes = []
    transitions = []
    displayed = {}
    for game in GAMES:
        d = exact[exact.game.eq(game)]
        order = d.groupby("action").size().sort_values(ascending=False).index.tolist()
        # Display nine common Mario button sets; retain every combination in tables.
        order = sorted(order) if game != GAMES[-1] else [0] + [a for a in order if a != 0][:8]
        displayed[game] = order
        cols = 3
        rows = (len(order)+2)//3
        fig, axes = plt.subplots(rows, cols, figsize=(13, rows*2.8), squeeze=False, layout="constrained")
        for act, ax in zip(order, axes.flat):
            sub = d[d.action.eq(act)]
            y, individual = group_density(sub)
            raw, _ = group_density(sub, smooth=False)
            ax.stairs(raw, 10**LOG_EDGES, color="#aab7b8", alpha=.55, fill=True, lw=.5)
            for curve in individual:
                ax.plot(10**LOG_CENTERS, curve, color="#5499c7", alpha=.28, lw=.8)
            ax.plot(10**LOG_CENTERS, y, color="#145a84", lw=2)
            style_axis(ax, f"{label(game, act)}  |  n={len(sub):,}")
            ax.set_ylabel("Log-time density")
        for ax in axes.flat[len(order):]:
            ax.axis("off")
        fig.suptitle(f"{game}: duration of each unchanged action", fontsize=16, weight="bold")
        fig.supxlabel("Gray: subject-averaged histogram. Dark: smoothed mean. Thin blue: each subject. Same time bins in all games.", fontsize=10)
        save(fig, out, f"02_{SLUG[game]}_actions")
        subjects = sorted(d.subject.unique())
        fig, axes = plt.subplots(len(subjects), len(order), figsize=(len(order)*2.15, len(subjects)*1.85), squeeze=False, layout="constrained")
        for i, subject in enumerate(subjects):
            for j, act in enumerate(order):
                ax = axes[i,j]
                sub = d[(d.subject == subject)&(d.action == act)]
                if len(sub):
                    ax.stairs(density(sub.duration_s, smooth=False), 10**LOG_EDGES, color="#5c9ec4", alpha=.6, fill=True, lw=.5)
                    ax.plot(10**LOG_CENTERS, density(sub.duration_s), color="#154b6b", lw=1.2)
                style_axis(ax, label(game,act) if i==0 else "", xlabel=False)
                ax.set_xlim(.005, 40)
                ax.set_xticks([.01, .1, 1, 10], [".01", ".1", "1", "10"])
                ax.tick_params(labelsize=8)
                ax.text(.96, .94, f"n={len(sub):,}", ha="right", va="top", transform=ax.transAxes, fontsize=7)
                if j==0:
                    ax.set_ylabel(f"Sub {subject:03d}\nLog-time density", fontsize=9)
                if i == len(subjects)-1:
                    ax.set_xlabel("Duration (s)", fontsize=9)
        fig.suptitle(f"{game}: subject × exact action", fontsize=16, weight="bold")
        save(fig, out, f"03_{SLUG[game]}_subject_actions")
        # Descriptive peak locations, not a statistical test of latent regimes.
        for act, sub in d.groupby("action"):
            for subject, sd in [("group", sub), *list(sub.groupby("subject"))]:
                curve = group_density(sd)[0] if subject == "group" else density(sd.duration_s)
                for bw in [.08, .10, .15]:
                    hist = group_density(sd, smooth=False)[0] if subject == "group" else density(sd.duration_s, smooth=False)
                    sm = gaussian_filter1d(hist, bw/BIN_WIDTH)
                    peaks, info = find_peaks(sm, prominence=max(sm)*.12, distance=max(1, int(.3/BIN_WIDTH)))
                    peak_s = 10**LOG_CENTERS[peaks]
                    modes.append(dict(game=game, action=int(act), action_name=label(game,act), subject=subject,
                        n_bouts=len(sd), bandwidth_dex=bw, n_visible_peaks=len(peaks),
                        peak_seconds=";".join(f"{v:.4g}" for v in peak_s)))
        switched = b[(b.game == game)&b.representation.eq("exact")&~b.right_censored]
        for (subject, source, target), sd in switched.groupby(["subject", "action", "next_action"]):
            transitions.append(dict(game=game, subject=subject, source=int(source), target=int(target),
                source_name=label(game,source), target_name=label(game,target), count=len(sd)))
    pd.DataFrame(modes).to_csv(out / "descriptive_log_density_peaks.csv", index=False)
    tr = pd.DataFrame(transitions)
    tr["conditional_probability"] = tr["count"] / tr.groupby(["game", "subject", "source"])["count"].transform("sum")
    tr.to_csv(out / "transition_counts_by_subject.csv", index=False)
    fig, axes = plt.subplots(2,2,figsize=(15,13),layout="constrained")
    grouptr = []
    for game, ax in zip(GAMES,axes.flat):
        order=displayed[game]
        gd=tr[tr.game.eq(game)]
        mat=np.zeros((len(order),len(order)))
        for i,src in enumerate(order):
            subjects=gd.loc[gd.source.eq(src),"subject"].unique()
            for j,dst in enumerate(order):
                vals=gd[(gd.source==src)&(gd.target==dst)].set_index("subject").conditional_probability.reindex(subjects,fill_value=0)
                mat[i,j]=vals.mean() if len(vals) else np.nan
                grouptr.append(dict(game=game, source=src,target=dst, probability=mat[i,j]))
        np.fill_diagonal(mat,np.nan)
        im=ax.imshow(mat,vmin=0,vmax=1,cmap="Blues")
        names=[label(game,a) for a in order]
        ax.set_xticks(range(len(order)),names,rotation=45,ha="right",fontsize=9)
        ax.set_yticks(range(len(order)),names,fontsize=9)
        for i in range(len(order)):
            for j in range(len(order)):
                if np.isfinite(mat[i,j]) and mat[i,j]>=.025:
                    ax.text(j,i,f"{100*mat[i,j]:.0f}",ha="center",va="center",fontsize=9,color="white" if mat[i,j]>.55 else "#253746")
        ax.set_title(game,weight="bold");ax.set_xlabel("Next exact action");ax.set_ylabel("Current exact action")
    fig.colorbar(im,ax=axes,label="P(next action | a switch occurred)",shrink=.65)
    fig.suptitle("Where actions switch to (equal subject weight)",fontsize=16,weight="bold")
    fig.supxlabel("Values are percentages. Mario displays nine common button sets; rare destinations remain in the denominator.",fontsize=10)
    save(fig,out,"04_switch_destinations")
    pd.DataFrame(grouptr).to_csv(out/"transition_group_displayed.csv",index=False)
    # Sensitivity to the canonical9 vocabulary and to onset/end censoring.
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    for game,ax in zip(GAMES,axes.flat):
        d=b[(b.game==game)&b.representation.eq("exact")]
        variants=[("Both ends observed",complete(d)),("Include boundary-start bouts",d[~d.right_censored]),
            ("Include all (censored = lower bound)",d[d.duration_s>0])]
        for name,sd in variants:
            y,_=group_density(sd);ax.plot(10**LOG_CENTERS,y,label=name,lw=1.8)
        if game==GAMES[-1]:
            sd=c[(c.game==game)&c.representation.eq("canonical9")]
            y,_=group_density(sd);ax.plot(10**LOG_CENTERS,y,label="Canonical9: both ends observed",ls="--",lw=1.6)
        style_axis(ax,game);ax.set_ylabel("Log-time density");ax.legend(fontsize=8,frameon=False)
    fig.suptitle("Sensitivity to boundaries and the Mario action vocabulary",fontsize=16,weight="bold")
    save(fig,out,"05_boundary_vocabulary_sensitivity")
    # Native sample counts match the unit in the user's historical NOOP figure.
    old_edges=np.geomspace(.95,5000,65)
    fig,axes=plt.subplots(3,6,figsize=(20,9),layout="constrained")
    for i,game in enumerate(GAMES[:3]):
        for j,subject in enumerate(range(1,7)):
            ax=axes[i,j]
            sd=b[(b.game==game)&(b.subject==subject)&b.action.eq(0)&~b.right_censored]
            ax.hist(sd.n_samples,bins=old_edges,color="#5c9ec4")
            ax.set_xscale("log");ax.set_xlim(1,5000);ax.grid(alpha=.2)
            ax.set_title(f"{game} / Sub {subject:03d}\nn={len(sd):,}",fontsize=10)
            ax.set_xlabel("Native recorded samples");ax.set_ylabel("Bout count")
    fig.suptitle("NOOP: native sample counts for comparison with the historical figure",fontsize=16,weight="bold")
    fig.supxlabel("Includes boundary-start bouts if followed by a switch; sample duration differs between games/files. No short/long threshold imposed.",fontsize=10)
    save(fig,out,"06_noop_native_samples")


def supplements(b, audit, out):
    """Compact action comparison, all Mario combinations, and input/plot audits."""
    check = extract(np.array([0,0,1,1,0,0,2,2]), np.arange(8)*.1,
        np.array([True,False,False,False,False,True,False,False]), {}, "exact")
    assert check.n_samples.sum() == 8
    assert np.allclose(complete(check).duration_s, [.2])
    assert check.right_censored.tolist() == [False,False,True,False,True]
    assert check.left_censored.tolist() == [True,False,False,True,False]
    exact = complete(b[b.representation.eq("exact")])
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    display_records = []
    for game, ax in zip(GAMES, axes.flat):
        d = exact[exact.game.eq(game)]
        counts = d.groupby("action").size().sort_index()
        # Main comparison includes actions contributing at least 1% of bouts.
        # All other actions are retained in the individual-action supplements.
        order = counts[counts >= len(d)*.01].index.tolist()
        palette = plt.get_cmap("tab20").colors
        for i, act in enumerate(order):
            sd = d[d.action.eq(act)]
            y, _ = group_density(sd)
            color = palette[(2*i) % 20 + i//10]
            ax.plot(10**LOG_CENTERS, y, lw=2, color=color,
                ls="-" if i < 10 else "--", label=label(game, act))
            display_records.append(dict(game=game, action=act, action_name=label(game,act),
                n_bouts=len(sd), fraction_of_bouts=len(sd)/len(d)))
        style_axis(ax, game)
        ax.set_xlim(.008, 30)
        ax.set_xticks([.01, .1, 1, 10], ["0.01", "0.1", "1", "10"])
        ax.set_ylabel("Density per log10(second)")
        ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=1)
    fig.suptitle("Action-specific persistence differs between games", fontsize=17, weight="bold")
    fig.supxlabel("Actions with >=1% of bouts shown; all others retained in supplements. Equal subject weight; smoothing = 0.10 log10(s).", fontsize=10)
    save(fig,out,"07_four_game_action_comparison")
    pd.DataFrame(display_records).to_csv(out/"main_figure_action_coverage.csv",index=False)

    game=GAMES[-1]
    d=exact[exact.game.eq(game)]
    actions=sorted(d.action.unique())
    fig,axes=plt.subplots((len(actions)+5)//6,6,figsize=(18,15),layout="constrained")
    for act,ax in zip(actions,axes.flat):
        sd=d[d.action.eq(act)];y,individual=group_density(sd);hist,_=group_density(sd,smooth=False)
        ax.stairs(hist,10**LOG_EDGES,fill=True,alpha=.35,color="#9db4c0",lw=.4)
        for curve in individual:
            ax.plot(10**LOG_CENTERS,curve,color="#5499c7",lw=.6,alpha=.2)
        ax.plot(10**LOG_CENTERS,y,color="#145a84",lw=1.5)
        style_axis(ax,"",xlabel=False);ax.set_xlim(.008,30)
        ax.set_xticks([.01,.1,1,10],[".01",".1","1","10"])
        ax.set_title(f"{label(game,act)}\nn={len(sd):,}",fontsize=9,loc="left")
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(actions):]:
        ax.axis("off")
    fig.suptitle(f"Super Mario Bros.: all {len(actions)} observed button combinations",fontsize=17,weight="bold")
    fig.supxlabel("Duration (seconds, log scale). Subject-normalized log-time densities. Rare combinations are descriptive only.",fontsize=11)
    save(fig,out,"08_mario_all_button_combinations")

    # Assess how the group NOOP curve changes when Mario sub001 is left out.
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    md=d[d.action.eq(0)]
    for subject,sd in md.groupby("subject"):
        axes[0].plot(10**LOG_CENTERS,density(sd.duration_s),label=f"Sub {subject:03d}",lw=1.8)
    for name,sd in [("All 5 subjects",md),("Without sub001",md[md.subject.ne(1)])]:
        y,_=group_density(sd);axes[1].plot(10**LOG_CENTERS,y,label=name,lw=2)
    for ax,title in zip(axes,["Mario NOOP: individual subjects","Mario NOOP: equal-weight mean"]):
        style_axis(ax,title);ax.set_xlim(.008,30);ax.set_ylabel("Log-time density");ax.legend(frameon=False,fontsize=9)
    save(fig,out,"09_mario_noop_subject_sensitivity")

    rows=[]
    for (game,subject),d in b[b.representation.eq("exact")].groupby(["game","subject"]):
        completed=complete(d);switched=d[~d.right_censored]
        rows.append(dict(game=game,subject=int(subject),n_bouts=len(completed),
            fraction_one_sample=float(completed.n_samples.eq(1).mean()),
            fraction_under_100ms=float(completed.duration_s.lt(.1).mean()),
            noop_bout_fraction=float(completed.action.eq(0).mean()),
            switch_involving_noop_fraction=float(((switched.action==0)|(switched.next_action==0)).mean())))
    switch_summary=pd.DataFrame(rows)
    switch_summary.to_csv(out/"switch_behavior_by_subject.csv",index=False)
    switch_summary.groupby("game").mean(numeric_only=True).drop(columns="subject").to_csv(out/"switch_behavior_group.csv")
    sd=exact[(exact.game==GAMES[-1])]
    stage_summary=sd.groupby(["subject","stage","action"]).agg(n_bouts=("duration_s","size"),
        median_seconds=("duration_s","median"),q90_seconds=("duration_s",lambda x: x.quantile(.9))).reset_index()
    stage_summary["action_name"]=stage_summary.action.map(raw_name)
    stage_summary.to_csv(out/"mario_stage_action_summary.csv",index=False)

    history=Path("/home/jeongmin/thinker/research_script/outputs/01_behavioral_analysis/atari/legacy_cross2/1-7_bout_short_long_summary.csv")
    historical_checks=[]
    if history.exists():
        for r in pd.read_csv(history).to_dict("records"):
            game={1:"Pong",2:"Space Invaders"}[r["game"]];subject=int(r["subject"].split("_")[-1])
            sd=b[(b.game==game)&(b.subject==subject)&b.action.eq(0)&~b.right_censored]
            historical_checks.append(dict(game=game,subject=subject,historical_count=r["n_bouts"],
                reproduced_count=len(sd),match=int(r["n_bouts"])==len(sd)))
        pd.DataFrame(historical_checks).to_csv(out/"historical_noop_reproduction.csv",index=False)
    ma=audit[audit.game.eq(GAMES[-1])]
    switch_checks=[]
    for (game,rep),sd in b.groupby(["game","representation"]):
        col="switches_exact" if rep=="exact" else "switches_canonical"
        expected=int(audit.loc[audit.game.eq(game),col].sum())
        actual=int((~sd.right_censored).sum())
        if actual!=expected:
            raise ValueError(f"Switch count mismatch {game} {rep}: {actual} != {expected}")
        switch_checks.append(dict(game=game,representation=rep,n_switches=actual,match=True))
    qc=dict(extraction_partition_and_boundary_synthetic_check="passed",
        switch_counts=switch_checks,historical_noop_counts_match_all=all(r["match"] for r in historical_checks),
        mario_raw_canonical_alignment_mismatches=int(ma.mapped_mismatch_frames.sum()),
        mario_frames_outside_canonical_vocabulary=int(ma.nonvocab_frames.sum()),
        mario_nonvocab_frame_fraction=float(ma.nonvocab_frames.sum()/ma.n_frames.sum()),
        mario_exact_switches=int(ma.switches_exact.sum()),mario_canonical_switches=int(ma.switches_canonical.sum()),
        mario_switches_lost_fraction=float(1-ma.switches_canonical.sum()/ma.switches_exact.sum()),
        complete_bouts_outside_plot_005_to_100_s=int(((exact.duration_s<.005)|(exact.duration_s>100)).sum()),
        complete_bouts_outside_histogram_range=int(((exact.duration_s<10**LOG_EDGES[0])|(exact.duration_s>10**LOG_EDGES[-1])).sum()))
    (out/"quality_checks.json").write_text(json.dumps(qc,indent=2))
    print(json.dumps(qc,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"research_script/outputs/01/action_bout_distributions_20261123")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--reuse-extraction",action="store_true")
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    if args.reuse_extraction:
        b=pd.read_pickle(out/"bouts.pkl.gz")
        audit=pd.read_csv(out/"source_audit.csv")
    else:
        tasks=tasks_for_data();frames=[];audits=[]
        print(f"Extracting {len(tasks)} human recordings",flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for i,(frame,audit) in enumerate(executor.map(process_source,tasks,chunksize=4),1):
                frames.append(frame);audits.append(audit)
                if i%100==0 or i==len(tasks):
                    print(f"{i}/{len(tasks)} recordings; {time.time()-started:.1f}s",flush=True)
        b=pd.concat(frames,ignore_index=True);audit=pd.DataFrame(audits)
        b.to_pickle(out/"bouts.pkl.gz",compression="gzip")
        b.to_csv(out/"bouts.csv.gz",index=False,compression="gzip")
        audit.to_csv(out/"source_audit.csv",index=False)
        print(f"Saved {len(b):,} bouts (including Mario canonical sensitivity)",flush=True)
    figures_and_modes(b,audit,out)
    supplements(b,audit,out)
    provenance=dict(status="COMPLETED",created_at=pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        elapsed_seconds=time.time()-started,source=str(Path(__file__)),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_files=len(audit),n_bout_rows=len(b),primary="exact action, observed onset and following switch",
        time_unit="seconds from stored time; Mario nominal 60 Hz, Atari irregular original timestamps",
        duration="time[next different action] - time[action onset]",
        boundaries="file, is_first, previous is_terminal, nonincreasing timestamp, gap > max(0.25 s, 10 median dt)",
        mario_alignment="BK2 patterns[1:1+N] exactly matches every NPZ canonical action",
        averaging="subject-normalized densities averaged equally; absent subject/action cells omitted",
        smoothing_dex=SMOOTH_DEX,peak_caution="Descriptive log-duration density peaks, not formal evidence for two latent regimes",
        image_arrays_read=False)
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2),flush=True)


if __name__=="__main__":
    main()
