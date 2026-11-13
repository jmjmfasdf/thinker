#!/usr/bin/env python3
"""Dynamic VoC, imagined returns, and human action transitions.

Read-only native-cache analysis, replacing the fixed-budget Actor readout in 03.
Q1: pre-choice VoC -> voluntary CONTINUE (PROCEED or RESET), depth and budget.
Q2: available imagined reward sums/bootstrapped returns -> VoC and its update.
Q3: initial and within-search VoC -> human switching, with return/budget controls.
Q4: session-held-out human prediction, aligned to the RL leave-session checkpoint.

EMA VoC is primary; online VoC is a separately labelled sensitivity analysis.
Model-predicted reward is never called observed utility. Forced STOP is censored
for gate analyses. No future imagination outcome predicts an earlier choice.
Default execution uses pooled within-participant regression with categorical
session fixed effects; see dynamic_voc_session_fixed.py. The native loader and
previous runwise methods below remain available for archived-result validation.
Use --runwise-legacy explicitly to execute that historical analysis.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "outputs/00_dynamic_cache"
DEFAULT_OUTPUT = ROOT / "outputs/03_dynamic_voc_mechanisms"
VERSION = "dynamic-voc-native-v1"
KEYS = ["run_label", "real_step", "real_step_id"]
GAMES = {0: "Enduro", 1: "Pong", 2: "Space Invaders"}
SOURCES = ("ema", "online")
POPULATIONS = ("overall", "noop_to_active", "active_to_noop", "active_to_active")
POP_LABELS = ("Any action change", "NOOP → active", "Active → NOOP", "Active → other active")
RETURN_METRICS = ("reward_sum", "discounted_reward", "bootstrap_return", "return_gain")
RETURN_LABELS = ("Undiscounted imagined rewards", "Discounted imagined rewards",
                 "Bootstrapped imagined return", "Return minus initial root value")
BASE_CONTROL = ["margin", "root_value", "root_entropy", "initial_q_margin",
                "log_age", "log_age_sq", "recent10", "recent30", "prior_reward",
                "prior_done", "run_progress", "run_progress_sq", "carried"]
SEARCH_CONTROL = ["log_budget", "max_depth", "forced_stop", "terminal_return_gain",
                  "terminal_discounted_reward", "mean_abs_return_update"]
COLORS = ("#386D9C", "#CF7B35", "#7B5EA7", "#16876B")


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_columns(path, columns):
    return pd.read_csv(path, usecols=list(dict.fromkeys(columns)), low_memory=False)


def session_of(run):
    m = re.search(r"_ses(\d+)_", str(run))
    if m is None:
        raise ValueError(f"Run lacks session identity: {run}")
    return int(m.group(1))


def validate_keys(frame, label, extra=()):
    keys = KEYS + list(extra)
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Invalid or duplicated {label} keys")


def masked(frame, name, valid):
    return pd.to_numeric(frame[name], errors="coerce").where(frame[valid].eq(1))


def action_controls(frame):
    return [c for c in frame if c.startswith("previous_action_")]


def discover(cache, subject, game, checkpoints):
    paths = sorted(Path(cache).glob("checkpoint_leave*"))
    if not paths and (Path(cache) / f"sub{subject:03d}_game{game}").exists():
        paths = [Path(cache)]
    found = []
    for parent in paths:
        p = parent / f"sub{subject:03d}_game{game}"
        if not p.exists():
            continue
        binding = json.loads((p / "checkpoint_binding.json").read_text())
        cp = int(binding["leave_session"])
        if checkpoints and cp not in checkpoints:
            continue
        if binding["status"] != "COMPLETED" or (binding["subject"], binding["game"]) != (subject, game):
            raise ValueError(f"Invalid checkpoint binding: {p}")
        if binding["session_roles"][str(cp)] != "held_out":
            raise ValueError(f"Invalid held-out role: {p}")
        manifest = json.loads((p / "manifest.json").read_text())
        if manifest["status"] != "COMPLETED":
            raise ValueError(f"Incomplete cache: {p}")
        if sha256(p / "manifest.json") != binding["cache_manifest_sha256"]:
            raise ValueError(f"Cache manifest differs from checkpoint binding: {p}")
        config_path = Path(binding["checkpoint"]) / "config_c.yaml"
        config = yaml.safe_load(config_path.read_text())
        found.append((cp, p, config, binding))
    if not found:
        raise FileNotFoundError(f"No bound dynamic caches for subject={subject}, game={game}")
    ids = [x[0] for x in found]
    if len(ids) != len(set(ids)) or (checkpoints and set(ids) != set(checkpoints)):
        raise ValueError(f"Incomplete/duplicate checkpoint selection: {ids}")
    return found


def reconstruct_reward_sums(events, gamma):
    """Undiscounted path rewards, reset at root/decision, checked against native
    discounted sums. Unknown carry/path jumps remain missing, never zero-filled.
    This follows the path's rewards, not the sum over all imagination events.
    """
    pre = np.full(len(events), np.nan)
    post = np.full(len(events), np.nan)
    cols = ["run_label", "real_step", "current_node_depth_pre", "current_node_depth_post",
            "root_trail_r_pre", "root_trail_r_post", "cur_r_post", "search_computation", "actor_reset"]
    last_key = None
    reward_sum, discounted, depth = np.nan, np.nan, -1
    failures = 0
    for i, row in enumerate(events[cols].itertuples(index=False, name=None)):
        run, step, before, after, rpre, rpost, reward, computation, reset = row
        key = (run, step)
        if key != last_key:
            reward_sum, discounted, depth = np.nan, np.nan, -1
            last_key = key
        if before == 0 and abs(rpre) < 1e-5:
            reward_sum, discounted, depth = 0.0, 0.0, 0
        elif before != depth or not np.isclose(rpre, discounted, atol=2e-4, rtol=2e-4):
            reward_sum, discounted, depth = np.nan, np.nan, -1
        pre[i] = reward_sum
        if computation == 1:
            if after == 0 and abs(rpost) < 1e-5:
                reward_sum, discounted, depth = 0.0, 0.0, 0
            elif reset != 1 and after == before + 1 and np.isfinite(reward_sum):
                predicted = discounted + gamma ** int(before) * reward
                if np.isclose(predicted, rpost, atol=2e-4, rtol=2e-4):
                    reward_sum += reward
                    discounted, depth = rpost, after
                else:
                    failures += 1
                    reward_sum, discounted, depth = np.nan, np.nan, -1
            else:
                reward_sum, discounted, depth = np.nan, np.nan, -1
        post[i] = reward_sum
    return pre, post, failures


def event_features(events, gamma):
    e = events.loc[events.is_control_choice.eq(1)].copy()
    e = e.sort_values(KEYS[:2] + ["event_order"], kind="stable").reset_index(drop=True)
    validate_keys(e, "control events", ["event_order"])
    if not e.accepted_control.isin([0, 1, 2]).all():
        raise ValueError("Unrecognized native control code")
    for source in SOURCES:
        e[f"voc_{source}"] = masked(e, f"voc_{source}_voc", f"voc_{source}_q_valid")
        expected = e[f"voc_{source}_q_continue"] - e[f"voc_{source}_q_stop"]
        good = e[f"voc_{source}_q_valid"].eq(1)
        if not np.allclose(e.loc[good, f"voc_{source}"], expected[good], atol=1e-6, rtol=1e-5):
            raise ValueError("VoC is not Q(continue)-Q(stop)")
    e["continue"] = e.accepted_control.ne(2).astype(float)
    if "voc_soft_continue_probability_valid" in e:
        e["voc_soft_continue_probability"] = masked(e, "voc_soft_continue_probability", "voc_soft_continue_probability_valid")
    # A legal CONTINUE can itself reach the computation cap. It remains an
    # observed continuation; the unobserved *next* stopping choice is censored.
    e["voluntary"] = (e.control_valid.eq(1) & e.forced_stop_valid.eq(1) &
                      (e.forced_stop.eq(0) | e.accepted_control.ne(2)))
    e["choice_index"] = e.groupby(KEYS, sort=False).cumcount()
    e["log_search_steps"] = np.log1p(e.search_steps_pre.clip(lower=0))
    e["depth"] = e.current_node_depth_pre
    e["discounted_reward"] = e.root_trail_r_pre
    e["bootstrap_return"] = e.rollout_return_pre
    e["bootstrap_tail"] = e.depth_discount_pre * e.cur_v_pre
    error = e.bootstrap_return - e.discounted_reward - e.bootstrap_tail
    finite = np.isfinite(error)
    if finite.any() and not np.allclose(error[finite], 0, atol=3e-4, rtol=0):
        raise ValueError(f"Return decomposition failed: max residual={error[finite].abs().max()}")
    e["reward_sum"], e["reward_sum_post"], failures = reconstruct_reward_sums(e, gamma)
    # Choices and all their pre/post states remain inside the same real decision.
    group = e.groupby(KEYS, sort=False)
    next_index = group.choice_index.shift(-1)
    e["has_next_choice"] = e["continue"].eq(1) & next_index.eq(e.choice_index + 1)
    e["return_update"] = e.rollout_return_post - e.rollout_return_pre
    for source in SOURCES:
        e[f"next_voc_{source}"] = group[f"voc_{source}"].shift(-1).where(e.has_next_choice)
        e[f"delta_voc_{source}"] = e[f"next_voc_{source}"] - e[f"voc_{source}"]
    qa = {"control_choices": len(e), "forced_choices": int(e.forced_stop.eq(1).sum()),
          "voluntary_choices": int(e.voluntary.sum()),
          "return_identity_max_error": float(error.abs().max()) if finite.any() else None,
          "reward_path_validation_failures": failures,
          "reward_sum_coverage": float(e.reward_sum.notna().mean())}
    return e, qa


def validate_endings(e, decisions):
    last = e.groupby(KEYS, sort=False).tail(1)
    counts = e.assign(observed_stop=e.accepted_control.eq(2)).groupby(KEYS).observed_stop.sum()
    if counts.gt(1).any():
        raise ValueError("Multiple STOP choices in one decision")
    terminal = last[KEYS + ["accepted_control"]].merge(
        decisions[KEYS + ["forced_stop", "stop_choice_censored"]], on=KEYS, validate="one_to_one")
    stopped = terminal.accepted_control.eq(2)
    censored = terminal.forced_stop.eq(1) & terminal.stop_choice_censored.eq(1)
    if not (stopped | censored).all():
        raise ValueError("A decision ends without either STOP or recorded cap censoring")
    return int((~stopped & censored).sum())


def add_behavior_controls(d):
    d = d.sort_values(["run_label", "real_step"], kind="stable").reset_index(drop=True)
    incumbent = d.incumbent_action.to_numpy(dtype=int)
    n_actions = len([c for c in d if c.startswith("model_root_p_")])
    d["n_actions"] = n_actions
    probs = d[[f"model_root_p_{a}" for a in range(n_actions)]].to_numpy(float)
    if not np.all(np.isfinite(probs)) or (probs < 0).any() or not np.allclose(probs.sum(1), 1, atol=1e-4):
        raise ValueError("Invalid pre-search model policy")
    logp = np.log(np.clip(probs, 1e-10, 1))
    other = logp.copy()
    valid = (incumbent >= 0) & (incumbent < n_actions)
    rows = np.arange(len(d))
    other[rows, incumbent.clip(0, n_actions-1)] = -np.inf
    d["margin"] = np.where(valid, logp[rows, incumbent.clip(0, n_actions-1)] - other.max(1), np.nan)
    d["root_entropy"] = -(probs * logp).sum(1)
    q = d[[f"root_qs_mean_initial_{a}" for a in range(n_actions)]].to_numpy(float)
    visits = d[[f"root_ns_initial_{a}" for a in range(n_actions)]].to_numpy(float)
    q[visits <= 0] = np.nan
    own = q[rows, incumbent.clip(0, n_actions-1)].copy()
    q[rows, incumbent.clip(0, n_actions-1)] = np.nan
    best = np.max(np.where(np.isfinite(q), q, -np.inf), axis=1)
    d["initial_q_margin"] = np.where(np.isfinite(best) & valid, own - best, np.nan)
    for a in range(n_actions):
        d[f"log_policy_{a}"] = logp[:, a]
    d["human_switch"] = d.human_action.ne(d.incumbent_action).astype(float)
    # Episode boundaries and gaps reset history. The current outcome is excluded.
    segment = (d.run_label.ne(d.run_label.shift()) | d.episode_id.ne(d.episode_id.shift()) |
               d.real_step.diff().ne(1) | d.prior_done.eq(1)).cumsum()
    past = d.human_switch.groupby(segment, sort=False).shift(1)
    for width, label in ((10, "recent10"), (30, "recent30")):
        d[label] = past.groupby(segment, sort=False).transform(lambda x: x.rolling(width, min_periods=1).sum()).fillna(0)
    d["log_age"] = np.log1p(d.bout_age.clip(lower=0))
    d["log_age_sq"] = d.log_age ** 2
    d["run_progress_sq"] = d.run_progress ** 2
    for a in range(1, n_actions):
        d[f"previous_action_{a}"] = d.incumbent_action.eq(a).astype(float)
    d["carried"] = masked(d, "root_carried", "root_carried_valid")
    d["log_budget"] = np.log1p(d.additional_computations.clip(lower=0))
    d["max_depth"] = d.max_path_depth
    return d


def load_cache(path, cp, config, subject, game):
    n_actions = int(json.loads((path / "manifest.json").read_text())["num_actions"])
    decision_cols = KEYS + ["episode_id", "human_action", "first_control_native_event",
        "last_control_native_event", "real_action_native_event", "additional_computations",
        "max_path_depth", "forced_stop", "stop_choice_censored", "root_carried", "root_carried_valid"]
    for s in SOURCES:
        for suffix in ("initial", "terminal"):
            decision_cols += [f"voc_{s}_voc_{suffix}", f"voc_{s}_voc_{suffix}_valid"]
    decision_cols += [f"root_{kind}_initial_{a}" for kind in ("qs_mean", "ns") for a in range(n_actions)]
    d = read_columns(path / "native_decisions.csv.gz", decision_cols)
    model_cols = KEYS + ["human_action", "incumbent_action", "root_value", "bout_age", "prior_reward",
                        "prior_done", "run_progress", "scored"] + [f"model_root_p_{a}" for a in range(n_actions)]
    model = read_columns(path / "model_control_windows.csv.gz", model_cols)
    validate_keys(d, "decisions"); validate_keys(model, "model windows")
    if len(d) != len(model):
        raise ValueError("Decision/model row counts disagree")
    d = d.merge(model, on=KEYS, validate="one_to_one", suffixes=("", "_model"), how="left")
    if not d.human_action.eq(d.human_action_model).all():
        raise ValueError("Native and behavioral action alignment disagree")
    if not ((d.first_control_native_event <= d.last_control_native_event) &
            (d.last_control_native_event < d.real_action_native_event)).all():
        raise ValueError("Search is not strictly before human action replay")
    for source in SOURCES:
        for suffix in ("initial", "terminal"):
            d[f"{source}_{suffix}"] = masked(d, f"voc_{source}_voc_{suffix}", f"voc_{source}_voc_{suffix}_valid")
        d[f"{source}_change"] = d[f"{source}_terminal"] - d[f"{source}_initial"]
    d = add_behavior_controls(d)
    alignment = read_columns(path / "visual_alignment.csv.gz", KEYS + ["actor_row"])
    validate_keys(alignment, "visual alignment")
    with np.load(path / "visual_controls.npz", allow_pickle=False) as z:
        visual = z["descriptor"]
    if len(alignment) != len(visual) or alignment.actor_row.duplicated().any():
        raise ValueError("Invalid visual control mapping")
    d = d.merge(alignment, on=KEYS, validate="one_to_one", how="left")
    if d.actor_row.isna().any():
        raise ValueError("Missing visual rows")
    rows = d.actor_row.to_numpy(int)
    if (rows < 0).any() or (rows >= len(visual)).any():
        raise ValueError("Visual row outside descriptor matrix")
    visual_names = [f"visual_{i}" for i in range(visual.shape[1])]
    d[visual_names] = visual[rows]
    event_cols = KEYS + ["event_order", "native_event", "is_control_choice", "accepted_control", "control_valid",
        "forced_stop", "forced_stop_valid", "search_steps_pre", "search_computation", "actor_reset",
        "current_node_depth_pre", "current_node_depth_post", "root_trail_r_pre", "root_trail_r_post",
        "rollout_return_pre", "rollout_return_post", "cur_r_post", "cur_v_pre", "depth_discount_pre",
        "voc_soft_continue_probability", "voc_soft_continue_probability_valid"]
    for source in SOURCES:
        event_cols += [f"voc_{source}_voc", f"voc_{source}_q_valid", f"voc_{source}_q_continue", f"voc_{source}_q_stop"]
    chunks = []
    for chunk in pd.read_csv(path / "native_events.csv.gz", usecols=event_cols, chunksize=200000):
        chunks.append(chunk.loc[chunk.is_control_choice.eq(1)].copy())
    e, qa = event_features(pd.concat(chunks, ignore_index=True), float(config["discounting"]))
    first = e.groupby(KEYS, sort=False).head(1)
    last = e.groupby(KEYS, sort=False).tail(1)
    qa["censored_without_stop_choice"] = validate_endings(e, d)
    paired = d.merge(first[KEYS + ["native_event", "voc_ema", "voc_online"]], on=KEYS, validate="one_to_one")
    if len(paired) != len(d) or not paired.native_event.eq(paired.first_control_native_event).all():
        raise ValueError("First control-choice alignment failed")
    for source in SOURCES:
        if not np.allclose(paired[f"{source}_initial"], paired[f"voc_{source}"], equal_nan=True, atol=1e-6):
            raise ValueError("Decision initial VoC differs from first native choice")
    counts = e.groupby(KEYS, as_index=False).search_computation.sum().rename(columns={"search_computation":"native_budget"})
    check = d.merge(counts, on=KEYS, validate="one_to_one")
    if not check.additional_computations.eq(check.native_budget).all():
        raise ValueError("Native search count and decision budget disagree")
    terminal = last[KEYS + ["native_event", "bootstrap_return", "discounted_reward", "reward_sum"]].rename(columns={
        "native_event":"terminal_native_event", "bootstrap_return":"terminal_return",
        "discounted_reward":"terminal_discounted_reward", "reward_sum":"terminal_reward_sum"})
    d = d.merge(terminal, on=KEYS, validate="one_to_one", how="left")
    if not d.terminal_native_event.eq(d.last_control_native_event).all():
        raise ValueError("Terminal search alignment failed")
    terminal_voc = d.merge(last[KEYS + ["voc_ema", "voc_online"]], on=KEYS, validate="one_to_one")
    for source in SOURCES:
        if not np.allclose(terminal_voc[f"{source}_terminal"], terminal_voc[f"voc_{source}"], equal_nan=True, atol=1e-6):
            raise ValueError("Decision terminal VoC differs from terminal native choice")
        # Terminal VoC at a voluntary STOP is undefined for cap-ended decisions.
        d.loc[d.stop_choice_censored.eq(1) | d.forced_stop.eq(1), f"{source}_change"] = np.nan
    updates = e.loc[e.search_computation.eq(1), KEYS + ["return_update"]].copy()
    updates["mean_abs_return_update"] = updates.return_update.abs()
    updates = updates.groupby(KEYS, as_index=False).mean_abs_return_update.mean()
    d = d.merge(updates, on=KEYS, how="left", validate="one_to_one")
    d.loc[d.additional_computations.eq(0), "mean_abs_return_update"] = 0.0
    d["terminal_return_gain"] = d.terminal_return - d.root_value
    shared = KEYS + BASE_CONTROL + action_controls(d) + ["human_switch", "human_action", "incumbent_action", "scored", "n_actions"]
    e = e.merge(d[shared], on=KEYS, validate="many_to_one", how="left")
    e["return_gain"] = e.bootstrap_return - e.root_value
    for frame in (d, e):
        frame["subject"], frame["game"], frame["checkpoint"] = subject, game, cp
        frame["session"] = frame.run_label.map(session_of)
        frame["held_out"] = frame.session.eq(cp)
    qa.update({"decisions": len(d), "runs": int(d.run_label.nunique()), "num_actions":n_actions,
               "zero_computation_fraction": float(d.additional_computations.eq(0).mean()),
               "forced_decision_fraction": float(d.forced_stop.eq(1).mean()),
               "max_budget": int(d.additional_computations.max()), "max_depth": int(d.max_depth.max()),
               "checkpoint": cp, "cache": str(path)})
    return d, e, qa


def prepare_matrix(frame, columns, train):
    """Training-only imputation, missing indicators, scaling and variable removal."""
    x = frame[columns].to_numpy(float)
    x[~np.isfinite(x)] = np.nan
    present = np.isfinite(x[train]).any(0)
    if not present.any():
        return np.empty((len(frame), 0)), []
    x = x[:, present]
    names = np.asarray(columns)[present].tolist()
    missing = ~np.isfinite(x)
    median = np.nanmedian(x[train], axis=0)
    x = np.where(missing, median, x)
    indicators = missing[train].any(0)
    if indicators.any():
        names += [f"{name}_missing" for name, keep in zip(np.asarray(columns)[present], indicators) if keep]
        x = np.column_stack([x, missing[:, indicators].astype(float)])
    mean, scale = x[train].mean(0), x[train].std(0)
    keep = scale > 1e-10
    return (x[:, keep] - mean[keep]) / scale[keep], np.asarray(names)[keep].tolist()


def fit_association(frame, target, predictors, controls, binary):
    required = [target] + predictors
    d = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    result = {p: np.nan for p in predictors}
    if len(d) < 80 or d[target].nunique() < 2:
        return result, len(d), "insufficient"
    columns = list(dict.fromkeys(predictors + controls))
    x, names = prepare_matrix(d, columns, np.ones(len(d), dtype=bool))
    if x.shape[1] == 0:
        return result, len(d), "constant"
    y = d[target].to_numpy(float)
    if binary and min(np.sum(y == 0), np.sum(y == 1)) < 5:
        return result, len(d), "rare_outcome"
    if binary:
        fit = LogisticRegression(C=1.0, max_iter=100, solver="newton-cholesky", tol=1e-6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            fit.fit(x, y)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            return result, len(d), "nonconverged"
        beta = fit.coef_[0]
    else:
        y = (y - y.mean()) / y.std()
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1:]
    for p in predictors:
        if p in names:
            result[p] = float(beta[names.index(p)])
    return result, len(d), "ok"


def spearman(x, y):
    d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 30 or d.x.nunique() < 2 or d.y.nunique() < 2:
        return np.nan
    return float(stats.spearmanr(d.x, d.y).statistic)


def risk_set(frame, population):
    n_actions = frame.n_actions if "n_actions" in frame else 6
    valid = frame.incumbent_action.ge(0) & frame.incumbent_action.lt(n_actions) & frame.human_action.ge(0) & frame.human_action.lt(n_actions) & frame.scored.eq(1)
    if population == "overall":
        keep, y = valid, frame.human_switch
    elif population == "noop_to_active":
        keep, y = valid & frame.incumbent_action.eq(0), frame.human_action.ne(0).astype(float)
    elif population == "active_to_noop":
        keep = valid & frame.incumbent_action.ne(0) & (frame.human_action.eq(0) | frame.human_action.eq(frame.incumbent_action))
        y = frame.human_action.eq(0).astype(float)
    elif population == "active_to_active":
        keep, y = valid & frame.incumbent_action.ne(0) & frame.human_action.ne(0), frame.human_switch
    else:
        raise ValueError(population)
    d = frame.loc[keep].copy()
    d["outcome"] = y.loc[keep]
    return d


def base_columns(d):
    return BASE_CONTROL + action_controls(d) + [c for c in d if c.startswith("log_policy_")] + [c for c in d if c.startswith("visual_")]


def analyze_associations(d, e):
    effects, curves = [], []
    keys = ["subject", "game", "checkpoint", "run_label", "session", "held_out"]
    def record(meta, family, source, population, term, value, n, status="ok", kind="beta"):
        effects.append(dict(meta, family=family, source=source, population=population, term=term,
                            value=value, n_rows=n, status=status, kind=kind))
    def fit(meta, data, family, source, population, target, predictors, controls, binary):
        values, n, status = fit_association(data, target, predictors, controls, binary)
        for term, val in values.items():
            record(meta, family, source, population, term, val, n, status)
    def curve(meta, data, family, source, population, x, outcomes):
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[x]).copy()
        if len(data) < 30 or data[x].nunique() < 2:
            return
        # Mid-ranks do not split identical VoC values across quantiles.
        data["bin"] = np.minimum(((data[x].rank(method="average") - .5) / len(data) * 7).astype(int), 6)
        for b, part in data.groupby("bin", sort=True):
            for outcome in outcomes:
                good = part[[x, outcome]].dropna()
                if len(good):
                    curves.append(dict(meta, family=family, source=source, population=population, bin=int(b),
                                       x_name=x, outcome=outcome, x=float(good[x].mean()), y=float(good[outcome].mean()), n=len(good)))
    for run, decisions in d.groupby("run_label", sort=False):
        meta = {k: decisions[k].iloc[0].item() if isinstance(decisions[k].iloc[0], np.generic) else decisions[k].iloc[0] for k in keys}
        events = e.loc[e.run_label.eq(run)]
        voluntary = events.loc[events.voluntary & events.scored.eq(1)]
        event_controls = ["margin", "root_value", "root_entropy", "carried", "depth", "log_search_steps"] + action_controls(d)
        for source in SOURCES:
            voc = f"voc_{source}"
            for population, part in (("all_choices", voluntary), ("first_choice", voluntary.loc[voluntary.choice_index.eq(0)]),
                                     ("after_computation", voluntary.loc[voluntary.search_steps_pre.gt(0)])):
                fit(meta, part, "Q1_gate", source, population, "continue", [voc], event_controls + ["return_gain"], True)
                curve(meta, part, "Q1_gate", source, population, voc, ["continue", "voc_soft_continue_probability"])
            for outcome in ("additional_computations", "max_depth"):
                value = spearman(decisions[f"{source}_initial"], decisions[outcome])
                record(meta, "Q1_allocation", source, "all_decisions", outcome, value, len(decisions), kind="spearman")
            curve(meta, decisions, "Q1_allocation", source, "all_decisions", f"{source}_initial", ["additional_computations", "max_depth"])
            searched = voluntary.loc[voluntary.search_steps_pre.gt(0)]
            for metric in RETURN_METRICS:
                fit(meta, searched, "Q2_return", source, "after_computation", voc, [metric], event_controls, False)
                curve(meta, searched, "Q2_return", source, "after_computation", metric, [voc])
                # Fixed-depth sensitivity separates return magnitude from horizon.
                for depth in (1, 2, 3):
                    fixed_depth = searched.loc[searched.depth.eq(depth)]
                    fit(meta, fixed_depth, "Q2_fixed_depth", source, f"depth_{depth}", voc, [metric], event_controls, False)
            updates = events.loc[events.has_next_choice & events.search_computation.eq(1) & events.scored.eq(1)]
            fit(meta, updates, "Q2_update", source, "executed_computations", f"delta_voc_{source}",
                ["return_update"], event_controls + [voc, "bootstrap_return"], False)
            for population in POPULATIONS:
                part = risk_set(decisions, population)
                part = part.loc[part.stop_choice_censored.eq(0) & part.forced_stop.eq(0)]
                predictors = [f"{source}_initial", f"{source}_change"]
                fit(meta, part, "Q3_human", source, population, "outcome", predictors,
                    base_columns(d) + SEARCH_CONTROL, True)
                curve(meta, part, "Q3_human", source, population, f"{source}_initial", ["outcome"])
    return pd.DataFrame(effects), pd.DataFrame(curves)


def prediction_blocks(d):
    baseline = base_columns(d)
    return {"B0": baseline, "B0_VoC": baseline + ["ema_initial", "ema_change"],
            "B1": baseline + SEARCH_CONTROL,
            "B2": baseline + SEARCH_CONTROL + ["ema_initial", "ema_change"]}


def fit_predict(d, y, columns, train, test):
    x, names = prepare_matrix(d, columns, train)
    if len(np.unique(y[train])) < 2 or train.sum() < 100:
        return None, {"status": "insufficient_training"}
    model = LogisticRegression(C=1.0, solver="newton-cholesky", max_iter=100, tol=1e-6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x[train], y[train])
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        return None, {"status": "nonconverged"}
    return model.predict_proba(x[test])[:, 1], {"status": "ok", "n_train": int(train.sum()),
                                              "n_test": int(test.sum()), "n_features": len(names)}


def predict_heldout_session(d):
    """Both RL checkpoint and behavioral readout exclude the whole test session.
    B0..B2 use exactly the same train/test decision rows. No test-fitted scaling.
    """
    metrics, predictions, qa = [], [], []
    blocks = prediction_blocks(d)
    cp = int(d.checkpoint.iloc[0])
    for population in POPULATIONS:
        part = risk_set(d, population)
        part = part.loc[part.ema_initial.notna() & part.ema_change.notna()].reset_index(drop=True)
        test = part.session.eq(cp).to_numpy()
        train = ~test
        if not test.any() or not train.any():
            qa.append(dict(checkpoint=cp, population=population, model="all", status="missing_train_or_test"))
            continue
        y = part.outcome.to_numpy(float)
        all_p = {}
        for name, columns in blocks.items():
            pred, audit = fit_predict(part, y, columns, train, test)
            qa.append(dict(checkpoint=cp, population=population, model=name, **audit))
            if pred is not None:
                all_p[name] = pred
        if len(all_p) != len(blocks):
            continue
        scored = part.loc[test, KEYS + ["subject", "game", "checkpoint", "session", "held_out", "outcome"]].copy()
        scored["population"] = population
        for name, p in all_p.items():
            scored[f"p_{name}"] = p
        predictions.append(scored)
        for run, x in scored.groupby("run_label", sort=False):
            truth = x.outcome.to_numpy()
            for name in blocks:
                p = x[f"p_{name}"].to_numpy()
                metrics.append({k: x[k].iloc[0] for k in ["subject", "game", "checkpoint", "session", "held_out"]} | {
                    "run_label":run, "population":population, "model":name, "n_test":len(x), "n_events":int(truth.sum()),
                    "log_loss":log_loss(truth, p, labels=[0, 1]), "brier":brier_score_loss(truth, p),
                    "auroc":roc_auc_score(truth, p) if len(np.unique(truth)) == 2 else np.nan})
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(), qa


def prediction_effects(metrics):
    records = []
    keys = ["subject", "game", "checkpoint", "session", "held_out", "run_label", "population"]
    for key, part in metrics.groupby(keys, sort=False):
        values = part.set_index("model")
        for a, b in (("B0", "B0_VoC"), ("B0", "B1"), ("B1", "B2")):
            for metric in ("log_loss", "auroc", "brier"):
                v = float(values.loc[a, metric] - values.loc[b, metric])
                if metric == "auroc": v = -v
                records.append(dict(zip(keys, key)) | {"family":"Q4_prediction", "source":"ema", "term":f"{a}_to_{b}:{metric}",
                    "kind":metric, "value":v, "status":"ok", "n_rows":int(values.loc[a, "n_test"])})
    return pd.DataFrame(records)


def exact_sign_flip(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    if len(x) > 16:
        raise ValueError("Inference requires run/subject units, not event rows")
    patterns = np.asarray(list(itertools.product([-1, 1], repeat=len(x))))
    return float(np.mean(np.abs(patterns @ x) >= abs(x.sum()) - 1e-12))


def bh(p):
    p = np.asarray(p, float); out = np.full(len(p), np.nan)
    good = np.flatnonzero(np.isfinite(p))
    order = good[np.argsort(p[good])]
    if len(order):
        q = p[order] * len(order) / np.arange(1, len(order) + 1)
        out[order] = np.minimum(1, np.minimum.accumulate(q[::-1])[::-1])
    return out


def inference(effects, unit="subject"):
    records = []
    identities = ["game", "family", "source", "population", "term", "kind"]
    for estimand in ("held_out", "all_checkpoints"):
        data = effects.loc[effects.held_out.eq(1)] if estimand == "held_out" else effects
        if estimand == "all_checkpoints":
            data = data.loc[~data.family.eq("Q4_prediction")]
        per_run = data.groupby(["subject", "run_label", *identities], dropna=False).value.mean().reset_index()
        collapsed = per_run.groupby(["subject", *identities], dropna=False).value.mean().reset_index() if unit == "subject" else per_run
        for key, part in collapsed.groupby(identities, dropna=False, sort=False):
            values = part.value.dropna().to_numpy()
            n = len(values)
            mean = float(values.mean()) if n else np.nan
            ci = float(stats.t.ppf(.975, n-1) * stats.sem(values)) if n > 1 else np.nan
            records.append(dict(zip(identities, key)) | {"estimand":estimand, "unit":unit,
                "mean":mean, "ci_low":mean-ci, "ci_high":mean+ci, "n_units":n,
                "n_positive":int((values > 0).sum()), "n_negative":int((values < 0).sum()),
                "p":exact_sign_flip(values)})
    result = pd.DataFrame(records)
    if result.empty: return result
    result["q"] = np.nan
    # Identical families in CSVs and figures. Prediction primary = log loss;
    # AUROC/Brier are separately labelled sensitivity families.
    for _, ix in result.groupby(["game", "family", "source", "estimand", "kind"], dropna=False).groups.items():
        result.loc[ix, "q"] = bh(result.loc[ix, "p"])
    return result


def unit_curves(curves, unit):
    keys = ["family", "source", "population", "bin", "x_name", "outcome"]
    c = curves.loc[curves.held_out.eq(1)]
    per_run = c.groupby(["subject", "run_label", *keys])[["x", "y"]].mean().reset_index()
    if unit == "subject":
        return per_run.groupby(["subject", *keys])[["x", "y"]].mean().reset_index()
    return per_run


def draw_curve(ax, data, label, color):
    if data.empty:
        return
    grain = "run_label" if "run_label" in data else "subject"
    for _, part in data.groupby(grain):
        part = part.sort_values("bin")
        ax.plot(part.x, part.y, color=color, alpha=.18, lw=.8)
    summary = data.groupby("bin")[["x", "y"]].mean()
    ax.plot(summary.x, summary.y, "o-", color=color, lw=2, ms=4, label=label)


def draw_effects(ax, table, terms, labels):
    available = table.loc[table.term.isin(terms), "mean"]
    if available.empty or not np.isfinite(available).any():
        ax.text(.5,.5,"Not estimable",ha="center",va="center",transform=ax.transAxes)
    for i, (term, label) in enumerate(zip(terms, labels)):
        row = table.loc[table.term.eq(term)]
        if row.empty: continue
        r = row.iloc[0]
        if not np.isfinite(r["mean"]): continue
        ax.errorbar(r["mean"], i, xerr=[[r["mean"]-r.ci_low], [r.ci_high-r["mean"]]], fmt="o", color=COLORS[i % 4], capsize=3)
        label += f"  q={r.q:.3f}, n={int(r.n_units)}"
        ax.text(.02, i+.18, label, transform=ax.get_yaxis_transform(), fontsize=8, va="bottom")
    ax.axvline(0, color=".5", ls="--", lw=.8)
    ax.set_yticks([]); ax.set_ylim(-.6, len(terms)+.1)


def render(fig_dir, effects, curves, summary, game, unit):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    c = unit_curves(curves, unit)
    s = summary.loc[summary.estimand.eq("held_out") & summary.source.eq("ema")]
    def save(fig, name):
        fig.suptitle(f"{GAMES[game]} · held-out sessions · EMA VoC", fontsize=14)
        fig.text(.02, .01, f"Thin lines: {unit}s. Thick lines: equal-{unit} means. Coefficients: mean ± 95% t interval; q from saved analysis-family FDR.", fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.96))
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir/f"{name}.{ext}", dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    fig, grid = plt.subplots(2, 3, figsize=(16, 9))
    axs = grid[0]
    for i, population in enumerate(("first_choice", "after_computation")):
        for outcome, label, color in (("continue", "Executed CONTINUE", COLORS[0]), ("voc_soft_continue_probability", "Recorded soft gate probability", COLORS[1])):
            part = c.loc[c.family.eq("Q1_gate") & c.source.eq("ema") & c.population.eq(population) & c.outcome.eq(outcome)]
            draw_curve(axs[i], part, label, color)
        axs[i].set(xlabel="VoC before choice", ylabel="Probability", title=population.replace("_", " "))
        axs[i].legend(fontsize=8)
    for metric, label, color in (("additional_computations", "Computation count", COLORS[0]), ("max_depth", "Maximum path depth", COLORS[1])):
        draw_curve(axs[2], c.loc[c.family.eq("Q1_allocation") & c.source.eq("ema") & c.outcome.eq(metric)], label, color)
    axs[2].set(xlabel="Initial VoC", ylabel="Count / depth", title="Executed search (caps retained)"); axs[2].legend(fontsize=8)
    for i, pop in enumerate(("first_choice", "after_computation")):
        selected = s.loc[s.family.eq("Q1_gate") & s.population.eq(pop)]
        draw_effects(grid[1,i], selected, ["voc_ema"], ["VoC before choice"])
        grid[1,i].set(title="Adjusted voluntary CONTINUE", xlabel="Log-odds per SD")
    draw_effects(grid[1,2], s.loc[s.family.eq("Q1_allocation")], ["additional_computations", "max_depth"], ["Computation count", "Maximum path depth"])
    grid[1,2].set(title="Initial VoC and executed allocation", xlabel="Within-run Spearman correlation")
    save(fig, "figure1_voc_and_search_continuation")
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, metric, label in zip(axs.flat, RETURN_METRICS, RETURN_LABELS):
        draw_curve(ax, c.loc[c.family.eq("Q2_return") & c.source.eq("ema") & c.x_name.eq(metric)], "Available before choice", COLORS[0])
        ax.set(xlabel=label, ylabel="VoC", title="After ≥1 computation")
    draw_effects(axs[1,1], s.loc[s.family.eq("Q2_return")], list(RETURN_METRICS), ["Reward sum", "Discounted reward", "Bootstrapped return", "Gain over root value"])
    axs[1,1].set_title("Adjusted return–VoC relationships"); axs[1,1].set_xlabel("Standardized slope")
    draw_effects(axs[1,2], s.loc[s.family.eq("Q2_update")], ["return_update"], ["Return update → next VoC update"])
    axs[1,2].set_title("Strictly subsequent VoC"); axs[1,2].set_xlabel("Standardized slope")
    save(fig, "figure2_voc_and_imagined_returns")
    fig, axs = plt.subplots(2, 4, figsize=(20, 8.5))
    for j, (pop, label) in enumerate(zip(POPULATIONS, POP_LABELS)):
        draw_curve(axs[0,j], c.loc[c.family.eq("Q3_human") & c.source.eq("ema") & c.population.eq(pop)], "Unadjusted", COLORS[0])
        axs[0,j].set(xlabel="Initial VoC", ylabel="Human transition probability", title=label)
        draw_effects(axs[1,j], s.loc[s.family.eq("Q3_human") & s.population.eq(pop)], ["ema_initial", "ema_change"], ["Initial VoC", "Terminal − initial VoC"])
        axs[1,j].set(xlabel="Adjusted log-odds per SD", title="With initial state, return, budget and visual controls")
    save(fig, "figure3_voc_and_human_transitions")
    pred = s.loc[s.family.eq("Q4_prediction")]
    if not pred.empty:
        fig, axs = plt.subplots(2, 4, figsize=(20, 9))
        for j, (pop, label) in enumerate(zip(POPULATIONS, POP_LABELS)):
            for i, metric in enumerate(("log_loss", "auroc")):
                terms = [f"{a}:{metric}" for a in ("B0_to_B0_VoC", "B0_to_B1", "B1_to_B2")]
                draw_effects(axs[i,j], pred.loc[pred.population.eq(pop)], terms, ["VoC beyond baseline", "Return/search beyond baseline", "VoC beyond return/search"])
                axs[i,j].set(title=label, xlabel=("Held-out log-loss reduction" if i == 0 else "Held-out AUROC increase"))
        save(fig, "figure4_heldout_human_prediction")


def write_readme(out):
    (out/"README.md").write_text("""# Analysis 03: dynamic VoC and human action transitions

Primary population: each leave-session checkpoint's held-out human sessions.
EMA VoC is primary; online VoC is a sensitivity analysis. The recorded soft
gate probability is not assumed to equal the actual execution probability in
epsilon-greedy or other projected policies; inspect manifest gate flags.

1. Figure 1: native pre-choice VoC versus voluntary CONTINUE (PROCEED/RESET),
   initial VoC versus observed computation count and maximum path depth.
   Forced STOP choices are excluded from the gate response. A legal CONTINUE
   that reaches the cap is retained as CONTINUE; the next choice is censored.
   Capped budgets
   remain observed, capped counts; these correlations are not latent duration estimates.
2. Figure 2: VoC versus available imagined undiscounted reward sum,
   discounted reward sum, bootstrapped return, and return minus initial root
   value. Return–VoC regressions adjust initial margin/value, depth, elapsed
   computation, carry and action identity. Return updates are paired only with
   the next control choice in the SAME real decision.
   Q2_fixed_depth tables repeat the comparisons at depths 1, 2 and 3.
3. Figure 3: human switching overall and three conditional transition contrasts.
   Initial VoC and terminal-minus-initial VoC are adjusted for initial policy,
   value, behavior duration/history, reward, visual descriptors and search
   budget/depth/returns. Active-to-NOOP excludes competing active-to-active;
   active-to-active excludes active-to-NOOP. Overall includes both.
   These terminal-VoC analyses exclude cap-censored decisions because no
   voluntary STOP evaluation was observed. Exclusion counts are saved in QA.
4. Figure 4: logistic readout and RL checkpoint both exclude the test SESSION.
   B0=pre-search/behavior/visual controls; B0_VoC=B0+VoC; B1=B0+search returns,
   budget/depth; B2=B1+VoC. Identical decision rows for all four models.
   Primary metric: log loss; AUROC and Brier are sensitivity metrics.

VoC = Q(CONTINUE)-Q(STOP). Predicted imagined returns are NOT actual realized
utility or causal STOP-versus-CONTINUE benefits. Root trailing reward is the
discounted reward sum; rollout return additionally includes gamma^depth V.
Undiscounted path rewards are reconstructed only where sequential depth and
native discounted reward checks succeed. Resets restart the path, real actions
end it, and unidentifiable path transitions remain missing.

All regressions are per checkpoint/run. Binary regressions use standardized
predictors and fixed L2 C=1, with convergence auditing. No per-event p-values.
Checkpoint effects average WITHIN run, runs average within subject; group
inference is an exact two-sided sign flip over subjects. BH-FDR is within
game/question/source/estimand/metric-kind, encompassing all transition directions.
Figures and CSVs use the SAME correction. Subject p-values use runs and are
exploratory because runs within sessions need not be independent. Held-out
imitation sessions are not a new cohort or a preregistered replication.

Old fixed-budget Actor integration source is preserved under research_script/legacy.
Source caches are read-only. Manifest records source hashes, QA and gate configuration.

Action count comes from each cache manifest (Enduro: 9; Pong/Space Invaders: 6).
Insufficient outcome variation and nonconvergence are explicitly recorded;
neither is treated as a zero effect. Figures with too few observations say
Not estimable. No significance claim is made from a single-subject validation.

Example (all 6 participants × 3 games × 4 checkpoints, CPU only):
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research_script/03_dynamic_voc_mechanisms.py --subjects 1 2 3 4 5 6 --games 0 1 2 --workers 3
Example (one participant, one game):
    python research_script/03_actor_integration_mechanisms.py --subject 1 --game 1
Example (aggregate completed outputs only):
    python research_script/03_dynamic_voc_mechanisms.py --aggregate-subjects 1 2 3 4 5 6 --game 1
""", encoding="utf-8")


def run_subject(args, subject, game):
    source_hash = sha256(__file__)
    out = Path(args.output_root) / f"sub{subject:03d}_game{game}"
    tables = out / "tables"; tables.mkdir(parents=True, exist_ok=True)
    all_effects, all_curves, all_metrics, audits, provenance = [], [], [], [], []
    for cp, path, config, binding in discover(args.shared_cache_root, subject, game, args.checkpoints):
        log(f"sub{subject:03d} {GAMES[game]} checkpoint {cp}: loading native events")
        d, e, qa = load_cache(path, cp, config, subject, game)
        log(f"sub{subject:03d} game{game} checkpoint {cp}: {len(d):,} decisions, {len(e):,} control choices")
        effects, curves = analyze_associations(d, e)
        all_effects.append(effects); all_curves.append(curves)
        if not args.skip_prediction:
            metrics, predictions, prediction_qa = predict_heldout_session(d)
            if not metrics.empty:
                all_metrics.append(metrics); all_effects.append(prediction_effects(metrics))
                predictions.to_csv(tables/f"heldout_predictions_checkpoint{cp}.csv.gz", index=False)
            qa["prediction"] = prediction_qa
        audits.append(qa)
        files = ["manifest.json", "checkpoint_binding.json", "native_decisions.csv.gz", "native_events.csv.gz",
                 "model_control_windows.csv.gz", "visual_controls.npz", "visual_alignment.csv.gz"]
        provenance.append({"checkpoint":cp, "cache":str(path), "hashes":{n:sha256(path/n) for n in files},
                           "checkpoint_binding":binding, "configuration":{k:config.get(k) for k in
                           ["discounting", "max_search_steps", "max_depth", "tree_carry", "voc_gate_q_temperature",
                            "voc_gate_exact_projection", "voc_gate_epsilon_greedy_execution", "voc_eval_stochastic"]}})
        del d, e
    effects = pd.concat(all_effects, ignore_index=True); curves = pd.concat(all_curves, ignore_index=True)
    summary = inference(effects, "run")
    effects.to_csv(tables/"run_effects_by_checkpoint.csv", index=False)
    curves.to_csv(tables/"curves_by_checkpoint.csv", index=False)
    summary.to_csv(tables/"run_inference.csv", index=False)
    if all_metrics: pd.concat(all_metrics, ignore_index=True).to_csv(tables/"heldout_prediction_metrics.csv", index=False)
    render(out/"figures", effects, curves, summary, game, "run")
    if sha256(__file__) != source_hash:
        raise RuntimeError("Source changed during analysis; rerun to create a consistent manifest")
    write_json(out/"analysis_manifest.json", {"status":"COMPLETED", "version":VERSION, "subject":subject, "game":game,
        "script":str(Path(__file__).resolve()), "script_sha256":source_hash, "created_at":datetime.now(timezone.utc).isoformat(),
        "prediction_requested":not args.skip_prediction, "primary_population":"held_out_sessions", "primary_voc":"ema",
        "source_caches":provenance, "qa":audits, "causal_claim":False})
    log(f"Completed {out}")


def aggregate(args, subjects, game):
    effects, curves = [], []
    manifests = []
    for subject in subjects:
        root = Path(args.output_root)/f"sub{subject:03d}_game{game}"
        manifest = json.loads((root/"analysis_manifest.json").read_text())
        if manifest["status"] != "COMPLETED" or manifest["script_sha256"] != sha256(__file__):
            raise ValueError(f"Stale/incomplete subject output: {root}")
        manifests.append(manifest)
        effects.append(pd.read_csv(root/"tables/run_effects_by_checkpoint.csv"))
        curves.append(pd.read_csv(root/"tables/curves_by_checkpoint.csv"))
    selection = [{x["checkpoint"] for x in m["source_caches"]} for m in manifests]
    if any(s != selection[0] for s in selection):
        raise ValueError("Cannot aggregate different checkpoint selections")
    e, c = pd.concat(effects, ignore_index=True), pd.concat(curves, ignore_index=True)
    summary = inference(e, "subject")
    out = Path(args.output_root)/f"group_game{game}"; (out/"tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv(out/"tables/group_inference.csv", index=False)
    e.to_csv(out/"tables/run_effects_by_checkpoint.csv", index=False)
    c.to_csv(out/"tables/curves_by_checkpoint.csv", index=False)
    render(out/"figures", e, c, summary, game, "subject")
    write_json(out/"analysis_manifest.json", {"status":"COMPLETED", "version":VERSION, "game":game,
        "subjects":subjects, "script_sha256":sha256(__file__), "checkpoints":sorted(selection[0]),
        "n_independent_subjects":len(subjects), "inference":"exact subject sign-flip; FDR across both/all directions within analysis family"})
    log(f"Completed {out}")


def worker(task):
    args, subject, game = task
    run_subject(args, subject, game)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shared-cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subject", type=int)
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--game", type=int, choices=GAMES)
    parser.add_argument("--games", type=int, nargs="+", choices=GAMES)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[1,2,3,4])
    parser.add_argument("--aggregate-subjects", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-prediction", action="store_true", help="Association-only diagnostic; no held-out predictive claim.")
    args = parser.parse_args()
    subjects = args.aggregate_subjects or args.subjects or ([args.subject] if args.subject else [])
    games = args.games or ([args.game] if args.game is not None else [])
    if not subjects or not games or len(set(subjects)) != len(subjects) or args.workers < 1:
        parser.error("Supply unique subjects and games, and positive workers")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_readme(args.output_root)
    if args.aggregate_subjects:
        for game in games: aggregate(args, subjects, game)
        return
    tasks = [(args, s, g) for g in games for s in subjects]
    if args.workers == 1:
        for task in tasks: worker(task)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, tasks))
    if len(subjects) > 1:
        for game in games: aggregate(args, subjects, game)


if __name__ == "__main__":
    import sys
    if "--runwise-legacy" in sys.argv:
        sys.argv.remove("--runwise-legacy")
        main()
    else:
        from dynamic_voc_session_fixed import main as session_fixed_main
        session_fixed_main()
