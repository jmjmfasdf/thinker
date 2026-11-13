"""Verified native dynamic-cache loader extracted from the pre-theme analysis 03.
Decisions and variable-length events are retained; human outcomes follow search.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd
import yaml
KEYS = ["run_label", "real_step", "real_step_id"]
SOURCES = ("ema", "online")
BASE_CONTROL = ["margin", "root_value", "root_entropy", "initial_q_margin", "log_age", "log_age_sq", "recent10", "recent30", "prior_reward", "prior_done", "run_progress", "run_progress_sq", "carried"]
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


def bh(p):
    p = np.asarray(p, float); out = np.full(len(p), np.nan)
    good = np.flatnonzero(np.isfinite(p))
    order = good[np.argsort(p[good])]
    if len(order):
        q = p[order] * len(order) / np.arange(1, len(order) + 1)
        out[order] = np.minimum(1, np.minimum.accumulate(q[::-1])[::-1])
    return out

