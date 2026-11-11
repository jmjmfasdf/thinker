"""Native event extraction; human time and search order are different axes."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

import common as C


def native_reader(path):
    # Import only the IO module, not Thinker/Gym/model initialization.
    source = Path(__file__).resolve().parents[2] / "thinker/thinker/behavior_trace.py"
    spec = importlib.util.spec_from_file_location("continuous_native_io", source)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.NativeVideoStat(path)


def pooled(images, n=4):
    # Exact disjoint cells (including non-divisible 84->16); no temporal interpolation.
    h, w = images.shape[1:3]
    return np.stack([images[:, i*h//n:(i+1)*h//n, j*w//n:(j+1)*w//n].mean(axis=(1, 2))
                     for i in range(n) for j in range(n)], axis=1)


def static_visual(images):
    x = np.moveaxis(images, 1, -1).astype(np.float32) / 255.
    gray = x @ np.array([.2989, .5870, .1140], dtype=np.float32)
    return np.column_stack((pooled(gray), x.mean((1, 2)), x.std((1, 2)),
                            np.abs(np.diff(gray, axis=2)).mean((1, 2)),
                            np.abs(np.diff(gray, axis=1)).mean((1, 2)))).astype(np.float32)


def visual_features(images):
    static = static_visual(images)
    x = np.moveaxis(images, 1, -1).astype(np.float32) / 255.
    gray = x @ np.array([.2989, .5870, .1140], dtype=np.float32)
    motion = pooled(np.abs(gray - np.concatenate((gray[:1], gray[:-1]))))
    motion[0] = np.nan
    descriptor = np.column_stack((static[:, :16], motion, static[:, 16:]))
    low = pooled(x, 16).reshape(len(x), -1)
    delta = low - np.concatenate((low[:1], low[:-1]))
    delta[0] = np.nan
    return descriptor.astype(np.float32), np.column_stack((low, delta)).astype(np.float32)


def reconstruct_paths(indices, actions, pre_depth, post_depth, reset):
    current = []
    result = []
    for i in indices:
        if len(current) != int(pre_depth[i]):
            raise ValueError(f"Unreconstructable path before native event {i}")
        if int(post_depth[i]) == 0 and reset[i]:
            path = []
        else:
            path = current + [int(actions[i])]
            if len(path) != int(post_depth[i]):
                raise ValueError(f"Unreconstructable path after native event {i}")
        result.append(path)
        current = [] if reset[i] else path
    return result


def forecast_alignment(origin, path, actions):
    """Path action d targets a[t+d-1]; its resulting image targets x[t+d]."""
    depth = len(path)
    action_target = origin + depth - 1
    state_target = origin + depth
    if depth < 1 or action_target >= len(actions):
        return action_target, state_target, False, False
    prefix_before = np.array_equal(path[:-1], actions[origin:action_target])
    prefix_full = prefix_before and path[-1] == actions[action_target]
    return action_target, state_target, bool(prefix_before), bool(prefix_full)


def input_episodes(args):
    index = args.capture_index or C.CAMPAIGN / "subjects" / (C.tag(args) + ".json")
    idx = json.loads(index.read_text())
    if idx["status"] != "COMPLETED" or (idx["subject"], idx["game"]) != (args.subject, args.game):
        raise ValueError("Capture index identity/status mismatch")
    records, signatures = [], []
    for capture in idx["capture_roots"]:
        capture = Path(capture)
        manifest = json.loads((capture / "manifest.json").read_text())
        signatures.append(dict(path=str(capture), sha256=C.sha(capture / "manifest.json")))
        checkpoint = manifest["semantic"]["checkpoint"]
        # The config is a verified provenance sidecar, not an alternate behavioral input.
        config = C.REPO / "logs/thinker" / checkpoint["logical_xpid"] / checkpoint["config"]["name"]
        if C.sha(config) != checkpoint["config"]["sha256"]:
            raise ValueError(f"Capture checkpoint config changed: {config}")
        import yaml
        flags = yaml.safe_load(config.read_text())
        if flags.get("model_enc_type", 0) != 0 or flags.get("stat_mask_type", 0) != 0:
            raise ValueError("Non-identity value encoding/masked tree requires an explicit decoder")
        for episode in manifest["episodes"]:
            meta = episode["metadata"]
            if (meta["subject"], meta["game"]) != (args.subject, args.game):
                raise ValueError("Mixed subject/game in capture")
            if f"leavesession{meta['session']}" not in checkpoint["logical_xpid"]:
                raise ValueError("Checkpoint does not hold out its replay session")
            parse_ids = lambda key: {int(v) for v in str(flags[key]).split(",")}
            if meta["session"] in parse_ids("icopro_train_sessions") or meta["session"] not in parse_ids("icopro_holdout_sessions"):
                raise ValueError("Replay session was not genuinely excluded by the training configuration")
            if args.subject not in parse_ids("icopro_subjects") or int(flags["icopro_game_id"]) != args.game:
                raise ValueError("Checkpoint training subject/game differs from replay")
            records.append(dict(path=capture / episode["path"], meta=meta,
                                checkpoint=checkpoint["bundle_sha256"], flags=flags))
    records.sort(key=lambda e: (e["meta"]["session"], e["meta"]["block"], e["meta"]["source_episode_index"]))
    if len({r["meta"]["source_id"] for r in records}) != 11:
        raise ValueError("Expected all 11 source runs per subject/game")
    if args.max_episodes:
        records = records[:args.max_episodes]
    return records, dict(index=str(index), index_sha256=C.sha(index), captures=signatures,
                          max_episodes=args.max_episodes)


def extract_episode(record, subject, game, offset):
    p, meta = record["path"], record["meta"]
    print(f"native-cache: verifying and extracting {p.name}", flush=True)
    native = native_reader(p)
    n_actions = native.num_actions
    if n_actions != {0: 9, 1: 6}[game]:
        raise ValueError("Unexpected game/action mapping")
    scalar_names = ["real_transition", "initial_root", "decision_time", "decision_time_valid",
        "source_frame_index", "pre_observation_source_index", "post_observation_source_index",
        "source_frame_index_valid", "human_action", "human_action_valid", "score_mask", "done", "truncated",
        "search_computation", "search_expansion", "search_revisit", "actor_reset", "forced_stop",
        "stop", "control_valid", "accepted_control", "accepted_primary_action", "phase_pre", "phase_post",
        "current_node_depth_pre", "current_node_depth_post", "root_carried", "useful_carry",
        "carried_descendant_expanded_count", "carried_descendant_visit_count", "imagined_image_valid",
        "real_image_valid", "actor_policy", "actor_policy_valid", "policy_type", "reward",
        "voc_soft_continue_probability", "voc_soft_continue_probability_valid", "voc_ema_q", "voc_ema_q_valid"]
    f = {name: np.asarray(native.field(name)) for name in scalar_names}
    post = np.asarray(native.field("tree_reps_post"))
    pre = np.asarray(native.field("tree_reps_pre"))
    def tree(key, before=False):
        value = (pre if before else post)[:, slice(*native.tree_rep_schema[key])].astype(float)
        return np.expm1(value) if key.endswith("_ns") else value
    qp, qa = tree("root_qs_mean", True), tree("root_qs_mean")
    np_, na = tree("root_ns", True), tree("root_ns")
    root_p = C.probabilities(tree("root_policy", True))
    cur_p = C.probabilities(tree("cur_policy", True))
    rv = tree("root_v", True)[:, 0]
    real = np.flatnonzero(f["real_transition"])
    anchors = np.r_[0, real]
    if len(real) != meta["replay_edge_count"] or not f["initial_root"][0]:
        raise ValueError("Real transition accounting mismatch")
    times = f["decision_time"][real]
    if not f["decision_time_valid"].all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid source decision time")
    if not np.allclose(np.diff(times), 1/15., atol=1e-7):
        raise ValueError("Timestamp units/grid require review; refusing to invent seconds")
    if not f["human_action_valid"][real].all():
        raise ValueError("Missing real human actions")
    actions = f["human_action"][real].astype(int)
    if np.any((actions < 0) | (actions >= n_actions)):
        raise ValueError("Human action out of bounds")
    # These are x_t, not x_(t+1) from the matching real-action event.
    images = native.read_field_rows("real_image", anchors)
    if not f["real_image_valid"][anchors].all():
        raise ValueError("Missing pre-action image")
    descriptor, lowres = visual_features(images[:-1])
    static = static_visual(images)
    run = f"sub{subject:03d}_ses{meta['session']:02d}_block{meta['block']}_game{game}"
    episode = meta["episode_id"]
    decisions, events, forecasts = [], [], []
    for local, right in enumerate(real):
        left = anchors[local]
        stage = np.arange(left + 1, right)
        computations = stage[f["search_computation"][stage]]
        controls = stage[(f["phase_pre"][stage] == 0) & f["control_valid"][stage]]
        if len(controls) < 1:
            raise ValueError("Decision has no recorded control choice")
        first = controls[0]
        if int(f["source_frame_index"][left]) != int(f["pre_observation_source_index"][right]):
            raise ValueError("Pre-action image/source mismatch")
        if not np.allclose(f["decision_time"][stage], times[local]):
            raise ValueError("Search events changed human-time origin")
        if f["policy_type"][right] != 2 or not f["actor_policy_valid"][right]:
            raise ValueError("Terminal Actor must be the REAL head")
        inc = actions[local - 1] if local else -1
        row = dict(decision_id=offset+local, subject=subject, game=game, session=meta["session"],
            block=meta["block"], run=run, episode=episode, checkpoint=record["checkpoint"], local=local,
            time=times[local], origin_time=times[local], available_time=times[local],
            human_action=actions[local], incumbent=inc, switch=float(actions[local] != inc) if local else np.nan,
            scored=bool(f["score_mask"][right]), episode_start=local == 0,
            terminal=bool(f["done"][right] or f["truncated"][right]),
            previous_reward=float(f["reward"][left].reshape(-1)[0]) if local else np.nan,
            pre_image_event=int(left), action_event=int(right), capture_episode=str(p),
            source_frame=int(f["source_frame_index"][left]),
            budget=len(computations), begin=int(len(computations)>0),
            expansion=int(f["search_expansion"][computations].sum()),
            revisit=int(f["search_revisit"][computations].sum()),
            reset=int((f["accepted_control"][computations] == 1).sum()),
            forced_stop=int(f["forced_stop"][stage].any()),
            voluntary_stop=int(f["stop"][stage].any()),
            root_value=rv[first], root_entropy=C.entropy(root_p[first]),
            carried=int(f["root_carried"][left]), useful_carry=int(f["useful_carry"][left]),
            carried_visits=int(f["carried_descendant_visit_count"][left]),
            carried_expanded=int(f["carried_descendant_expanded_count"][left]),
            cap=int(record["flags"].get("max_search_steps", 0)),
            max_allowed_depth=int(record["flags"].get("max_depth", 0)))
        if row["forced_stop"] + row["voluntary_stop"] != 1:
            raise ValueError("Each stage needs one voluntary stop or a forced terminal computation")
        paths = reconstruct_paths(computations, f["accepted_primary_action"],
                                  f["current_node_depth_pre"], f["current_node_depth_post"], f["actor_reset"])
        row["max_depth"] = max([len(path) for path in paths] + [0])
        row["branches"] = len({path[0] for path in paths if path})
        final = stage[-1]
        visited_before, visited_after = np_[first] > 0, na[final] > 0
        touched = (na[final] > np_[first] + 1e-8)
        known_revision = visited_before & visited_after & touched
        first_evaluation = ~visited_before & visited_after & touched
        revision = np.where(known_revision, qa[final] - qp[first], np.nan)
        new_return = np.where(first_evaluation, qa[final] - rv[first], np.nan)
        support = np.where(known_revision, revision, new_return)
        terminal_p = C.probabilities(f["actor_policy"][right])
        for a in range(n_actions):
            row.update({f"root_p_{a}": root_p[first, a], f"root_q_{a}": qp[first, a] if visited_before[a] else np.nan,
                        f"root_n_{a}": np_[first, a], f"evaluated_{a}": int(touched[a]),
                        f"revision_{a}": revision[a], f"first_return_{a}": new_return[a],
                        f"support_{a}": support[a], f"actor_p_{a}": terminal_p[a]})
        valid = np.isfinite(support)
        row["evidence_abs"] = float(np.sum(np.abs(support[valid])))
        row["evidence_available"] = int(valid.any())
        row["incumbent_change"] = support[inc] if inc >= 0 else np.nan
        alts = np.arange(n_actions) != inc
        row["alternative_change"] = np.max(support[valid & alts]) if (valid & alts).any() else np.nan
        row["evidence_signed"] = np.nan_to_num(row["alternative_change"]) - np.nan_to_num(row["incumbent_change"])
        row["root_margin"] = (np.log(root_p[first, inc]+1e-12)-np.max(np.log(root_p[first, alts]+1e-12))) if inc >= 0 else np.nan
        cumulative_net = []
        for event in computations:
            changed = na[event] > np_[first]+1e-8
            support_after = np.where(changed, qa[event]-np.where(visited_before, qp[first], rv[first]), np.nan)
            if inc >= 0:
                available_alts = np.isfinite(support_after) & alts
                alt = np.max(support_after[available_alts]) if available_alts.any() else 0.
                cumulative_net.append(float(alt-np.nan_to_num(support_after[inc])))
        row["evidence_slope"] = float(np.polyfit(np.arange(len(cumulative_net)), cumulative_net, 1)[0]) if len(cumulative_net)>=3 else np.nan
        row["evidence_slope_events"] = len(cumulative_net)
        for j, value in enumerate(descriptor[local]):
            row[f"visual_{j}"] = value
        decisions.append(row)
        for control in controls:
            j = int(np.sum(computations < control))
            previous = computations[computations < control]
            previous_change = 0.
            if len(previous):
                last = previous[-1]
                changed = na[last] > np_[last]+1e-8
                last_reference = np.where(np_[last]>0, qp[last], rv[first])
                previous_change = float(np.abs(qa[last]-last_reference)[changed].sum())
            known = np_[control] > 0
            cumulative = np.where(known & visited_before, qp[control]-qp[first], 0.)
            voc = f["voc_ema_q"][control]
            events.append(dict(decision_id=offset+local, subject=subject, game=game, run=run, session=meta["session"],
                episode=episode, checkpoint=record["checkpoint"], time=times[local], origin_time=times[local],
                available_time=times[local], event=int(control), event_index=j,
                incumbent=inc, stop_choice=int(f["accepted_control"][control] == 2),
                control=int(f["accepted_control"][control]), forced_terminal=int(f["forced_stop"][control]),
                depth_pre=int(f["current_node_depth_pre"][control]),
                node_entropy=C.entropy(cur_p[control]), prior_evidence_abs=float(np.abs(cumulative).sum()),
                last_change=previous_change, root_entropy=row["root_entropy"],
                root_margin=row["root_margin"], carried=row["carried"],
                gate_continue=float(f["voc_soft_continue_probability"][control]) if f["voc_soft_continue_probability_valid"][control] else np.nan,
                voc_continue_minus_stop=float(voc[0]-voc[1]) if f["voc_ema_q_valid"][control] else np.nan))
            event_record = events[-1]
            event_record.update(computation=bool(f["search_computation"][control]),
                expansion=bool(f["search_expansion"][control]), revisit=bool(f["search_revisit"][control]),
                actor_reset=bool(f["actor_reset"][control]), accepted_action=int(f["accepted_primary_action"][control]),
                depth_post=int(f["current_node_depth_post"][control]), root_value_pre=rv[control])
            for a in range(n_actions):
                event_record.update({f"root_q_before_{a}": qp[control, a] if np_[control, a]>0 else np.nan,
                    f"root_q_after_{a}": qa[control, a] if na[control, a]>0 else np.nan,
                    f"root_n_before_{a}": np_[control, a], f"root_n_after_{a}": na[control, a]})
        # Read only actual prediction-bearing images. A root-only reset has no future state.
        for event, path in zip(computations, paths):
            if not path or not f["imagined_image_valid"][event]:
                continue
            action_target, state_target, prefix_before, prefix_full = forecast_alignment(local, path, actions)
            target_valid = action_target < len(actions)
            state_valid = state_target < len(actions)  # terminal x_(T) has no next action/risk row
            forecasts.append(dict(origin_id=offset+local, origin_local=local, origin_time=times[local],
                available_time=times[local], run=run, episode=episode, subject=subject, game=game,
                depth=len(path), event=int(event), path=json.dumps(path), imagined_action=path[-1],
                origin_incumbent=inc, target_id=offset+action_target if target_valid else -1,
                state_target_id=offset+state_target if state_valid else -1,
                target_time=times[action_target] if target_valid else np.nan,
                state_time=times[state_target] if state_valid else np.nan,
                prefix_before=prefix_before, prefix_match=prefix_full,
                action_match=int(path[-1] == actions[action_target]) if target_valid else np.nan,
                persistence_match=int(inc == actions[action_target]) if target_valid and inc >= 0 else np.nan,
                root_match=int(root_p[first].argmax() == actions[action_target]) if target_valid else np.nan,
                target_action=actions[action_target] if target_valid else -1,
                pixel_error=np.nan, descriptor_error=np.nan, persistence_pixel_error=np.nan,
                state_comparison_available=bool(state_valid), image_valid=True))
    # Blockwise reads avoid random decompression of the same image chunk for each forecast.
    for start in range(0, len(forecasts), 256):
        batch = forecasts[start:start+256]
        predicted = native.read_field_rows("imagined_image", [r["event"] for r in batch])
        pred_static = static_visual(predicted)
        for j, record_ in enumerate(batch):
            if not record_["state_comparison_available"]:
                continue
            target = record_["state_target_id"]-offset
            origin = record_["origin_local"]
            actual = images[target].astype(np.float32)/255.
            record_["pixel_error"] = float(np.mean((predicted[j].astype(np.float32)/255.-actual)**2))
            record_["persistence_pixel_error"] = float(np.mean((images[origin].astype(np.float32)/255.-actual)**2))
            record_["descriptor_error"] = float(np.sqrt(np.mean((pred_static[j]-static[target])**2)))
    return pd.DataFrame(decisions), pd.DataFrame(events), pd.DataFrame(forecasts), lowres


def add_history(frame, forecasts, actions):
    """All historical covariates are left-closed in information, never in outcome."""
    d = frame.copy()
    n_actions = actions
    extra = {}
    def blank(name):
        if name not in extra:
            extra[name] = np.full(len(d), np.nan)
        return extra[name]
    # Past action/preference history never crosses an episode (including split source files).
    for _, indices in d.groupby("episode", sort=False).groups.items():
        ix = np.asarray(indices); e = d.loc[ix]; t = e.time.to_numpy(); a = e.human_action.to_numpy()
        evidence = e.evidence_signed.to_numpy(); absolute = e.evidence_abs.to_numpy()
        support_matrix = e[[f"support_{j}" for j in range(n_actions)]].to_numpy()
        oriented_by_action = []
        for inc in range(n_actions):
            alt_support = support_matrix[:, np.arange(n_actions) != inc]
            finite_alt = np.isfinite(alt_support)
            alt_max = np.max(np.where(finite_alt, alt_support, -np.inf), axis=1)
            alt_max[~finite_alt.any(axis=1)] = 0.
            oriented_by_action.append(alt_max-np.nan_to_num(support_matrix[:, inc]))
        bout_start = 0; observed_onset = False
        for k, index in enumerate(ix):
            if k > 1 and a[k-1] != a[k-2]:
                bout_start = k-1; observed_onset = True
            start_time = t[bout_start]
            blank("bout_age")[index] = t[k]-start_time if k else np.nan
            blank("bout_left_censored")[index] = int(not observed_onset)
            blank("history_available_time")[index] = t[k-1] if k else t[k]
            blank("history_origin_time")[index] = start_time
            # Re-express old evidence relative to the CURRENT incumbent, including
            # the onset decision whose former incumbent was a different action.
            if k:
                oriented = oriented_by_action[a[k-1]]
            else:
                oriented = evidence
            blank("history_previous_evidence")[index] = oriented[k-1] if k else np.nan
            blank("history_previous_abs")[index] = absolute[k-1] if k else np.nan
            previous = oriented[bout_start:k]
            blank("history_cumulative")[index] = previous.sum() if len(previous) else np.nan
            blank("history_cumulative_abs")[index] = absolute[bout_start:k].sum() if k > bout_start else np.nan
            blank("history_cumulative_budget")[index] = e.budget.to_numpy()[bout_start:k].sum() if k > bout_start else np.nan
            blank("history_consistency")[index] = abs(previous.sum())/(np.abs(previous).sum()+1e-8) if len(previous) else np.nan
            nonzero = np.sign(previous[previous != 0])
            blank("history_reversals")[index] = (nonzero[1:] != nonzero[:-1]).sum() if len(nonzero) else np.nan
            for back in (1, 2, 3, 4):
                blank(f"previous_action_{back}")[index] = a[k-back] if k >= back else -1
            for seconds in (1., 4.):
                left = np.searchsorted(t, t[k]-seconds, side="left")
                prior = a[left:k]
                for action in range(n_actions):
                    blank(f"action_fraction_{int(seconds)}_{action}")[index] = np.mean(prior == action) if len(prior) else np.nan
                blank(f"switch_count_{int(seconds)}")[index] = np.sum(prior[1:] != prior[:-1]) if len(prior) else np.nan
            for tau in (.25, 1., 4.):
                left = max(bout_start, np.searchsorted(t, t[k]-4., side="left"))
                weights = np.exp(-(t[k]-t[left:k])/tau)
                blank(f"history_ewma_{tau:g}")[index] = np.average(oriented[left:k], weights=weights) if len(weights) else np.nan
            for lag, (lo, hi) in enumerate(C.LAGS):
                selected = np.flatnonzero((t < t[k]-lo-1e-8) & (t >= t[k]-hi-1e-8))
                selected = selected[selected < k]
                blank(f"lag{lag}_coverage")[index] = len(selected)
                blank(f"lag{lag}_origin_time")[index] = t[selected[0]] if len(selected) else t[k]
                blank(f"lag{lag}_available_time")[index] = t[selected[-1]] if len(selected) else t[k]
                if len(selected):
                    for feature in [f"visual_{j}" for j in range(40)] + ["root_margin", "root_value", "evidence_signed", "evidence_abs", "budget"]:
                        values = e[feature].to_numpy()[selected]
                        blank(f"lag{lag}_{feature}")[index] = np.nanmean(values) if np.isfinite(values).any() else np.nan
    d = pd.concat((d, pd.DataFrame(extra)), axis=1)
    d["run_progress"] = d.groupby("run")["time"].transform(lambda x: x-x.min()) / 480.
    # Earlier predictions of the CURRENT action are available only if the past action prefix matched.
    for a in range(n_actions):
        d[f"old_plan_support_{a}"] = np.nan
    d["old_plan_count"] = 0.
    d["old_plan_origin_time"] = d.time
    d["mismatch_pixel"] = np.nan; d["mismatch_descriptor"] = np.nan
    d["mismatch_persistence"] = np.nan; d["mismatch_count"] = 0.
    d["mismatch_available_time"] = d.time
    if len(forecasts):
        eligible = forecasts[(forecasts.target_id >= 0) & (forecasts.origin_id < forecasts.target_id) & forecasts.prefix_before]
        for target, rows in eligible.groupby("target_id"):
            if rows.episode.nunique() != 1 or d.loc[target, "episode"] != rows.episode.iloc[0]:
                raise ValueError("Forecast crossed episode")
            for a in range(n_actions):
                d.loc[target, f"old_plan_support_{a}"] = np.mean(rows.imagined_action == a)
            d.loc[target, "old_plan_count"] = len(rows)
            d.loc[target, "old_plan_origin_time"] = rows.origin_time.min()
        arrived = forecasts[(forecasts.state_target_id >= 0) & forecasts.prefix_match]
        for target, rows in arrived.groupby("state_target_id"):
            C.assert_available(rows.origin_time, rows.state_time, d.loc[target, "time"])
            d.loc[target, "mismatch_pixel"] = rows.pixel_error.mean()
            d.loc[target, "mismatch_descriptor"] = rows.descriptor_error.mean()
            d.loc[target, "mismatch_persistence"] = rows.persistence_pixel_error.mean()
            d.loc[target, "mismatch_count"] = len(rows)
    C.assert_available(d.origin_time, d.available_time, d.time)
    C.assert_available(d.history_origin_time, d.history_available_time, d.time)
    C.assert_available(d.old_plan_origin_time, d.time, d.time)
    d["mismatch_x_evidence"] = d.mismatch_pixel*d.evidence_signed
    return d


def main():
    args = C.parser(__doc__).parse_args()
    if not C.settings(args)["verify_native_hashes"]:
        raise ValueError("Full native integrity validation is mandatory for this release")
    episodes, inputs = input_episodes(args)
    root, manifest, done = C.start_manifest(args, "00_native_cache", inputs)
    if done:
        print(f"Validated complete cache: {root}", flush=True); return
    ds, es, ps, visuals = [], [], [], []
    count = 0
    for record in episodes:
        d, e, p, v = extract_episode(record, args.subject, args.game, count)
        ds.append(d); es.append(e); ps.append(p); visuals.append(v); count += len(d)
    d = pd.concat(ds, ignore_index=True); e = pd.concat(es, ignore_index=True)
    forecasts = pd.concat(ps, ignore_index=True)
    d = add_history(d, forecasts, {0: 9, 1: 6}[args.game])
    if d.decision_id.duplicated().any() or not np.array_equal(d.decision_id, np.arange(len(d))):
        raise ValueError("Decision IDs are not a contiguous unique key")
    d.to_csv(root / "decisions.csv.gz", index=False)
    e.to_csv(root / "search_events.csv.gz", index=False)
    forecasts.to_csv(root / "forecasts.csv.gz", index=False)
    forecasts[forecasts.state_comparison_available].to_csv(root / "realized_forecasts.csv.gz", index=False)
    np.savez_compressed(root / "visual.npz", lowres=np.concatenate(visuals))
    # Column-level provenance plus row-level timestamps form a compact per-feature contract.
    contract = {}
    metadata = {"human_action", "switch", "terminal", "action_event", "scored"}
    for column in d:
        kind = "outcome_or_metadata" if column in metadata else "pre_action_covariate_or_key"
        origin = "history_origin_time" if column.startswith("history_") else "origin_time"
        if column.startswith("old_plan_"): origin = "old_plan_origin_time"
        available = "history_available_time" if column.startswith(("history_", "previous_action_", "action_fraction_", "switch_count_")) else "available_time"
        if column.startswith("lag"):
            prefix = column.split("_", 1)[0]
            origin, available = prefix+"_origin_time", prefix+"_available_time"
        contract[column] = dict(kind=kind, origin_time_column=origin, available_time_column=available)
    C.write_json(root / "feature_contract.json", contract)
    qa = dict(complete_real=len(d), scored_real=int(d.scored.sum()), episodes=len(episodes),
              source_runs=d.run.nunique(), search_events=int(d.budget.sum()), control_rows=len(e),
              budget_counts=d.budget.value_counts().sort_index().to_dict(),
              forced_stops=int(d.forced_stop.sum()), voluntary_stops=int(d.voluntary_stop.sum()),
              resets=int(d.reset.sum()), native_hashes_verified=True,
              pre_action_image_alignment_verified=True, no_cross_episode_history=True,
              forecasts=len(forecasts), same_head_actor_revision_available=False)
    if int(d.scored.sum()) != sum(r["meta"]["scored_edge_count"] for r in episodes):
        raise ValueError("Score mask count mismatch")
    C.write_json(root / "qa.json", qa)
    C.finish(root, manifest, **qa, num_actions={0: 9, 1: 6}[args.game])
    print(json.dumps(qa), flush=True)


if __name__ == "__main__":
    main()
