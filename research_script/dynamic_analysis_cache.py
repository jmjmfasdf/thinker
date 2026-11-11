"""Adapt completed native captures to the existing analysis table contract.

Only numeric/tree fields are read during extraction. Images are read at real
anchors in bounded batches. Raw behavioral archives are never opened.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys
import time
import hashlib
import re
import numpy as np
import pandas as pd

from shared_analysis_cache import cache_paths, validate_one_to_one_real_steps, SHARED_CACHE_VERSION
from shared_actor_policy_cache import window_coordinates
from dynamic_native_features import NativeFeatures, pack_paths, prediction_summaries, schema_contract, TREE_COMPONENTS

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def trace_module():
    # Import this independent I/O module without importing the torch/cenv package.
    return load_module("analysis_behavior_trace", ROOT / "thinker/thinker/behavior_trace.py")


def policy(logits):
    values = np.asarray(logits, dtype=float)
    exp = np.exp(values - np.max(values, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def reconstruct_paths(search, action, pre_depth, post_depth, reset):
    """Return exact paths in execution order; RESET-to-root has no branch."""
    paths = []
    current = []
    for index in search:
        if len(current) != int(pre_depth[index]):
            raise ValueError(f"Path/pre-depth mismatch at event {index}")
        if int(post_depth[index]) == 0 and reset[index]:
            path = []
        else:
            path = current + [int(action[index])]
            if len(path) != int(post_depth[index]):
                raise ValueError(f"Path/post-depth mismatch at event {index}")
        paths.append(path)
        current = [] if reset[index] else path
    width = max([1] + [len(p) for p in paths])
    matrix = np.full((len(paths), width), -1, dtype=np.int16)
    for row, path in enumerate(paths):
        matrix[row, :len(path)] = path
    return matrix, np.array([len(p) for p in paths], dtype=np.int16)


def visual_descriptor(current, past):
    # The same 40-dimensional formula used by 04, on native CHW observations.
    current = np.moveaxis(current, 1, -1).astype(np.float32) / 255
    past = np.moveaxis(past, 1, -1).astype(np.float32) / 255
    weights = np.array([.2989, .5870, .1140], dtype=np.float32)
    gray = np.einsum("nhwc,c->nhw", current, weights)
    old = np.einsum("nhwc,c->nhw", past, weights)
    def pooled(x):
        h, w = x.shape[1] // 4 * 4, x.shape[2] // 4 * 4
        return x[:, :h, :w].reshape(len(x), 4, h//4, 4, w//4).mean((2, 4)).reshape(len(x), 16)
    return np.column_stack([pooled(gray), pooled(abs(gray-old)), current.mean((1, 2)),
                            current.std((1, 2)), abs(np.diff(gray, axis=2)).mean((1, 2)),
                            abs(np.diff(gray, axis=1)).mean((1, 2))]).astype(np.float32)


def capture_inputs(args):
    """Bind every episode to its real capture; never merge capture manifests."""
    index_path = getattr(args, "capture_index", None)
    index = json.loads(index_path.read_text()) if index_path else None
    if index:
        if (index.get("status"), index.get("subject"), index.get("game")) != ("COMPLETED", args.subject, args.game):
            raise ValueError("Incomplete or wrong subject/game collection index")
        roots = [Path(p).resolve() for p in index["capture_roots"]]
        if len(roots) != 4 or len(set(roots)) != 4 or len(index["checkpoints"]) != 4:
            raise ValueError("Held-out collection requires four distinct captures/checkpoints")
        campaign = json.loads((index_path.resolve().parent.parent / "campaign.json").read_text())
        tasks = {Path(t["capture"]).resolve(): t for t in campaign["tasks"]
                 if (t["subject"], t["game"]) == (args.subject, args.game)}
        if set(tasks) != set(roots):
            raise ValueError("Collection/campaign capture coverage mismatch")
    else:
        roots, tasks = [args.capture_dir.resolve()], {}
    episodes, origins, source_runs, seen_episodes = [], [], {}, set()
    sessions = set()
    for position, capture in enumerate(roots):
        completed = json.loads((capture / "COMPLETED.json").read_text())
        raw = (capture / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
        if (completed.get("status") != "COMPLETED" or completed.get("manifest_sha256") != digest
                or manifest.get("manifest_sha256") != digest):
            raise ValueError(f"Incomplete capture or manifest hash mismatch: {capture}")
        selected = manifest["episodes"]
        if len(selected) != completed["episode_count"] or not selected:
            raise ValueError(f"Episode coverage mismatch: {capture}")
        task = tasks.get(capture)
        if task:
            checkpoint = task["checkpoint"]
            match = re.search(r"leavesession0*(\d+)$", checkpoint)
            if (not match or int(match[1]) != task["session"]
                    or checkpoint != index["checkpoints"][position]):
                raise ValueError("Held-out checkpoint/session mismatch")
            sessions.add(task["session"])
            identity = manifest["semantic"]["checkpoint"]
            if (identity["logical_xpid"] != Path(checkpoint).name
                    or identity["milestone_sha256"] != task["milestone_sha256"]):
                raise ValueError("Capture checkpoint differs from campaign")
            actual_sources = {e["metadata"]["source_id"]: e["metadata"]["source_sha256"] for e in selected}
            if actual_sources != {s["source_id"]: s["sha256"] for s in task["sources"]}:
                raise ValueError("Held-out source coverage/hash mismatch")
        for ep in selected:
            meta = ep["metadata"]
            if (meta["subject"], meta["game"]) != (args.subject, args.game):
                raise ValueError("Capture subject/game differs from requested identity")
            if task and (meta["session"] != task["session"] or meta["checkpoint_real_step"] != campaign["target_step"]):
                raise ValueError("Capture session/checkpoint step differs from campaign")
            identity = (meta["source_id"], meta["source_episode_index"])
            if identity in seen_episodes:
                raise ValueError("Duplicate source episode in capture collection")
            seen_episodes.add(identity)
            run = f"sub{args.subject:03d}_ses{meta['session']:02d}_block{meta['block']}_game{args.game}"
            binding = (run, str(capture), meta["source_sha256"])
            if source_runs.setdefault(meta["source_id"], binding) != binding:
                raise ValueError("Source maps to multiple runs/checkpoints")
            episode_path = (capture / ep["path"]).resolve()
            if not episode_path.is_relative_to(capture) or not episode_path.is_dir():
                raise ValueError("Missing or out-of-capture episode path")
            episodes.append(dict(ep, capture_dir=str(capture)))
        origins.append(dict(capture_dir=str(capture), manifest_sha256=digest,
                            checkpoint=task["checkpoint"] if task else None,
                            session=task["session"] if task else None,
                            checkpoint_hashes={k: selected[0]["metadata"].get(k) for k in
                                ("checkpoint_actor_sha256", "checkpoint_model_sha256", "checkpoint_bundle_sha256")}))
    if len({v[0] for v in source_runs.values()}) != len(source_runs):
        raise ValueError("Multiple source files map to the same LORO run")
    if index and (sessions != {1, 2, 3, 4} or len(source_runs) != index["source_count"]):
        raise ValueError("Incomplete held-out sessions/source files")
    episodes.sort(key=lambda e: (e["metadata"]["source_id"], e["metadata"]["source_episode_index"]))
    if args.sessions:
        sessions = set(map(int, args.sessions.split(",")))
        episodes = [e for e in episodes if e["metadata"]["session"] in sessions]
    if args.max_files:
        if args.max_files <= 0:
            raise ValueError("--max-files must be positive")
        sources = list(dict.fromkeys(e["metadata"]["source_id"] for e in episodes))[:args.max_files]
        episodes = [e for e in episodes if e["metadata"]["source_id"] in sources]
    if not episodes:
        raise ValueError("No selected episodes")
    provenance = dict(capture_index=str(index_path.resolve()) if index_path else None,
        captures=origins, assignment=index.get("assignment") if index else None,
        source_runs={key: dict(run_label=value[0], capture_dir=value[1], source_sha256=value[2])
                     for key, value in source_runs.items() if key in {e["metadata"]["source_id"] for e in episodes}})
    return episodes, provenance


def build_capture_cache(args):
    options = dict(images=bool(getattr(args, "include_prediction_summaries", False)),
                   latents=bool(getattr(args, "include_latent_summaries", False)),
                   batch_size=int(getattr(args, "native_summary_batch_size", 128)))
    if options["batch_size"] < 1:
        raise ValueError("--native-summary-batch-size must be positive")
    feature_sources = {name: hashlib.sha256((SCRIPT_DIR/name).read_bytes()).hexdigest()
                       for name in ("dynamic_analysis_cache.py", "dynamic_native_features.py")}
    episodes, provenance = capture_inputs(args)
    paths = cache_paths(args.cache_root, args.subject, args.game)
    signature = hashlib.sha256(json.dumps([episodes, provenance], sort_keys=True).encode()).hexdigest()
    if paths.manifest.exists() and not args.overwrite:
        prior = json.loads(paths.manifest.read_text())
        if (prior.get("source_signature") == signature and prior.get("adapter_version") == 3
                and prior.get("native_feature_options") == options
                and prior.get("native_feature_sources") == feature_sources):
            required = [paths.actor_metrics, paths.root_policy_metrics, paths.persistence_windows, paths.persistence_events,
                        paths.persistence_bouts, paths.model_control_windows, paths.rollout_windows,
                        paths.rollout_prefix_events, paths.visual_controls, paths.visual_alignment,
                        paths.native_decisions, paths.native_events, paths.imagination_paths,
                        paths.forecast_outcomes, paths.native_features_schema, paths.native_features_qa]
            if options["images"] or options["latents"]:
                required.append(paths.representation_summaries)
            if all(p.is_file() and p.stat().st_size for p in required):
                print(f"Reusing complete dynamic cache: {paths.root}", flush=True)
                return
        raise ValueError("Existing cache is incomplete or differs; select another cache root or --overwrite")
    if paths.root.exists() and args.overwrite:
        paths.root.rename(paths.root.with_name(paths.root.name + time.strftime(".backup_%Y%m%d_%H%M%S")))
    paths.root.mkdir(parents=True, exist_ok=True)
    io = trace_module()
    geometry = load_module("dynamic_geometry_formulas", SCRIPT_DIR / "legacy/02_persistence_action_geometry.py")
    windows, events, bouts, prefixes, alignment, descriptors = [], [], [], [], [], []
    audits = []
    native_decisions, path_rows, feature_audits = [], [], []
    run_step = {}
    num_actions = None
    for ep_index, ep in enumerate(episodes):
        capture = Path(ep["capture_dir"])
        print(f"Extracting episode {ep_index+1}/{len(episodes)}: {ep['metadata']['source_id']} #{ep['metadata']['source_episode_index']}", flush=True)
        projection = io.open_video_stat(capture / ep["path"], materialize=False)
        native = projection.native
        if num_actions is not None and native.num_actions != num_actions:
            raise ValueError("Action count changed within capture")
        num_actions = native.num_actions
        if num_actions != {0: 9, 1: 6}.get(args.game, num_actions):
            raise ValueError("Unexpected action count for game")
        tree = projection["tree_reps"]
        source_rows = projection.source_event_index
        status = projection["status"]
        anchors = source_rows[status == 0]
        # Native pre/post values avoid temporal guesses at STOP/RESET boundaries.
        names = ["human_action", "thinker_action", "actor_policy", "cur_rewards", "done", "score_mask",
                 "accepted_primary_action", "current_node_depth_pre", "current_node_depth_post", "actor_reset",
                 "search_computation", "source_frame_index", "pre_observation_source_index", "post_observation_source_index"]
        fields = {k: np.asarray(native.field(k)) for k in names}
        post = np.asarray(native.field("tree_reps_post"))
        pre = np.asarray(native.field("tree_reps_pre"))
        decoded = {}
        decoded_pre = {}
        for key, descriptor in native.tree_rep_schema.items():
            decoded[key] = post[:, slice(*descriptor)]
            decoded_pre[key] = pre[:, slice(*descriptor)]
        # Log counts are a native-format encoding, not a fraction of a fixed budget.
        for key in ("root_ns", "cur_ns"):
            decoded[key] = np.expm1(decoded[key].astype(float))
            decoded_pre[key] = np.expm1(decoded_pre[key].astype(float))
        meta = ep["metadata"]
        run = f"sub{args.subject:03d}_ses{meta['session']:02d}_block{meta['block']}_game{args.game}"
        offset = run_step.get(run, 0)
        base = len(windows)
        native_features = NativeFeatures(native, anchors, decoded_pre, decoded, run,
                                        ep["episode_id"], base, offset)
        image_rows = anchors[:-1]
        descriptor = np.full((len(image_rows), 40), np.nan, dtype=np.float32)
        for start in range(1, len(image_rows), 256):
            stop = min(start + 256, len(image_rows))
            current_images = native.read_field_rows("real_image", image_rows[start:stop])
            past_images = native.read_field_rows("real_image", image_rows[start-1:stop-1])
            if not np.all(native.read_field_rows("real_image_valid", image_rows[start-1:stop])):
                raise ValueError("Missing real observation at visual-control anchor")
            descriptor[start:stop] = visual_descriptor(current_images, past_images)
        descriptors.append(descriptor)
        age = 0
        for local, (left, right) in enumerate(zip(anchors[:-1], anchors[1:])):
            if int(fields["source_frame_index"][left]) != int(fields["pre_observation_source_index"][right]):
                raise ValueError("Current image is not the real action's pre-observation")
            wi = len(windows)
            real_step = offset + local + 1
            human = int(fields["human_action"][right])
            incumbent = int(fields["human_action"][left]) if local else -1
            before = int(fields["human_action"][anchors[local-1]]) if local > 1 else -1
            age = age + 1 if local > 1 and incumbent == before else 1
            selected = np.flatnonzero(fields["search_computation"][left+1:right]) + left+1
            length = len(selected)
            path_matrix, depth = reconstruct_paths(selected, fields["accepted_primary_action"],
                fields["current_node_depth_pre"], fields["current_node_depth_post"], fields["actor_reset"])
            native_features.decision(local, selected, path_matrix, depth)
            prior = policy(decoded["root_policy"][left])
            terminal = policy(fields["actor_policy"][right])
            search_policy = policy(fields["actor_policy"][selected]) if length else np.empty((0, num_actions))
            initial = search_policy[0] if length else terminal
            root = geometry.RootState(prior, decoded["root_qs_mean"][left],
                decoded["root_ns"][left]/(length+1), float(decoded["root_v"][left,0]),
                float(decoded["rollout_return"][left,0]), float(decoded["max_rollout_return"][left,0]),
                max(incumbent, 0), float(fields["cur_rewards"][left].flat[0]),
                float(fields["done"][left]), meta["source_episode_index"])
            batch = SimpleNamespace(n_status1=int(fields["actor_reset"][selected].sum()), n_status2=length, n_status3=0,
                paths=path_matrix, path_valid=depth>0, depth=depth,
                cur_reset=fields["actor_reset"][selected].astype(np.int8),
                root_q=decoded["root_qs_mean"][selected], root_n=decoded["root_ns"][selected]/(length+1),
                cur_policy=search_policy, rollout_return=decoded["rollout_return"][selected,0])
            # Geometry's empty-window branch already preserves undefined means as NaN.
            metrics, local_bouts, local_events = geometry.summarize_window(root, batch, search_policy,
                human, "actor" if length else "model", "root_v", num_actions, 1.0)
            coords = geometry.persistence_coordinates(prior, incumbent, num_actions)
            actor_coords = window_coordinates(initial, terminal, search_policy, incumbent, geometry.persistence_coordinates)
            key = dict(run_label=run, real_step=real_step, real_step_id=f"{run}_real{real_step:06d}")
            row = dict(metrics)
            row.update(key)
            row.update(subject=args.subject, session=meta["session"], block=meta["block"], game=args.game,
                chunk=meta["source_episode_index"], source_chunk=meta["source_episode_index"] if local else np.nan,
                episode_id=ep["episode_id"], episode_start=int(local==0),
                human_action=human, previous_human_action=incumbent, incumbent_action=incumbent,
                action_before_incumbent=before, thinker_action=int(fields["thinker_action"][right]),
                switch=int(human != incumbent) if incumbent >= 0 else np.nan,
                previous_reward=root.reward, prior_reward=root.reward,
                previous_done=int(local==0 or fields["done"][left]), prior_done=int(local==0 or fields["done"][left]),
                bout_age=age, noop_run_length_prev=age if incumbent==0 else 0,
                actor_policy_source_real_step=real_step, initial_actor_policy_complete=1, terminal_actor_policy_complete=1,
                model_candidate=int(np.argmax(prior)), actor_initial_candidate=int(np.argmax(initial)),
                scored=int(fields["score_mask"][right]), native_anchor_before=int(left), native_anchor_after=int(right),
                current_frame=int(fields["source_frame_index"][left]),
                previous_frame=int(fields["source_frame_index"][anchors[local-1]]) if local else -1,
                capture_episode_path=str(capture/ep["path"]))
            row.update(actor_coords)
            # Preserve 03's actor-defined persistence coordinate and 02's model control coordinate separately.
            row["persistence_evidence"] = actor_coords["actor_current_action_margin_start"]
            row["switch_target_uncertainty"] = actor_coords["actor_current_action_uncertainty_start"]
            row["strongest_competitor_margin"] = coords["strongest_competitor_margin"]
            row["model_switch_target_uncertainty"] = coords["switch_target_uncertainty"]
            for a in range(num_actions):
                row[f"model_root_p_{a}"] = prior[a]
                row[f"actor_initial_p_{a}"] = initial[a]
                row[f"actor_terminal_p_{a}"] = terminal[a]
            windows.append(row)
            for record in local_events:
                record.update(key)
                if incumbent < 0:
                    record["directed_value_revision"] = np.nan
                    record["directed_prediction_error"] = np.nan
                events.append(record)
            # Root-only RESET computations count towards budget but have no sampled branch.
            for pos in np.flatnonzero(depth == 0):
                events.append(dict(key, status2_position=int(pos), bout_index=-1, branch=-1, depth=0,
                    rollout_return=float(batch.rollout_return[pos]), value_revision_valid=0))
            for record in local_bouts:
                bouts.append(dict(record, **key))
            for j in range(length):
                pair = j+1 < length and depth[j]>0 and depth[j+1]>0
                same = pair and path_matrix[j,0] == path_matrix[j+1,0]
                deepen = pair and not batch.cur_reset[j] and depth[j+1]>depth[j] and np.array_equal(path_matrix[j,:depth[j]],path_matrix[j+1,:depth[j]])
                prefixes.append(dict(window_index=wi, event_index=j,
                    return_advantage=float(batch.rollout_return[j]-root.root_value),
                    event_progress=j/max(length-2,1), pair_valid=int(pair),
                    same_branch_next=int(same), branch_switch_next=int(pair and not same),
                    same_path_deepen_next=int(deepen), reset_after=int(batch.cur_reset[j]),
                    has_next=int(j+1 < length)))
            alignment.append(dict(key, actor_row=wi, episode_id=ep["episode_id"],
                current_frame=row["current_frame"], previous_frame=row["previous_frame"],
                native_current_anchor=int(left), native_past_anchor=int(anchors[local-1]) if local else -1))
        run_step[run] = offset + len(anchors)-1 + 1  # gap marks the episode boundary
        native_decisions.extend(native_features.records)
        path_rows.extend(native_features.path_rows)
        native_events = native_features.event_table()
        forecast, representations = prediction_summaries(native_features,
            include_images=options["images"], include_latents=options["latents"],
            batch_size=options["batch_size"])
        # Concatenated gzip members keep episode memory bounded, without raw-image copies.
        for data, path in ((native_events, paths.native_events), (forecast, paths.forecast_outcomes)):
            if ep_index and list(pd.read_csv(path,nrows=0).columns) != list(data.columns):
                raise ValueError(f"Native feature schema changed across episodes: {path}")
            data.to_csv(path, index=False, compression="gzip", mode="w" if ep_index == 0 else "a", header=ep_index == 0)
        if options["images"] or options["latents"]:
            if ep_index and list(pd.read_csv(paths.representation_summaries,nrows=0).columns) != list(representations.columns):
                raise ValueError("Representation summary schema changed across episodes")
            representations.to_csv(paths.representation_summaries, index=False, compression="gzip",
                mode="w" if ep_index == 0 else "a", header=ep_index == 0)
        controls = native_events[native_events.is_control_choice == 1]
        feature_audits.append(dict(episode_id=ep["episode_id"], native_events=len(native_events),
            controls=len(controls), forecasts=len(forecast), root_only=int(sum(r["additional_computations"] == 0 for r in native_features.records)),
            stops=int(sum(r["stop"] for r in native_features.records)),
            forced=int(sum(r["forced_stop"] for r in native_features.records)),
            field_validity={name:dict(valid=int(valid.sum()),total=len(valid)) for name,valid in native_features.valid.items()},
            control_voc_validity={name:int(controls[name+"_valid"].sum()) for name in
                ("voc_online_q","voc_ema_q","voc_soft_continue_probability")},
            missing_tree_components=[name for name in TREE_COMPONENTS if name not in decoded]))
        if len(anchors)-1 != meta["replay_edge_count"]:
            raise ValueError("Real-window count differs from captured replay edges")
        audits.append(dict(source_id=meta["source_id"], episode_id=ep["episode_id"], real_steps=len(anchors)-1,
                           expected_real_steps=meta["replay_edge_count"], capture_dir=str(capture),
                           search_events=int(fields["search_computation"].sum()), path_depth_checked=True))
        print(f"  cached {len(windows):,} windows / {len(events):,} computations", flush=True)
    frame = pd.DataFrame(windows)
    frame["run_progress"] = frame.groupby("run_label", sort=False).cumcount() / frame.groupby("run_label")["real_step"].transform("size").clip(lower=1)
    model = frame.copy()
    model["switch_target_uncertainty"] = model["model_switch_target_uncertainty"]
    # A source containing only B=0 decisions still has valid empty side tables.
    event_frame = pd.DataFrame(events) if events else pd.DataFrame(columns=[
        "run_label", "real_step", "real_step_id", "bout_index", "status2_position",
        "depth", "branch", "rollout_return", "return_gain_from_root",
        "persistence_x", "persistence_y", "persistence_evidence", "switch_target_uncertainty",
        "root_q_before_sampled", "root_q_after_sampled", "root_n_before_sampled",
        "root_n_after_sampled", "root_q_revision", "rollout_prediction_error",
        "revision_direction", "directed_value_revision", "directed_prediction_error",
        "value_revision_valid"])
    event_frame = event_frame.sort_values(["run_label", "real_step", "status2_position"])
    bout_frame = pd.DataFrame(bouts) if bouts else pd.DataFrame(columns=[
        "run_label", "real_step", "real_step_id", "bout_index", "branch",
        "first_status2_position", "terminal_status2_position", "reset_known", "terminal_depth", "path",
        "first_rollout_return", "terminal_rollout_return", "return_gain_from_root",
        "return_gain_within_bout", "root_reference", "root_reference_name",
        "incumbent_branch", "presearch_candidate_branch", "start_persistence_evidence",
        "start_switch_target_uncertainty", "terminal_persistence_evidence",
        "terminal_switch_target_uncertainty", "terminal_persistence_x", "terminal_persistence_y",
        "terminal_delta_persistence_evidence", "terminal_delta_switch_target_uncertainty",
        "within_bout_policy_js"])
    for data, path in ((frame, paths.actor_metrics), (frame, paths.root_policy_metrics),
            (frame, paths.persistence_windows), (bout_frame, paths.persistence_bouts),
            (event_frame, paths.persistence_events), (model, paths.model_control_windows),
            (frame, paths.rollout_windows), (pd.DataFrame(alignment), paths.visual_alignment)):
        data.to_csv(path, index=False, compression="gzip")
    prefix = pd.DataFrame(prefixes) if prefixes else pd.DataFrame(columns=[
        "window_index", "event_index", "return_advantage", "event_progress", "pair_valid",
        "same_branch_next", "branch_switch_next", "same_path_deepen_next", "reset_after", "has_next"], dtype=float)
    np.savez_compressed(paths.rollout_prefix_events, **{k: prefix[k].to_numpy() for k in prefix})
    np.savez_compressed(paths.visual_controls, descriptor=np.concatenate(descriptors))
    pd.DataFrame(native_decisions).to_csv(paths.native_decisions, index=False, compression="gzip")
    np.savez_compressed(paths.imagination_paths, **pack_paths(path_rows))
    paths.native_features_schema.write_text(json.dumps(schema_contract(options), indent=2))
    paths.native_features_qa.write_text(json.dumps(dict(episodes=feature_audits,
        decisions=len(native_decisions), paths=len(path_rows),
        human_time_units="seconds; native timestamps, never event index / assumed FPS",
        summary_options=options), indent=2))
    budget = frame.n_status2
    qa = dict(real_steps=len(frame), runs=int(frame.run_label.nunique()), num_actions=num_actions,
              mean_budget=float(budget.mean()), max_budget=int(budget.max()), search_events=int(budget.sum()),
              episode_count=len(episodes), source_count=len({e['metadata']['source_id'] for e in episodes}), episodes=audits)
    if qa["runs"] != qa["source_count"]:
        raise ValueError("Source/run mapping is not one-to-one")
    if len(event_frame) != int(budget.sum()):
        raise ValueError("Some native search computations were lost")
    if len(path_rows) != int(budget.sum()) or len(native_decisions) != len(frame):
        raise ValueError("Native supplement changed decision/computation coverage")
    validate_one_to_one_real_steps({"windows": frame})
    for path in (paths.actor_qa, paths.persistence_qa, paths.model_control_qa, paths.rollout_qa):
        path.write_text(json.dumps(qa, indent=2))
    result = dict(qa, status="COMPLETED", adapter_version=3, shared_cache_version=SHARED_CACHE_VERSION,
        native_feature_options=options, native_feature_version=1,
        native_feature_sources=feature_sources,
        subject=args.subject, game=args.game, source_signature=signature, **provenance,
        capture_dir=episodes[0]["capture_dir"] if len(provenance["captures"]) == 1 else None,
        source_format="native_zarr_v2", num_actions=num_actions, n_actions=num_actions,
        visit_encoding="expm1(native_log1p_visits)/(n_status2+1)",
        actor_alignment="first search actor (or terminal for budget=0) to same-window terminal real-head actor",
        visual_alignment="current pre-action real anchor and preceding real anchor, within episode",
        components={k: {"version":3} for k in ("actor_policy","persistence_action_geometry","model_control","rollout_value")})
    paths.manifest.write_text(json.dumps(result, indent=2))
    print(f"Dynamic cache completed: {paths.root} ({len(frame):,} real steps)", flush=True)
