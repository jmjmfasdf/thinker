"""Causal image-only candidates with a mandatory external annotation gate.

These simple detectors are deliberately not treated as validated game-state
measurements. No urgency inference is permitted before the 200-frame gate.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import label, find_objects
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C


def read_images(path, rows):
    path = Path(path)
    manifest = json.loads((path/"manifest.json").read_text())
    if manifest["state"] != "complete":
        raise ValueError("Cannot read incomplete image store")
    schema = {f["name"]: f for f in manifest["field_schema"]}
    root = zarr.open_consolidated(str(path), mode="r")
    f = schema["real_image"]
    return np.asarray(root["packs"][f["storage_group"]].oindex[np.asarray(rows, int), f["storage_index"]])


def components(mask):
    regions, count = label(mask)
    result = []
    for i, region in enumerate(find_objects(regions), 1):
        if region is None: continue
        y, x = np.nonzero(regions[region]==i)
        if not len(x): continue
        result.append(dict(x=float(x.mean()+region[1].start), y=float(y.mean()+region[0].start),
                           width=region[1].stop-region[1].start, height=region[0].stop-region[0].start, area=len(x)))
    return result


def detect(image, game):
    rgb = np.moveaxis(image, 0, -1).astype(float)
    gray = rgb.mean(axis=2)
    if game == 1:
        mask = gray > 180
        mask[:10] = False
        candidates = components(mask)
        balls = [c for c in candidates if 2 <= c["area"] <= 35 and c["width"] <= 6 and c["height"] <= 6 and 5<c["x"]<79]
        paddles = [c for c in candidates if c["height"] >= 5 and c["width"] <= 5 and c["x"]>55]
        output = []
        if balls: output.append(dict(min(balls, key=lambda c: abs(c["area"]-8)), role="ball"))
        if paddles: output.append(dict(max(paddles, key=lambda c: c["height"]), role="player"))
        return output
    background = np.median(rgb, axis=1, keepdims=True)
    mask = np.linalg.norm(rgb-background, axis=2)>65
    mask[:20] = False; mask[:, :8] = False; mask[:, 76:] = False; mask[80:] = False
    candidates = [c for c in components(mask) if 4 <= c["area"] <= 100 and 2 <= c["width"] <= 16 and 2 <= c["height"] <= 12]
    player = [c for c in candidates if c["y"]>65 and abs(c["x"]-42)<24]
    output = []
    if player:
        selected = max(player, key=lambda c: c["y"])
        output.append(dict(selected, role="player"))
        output += [dict(c, role="obstacle") for c in candidates if c["y"]<selected["y"]-4]
    return output


class Tracker:
    def __init__(self, game):
        self.game, self.previous, self.previous_time = game, [], None

    def step(self, image, time):
        objects = detect(image, self.game)
        result = dict(urgency=np.nan, ttc=np.nan, required_movement=np.nan, closing=0, tracking_valid=0)
        player = next((o for o in objects if o["role"]=="player"), None)
        targets = [o for o in objects if o["role"] != "player"]
        estimates = []
        if player is not None and self.previous_time is not None and time>self.previous_time:
            dt = time-self.previous_time
            for target in targets:
                before = [o for o in self.previous if o["role"]==target["role"]]
                if not before: continue
                old = min(before, key=lambda o: (o["x"]-target["x"])**2+(o["y"]-target["y"])**2)
                if np.hypot(old["x"]-target["x"], old["y"]-target["y"])>15: continue
                vx, vy = (target["x"]-old["x"])/dt, (target["y"]-old["y"])/dt
                result["tracking_valid"] = 1
                if self.game == 1 and vx>1 and player["x"]>target["x"]:
                    ttc = (player["x"]-target["x"])/vx
                    # Reflect constant-velocity intercept within the fixed visible playfield.
                    top, bottom = 10., 82.
                    unfolded = (target["y"]+vy*ttc-top) % (2*(bottom-top))
                    intercept = top+min(unfolded, 2*(bottom-top)-unfolded)
                    estimates.append((ttc, max(0., abs(intercept-player["y"])-player["height"]/2)))
                elif self.game == 0 and vy>1 and target["y"]<player["y"] and abs(target["x"]-player["x"]) < (target["width"]+player["width"])/2:
                    estimates.append(((player["y"]-target["y"])/vy, abs(target["x"]-player["x"])))
        if estimates:
            ttc, movement = min(estimates)
            if 0 < ttc <= 10:
                result.update(ttc=ttc, urgency=1./max(ttc, 1/15.), required_movement=movement, closing=1)
        self.previous, self.previous_time = objects, time
        return result, objects


def validate_annotations(frame, expected_ids, config):
    required = {"frame_id", "game", "role", "visible", "x", "y", "annotator", "predictions"}
    if not required.issubset(frame):
        return dict(passed=False, reason="annotation columns missing")
    if frame.frame_id.nunique() != config["annotations_per_game"] or set(frame.frame_id) != set(expected_ids):
        return dict(passed=False, reason="requires exactly the preselected 200 game frames")
    if frame[["visible", "annotator"]].isna().any().any() or (frame.annotator.astype(str).str.strip()=="").any():
        return dict(passed=False, reason="independent annotation incomplete")
    if not frame.visible.isin([0, 1]).all():
        return dict(passed=False, reason="visible must be exactly 0 or 1")
    visible = frame.visible == 1
    if frame.loc[visible, ["x", "y"]].isna().any().any():
        return dict(passed=False, reason="visible objects have missing reference coordinates")
    successes = []
    # Each frame must have a player annotation and the game's relevant target role.
    for _, rows in frame.groupby("frame_id"):
        game = int(rows.game.iloc[0]); roles = set(rows.role)
        if not {"player", "ball" if game==1 else "obstacle"}.issubset(roles):
            successes.append(False); continue
        pred = json.loads(rows.predictions.iloc[0])
        used = set(); valid = True
        for row in rows.itertuples():
            candidates = [(i, o) for i, o in enumerate(pred) if o["role"]==row.role and i not in used]
            if not row.visible:
                if candidates: valid=False
                continue
            if not candidates:
                valid=False; continue
            index, best = min(candidates, key=lambda item: np.hypot(item[1]["x"]-row.x, item[1]["y"]-row.y))
            if np.hypot(best["x"]-row.x, best["y"]-row.y)>config["tracker_error_pixels"]: valid=False
            used.add(index)
        if any(i not in used for i in range(len(pred))): valid=False
        successes.append(valid)
    accuracy = float(np.mean(successes))
    return dict(passed=accuracy>=config["tracker_accuracy"], frame_accuracy=accuracy,
                frames=len(successes), reason="" if accuracy>=config["tracker_accuracy"] else "tracker failed 90% within-2-pixel criterion")


def main():
    p = C.parser(__doc__)
    p.add_argument("--annotations", type=Path)
    args = p.parse_args()
    d, pixels, cache = C.load_cache(args, visuals=True)
    inputs = dict(cache=cache["signature"], annotations=C.sha(args.annotations) if args.annotations else None)
    module_name = C.MODULES["06"]
    if args.annotations:
        module_name += "_validated_"+C.sha(args.annotations)[:12]
    root, manifest, done = C.start_manifest(args, module_name, inputs)
    if done: return
    manifest["smoke"] = cache["smoke"]
    rng = np.random.default_rng(C.settings(args)["seed"]+args.subject*10+args.game)
    quota = 34 if args.subject<=2 else 33
    selected = []
    strata = list(d.groupby("session", sort=True))
    for i, (_, rows) in enumerate(strata):
        take = quota//len(strata)+int(i<quota%len(strata))
        selected.extend(rng.choice(rows.index, size=min(take, len(rows)), replace=False).tolist())
    selected = set(selected)
    (root/"annotation_images").mkdir(exist_ok=True)
    records, samples, thumbnails = [], [], []
    for episode, rows in d.groupby("episode", sort=False):
        tracker = Tracker(args.game)
        image_rows = rows.pre_image_event.to_numpy(int)
        path = rows.capture_episode.iloc[0]
        for start in range(0, len(rows), 256):
            images = read_images(path, image_rows[start:start+256])
            for (idx, row), image in zip(rows.iloc[start:start+256].iterrows(), images):
                value, objects = tracker.step(image, row.time)
                records.append(dict(decision_id=row.decision_id, origin_time=row.time, available_time=row.time, **value))
                if idx in selected:
                    frame_id = f"{C.tag(args)}_{row.decision_id}"
                    samples.append(dict(frame_id=frame_id, subject=args.subject, game=args.game, session=row.session,
                        episode=episode, time=row.time, capture_episode=path, event=row.pre_image_event,
                        predictions=json.dumps(objects)))
                    thumbnails.append((frame_id, image, objects))
                    plt.imsave(root/"annotation_images"/(frame_id+".png"), np.moveaxis(image, 0, -1))
    features = pd.DataFrame(records)
    features.to_csv(root/"urgency_candidates.csv.gz", index=False)
    sample_table = pd.DataFrame(samples)
    sample_table.to_csv(root/"annotation_frames.csv", index=False)
    template = []
    for row in samples:
        for role in ("player", "ball" if args.game==1 else "obstacle"):
            template.append(dict(row, role=role, visible=np.nan, x=np.nan, y=np.nan, annotator=""))
    pd.DataFrame(template).to_csv(root/"annotation_template.csv", index=False)
    fig, ax = plt.subplots(2, 3, figsize=(10, 7))
    for axis, (name, image, objects) in zip(ax.ravel(), thumbnails[:6]):
        axis.imshow(np.moveaxis(image, 0, -1)); axis.set_title(name, fontsize=8)
        for obj in objects:
            axis.plot(obj["x"], obj["y"], "rx"); axis.text(obj["x"], obj["y"], obj["role"], color="yellow", fontsize=6)
    fig.suptitle("Unvalidated causal object candidates: no urgency inference yet")
    C.save_figure(fig, root, "figure6A_tracker_candidates")
    gate = dict(passed=False, reason="independent 200-frame game annotation has not been supplied")
    if args.annotations:
        annotation = pd.read_csv(args.annotations)
        if "game" not in annotation:
            raise ValueError("Annotations require game identity")
        annotation = annotation[annotation.game==args.game].copy()
        all_frames = []
        for subject in range(1, 7):
            f = args.output_root/C.MODULES["06"]/f"sub{subject:03d}_game{args.game}"/"annotation_frames.csv"
            if not f.exists(): raise ValueError("Prepare annotation frames for all six subjects before validation")
            all_frames.append(pd.read_csv(f))
        all_frames = pd.concat(all_frames, ignore_index=True)
        annotation = annotation.drop(columns=["predictions"], errors="ignore").merge(
            all_frames[["frame_id", "predictions"]], on="frame_id", how="left", validate="many_to_one")
        gate = validate_annotations(annotation, all_frames.frame_id, C.settings(args))
    C.write_json(root/"tracker_validation.json", gate)
    if gate["passed"]:
        from analysis import Reporter, effect_plot, calibration, overview_axes
        from models import make_spec
        merged = d.merge(features.drop(columns=["origin_time", "available_time"]), on="decision_id", validate="one_to_one")
        for col in ("evidence_signed", "budget"):
            merged["urgency_x_"+col] = merged.urgency*merged[col]
        r = Reporter(args, "06", merged, pixels, cache, existing=(root, manifest, False))
        # Reuse this module's already-created manifest with its annotation identity.
        r.root, r.manifest, r.done = root, manifest, False
        b = r.blocks
        main = ["urgency", "ttc", "required_movement", "closing", "tracking_valid"]
        r.specs["urgency_main"] = make_spec("urgency_main", b, ["H", "V", "VL", "P", "B", "E"], extra=main)
        r.specs["urgency_interaction"] = make_spec("urgency_interaction", b, ["H", "V", "VL", "P", "B", "E"],
            extra=main+["urgency_x_evidence_signed", "urgency_x_budget"])
        r.contrast("compute", "urgency_main", "object_environment_increment")
        r.contrast("urgency_main", "urgency_interaction", "urgency_interactions")
        fig, ax = overview_axes("Validated urgency associations (not exact deadlines)")
        effect_plot(ax[0], r.effects)
        pred, _ = r.fit("urgency_interaction"); calibration(ax[1], pred)
        bins = pd.qcut(merged.urgency, 5, duplicates="drop")
        stats = merged.groupby(bins, observed=True)[["budget", "switch", "evidence_signed"]].mean()
        ax[2].plot(stats.budget.to_numpy(), "o-"); ax[2].set(title="Budget by urgency quantile")
        ax[3].plot(stats.switch.to_numpy(), "o-"); ax[3].set(title="Observed switch rate by urgency quantile")
        C.save_figure(fig, root, "figure6_urgency_modulation")
        r.finish(dict(tracker_validation=gate, urgency_inference="EXECUTED"))
    else:
        pd.DataFrame([dict(subject=args.subject, game=args.game, contrast="urgency_interactions", family="06", primary="",
            outcome="switch", delta_ll=np.nan, estimable=False, reason=gate["reason"])]).to_csv(root/"effects.csv", index=False)
        pd.DataFrame(columns=["contrast", "run", "delta_ll"]).to_csv(root/"run_effects.csv", index=False)
        C.write_json(root/"qa.json", dict(urgency_inference="NOT_ESTIMABLE", reason=gate["reason"],
            fallback="Validated current/past visual and motion controls remain in analyses 01-05"))
        C.finish(root, manifest, urgency_inference="NOT_ESTIMABLE", tracker_validation=gate,
                 annotation_frames=len(samples))


if __name__ == "__main__":
    main()
