"""Verify artifacts after the three subject analyses complete."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from types import SimpleNamespace
from dynamic_analysis_cache import capture_inputs

ANALYSES = ("02_evaluation_revision_geometry", "03_actor_integration_mechanisms",
            "04_search_evidence_geometry_pathway_v2")


def verify_04_manifest(root, manifest):
    """Verify solver/source provenance and every declared numeric artifact."""
    script = Path(__file__).resolve().parent / "04_search_evidence_geometry_pathway_v2.py"
    assert manifest["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert manifest["actor_softmax_maxiter"] == 1000
    assert manifest["stage3_min_train_rows"] == 400
    for table in manifest["tables"].values():
        path = Path(table["path"])
        assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size > 0
        assert hashlib.sha256(path.read_bytes()).hexdigest() == table["sha256"]


def verify_group(outputs, game, only_04=False):
    subjects = list(range(1, 7))
    reports = [json.loads((outputs / "dynamic_jobs" / f"sub{s:03d}_game{game}.completed.json").read_text()) for s in subjects]
    assert all(r["status"] == "COMPLETED" for r in reports)
    tables = (("results/subject_effects.csv", "results/group_effects.csv"),
              ("tables/subject_effects.csv", "tables/group_inference.csv"),
              ("tables/group_subject_effect_matrix_long.csv", "tables/group_stagewise_exact_inference.csv"))
    result = dict(status="COMPLETED", game=game, subjects=subjects, analyses={})
    for name, required in zip(ANALYSES, tables):
        if only_04 and name != ANALYSES[-1]:
            continue
        root = outputs / name / f"group_game{game}"
        manifest = json.loads((root / "analysis_manifest.json").read_text())
        assert manifest["subjects"] == subjects
        if only_04:
            verify_04_manifest(root, manifest)
        figures = list((root / "figures").rglob("*.png"))
        assert figures and all(f.stat().st_size for f in figures)
        frames = [pd.read_csv(root / path) for path in required]
        assert all(len(f) for f in frames)
        assert set(frames[0].subject.unique()) == set(subjects)
        result["analyses"][name] = dict(figures=len(figures), table_rows=dict(zip(required, map(len, frames))))
    result["stage1_not_estimable_effect_rows"] = sum(r["stage1_not_estimable_effect_rows"] for r in reports)
    (outputs / "dynamic_jobs" / f"group_game{game}.completed.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def verify_cache_alignment(cache_root):
    keys = ["run_label", "real_step", "real_step_id"]
    windows = pd.read_csv(cache_root / "actor_metrics.csv.gz", usecols=keys + [
        "episode_id", "episode_start", "incumbent_action", "n_status2",
        "native_anchor_before", "native_anchor_after", "current_frame", "previous_frame"])
    assert not windows.duplicated(keys).any()
    starts = windows.groupby("episode_id", sort=False).cumcount().eq(0)
    assert windows.episode_start.eq(starts.astype(int)).all()
    assert windows.loc[starts, "incumbent_action"].eq(-1).all()
    assert windows.loc[starts, "previous_frame"].eq(-1).all()
    assert (windows.loc[~starts, "previous_frame"] <= windows.loc[~starts, "current_frame"]).all()
    assert (windows.native_anchor_before < windows.native_anchor_after).all()
    for _, episode in windows.groupby("episode_id", sort=False):
        np.testing.assert_array_equal(episode.native_anchor_before.to_numpy()[1:],
                                      episode.native_anchor_after.to_numpy()[:-1])
        np.testing.assert_array_equal(episode.previous_frame.to_numpy()[1:],
                                      episode.current_frame.to_numpy()[:-1])
    for _, run in windows.groupby("run_label", sort=False):
        boundary = run.episode_id.ne(run.episode_id.shift()) & run.index.to_series().ne(run.index[0])
        assert run.real_step.diff().loc[boundary].gt(1).all()
    alignment = pd.read_csv(cache_root / "visual_alignment.csv.gz")
    assert windows[keys].equals(alignment[keys])
    with np.load(cache_root / "visual_controls.npz", allow_pickle=False) as data:
        descriptor = data["descriptor"]
    assert descriptor.shape == (len(windows), 40)
    assert np.isnan(descriptor[starts]).all()
    assert np.isfinite(descriptor[~starts]).all()
    events = pd.read_csv(cache_root / "persistence_events.csv.gz", usecols=keys + ["status2_position"])
    assert not events.duplicated(keys + ["status2_position"]).any()
    counts = events.groupby(keys).size().reindex(pd.MultiIndex.from_frame(windows[keys]), fill_value=0)
    np.testing.assert_array_equal(counts.to_numpy(), windows.n_status2.to_numpy())
    return {"episode_boundaries_checked": int(starts.sum()), "visual_past_only_checked": True,
            "ragged_event_count_checked": len(events), "budget_zero_windows": int(windows.n_status2.eq(0).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--outputs-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--capture-index", type=Path)
    parser.add_argument("--legacy-sub001-fixture", action="store_true")
    parser.add_argument("--group", action="store_true")
    parser.add_argument("--only-04", action="store_true")
    parser.add_argument("--shared-cache-root", type=Path)
    args = parser.parse_args()
    outputs = args.outputs_root.resolve()
    if args.group:
        verify_group(outputs, args.game, only_04=args.only_04)
        return
    tag = f"sub{args.subject:03d}_game{args.game}"
    cache_root = args.shared_cache_root or outputs / "00_dynamic_cache"
    cache = json.loads((cache_root / tag / "manifest.json").read_text())
    assert (cache["status"], cache["subject"], cache["game"]) == ("COMPLETED", args.subject, args.game)
    assert cache["num_actions"] == {0:9, 1:6}[args.game]
    assert cache["real_steps"] > 0
    if args.legacy_sub001_fixture:
        assert args.subject == 1
        expected = {0: (79136, .7875935099069955, 30, 9), 1:(79137, .9363508851738125, 15, 6)}[args.game]
        assert (cache["real_steps"], cache["max_budget"], cache["num_actions"]) == (expected[0], expected[2], expected[3])
        assert np.isclose(cache["mean_budget"], expected[1], atol=1e-9)
    if args.capture_index:
        episodes, provenance = capture_inputs(SimpleNamespace(capture_index=args.capture_index,
            capture_dir=None, subject=args.subject, game=args.game, sessions=None, max_files=None))
        assert cache["captures"] == provenance["captures"]
        assert cache["source_runs"] == provenance["source_runs"]
        assert cache["episode_count"] == len(episodes)
        assert cache["real_steps"] == sum(e["metadata"]["replay_edge_count"] for e in episodes)
    assert cache["runs"] == cache["source_count"] == 11
    alignment_qa = verify_cache_alignment(cache_root / tag)
    assert alignment_qa["ragged_event_count_checked"] == cache["search_events"]
    analyses = {
        "02_evaluation_revision_geometry": ("results/evaluation_revision_window_metrics.csv.gz", "results/run_effects.csv"),
        "03_actor_integration_mechanisms": ("tables/run_effects.csv", "tables/loro_readout_folds.csv"),
        "04_search_evidence_geometry_pathway_v2": ("tables/stage1_event_to_future_geometry_run_effects.csv", "tables/stage2_early_late_run_effects.csv", "tables/stage3_nested_loro_fold_effects.csv"),
    }
    result = {"status":"COMPLETED", "subject":args.subject, "game":args.game, "cache":cache,
              "shared_cache_root": str(cache_root.resolve()), "only_04": args.only_04,
              "alignment_qa": alignment_qa, "analyses":{}}
    for name, tables in analyses.items():
        if args.only_04 and name != ANALYSES[-1]:
            continue
        root = outputs / name / tag
        analysis_manifest = json.loads((root / "analysis_manifest.json").read_text())
        if args.only_04:
            verify_04_manifest(root, analysis_manifest)
            assert (analysis_manifest["subject"], analysis_manifest["game"]) == (args.subject, args.game)
            actor_qa = json.loads((root / "qa/actor_loro_from_basic_cache.json").read_text())["actor_qa"]
            assert actor_qa["loro_folds"] == 11 and actor_qa["actor_softmax_maxiter"] == 1000
            folds = pd.read_csv(root / "tables/actor_loro_folds_from_basic_cache.csv")
            assert len(folds) == folds.run_label.nunique() == 11
            assert analysis_manifest["stage3_qa"]["minimum_train_rows"] == 400
            stage3_folds = pd.read_csv(root / "tables/stage3_nested_loro_fold_effects.csv")
            assert len(stage3_folds) == 66
            assert stage3_folds.groupby(["transition_type", "increment"]).run_label.nunique().eq(11).all()
            assert stage3_folds.n_train.ge(400).all() and stage3_folds.n_test.gt(0).all()
            assert (stage3_folds.train_events.gt(0) & stage3_folds.train_events.lt(stage3_folds.n_train)).all()
            assert np.isfinite(stage3_folds.beta).all()
        figures = list((root / "figures").rglob("*.png"))
        assert figures and all(f.stat().st_size > 0 for f in figures)
        counts = {}
        for filename in tables:
            table = pd.read_csv(root / filename)
            assert len(table) > 0, filename
            counts[filename] = len(table)
        result["analyses"][name] = {"figures": len(figures), "table_rows": counts}
        if name.startswith("02_"):
            assert json.loads((root / "results/qa_summary.json").read_text())
        elif name.startswith("03_"):
            assert analysis_manifest["real_head_alignment_qa"]
            assert len(pd.read_csv(root / "tables/coverage.csv"))
        else:
            for qa_name in ("actor_loro_from_basic_cache.json", "visual_state_controls.json"):
                assert json.loads((root / "qa" / qa_name).read_text())
    stage1 = pd.read_csv(outputs / "04_search_evidence_geometry_pathway_v2" / tag / analyses["04_search_evidence_geometry_pathway_v2"][0])
    missing = stage1.loc[stage1.beta.isna()]
    assert missing.not_estimable_reason.fillna("").str.len().gt(0).all()
    missing_qa = pd.read_csv(outputs / "04_search_evidence_geometry_pathway_v2" / tag / "qa/stage1_not_estimable.csv")
    assert len(missing_qa) == len(missing)
    result["stage1_not_estimable_effect_rows"] = len(missing)
    result["stage1_not_estimable_reasons"] = missing.not_estimable_reason.value_counts().to_dict()
    path = outputs / "dynamic_jobs" / f"{tag}.completed.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "cache"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
