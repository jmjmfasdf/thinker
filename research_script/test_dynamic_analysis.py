"""Regression tests for variable-length input and temporal alignment."""
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import json
import hashlib
import numpy as np
import pandas as pd
from dynamic_analysis_cache import load_module, reconstruct_paths, visual_descriptor, capture_inputs

HERE = Path(__file__).resolve().parent
P = load_module("test_pathway", HERE / "04_search_evidence_geometry_pathway_v2.py")
I = load_module("test_integration", HERE / "03_actor_integration_mechanisms.py")
S = load_module("test_submission", HERE / "submit_heldout_analysis.py")


def fixture(lengths):
    windows, events = [], []
    for row, length in enumerate(lengths):
        window = dict(run_label="run", real_step=row+1, real_step_id=f"r{row}", actor_row=row,
            subject=1, game=1, incumbent_action=0, human_action=1, action_before_incumbent=2,
            strongest_competitor_margin=.2, bout_age=1., prior_reward=0., prior_done=0.,
            run_progress=.1, value_revision_event_fraction=1., n_status2=length)
        for a in range(6):
            for key in ("model_root_p", "actor_terminal_p"):
                window[f"{key}_{a}"] = 1/6
            window[f"root_q_initial_{a}"] = a
            window[f"root_n_initial_{a}"] = 1.
        windows.append(window)
        for j in range(length):
            events.append(dict({k: window[k] for k in P.KEYS}, status2_position=j,
                branch=j%6, depth=j+1, return_advantage=float(j), root_q_revision=.1*j,
                directed_value_revision=.1*j, valid_repeated_q_revision=1, root_n_before_sampled=1.))
    return pd.DataFrame(windows), pd.DataFrame(events)


class DynamicAnalysisTests(unittest.TestCase):
    def test_stage3_binary_minimum_is_400(self):
        rng = np.random.default_rng(2026)
        self.assertEqual(P.STAGE3_MIN_TRAIN_ROWS, 400)
        for count in (399, 400, 424, 494):
            x = rng.normal(size=(count, 3))
            y = np.arange(count) % 2
            fit = P._fit_binary_l2(x, y)
            if count < 400:
                self.assertIsNone(fit)
            else:
                self.assertIsNotNone(fit)
                self.assertTrue(np.isfinite(P._predict_binary(fit, x)).all())
                self.assertEqual(fit["model"].C, 1.0)
        self.assertIsNone(P._fit_binary_l2(np.zeros((424, 3)), np.zeros(424)))

    def test_stage3_loro_accepts_400_per_training_fold(self):
        rng = np.random.default_rng(2026)
        records = []
        continuous = ["transition_margin_M", "R_log", "prior_reward", "prior_done",
            "bout_age", "run_progress", "value_revision_event_fraction", "aligned_E_mean",
            "aligned_E_slope", "aligned_signed_q_mean", "aligned_signed_q_slope",
            "directed_q_mean", "directed_q_slope", "signed_q_available", "signed_q_event_count",
            "destination_share_shift", "destination_depth_shift", "actor_residual_beyond_P",
            "actor_geometry_transition_increment"]
        for run in range(11):
            for incumbent in (0, 1):
                for position in range(40):
                    records.append(dict(zip(continuous, rng.uniform(.1, 1., len(continuous))),
                        run_label=f"run{run}", actor_row=len(records), incumbent_action=incumbent,
                        human_action=position % 2, action_before_incumbent=1-incumbent))
        summary = pd.DataFrame(records)
        visual = rng.normal(size=(len(summary), 40))
        with patch.object(P, "N_ACTIONS", 6):
            folds, predictions, coefficients, qa = P.fit_stage3_loro(summary, visual, 1, 1)
        self.assertEqual(len(folds), 66)
        self.assertTrue(folds.n_train.eq(400).all())
        self.assertTrue(folds.n_test.eq(40).all())
        self.assertTrue(np.isfinite(folds.beta).all())
        self.assertEqual(qa["minimum_train_rows"], 400)
        self.assertEqual(len(predictions), len(summary))

    def test_submission_uses_only_slave5_to_8_and_balances_subjects(self):
        nodes = [S.assigned_node(subject, game) for subject in range(1, 7) for game in (0, 1)]
        self.assertEqual(set(nodes), {f"bml2-slave{i}" for i in range(5, 9)})
        self.assertTrue(all(nodes.count(node) == 3 for node in set(nodes)))
        self.assertTrue(all(S.assigned_node(0, game) in nodes for game in (0, 1)))

    def test_actor_solver_uses_1000_and_rejects_nonconvergence(self):
        n = 500
        features = {"x": np.zeros((n, P.N_ACTIONS))}
        target = np.full((n, P.N_ACTIONS), 1 / P.N_ACTIONS)
        result = SimpleNamespace(success=True, x=np.zeros(P.N_ACTIONS), fun=0.,
                                 nit=205, status=0, message="converged")
        with patch.object(P, "minimize", return_value=result) as minimize:
            fit = P.fit_softmax_soft_targets(features, target, np.ones(n, dtype=bool))
            self.assertIsNotNone(fit)
            self.assertEqual(minimize.call_args.kwargs["options"], {"maxiter": 1000})
            self.assertEqual(minimize.call_args.kwargs["method"], "L-BFGS-B")
            result.success, result.nit, result.status = False, 1000, 1
            result.message = "TOTAL NO. OF ITERATIONS REACHED LIMIT"
            self.assertIsNone(P.fit_softmax_soft_targets(features, target, np.ones(n, dtype=bool)))

    def collection(self, root):
        roots, checkpoints, tasks = [], [], []
        for session in range(1, 5):
            capture = root / f"ses{session}"
            capture.mkdir()
            checkpoint = f"/checkpoints/sub1-leavesession{session}"
            episodes = []
            for episode in range(2 if session == 1 else 1):
                path = f"episode{episode}.zarr"
                (capture / path).mkdir()
                episodes.append(dict(path=path, episode_id=f"s{session}e{episode}", metadata=dict(
                    subject=1, game=0, session=session, block=1, source_id=f"source{session}",
                    source_sha256=f"hash{session}", source_episode_index=episode, checkpoint_real_step=5000000)))
            manifest = dict(episodes=episodes, semantic=dict(checkpoint=dict(
                logical_xpid=Path(checkpoint).name, milestone_sha256=f"milestone{session}")))
            digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            manifest["manifest_sha256"] = digest
            (capture / "manifest.json").write_text(json.dumps(manifest))
            (capture / "COMPLETED.json").write_text(json.dumps(dict(status="COMPLETED", manifest_sha256=digest, episode_count=len(episodes))))
            roots.append(str(capture))
            checkpoints.append(checkpoint)
            tasks.append(dict(capture=str(capture), checkpoint=checkpoint, subject=1, game=0,
                session=session, milestone_sha256=f"milestone{session}", sources=[dict(source_id=f"source{session}", sha256=f"hash{session}")]))
        (root / "subjects").mkdir()
        index = root / "subjects/sub001_game0.json"
        index.write_text(json.dumps(dict(status="COMPLETED", subject=1, game=0, capture_roots=roots,
            checkpoints=checkpoints, source_count=4, assignment="held-out")))
        (root / "campaign.json").write_text(json.dumps(dict(target_step=5000000, tasks=tasks)))
        return SimpleNamespace(capture_index=index, capture_dir=None, subject=1, game=0, sessions=None, max_files=None)

    def test_collection_preserves_sources_and_episode_origins(self):
        with TemporaryDirectory() as temp:
            args = self.collection(Path(temp))
            episodes, provenance = capture_inputs(args)
            self.assertEqual(len(episodes), 5)
            self.assertEqual(len(provenance["source_runs"]), 4)
            self.assertEqual(len(provenance["captures"]), 4)
            args.max_files = 1
            selected, _ = capture_inputs(args)
            self.assertEqual(len(selected), 2)  # do not truncate a split source
            args.capture_dir = Path(episodes[0]["capture_dir"])
            args.capture_index = None
            self.assertEqual(len(capture_inputs(args)[0]), 2)

    def test_collection_rejects_wrong_identity_and_duplicate_capture(self):
        with TemporaryDirectory() as temp:
            args = self.collection(Path(temp))
            args.game = 1
            with self.assertRaisesRegex(ValueError, "subject/game"):
                capture_inputs(args)
            args.game = 0
            index = json.loads(args.capture_index.read_text())
            index["capture_roots"][1] = index["capture_roots"][0]
            args.capture_index.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "distinct"):
                capture_inputs(args)

    def test_collection_rejects_wrong_checkpoint_assignment(self):
        with TemporaryDirectory() as temp:
            args = self.collection(Path(temp))
            path = Path(temp) / "campaign.json"
            campaign = json.loads(path.read_text())
            campaign["tasks"][0]["session"] = 2
            path.write_text(json.dumps(campaign))
            with self.assertRaisesRegex(ValueError, "checkpoint/session"):
                capture_inputs(args)

    def test_collection_rejects_changed_manifest(self):
        with TemporaryDirectory() as temp:
            args = self.collection(Path(temp))
            path = Path(temp) / "ses1/manifest.json"
            manifest = json.loads(path.read_text())
            manifest["episodes"][0]["metadata"]["subject"] = 2
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                capture_inputs(args)

    def test_all_lengths_and_strict_future(self):
        windows, events = fixture([0, 1, 2, 29, 30])
        result, _ = P.build_transition_suffix_run(events, windows)
        counts = result.groupby("real_step").size().to_dict()
        self.assertEqual(counts, {3:1, 4:28, 5:29})
        self.assertTrue((result.n_future_events >= 1).all())
        last = result[(result.real_step==5)&(result.event_index==28)].iloc[0]
        self.assertEqual(last.n_future_events, 1)
        self.assertEqual(last.future_destination_depth_score, 30.)

    def test_fixed_29_horizon_compatibility(self):
        windows, events = fixture([29])
        old = P.MIN_FUTURE
        try:
            P.MIN_FUTURE = 6
            result, _ = P.build_transition_suffix_run(events, windows)
        finally:
            P.MIN_FUTURE = old
        self.assertEqual(len(result), 23)
        for _, row in result.iterrows():
            j = int(row.event_index)
            branches = np.arange(j+1,29)%6
            signs = np.where(branches==0, -1., 1.)
            self.assertAlmostEqual(row.future_destination_share_contrast, signs.mean())
        self.assertEqual(29//2, 14)

    def test_truncated_window_is_rejected(self):
        windows, events = fixture([3])
        with self.assertRaises(ValueError):
            list(P._event_length_groups(events.iloc[:-1], windows))

    def test_early_late_uses_actual_window_halves(self):
        windows, events = fixture([0,1,2,29,30])
        actor = windows[["actor_row"]].copy()
        for model in ("P", "A2", "A3"):
            for a in range(6):
                actor[f"actor_{model}_oof_p_{a}"] = 1/6
        result, _ = P.build_window_pathway_summary(events, windows, actor)
        self.assertEqual(result.n_status2.tolist(), [2,29,30])
        for _, row in result.iterrows():
            length = int(row.n_status2)
            sign = np.where(np.arange(length)%6==0,-1.,1.)
            self.assertAlmostEqual(row.early_destination_share, sign[:length//2].mean())
            self.assertAlmostEqual(row.late_destination_share, sign[length//2:].mean())
        self.assertTrue(np.isnan(result.iloc[0].aligned_E_slope))

    def test_reset_path_and_depth_consistency(self):
        paths, depth = reconstruct_paths(np.arange(4), np.array([1,2,3,4]),
            np.array([0,1,2,0]), np.array([1,2,0,1]), np.array([False,False,True,False]))
        np.testing.assert_array_equal(depth, [1,2,0,1])
        np.testing.assert_array_equal(paths[1,:2], [1,2])
        np.testing.assert_array_equal(paths[3,:1], [4])
        with self.assertRaises(ValueError):
            reconstruct_paths(np.arange(1), np.array([1]), np.array([2]), np.array([3]), np.array([False]))

    def test_short_budget_history_and_episode_boundary(self):
        windows, events = fixture([1, 3, 0, 3])
        windows["episode_id"] = ["first", "first", "second", "second"]
        windows["incumbent_action"] = [4, 0, 0, 0]
        actor = windows[["actor_row"]].copy()
        for model in ("P", "A2", "A3"):
            for a in range(6):
                actor[f"actor_{model}_oof_p_{a}"] = 1/6
        result, _ = P.build_window_pathway_summary(events, windows, actor)
        self.assertEqual(result.actor_row.tolist(), [1, 3])
        self.assertEqual(result.pre_bout_active_action.iloc[0], 4)
        self.assertTrue(np.isnan(result.pre_bout_active_action.iloc[1]))

    def test_missing_moderation_has_effect_specific_reason(self):
        windows, events = fixture([3])
        suffix, _ = P.build_transition_suffix_run(events, windows)
        fit = {"focal_beta": .2, "interaction_beta": np.nan, "n_rows": 100, "n_windows": 50}
        with patch.object(P, "_event_design_fit", return_value=(fit, {})):
            effects, _ = P.fit_stage1_run(suffix, 1, 1)
        moderation = effects.loc[effects.effect_role.eq("margin_moderation")]
        self.assertTrue((~moderation.estimable).all())
        self.assertTrue(moderation.not_estimable_reason.str.len().gt(0).all())

    def test_native_visual_formula_matches_existing(self):
        rng = np.random.default_rng(22)
        current = rng.integers(0,256,(3,3,84,84),dtype=np.uint8)
        past = rng.integers(0,256,(3,3,84,84),dtype=np.uint8)
        actual = visual_descriptor(current, past)
        expected = P.low_level_visual_descriptor(np.moveaxis(current,1,-1),np.moveaxis(past,1,-1))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_missing_effect_family_never_becomes_zero(self):
        effects = pd.DataFrame({"run_label":["a","b"],"effect_id":["test","test"],"beta":[1.,np.nan]})
        result, _ = P.exact_unit_inference(effects,["test"],"run_label")
        self.assertTrue(np.isnan(result.synchronized_max_t_p.iloc[0]))
        self.assertEqual(result.mean_beta.iloc[0],1.)


if __name__ == "__main__":
    unittest.main()
