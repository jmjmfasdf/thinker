"""Deterministic contract tests; real-data smoke is a separate acceptance stage."""
import json
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

import common as C
from native_cache import reconstruct_paths, forecast_alignment, visual_features, add_history
from models import Choice, outer_folds, inner_folds, compare, make_spec, check_control_features, Spec, Design
from urgency import Tracker, validate_annotations


class AlignmentTests(unittest.TestCase):
    def test_ragged_budgets(self):
        for budget in (0, 1, 2, 29, 30):
            ix = np.arange(budget)
            paths = reconstruct_paths(ix, np.ones(budget, int), ix, ix+1, np.zeros(budget, bool))
            self.assertEqual(len(paths), budget)
            self.assertEqual(max([len(p) for p in paths]+[0]), budget)

    def test_reset_and_forced_terminal_are_not_voluntary_stop(self):
        actions = np.array([2, 1, 0, 3])
        paths = reconstruct_paths(np.arange(4), actions, [0, 1, 2, 0], [1, 2, 0, 1], [False, False, True, False])
        self.assertEqual(paths, [[2], [2, 1], [], [3]])
        accepted = np.array([0, 0, 1]); forced = np.array([False, False, True])
        self.assertEqual(int((accepted==2).sum()), 0)
        self.assertTrue(forced[-1])
        self.assertEqual(len(accepted), 3)  # no fabricated fourth STOP opportunity

    def test_forecast_off_by_one(self):
        a = np.array([2, 3, 1, 0])
        self.assertEqual(forecast_alignment(0, [2, 3], a), (1, 2, True, True))
        self.assertEqual(forecast_alignment(0, [2, 0], a), (1, 2, True, False))
        self.assertEqual(forecast_alignment(1, [0, 1], a), (2, 3, False, False))

    def test_future_availability_fails(self):
        C.assert_available([0, 1], [1, 2], [1, 2])
        with self.assertRaises(ValueError): C.assert_available(0, 2, 1)
        with self.assertRaises(ValueError): C.assert_available(2, 1, 2)

    def test_visual_causal(self):
        rng = np.random.default_rng(4)
        images = rng.integers(0, 255, (5, 3, 84, 84), dtype=np.uint8)
        d, raw = visual_features(images)
        changed = images.copy(); changed[3:] = 0
        d2, raw2 = visual_features(changed)
        np.testing.assert_allclose(d[:3], d2[:3], equal_nan=True)
        np.testing.assert_allclose(raw[:3], raw2[:3], equal_nan=True)
        self.assertEqual(d.shape, (5, 40)); self.assertEqual(raw.shape, (5, 1536))

    def test_head_semantics(self):
        # There is no pre-search REAL head in the feature contract/specs.
        with self.assertRaises(ValueError): make_spec("bad", {"H": ["human_action"]}, ["H"])
        with self.assertRaises(ValueError): make_spec("bad", {"H": ["forced_stop"]}, ["H"])

    def test_no_future_budget_in_control_model(self):
        for outcome in ("begin", "stop"):
            with self.assertRaises(ValueError): check_control_features(Spec("bad", ("budget",)), outcome)
            with self.assertRaises(ValueError): check_control_features(Spec("bad", ("actor_p_0",)), outcome)
            check_control_features(Spec("good", ("event_index", "prior_evidence_abs")), outcome)

    def test_episode_boundary_and_reoriented_bout_evidence(self):
        records = []
        for i, action in enumerate([0, 1, 1, 1, 0, 0]):
            episode = "A" if i<4 else "B"
            local = i if i<4 else i-4
            record = dict(decision_id=i, episode=episode, run="one", time=local/15., origin_time=local/15.,
                available_time=local/15., human_action=action, evidence_signed=-3., evidence_abs=3., budget=1,
                root_margin=0., root_value=0., support_0=3., support_1=0.)
            record.update({f"visual_{v}": float(i) for v in range(40)})
            records.append(record)
        result = add_history(pd.DataFrame(records), pd.DataFrame(), 2)
        self.assertEqual(result.loc[2, "history_previous_evidence"], 3.)
        self.assertTrue(np.isnan(result.loc[4, "history_previous_evidence"]))
        self.assertEqual(result.loc[4, "previous_action_1"], -1)
        self.assertEqual(result.loc[4, "lag0_coverage"], 0)


class CVTests(unittest.TestCase):
    def fixture(self):
        records = []
        for session in (1, 2):
            for block in (1, 2, 3):
                for i in range(60):
                    records.append(dict(session=session, run=f"s{session}b{block}", episode=f"s{session}b{block}e{i//30}",
                        checkpoint=f"heldout{session}", time=i, row_id=len(records)))
        return pd.DataFrame(records)

    def test_same_session_checkpoint_and_whole_source_fold(self):
        d = self.fixture()
        for _, train, test in outer_folds(d):
            self.assertEqual(d.loc[train, "session"].nunique(), 1)
            self.assertEqual(set(d.loc[train, "checkpoint"]), set(d.loc[test, "checkpoint"]))
            self.assertFalse(set(d.loc[train, "run"]) & set(d.loc[test, "run"]))
            self.assertFalse(set(d.loc[train, "episode"]) & set(d.loc[test, "episode"]))

    def test_forward_inner_embargo(self):
        d = self.fixture().iloc[:60].copy()
        d["episode"] = "one"; d["time"] = np.arange(60)
        for train, test in inner_folds(d, d.index.to_numpy(), 8.):
            self.assertLess(d.loc[train, "time"].max(), d.loc[test, "time"].min()-8)

    def test_constant_and_missing_group_effects(self):
        t, raw, corrected = C.exact_max_t(np.column_stack((np.ones(6), np.zeros(6))))
        self.assertAlmostEqual(raw[0], 2/64)
        self.assertEqual(corrected[1], 1.)
        x = np.ones((6, 1)); x[0] = np.nan
        self.assertTrue(np.isnan(C.exact_max_t(x)[2][0]))
        with self.assertRaises(ValueError): C.exact_max_t(np.ones((5, 1)))

    def test_conditional_action_space(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(120, 2)); inc = np.arange(120)%3
        y = (inc+1+(x[:, 0]>0))%3
        model = Choice().fit(x, y, np.ones(120)/120, 3, 1., 1000, 1e-6, inc)
        p = model.predict(x, inc)
        np.testing.assert_allclose(p.sum(1), 1.)
        np.testing.assert_array_equal(p[np.arange(120), inc], 0.)
        self.assertTrue(np.isfinite(p).all())

    def test_one_class_is_not_success(self):
        with self.assertRaises(ValueError):
            Choice().fit(np.ones((20, 2)), np.zeros(20, int), np.ones(20)/20, 2, 1., 100, 1e-6)

    def test_known_signal_and_null(self):
        rng = np.random.default_rng(9)
        train = rng.normal(size=(1000, 2)); test = rng.normal(size=(1000, 2))
        y = (train[:, 0]+rng.normal(size=1000)*.5>0).astype(int)
        yt = (test[:, 0]+rng.normal(size=1000)*.5>0).astype(int)
        full = Choice().fit(train, y, np.ones(1000)/1000, 2, 1., 1000, 1e-6).predict(test)
        noise = Choice().fit(train[:, 1:], y, np.ones(1000)/1000, 2, 1., 1000, 1e-6).predict(test[:, 1:])
        score = lambda p: np.log(p[np.arange(1000), yt]).mean()
        self.assertGreater(score(full)-score(noise), .15)
        self.assertLess(abs(score(noise)+np.log(2)), .03)

    def test_preprocessing_training_only(self):
        rng = np.random.default_rng(19)
        d = pd.DataFrame(dict(decision_id=np.arange(20), value=rng.normal(size=20)))
        d.attrs["num_actions"] = 3
        pixels = rng.normal(size=(20, 12))
        config = dict(C.DEFAULT_CONFIG, pca_components=3, pca_fit_rows=20)
        train, test = np.arange(12), np.arange(12, 20)
        with tempfile.TemporaryDirectory() as tmp:
            design = Design(d, pixels, config, Path(tmp)/"first")
            xt, _, _ = design.prepare(train, test, Spec("test", ("value",)))
            changed = d.copy(); changed.loc[test, "value"] = 1e8; changed.attrs=d.attrs
            other_pixels = pixels.copy(); other_pixels[test] = 1e8
            other = Design(changed, other_pixels, config, Path(tmp)/"second")
            xt2, _, _ = other.prepare(train, test, Spec("test", ("value",)))
            np.testing.assert_allclose(xt, xt2, atol=1e-10)


class UrgencyTests(unittest.TestCase):
    def test_annotation_is_mandatory(self):
        gate = validate_annotations(pd.DataFrame(), [], C.DEFAULT_CONFIG)
        self.assertFalse(gate["passed"])

    def test_tracker_uses_only_past(self):
        blank = np.zeros((3, 84, 84), np.uint8)
        tracker = Tracker(1)
        value, _ = tracker.step(blank, 0.)
        self.assertEqual(value["tracking_valid"], 0)
        self.assertTrue(np.isnan(value["urgency"]))


if __name__ == "__main__":
    unittest.main()
