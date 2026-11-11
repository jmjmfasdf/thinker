"""Synthetic full-cohort integration of read-only supplemental reporting."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import common as C


class SupplementaryTests(unittest.TestCase):
    def test_full_cohort_selection_correction_and_preservation(self):
        spec = importlib.util.spec_from_file_location("supp", Path(__file__).with_name("09_supplementary_checks.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            effects = []
            for game in (0, 1):
                for subject in range(1, 7):
                    tag = f"sub{subject:03d}_game{game}"
                    runs = [f"sub{subject:03d}_ses{session:02d}_block1_game{game}" for session in (1, 2)]
                    revision = []
                    for name, value in (("incumbent_change_given_alternative", .001*subject),
                                        ("alternative_change_given_incumbent", .003*subject)):
                        effects.append(dict(subject=subject, game=game, family="03", primary="", contrast=name, delta_ll=value))
                        revision.extend(dict(run=run, contrast=name, delta_ll=value) for run in runs)
                    effects.append(dict(subject=subject, game=game, family="03", primary="H3", contrast="primary_history", delta_ll=.01))
                    effects.append(dict(subject=subject, game=game, family="05", primary="", contrast="existing_continuity", delta_ll=.002*subject))
                    root = out/C.MODULES["03"]/tag
                    root.mkdir(parents=True)
                    pd.DataFrame(revision).to_csv(root/"run_effects.csv", index=False)
                    C.finish(root, {"signature": tag+"03"})
                    root = out/C.MODULES["05"]/tag
                    root.mkdir(parents=True)
                    horizons = []
                    for low, high in C.HORIZONS:
                        for run in runs:
                            for mode in ("all_paths", "prefix_matched"):
                                if low==16 and mode=="prefix_matched" and "ses02" in run:
                                    continue
                                horizons.append(dict(run=run, horizon=f"{low}-{high}", mode=mode, n=12, targets=8,
                                    action_match=1 if mode=="prefix_matched" else .4+.001*subject,
                                    persistence_match=.5, root_match=.3, pixel_error=.02,
                                    persistence_pixel_error=.04+.001*subject))
                    pd.DataFrame(horizons).to_csv(root/"horizon_alignment.csv", index=False)
                    C.finish(root, {"signature": tag+"05"})
                    cache = out/"00_native_cache"/tag
                    cache.mkdir(parents=True)
                    pd.DataFrame(dict(decision_id=[0, 1, 2], root_entropy=[.1, .2, .3],
                        begin=[True, True, False], budget=[1, 2, 0], scored=[True, True, False])).to_csv(cache/"decisions.csv.gz", index=False)
                    pd.DataFrame(dict(decision_id=[0, 1, 2], event_index=[1, 1, 1],
                        prior_evidence_abs=[.1, .2, .3], stop_choice=[True, False, True],
                        forced_terminal=[False, True, False], control=[2, 0, 2])).to_csv(cache/"search_events.csv.gz", index=False)
                    C.finish(cache, {"signature": tag+"cache"})
            group = out/"08_group_report"
            group.mkdir()
            pd.DataFrame(effects).to_csv(group/"subject_effects.csv", index=False)
            C.finish(group, {"signature": "synthetic_group"})
            before = {str(p): C.sha(p) for p in out.rglob("*") if p.is_file()}
            with patch("sys.argv", ["09_supplementary_checks.py", "--output-root", str(out)]):
                module.main()
                module.main()  # Compatible completed report is reusable and verified.
            self.assertEqual(before, {p: C.sha(Path(p)) for p in before})
            report = out/"09_supplementary_checks"
            module.group_module().verify_artifacts(report)
            added = pd.read_csv(report/"additional_subject_effects.csv")
            agreement = added[added.contrast.str.startswith("path_vs_persistence")]
            self.assertTrue((agreement.delta_ll < 0).all())  # Never select full-prefix=1.
            last = added[added.contrast.eq("state_vs_persistence_h16-30")]
            self.assertTrue(last.delta_ll.isna().all())  # One-session availability is not a full effect.
            direct = added[added.contrast.eq("incumbent_versus_alternative_direct")]
            np.testing.assert_allclose(direct.delta_ll, -.002*direct.subject)
            inference = pd.read_csv(report/"enlarged_secondary_family_max_t.csv")
            self.assertFalse(inference.contrast.eq("primary_history").any())
            self.assertTrue(inference[inference.contrast.eq("state_vs_persistence_h16-30")].p_max_t.isna().all())
            self.assertTrue(inference[inference.contrast.eq("incumbent_versus_alternative_direct")].decision.eq("차이 지지됨").all())
            curves = pd.read_csv(report/"computation_probability_curves.csv")
            self.assertTrue((curves.groupby(["subject", "game", "kind"])["count"].sum()==2).all())
            self.assertEqual(len(list((report/"figures").glob("*.png"))), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
