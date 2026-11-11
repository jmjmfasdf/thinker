"""Submission scope and fail-closed visualization/cache chaining."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import spaceinvaders_visual_cache as campaign


class ChainTests(unittest.TestCase):
    def fixture(self, root, reused):
        cp = root/'checkpoint'
        cp.mkdir()
        (cp/'milestone_5000000.json').write_text('{}')
        task = dict(index=0, subject=6, game=2, leave_session=1, runtime='group',
                    checkpoint=str(cp), milestone_sha256=campaign.digest(cp/'milestone_5000000.json'),
                    capture=str(root/'capture'), cache=str(root/'cache/sub006_game2'),
                    cache_root=str(root/'cache'), reuse_visualization=reused,
                    capture_manifest_sha256='bound-hash', sources=[dict(source_id='same-subject-game2.npz')])
        capture = dict(manifest_sha256='bound-hash', episodes=[dict(metadata=dict(
            source_id='run%d' % i, replay_edge_count=2, store_predicted_xs=False,
            compression_level=9)) for i in range(11)])
        return task, capture

    def exercise(self, reused, fail_visual=False, fail_validation=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task, capture = self.fixture(root, reused)
            calls = []
            def monitored(args, output):
                kind = 'cache' if any('00_cache.py' in p for p in args) else 'visualization'
                calls.append(kind)
                if fail_visual and kind == 'visualization':
                    raise RuntimeError('visualization failed')
            def validate(path):
                calls.append('validation')
                if fail_validation:
                    raise RuntimeError('invalid capture')
                return capture
            module = types.SimpleNamespace(validate_capture_output=validate)
            with patch.dict(sys.modules, visual_behav=module), \
                 patch.object(campaign, 'preflight'), \
                 patch.object(campaign, 'monitor', side_effect=monitored), \
                 patch.object(campaign, 'completed_capture', return_value=capture), \
                 patch.object(campaign, 'verify_cache', side_effect=lambda *a: calls.append('cache_validation')), \
                 patch.object(campaign, 'aggregate'):
                if fail_visual or fail_validation:
                    with self.assertRaises(RuntimeError):
                        campaign.run(root, task)
                else:
                    campaign.run(root, task)
            return calls

    def test_reused_capture_never_replayed(self):
        self.assertEqual(self.exercise(True), ['validation', 'cache', 'cache_validation'])

    def test_new_capture_chain_order(self):
        self.assertEqual(self.exercise(False), ['visualization', 'validation', 'cache', 'cache_validation'])

    def test_visual_failure_prevents_cache(self):
        self.assertEqual(self.exercise(False, fail_visual=True), ['visualization'])

    def test_capture_validation_failure_prevents_cache(self):
        self.assertEqual(self.exercise(True, fail_validation=True), ['validation'])

    def test_cache_flags_keep_all_sessions_and_extended_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, _ = self.fixture(Path(tmp), True)
            args = campaign.make_cache_command(Path(tmp), task)
            self.assertEqual(args[args.index('--game')+1], '2')
            self.assertEqual(args[args.index('--capture-dir')+1], task['capture'])
            for flag in ('--include-prediction-summaries', '--include-latent-summaries'):
                self.assertIn(flag, args)
            for flag in ('--overwrite', '--sessions', '--max-files', '--capture-index'):
                self.assertNotIn(flag, args)

    def test_prepare_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                campaign.prepare(Path(tmp))


if __name__ == '__main__':
    unittest.main()
