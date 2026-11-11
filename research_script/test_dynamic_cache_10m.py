"""Cache-only dispatch contracts; no jobs or persistent outputs are created."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import dynamic_cache_10m as campaign


class CacheCampaignTests(unittest.TestCase):
    def test_full_grid(self):
        keys = campaign.expected_keys()
        self.assertEqual(len(keys), 72)
        for game in range(3):
            self.assertEqual(sum(g == game for _, g, _ in keys), 24)

    def test_command_is_cache_only_and_keeps_summaries(self):
        task = dict(subject=2, game=0, capture='/capture', cache_root='/cache/checkpoint_leave01')
        cmd = campaign.make_command(Path('/output'), task)
        self.assertIn('/output/source/research_script/00_cache.py', cmd)
        self.assertIn('--include-prediction-summaries', cmd)
        self.assertIn('--include-latent-summaries', cmd)
        self.assertNotIn('--overwrite', cmd)
        self.assertNotIn('--capture-index', cmd)
        self.assertNotIn('--trace-root', cmd)

    def test_atomic_queue_claims_exactly_once(self):
        with TemporaryDirectory() as d:
            out = Path(d)
            (out / 'reports/claims').mkdir(parents=True)
            tasks = [dict(index=i) for i in range(72)]
            def worker(number):
                claimed = []
                while True:
                    task = campaign.claim_next(out, tasks, number)
                    if task is None:
                        return claimed
                    claimed.append(task['index'])
            with ThreadPoolExecutor(max_workers=24) as pool:
                ids = sum(pool.map(worker, range(24)), [])
            self.assertEqual(sorted(ids), list(range(72)))
            self.assertIsNone(campaign.claim_next(out, tasks, 99))
            owners = list((out / 'reports/claims').glob('*/owner.json'))
            self.assertEqual(len(owners), 72)

    def test_pbs_input_read_only_and_eight_explicit_hosts(self):
        text = Path(campaign.__file__).with_suffix('.pbs').read_text()
        self.assertIn('"$VISUAL:$VISUAL:ro"', text)
        self.assertIn('"$HOST" == "bml2-slave$NODE"', text)
        self.assertIn('^[1-8]$', text)
        self.assertIn('168:00:00', text)

    def test_verifier_checks_scored_column_and_preserves_zero_search(self):
        import numpy as np
        import pandas as pd
        from shared_analysis_cache import cache_paths
        with TemporaryDirectory() as d:
            task = dict(subject=1, game=0, num_actions=9, cache_root=d,
                        expected_real_steps=22, expected_scored_steps=11,
                        expected_episode_count=11, expected_native_events=44)
            p = cache_paths(d, 1, 0)
            p.root.mkdir()
            for name, path in vars(p).items():
                if name not in ('root', 'root_state_dir', 'root_state_index', 'representation_dir'):
                    path.write_text('placeholder\n')
            m = dict(status='COMPLETED', adapter_version=3, subject=1, game=0, num_actions=9,
                     source_count=11, runs=11, real_steps=22, episode_count=11, search_events=0,
                     captures=[dict(manifest_sha256='input')],
                     native_feature_options=dict(images=True, latents=True, batch_size=64))
            campaign.write(p.manifest, m)
            ids = ['real%d' % i for i in range(22)]
            pd.DataFrame(dict(real_step_id=ids, stop=1, forced_stop=0, additional_computations=0)).to_csv(
                p.native_decisions, index=False, compression='gzip')
            windows = pd.DataFrame(dict(real_step_id=ids, scored=[0, 1]*11,
                session=[1+i%4 for i in range(22)], run_label=['run%d' % (i//2) for i in range(22)]))
            windows.to_csv(p.persistence_windows, index=False, compression='gzip')
            campaign.write(p.native_features_qa, dict(episodes=[dict(native_events=44)], paths=0, decisions=22))
            np.savez_compressed(p.imagination_paths, path_lengths=np.array([], dtype=int),
                                path_offsets=np.array([0]), path_actions=np.array([], dtype=int))
            pd.DataFrame(dict(search_computation=np.zeros(44))).to_csv(p.native_events, index=False, compression='gzip')
            pd.DataFrame(columns=['native_event']).to_csv(p.forecast_outcomes, index=False, compression='gzip')
            self.assertEqual(campaign.verify_cache(task, dict(manifest_sha256='input')), m)
            windows.loc[0, 'scored'] = 1
            windows.to_csv(p.persistence_windows, index=False, compression='gzip')
            with self.assertRaisesRegex(ValueError, 'Scored-step'):
                campaign.verify_cache(task, dict(manifest_sha256='input'))


if __name__ == '__main__':
    unittest.main()
