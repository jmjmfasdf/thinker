"""Native-cache additions: temporal/validity contracts, no output-folder writes."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import dynamic_analysis_cache as adapter
from dynamic_native_features import (
    CARRY, FLAGS, TREE_COMPONENTS, TREE_METRICS, NativeFeatures,
    pack_paths, prediction_summaries, read_optional, xs_display,
)
from shared_analysis_cache import cache_paths


class FakeNative:
    def __init__(self, fields):
        self.fields = fields
        self.event_count = len(fields['decision_time'])
        self.num_actions = 2
        self.path = Path('/synthetic/episode.zarr')
        self.validity = {}
        self.reads = []

    def has_field(self, name):
        return name in self.fields

    def field(self, name):
        return self.fields[name]

    def explicit_valid(self, name):
        return self.validity.get(name, np.ones(self.event_count, bool))

    def read_field_rows(self, name, rows):
        self.reads.append((name, len(rows)))
        return self.fields[name][rows]


def fixture(budgets=(0, 1, 2, 29, 30), forced=(), build=True):
    anchors = [0]
    for i, budget in enumerate(budgets):
        anchors.append(anchors[-1]+budget+1+(i not in forced))
    anchors = np.array(anchors)
    n = anchors[-1]+1
    fields = {name: np.zeros(n) for name in FLAGS+CARRY}
    for name in TREE_METRICS:
        for suffix in ('pre', 'post', 'delta'):
            fields[name+'_'+suffix] = np.ones(n)
    fields.update(voc_online_q=np.column_stack((np.arange(n)*.01+.1, np.zeros(n))),
        voc_ema_q=np.tile([.3, .1], (n, 1)),
        voc_soft_continue_probability=np.full(n, .6),
        voc_state_value=np.full(n, .2), voc_effective_think_cost=np.full(n, .01),
        actor_control_logits=np.tile([.1, -.2, .3], (n, 1)),
        actor_action_probability=np.tile([.8, .2], (n, 1)),
        reward=np.zeros((n, 1)))
    fields['initial_root'][0] = 1
    fields['current_node_has_real_state_post'][anchors] = 1
    fields['root_carried'][0] = 1
    fields['useful_carry'][0] = 1
    fields['decision_time'][0] = 1/15
    for i, (left, right) in enumerate(zip(anchors[:-1], anchors[1:])):
        comp = np.arange(left+1, left+1+budgets[i])
        fields['decision_time'][left+1:right+1] = (i+1)/15
        fields['search_computation'][comp] = 1
        fields['search_expansion'][comp] = 1
        fields['control_valid'][left+1:right] = 1
        fields['accepted_primary_action'][comp] = np.arange(len(comp)) % 2
        fields['current_node_depth_pre'][comp] = np.arange(len(comp))
        fields['current_node_depth_post'][comp] = np.arange(1, len(comp)+1)
        fields['real_transition'][right] = 1
        fields['phase_pre'][right] = 1
        fields['human_action'][right] = i % 2
        fields['reward'][right] = i+1
        fields['search_steps_post'][comp] = np.arange(1, len(comp)+1)
        fields['search_steps_pre'][comp] = np.arange(len(comp))
        if i in forced:
            fields['forced_stop'][right-1] = 1
        else:
            fields['stop'][right-1] = 1
            fields['accepted_control'][right-1] = 2
            fields['search_steps_pre'][right-1] = budgets[i]
            fields['search_steps_post'][right-1] = budgets[i]
    tree = {}
    for name in TREE_COMPONENTS:
        width = 2 if name in ('root_qs_max','root_qs_mean','root_ns','root_policy',
                              'cur_policy','cur_qs_mean','cur_qs_max','cur_ns') else 1
        tree[name] = np.tile(np.arange(n)[:, None]*.1, (1, width))
    native = FakeNative(fields)
    features = NativeFeatures(native, anchors, tree, tree, 'run', 'episode', 0, 0)
    if build:
        fill(features)
    return features


def fill(features):
    fields = features.native.fields
    for i, (left, right) in enumerate(zip(features.anchors[:-1], features.anchors[1:])):
        computations = np.flatnonzero(fields['search_computation'][left+1:right])+left+1
        paths, depths = adapter.reconstruct_paths(computations, fields['accepted_primary_action'],
            fields['current_node_depth_pre'], fields['current_node_depth_post'], fields['actor_reset'])
        features.decision(i, computations, paths, depths)


def add_images(features):
    native = features.native
    n = native.event_count
    native.fields['real_image'] = np.zeros((n, 3, 4, 4), dtype=np.uint8)
    native.fields['imagined_image'] = np.zeros((n, 3, 4, 4), dtype=np.uint8)
    native.fields['predicted_xs'] = np.zeros((n, 12, 4, 4), dtype=np.float32)
    native.fields['im_vectors'] = np.ones((n, 1, 2, 2, 2), dtype=np.float32)
    for i, anchor in enumerate(features.anchors):
        native.fields['real_image'][anchor] = i*20
        native.fields['im_vectors'][anchor] = i+1
    for forecast in features.forecasts:
        event, target = forecast['native_event'], forecast['state_target_native_event']
        if target < 0:
            continue
        native.fields['imagined_image'][event] = native.fields['real_image'][target]
        native.fields['predicted_xs'][event, -3:] = native.fields['real_image'][target]/255.
        native.fields['im_vectors'][event] = native.fields['im_vectors'][target]


class NativeFeatureTests(unittest.TestCase):
    def test_all_budgets_include_stop_and_zero_budget_values(self):
        f = fixture()
        for budget, row in zip((0, 1, 2, 29, 30), f.records):
            self.assertEqual(row['additional_computations'], budget)
            self.assertEqual(row['control_event_count'], budget+1)
            self.assertEqual(row['stop'], 1)
            self.assertEqual(row['stop_choice_censored'], 0)
            self.assertTrue(np.isfinite(row['voc_online_voc_initial']))
            self.assertTrue(np.isfinite(row['voc_online_q_continue_at_stop']))
            self.assertEqual(row['voc_online_voc_valid_events'], budget+1)
            if budget:
                self.assertAlmostEqual(row['voc_online_voc_slope'], .01)
            else:
                self.assertTrue(np.isnan(row['voc_online_voc_slope']))
        self.assertEqual(f.event_table().is_control_choice.sum(), 67)
        self.assertEqual(len(f.path_rows), 62)

    def test_forced_stop_is_censored_not_voluntary(self):
        row = fixture((2,), forced=(0,)).records[0]
        self.assertEqual(row['control_event_count'], 2)
        self.assertEqual(row['stop'], 0)
        self.assertEqual(row['forced_stop'], 1)
        self.assertEqual(row['stop_choice_censored'], 1)
        self.assertTrue(np.isnan(row['voc_online_q_continue_at_stop']))

    def test_carry_tree_bounds_and_actor_probability(self):
        f = fixture((2,))
        row = f.records[0]
        self.assertEqual(row['root_carried'], 1)
        self.assertEqual(f.scalar('root_carried')[f.anchors[-1]], 0)
        self.assertEqual(row['root_r_initial'], f.pre['root_r'][1, 0])
        self.assertEqual(row['root_r_terminal'], f.post['root_r'][3, 0])
        self.assertNotEqual(row['root_r_terminal'], f.post['root_r'][4, 0])
        self.assertEqual(row['actor_action_probability_0'], .8)
        self.assertEqual(row['actor_action_probability_1'], .2)

    def test_missing_invalid_and_true_zero_are_distinct(self):
        f = fixture((0,))
        native = f.native
        native.fields['voc_state_value'][:] = 0
        valid = np.array([True, False, True])
        valid.flags.writeable = False
        native.validity['voc_state_value'] = valid
        value, ok = read_optional(native, 'voc_state_value')
        self.assertEqual(value[0, 0], 0)
        self.assertTrue(ok[0])
        self.assertTrue(np.isnan(value[1, 0]))
        self.assertFalse(ok[1])
        value, ok = read_optional(native, 'absent')
        self.assertTrue(np.isnan(value).all())
        self.assertFalse(ok.any())
        native.fields['actor_control_logits'][0, 1] = -np.inf
        self.assertTrue(read_optional(native, 'actor_control_logits', 3)[1][0])

    def test_reset_ragged_paths(self):
        f = fixture((3,), build=False)
        fields = f.native.fields
        fields['current_node_depth_pre'][[1,2,3]] = [0,1,0]
        fields['current_node_depth_post'][[1,2,3]] = [1,0,1]
        fields['actor_reset'][2] = 1
        fields['accepted_control'][2] = 1
        # Scalar arrays were copied into the helper; refresh after synthetic edits.
        f = NativeFeatures(f.native, f.anchors, f.pre, f.post, 'run', 'episode', 0, 0)
        fill(f)
        packed = pack_paths(f.path_rows)
        self.assertEqual(packed['path_lengths'].tolist(), [1,0,1])
        self.assertEqual(packed['path_offsets'].tolist(), [0,1,1,2])
        self.assertEqual(f.records[0]['reset_count'], 1)
        self.assertEqual(f.forecasts[1]['state_target_valid'], 0)
        with TemporaryDirectory() as temp:
            path = Path(temp)/'paths.npz'
            np.savez_compressed(path, **packed)
            with np.load(path, allow_pickle=False) as result:
                self.assertEqual(result['path_actions'].tolist(), [0,0])

    def test_action_state_alignment_and_episode_boundary(self):
        f = fixture((2,0,1))
        first, second, final = f.forecasts
        self.assertEqual(first['action_target_local'], 0)
        self.assertEqual(first['state_target_local'], 1)
        self.assertEqual(second['action_target_local'], 1)
        self.assertEqual(second['state_target_local'], 2)
        self.assertEqual(second['prefix_full_match'], 1)
        self.assertGreater(second['outcome_available_time'], second['origin_time'])
        self.assertEqual(final['state_target_valid'], 1)
        self.assertTrue(np.isnan(final['outcome_available_time']))
        beyond = f.forecast(2, final['native_event'], 1, [0,1])
        self.assertEqual(beyond['action_target_valid'], 0)
        self.assertEqual(beyond['state_target_valid'], 0)
        events = f.event_table()
        self.assertAlmostEqual(events.loc[f.anchors[1], 'available_time'], 2/15)
        self.assertTrue(np.isnan(events.loc[f.anchors[-1], 'available_time']))

    def test_default_does_not_read_heavy_fields(self):
        f = fixture((2,0,0))
        add_images(f)
        forecasts, _ = prediction_summaries(f)
        self.assertFalse(f.native.reads)
        self.assertEqual(len(forecasts), 2)
        self.assertNotIn('im_vectors_mse', forecasts)
        self.assertEqual(forecasts.observed_reward.tolist(), [1,2])

    def test_image_latent_distances_and_bounded_reads(self):
        f = fixture((2,0,0))
        add_images(f)
        f.forecasts[1]['prefix_full_match'] = 0
        f.forecasts[1]['state_comparison_primary'] = 0
        forecasts, summaries = prediction_summaries(f, True, True, batch_size=1)
        self.assertEqual(len(forecasts), 2)  # divergent prefix is retained
        for name in ('imagined_image','predicted_xs','im_vectors'):
            self.assertTrue(forecasts[name+'_comparison_valid'].eq(1).all())
            np.testing.assert_allclose(forecasts[name+'_mse'], 0, atol=1e-10)
            self.assertTrue(forecasts[name+'_baseline_mse'].gt(0).all())
        self.assertIn('im_vectors_mean_1', summaries)
        self.assertIn('real_image_descriptor_23', summaries)
        self.assertLessEqual(max(count for _, count in f.native.reads), 1)

    def test_invalid_predictions_and_carried_latent_not_ground_truth(self):
        f = fixture((2,0,0))
        add_images(f)
        f.native.validity['imagined_image'] = np.zeros(f.native.event_count, bool)
        f.values['current_node_has_real_state_post'][f.anchors[1], 0] = 0
        result, _ = prediction_summaries(f, True, True)
        self.assertTrue(result.imagined_image_mse.isna().all())
        self.assertEqual(result.loc[0, 'im_vectors_comparison_valid'], 0)
        self.assertTrue(np.isnan(result.loc[0, 'im_vectors_mse']))
        self.assertEqual(result.loc[1, 'im_vectors_comparison_valid'], 1)

    def test_stacked_rgb_uses_latest_frame(self):
        value = np.zeros((1,12,4,4), np.float32)
        value[:, :9] = 50
        value[:, -3:] = .2
        np.testing.assert_allclose(xs_display(value), 51)

    def test_compact_capture_missing_xs_retains_image_and_latent_summaries(self):
        f = fixture((2,0,0))
        add_images(f)
        del f.native.fields['predicted_xs']
        result, summaries = prediction_summaries(f, True, True, batch_size=1)
        self.assertTrue(result.predicted_xs_mse.isna().all())
        self.assertTrue(result.predicted_xs_source_valid.eq(0).all())
        self.assertTrue(summaries.predicted_xs_valid.eq(0).all())
        self.assertTrue(result.imagined_image_comparison_valid.eq(1).all())
        self.assertTrue(result.im_vectors_comparison_valid.eq(1).all())
        np.testing.assert_allclose(result.imagined_image_mse, 0)
        np.testing.assert_allclose(result.im_vectors_mse, 0)

    def test_empty_search_headers_and_streaming(self):
        f = fixture((0,0))
        add_images(f)
        empty, _ = prediction_summaries(f, True, True)
        other = fixture((2,0,0))
        add_images(other)
        nonempty, _ = prediction_summaries(other, True, True)
        self.assertEqual(list(empty), list(nonempty))
        packed = pack_paths([])
        self.assertEqual(packed['path_offsets'].tolist(), [0])
        self.assertEqual(len(packed['path_actions']), 0)
        with TemporaryDirectory() as temp:
            path = Path(temp)/'out.csv.gz'
            empty.to_csv(path, index=False, compression='gzip')
            nonempty.to_csv(path, index=False, header=False, mode='a', compression='gzip')
            self.assertEqual(len(pd.read_csv(path)), 2)

    def test_old_cache_refused_without_touching_contents(self):
        with TemporaryDirectory() as temp:
            args = SimpleNamespace(cache_root=Path(temp), subject=1, game=0, overwrite=False)
            paths = cache_paths(args.cache_root, 1, 0)
            paths.root.mkdir()
            original = json.dumps(dict(adapter_version=2))
            paths.manifest.write_text(original)
            with patch.object(adapter, 'capture_inputs', return_value=([],{})):
                with self.assertRaisesRegex(ValueError, 'Existing cache'):
                    adapter.build_capture_cache(args)
            self.assertEqual(paths.manifest.read_text(), original)

    def test_adapter_all_zero_budget_source_and_reuse_contract(self):
        f = fixture((0,0))
        add_images(f)
        native, fields = f.native, f.native.fields
        native.tree_rep_schema = {}
        offset = 0
        for key, values in f.pre.items():
            native.tree_rep_schema[key] = [offset, offset+values.shape[1]]
            offset += values.shape[1]
        fields['tree_reps_pre'] = np.concatenate(list(f.pre.values()), axis=1)
        fields['tree_reps_post'] = np.concatenate(list(f.post.values()), axis=1)
        fields['cur_rewards'] = fields['reward'].copy()
        fields['thinker_action'] = fields['human_action'].copy()
        fields['actor_policy'] = np.zeros((native.event_count, 2))
        fields['source_frame_index'][f.anchors] = np.arange(len(f.anchors))
        fields['pre_observation_source_index'][f.anchors[1:]] = np.arange(len(f.times))
        fields['real_image_valid'] = np.ones(native.event_count, bool)
        class Projection(dict):
            pass
        projection = Projection(tree_reps=fields['tree_reps_post'], status=np.zeros(len(f.anchors)))
        projection.native = native
        projection.source_event_index = f.anchors
        episode = dict(capture_dir='/synthetic', path='episode.zarr', episode_id='episode',
            metadata=dict(subject=1, game=2, session=1, block=1, source_id='run',
                          source_episode_index=0, replay_edge_count=2))
        with TemporaryDirectory() as temp:
            args = SimpleNamespace(cache_root=Path(temp), subject=1, game=2, overwrite=False)
            with patch.object(adapter, 'capture_inputs', return_value=([episode],dict(captures=['synthetic']))), \
                 patch.object(adapter, 'trace_module', return_value=SimpleNamespace(open_video_stat=lambda *a,**kw:projection)):
                adapter.build_capture_cache(args)
                paths = cache_paths(temp, 1, 2)
                self.assertEqual(len(pd.read_csv(paths.native_decisions)), 2)
                self.assertEqual(len(pd.read_csv(paths.persistence_events)), 0)
                self.assertEqual(len(pd.read_csv(paths.persistence_bouts)), 0)
                with np.load(paths.rollout_prefix_events, allow_pickle=False) as data:
                    self.assertEqual(len(data['window_index']), 0)
                with patch.object(adapter, 'trace_module', side_effect=AssertionError('re-extraction')):
                    adapter.build_capture_cache(args)
                args.include_prediction_summaries = True
                with self.assertRaisesRegex(ValueError, 'Existing cache'):
                    adapter.build_capture_cache(args)
                args.include_prediction_summaries = False
                manifest = json.loads(paths.manifest.read_text())
                manifest['native_feature_sources']['dynamic_native_features.py'] = 'stale'
                paths.manifest.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'Existing cache'):
                    adapter.build_capture_cache(args)
                args.overwrite = True
                args.include_prediction_summaries = True
                args.include_latent_summaries = True
                adapter.build_capture_cache(args)
                self.assertTrue(paths.representation_summaries.is_file())
                self.assertEqual(len(pd.read_csv(paths.forecast_outcomes)), 0)
                self.assertEqual(len(list(Path(temp).glob('sub001_game2.backup_*'))), 1)


if __name__ == '__main__':
    unittest.main()
