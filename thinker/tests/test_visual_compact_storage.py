"""Compact storage must only omit XS; every other event value is lossless."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import visual_behav as recorder
from test_visual_behav_recorder import _event, _CollectingOutput, _checkpoint_bundle, _capture_config


def test_compact_sink_changes_only_two_storage_fields():
    outputs = []
    for store in (True, False):
        out = _CollectingOutput()
        sink = recorder._CanonicalEventSink(out, images="all", store_predicted_xs=store,
            source_frame_indices=np.array([10,11]), decision_times=np.array([.1,.2]),
            grayscale=False, observation_provider=np.zeros((2,6,2,2), np.uint8))
        for event in (_event(0, initial=True), _event(1, search=True),
                      _event(2, real=True, cursor_after=1)):
            before = event['predicted_xs'].copy()
            sink.write_event(event)
            np.testing.assert_array_equal(event['predicted_xs'], before)
        sink.assert_complete()
        outputs.append(out.events)
    for original, compact in zip(*outputs):
        assert set(original)-set(compact) == {'predicted_xs','predicted_xs_valid'}
        for key in compact:
            np.testing.assert_array_equal(compact[key], original[key], err_msg=key)


def test_level9_roundtrip_and_compact_validation(tmp_path):
    from thinker import behavior_trace as trace
    path = tmp_path/'episode.zarr'
    output = recorder._BoundedWriter(trace, partial_path=tmp_path/'partial', final_path=path,
        metadata=dict(compression_level=9, store_predicted_xs=False), num_actions=3,
        tree_rep_schema={'all':[0,4,None]}, chunk_events=2, images='all', queue_size=1,
        compression_level=9)
    sink = recorder._CanonicalEventSink(output, images='all', store_predicted_xs=False,
        source_frame_indices=np.array([10,11]), decision_times=np.array([.1,.2]),
        grayscale=False, observation_provider=np.zeros((2,6,2,2),np.uint8))
    for event in (_event(0,initial=True), _event(1,real=True,cursor_after=1)):
        event['trace_schema_version'] = np.int16(recorder.TRACE_EVENT_SCHEMA_VERSION)
        sink.write_event(event)
    output.finalize()
    manifest = trace.validate_episode_store(path)
    recorder._validate_native_trace_contract(trace, path, manifest, recorder.TRACE_EVENT_SCHEMA_VERSION)
    native = trace.open_video_stat(path).native
    assert not native.has_field('predicted_xs')
    np.testing.assert_array_equal(native.field('im_vectors'), [[[0,0]],[[1,1]]])
    assert native.has_field('imagined_image') and native.has_field('real_image')
    for pack in manifest['pack_schema']:
        codec = json.loads((path/'packs'/pack['name']/'.zarray').read_text())['compressor']
        assert (codec['cname'],codec['clevel'],codec['shuffle']) == ('zstd',9,2)
    manifest['metadata']['store_predicted_xs'] = True
    with pytest.raises(recorder.VisualBehaviorError, match='lacks'):
        recorder._validate_native_trace_contract(trace,path,manifest,recorder.TRACE_EVENT_SCHEMA_VERSION)


def test_compact_profile_requires_images_and_valid_level(tmp_path):
    root, _ = _checkpoint_bundle(tmp_path)
    bundle = recorder.resolve_checkpoint_bundle(root, stable_polls=1, stable_interval_seconds=0)
    config = _capture_config(tmp_path,bundle)
    with pytest.raises(recorder.VisualBehaviorError, match='requires retaining'):
        replace(config, images='none', store_predicted_xs=False)
    with pytest.raises(recorder.VisualBehaviorError, match='compression_level'):
        replace(config, compression_level=10)
    assert replace(config, images='all', store_predicted_xs=False, compression_level=9).compression_level == 9
