import json
from pathlib import Path

import pytest
import torch
import visual_behav as v
from test_visual_behav_recorder import _checkpoint_bundle, _sha256


def ten_m(tmp_path):
    root, flags = _checkpoint_bundle(tmp_path, step=10_000_000)
    steps = {'actor': 10_000_048, 'model': 10_000_064}
    files = {}
    for role, step in steps.items():
        original = root / ('ckp_%s.tar_step_10000000' % role)
        checkpoint = torch.load(original, weights_only=False)
        checkpoint['real_step'] = step
        path = root / ('ckp_%s.tar_step_%d' % (role, step))
        torch.save(checkpoint, path)
        files[role] = dict(path='/training/host/' + path.name, sha256=_sha256(path))
    config = root / 'config_c.yaml'
    files['config'] = dict(path='/training/host/' + config.name, sha256=_sha256(config))
    data = dict(files=files, nominal_milestone=10_000_000, real_step=steps['actor'],
                actor_real_step=steps['actor'], model_real_step=steps['model'],
                pair_step_difference=16)
    (root / 'bundle.json').write_text(json.dumps(data))
    return root, data


def test_published_10m_pair_allows_bound_asynchronous_steps(tmp_path):
    root, _ = ten_m(tmp_path)
    bundle = v.resolve_checkpoint_bundle(root, requested_step=10_000_000, stable_polls=1)
    result = v.validate_checkpoint_payload_bundle(bundle)
    assert result['actor_real_step'] == 10_000_048
    assert result['model_real_step'] == 10_000_064
    assert bundle.milestone.path == str(root / 'bundle.json')


@pytest.mark.parametrize('mutation', ['hash', 'step', 'difference', 'nominal'])
def test_10m_bundle_fails_closed_on_contradiction(tmp_path, mutation):
    root, data = ten_m(tmp_path)
    if mutation == 'hash': data['files']['actor']['sha256'] = '0' * 64
    elif mutation == 'step': data['actor_real_step'] = 10_100_000
    elif mutation == 'difference': data['pair_step_difference'] = 0
    else: data['nominal_milestone'] = 5_000_000
    (root / 'bundle.json').write_text(json.dumps(data))
    with pytest.raises(v.VisualBehaviorError):
        v.resolve_checkpoint_bundle(root, requested_step=10_000_000, stable_polls=1)
