import json
import numpy as np
import pytest
from scipy.special import expit
from expanded_voc_sampling import sampling_shards, read_arm
from voc_allocation_regression import fractional_logit


def test_sampling_adds_scenes_not_only_replicates():
    pairs=[]
    for s in sampling_shards():
        for stream in range(s['batch_size']):
            for prefix in s['prefixes']:
                for count in s['counts']:
                    for rep in range(s['repetitions']):pairs.append((s['seed'],stream,prefix,count,rep))
    assert len(pairs)==len(set(pairs))==512
    assert len({p[:3] for p in pairs})==128
    assert len({p[:4] for p in pairs})==256
    assert len({p[:2] for p in pairs})==32


def test_resume_rejects_changed_protocol(tmp_path):
    p=tmp_path/'arm.json';p.write_text(json.dumps(dict(protocol_sha256='old',result={})))
    with pytest.raises(ValueError):read_arm(p,'new')
    assert read_arm(p,'old')=={}


def test_continuous_regression_recovers_known_probability_curve():
    x=np.linspace(-2,3,51);y=expit(-.4+.8*x)
    b,ok=fractional_logit(x,y)
    assert ok
    np.testing.assert_allclose(b,[-.4,.8],atol=1e-6)


def test_constant_gate_has_no_utility_slope():
    b,ok=fractional_logit(np.linspace(-2,3,40),np.full(40,.01))
    assert ok and abs(b[1])<1e-8
