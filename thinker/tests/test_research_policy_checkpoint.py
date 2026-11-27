"""06 can read an owning capture checkpoint in a fresh, cache-reusing process."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'research_script'))
from analyses import policy_continuity
from common import native_replay, voc_dynamics


@pytest.mark.parametrize('invalid', [False, True])
def test_policy_head_uses_owning_frozen_buffer_before_unpickling(tmp_path, monkeypatch, invalid):
    campaign = tmp_path / 'capture_campaign'
    folder = campaign / 'checkpoints/run/milestones/10000000'
    folder.mkdir(parents=True)
    actor = folder / 'ckp_actor.tar'
    config = folder / 'config.yaml'
    config.write_text('critic_enc_type: 0\nlast_layer_n: 0\nsep_actor_critic: '+str(invalid).lower()+'\n')
    metadata = {'checkpoint': {'logical_xpid': 'run'}}
    calls = []
    def checkpoint(received):
        assert received is metadata
        calls.append('resolve')
        return folder, actor, config
    def frozen(root, module):
        assert root == campaign and module == 'buffer'
        calls.append('buffer')
    def head(audit):
        assert calls == ['resolve', 'buffer']
        assert audit['checkpoint'] == str(folder) and audit['num_actions'] == 6
        calls.append('weights')
        return audit
    monkeypatch.setattr(voc_dynamics, 'checkpoint_for_capture', checkpoint)
    monkeypatch.setattr(native_replay, 'frozen_import', frozen)
    monkeypatch.setattr(native_replay, 'ActorHeads', head)
    if invalid:
        with pytest.raises(ValueError, match='linear actor head'):
            policy_continuity._captured_actor_head(metadata, 6)
        assert calls == ['resolve']
    else:
        result = policy_continuity._captured_actor_head(metadata, 6)
        assert result['num_actions'] == 6
        assert calls == ['resolve', 'buffer', 'weights']


@pytest.mark.parametrize('number,failed', [('05', False), ('07', False), ('07', True)])
def test_revision_behavior_reports_fit_support(monkeypatch, number, failed):
    import pandas as pd
    from analyses import revision_behavior
    def result(*args):
        return {'prediction_fit_qa': pd.DataFrame({'status': ['nonconverged' if failed else 'ok']})}, {}, {}
    monkeypatch.setattr(revision_behavior, 'revision', result)
    monkeypatch.setattr(revision_behavior, 'behavior', result)
    tables,qa,_ = revision_behavior.analyze(number, None, 1, 'pong', {'decisions': pd.DataFrame()}, None)
    assert qa['status'] == ('PARTIAL_FITS' if failed else 'COMPLETED')
    assert qa['failed_readout_fits'] == int(failed)
    assert tables['prediction_fit_qa'].subject.eq(1).all()
