import json

import pytest
import yaml

from tools import visual_heldout_campaign as campaign


@pytest.mark.parametrize('value,expected', [('1', [1]), ('1,2,3', [1, 2, 3]), ([4], [4]), (2, [2])])
def test_integer_ids(value, expected):
    assert campaign.integer_ids(value) == expected


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'REPO', tmp_path)
    for game, name in ((0, 'enduro'), (1, 'pong')):
        for subject in range(1, 7):
            for session in range(1, 5):
                checkpoint = tmp_path / 'logs/thinker' / f'dynamic-imitation-{name}-voc-sub{subject}-leavesession{session}'
                checkpoint.mkdir(parents=True)
                config = dict(icopro_subjects=str(subject), icopro_holdout_sessions=str(session),
                              icopro_train_sessions=','.join(str(s) for s in range(1, 5) if s != session),
                              icopro_game_id=game)
                (checkpoint / 'config_c.yaml').write_text(yaml.safe_dump(config))
                (checkpoint / 'milestone_5000000.json').write_text(json.dumps(dict(
                    logical=dict(subject=subject, game_id=game, leave_session=session))))
                data = tmp_path / 'behavioral_data_block' / f'sub-{subject:03d}/ses-{session:02d}'
                data.mkdir(parents=True, exist_ok=True)
                for block in range(2 if session == 1 else 3):
                    (data / f'sub{subject:03d}-ses{session:02d}-block{block}-game{game}.npz').write_bytes(b'fixture')
    return tmp_path / 'output'


def test_prepare_routes_all_sources_once(prepared_inputs):
    campaign.prepare(prepared_inputs)
    manifest = json.loads((prepared_inputs / 'campaign.json').read_text())
    assert len(manifest['tasks']) == 48
    sources = []
    for task in manifest['tasks']:
        for source in task['sources']:
            source_id = source['source_id']
            assert source_id.startswith(f"sub-{task['subject']:03d}/ses-{task['session']:02d}/")
            assert source_id.endswith(f"-game{task['game']}.npz")
            sources.append(source_id)
    assert len(sources) == len(set(sources)) == 132
    # Identical reruns are allowed, but replacing bound source bytes is not.
    campaign.prepare(prepared_inputs)
    (campaign.REPO / 'behavioral_data_block' / sources[0]).write_bytes(b'changed')
    with pytest.raises(AssertionError, match='different inputs'):
        campaign.prepare(prepared_inputs)


def test_prepare_rejects_heldout_session_in_training(prepared_inputs):
    config_path = campaign.REPO / 'logs/thinker/dynamic-imitation-enduro-voc-sub1-leavesession1/config_c.yaml'
    config = yaml.safe_load(config_path.read_text())
    config['icopro_train_sessions'] = '1,2,3,4'
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(AssertionError):
        campaign.prepare(prepared_inputs)
    assert not (prepared_inputs / 'campaign.json').exists()


def test_subject_index_requires_four_complete_folds(prepared_inputs):
    campaign.prepare(prepared_inputs)
    for index in range(3):
        campaign.write_json(prepared_inputs / 'reports' / f'fold-{index:02d}.completed.json', dict(status='COMPLETED'))
    campaign.status(prepared_inputs)
    assert not list((prepared_inputs / 'subjects').iterdir())
    campaign.write_json(prepared_inputs / 'reports/fold-03.completed.json', dict(status='COMPLETED'))
    campaign.status(prepared_inputs)
    index = json.loads((prepared_inputs / 'subjects/sub001_game0.json').read_text())
    assert index['source_count'] == 11
    assert len(index['checkpoints']) == len(index['capture_roots']) == 4
    assert not (prepared_inputs / 'COMPLETED.json').exists()
