"""Opt-in, three-checkpoint integration test using short real-image prefixes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('VISUAL_ALLFRAMES_LIVE')!='1',reason='requires local 10M campaign/checkpoints')
REPO=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('game',['enduro','pong','spaceinvaders'])
def test_original_order_capture_real_checkpoint(tmp_path,game):
    campaign=REPO/'logs/visualization/campaigns/visualization_hxln_10m_allframes_20261126'
    manifest=json.loads((campaign/'manifest.json').read_text())
    task=next(t for t in manifest['runs'] if t['subject']==1 and t['game']==game)
    source=task['sources'][0]['source_id']
    raw=tmp_path/'data'/source;raw.parent.mkdir(parents=True)
    with np.load(REPO/'behavioral_data_block'/source,allow_pickle=False) as archive:
        n=len(archive['action'])
        prefix={k:archive[k][:16] if archive[k].ndim and len(archive[k])==n else archive[k] for k in archive.files}
    np.savez_compressed(raw,**prefix)
    output=tmp_path/'capture'
    env=dict(os.environ,PYTHONPATH=str(campaign/'runtime_source/thinker'))
    subprocess.run([sys.executable,'-B',str(campaign/'runtime_source/thinker/visual_behav.py'),'capture',
        '--checkpoint-dir',task['checkpoint_dir'],'--checkpoint-step','10000000',
        '--data-root',str(tmp_path/'data'),'--output',str(output),'--scratch-dir',str(tmp_path/'scratch'),
        '--devices','cpu','--cpu-workers','1','--torch-threads-per-worker','1',
        '--images','all','--omit-predicted-xs','--stable-polls','1','--stable-interval','0'],
        env=env,check=True,timeout=180)
    import visual_behav as visual
    capture=visual.validate_capture_output(output)
    assert capture['semantic']['replay_sampling']=='original-order-v1'
    assert len(capture['episodes'])==1
    from thinker.behavior_trace import open_video_stat
    trace=open_video_stat(output/capture['episodes'][0]['path']).native
    real=np.asarray(trace.field('real_transition')[:],bool)
    np.testing.assert_array_equal(np.asarray(trace.field('pre_observation_source_index')[:])[real],np.arange(15))
    np.testing.assert_array_equal(np.asarray(trace.field('score_mask')[:])[real],[False,False]+[True]*13)
