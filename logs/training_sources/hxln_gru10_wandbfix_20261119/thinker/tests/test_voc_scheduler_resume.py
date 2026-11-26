import copy
import pytest
import torch
from thinker import util
from thinker.learn_actor import SActorLearner
from tests.test_dynamic_voc_learner import (
    _exact_projection_learner_flags, _rollout, _NullWriter,
    _select_rollout_actor_stream,
)


def test_checkpoint_after_unroll_without_q_support_restores_schedule(tmp_path,monkeypatch):
    flags=_exact_projection_learner_flags(tmp_path,'voc-no-support-resume')
    (tmp_path/flags.xpid).mkdir()
    actor,rollout,core=_rollout(flags)
    monkeypatch.setattr('thinker.learn_actor.FileWriter',lambda **kw:_NullWriter())
    learner=SActorLearner(None,{},flags,actor,torch.device('cpu'))
    learner.consume_data((rollout,core))
    epoch=learner.voc_scheduler.last_epoch
    updates=learner.voc_update_count
    learner.consume_data(_select_rollout_actor_stream(rollout,core,0))
    assert learner.real_step>epoch
    assert learner.voc_update_count==updates
    learner.save_checkpoint()
    saved=torch.load(learner.ckp_path,map_location='cpu',weights_only=False)
    assert saved['voc_lr_last_successful_real_step']==epoch
    util.validate_voc_checkpoint_components(saved,flags=flags)
    bad=copy.deepcopy(saved);bad['voc_lr_last_successful_real_step']=bad['real_step']+1
    with pytest.raises(ValueError,match='last successful step'):
        util.validate_voc_checkpoint_components(bad,flags=flags)
    resume=copy.deepcopy(flags);resume.ckp=True
    restored=SActorLearner(None,{},resume,copy.deepcopy(actor),torch.device('cpu'))
    assert restored.voc_scheduler.state_dict()==learner.voc_scheduler.state_dict()
    assert restored.real_step==learner.real_step


def test_opt_in_production_gru_connection_commits_gradient(tmp_path,monkeypatch):
    flags=_exact_projection_learner_flags(tmp_path,'voc-gru-native')
    flags.voc_gru_connected=True
    actor,rollout,core=_rollout(flags)
    monkeypatch.setattr('thinker.learn_actor.FileWriter',lambda **kw:_NullWriter())
    learner=SActorLearner(None,{},flags,actor,torch.device('cpu'))
    learner.consume_data((rollout,core))
    learner.consume_data((rollout,core))
    metrics=learner._voc_gru_adapter.last_metrics
    assert metrics['actor_optimizer_stepped']
    assert metrics['voc_gru_applied_norm']>0
    assert metrics['gru_update_norm']>0
