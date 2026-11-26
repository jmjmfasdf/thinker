"""Regressions exposed by real synchronous RL+BC pipeline runs."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from thinker import util
from thinker.buffer import SModelBuffer
from thinker.main import Env
from thinker.learn_actor import SActorLearner
from thinker.dynamic_imitation import DynamicImitationRunner
from tests.test_dynamic_imitation_actor import _batch, _StopThenProposeActor, _FrozenFakeModel


def test_training_batch_preserves_per_sequence_source_indices_and_gradients():
    flags = util.create_setting(args=['--dynamic_search','true','--max_search_steps','2',
        '--max_depth','2','--see_h','false','--see_x','false','--parallel','false','--float16','false'], save_flags=False)
    batch = _batch()
    batch['observation_source_index'] = np.arange(12).reshape(2,6)
    actor = _StopThenProposeActor(5)
    runner = DynamicImitationRunner(actor, _FrozenFakeModel(5), flags, device='cpu')
    source = batch['observation_source_index'].copy()
    result = runner.rollout(batch, training=True)
    assert result.count == 8
    result.loss.backward()
    assert torch.isfinite(actor.real_logits.grad).all()
    np.testing.assert_array_equal(batch['observation_source_index'], source)
    batch['observation_source_index'] = np.zeros((3,6), dtype=np.int64)
    with pytest.raises(ValueError, match='observation_source_index'):
        runner.rollout(batch, training=True)
    runner.close()


def test_synchronous_model_write_counts_only_real_rows():
    buffer = SModelBuffer(buffer_n=32,max_rank=1,batch_size=2,warm_up_n=0)
    env = SimpleNamespace(parallel=False, model_buffer=buffer, rank=0,
        frame_stack_n=1, tuple_action=False, env_n=2)
    info = dict(baseline=torch.zeros(2,1),real_states_np=np.zeros((2,1,2,2),dtype=np.uint8))
    Env._write_send_model_buffer(env, None, torch.tensor([2.,3.]), torch.zeros(2,dtype=torch.bool),
        torch.zeros(2,dtype=torch.bool), info, torch.tensor([1,2]), torch.ones(2,3)/3,
        real_step_mask=torch.tensor([True,False]))
    assert buffer.processed_n == 1
    assert buffer.filled_t.tolist() == [1,0]
    assert buffer.buffer['action'][0,0] == 1


def test_synchronous_model_learning_uses_trajectory_dimensions_and_restores_eval():
    model = torch.nn.Linear(2,2).eval()
    calls = []
    class Buffer:
        def read(self,t,b,beta,add_t):
            calls.append((t,b,beta,add_t))
            return {'processed_n':12} if len(calls)==1 else None
        def get_status(self): return {'finish':False}
    class Learner:
        model_T=5; model_B=4; real_step=12
        def compute_beta(self): return .4
        def init_psteps(self,data): pass
        def update_real_step(self,data): pass
        def step_per_transition(self): return .5
        def consume_data(self,data,model_buffer):
            assert model.training and torch.is_grad_enabled()
        def log_preload(self,status): pass
    env = SimpleNamespace(model_net=model,model_learner=Learner(),model_buffer=Buffer(),
        flags=SimpleNamespace(model_return_n=5,max_replay_ratio=1.,total_steps=100,ckp=False))
    with torch.no_grad():
        Env._train_model(env)
    assert calls == [(5,4,.4,6)]*2
    assert not model.training


def test_synchronous_imitation_reads_live_model_without_aliasing_or_gradients():
    class Model(torch.nn.Linear):
        def set_weights(self,weights): self.load_state_dict(weights)
    source, target = Model(2,2), Model(2,2)
    learner=SimpleNamespace(imitation_model_source=source,bc_model_net=target,
        model_param_buffer=None,flags=SimpleNamespace(ckp=False,preload=''))
    assert SActorLearner._refresh_imitation_model(learner,require_weights=True)
    assert torch.equal(source.weight,target.weight)
    assert not target.weight.requires_grad and source.weight.requires_grad
    with torch.no_grad(): source.weight.add_(1)
    assert not torch.equal(source.weight,target.weight)
    SActorLearner._refresh_imitation_model(learner,require_weights=True)
    assert torch.equal(source.weight,target.weight)


def test_resume_counts_new_warmup_transitions_in_cold_local_buffer():
    from thinker.learn_model import SModelLearner
    learner=SimpleNamespace(start_training=False,last_psteps=0,real_step=100,
        flags=SimpleNamespace(ckp=True),voc_model_input_seal_runtime=False,
        model_T=5,model_B=4)
    learner.compute_beta=lambda:.4
    learner.init_psteps=lambda data:SModelLearner.init_psteps(learner,data)
    learner.update_real_step=lambda data:SModelLearner.update_real_step(learner,data)
    learner.step_per_transition=lambda:2.
    buffer=SimpleNamespace(read=lambda *a,**k:{'processed_n':60},get_status=lambda:{'finish':False})
    env=SimpleNamespace(model_learner=learner,model_buffer=buffer,
        flags=SimpleNamespace(ckp=True,model_return_n=5,max_replay_ratio=1.,total_steps=1000))
    Env._train_model(env)
    assert learner.real_step == 160
    assert learner.last_psteps == 60
