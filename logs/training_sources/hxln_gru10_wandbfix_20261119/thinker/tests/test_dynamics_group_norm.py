import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import numpy as np
from gymnasium import spaces

from thinker.model_net import DynamicModel, ModelNet, use_dynamics_group_norm
from thinker import util


def network():
    torch.manual_seed(17)
    net=DynamicModel(dim_rep_actions=6,inplanes=16,oned_input=False,
                     size_nn=1,disable_bn=False)
    use_dynamics_group_norm(net)
    return net


def test_group_norm_is_independent_of_mode_and_other_tree_rows():
    # Compare at high precision to separate cross-sample dependence from
    # convolution kernel roundoff when the batch shape changes.
    net=network().double()
    h=torch.randn(3,16,6,6,dtype=torch.float64)
    actions=torch.eye(6,dtype=torch.float64)[:3]
    net.train()
    train=net(h,actions)
    net.eval()
    inference=net(h,actions)
    single=net(h[:1],actions[:1])
    torch.testing.assert_close(train,inference)
    torch.testing.assert_close(inference[:1],single,atol=1e-10,rtol=1e-10)
    other_rows_changed=h.clone()
    other_rows_changed[1:]*=1e6
    torch.testing.assert_close(inference[:1],net(other_rows_changed,actions)[:1],
                               atol=1e-10,rtol=1e-10)
    assert any(isinstance(m,nn.GroupNorm) for m in net.modules())
    assert not any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in net.modules())


def test_conversion_preserves_affine_parameters_and_rng():
    torch.manual_seed(17)
    net=DynamicModel(dim_rep_actions=6,inplanes=16,oned_input=False,
                     size_nn=1,disable_bn=False)
    parameters={k:v.detach().clone() for k,v in net.named_parameters()}
    rng=torch.get_rng_state().clone()
    use_dynamics_group_norm(net)
    torch.testing.assert_close(torch.get_rng_state(),rng)
    for name,value in net.named_parameters():torch.testing.assert_close(value,parameters[name])


def test_thirty_step_dynamics_gradients_and_checkpoint_roundtrip():
    net=network()
    h=torch.randn(2,16,6,6)
    action=torch.eye(6)[:2]
    for _ in range(30):h=net(h,action)
    h.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    restored=network()
    restored.load_state_dict(copy.deepcopy(net.state_dict()))
    torch.testing.assert_close(restored(h.detach(),action),net(h.detach(),action))


def test_norm_option_is_explicit_and_rejects_unknown_values():
    parser=util.add_parse(['default_thinker.yaml','default_actor.yaml'])
    assert parser.parse_args([]).model_sr_dynamics_norm is None
    assert parser.parse_args(['--model_sr_dynamics_norm','group']).model_sr_dynamics_norm=='group'
    with pytest.raises(SystemExit):parser.parse_args(['--model_sr_dynamics_norm','invalid'])


def test_flag_roundtrip_selects_only_sr_dynamics_and_keeps_legacy_keyset():
    legacy=util.create_flags(['default_thinker.yaml','default_actor.yaml'],
                             save_flags=False,model_disable_bn=False)
    flags=util.create_flags(['default_thinker.yaml','default_actor.yaml'],
                            save_flags=False,model_disable_bn=False,
                            model_sr_dynamics_norm='group')
    assert not hasattr(legacy,'model_sr_dynamics_norm')
    assert flags.model_sr_dynamics_norm=='group'
    model=ModelNet(spaces.Box(0,255,(12,84,84),dtype=np.uint8),
                   spaces.Discrete(6),flags,frame_stack_n=4)
    assert any(isinstance(m,nn.GroupNorm) for m in model.sr_net.RNN.modules())
    assert not any(isinstance(m,nn.BatchNorm2d) for m in model.sr_net.RNN.modules())
    assert any(isinstance(m,nn.BatchNorm2d) for m in model.sr_net.encoder.modules())
