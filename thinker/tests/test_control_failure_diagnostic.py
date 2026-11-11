import logging
from types import SimpleNamespace

import pytest
import torch
from thinker import util
from thinker.self_play import SelfPlayWorker


def test_failed_control_keeps_predecision_mask_and_propagates_assertion(tmp_path):
    def fail(**kwargs):
        raise AssertionError('PROCEED is illegal at max_depth')

    env_out = util.EnvOut(**{
        key: (torch.tensor([[[False, True, True]]]) if key == 'legal_control_mask'
              else torch.tensor([[0]]) if key == 'phase' else None)
        for key in util.EnvOut._fields
    })
    actor_out = SimpleNamespace(action=(torch.tensor([0]), torch.tensor([0])),
                                action_prob=torch.tensor([[[1., 0.]]]),
                                search_control_logits=torch.tensor([[[-1e9, -1e10, 0.]]]))
    worker = SimpleNamespace(actor_net=lambda **kw: (actor_out, ()),
                             disable_thinker=False, dynamic_search=True,
                             env=SimpleNamespace(step=fail),
                             flags=SimpleNamespace(ckpdir=str(tmp_path)), rank=0,
                             _logger=logging.getLogger(__name__))
    with pytest.raises(AssertionError, match='PROCEED is illegal'):
        SelfPlayWorker.__ray_metadata__.modified_class.env_step(worker, env_out, ())
    saved = torch.load(tmp_path / 'control_failure_rank0.pt', weights_only=True)
    assert saved['env_out']['legal_control_mask'].tolist() == [[[False, True, True]]]
    assert saved['search_control'].tolist() == [0]
