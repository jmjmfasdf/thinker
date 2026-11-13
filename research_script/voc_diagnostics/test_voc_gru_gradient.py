import copy
import os
from types import SimpleNamespace
import pytest
import torch
from voc_gru_gradient import SelectiveVOCGradient


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tree_rep_encoder=torch.nn.Linear(3,4)
        self.other=torch.nn.Linear(3,4)
        self.mix=torch.nn.Linear(8,8)
        self.policy=torch.nn.Linear(8,2)
        self.voc_head=torch.nn.Linear(8,2)
    def forward(self,x):
        features=self.mix(torch.cat([self.tree_rep_encoder(x),self.other(x)],-1)).tanh()
        return self.policy(features),self.voc_head(features.detach()),features


class Learner:
    def __init__(self,model):
        self.actor_net=model
        self.optimizer=torch.optim.SGD([p for n,p in model.named_parameters()
                                        if not n.startswith('voc_head.')],lr=.01)
        self.head=torch.optim.SGD(model.voc_head.parameters(),lr=.01)
    def compute_losses(self,x):
        policy,q,_=self.actor_net(x)
        return {'total_loss':policy.square().mean(),'_voc_total_loss':(q-1).square().mean()},q
    def _step_actor_optimizer(self,params,T,B):
        self.optimizer.step()
        return SimpleNamespace(optimizer_stepped=True)
    def update(self,x):
        losses,q=self.compute_losses(x)
        self.optimizer.zero_grad();self.head.zero_grad()
        losses['total_loss'].backward()
        losses['_voc_total_loss'].backward()
        self.head.step()
        self._step_actor_optimizer(self.optimizer.param_groups[0]['params'],1,len(x))
        return q.detach()


class AmpLearner(Learner):
    def __init__(self,model):
        super().__init__(model)
        self.actor_scaler=torch.amp.GradScaler('cuda',init_scale=1024)
        self.voc_scaler=torch.amp.GradScaler('cuda',init_scale=16)
    def _step_actor_optimizer(self,params,T,B):
        self.actor_scaler.step(self.optimizer);self.actor_scaler.update()
        return SimpleNamespace(optimizer_stepped=True)
    def update(self,x):
        losses,q=self.compute_losses(x)
        self.optimizer.zero_grad();self.head.zero_grad()
        self.actor_scaler.scale(losses['total_loss']).backward()
        self.voc_scaler.scale(losses['_voc_total_loss']).backward()
        self.actor_scaler.unscale_(self.optimizer)
        self.voc_scaler.unscale_(self.head)
        self.voc_scaler.step(self.head);self.voc_scaler.update()
        self._step_actor_optimizer(self.optimizer.param_groups[0]['params'],1,len(x))
        return q.detach()


@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('amp',[False,True])
def test_update_matches_explicit_selective_autograd(enabled,amp):
    if amp and (not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available()):pytest.skip('CUDA AMP test requires SLURM')
    torch.manual_seed(412)
    device='cuda' if amp else 'cpu'
    initial=Toy().to(device);ref=copy.deepcopy(initial);model=copy.deepcopy(initial)
    x=torch.randn(7,3,device=device)
    policy,q,features=ref(x)
    actor_loss=policy.square().mean();voc=(ref.voc_head(features)-1).square().mean()
    named=list(ref.named_parameters());params=[p for _,p in named]
    ag=torch.autograd.grad(actor_loss,params,allow_unused=True,retain_graph=True)
    vg=torch.autograd.grad(voc,params,allow_unused=True)
    expected={}
    for (name,p),a,v in zip(named,ag,vg):
        grad=torch.zeros_like(p)
        if a is not None:grad+=a
        if v is not None and (name.startswith('voc_head.') or (enabled and name.startswith('tree_rep_encoder.'))):grad+=v
        expected[name]=p.detach()-.01*grad
    learner=(AmpLearner if amp else Learner)(model);adapter=SelectiveVOCGradient(learner,enabled)
    actual_q=learner.update(x)
    torch.testing.assert_close(actual_q,q.detach(),rtol=0,atol=0)
    for name,p in model.named_parameters():
        torch.testing.assert_close(p,expected[name],rtol=1e-6,atol=1e-7)
    assert adapter.last_metrics['voc_gru_candidate_norm']>0
    assert (adapter.last_metrics['voc_gru_applied_norm']>0)==enabled
    assert adapter.last_metrics['gru_update_norm']>0
    adapter.close()
