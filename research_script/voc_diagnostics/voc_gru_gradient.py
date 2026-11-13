"""Selective VOC -> tree-GRU gradient for the matched continuation experiment.

The production detached VOC forward and separate AMP/optimizer paths remain
intact. A leaf at the VOC input measures dL/dfeatures. A VJP through the live
feature graph selects only GRU parameters. Add that unscaled gradient after
actor AMP unscale, before the ordinary actor clipping and optimizer step.
This is mathematically selective un-detaching, without sending gradients to
the other encoders, phase embedding, targets, or EMA teacher.
"""
import torch


def selected_feature_vjp(loss, records, parameters):
    if loss is None:
        return tuple(None for _ in parameters)
    leaves=[leaf for _,leaf in records]
    if not leaves:
        raise AssertionError('VOC loss has no captured feature leaf')
    upstream=torch.autograd.grad(loss,leaves,allow_unused=True,retain_graph=True)
    used=[(live,g) for (live,_),g in zip(records,upstream) if g is not None]
    if not used:
        raise AssertionError('VOC loss is disconnected from every feature leaf')
    gradients=torch.autograd.grad([x for x,_ in used],parameters,
        grad_outputs=[g for _,g in used],allow_unused=True,retain_graph=True)
    return tuple(None if g is None else g.detach() for g in gradients)


class SelectiveVOCGradient:
    def __init__(self, learner, enabled):
        self.learner=learner
        self.actor=learner.actor_net
        self.enabled=bool(enabled)
        for name in ['tree_rep_encoder','policy','voc_head']:
            if not hasattr(self.actor,name):
                raise ValueError('This diagnostic requires ActorNetSingle with '+name)
        self.parameters=tuple(self.actor.tree_rep_encoder.parameters())
        owned={id(p) for group in learner.optimizer.param_groups for p in group['params']}
        if not all(id(p) in owned for p in self.parameters):
            raise AssertionError('Tree GRU must belong to the actor optimizer')
        self.active=False;self.live=None;self.records=[];self.pending=None
        self.last_metrics={};self.updates=0
        self.hooks=[self.actor.policy.register_forward_pre_hook(self._capture),
                    self.actor.voc_head.register_forward_pre_hook(self._leaf)]
        self.original_compute=learner.compute_losses
        self.original_step=learner._step_actor_optimizer
        learner.compute_losses=self._compute
        learner._step_actor_optimizer=self._step

    def _capture(self,module,args):
        if self.active and torch.is_grad_enabled():
            self.live=args[0]

    def _leaf(self,module,args):
        if not self.active or not torch.is_grad_enabled():
            return None
        if self.live is None or not self.live.requires_grad:
            raise AssertionError('Missing differentiable live actor features')
        if not torch.equal(args[0].detach(),self.live.detach()):
            raise AssertionError('VOC/policy feature identity changed')
        leaf=args[0].detach().requires_grad_(True)
        self.records.append((self.live,leaf))
        return (leaf,)+args[1:]

    def _compute(self,*args,**kwargs):
        if self.pending is not None:
            raise AssertionError('Unconsumed VOC-GRU gradient')
        self.records=[];self.live=None;self.active=True
        try:
            result=self.original_compute(*args,**kwargs)
            self.pending=selected_feature_vjp(result[0].get('_voc_total_loss'),
                                              self.records,self.parameters)
            if any(g is not None and not torch.isfinite(g).all() for g in self.pending):
                raise FloatingPointError('Nonfinite unscaled VOC -> GRU gradient')
            return result
        finally:
            self.active=False;self.records=[];self.live=None

    def _step(self,optimize_params,T,B):
        if self.pending is None:
            raise AssertionError('Actor step without staged VOC-GRU gradient')
        before=[p.detach().clone() for p in self.parameters]
        norm2=0.;actor2=0.;dot=0.
        for p,g in zip(self.parameters,self.pending):
            if p.grad is not None:
                actor2+=float(p.grad.detach().double().square().sum())
            if g is not None:
                norm2+=float(g.double().square().sum())
                if p.grad is not None:
                    dot+=float((p.grad.detach().double()*g.double()).sum())
                if self.enabled:
                    if p.grad is None:p.grad=g.clone()
                    else:p.grad.add_(g)
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError('Nonfinite merged actor/VOC GRU gradient')
        self.last_metrics=dict(voc_gru_candidate_norm=norm2**.5,
            voc_gru_applied_norm=norm2**.5 if self.enabled else 0.,
            actor_gru_gradient_norm=actor2**.5,
            voc_actor_gru_cosine=dot/(norm2*actor2)**.5 if norm2*actor2>0 else 0.)
        # Gradients added above are finite and already unscaled; the unchanged
        # actor GradScaler retains its earlier overflow decision for RL/BC.
        result=self.original_step(optimize_params,T,B)
        self.last_metrics['gru_update_norm']=sum(float((p.detach()-old).double().square().sum())
                                                for p,old in zip(self.parameters,before))**.5
        self.last_metrics['actor_optimizer_stepped']=bool(result.optimizer_stepped)
        self.pending=None;self.updates+=1
        return result

    def close(self):
        self.learner.compute_losses=self.original_compute
        self.learner._step_actor_optimizer=self.original_step
        for hook in self.hooks:hook.remove()
