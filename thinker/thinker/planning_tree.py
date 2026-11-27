"""Differentiable arithmetic for the native Dynamic Thinker state machine.

Topology, legal actions and environment scheduling remain in cenv.  Native
nodes own a TorchNode only for the opt-in ``torch`` backend.  Return expressions
are stored as reward/value terms, not evaluated tensors: changing gamma can
revalue a carried tree without clearing its evidence or retaining old graphs.
"""
from dataclasses import dataclass
import copy
import math

import torch


@dataclass(frozen=True)
class ReturnTerm:
    r: float
    v: float
    raw_value: bool = False


class PlanningTreeContext:
    def __init__(self, gamma, device="cpu", *, reward_clip=-1.0):
        self.device = torch.device(device)
        self.reward_clip = float(reward_clip)
        self.last_token = None
        self.gamma = torch.as_tensor(gamma, device=self.device)
        if not self.gamma.is_floating_point():
            self.gamma = self.gamma.float()
        self.set_gamma(self.gamma)

    def set_gamma(self, gamma):
        gamma = torch.as_tensor(gamma, device=self.device)
        if not gamma.is_floating_point():
            gamma = gamma.float()
        if gamma.numel() != 1 or not bool(torch.isfinite(gamma)) or not 0 < float(gamma.detach()) < 1:
            raise ValueError("planning gamma must be a finite scalar in (0, 1)")
        self.gamma = gamma.reshape(())

    def scalar(self, value):
        return self.gamma.new_tensor(value)

    def value(self, value, raw=False):
        value = self.scalar(value)
        if raw and self.reward_clip > 0:
            bound = self.reward_clip / (1 - self.gamma)
            value = torch.clamp(value, -bound, bound)
        return value

    def evaluate(self, terms, value, raw=False):
        result = self.value(value, raw)
        for reward in reversed(terms):
            result = self.scalar(reward) + self.gamma * result
        return result

    def cached_tree_reps(self):
        if self.last_token is None:
            return None
        return tree_representation(*self.last_token, record=False)


class TorchNode:
    def __init__(self, context, parent, action, logit, num_actions):
        self.context = context
        self.parent = parent
        self.action = int(action)
        self.logit = float(logit)
        self.num_actions = int(num_actions)
        self.children = []
        if parent is not None:
            parent.children.append(self)
        self.r = self.v = 0.0
        self.raw_value = False
        self.done = self.visited = False
        self.rollout_n = 0
        # Each saved path runs leaf -> this node.  Native node_refresh updates
        # r along every matching path, but v only at the path's leaf.
        self.paths = []
        self.trail = []
        self.trail_value = 0.0
        self.trail_value_raw = False

    def expand(self, reward, value, done, logits, raw_value=None):
        self.r, self.v, self.done = float(reward), float(value), bool(done)
        self.raw_value = raw_value is not None and not done
        if self.raw_value:
            self.v = float(raw_value)
        for child, logit in zip(self.children, logits):
            child.logit = float(logit)

    def visit(self):
        self.trail = []
        new_rollout = not self.visited
        path = []
        node = self
        while node is not None:
            node.trail.append(self.r)
            node.trail_value = self.v
            node.trail_value_raw = self.raw_value
            path.append(node)
            if new_rollout:
                node.paths.append(tuple(path))
            node = node.parent
        self.visited = True
        self.rollout_n += 1

    def returns(self):
        return [self.context.evaluate([n.r for n in reversed(path)], path[0].v,
                                      path[0].raw_value) for path in self.paths]

    def frozen_token(self, children=True):
        node = copy.copy(self)
        node.parent = None
        node.trail = list(self.trail)
        node.paths = [tuple(ReturnTerm(n.r, n.v, n.raw_value) for n in path) for path in self.paths]
        node.children = [n.frozen_token(False) for n in self.children] if children else []
        return node

    def mean(self):
        values = self.returns()
        return torch.stack(values).mean() if values else self.context.scalar(0)

    def maximum(self):
        values = self.returns()
        # max(dim) chooses the first tie, as the native strict > loop does.
        return torch.stack(values).max(dim=0).values if values else self.context.scalar(0)

    def max_q(self):
        return (self.maximum() - self.r) / self.context.gamma

    def statistics(self, detailed, enc_type=0, enc_f_type=0, mask_type=0):
        ctx = self.context
        width = 5 * self.num_actions + (6 if detailed else 3)
        if mask_type == 3:
            return ctx.gamma.new_zeros(width)

        def encode(x):
            if not enc_type:
                return x
            if enc_f_type == 0:
                return x.sign() * (torch.sqrt(x.abs() + 1) - 1) + .001 * x
            return x.sign() * torch.log1p(x.abs())

        zero = ctx.scalar(0)
        action = [ctx.scalar(int(i == self.action)) for i in range(self.num_actions)]
        out = action + [encode(ctx.scalar(self.r)), ctx.scalar(self.done),
                        encode(ctx.value(self.v, self.raw_value)) if mask_type != 2 else zero]
        policies, means, maxima, counts = [], [], [], []
        for index in range(self.num_actions):
            child = self.children[index] if index < len(self.children) else None
            policies.append(ctx.scalar(child.logit) if child is not None and mask_type != 2 else zero)
            visible = child is not None and mask_type not in (1, 2) and not (mask_type == 5 and not detailed)
            means.append(encode(child.mean()) if visible else zero)
            maxima.append(encode(child.maximum()) if visible and mask_type != 4 else zero)
            counts.append(ctx.scalar(math.log1p(child.rollout_n)) if visible else zero)
        out += policies + means + maxima + counts
        if detailed:
            if mask_type in (1, 2, 4):
                out += [zero, zero, zero]
            else:
                trail_r = ctx.evaluate(self.trail, 0.)
                rollout = ctx.evaluate(self.trail, self.trail_value, self.trail_value_raw)
                out += [encode((trail_r - self.r) / ctx.gamma),
                        encode((rollout - self.r) / ctx.gamma), encode(self.max_q())]
        return torch.stack(out)


def tree_representation(roots, cursors, context, resets, depths, steps, phases,
                        enc_type=0, enc_f_type=0, mask_type=0, *, record=True):
    # cenv's control engine deliberately runs under no_grad.  The tree-only
    # scope reconnects gamma without making the frozen world model trainable.
    if record:
        context.last_token = ([n.frozen_token() for n in roots], [n.frozen_token() for n in cursors],
                              context, tuple(resets), tuple(depths), tuple(steps), tuple(phases),
                              enc_type, enc_f_type, mask_type)
    with torch.set_grad_enabled(context.gamma.requires_grad):
        rows = []
        for root, cursor, reset, depth, step, phase in zip(roots, cursors, resets, depths, steps, phases):
            tail = torch.stack((context.scalar(reset), context.gamma ** int(depth),
                                context.scalar(math.log1p(step)), context.scalar(math.log1p(depth)),
                                context.scalar(int(phase == 0 and step == 0))))
            rows.append(torch.cat((root.statistics(True, enc_type, enc_f_type, mask_type),
                                   cursor.statistics(False, enc_type, enc_f_type, mask_type), tail)))
        return torch.stack(rows)
