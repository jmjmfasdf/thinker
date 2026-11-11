"""Stable reductions without changing loss weights or clipping thresholds."""
import torch


def masked_squared_sum(value, dim=-1, mask=None):
    if mask is not None:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=value.device)
        if tuple(value.shape[:mask.ndim]) != tuple(mask.shape):
            raise ValueError("squared-sum mask must match leading dimensions")
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        value = torch.where(mask, value, torch.zeros_like(value))
    return value.to(torch.float64).square().sum(dim=dim)


def stable_grad_norm(parameters, norm_type=2.):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0., dtype=torch.float64)
    device = grads[0].device
    norms = [torch.linalg.vector_norm(g.detach(), ord=float(norm_type),
                                      dtype=torch.float64).to(device) for g in grads]
    return torch.linalg.vector_norm(torch.stack(norms), ord=float(norm_type))


@torch.no_grad()
def stable_clip_grad_norm_(parameters, max_norm, norm_type=2.):
    parameters = list(parameters)
    norm = stable_grad_norm(parameters, norm_type)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("non-finite gradient in stable clipping")
    coefficient = (float(max_norm) / (norm + 1e-6)).clamp(max=1.)
    for p in parameters:
        if p.grad is not None:
            # Multiply in FP64 as well: the coefficient itself can be smaller
            # than the gradient dtype can represent, despite a valid result.
            p.grad.copy_(p.grad.to(torch.float64) * coefficient.to(p.grad.device))
    return norm
