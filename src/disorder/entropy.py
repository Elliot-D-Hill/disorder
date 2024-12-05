import numpy as np
import torch

MAX_ORDER = 100


def entropy(
    p: torch.Tensor, alpha: float, z: torch.Tensor | None = None, atol: float = 1e-8
):
    zp = p if z is None else z @ p
    is_zero = torch.abs(p) < atol
    if np.isclose(alpha, 1.0):
        zp.masked_fill_(is_zero, 1.0)
        return -torch.sum(p * zp.log(), dim=0)
    if alpha <= -MAX_ORDER:
        zp.masked_fill_(is_zero, torch.inf)
        return -torch.amin(zp, dim=0).log()
    if alpha >= MAX_ORDER:
        zp.masked_fill_(is_zero, -torch.inf)
        return -torch.amax(zp, dim=0).log()
    zp.masked_fill(is_zero, 1.0)
    return (1.0 / (1.0 - alpha)) * (p * (1 / zp) ** (1 - alpha)).sum(dim=0).log()


def cross_entropy(
    p: torch.Tensor,
    q: torch.Tensor,
    alpha: float,
    z: torch.Tensor | None = None,
    atol: float = 1e-8,
):
    zq = q if z is None else z @ q
    is_zero = torch.abs(p) < atol
    if np.isclose(alpha, 1.0):
        zq.masked_fill_(is_zero, 1.0)
        return (p * (1.0 / zq).log()).sum(dim=0)
    if alpha <= -MAX_ORDER:
        zq.masked_fill_(is_zero, torch.inf)
        return -torch.amin(zq, dim=0).log()
    if alpha >= MAX_ORDER:
        zq.masked_fill_(is_zero, -torch.inf)
        return -torch.amax(zq, dim=0).log()
    zq.masked_fill_(is_zero, 1.0)
    return (1.0 / (1.0 - alpha)) * (p * (1 / zq) ** (1.0 - alpha)).sum(dim=0).log()


def relative_entropy(
    p: torch.Tensor,
    q: torch.Tensor,
    alpha: float,
    z: torch.Tensor | None = None,
    atol: float = 1e-8,
) -> torch.Tensor:
    zp, zq = (p, q) if z is None else (z @ p, z @ q)
    is_zero = torch.abs(p) < atol
    if np.isclose(alpha, 1.0):
        zq.masked_fill_(is_zero, 1.0)
        return (p * (zp / zq).log()).sum(dim=0)
    elif alpha <= -MAX_ORDER:
        ratio = zp / zq
        ratio.masked_fill_(is_zero, torch.inf)
        return torch.amin(ratio, dim=0).log()
    elif alpha >= MAX_ORDER:
        ratio = zp / zq
        ratio.masked_fill_(is_zero, -torch.inf)
        return torch.amax(ratio, dim=0).log()
    else:
        zq.masked_fill_(is_zero, 1.0)
        return (1.0 / (alpha - 1.0)) * (p * (zp / zq) ** (alpha - 1.0)).sum(dim=0).log()
