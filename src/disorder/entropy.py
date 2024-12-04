import numpy as np
import torch

MAX_ORDER = 100


def entropy(
    p: torch.Tensor, q: float, z: torch.Tensor | None = None, atol: float = 1e-8
):
    zp = p if z is None else z @ p
    is_zero = torch.abs(p) < atol
    if np.isclose(q, 1.0):
        return -torch.sum(p * zp.masked_fill(is_zero, 1.0).log(), dim=0)
    if q <= -MAX_ORDER:
        return -torch.amin(zp.masked_fill(is_zero, torch.inf), dim=0).log()
    if q >= MAX_ORDER:
        return -torch.amax(zp.masked_fill(is_zero, -torch.inf), dim=0).log()
    return (1.0 / (1.0 - q)) * (p / zp ** (1 - q)).masked_fill(is_zero, 0.0).sum(
        dim=0
    ).log()


def cross_entropy(
    p: torch.Tensor,
    r: torch.Tensor,
    q: float,
    z: torch.Tensor | None = None,
    atol: float = 1e-8,
):
    zr = r if z is None else z @ r
    is_zero = torch.abs(zr) < atol
    if np.isclose(q, 1.0):
        return (p * (1.0 / zr.masked_fill(is_zero, 1.0)).log()).sum(dim=0)
    if q <= -MAX_ORDER:
        return -torch.amin(zr.masked_fill(is_zero, torch.inf), dim=0).log()
    if q >= MAX_ORDER:
        return -torch.amax(zr.masked_fill(is_zero, -torch.inf), dim=0).log()
    return (1.0 / (1.0 - q)) * (p / zr.masked_fill(is_zero, 1.0) ** (1.0 - q)).sum(
        dim=0
    ).log()


def relative_entropy(
    p: torch.Tensor,
    r: torch.Tensor,
    q: float,
    z: torch.Tensor | None = None,
    atol: float = 1e-8,
) -> torch.Tensor:
    zp, zr = (p, r) if z is None else (z @ p, z @ r)
    is_zero = torch.abs(r) < atol
    if np.isclose(q, 1.0):
        return (p * (zp / zr.masked_fill(is_zero, 1.0)).log()).sum(dim=0)
    elif q <= -MAX_ORDER:
        return torch.amin((zp / zr).masked_fill(is_zero, torch.inf), dim=0).log()
    elif q >= MAX_ORDER:
        return torch.amax((zp / zr).masked_fill(is_zero, -torch.inf), dim=0).log()
    else:
        return (1.0 / (q - 1.0)) * (
            p * (zp / zr.masked_fill(is_zero, 1.0)) ** (q - 1.0)
        ).sum(dim=0).log()
