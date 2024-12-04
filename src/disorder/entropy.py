import numpy as np
import torch

MAX_ORDER = 100


def entropy(p: torch.Tensor, q: float, z: torch.Tensor | None = None):
    p = z @ p if z is not None else p
    return (1.0 / (1.0 - q)) * torch.log(torch.sum(p**q, dim=0))


def cross_entropy(
    p: torch.Tensor,
    r: torch.Tensor,
    q: float,
    z: torch.Tensor | None = None,
    atol: float = 1e-8,
):
    zr = r if z is None else z @ r
    is_zero = torch.abs(zr) < atol
    zr = zr.masked_fill(is_zero, 1.0)
    return (1.0 / (1.0 - q)) * torch.log(torch.sum(p * (1.0 / zr) ** (1.0 - q), dim=0))


def relative_entropy(
    p: torch.Tensor,
    r: torch.Tensor,
    q: float,
    z: torch.Tensor | None = None,
    atol: float = 1e-8,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    if z is None:
        zp = p
        zr = r
    else:
        zp = z @ p
        zr = z @ r
    is_zero = torch.abs(r) < atol
    if np.isclose(q, 1.0, atol=epsilon):
        r = r.masked_fill(is_zero, 1.0)
        return (p * (zp / zr).log()).sum(dim=0)
    elif q <= -MAX_ORDER:
        return torch.amin(p.masked_fill(is_zero, float("inf")), dim=0).log()
    elif q >= MAX_ORDER:
        return torch.amax(p.masked_fill(is_zero, float("-inf")), dim=0).log()
    else:
        r = r.masked_fill(is_zero, 1.0)
        return (1.0 / (q - 1.0)) * torch.log(
            torch.sum(p * (zp / zr) ** (q - 1.0), dim=0)
        )
