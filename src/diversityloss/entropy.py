import numpy as np
import torch

MAX_ORDER = 100


def relative_entropy(
    p: torch.Tensor,
    r: torch.Tensor,
    q: float | torch.Tensor,
    atol: float = 1e-8,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    weight_is_zero = torch.abs(r) < atol
    if np.isclose(q, 1.0, atol=epsilon):
        # return torch.sum(p * torch.log(p / r))
        weighted_sum = p * (torch.log(p) - torch.log(r))
        weighted_sum = weighted_sum.masked_fill(weight_is_zero, 0.0)
        return torch.sum(weighted_sum, dim=0)
    elif q <= -MAX_ORDER:
        return torch.amin(p.masked_fill(weight_is_zero, float("inf")), dim=0)
    elif q >= MAX_ORDER:
        return torch.amax(p.masked_fill(weight_is_zero, float("-inf")), dim=0)
    else:
        return (1 / (q - 1)) * torch.log(
            torch.sum(torch.pow(p, q) * torch.pow(r, 1 - q))
        )
