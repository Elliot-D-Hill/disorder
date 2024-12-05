from functools import partial
from typing import Callable

import torch

MAX_ORDER = 64


def geometric_mean_expansion(
    p: torch.Tensor, x: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    log_x = x.log()
    mu = p.mul(log_x).sum(dim=0)
    sigma_sq = p.mul(log_x.pow(2.0)).sum(dim=0).sub(mu.pow(2.0))
    return mu.exp().mul(1.0 + (sigma_sq.mul(t)) * 0.5)


def weighted_power_mean(
    x: torch.Tensor,
    p: torch.Tensor,
    t: float | torch.Tensor,
    atol: float = 1e-8,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    is_zero = torch.abs(p) < atol
    if t == 0.0:
        x.masked_fill_(is_zero, 1.0)
        return (x**p).prod(dim=0)
    if t <= -MAX_ORDER:
        x.masked_fill_(is_zero, float("inf"))
        return torch.amin(x, dim=0)
    if t >= MAX_ORDER:
        x.masked_fill_(is_zero, float("-inf"))
        return torch.amax(x, dim=0)
    if abs(t) < epsilon:
        return geometric_mean_expansion(p=p, x=x, t=t)
    x.masked_fill_(is_zero, 1.0)
    return (p * x**t).sum(dim=0) ** (1.0 / t)


def weight_abundance(
    abundance: torch.Tensor, similarity: torch.Tensor | None = None
) -> torch.Tensor:
    if similarity is None:
        return abundance
    return similarity @ abundance


def alpha(
    abundance: torch.Tensor, similarity: torch.Tensor | None = None
) -> torch.Tensor:
    return 1 / weight_abundance(abundance, similarity)


def rho(
    abundance: torch.Tensor,
    metacommunity_abundance: torch.Tensor,
    similarity: torch.Tensor | None = None,
) -> torch.Tensor:
    subcommunity_similarity = weight_abundance(abundance, similarity)
    metacommunity_similarity = weight_abundance(metacommunity_abundance, similarity)
    return metacommunity_similarity / subcommunity_similarity


def gamma(
    abundance: torch.Tensor,
    metacommunity_abundance: torch.Tensor,
    similarity: torch.Tensor | None = None,
) -> torch.Tensor:
    metacommunity_abundance = torch.broadcast_to(
        metacommunity_abundance, abundance.shape
    )
    metacommunity_similarity = weight_abundance(metacommunity_abundance, similarity)
    return 1 / metacommunity_similarity


MEASURES: dict[str, Callable] = {
    "alpha": alpha,
    "beta": rho,
    "rho": rho,
    "gamma": gamma,
}


def community_ratio(
    measure: str,
    abundance: torch.Tensor,
    normalized_abundance: torch.Tensor,
    normalize: bool,
    similarity: torch.Tensor | None = None,
):
    f = MEASURES[measure]
    if f in {gamma, rho}:
        metacommunity_abundance = abundance.sum(dim=1, keepdim=True)
        f = partial(f, metacommunity_abundance=metacommunity_abundance)
    if normalize:
        f = partial(f, abundance=normalized_abundance)
    else:
        f = partial(f, abundance=abundance)
    if similarity is not None:
        f = partial(f, similarity=similarity)
    return f()


def subcommunity_diversity(
    abundance: torch.Tensor,
    normalizing_constants: torch.Tensor,
    viewpoint: float | torch.Tensor,
    measure: str,
    normalize: bool = True,
    similarity: torch.Tensor | None = None,
) -> torch.Tensor:
    order = 1.0 - viewpoint
    normalized_abundance = abundance / normalizing_constants
    ratio = community_ratio(
        measure=measure,
        abundance=abundance,
        normalized_abundance=normalized_abundance,
        normalize=normalize,
        similarity=similarity,
    )
    subcommunity_diversity = weighted_power_mean(
        x=ratio, p=normalized_abundance, t=order
    )
    if measure == "beta":
        subcommunity_diversity = 1 / subcommunity_diversity
    return subcommunity_diversity


def _validate_args(measure: str, normalize: bool) -> None:
    if measure not in {"alpha", "beta", "rho", "gamma"}:
        raise ValueError(
            f"Invalid 'measure' argument: {measure}. Expected one of: 'alpha', 'beta', 'rho', 'gamma'."
        )
    if not isinstance(normalize, bool):
        raise ValueError(
            f"Invalid 'normalize' argument: {normalize}. Expected type bool."
        )


def diversity(
    abundance: torch.Tensor,
    viewpoint: float | torch.Tensor,
    measure: str,
    normalize: bool = True,
    similarity: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_args(measure, normalize)
    order = 1.0 - viewpoint
    normalizing_constants = abundance.sum(dim=0)
    sub_diversity = subcommunity_diversity(
        abundance=abundance,
        normalizing_constants=normalizing_constants,
        viewpoint=viewpoint,
        measure=measure,
        normalize=normalize,
        similarity=similarity,
    )
    return weighted_power_mean(
        x=sub_diversity,
        p=normalizing_constants,
        t=order,
    )


class Diversity(torch.nn.Module):
    def __init__(self, viewpoint: float, measure: str, normalize: bool = True) -> None:
        super().__init__()
        self.viewpoint = viewpoint
        self.measure = measure
        self.normalize = normalize

    def forward(
        self, abundance: torch.Tensor, similarity: torch.Tensor | None = None
    ) -> torch.Tensor:
        return diversity(
            abundance=abundance,
            viewpoint=self.viewpoint,
            measure=self.measure,
            normalize=self.normalize,
            similarity=similarity,
        )
