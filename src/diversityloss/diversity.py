from functools import partial
from typing import Callable

import numpy as np
import torch

MAX_ORDER = 32


def weighted_power_mean(
    items: torch.Tensor, weights: torch.Tensor, order: float, atol: float = 1e-6
) -> torch.Tensor:
    weight_is_zero = torch.abs(weights) < atol
    if np.isclose(order, 0.0, atol=atol):
        weighted_items = torch.pow(items, weights).masked_fill(weight_is_zero, 1.0)
        return torch.prod(weighted_items, dim=0)
    elif order < -MAX_ORDER:
        return torch.amin(items.masked_fill(weight_is_zero, float("inf")), dim=0)
    elif order > MAX_ORDER:
        return torch.amax(items.masked_fill(weight_is_zero, float("-inf")), dim=0)
    else:
        exponentiated_items = torch.pow(items, order).masked_fill(weight_is_zero, 0.0)
        return torch.sum(exponentiated_items * weights, dim=0) ** (1.0 / order)


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
    viewpoint: float,
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
        items=ratio, weights=normalized_abundance, order=order
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
    viewpoint: float,
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
        items=sub_diversity,
        weights=normalizing_constants,
        order=order,
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
