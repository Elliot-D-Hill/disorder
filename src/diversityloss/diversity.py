from typing import Callable
from torch import (
    Tensor,
    broadcast_to,
    prod,
    pow,
    amin,
    amax,
    sum,
    where,
    ones_like,
    zeros_like,
    abs,
)
from torch.nn import Module
from numpy import isclose


MAX_ORDER = 32


# TODO tolerance (atol) should be higher if using entropy instead of diversity
# because entropy uses logs, which are more numerically stable; we can also can
# increase the max and min order
def weighted_power_mean(
    items: Tensor, weights: Tensor, order: float, atol: float = 1e-6
) -> Tensor:
    weight_is_nonzero = abs(weights) >= atol
    if isclose(order, 0.0, atol=atol):
        weighted_items = where(weight_is_nonzero, pow(items, weights), ones_like(items))
        return prod(weighted_items, dim=0)
    elif order < -MAX_ORDER:
        return amin(where(weight_is_nonzero, items, float("inf")), dim=0)
    elif order > MAX_ORDER:
        return amax(where(weight_is_nonzero, items, float("-inf")), dim=0)
    else:
        exponentiated_items = pow(items, exponent=order)
        weighted_items = where(
            weight_is_nonzero, exponentiated_items * weights, zeros_like(items)
        )
        weighted_item_sum = sum(weighted_items, dim=0)
        return pow(weighted_item_sum, exponent=1 / order)


def weight_abundance(abundance: Tensor, similarity: Tensor | None = None) -> Tensor:
    if similarity is None:
        return abundance
    return similarity @ abundance


def alpha(abundance: Tensor, _, similarity: Tensor | None = None) -> Tensor:
    return 1 / weight_abundance(abundance, similarity)


def rho(abundance: Tensor, _, similarity: Tensor | None = None) -> Tensor:
    subcommunity_similarity = weight_abundance(abundance, similarity)
    metacommunity_abundance = abundance.sum(dim=1, keepdim=True)
    metacommunity_similarity = weight_abundance(metacommunity_abundance, similarity)
    return metacommunity_similarity / subcommunity_similarity


def beta(abundance: Tensor, _, similarity: Tensor | None = None) -> Tensor:
    return 1 / rho(abundance, _, similarity)


def gamma(abundance: Tensor, _, similarity: Tensor | None = None) -> Tensor:
    metacommunity_abundance = abundance.sum(dim=1, keepdim=True)
    metacommunity_abundance = broadcast_to(metacommunity_abundance, abundance.shape)
    metacommunity_similarity = weight_abundance(metacommunity_abundance, similarity)
    return 1 / metacommunity_similarity


def normalized_alpha(_, normalized_abundance: Tensor, similarity: Tensor | None = None):
    return 1 / weight_abundance(normalized_abundance, similarity)


def normalized_rho(
    abundance: Tensor, normalized_abundance: Tensor, similarity: Tensor | None = None
) -> Tensor:
    normalized_subcommunity_similarity = weight_abundance(
        normalized_abundance, similarity
    )
    metacommunity_abundance = abundance.sum(dim=1, keepdim=True)
    metacommunity_similarity = weight_abundance(metacommunity_abundance, similarity)
    return metacommunity_similarity / normalized_subcommunity_similarity


def normalized_beta(
    abundance: Tensor, normalized_abundance: Tensor, similarity: Tensor | None = None
) -> Tensor:
    return 1 / normalized_rho(abundance, normalized_abundance, similarity)


# Note: gamma cannot be normalized, so it will be returned whether normalize = True or False
MEASURES: dict[tuple[str, bool], Callable[[Tensor, Tensor, Tensor | None], Tensor]] = {
    ("alpha", False): alpha,
    ("beta", False): beta,
    ("rho", False): rho,
    ("alpha", True): normalized_alpha,
    ("beta", True): normalized_beta,
    ("rho", True): normalized_rho,
    ("gamma", True): gamma,
    ("gamma", False): gamma,
}


class Diversity(Module):
    def __init__(self, viewpoint: float, measure: str, normalize: bool = True) -> None:
        super().__init__()
        self._validate_args(measure, normalize)
        self.order = 1.0 - viewpoint
        self.measure = MEASURES[(measure, normalize)]

    def _validate_args(self, measure: str, normalize: bool) -> None:
        if measure not in {"alpha", "beta", "rho", "gamma"}:
            raise ValueError(
                f"Invalid 'measure' argument: {measure}. Expected one of: 'alpha', 'beta', 'rho', 'gamma'."
            )
        if not isinstance(normalize, bool):
            raise ValueError(
                f"Invalid 'normalize' argument: {normalize}. Expected a bool."
            )

    def subcommunity_diversity(
        self,
        abundance: Tensor,
        normalizing_constants: Tensor,
        similarity: Tensor | None = None,
    ) -> Tensor:
        normalized_abundance = abundance / normalizing_constants
        community_ratio = self.measure(abundance, normalized_abundance, similarity)
        return weighted_power_mean(
            items=community_ratio, weights=normalized_abundance, order=self.order
        )

    def forward(self, abundance: Tensor, similarity: Tensor | None = None) -> Tensor:
        normalizing_constants = abundance.sum(dim=0)
        subcommunity_diversity = self.subcommunity_diversity(
            abundance=abundance,
            normalizing_constants=normalizing_constants,
            similarity=similarity,
        )
        return weighted_power_mean(
            items=subcommunity_diversity,
            weights=normalizing_constants,
            order=self.order,
        )
