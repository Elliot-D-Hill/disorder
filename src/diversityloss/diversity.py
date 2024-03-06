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


def validate_args(measure: str, normalize: bool) -> None:
    if measure not in {"alpha", "beta", "rho", "gamma"}:
        raise ValueError(
            f"Invalid 'measure' argument: {measure}. Expected one of: 'alpha', 'beta', 'rho', 'gamma'."
        )
    if not isinstance(normalize, bool):
        raise ValueError(f"Invalid 'normalize' argument: {normalize}. Expected a bool.")


# TODO tolerance (atol) should be higher if using entropy instead of diversity
# because entropy uses logs, which are more numerically stable
def power_mean(
    order: float, weights: Tensor, items: Tensor, atol: float = 1e-6
) -> Tensor:
    weight_is_nonzero = abs(weights) >= atol
    if isclose(order, 0.0, atol=atol):
        weighted_items = where(weight_is_nonzero, pow(items, weights), ones_like(items))
        return prod(weighted_items, dim=0)
    elif order < -100:
        return amin(where(weight_is_nonzero, items, float("inf")), dim=0)
    elif order > 100:
        return amax(where(weight_is_nonzero, items, float("-inf")), dim=0)
    else:
        pow_items = pow(items, order)
        weighted_pow_items = where(
            weight_is_nonzero, pow_items * weights, zeros_like(items)
        )
        items_sum = sum(weighted_pow_items, dim=0)
        return pow(items_sum, 1 / order)


def alpha(abundance: Tensor, _) -> Tensor:
    return 1 / abundance


def rho(abundance: Tensor, _) -> Tensor:
    metacommunity_abundance = abundance.sum(dim=1, keepdim=True)
    return abundance / metacommunity_abundance


def beta(abundance: Tensor, _) -> Tensor:
    return 1 / rho(abundance, _)


def gamma(abundance: Tensor, _) -> Tensor:
    metacommunity_abundance = abundance.sum(dim=1, keepdim=True)
    metacommunity_abundance = broadcast_to(metacommunity_abundance, abundance.shape)
    return 1 / metacommunity_abundance


def normalized_alpha(_, normalized_abundance: Tensor) -> Tensor:
    return 1 / normalized_abundance


def normalized_rho(abundance: Tensor, normalized_abundance: Tensor) -> Tensor:
    metacommunity_abundance = abundance.sum(dim=1, keepdim=True)
    return metacommunity_abundance / normalized_abundance


def normalized_beta(abundance: Tensor, normalized_abundance: Tensor) -> Tensor:
    return 1 / normalized_rho(abundance, normalized_abundance)


FREQUENCY_MEASURES: dict[tuple[str, bool], Callable[[Tensor, Tensor], Tensor]] = {
    ("alpha", False): alpha,
    ("beta", False): beta,
    ("rho", False): rho,
    ("gamma", False): gamma,
    ("alpha", True): normalized_alpha,
    ("beta", True): normalized_beta,
    ("rho", True): normalized_rho,
}


class FrequencySensitiveDiversity(Module):
    def __init__(self, viewpoint: float, measure: str, normalize: bool = True) -> None:
        super().__init__()
        validate_args(measure, normalize)
        self.viewpoint = viewpoint
        self.measure = FREQUENCY_MEASURES[(measure, normalize)]

    def forward(self, abundance: Tensor) -> Tensor:
        normalizing_constants = abundance.sum(dim=0)
        normalized_abundance = abundance / normalizing_constants
        community_ratio = self.measure(abundance, normalized_abundance)
        community_ratio = where(
            community_ratio != 0.0, community_ratio, zeros_like(community_ratio)
        )
        order = 1.0 - self.viewpoint
        subcommunity_diversity = power_mean(
            weights=normalized_abundance, items=community_ratio, order=order
        )
        return power_mean(
            weights=normalizing_constants, items=subcommunity_diversity, order=order
        )


def alpha_similarity(abundance: Tensor, _, similarity: Tensor) -> Tensor:
    return 1 / (similarity @ abundance)


def rho_similarity(abundance: Tensor, _, similarity: Tensor) -> Tensor:
    subcommunity_similarity = similarity @ abundance
    metacommunity_abundance = abundance.sum(dim=1, keepdim=True)
    metacommunity_similarity = similarity @ metacommunity_abundance
    return metacommunity_similarity / subcommunity_similarity


def beta_similarity(abundance: Tensor, _, similarity: Tensor) -> Tensor:
    return 1 / rho_similarity(abundance, _, similarity)


def gamma_similarity(abundance: Tensor, _, similarity: Tensor) -> Tensor:
    metacommunity_abundance = abundance.sum(dim=1, keepdim=True)
    metacommunity_abundance = broadcast_to(metacommunity_abundance, abundance.shape)
    metacommunity_similarity = similarity @ metacommunity_abundance
    return 1 / metacommunity_similarity


def normalized_alpha_similarity(_, normalized_abundance: Tensor, similarity: Tensor):
    return 1 / (similarity @ normalized_abundance)


def normalized_rho_similarity(
    abundance: Tensor, normalized_abundance: Tensor, similarity: Tensor
) -> Tensor:
    normalized_subcommunity_similarity = similarity @ normalized_abundance
    metacommunity_abundance = abundance.sum(dim=1, keepdim=True)
    metacommunity_similarity = similarity @ metacommunity_abundance
    return metacommunity_similarity / normalized_subcommunity_similarity


def normalized_beta_similarity(
    abundance: Tensor, normalized_abundance: Tensor, similarity: Tensor
) -> Tensor:
    return 1 / normalized_rho_similarity(abundance, normalized_abundance, similarity)


SIMILARITY_MEASURES: dict[
    tuple[str, bool], Callable[[Tensor, Tensor, Tensor], Tensor]
] = {
    ("alpha", False): alpha_similarity,
    ("beta", False): beta_similarity,
    ("rho", False): rho_similarity,
    ("gamma", False): gamma_similarity,
    ("alpha", True): normalized_alpha_similarity,
    ("beta", True): normalized_beta_similarity,
    ("rho", True): normalized_rho_similarity,
}


class SimilaritySensitiveDiversity(Module):
    def __init__(self, viewpoint: float, measure: str, normalize: bool = True) -> None:
        super().__init__()
        validate_args(measure, normalize)
        self.viewpoint = viewpoint
        self.measure = SIMILARITY_MEASURES[(measure, normalize)]

    def forward(self, abundance: Tensor, similarity) -> Tensor:
        normalizing_constants = abundance.sum(dim=0)
        normalized_abundance = abundance / normalizing_constants
        community_ratio = self.measure(abundance, normalized_abundance, similarity)
        community_ratio = where(
            community_ratio != 0.0, community_ratio, zeros_like(community_ratio)
        )
        order = 1.0 - self.viewpoint
        subcommunity_diversity = power_mean(
            weights=normalized_abundance, items=community_ratio, order=order
        )
        return power_mean(
            weights=normalizing_constants, items=subcommunity_diversity, order=order
        )
