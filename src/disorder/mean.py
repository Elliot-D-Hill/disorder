import torch

MAX_ORDER = 100


def geometric_mean_expansion(
    input: torch.Tensor,
    weights: torch.Tensor,
    order: float | torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    log_input = input.log()
    mu = weights.mul(log_input).sum(dim=dim)
    sigma_sq = weights.mul(log_input.pow(2.0)).sum(dim=0).sub(mu.pow(2.0))
    return mu.exp().mul(1.0 + (sigma_sq.mul(order)) * 0.5)


def geometric_mean_2nd_order_expansion(
    input: torch.Tensor,
    weights: torch.Tensor,
    order: float | torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """
    Second-order Taylor expansion (around t=0) for the weighted power mean M_t,
    which becomes a more accurate approximation to the geometric mean for t close to 0.

    M_t = ( sum_i [ w_i * x_i^t ] )^(1/t)
        ~ exp(mu) * [ 1 + (sigma_sq / 2)*t + ( c3 + sigma_sq^2/8 ) * t^2 ]

    where:
      mu       = sum_i [ w_i * log(x_i) ]
      sigma_sq = sum_i [ w_i * (log(x_i))^2 ] - mu^2
      alpha3   = sum_i [ w_i * (log(x_i))^3 ]
      c3       = alpha3/6 - (mu * alpha2)/2 + mu^3/3
                (with alpha2 = sum_i w_i (log(x_i))^2, so alpha2 - mu^2 = sigma_sq)
    """
    log_input = input.log()
    mu = (weights * log_input).sum(dim=dim)
    alpha2 = (weights * log_input.pow(2)).sum(dim=dim)
    alpha3 = (weights * log_input.pow(3)).sum(dim=dim)
    sigma_sq = alpha2 - mu.pow(2)
    c3 = alpha3 / 6.0 - mu * alpha2 / 2.0 + mu.pow(3) / 3.0
    factor = (
        1.0 + (sigma_sq * 0.5) * order + (c3 + (sigma_sq.pow(2) / 8.0)) * (order**2)
    )
    return mu.exp() * factor


def weighted_power_mean(
    input: torch.Tensor,
    weights: torch.Tensor,
    order: float | torch.Tensor,
    dim: int = 0,
    weight_epsilon: float = 1e-8,
    order_epsilon: float = 1e-2,
) -> torch.Tensor:
    is_zero = torch.abs(weights) < weight_epsilon
    if order == 0.0:
        input.masked_fill_(is_zero, 1.0)
        return (weights * input.log()).sum(dim=dim).exp()
    if abs(order) < order_epsilon:
        input.masked_fill_(is_zero, 1.0)
        return geometric_mean_2nd_order_expansion(
            input=input, weights=weights, order=order
        )
    if order <= -MAX_ORDER:
        input.masked_fill_(is_zero, float("inf"))
        return torch.amin(input, dim=dim)
    if order >= MAX_ORDER:
        input.masked_fill_(is_zero, float("-inf"))
        return torch.amax(input, dim=dim)
    input.masked_fill_(is_zero, 1.0)
    a = weights.log() + order * input.log()
    a_max = a.amax(dim=dim, keepdim=True)
    sum_exp = (a - a_max).exp().sum(dim=dim, keepdim=True)
    log_sum = a_max + sum_exp.log()
    mean_log = log_sum / order
    return mean_log.exp().squeeze(dim)
