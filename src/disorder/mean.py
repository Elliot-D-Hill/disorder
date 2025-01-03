import torch

MAX_ORDER = 100


def geometric_mean_expansion(
    x: torch.Tensor, weights: torch.Tensor, order: float | torch.Tensor
) -> torch.Tensor:
    log_x = x.log()
    mu = weights.mul(log_x).sum(dim=0)
    sigma_sq = weights.mul(log_x.pow(2.0)).sum(dim=0).sub(mu.pow(2.0))
    return mu.exp().mul(1.0 + (sigma_sq.mul(order)) * 0.5)


def geometric_mean_expansion_2nd_order(
    x: torch.Tensor,
    weights: torch.Tensor,
    order: float | torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """
    Second-order Taylor expansion (around t=0) for the weighted power mean M_t,
    which becomes a more accurate approximation to the geometric mean for small t.

    M_t = ( sum_i [ w_i * x_i^t ] )^(1/t)
        ~ exp(mu) * [ 1 + (sigma_sq / 2)*t + ( c3 + sigma_sq^2/8 ) * t^2 ]

    where:
      mu       = sum_i [ w_i * log(x_i) ]
      sigma_sq = sum_i [ w_i * (log(x_i))^2 ] - mu^2
      alpha3   = sum_i [ w_i * (log(x_i))^3 ]
      c3       = alpha3/6 - (mu * alpha2)/2 + mu^3/3
                (with alpha2 = sum_i w_i (log(x_i))^2, so alpha2 - mu^2 = sigma_sq)

    This is accurate for order near 0.  For large |order|, use the log-sum-exp version.
    """
    log_x = x.log()
    mu = (weights * log_x).sum(dim=dim)  # sum_i w_i log(x_i)
    alpha2 = (weights * log_x.pow(2)).sum(dim=dim)  # sum_i w_i (log(x_i))^2
    alpha3 = (weights * log_x.pow(3)).sum(dim=dim)  # sum_i w_i (log(x_i))^3

    sigma_sq = alpha2 - mu.pow(2)  # variance in log(x)
    c3 = (
        alpha3 / 6.0 - mu * alpha2 / 2.0 + mu.pow(3) / 3.0
    )  # coefficient for t^3 in f(t)

    # Second-order expansion:  M_t ~ exp(mu) * [ 1 + (sigma_sq/2)*t + (c3 + sigma_sq^2/8)*t^2 ]
    # sigma_sq^2 = (sigma_sq).pow(2)
    factor = (
        1.0 + (sigma_sq * 0.5) * order + (c3 + (sigma_sq.pow(2) / 8.0)) * (order**2)
    )
    return mu.exp() * factor


def weighted_power_mean(
    x: torch.Tensor,
    weights: torch.Tensor,
    order: float | torch.Tensor,
    dim: int = 0,
    eps: float = 1e-10,
) -> torch.Tensor:
    is_zero = torch.abs(weights) < eps
    if order == 0.0:
        x.masked_fill_(is_zero, 1.0)
        return (weights * x.log()).sum(dim=dim).exp()
    if abs(order) < 1e-3:
        x.masked_fill_(is_zero, 1.0)
        return geometric_mean_expansion_2nd_order(x=x, weights=weights, order=order)
    if order <= -MAX_ORDER:
        x.masked_fill_(is_zero, float("inf"))
        return torch.amin(x, dim=dim)
    if order >= MAX_ORDER:
        x.masked_fill_(is_zero, float("-inf"))
        return torch.amax(x, dim=dim)
    x.masked_fill_(is_zero, 1.0)
    a = weights.log() + order * x.log()
    a_max = a.amax(dim=dim, keepdim=True)
    sum_exp = (a - a_max).exp().sum(dim=dim, keepdim=True)
    log_sum = a_max + sum_exp.log()
    mean_log = log_sum / order
    return mean_log.exp().squeeze(dim)


# def weighted_power_mean(
#     x: torch.Tensor,
#     p: torch.Tensor,
#     t: float | torch.Tensor,
#     atol: float = 1e-8,
#     epsilon: float = 1e-3,
# ) -> torch.Tensor:
#     is_zero = torch.abs(p) < atol
#     if t == 0.0:
#         x.masked_fill_(is_zero, 1.0)
#         return (x**p).prod(dim=0)
#     if t <= -MAX_ORDER:
#         x.masked_fill_(is_zero, float("inf"))
#         return torch.amin(x, dim=0)
#     if t >= MAX_ORDER:
#         x.masked_fill_(is_zero, float("-inf"))
#         return torch.amax(x, dim=0)
#     if abs(t) < epsilon:
#         return geometric_mean_expansion(p=p, x=x, t=t)
#     x.masked_fill_(is_zero, 1.0)
#     return (p * x**t).sum(dim=0) ** (1.0 / t)
