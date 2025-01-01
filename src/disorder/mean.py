import torch

MAX_ORDER = 100


def geometric_mean_expansion(
    p: torch.Tensor, x: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    log_x = x.log()
    mu = p.mul(log_x).sum(dim=0)
    sigma_sq = p.mul(log_x.pow(2.0)).sum(dim=0).sub(mu.pow(2.0))
    return mu.exp().mul(1.0 + (sigma_sq.mul(t)) * 0.5)


def logsumexp_power_mean(x, p, t):
    a = torch.log(p) + t * torch.log(x)
    a_max = torch.max(a, dim=0).values
    sum_exp = torch.exp(a - a_max).sum(dim=0)
    log_num = a_max + torch.log(sum_exp)
    log_den = torch.log(torch.sum(p, dim=0))
    exponent = (log_num - log_den) / t
    return torch.exp(exponent)


def weighted_power_mean(
    x: torch.Tensor,
    p: torch.Tensor,
    t: float | torch.Tensor,
    atol: float = 1e-8,
    epsilon: float = 1e-3,
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
    # return logsumexp_power_mean(x=x, p=p, t=t)


# def geometric_mean_expansion(
#     p: torch.Tensor, x: torch.Tensor, t: float | torch.Tensor
# ) -> torch.Tensor:
#     """
#     Approximate the geometric mean for small t by second-order expansion.
#     (Optional feature if you want a smoother approximation around t ≈ 0.)
#     """
#     log_x = x.log()
#     mu = p.mul(log_x).sum(dim=0)
#     sigma_sq = p.mul(log_x.pow(2.0)).sum(dim=0) - mu.pow(2.0)
#     return mu.exp().mul(1.0 + sigma_sq.mul(t) * 0.5)


# def weighted_power_mean(
#     x: torch.Tensor,
#     p: torch.Tensor,
#     t: float | torch.Tensor,
#     atol: float = 1e-8,
#     epsilon: float = 1e-4,
#     max_order: int = MAX_ORDER,
#     use_geometric_expansion: bool = True,
# ) -> torch.Tensor:
#     """
#     A single function that computes the weighted power mean using:
#       1) Special handling for t=0 (geometric mean).
#       2) Handling for large ±t (min or max).
#       3) (Optional) geometric expansion for very small |t|.
#       4) Log-sum-exp for general t, ensuring numerical stability.

#     Parameters
#     ----------
#     x : torch.Tensor
#         Data tensor of shape (n, ...) for which we compute the power mean along dim=0.
#     p : torch.Tensor
#         Weights (same shape as x, or broadcastable) that sum to 1 (or partial weights).
#     t : float or torch.Tensor
#         Exponent.  t=0 => geometric mean, t=1 => arithmetic mean, etc.
#     atol : float
#         Threshold for considering weights p "zero".
#     epsilon : float
#         Threshold for deciding when |t| is close to 0 (if `use_geometric_expansion`=True).
#     max_order : int
#         Large magnitude cutoff for t => min or max operation.
#     use_geometric_expansion : bool
#         If True, use a second-order expansion around t=0 for improved continuity.

#     Returns
#     -------
#     torch.Tensor
#         The weighted power mean of x along dim=0, shape (...).
#     """
#     # 1) Mask out small weights so we don't log(0)
#     is_zero = torch.abs(p) < atol
#     # set x_i=1 for p_i≈0 to remove it from the product/sum
#     x.masked_fill_(is_zero, 1.0)

#     # 2) Large negative or large positive t => min or max
#     if t <= -max_order:
#         # Weighted power mean with t -> -∞ => minimum
#         # If p_i=0 => ignore that x_i (effectively "∞" for min).
#         x.masked_fill_(is_zero, float("inf"))
#         return torch.amin(x, dim=0)

#     if t >= max_order:
#         # Weighted power mean with t -> +∞ => maximum
#         # If p_i=0 => ignore that x_i (effectively "-∞" for max).
#         x.masked_fill_(is_zero, float("-inf"))
#         return torch.amax(x, dim=0)

#     # 3) Exactly t=0 => geometric mean
#     #    (or near-zero t => optional geometric expansion)
#     if isinstance(t, float) and abs(t) < atol:
#         # True geometric mean
#         return (x.log().mul(p)).sum(dim=0).exp()
#     elif use_geometric_expansion and isinstance(t, float) and abs(t) < epsilon:
#         # Approximate geometric mean for small t
#         return geometric_mean_expansion(p=p, x=x, t=t)

#     # 4) General case => log-sum-exp approach
#     # Weighted sum of x^t = sum_i p_i * x_i^t
#     # log-sum-exp to avoid overflow in x_i^t for large x_i,t
#     # => M_t = ( sum_i [ p_i * x_i^t ] )^(1/t)
#     # We'll do: log( p_i ) + t * log( x_i )
#     a = torch.log(p) + t * torch.log(x)
#     a_max = torch.max(a, dim=0).values
#     sum_exp = torch.exp(a - a_max).sum(dim=0)
#     log_num = a_max + torch.log(sum_exp)  # log of numerator
#     log_den = torch.log(p.sum(dim=0))  # log of sum of weights
#     exponent = (log_num - log_den) / t
#     return torch.exp(exponent)
