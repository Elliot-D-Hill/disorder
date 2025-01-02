import numpy as np
import torch
import torch.nn.functional as F

MAX_ORDER = 100


def entropy(
    p: torch.Tensor, alpha: float, z: torch.Tensor | None = None, atol: float = 1e-8
):
    zp = p if z is None else z @ p
    is_zero = torch.abs(p) < atol
    if np.isclose(alpha, 1.0):
        zp.masked_fill_(is_zero, 1.0)
        return -torch.sum(p * zp.log(), dim=0)
    if alpha <= -MAX_ORDER:
        zp.masked_fill_(is_zero, torch.inf)
        return -torch.amin(zp, dim=0).log()
    if alpha >= MAX_ORDER:
        zp.masked_fill_(is_zero, -torch.inf)
        return -torch.amax(zp, dim=0).log()
    zp.masked_fill(is_zero, 1.0)
    return (1.0 / (1.0 - alpha)) * (p * (1 / zp) ** (1 - alpha)).sum(dim=0).log()


# def cross_entropy(
#     p: torch.Tensor,
#     q: torch.Tensor,
#     alpha: float,
#     z: torch.Tensor | None = None,
#     atol: float = 1e-8,
# ):
#     zq = q if z is None else z @ q
#     q = F.softmax(q, dim=0)
#     is_zero = torch.abs(p) < atol
#     if np.isclose(alpha, 1.0):
#         zq.masked_fill_(is_zero, 1.0)
#         return (p * (1.0 / zq).log()).sum(dim=0)
#     if alpha <= -MAX_ORDER:
#         zq.masked_fill_(is_zero, torch.inf)
#         return -torch.amin(zq, dim=0).log()
#     if alpha >= MAX_ORDER:
#         zq.masked_fill_(is_zero, -torch.inf)
#         return -torch.amax(zq, dim=0).log()
#     zq.masked_fill_(is_zero, 1.0)
#     return (1.0 / (1.0 - alpha)) * (p * (1 / zq) ** (1.0 - alpha)).sum(dim=0).log()


def relative_entropy(
    p: torch.Tensor,
    q: torch.Tensor,
    alpha: float,
    z: torch.Tensor | None = None,
    atol: float = 1e-8,
) -> torch.Tensor:
    zp, zq = (p, q) if z is None else (z @ p, z @ q)
    is_zero = torch.abs(p) < atol
    if np.isclose(alpha, 1.0):
        zq.masked_fill_(is_zero, 1.0)
        return (p * (zp / zq).log()).sum(dim=0)
    elif alpha <= -MAX_ORDER:
        ratio = zp / zq
        ratio.masked_fill_(is_zero, torch.inf)
        return torch.amin(ratio, dim=0).log()
    elif alpha >= MAX_ORDER:
        ratio = zp / zq
        ratio.masked_fill_(is_zero, -torch.inf)
        return torch.amax(ratio, dim=0).log()
    else:
        zq.masked_fill_(is_zero, 1.0)
        return (1.0 / (alpha - 1.0)) * (p * (zp / zq) ** (alpha - 1.0)).sum(dim=0).log()


def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    similarity: torch.Tensor | None = None,
    order: float = 1.0,
    dim: int = -1,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""
    Computes the Rényi cross-entropy (or the Leinster–Cobbold cross-entropy if
    a similarity matrix is provided) in a numerically stable manner, mirroring
    PyTorch's classification-based API.

    This function:
    1) Applies :func:`torch.nn.functional.log_softmax` to ``input`` along dimension ``dim``.
    2) Optionally smooths the target distribution with a class similarity matrix (Leinster–Cobbold).
    3) Uses a log-sum-exp approach for ``order != 1`` to compute the Rényi cross-entropy.
    4) Recovers the Shannon cross-entropy (negative log-likelihood) when ``order == 1``.

    Args:
        input:
            Unnormalized scores of shape ``(N, C)`` where ``N`` is the batch size
            and ``C`` is the number of classes.
        target:
            Soft labels (probabilities) of shape ``(N, C)``.
        similarity:
            A class-similarity matrix of shape ``(C, C)``. If provided, the target
            distribution is transformed by this matrix before computing the loss
            (Leinster–Cobbold cross-entropy). Defaults to ``None`` (no transformation),
            which is equivalent to setting the similarity matrix to the identity matrix.
        order:
            The Rényi entropy order parameter :math:`\alpha`. Defaults to ``1.0``
            (which yields the standard Shannon cross-entropy).
        dim:
            The dimension along which :func:`log_softmax` is applied. Typically ``-1``
            for inputs of shape ``(N, C)``. Defaults to ``-1``.
        reduction:
            Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
        eps:
            A small constant added to probabilities before taking the logarithm
            to avoid numerical issues with :math:`\log(0)`. Defaults to ``1e-12``.

    Returns:
        Tensor:
            The scalar cross-entropy loss (if ``reduction != 'none'``) or a
            per-sample tensor of shape ``(N,)`` (if ``reduction == 'none'``).

    Shape:
        - ``input``: :math:`(N, C)`
        - ``target``: :math:`(N, C)`
        - ``similarity``: :math:`(C, C)` (optional)
        - Output: depends on ``reduction``.

    Example::

        >>> from disorder.entropy import cross_entropy
        >>> import torch
        >>> torch.manual_seed(0)
        >>> n_classes = 3
        >>> n_instances = 2
        >>> n_features = 4
        >>> input = torch.randn(n_instances, n_classes)
        >>> target = torch.rand(n_instances, n_classes)
        >>> features = torch.randn(n_classes, n_features)
        >>> similarity = torch.exp(-torch.cdist(features, features))
        >>> cross_entropy(input, target, similarity=similarity, order=2.0)
    """
    log_q = F.log_softmax(input, dim=dim)
    if similarity is not None:
        log_q = log_q.exp() @ similarity
        log_q = torch.log(log_q + eps)
    if abs(order - 1.0) < 1e-9:
        ce = -(target * log_q).sum(dim=dim)
    else:
        log_p = (target + eps).log()
        a = order * log_p + (1.0 - order) * log_q
        a_max, _ = a.max(dim=dim, keepdim=True)
        sum_exp = torch.exp(a - a_max).sum(dim=dim, keepdim=True)
        log_sum_exp = a_max + sum_exp.log()
        ce = (1.0 / (order - 1.0)) * log_sum_exp.squeeze(dim)
    if reduction == "none":
        return ce
    elif reduction == "mean":
        return ce.mean()
    elif reduction == "sum":
        return ce.sum()
    else:
        raise ValueError(
            f"Invalid reduction='{reduction}'. Use 'none', 'mean', or 'sum'."
        )
