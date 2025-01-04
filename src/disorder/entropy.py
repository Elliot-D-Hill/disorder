import torch
import torch.nn.functional as F


def reduce_tensor(input: torch.Tensor, reduction: str) -> torch.Tensor:
    match reduction:
        case "none":
            return input
        case "mean":
            return input.mean()
        case "sum":
            return input.sum()
        case _:
            raise ValueError(
                f"Invalid reduction='{reduction}'. Use 'none', 'mean', or 'sum'"
            )


def entropy(
    input: torch.Tensor,
    order: float = 1.0,
    similarity: torch.Tensor | None = None,
    reduction: str = "mean",
    dim: int = -1,
    eps: float = 1e-12,
):
    log_probs = F.log_softmax(input, dim=dim)
    probs = log_probs.exp()
    if similarity is not None:
        probs = probs @ similarity
    probs = torch.clamp(probs, min=eps)
    log_probs = probs.log()
    if abs(order - 1.0) < 1e-9:
        out = -(probs * log_probs).sum(dim=dim)
    else:
        a = order * log_probs
        a_max = a.amax(dim=dim, keepdim=True)
        sum_exp = (a - a_max).exp().sum(dim=dim, keepdim=True)
        log_sum = a_max + sum_exp.log()
        out = (1.0 / (1.0 - order)) * log_sum.squeeze(dim)
    return reduce_tensor(out, reduction=reduction)


def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    similarity: torch.Tensor | None = None,
    order: float = 1.0,
    reduction: str = "mean",
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    log_probs = F.log_softmax(input, dim=dim)
    if similarity is not None:
        weighted_probs = log_probs.exp() @ similarity
        weighted_probs = torch.clamp(weighted_probs, min=eps)
        log_probs = weighted_probs.log()
    target = torch.clamp(target, min=eps)
    if abs(order - 1.0) < 1e-9:
        out = -(target * log_probs).sum(dim=dim)
    else:
        a = order * target.log() + (1.0 - order) * log_probs
        a_max = a.amax(dim=dim, keepdim=True)
        sum_exp = torch.exp(a - a_max).sum(dim=dim, keepdim=True)
        log_sum_exp = a_max + sum_exp.log()
        out = (1.0 / (order - 1.0)) * log_sum_exp.squeeze(dim)
    return reduce_tensor(out, reduction=reduction)


def relative_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    similarity: torch.Tensor | None = None,
    order: float = 1.0,
    reduction: str = "mean",
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    target = torch.clamp(target, min=eps)
    if similarity is not None:
        log_target = torch.clamp(target @ similarity, min=eps).log()
        input = input.exp() @ similarity
        input = torch.clamp(input, min=eps).log()
    else:
        log_target = target.log()
    if abs(order - 1.0) < 1e-9:
        out = (target * (log_target - input)).sum(dim=dim)
    else:
        a = order * log_target - (order - 1.0) * input
        a_max = a.amax(dim=dim, keepdim=True)
        sum_exp = (a - a_max).exp().sum(dim=dim, keepdim=True)
        log_sum = a_max + sum_exp.log()
        out = (1.0 / (order - 1.0)) * log_sum.squeeze(dim)
    return reduce_tensor(out, reduction=reduction)
