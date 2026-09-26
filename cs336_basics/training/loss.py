import torch


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, dim: int = -1):
    logits_per_pos = torch.gather(logits, dim=dim, index=targets.unsqueeze(dim))

    maxes = torch.max(logits, dim=dim, keepdim=True)[0]
    assert logits_per_pos.shape == maxes.shape, "Subtraction is not elementwise, broadcasting may happen, leading to possibly larger loss numbers"
    exponentiated = torch.exp(logits - maxes)
    expo_sum = torch.sum(exponentiated, dim=dim, keepdim=True)

    total_loss = torch.log(expo_sum) - (logits_per_pos - maxes)
    return torch.mean(total_loss)