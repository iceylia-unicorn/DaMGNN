from __future__ import annotations

import torch
import torch.nn.functional as F


def contrast_loss(anchor: torch.Tensor, positive: torch.Tensor, *_: object, temperature: float = 0.2) -> torch.Tensor:
    if anchor.ndim != 2 or positive.ndim != 2 or anchor.shape != positive.shape:
        raise ValueError("anchor and positive must be equally shaped [num_nodes, hidden_dim] tensors")
    if anchor.shape[0] < 2:
        return anchor.new_zeros(())
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    labels = torch.arange(a.shape[0], device=a.device)
    logits = a @ p.T / temperature
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
