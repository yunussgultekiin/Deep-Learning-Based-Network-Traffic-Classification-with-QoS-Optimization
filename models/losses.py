from __future__ import annotations
from typing import Optional, Dict, Tuple
import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.long()

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.weight is not None:
            alpha_t = self.weight.gather(0, targets)
        else:
            alpha_t = 1.0

        loss = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

class ConfusionPenaltyLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,
        confused_pairs: Dict[Tuple[int, int], float],
        alpha: float = 0.3,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.confused_pairs = confused_pairs
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(logits, targets)

        probs = torch.softmax(logits, dim=1)
        penalty = torch.zeros(logits.size(0), device=logits.device)

        for (true_cls, wrong_cls), w in self.confused_pairs.items():
            mask = targets == true_cls
            if mask.any():
                penalty[mask] = penalty[mask] + w * probs[mask, wrong_cls]

        return base + self.alpha * penalty.mean()