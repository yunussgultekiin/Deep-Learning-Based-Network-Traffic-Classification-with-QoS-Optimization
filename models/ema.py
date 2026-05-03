from __future__ import annotations
import copy
import torch
from torch import nn

class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup: dict[str, torch.Tensor] | None = None

    def update(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            if name in self.shadow:
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name] = param.detach().clone()

    def apply_to(self, model: nn.Module) -> None:
        self.backup = copy.deepcopy(model.state_dict())
        device = next(model.parameters()).device
        shadow = {k: v.to(device) for k, v in self.shadow.items()}
        model.load_state_dict(shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            return
        model.load_state_dict(self.backup, strict=True)
        self.backup = None
