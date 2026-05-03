from typing import Iterable
import torch
from torch import nn
from .ft_transformer import FTTransformer

class DNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: Iterable[int],
        dropout: float,
        batch_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def build_model(
    arch: str,
    input_dim: int,
    num_classes: int,
    cfg,
) -> nn.Module:
    if arch == "ft_transformer":
        return FTTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            dim=cfg.ft_dim,
            depth=cfg.ft_depth,
            heads=cfg.ft_heads,
            attn_dropout=cfg.ft_attn_dropout,
            ff_dropout=cfg.ft_ff_dropout,
            ff_mult=cfg.ft_ff_mult,
            emb_dropout=cfg.ft_emb_dropout,
            cls_dropout=cfg.ft_cls_dropout,
        )
    if arch == "dnn":
        return DNNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_sizes=[512, 256, 128],
            dropout=0.3,
            batch_norm=True,
        )
    raise ValueError(f"Unknown architecture: {arch!r}. Options: 'ft_transformer', 'dnn'")
