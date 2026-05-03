from __future__ import annotations
import torch
from torch import nn

class FeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, dim))
        self.bias = nn.Parameter(torch.zeros(num_features, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, attn_dropout: float, ff_dropout: float, ff_mult: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.ln1(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x

class FTTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dim: int = 128,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        ff_mult: int = 4,
        emb_dropout: float = 0.1,
        cls_dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(input_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, attn_dropout, ff_dropout, ff_mult) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(cls_dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])
