from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dim_feedforward: int = 256):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=nhead, batch_first=True, dropout=0.0)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class OsuPatternTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 11,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 8,
    ):
        super().__init__()
        self.emb = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.emb(x)
        x = self.pos(x)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        return self.fc(x)
