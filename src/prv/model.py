from __future__ import annotations

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        raise NotImplementedError("TODO: implement sinusoidal positional encoding")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: add position encoding to input tensor")


class OsuPatternTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        n_level1: int = 4,
        n_level2: int = 6,
    ):
        super().__init__()
        raise NotImplementedError("TODO: build encoder and classification heads")

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        raise NotImplementedError("TODO: run forward pass and return level1/level2 logits")
