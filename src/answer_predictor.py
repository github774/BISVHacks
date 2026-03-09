"""
Answer predictor: simple pairwise multiplication + learned weights + bias.

- out = (v1 * v2) * w + b
- v1, v2: (batch, 384), w, b: (384,)
- Returns: (batch, 384)
"""

import torch
import torch.nn as nn

DIM = 384


class AnswerPredictor(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        v1, v2: (batch, 384)
        Returns: (batch, 384)
        """
        out = (v1 * v2) * self.w + self.b
        return out
