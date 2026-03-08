"""
Answer predictor: 384×384 matrix from two 384D vectors, 3 residual blocks,
weighted sum by row, then linear.

- Build M = outer(v1, v2)  -> (batch, 384, 384)
- 3 blocks: M = M - (M * W + b) each
- Pool: weighted sum by row -> (batch, 384) using learned row_weights (384,)
- Final: linear(pooled) + bias -> (batch, 384)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 384


class AnswerPredictor(nn.Module):
    def __init__(self, dim: int = DIM, num_blocks: int = 3):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        # Blocks: M = M - (M * W + b); store as (num_blocks, dim, dim) for checkpoint compat
        self.block_W = nn.Parameter(torch.randn(num_blocks, dim, dim) * 0.02)
        self.block_b = nn.Parameter(torch.zeros(num_blocks, dim, dim))
        # Pool: weighted sum by row (one weight per row)
        self.row_weights = nn.Parameter(torch.ones(dim) / dim)
        # Final
        self.linear = nn.Linear(dim, dim)

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        v1, v2: (batch, 384)
        Returns: (batch, 384)
        """
        # Outer product: (batch, 384, 1) @ (batch, 1, 384) -> (batch, 384, 384)
        M = v1.unsqueeze(2) * v2.unsqueeze(1)
        for i in range(self.num_blocks):
            M = M - (M * self.block_W[i] + self.block_b[i])
        # Weighted sum by row: (batch, 384, 384), row_weights (384,) -> (batch, 384)
        # pooled[b, j] = sum_i row_weights[i] * M[b, i, j]
        pooled = torch.einsum("i,bij->bj", self.row_weights, M)
        out = self.linear(pooled)
        return out
