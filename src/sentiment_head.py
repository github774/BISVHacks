"""SentimentHead: maps 384D embedding -> Good/Bad sentiment (2-class)."""

import torch
import torch.nn as nn


class SentimentHead(nn.Module):
    """Input: (B, 384) embedding. Output: (B, 2) raw [neg, pos] matching VADER (Bad, Good)."""

    def __init__(self, emb_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
