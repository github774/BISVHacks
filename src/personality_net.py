"""
Neural network for personality vector prediction from Q&A pairs.

Architecture:
- Input: 10 questions (384-d) and 10 answers (384-d) on a grid
- Diagonal pairs (q1,a1), (q2,a2), ..., (q10,a10) -> Cantor-pair each dim -> scale to [0,10]
- Stack -> 10x384 matrix
- Pairwise multiply by learned weights + bias
- ReLU
- Linear (learned weight + bias)
- Pool to 1x384 (mean over rows)
- Final linear + bias
- L2 normalize -> personality vector

Loss: 1 - cosine_similarity(output, target)
where target = normalized sum of semantic adjective embeddings derived from answers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cantor import cantor_pair

CANTOR_SCALE = 1e6
CANTOR_DIVISOR = 200_000_200_000  # scale Cantor output to ~[0, 10]
DIM = 384


def _float_to_int(x: torch.Tensor) -> torch.Tensor:
    """Map floats in [-1,1] to integers in [0, 1e6] for Cantor pairing."""
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    return (x * CANTOR_SCALE).long().clamp(0, int(CANTOR_SCALE))


def cantor_pair_batch(q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Apply Cantor pairing element-wise to matching dimensions of q and a.
    q, a: (batch, dim) float tensors
    Returns: (batch, dim) float tensor, scaled to ~[0, 10]
    """
    q_int = _float_to_int(q)
    a_int = _float_to_int(a)
    # Cantor is not vectorizable in pure PyTorch easily; use numpy or loop
    device = q.device
    q_np = q_int.cpu().numpy()
    a_np = a_int.cpu().numpy()
    out = np.empty_like(q_np, dtype=np.float32)
    for i in range(q_np.shape[0]):
        for d in range(q_np.shape[1]):
            c = cantor_pair(int(q_np[i, d]), int(a_np[i, d]))
            out[i, d] = c / CANTOR_DIVISOR
    return torch.from_numpy(out).to(device)


def cantor_pair_batch_vectorized(q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Vectorized Cantor pairing using the formula: (a+b)*(a+b+1)//2 + b
    Uses float arithmetic for GPU compatibility (approximate for large ints).
    """
    q_int = _float_to_int(q).float()
    a_int = _float_to_int(a).float()
    s = q_int + a_int
    c = (s * (s + 1)) / 2 + a_int
    return c / CANTOR_DIVISOR


class PersonalityNet(nn.Module):
    def __init__(self, dim: int = DIM, num_pairs: int = 10):
        super().__init__()
        self.dim = dim
        self.num_pairs = num_pairs

        # Pairwise multiply by learned weights, add bias
        self.pairwise_weight = nn.Parameter(torch.randn(num_pairs, dim) * 0.02)
        self.pairwise_bias = nn.Parameter(torch.zeros(num_pairs, dim))

        # After ReLU: shared linear 384 -> 384
        self.linear = nn.Linear(dim, dim)

        # Pool to 1x384, then final
        self.final = nn.Linear(dim, dim)

    def forward(self, questions: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        """
        questions: (batch, 10, 384) - 10 question embeddings per sample
        answers: (batch, 10, 384) - 10 answer embeddings per sample
        Returns: (batch, 384) L2-normalized personality vector
        """
        batch = questions.shape[0]
        device = questions.device

        # Diagonal pairs: (q_i, a_i) for i in 0..9
        # Shape: (batch, 10, 384) each
        pairs = []
        for i in range(self.num_pairs):
            q_i = questions[:, i, :]  # (batch, 384)
            a_i = answers[:, i, :]   # (batch, 384)
            c = cantor_pair_batch_vectorized(q_i, a_i)  # (batch, 384)
            pairs.append(c)
        x = torch.stack(pairs, dim=1)  # (batch, 10, 384)

        # Pairwise multiply by learned weights, add biases
        x = x * self.pairwise_weight.unsqueeze(0) + self.pairwise_bias.unsqueeze(0)
        x = F.relu(x)

        # Linear 384 -> 384 (shared across rows)
        x = self.linear(x)  # (batch, 10, 384)

        # Transform to 1x384: mean pool over the 10 rows
        x = x.mean(dim=1)  # (batch, 384)

        # Final linear + bias
        x = self.final(x)
        return F.normalize(x, p=2, dim=-1)


def personality_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity loss. Minimize 1 - cos_sim so that pred aligns with target.
    pred: (batch, 384) - model output (already L2-normalized)
    target: (batch, 384) - normalized sum of adjective embeddings (L2-normalized)
    """
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    return (1 - cos_sim).mean()


def build_adjective_target(
    answer_texts: list[list[str]],
    adjectives: list[str],
    encoder,
    device: torch.device,
    derive_from_answers: bool = True,
) -> torch.Tensor:
    """
    Derive target personality vector from answers using semantic adjective embeddings.

    If derive_from_answers: weight each adjective by max similarity to any answer.
    Else: use uniform sum of adjective embeddings.

    answer_texts: list of (10 answer strings) per sample
    adjectives: e.g. ['adventurous', 'creative', 'thoughtful', 'ambitious', ...]
    encoder: sentence-transformers model
    """
    adj_emb = encoder.encode(adjectives, convert_to_tensor=True, device=device)
    adj_emb = F.normalize(adj_emb, p=2, dim=-1)  # (n_adj, 384)

    if not derive_from_answers:
        adj_avg = adj_emb.mean(dim=0, keepdim=True)
        target = F.normalize(adj_avg, p=2, dim=-1)
        return target.expand(len(answer_texts), -1)

    targets = []
    for answers in answer_texts:
        ans_emb = encoder.encode(answers, convert_to_tensor=True, device=device)
        ans_emb = F.normalize(ans_emb, p=2, dim=-1)  # (10, 384)
        # Similarity: adj_emb (n_adj, 384) @ ans_emb.T (384, 10) -> (n_adj, 10)
        sim = adj_emb @ ans_emb.T  # (n_adj, 10)
        weights = sim.max(dim=1).values  # (n_adj,) - max sim of each adj to any answer
        weights = F.softmax(weights, dim=0)  # normalize weights
        weighted_sum = (weights.unsqueeze(1) * adj_emb).sum(dim=0)
        target = F.normalize(weighted_sum.unsqueeze(0), p=2, dim=-1)
        targets.append(target)
    return torch.cat(targets, dim=0)
