"""
Preprocessor: maps a description embedding -> weighted blend of persona vectors.

Steps:
1. Cosine similarity between input embedding and each description embedding
2. Keep top 5 similarities, zero the rest
3. Round to 0.0001
4. Multiply by 10000
5. Softmax
6. Weighted sum of persona vectors
7. Return 384D vector
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def cosine_similarity(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """Cosine similarity between vectors. a, b can be 1D or 2D."""
    a_norm = a / (np.linalg.norm(a, axis=axis, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=axis, keepdims=True) + 1e-9)
    return np.sum(a_norm * b_norm, axis=axis)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.array(x, dtype=np.float64)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-9)


_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def preprocess(
    embedding: np.ndarray | list[float],
    descriptions_path: str | Path = "data/archetype_descriptions.json",
    persona_vectors_path: str | Path = "data/persona_vectors.json",
    top_k: int = 10,
    round_to: float = 0.0001,
    scale: float = 10000,
    verbose: bool = False,
) -> tuple[np.ndarray, Optional[dict]]:
    """
    Map a 384D description embedding to a 384D persona blend.

    Returns:
        output: 384D vector (weighted blend of persona vectors)
        debug: if verbose, dict with all intermediate steps for inspection
    """
    embedding = np.array(embedding, dtype=np.float64).flatten()
    assert embedding.shape == (384,), f"Expected 384D, got {embedding.shape}"

    cache_key = (str(Path(descriptions_path)), str(Path(persona_vectors_path)))
    if cache_key not in _CACHE:
        with open(Path(descriptions_path)) as f:
            descriptions = json.load(f)
        with open(Path(persona_vectors_path)) as f:
            personas = json.load(f)
        desc_embeddings = np.array([d["embedding"] for d in descriptions], dtype=np.float64)
        persona_vectors = np.array([p["persona_vector"] for p in personas], dtype=np.float64)
        _CACHE[cache_key] = (desc_embeddings, persona_vectors)
    desc_embeddings, persona_vectors = _CACHE[cache_key]

    debug = {} if verbose else None

    # Step 1: Cosine similarity with every description embedding
    similarities = cosine_similarity(
        embedding[np.newaxis, :], desc_embeddings, axis=1
    ).flatten()
    if verbose:
        debug["step1_similarities"] = similarities.copy()

    # Step 2: Keep top 5, zero the rest
    top_indices = np.argsort(similarities)[-top_k:]
    weight_vector = np.zeros_like(similarities)
    weight_vector[top_indices] = similarities[top_indices]
    if verbose:
        debug["step2_top_indices"] = top_indices.tolist()
        debug["step2_weight_vector"] = weight_vector.copy()

    # Step 3: Round to nearest round_to
    weight_vector = np.round(weight_vector / round_to) * round_to
    if verbose:
        debug["step3_rounded"] = weight_vector.copy()

    # Step 4: Multiply by scale
    scaled = weight_vector * scale
    if verbose:
        debug["step4_scaled"] = scaled.copy()

    # Step 5: Softmax
    probs = softmax(scaled)
    if verbose:
        debug["step5_probs"] = probs.copy()

    # Step 6: Weighted sum of persona vectors
    output = np.sum(probs[:, np.newaxis] * persona_vectors, axis=0)
    if verbose:
        debug["step6_output"] = output.copy()

    return output.astype(np.float32), debug


def _get_mps_data(
    descriptions_path: str | Path = "data/archetype_descriptions.json",
    persona_vectors_path: str | Path = "data/persona_vectors.json",
) -> tuple[np.ndarray, np.ndarray]:
    """Load desc_embeddings, persona_vectors (float32). Reuses preprocess cache."""
    cache_key = (str(Path(descriptions_path)), str(Path(persona_vectors_path)))
    if cache_key not in _CACHE:
        with open(Path(descriptions_path)) as f:
            descriptions = json.load(f)
        with open(Path(persona_vectors_path)) as f:
            personas = json.load(f)
        desc = np.array([d["embedding"] for d in descriptions], dtype=np.float64)
        pers = np.array([p["persona_vector"] for p in personas], dtype=np.float64)
        _CACHE[cache_key] = (desc, pers)
    desc, pers = _CACHE[cache_key]
    return desc.astype(np.float32), pers.astype(np.float32)


def preprocess_batched_mps(
    embeddings: np.ndarray,
    descriptions_path: str | Path = "data/archetype_descriptions.json",
    persona_vectors_path: str | Path = "data/persona_vectors.json",
    top_k: int = 10,
    round_to: float = 0.0001,
    scale: float = 10000,
) -> np.ndarray:
    """
    Batched MPS preprocess. embeddings: (B, 384). Returns (B, 384).
    Uses PyTorch on MPS (Apple Silicon) with batching, inference_mode, no grad.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for preprocess_batched_mps")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    desc_np, pers_np = _get_mps_data(descriptions_path, persona_vectors_path)

    q = torch.from_numpy(embeddings.astype(np.float32)).to(device)
    desc = torch.from_numpy(desc_np).to(device)
    pers = torch.from_numpy(pers_np).to(device)

    with torch.inference_mode():
        # Step 1: Cosine similarity (B, 384) @ (384, N) -> (B, N)
        q_norm = F.normalize(q, p=2, dim=-1)
        desc_norm = F.normalize(desc, p=2, dim=-1)
        sim = q_norm @ desc_norm.T  # (B, N)

        # Step 2: Top-k only
        top_vals, top_idx = torch.topk(sim, k=top_k, dim=-1)
        weights = torch.zeros_like(sim, device=device, dtype=sim.dtype)
        weights.scatter_(-1, top_idx, top_vals)

        # Step 3: Round
        weights = torch.round(weights / round_to) * round_to

        # Step 4: Scale
        scaled = weights * scale

        # Step 5: Softmax (subtract max for stability)
        scaled = scaled - scaled.max(dim=-1, keepdim=True).values
        probs = F.softmax(scaled, dim=-1)

        # Step 6: Weighted sum (B, N) @ (N, 384) -> (B, 384)
        out = probs @ pers

    return out.cpu().numpy().astype(np.float32)
