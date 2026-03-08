"""
Marginalization spread network: simulates how marginalization propagates
through groups in a population. The logistic curve is applied only to similarity;
effect on B from A = A_damage * B_vulnerability * (influence_A * logistic(similarity(A,B))).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_archetypes(path: str | Path) -> tuple[list[dict], dict[str, str]]:
    """Load archetypes.json; return list of archetype dicts and archetype_keys."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["archetypes"], data["archetype_keys"]


def load_persona_vectors(path: str | Path) -> np.ndarray:
    """Load persona_vectors.json; return (N, D) array indexed by archetype_id."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    # Assume archetype_id in 0..N-1; build dense matrix
    n = len(items)
    d = len(items[0]["persona_vector"])
    matrix = np.zeros((n, d), dtype=np.float64)
    for item in items:
        i = item["archetype_id"]
        matrix[i] = item["persona_vector"]
    return matrix


# ---------------------------------------------------------------------------
# 2. Vulnerability from profile
# ---------------------------------------------------------------------------

# Dimensions: 35 Health insurance, 42 Housing stability, 83 Legal vulnerability, 84 Exposure to discrimination, 90 Financial resilience
VULNERABILITY_DIMS = (35, 42, 83, 84, 90)

# Categorical -> vulnerability score (higher = more vulnerable). Used for vulnerability-increasing values.
VULN_VALUE_MAP = {
    "none": 1.0,
    "very low": 1.0,
    "low": 0.8,
    "limited": 0.7,
    "unaffordable": 0.9,
    "unstable": 0.9,
    "unsafe": 0.9,
    "high": 0.2,  # e.g. exposure to discrimination -> high vuln
    "strong": 0.2,
    "moderate": 0.5,
    "medium": 0.5,
    "basic": 0.6,
    "public": 0.6,
    "private": 0.3,
    "affordable": 0.4,
    "stable": 0.3,
    "good": 0.3,
    "easy": 0.3,
    "frequent": 0.4,
    "occasional": 0.5,
    "downward": 0.8,
    "upward": 0.2,
}
DEFAULT_VULN = 0.5


def vulnerability_score(value: str) -> float:
    return VULN_VALUE_MAP.get(value.lower() if isinstance(value, str) else str(value), DEFAULT_VULN)


def compute_vulnerability(archetypes: list[dict]) -> np.ndarray:
    """Per-node vulnerability in [0, 1]. Higher = more vulnerable."""
    n = len(archetypes)
    vuln = np.zeros(n, dtype=np.float64)
    for i, arch in enumerate(archetypes):
        profile = arch.get("profile") or {}
        scores = []
        for dim in VULNERABILITY_DIMS:
            key = str(dim)
            if key in profile:
                scores.append(vulnerability_score(profile[key]))
        vuln[i] = np.mean(scores) if scores else DEFAULT_VULN
    # Normalize to [0, 1] across nodes (optional; already in ~[0,1])
    lo, hi = vuln.min(), vuln.max()
    if hi > lo:
        vuln = (vuln - lo) / (hi - lo)
    return np.clip(vuln, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. Influence (dim 101) per node
# ---------------------------------------------------------------------------

INFLUENCE_MAP = {"high": 1.0, "moderate": 0.5, "low": 0.2}
DEFAULT_INFLUENCE = 0.5
INFLUENCE_DIM = 101


def compute_influence(archetypes: list[dict]) -> np.ndarray:
    """Per-node influence scalar; used as multiplier for edge weight from A."""
    n = len(archetypes)
    inf = np.zeros(n, dtype=np.float64)
    for i, arch in enumerate(archetypes):
        profile = arch.get("profile") or {}
        val = profile.get(str(INFLUENCE_DIM), "moderate")
        inf[i] = INFLUENCE_MAP.get(
            val.lower() if isinstance(val, str) else str(val), DEFAULT_INFLUENCE
        )
    return inf


# ---------------------------------------------------------------------------
# 4. Similarity from persona_vectors (cosine, then shift to [0,1])
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """(N, D) -> (N, N) cosine similarity. Then shift to [0, 1]: (sim + 1) / 2."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = vectors / norms
    sim = unit @ unit.T
    sim = np.clip(sim, -1.0, 1.0)
    return (sim + 1.0) / 2.0


# ---------------------------------------------------------------------------
# 5. Edge weight A -> B = influence_A * similarity(A, B)
# ---------------------------------------------------------------------------

def edge_weights(influence: np.ndarray, similarity: np.ndarray) -> np.ndarray:
    """(N,) influence, (N,N) similarity -> (N,N) weight[A,B] = influence[A] * similarity[A,B]."""
    return influence[:, np.newaxis] * similarity


# ---------------------------------------------------------------------------
# 6. Logistic (applied only to similarity when building weights)
# ---------------------------------------------------------------------------

def logistic(x: np.ndarray, k: float = 1.0, x0: float = 0.5) -> np.ndarray:
    """Sigmoid: 1 / (1 + exp(-k * (x - x0))). Used only to transform similarity."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


# ---------------------------------------------------------------------------
# 7. Effect: raw = damage[A] * vulnerability[B] * weight(A->B); no logistic
# ---------------------------------------------------------------------------

def compute_effects(
    damage: np.ndarray,
    vulnerability: np.ndarray,
    weight: np.ndarray,
    aggregate: str = "sum_then_logistic",
) -> np.ndarray:
    """
    damage (N,), vulnerability (N,), weight (N,N).
    Weight already contains logistic(similarity) in its definition.
    Option A (aggregate='sum_then_logistic'): effect[B] = clip(sum_A damage[A]*vuln[B]*weight[A,B], 0, 1)
    Option B (aggregate='per_neighbor'): effect[B] = 1 - prod_A (1 - clip(raw, 0, 1))
    """
    raw = damage[:, np.newaxis] * vulnerability[np.newaxis, :] * weight
    if aggregate == "sum_then_logistic":
        total_raw = np.sum(raw, axis=0)
        return np.clip(total_raw, 0.0, 1.0)
    per = np.clip(raw, 0.0, 1.0)
    return 1.0 - np.prod(1.0 - per, axis=0)


# ---------------------------------------------------------------------------
# 8. Cascading propagation
# ---------------------------------------------------------------------------

def propagate(
    initial_damage: np.ndarray,
    vulnerability: np.ndarray,
    weight: np.ndarray,
    steps: int = 3,
    accumulation: str = "replace",
) -> np.ndarray:
    """
    Run cascading propagation for `steps` steps.
    accumulation: 'replace' -> damage = effect each step; 'add' -> damage += effect.
    Returns (steps+1, N): damage at step 0, 1, ..., steps.
    """
    n = initial_damage.size
    history = np.zeros((steps + 1, n), dtype=np.float64)
    damage = np.array(initial_damage, dtype=np.float64)
    history[0] = damage
    for t in range(1, steps + 1):
        effect = compute_effects(damage, vulnerability, weight)
        if accumulation == "replace":
            damage = effect
        else:
            damage = np.clip(damage + effect, 0.0, 1.0)
        history[t] = damage
    return history


# ---------------------------------------------------------------------------
# 9. Output: per-node marginalization, most affected or all
# ---------------------------------------------------------------------------

def most_affected(
    effect: np.ndarray,
    top_k: int = 20,
    archetype_ids: Optional[list[int]] = None,
) -> list[tuple[int, float]]:
    """Return list of (archetype_id, effect) for top_k most affected nodes."""
    if archetype_ids is None:
        archetype_ids = list(range(len(effect)))
    indexed = [(archetype_ids[i], float(effect[i])) for i in range(len(effect))]
    indexed.sort(key=lambda x: -x[1])
    return indexed[:top_k]

def all_affected(
    effect: np.ndarray,
    archetype_ids: Optional[list[int]] = None,
) -> list[tuple[int, float]]:
    """Return list of (archetype_id, effect) for all affected nodes."""
    if archetype_ids is None:
        archetype_ids = list(range(len(effect)))
    indexed = [(archetype_ids[i], float(effect[i])) for i in range(len(effect))]
    return indexed


# ---------------------------------------------------------------------------
# MarginalizationNetwork: single entry point
# ---------------------------------------------------------------------------

class MarginalizationNetwork:
    """
    Network of archetypes: load data, precompute vulnerability, influence,
    similarity, and edge weights; then run one-step or cascading effect.
    """

    def __init__(
        self,
        archetypes_path: str | Path = "archetypes.json",
        persona_vectors_path: str | Path = "persona_vectors.json",
        similarity_threshold: Optional[float] = None,
        logistic_k: float = 1.0,
        logistic_x0: float = 0.5,
        max_archetypes: Optional[int] = None,
    ):
        self.archetypes_path = Path(archetypes_path)
        self.persona_vectors_path = Path(persona_vectors_path)
        self.similarity_threshold = similarity_threshold
        self.logistic_k = logistic_k
        self.logistic_x0 = logistic_x0
        self.max_archetypes = max_archetypes

        self.archetypes: list[dict] = []
        self.archetype_keys: dict[str, str] = {}
        self.n_nodes = 0

        self._vulnerability: Optional[np.ndarray] = None
        self._influence: Optional[np.ndarray] = None
        self._similarity: Optional[np.ndarray] = None
        self._weight: Optional[np.ndarray] = None
        self._vectors: Optional[np.ndarray] = None

    def load(self) -> None:
        """Load JSON and build vulnerability, influence, similarity, weights."""
        self.archetypes, self.archetype_keys = load_archetypes(self.archetypes_path)
        self._vectors = load_persona_vectors(self.persona_vectors_path)
        if self.max_archetypes is not None:
            n = min(self.max_archetypes, len(self.archetypes), len(self._vectors))
            self.archetypes = self.archetypes[:n]
            self._vectors = self._vectors[:n]
        self.n_nodes = len(self.archetypes)

        self._vulnerability = compute_vulnerability(self.archetypes)
        self._influence = compute_influence(self.archetypes)
        self._similarity = cosine_similarity_matrix(self._vectors)
        # Apply logistic only to similarity (not to final effect)
        self._similarity = logistic(
            self._similarity, k=self.logistic_k, x0=self.logistic_x0
        )
        if self.similarity_threshold is not None:
            self._similarity = np.where(
                self._similarity >= self.similarity_threshold, self._similarity, 0.0
            )
        self._weight = edge_weights(self._influence, self._similarity)

    @property
    def vulnerability(self) -> np.ndarray:
        if self._vulnerability is None:
            self.load()
        return self._vulnerability

    @property
    def influence(self) -> np.ndarray:
        if self._influence is None:
            self.load()
        return self._influence

    @property
    def similarity(self) -> np.ndarray:
        if self._similarity is None:
            self.load()
        return self._similarity

    @property
    def weight(self) -> np.ndarray:
        if self._weight is None:
            self.load()
        return self._weight

    def run_one_step(
        self,
        damage: np.ndarray,
        aggregate: str = "sum_then_logistic",
    ) -> np.ndarray:
        """Single-step effect from current damage. Returns effect per node."""
        return compute_effects(
            damage,
            self.vulnerability,
            self.weight,
            aggregate=aggregate,
        )

    def run_cascade(
        self,
        initial_damage: np.ndarray,
        steps: int = 3,
        accumulation: str = "replace",
    ) -> np.ndarray:
        """(steps+1, N) damage over time."""
        return propagate(
            initial_damage,
            self.vulnerability,
            self.weight,
            steps=steps,
            accumulation=accumulation,
        )

    def most_affected(
        self,
        effect: np.ndarray,
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        return most_affected(effect, top_k=top_k)
    
    def all_affected(
        self,
        effect: np.ndarray,
    ) -> list[tuple[int, float]]:
        return all_affected(effect)


# ---------------------------------------------------------------------------
# Example / CLI
# ---------------------------------------------------------------------------

def main() -> None:
    base = Path(__file__).resolve().parent
    net = MarginalizationNetwork(
        archetypes_path=base / "archetypes.json",
        persona_vectors_path=base / "persona_vectors.json",
        logistic_k=1.0,
        logistic_x0=0.5,
    )
    net.load()
    print(f"Nodes: {net.n_nodes}")

    # Single node initially marginalized (e.g. one business, id=0)
    damage = np.zeros(net.n_nodes, dtype=np.float64)
    damage[0] = 1.0

    effect = net.run_one_step(damage)
    top = net.most_affected(effect, top_k=10)
    print("One-step effect from node 0; top 10 affected:")
    for aid, val in top:
        print(f"  archetype_id={aid}  effect={val:.4f}")

    # Cascading
    history = net.run_cascade(damage, steps=3, accumulation="replace")
    print("\nCascading (replace), final step top 5:")
    for aid, val in net.most_affected(history[-1], top_k=5):
        print(f"  archetype_id={aid}  damage={val:.4f}")


# if __name__ == "__main__":
#     main()
