"""
Marginalization spread network: simulates how marginalization propagates
through groups in a population. The logistic curve is applied only to similarity;
effect on B from A = A_damage * B_vulnerability * (influence_A * logistic(similarity(A,B))).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add src to path for genagents import
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from genagents import gen_agents, agent_enc, load_agents_from_archetypes_json, Agent


# ---------------------------------------------------------------------------
# 1. Generate or load data
# ---------------------------------------------------------------------------

def generate_agents_with_embeddings(n_agents: int, device: str = "cpu") -> list[Agent]:
    """Generate agents with descriptions, embeddings, and persona vectors."""
    agents = gen_agents(n_agents)
    agents = agent_enc(agents, device=device)
    return agents


def agents_to_archetypes(agents: list[Agent]) -> list[dict]:
    """Convert Agent objects to archetype-like dicts for compatibility."""
    archetypes = []
    for agent in agents:
        archetype = {
            "archetype_id": agent.id,
            "profile": agent.attrs,  # attrs already has string keys
            "description": agent.desc_str,
        }
        archetypes.append(archetype)
    return archetypes


def agents_to_persona_matrix(agents: list[Agent]) -> np.ndarray:
    """Extract persona vectors from agents into (N, D) matrix."""
    n = len(agents)
    d = len(agents[0].persona_v) if n > 0 else 0
    matrix = np.zeros((n, d), dtype=np.float64)
    for i, agent in enumerate(agents):
        matrix[i] = agent.persona_v
    return matrix


# ---------------------------------------------------------------------------
# 2. Vulnerability from profile
# ---------------------------------------------------------------------------

# Dimensions: 35 Health insurance, 42 Housing stability, 83 Legal vulnerability, 84 Exposure to discrimination, 90 Financial resilience
VULNERABILITY_DIMS = (35, 42, 83, 84, 90)
VULN_FACTOR = 1.0

# Categorical -> vulnerability score (higher = more vulnerable).
# Rich / resourced / stable -> low vuln; poor / exposed / unstable -> high vuln.
# Covers profile values across dimensions; same word can have different meaning per dimension (one map used for all).
VULN_VALUE_MAP = {
    # Highest vulnerability: no resources, exposed, unstable
    "none": 1.0,
    "very low": 1.0,
    "unaffordable": 0.95,
    "unstable": 0.9,
    "unsafe": 0.9,
    "unemployed": 0.95,
    "homeless": 1.0,
    "undocumented": 0.95,
    "downward": 0.85,
    "far": 0.75,
    "long": 0.7,
    # Low / limited / poor
    "low": 0.8,
    "limited": 0.75,
    "poor": 0.85,
    "small": 0.7,
    "rare": 0.7,
    "rarely": 0.7,
    # Moderate / middle
    "moderate": 0.5,
    "medium": 0.5,
    "occasional": 0.55,
    "basic": 0.6,
    "some": 0.55,
    # Resourced / stable (lower vulnerability)
    "public": 0.6,
    "affordable": 0.45,
    "stable": 0.35,
    "good": 0.35,
    "easy": 0.35,
    "high": 0.25,
    "strong": 0.25,
    "private": 0.25,
    "frequent": 0.4,
    "upward": 0.25,
    # Rich / wealthy / secure (low vulnerability — how can you be rich and vulnerable)
    "rich": 0.15,
    "wealthy": 0.15,
    "very high": 0.2,
    "excellent": 0.2,
    "full": 0.25,
    "secure": 0.2,
    "safe": 0.25,
    "abundant": 0.2,
    "leadership": 0.25,
    "owner": 0.3,
    "fluent": 0.35,
    "advanced": 0.3,
    "personal": 0.35,
    "yes": 0.45,
    "no": 0.5,
    # Demographics / neutral (context-dependent)
    "child": 0.65,
    "elder": 0.6,
    "senior": 0.5,
    "mid-career": 0.45,
    "young": 0.5,
    "married": 0.45,
    "single": 0.55,
    "widowed": 0.6,
    "domestic": 0.4,
    "citizen": 0.4,
    "permanent resident": 0.5,
    "member": 0.45,
    "dual earner": 0.4,
    "renter": 0.65,
    "suburban": 0.45,
    "urban": 0.5,
    "rural": 0.55,
    "growing": 0.4,
    "manual": 0.6,
    "part-time": 0.7,
    "gig": 0.75,
    "full-time": 0.45,
    "college": 0.45,
    "agriculture": 0.6,
    "manufacturing": 0.5,
    "english": 0.45,
    "majority": 0.45,
    "lgbtq": 0.55,
    "nonbinary": 0.5,
    "male": 0.45,
    "female": 0.5,
    "other": 0.5,
    "6+": 0.6,
    "4–5": 0.5,
    "4-5": 0.5,
    "<5": 0.55,
    "15+": 0.5,
    "multiple": 0.5,
    "mandarin": 0.5,
    "spanish": 0.5,
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
    # lo, hi = vuln.min(), vuln.max()
    # if hi > lo:
    #     vuln = (vuln - lo) / (hi - lo)
    return np.clip(vuln * VULN_FACTOR, 0.0, 1.0)


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
# 4. Similarity from persona_vectors (cosine, then scale to [0,1])
# ---------------------------------------------------------------------------

# Scale so that raw cosine 0.999995 -> 0 and 1.0 -> 1 (spreads the narrow band)
COS_SIM_SCALE_MIN = 0.999995
COS_SIM_SCALE_MAX = 1.0


def scale_similarity(raw_cos: np.ndarray) -> np.ndarray:
    """Linearly scale raw cosine from [COS_SIM_SCALE_MIN, COS_SIM_SCALE_MAX] to [0, 1]. Clip outside."""
    scaled = (raw_cos - COS_SIM_SCALE_MIN) / (COS_SIM_SCALE_MAX - COS_SIM_SCALE_MIN)
    return np.clip(scaled, 0.0, 1.0)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """(N, D) -> (N, N) raw cosine similarity in [-1, 1]."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = vectors / norms
    sim = unit @ unit.T
    return np.clip(sim, -1.0, 1.0)


def similarity_matrix_scaled(vectors: np.ndarray) -> np.ndarray:
    """(N, D) -> (N, N) cosine similarity scaled so 0.999995 -> 0, 1 -> 1."""
    raw = cosine_similarity_matrix(vectors)
    return scale_similarity(raw)


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
    damage (N,), vulnerability (N,), weight (N,N). Damage and effect are in [-1, 1].
    Weight already contains scaled similarity. Damage is clipped to [-1, 1] before use.
    Option A (aggregate='sum_then_logistic'): effect[B] = clip(sum_A damage[A]*vuln[B]*weight[A,B], -1, 1)
    Option B (aggregate='per_neighbor'): per = clip(raw, -1, 1), then combine and clip to [-1, 1].
    """
    damage = np.clip(damage, -1.0, 1.0)
    raw = damage[:, np.newaxis] * vulnerability[np.newaxis, :] * weight
    if aggregate == "sum_then_logistic":
        total_raw = np.sum(raw, axis=0)
        return np.clip(total_raw, -1.0, 1.0)
    per = np.clip(raw, -1.0, 1.0)
    combined = 1.0 - np.prod(1.0 - per, axis=0)
    return np.clip(combined, -1.0, 1.0)


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
    Run cascading propagation for `steps` steps. Damage and effect in [-1, 1].
    accumulation: 'replace' -> damage = effect each step; 'add' -> damage += effect.
    Returns (steps+1, N): damage at step 0, 1, ..., steps.
    """
    n = initial_damage.size
    history = np.zeros((steps + 1, n), dtype=np.float64)
    damage = np.clip(np.array(initial_damage, dtype=np.float64), -1.0, 1.0)
    history[0] = damage
    for t in range(1, steps + 1):
        effect = compute_effects(damage, vulnerability, weight)
        if accumulation == "replace":
            damage = effect
        else:
            damage = np.clip(damage + effect, -1.0, 1.0)
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


def all_affected_with_attributes(
    effect: np.ndarray,
    vulnerability: np.ndarray,
    influence: np.ndarray,
    weight: np.ndarray,
    damage: np.ndarray,
    archetype_ids: Optional[list[int]] = None,
) -> list[dict]:
    """
    Return list of dicts with per-node: archetype_id, effect, vulnerability,
    influence (node's own edge-weight scalar), incoming_weight (from other damaged nodes to this node; self-loops excluded).
    """
    n = len(effect)
    if archetype_ids is None:
        archetype_ids = list(range(n))
    # incoming_weight[B] = sum over A != B of (damage[A] * weight[A, B])
    weight_no_self = weight.copy()
    weight_no_self[np.arange(n), np.arange(n)] = 0.0
    incoming_weight = (damage @ weight_no_self) if damage.ndim == 1 else (damage @ weight_no_self).flatten()
    return [
        {
            "archetype_id": archetype_ids[i],
            "effect": float(effect[i]),
            "vulnerability": float(vulnerability[i]),
            "influence": float(influence[i]),
            "incoming_weight": float(incoming_weight[i]),
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# MarginalizationNetwork: single entry point
# ---------------------------------------------------------------------------

class MarginalizationNetwork:
    """
    Network of archetypes: generate or load data, precompute vulnerability, influence,
    similarity, and edge weights; then run one-step or cascading effect.
    """

    def __init__(
        self,
        n_agents: Optional[int] = None,
        use_generated: bool = True,
        use_archetypes_json: bool = False,
        device: str = "cpu",
        archetypes_path: Optional[str | Path] = None,
        persona_vectors_path: Optional[str | Path] = None,
        archetype_descriptions_path: Optional[str | Path] = None,
        similarity_threshold: Optional[float] = None,
        logistic_k: float = 1.0,
        logistic_x0: float = 0.5,
    ):
        """
        Initialize network.
        
        Args:
            n_agents: Number of agents to generate (if use_generated=True)
            use_generated: If True, generate agents; if False, load from JSON files
            device: Device for encoding ('cpu', 'mps', 'cuda')
            archetypes_path: Path to archetypes.json (only if use_generated=False)
            persona_vectors_path: Path to persona_vectors.json (only if use_generated=False)
            similarity_threshold: Threshold for similarity matrix
            logistic_k: Logistic curve steepness
            logistic_x0: Logistic curve midpoint
        """
        self.n_agents = n_agents
        self.use_generated = use_generated
        self.use_archetypes_json = use_archetypes_json
        self.device = device
        self.archetypes_path = Path(archetypes_path) if archetypes_path else None
        self.persona_vectors_path = Path(persona_vectors_path) if persona_vectors_path else None
        self.archetype_descriptions_path = Path(archetype_descriptions_path) if archetype_descriptions_path else None
        self.similarity_threshold = similarity_threshold
        self.logistic_k = logistic_k
        self.logistic_x0 = logistic_x0

        self.agents: list[Agent] = []
        self.archetypes: list[dict] = []
        self.n_nodes = 0

        self._vulnerability: Optional[np.ndarray] = None
        self._influence: Optional[np.ndarray] = None
        self._similarity: Optional[np.ndarray] = None
        self._weight: Optional[np.ndarray] = None
        self._vectors: Optional[np.ndarray] = None

    def load(self) -> None:
        """Generate or load data and build vulnerability, influence, similarity, weights."""
        if self.use_archetypes_json:
            # Load agents by randomly sampling from archetypes.json
            if self.n_agents is None:
                raise ValueError("n_agents must be specified when use_archetypes_json=True")
            data_dir = self.archetypes_path.parent if self.archetypes_path else ROOT / "data"
            arch_path = self.archetypes_path or data_dir / "archetypes.json"
            desc_path = self.archetype_descriptions_path or data_dir / "archetype_descriptions.json"
            pers_path = self.persona_vectors_path or data_dir / "persona_vectors.json"
            print(f"Loading {self.n_agents} agents from archetypes.json (random sample)...")
            self.agents = load_agents_from_archetypes_json(
                self.n_agents, arch_path, desc_path, pers_path,
            )
            self.archetypes = agents_to_archetypes(self.agents)
            self._vectors = agents_to_persona_matrix(self.agents)
        elif self.use_generated:
            # Generate agents
            if self.n_agents is None:
                raise ValueError("n_agents must be specified when use_generated=True")
            print(f"Generating {self.n_agents} agents...")
            self.agents = generate_agents_with_embeddings(self.n_agents, device=self.device)
            self.archetypes = agents_to_archetypes(self.agents)
            self._vectors = agents_to_persona_matrix(self.agents)
        else:
            # Load from JSON files (legacy mode)
            if self.archetypes_path is None or self.persona_vectors_path is None:
                raise ValueError("archetypes_path and persona_vectors_path required when use_generated=False")
            # Note: load_archetypes and load_persona_vectors functions removed
            # This is legacy mode and not the primary use case
            raise NotImplementedError("Legacy JSON loading removed. Use use_generated=True.")
        
        self.n_nodes = len(self.archetypes)

        self._vulnerability = compute_vulnerability(self.archetypes)
        self._influence = compute_influence(self.archetypes)
        # Scaled so raw cosine 0.999995 -> 0, 1.0 -> 1
        self._similarity = similarity_matrix_scaled(self._vectors)
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

    @property
    def vectors(self) -> np.ndarray:
        """Persona vectors (N, D). Loads data if needed."""
        if self._vectors is None:
            self.load()
        return self._vectors

    def cosine_similarity_from_node(self, node_index: int) -> np.ndarray:
        """Scaled similarity from node_index to every other node (0.999995->0, 1->1), in [0, 1]."""
        raw = cosine_similarity_matrix(self.vectors)
        row = raw[node_index]
        return scale_similarity(row)

    def least_similar_pair(self) -> tuple[int, int, float]:
        """Return (i, j, scaled_similarity) for the pair of distinct nodes with minimum (scaled) similarity."""
        raw = cosine_similarity_matrix(self.vectors)
        scaled = scale_similarity(raw)
        n = scaled.shape[0]
        scaled_no_diag = scaled.copy()
        scaled_no_diag[np.arange(n), np.arange(n)] = 2.0
        idx = np.argmin(scaled_no_diag)
        i, j = np.unravel_index(idx, scaled_no_diag.shape)
        return (int(i), int(j), float(scaled[i, j]))

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

    def all_affected_with_attributes(
        self,
        effect: np.ndarray,
        damage: np.ndarray,
    ) -> list[dict]:
        """Return per-node effect, vulnerability, influence, incoming_weight."""
        return all_affected_with_attributes(
            effect,
            self.vulnerability,
            self.influence,
            self.weight,
            damage,
        )


# ---------------------------------------------------------------------------
# Example / CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # Generate agents and build network
    print("Generating agents and building network...")
    net = MarginalizationNetwork(
        n_agents=50,  # Generate 50 agents
        use_generated=True,
        device="cpu",  # Use 'mps', 'cuda', or 'cpu'
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
