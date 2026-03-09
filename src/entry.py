"""
Entry point: Load network, analyze policy semantically, estimate damage, and propagate.

This module:
1. Loads the marginalization network and archetype data
2. Semantically analyzes a policy text against archetype descriptions
3. Estimates damage to each node based on policy impact
4. Propagates damage through the network
5. Reports affected archetypes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import torch
import torch.nn.functional as F

from predict_policy_answer import _load_resources
from preprocessor import _get_mps_data, preprocess_batched_mps_preloaded

# Add parent directory to path to import network module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from network.network import MarginalizationNetwork

# Configuration constants
NUM_NODES = 1000  # Number of agents to generate
BENEFIT_TUNING = 1.0  # Multiplier for benefit in net impact calculation
DAMAGE_TUNING = 1.0 # Multiplier for damage in net impact calculation


class PolicyAnalyzer:
    """
    Semantic policy analyzer that estimates damage to archetypes based on
    policy text similarity to archetype descriptions.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize policy analyzer.
        
        Args:
            model_name: HuggingFace sentence-transformer model name
            data_dir: Directory containing data files (defaults to ../data)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = Path(data_dir)
        
        self.archetype_descriptions: list[dict] = []
        self.archetype_embeddings: Optional[np.ndarray] = None
        
    def load_archetype_descriptions(self) -> None:
        """Load archetype descriptions and their embeddings."""
        desc_path = self.data_dir / "archetype_descriptions.json"
        with open(desc_path, encoding="utf-8") as f:
            self.archetype_descriptions = json.load(f)
        
        # Extract embeddings
        embeddings = []
        for item in self.archetype_descriptions:
            embeddings.append(item["embedding"])
        self.archetype_embeddings = np.array(embeddings, dtype=np.float32)
        
        print(f"Loaded {len(self.archetype_descriptions)} archetype descriptions")
    
    
    def analyze_policy_with_avnehs_code(
        self,
        policy_text: str,
        archetypes: list,
        *,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Compute net impact = (benefit * BENEFIT_TUNING) - (damage * DAMAGE_TUNING) for each archetype.
        Loads encoder, model, SentimentHead, preprocessor data once; runs batched inference."""
        import predict_policy_answer as m

        _load_resources(self.data_dir, self.data_dir / "answer_predictor.pt")
        device = next(m._MODEL.parameters()).device
        sentiment_head = m._SENTIMENT_HEAD
        encoder = m._ENCODER
        model = m._MODEL

        desc_path = str(self.data_dir / "archetype_descriptions.json")
        pers_path = str(self.data_dir / "persona_vectors.json")
        desc_np, pers_np = _get_mps_data(desc_path, pers_path)
        desc_t = torch.from_numpy(desc_np).to(device)
        pers_t = torch.from_numpy(pers_np).to(device)

        descriptions = [a["description"] for a in archetypes]
        N = len(descriptions)

        desc_embs = encoder.encode(descriptions, convert_to_numpy=True, show_progress_bar=False)
        desc_embs = torch.from_numpy(desc_embs.astype("float32")).to(device)
        q_emb = encoder.encode(policy_text, convert_to_numpy=True)
        q_emb = torch.from_numpy(q_emb.astype("float32")).to(device)

        net_impacts = []
        for i in range(0, N, batch_size):
            batch = desc_embs[i : i + batch_size]
            b = batch.shape[0]
            persona = preprocess_batched_mps_preloaded(batch, desc_t, pers_t)
            q_batch = q_emb.unsqueeze(0).expand(b, -1)
            with torch.inference_mode():
                out = model(q_batch, persona)
                sent = sentiment_head(out)  # [bad, good] = [neg, pos]
            for j in range(b):
                damage = float(sent[j, 0].item())
                benefit = float(sent[j, 1].item())
                net_impacts.append((benefit * BENEFIT_TUNING) + (damage * DAMAGE_TUNING))
        return np.array(net_impacts)

    
    def analyze_policy(
        self,
        policy_text: str,
        impact_threshold: float = 0.5,
        damage_scale: float = 1.0,
        method: str = "avneh",
    ) -> np.ndarray:
        """
        Analyze policy text and estimate damage to each archetype.
        
        Args:
            policy_text: Policy description to analyze
            impact_threshold: Minimum similarity for impact (0-1)
            damage_scale: Multiplier for damage scores (0-1)
            method: 'similarity' (higher similarity = more damage) or
                    'negative' (analyze for negative impact)
        
        Returns:
            damage: (N,) array of damage scores per archetype in [0, 1]
        """
        if not policy_text.strip():
            raise ValueError("Policy text is empty")
        
        if self.archetype_embeddings is None:
            self.load_archetype_descriptions()
        
        # Encode policy text
        print(f"Encoding policy text ({len(policy_text)} chars)...")
        policy_embedding = self.model.encode(
            policy_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        # Compute cosine similarity to each archetype description
        # archetype_embeddings are already normalized from training
        similarities = self.archetype_embeddings @ policy_embedding
        similarities = np.clip(similarities, -1.0, 1.0)
        
        # Map similarity to damage
        if method == "similarity":
            # Higher similarity = more relevant = potentially more impacted
            # Scale from [-1, 1] to [0, 1]
            damage = (similarities + 1.0) / 2.0
        elif method == "negative":
            # Detect negative keywords, then scale by similarity
            negative_keywords = [
                "ban", "prohibit", "restrict", "cut", "reduce", "eliminate",
                "decrease", "remove", "deny", "reject", "exclude", "closure",
                "shutdown", "deportation", "enforcement", "penalty", "fine",
            ]
            policy_lower = policy_text.lower()
            is_negative = any(kw in policy_lower for kw in negative_keywords)
            
            if is_negative:
                # High similarity to negative policy = high damage
                damage = (similarities + 1.0) / 2.0
            else:
                # Positive policy: low or no damage
                damage = np.zeros_like(similarities)
        elif method == "avneh":
            damage = self.analyze_policy_with_avnehs_code(policy_text, self.archetype_descriptions)
            # Note: 'damage' variable now contains net_impact = (benefit * BENEFIT_TUNING) - (damage * DAMAGE_TUNING)
            # Negative values mean net harm, positive values mean net benefit

        else:
            raise ValueError(f"Unknown method: {method}")
        
        # For avneh method, damage is already net_impact, no need for threshold/scale
        if method != "avneh":
            # Apply threshold and scale
            damage = np.where(damage >= impact_threshold, damage, 0.0)
            damage = np.clip(damage * damage_scale, 0.0, 1.0)
        
        print(f"Policy analysis: {np.sum(damage > 0)} archetypes impacted (damage > 0)")
        return damage


class PolicyImpactSimulator:
    """
    End-to-end simulator: generate agents, build network, analyze policy, propagate damage.
    """
    
    def __init__(
        self,
        n_agents: int = 1000,
        data_dir: Optional[Path] = None,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """
        Initialize simulator.
        
        Args:
            n_agents: Number of agents to generate
            data_dir: Directory containing data files
            model_name: Sentence transformer model
            device: Device for encoding ('cpu', 'mps', 'cuda')
        """
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.n_agents = n_agents
        self.device = device
        
        # Initialize components: pull archetypes randomly from data/archetypes.json
        self.network = MarginalizationNetwork(
            n_agents=n_agents,
            use_generated=False,
            use_archetypes_json=True,
            device=device,
            archetypes_path=self.data_dir / "archetypes.json",
            persona_vectors_path=self.data_dir / "persona_vectors.json",
            archetype_descriptions_path=self.data_dir / "archetype_descriptions.json",
        )
        self.analyzer = PolicyAnalyzer(
            model_name=model_name,
            data_dir=data_dir,
        )
    
    def load(self) -> None:
        """Generate agents and build network."""
        print("Generating network...")
        self.network.load()
        print(f"Network generated: {self.network.n_nodes} nodes")
        
        # Create archetype descriptions from generated agents for policy analyzer
        print("Preparing archetype descriptions for policy analysis...")
        self.analyzer.archetype_descriptions = [
            {
                "archetype_id": arch["archetype_id"],
                "description": arch["description"],
                "embedding": self.network.agents[i].desc_emb.tolist(),
            }
            for i, arch in enumerate(self.network.archetypes)
        ]
        self.analyzer.archetype_embeddings = np.array(
            [agent.desc_emb for agent in self.network.agents],
            dtype=np.float32
        )
        print(f"Prepared {len(self.analyzer.archetype_descriptions)} archetype descriptions")
    
    def run_simulation(
        self,
        policy_text: str,
        impact_threshold: float = 0.3,
        damage_scale: float = 1.0,
        method: str = "similarity",
        cascade_steps: int = 3,
        accumulation: str = "replace",
        top_k: int = 20,
    ) -> dict:
        """
        Run full simulation: analyze policy, estimate damage, propagate.
        
        Args:
            policy_text: Policy to analyze
            impact_threshold: Minimum similarity for impact
            damage_scale: Multiplier for initial damage
            method: 'similarity' or 'negative'
            cascade_steps: Number of propagation steps
            accumulation: 'replace' or 'add' for cascade
            top_k: Number of top affected to return
        
        Returns:
            results: dict with initial_damage, cascade_history, top_affected
        """
        # Analyze policy and estimate initial damage
        print("\n" + "="*70)
        print("POLICY ANALYSIS")
        print("="*70)
        initial_damage = self.analyzer.analyze_policy(
            policy_text,
            impact_threshold=impact_threshold,
            damage_scale=damage_scale,
            method=method,
        )
        # Rescale to [-1, 1]: lowest -> -1, highest -> 1, linear in between
        lo, hi = np.min(initial_damage), np.max(initial_damage)
        if hi > lo:
            initial_damage = 2.0 * (initial_damage - lo) / (hi - lo) - 1.0
        else:
            initial_damage = np.zeros_like(initial_damage)  # all same -> 0

        # Show initial impact
        print(f"\nInitial net impact values (benefit*{BENEFIT_TUNING} - damage*{DAMAGE_TUNING}):")
        print(f"  {initial_damage.tolist()}")
        
        print(f"\nInitial impact statistics:")
        print(f"  Mean: {np.mean(initial_damage):.4f}")
        print(f"  Min: {np.min(initial_damage):.4f}")
        print(f"  Max: {np.max(initial_damage):.4f}")
        print(f"  Positive (net benefit): {np.sum(initial_damage > 0)}")
        print(f"  Negative (net harm): {np.sum(initial_damage < 0)}")
        
        # Propagate damage through network
        print("\n" + "="*70)
        print("DAMAGE PROPAGATION")
        print("="*70)
        print(f"Running cascade for {cascade_steps} steps (accumulation={accumulation})...")
        cascade_history = self.network.run_cascade(
            initial_damage,
            steps=cascade_steps,
            accumulation=accumulation,
        )
        
        # Report cascade progression
        print("\nCascade progression:")
        for step in range(cascade_steps + 1):
            damage_at_step = cascade_history[step]
            print(f"  Step {step}: mean={np.mean(damage_at_step):.4f}, "
                  f"max={np.max(damage_at_step):.4f}, "
                  f"damaged nodes={np.sum(damage_at_step > 0)}")
        
        # Get top affected at final step
        final_damage = cascade_history[-1]
        
        # For net impact, show both most harmed (lowest/most negative) and most benefited (highest/most positive)
        sorted_indices = np.argsort(final_damage)  # Ascending order
        most_harmed_indices = sorted_indices[:top_k]  # Lowest values = most net harm
        most_benefited_indices = sorted_indices[-top_k:][::-1]  # Highest values = most net benefit
        
        # Legacy top_affected for compatibility
        top_affected = self.network.most_affected(final_damage, top_k=top_k)
        
        # Get all affected with attributes for final step
        all_affected = self.network.all_affected_with_attributes(
            final_damage,
            damage=final_damage,
        )
        
        print("\n" + "="*70)
        print(f"TOP {top_k} MOST HARMED ARCHETYPES (Final Step - Most Negative Net Impact)")
        print("="*70)
        for rank, idx in enumerate(most_harmed_indices, 1):
            net_impact = final_damage[idx]
            desc = self.analyzer.archetype_descriptions[idx]["description"]
            desc_short = desc[:150] + "..." if len(desc) > 150 else desc
            print(f"\n{rank}. Archetype {idx} (net impact={net_impact:+.4f})")
            print(f"   {desc_short}")
        
        print("\n" + "="*70)
        print(f"TOP {top_k} MOST BENEFITED ARCHETYPES (Final Step - Most Positive Net Impact)")
        print("="*70)
        for rank, idx in enumerate(most_benefited_indices, 1):
            net_impact = final_damage[idx]
            desc = self.analyzer.archetype_descriptions[idx]["description"]
            desc_short = desc[:150] + "..." if len(desc) > 150 else desc
            print(f"\n{rank}. Archetype {idx} (net impact={net_impact:+.4f})")
            print(f"   {desc_short}")
        
        return {
            "policy_text": policy_text,
            "initial_damage": initial_damage,
            "cascade_history": cascade_history,
            "top_affected": top_affected,
            "all_affected": all_affected,
            "parameters": {
                "impact_threshold": impact_threshold,
                "damage_scale": damage_scale,
                "method": method,
                "cascade_steps": cascade_steps,
                "accumulation": accumulation,
            },
        }
    
    def save_results(
        self,
        results: dict,
        output_path: Optional[Path] = None,
    ) -> None:
        """Save simulation results to JSON."""
        if output_path is None:
            output_path = self.data_dir / "simulation_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable = {
            "policy_text": results["policy_text"],
            "initial_damage": results["initial_damage"].tolist(),
            "cascade_history": results["cascade_history"].tolist(),
            "top_affected": results["top_affected"],
            "all_affected": results["all_affected"],
            "parameters": results["parameters"],
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}")


def main():
    """Main entry point: load policy from file and run simulation."""
    # Paths
    data_dir = Path(__file__).resolve().parent.parent / "data"
    policy_path = data_dir / "policy.txt"
    
    # Load policy
    if not policy_path.exists():
        print(f"Error: Policy file not found: {policy_path}")
        print("Please create a policy.txt file in the data directory.")
        return
    
    with open(policy_path, encoding="utf-8") as f:
        policy_text = f.read().strip()
    
    if not policy_text:
        print("Error: policy.txt is empty. Please add policy text to analyze.")
        print("\nExample policy text:")
        print("-" * 70)
        print("The government will implement stricter immigration enforcement,")
        print("including increased workplace raids and expedited deportations.")
        print("This policy aims to reduce illegal immigration but may impact")
        print("communities with large immigrant populations.")
        print("-" * 70)
        return
    
    print(f"Policy loaded from: {policy_path}")
    print(f"Policy length: {len(policy_text)} characters")
    print(f"\nPolicy text preview:")
    print("-" * 70)
    print(policy_text[:500] + ("..." if len(policy_text) > 500 else ""))
    print("-" * 70)
    
    # Initialize simulator with generated agents
    print("\nInitializing policy impact simulator...")
    print(f"Configuration: NUM_NODES={NUM_NODES}, BENEFIT_TUNING={BENEFIT_TUNING}, DAMAGE_TUNING={DAMAGE_TUNING}")
    simulator = PolicyImpactSimulator(
        n_agents=NUM_NODES,
        data_dir=data_dir,
        model_name="all-MiniLM-L6-v2",
        device="cpu",  # Use 'mps' for Mac, 'cuda' for NVIDIA GPU, 'cpu' otherwise
    )
    
    # Generate agents and build network
    simulator.load()
    
    # Run simulation
    results = simulator.run_simulation(
        policy_text=policy_text,
        impact_threshold=0.0,  # No threshold - use raw damage scores
        damage_scale=0.1,
        method="avneh",  # Use predict_policy_answer for accurate damage estimation
        cascade_steps=0,  # No propagation - just show direct policy damage
        accumulation="replace",
        top_k=20,
    )
    
    # Save results
    simulator.save_results(results)


if __name__ == "__main__":
    main()
