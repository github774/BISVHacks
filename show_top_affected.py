#!/usr/bin/env python3
"""Show full descriptions of top affected agents."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from src.entry import PolicyImpactSimulator

# Load results
results_path = Path(__file__).resolve().parent / "data" / "simulation_results.json"
with open(results_path) as f:
    results = json.load(f)

# Generate same agents
print("Generating agents to match results...")
sim = PolicyImpactSimulator(n_agents=100, device="cpu")
sim.load()

print("\n" + "="*80)
print("TOP 10 MOST AFFECTED AGENTS (Full Descriptions)")
print("="*80)

top_ids = [x[0] for x in results['top_affected'][:10]]
for i, aid in enumerate(top_ids, 1):
    damage = results['initial_damage'][aid]
    desc = sim.network.archetypes[aid]['description']
    print(f"\n{i}. Agent {aid} (damage={damage:.4f}):")
    print(desc)
    print()
