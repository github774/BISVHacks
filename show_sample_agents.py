#!/usr/bin/env python3
"""Show sample agent descriptions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from genagents import gen_agents, agent_enc

# Generate sample agents
print("Generating 10 sample agents...")
agents = gen_agents(10)
agents = agent_enc(agents, 'cpu')

print("\n" + "="*80)
print("SAMPLE AGENT DESCRIPTIONS WITH IMMIGRATION ATTRIBUTES")
print("="*80)

for agent in agents[:5]:
    print(f"\nAgent {agent.id}:")
    print(agent.desc_str)
    print(f"  Immigration Status: {agent.attrs.get('15')}")
    print(f"  Primary Language: {agent.attrs.get('19')}")
    print(f"  English Proficiency: {agent.attrs.get('18')}")
    print(f"  Ethnicity: {agent.attrs.get('14')}")
    print()
