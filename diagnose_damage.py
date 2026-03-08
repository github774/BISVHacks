"""
Diagnostic script to understand why damage is saturating.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.genagents import gen_agents, agent_enc
from src.predict_policy_answer import predict_policy_answer
from src.entry import DAMAGE_TUNING, BENEFIT_TUNING

# Generate sample agents
print("Generating 20 agents...")
agents = gen_agents(20)
agents = agent_enc(agents, device="cpu")

# Immigration policy
policy_text = "Restrict immigration and reduce access to public services for undocumented individuals."

print("\n" + "="*80)
print("DAMAGE ANALYSIS BY AGENT PROFILE")
print("="*80)

# Calculate damage for each
damages = []
for i, agent in enumerate(agents):
    benefit, damage = predict_policy_answer(policy_text, agent.desc_str)
    damages.append((i, benefit, damage, agent))
    
# Sort by damage
damages.sort(key=lambda x: -x[2])

print("\nTop 10 most damaged agents:")
print(f"{'Rank':<5} {'Benefit':<8} {'Damage':<8} Profile Summary")
print("-" * 120)

for rank, (aid, benefit, damage, agent) in enumerate(damages[:10], 1):
    # Extract key attributes
    attrs = agent.attrs
    income = attrs.get('1', '?')
    education = attrs.get('3', '?')
    employment = attrs.get('6', '?')
    immigration = attrs.get('15', '?')
    language = attrs.get('19', '?')
    
    profile_summary = f"income={income}, edu={education}, emp={employment}, immig={immigration}, lang={language}"
    print(f"{rank:<5} {benefit:<8.4f} {damage:<8.4f} {profile_summary}")

print("\n\nBottom 10 least damaged agents:")
print(f"{'Rank':<5} {'Benefit':<8} {'Damage':<8} Profile Summary")
print("-" * 120)

for rank, (aid, benefit, damage, agent) in enumerate(damages[-10:], 1):
    # Extract key attributes
    attrs = agent.attrs
    income = attrs.get('1', '?')
    education = attrs.get('3', '?')
    employment = attrs.get('6', '?')
    immigration = attrs.get('15', '?')
    language = attrs.get('19', '?')
    
    profile_summary = f"income={income}, edu={education}, emp={employment}, immig={immigration}, lang={language}"
    print(f"{rank:<5} {benefit:<8.4f} {damage:<8.4f} {profile_summary}")

# Statistics
damage_values = [d[2] for d in damages]
benefit_values = [d[1] for d in damages]

print("\n" + "="*80)
print("STATISTICS")
print("="*80)
print(f"Damage:  min={min(damage_values):.4f}, max={max(damage_values):.4f}, mean={np.mean(damage_values):.4f}, std={np.std(damage_values):.4f}")
print(f"Benefit: min={min(benefit_values):.4f}, max={max(benefit_values):.4f}, mean={np.mean(benefit_values):.4f}, std={np.std(benefit_values):.4f}")
print(f"Range (damage): {max(damage_values) - min(damage_values):.4f}")
print(f"Range (benefit): {max(benefit_values) - min(benefit_values):.4f}")

# Check if benefit-damage might be better
net_impact = [benefit - damage for benefit, damage in zip(benefit_values, damage_values)]
net_impact_tuned = [(benefit * BENEFIT_TUNING) - (damage * DAMAGE_TUNING) for benefit, damage in zip(benefit_values, damage_values)]

print(f"\nNet impact (benefit - damage):")
print(f"  min={min(net_impact):.4f}, max={max(net_impact):.4f}, mean={np.mean(net_impact):.4f}, std={np.std(net_impact):.4f}")
print(f"  range={max(net_impact) - min(net_impact):.4f}")

print(f"\nNet impact with tuning (benefit*{BENEFIT_TUNING} - damage*{DAMAGE_TUNING}):")
print(f"  min={min(net_impact_tuned):.4f}, max={max(net_impact_tuned):.4f}, mean={np.mean(net_impact_tuned):.4f}, std={np.std(net_impact_tuned):.4f}")
print(f"  range={max(net_impact_tuned) - min(net_impact_tuned):.4f}")

# Show top benefited agents
print("\n" + "="*80)
print("BENEFIT ANALYSIS")
print("="*80)

benefits_sorted = sorted(damages, key=lambda x: -x[1])

print("\nTop 10 most benefited agents:")
print(f"{'Rank':<5} {'Benefit':<8} {'Damage':<8} {'Net':<8} Profile Summary")
print("-" * 120)

for rank, (aid, benefit, damage, agent) in enumerate(benefits_sorted[:10], 1):
    attrs = agent.attrs
    income = attrs.get('1', '?')
    education = attrs.get('3', '?')
    employment = attrs.get('6', '?')
    immigration = attrs.get('15', '?')
    language = attrs.get('19', '?')
    net = (benefit * BENEFIT_TUNING) - (damage * DAMAGE_TUNING)
    
    profile_summary = f"income={income}, edu={education}, emp={employment}, immig={immigration}, lang={language}"
    print(f"{rank:<5} {benefit:<8.4f} {damage:<8.4f} {net:<+8.4f} {profile_summary}")

print("\nBottom 10 least benefited agents:")
print(f"{'Rank':<5} {'Benefit':<8} {'Damage':<8} {'Net':<8} Profile Summary")
print("-" * 120)

for rank, (aid, benefit, damage, agent) in enumerate(benefits_sorted[-10:], 1):
    attrs = agent.attrs
    income = attrs.get('1', '?')
    education = attrs.get('3', '?')
    employment = attrs.get('6', '?')
    immigration = attrs.get('15', '?')
    language = attrs.get('19', '?')
    net = (benefit * BENEFIT_TUNING) - (damage * DAMAGE_TUNING)
    
    profile_summary = f"income={income}, edu={education}, emp={employment}, immig={immigration}, lang={language}"
    print(f"{rank:<5} {benefit:<8.4f} {damage:<8.4f} {net:<+8.4f} {profile_summary}")

# Check vulnerable groups specifically
print("\n" + "="*80)
print("VULNERABLE GROUP ANALYSIS")
print("="*80)

undocumented = [d for d in damages if d[3].attrs.get('15') == 'undocumented']
documented = [d for d in damages if d[3].attrs.get('15') in ['citizen', 'permanent resident']]
spanish_speakers = [d for d in damages if d[3].attrs.get('19') in ['Spanish', 'spanish']]
english_speakers = [d for d in damages if d[3].attrs.get('19') in ['English', 'english']]
low_education = [d for d in damages if d[3].attrs.get('3') in ['none', 'high school']]
high_education = [d for d in damages if d[3].attrs.get('3') in ['bachelor', 'graduate']]
low_income = [d for d in damages if d[3].attrs.get('1') in ['very low', 'low']]
high_income = [d for d in damages if d[3].attrs.get('1') in ['very high', 'high']]

def print_group_analysis(name, group):
    if group:
        avg_benefit = np.mean([d[1] for d in group])
        avg_damage = np.mean([d[2] for d in group])
        avg_net = (avg_benefit * BENEFIT_TUNING) - (avg_damage * DAMAGE_TUNING)
        print(f"{name:30} ({len(group):2} agents): benefit={avg_benefit:.4f}, damage={avg_damage:.4f}, net={avg_net:+.4f}")
    else:
        print(f"{name:30} ( 0 agents): N/A")

print("\nVulnerable groups:")
print_group_analysis("Undocumented", undocumented)
print_group_analysis("Spanish speakers", spanish_speakers)
print_group_analysis("Low education", low_education)
print_group_analysis("Low income", low_income)

print("\nPrivileged groups:")
print_group_analysis("Documented (citizen/resident)", documented)
print_group_analysis("English speakers", english_speakers)
print_group_analysis("High education", high_education)
print_group_analysis("High income", high_income)

print("\n" + "="*80)
print("DIFFERENTIAL IMPACT RATIOS")
print("="*80)

if undocumented and documented:
    damage_ratio = np.mean([d[2] for d in undocumented]) / np.mean([d[2] for d in documented])
    benefit_ratio = np.mean([d[1] for d in undocumented]) / np.mean([d[1] for d in documented])
    print(f"Undocumented vs Documented: damage ratio={damage_ratio:.2f}x, benefit ratio={benefit_ratio:.2f}x")
    
if spanish_speakers and english_speakers:
    damage_ratio = np.mean([d[2] for d in spanish_speakers]) / np.mean([d[2] for d in english_speakers])
    benefit_ratio = np.mean([d[1] for d in spanish_speakers]) / np.mean([d[1] for d in english_speakers])
    print(f"Spanish vs English speakers:  damage ratio={damage_ratio:.2f}x, benefit ratio={benefit_ratio:.2f}x")
    
if low_income and high_income:
    damage_ratio = np.mean([d[2] for d in low_income]) / np.mean([d[2] for d in high_income])
    benefit_ratio = np.mean([d[1] for d in low_income]) / np.mean([d[1] for d in high_income])
    print(f"Low vs High income:           damage ratio={damage_ratio:.2f}x, benefit ratio={benefit_ratio:.2f}x")
    
if low_education and high_education:
    damage_ratio = np.mean([d[2] for d in low_education]) / np.mean([d[2] for d in high_education])
    benefit_ratio = np.mean([d[1] for d in low_education]) / np.mean([d[1] for d in high_education])
    print(f"Low vs High education:        damage ratio={damage_ratio:.2f}x, benefit ratio={benefit_ratio:.2f}x")

# Overall average
print(f"Overall average damage: {np.mean(damage_values):.4f}")
