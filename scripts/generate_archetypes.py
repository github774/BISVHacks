#!/usr/bin/env python3
"""
Generate 1000+ unique socioeconomic archetypes with 10 governmental policy questions
and 10 situation-tailored answers each.

- 3 questions with positive policy effect
- 3 questions with negative policy effect  
- 3 questions with neutral policy effect
- 1 question about chained effect elsewhere in the community

Output: JSON file with archetype_id, profile (100 variables), questions, answers.
"""

import json
import random
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from src.archetype_definitions import ARCHETYPE_VARIABLES
from src.policy_templates import (
    generate_answers_for_archetype,
    generate_questions_for_archetype,
)

NUM_ARCHETYPES = 1050  # 1000+ unique archetypes
SEED = 42
OUTPUT_PATH = ROOT / "data" / "archetypes.json"


def sample_archetype(rng: random.Random) -> dict[int, str]:
    """Sample one random archetype profile from the 100-variable space."""
    profile: dict[int, str] = {}
    for var_id, var_def in ARCHETYPE_VARIABLES.items():
        values = var_def["values"]
        profile[var_id] = rng.choice(values)
    return profile


def sample_unique_archetypes(n: int, seed: int) -> list[dict[int, str]]:
    """Sample n unique archetypes. Uses frozenset of (var_id, value) tuples for uniqueness."""
    rng = random.Random(seed)
    seen: set[frozenset[tuple[int, str]]] = set()
    archetypes: list[dict[int, str]] = []
    attempts = 0
    max_attempts = n * 100  # Avoid infinite loop
    while len(archetypes) < n and attempts < max_attempts:
        candidate = sample_archetype(rng)
        key = frozenset(candidate.items())
        if key not in seen:
            seen.add(key)
            archetypes.append(candidate)
        attempts += 1
    if len(archetypes) < n:
        print(f"Warning: Only generated {len(archetypes)} unique archetypes after {max_attempts} attempts.")
    return archetypes


def archetype_to_dict(profile: dict[int, str]) -> dict[str, str]:
    """Convert int-keyed profile to string-keyed for JSON serialization."""
    return {str(k): v for k, v in profile.items()}


def main() -> None:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else str(OUTPUT_PATH)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_ARCHETYPES

    print(f"Generating {n} unique archetypes...")
    archetypes = sample_unique_archetypes(n, SEED)
    print(f"Sampled {len(archetypes)} unique profiles.")

    rng = random.Random(SEED)
    results: list[dict] = []

    for i, profile in enumerate(archetypes):
        archetype_dict = {k: v for k, v in profile.items()}  # Use int keys for template logic
        arch_rng = random.Random(SEED + i)  # Reproducible per-archetype

        questions, metadata = generate_questions_for_archetype(i, arch_rng)
        answers = generate_answers_for_archetype(archetype_dict, metadata, arch_rng) #type: ignore

        results.append({
            "archetype_id": i,
            "profile": archetype_to_dict(profile),
            "questions": questions,
            "answers": answers,
        })

    archetype_keys = {str(k): v["name"] for k, v in ARCHETYPE_VARIABLES.items()}

    output = {
        "archetype_keys": archetype_keys,
        "archetypes": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(results)} archetypes to {output_path}")


if __name__ == "__main__":
    main()
