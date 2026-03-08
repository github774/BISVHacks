#!/usr/bin/env python3
"""
Generate 1000 NEW personality descriptions (not in existing archetypes) with:
- 10 unique policy questions (3 positive, 3 negative, 3 neutral, 1 chained)
- 10 answers tailored to their situation
- 384D embeddings for descriptions, questions, and answers

Output: data/new_personas.json
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sentence_transformers import SentenceTransformer

from archetype_definitions import ARCHETYPE_VARIABLES
from policy_templates import (
    generate_answers_for_archetype,
    generate_questions_for_archetype,
)

NUM_PERSONAS = 1000
SEED_NEW = 12345  # Different seed to avoid overlap with archetypes
OUTPUT_PATH = ROOT / "data" / "new_personas.json"


def _format_val(val: str) -> str:
    return str(val).replace("_", " ").replace("–", "-")


def build_description(profile: dict) -> str:
    """Generate 3-sentence description from profile."""
    p = {int(k): v for k, v in profile.items()}
    income = _format_val(p.get(1, "unknown"))
    employment = _format_val(p.get(6, "unknown"))
    industry = _format_val(p.get(7, "unknown"))
    education = _format_val(p.get(3, "unknown"))
    age = _format_val(p.get(11, "unknown"))
    gender = _format_val(p.get(12, "unknown"))
    marital = _format_val(p.get(21, "unknown"))
    household = _format_val(p.get(22, "unknown"))
    housing = _format_val(p.get(41, "unknown"))
    housing_afford = _format_val(p.get(43, "unknown"))
    health_ins = _format_val(p.get(35, "unknown"))
    health_access = _format_val(p.get(36, "unknown"))
    geography = _format_val(p.get(51, "unknown"))
    internet = _format_val(p.get(58, "unknown"))
    isolation = _format_val(p.get(76, "unknown"))
    discrimination = _format_val(p.get(84, "unknown"))
    automation = _format_val(p.get(88, "unknown"))
    community = _format_val(p.get(72, "unknown"))

    s1 = (
        f"This person has {income} income, works {employment} in the {industry} industry "
        f"with {education} education, and is in the {age} age group, {gender}."
    )
    s2 = (
        f"They are {marital} with a household of {household}, live as a {housing} "
        f"with {housing_afford} costs, and have {health_ins} health insurance with "
        f"{health_access} healthcare access."
    )
    s3 = (
        f"They reside in a {geography} area with {internet} internet, experience "
        f"{isolation} social isolation, face {discrimination} exposure to discrimination, "
        f"and have {automation} automation risk in their line of work."
    )
    return f"{s1} {s2} {s3}"


def sample_archetype(rng: random.Random) -> dict[int, str]:
    profile: dict[int, str] = {}
    for var_id, var_def in ARCHETYPE_VARIABLES.items():
        profile[var_id] = rng.choice(var_def["values"])
    return profile


def load_existing_profile_keys(path: Path) -> set[frozenset[tuple[int, str]]]:
    """Load frozenset keys of existing archetype profiles to exclude."""
    if not path.exists():
        return set()
    with open(path) as f:
        data = json.load(f)
    keys = set()
    for a in data.get("archetypes", []):
        profile = {int(k): v for k, v in a["profile"].items()}
        keys.add(frozenset(profile.items()))
    return keys


def main():
    device = "cuda"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    existing = load_existing_profile_keys(ROOT / "data" / "archetypes.json")
    print(f"Excluding {len(existing)} existing archetype profiles")

    rng = random.Random(SEED_NEW)
    seen: set[frozenset[tuple[int, str]]] = set(existing)
    personas: list[dict] = []
    attempts = 0
    max_attempts = NUM_PERSONAS * 100

    while len(personas) < NUM_PERSONAS and attempts < max_attempts:
        candidate = sample_archetype(rng)
        key = frozenset(candidate.items())
        if key not in seen:
            seen.add(key)
            personas.append(candidate)
        attempts += 1

    if len(personas) < NUM_PERSONAS:
        print(f"Warning: Only {len(personas)} unique new profiles (space may be saturated)")

    print(f"Generated {len(personas)} new unique personas")

    # Build descriptions, questions, answers
    descriptions = []
    all_questions = []
    all_answers = []
    for i, profile in enumerate(personas):
        arch_dict = {k: v for k, v in profile.items()}
        arch_rng = random.Random(SEED_NEW + i)
        descriptions.append(build_description(arch_dict))
        questions, metadata = generate_questions_for_archetype(i, arch_rng)
        answers = generate_answers_for_archetype(arch_dict, metadata, arch_rng)
        all_questions.append(questions)
        all_answers.append(answers)

    # Embed
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print("Embedding descriptions...")
    desc_embs = encoder.encode(descriptions, convert_to_numpy=True)
    print("Embedding questions...")
    q_flat = [q for p in all_questions for q in p]
    q_embs = encoder.encode(q_flat, convert_to_numpy=True)
    q_embs = q_embs.reshape(len(personas), 10, -1)
    print("Embedding answers...")
    a_flat = [a for p in all_answers for a in p]
    a_embs = encoder.encode(a_flat, convert_to_numpy=True)
    a_embs = a_embs.reshape(len(personas), 10, -1)

    result = []
    for i in range(len(personas)):
        result.append({
            "persona_id": i,
            "description": descriptions[i],
            "description_embedding": desc_embs[i].tolist(),
            "questions": all_questions[i],
            "question_embeddings": q_embs[i].tolist(),
            "answers": all_answers[i],
            "answer_embeddings": a_embs[i].tolist(),
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {len(result)} personas to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
