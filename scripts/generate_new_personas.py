#!/usr/bin/env python3
"""
Generate 1000 NEW personality descriptions (not in existing archetypes) with:
- ~8-sentence descriptions covering all 100 attributes
- 6 unique policy questions (3 positive, 3 negative)
- 6 answers tailored to their situation
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


def _get(p: dict, i: int) -> str:
    return _format_val(p.get(i, "unknown"))


def build_description(profile: dict) -> str:
    """Generate compact description covering all 100 attributes, under 250 tokens."""
    p = {int(k): v for k, v in profile.items()}
    parts = [
        f"{_get(p, 1)} income, {_get(p, 2)} wealth, {_get(p, 3)}. {_get(p, 4)} sector, {_get(p, 6)} {_get(p, 7)}, {_get(p, 5)}, {_get(p, 8)}, {_get(p, 9)} union, {_get(p, 10)} mobility.",
        f"{_get(p, 11)}, {_get(p, 12)}, {_get(p, 13)}, {_get(p, 14)}. {_get(p, 15)}, {_get(p, 16)}, {_get(p, 17)}, {_get(p, 18)} English, {_get(p, 19)}, {_get(p, 20)}.",
        f"{_get(p, 21)}, {_get(p, 22)} hh, {_get(p, 23)} kids, {_get(p, 24)} single parent, {_get(p, 25)} elders, {_get(p, 26)}, {_get(p, 27)}, {_get(p, 28)} childcare, {_get(p, 29)}, {_get(p, 30)}.",
        f"{_get(p, 31)} physical, {_get(p, 32)} mental, {_get(p, 33)} chronic, {_get(p, 34)}. {_get(p, 35)} ins, {_get(p, 36)} access, {_get(p, 37)}, {_get(p, 38)} substance, {_get(p, 39)} life risk, {_get(p, 40)}.",
        f"{_get(p, 41)}, {_get(p, 42)}, {_get(p, 43)}. Nbr: {_get(p, 44)}, {_get(p, 45)}, {_get(p, 46)}, {_get(p, 47)} poll, {_get(p, 48)} green, {_get(p, 49)}, {_get(p, 50)}. {_get(p, 51)} {_get(p, 52)}, {_get(p, 53)}, {_get(p, 54)}, {_get(p, 55)}, {_get(p, 56)} transit, {_get(p, 57)}.",
        f"{_get(p, 58)} net, {_get(p, 59)} jobs, {_get(p, 60)} COL. {_get(p, 61)} schools, {_get(p, 62)} college, {_get(p, 63)} train, {_get(p, 64)} legal, {_get(p, 65)}. {_get(p, 66)} bank, {_get(p, 67)} credit, {_get(p, 68)} rep, {_get(p, 69)} vote, {_get(p, 70)} trust.",
        f"{_get(p, 71)} net, {_get(p, 72)} community, {_get(p, 73)} relig, {_get(p, 74)} civic, {_get(p, 75)} volunteer, {_get(p, 76)} isolation. {_get(p, 77)} informal, {_get(p, 78)} mentor, {_get(p, 79)} prof, {_get(p, 80)}. {_get(p, 81)} security, {_get(p, 82)} volatility, {_get(p, 83)} legal, {_get(p, 84)} discrim, {_get(p, 85)}. {_get(p, 86)} power, {_get(p, 87)} safe, {_get(p, 88)} autom, {_get(p, 89)} fin, {_get(p, 90)}.",
        f"{_get(p, 91)} digital, {_get(p, 92)} comp, {_get(p, 93)} phone, {_get(p, 94)} online, {_get(p, 95)} media, {_get(p, 96)}. {_get(p, 97)} remote, {_get(p, 98)} presence, {_get(p, 99)} econ, {_get(p, 100)} AI.",
    ]
    return " ".join(parts)


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
        questions, metadata = generate_questions_for_archetype(
            i, arch_rng, include_neutral_chained=False
        )
        answers = generate_answers_for_archetype(
            arch_dict, metadata, arch_rng, include_neutral_chained=False
        )
        all_questions.append(questions)
        all_answers.append(answers)

    # Embed
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print("Embedding descriptions...")
    desc_embs = encoder.encode(descriptions, convert_to_numpy=True)
    print("Embedding questions...")
    q_flat = [q for p in all_questions for q in p]
    q_embs = encoder.encode(q_flat, convert_to_numpy=True)
    q_embs = q_embs.reshape(len(personas), 6, -1)
    print("Embedding answers...")
    a_flat = [a for p in all_answers for a in p]
    a_embs = encoder.encode(a_flat, convert_to_numpy=True)
    a_embs = a_embs.reshape(len(personas), 6, -1)

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
