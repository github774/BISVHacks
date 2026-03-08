#!/usr/bin/env python3
"""
Generate a 3-sentence detailed description for each archetype and embed it.
Output: data/archetype_descriptions.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sentence_transformers import SentenceTransformer

# Key profile fields for readable descriptions (variable_id -> label)
KEY_FIELDS = {
    1: "income",
    2: "wealth",
    3: "education",
    6: "employment",
    7: "industry",
    8: "job seniority",
    11: "age group",
    12: "gender",
    14: "race/ethnicity",
    15: "immigration status",
    21: "marital status",
    22: "household size",
    23: "children",
    24: "single parent",
    35: "health insurance",
    36: "healthcare access",
    41: "housing status",
    42: "housing stability",
    43: "housing affordability",
    51: "geography",
    55: "commute time",
    58: "internet access",
    76: "social isolation",
    84: "exposure to discrimination",
    88: "automation risk",
}


def _format_val(val: str) -> str:
    """Format value for natural language (e.g. 'very low' -> 'very low')."""
    return str(val).replace("_", " ").replace("–", "-")


def build_description(profile: dict) -> str:
    """Generate a 3-sentence detailed description from archetype profile."""
    p = {int(k): v for k, v in profile.items()}

    # Sentence 1: Demographics and employment
    income = _format_val(p.get(1, "unknown"))
    employment = _format_val(p.get(6, "unknown"))
    industry = _format_val(p.get(7, "unknown"))
    education = _format_val(p.get(3, "unknown"))
    age = _format_val(p.get(11, "unknown"))
    gender = _format_val(p.get(12, "unknown"))

    s1 = (
        f"This person has {income} income, works {employment} in the {industry} industry "
        f"with {education} education, and is in the {age} age group, {gender}."
    )

    # Sentence 2: Family, housing, and health
    marital = _format_val(p.get(21, "unknown"))
    household = _format_val(p.get(22, "unknown"))
    housing = _format_val(p.get(41, "unknown"))
    housing_afford = _format_val(p.get(43, "unknown"))
    health_ins = _format_val(p.get(35, "unknown"))
    health_access = _format_val(p.get(36, "unknown"))

    s2 = (
        f"They are {marital} with a household of {household}, live as a {housing} "
        f"with {housing_afford} costs, and have {health_ins} health insurance with "
        f"{health_access} healthcare access."
    )

    # Sentence 3: Geography, context, and vulnerability
    geography = _format_val(p.get(51, "unknown"))
    internet = _format_val(p.get(58, "unknown"))
    isolation = _format_val(p.get(76, "unknown"))
    discrimination = _format_val(p.get(84, "unknown"))
    automation = _format_val(p.get(88, "unknown"))

    s3 = (
        f"They reside in a {geography} area with {internet} internet, experience "
        f"{isolation} social isolation, face {discrimination} exposure to discrimination, "
        f"and have {automation} automation risk in their line of work."
    )

    return f"{s1} {s2} {s3}"


def main():
    device = "cuda"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    with open(ROOT / "data" / "archetypes.json") as f:
        data = json.load(f)
    archetypes = data["archetypes"]

    result = []
    descriptions = []

    print("Generating descriptions...")
    for a in archetypes:
        desc = build_description(a["profile"])
        descriptions.append(desc)
        result.append({
            "archetype_id": a["archetype_id"],
            "description": desc,
        })

    print("Embedding descriptions...")
    embeddings = encoder.encode(
        descriptions, convert_to_tensor=True, device=device
    )

    for r, emb in zip(result, embeddings):
        r["embedding"] = emb.cpu().tolist() if hasattr(emb, "cpu") else emb.tolist()

    output_path = ROOT / "data" / "archetype_descriptions.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {len(result)} descriptions to {output_path}")


if __name__ == "__main__":
    main()
