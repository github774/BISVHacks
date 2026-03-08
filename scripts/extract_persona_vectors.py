#!/usr/bin/env python3
"""
Extract 384D persona vectors for all archetypes using the trained PersonalityNet.
Output: data/persona_vectors.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from sentence_transformers import SentenceTransformer

from personality_net import PersonalityNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersonalityNet().to(device)
    model.load_state_dict(torch.load(ROOT / "data" / "personality_net.pt", map_location=device))
    model.eval()

    encoder = SentenceTransformer("all-MiniLM-L6-v2").to(device)

    with open(ROOT / "data" / "archetypes.json") as f:
        data = json.load(f)
    archetypes = data["archetypes"]

    questions = [a["questions"] for a in archetypes]
    answers = [a["answers"] for a in archetypes]

    q_flat = [q for sample in questions for q in sample]
    a_flat = [a for sample in answers for a in sample]

    print("Encoding questions and answers...")
    q_emb = encoder.encode(q_flat, convert_to_tensor=True, device=device)
    a_emb = encoder.encode(a_flat, convert_to_tensor=True, device=device)
    q_emb = q_emb.view(len(archetypes), 10, -1)
    a_emb = a_emb.view(len(archetypes), 10, -1)

    print("Extracting persona vectors...")
    with torch.no_grad():
        vectors = model(q_emb, a_emb)  # (N, 384)

    result = [
        {
            "archetype_id": a["archetype_id"],
            "persona_vector": v.cpu().tolist(),
        }
        for a, v in zip(archetypes, vectors)
    ]

    output_path = ROOT / "data" / "persona_vectors.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {len(result)} persona vectors to {output_path}")


if __name__ == "__main__":
    main()
