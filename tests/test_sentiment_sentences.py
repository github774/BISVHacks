#!/usr/bin/env python3
"""Run SentimentHead on 5 random answers from new_personas.json; compare to VADER [neg, pos]."""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sentiment_head import SentimentHead


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    model = SentimentHead().to(device)
    ckpt = ROOT / "data" / "sentiment_head.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    with open(ROOT / "data" / "new_personas.json") as f:
        personas = json.load(f)
    desc_answer_pairs = [(p["description"], a) for p in personas for a in p["answers"]]
    random.seed(42)
    chosen = random.sample(desc_answer_pairs, min(5, len(desc_answer_pairs)))
    answers = [a for _, a in chosen]
    embs = encoder.encode(answers, convert_to_numpy=True)
    embs_t = torch.tensor(embs, dtype=torch.float32, device=device)

    with torch.inference_mode():
        out = model(embs_t)

    analyzer = SentimentIntensityAnalyzer()
    print("SentimentHead vs VADER [bad/neg, good/pos] for 5 random answers:\n")
    for (desc, answer), vec in zip(chosen, out.tolist()):
        vs = analyzer.polarity_scores(answer)
        vader = [vs["neg"], vs["pos"]]
        print(f"  Description: {(desc[:120] + '...') if len(desc) > 120 else desc}")
        print(f"  Answer:      {answer!r}")
        print(f"  SentimentHead: [bad={vec[0]:.4f}, good={vec[1]:.4f}]")
        print(f"  VADER:        [neg={vader[0]:.4f}, pos={vader[1]:.4f}]")
        print(f"  MAE:          {abs(vec[0] - vader[0]) + abs(vec[1] - vader[1]):.4f}\n")


if __name__ == "__main__":
    main()
