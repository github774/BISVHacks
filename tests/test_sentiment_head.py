#!/usr/bin/env python3
"""Test SentimentHead: forward, trained model MAE on new_personas."""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sentiment_head import SentimentHead


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Forward shape check
    model = SentimentHead().to(device)
    x = torch.randn(8, 384, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 2), f"Expected (8, 2), got {out.shape}"
    print("Forward shape: (8, 384) -> (8, 2) ✓")

    # Load trained model and test on new_personas
    ckpt = ROOT / "data" / "sentiment_head.pt"
    personas_path = ROOT / "data" / "new_personas.json"
    if ckpt.exists() and personas_path.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        analyzer = SentimentIntensityAnalyzer()
        with open(personas_path) as f:
            personas = json.load(f)
        samples = []
        for p in personas:
            for ans_text, ans_emb in zip(p["answers"], p["answer_embeddings"]):
                vs = analyzer.polarity_scores(ans_text)
                target = [vs["neg"], vs["pos"]]
                samples.append((ans_emb, target))

        random.Random(42).shuffle(samples)
        test_data = samples[len(samples) // 2 :]
        X = torch.tensor([s[0] for s in test_data], dtype=torch.float32, device=device)
        y = torch.tensor([s[1] for s in test_data], dtype=torch.float32, device=device)

        with torch.no_grad():
            pred = model(X)
            mae = F.l1_loss(pred, y).item()
        print(f"Test MAE (n={len(test_data)}): {mae:.4f}")
    else:
        print("No trained model or new_personas, skipping MAE test")


if __name__ == "__main__":
    main()
