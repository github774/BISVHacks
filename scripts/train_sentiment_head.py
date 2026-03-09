#!/usr/bin/env python3
"""
Train SentimentHead on new_personas.json.

- Input: 384D answer embedding (with 0.5% random perturbation during training)
- Target: VADER neg/pos scores [neg, pos] (keywords Bad, Good)
- Output: 2D vector; loss = MAE(pred, target); no sigmoid
"""

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
    model = SentimentHead().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    analyzer = SentimentIntensityAnalyzer()

    with open(ROOT / "data" / "new_personas.json") as f:
        personas = json.load(f)

    samples = []
    for p in personas:
        for ans_text, ans_emb in zip(p["answers"], p["answer_embeddings"]):
            vs = analyzer.polarity_scores(ans_text)
            target = [vs["neg"], vs["pos"]]  # VADER neg/pos (Bad, Good)
            samples.append((ans_emb, target))

    random.Random(42).shuffle(samples)
    split = int(0.8 * len(samples))
    train_data = samples[:split]
    test_data = samples[split:]

    X_train = torch.tensor([s[0] for s in train_data], dtype=torch.float32, device=device)
    y_train = torch.tensor([s[1] for s in train_data], dtype=torch.float32, device=device)  # (N, 2)
    X_test = torch.tensor([s[0] for s in test_data], dtype=torch.float32, device=device)
    y_test = torch.tensor([s[1] for s in test_data], dtype=torch.float32, device=device)  # (N, 2)

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    n_epochs = 100
    batch_size = 64

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(train_data), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_data), batch_size):
            idx = perm[i : i + batch_size]
            x = X_train[idx] + 0.005 * torch.randn_like(X_train[idx], device=device)  # 0.5% perturbation
            y = y_train[idx]
            pred = model(x)
            loss = F.l1_loss(pred, y)  # MAE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                pred = model(X_test)
                mae = F.l1_loss(pred, y_test).item()
            print(f"Epoch {epoch + 1}/{n_epochs}, loss={avg_loss:.4f}, test_MAE={mae:.4f}")

    torch.save(model.state_dict(), ROOT / "data" / "sentiment_head.pt")
    print(f"Saved to data/sentiment_head.pt")


if __name__ == "__main__":
    main()
