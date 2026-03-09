#!/usr/bin/env python3
"""
Train AnswerPredictor on new_personas.json.

- v1 = question embedding, v2 = description embedding
- target = answer embedding

Loss: sentiment MAE + cosine to gt + spread (penalize batch similarity).
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from answer_predictor import AnswerPredictor
from sentiment_head import SentimentHead


def _spread_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalize high pairwise similarity within batch (encourage diverse outputs)."""
    pred_n = F.normalize(pred, p=2, dim=-1)
    sim = pred_n @ pred_n.T  # (B, B)
    B = pred.shape[0]
    mask = 1 - torch.eye(B, device=pred.device)
    off_diag = (sim * mask).sum() / (mask.sum() + 1e-9)  # mean pairwise cos sim
    return off_diag  # minimize -> outputs less similar


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnswerPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    sentiment_head = SentimentHead().to(device)
    sent_ckpt = ROOT / "data" / "sentiment_head.pt"
    if sent_ckpt.exists():
        ckpt = torch.load(sent_ckpt, map_location=device)
        sentiment_head.load_state_dict(ckpt, strict=False)
    else:
        print("Warning: sentiment_head.pt not found, using random init. Run scripts/train_sentiment_head.py first.")
    sentiment_head.eval()
    for p in sentiment_head.parameters():
        p.requires_grad = False

    with open(ROOT / "data" / "new_personas.json") as f:
        personas = json.load(f)

    # Flatten to (q_emb, desc_emb, a_emb) per pair
    samples = []
    for p in personas:
        desc = p["description_embedding"]
        for q, a in zip(p["question_embeddings"], p["answer_embeddings"]):
            samples.append((q, desc, a))

    random.Random(42).shuffle(samples)
    split = len(samples) // 2
    train_data = samples[:split]
    test_data = samples[split:]

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    q_train = torch.tensor([s[0] for s in train_data], dtype=torch.float32, device=device)
    desc_train = torch.tensor([s[1] for s in train_data], dtype=torch.float32, device=device)
    a_train = torch.tensor([s[2] for s in train_data], dtype=torch.float32, device=device)

    q_test = torch.tensor([s[0] for s in test_data], dtype=torch.float32, device=device)
    desc_test = torch.tensor([s[1] for s in test_data], dtype=torch.float32, device=device)
    a_test = torch.tensor([s[2] for s in test_data], dtype=torch.float32, device=device)

    a_train_norm = F.normalize(a_train, p=2, dim=-1)
    a_test_norm = F.normalize(a_test, p=2, dim=-1)

    n_epochs = 500
    batch_size = 256

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(train_data), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_data), batch_size):
            idx = perm[i : i + batch_size]
            q = q_train[idx]
            desc = desc_train[idx]
            a_orig = a_train_norm[idx]

            pred = model(q, desc)
            pred_sent = torch.sigmoid(sentiment_head(pred))
            gt_sent = torch.sigmoid(sentiment_head(a_orig))
            loss_sent = F.l1_loss(pred_sent, gt_sent)
            loss_cos = (1 - F.cosine_similarity(F.normalize(pred, p=2, dim=-1), a_orig, dim=-1)).mean()
            loss_spread = _spread_loss(pred)
            loss = loss_sent + loss_cos + loss_spread

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, loss = {avg_loss:.4f}")

    # Eval
    model.eval()
    with torch.no_grad():
        pred = model(q_test, desc_test)
        pred_sent = torch.sigmoid(sentiment_head(pred))
        gt_sent = torch.sigmoid(sentiment_head(a_test_norm))
        sent_mae = F.l1_loss(pred_sent, gt_sent).item()
        cos_sim = F.cosine_similarity(F.normalize(pred, p=2, dim=-1), a_test_norm, dim=-1).mean().item()
    print(f"Test sentiment MAE: {sent_mae:.4f}, cos sim to gt: {cos_sim:.4f}")

    torch.save(model.state_dict(), ROOT / "data" / "answer_predictor.pt")
    print("Saved model to data/answer_predictor.pt")


if __name__ == "__main__":
    main()
