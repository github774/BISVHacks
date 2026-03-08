#!/usr/bin/env python3
"""
Train AnswerPredictor on new_personas.json.

- v1 = question embedding, v2 = description embedding
- target = answer embedding

Loss: margin ranking (preserve beneficial>damaging or vice versa by ≥5%) + cosine similarity.
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
from sentence_transformers import SentenceTransformer


def _beneficial_damaging_sims(emb: torch.Tensor, ben: torch.Tensor, dam: torch.Tensor) -> torch.Tensor:
    """emb (B,384), ben/dam (384,). Returns (B,2) raw cosine sims [ben, dam]."""
    emb_n = F.normalize(emb, p=2, dim=-1)
    sim_ben = emb_n @ ben
    sim_dam = emb_n @ dam
    return torch.stack([sim_ben, sim_dam], dim=-1)


def _margin_ranking_loss(diff_orig: torch.Tensor, diff_pred: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """Preserve sign: if orig has ben>dam, pred should have ben-dam >= margin (and vice versa)."""
    sign = torch.sign(diff_orig)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return F.relu(margin - sign * diff_pred).mean()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnswerPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    ben_emb = F.normalize(
        torch.tensor(
            encoder.encode("This policy is beneficial to me.", convert_to_numpy=True),
            dtype=torch.float32, device=device,
        ),
        p=2, dim=0,
    )
    dam_raw = F.normalize(
        torch.tensor(
            encoder.encode("This policy is damaging to me.", convert_to_numpy=True),
            dtype=torch.float32, device=device,
        ),
        p=2, dim=0,
    )
    # Orthogonalize: dam_emb ⊥ ben_emb so cos(ben, dam) = 0
    dam_emb = F.normalize(dam_raw - (dam_raw @ ben_emb) * ben_emb, p=2, dim=0)

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

    n_epochs = 100
    batch_size = 32

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
            v_orig = _beneficial_damaging_sims(a_orig, ben_emb, dam_emb)  # (B, 2)
            v_pred = _beneficial_damaging_sims(pred, ben_emb, dam_emb)  # (B, 2)
            diff_orig = v_orig[:, 0] - v_orig[:, 1]
            diff_pred = v_pred[:, 0] - v_pred[:, 1]
            loss_margin = _margin_ranking_loss(diff_orig, diff_pred, margin=0.05)
            loss_cos = (1 - F.cosine_similarity(F.normalize(pred, p=2, dim=-1), a_orig, dim=-1)).mean()
            loss = loss_margin + loss_cos

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
        v_orig = _beneficial_damaging_sims(a_test_norm, ben_emb, dam_emb)
        v_pred = _beneficial_damaging_sims(pred, ben_emb, dam_emb)
        diff_orig = v_orig[:, 0] - v_orig[:, 1]
        diff_pred = v_pred[:, 0] - v_pred[:, 1]
        sign_orig = torch.sign(diff_orig)
        correct = ((sign_orig * diff_pred) >= 0.05).float().mean().item()
        cos = F.cosine_similarity(F.normalize(pred, p=2, dim=-1), a_test_norm, dim=-1).mean().item()
    print(f"Test margin acc (≥5%): {correct:.4f}, answer cos sim: {cos:.4f}")

    torch.save(model.state_dict(), ROOT / "data" / "answer_predictor.pt")
    print("Saved model to data/answer_predictor.pt")


if __name__ == "__main__":
    main()
