#!/usr/bin/env python3
"""Test AnswerPredictor: accuracy on new_personas, MPS batched speed test."""

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from answer_predictor import AnswerPredictor, DIM
from sentence_transformers import SentenceTransformer


def _beneficial_damaging_sims(emb, ben, dam):
    emb_n = F.normalize(emb, p=2, dim=-1)
    sim_ben = emb_n @ ben
    sim_dam = emb_n @ dam
    return torch.stack([sim_ben, sim_dam], dim=-1)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AnswerPredictor().to(device)
    ckpt = ROOT / "data" / "answer_predictor.pt"
    if ckpt.exists():
        sd = torch.load(ckpt, map_location=device)
        # Handle old checkpoint: block_W/block_b (384,384) -> expand to (3,384,384)
        if "block_W" in sd and sd["block_W"].dim() == 2:
            sd["block_W"] = sd["block_W"].unsqueeze(0).expand(3, -1, -1).clone()
            sd["block_b"] = sd["block_b"].unsqueeze(0).expand(3, -1, -1).clone()
        model.load_state_dict(sd)
        model.eval()
    else:
        print("No trained model at data/answer_predictor.pt, using random init for speed test")

    # Param count
    total = sum(p.numel() for p in model.parameters())
    print("AnswerPredictor parameter count:")
    for name, p in model.named_parameters():
        print(f"  {name}: {p.numel():,}")
    print(f"  Total: {total:,}")

    # Forward sanity check
    batch = 4
    v1 = torch.randn(batch, DIM, device=device)
    v2 = torch.randn(batch, DIM, device=device)
    with torch.inference_mode():
        out = model(v1, v2)
    print(f"\nForward: v1/v2 ({batch}, {DIM}) -> out {out.shape}")

    # Accuracy test (if model trained and new_personas exists)
    personas_path = ROOT / "data" / "new_personas.json"
    if ckpt.exists() and personas_path.exists():
        with open(personas_path) as f:
            personas = json.load(f)
        samples = []
        for p in personas:
            desc = p["description_embedding"]
            for q, a in zip(p["question_embeddings"], p["answer_embeddings"]):
                samples.append((q, desc, a))
        random.Random(42).shuffle(samples)
        test_data = samples[len(samples) // 2:]
        q_test = torch.tensor([s[0] for s in test_data], dtype=torch.float32, device=device)
        desc_test = torch.tensor([s[1] for s in test_data], dtype=torch.float32, device=device)
        a_test = torch.tensor([s[2] for s in test_data], dtype=torch.float32, device=device)
        a_test_norm = F.normalize(a_test, p=2, dim=-1)

        encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        ben_emb = F.normalize(
            torch.tensor(encoder.encode("This policy is beneficial to me.", convert_to_numpy=True), dtype=torch.float32, device=device),
            p=2, dim=0,
        )
        dam_raw = F.normalize(
            torch.tensor(encoder.encode("This policy is damaging to me.", convert_to_numpy=True), dtype=torch.float32, device=device),
            p=2, dim=0,
        )
        dam_emb = F.normalize(dam_raw - (dam_raw @ ben_emb) * ben_emb, p=2, dim=0)

        with torch.inference_mode():
            pred = model(q_test, desc_test)
            cos_sim = F.cosine_similarity(F.normalize(pred, p=2, dim=-1), a_test_norm, dim=-1).mean().item()
            v_orig = _beneficial_damaging_sims(a_test_norm, ben_emb, dam_emb)
            v_pred = _beneficial_damaging_sims(pred, ben_emb, dam_emb)
            diff_orig = v_orig[:, 0] - v_orig[:, 1]
            diff_pred = v_pred[:, 0] - v_pred[:, 1]
            sign_orig = torch.sign(diff_orig)
            margin_acc = ((sign_orig * diff_pred) >= 0.05).float().mean().item()
        print(f"\nAccuracy (test set, n={len(test_data)}):")
        print(f"  Answer cos sim: {cos_sim:.4f}")
        print(f"  Margin acc (≥5%): {margin_acc:.4f}")

    # MPS batched speed test
    if torch.backends.mps.is_available():
        batch_size = 256
        n_runs = 500
        v1 = torch.randn(batch_size, DIM, device=device, dtype=torch.float32)
        v2 = torch.randn(batch_size, DIM, device=device, dtype=torch.float32)
        # Warmup
        for _ in range(10):
            _ = model(v1, v2)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(v1, v2)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        samples = n_runs * batch_size
        print(f"\nMPS batched speed test:")
        print(f"  {n_runs} runs × batch {batch_size} = {samples} samples")
        print(f"  Total: {elapsed:.3f}s | Per sample: {elapsed/samples*1000:.3f}ms | {samples/elapsed:.0f} samples/s")
    else:
        print("\nMPS not available, skipping batched speed test")


if __name__ == "__main__":
    main()
