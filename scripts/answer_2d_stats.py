#!/usr/bin/env python3
"""
Generate 2D [ben, dam] vectors for all answers in new_personas via SentimentHead.
Show top extremes: highest (ben - dam) and highest (dam - ben).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from sentiment_head import SentimentHead


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentiment_head = SentimentHead().to(device)
    ckpt = ROOT / "data" / "sentiment_head.pt"
    if ckpt.exists():
        sentiment_head.load_state_dict(torch.load(ckpt, map_location=device))
    sentiment_head.eval()

    with open(ROOT / "data" / "new_personas.json") as f:
        personas = json.load(f)

    rows = []
    for pi, p in enumerate(personas):
        for qi, (q, a, a_emb) in enumerate(
            zip(p["questions"], p["answers"], p["answer_embeddings"])
        ):
            rows.append((pi, qi, q, a, a_emb))

    embs = np.array([r[4] for r in rows], dtype=np.float32)
    embs_t = torch.from_numpy(embs).to(device)
    with torch.inference_mode():
        sent = sentiment_head(embs_t)
    sim_dam = sent[:, 0].cpu().numpy()
    sim_ben = sent[:, 1].cpu().numpy()
    diff = sim_ben - sim_dam

    n_show = 5
    idxs_ben = np.argsort(diff)[-n_show:][::-1]
    idxs_dam = np.argsort(diff)[:n_show]

    def show(idx, rank):
        pi, qi, q, a, _ = rows[idx]
        print(f"  #{rank} diff={diff[idx]:.4f} (ben={sim_ben[idx]:.4f}, dam={sim_dam[idx]:.4f})")
        print(f"      Q: {q[:80]}...")
        print(f"      A: {a[:120]}...")

    print("=== Biggest ben - dam (most beneficial) ===")
    for r, i in enumerate(idxs_ben, 1):
        show(i, r)

    print("\n=== Biggest dam - ben (most damaging) ===")
    for r, i in enumerate(idxs_dam, 1):
        show(i, r)


if __name__ == "__main__":
    main()
