#!/usr/bin/env python3
"""Test preprocessor. Run from project root: python tests/test_preprocessor.py"""

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from preprocessor import preprocess, preprocess_batched_mps


def main():
    random.seed(42)
    data_dir = ROOT / "data"

    with open(data_dir / "archetype_descriptions.json") as f:
        descriptions = json.load(f)

    idx = random.randint(0, len(descriptions) - 1)
    sample = descriptions[idx]
    embedding = sample["embedding"]

    print("PREPROCESSOR TEST")
    output, _ = preprocess(embedding, str(data_dir / "archetype_descriptions.json"), str(data_dir / "persona_vectors.json"))
    print(f"Output shape: {output.shape}")

    # Speed test
    t0 = time.perf_counter()
    for _ in range(100):
        _ = preprocess(embedding, str(data_dir / "archetype_descriptions.json"), str(data_dir / "persona_vectors.json"))
    print(f"Numpy 100 runs: {time.perf_counter() - t0:.3f}s")

    # MPS batched
    try:
        import torch
        if torch.backends.mps.is_available():
            all_emb = np.array([d["embedding"] for d in descriptions], dtype=np.float32)
            batched = preprocess_batched_mps(all_emb[:64], str(data_dir / "archetype_descriptions.json"), str(data_dir / "persona_vectors.json"))
            print(f"MPS batched 64 samples: OK, shape {batched.shape}")
    except Exception as e:
        print(f"MPS skipped: {e}")


if __name__ == "__main__":
    main()
