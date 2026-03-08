#!/usr/bin/env python3
"""
Test predict_policy_answer: one example with verbose steps, then speed tests on N random personas.
Compares predicted [ben, dam] to real answer projection from new_personas.
Run from project root: python tests/test_predict_policy_answer.py
"""

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from predict_policy_answer import predict_policy_answer, _load_resources
from preprocessor import preprocess
from answer_predictor import AnswerPredictor
from sentence_transformers import SentenceTransformer


def _real_2d_from_answer_embedding(ans_emb: list[float], ben: torch.Tensor, dam: torch.Tensor, device: torch.device) -> list[float]:
    """Project answer embedding onto ben/dam to get ground-truth [sim_ben, sim_dam]."""
    t = torch.tensor(ans_emb, dtype=torch.float32, device=device)
    t_n = F.normalize(t, p=2, dim=0)
    return [float((t_n @ ben).item()), float((t_n @ dam).item())]


def _verbose_one_example():
    """Print all steps for a single prediction. Uses new_personas for real comparison."""
    print("=" * 60)
    print("VERBOSE EXAMPLE: one prediction with all steps")
    print("=" * 60)

    data_dir = ROOT / "data"
    with open(data_dir / "new_personas.json") as f:
        personas = json.load(f)
    p = random.Random(42).choice(personas)
    q_idx = random.Random(43).randint(0, len(p["questions"]) - 1)
    description = p["description"]
    policy_question = p["questions"][q_idx]
    real_answer_emb = p["answer_embeddings"][q_idx]
    model_path = data_dir / "answer_predictor.pt"

    # 1. Load resources
    print("\n1. Loading resources (encoder, model, ben/dam embeddings)...")
    t0 = time.perf_counter()
    _load_resources(data_dir, model_path)
    print(f"   Load time: {(time.perf_counter() - t0) * 1000:.1f} ms")

    import predict_policy_answer as m
    encoder = m._ENCODER

    # 2. Encode
    print("\n2. Encoding description and policy question...")
    t0 = time.perf_counter()
    desc_emb = encoder.encode(description, convert_to_numpy=True)
    q_emb = encoder.encode(policy_question, convert_to_numpy=True)
    print(f"   Encoding time: {(time.perf_counter() - t0) * 1000:.1f} ms")
    print(f"   Description embedding shape: {desc_emb.shape}")
    print(f"   Question embedding shape: {q_emb.shape}")

    # 3. Preprocess
    print("\n3. Preprocessing description embedding -> persona blend...")
    t0 = time.perf_counter()
    persona_blend, _ = preprocess(
        desc_emb,
        descriptions_path=str(data_dir / "archetype_descriptions.json"),
        persona_vectors_path=str(data_dir / "persona_vectors.json"),
        verbose=False,
    )
    print(f"   Preprocess time: {(time.perf_counter() - t0) * 1000:.1f} ms")
    print(f"   Persona blend shape: {persona_blend.shape}")

    # 4. Answer predictor
    model = m._MODEL
    ben = m._BEN_EMB
    dam = m._DAM_EMB
    device = next(model.parameters()).device

    q_t = torch.tensor(q_emb, dtype=torch.float32, device=device).unsqueeze(0)
    persona_t = torch.tensor(persona_blend, dtype=torch.float32, device=device).unsqueeze(0)

    print("\n4. AnswerPredictor forward...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(q_t, persona_t)
        emb_n = F.normalize(out, p=2, dim=-1)
        sim_ben = (emb_n @ ben).item()
        sim_dam = (emb_n @ dam).item()
    print(f"   Forward time: {(time.perf_counter() - t0) * 1000:.1f} ms")

    # 5. List conversion
    result = [float(sim_ben), float(sim_dam)]
    print("\n5. Final 2D vector (list conversion)...")
    print(f"   Result: {result}")

    # Full function call
    print("\n6. Full predict_policy_answer() call:")
    result_full = predict_policy_answer(description, policy_question)
    print(f"   Predicted [ben, dam]: {result_full}")

    # Compare to real answer embedding projection
    real_2d = _real_2d_from_answer_embedding(real_answer_emb, ben, dam, device)
    print(f"\n7. Real answer [ben, dam] (from answer_embedding): {real_2d}")
    print(f"   Diff (pred - real): [{result_full[0] - real_2d[0]:.4f}, {result_full[1] - real_2d[1]:.4f}]")


def _speed_test_1000():
    """Run N predictions on random personas from new_personas, batched, preloaded, on MPS."""
    N = 1000
    BATCH = 64
    random.seed(123)

    print("\n" + "=" * 60)
    print(f"SPEED TEST: {N} random users (batched, preloaded, MPS)")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")

    data_dir = ROOT / "data"
    with open(data_dir / "new_personas.json") as f:
        personas = json.load(f)

    # Sample N (persona_idx, q_idx) without replacement when possible
    samples = []
    for _ in range(N):
        p = random.choice(personas)
        q_idx = random.randint(0, len(p["questions"]) - 1)
        samples.append((p["description"], p["questions"][q_idx], p["answer_embeddings"][q_idx]))
    descs = [s[0] for s in samples]
    questions = [s[1] for s in samples]
    real_answer_embs = [s[2] for s in samples]
    desc_path = str(data_dir / "archetype_descriptions.json")
    pers_path = str(data_dir / "persona_vectors.json")

    print("\nPreloading into RAM/MPS...")
    t_preload = time.perf_counter()

    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    model = AnswerPredictor().to(device)
    ckpt = data_dir / "answer_predictor.pt"
    sd = torch.load(ckpt, map_location=device)
    if "block_W" in sd and sd["block_W"].dim() == 2:
        sd["block_W"] = sd["block_W"].unsqueeze(0).expand(3, -1, -1).clone()
        sd["block_b"] = sd["block_b"].unsqueeze(0).expand(3, -1, -1).clone()
    model.load_state_dict(sd)
    model.eval()

    ben_raw = encoder.encode("This policy is beneficial to me.", convert_to_numpy=True)
    dam_raw = encoder.encode("This policy is damaging to me.", convert_to_numpy=True)
    ben = F.normalize(torch.tensor(ben_raw, dtype=torch.float32, device=device), p=2, dim=0)
    dam_raw_t = F.normalize(torch.tensor(dam_raw, dtype=torch.float32, device=device), p=2, dim=0)
    dam = F.normalize(dam_raw_t - (dam_raw_t @ ben) * ben, p=2, dim=0)

    from preprocessor import _get_mps_data, preprocess_batched_mps_preloaded
    desc_np, pers_np = _get_mps_data(desc_path, pers_path)
    desc_t = torch.from_numpy(desc_np).to(device)
    pers_t = torch.from_numpy(pers_np).to(device)

    desc_embs = encoder.encode(descs, convert_to_numpy=True, show_progress_bar=False)
    q_embs = encoder.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    desc_embs = torch.from_numpy(desc_embs.astype("float32")).to(device)
    q_embs_t = torch.from_numpy(q_embs.astype("float32")).to(device)

    print(f"   Preload time: {(time.perf_counter() - t_preload):.2f} s")

    def run_batches(desc_tensor, q_tensor):
        out_list = []
        for i in range(0, N, BATCH):
            emb = desc_tensor[i : i + BATCH]
            qb = q_tensor[i : i + BATCH]
            b = emb.shape[0]
            persona = preprocess_batched_mps_preloaded(emb, desc_t, pers_t)
            with torch.inference_mode():
                out = model(qb, persona)
                emb_n = F.normalize(out, p=2, dim=-1)
                sim_ben = emb_n @ ben
                sim_dam = emb_n @ dam
            for j in range(b):
                out_list.append([float(sim_ben[j].item()), float(sim_dam[j].item())])
        return out_list

    print("\n--- Time 1: Batched full (preproc + answer_predictor + list), embeddings preloaded ---")
    t0 = time.perf_counter()
    results = run_batches(desc_embs, q_embs_t)
    t1 = time.perf_counter() - t0
    print(f"   {N} users (batch={BATCH}): {t1:.3f} s  ({t1 / N * 1000:.2f} ms per user)")

    print("\n--- Time 2: Batched preproc + answer_predictor only ---")
    t0 = time.perf_counter()
    for i in range(0, N, BATCH):
        emb = desc_embs[i : i + BATCH]
        qb = q_embs_t[i : i + BATCH]
        b = emb.shape[0]
        persona = preprocess_batched_mps_preloaded(emb, desc_t, pers_t)
        with torch.inference_mode():
            out = model(qb, persona)
            emb_n = F.normalize(out, p=2, dim=-1)
            _ = emb_n @ ben
            _ = emb_n @ dam
    t2 = time.perf_counter() - t0
    print(f"   {N} users (batch={BATCH}): {t2:.3f} s  ({t2 / N * 1000:.2f} ms per user)")

    print("\n--- Time 3: Encoder + batched preproc + answer_predictor + list ---")
    t0 = time.perf_counter()
    desc_embs_fresh = encoder.encode(descs, convert_to_numpy=True, show_progress_bar=False)
    q_embs_fresh = encoder.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    desc_embs_3 = torch.from_numpy(desc_embs_fresh.astype("float32")).to(device)
    q_embs_3 = torch.from_numpy(q_embs_fresh.astype("float32")).to(device)
    results3 = run_batches(desc_embs_3, q_embs_3)
    t3 = time.perf_counter() - t0
    print(f"   {N} users (batch={BATCH}): {t3:.3f} s  ({t3 / N * 1000:.2f} ms per user)")

    # Compare predicted vs real
    real_2ds = [_real_2d_from_answer_embedding(emb, ben, dam, device) for emb in real_answer_embs]
    pred = np.array(results)
    real = np.array(real_2ds)
    mae = np.abs(pred - real).mean()
    cos_sim = np.sum(pred * real, axis=1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(real, axis=1) + 1e-9)
    print("\n--- Predicted vs real [ben, dam] (from answer_embedding) ---")
    print(f"   Mean absolute error: {mae:.4f}")
    print(f"   Mean cosine similarity (2D): {cos_sim.mean():.4f}")


def main():
    _verbose_one_example()
    _speed_test_1000()


if __name__ == "__main__":
    main()
