"""
Training script for PersonalityNet.
"""

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from sentence_transformers import SentenceTransformer
from personality_net import PersonalityNet, personality_loss, build_adjective_target

DEFAULT_ADJECTIVES = [
    # Positive Traits
    "Adventurous", "Analytical", "Artistic", "Assertive", "Bold",
    "Calm", "Caring", "Charismatic", "Cheerful", "Clever",
    "Compassionate", "Confident", "Curious", "Daring", "Determined",
    "Easygoing", "Empathetic", "Energetic", "Humorous", "Imaginative",
    "Independent", "Insightful", "Loyal", "Observant", "Optimistic",
    "Playful", "Practical", "Resilient", "Spontaneous", "Thoughtful",
    "Witty",

    # Negative Traits
    "Abrasive", "Aloof", "Arrogant", "Bitter", "Bossy",
    "Callous", "Careless", "Clingy", "Cynical", "Deceitful",
    "Defiant", "Demanding", "Distrustful", "Egocentric", "Impulsive",
    "Inconsiderate", "Indecisive", "Insecure", "Jealous", "Lazy",
    "Moody", "Naive", "Pessimistic", "Reckless", "Rigid",
    "Selfish", "Stubborn", "Suspicious", "Tactless", "Timid",
    "Unreliable"
]


def train_step(
    model: PersonalityNet,
    q_emb: torch.Tensor,
    a_emb: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    pred = model(q_emb, a_emb)
    loss = personality_loss(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersonalityNet().to(device)
    encoder = SentenceTransformer("all-MiniLM-L6-v2").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Load archetypes
    with open(ROOT / "data" / "archetypes.json") as f:
        data = json.load(f)
    archetypes = data["archetypes"]
    questions_text = [a["questions"] for a in archetypes]
    answers_text = [a["answers"] for a in archetypes]

    # 50/50 train/test split
    indices = list(range(len(archetypes)))
    random.Random(42).shuffle(indices)
    split = len(indices) // 2
    train_idx, test_idx = indices[:split], indices[split:]
    train_questions = [questions_text[i] for i in train_idx]
    train_answers = [answers_text[i] for i in train_idx]
    test_questions = [questions_text[i] for i in test_idx]
    test_answers = [answers_text[i] for i in test_idx]

    print(f"Train: {len(train_questions)}, Test: {len(test_questions)}")

    # Encode train set
    q_flat = [q for sample in train_questions for q in sample]
    a_flat = [a for sample in train_answers for a in sample]
    q_emb = encoder.encode(q_flat, convert_to_tensor=True, device=device)
    a_emb = encoder.encode(a_flat, convert_to_tensor=True, device=device)
    q_emb = q_emb.view(len(train_questions), 10, -1)
    a_emb = a_emb.view(len(train_answers), 10, -1)

    target = build_adjective_target(
        train_answers, DEFAULT_ADJECTIVES, encoder, device, derive_from_answers=True
    )

    for step in range(100):
        loss = train_step(model, q_emb, a_emb, target, optimizer, device)
        if step % 20 == 0:
            print(f"Step {step}, loss = {loss:.4f}")

    with torch.no_grad():
        pred = model(q_emb, a_emb)
        cos_train = torch.nn.functional.cosine_similarity(
            pred, target, dim=-1
        ).mean().item()
        print(f"Train cosine sim: {cos_train:.4f}")

    torch.save(model.state_dict(), ROOT / "data" / "personality_net.pt")
    print("Saved model to personality_net.pt")

    # Evaluate on test split
    model.eval()
    q_flat = [q for sample in test_questions for q in sample]
    a_flat = [a for sample in test_answers for a in sample]
    q_test = encoder.encode(q_flat, convert_to_tensor=True, device=device)
    a_test = encoder.encode(a_flat, convert_to_tensor=True, device=device)
    q_test = q_test.view(len(test_questions), 10, -1)
    a_test = a_test.view(len(test_answers), 10, -1)
    target_test = build_adjective_target(
        test_answers, DEFAULT_ADJECTIVES, encoder, device, derive_from_answers=True
    )
    with torch.no_grad():
        pred_test = model(q_test, a_test)
        cos_test = torch.nn.functional.cosine_similarity(
            pred_test, target_test, dim=-1
        ).mean().item()
    print(f"Test cosine sim: {cos_test:.4f}")

    # Speed benchmark: 1000 runs
    model.eval()
    batch_size = q_test.shape[0]
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(1000):
            _ = model(q_test, a_test)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"Speed (1000 runs, batch={batch_size}): {elapsed:.3f}s total, {elapsed/1000*1000:.2f}ms/run, {1000/elapsed:.0f} runs/s")


if __name__ == "__main__":
    main()
