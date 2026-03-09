"""
Predict policy answer alignment: [beneficial, damaging] via SentimentHead.

Input: description string, policy question string.
Output: 2D vector [beneficial, damaging] = [good, bad] from SentimentHead [neg, pos].
"""

from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parent.parent

_ENCODER = None
_MODEL = None
_SENTIMENT_HEAD = None


def _load_resources(
    data_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> None:
    global _ENCODER, _MODEL, _SENTIMENT_HEAD
    if _ENCODER is not None:
        return
    from sentence_transformers import SentenceTransformer
    from src.answer_predictor import AnswerPredictor
    from src.sentiment_head import SentimentHead

    data_dir = data_dir or ROOT / "data"
    model_path = model_path or data_dir / "answer_predictor.pt"

    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    _ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    _MODEL = AnswerPredictor().to(device)
    _MODEL.load_state_dict(torch.load(model_path, map_location=device))
    _MODEL.eval()

    _SENTIMENT_HEAD = SentimentHead().to(device)
    sent_path = data_dir / "sentiment_head.pt"
    if sent_path.exists():
        _SENTIMENT_HEAD.load_state_dict(torch.load(sent_path, map_location=device))
    _SENTIMENT_HEAD.eval()


def encode_policy(policy_question: str) -> torch.Tensor:
    """Encode policy question to embedding."""
    if _ENCODER is None:
        _load_resources()
    device = next(_MODEL.parameters()).device
    q_emb = _ENCODER.encode(policy_question, convert_to_numpy=True)
    q_t = torch.tensor(q_emb, dtype=torch.float32, device=device).unsqueeze(0)
    return q_t

def encode_description(description: str) -> torch.Tensor:
    """Encode description to embedding."""
    if _ENCODER is None:
        _load_resources()
    device = next(_MODEL.parameters()).device
    desc_emb = _ENCODER.encode(description, convert_to_numpy=True)
    desc_t = torch.tensor(desc_emb, dtype=torch.float32, device=device).unsqueeze(0)
    return desc_t

def predict_policy_answer(
    description: str,
    policy_question: str,
    data_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> list[float]:
    """
    Encode description and question, preprocess description, predict answer, project to 2D.

    Returns: [sim_to_beneficial, sim_to_damaging] as python list.
    """
    data_dir = data_dir or ROOT / "data"
    model_path = model_path or data_dir / "answer_predictor.pt"
    _load_resources(data_dir, model_path)

    device = next(_MODEL.parameters()).device
    desc_emb = _ENCODER.encode(description, convert_to_numpy=True)
    q_emb = _ENCODER.encode(policy_question, convert_to_numpy=True)

    from src.preprocessor import preprocess
    persona_blend, _ = preprocess(
        desc_emb,
        descriptions_path=str(data_dir / "archetype_descriptions.json"),
        persona_vectors_path=str(data_dir / "persona_vectors.json"),
        verbose=False,
    )

    q_t = torch.tensor(q_emb, dtype=torch.float32, device=device).unsqueeze(0)
    persona_t = torch.tensor(persona_blend, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.inference_mode():
        out = _MODEL(q_t, persona_t)
        print(out)
        sent = _SENTIMENT_HEAD(out)  # [bad, good] = [neg, pos]
    return [float(sent[0, 1].item()), float(sent[0, 0].item())]  # [beneficial, damaging] = [good, bad]
