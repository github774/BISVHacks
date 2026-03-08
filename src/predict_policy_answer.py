"""
Predict policy answer alignment: [sim_to_beneficial, sim_to_damaging].

Input: description string, policy question string.
Output: 2D vector [beneficial, damaging] as python list.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent

_ENCODER = None
_MODEL = None
_BEN_EMB = None
_DAM_EMB = None


def _load_resources(
    data_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> None:
    global _ENCODER, _MODEL, _BEN_EMB, _DAM_EMB
    if _ENCODER is not None:
        return
    from sentence_transformers import SentenceTransformer
    from answer_predictor import AnswerPredictor

    data_dir = data_dir or ROOT / "data"
    model_path = model_path or data_dir / "answer_predictor.pt"

    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    _ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    _MODEL = AnswerPredictor().to(device)
    _MODEL.load_state_dict(torch.load(model_path, map_location=device))
    _MODEL.eval()

    ben_raw = _ENCODER.encode("This policy is beneficial to me.", convert_to_numpy=True)
    dam_raw = _ENCODER.encode("This policy is damaging to me.", convert_to_numpy=True)
    ben = F.normalize(torch.tensor(ben_raw, dtype=torch.float32, device=device), p=2, dim=0)
    dam_raw_t = F.normalize(torch.tensor(dam_raw, dtype=torch.float32, device=device), p=2, dim=0)
    dam = F.normalize(dam_raw_t - (dam_raw_t @ ben) * ben, p=2, dim=0)
    _BEN_EMB = ben
    _DAM_EMB = dam


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

    from preprocessor import preprocess
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
        emb_n = F.normalize(out, p=2, dim=-1)
        sim_ben = (emb_n @ _BEN_EMB).item()
        sim_dam = (emb_n @ _DAM_EMB).item()

    return [float(sim_ben), float(sim_dam)]
