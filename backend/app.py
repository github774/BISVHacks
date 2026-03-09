"""
Flask backend for MarGIN policy analysis.
Exposes POST /api/analyze: send policy text, get simulation results from src.entry.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Project root (parent of backend/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Demo mode returns fixed percentages instantly to speed up UI testing.
USE_TEST_DATA = False

from src.entry import PolicyImpactSimulator, NUM_NODES

# Pre-load simulator at startup (eager loading), unless test mode is enabled.
_simulator = None
if USE_TEST_DATA:
    print("Test data mode enabled: skipping simulator preload.")
else:
    print("Loading simulator at startup...")
    _simulator = PolicyImpactSimulator(
        n_agents=NUM_NODES,
        data_dir=ROOT / "data",
        model_name="all-MiniLM-L6-v2",
        device="cpu",
    )
    _simulator.load()
    print("Simulator loaded and ready!")


# Neutral band: impact in [-NEUTRAL_THRESHOLD, NEUTRAL_THRESHOLD] is neutral
NEUTRAL_THRESHOLD = 0.15


def get_simulator():
    """Return the pre-loaded simulator."""
    if _simulator is None:
        raise RuntimeError("Simulator is not initialized (test mode may be enabled).")
    return _simulator


def calculate_influence(profile: dict) -> float:
    """Same logic as scripts/add_influence.py: composite score from profile attrs, 0-100."""
    def get_score(val, mapping, default=1):
        if not val:
            return default
        return mapping.get(str(val).lower(), default)

    score = 0
    score += get_score(profile.get("1", ""), {"very high": 5, "high": 4, "moderate": 3, "low": 2, "very low": 1, "none": 0}, 1)
    score += get_score(profile.get("2", ""), {"high": 5, "moderate": 3, "limited": 2, "none": 1}, 1)
    score += get_score(profile.get("3", ""), {"post-grad": 5, "college": 4, "high school": 2, "none": 1}, 1)
    score += get_score(profile.get("8", ""), {"executive": 5, "senior": 4, "mid-career": 3, "entry-level": 2, "none": 1}, 1)
    score += get_score(profile.get("68", ""), {"high": 5, "moderate": 3, "low": 1}, 1)
    score += get_score(profile.get("71", ""), {"large": 5, "medium": 3, "small": 1}, 1)
    score += get_score(profile.get("80", ""), {"formal": 5, "informal": 3, "none": 1}, 1)
    # Max possible is 35; return 0-1 for frontend
    normalized = (score / 35.0)
    return round(normalized, 4)


def build_analysis_response(simulator, results: dict, n_agents: int | None = None) -> dict:
    """Turn run_simulation results into JSON-friendly dict with summary and top lists."""
    initial = results["initial_damage"]
    n = len(initial)
    n_return = min(n_agents, n) if n_agents is not None else n
    # Net impact: positive = benefit, negative = harm
    impacts_full = [float(x) for x in initial]
    impacts = impacts_full[:n_return]
    # Neutral: -NEUTRAL_THRESHOLD < impact < NEUTRAL_THRESHOLD; support: >=; oppose: <=
    n_benefited = sum(1 for x in impacts if x >= NEUTRAL_THRESHOLD)
    n_harmed = sum(1 for x in impacts if x <= -NEUTRAL_THRESHOLD)
    total = max(len(impacts), 1)

    # Add 1-5% random noise to support and oppose; neutral absorbs remainder so counts sum to total
    noise_pct = random.uniform(0.01, 0.05)
    n_benefited = max(0, round(n_benefited * (1 + random.uniform(-1, 1) * noise_pct)))
    n_harmed = max(0, round(n_harmed * (1 + random.uniform(-1, 1) * noise_pct)))
    n_benefited = min(n_benefited, total)
    n_harmed = min(n_harmed, total - n_benefited)
    n_neutral = total - n_benefited - n_harmed  # ensures support + oppose + neutral = total

    pct_support = round(100 * n_benefited / total, 1)
    pct_oppose = round(100 * n_harmed / total, 1)
    pct_neutral = round(100 * n_neutral / total, 1)

    # Top harmed (most negative) and top benefited (most positive)
    top_k = 5
    indexed = [(i, impacts[i]) for i in range(len(impacts))]
    indexed.sort(key=lambda x: x[1])
    most_harmed = [
        {
            "archetype_id": i,
            "net_impact": round(imp, 4),
            "description": (simulator.analyzer.archetype_descriptions[i]["description"])[:300],
        }
        for i, imp in indexed[:top_k]
    ]
    most_benefited = [
        {
            "archetype_id": i,
            "net_impact": round(imp, 4),
            "description": (simulator.analyzer.archetype_descriptions[i]["description"])[:300],
        }
        for i, imp in indexed[-top_k:][::-1]
    ]

    def _stance(imp: float) -> str:
        if imp >= NEUTRAL_THRESHOLD:
            return "support"
        if imp <= -NEUTRAL_THRESHOLD:
            return "oppose"
        return "neutral"

    def _profile(i: int) -> dict:
        if i < len(simulator.network.agents):
            a = simulator.network.agents[i]
            return {str(k): str(v) for k, v in a.attrs.items()}
        return {}

    agent_descriptions = [
        {
            "archetype_id": i,
            "description": (simulator.analyzer.archetype_descriptions[i]["description"])[:300],
            "stance": _stance(impacts[i]),
            "influence": calculate_influence(_profile(i)),
        }
        for i in range(n_return)
    ]

    return {
        "policy_text": results["policy_text"],
        "summary": {
            "n_agents": len(impacts),
            "n_benefited": n_benefited,
            "n_harmed": n_harmed,
            "n_neutral": n_neutral,
            "pct_support": pct_support,
            "pct_oppose": pct_oppose,
            "pct_neutral": pct_neutral,
        },
        "initial_damage": impacts,
        "agent_descriptions": agent_descriptions,
        "most_harmed": most_harmed,
        "most_benefited": most_benefited,
        "parameters": results["parameters"],
        "neutral_threshold": NEUTRAL_THRESHOLD,
    }


def build_test_analysis_response(policy_text: str, n_agents: int) -> dict:
    """Return deterministic test payload: 0% benefited, 75% harmed, 25% neutral, with 1-5% noise."""
    n_harmed_raw = int(round(n_agents * 0.75))
    n_benefited_raw = 0
    noise_pct = random.uniform(0.01, 0.05)
    n_harmed = max(0, min(n_agents, round(n_harmed_raw * (1 + random.uniform(-1, 1) * noise_pct))))
    n_benefited = max(0, min(n_agents - n_harmed, round(n_benefited_raw * (1 + random.uniform(-1, 1) * noise_pct))))
    n_neutral = n_agents - n_harmed - n_benefited

    impacts = ([-0.8] * n_harmed) + ([0.0] * n_neutral)

    most_harmed = [
        {
            "archetype_id": i,
            "net_impact": -0.8,
            "description": "Test harmed archetype for fast UI preview.",
        }
        for i in range(min(5, n_harmed))
    ]

    agent_descriptions = [
        {
            "archetype_id": i,
            "description": "Test harmed archetype for fast UI preview.",
            "stance": "oppose" if i < n_harmed else "neutral",
            "influence": 0.5,
        }
        for i in range(n_agents)
    ]

    return {
        "policy_text": policy_text,
        "summary": {
            "n_agents": n_agents,
            "n_benefited": n_benefited,
            "n_harmed": n_harmed,
            "n_neutral": n_neutral,
            "pct_support": 0.0,
            "pct_oppose": 75.0,
            "pct_neutral": 25.0,
        },
        "initial_damage": impacts,
        "agent_descriptions": agent_descriptions,
        "most_harmed": most_harmed,
        "most_benefited": [],
        "parameters": {
            "mode": "test_data",
            "target_distribution": "0/75/25",
        },
        "neutral_threshold": NEUTRAL_THRESHOLD,
    }


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    policy_text = (data.get("policy_text") or "").strip()
    n_agents = data.get("n_agents")
    if n_agents is not None:
        n_agents = int(n_agents)
        n_agents = max(10, min(n_agents, 10000))
    if not policy_text:
        return jsonify({"error": "policy_text is required and cannot be empty"}), 400

    try:
        if USE_TEST_DATA:
            return jsonify(build_test_analysis_response(policy_text, n_agents or NUM_NODES))

        simulator = get_simulator()
        results = simulator.run_simulation(
            policy_text=policy_text,
            impact_threshold=0.0,
            damage_scale=0.1,
            method="avneh",
            cascade_steps=1,
            accumulation="replace",
            top_k=20,
        )
        out = build_analysis_response(simulator, results, n_agents=n_agents)
        return jsonify(out)
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2026, debug=True)
