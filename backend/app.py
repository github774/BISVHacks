"""
Flask backend for MarGIN policy analysis.
Exposes POST /api/analyze: send policy text, get simulation results from src.entry.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root (parent of backend/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Lazy-loaded simulator (heavy: sentence_transformers, sklearn, scipy load only on first request)
_simulator = None


def get_simulator():
    global _simulator
    if _simulator is None:
        from src.entry import PolicyImpactSimulator, NUM_NODES
        _simulator = PolicyImpactSimulator(
            n_agents=NUM_NODES,
            data_dir=ROOT / "data",
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )
        _simulator.load()
    return _simulator


def build_analysis_response(simulator, results: dict) -> dict:
    """Turn run_simulation results into JSON-friendly dict with summary and top lists."""
    initial = results["initial_damage"]
    n = len(initial)
    # Net impact: positive = benefit, negative = harm
    impacts = [float(x) for x in initial]
    n_benefited = sum(1 for x in impacts if x > 0)
    n_harmed = sum(1 for x in impacts if x < 0)
    n_neutral = n - n_benefited - n_harmed
    total = max(n, 1)
    pct_support = round(100 * n_benefited / total, 1)
    pct_oppose = round(100 * n_harmed / total, 1)
    pct_neutral = round(100 * n_neutral / total, 1)

    # Top harmed (most negative) and top benefited (most positive)
    top_k = 5
    indexed = [(i, impacts[i]) for i in range(n)]
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

    return {
        "policy_text": results["policy_text"],
        "summary": {
            "n_agents": n,
            "n_benefited": n_benefited,
            "n_harmed": n_harmed,
            "n_neutral": n_neutral,
            "pct_support": pct_support,
            "pct_oppose": pct_oppose,
            "pct_neutral": pct_neutral,
        },
        "initial_damage": impacts,
        "most_harmed": most_harmed,
        "most_benefited": most_benefited,
        "parameters": results["parameters"],
    }


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    policy_text = (data.get("policy_text") or "").strip()
    if not policy_text:
        return jsonify({"error": "policy_text is required and cannot be empty"}), 400

    try:
        simulator = get_simulator()
        results = simulator.run_simulation(
            policy_text=policy_text,
            impact_threshold=0.0,
            damage_scale=0.1,
            method="avneh",
            cascade_steps=0,
            accumulation="replace",
            top_k=20,
        )
        out = build_analysis_response(simulator, results)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
