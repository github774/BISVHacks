# MarGIN - Marginalization Generative Information Network

A policy-impact and marginalization-spread toolkit. It lets you (1) **predict how a described agent would react to a policy question** (beneficial vs. damaging), and (2) **simulate how marginalization or “damage” propagates** through a large population of **agents** (on the order of thousands) connected by similarity and influence. Each entity in part 1 is an **agent**; the marginalization simulation runs on a configurable number of such agents, using either generated agents or pre-built archetypes from data files.

---

## What This Project Does

### 1. Policy answer prediction (beneficial vs. damaging)

Given:

- A **text description** of an **agent** (e.g. demographics, income, housing, health, employment), and  
- A **policy question** (e.g. “What impact would a universal healthcare expansion have on you?”),

the system outputs a **2D alignment** via a **SentimentHead**: `[neg, pos]` (bad, good), indicating how much the predicted “answer” aligns with harmful vs. beneficial. The same agent representation is used to populate the marginalization simulation.

**How it works:**

- **Archetypes**: The project uses 100+ **socioeconomic dimensions** (income, education, health insurance, housing, immigration status, etc.) to define **archetype profiles**. Over 1,000 unique archetypes can be generated, each with a profile and synthetic **policy Q&A** (e.g. 3 positive, 3 negative, 3 neutral, 1 chained-effect question).
- **Persona vectors**: From each archetype’s Q&A, a **384‑dimensional “persona” vector** is learned (via a small neural net using Cantor-paired question/answer embeddings). These are stored in `data/persona_vectors.json` and linked to archetype descriptions in `data/archetype_descriptions.json`.
- **Preprocessor**: A free-form **description** is embedded (e.g. with SentenceTransformer `all-MiniLM-L6-v2`). That embedding is matched by **cosine similarity** to archetype description embeddings; a **softmax-weighted blend** of the corresponding persona vectors is produced → one **384D persona blend** for that description.
- **Answer predictor**: A neural model (outer product of question and persona vectors, residual blocks, pooling, linear layer) **predicts an “answer” embedding** from (policy question embedding, persona blend). A **SentimentHead** (384D → 2D) maps this embedding to **\[neg, pos]** (bad, good), giving the final alignment.

So: **description + policy question → persona blend → predicted answer embedding → SentimentHead → [bad, good] alignment.**

### 2. Marginalization spread network

A second part of the project simulates **how “damage” or marginalization spreads** through a **population of agents** (e.g. 1,000 by default, configurable).

- **Nodes**: Each node is an **agent** with:
  - **Vulnerability** (0–1): derived from profile dimensions (e.g. health insurance, housing stability, legal vulnerability, exposure to discrimination, financial resilience). Higher = more vulnerable.
  - **Influence** (dimension 101): how much this agent’s state affects others (high / moderate / low).
  - **Persona vector**: 384D, used to compute **similarity** between agents.
- **Edges**: Similarity between two agents is **cosine similarity** of their persona vectors, then scaled and optionally passed through a **logistic** so it sits in a usable band. Edge weight **A → B** = `influence[A] × f(similarity(A, B))`.
- **Effect**: The **effect on agent B** from current damage on all agents is:  
  `effect[B] = Σ_A ( damage[A] × vulnerability[B] × weight(A→B) )`, with damage/effect clipped to [-1, 1]. So damage spreads according to who is influential, who is similar, and who is vulnerable.
- **Propagation**: You can run **one step** (effect from current damage) or **multiple steps** (cascading), with options for how damage accumulates (e.g. replace vs add). The API can return the **most affected** agents or **all affected** with attributes (vulnerability, influence, incoming weight, etc.).

So: **agents** (persona vectors + vulnerability + influence) → similarity & weights → given initial damage, compute effect and optionally cascade.

### 3. Web application

- **Backend**: A **Flask** API (`backend/app.py`) exposes:
  - **POST /api/analyze** — send policy text, get simulation results (summary counts, most harmed/benefited archetypes, net impact).
  - **GET /api/health** — health check.
- **Frontend**: A **Next.js** app that embeds the **MARGIN - Global Inequality Digital Twin** UI (`frontend/public/equalcity.html`). The UI calls the backend at `http://localhost:2026` to run policy analysis and display results (support/oppose/neutral percentages, top harmed/benefited).
- **End-to-end flow**: `PolicyImpactSimulator` in `src/entry.py` loads the network from archetype data, analyzes policy (with method `"avneh"` using the answer predictor + SentimentHead for net impact), propagates damage through the network, and returns initial damage, cascade history, and top harmed/benefited archetypes.

---

## Project structure

```
BISVHacks/
├── src/                          # Core logic
│   ├── archetype_definitions.py  # Socioeconomic dimensions and allowed values
│   ├── policy_templates.py      # Policy question templates and answer generation
│   ├── cantor.py                # Cantor pairing for the personality net
│   ├── personality_net.py      # NN: Q&A embeddings → persona vector (Cantor + layers + L2 norm)
│   ├── preprocessor.py          # Description embedding → weighted blend of persona vectors
│   ├── answer_predictor.py      # NN: (question_emb, persona_emb) → answer embedding
│   ├── sentiment_head.py        # 384D embedding → [neg, pos] (bad, good) 2-class head
│   ├── predict_policy_answer.py # High-level: description + policy question → [beneficial, damaging] via SentimentHead
│   ├── genagents.py             # Generate or load agents (from archetypes JSON) for the network
│   ├── entry.py                 # PolicyAnalyzer, PolicyImpactSimulator; CLI entry for running simulations
│   └── ...
├── network/
│   ├── network.py               # MarginalizationNetwork: load/generate agents, similarity, weights, effects/cascade
│   └── test_network.py          # Tests for the network
├── backend/
│   └── app.py                   # Flask API: /api/analyze, /api/health (port 2026)
├── frontend/
│   ├── app/                     # Next.js app (page.tsx embeds equalcity.html)
│   └── public/
│       └── equalcity.html       # MARGIN - Global Inequality Digital Twin UI
├── scripts/
│   ├── generate_archetypes.py           # Generate 1000+ archetypes with profiles and policy Q&A
│   ├── generate_archetype_descriptions.py # Archetype descriptions (and embeddings) for matching
│   ├── generate_new_personas.py         # New persona descriptions + Q&A + embeddings (for training)
│   ├── extract_persona_vectors.py       # Extract 384D persona vectors from archetype Q&A
│   ├── train_personality.py             # Train the personality net (Q&A → persona vector)
│   ├── train_answer_predictor.py        # Train answer predictor on (question, description, answer) triples
│   ├── train_sentiment_head.py          # Train SentimentHead (384D → [neg, pos])
│   ├── answer_2d_stats.py               # Stats for 2D answer alignment
│   ├── add_influence.py                 # Add influence dimension to archetypes
│   └── ytconv.py                        # Utility script
├── data/
│   ├── archetypes.json                  # Archetype profiles and policy Q&A (generated)
│   ├── archetype_descriptions.json      # Archetype descriptions and embeddings
│   ├── persona_vectors.json             # 384D persona vector per archetype
│   ├── new_personas.json                # New personas for training the answer predictor
│   ├── policy.txt                       # Sample policy text for CLI simulation
│   ├── simulation_results.json         # Saved simulation output (from entry.py)
│   └── real_world_ratios.json           # Optional real-world ratios data
├── tests/
│   ├── test_preprocessor.py
│   ├── test_answer_predictor.py
│   ├── test_predict_policy_answer.py
│   ├── test_sentiment_head.py
│   ├── test_sentiment_sentences.py
│   └── ...
├── requirements.txt             # Python dependencies (torch, sentence-transformers, flask, flask-cors, etc.)
├── package.json                 # Root npm scripts (dev/build/start frontend via frontend/)
└── README.md
```

---

## Setup and run

### 1. Python environment and data

Create a virtual environment, then:

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `sentence-transformers`, `numpy`, `scikit-learn`, `faiss-cpu`, `flask`, `flask-cors`, and others (see `requirements.txt`).

The pipeline expects (or generates):

- `data/archetypes.json` — from `scripts/generate_archetypes.py`
- `data/archetype_descriptions.json` — from `scripts/generate_archetype_descriptions.py`
- `data/persona_vectors.json` — from `scripts/extract_persona_vectors.py` (and training)
- `data/answer_predictor.pt` — from `scripts/train_answer_predictor.py`
- `data/sentiment_head.pt` — from `scripts/train_sentiment_head.py` (optional; used for method `"avneh"`)

For the answer predictor you may also use `data/new_personas.json` (from `scripts/generate_new_personas.py`).

### 2. Policy answer from description + question

From project root, with `src` on the path:

```python
from src.predict_policy_answer import predict_policy_answer

vec = predict_policy_answer(
    description="Single parent, part-time retail, renter, public health insurance.",
    policy_question="What impact would a universal healthcare expansion have on you?",
)
# vec = [sim_to_beneficial, sim_to_damaging] (via SentimentHead [neg, pos])
```

### 3. Marginalization network (programmatic)

Use `network/network.py` with archetype data (e.g. from `entry.py`):

```python
from network.network import MarginalizationNetwork

net = MarginalizationNetwork(
    n_agents=1000,
    use_generated=False,
    use_archetypes_json=True,
    device="cpu",
    archetypes_path="data/archetypes.json",
    persona_vectors_path="data/persona_vectors.json",
    archetype_descriptions_path="data/archetype_descriptions.json",
)
net.load()

# One-step effect from some initial damage vector (one value per agent)
import numpy as np
damage = np.zeros(net.n_nodes)
damage[0] = 0.8  # Example: one node heavily damaged
effect = net.run_one_step(damage)

# Or cascade for multiple steps
history = net.run_cascade(damage, steps=3)
```

You can then use helpers like `most_affected(effect, top_k=20)` or `all_affected_with_attributes(...)` to inspect results.

### 4. End-to-end simulation (CLI)

Run a full simulation from a policy file:

```bash
# From project root (with data/policy.txt containing policy text)
python -m src.entry
```

This uses `PolicyImpactSimulator`: loads the network from archetypes, analyzes `data/policy.txt` with the chosen method (e.g. `"avneh"` for full model + SentimentHead), propagates damage, and can save results to `data/simulation_results.json`.

### 5. Web app (backend + frontend)

**Backend** (Flask, port 2026):

```bash
python backend/app.py
```

**Frontend** (Next.js):

```bash
npm install
npm run dev
```

Then open the frontend in the browser. The **MARGIN - Global Inequality Digital Twin** UI is served by the Next.js app and calls `http://localhost:2026` for **POST /api/analyze**. Set `USE_TEST_DATA = False` in `backend/app.py` to run real simulations; when `True`, the API returns fixed demo percentages for quick UI testing.

---

## Summary

- **Policy side**: Turn a **text description** of an **agent** and a **policy question** into a **beneficial vs. damaging** alignment by mapping the description to a blend of learned persona vectors, predicting an answer embedding, and passing it through the **SentimentHead** to get [neg, pos].
- **Network side**: Simulate **how damage or marginalization propagates** across a **population of agents** using persona similarity, influence, and vulnerability, with one-step or multi-step propagation and helpers to list the most (or all) affected agents.
- **App side**: **Flask** backend and **Next.js** frontend provide a web UI (**MARGIN - Global Inequality Digital Twin**) that runs policy analysis and displays support/oppose/neutral and top harmed/benefited archetypes.

Together, this supports analyzing policy impact for described agents and studying spillover of marginalization across similar and vulnerable groups, both programmatically and via the web interface.
