# MarGIN - Marginalization Generative Information Network

A policy-impact and marginalization-spread toolkit. It lets you (1) **predict how a described agent would react to a policy question** (beneficial vs. damaging), and (2) **simulate how marginalization or “damage” propagates** through a large population of **agents** (on the order of 10,000–100,000) connected by similarity and influence. Each entity in part 1 is an **agent**; the marginalization simulation is designed to run on tens of thousands of such agents (using archetypes was a temporary solution during development).

---

## What This Project Does

### 1. Policy answer prediction (beneficial vs. damaging)

Given:

- A **text description** of an **agent** (e.g. demographics, income, housing, health, employment), and  
- A **policy question** (e.g. “What impact would a universal healthcare expansion have on you?”),

the system outputs a **2D alignment vector** `[sim_to_beneficial, sim_to_damaging]` indicating how much the predicted “answer” aligns with “beneficial to me” vs “damaging to me.” Each such entity is an **agent**; the same agent representation is used to populate the marginalization simulation.

**How it works:**

- **Archetypes**: The project uses 100+ **socioeconomic dimensions** (income, education, health insurance, housing, immigration status, etc.) to define **archetype profiles**. Over 1,000 unique archetypes are generated, each with a profile and synthetic **policy Q&A** (e.g. 3 positive, 3 negative, 3 neutral, 1 chained-effect question).
- **Persona vectors**: From each archetype’s Q&A, a **384‑dimensional “persona” vector** is learned (via a small neural net that uses Cantor-paired question/answer embeddings). These are stored in `data/persona_vectors.json` and linked to archetype descriptions in `data/archetype_descriptions.json`.
- **Preprocessor**: A free-form **description** is embedded (e.g. with SentenceTransformer `all-MiniLM-L6-v2`). That embedding is matched by **cosine similarity** to archetype description embeddings; a **softmax-weighted blend** of the corresponding persona vectors is produced → one **384D persona blend** for that description.
- **Answer predictor**: A neural model (outer product of question and persona vectors, residual blocks, pooling, linear layer) **predicts an “answer” embedding** from (policy question embedding, persona blend). The predicted embedding is then projected onto **beneficial** and **damaging** reference directions (with damaging orthogonalized to beneficial) to get the final `[sim_to_beneficial, sim_to_damaging]`.

So: **description + policy question → persona blend → predicted answer embedding → [beneficial, damaging] alignment.**

### 2. Marginalization spread network

A second part of the project simulates **how “damage” or marginalization spreads** through a **large population of agents**—on the order of **10,000–100,000 agents**. (Using a smaller set of archetypes in the codebase was a temporary solution; the simulation is designed for this agent-scale population.)

- **Nodes**: Each node is an **agent** with:
  - **Vulnerability** (0–1): derived from profile dimensions (e.g. health insurance, housing stability, legal vulnerability, exposure to discrimination, financial resilience). Higher = more vulnerable.
  - **Influence** (dimension 101): how much this agent’s state affects others (high / moderate / low).
  - **Persona vector**: 384D, used to compute **similarity** between agents.
- **Edges**: Similarity between two agents is **cosine similarity** of their persona vectors, then scaled and optionally passed through a **logistic** so it sits in a usable band. Edge weight **A → B** = `influence[A] × f(similarity(A, B))`.
- **Effect**: The **effect on agent B** from current damage on all agents is:  
  `effect[B] = Σ_A ( damage[A] × vulnerability[B] × weight(A→B) )`, with damage/effect clipped to [-1, 1]. So damage spreads according to who is influential, who is similar, and who is vulnerable.
- **Propagation**: You can run **one step** (effect from current damage) or **multiple steps** (cascading), with options for how damage accumulates (e.g. replace vs add). The API can return the **most affected** agents or **all affected** with attributes (vulnerability, influence, incoming weight, etc.).

So: **10k–100k agents** (persona vectors + vulnerability + influence) → similarity & weights → given initial damage, compute effect and optionally cascade.

---

## Project structure

```
BISVHacks/
├── src/                      # Core logic
│   ├── archetype_definitions.py   # 100 socioeconomic dimensions and allowed values
│   ├── policy_templates.py       # Policy question templates and answer generation
│   ├── cantor.py                 # Cantor pairing (encode two ints → one) for the personality net
│   ├── personality_net.py        # NN: Q&A embeddings → persona vector (Cantor + layers + L2 norm)
│   ├── preprocessor.py          # Description embedding → weighted blend of persona vectors
│   ├── answer_predictor.py       # NN: (question_emb, persona_emb) → answer embedding
│   ├── predict_policy_answer.py  # High-level: description + policy question → [beneficial, damaging]
│   └── ...
├── network/
│   ├── network.py                # Marginalization network: load archetypes, similarity, weights, run effects/cascade
│   └── test_network.py           # Tests for the network
├── scripts/
│   ├── generate_archetypes.py    # Generate 1000+ archetypes with profiles and policy Q&A
│   ├── generate_archetype_descriptions.py  # Archetype descriptions (and embeddings) for matching
│   ├── generate_new_personas.py  # New persona descriptions + Q&A + embeddings (for training)
│   ├── extract_persona_vectors.py # Extract 384D persona vectors from archetype Q&A
│   ├── train_personality.py      # Train the personality net (Q&A → persona vector)
│   └── train_answer_predictor.py # Train answer predictor on (question, description, answer) triples
├── data/
│   ├── archetypes.json           # Archetype profiles and policy Q&A (generated)
│   ├── archetype_descriptions.json # Archetype descriptions and embeddings
│   ├── persona_vectors.json      # 384D persona vector per archetype
│   └── new_personas.json         # New personas for training the answer predictor
├── tests/
│   ├── test_preprocessor.py
│   ├── test_answer_predictor.py
│   └── test_predict_policy_answer.py
├── requirements.txt
└── README.md
```

---

## Setup and run

1. **Environment**  
   Create a virtual environment, then:

   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies: `torch`, `sentence-transformers`, `numpy`, `scikit-learn`, `faiss-cpu`, and others (see `requirements.txt`).

2. **Data and models**  
   The pipeline expects (or generates):

   - `data/archetypes.json` — from `scripts/generate_archetypes.py`
   - `data/archetype_descriptions.json` — from `scripts/generate_archetype_descriptions.py`
   - `data/persona_vectors.json` — from `scripts/extract_persona_vectors.py` (and training)
   - `data/answer_predictor.pt` — from `scripts/train_answer_predictor.py`

   For the answer predictor you may also use `data/new_personas.json` (from `scripts/generate_new_personas.py`).

3. **Policy answer from description + question**  
   From project root, with `src` on the path:

   ```python
   from src.predict_policy_answer import predict_policy_answer

   vec = predict_policy_answer(
       description="Single parent, part-time retail, renter, public health insurance.",
       policy_question="What impact would a universal healthcare expansion have on you?",
   )
   # vec = [sim_to_beneficial, sim_to_damaging]
   ```

4. **Marginalization network**  
   Use `network/network.py`:

   ```python
   from network.network import MarginalizationNetwork

   net = MarginalizationNetwork(
       archetypes_path="data/archetypes.json",
       persona_vectors_path="data/persona_vectors.json",
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

---

## Summary

- **Policy side**: Turn a **text description** of an **agent** and a **policy question** into a **beneficial vs. damaging** alignment by mapping the description to a blend of learned persona vectors and predicting an answer embedding, then projecting it onto beneficial/damaging axes.
- **Network side**: Simulate **how damage or marginalization propagates** across a **large population of agents** (10k–100k) using persona similarity, influence, and vulnerability, with one-step or multi-step propagation and helpers to list the most (or all) affected agents.

Together, this supports analyzing policy impact for described agents and studying spillover of marginalization across similar and vulnerable groups at scale. (The current code may still use archetype data as a temporary stand-in for the full agent population.)
