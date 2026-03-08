"""
Test the marginalization network with an example of NUM_NODES archetypes.
Uses the first NUM_NODES archetypes from archetypes.json / persona_vectors.json.
"""

from pathlib import Path

import numpy as np

from network import MarginalizationNetwork

# Number of archetypes (nodes) to use in the test. Change this to run with a different size.
NUM_NODES = 1000


def test_network() -> None:
    # JSON data lives in project root data/
    data_dir = Path(__file__).resolve().parent.parent / "data"
    net = MarginalizationNetwork(
        archetypes_path=data_dir / "archetypes.json",
        persona_vectors_path=data_dir / "persona_vectors.json",
        max_archetypes=NUM_NODES,
        logistic_k=1.0,
        logistic_x0=0.5,
    )
    net.load()

    assert net.n_nodes == NUM_NODES, f"Expected {NUM_NODES} nodes, got {net.n_nodes}"
    assert net.vulnerability.shape == (NUM_NODES,)
    assert net.influence.shape == (NUM_NODES,)
    assert net.similarity.shape == (NUM_NODES, NUM_NODES)
    assert net.weight.shape == (NUM_NODES, NUM_NODES)

    # Cosine similarity from first node to all others
    cos_from_0 = net.cosine_similarity_from_node(0)
    assert cos_from_0.shape == (NUM_NODES,)
    assert np.all(cos_from_0 >= 0) and np.all(cos_from_0 <= 1)
    assert np.isclose(cos_from_0[0], 1.0), "Node 0 vs self should be 1.0"

    # Least similar pair of nodes
    node_i, node_j, least_sim = net.least_similar_pair()
    assert node_i != node_j
    assert 0 <= least_sim <= 1

    # Effect is in [0, 1]
    damage = np.zeros(NUM_NODES, dtype=np.float64)
    damage[0] = 0.5
    effect = net.run_one_step(damage)
    assert effect.shape == (NUM_NODES,)
    assert np.all(effect >= 0) and np.all(effect <= 1), "Effect should be in [0, 1]"

    # Most affected returns top nodes
    # top_k = 15
    # top = net.most_affected(effect, top_k=top_k)
    # assert len(top) == top_k
    # assert all(isinstance(aid, int) and isinstance(val, (int, float)) for aid, val in top)
    # # Top effect should be >= others
    # effects_sorted = np.sort(effect)[::-1]
    # assert np.allclose([v for _, v in top], effects_sorted[:top_k])
    
    all_affected = net.all_affected(effect)
    assert len(all_affected) == NUM_NODES
    assert all(isinstance(aid, int) and isinstance(val, (int, float)) for aid, val in all_affected)
    assert np.allclose([v for _, v in all_affected], effect)

    # Per-node attributes: effect, vulnerability, influence, incoming_weight
    rows = net.all_affected_with_attributes(effect, damage)
    assert len(rows) == NUM_NODES
    assert all(
        set(r.keys()) == {"archetype_id", "effect", "vulnerability", "influence", "incoming_weight"}
        for r in rows
    )

    # Cascading: history shape (steps+1, N)
    cascade_steps = 3
    history = net.run_cascade(damage, steps=cascade_steps, accumulation="replace")
    assert history.shape == (cascade_steps + 1, NUM_NODES)
    assert np.all(history >= 0) and np.all(history <= 1)

    # Accumulation "add": damage can grow then cap at 1
    add_steps = 2
    history_add = net.run_cascade(damage, steps=add_steps, accumulation="add")
    assert history_add.shape == (add_steps + 1, NUM_NODES)
    assert np.all(history_add >= 0) and np.all(history_add <= 1)

    print("test_network: all checks passed.")
    print(f"  Nodes: {net.n_nodes}")
    print(f"  One-step from node 0: min effect = {effect.min():.4f}, max = {effect.max():.4f}")
    print()
    print(f"  Least similar pair: nodes ({node_i}, {node_j}) with scaled similarity = {least_sim:.6f}")
    print()
    print("  Cosine similarity (scaled: 0.999995->0, 1->1) from node 0 to all others:")
    print(f"  (range: min = {cos_from_0.min():.6f}, max = {cos_from_0.max():.6f})")
    print("  archetype_id  cos_sim")
    print("  " + "-" * 30)
    for j in range(min(20, NUM_NODES)):
        print(f"    {j:4d}         {cos_from_0[j]:.6f}")
    if NUM_NODES > 20:
        print("    ...")
        # Top 10 most similar to node 0
        order = np.argsort(-cos_from_0)
        top_sim = [(int(order[k]), float(cos_from_0[order[k]])) for k in range(min(10, NUM_NODES))]
        print("  Top 10 most similar to node 0:")
        for aid, sim in top_sim:
            print(f"    {aid:4d}         {sim:.6f}")
        # 10 least similar to node 0
        order_asc = np.argsort(cos_from_0)
        least_sim_list = [(int(order_asc[k]), float(cos_from_0[order_asc[k]])) for k in range(min(10, NUM_NODES))]
        print("  10 least similar to node 0:")
        for aid, sim in least_sim_list:
            print(f"    {aid:4d}         {sim:.6f}")
    print()
    print("  Per-node: archetype_id | effect | vulnerability | influence | incoming_weight")
    print("  " + "-" * 72)
    for r in rows[:25]:
        print(
            f"    {r['archetype_id']:4d}        | {r['effect']:.4f} | "
            f"{r['vulnerability']:.4f}         | {r['influence']:.4f}    | {r['incoming_weight']:.4f}"
        )
    if len(rows) > 25:
        print("    ...")
        top_by_effect = sorted(rows, key=lambda x: -x["effect"])[:10]
        print("  Top 10 by effect:")
        for r in top_by_effect:
            print(
                f"    {r['archetype_id']:4d}        | {r['effect']:.4f} | "
                f"{r['vulnerability']:.4f}         | {r['influence']:.4f}    | {r['incoming_weight']:.4f}"
            )
        least_affected = sorted(rows, key=lambda x: x["effect"])[:10]
        print("  10 least affected:")
        for r in least_affected:
            print(
                f"    {r['archetype_id']:4d}        | {r['effect']:.4f} | "
                f"{r['vulnerability']:.4f}         | {r['influence']:.4f}    | {r['incoming_weight']:.4f}"
            )


if __name__ == "__main__":
    test_network()
