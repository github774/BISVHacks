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
    base = Path(__file__).resolve().parent
    net = MarginalizationNetwork(
        archetypes_path=base / "archetypes.json",
        persona_vectors_path=base / "persona_vectors.json",
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

    # Effect is in [0, 1]
    damage = np.zeros(NUM_NODES, dtype=np.float64)
    damage[0] = 1
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
    print("  Top most affected (one-step):")
    for aid, val in all_affected:
        print(f"    archetype_id={aid}  effect={val:.4f}")


if __name__ == "__main__":
    test_network()
