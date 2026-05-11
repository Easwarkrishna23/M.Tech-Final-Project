"""
Edge Flip Attack — evasion attack at test time.

Flips edges in the neighbourhood of test nodes.
Two strategies:
  'random'    — uniformly random edge flips near test nodes
  'degree'    — preferentially target high-degree test nodes (disrupts
                hub nodes whose neighbourhoods influence many predictions)
"""
import numpy as np
from datasets.cora_loader import GraphData
from attacks.base import AttackResult, edge_budget, diff_edges
from utils.graph_utils import normalize_adjacency


def edge_flip_attack(
    graph: GraphData,
    budget_ratio: float = 0.05,
    strategy: str = "random",
    seed: int = 42,
) -> AttackResult:
    """
    Flip edges adjacent to test nodes (test-time evasion).

    Args:
        graph:         Clean GraphData.
        budget_ratio:  Fraction of total edges to flip.
        strategy:      'random' or 'degree'.
        seed:          RNG seed.

    Returns:
        AttackResult with modified adjacency; features unchanged.
    """
    rng    = np.random.default_rng(seed)
    adj    = graph.adj.copy()
    n      = adj.shape[0]
    budget = edge_budget(adj, budget_ratio)
    test_nodes = np.where(graph.test_mask)[0]

    if strategy == "degree":
        degrees = adj.sum(axis=1)[test_nodes]
        probs   = degrees / degrees.sum()
        ordered = test_nodes[np.argsort(-degrees)]   # high-degree first
    else:
        ordered = rng.permutation(test_nodes)

    print(f"  [Edge Flip] Strategy={strategy}, budget={budget}")

    flipped = 0
    for v in ordered:
        if flipped >= budget:
            break

        # Candidates: all edges and non-edges incident to v
        neighbors     = np.where(adj[v] > 0)[0]
        non_neighbors = np.where(adj[v] == 0)[0]
        non_neighbors = non_neighbors[non_neighbors != v]

        # Remove one existing edge (if any)
        if len(neighbors) > 0 and flipped < budget:
            nb = rng.choice(neighbors)
            adj[v, nb] = 0.0
            adj[nb, v] = 0.0
            flipped += 1

        # Add one new edge (if any non-neighbors remain)
        if len(non_neighbors) > 0 and flipped < budget:
            nb = rng.choice(non_neighbors)
            adj[v, nb] = 1.0
            adj[nb, v] = 1.0
            flipped += 1

    perturbed = graph.copy()
    perturbed = perturbed.update_adj(adj)
    perturbed.name = "edge_flip"

    added, removed = diff_edges(graph.adj, adj)
    return AttackResult(
        perturbed_graph=perturbed,
        attack_name="Edge Flip",
        n_edges_added=added,
        n_edges_removed=removed,
        n_features_perturbed=0,
        budget_used=flipped,
    )
