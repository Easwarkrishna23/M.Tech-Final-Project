"""
Edge Flip Attack — evasion attack at test time.

Enhanced strategy (bridge-node targeting):
  'random'    — uniformly random edge flips near test nodes (baseline)
  'degree'    — prefer high-degree test nodes (existing behaviour)
  'bridge'    — rank test nodes by betweenness centrality; attack highest-
                betweenness test nodes first. Bridge-position test nodes sit
                on many shortest paths — removing/flipping their edges causes
                the maximum disruption to information flow at inference time.

Why betweenness beats degree:
  Degree counts local connections; betweenness measures global influence.
  A degree-10 bridge node may control information between two large communities,
  while a degree-30 hub in a clique affects only its immediate cluster.
"""
import numpy as np
import networkx as nx
from datasets.cora_loader import GraphData
from attacks.base import AttackResult, edge_budget, diff_edges
from utils.graph_utils import normalize_adjacency


def edge_flip_attack(
    graph: GraphData,
    budget_ratio: float = 0.35,
    strategy: str = "bridge",
    seed: int = 42,
) -> AttackResult:
    """
    Flip edges adjacent to test nodes (test-time evasion).

    Args:
        graph:         Clean GraphData.
        budget_ratio:  Fraction of total edges to flip.
        strategy:      'random', 'degree', or 'bridge'.
        seed:          RNG seed.

    Returns:
        AttackResult with modified adjacency; features unchanged.
    """
    rng    = np.random.default_rng(seed)
    adj    = graph.adj.copy()
    n      = adj.shape[0]
    budget = edge_budget(adj, budget_ratio)
    test_nodes = np.where(graph.test_mask)[0]

    if strategy == "bridge":
        G  = nx.from_numpy_array(adj)
        print(f"  [Edge Flip] Computing betweenness centrality (approx, k=300)...")
        bc = nx.betweenness_centrality(G, k=min(300, n), normalized=True, seed=int(seed))
        bc_test = np.array([bc[v] for v in test_nodes])
        ordered = test_nodes[np.argsort(-bc_test)]   # high-betweenness first
        print(f"  [Edge Flip] Bridge strategy: top bc={bc_test.max():.4f}")
    elif strategy == "degree":
        degrees = adj.sum(axis=1)[test_nodes]
        ordered = test_nodes[np.argsort(-degrees)]   # high-degree first
    else:
        ordered = rng.permutation(test_nodes)

    print(f"  [Edge Flip] Strategy={strategy}, budget={budget}")

    flipped = 0
    for v in ordered:
        if flipped >= budget:
            break

        neighbors     = np.where(adj[v] > 0)[0]
        non_neighbors = np.where(adj[v] == 0)[0]
        non_neighbors = non_neighbors[non_neighbors != v]

        # Remove one existing edge
        if len(neighbors) > 0 and flipped < budget:
            nb = rng.choice(neighbors)
            adj[v, nb] = 0.0
            adj[nb, v] = 0.0
            flipped += 1

        # Add one new cross-class edge (adversarial structural bottleneck bias).
        # Inserting a cross-class edge at a bridge node injects maximum
        # cross-class noise at a high-leverage position in the graph.
        if len(non_neighbors) > 0 and flipped < budget:
            v_label = int(graph.labels[v]) if graph.labels is not None else -1
            if v_label >= 0:
                cross = non_neighbors[
                    (graph.labels[non_neighbors] != v_label) &
                    (graph.labels[non_neighbors] >= 0)
                ]
                pool = cross if len(cross) > 0 else non_neighbors
            else:
                pool = non_neighbors
            nb = rng.choice(pool)
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
