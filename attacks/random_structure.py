"""
Random Structure Attack — baseline poisoning attack.

Randomly adds and removes edges with equal probability up to a budget.
Serves as the lower-bound baseline: if the model is robust to random noise,
it should certainly be robust to smarter attacks (and vice versa).
"""
import numpy as np
from datasets.cora_loader import GraphData
from attacks.base import AttackResult, edge_budget, diff_edges
from utils.graph_utils import normalize_adjacency


def random_structure_attack(
    graph: GraphData,
    budget_ratio: float = 0.05,
    seed: int = 42,
) -> AttackResult:
    """
    Randomly flip edges up to budget (50% add, 50% remove).

    Args:
        graph:        Clean GraphData.
        budget_ratio: Fraction of existing edges to perturb.
        seed:         RNG seed for reproducibility.

    Returns:
        AttackResult with randomly perturbed adjacency.
    """
    rng    = np.random.default_rng(seed)
    adj    = graph.adj.copy()
    n      = adj.shape[0]
    budget = edge_budget(adj, budget_ratio)
    half   = budget // 2

    print(f"  [Random Structure] Budget={budget} "
          f"(+{half} add / -{budget-half} remove)")

    # Remove half the budget from existing edges
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    if len(rows) >= budget - half:
        chosen = rng.choice(len(rows), size=budget - half, replace=False)
        for idx in chosen:
            adj[rows[idx], cols[idx]] = 0.0
            adj[cols[idx], rows[idx]] = 0.0

    # Add the other half to non-existing entries
    added = 0
    attempts = 0
    while added < half and attempts < half * 100:
        i, j = rng.integers(0, n, size=2)
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
            added += 1
        attempts += 1

    perturbed = graph.copy()
    perturbed = perturbed.update_adj(adj)
    perturbed.name = "random_structure"

    a, r = diff_edges(graph.adj, adj)
    return AttackResult(
        perturbed_graph=perturbed,
        attack_name="Random Structure",
        n_edges_added=a,
        n_edges_removed=r,
        n_features_perturbed=0,
        budget_used=budget,
    )
