"""
Random Structure Attack — baseline poisoning attack.

Enhanced strategy: high-betweenness node targeting.
  - Half the deletion budget targets edges incident to high-betweenness nodes
    (bridges between communities) — more disruptive than purely random removal.
  - Addition budget is still random cross-graph (preserves baseline character).
  - At 40% budget this now causes meaningful accuracy drops on GCN.
"""
import numpy as np
import networkx as nx
from datasets.cora_loader import GraphData
from attacks.base import AttackResult, edge_budget, diff_edges
from utils.graph_utils import normalize_adjacency


def random_structure_attack(
    graph: GraphData,
    budget_ratio: float = 0.40,
    seed: int = 42,
    centrality_fraction: float = 0.50,
) -> AttackResult:
    """
    Randomly flip edges up to budget with centrality-guided deletion.

    Args:
        graph:               Clean GraphData.
        budget_ratio:        Fraction of existing edges to perturb.
        seed:                RNG seed.
        centrality_fraction: Fraction of deletion budget spent on
                             high-betweenness edges (rest is random).

    Returns:
        AttackResult with perturbed adjacency.
    """
    rng    = np.random.default_rng(seed)
    adj    = graph.adj.copy()
    n      = adj.shape[0]
    budget = edge_budget(adj, budget_ratio)
    half   = budget // 2

    # ── Betweenness centrality for deletion targeting ─────────────────────
    G = nx.from_numpy_array(adj)
    print(f"  [Random Structure] Computing betweenness centrality (approx, k=300)...")
    bc = nx.betweenness_centrality(G, k=min(300, n), normalized=True, seed=int(seed))
    bc_arr  = np.array([bc[i] for i in range(n)])
    bc_thr  = np.percentile(bc_arr, 80)    # top-20% bridge nodes
    bridge_set = set(np.where(bc_arr >= bc_thr)[0])

    print(f"  [Random Structure] Budget={budget} "
          f"(+{half} add / -{budget-half} remove), "
          f"bridge-targeted deletion={centrality_fraction:.0%}")

    rows, cols = np.where(np.triu(adj, k=1) > 0)
    bridge_edge_mask = np.array([
        (rows[i] in bridge_set or cols[i] in bridge_set)
        for i in range(len(rows))
    ])
    bridge_idx = np.where(bridge_edge_mask)[0]
    other_idx  = np.where(~bridge_edge_mask)[0]

    n_bridge_del = min(int((budget - half) * centrality_fraction), len(bridge_idx))
    n_other_del  = min((budget - half) - n_bridge_del, len(other_idx))

    del_idx = np.concatenate([
        rng.choice(bridge_idx, n_bridge_del, replace=False) if n_bridge_del > 0 else [],
        rng.choice(other_idx,  n_other_del,  replace=False) if n_other_del > 0  else [],
    ]).astype(int)
    for idx in del_idx:
        adj[rows[idx], cols[idx]] = 0.0
        adj[cols[idx], rows[idx]] = 0.0

    # ── Random additions ──────────────────────────────────────────────────
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
