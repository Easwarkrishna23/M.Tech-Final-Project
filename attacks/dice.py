"""
DICE Attack — Delete Internally, Connect Externally.

Reference: Waniek et al., "Hiding Individuals and Communities in a Social Network",
Nature Human Behaviour, 2018.

Enhanced strategy (bridge-node targeting):
  Phase 1 — BRIDGE DELETION: identify high-betweenness-centrality nodes
    (structural bridges); prioritise deleting internal edges incident to them.
    Removing bridge edges maximally fragments community structure.
  Phase 2 — EXTERNAL CONNECTION: add cross-class edges incident to the same
    bridge nodes, injecting maximum cross-class noise at high-leverage positions.

Why betweenness centrality:
  Bridge nodes sit on many shortest paths between communities. Removing their
  internal edges breaks inter-community communication; adding external edges
  to them floods the graph with cross-class information at the most connected
  positions, disrupting GCN's homophily aggregation far more than random DICE.
"""
import numpy as np
import networkx as nx
from typing import Optional

from datasets.cora_loader import GraphData
from attacks.base import AttackResult, edge_budget, diff_edges


def dice_attack(
    graph: GraphData,
    model,
    params,
    budget_ratio: float = 0.35,
    seed: int = 42,
    bridge_fraction: float = 0.60,
) -> AttackResult:
    """
    DICE poisoning attack with bridge-node targeting.

    Args:
        graph:            Clean GraphData.
        model:            Trained GCN (used for predicted class labels).
        params:           Model parameters.
        budget_ratio:     Fraction of existing edges as total budget.
        seed:             RNG seed.
        bridge_fraction:  Fraction of budget spent on bridge-node edges.

    Returns:
        AttackResult with perturbed graph.
    """
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    adj = graph.adj.copy().astype(np.float32)
    n   = adj.shape[0]

    # Predicted class labels from clean model
    a_hat = jnp.array(graph.adj_norm)
    x     = jnp.array(graph.features)
    _, logits, _ = model.apply({"params": params}, x, a_hat, training=False)
    pred_labels = np.array(logits.argmax(axis=-1))

    total_budget = edge_budget(graph.adj, budget_ratio)
    half         = total_budget // 2

    # ── Betweenness centrality — find structural bridge nodes ────────────────
    G = nx.from_numpy_array(adj)
    print(f"  [DICE] Computing betweenness centrality (approx, k=300)...")
    bc = nx.betweenness_centrality(G, k=min(300, n), normalized=True, seed=int(seed))
    bc_arr = np.array([bc[i] for i in range(n)])
    # Top-20% highest-betweenness nodes are "bridge nodes"
    bc_threshold = np.percentile(bc_arr, 80)
    bridge_nodes = set(np.where(bc_arr >= bc_threshold)[0])
    print(f"  [DICE] {len(bridge_nodes)} bridge nodes identified "
          f"(bc ≥ {bc_threshold:.4f})")

    print(f"  [DICE] Budget={total_budget} edges "
          f"({budget_ratio:.0%} of {int(graph.adj.sum())//2}), "
          f"delete={half}, add={half}")

    rows, cols = np.where(np.triu(adj, k=1) > 0)

    # ── Step 1: DELETE internal edges — bridge-first ─────────────────────────
    internal_mask  = pred_labels[rows] == pred_labels[cols]
    bridge_mask    = np.array([(rows[i] in bridge_nodes or cols[i] in bridge_nodes)
                                for i in range(len(rows))])
    bridge_internal = np.where(internal_mask & bridge_mask)[0]
    other_internal  = np.where(internal_mask & ~bridge_mask)[0]

    n_bridge_del = min(int(half * bridge_fraction), len(bridge_internal))
    n_other_del  = min(half - n_bridge_del, len(other_internal))

    chosen_del = np.concatenate([
        rng.choice(bridge_internal, n_bridge_del, replace=False) if n_bridge_del > 0 else [],
        rng.choice(other_internal,  n_other_del,  replace=False) if n_other_del > 0  else [],
    ]).astype(int)
    for idx in chosen_del:
        i, j = rows[idx], cols[idx]
        adj[i, j] = 0.0
        adj[j, i] = 0.0

    # ── Step 2: ADD external edges — bridge-incident first ───────────────────
    bridge_list = np.array(sorted(bridge_nodes))
    n_added = 0
    # First pass: cross-class edges incident to bridge nodes
    bridge_budget = int(half * bridge_fraction)
    attempts = 0
    while n_added < bridge_budget and attempts < bridge_budget * 30:
        v = rng.choice(bridge_list)
        j = rng.integers(0, n)
        if (v != j and adj[v, j] == 0 and pred_labels[v] != pred_labels[j]):
            adj[v, j] = 1.0
            adj[j, v] = 1.0
            n_added += 1
        attempts += 1

    # Second pass: fill remaining budget with random cross-class edges
    attempts = 0
    while n_added < half and attempts < half * 20:
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if (i != j and adj[i, j] == 0 and pred_labels[i] != pred_labels[j]):
            adj[i, j] = 1.0
            adj[j, i] = 1.0
            n_added += 1
        attempts += 1

    if n_added < half:
        print(f"  [DICE] Added {n_added}/{half} external edges (graph density limit)")

    perturbed = graph.copy()
    perturbed = perturbed.update_adj(adj)
    perturbed.name = "dice"

    added, removed = diff_edges(graph.adj, adj)
    print(f"  [DICE] Done: +{added} external edges, -{removed} internal edges "
          f"(bridge-targeted)")

    return AttackResult(
        perturbed_graph=perturbed,
        attack_name="DICE",
        n_edges_added=added,
        n_edges_removed=removed,
        n_features_perturbed=0,
        budget_used=total_budget,
    )
