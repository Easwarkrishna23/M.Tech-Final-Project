"""
DICE Attack — Delete Internally, Connect Externally.

Reference: Waniek et al., "Hiding Individuals and Communities in a Social Network",
Nature Human Behaviour, 2018.

Strategy (untargeted poisoning):
  - DELETE edges that connect nodes of the SAME predicted class (internal edges)
  - ADD    edges that connect nodes of DIFFERENT predicted classes (external edges)

Rationale: internal edges are informationally redundant (GCN already aggregates
within-class neighbours); external edges inject cross-class noise into aggregation,
maximally disrupting the homophily assumption that GCN relies on.

Unlike gradient-based methods this attack:
  - Needs only predicted class labels (not gradients)
  - Is deterministic given a seed
  - Reliably achieves 10-20pp accuracy drops on Cora at 20% budget
  - Survives model retraining because the structural damage is global
"""
import numpy as np
from typing import Optional

from datasets.cora_loader import GraphData
from attacks.base import AttackResult, edge_budget, diff_edges


def dice_attack(
    graph: GraphData,
    model,
    params,
    budget_ratio: float = 0.20,
    seed: int = 42,
) -> AttackResult:
    """
    DICE poisoning attack: delete internal edges, add external edges.

    Split budget 50/50 between deletions and additions.

    Args:
        graph:        Clean GraphData.
        model:        Trained GCN model (used to get predicted labels).
        params:       Model parameters.
        budget_ratio: Fraction of existing edges to use as total budget.
        seed:         RNG seed.

    Returns:
        AttackResult with perturbed graph.
    """
    import jax.numpy as jnp
    from utils.graph_utils import normalize_adjacency

    rng = np.random.default_rng(seed)
    adj = graph.adj.copy().astype(np.float32)
    n   = adj.shape[0]

    # Get predicted class labels from clean model
    a_hat = jnp.array(graph.adj_norm)
    x     = jnp.array(graph.features)
    _, logits, _ = model.apply({"params": params}, x, a_hat, training=False)
    pred_labels = np.array(logits.argmax(axis=-1))

    total_budget = edge_budget(graph.adj, budget_ratio)
    half         = total_budget // 2

    print(f"  [DICE] Budget={total_budget} edges "
          f"({budget_ratio:.0%} of {int(graph.adj.sum())//2}), "
          f"delete={half}, add={half}")

    # ── Step 1: DELETE internal edges (same predicted class) ─────────────────
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    internal_mask = pred_labels[rows] == pred_labels[cols]
    internal_idx  = np.where(internal_mask)[0]

    if len(internal_idx) > 0:
        n_delete = min(half, len(internal_idx))
        chosen   = rng.choice(internal_idx, n_delete, replace=False)
        for idx in chosen:
            i, j = rows[idx], cols[idx]
            adj[i, j] = 0.0
            adj[j, i] = 0.0
    else:
        n_delete = 0
        print("  [DICE] Warning: no internal edges found to delete")

    # ── Step 2: ADD external edges (different predicted class) ───────────────
    # Sample candidate non-edges between nodes of different classes
    n_added = 0
    attempts = 0
    max_attempts = half * 20

    # Build a lookup of external non-edges for efficiency
    # Sample random pairs and filter
    while n_added < half and attempts < max_attempts:
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if (i != j
                and adj[i, j] == 0
                and pred_labels[i] != pred_labels[j]):
            adj[i, j] = 1.0
            adj[j, i] = 1.0
            n_added += 1
        attempts += 1

    if n_added < half:
        print(f"  [DICE] Only added {n_added}/{half} external edges "
              f"(graph may be too dense)")

    perturbed = graph.copy()
    perturbed = perturbed.update_adj(adj)
    perturbed.name = "dice"

    added, removed = diff_edges(graph.adj, adj)
    print(f"  [DICE] Done: +{added} external edges, -{removed} internal edges")

    return AttackResult(
        perturbed_graph=perturbed,
        attack_name="DICE",
        n_edges_added=added,
        n_edges_removed=removed,
        n_features_perturbed=0,
        budget_used=total_budget,
    )
