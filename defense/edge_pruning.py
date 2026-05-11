"""
Defense Step 1 — Edge Pruning.

Goal: Remove adversarial edges inserted by structural attacks.

Method:
  Compute cosine similarity between every pair of connected nodes.
  If sim(u, v) < threshold → the edge is likely adversarial → remove it.

  Adversarial edges tend to connect structurally dissimilar nodes
  (attacker bridges different communities to disrupt aggregation).
  Legitimate edges in citation/transaction networks connect similar nodes.

Safety constraint:
  Never remove more than (1 - min_edges_ratio) fraction of edges.
  If removing all low-sim edges would disconnect the graph, restore
  the minimum-weight edges needed to keep it connected.
"""
import numpy as np
from datasets.cora_loader import GraphData
from utils.graph_utils import (
    compute_cosine_similarity,
    normalize_adjacency,
    check_connectivity,
)
from utils.config import DefenseConfig


def edge_pruning(
    graph: GraphData,
    cfg: DefenseConfig,
) -> tuple[GraphData, dict]:
    """
    Remove low-similarity edges from the attacked graph.

    Uses percentile-based threshold (prune_percentile) rather than a fixed
    value — this is dataset-agnostic and avoids over-pruning on datasets
    like Cora whose BoW features have naturally low absolute cosine similarity.

    Args:
        graph: Attacked GraphData.
        cfg:   DefenseConfig with prune_percentile and min_edges_ratio.

    Returns:
        (pruned_graph, stats) where stats reports how many edges were removed.
    """
    adj  = graph.adj.copy()
    feats = graph.features
    n    = adj.shape[0]

    # Pairwise cosine similarity [N, N]
    sim = compute_cosine_similarity(feats)

    # Compute adaptive threshold from existing-edge similarity distribution
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    if len(rows) > 0 and cfg.prune_percentile > 0:
        edge_sims = sim[rows, cols]
        threshold = float(np.percentile(edge_sims, cfg.prune_percentile))
    else:
        threshold = cfg.cosine_threshold

    # Build pruning mask: keep edge (i,j) if sim >= threshold
    edge_mask = sim >= threshold
    adj_pruned = adj * edge_mask.astype(np.float32)
    adj_pruned = np.maximum(adj_pruned, adj_pruned.T)   # keep symmetric

    # Safety: do not remove too many edges
    orig_edges  = int(np.triu(adj, k=1).sum())
    pruned_edges = int(np.triu(adj_pruned, k=1).sum())
    min_keep    = int(orig_edges * cfg.min_edges_ratio)

    if pruned_edges < min_keep:
        # Restore lowest-similarity removed edges until we hit min_keep
        adj_pruned = _restore_min_edges(adj, adj_pruned, sim, min_keep)
        pruned_edges = int(np.triu(adj_pruned, k=1).sum())

    # Connectivity guard: restore MST edges if graph becomes disconnected
    if not check_connectivity(adj_pruned):
        adj_pruned = _restore_connectivity(adj, adj_pruned, sim)

    n_removed = orig_edges - int(np.triu(adj_pruned, k=1).sum())
    stats = {
        "edges_before": orig_edges,
        "edges_after":  int(np.triu(adj_pruned, k=1).sum()),
        "edges_removed": n_removed,
        "removal_rate": n_removed / max(orig_edges, 1),
    }
    print(f"  [Edge Pruning] p{cfg.prune_percentile}%→threshold={threshold:.4f} | "
          f"removed {n_removed}/{orig_edges} edges "
          f"({stats['removal_rate']:.1%})")

    pruned = graph.copy()
    pruned = pruned.update_adj(adj_pruned)
    pruned.name = graph.name + "_pruned"
    return pruned, stats


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _restore_min_edges(adj_orig: np.ndarray, adj_pruned: np.ndarray,
                        sim: np.ndarray, min_keep: int) -> np.ndarray:
    """Re-add lowest-sim pruned edges until edge count >= min_keep."""
    adj_new = adj_pruned.copy()
    rows, cols = np.where(np.triu(adj_orig - adj_pruned, k=1) > 0)
    if len(rows) == 0:
        return adj_new
    sims = sim[rows, cols]
    order = np.argsort(-sims)    # highest sim first among removed edges
    current = int(np.triu(adj_new, k=1).sum())
    for idx in order:
        if current >= min_keep:
            break
        i, j = rows[idx], cols[idx]
        adj_new[i, j] = 1.0
        adj_new[j, i] = 1.0
        current += 1
    return adj_new


def _restore_connectivity(adj_orig: np.ndarray,
                           adj_pruned: np.ndarray,
                           sim: np.ndarray) -> np.ndarray:
    """Add back edges from original graph to restore connectivity using BFS."""
    import networkx as nx
    adj_new = adj_pruned.copy()
    G = nx.from_numpy_array(adj_new)
    components = list(nx.connected_components(G))

    while len(components) > 1:
        # Find the best cross-component edge from original graph
        best_sim, best_i, best_j = -1.0, -1, -1
        for c_idx in range(len(components) - 1):
            src_nodes = list(components[c_idx])
            dst_nodes = list(components[c_idx + 1])
            for u in src_nodes:
                for v in dst_nodes:
                    if adj_orig[u, v] > 0 and sim[u, v] > best_sim:
                        best_sim = sim[u, v]
                        best_i, best_j = u, v
        if best_i < 0:
            break
        adj_new[best_i, best_j] = 1.0
        adj_new[best_j, best_i] = 1.0
        G = nx.from_numpy_array(adj_new)
        components = list(nx.connected_components(G))

    return adj_new
