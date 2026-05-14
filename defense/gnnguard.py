"""
Defense 1 — GNNGUARD Algorithm.

Reference: Zhang & Zitnik, "GNNGuard: Defending Graph Neural Networks
against Adversarial Attacks", NeurIPS 2020.

Algorithm:
  For each layer l of the GCN:
    1. Compute cosine similarity between representations h_u^(l) and h_v^(l)
       for every connected pair (u, v).
    2. Prune edges whose similarity < P0 (layer-wise threshold).
    3. Re-normalise the pruned adjacency and pass to layer l+1.

  This "layer-wise graph memory" means each GCN layer only aggregates over
  edges that are structurally coherent at that representation level, preventing
  adversarial edges from propagating corrupted signals across layers.

Simplification for this implementation:
  We approximate the layer-wise forward pass by:
    Layer 0 : use raw node features for similarity (pre-GCN)
    Layer 1 : use GCN layer-1 embeddings (from clean model forward pass)
  This gives a two-level importance estimate without requiring custom GCN surgery.
  The defended adjacency is the intersection of both pruned graphs.
"""
import numpy as np
import jax.numpy as jnp
from typing import Any

from datasets.cora_loader import GraphData
from utils.graph_utils import normalize_adjacency, compute_cosine_similarity, check_connectivity
from utils.config import GNNGuardConfig


def gnnguard_defense(
    attacked_graph: GraphData,
    model,
    params: Any,
    cfg: GNNGuardConfig,
) -> tuple[GraphData, dict]:
    """
    Apply GNNGUARD to the attacked graph.

    Steps:
      1. Compute cosine similarity at feature level (layer 0).
      2. Compute cosine similarity at embedding level (layer 1) using clean params.
      3. Prune edges below P0 at both levels (layer-wise graph memory).
      4. Final adjacency = intersection of both pruning decisions.
      5. Safety: restore minimum edges if too many removed.

    Args:
        attacked_graph: GraphData after attack.
        model:          GCN module (Flax).
        params:         Params from baseline (clean) training.
        cfg:            GNNGuardConfig with p0, min_edges_ratio, layer_wise.

    Returns:
        (defended_graph, stats)
    """
    adj   = attacked_graph.adj.copy().astype(np.float32)
    feats = attacked_graph.features
    n     = adj.shape[0]

    rows, cols = np.where(np.triu(adj, k=1) > 0)
    n_orig = len(rows)

    # ── Layer 0: feature-space cosine similarity ────────────────────────────
    sim_feat = compute_cosine_similarity(feats)
    edge_sim_feat = sim_feat[rows, cols]
    threshold_feat = float(np.percentile(edge_sim_feat, cfg.p0 * 100))
    keep_feat = edge_sim_feat >= threshold_feat

    print(f"  [GNNGUARD] Layer-0 (features): threshold={threshold_feat:.4f}, "
          f"pruning {(~keep_feat).sum()}/{n_orig} edges")

    if cfg.layer_wise and cfg.use_embedding_sim:
        # ── Layer 1: embedding-space cosine similarity ───────────────────────
        a_hat = jnp.array(attacked_graph.adj_norm)
        x_jax = jnp.array(feats)
        embeddings, _, _ = model.apply({"params": params}, x_jax, a_hat,
                                        training=False)
        emb_np = np.array(embeddings)          # [N, hidden_dim]

        sim_emb = compute_cosine_similarity(emb_np)
        edge_sim_emb = sim_emb[rows, cols]
        threshold_emb = float(np.percentile(edge_sim_emb, cfg.p0 * 100))
        keep_emb = edge_sim_emb >= threshold_emb

        print(f"  [GNNGUARD] Layer-1 (embeddings): threshold={threshold_emb:.4f}, "
              f"pruning {(~keep_emb).sum()}/{n_orig} edges")

        # Layer-wise graph memory: keep edge only if it passes BOTH levels
        keep = keep_feat & keep_emb
    else:
        keep = keep_feat

    # Apply pruning
    adj_pruned = adj.copy()
    for idx in range(len(rows)):
        if not keep[idx]:
            i, j = rows[idx], cols[idx]
            adj_pruned[i, j] = 0.0
            adj_pruned[j, i] = 0.0

    n_after = int(np.triu(adj_pruned, k=1).sum())
    n_removed = n_orig - n_after

    # ── Safety: restore minimum edges ───────────────────────────────────────
    min_keep = int(n_orig * cfg.min_edges_ratio)
    if n_after < min_keep:
        adj_pruned = _restore_min_edges(adj, adj_pruned, sim_feat, min_keep, rows, cols)
        n_after = int(np.triu(adj_pruned, k=1).sum())
        n_removed = n_orig - n_after
        print(f"  [GNNGUARD] Restored edges to meet min_ratio={cfg.min_edges_ratio:.0%}")

    # Connectivity guard
    if not check_connectivity(adj_pruned):
        adj_pruned = _restore_connectivity_mst(adj, adj_pruned, sim_feat)
        print(f"  [GNNGUARD] Restored MST edges for connectivity")

    stats = {
        "edges_before":  n_orig,
        "edges_after":   int(np.triu(adj_pruned, k=1).sum()),
        "edges_pruned":  n_removed,
        "prune_rate":    n_removed / max(n_orig, 1),
        "p0_threshold":  threshold_feat,
    }
    print(f"  [GNNGUARD] Final: {n_orig} → {stats['edges_after']} edges "
          f"({stats['prune_rate']:.1%} removed)")

    defended = attacked_graph.copy()
    defended = defended.update_adj(adj_pruned)
    defended.name = attacked_graph.name + "_gnnguard"
    return defended, stats


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _restore_min_edges(adj_orig, adj_pruned, sim, min_keep, rows, cols):
    adj_new = adj_pruned.copy()
    removed_idx = [i for i in range(len(rows))
                   if adj_new[rows[i], cols[i]] == 0 and adj_orig[rows[i], cols[i]] > 0]
    if not removed_idx:
        return adj_new
    sims = sim[rows[removed_idx], cols[removed_idx]]
    order = np.argsort(-sims)  # restore highest-sim edges first
    current = int(np.triu(adj_new, k=1).sum())
    for o in order:
        if current >= min_keep:
            break
        i, j = rows[removed_idx[o]], cols[removed_idx[o]]
        adj_new[i, j] = 1.0
        adj_new[j, i] = 1.0
        current += 1
    return adj_new


def _restore_connectivity_mst(adj_orig, adj_pruned, sim):
    import networkx as nx
    adj_new = adj_pruned.copy()
    G = nx.from_numpy_array(adj_new)
    components = list(nx.connected_components(G))
    while len(components) > 1:
        best_sim, best_i, best_j = -1.0, -1, -1
        for c1 in range(len(components) - 1):
            for u in components[c1]:
                for v in components[c1 + 1]:
                    if adj_orig[u, v] > 0 and sim[u, v] > best_sim:
                        best_sim, best_i, best_j = sim[u, v], u, v
        if best_i < 0:
            break
        adj_new[best_i, best_j] = 1.0
        adj_new[best_j, best_i] = 1.0
        G = nx.from_numpy_array(adj_new)
        components = list(nx.connected_components(G))
    return adj_new
