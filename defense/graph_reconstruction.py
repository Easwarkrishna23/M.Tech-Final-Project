"""
Defense Step 3 — Graph Reconstruction.

Goal: Rebuild graph topology using smoothed node features as a similarity signal,
replacing adversarially corrupted edges with structurally consistent ones.

Method:
  1. Compute pairwise cosine similarity on the SMOOTHED features.
  2. For each node, keep its top-k most similar neighbours (k-NN graph).
  3. Merge k-NN graph with the pruned adjacency (union):
       A_final = max(A_pruned, A_knn)
     This preserves legitimate existing edges while adding high-confidence
     new ones derived from feature similarity.

Why union instead of replacing?
  Replacing would discard legitimate edges that happen to connect nodes
  with moderate (not top-k) similarity. Union is conservative: we only
  add edges, never further remove them at this step.
"""
import numpy as np
from datasets.cora_loader import GraphData
from utils.graph_utils import build_knn_graph, normalize_adjacency
from utils.config import DefenseConfig


def graph_reconstruction(
    graph: GraphData,
    cfg: DefenseConfig,
) -> tuple[GraphData, dict]:
    """
    Rebuild graph by merging pruned adjacency with k-NN graph on smoothed features.

    Args:
        graph: Graph after feature smoothing (Step 2).
        cfg:   DefenseConfig with knn_k.

    Returns:
        (reconstructed_graph, stats).
    """
    feats     = graph.features         # smoothed features from Step 2
    adj_pruned = graph.adj             # pruned adjacency from Step 1

    # Build k-NN graph on smoothed features, merged with pruned adj
    adj_reconstructed = build_knn_graph(feats, k=cfg.knn_k,
                                         existing_adj=adj_pruned)

    edges_pruned = int(np.triu(adj_pruned, k=1).sum())
    edges_recon  = int(np.triu(adj_reconstructed, k=1).sum())
    edges_added  = edges_recon - edges_pruned

    stats = {
        "edges_before_reconstruction": edges_pruned,
        "edges_after_reconstruction":  edges_recon,
        "edges_added_by_knn":          edges_added,
        "knn_k": cfg.knn_k,
    }
    print(f"  [Graph Reconstruction] k={cfg.knn_k} | "
          f"edges: {edges_pruned} → {edges_recon} "
          f"(+{edges_added} from k-NN)")

    reconstructed = graph.copy()
    reconstructed = reconstructed.update_adj(adj_reconstructed)
    reconstructed.name = graph.name.replace("_smoothed", "") + "_defended"
    return reconstructed, stats
