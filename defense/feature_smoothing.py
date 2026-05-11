"""
Defense Step 2 — Feature Smoothing.

Goal: Suppress adversarial noise injected into node features.

Method:
    X' = A_hat @ X

  One step of graph-based diffusion propagates each node's features
  as a weighted average of its neighbours' features.

Effect:
  - Isolated adversarial spikes in a single node's features get averaged
    out by clean neighbours (neighbourhood majority vote).
  - Nodes with many clean neighbours recover faster.
  - Works on both feature perturbation and gradient-based attacks.

Note: Uses the PRUNED adjacency (from Step 1), not the original.
This ensures adversarial edges do not participate in smoothing.
"""
import numpy as np
from datasets.cora_loader import GraphData
from utils.graph_utils import normalize_adjacency


def feature_smoothing(graph: GraphData) -> tuple[GraphData, dict]:
    """
    Apply one-step graph diffusion smoothing: X' = A_hat @ X.

    Args:
        graph: Graph after edge pruning (Step 1).

    Returns:
        (smoothed_graph, stats) with updated features.
    """
    a_hat   = graph.adj_norm          # already recomputed after pruning
    feats   = graph.features          # [N, F]

    feats_smoothed = a_hat @ feats    # [N, F]  — pure matrix multiply

    # Measure how much features changed (L2 norm of delta per node)
    delta     = feats_smoothed - feats
    mean_l2   = float(np.linalg.norm(delta, axis=1).mean())
    max_l2    = float(np.linalg.norm(delta, axis=1).max())

    stats = {
        "mean_feature_delta_l2": mean_l2,
        "max_feature_delta_l2":  max_l2,
    }
    print(f"  [Feature Smoothing] X' = A_hat @ X | "
          f"mean Δ‖x‖={mean_l2:.4f}, max Δ‖x‖={max_l2:.4f}")

    smoothed = graph.copy()
    smoothed = smoothed.update_features(feats_smoothed.astype(np.float32))
    smoothed.name = graph.name.replace("_pruned", "") + "_smoothed"
    return smoothed, stats
