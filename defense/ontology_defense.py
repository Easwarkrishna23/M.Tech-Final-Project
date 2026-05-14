"""
Defense 2 — Ontology-Driven Self-Healing.

Conceptual model (mapped to Cora citation network):
  CitationEdge    : any edge (u, v) in the graph
  TopicSimilarity : cosine(features[u], features[v])  — BoW topic overlap
  SuspiciousEdge  : CitationEdge where TopicSimilarity < topic_sim_threshold
  TopicMismatchVulnerability : detected when fraction of SuspiciousEdges
                               exceeds mismatch_alert_ratio

Dynamic Orchestration:
  IF TopicMismatchVulnerability detected:
    Execute: Filtering → Feature Denoising → Retraining
  ELSE:
    Execute: Feature Denoising → Retraining   (lighter plan)

Plan steps:
  Step 1 — Filtering:
    Remove all SuspiciousEdges (ontology rule-based filtering).
    This is a per-class semantic decision, not a global percentile cut.

  Step 2 — Feature Denoising:
    X' = (A_hat)^k @ X   with k = denoising_steps.
    Multi-step diffusion averages over deeper neighbourhoods, suppressing
    localised adversarial feature spikes more aggressively than single-step
    smoothing. On clean-filtered edges this propagates ONLY legitimate
    topic-coherent information.

  Step 3 — Retraining:
    GCN retrained on filtered + denoised graph.
    Called by the pipeline after this module returns the defended graph.
"""
import numpy as np
from typing import Any

from datasets.cora_loader import GraphData
from utils.graph_utils import normalize_adjacency, check_connectivity
from utils.config import OntologyDefenseConfig


# ── Vulnerability detection ──────────────────────────────────────────────────

def detect_topic_mismatch_vulnerability(
    adj: np.ndarray,
    features: np.ndarray,
    cfg: OntologyDefenseConfig,
) -> dict:
    """
    Scan all edges for TopicMismatchVulnerability.

    Returns a dict with vulnerability flag, suspicious edge count/ratio,
    and indices for downstream filtering.
    """
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    n_edges = len(rows)
    if n_edges == 0:
        return {"vulnerable": False, "n_suspicious": 0, "ratio": 0.0,
                "suspicious_rows": np.array([]), "suspicious_cols": np.array([])}

    # TopicSimilarity = cosine(f_u, f_v)
    norms = np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-8)
    feats_normed = features / norms
    sims = (feats_normed[rows] * feats_normed[cols]).sum(axis=1)   # [E]

    suspicious_mask = sims < cfg.topic_sim_threshold
    n_suspicious = int(suspicious_mask.sum())
    ratio = n_suspicious / n_edges

    return {
        "vulnerable":      ratio >= cfg.mismatch_alert_ratio,
        "n_suspicious":    n_suspicious,
        "n_total_edges":   n_edges,
        "ratio":           ratio,
        "threshold":       cfg.topic_sim_threshold,
        "suspicious_rows": rows[suspicious_mask],
        "suspicious_cols": cols[suspicious_mask],
        "edge_sims":       sims,
    }


# ── Ontology filtering ───────────────────────────────────────────────────────

def ontology_filtering(
    adj: np.ndarray,
    features: np.ndarray,
    detection: dict,
    cfg: OntologyDefenseConfig,
) -> tuple[np.ndarray, dict]:
    """
    Remove all SuspiciousEdges (TopicSimilarity < threshold).

    Safety: preserve at least min_edges_ratio fraction of original edges.
    If filtering would go below floor, restore highest-sim suspicious edges.
    """
    adj_filtered = adj.copy()
    sus_rows = detection["suspicious_rows"]
    sus_cols = detection["suspicious_cols"]

    n_orig = int(np.triu(adj, k=1).sum())
    min_keep = int(n_orig * cfg.min_edges_ratio)

    # Sort suspicious edges by sim descending (remove lowest-sim first)
    sims = detection["edge_sims"]
    sus_sims = sims[
        # re-index: find positions of suspicious edges in full edge list
        np.where(
            np.isin(
                np.arange(len(detection.get("edge_sims", []))),
                _suspicious_indices(adj, detection)
            )
        )[0]
    ] if len(sus_rows) > 0 else np.array([])

    # Remove suspicious edges in order of ascending similarity
    # (most suspicious = lowest sim removed first)
    if len(sus_rows) > 0:
        # All suspicious removed first, then check floor
        for i, j in zip(sus_rows, sus_cols):
            adj_filtered[i, j] = 0.0
            adj_filtered[j, i] = 0.0

        n_after = int(np.triu(adj_filtered, k=1).sum())
        if n_after < min_keep:
            # Restore highest-sim suspicious edges until we hit floor
            sus_sim_vals = (features[sus_rows] / np.linalg.norm(features[sus_rows],
                            axis=1, keepdims=True).clip(min=1e-8) *
                            features[sus_cols] / np.linalg.norm(features[sus_cols],
                            axis=1, keepdims=True).clip(min=1e-8)).sum(axis=1)
            restore_order = np.argsort(-sus_sim_vals)
            for idx in restore_order:
                if n_after >= min_keep:
                    break
                i, j = sus_rows[idx], sus_cols[idx]
                adj_filtered[i, j] = 1.0
                adj_filtered[j, i] = 1.0
                n_after += 1

    n_final = int(np.triu(adj_filtered, k=1).sum())
    stats = {
        "edges_before":     n_orig,
        "edges_after":      n_final,
        "suspicious_edges": len(sus_rows),
        "edges_removed":    n_orig - n_final,
        "removal_rate":     (n_orig - n_final) / max(n_orig, 1),
    }
    return adj_filtered, stats


def _suspicious_indices(adj, detection):
    """Get position indices of suspicious edges in the upper-triangle edge list."""
    rows_all, cols_all = np.where(np.triu(adj, k=1) > 0)
    sus_set = set(zip(detection["suspicious_rows"].tolist(),
                      detection["suspicious_cols"].tolist()))
    indices = [i for i, (r, c) in enumerate(zip(rows_all, cols_all))
               if (r, c) in sus_set]
    return np.array(indices)


# ── Feature Denoising ────────────────────────────────────────────────────────

def multi_step_feature_denoising(
    adj_filtered: np.ndarray,
    features: np.ndarray,
    k: int = 3,
) -> tuple[np.ndarray, dict]:
    """
    Apply k-step graph diffusion: X' = (A_hat)^k @ X.

    Each step further averages each node's features over its neighbourhood,
    suppressing localised adversarial spikes that survived edge filtering.
    k=3 reaches ~3-hop neighbourhoods, providing stronger denoising than
    the single-step X' = A_hat @ X in the legacy pipeline.

    Args:
        adj_filtered: Filtered adjacency (after ontology step).
        features:     Node feature matrix [N, F].
        k:            Number of diffusion steps.

    Returns:
        (denoised_features, stats)
    """
    a_hat = normalize_adjacency(adj_filtered)
    x = features.copy().astype(np.float32)

    for step in range(k):
        x = a_hat @ x

    delta = x - features
    mean_l2 = float(np.linalg.norm(delta, axis=1).mean())
    max_l2  = float(np.linalg.norm(delta, axis=1).max())

    stats = {
        "denoising_steps": k,
        "mean_feature_delta_l2": mean_l2,
        "max_feature_delta_l2":  max_l2,
    }
    return x.astype(np.float32), stats


# ── Full Self-Healing Pipeline ───────────────────────────────────────────────

def ontology_self_healing(
    attacked_graph: GraphData,
    cfg: OntologyDefenseConfig,
) -> tuple[GraphData, dict]:
    """
    Full Ontology-Driven Self-Healing pipeline.

    Dynamically orchestrates:
      IF TopicMismatchVulnerability detected (ratio ≥ mismatch_alert_ratio):
        Filtering → Feature Denoising → (caller does Retraining)
      ELSE:
        Feature Denoising only → (caller does Retraining)

    Args:
        attacked_graph: GraphData after attack.
        cfg:            OntologyDefenseConfig.

    Returns:
        (defended_graph, stats)
    """
    adj   = attacked_graph.adj.copy()
    feats = attacked_graph.features.copy()

    # ── Vulnerability detection ───────────────────────────────────────────
    detection = detect_topic_mismatch_vulnerability(adj, feats, cfg)
    vulnerable = detection["vulnerable"]

    print(f"  [Ontology] Suspicious edges: {detection['n_suspicious']}"
          f"/{detection['n_total_edges']} "
          f"({detection['ratio']:.1%}, threshold={cfg.topic_sim_threshold})")

    all_stats = {"detection": detection}

    if vulnerable:
        print(f"  [Ontology] TopicMismatchVulnerability DETECTED → "
              f"Filtering + Feature Denoising + Retraining")

        # Step 1: Filtering
        adj_filtered, filter_stats = ontology_filtering(adj, feats, detection, cfg)
        all_stats["filtering"] = filter_stats
        print(f"  [Ontology] Filtering: {filter_stats['edges_before']} → "
              f"{filter_stats['edges_after']} edges "
              f"({filter_stats['removal_rate']:.1%} removed)")

        # Connectivity guard after filtering
        if not check_connectivity(adj_filtered):
            adj_filtered = _restore_conn(adj, adj_filtered, feats)
            print(f"  [Ontology] Restored connectivity after filtering")
    else:
        print(f"  [Ontology] No significant vulnerability → Feature Denoising only")
        adj_filtered = adj.copy()
        all_stats["filtering"] = {"skipped": True, "reason": "below alert threshold"}

    # Step 2: Feature Denoising (always applied)
    # Adaptive k: scale denoising strength with detected vulnerability ratio.
    # More suspicious edges = heavier attack = deeper neighbourhood averaging needed.
    k = cfg.denoising_steps
    if getattr(cfg, 'adaptive_denoising', True):
        ratio = detection.get("ratio", 0.0)
        max_k = getattr(cfg, 'max_denoising_steps', 7)
        if ratio > 0.30:
            k = min(max_k, cfg.denoising_steps + 3)
        elif ratio > 0.20:
            k = min(max_k, cfg.denoising_steps + 2)
        elif ratio > 0.10:
            k = min(max_k, cfg.denoising_steps + 1)
        if k != cfg.denoising_steps:
            print(f"  [Ontology] Adaptive denoising: k={cfg.denoising_steps} → {k} "
                  f"(ratio={ratio:.1%})")

    feats_denoised, denoise_stats = multi_step_feature_denoising(
        adj_filtered, feats, k=k
    )
    all_stats["denoising"] = denoise_stats
    print(f"  [Ontology] {k}-step diffusion: "
          f"mean Δ‖x‖={denoise_stats['mean_feature_delta_l2']:.4f}")

    all_stats["plan"] = ("Filtering + Feature Denoising + Retraining"
                         if vulnerable else "Feature Denoising + Retraining")

    defended = attacked_graph.copy()
    defended = defended.update_adj(adj_filtered)
    defended = defended.update_features(feats_denoised)
    defended.name = attacked_graph.name + "_ontology"
    return defended, all_stats


# ── Temporal Drift Detection & Self-Healing (Elliptic) ──────────────────────

def detect_temporal_drift(
    features_current: np.ndarray,
    features_prev: np.ndarray,
    drift_sigma_threshold: float = 2.5,
) -> dict:
    """
    Detect SuspiciousNodes: nodes with abnormally large feature drift
    between consecutive snapshots (timestep t-1 → t).

    Normal drift follows an approximately Gaussian distribution across nodes.
    SuspiciousNode: z_score[v] = (drift[v] - μ) / σ > drift_sigma_threshold.

    Temporal Perturbation attacks amplify the delta X_t - X_{t-1} for
    a fraction of nodes, making their drift z-scores far exceed the threshold.

    Args:
        features_current:      [N, F] node features at time t.
        features_prev:         [N, F] node features at time t-1.
        drift_sigma_threshold: Z-score cutoff for anomaly classification.

    Returns dict with vulnerable flag, suspicious node indices, drift stats.
    """
    n = min(features_current.shape[0], features_prev.shape[0])
    if n == 0:
        return {"vulnerable": False, "suspicious_nodes": np.array([]),
                "n_suspicious": 0, "ratio": 0.0, "drift_norms": np.array([])}

    fc = features_current[:n].astype(np.float32)
    fp = features_prev[:n].astype(np.float32)

    drift_norms = np.linalg.norm(fc - fp, axis=1)   # [N]

    # Use Median Absolute Deviation (MAD) for robustness against 40-50% contamination.
    # Mean/std break down when 30%+ of nodes are attacked; MAD has 50% breakdown point.
    median_drift = float(np.median(drift_norms))
    mad_drift    = float(np.median(np.abs(drift_norms - median_drift))) * 1.4826 + 1e-8
    z_scores     = (drift_norms - median_drift) / mad_drift

    suspicious_mask  = z_scores > drift_sigma_threshold
    suspicious_nodes = np.where(suspicious_mask)[0]

    return {
        "vulnerable":       len(suspicious_nodes) > 0,
        "suspicious_nodes": suspicious_nodes,
        "n_suspicious":     int(len(suspicious_nodes)),
        "ratio":            float(len(suspicious_nodes)) / max(n, 1),
        "mean_drift":       float(drift_norms.mean()),
        "median_drift":     median_drift,
        "mad_drift":        mad_drift,
        "max_drift":        float(drift_norms.max()),
        "threshold_sigma":  drift_sigma_threshold,
        "drift_norms":      drift_norms,
    }


def temporal_self_healing(
    attacked_graph: GraphData,
    prev_features: np.ndarray,
    cfg: OntologyDefenseConfig,
) -> tuple[GraphData, dict]:
    """
    Temporal Self-Healing: isolate nodes with abnormal temporal feature drift,
    then apply multi-step feature denoising on the cleaned graph.

    Steps:
      1. Detect SuspiciousNodes — nodes whose drift ||x_t - x_{t-1}|| is an
         outlier by more than drift_sigma_threshold standard deviations.
      2. Isolation — zero out adjacency rows/cols for suspicious nodes so they
         cannot inject corrupted features into their neighbors' aggregations.
      3. Feature Denoising — k-step graph diffusion on the isolated graph,
         propagating only features from trustworthy (non-suspicious) nodes.

    Isolation without removal: isolated nodes retain self-loop values so the
    GCN still produces predictions for them (with reduced influence on others).

    Args:
        attacked_graph: GraphData at time t (possibly perturbed).
        prev_features:  Feature matrix [N, F] from time t-1 (reference).
        cfg:            OntologyDefenseConfig (uses temporal_drift_sigma if set).

    Returns:
        (defended_graph, stats)
    """
    sigma = getattr(cfg, 'temporal_drift_sigma', 2.5)
    drift = detect_temporal_drift(attacked_graph.features, prev_features, sigma)

    adj   = attacked_graph.adj.copy()
    feats = attacked_graph.features.copy()

    print(f"  [Temporal Ontology] Suspicious nodes: {drift['n_suspicious']} "
          f"({drift['ratio']:.1%})  mean_drift={drift['mean_drift']:.4f}  "
          f"max_drift={drift['max_drift']:.4f}")

    all_stats = {
        "drift":  drift,
        "plan":   "Temporal Isolation + Feature Denoising + Retraining",
    }

    if drift["vulnerable"] and len(drift["suspicious_nodes"]) > 0:
        sus = drift["suspicious_nodes"]
        # Isolate: zero adjacency for suspicious nodes (do not remove self-loops)
        adj_isolated = adj.copy()
        adj_isolated[sus, :] = 0.0
        adj_isolated[:, sus] = 0.0
        print(f"  [Temporal Ontology] Isolated {len(sus)} suspicious nodes "
              f"from neighborhood aggregation")
        all_stats["isolated"] = int(len(sus))
    else:
        adj_isolated = adj
        print(f"  [Temporal Ontology] No suspicious drift detected — denoising only")
        all_stats["isolated"] = 0

    # Adaptive denoising: use more steps when more nodes are isolated
    k = cfg.denoising_steps
    if getattr(cfg, 'adaptive_denoising', True) and drift["ratio"] > 0.1:
        max_k = getattr(cfg, 'max_denoising_steps', 7)
        k = min(max_k, cfg.denoising_steps + max(1, int(drift["ratio"] * 10)))
        print(f"  [Temporal Ontology] Adaptive denoising k={k} (drift_ratio={drift['ratio']:.1%})")

    feats_denoised, denoise_stats = multi_step_feature_denoising(adj_isolated, feats, k=k)
    all_stats["denoising"] = denoise_stats

    defended = attacked_graph.copy()
    defended = defended.update_adj(adj_isolated)
    defended = defended.update_features(feats_denoised)
    defended.name = attacked_graph.name + "_temporal_ontology"
    return defended, all_stats


def _restore_conn(adj_orig, adj_filtered, features):
    """Restore connectivity using highest-similarity original edges."""
    import networkx as nx
    norms = np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-8)
    fn = features / norms
    adj_new = adj_filtered.copy()
    G = nx.from_numpy_array(adj_new)
    components = list(nx.connected_components(G))
    while len(components) > 1:
        best_sim, best_i, best_j = -1.0, -1, -1
        for c1 in range(len(components) - 1):
            for u in list(components[c1]):
                for v in list(components[c1 + 1]):
                    if adj_orig[u, v] > 0:
                        s = float(fn[u] @ fn[v])
                        if s > best_sim:
                            best_sim, best_i, best_j = s, u, v
        if best_i < 0:
            break
        adj_new[best_i, best_j] = 1.0
        adj_new[best_j, best_i] = 1.0
        G = nx.from_numpy_array(adj_new)
        components = list(nx.connected_components(G))
    return adj_new
