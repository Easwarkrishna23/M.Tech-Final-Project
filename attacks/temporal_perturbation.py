"""
Temporal Perturbation Attack — breaks temporal consistency in time-series graphs.

Designed for the Elliptic Bitcoin Dataset (49 timesteps) but applicable to
any graph sequence where nodes have consistent feature trajectories over time.

Why standard feature perturbation fails on Elliptic:
  Applying ε uniformly to all features shifts every node equally. The
  classifier adapts because the class-conditional distributions shift together.
  The result: ~0% accuracy change (observed in baseline experiments).

This attack instead:
  Amplifies the TEMPORAL DELTA for a fraction of nodes:
    X_attacked[v] = X_t[v] + ε * (X_t[v] - X_{t-1}[v])

  This makes attacked nodes appear to have 'teleported' in feature space —
  their trajectory from t-1 to t is far larger than legitimate transactions.
  The shift is class-inconsistent (not all nodes shift the same way), so the
  classifier cannot simply re-normalise.

  When no previous snapshot is available (t=0), the attack uses per-feature
  standard deviation as a proxy for typical drift:
    δ[v] = std_col * sign(randn[v])
    X_attacked[v] = X_t[v] + ε * δ[v]

Why this beats simple ε-perturbation:
  - The temporal delta is heterogeneous across nodes (each node's delta
    is unique) → classifier cannot recalibrate with a global shift
  - The attack is detectable only by comparing X_t vs X_{t-1}, which is
    exactly the Temporal Drift Ontology's reasoning mechanism
  - Magnitude is data-driven: larger deltas in naturally drifting nodes
    are amplified further, increasing attack impact at those positions
"""
import numpy as np
from typing import Optional

from datasets.cora_loader import GraphData
from attacks.base import AttackResult


def temporal_perturbation_attack(
    graph: GraphData,
    prev_features: Optional[np.ndarray] = None,
    epsilon: float = 0.5,
    fraction: float = 0.40,
    seed: int = 42,
) -> AttackResult:
    """
    Temporal consistency attack for time-series graphs.

    Args:
        graph:          Current snapshot GraphData (time t).
        prev_features:  Feature matrix [N, F] from snapshot t-1.
                        If None, uses per-feature std as drift proxy.
        epsilon:        Amplification factor on the temporal delta.
        fraction:       Fraction of test nodes to attack (default 40%).
        seed:           RNG seed.

    Returns:
        AttackResult with perturbed features; adjacency unchanged.
    """
    rng   = np.random.default_rng(seed)
    feats = graph.features.copy().astype(np.float32)
    n, d  = feats.shape

    n_attacked = max(1, int(n * fraction))

    # ── Compute temporal delta per node ──────────────────────────────────────
    if prev_features is not None:
        n_ref = min(n, prev_features.shape[0])
        delta = np.zeros_like(feats)
        delta[:n_ref] = feats[:n_ref] - prev_features[:n_ref].astype(np.float32)
        src = f"temporal (X_t - X_{{t-1}}), n_ref={n_ref}"
    else:
        # Fallback: column-wise std × random sign
        feat_std = feats.std(axis=0) + 1e-8   # [D]
        signs    = rng.choice([-1.0, 1.0], size=(n, d)).astype(np.float32)
        delta    = feat_std[None, :] * signs
        src = "synthetic (std × sign)"

    # ── Select target nodes ──────────────────────────────────────────────────
    # Attack test-set nodes by descending delta magnitude — nodes with the
    # largest natural drift are amplified the most, maximising detection impact
    test_nodes = np.where(graph.test_mask)[0]
    if len(test_nodes) >= n_attacked:
        delta_norms = np.linalg.norm(delta[test_nodes], axis=1)
        order = np.argsort(-delta_norms)   # largest delta first
        targets = test_nodes[order][:n_attacked]
    else:
        # Fall back to all test nodes + random non-test nodes
        extra = rng.choice(n, n_attacked - len(test_nodes), replace=False)
        targets = np.concatenate([test_nodes, extra])

    print(f"  [Temporal Perturbation] ε={epsilon}, fraction={fraction:.0%}, "
          f"n_attacked={len(targets)}, delta_src={src}")
    if prev_features is not None:
        mean_drift = float(np.linalg.norm(delta[targets], axis=1).mean())
        print(f"  [Temporal Perturbation] Mean natural drift of targets: {mean_drift:.4f}")

    # ── Apply amplified perturbation ─────────────────────────────────────────
    feats[targets] = feats[targets] + epsilon * delta[targets]

    # Clip to a generous range around the original feature distribution
    feat_min = float(graph.features.min())
    feat_max = float(graph.features.max())
    feat_range = max(feat_max - feat_min, 1.0)
    feats = np.clip(feats, feat_min - feat_range, feat_max + feat_range)

    perturbed = graph.copy()
    perturbed = perturbed.update_features(feats)
    perturbed.name = "temporal_perturbation"

    feat_diff = int((feats != graph.features).any(axis=1).sum())

    return AttackResult(
        perturbed_graph=perturbed,
        attack_name="Temporal Perturbation",
        n_edges_added=0,
        n_edges_removed=0,
        n_features_perturbed=feat_diff,
        budget_used=len(targets),
    )
