"""
Feature Perturbation Attack — evasion attack at test time.

Adds bounded noise to node features. Applies to ALL nodes (untargeted)
or only test nodes depending on `targeted_test_only`.

Two noise modes:
  'uniform' — add uniform noise in [-ε, +ε]  (default)
  'gaussian' — add Gaussian noise with std=ε
"""
import numpy as np
from datasets.cora_loader import GraphData
from attacks.base import AttackResult


def feature_perturbation_attack(
    graph: GraphData,
    epsilon: float = 0.1,
    noise_mode: str = "uniform",
    test_only: bool = False,
    seed: int = 42,
) -> AttackResult:
    """
    Add bounded noise to node features (test-time evasion).

    Args:
        graph:      Clean GraphData.
        epsilon:    Noise magnitude bound.
        noise_mode: 'uniform' or 'gaussian'.
        test_only:  If True, only perturb test node features.
        seed:       RNG seed.

    Returns:
        AttackResult with perturbed features; adjacency unchanged.
    """
    rng   = np.random.default_rng(seed)
    feats = graph.features.copy()
    n, f  = feats.shape

    node_mask = graph.test_mask if test_only else np.ones(n, dtype=bool)
    n_perturbed = int(node_mask.sum())

    if noise_mode == "uniform":
        noise = rng.uniform(-epsilon, epsilon, size=(n_perturbed, f)).astype(np.float32)
    elif noise_mode == "gaussian":
        noise = rng.normal(0, epsilon, size=(n_perturbed, f)).astype(np.float32)
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}")

    feats[node_mask] = np.clip(feats[node_mask] + noise, 0.0, 1.0)

    print(f"  [Feature Perturbation] ε={epsilon}, mode={noise_mode}, "
          f"perturbed {n_perturbed} nodes")

    perturbed = graph.copy()
    perturbed = perturbed.update_features(feats)
    perturbed.name = "feature_perturbation"

    return AttackResult(
        perturbed_graph=perturbed,
        attack_name="Feature Perturbation",
        n_edges_added=0,
        n_edges_removed=0,
        n_features_perturbed=n_perturbed,
        budget_used=n_perturbed,
    )
