"""
Structural Defense Pipeline — strict order per PROMPT:

  Attacked Graph
       ↓
  Step 1: Edge Pruning        (remove low-cosine-sim edges)
       ↓
  Step 2: Feature Smoothing   (X' = A_hat @ X)
       ↓
  Step 3: Graph Reconstruction (k-NN merge)
       ↓
  Clean Graph

After reconstruction the GCN is retrained on the defended graph
and re-evaluated, giving the 'After Defense' metrics.
"""
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
import numpy as np
from flax import linen as nn

from datasets.cora_loader import GraphData
from defense.edge_pruning import edge_pruning
from defense.feature_smoothing import feature_smoothing
from defense.graph_reconstruction import graph_reconstruction
from utils.config import DefenseConfig, ModelConfig


@dataclass
class DefenseResult:
    defended_graph: GraphData
    pruning_stats: dict
    smoothing_stats: dict
    reconstruction_stats: dict
    defended_params: Any       # model retrained on defended graph


def run_defense(
    attacked_graph: GraphData,
    model: nn.Module,
    attack_type: str,               # 'poisoning' or 'evasion'
    attacked_params: Any,           # params from attacked evaluation
    defense_cfg: DefenseConfig,
    model_cfg: ModelConfig,
    seed: int = 42,
) -> DefenseResult:
    """
    Apply the full 3-step structural defense pipeline.

    For poisoning attacks: retrain on defended graph (same as attack eval).
    For evasion attacks:   reuse clean params (defense only fixes the graph).

    Args:
        attacked_graph:  Perturbed GraphData from Phase 4.
        model:           GCN/GAT module.
        attack_type:     'poisoning' or 'evasion'.
        attacked_params: Params used during attack evaluation.
        defense_cfg:     DefenseConfig thresholds.
        model_cfg:       ModelConfig for retraining.
        seed:            RNG seed.

    Returns:
        DefenseResult with clean graph and evaluation-ready params.
    """
    print(f"  [Defense Pipeline] Starting on {attacked_graph.name}")

    # Step 1: Edge Pruning
    pruned_graph, pruning_stats = edge_pruning(attacked_graph, defense_cfg)

    # Step 2: Feature Smoothing
    smoothed_graph, smoothing_stats = feature_smoothing(pruned_graph)

    # Step 3: Graph Reconstruction
    defended_graph, recon_stats = graph_reconstruction(smoothed_graph, defense_cfg)

    # Retrain on defended graph
    print(f"  [Defense] Retraining model on defended graph...")
    from models.train import train_model
    result = train_model(model, defended_graph, model_cfg,
                         seed=seed, verbose=False)
    print(f"  [Defense] Defended val acc: {result.best_val_acc:.4f}")

    return DefenseResult(
        defended_graph=defended_graph,
        pruning_stats=pruning_stats,
        smoothing_stats=smoothing_stats,
        reconstruction_stats=recon_stats,
        defended_params=result.best_params,
    )


def run_all_defenses(
    attack_results: dict,
    model: nn.Module,
    defense_cfg: DefenseConfig,
    model_cfg: ModelConfig,
    seed: int = 42,
    save_dir: Optional[Path] = None,
) -> dict[str, DefenseResult]:
    """
    Run defense pipeline for every attack result from Phase 4.

    Returns:
        Dict mapping attack_name → DefenseResult.
    """
    from attacks.runner import EvaluatedAttack
    defense_results = {}

    print(f"\n{'='*60}")
    print("PHASE 5 — Structural Defense Pipeline")
    print(f"{'='*60}")

    for attack_name, ea in attack_results.items():
        print(f"\n[Defense for: {attack_name}]")
        dr = run_defense(
            attacked_graph=ea.attack_result.perturbed_graph,
            model=model,
            attack_type=ea.attack_type,
            attacked_params=ea.eval_params,
            defense_cfg=defense_cfg,
            model_cfg=model_cfg,
            seed=seed,
        )
        defense_results[attack_name] = dr

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, dr in defense_results.items():
            g = dr.defended_graph
            np.savez(save_dir / f"defended_{name}.npz",
                     adj=g.adj, features=g.features, labels=g.labels,
                     train_mask=g.train_mask, val_mask=g.val_mask,
                     test_mask=g.test_mask)
        print(f"\n[Defense] Saved {len(defense_results)} defended graphs → {save_dir}")

    return defense_results
