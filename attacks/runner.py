"""
Attack runner — applies all 6 attacks one-by-one and collects results.

Rules enforced per PROMPT:
  - ONE attack at a time; each starts from the CLEAN graph
  - Poisoning attacks  → retrain model on poisoned graph, then evaluate
  - Evasion attacks    → keep clean model, evaluate on modified graph
  - Attacked graphs saved separately to disk
  - Results evaluated independently

Poisoning vs Evasion distinction:
  Poisoning (Nettack, Meta, Random Structure):
    corrupt training data → retrain GCN → evaluate retrained model on test nodes
  Evasion (Feature Perturbation, Edge Flip, Gradient Attack):
    keep clean trained model → modify graph at test time → evaluate
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import numpy as np
from flax import linen as nn

from datasets.cora_loader import GraphData
from attacks.base import AttackResult
from attacks.nettack import nettack
from attacks.meta_attack import meta_attack
from attacks.random_structure import random_structure_attack
from attacks.feature_perturbation import feature_perturbation_attack
from attacks.edge_flip import edge_flip_attack
from attacks.gradient_attack import gradient_attack
from attacks.dice import dice_attack
from utils.config import AttackConfig, ModelConfig


POISONING_ATTACKS = {"nettack", "meta_attack", "random_structure", "dice"}
EVASION_ATTACKS   = {"feature_perturbation", "edge_flip", "gradient_attack"}
ATTACK_NAMES      = list(POISONING_ATTACKS) + list(EVASION_ATTACKS)


@dataclass
class EvaluatedAttack:
    """Combines AttackResult with its evaluation model and metrics."""
    attack_result: AttackResult
    attack_type: str                      # 'poisoning' or 'evasion'
    eval_params: Any                      # params used for evaluation
    retrained: bool                       # True if model was retrained after attack


def run_all_attacks(
    graph: GraphData,
    model: nn.Module,
    clean_params: Any,
    attack_cfg: AttackConfig,
    model_cfg: ModelConfig,
    seed: int = 42,
    save_dir: Optional[Path] = None,
) -> dict[str, EvaluatedAttack]:
    """
    Apply all 6 attacks to a clean graph, one-by-one.

    Poisoning attacks trigger a full model retrain on the poisoned graph.
    Evasion attacks use the pre-trained clean model.

    Returns:
        Dict mapping attack_name → EvaluatedAttack (with correct eval params).
    """
    from models.train import train_model
    from models.gcn import create_gcn

    results: dict[str, EvaluatedAttack] = {}

    print(f"\n{'='*60}")
    print(f"PHASE 4 — Adversarial Attacks on {graph.name.upper()}")
    print(f"{'='*60}")

    # ── Poisoning Attack 1: Nettack (margin-based) ───────────────
    print("\n[Poisoning 1/4] Nettack (margin scoring)")
    r = nettack(graph, model, clean_params,
                n_perturbations=attack_cfg.nettack_n_perturbations,
                direct_attack=attack_cfg.nettack_direct)
    print(f"  {r.summary()}")
    poisoned_params = _retrain(r.perturbed_graph, model, model_cfg, seed, "Nettack")
    results["nettack"] = EvaluatedAttack(r, "poisoning", poisoned_params, retrained=True)

    # ── Poisoning Attack 2: DICE ──────────────────────────────────
    print("\n[Poisoning 2/4] DICE Attack")
    r = dice_attack(graph, model, clean_params,
                    budget_ratio=attack_cfg.meta_budget_ratio,
                    seed=seed)
    print(f"  {r.summary()}")
    poisoned_params = _retrain(r.perturbed_graph, model, model_cfg, seed, "DICE")
    results["dice"] = EvaluatedAttack(r, "poisoning", poisoned_params, retrained=True)

    # ── Poisoning Attack 3: Meta Attack (with inner-loop retrain) ─
    print("\n[Poisoning 3/4] Meta Attack (bilevel approx.)")
    r = meta_attack(graph, model, clean_params,
                    budget_ratio=attack_cfg.meta_budget_ratio,
                    n_steps=attack_cfg.meta_epochs)
    print(f"  {r.summary()}")
    poisoned_params = _retrain(r.perturbed_graph, model, model_cfg, seed, "Meta Attack")
    results["meta_attack"] = EvaluatedAttack(r, "poisoning", poisoned_params, retrained=True)

    # ── Poisoning Attack 4: Random Structure ──────────────────────
    print("\n[Poisoning 4/4] Random Structure Attack")
    r = random_structure_attack(graph,
                                budget_ratio=attack_cfg.random_budget_ratio,
                                seed=seed)
    print(f"  {r.summary()}")
    poisoned_params = _retrain(r.perturbed_graph, model, model_cfg, seed, "Random Structure")
    results["random_structure"] = EvaluatedAttack(r, "poisoning", poisoned_params, retrained=True)

    # ── Evasion Attack 1: Feature Perturbation ────────────────────
    print("\n[Evasion 1/3] Feature Perturbation Attack")
    r = feature_perturbation_attack(graph, epsilon=attack_cfg.feature_epsilon, seed=seed)
    print(f"  {r.summary()}")
    results["feature_perturbation"] = EvaluatedAttack(r, "evasion", clean_params, retrained=False)

    # ── Evasion Attack 2: Edge Flip ───────────────────────────────
    print("\n[Evasion 2/3] Edge Flip Attack")
    r = edge_flip_attack(graph, budget_ratio=attack_cfg.edge_flip_budget_ratio, seed=seed)
    print(f"  {r.summary()}")
    results["edge_flip"] = EvaluatedAttack(r, "evasion", clean_params, retrained=False)

    # ── Evasion Attack 3: Gradient-Based ─────────────────────────
    print("\n[Evasion 3/3] Gradient-Based Attack")
    r = gradient_attack(graph, model, clean_params,
                        epsilon=attack_cfg.grad_epsilon,
                        steps=attack_cfg.grad_steps)
    print(f"  {r.summary()}")
    results["gradient_attack"] = EvaluatedAttack(r, "evasion", clean_params, retrained=False)

    # ── Save attacked graphs ──────────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, ea in results.items():
            g = ea.attack_result.perturbed_graph
            np.savez(save_dir / f"{graph.name}_{name}.npz",
                     adj=g.adj, features=g.features, labels=g.labels,
                     train_mask=g.train_mask, val_mask=g.val_mask,
                     test_mask=g.test_mask)
        print(f"\n[Runner] Saved {len(results)} attacked graphs → {save_dir}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Retrain helper (poisoning evaluation)
# ──────────────────────────────────────────────────────────────────────────────

def _retrain(poisoned_graph: GraphData, model: nn.Module,
             model_cfg: ModelConfig, seed: int, attack_name: str) -> Any:
    """Retrain model on poisoned graph and return best params."""
    from models.train import train_model
    print(f"  [Retrain after {attack_name}] Training on poisoned graph...")
    result = train_model(model, poisoned_graph, model_cfg,
                         seed=seed, verbose=False)
    print(f"  [Retrain] Best val acc on poisoned graph: {result.best_val_acc:.4f}")
    return result.best_params
