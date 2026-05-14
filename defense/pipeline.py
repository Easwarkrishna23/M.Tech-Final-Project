"""
Dual Defense Pipeline — compares two defense philosophies side-by-side.

Defense 1 — GNNGUARD (Zhang & Zitnik, NeurIPS 2020):
  Neighbor importance estimation (cosine similarity), layer-wise graph memory,
  edge pruning based on similarity threshold P0. Retrain on pruned graph.

Defense 2 — Ontology-Driven Self-Healing:
  Semantic rules: CitationEdge with TopicSimilarity < 0.20 = SuspiciousEdge.
  Dynamic orchestration: Filtering → Feature Denoising (k-step) → Retraining.
  Plan generated automatically based on detected TopicMismatchVulnerability.

Both defenses are applied to every attacked graph and their results compared.
The pipeline reports accuracy, F1, and the new advanced metrics for each.
"""
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import numpy as np
from flax import linen as nn

from datasets.cora_loader import GraphData
from defense.gnnguard import gnnguard_defense
from defense.ontology_defense import ontology_self_healing
from defense.edge_pruning import edge_pruning
from defense.feature_smoothing import feature_smoothing
from defense.graph_reconstruction import graph_reconstruction
from utils.config import DefenseConfig, ModelConfig


@dataclass
class SingleDefenseResult:
    """Result from one defense strategy applied to one attack."""
    defended_graph: GraphData
    defense_stats: dict
    defended_params: Any         # params after retraining on defended graph
    defense_name: str


@dataclass
class DefenseResult:
    """Combined result: both defenses applied to one attacked graph."""
    gnnguard: SingleDefenseResult
    ontology:  SingleDefenseResult
    # Legacy 3-step result (kept for backward compatibility with visualizations)
    legacy:    Optional[SingleDefenseResult] = None

    @property
    def best(self) -> SingleDefenseResult:
        """Return the defense with higher retraining val accuracy."""
        g_acc = self.gnnguard.defense_stats.get("defended_val_acc", 0.0)
        o_acc = self.ontology.defense_stats.get("defended_val_acc", 0.0)
        return self.gnnguard if g_acc >= o_acc else self.ontology

    # Legacy .defended_params — returns best defense params for downstream code
    @property
    def defended_params(self) -> Any:
        return self.best.defended_params

    @property
    def defended_graph(self) -> GraphData:
        return self.best.defended_graph

    @property
    def pruning_stats(self) -> dict:
        return self.gnnguard.defense_stats

    @property
    def smoothing_stats(self) -> dict:
        return self.ontology.defense_stats

    @property
    def reconstruction_stats(self) -> dict:
        return {"best_defense": self.best.defense_name}


def _retrain_on_defended(
    defended_graph: GraphData,
    model: nn.Module,
    model_cfg: ModelConfig,
    seed: int,
    label: str,
) -> tuple[Any, float]:
    """Retrain GCN on defended graph, return (best_params, best_val_acc)."""
    from models.train import train_model
    result = train_model(model, defended_graph, model_cfg, seed=seed, verbose=False)
    print(f"  [{label}] Val acc after defense: {result.best_val_acc:.4f}")
    return result.best_params, result.best_val_acc


def run_defense(
    attacked_graph: GraphData,
    model: nn.Module,
    attack_type: str,
    attacked_params: Any,
    defense_cfg: DefenseConfig,
    model_cfg: ModelConfig,
    seed: int = 42,
    baseline_acc: float = 0.0,
    attacked_acc: float = 0.0,
    damage_threshold: float = 0.05,
    run_legacy: bool = False,
) -> DefenseResult:
    """
    Apply both GNNGUARD and Ontology Self-Healing defenses to the attacked graph.

    The damage_threshold gate applies only to the legacy pipeline (kept for
    comparison). Both new defenses always run — they have their own internal
    gates (GNNGUARD prunes only edges below P0; Ontology activates filtering
    only when mismatch ratio > alert threshold).

    Args:
        attacked_graph:   Perturbed GraphData from Phase 4.
        model:            GCN/GAT module.
        attack_type:      'poisoning' or 'evasion'.
        attacked_params:  Params from attacked evaluation.
        defense_cfg:      DefenseConfig (includes GNNGuardConfig, OntologyDefenseConfig).
        model_cfg:        ModelConfig for retraining.
        seed:             RNG seed.
        baseline_acc:     Clean model accuracy.
        attacked_acc:     Post-attack accuracy.
        damage_threshold: Legacy gate threshold.
        run_legacy:       Also run the legacy 3-step pipeline for comparison.

    Returns:
        DefenseResult containing both defense results.
    """
    damage = max(0.0, baseline_acc - attacked_acc)
    print(f"\n  [Defense Pipeline] {attacked_graph.name} "
          f"(damage={damage:.3f}, attack_type={attack_type})")

    # ── Defense 1: GNNGUARD ───────────────────────────────────────────────────
    print(f"  --- Defense 1: GNNGUARD ---")
    gnnguard_graph, gnnguard_stats = gnnguard_defense(
        attacked_graph, model, attacked_params, defense_cfg.gnnguard
    )
    gg_params, gg_val_acc = _retrain_on_defended(
        gnnguard_graph, model, model_cfg, seed, "GNNGUARD"
    )
    gnnguard_stats["defended_val_acc"] = gg_val_acc
    gnnguard_result = SingleDefenseResult(
        defended_graph=gnnguard_graph,
        defense_stats=gnnguard_stats,
        defended_params=gg_params,
        defense_name="GNNGUARD",
    )

    # ── Defense 2: Ontology Self-Healing ─────────────────────────────────────
    print(f"  --- Defense 2: Ontology Self-Healing ---")
    ontology_graph, ontology_stats = ontology_self_healing(
        attacked_graph, defense_cfg.ontology
    )
    ont_params, ont_val_acc = _retrain_on_defended(
        ontology_graph, model, model_cfg, seed, "Ontology"
    )
    ontology_stats["defended_val_acc"] = ont_val_acc
    ontology_result = SingleDefenseResult(
        defended_graph=ontology_graph,
        defense_stats=ontology_stats,
        defended_params=ont_params,
        defense_name="Ontology Self-Healing",
    )

    # ── Legacy 3-step pipeline (optional) ────────────────────────────────────
    legacy_result = None
    if run_legacy:
        print(f"  --- Defense 3: Legacy Pipeline (edge prune → smooth → k-NN) ---")
        pruned_graph, pruning_stats = edge_pruning(attacked_graph, defense_cfg)

        if damage >= damage_threshold:
            smoothed_graph, smoothing_stats = feature_smoothing(pruned_graph)
            final_graph, recon_stats = graph_reconstruction(smoothed_graph, defense_cfg)
        else:
            final_graph    = pruned_graph
            smoothing_stats = {"skipped": True}
            recon_stats     = {"skipped": True}

        leg_params, leg_val_acc = _retrain_on_defended(
            final_graph, model, model_cfg, seed, "Legacy"
        )
        legacy_result = SingleDefenseResult(
            defended_graph=final_graph,
            defense_stats={**pruning_stats,
                           "smoothing": smoothing_stats,
                           "reconstruction": recon_stats,
                           "defended_val_acc": leg_val_acc},
            defended_params=leg_params,
            defense_name="Legacy 3-Step",
        )

    # ── Recovery boost: if both defenses remain >2pp below baseline, retry
    # with a more aggressive ontology config (lower alert threshold, higher
    # sim cutoff, +3 denoising steps). This ensures baseline recovery even
    # against the strongest attacks.
    if (baseline_acc > 0
            and gg_val_acc < baseline_acc - 0.02
            and ont_val_acc < baseline_acc - 0.02):
        print(f"  [Recovery Boost] Both defenses below baseline "
              f"(best={max(gg_val_acc, ont_val_acc):.4f} < {baseline_acc:.4f}). "
              f"Retrying with stronger ontology config...")
        from utils.config import OntologyDefenseConfig
        boost_cfg = OntologyDefenseConfig(
            # Keep similarity threshold unchanged — raising it (e.g. 0.20→0.30)
            # flags >90% of Cora edges as suspicious and destroys the graph.
            # Only increase denoising depth and lower the alert trigger.
            topic_sim_threshold=defense_cfg.ontology.topic_sim_threshold,
            mismatch_alert_ratio=max(0.03, defense_cfg.ontology.mismatch_alert_ratio * 0.4),
            denoising_steps=min(7, defense_cfg.ontology.denoising_steps + 3),
            min_edges_ratio=defense_cfg.ontology.min_edges_ratio,
            adaptive_denoising=False,  # already at boosted k — don't add more
            max_denoising_steps=7,
        )
        ont_graph_b, _ = ontology_self_healing(attacked_graph, boost_cfg)
        ont_b_params, ont_b_acc = _retrain_on_defended(
            ont_graph_b, model, model_cfg, seed, "Ontology-Boost"
        )
        if ont_b_acc > max(gg_val_acc, ont_val_acc):
            print(f"  [Recovery Boost] Improved: "
                  f"{max(gg_val_acc, ont_val_acc):.4f} → {ont_b_acc:.4f}")
            ont_val_acc = ont_b_acc
            ontology_result = SingleDefenseResult(
                defended_graph=ont_graph_b,
                defense_stats={**ontology_result.defense_stats,
                               "defended_val_acc": ont_b_acc,
                               "boosted": True},
                defended_params=ont_b_params,
                defense_name="Ontology Self-Healing (Boosted)",
            )

    best = gnnguard_result if gg_val_acc >= ont_val_acc else ontology_result
    print(f"  [Defense Summary] Best defense: {best.defense_name} "
          f"(val_acc={best.defense_stats['defended_val_acc']:.4f})")

    return DefenseResult(
        gnnguard=gnnguard_result,
        ontology=ontology_result,
        legacy=legacy_result,
    )


def run_all_defenses(
    attack_results: dict,
    model: nn.Module,
    defense_cfg: DefenseConfig,
    model_cfg: ModelConfig,
    seed: int = 42,
    save_dir: Optional[Path] = None,
    baseline_acc: float = 0.0,
    attack_accs: Optional[dict] = None,
    damage_threshold: float = 0.05,
    run_legacy: bool = False,
) -> dict[str, DefenseResult]:
    """
    Run both defenses for every attack result from Phase 4.

    Returns:
        Dict mapping attack_name → DefenseResult.
    """
    from attacks.runner import EvaluatedAttack
    defense_results = {}
    attack_accs = attack_accs or {}

    print(f"\n{'='*60}")
    print("PHASE 5 — Dual Defense Pipeline (GNNGUARD + Ontology)")
    print(f"{'='*60}")

    for attack_name, ea in attack_results.items():
        print(f"\n[Defending against: {attack_name}]")
        dr = run_defense(
            attacked_graph=ea.attack_result.perturbed_graph,
            model=model,
            attack_type=ea.attack_type,
            attacked_params=ea.eval_params,
            defense_cfg=defense_cfg,
            model_cfg=model_cfg,
            seed=seed,
            baseline_acc=baseline_acc,
            attacked_acc=attack_accs.get(attack_name, 0.0),
            damage_threshold=damage_threshold,
            run_legacy=run_legacy,
        )
        defense_results[attack_name] = dr

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, dr in defense_results.items():
            for dname, single in [("gnnguard", dr.gnnguard), ("ontology", dr.ontology)]:
                g = single.defended_graph
                np.savez(
                    save_dir / f"defended_{name}_{dname}.npz",
                    adj=g.adj, features=g.features, labels=g.labels,
                    train_mask=g.train_mask, val_mask=g.val_mask,
                    test_mask=g.test_mask,
                )
        print(f"\n[Defense] Saved defended graphs → {save_dir}")

    return defense_results
