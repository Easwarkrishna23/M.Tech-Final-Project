"""
Main experiment pipeline.

Pipeline:
  Dataset (Static + Dynamic)
  → Baseline Training (GCN)
  → Baseline Evaluation
  → Apply Attacks (one-by-one)
  → Evaluate Attack Impact
  → Apply Structural Defense
  → Reconstruct Graph + Features
  → Retrain / Re-evaluate
  → Final Evaluation + Visualization
"""
import sys
import numpy as np
from pathlib import Path

from utils.config import cfg
from utils.metrics import classification_metrics, robustness_summary
from datasets import load_cora, load_elliptic
from models import create_gcn, create_gat, train_model, predict, save_params


def setup():
    """Seed everything and create output directories."""
    np.random.seed(cfg.seed)
    cfg.make_dirs()
    print(f"[Setup] Seed={cfg.seed}, results → {cfg.results_dir}")


def phase1_load_datasets():
    """Load Cora (static) and Elliptic Bitcoin (dynamic, 49 timesteps)."""
    print("\n" + "="*60)
    print("PHASE 1 — Dataset Loading")
    print("="*60)

    # Static: Cora
    cora = load_cora(cfg.data_dir)
    print(f"  Cora stats: {cora.stats()}")

    # Dynamic: Elliptic Bitcoin (49 real timesteps)
    elliptic = load_elliptic(cfg.data_dir)
    final_snap = elliptic.final_snapshot()
    print(f"  Elliptic final snapshot stats: {final_snap.stats()}")
    print(f"  Timesteps: {elliptic.num_timesteps}")

    return cora, elliptic


def phase3_baseline(cora, elliptic_snap):
    """Train GCN baseline on both datasets and report metrics."""
    print("\n" + "="*60)
    print("PHASE 3 — Baseline Training (GCN)")
    print("="*60)
    results = {}

    for name, graph in [("Cora", cora), ("Elliptic", elliptic_snap)]:
        print(f"\n[{name}] Training 2-layer GCN ...")
        model = create_gcn(
            hidden_dim=cfg.model.hidden_dim,
            num_classes=graph.num_classes,
            dropout_rate=cfg.model.dropout_rate,
        )
        result = train_model(model, graph, cfg.model, seed=cfg.seed)

        embeddings, preds, probs = predict(model, result.best_params, graph)
        metrics = classification_metrics(graph.labels, preds, mask=graph.test_mask)

        print(f"  [{name}] Test  → acc={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1']:.4f}  "
              f"precision={metrics['precision']:.4f}  "
              f"recall={metrics['recall']:.4f}")
        print(f"  [{name}] Best val acc={result.best_val_acc:.4f} @ epoch {result.best_epoch}")

        # Save checkpoint
        ckpt_path = str(cfg.checkpoints_dir / f"gcn_{name.lower()}_baseline")
        save_params(result.best_params, ckpt_path)

        results[name] = {
            "model": model,
            "params": result.best_params,
            "metrics": metrics,
            "embeddings": embeddings,
            "preds": preds,
            "train_result": result,
        }

    # Optional: GAT comparison on Cora
    print("\n[Cora] Training optional GAT for comparison...")
    gat = create_gat(hidden_dim=cfg.model.hidden_dim,
                     num_classes=cora.num_classes,
                     dropout_rate=0.6)
    gat_result = train_model(gat, cora, cfg.model, seed=cfg.seed)
    _, gat_preds, _ = predict(gat, gat_result.best_params, cora)
    gat_metrics = classification_metrics(cora.labels, gat_preds, mask=cora.test_mask)
    print(f"  [Cora/GAT] Test → acc={gat_metrics['accuracy']:.4f}  "
          f"f1={gat_metrics['f1']:.4f}")
    results["Cora_GAT"] = {"model": gat, "params": gat_result.best_params,
                            "metrics": gat_metrics}

    return results.get("Cora"), results.get("Elliptic")


def main():
    setup()

    # ── Phase 1: Load datasets ─────────────────────────────────────
    cora, elliptic = phase1_load_datasets()

    # ── Phase 3: Baseline training ─────────────────────────────────
    cora_results, elliptic_results = phase3_baseline(cora, elliptic.final_snapshot())

    # ── Phase 4-8: Attacks, defense, evaluation, visualization ─────
    print("\n[Phase 4-8] Not yet implemented — coming next")


if __name__ == "__main__":
    main()
