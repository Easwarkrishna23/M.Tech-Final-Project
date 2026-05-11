"""
Elliptic Temporal Evaluation — Phase 7 (dynamic dataset).

Runs the two most impactful attacks (Gradient/PGD + Feature Perturbation)
and the structural defense on the Elliptic final snapshot (t=49).

Also generates the temporal accuracy line plot showing robustness across
all 49 timesteps for the baseline model.

Outputs:
  results/figures/temporal_gradient_attack_elliptic.png
  results/figures/temporal_feature_perturbation_elliptic.png
  results/figures/accuracy_bar_elliptic.png
  results/tables/elliptic_results.md
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.config import cfg
from datasets.elliptic_loader import load_elliptic
from models.gcn import create_gcn
from models.train import train_model, predict, save_params, load_params
from utils.metrics import classification_metrics, recovery_rate
from attacks.gradient_attack import gradient_attack
from attacks.feature_perturbation import feature_perturbation_attack
from defense.pipeline import run_defense
from visualization.bar_charts import plot_accuracy_bar
from visualization.line_plots import plot_temporal_accuracy

import jax
import jax.numpy as jnp
from utils.graph_utils import normalize_adjacency


FIGURES_DIR = cfg.figures_dir
TABLES_DIR  = cfg.tables_dir


def _init_params(model, graph):
    a_hat = jnp.array(normalize_adjacency(graph.adj))
    x     = jnp.array(graph.features)
    key   = jax.random.PRNGKey(0)
    return model.init({"params": key, "dropout": key}, x, a_hat, training=False)["params"]


def train_or_load_elliptic(model, graph, tag="elliptic_baseline"):
    ckpt = cfg.checkpoints_dir / f"gcn_{tag}"
    ckpt_file = Path(str(ckpt) + ".npz")
    if ckpt_file.exists():
        print(f"  [Elliptic] Loading cached checkpoint: {ckpt_file}")
        template = _init_params(model, graph)
        return load_params(template, str(ckpt))
    print(f"  [Elliptic] Training GCN on final snapshot …")
    result = train_model(model, graph, cfg.model, seed=cfg.seed, verbose=False)
    save_params(result.best_params, str(ckpt))
    return result.best_params


def run_temporal_baseline(elliptic, model, save_path):
    """Evaluate baseline GCN across all 49 timesteps."""
    print("\n[Temporal] Evaluating baseline across 49 timesteps …")
    baseline_accs = []

    # Train once on snapshot 34 (end of training era)
    train_snap = elliptic.get_snapshot(33)   # 0-indexed → t=34
    params = train_or_load_elliptic(model, train_snap, tag="elliptic_train_era")

    for t, snap in enumerate(elliptic.snapshots):
        _, preds, _ = predict(model, params, snap)
        m = classification_metrics(snap.labels, np.array(preds), mask=snap.test_mask)
        baseline_accs.append(m["accuracy"])
        if (t + 1) % 10 == 0:
            print(f"  t={t+1:2d}  acc={m['accuracy']:.3f}")

    return baseline_accs, params


def run_attack_defense_on_final(elliptic, model, params, save_path):
    """Run PGD and Feature Perturbation + defense on the final snapshot (t=49)."""
    final_snap = elliptic.final_snapshot()
    _, clean_preds, _ = predict(model, params, final_snap)
    baseline_m = classification_metrics(
        final_snap.labels, np.array(clean_preds), mask=final_snap.test_mask
    )
    baseline_acc = baseline_m["accuracy"]
    print(f"\n[Elliptic final t=49] Baseline acc={baseline_acc:.4f}  "
          f"f1={baseline_m['f1']:.4f}")

    results = {}

    for attack_fn, attack_name, kwargs in [
        (gradient_attack, "gradient_attack",
         {"epsilon": cfg.attack.grad_epsilon, "steps": cfg.attack.grad_steps}),
        (feature_perturbation_attack, "feature_perturbation",
         {"epsilon": cfg.attack.feature_epsilon}),
    ]:
        print(f"\n  [{attack_name}] Attacking Elliptic final snapshot …")
        ar = attack_fn(final_snap, model, params, **kwargs)
        print(f"    {ar.summary()}")

        # Evasion: use clean params, evaluate on perturbed graph
        _, atk_preds, _ = predict(model, params, ar.perturbed_graph)
        atk_m = classification_metrics(
            final_snap.labels, np.array(atk_preds), mask=final_snap.test_mask
        )
        print(f"    After attack: acc={atk_m['accuracy']:.4f}  f1={atk_m['f1']:.4f}")

        # Defense
        dr = run_defense(
            attacked_graph=ar.perturbed_graph,
            model=model,
            attack_type="evasion",
            attacked_params=params,
            defense_cfg=cfg.defense,
            model_cfg=cfg.model,
            seed=cfg.seed,
            baseline_acc=baseline_acc,
            attacked_acc=atk_m["accuracy"],
            damage_threshold=0.05,
        )

        _, def_preds, _ = predict(model, dr.defended_params, dr.defended_graph)
        def_m = classification_metrics(
            final_snap.labels, np.array(def_preds), mask=final_snap.test_mask
        )
        rr = recovery_rate(baseline_acc, atk_m["accuracy"], def_m["accuracy"])
        print(f"    After defense: acc={def_m['accuracy']:.4f}  "
              f"f1={def_m['f1']:.4f}  recovery={rr:.1%}" if rr is not None
              else f"    After defense: acc={def_m['accuracy']:.4f}  f1={def_m['f1']:.4f}  recovery=N/A")

        results[attack_name] = {
            "attacked_acc": atk_m["accuracy"],
            "defended_acc": def_m["accuracy"],
            "recovery_rate": rr,
        }

    return baseline_acc, results


def build_temporal_lines(elliptic, model, params, baseline_accs,
                          attack_fn, attack_kwargs, save_path, attack_name):
    """Build per-timestep attacked + defended accuracy arrays and plot."""
    print(f"\n[Temporal] Building {attack_name} temporal lines …")
    attacked_accs = []
    defended_accs = []

    # Only process every 3rd timestep for speed, interpolate rest
    SAMPLE = 3
    sampled_t = list(range(0, elliptic.num_timesteps, SAMPLE))

    for t in sampled_t:
        snap = elliptic.get_snapshot(t)

        ar = attack_fn(snap, model, params, **attack_kwargs)
        _, atk_preds, _ = predict(model, params, ar.perturbed_graph)
        atk_m = classification_metrics(
            snap.labels, np.array(atk_preds), mask=snap.test_mask
        )

        dr = run_defense(
            attacked_graph=ar.perturbed_graph,
            model=model,
            attack_type="evasion",
            attacked_params=params,
            defense_cfg=cfg.defense,
            model_cfg=cfg.model,
            seed=cfg.seed,
            baseline_acc=baseline_accs[t],
            attacked_acc=atk_m["accuracy"],
            damage_threshold=0.05,
        )
        _, def_preds, _ = predict(model, dr.defended_params, dr.defended_graph)
        def_m = classification_metrics(
            snap.labels, np.array(def_preds), mask=snap.test_mask
        )

        attacked_accs.append(atk_m["accuracy"])
        defended_accs.append(def_m["accuracy"])
        print(f"  t={t+1:2d}  base={baseline_accs[t]:.3f}  "
              f"atk={atk_m['accuracy']:.3f}  def={def_m['accuracy']:.3f}")

    # Interpolate back to 49 timesteps
    full_atk = np.interp(range(elliptic.num_timesteps), sampled_t, attacked_accs).tolist()
    full_def = np.interp(range(elliptic.num_timesteps), sampled_t, defended_accs).tolist()

    plot_temporal_accuracy(
        timestep_accs={
            "baseline": baseline_accs,
            "attacked": full_atk,
            "defended": full_def,
        },
        attack_name=attack_name,
        dataset_name="Elliptic",
        save_path=save_path,
    )


def write_results_md(baseline_acc, attack_results, save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    fpath = save_path / "elliptic_results.md"
    lines = [
        "# Elliptic Bitcoin Dataset — Attack & Defense Results",
        "",
        f"**Dataset:** Elliptic Bitcoin (49 timesteps, final snapshot t=49)**",
        f"**Baseline accuracy:** {baseline_acc:.4f}",
        "",
        "## Attack & Defense on Final Snapshot (t=49)",
        "",
        "| Attack | After Attack | After Defense | Recovery Rate |",
        "| --- | --- | --- | --- |",
    ]
    for atk, m in attack_results.items():
        rr = m["recovery_rate"]
        rr_str = f"{rr:.1%}" if rr is not None else "N/A"
        lines.append(
            f"| {atk} | {m['attacked_acc']:.4f} | {m['defended_acc']:.4f} | {rr_str} |"
        )
    lines += [
        "",
        "## Notes",
        "- Evasion attacks only (gradient + feature); clean model used for evaluation.",
        "- Temporal line plots show per-timestep accuracy (sampled every 3 steps, interpolated).",
    ]
    fpath.write_text("\n".join(lines))
    print(f"  [Tables] Saved → {fpath}")


def main():
    cfg.make_dirs()

    print("=" * 60)
    print("PHASE 7 — Elliptic Temporal Evaluation")
    print("=" * 60)

    # Load dataset
    elliptic = load_elliptic(cfg.data_dir)
    final_snap = elliptic.final_snapshot()
    print(f"  Timesteps: {elliptic.num_timesteps}  "
          f"Final snapshot: {final_snap.stats()}")

    # Build model
    model = create_gcn(
        hidden_dim=cfg.model.hidden_dim,
        num_classes=final_snap.num_classes,
        dropout_rate=cfg.model.dropout_rate,
    )

    # Temporal baseline + params trained on era 1-34
    baseline_accs, params = run_temporal_baseline(
        elliptic, model, FIGURES_DIR
    )
    print(f"\n  Mean baseline acc across 49 timesteps: {np.mean(baseline_accs):.4f}")

    # Attack + defense on final snapshot
    baseline_acc, attack_results = run_attack_defense_on_final(
        elliptic, model, params, FIGURES_DIR
    )

    # Accuracy bar chart for Elliptic
    attacked_accs = {k: v["attacked_acc"] for k, v in attack_results.items()}
    defended_accs = {k: v["defended_acc"] for k, v in attack_results.items()}
    plot_accuracy_bar(
        baseline_acc=baseline_acc,
        attacked_accs=attacked_accs,
        defended_accs=defended_accs,
        dataset_name="Elliptic",
        save_path=FIGURES_DIR,
    )

    # Temporal line plots (sampled for speed)
    build_temporal_lines(
        elliptic, model, params, baseline_accs,
        attack_fn=gradient_attack,
        attack_kwargs={"epsilon": cfg.attack.grad_epsilon, "steps": cfg.attack.grad_steps},
        save_path=FIGURES_DIR,
        attack_name="gradient_attack",
    )
    build_temporal_lines(
        elliptic, model, params, baseline_accs,
        attack_fn=feature_perturbation_attack,
        attack_kwargs={"epsilon": cfg.attack.feature_epsilon},
        save_path=FIGURES_DIR,
        attack_name="feature_perturbation",
    )

    # Write markdown table
    write_results_md(baseline_acc, attack_results, TABLES_DIR)

    print("\n" + "=" * 60)
    print("[Phase 7 Complete]")
    print("=" * 60)


if __name__ == "__main__":
    main()
