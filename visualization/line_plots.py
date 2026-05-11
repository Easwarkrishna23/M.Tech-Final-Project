"""
Line plots:
  1. Attack vs Defense accuracy per attack (horizontal comparison)
  2. Temporal accuracy over Elliptic's 49 timesteps (clean / attacked / defended)
  3. Training loss/accuracy curves per model
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


COLORS = {
    "baseline":  "#2196F3",
    "attacked":  "#F44336",
    "defended":  "#4CAF50",
}

ATTACK_LABELS = {
    "nettack":              "Nettack",
    "meta_attack":          "Meta Attack",
    "random_structure":     "Random Structure",
    "feature_perturbation": "Feature Perturbation",
    "edge_flip":            "Edge Flip",
    "gradient_attack":      "Gradient Attack (PGD)",
}


def plot_attack_defense_line(
    baseline_acc: float,
    attacked_accs: dict[str, float],
    defended_accs: dict[str, float],
    dataset_name: str = "Cora",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Line plot showing accuracy trajectory: Baseline → Attacked → Defended
    for each attack. Each attack is one line connecting three points.
    """
    attacks = list(attacked_accs.keys())
    x_pos = [0, 1, 2]
    x_labels = ["Baseline", "After Attack", "After Defense"]

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.get_cmap("tab10", len(attacks))

    for idx, atk in enumerate(attacks):
        y = [baseline_acc, attacked_accs[atk], defended_accs[atk]]
        color = cmap(idx)
        ax.plot(x_pos, y, marker="o", linewidth=2, markersize=8,
                color=color, label=ATTACK_LABELS.get(atk, atk))
        # Annotate final defended value
        ax.annotate(f"{defended_accs[atk]:.3f}",
                    xy=(2, defended_accs[atk]),
                    xytext=(2.05, defended_accs[atk]),
                    fontsize=8, color=color, va="center")

    ax.axhline(baseline_acc, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.fill_between([0, 2], 0.75, 0.82, color="green", alpha=0.06,
                    label="Target recovery zone")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"Attack vs Defense — Accuracy Trajectory ({dataset_name})",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fname = Path(save_path) / f"attack_defense_line_{dataset_name.lower()}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved → {fname}")

    return fig


def plot_temporal_accuracy(
    timestep_accs: dict[str, list],   # {'baseline': [...], 'attacked': [...], 'defended': [...]}
    attack_name: str,
    dataset_name: str = "Elliptic",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Line plot of accuracy over 49 Elliptic timesteps for baseline/attacked/defended.
    Shows how robustness evolves over time — unique contribution of dynamic dataset.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    t = np.arange(1, len(timestep_accs["baseline"]) + 1)

    for cond, color in COLORS.items():
        if cond in timestep_accs:
            ax.plot(t, timestep_accs[cond], color=color,
                    linewidth=2, label=cond.capitalize(), alpha=0.85)
            ax.fill_between(t, timestep_accs[cond],
                            alpha=0.08, color=color)

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(
        f"Temporal Robustness — {ATTACK_LABELS.get(attack_name, attack_name)} "
        f"({dataset_name}, 49 timesteps)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlim(1, len(t))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fname = Path(save_path) / f"temporal_{attack_name}_{dataset_name.lower()}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved → {fname}")

    return fig


def plot_training_curves(
    train_losses: list,
    val_accs: list,
    model_name: str = "GCN",
    dataset_name: str = "Cora",
    save_path: Path | None = None,
) -> plt.Figure:
    """Training loss and validation accuracy curves."""
    epochs = np.arange(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, color="#E91E63", linewidth=1.5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training Loss")
    ax1.set_title(f"{model_name} — Training Loss ({dataset_name})")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_accs, color="#2196F3", linewidth=1.5)
    ax2.axhline(max(val_accs), color="green", linestyle="--",
                linewidth=1, label=f"Best: {max(val_accs):.4f}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation Accuracy")
    ax2.set_title(f"{model_name} — Val Accuracy ({dataset_name})")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fname = Path(save_path) / f"training_curves_{model_name.lower()}_{dataset_name.lower()}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved → {fname}")

    return fig
