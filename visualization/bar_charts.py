"""
Bar charts — accuracy comparison across baseline / attacked / defended.
Produces Figure 1 of the paper.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


COLORS = {
    "baseline":  "#2196F3",   # blue
    "attacked":  "#F44336",   # red
    "defended":  "#4CAF50",   # green
}

ATTACK_LABELS = {
    "nettack":              "Nettack",
    "meta_attack":          "Meta Attack",
    "random_structure":     "Random\nStructure",
    "feature_perturbation": "Feature\nPerturbation",
    "edge_flip":            "Edge Flip",
    "gradient_attack":      "Gradient\nAttack (PGD)",
}


def plot_accuracy_bar(
    baseline_acc: float,
    attacked_accs: dict[str, float],
    defended_accs: dict[str, float],
    dataset_name: str = "Cora",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: for each attack show baseline / attacked / defended accuracy.

    Args:
        baseline_acc:  Single baseline accuracy value.
        attacked_accs: Dict {attack_name: accuracy}.
        defended_accs: Dict {attack_name: accuracy}.
        dataset_name:  Used in title and filename.
        save_path:     Directory to save figure; None = show only.
    """
    attacks = list(attacked_accs.keys())
    n = len(attacks)
    x = np.arange(n)
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 6))

    # Baseline bar (same height for all attacks)
    ax.bar(x - width, [baseline_acc] * n, width,
           color=COLORS["baseline"], alpha=0.85, label="Baseline")
    ax.bar(x,          [attacked_accs[a] for a in attacks], width,
           color=COLORS["attacked"], alpha=0.85, label="After Attack")
    ax.bar(x + width,  [defended_accs[a] for a in attacks], width,
           color=COLORS["defended"], alpha=0.85, label="After Defense")

    # Value labels on bars
    for i, atk in enumerate(attacks):
        for val, offset in [(baseline_acc, -width),
                            (attacked_accs[atk], 0),
                            (defended_accs[atk], +width)]:
            ax.text(i + offset, val + 0.008, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=45)

    # Target band
    ax.axhline(0.75, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.82, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.fill_between([-0.5, n - 0.5], 0.75, 0.82,
                    color="green", alpha=0.05, label="Target recovery zone")

    ax.set_xlabel("Attack", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"Accuracy Comparison — Baseline vs Attacked vs Defended ({dataset_name})",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fname = Path(save_path) / f"accuracy_bar_{dataset_name.lower()}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved → {fname}")

    return fig


def plot_metrics_grouped(
    metrics_table: dict[str, dict],     # {condition: {attack: {acc,f1,...}}}
    metric: str = "f1",
    dataset_name: str = "Cora",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Side-by-side grouped bars for a chosen metric across all attacks.
    metrics_table = {'baseline': {...}, 'attacked': {...}, 'defended': {...}}
    """
    attacks = list(metrics_table.get("attacked", {}).keys())
    n = len(attacks)
    x = np.arange(n)
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (cond, color) in enumerate(zip(["baseline", "attacked", "defended"],
                                           [COLORS["baseline"], COLORS["attacked"], COLORS["defended"]])):
        vals = [metrics_table[cond].get(a, {}).get(metric, 0) for a in attacks]
        ax.bar(x + (i - 1) * width, vals, width,
               color=color, alpha=0.85, label=cond.capitalize())

    ax.set_xlabel("Attack", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"{metric.upper()} Score Comparison ({dataset_name})",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fname = Path(save_path) / f"{metric}_bar_{dataset_name.lower()}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved → {fname}")

    return fig
