"""
Embedding visualizations — t-SNE / UMAP of GCN hidden-layer representations.

Shows how the latent space changes across clean / attacked / defended conditions,
providing qualitative evidence that the defense restores class separability.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def _reduce(embeddings: np.ndarray, method: str = "tsne", seed: int = 42) -> np.ndarray:
    """Reduce N×D embeddings to N×2 using t-SNE or UMAP."""
    if method == "umap":
        if not HAS_UMAP:
            print("  [Embeddings] umap-learn not installed; falling back to t-SNE")
            method = "tsne"
        else:
            reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15,
                                min_dist=0.1, metric="cosine")
            return reducer.fit_transform(embeddings)

    if not HAS_TSNE:
        raise ImportError("scikit-learn required for t-SNE embeddings")

    import sklearn
    tsne_kwargs = dict(n_components=2, random_state=seed, perplexity=30,
                       learning_rate="auto", init="pca")
    # sklearn ≥1.2 renamed n_iter → max_iter
    sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if sk_version >= (1, 2):
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    reducer = TSNE(**tsne_kwargs)
    return reducer.fit_transform(embeddings)


def plot_embeddings_comparison(
    emb_clean: np.ndarray,
    emb_attacked: np.ndarray,
    emb_defended: np.ndarray,
    labels: np.ndarray,
    attack_name: str = "nettack",
    dataset_name: str = "Cora",
    method: str = "tsne",
    max_nodes: int = 2000,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Three-panel t-SNE/UMAP scatter: Clean / After Attack / After Defense.
    Each point is a node; color = ground-truth class label.

    Args:
        emb_*:     Node embeddings (N, D) from GCN layer-1 hidden activations.
        labels:    Ground-truth class labels (N,).
        method:    'tsne' or 'umap'.
        max_nodes: Subsample for speed if N > max_nodes.
    """
    n_classes = int(labels.max()) + 1
    cmap      = plt.cm.get_cmap("tab10", n_classes)

    # Subsample for large graphs
    n = len(labels)
    if n > max_nodes:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_nodes, replace=False)
    else:
        idx = np.arange(n)

    embs = [emb_clean[idx], emb_attacked[idx], emb_defended[idx]]
    lbls = labels[idx]

    print(f"  [Embeddings] Running {method.upper()} on {len(idx)} nodes × "
          f"{emb_clean.shape[1]}D embeddings …")

    reduced = [_reduce(e, method=method) for e in embs]

    titles = ["Clean Embeddings",
              f"After Attack ({attack_name})",
              "After Defense"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, coords, title in zip(axes, reduced, titles):
        for c in range(n_classes):
            mask = lbls == c
            if mask.sum() == 0:
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=12, alpha=0.6, color=cmap(c), label=f"Class {c}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"{method.upper()}-1", fontsize=9)
        ax.set_ylabel(f"{method.upper()}-2", fontsize=9)

    # Shared legend
    patches = [mpatches.Patch(color=cmap(c), label=f"Class {c}")
               for c in range(n_classes)]
    fig.legend(handles=patches, loc="lower center", ncol=n_classes,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"{method.upper()} Embeddings — {attack_name} — {dataset_name}",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fname = Path(save_path) / f"embeddings_{method}_{attack_name}_{dataset_name.lower()}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved → {fname}")

    return fig


def plot_embeddings_overlay(
    emb_clean: np.ndarray,
    emb_attacked: np.ndarray,
    emb_defended: np.ndarray,
    labels: np.ndarray,
    attack_name: str = "nettack",
    dataset_name: str = "Cora",
    method: str = "tsne",
    max_nodes: int = 1500,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Single-panel overlay: same t-SNE space for all three conditions.
    Useful for showing that defense restores cluster structure.
    Each condition gets a distinct marker shape; color = class label.
    """
    n_classes = int(labels.max()) + 1
    cmap      = plt.cm.get_cmap("tab10", n_classes)

    n = len(labels)
    if n > max_nodes:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_nodes, replace=False)
    else:
        idx = np.arange(n)

    lbls = labels[idx]

    # Fit t-SNE on concatenated embeddings so all three share one coordinate space
    all_embs = np.concatenate(
        [emb_clean[idx], emb_attacked[idx], emb_defended[idx]], axis=0
    )
    print(f"  [Embeddings Overlay] Running {method.upper()} on "
          f"{len(all_embs)} points …")
    all_2d = _reduce(all_embs, method=method)

    seg = len(idx)
    coords_clean    = all_2d[:seg]
    coords_attacked = all_2d[seg:2*seg]
    coords_defended = all_2d[2*seg:]

    markers = {"clean": "o", "attacked": "^", "defended": "s"}
    cond_data = {
        "clean":    coords_clean,
        "attacked": coords_attacked,
        "defended": coords_defended,
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    for cond, coords in cond_data.items():
        for c in range(n_classes):
            mask = lbls == c
            if mask.sum() == 0:
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=14, alpha=0.5, color=cmap(c),
                       marker=markers[cond])

    # Legend: class colors + condition markers
    class_patches = [mpatches.Patch(color=cmap(c), label=f"Class {c}")
                     for c in range(n_classes)]
    from matplotlib.lines import Line2D
    cond_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=7, label="Clean"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None",
               markersize=7, label="Attacked"),
        Line2D([0], [0], marker="s", color="gray", linestyle="None",
               markersize=7, label="Defended"),
    ]
    ax.legend(handles=class_patches + cond_handles,
              loc="best", fontsize=8, ncol=3)

    ax.set_title(
        f"{method.upper()} Overlay — {attack_name} ({dataset_name})\n"
        "○ Clean  △ Attacked  □ Defended",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fname = Path(save_path) / f"emb_overlay_{method}_{attack_name}_{dataset_name.lower()}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved → {fname}")

    return fig
