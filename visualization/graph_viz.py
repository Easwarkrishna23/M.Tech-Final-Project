"""
Graph visualizations — draw original / attacked / defended subgraphs.
Nodes are colored by class label; added/removed edges are highlighted.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


def _adj_to_nx(adj: np.ndarray, labels: np.ndarray, node_ids=None):
    """Build a NetworkX graph from a dense adjacency matrix."""
    if not HAS_NX:
        raise ImportError("networkx required for graph_viz")
    G = nx.from_numpy_array(np.triu(adj, k=1))
    nx.set_node_attributes(G, {i: int(labels[i]) for i in range(len(labels))}, "label")
    if node_ids is not None:
        nx.set_node_attributes(G, {i: int(node_ids[i]) for i in range(len(node_ids))}, "node_id")
    return G


def _subgraph(G, center_nodes, hops: int = 2):
    """Return the k-hop ego subgraph around a set of center nodes."""
    nodes = set(center_nodes)
    for _ in range(hops):
        expand = set()
        for n in nodes:
            expand.update(G.neighbors(n))
        nodes.update(expand)
    return G.subgraph(nodes).copy()


def plot_graph_comparison(
    adj_clean: np.ndarray,
    adj_attacked: np.ndarray,
    adj_defended: np.ndarray,
    labels: np.ndarray,
    center_nodes: list,
    attack_name: str = "nettack",
    dataset_name: str = "Cora",
    hops: int = 2,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Three-panel figure: Original / After Attack / After Defense.
    Only renders a local subgraph (ego graph around center_nodes)
    to keep the plot legible.

    Added edges highlighted in red, removed edges in dashed gray.
    """
    if not HAS_NX:
        print("  [GraphViz] networkx not available — skipping graph comparison")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "networkx not installed", ha="center", transform=ax.transAxes)
        return fig

    n_classes = int(labels.max()) + 1
    cmap = plt.cm.get_cmap("tab10", n_classes)

    G_clean    = _adj_to_nx(adj_clean,    labels)
    G_attacked = _adj_to_nx(adj_attacked, labels)
    G_defended = _adj_to_nx(adj_defended, labels)

    # Build ego subgraph from clean graph; use same node set for all panels
    sub_clean = _subgraph(G_clean, center_nodes, hops)
    node_set  = list(sub_clean.nodes())
    sub_atk   = G_attacked.subgraph(node_set).copy()
    sub_def   = G_defended.subgraph(node_set).copy()

    # Fixed layout from clean graph
    pos = nx.spring_layout(sub_clean, seed=42, k=1.5 / np.sqrt(len(node_set)))

    node_colors = [cmap(sub_clean.nodes[n]["label"]) for n in node_set]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    titles   = ["Original (Clean)", f"After Attack\n({attack_name})", "After Defense"]
    subgraphs = [sub_clean, sub_atk, sub_def]

    clean_edges = set(sub_clean.edges())

    for ax, G_sub, title in zip(axes, subgraphs, titles):
        cur_edges = set(G_sub.edges())

        # Classify edges
        common  = clean_edges & cur_edges
        added   = cur_edges - clean_edges      # adversarial additions
        removed = clean_edges - cur_edges      # adversarial removals (shown as dashed)

        nx.draw_networkx_nodes(G_sub, pos, nodelist=node_set,
                               node_color=node_colors, node_size=80,
                               alpha=0.9, ax=ax)
        if common:
            nx.draw_networkx_edges(G_sub, pos, edgelist=list(common),
                                   width=0.8, alpha=0.4, edge_color="gray", ax=ax)
        if added:
            nx.draw_networkx_edges(G_sub, pos, edgelist=list(added),
                                   width=1.8, alpha=0.8, edge_color="red", ax=ax)
        if removed:
            nx.draw_networkx_edges(G_sub, pos, edgelist=list(removed),
                                   width=1.0, alpha=0.5, edge_color="gray",
                                   style="dashed", ax=ax)
        # Mark center nodes
        nx.draw_networkx_nodes(G_sub, pos, nodelist=center_nodes,
                               node_color="yellow", node_size=180,
                               edgecolors="black", linewidths=1.5, ax=ax)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Legend: class colors
    class_patches = [mpatches.Patch(color=cmap(c), label=f"Class {c}")
                     for c in range(n_classes)]
    extra_patches  = [
        mpatches.Patch(color="red",   label="Added edge"),
        mpatches.Patch(color="gray",  label="Removed edge (dashed)"),
        mpatches.Patch(color="yellow", label="Target node"),
    ]
    fig.legend(handles=class_patches + extra_patches,
               loc="lower center", ncol=n_classes + 3,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Graph Structure — {attack_name} — {dataset_name} "
        f"({len(node_set)} nodes, {hops}-hop neighborhood)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fname = Path(save_path) / f"graph_viz_{attack_name}_{dataset_name.lower()}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved → {fname}")

    return fig


def plot_degree_distribution(
    adj_clean: np.ndarray,
    adj_attacked: np.ndarray,
    adj_defended: np.ndarray,
    attack_name: str = "nettack",
    dataset_name: str = "Cora",
    save_path: Path | None = None,
) -> plt.Figure:
    """Overlay histogram of node degrees for clean / attacked / defended."""
    def degrees(adj):
        return np.array(adj.sum(axis=1)).flatten()

    d_clean   = degrees(adj_clean)
    d_attacked = degrees(adj_attacked)
    d_defended = degrees(adj_defended)

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(0, max(d_clean.max(), d_attacked.max(), d_defended.max()) + 1, 40)

    ax.hist(d_clean,    bins=bins, alpha=0.5, color="#2196F3", label="Clean")
    ax.hist(d_attacked, bins=bins, alpha=0.5, color="#F44336", label="Attacked")
    ax.hist(d_defended, bins=bins, alpha=0.5, color="#4CAF50", label="Defended")

    ax.set_xlabel("Node Degree", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Degree Distribution — {attack_name} ({dataset_name})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fname = Path(save_path) / f"degree_dist_{attack_name}_{dataset_name.lower()}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved → {fname}")

    return fig
