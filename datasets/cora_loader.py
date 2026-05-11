"""
Cora dataset loader (static graph).
Downloads via torch_geometric, converts to NumPy for framework-agnostic usage.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from utils.graph_utils import sparse_to_dense, normalize_adjacency, graph_stats


@dataclass
class GraphData:
    """Container for a single graph dataset."""
    adj: np.ndarray          # [N, N] dense adjacency (binary, symmetric)
    adj_norm: np.ndarray     # [N, N] normalized adjacency A_hat
    features: np.ndarray     # [N, F] node feature matrix
    labels: np.ndarray       # [N]    integer class labels
    train_mask: np.ndarray   # [N]    boolean
    val_mask: np.ndarray
    test_mask: np.ndarray
    num_nodes: int
    num_features: int
    num_classes: int
    name: str = "cora"

    def copy(self) -> "GraphData":
        """Deep copy — use this before applying attacks."""
        return GraphData(
            adj=self.adj.copy(),
            adj_norm=self.adj_norm.copy(),
            features=self.features.copy(),
            labels=self.labels.copy(),
            train_mask=self.train_mask.copy(),
            val_mask=self.val_mask.copy(),
            test_mask=self.test_mask.copy(),
            num_nodes=self.num_nodes,
            num_features=self.num_features,
            num_classes=self.num_classes,
            name=self.name,
        )

    def stats(self) -> dict:
        return graph_stats(self.adj)

    def update_adj(self, new_adj: np.ndarray) -> "GraphData":
        """Return a new GraphData with updated adjacency and recomputed A_hat."""
        d = self.copy()
        d.adj = new_adj.astype(np.float32)
        d.adj_norm = normalize_adjacency(new_adj)
        return d

    def update_features(self, new_features: np.ndarray) -> "GraphData":
        d = self.copy()
        d.features = new_features.astype(np.float32)
        return d


def load_cora(data_dir: str | Path = "data") -> GraphData:
    """
    Load the Cora citation network via torch_geometric.
    Returns a GraphData with numpy arrays.

    Cora:
      - 2708 nodes, 7 classes, 1433 features
      - Train: 140 nodes (20 per class)
      - Val: 500 nodes
      - Test: 1000 nodes
    """
    from torch_geometric.datasets import Planetoid
    import torch

    data_dir = Path(data_dir)
    dataset = Planetoid(root=str(data_dir / "Cora"), name="Cora")
    data = dataset[0]

    edge_index = data.edge_index.numpy()
    features = data.x.numpy().astype(np.float32)
    labels = data.y.numpy().astype(np.int64)
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    num_nodes = features.shape[0]
    adj = sparse_to_dense(edge_index, num_nodes)
    adj = np.maximum(adj, adj.T)   # ensure symmetry
    adj_norm = normalize_adjacency(adj)

    print(f"[Cora] Loaded: {num_nodes} nodes, {int(adj.sum())//2} edges, "
          f"{features.shape[1]} features, {int(labels.max())+1} classes")
    print(f"       Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    return GraphData(
        adj=adj,
        adj_norm=adj_norm,
        features=features,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
        num_features=features.shape[1],
        num_classes=int(labels.max()) + 1,
        name="cora",
    )
