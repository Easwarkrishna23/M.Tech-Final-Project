"""Core graph utility functions — all pure, JAX/NumPy compatible."""
import numpy as np
import scipy.sparse as sp
from typing import Tuple


def normalize_adjacency(adj: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Compute symmetric normalized adjacency: D^{-1/2} (A + I) D^{-1/2}."""
    if sp.issparse(adj):
        adj = adj.toarray().astype(np.float32)
    else:
        adj = adj.astype(np.float32)

    n = adj.shape[0]
    adj_hat = adj + np.eye(n, dtype=np.float32)   # A + I (self-loops)
    degree = adj_hat.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    # Element-wise scaling: (D^{-1/2} @ A_hat @ D^{-1/2})[i,j] = d[i]*A[i,j]*d[j]
    # Equivalent to D @ adj_hat @ D but O(N²) instead of O(N³).
    return d_inv_sqrt[:, None] * adj_hat * d_inv_sqrt[None, :]


def sparse_to_dense(edge_index: np.ndarray, num_nodes: int,
                    weighted: bool = False,
                    edge_weights: np.ndarray | None = None) -> np.ndarray:
    """Convert COO edge_index [2, E] to dense adjacency [N, N]."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    rows, cols = edge_index[0], edge_index[1]
    if weighted and edge_weights is not None:
        adj[rows, cols] = edge_weights.astype(np.float32)
    else:
        adj[rows, cols] = 1.0
    return adj


def dense_to_sparse(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert dense adjacency to COO (edge_index [2,E], edge_weights [E])."""
    rows, cols = np.nonzero(adj)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_weights = adj[rows, cols].astype(np.float32)
    return edge_index, edge_weights


def check_connectivity(adj: np.ndarray) -> bool:
    """Return True if the graph is connected (undirected check)."""
    import networkx as nx
    G = nx.from_numpy_array(adj)
    return nx.is_connected(G)


def largest_connected_component(adj: np.ndarray,
                                 features: np.ndarray,
                                 labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return LCC subgraph and re-indexed node ids."""
    import networkx as nx
    G = nx.from_numpy_array(adj)
    lcc = max(nx.connected_components(G), key=len)
    node_ids = np.array(sorted(lcc))

    # re-index
    id_map = {old: new for new, old in enumerate(node_ids)}
    sub_adj = adj[np.ix_(node_ids, node_ids)]
    sub_feat = features[node_ids]
    sub_labels = labels[node_ids]
    return sub_adj, sub_feat, sub_labels, node_ids


def compute_cosine_similarity(x: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix [N, N] from feature matrix [N, F]."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    x_norm = x / norms
    return x_norm @ x_norm.T


def edge_cosine_similarity(adj: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Return cosine similarity for each edge. Returns array of shape [N, N] masked to edges."""
    sim = compute_cosine_similarity(features)
    return sim * (adj > 0).astype(np.float32)


def build_knn_graph(features: np.ndarray, k: int,
                    existing_adj: np.ndarray | None = None) -> np.ndarray:
    """
    Build k-NN graph from features using cosine similarity.
    Optionally merges with existing_adj (union).
    Returns symmetric adjacency matrix.
    """
    sim = compute_cosine_similarity(features)
    np.fill_diagonal(sim, -np.inf)   # exclude self-loops

    n = features.shape[0]
    knn_adj = np.zeros((n, n), dtype=np.float32)
    top_k_idx = np.argsort(sim, axis=1)[:, -k:]
    rows = np.repeat(np.arange(n), k)
    cols = top_k_idx.flatten()
    knn_adj[rows, cols] = 1.0
    knn_adj = np.maximum(knn_adj, knn_adj.T)   # make symmetric

    if existing_adj is not None:
        knn_adj = np.maximum(knn_adj, (existing_adj > 0).astype(np.float32))

    return knn_adj


def add_random_edges(adj: np.ndarray, budget: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Randomly add `budget` edges to adjacency matrix."""
    adj = adj.copy()
    n = adj.shape[0]
    added = 0
    while added < budget:
        i, j = rng.integers(0, n, size=2)
        if i != j and adj[i, j] == 0:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
            added += 1
    return adj


def remove_random_edges(adj: np.ndarray, budget: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Randomly remove `budget` edges from adjacency matrix."""
    adj = adj.copy()
    rows, cols = np.where((adj > 0) & (np.triu(np.ones_like(adj), k=1) > 0))
    if len(rows) == 0:
        return adj
    budget = min(budget, len(rows))
    chosen = rng.choice(len(rows), size=budget, replace=False)
    for idx in chosen:
        i, j = rows[idx], cols[idx]
        adj[i, j] = 0.0
        adj[j, i] = 0.0
    return adj


def graph_stats(adj: np.ndarray) -> dict:
    """Return basic graph statistics."""
    n = adj.shape[0]
    edges = int((adj > 0).sum()) // 2
    degrees = (adj > 0).sum(axis=1)
    return {
        "num_nodes": n,
        "num_edges": edges,
        "avg_degree": float(degrees.mean()),
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "density": edges / (n * (n - 1) / 2),
    }
