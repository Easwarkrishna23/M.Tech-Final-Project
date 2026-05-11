"""
Dynamic graph dataset using a Temporal Stochastic Block Model (T-SBM).

Why T-SBM instead of random Cora perturbation:
  - Ground-truth community labels → proper node classification task
  - Principled temporal evolution (community drift + edge rewiring)
  - Controllable difficulty: adjust p_in/p_out ratio
  - Widely used benchmark in GNN robustness literature
  - Generates realistic mesoscale structure (not just noise)

Graph evolution per timestep:
  1. Community switching: a fraction of nodes migrate to an adjacent community
  2. Edge rewiring: add intra-community edges for nodes that switched,
                    remove inter-community edges that are now "stale"
  3. Feature drift: small Gaussian noise added to features,
                    node class centroid features updated
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import numpy as np

from datasets.cora_loader import GraphData
from utils.graph_utils import normalize_adjacency, graph_stats
from utils.config import DynamicGraphConfig


@dataclass
class DynamicGraphData:
    """Sequence of graph snapshots forming a dynamic graph."""
    snapshots: List[GraphData]
    communities: np.ndarray           # [T, N] community assignment per timestep
    config: DynamicGraphConfig

    @property
    def num_timesteps(self) -> int:
        return len(self.snapshots)

    def get_snapshot(self, t: int) -> GraphData:
        return self.snapshots[t]

    def final_snapshot(self) -> GraphData:
        """The last timestep is used as input for downstream tasks."""
        return self.snapshots[-1]

    def stats_over_time(self) -> List[dict]:
        return [s.stats() for s in self.snapshots]


class SBMDynamicGraph:
    """
    Builder for Temporal SBM graphs.

    The initial graph is a planted partition model (k communities, p_in >> p_out).
    Each timestep evolves:
      - community_switch_rate fraction of nodes change community
      - edge_change_rate fraction of edges are rewired to reflect new memberships
      - small Gaussian feature noise is added

    Node features are drawn from per-community Gaussian distributions:
      x_i ~ N(mu_c, sigma) where c = community(i)
    """

    def __init__(self, cfg: DynamicGraphConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> DynamicGraphData:
        cfg = self.cfg
        print(f"[SBM-Dynamic] Building T-SBM: {cfg.num_nodes} nodes, "
              f"{cfg.num_communities} communities, {cfg.timesteps} timesteps")

        communities, features = self._init_communities_and_features()
        adj = self._init_adjacency(communities)

        snapshots = []
        all_communities = [communities.copy()]

        for t in range(cfg.timesteps):
            gd = self._make_graph_data(adj, features, communities, t)
            snapshots.append(gd)

            if t < cfg.timesteps - 1:
                communities, adj, features = self._evolve(communities, adj, features)
                all_communities.append(communities.copy())

        # Stack community history [T, N]
        community_history = np.stack(all_communities, axis=0)

        print(f"[SBM-Dynamic] Done. Final snapshot: {snapshots[-1].num_nodes} nodes, "
              f"{int(snapshots[-1].adj.sum())//2} edges")

        return DynamicGraphData(
            snapshots=snapshots,
            communities=community_history,
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_communities_and_features(self):
        cfg = self.cfg
        n, k, d = cfg.num_nodes, cfg.num_communities, cfg.feature_dim

        # Assign nodes to communities (balanced)
        communities = np.repeat(np.arange(k), n // k)
        remainder = n - len(communities)
        if remainder:
            communities = np.concatenate([communities,
                                          self.rng.integers(0, k, size=remainder)])
        self.rng.shuffle(communities)

        # Per-community feature centroids — sparse binary-like (matches Cora style)
        centroids = self._generate_sparse_centroids(k, d)

        # Per-node features: centroid + small noise
        features = centroids[communities] + self.rng.normal(0, 0.05, (n, d)).astype(np.float32)
        # Binarise to match Cora BoW features
        features = (features > 0.5).astype(np.float32)

        self._centroids = centroids   # keep for feature drift
        return communities, features

    def _generate_sparse_centroids(self, k: int, d: int) -> np.ndarray:
        """Generate k sparse centroids in R^d (mimics bag-of-words)."""
        centroids = np.zeros((k, d), dtype=np.float32)
        words_per_class = d // k
        for c in range(k):
            start = c * words_per_class
            end = min(start + words_per_class, d)
            centroids[c, start:end] = 1.0
            # some overlap to make task non-trivial
            extra = self.rng.integers(0, d, size=words_per_class // 4)
            centroids[c, extra] = 0.5
        return centroids

    def _init_adjacency(self, communities: np.ndarray) -> np.ndarray:
        """Sample SBM adjacency matrix."""
        cfg = self.cfg
        n = cfg.num_nodes
        adj = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                p = cfg.p_in if communities[i] == communities[j] else cfg.p_out
                if self.rng.random() < p:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        print(f"  [SBM-init] {int(adj.sum())//2} edges sampled "
              f"({int((adj.sum(axis=1)>0).sum())} connected nodes)")
        return adj

    # ------------------------------------------------------------------
    # Evolution step
    # ------------------------------------------------------------------

    def _evolve(self, communities: np.ndarray, adj: np.ndarray,
                features: np.ndarray):
        cfg = self.cfg
        n = cfg.num_nodes
        k = cfg.num_communities

        # 1. Community switching
        n_switch = max(1, int(n * cfg.community_switch_rate))
        switch_nodes = self.rng.choice(n, size=n_switch, replace=False)
        new_communities = communities.copy()
        for node in switch_nodes:
            current = communities[node]
            # move to adjacent community (cyclic)
            new_communities[node] = (current + self.rng.integers(1, k)) % k

        # 2. Edge rewiring for switched nodes
        adj_new = adj.copy()
        existing_edges = int(adj.sum()) // 2
        n_rewire = max(1, int(existing_edges * cfg.edge_change_rate))

        for node in switch_nodes:
            old_c = communities[node]
            new_c = new_communities[node]
            neighbors = np.where(adj[node] > 0)[0]

            # Remove intra-community edges that no longer match
            for nb in neighbors:
                if communities[nb] == old_c and self.rng.random() < 0.5:
                    adj_new[node, nb] = 0.0
                    adj_new[nb, node] = 0.0

            # Add intra-community edges for new community
            new_community_nodes = np.where(new_communities == new_c)[0]
            new_community_nodes = new_community_nodes[new_community_nodes != node]
            if len(new_community_nodes) > 0:
                add_count = min(2, len(new_community_nodes))
                new_nbs = self.rng.choice(new_community_nodes, size=add_count, replace=False)
                for nb in new_nbs:
                    adj_new[node, nb] = 1.0
                    adj_new[nb, node] = 1.0

        # Global edge budget: add/remove random intra-community edges
        all_community_pairs = [
            (i, j)
            for c in range(k)
            for i in np.where(new_communities == c)[0]
            for j in np.where(new_communities == c)[0]
            if i < j and adj_new[i, j] == 0
        ]
        if all_community_pairs:
            add_pairs = self.rng.choice(len(all_community_pairs),
                                         size=min(n_rewire // 2, len(all_community_pairs)),
                                         replace=False)
            for idx in add_pairs:
                i, j = all_community_pairs[idx]
                adj_new[i, j] = 1.0
                adj_new[j, i] = 1.0

        # 3. Feature drift: small noise + centroid drift for switched nodes
        features_new = features + self.rng.normal(0, cfg.feature_noise_std,
                                                   features.shape).astype(np.float32)
        for node in switch_nodes:
            new_c = new_communities[node]
            features_new[node] = (0.8 * features[node] +
                                   0.2 * self._centroids[new_c] +
                                   self.rng.normal(0, 0.05, features.shape[1]).astype(np.float32))
        features_new = np.clip(features_new, 0.0, 1.0)

        return new_communities, adj_new, features_new

    # ------------------------------------------------------------------
    # Graph snapshot builder
    # ------------------------------------------------------------------

    def _make_graph_data(self, adj: np.ndarray, features: np.ndarray,
                          communities: np.ndarray, t: int) -> GraphData:
        cfg = self.cfg
        n = cfg.num_nodes
        adj_norm = normalize_adjacency(adj)

        # Fixed 60/20/20 split (stratified)
        train_mask, val_mask, test_mask = self._stratified_split(communities, cfg.num_communities)

        return GraphData(
            adj=adj.astype(np.float32),
            adj_norm=adj_norm,
            features=features.astype(np.float32),
            labels=communities.astype(np.int64),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=n,
            num_features=cfg.feature_dim,
            num_classes=cfg.num_communities,
            name=f"sbm_dynamic_t{t}",
        )

    def _stratified_split(self, labels: np.ndarray, num_classes: int,
                           train_ratio: float = 0.6,
                           val_ratio: float = 0.2) -> tuple:
        n = len(labels)
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)

        for c in range(num_classes):
            idx = np.where(labels == c)[0]
            self.rng.shuffle(idx)
            n_train = max(1, int(len(idx) * train_ratio))
            n_val = max(1, int(len(idx) * val_ratio))
            train_mask[idx[:n_train]] = True
            val_mask[idx[n_train:n_train + n_val]] = True
            test_mask[idx[n_train + n_val:]] = True

        return train_mask, val_mask, test_mask


# ------------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------------

def save_dynamic_graph(dg: DynamicGraphData, path: str | Path):
    """Save dynamic graph snapshots to disk as .npz archive."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "communities.npy", dg.communities)
    for t, snap in enumerate(dg.snapshots):
        np.savez(path / f"snapshot_t{t}.npz",
                 adj=snap.adj, features=snap.features,
                 labels=snap.labels, train_mask=snap.train_mask,
                 val_mask=snap.val_mask, test_mask=snap.test_mask)
    print(f"[SBM-Dynamic] Saved {len(dg.snapshots)} snapshots to {path}")


def load_dynamic_graph(path: str | Path,
                        cfg: DynamicGraphConfig) -> DynamicGraphData:
    """Load a previously saved dynamic graph from disk."""
    path = Path(path)
    communities = np.load(path / "communities.npy")
    snapshots = []
    t = 0
    while (path / f"snapshot_t{t}.npz").exists():
        d = np.load(path / f"snapshot_t{t}.npz")
        adj = d["adj"]
        snapshots.append(GraphData(
            adj=adj,
            adj_norm=normalize_adjacency(adj),
            features=d["features"],
            labels=d["labels"],
            train_mask=d["train_mask"],
            val_mask=d["val_mask"],
            test_mask=d["test_mask"],
            num_nodes=adj.shape[0],
            num_features=d["features"].shape[1],
            num_classes=int(d["labels"].max()) + 1,
            name=f"sbm_dynamic_t{t}",
        ))
        t += 1
    print(f"[SBM-Dynamic] Loaded {t} snapshots from {path}")
    return DynamicGraphData(snapshots=snapshots, communities=communities, config=cfg)
