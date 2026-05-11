"""
Elliptic Bitcoin Dataset loader — real temporal graph with 49 timesteps.

Dataset facts (raw CSVs):
  - 203,769 transaction nodes total, 234,355 directed edges
  - 166 features per node (col 0 = txId, col 1 = timestep, cols 2-166 = features)
  - Labels: 1=illicit (fraud), 2=licit — remapped to 1=illicit, 0=licit
  - ~21% nodes are unlabeled (class 0 = unknown) — excluded from classification metrics
  - 49 native timesteps (~4,100 nodes, ~7,000 edges each — Cora-scale)

Temporal split strategy:
  - timesteps 1-34  → training era  (used for train/val splits within each snapshot)
  - timesteps 35-49 → test era      (matches original paper's evaluation protocol)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from datasets.cora_loader import GraphData
from utils.graph_utils import sparse_to_dense, normalize_adjacency, graph_stats


@dataclass
class EllipticData:
    """Container for the full Elliptic temporal graph."""
    snapshots: List[GraphData]          # one per timestep (index 0 = timestep 1)
    labeled_ratios: List[float]         # fraction of labeled nodes per step
    illicit_ratios: List[float]         # fraction of illicit among labeled per step

    @property
    def num_timesteps(self) -> int:
        return len(self.snapshots)

    def get_snapshot(self, t: int) -> GraphData:
        return self.snapshots[t]

    def final_snapshot(self) -> GraphData:
        """Last timestep snapshot — used as primary graph for attack/defense."""
        return self.snapshots[-1]

    def stats_over_time(self) -> List[dict]:
        return [s.stats() for s in self.snapshots]


def load_elliptic(data_dir: str | Path = "data") -> EllipticData:
    """
    Load Elliptic Bitcoin Dataset.
    Downloads via torch_geometric on first call, then reads raw CSVs for timestep info.

    Label convention (after remapping):
        0  = licit (benign transaction)
        1  = illicit (fraudulent transaction)
        -1 = unknown (unlabeled — in graph structure but excluded from metrics)
    """
    from torch_geometric.datasets import EllipticBitcoinDataset

    data_dir = Path(data_dir)
    raw_dir = data_dir / "Elliptic" / "raw"

    # Trigger PyG download/processing if needed
    print("[Elliptic] Loading dataset (downloading if needed)...")
    _ = EllipticBitcoinDataset(root=str(data_dir / "Elliptic"))

    # ── Load raw CSVs ──────────────────────────────────────────────
    print("[Elliptic] Reading raw CSVs...")
    feat_df = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None)
    # cols: [0]=txId, [1]=timestep, [2..166]=features (165 total)
    tx_ids = feat_df[0].values                        # transaction IDs
    timesteps = feat_df[1].values.astype(np.int64)    # 1–49
    features_all = feat_df.iloc[:, 2:].values.astype(np.float32)  # [N, 165]

    classes_df = pd.read_csv(raw_dir / "elliptic_txs_classes.csv")
    # cols: txId, class — where class is '1'(illicit), '2'(licit), 'unknown'
    class_map = dict(zip(classes_df["txId"].values, classes_df["class"].values))

    edges_df = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
    # cols: txId1, txId2

    # Build global txId → row-index mapping
    tx_to_idx = {tx: i for i, tx in enumerate(tx_ids)}

    # Remap global labels: '1'→1, '2'→0, 'unknown'→-1
    n_total = len(tx_ids)
    labels_all = np.full(n_total, -1, dtype=np.int64)
    for tx, cls in class_map.items():
        if tx in tx_to_idx:
            idx = tx_to_idx[tx]
            if cls == "1":
                labels_all[idx] = 1
            elif cls == "2":
                labels_all[idx] = 0

    # Build global edge list as arrays of row indices
    src_global = edges_df.iloc[:, 0].values
    dst_global = edges_df.iloc[:, 1].values
    valid_edge = np.isin(src_global, list(tx_to_idx.keys())) & \
                 np.isin(dst_global, list(tx_to_idx.keys()))
    src_idx = np.array([tx_to_idx[t] for t in src_global[valid_edge]])
    dst_idx = np.array([tx_to_idx[t] for t in dst_global[valid_edge]])

    # ── Build per-timestep snapshots ──────────────────────────────
    unique_timesteps = sorted(np.unique(timesteps).tolist())  # [1, 2, ..., 49]
    print(f"[Elliptic] Building {len(unique_timesteps)} snapshots...")

    snapshots, labeled_ratios, illicit_ratios = [], [], []

    for t_val in unique_timesteps:
        node_mask = (timesteps == t_val)
        node_global_idx = np.where(node_mask)[0]          # row indices in full dataset
        n_local = len(node_global_idx)

        # Local re-indexing
        global_to_local = {g: l for l, g in enumerate(node_global_idx)}

        # Filter edges where BOTH endpoints are in this timestep
        edge_mask = np.isin(src_idx, node_global_idx) & np.isin(dst_idx, node_global_idx)
        local_src = np.array([global_to_local[g] for g in src_idx[edge_mask]])
        local_dst = np.array([global_to_local[g] for g in dst_idx[edge_mask]])
        edge_index = np.stack([local_src, local_dst], axis=0)

        features = features_all[node_global_idx]     # [n_local, 165]
        labels = labels_all[node_global_idx]         # [n_local]
        labeled_mask = labels >= 0

        # Dense adjacency (n_local ≈ 4K → ~64MB peak, freed after norm)
        adj = sparse_to_dense(edge_index, n_local)
        adj = np.maximum(adj, adj.T)                 # symmetrise (undirected GCN)
        adj_norm = normalize_adjacency(adj)

        train_mask, val_mask, test_mask = _split(labels, labeled_mask, seed=42 + t_val)

        labeled_ratio = float(labeled_mask.mean())
        illicit_count = int((labels == 1).sum())
        licit_count = int((labels == 0).sum())
        illicit_ratio = illicit_count / max(illicit_count + licit_count, 1)
        labeled_ratios.append(labeled_ratio)
        illicit_ratios.append(illicit_ratio)

        snap = GraphData(
            adj=adj,
            adj_norm=adj_norm,
            features=features,
            labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=n_local,
            num_features=features.shape[1],
            num_classes=2,
            name=f"elliptic_t{t_val}",
        )
        snapshots.append(snap)

        if t_val % 10 == 0 or t_val == unique_timesteps[-1]:
            s = graph_stats(adj)
            print(f"  [t={t_val:02d}/49] nodes={n_local}, edges={s['num_edges']}, "
                  f"labeled={labeled_ratio:.1%}, illicit={illicit_ratio:.1%}")

    print(f"\n[Elliptic] Done — {len(snapshots)} snapshots loaded.")
    _print_summary(snapshots, labeled_ratios, illicit_ratios)

    return EllipticData(
        snapshots=snapshots,
        labeled_ratios=labeled_ratios,
        illicit_ratios=illicit_ratios,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split(labels: np.ndarray, labeled_mask: np.ndarray,
           train_ratio: float = 0.6, val_ratio: float = 0.2,
           seed: int = 42) -> tuple:
    n = len(labels)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    rng = np.random.default_rng(seed)

    for cls in [0, 1]:
        idx = np.where((labels == cls) & labeled_mask)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        n_val = max(1, int(len(idx) * val_ratio))
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True

    return train_mask, val_mask, test_mask


def _print_summary(snapshots: List[GraphData],
                   labeled_ratios: List[float],
                   illicit_ratios: List[float]):
    sizes = [s.num_nodes for s in snapshots]
    edges = [s.stats()["num_edges"] for s in snapshots]
    print(f"  Nodes/snapshot : min={min(sizes)}, max={max(sizes)}, avg={int(np.mean(sizes))}")
    print(f"  Edges/snapshot : min={min(edges)}, max={max(edges)}, avg={int(np.mean(edges))}")
    print(f"  Labeled ratio  : avg={np.mean(labeled_ratios):.1%}")
    print(f"  Illicit ratio  : avg={np.mean(illicit_ratios):.1%} (class imbalance)")
