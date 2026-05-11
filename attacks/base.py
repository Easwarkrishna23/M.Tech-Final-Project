"""Shared interface and utilities for all attacks."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from datasets.cora_loader import GraphData


@dataclass
class AttackResult:
    """Returned by every attack function."""
    perturbed_graph: GraphData
    attack_name: str
    n_edges_added: int
    n_edges_removed: int
    n_features_perturbed: int
    budget_used: int

    def summary(self) -> str:
        return (f"{self.attack_name}: "
                f"+{self.n_edges_added}/-{self.n_edges_removed} edges, "
                f"{self.n_features_perturbed} feature perturbations, "
                f"budget={self.budget_used}")


def edge_budget(adj: np.ndarray, ratio: float) -> int:
    """Convert a ratio of total edges into an integer edge budget."""
    n_edges = int((adj > 0).sum()) // 2
    return max(1, int(n_edges * ratio))


def diff_edges(adj_orig: np.ndarray, adj_new: np.ndarray) -> tuple[int, int]:
    """Count added and removed edges between two adjacency matrices."""
    orig = (adj_orig > 0).astype(np.int8)
    new  = (adj_new  > 0).astype(np.int8)
    added   = int(np.triu(new  - orig, k=1).clip(0).sum())
    removed = int(np.triu(orig - new,  k=1).clip(0).sum())
    return added, removed
