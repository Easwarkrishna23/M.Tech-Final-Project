"""Central configuration for all experiments."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 20


@dataclass
class AttackConfig:
    # Nettack — increased perturbations per node for stronger targeted impact
    nettack_n_perturbations: int = 20
    nettack_direct: bool = True

    # Meta Attack — 20% budget drives accuracy into 40-60% range
    meta_epochs: int = 200
    meta_lr: float = 0.1
    meta_budget_ratio: float = 0.20

    # Random Structure — 25% budget for strong baseline attack
    random_budget_ratio: float = 0.25

    # Feature Perturbation — ε=0.5 drives ~40% accuracy drop on Cora BoW features
    feature_epsilon: float = 0.5

    # Edge Flip — 20% budget to match structural attack strength
    edge_flip_budget_ratio: float = 0.20

    # Gradient-Based — ε=0.15 lands in 40-60% drop range (ε=0.3 was too extreme at 0%)
    grad_epsilon: float = 0.15
    grad_steps: int = 20


@dataclass
class DefenseConfig:
    # Edge Pruning — percentile-based: remove bottom prune_pct% of edges by cosine sim
    # This is dataset-agnostic (works regardless of absolute sim values)
    prune_percentile: float = 10.0    # remove bottom 10% least-similar edges
    cosine_threshold: float = 0.0     # fallback fixed threshold (used if prune_percentile=0)
    min_edges_ratio: float = 0.7      # keep at least 70% of original edges

    # Graph Reconstruction (k-NN)
    knn_k: int = 3


@dataclass
class DynamicGraphConfig:
    """SBM-based temporal graph parameters."""
    num_nodes: int = 2708          # match Cora size
    num_communities: int = 7       # match Cora classes
    timesteps: int = 10
    p_in: float = 0.008            # intra-community edge prob (tuned to match Cora density ~5K edges)
    p_out: float = 0.0002          # inter-community edge prob
    feature_dim: int = 1433        # match Cora feature dim
    community_switch_rate: float = 0.02   # fraction of nodes that switch community per step
    edge_change_rate: float = 0.05        # fraction of edges added/removed per step
    feature_noise_std: float = 0.01       # Gaussian noise added to features per step
    seed: int = 42


@dataclass
class Config:
    seed: int = 42
    data_dir: Path = field(default_factory=lambda: Path("data"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    figures_dir: Path = field(default_factory=lambda: Path("results/figures"))
    tables_dir: Path = field(default_factory=lambda: Path("results/tables"))
    checkpoints_dir: Path = field(default_factory=lambda: Path("checkpoints"))

    model: ModelConfig = field(default_factory=ModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    dynamic: DynamicGraphConfig = field(default_factory=DynamicGraphConfig)

    def make_dirs(self):
        for d in [self.data_dir, self.results_dir, self.figures_dir,
                  self.tables_dir, self.checkpoints_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Singleton used throughout the project
cfg = Config()
