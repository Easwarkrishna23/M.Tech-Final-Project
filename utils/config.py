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
    # Nettack — 120 HIGH-confidence targets + 40 perturbs each
    nettack_n_perturbations: int = 40   # was 20 — doubled for stronger impact
    nettack_target_count: int = 120     # was 80 — attack more test nodes
    nettack_direct: bool = True

    # Meta Attack — combined train+val loss + momentum; larger budget
    meta_epochs: int = 500
    meta_lr: float = 0.1
    meta_budget_ratio: float = 0.35
    meta_inner_epochs: int = 40         # 15 was too short (chaotic gradients), 75 was too long (stale)

    # DICE — 40% budget + aggressive bridge targeting
    dice_budget_ratio: float = 0.40     # was 0.35

    # Random Structure — 45% budget, targets high-betweenness nodes
    random_budget_ratio: float = 0.45   # was 0.40

    # Feature Perturbation — ε=0.5 drives ~40% accuracy drop on Cora BoW features
    feature_epsilon: float = 0.5

    # Edge Flip — 40% budget + cross-class bias at bridge positions
    edge_flip_budget_ratio: float = 0.40   # was 0.35

    # Gradient-Based — ε=0.15
    grad_epsilon: float = 0.15
    grad_steps: int = 20


@dataclass
class GNNGuardConfig:
    """GNNGUARD defense configuration (Zhang & Zitnik, NeurIPS 2020)."""
    # Edge similarity threshold P0: prune edges with cosine(h_u, h_v) < p0
    p0: float = 0.05                  # conservative — only prune clearly dissimilar edges
    use_embedding_sim: bool = True    # True=layer-1 embeddings, False=raw features
    min_edges_ratio: float = 0.60     # keep at least 60% of original edges
    layer_wise: bool = True           # apply per-layer pruning (as in original paper)


@dataclass
class OntologyDefenseConfig:
    """Ontology-Driven Self-Healing defense configuration."""
    topic_sim_threshold: float = 0.20   # CitationEdge flagged as SuspiciousEdge if sim < this
    mismatch_alert_ratio: float = 0.15  # trigger full plan if >15% of edges are suspicious
    denoising_steps: int = 3            # base k in X' = (A_hat)^k @ X
    min_edges_ratio: float = 0.75       # keep at least 75% of edges — was 0.50, too aggressive
    adaptive_denoising: bool = True     # scale k with detected vulnerability ratio
    max_denoising_steps: int = 7        # upper limit for adaptive k
    temporal_drift_sigma: float = 2.5   # z-score threshold for temporal anomaly detection


@dataclass
class DefenseConfig:
    # Legacy 3-step pipeline (still used for comparison)
    prune_percentile: float = 10.0
    cosine_threshold: float = 0.0
    min_edges_ratio: float = 0.7
    knn_k: int = 3

    # Defense 1: GNNGUARD
    gnnguard: GNNGuardConfig = field(default_factory=GNNGuardConfig)

    # Defense 2: Ontology-Driven Self-Healing
    ontology: OntologyDefenseConfig = field(default_factory=OntologyDefenseConfig)


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
