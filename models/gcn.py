"""
2-layer Graph Convolutional Network (GCN) — Kipf & Welling, 2017.

Forward pass per layer:
    H^(l+1) = σ( A_hat @ H^(l) @ W^(l) )

where A_hat = D^{-1/2}(A+I)D^{-1/2} is the normalised adjacency (pre-computed).

Outputs three values so every downstream component can pick what it needs:
  embeddings  — layer-1 activations  [N, hidden_dim]  (used for t-SNE)
  logits      — raw layer-2 output   [N, num_classes]
  probs       — softmax of logits    [N, num_classes]
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class GCN(nn.Module):
    hidden_dim: int
    num_classes: int
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,        # [N, F]  node features
        a_hat: jnp.ndarray,     # [N, N]  normalised adjacency
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        # ── Layer 1: A_hat @ X @ W0 → ReLU → Dropout ──────────────
        h = nn.Dense(self.hidden_dim, name="layer1")(a_hat @ x)
        h = nn.relu(h)
        h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)
        embeddings = h                                  # [N, hidden_dim]

        # ── Layer 2: A_hat @ H @ W1 → logits ──────────────────────
        logits = nn.Dense(self.num_classes, name="layer2")(a_hat @ h)
        probs = nn.softmax(logits, axis=-1)             # [N, num_classes]

        return embeddings, logits, probs


def create_gcn(hidden_dim: int, num_classes: int,
               dropout_rate: float = 0.5) -> GCN:
    return GCN(hidden_dim=hidden_dim, num_classes=num_classes,
               dropout_rate=dropout_rate)
