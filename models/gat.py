"""
Graph Attention Network (GAT) — Veličković et al., 2018.
Single-head attention implementation for comparison with GCN.

Attention coefficient for edge (i,j):
    e_ij  = LeakyReLU( a^T [ W*h_i || W*h_j ] )
    α_ij  = softmax_j( e_ij )   (over neighbours of i)
    h'_i  = σ( Σ_j α_ij * W*h_j )

Uses adjacency mask so attention only flows over existing edges.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class GATLayer(nn.Module):
    out_dim: int
    dropout_rate: float = 0.5
    negative_slope: float = 0.2    # LeakyReLU

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,        # [N, F]
        adj: jnp.ndarray,       # [N, N]  binary adjacency (with self-loops)
        training: bool = False,
    ) -> jnp.ndarray:           # [N, out_dim]

        n = x.shape[0]
        # Linear transform
        h = nn.Dense(self.out_dim, use_bias=False, name="W")(x)  # [N, out_dim]
        h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)

        # Attention: concatenate pairs [h_i || h_j] for each possible edge
        a_src = nn.Dense(1, use_bias=False, name="a_src")(h)  # [N, 1]
        a_dst = nn.Dense(1, use_bias=False, name="a_dst")(h)  # [N, 1]

        # e_ij = LeakyReLU(a_src_i + a_dst_j)  broadcast → [N, N]
        e = a_src + a_dst.T
        e = nn.leaky_relu(e, negative_slope=self.negative_slope)

        # Mask: only attend over existing edges (adj > 0) + self-loops
        adj_mask = (adj + jnp.eye(n)) > 0
        VERY_NEG = -1e9
        e = jnp.where(adj_mask, e, VERY_NEG)

        alpha = nn.softmax(e, axis=-1)                        # [N, N]
        alpha = nn.Dropout(self.dropout_rate,
                           deterministic=not training)(alpha)

        # Aggregate
        out = alpha @ h                                        # [N, out_dim]
        return out


class GAT(nn.Module):
    hidden_dim: int
    num_classes: int
    dropout_rate: float = 0.6

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        adj: jnp.ndarray,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        # ── Layer 1 ───────────────────────────────────────────────
        h = GATLayer(self.hidden_dim, self.dropout_rate,
                     name="gat1")(x, adj, training)
        h = nn.elu(h)
        embeddings = h                                         # [N, hidden_dim]

        # ── Layer 2 ───────────────────────────────────────────────
        logits = GATLayer(self.num_classes, self.dropout_rate,
                          name="gat2")(h, adj, training)
        probs = nn.softmax(logits, axis=-1)

        return embeddings, logits, probs


def create_gat(hidden_dim: int, num_classes: int,
               dropout_rate: float = 0.6) -> GAT:
    return GAT(hidden_dim=hidden_dim, num_classes=num_classes,
               dropout_rate=dropout_rate)
