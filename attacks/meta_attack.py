"""
Meta Attack — global poisoning via meta-gradients.

Reference: Zügner & Günnemann, "Adversarial Attacks on Graph Neural Networks
via Meta Learning", ICLR 2019.

Strategy (simplified greedy meta-gradient attack):
  Treat the adjacency matrix as a continuous variable.
  Compute the gradient of validation loss w.r.t. A (the meta-gradient).
  Greedily flip the edge that maximally increases validation loss,
  repeating for `budget` steps.

  score(i,j) = grad_A[i,j] * (1 - 2*A[i,j])
  — positive score = flipping (i,j) increases val loss = good for attacker.

This is a global untargeted attack: the entire graph structure is perturbed
to maximally hurt overall classification performance.
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from datasets.cora_loader import GraphData
from attacks.base import AttackResult, edge_budget, diff_edges
from utils.graph_utils import normalize_adjacency


def meta_attack(
    graph: GraphData,
    model: nn.Module,
    params: any,
    budget_ratio: float = 0.05,
    n_steps: int = 20,
) -> AttackResult:
    """
    Global untargeted poisoning via meta-gradient on adjacency.

    Args:
        graph:        Clean GraphData (training graph).
        model:        Trained GCN model (used as surrogate).
        params:       Baseline model params.
        budget_ratio: Fraction of edges to perturb.
        n_steps:      Meta-gradient computation steps (each step = 1 edge flip).

    Returns:
        AttackResult with globally perturbed graph.
    """
    budget = edge_budget(graph.adj, budget_ratio)
    budget = min(budget, n_steps)
    print(f"  [Meta Attack] Budget={budget} edges "
          f"({budget_ratio:.0%} of {int(graph.adj.sum())//2})")

    adj  = graph.adj.copy().astype(np.float32)
    x_j  = jnp.array(graph.features)
    lbl  = jnp.array(graph.labels)
    val_mask = jnp.array(graph.val_mask)
    n    = adj.shape[0]

    for step in range(budget):
        a_hat = jnp.array(normalize_adjacency(adj))

        # Meta-gradient: ∂(val_loss) / ∂A
        grad = _meta_grad(params, model, x_j, a_hat, lbl, val_mask, n)

        # Score: gradient * flip_direction
        flip_dir = 1.0 - 2.0 * adj
        scores = grad * flip_dir

        # Mask diagonal and already-max-budget entries
        np.fill_diagonal(scores, -np.inf)

        best = int(np.argmax(scores))
        i_e, j_e = best // n, best % n

        if scores[i_e, j_e] > -np.inf:
            adj[i_e, j_e] = 1.0 - adj[i_e, j_e]
            adj[j_e, i_e] = adj[i_e, j_e]

        if (step + 1) % 5 == 0:
            print(f"    step {step+1}/{budget}, best score={scores[i_e,j_e]:.4f}, "
                  f"flipped ({i_e},{j_e}) → edge={'added' if adj[i_e,j_e]==1 else 'removed'}")

    perturbed = graph.copy()
    perturbed = perturbed.update_adj(adj)
    perturbed.name = "meta_attack"

    added, removed = diff_edges(graph.adj, adj)
    return AttackResult(
        perturbed_graph=perturbed,
        attack_name="Meta Attack",
        n_edges_added=added,
        n_edges_removed=removed,
        n_features_perturbed=0,
        budget_used=budget,
    )


# ──────────────────────────────────────────────────────────────────────────────
# JAX gradient helper
# ──────────────────────────────────────────────────────────────────────────────

def _val_loss(params, model, x, a_hat_flat, labels, val_mask, n):
    a_hat = a_hat_flat.reshape(n, n)
    _, logits, _ = model.apply({"params": params}, x, a_hat, training=False)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    valid = val_mask & (labels >= 0)
    true_lp = log_probs[jnp.arange(n), jnp.where(labels >= 0, labels, 0)]
    loss = -jnp.where(valid, true_lp, 0.0).sum() / jnp.maximum(valid.sum(), 1)
    return loss


def _meta_grad(params, model, x, a_hat, labels, val_mask, n):
    a_flat = a_hat.reshape(-1)
    grad_fn = jax.grad(_val_loss, argnums=3)
    g = grad_fn(params, model, x, a_flat, labels, val_mask, n)
    return np.array(g).reshape(n, n)
