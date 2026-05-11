"""
Meta Attack — global poisoning via meta-gradients with inner-loop retraining.

Reference: Zügner & Günnemann, "Adversarial Attacks on Graph Neural Networks
via Meta Learning", ICLR 2019.

Strategy (greedy meta-gradient with approximate bilevel optimization):
  Treat the adjacency matrix as a continuous variable.
  Each step:
    1. Compute meta-gradient of validation loss w.r.t. A on CURRENT params.
    2. Greedily flip the edge with highest score.
    3. Do a short warm-start retrain (inner_epochs) on the perturbed graph
       so that subsequent gradient steps account for model adaptation.

  This approximates the bilevel structure of the full Meta Attack.
  Without step 3, perturbations are computed on stale params and their
  cumulative effect does not survive full retraining at evaluation time.

  score(i,j) = grad_A[i,j] * (1 - 2*A[i,j])
  — positive score = flipping (i,j) increases val loss = good for attacker.
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
    inner_epochs: int = 15,
) -> AttackResult:
    """
    Global untargeted poisoning via meta-gradient on adjacency.

    Uses approximate bilevel optimization: after each edge flip, runs
    `inner_epochs` of SGD warm-start on the perturbed graph so that
    subsequent gradient steps are computed on adapted (not stale) params.
    This is the key fix that allows accumulated perturbations to survive
    full model retraining at evaluation time.

    Args:
        graph:        Clean GraphData (training graph).
        model:        Trained GCN model (used as surrogate).
        params:       Baseline model params.
        budget_ratio: Fraction of edges to perturb.
        n_steps:      Meta-gradient steps (each step = 1 edge flip).
        inner_epochs: Warm-start retraining epochs per step.

    Returns:
        AttackResult with globally perturbed graph.
    """
    import optax
    from models.train import GNNTrainState, train_step

    budget = edge_budget(graph.adj, budget_ratio)
    budget = min(budget, n_steps)
    print(f"  [Meta Attack] Budget={budget} edges "
          f"({budget_ratio:.0%} of {int(graph.adj.sum())//2}), "
          f"inner_epochs={inner_epochs}")

    adj      = graph.adj.copy().astype(np.float32)
    x_j      = jnp.array(graph.features)
    lbl      = jnp.array(graph.labels)
    val_mask = jnp.array(graph.val_mask)
    tr_mask  = jnp.array(graph.train_mask)
    n        = adj.shape[0]

    # Maintain a live model state for inner-loop warm retraining
    tx = optax.adamw(learning_rate=0.01, weight_decay=5e-4)
    state = GNNTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dropout_key=jax.random.PRNGKey(0),
    )

    for step in range(budget):
        a_hat = jnp.array(normalize_adjacency(adj))

        # Meta-gradient on CURRENT (warm) params: ∂(val_loss) / ∂A
        grad = _meta_grad(state.params, model, x_j, a_hat, lbl, val_mask, n)

        flip_dir = 1.0 - 2.0 * adj
        scores   = grad * flip_dir
        np.fill_diagonal(scores, -np.inf)

        best  = int(np.argmax(scores))
        i_e, j_e = best // n, best % n

        if scores[i_e, j_e] > -np.inf:
            adj[i_e, j_e] = 1.0 - adj[i_e, j_e]
            adj[j_e, i_e] = adj[i_e, j_e]

        # Inner-loop warm retrain on current perturbed graph
        a_hat_new = jnp.array(normalize_adjacency(adj))
        for _ in range(inner_epochs):
            state, _ = train_step(state, model, x_j, a_hat_new, lbl, tr_mask)

        if (step + 1) % 5 == 0:
            print(f"    step {step+1}/{budget}, score={scores[i_e,j_e]:.4f}, "
                  f"edge={'added' if adj[i_e,j_e]==1 else 'removed'} ({i_e},{j_e})")

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
