"""
Meta Attack — global poisoning via momentum meta-gradients + combined surrogate loss.

Reference: Zügner & Günnemann, "Adversarial Attacks on Graph Neural Networks
via Meta Learning", ICLR 2019.

Strategy (greedy meta-gradient with approximate bilevel optimization):
  Treat the adjacency matrix as a continuous variable.
  Each step:
    1. Compute combined surrogate meta-gradient: α*val_loss + (1-α)*train_loss.
       Using BOTH masks makes the gradient attack the model at training time
       (disrupts learned representations) AND at evaluation time (degrades
       generalisation). Pure val_loss produces stale signals as the model
       re-fits the training set; the combined loss keeps perturbations coherent
       across both splits.
    2. Accumulate meta-gradient with momentum (β=0.9) to smooth out per-step
       noise. Momentum-smoothed scores select edges that consistently increase
       the combined loss across multiple steps, not just one-off spikes.
    3. Greedily flip the edge with highest momentum-weighted score.
    4. Do a short warm-start retrain (inner_epochs) on the perturbed graph
       so that subsequent gradient steps account for model adaptation.

  score(i,j) = momentum[i,j] * (1 - 2*A[i,j])
  — positive score = flipping (i,j) increases combined loss = good for attacker.
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
    budget_ratio: float = 0.35,
    n_steps: int = 500,
    inner_epochs: int = 75,
    alpha: float = 0.7,
    beta: float = 0.9,
) -> AttackResult:
    """
    Global untargeted poisoning via momentum meta-gradient on adjacency.

    Uses approximate bilevel optimization with two enhancements over the
    original Meta Attack:

    1. Combined surrogate loss: α*val_loss + (1-α)*train_loss.
       Attacks both the generalisation surface (val) and the training
       manifold (train), so perturbations survive full model retraining
       at evaluation time.

    2. Momentum smoothing (β=0.9): accumulates gradients across steps so
       edge selection is based on consistently high scores, not noisy
       single-step estimates.

    Args:
        graph:        Clean GraphData (training graph).
        model:        Trained GCN model (used as surrogate).
        params:       Baseline model params.
        budget_ratio: Fraction of edges to perturb.
        n_steps:      Meta-gradient steps (each step = 1 edge flip).
        inner_epochs: Warm-start retraining epochs per step.
        alpha:        Val/train loss mixing weight (higher = more emphasis on val).
        beta:         Momentum coefficient for gradient accumulation.

    Returns:
        AttackResult with globally perturbed graph.
    """
    import optax
    from models.train import GNNTrainState, train_step

    budget = edge_budget(graph.adj, budget_ratio)
    budget = min(budget, n_steps)
    print(f"  [Meta Attack] Budget={budget} edges "
          f"({budget_ratio:.0%} of {int(graph.adj.sum())//2}), "
          f"inner_epochs={inner_epochs}, α={alpha}, β={beta}")

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

    # JIT-compile the gradient function once (model is a Python closure)
    @jax.jit
    def _grad_jit(params, x, a_hat_flat):
        return jax.grad(_combined_loss, argnums=3)(
            params, model, x, a_hat_flat, lbl, val_mask, tr_mask, n, alpha
        )

    # Momentum accumulator — smooths gradient estimates over steps
    momentum = np.zeros((n, n), dtype=np.float32)
    # Cooldown guard: an edge can be re-flipped only after COOLDOWN steps.
    # This prevents short oscillations (the bug with no-flip-back: permanent ban
    # exhausted good candidates and wasted late steps on weak edges) while still
    # letting the attack revisit important edges once momentum has re-scored them.
    COOLDOWN = 50
    flip_last_step: dict = {}   # (i,j) -> step when last flipped

    for step in range(budget):
        a_hat = jnp.array(normalize_adjacency(adj))

        # Combined meta-gradient on CURRENT (warm) params (JIT-compiled)
        a_flat = a_hat.reshape(-1)
        grad = np.array(_grad_jit(state.params, x_j, a_flat)).reshape(n, n)

        # Momentum update: exponential moving average of gradients
        momentum = beta * momentum + (1.0 - beta) * grad

        flip_dir = 1.0 - 2.0 * adj
        scores   = momentum * flip_dir
        np.fill_diagonal(scores, -np.inf)

        # Block edges still in cooldown
        for (fi, fj), last in flip_last_step.items():
            if step - last < COOLDOWN:
                scores[fi, fj] = -np.inf
                scores[fj, fi] = -np.inf

        best  = int(np.argmax(scores))
        i_e, j_e = best // n, best % n

        if scores[i_e, j_e] > -np.inf:
            adj[i_e, j_e] = 1.0 - adj[i_e, j_e]
            adj[j_e, i_e] = adj[i_e, j_e]
            flip_last_step[(min(i_e, j_e), max(i_e, j_e))] = step

        # Inner-loop warm retrain on current perturbed graph
        a_hat_new = jnp.array(normalize_adjacency(adj))
        for _ in range(inner_epochs):
            state, _ = train_step(state, model, x_j, a_hat_new, lbl, tr_mask)

        if (step + 1) % 5 == 0:
            active_cooldowns = sum(
                1 for last in flip_last_step.values() if step - last < COOLDOWN
            )
            print(f"    step {step+1}/{budget}, score={scores[i_e,j_e]:.4f}, "
                  f"edge={'added' if adj[i_e,j_e]==1 else 'removed'} ({i_e},{j_e}), "
                  f"cooldown_active={active_cooldowns}")

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
# JAX gradient helpers
# ──────────────────────────────────────────────────────────────────────────────

def _combined_loss(params, model, x, a_hat_flat, labels, val_mask, train_mask, n, alpha):
    """α * val_loss + (1-α) * train_loss — combined surrogate for meta-gradient."""
    a_hat = a_hat_flat.reshape(n, n)
    _, logits, _ = model.apply({"params": params}, x, a_hat, training=False)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    true_lp = log_probs[jnp.arange(n), jnp.where(labels >= 0, labels, 0)]

    valid_val = val_mask & (labels >= 0)
    v_loss = -jnp.where(valid_val, true_lp, 0.0).sum() / jnp.maximum(valid_val.sum(), 1)

    valid_tr = train_mask & (labels >= 0)
    t_loss = -jnp.where(valid_tr, true_lp, 0.0).sum() / jnp.maximum(valid_tr.sum(), 1)

    return alpha * v_loss + (1.0 - alpha) * t_loss


def _combined_meta_grad(params, model, x, a_hat, labels, val_mask, train_mask, n, alpha):
    a_flat  = a_hat.reshape(-1)
    grad_fn = jax.grad(_combined_loss, argnums=3)
    g = grad_fn(params, model, x, a_flat, labels, val_mask, train_mask, n, alpha)
    return np.array(g).reshape(n, n)
