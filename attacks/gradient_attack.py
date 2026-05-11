"""
Gradient-Based Feature Attack — evasion attack at test time.

Computes the gradient of the classification loss w.r.t. node features,
then perturbs features in the gradient direction (maximises loss).

This is the graph-adapted Fast Gradient Sign Method (FGSM):
    x' = x + ε · sign(∇_x L(f(x), y))

Multi-step variant (PGD-style) supported via `steps` > 1:
    x_0 = x
    x_{t+1} = clip(x_t + α · sign(∇_x L), 0, 1)
    where α = ε / steps

Applies only to test nodes (evasion: attacker can observe but not
modify the training procedure).
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from datasets.cora_loader import GraphData
from attacks.base import AttackResult


def gradient_attack(
    graph: GraphData,
    model: nn.Module,
    params: any,
    epsilon: float = 0.1,
    steps: int = 10,
) -> AttackResult:
    """
    FGSM / PGD feature attack on test nodes.

    Args:
        graph:    Clean GraphData.
        model:    Trained GCN (used as white-box surrogate).
        params:   Baseline model parameters.
        epsilon:  Total L∞ perturbation budget per feature.
        steps:    Number of gradient steps (1 = FGSM, >1 = PGD).

    Returns:
        AttackResult with perturbed test-node features.
    """
    alpha = epsilon / steps    # step size
    x = jnp.array(graph.features)
    a_hat = jnp.array(graph.adj_norm)
    labels = jnp.array(graph.labels)
    test_mask = jnp.array(graph.test_mask)

    x_orig = x.copy()
    x_adv  = x

    print(f"  [Gradient Attack] ε={epsilon}, steps={steps}, α={alpha:.4f}")

    for step in range(steps):
        grad = _feature_grad(params, model, x_adv, a_hat, labels, test_mask)

        # Only perturb test nodes
        grad_masked = jnp.where(test_mask[:, None], grad, 0.0)

        # FGSM step
        x_adv = x_adv + alpha * jnp.sign(grad_masked)
        x_adv = jnp.clip(x_adv, 0.0, 1.0)

        # Project to ε-ball around original
        delta = jnp.clip(x_adv - x_orig, -epsilon, epsilon)
        x_adv = jnp.clip(x_orig + delta, 0.0, 1.0)

    perturbed_feats = np.array(x_adv)
    n_perturbed = int(graph.test_mask.sum())

    perturbed = graph.copy()
    perturbed = perturbed.update_features(perturbed_feats)
    perturbed.name = "gradient_attack"

    return AttackResult(
        perturbed_graph=perturbed,
        attack_name="Gradient Attack",
        n_edges_added=0,
        n_edges_removed=0,
        n_features_perturbed=n_perturbed,
        budget_used=n_perturbed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# JAX gradient computation
# ──────────────────────────────────────────────────────────────────────────────

def _test_loss(params, model, x, a_hat, labels, test_mask):
    """Loss on test nodes — what the attacker maximises."""
    _, logits, _ = model.apply({"params": params}, x, a_hat, training=False)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    valid = test_mask & (labels >= 0)
    true_lp = log_probs[jnp.arange(logits.shape[0]),
                         jnp.where(labels >= 0, labels, 0)]
    loss = -jnp.where(valid, true_lp, 0.0).sum() / jnp.maximum(valid.sum(), 1)
    return loss


# Gradient w.r.t. features (argnums=2)
_feature_grad = jax.jit(
    jax.grad(_test_loss, argnums=2),
    static_argnames=("model",),
)
