"""
Nettack — gradient-based targeted poisoning attack.

Reference: Zügner et al., "Adversarial Attacks on Neural Networks for
Graph Data", KDD 2018.

Strategy (simplified gradient-based approximation):
  For each target node v:
    1. Compute gradient of loss(v) w.r.t. every possible edge flip (i,j)
       using the surrogate: score = |grad_A[i,j]| * (1 - 2*A[i,j])
       The (1-2A) term ensures we reward flips, not reinforcing existing state.
    2. Separately compute feature perturbation scores via gradient of loss
       w.r.t. X[v,:].
    3. Greedily apply the top-scoring perturbations up to budget.

Perturbations modify:
  - Adjacency structure (add/remove edges around target nodes)
  - Node features of target nodes
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional

from datasets.cora_loader import GraphData
from models.train import eval_step, cross_entropy_loss
from attacks.base import AttackResult, edge_budget, diff_edges
from utils.graph_utils import normalize_adjacency
from flax import linen as nn


def nettack(
    graph: GraphData,
    model: nn.Module,
    params: any,
    n_perturbations: int = 5,
    target_nodes: Optional[np.ndarray] = None,
    direct_attack: bool = True,
) -> AttackResult:
    """
    Targeted poisoning attack on specific nodes.

    Args:
        graph:           Clean GraphData.
        model:           Trained GCN model.
        params:          Best model params from baseline training.
        n_perturbations: Number of perturbations per target node.
        target_nodes:    Nodes to attack. Defaults to correctly-classified test nodes.
        direct_attack:   If True, only perturb edges incident to target nodes.

    Returns:
        AttackResult with perturbed graph (ALL target nodes attacked).
    """
    adj   = graph.adj.copy()
    feats = graph.features.copy()
    labels = graph.labels

    # Select target nodes: correctly-classified test nodes
    if target_nodes is None:
        x_j   = jnp.array(graph.features)
        a_j   = jnp.array(graph.adj_norm)
        _, _, acc, _, preds = _eval_full(params, model, x_j, a_j, labels,
                                          graph.test_mask)
        correct = np.array(preds) == labels
        target_nodes = np.where(graph.test_mask & correct)[0][:20]  # attack up to 20

    print(f"  [Nettack] Attacking {len(target_nodes)} target nodes, "
          f"{n_perturbations} perturbations each...")

    for v in target_nodes:
        adj, feats = _attack_single_node(
            v, adj, feats, labels, model, params, n_perturbations, direct_attack
        )

    new_adj_norm = normalize_adjacency(adj)
    perturbed = graph.copy()
    perturbed = perturbed.update_adj(adj)
    perturbed = perturbed.update_features(feats)
    perturbed.name = "nettack"

    added, removed = diff_edges(graph.adj, adj)
    feat_diff = int((feats != graph.features).any(axis=1).sum())

    return AttackResult(
        perturbed_graph=perturbed,
        attack_name="Nettack",
        n_edges_added=added,
        n_edges_removed=removed,
        n_features_perturbed=feat_diff,
        budget_used=len(target_nodes) * n_perturbations,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-node attack
# ──────────────────────────────────────────────────────────────────────────────

def _attack_single_node(v, adj, feats, labels, model, params,
                         n_perturbations, direct_attack):
    n = adj.shape[0]

    for _ in range(n_perturbations):
        a_hat = normalize_adjacency(adj)
        x_j   = jnp.array(feats)
        a_j   = jnp.array(a_hat)
        lbl_j = jnp.array(labels)

        # Gradient of loss(v) w.r.t. adjacency
        grad_adj = _grad_adj(params, model, x_j, a_j, lbl_j, v, n)

        # Candidate scores: score(i,j) = |grad| * flip_direction
        # flip_direction = 1 - 2*A[i,j]  (positive when flipping adds value)
        flip_dir = 1.0 - 2.0 * adj
        scores_adj = np.abs(grad_adj) * flip_dir

        if direct_attack:
            # Only edges incident to v
            mask = np.zeros((n, n), dtype=bool)
            mask[v, :] = True
            mask[:, v] = True
            np.fill_diagonal(mask, False)
            scores_adj = np.where(mask, scores_adj, -np.inf)
        else:
            np.fill_diagonal(scores_adj, -np.inf)

        # Best edge flip
        best_edge = int(np.argmax(scores_adj))
        i_e, j_e = best_edge // n, best_edge % n
        if scores_adj[i_e, j_e] > -np.inf:
            adj[i_e, j_e] = 1.0 - adj[i_e, j_e]
            adj[j_e, i_e] = adj[i_e, j_e]

        # Gradient of loss(v) w.r.t. features[v]
        grad_feat = _grad_feat(params, model, x_j, a_j, lbl_j, v)
        feat_score = np.abs(grad_feat)
        best_f = int(np.argmax(feat_score))
        feats[v, best_f] = 1.0 - feats[v, best_f]   # flip binary feature

    return adj, feats


# ──────────────────────────────────────────────────────────────────────────────
# JAX gradient helpers
# ──────────────────────────────────────────────────────────────────────────────

def _loss_for_node(params, model, x, a_hat_flat, labels, v, n):
    a_hat = a_hat_flat.reshape(n, n)
    _, logits, _ = model.apply({"params": params}, x, a_hat, training=False)
    log_p = jax.nn.log_softmax(logits[v])
    return -log_p[jnp.where(labels[v] >= 0, labels[v], 0)]


def _grad_adj(params, model, x, a_hat, labels, v, n):
    a_flat = a_hat.reshape(-1)
    grad_fn = jax.grad(_loss_for_node, argnums=3)
    g = grad_fn(params, model, x, a_flat, labels, v, n)
    return np.array(g).reshape(n, n)


def _grad_feat(params, model, x, a_hat, labels, v):
    def loss_fn(x_):
        _, logits, _ = model.apply({"params": params}, x_, a_hat, training=False)
        log_p = jax.nn.log_softmax(logits[v])
        return -log_p[jnp.where(labels[v] >= 0, labels[v], 0)]
    g = jax.grad(loss_fn)(x)
    return np.array(g[v])


def _eval_full(params, model, x, a_hat, labels, mask):
    embeddings, logits, probs = model.apply({"params": params}, x, a_hat,
                                             training=False)
    preds = jnp.argmax(logits, axis=-1)
    valid = mask & (labels >= 0)
    correct = jnp.where(valid, preds == labels, 0).sum()
    acc = correct / jnp.maximum(valid.sum(), 1)
    return embeddings, logits, acc, probs, preds
