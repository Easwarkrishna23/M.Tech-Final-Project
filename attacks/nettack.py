"""
Nettack — gradient-based targeted poisoning attack.

Reference: Zügner et al., "Adversarial Attacks on Neural Networks for
Graph Data", KDD 2018.

Strategy (gradient-based scoring — fast path):
  For each target node v and each perturbation step:
    Structure: compute ∂(margin_loss) / ∂(A_hat) in ONE backward pass.
               Score each candidate edge by grad * (1 - 2*A[i,j]).
               Select the highest-scoring edge in v's 2-hop neighbourhood.

    Feature:   compute ∂(CE_loss) / ∂(x[v]) in ONE backward pass.
               Flip the feature with highest gradient magnitude.

  Both gradient functions are JIT-compiled once and reused for all target
  nodes. `model` is a Python closure so JAX traces through it statically;
  `v` and `true_label` are dynamic integer arrays so JAX re-uses one
  compiled program for every node.

  Old approach: ~150 forward passes per step (one per candidate edge).
  New approach: 2 JIT-compiled backward passes per step.
  Speedup: ~75× on Cora's 2708-node graph.

Target selection:
  Nodes sorted by DESCENDING margin (highest-confidence first).
  High-margin nodes form the backbone of correct test-set predictions —
  flipping them produces the largest aggregate accuracy drop.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional

from datasets.cora_loader import GraphData
from attacks.base import AttackResult, diff_edges
from utils.graph_utils import normalize_adjacency
from flax import linen as nn


def nettack(
    graph: GraphData,
    model: nn.Module,
    params: any,
    n_perturbations: int = 20,
    target_nodes: Optional[np.ndarray] = None,
    target_count: int = 80,
    direct_attack: bool = True,
) -> AttackResult:
    """
    Targeted poisoning attack on specific nodes.

    Selects targets by DESCENDING classification margin (highest-confidence
    correct predictions first). See module docstring for rationale.

    Args:
        graph:           Clean GraphData.
        model:           Trained GCN model.
        params:          Best model params from baseline training.
        n_perturbations: Perturbations per target node (default 20).
        target_nodes:    Override: explicit node array to attack.
        target_count:    Max number of targets when auto-selecting (default 80).
        direct_attack:   If True, restrict structure candidates to 2-hop.
    """
    adj   = graph.adj.copy()
    feats = graph.features.copy()
    labels = graph.labels

    # ── JIT-compile gradient functions once (model is a Python closure) ──────
    @jax.jit
    def _grad_adj_jit(params, a_hat, x, v, tl):
        """Gradient of negative margin loss w.r.t. A_hat."""
        def _loss(a_):
            _, logits, _ = model.apply({"params": params}, x, a_, training=False)
            nc = logits.shape[1]
            lv = logits[v]
            other_max = jnp.max(jnp.where(jnp.arange(nc) == tl, -jnp.inf, lv))
            return -(lv[tl] - other_max)
        return jax.grad(_loss)(a_hat)

    @jax.jit
    def _grad_feat_jit(params, x, a_hat, v, tl):
        """Gradient of CE loss w.r.t. all node features; index row v after."""
        def _loss(x_):
            _, logits, _ = model.apply({"params": params}, x_, a_hat, training=False)
            lp = jax.nn.log_softmax(logits[v])
            return -lp[tl]
        return jax.grad(_loss)(x)

    # ── Select target nodes: descending margin (highest-confidence first) ────
    if target_nodes is None:
        x_j   = jnp.array(graph.features)
        a_j   = jnp.array(graph.adj_norm)
        _, logits_all, _, _, preds = _eval_full(params, model, x_j, a_j, labels,
                                                 graph.test_mask)
        preds_np  = np.array(preds)
        logits_np = np.array(logits_all)
        correct_mask = (preds_np == labels) & graph.test_mask
        correct_idx  = np.where(correct_mask)[0]

        margins = np.array([
            _classification_margin(logits_np[v], int(labels[v]))
            for v in correct_idx
        ])
        order = np.argsort(-margins)   # descending: highest-confidence first
        target_nodes = correct_idx[order][:target_count]

    print(f"  [Nettack] Attacking {len(target_nodes)} high-confidence nodes, "
          f"{n_perturbations} perturbations each "
          f"(gradient-based, JIT-compiled)...")

    for v in target_nodes:
        adj, feats = _attack_single_node(
            v, adj, feats, labels, model, params, n_perturbations,
            direct_attack, _grad_adj_jit, _grad_feat_jit,
        )

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
# Per-node attack — gradient-based (fast path)
# ──────────────────────────────────────────────────────────────────────────────

def _attack_single_node(v, adj, feats, labels, model, params, n_perturbations,
                         direct_attack, grad_adj_fn, grad_feat_fn):
    n = adj.shape[0]
    true_label = int(labels[v])
    tl_arr = jnp.array(true_label)
    v_arr  = jnp.array(v)

    # Precompute 2-hop neighbourhood (updated after each structural flip)
    if direct_attack:
        neighbourhood = _neighbourhood(v, adj, n)
    else:
        neighbourhood = None

    for _ in range(n_perturbations):
        a_hat_np = normalize_adjacency(adj)
        a_j = jnp.array(a_hat_np)
        x_j = jnp.array(feats)

        # ── Structure: 1 backward pass, O(N²×hidden) ────────────────────────
        grad_adj = np.array(grad_adj_fn(params, a_j, x_j, v_arr, tl_arr))
        flip_dir = 1.0 - 2.0 * adj
        scores   = grad_adj * flip_dir
        np.fill_diagonal(scores, -np.inf)

        if neighbourhood is not None:
            mask = np.full((n, n), -np.inf, dtype=np.float32)
            for nb in neighbourhood:
                mask[v, nb] = scores[v, nb]
                mask[nb, v] = scores[nb, v]
            scores = mask

        best = int(np.argmax(scores))
        i_e, j_e = best // n, best % n
        if scores[i_e, j_e] > -np.inf:
            adj[i_e, j_e] = 1.0 - adj[i_e, j_e]
            adj[j_e, i_e] = adj[i_e, j_e]
            if direct_attack:
                neighbourhood = _neighbourhood(v, adj, n)

        # ── Feature: 1 backward pass, take gradient row v ───────────────────
        a_hat_cur = jnp.array(normalize_adjacency(adj))
        x_cur     = jnp.array(feats)
        grad_feat = np.abs(np.array(
            grad_feat_fn(params, x_cur, a_hat_cur, v_arr, tl_arr)
        )[v])
        best_f = int(np.argmax(grad_feat))
        feats[v, best_f] = 1.0 - feats[v, best_f]

    return adj, feats


def _neighbourhood(v, adj, n):
    hop1 = set(np.where(adj[v] > 0)[0])
    hop2 = set()
    for u in hop1:
        hop2.update(np.where(adj[u] > 0)[0])
    nb = sorted((hop1 | hop2) - {v})
    return nb if nb else list(set(range(n)) - {v})


def _classification_margin(logits_v: np.ndarray, true_label: int) -> float:
    true_score = logits_v[true_label]
    other_max  = float(np.max(np.delete(logits_v, true_label)))
    return float(true_score - other_max)


def _eval_full(params, model, x, a_hat, labels, mask):
    embeddings, logits, probs = model.apply({"params": params}, x, a_hat,
                                             training=False)
    preds = jnp.argmax(logits, axis=-1)
    valid = mask & (labels >= 0)
    correct = jnp.where(valid, preds == labels, 0).sum()
    acc = correct / jnp.maximum(valid.sum(), 1)
    return embeddings, logits, acc, probs, preds
