"""
Nettack — margin-based targeted poisoning attack.

Reference: Zügner et al., "Adversarial Attacks on Neural Networks for
Graph Data", KDD 2018.

Strategy (classification-margin scoring):
  For each target node v:
    1. Score each candidate edge flip (i,j) by its effect on the
       CLASSIFICATION MARGIN of v:
         margin = logit[true_class] - max(logit[other_classes])
       The flip that most REDUCES this margin is chosen (best for attacker).
       This is evaluated by a forward pass on a locally-perturbed adjacency,
       restricted to edges incident to v (direct attack) to keep cost tractable.
    2. Separately score feature flips by margin reduction on X[v,:].
    3. Greedily apply the top-scoring perturbations up to budget.

Why margin scoring beats raw gradient magnitude:
  Gradient magnitude measures sensitivity of loss, not of the actual decision
  boundary. Two perturbations may have equal gradient magnitude but very
  different effects on whether the node is misclassified. Margin scoring
  directly measures the decision-boundary effect and reliably achieves
  non-zero Attack Success Rates.
"""
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
    true_label = int(labels[v])

    for _ in range(n_perturbations):
        a_hat = normalize_adjacency(adj)
        x_j   = jnp.array(feats)
        a_j   = jnp.array(a_hat)
        lbl_j = jnp.array(labels)

        # ── Structural perturbation: margin-based scoring ────────────────────
        # Candidate set: 2-hop neighborhood of v (direct attack) or all edges.
        # Restricting to 2-hop keeps candidates ~50-150 nodes instead of 2708,
        # making margin scoring tractable without sacrificing targeting quality.
        if direct_attack:
            hop1 = set(np.where(adj[v] > 0)[0])
            hop2 = set()
            for u in hop1:
                hop2.update(np.where(adj[u] > 0)[0])
            neighborhood = (hop1 | hop2) - {v}
            if len(neighborhood) == 0:
                neighborhood = set(range(n)) - {v}
            candidates = [(v, j) for j in neighborhood] + \
                         [(i, v) for i in neighborhood if i != v]
            # deduplicate
            candidates = list({(min(a,b), max(a,b)) for a, b in candidates})
        else:
            rows, cols = np.where(np.triu(np.ones((n, n), dtype=bool), k=1))
            candidates = list(zip(rows.tolist(), cols.tolist()))

        best_margin_drop = -np.inf
        best_ie, best_je = -1, -1

        for i_e, j_e in candidates:
            # Simulate this flip
            adj_try = adj.copy()
            adj_try[i_e, j_e] = 1.0 - adj_try[i_e, j_e]
            adj_try[j_e, i_e] = adj_try[i_e, j_e]

            a_hat_try = jnp.array(normalize_adjacency(adj_try))
            _, logits_try, _ = model.apply(
                {"params": params}, x_j, a_hat_try, training=False
            )
            logits_v = np.array(logits_try[v])
            margin = _classification_margin(logits_v, true_label)
            # Lower margin = harder to classify correctly = better for attacker
            if -margin > best_margin_drop:
                best_margin_drop = -margin
                best_ie, best_je = i_e, j_e

        if best_ie >= 0:
            adj[best_ie, best_je] = 1.0 - adj[best_ie, best_je]
            adj[best_je, best_ie] = adj[best_ie, best_je]

        # ── Feature perturbation: gradient pre-filter then margin score ─────
        # Pre-filter: use gradient magnitude to select top-K feature candidates,
        # then do accurate margin scoring only on those K.
        # This avoids 1433 forward passes per step while keeping quality high.
        a_hat_cur = jnp.array(normalize_adjacency(adj))
        x_cur     = jnp.array(feats)

        def _feat_loss(x_):
            _, logits_, _ = model.apply({"params": params}, x_, a_hat_cur, training=False)
            lp = jax.nn.log_softmax(logits_[v])
            return -lp[jnp.where(labels[v] >= 0, labels[v], 0)]

        import jax
        grad_feat = np.abs(np.array(jax.grad(_feat_loss)(x_cur)[v]))
        top_k = min(50, feats.shape[1])
        top_feats = np.argsort(grad_feat)[-top_k:]   # top-K by gradient magnitude

        best_feat_drop = -np.inf
        best_f = -1
        for f in top_feats:
            feats_try = feats.copy()
            feats_try[v, f] = 1.0 - feats_try[v, f]
            x_try = jnp.array(feats_try)
            _, logits_try, _ = model.apply(
                {"params": params}, x_try, a_hat_cur, training=False
            )
            logits_v = np.array(logits_try[v])
            margin = _classification_margin(logits_v, true_label)
            if -margin > best_feat_drop:
                best_feat_drop = -margin
                best_f = f

        if best_f >= 0:
            feats[v, best_f] = 1.0 - feats[v, best_f]

    return adj, feats


def _classification_margin(logits_v: np.ndarray, true_label: int) -> float:
    """margin = logit[true_class] - max(logit[other_classes]). Lower = worse."""
    true_score  = logits_v[true_label]
    other_max   = float(np.max(np.delete(logits_v, true_label)))
    return float(true_score - other_max)


def _eval_full(params, model, x, a_hat, labels, mask):
    embeddings, logits, probs = model.apply({"params": params}, x, a_hat,
                                             training=False)
    preds = jnp.argmax(logits, axis=-1)
    valid = mask & (labels >= 0)
    correct = jnp.where(valid, preds == labels, 0).sum()
    acc = correct / jnp.maximum(valid.sum(), 1)
    return embeddings, logits, acc, probs, preds
