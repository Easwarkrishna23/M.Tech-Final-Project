"""All evaluation metrics for classification and robustness assessment."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Optional
from scipy.stats import entropy as scipy_entropy



def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> dict:
    """
    Compute accuracy, precision, recall, F1 on (optionally masked) nodes.
    Returns dict with all four metrics.
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def accuracy_drop(baseline_acc: float, attacked_acc: float) -> float:
    """Accuracy Drop = Baseline − Attacked. Higher = worse attack damage."""
    return baseline_acc - attacked_acc


def recovery_rate(baseline_acc: float, attacked_acc: float,
                  defended_acc: float) -> Optional[float]:
    """
    Recovery Rate = (Defended − Attacked) / (Baseline − Attacked).
    1.0 = full recovery, 0.0 = no recovery, >1.0 = surpassed baseline.
    Returns None when attack caused no degradation (denominator ≈ 0),
    which should be reported as 'N/A' rather than a misleading number.
    """
    denom = baseline_acc - attacked_acc
    if abs(denom) < 0.05:
        return None   # < 5pp damage — recovery rate is undefined/misleading
    return (defended_acc - attacked_acc) / denom


def attack_success_rate(target_nodes: np.ndarray, y_true: np.ndarray,
                         y_pred_clean: np.ndarray,
                         y_pred_attacked: np.ndarray) -> float:
    """
    Targeted attack success rate: fraction of originally correct target nodes
    that are misclassified after attack.
    """
    correct_before = y_pred_clean[target_nodes] == y_true[target_nodes]
    if correct_before.sum() == 0:
        return 0.0
    misclassified = y_pred_attacked[target_nodes[correct_before]] != y_true[target_nodes[correct_before]]
    return float(misclassified.mean())


def robustness_summary(baseline: dict, attacked: dict,
                        defended: dict) -> dict:
    """Build a summary dict from three classification_metrics dicts."""
    return {
        "baseline_acc":    baseline["accuracy"],
        "attacked_acc":    attacked["accuracy"],
        "defended_acc":    defended["accuracy"],
        "accuracy_drop":   accuracy_drop(baseline["accuracy"], attacked["accuracy"]),
        "recovery_rate":   recovery_rate(baseline["accuracy"], attacked["accuracy"],
                                         defended["accuracy"]),
        "baseline_f1":     baseline["f1"],
        "attacked_f1":     attacked["f1"],
        "defended_f1":     defended["f1"],
    }


def format_metrics_table(results: dict[str, dict],
                          metric_keys: list[str] | None = None) -> str:
    """Format a dict of {attack_name: metrics_dict} as a markdown table string."""
    if metric_keys is None:
        metric_keys = ["accuracy", "precision", "recall", "f1"]

    header = "| Attack | " + " | ".join(k.capitalize() for k in metric_keys) + " |"
    sep =    "| --- | " + " | ".join(["---"] * len(metric_keys)) + " |"
    rows = []
    for attack_name, m in results.items():
        vals = " | ".join(f"{m.get(k, 0.0):.4f}" for k in metric_keys)
        rows.append(f"| {attack_name} | {vals} |")

    return "\n".join([header, sep] + rows)


def format_defense_table(results: dict[str, dict]) -> str:
    """Format Table 2: defense performance table."""
    header = "| Attack | After Attack | After Defense | Recovery Rate |"
    sep =    "| --- | --- | --- | --- |"
    rows = []
    for attack_name, m in results.items():
        rr = m.get('recovery_rate', None)
        rr_str = f"{rr:.1%}" if rr is not None else "N/A (no damage)"
        rows.append(
            f"| {attack_name} "
            f"| {m.get('attacked_acc', 0.0):.4f} "
            f"| {m.get('defended_acc', 0.0):.4f} "
            f"| {rr_str} |"
        )
    return "\n".join([header, sep] + rows)


# ──────────────────────────────────────────────────────────────────────────────
# Advanced metrics (Phase 4+5 extension)
# ──────────────────────────────────────────────────────────────────────────────

def attack_success_rate_targeted(
    target_nodes: np.ndarray,
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_attacked: np.ndarray,
) -> float:
    """
    ASR for targeted attacks (Nettack): fraction of correctly-classified
    target nodes whose predicted label changed after the attack.
    Returns 0.0 if no target nodes were correctly classified initially.
    """
    initially_correct = target_nodes[y_pred_clean[target_nodes] == y_true[target_nodes]]
    if len(initially_correct) == 0:
        return 0.0
    flipped = (y_pred_attacked[initially_correct] != y_true[initially_correct]).sum()
    return float(flipped) / len(initially_correct)


def attack_success_rate_global(
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_attacked: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    ASR for global/untargeted attacks: fraction of test-set nodes whose
    prediction changed (regardless of whether it became wrong).
    """
    if mask is not None:
        y_pred_clean    = y_pred_clean[mask]
        y_pred_attacked = y_pred_attacked[mask]
    if len(y_pred_clean) == 0:
        return 0.0
    return float((y_pred_attacked != y_pred_clean).mean())


def node_level_recovery_rate(
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_attacked: np.ndarray,
    y_pred_defended: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Optional[float]:
    """
    Node-level Recovery Rate: of the nodes that were flipped by the attack
    (originally correct, now wrong), what fraction return to the correct
    label after defense?

    Returns None when no nodes were misclassified by the attack.
    """
    if mask is not None:
        y_true          = y_true[mask]
        y_pred_clean    = y_pred_clean[mask]
        y_pred_attacked = y_pred_attacked[mask]
        y_pred_defended = y_pred_defended[mask]

    # Nodes that were originally correct but are now wrong after attack
    flipped_mask = (y_pred_clean == y_true) & (y_pred_attacked != y_true)
    n_flipped = flipped_mask.sum()
    if n_flipped == 0:
        return None

    # Of those, how many are correctly classified after defense?
    recovered = (y_pred_defended[flipped_mask] == y_true[flipped_mask]).sum()
    return float(recovered) / n_flipped


def neighborhood_entropy(
    adj: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Mean neighborhood entropy over (masked) nodes.

    For each node v, compute the class distribution of its neighbors
    and calculate Shannon entropy. Higher entropy = more class-heterogeneous
    neighborhood = more structural chaos / cross-class mixing from attacks.

    Returns the mean entropy across masked nodes.
    """
    nodes = np.where(mask)[0] if mask is not None else np.arange(adj.shape[0])
    entropies = []
    for v in nodes:
        nbrs = np.where(adj[v] > 0)[0]
        if len(nbrs) == 0:
            entropies.append(0.0)
            continue
        nbr_labels = labels[nbrs]
        counts = np.bincount(nbr_labels.astype(int), minlength=num_classes).astype(float)
        p = counts / counts.sum()
        p = p[p > 0]   # remove zero-probability classes before entropy
        entropies.append(float(scipy_entropy(p)))
    return float(np.mean(entropies)) if entropies else 0.0


def embedding_drift(
    embeddings_clean: np.ndarray,
    embeddings_attacked: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Mean L2 drift of GCN node embeddings between clean and attacked graph.

    Higher drift = attack caused larger representation shift, indicating
    that the GCN's internal features were disrupted beyond the prediction level.

    Args:
        embeddings_clean:    [N, D] layer-1 activations on clean graph.
        embeddings_attacked: [N, D] layer-1 activations on attacked graph.
        mask:                Boolean mask for nodes to include.

    Returns:
        Scalar mean L2 drift over masked nodes.
    """
    delta = embeddings_attacked - embeddings_clean
    norms = np.linalg.norm(delta, axis=1)   # [N]
    if mask is not None:
        norms = norms[mask]
    return float(norms.mean()) if len(norms) > 0 else 0.0


def advanced_metrics_summary(
    adj_clean: np.ndarray,
    adj_attacked: np.ndarray,
    adj_defended: np.ndarray,
    labels: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_attacked: np.ndarray,
    y_pred_defended: np.ndarray,
    embeddings_clean: np.ndarray,
    embeddings_attacked: np.ndarray,
    embeddings_defended: np.ndarray,
    test_mask: np.ndarray,
    num_classes: int,
    target_nodes: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all advanced metrics in one call.

    Returns a dict suitable for printing/saving alongside standard metrics.
    """
    asr_global = attack_success_rate_global(
        labels, y_pred_clean, y_pred_attacked, mask=test_mask
    )
    asr_targeted = (
        attack_success_rate_targeted(
            target_nodes, labels, y_pred_clean, y_pred_attacked
        ) if target_nodes is not None else None
    )
    nlrr = node_level_recovery_rate(
        labels, y_pred_clean, y_pred_attacked, y_pred_defended, mask=test_mask
    )
    ent_clean    = neighborhood_entropy(adj_clean,    labels, num_classes, test_mask)
    ent_attacked = neighborhood_entropy(adj_attacked, labels, num_classes, test_mask)
    ent_defended = neighborhood_entropy(adj_defended, labels, num_classes, test_mask)
    drift_attack  = embedding_drift(embeddings_clean, embeddings_attacked,    test_mask)
    drift_defense = embedding_drift(embeddings_attacked, embeddings_defended, test_mask)

    h_drop  = homophily_drop(adj_clean, adj_attacked, labels)
    be_fit  = bose_einstein_fitness(adj_clean, adj_attacked)
    assort_c = assortativity_coefficient(adj_clean)
    assort_a = assortativity_coefficient(adj_attacked)
    assort_d = assortativity_coefficient(adj_defended)
    clr = clean_label_recovery(labels, y_pred_attacked, y_pred_defended, test_mask)

    return {
        "asr_global":           asr_global,
        "asr_targeted":         asr_targeted,
        "node_recovery_rate":   nlrr,
        "neighborhood_entropy_clean":    ent_clean,
        "neighborhood_entropy_attacked": ent_attacked,
        "neighborhood_entropy_defended": ent_defended,
        "entropy_delta_attack":  ent_attacked - ent_clean,
        "entropy_delta_defense": ent_defended - ent_attacked,
        "embedding_drift_attack":  drift_attack,
        "embedding_drift_defense": drift_defense,
        "homophily_drop":          h_drop,
        "bose_einstein_fitness":   be_fit,
        "assortativity_clean":     assort_c,
        "assortativity_attacked":  assort_a,
        "assortativity_defended":  assort_d,
        "assortativity_delta_attack":  assort_a - assort_c,
        "assortativity_delta_defense": assort_d - assort_a,
        "clean_label_recovery":    clr,
    }


# ── Structural superiority metrics (vs GNNGUARD baseline) ────────────────────

def edge_homophily(adj: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of edges connecting same-class nodes (edge homophily ratio)."""
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    if len(rows) == 0:
        return 0.0
    return float((labels[rows] == labels[cols]).mean())


def homophily_drop(
    adj_clean: np.ndarray,
    adj_attacked: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Reduction in edge homophily caused by the attack.
    Proves the attack successfully injected cross-class edges.
    Higher value = more structural damage to the homophily assumption.
    """
    return edge_homophily(adj_clean, labels) - edge_homophily(adj_attacked, labels)


def bose_einstein_fitness(
    adj_clean: np.ndarray,
    adj_attacked: np.ndarray,
) -> float:
    """
    KL-divergence between attacked and clean degree distributions.

    Quantifies how 'unnatural' the attack's degree changes are relative to
    the original degree sequence. A high score means the attack created
    anomalous degree patterns that cosine-similarity filters miss but our
    defense's structural analysis detects.

    Returns KL(P_attacked || P_clean). Higher = more distorted degree sequence.
    """
    def _normed_dist(adj):
        degrees = adj.sum(axis=1).astype(int)
        max_d = max(int(degrees.max()), 1)
        counts = np.bincount(degrees, minlength=max_d + 2).astype(float)
        counts += 1e-8
        return counts / counts.sum()

    p_c = _normed_dist(adj_clean)
    p_a = _normed_dist(adj_attacked)
    max_len = max(len(p_c), len(p_a))
    p_c = np.pad(p_c, (0, max_len - len(p_c)), constant_values=1e-8)
    p_a = np.pad(p_a, (0, max_len - len(p_a)), constant_values=1e-8)
    p_c /= p_c.sum()
    p_a /= p_a.sum()
    return float(scipy_entropy(p_a, p_c))


def assortativity_coefficient(adj: np.ndarray) -> float:
    """
    Degree assortativity coefficient (Newman 2002).

    Measures degree–degree correlations across edges. Adversarial edges that
    connect hubs to low-degree nodes shift this toward -1 (dissortative),
    revealing the topological perturbation even when feature similarity is intact.

    Returns float in [-1, 1]. Typical clean Cora ≈ -0.07 to -0.15.
    """
    import networkx as nx
    try:
        G = nx.from_numpy_array(adj)
        return float(nx.degree_assortativity_coefficient(G))
    except Exception:
        return 0.0


def clean_label_recovery(
    y_true: np.ndarray,
    y_pred_attacked: np.ndarray,
    y_pred_defended: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Clean Label Recovery: fraction of post-attack misclassified nodes that
    the defense correctly restores to their ground-truth label.

    Unlike node_level_recovery_rate (checks prediction-change only), this
    verifies y_pred_defended[v] == y_true[v], proving the defense restores
    truth rather than just changing the label.

    Returns 1.0 when no nodes are misclassified after attack.
    """
    if mask is not None:
        y_true          = y_true[mask]
        y_pred_attacked = y_pred_attacked[mask]
        y_pred_defended = y_pred_defended[mask]

    wrong_after_attack = y_pred_attacked != y_true
    n_wrong = int(wrong_after_attack.sum())
    if n_wrong == 0:
        return 1.0

    restored = int(
        (y_pred_defended[wrong_after_attack] == y_true[wrong_after_attack]).sum()
    )
    return float(restored) / n_wrong
