"""All evaluation metrics for classification and robustness assessment."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Optional



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
