"""Visualization package — research-paper-quality figures."""
from visualization.bar_charts import plot_accuracy_bar, plot_metrics_grouped
from visualization.line_plots import (
    plot_attack_defense_line,
    plot_temporal_accuracy,
    plot_training_curves,
)
from visualization.graph_viz import plot_graph_comparison, plot_degree_distribution
from visualization.embeddings import plot_embeddings_comparison, plot_embeddings_overlay

__all__ = [
    "plot_accuracy_bar",
    "plot_metrics_grouped",
    "plot_attack_defense_line",
    "plot_temporal_accuracy",
    "plot_training_curves",
    "plot_graph_comparison",
    "plot_degree_distribution",
    "plot_embeddings_comparison",
    "plot_embeddings_overlay",
]
