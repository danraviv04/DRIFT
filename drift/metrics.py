"""Public fusion descriptor API, shared with the installed metric-analysis package."""
from metric_analysis.pipeline.library.descriptor_metrics import (
    METRICS, REGISTRY, Metric, descriptor_values, ks_mean_equal,
    ks_residue_results, relative_change, residue_ks, residue_ks_statistics,
)

__all__ = [
    "METRICS", "REGISTRY", "Metric", "descriptor_values", "ks_mean_equal",
    "ks_residue_results", "relative_change", "residue_ks", "residue_ks_statistics",
]
