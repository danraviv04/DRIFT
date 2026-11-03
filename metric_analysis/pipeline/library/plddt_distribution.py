"""Distribution-sensitive WT-vs-mutant metrics for per-residue pLDDT ensembles.

The established D1 score measures only the absolute relative change in the
per-residue mean.  This module keeps D1 untouched and adds descriptive effect
sizes that compare the complete empirical distributions.  No class labels enter
any calculation in this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metric_analysis.pipeline.library.bootstrap_uncertainty import bootstrap_mean_sd


PLDDT_SCALE = 100.0
DEFAULT_OUTLIER_THRESHOLD = 0.01


def _threshold_from_frame(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(
        frame.get(column, pd.Series(dtype=float)), errors="coerce"
    ).dropna().unique()
    return float(values[0]) if len(values) else DEFAULT_OUTLIER_THRESHOLD


def _validate_inputs(
    wt_matrix: np.ndarray,
    mutant_matrix: np.ndarray,
    residue_indices: Sequence[object] | None,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wt = np.asarray(wt_matrix, dtype=float)
    mutant = np.asarray(mutant_matrix, dtype=float)
    if wt.ndim != 2 or mutant.ndim != 2:
        raise ValueError("WT and mutant pLDDT inputs must be two-dimensional")
    if wt.shape[0] != mutant.shape[0]:
        raise ValueError(
            "WT/mutant residue-count mismatch: "
            f"{wt.shape[0]} vs {mutant.shape[0]}"
        )
    if wt.shape[0] == 0:
        raise ValueError("pLDDT matrices must contain at least one residue")
    if int(min_samples) < 1:
        raise ValueError("min_samples must be a positive integer")
    if residue_indices is None:
        indices = np.arange(wt.shape[0])
    else:
        indices = np.asarray(list(residue_indices), dtype=object)
        if indices.shape != (wt.shape[0],):
            raise ValueError(
                "residue_indices length must equal the number of matrix rows"
            )
    return wt, mutant, indices


def _metrics_for_finite_rows(
    wt: np.ndarray,
    mutant: np.ndarray,
    *,
    outlier_threshold: float,
) -> dict[str, np.ndarray]:
    """Vectorized exact one-dimensional empirical metrics for finite rows.

    The merged ordered samples provide all required integrals.  This supports
    unequal WT/mutant ensemble sizes and handles ties without arbitrary within-tie
    ordering effects.
    """
    n_wt, n_mutant = wt.shape[1], mutant.shape[1]
    values = np.concatenate([wt, mutant], axis=1)
    is_wt = np.broadcast_to(
        np.r_[np.ones(n_wt, dtype=bool), np.zeros(n_mutant, dtype=bool)],
        values.shape,
    )
    order = np.argsort(values, axis=1, kind="stable")
    ordered = np.take_along_axis(values, order, axis=1)
    ordered_is_wt = np.take_along_axis(is_wt, order, axis=1)
    cumulative_wt = np.cumsum(ordered_is_wt, axis=1, dtype=float)
    cumulative_mutant = np.cumsum(~ordered_is_wt, axis=1, dtype=float)
    ecdf_difference = (
        cumulative_wt / float(n_wt)
        - cumulative_mutant / float(n_mutant)
    )
    gaps = np.diff(ordered, axis=1)

    # W1 = integral |F-G| dt. Zero-width tied intervals contribute exactly zero.
    wasserstein = np.sum(np.abs(ecdf_difference[:, :-1]) * gaps, axis=1)

    # Pairwise absolute-difference sums from counts on either side of each gap.
    left_wt = cumulative_wt[:, :-1]
    left_mutant = cumulative_mutant[:, :-1]
    cross_sum = np.sum(
        gaps
        * (
            left_wt * (n_mutant - left_mutant)
            + left_mutant * (n_wt - left_wt)
        ),
        axis=1,
    )
    within_wt_sum = np.sum(
        gaps * 2.0 * left_wt * (n_wt - left_wt), axis=1
    )
    within_mutant_sum = np.sum(
        gaps * 2.0 * left_mutant * (n_mutant - left_mutant), axis=1
    )
    energy_squared_raw = (
        2.0 * cross_sum / float(n_wt * n_mutant)
        - within_wt_sum / float(n_wt * n_wt)
        - within_mutant_sum / float(n_mutant * n_mutant)
    )
    # Conventional energy distance on pLDDT/100. Tiny negative values can arise
    # only through floating-point cancellation; the population V-statistic is >=0.
    energy = np.sqrt(np.maximum(energy_squared_raw / PLDDT_SCALE, 0.0))

    # CvM effect size: integral (F-G)^2 dH for the pooled empirical measure H.
    # Evaluate after complete tie groups and weight by each group's pooled mass.
    group_start = np.ones(values.shape, dtype=bool)
    group_start[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    group_end = np.ones(values.shape, dtype=bool)
    group_end[:, :-1] = ordered[:, :-1] != ordered[:, 1:]
    positions = np.arange(values.shape[1], dtype=int)[None, :]
    latest_start = np.maximum.accumulate(
        np.where(group_start, positions, -1), axis=1
    )
    group_size = positions - latest_start + 1
    cvm = np.sum(
        np.where(
            group_end,
            ecdf_difference**2 * group_size / float(values.shape[1]),
            0.0,
        ),
        axis=1,
    )

    wt_mean = np.mean(wt, axis=1)
    mutant_mean = np.mean(mutant, axis=1)
    location = np.abs(mutant_mean - wt_mean)
    # Kantorovich-Rubinstein implies W1 >= |E[X]-E[Y]|. Clipping only removes
    # round-off and makes the decomposition numerically stable.
    shape_excess = np.maximum(wasserstein - location, 0.0)
    valid_denominator = np.abs(wt_mean) > np.finfo(float).eps
    relative = np.divide(
        wasserstein,
        np.abs(wt_mean),
        out=np.full_like(wasserstein, np.nan),
        where=valid_denominator,
    )
    location_relative = np.divide(
        location,
        np.abs(wt_mean),
        out=np.full_like(location, np.nan),
        where=valid_denominator,
    )
    shape_relative = np.divide(
        shape_excess,
        np.abs(wt_mean),
        out=np.full_like(shape_excess, np.nan),
        where=valid_denominator,
    )
    shape_fraction = np.divide(
        shape_excess,
        wasserstein,
        out=np.zeros_like(shape_excess),
        where=wasserstein > np.finfo(float).eps,
    )

    # Supervisor-defined thresholded outlier metric.  For residue r and mutant
    # conformation i, L_ri = |m_ri / median(WT_r) - 1|.  S_r is the sum of L_ri
    # above tau divided by every finite mutant conformation, so it jointly reflects
    # outlier prevalence and magnitude.  No WT/mutant conformation pairing is used.
    wt_median = np.median(wt, axis=1)
    valid_wt_median = np.abs(wt_median) > np.finfo(float).eps
    relative_deviation = np.divide(
        mutant,
        wt_median[:, None],
        out=np.full_like(mutant, np.nan),
        where=valid_wt_median[:, None],
    )
    relative_deviation = np.abs(relative_deviation - 1.0)
    selected = relative_deviation > float(outlier_threshold)
    selected_sum = np.nansum(np.where(selected, relative_deviation, 0.0), axis=1)
    outlier_count = selected.sum(axis=1).astype(float)
    outlier_score = selected_sum / float(n_mutant)
    outlier_fraction = outlier_count / float(n_mutant)
    outlier_mean_deviation = np.divide(
        selected_sum,
        outlier_count,
        out=np.zeros_like(selected_sum),
        where=outlier_count > 0,
    )
    for values_out in (
        outlier_score, outlier_fraction, outlier_mean_deviation, outlier_count,
    ):
        values_out[~valid_wt_median] = np.nan
    return {
        "wasserstein": wasserstein,
        "wasserstein_unit": wasserstein / PLDDT_SCALE,
        "wasserstein_relative": relative,
        "location_shift": location,
        "location_shift_relative": location_relative,
        "shape_excess": shape_excess,
        "shape_excess_relative": shape_relative,
        "shape_fraction": shape_fraction,
        "energy_distance": energy,
        "cvm_effect": cvm,
        "wt_median": wt_median,
        "outlier_score": outlier_score,
        "outlier_fraction": outlier_fraction,
        "outlier_mean_deviation": outlier_mean_deviation,
        "outlier_count": outlier_count,
    }


def compute_per_residue_plddt_distribution_metrics(
    wt_matrix: np.ndarray,
    mutant_matrix: np.ndarray,
    *,
    residue_indices: Sequence[object] | None = None,
    min_samples: int = 2,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> pd.DataFrame:
    """Compare complete WT and mutant pLDDT distributions at each residue.

    Returned effect sizes are descriptive, not p-values. ``wasserstein`` and
    ``shape_excess`` are in pLDDT points; ``wasserstein_unit`` uses pLDDT/100;
    relative fields divide by the absolute WT residue mean. ``energy_distance``
    is the conventional energy distance after scaling pLDDT to [0, 1], and
    ``cvm_effect`` is the unscaled empirical integral against the pooled CDF.
    ``outlier_score`` is the thresholded mean absolute relative deviation from
    the WT residue median, using every finite mutant conformation in its
    denominator. Non-finite observations are removed separately for each residue.
    """
    wt, mutant, indices = _validate_inputs(
        wt_matrix, mutant_matrix, residue_indices, min_samples
    )
    outlier_threshold = float(outlier_threshold)
    if not np.isfinite(outlier_threshold) or outlier_threshold < 0.0:
        raise ValueError("outlier_threshold must be finite and non-negative")
    metric_names = (
        "wasserstein", "wasserstein_unit", "wasserstein_relative",
        "location_shift", "location_shift_relative", "shape_excess",
        "shape_excess_relative", "shape_fraction", "energy_distance",
        "cvm_effect", "wt_median", "outlier_score", "outlier_fraction",
        "outlier_mean_deviation", "outlier_count",
    )
    output = {name: np.full(wt.shape[0], np.nan) for name in metric_names}
    n_wt = np.isfinite(wt).sum(axis=1).astype(int)
    n_mutant = np.isfinite(mutant).sum(axis=1).astype(int)

    fully_finite = np.isfinite(wt).all(axis=1) & np.isfinite(mutant).all(axis=1)
    eligible = (n_wt >= int(min_samples)) & (n_mutant >= int(min_samples))
    fast_rows = fully_finite & eligible
    if fast_rows.any():
        fast = _metrics_for_finite_rows(
            wt[fast_rows], mutant[fast_rows],
            outlier_threshold=outlier_threshold,
        )
        for name in metric_names:
            output[name][fast_rows] = fast[name]

    # Missing values make per-row sample counts unequal. Keep the common case
    # vectorized and use the same exact implementation only for affected rows.
    for row_idx in np.flatnonzero(eligible & ~fully_finite):
        wt_values = wt[row_idx, np.isfinite(wt[row_idx])][None, :]
        mutant_values = mutant[row_idx, np.isfinite(mutant[row_idx])][None, :]
        row = _metrics_for_finite_rows(
            wt_values, mutant_values, outlier_threshold=outlier_threshold,
        )
        for name in metric_names:
            output[name][row_idx] = row[name][0]

    return pd.DataFrame(
        {
            "residue_index": indices,
            **output,
            "outlier_threshold": np.full(wt.shape[0], outlier_threshold),
            "n_wt": n_wt,
            "n_mutant": n_mutant,
        }
    )


def summarize_per_residue_plddt_distribution_metrics(
    residue_results: pd.DataFrame,
    *,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> dict[str, float | int]:
    """Reduce a residue profile to predeclared mutation-level effect summaries."""
    required = {
        "wasserstein", "wasserstein_unit", "wasserstein_relative",
        "shape_excess", "shape_excess_relative", "shape_fraction",
        "energy_distance", "cvm_effect", "location_shift_relative",
        "outlier_score", "outlier_fraction", "outlier_mean_deviation",
        "outlier_count", "outlier_threshold", "wt_median",
        "n_wt", "n_mutant",
    }
    missing = sorted(required.difference(residue_results.columns))
    if missing:
        raise ValueError(
            f"Distribution residue table is missing required columns: {missing}"
        )
    valid = pd.to_numeric(
        residue_results["wasserstein"], errors="coerce"
    ).notna()
    thresholds = pd.to_numeric(
        residue_results["outlier_threshold"], errors="coerce"
    ).dropna().unique()
    if len(thresholds) > 1:
        raise ValueError("Distribution residue table mixes outlier thresholds")
    threshold = float(thresholds[0]) if len(thresholds) else float(outlier_threshold)
    if not valid.any():
        return {
            "plddt_w1_mean": np.nan,
            "plddt_w1_median": np.nan,
            "plddt_w1_q90": np.nan,
            "plddt_w1_unit_mean": np.nan,
            "plddt_w1_relative_mean": np.nan,
            "plddt_w1_shape_excess_mean": np.nan,
            "plddt_w1_shape_excess_relative_mean": np.nan,
            "plddt_w1_shape_fraction": np.nan,
            "plddt_energy_mean": np.nan,
            "plddt_cvm_mean": np.nan,
            "plddt_outlier_score_mean": np.nan,
            "plddt_outlier_score_median": np.nan,
            "plddt_outlier_score_q90": np.nan,
            "plddt_outlier_fraction_mean": np.nan,
            "plddt_outlier_magnitude_mean": np.nan,
            "plddt_outlier_threshold": threshold,
            "plddt_outlier_n_residues": 0,
            "d1_from_distribution_mean": np.nan,
            "plddt_distribution_n_residues": 0,
            "plddt_distribution_min_wt_samples": 0,
            "plddt_distribution_min_mutant_samples": 0,
        }

    def mean(column: str) -> float:
        return float(pd.to_numeric(
            residue_results.loc[valid, column], errors="coerce"
        ).mean())

    w1 = pd.to_numeric(residue_results.loc[valid, "wasserstein"], errors="coerce")
    outlier_score = pd.to_numeric(
        residue_results.loc[valid, "outlier_score"], errors="coerce"
    )
    total_w1 = float(w1.sum())
    total_shape = float(pd.to_numeric(
        residue_results.loc[valid, "shape_excess"], errors="coerce"
    ).sum())
    return {
        "plddt_w1_mean": float(w1.mean()),
        "plddt_w1_median": float(w1.median()),
        "plddt_w1_q90": float(w1.quantile(0.90)),
        "plddt_w1_unit_mean": mean("wasserstein_unit"),
        "plddt_w1_relative_mean": mean("wasserstein_relative"),
        "plddt_w1_shape_excess_mean": mean("shape_excess"),
        "plddt_w1_shape_excess_relative_mean": mean("shape_excess_relative"),
        "plddt_w1_shape_fraction": (
            total_shape / total_w1 if total_w1 > np.finfo(float).eps else 0.0
        ),
        "plddt_energy_mean": mean("energy_distance"),
        "plddt_cvm_mean": mean("cvm_effect"),
        "plddt_outlier_score_mean": float(outlier_score.mean()),
        "plddt_outlier_score_median": float(outlier_score.median()),
        "plddt_outlier_score_q90": float(outlier_score.quantile(0.90)),
        "plddt_outlier_fraction_mean": mean("outlier_fraction"),
        "plddt_outlier_magnitude_mean": mean("outlier_mean_deviation"),
        "plddt_outlier_threshold": threshold,
        "plddt_outlier_n_residues": int(outlier_score.notna().sum()),
        # Algebraically identical to the established per-residue D1 convention;
        # retained as an audit column, not proposed as a new predictor.
        "d1_from_distribution_mean": mean("location_shift_relative"),
        "plddt_distribution_n_residues": int(valid.sum()),
        "plddt_distribution_min_wt_samples": int(
            pd.to_numeric(residue_results.loc[valid, "n_wt"]).min()
        ),
        "plddt_distribution_min_mutant_samples": int(
            pd.to_numeric(residue_results.loc[valid, "n_mutant"]).min()
        ),
    }


def plot_plddt_distribution_profile_by_class(
    residue_results: pd.DataFrame,
    variant_summary: pd.DataFrame,
    *,
    residue_start: int | None = None,
    hotspot_range: Sequence[int] | None = None,
    out_path: Path | str | None = None,
    bootstrap_resamples: int = 1_000,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    """Plot class-mean profiles with mutation-bootstrap SD ribbons."""
    labels = variant_summary[["mutation_code", "mutation_type"]].drop_duplicates(
        "mutation_code"
    )
    work = residue_results.merge(labels, on="mutation_code", how="inner")
    work = work[work["mutation_type"].isin(["Driver", "Passenger"])].copy()
    for col in (
        "residue_index", "wasserstein_unit", "outlier_score",
        "shape_excess_relative",
    ):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if work.empty or set(work["mutation_type"]) != {"Driver", "Passenger"}:
        raise ValueError("Distribution profile needs finite results from both classes")

    colors = {"Driver": "#d1495b", "Passenger": "#2e86ab"}
    threshold = _threshold_from_frame(work, "outlier_threshold")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    specs = (
        ("wasserstein_unit", "Wasserstein-1 (pLDDT / 100)"),
        ("outlier_score", f"Thresholded outlier score (tau={threshold:g})"),
        ("shape_excess_relative", "Shape-excess W1 / |WT mean|"),
    )
    class_counts: dict[str, int] = {}
    rng = np.random.default_rng(int(bootstrap_seed))
    for class_name in ("Driver", "Passenger"):
        subset = work[work["mutation_type"] == class_name]
        class_counts[class_name] = int(subset["mutation_code"].nunique())
        for ax, (metric, ylabel) in zip(axes, specs):
            matrix = subset.pivot_table(
                index="mutation_code", columns="residue_index", values=metric,
                aggfunc="mean",
            ).sort_index(axis=1)
            values = matrix.to_numpy(dtype=float)
            mean = np.nanmean(values, axis=0)
            bootstrap_sd = bootstrap_mean_sd(
                values, n_resamples=bootstrap_resamples, rng=rng,
            )
            x = matrix.columns.to_numpy(dtype=float)
            if residue_start is not None:
                x = x + int(residue_start)
            y = np.asarray(mean, dtype=float)
            err = np.asarray(bootstrap_sd, dtype=float)
            ax.fill_between(
                x, np.maximum(y - err, 0.0), y + err,
                color=colors[class_name], alpha=0.18, linewidth=0,
            )
            ax.plot(
                x, y, color=colors[class_name], lw=1.4,
                label=(f"{class_name} mean ±bootstrap SD "
                       f"(n={class_counts[class_name]}, B={bootstrap_resamples})"),
            )
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
    if hotspot_range is not None and len(hotspot_range) == 2:
        for ax in axes:
            ax.axvspan(
                float(hotspot_range[0]), float(hotspot_range[1]),
                color="#f4a261", alpha=0.12, label="Configured hotspot",
            )
    axes[0].legend(fontsize=9)
    axes[-1].set_xlabel("Residue")
    fig.suptitle("Per-residue pLDDT distribution shift by class")
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return {
        "figure": fig,
        "class_counts": class_counts,
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
    }


def plot_plddt_distribution_summary(
    variant_summary: pd.DataFrame,
    *,
    dataset_name: str = "Dataset",
    out_path: Path | str | None = None,
) -> dict[str, object]:
    """Class distributions for primary metrics and outlier-score prevalence."""
    threshold = _threshold_from_frame(
        variant_summary, "plddt_outlier_threshold"
    )
    metrics = (
        ("plddt_w1_unit_mean", "Mean W1", "W1 (pLDDT / 100)"),
        (
            "plddt_outlier_score_mean",
            f"Outlier score (tau={threshold:g})",
            "Mean thresholded outlier score",
        ),
        (
            "plddt_outlier_fraction_mean", "Outlier prevalence",
            "Mean fraction above threshold",
        ),
        ("plddt_w1_shape_fraction", "Shape share of W1", "Shape excess / total W1"),
        ("plddt_energy_mean", "Energy distance", "Energy distance"),
        ("plddt_cvm_mean", "CvM effect", "CvM effect"),
    )
    work = variant_summary[
        variant_summary["mutation_type"].isin(["Driver", "Passenger"])
    ].copy()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = {"Driver": "#d1495b", "Passenger": "#2e86ab"}
    for ax, (metric, title, ylabel) in zip(axes.ravel(), metrics):
        groups = []
        positions = []
        for pos, class_name in enumerate(("Driver", "Passenger"), start=1):
            vals = pd.to_numeric(
                work.loc[work["mutation_type"] == class_name, metric],
                errors="coerce",
            ).dropna().to_numpy(float)
            groups.append(vals)
            positions.append(pos)
        parts = ax.violinplot(groups, positions=positions, showextrema=False)
        for body, class_name in zip(parts["bodies"], ("Driver", "Passenger")):
            body.set_facecolor(colors[class_name])
            body.set_edgecolor("black")
            body.set_alpha(0.65)
        for pos, vals, class_name in zip(
            positions, groups, ("Driver", "Passenger")
        ):
            if vals.size:
                jitter = np.linspace(-0.07, 0.07, vals.size)
                ax.scatter(
                    pos + jitter, vals, s=10, alpha=0.35,
                    color=colors[class_name], edgecolors="none",
                )
                ax.scatter(pos, np.median(vals), marker="D", s=28, color="black")
        ax.set_xticks(positions, ["Driver", "Passenger"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{dataset_name}: full-distribution pLDDT metrics")
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return {"figure": fig}


def plot_plddt_distribution_vs_d1(
    variant_summary: pd.DataFrame,
    *,
    dataset_name: str = "Dataset",
    out_path: Path | str | None = None,
) -> dict[str, object]:
    """Compare new distribution metrics with D1 and report rank correlation."""
    threshold = _threshold_from_frame(
        variant_summary, "plddt_outlier_threshold"
    )
    metrics = (
        ("plddt_w1_unit_mean", "Mean W1"),
        ("plddt_outlier_score_mean", f"Outlier score (tau={threshold:g})"),
        ("plddt_w1_shape_fraction", "Shape share of W1"),
        ("plddt_energy_mean", "Energy distance"),
        ("plddt_cvm_mean", "CvM effect"),
    )
    work = variant_summary[
        variant_summary["mutation_type"].isin(["Driver", "Passenger"])
    ].copy()
    colors = {"Driver": "#d1495b", "Passenger": "#2e86ab"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    correlations: dict[str, float] = {}
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        x = pd.to_numeric(work["d1_mean"], errors="coerce")
        y = pd.to_numeric(work[metric], errors="coerce")
        valid = x.notna() & y.notna()
        rho = float(x[valid].corr(y[valid], method="spearman"))
        correlations[metric] = rho
        for class_name in ("Driver", "Passenger"):
            mask = valid & work["mutation_type"].eq(class_name)
            ax.scatter(
                x[mask], y[mask], s=18, alpha=0.55,
                color=colors[class_name], label=class_name,
            )
        ax.set_title(f"{title} (Spearman rho={rho:.3f})")
        ax.set_xlabel("D1 mean")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    axes[0, 0].legend()
    fig.suptitle(f"{dataset_name}: distribution metrics versus mean-only D1")
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return {"figure": fig, "spearman": correlations}


__all__ = [
    "DEFAULT_OUTLIER_THRESHOLD",
    "compute_per_residue_plddt_distribution_metrics",
    "summarize_per_residue_plddt_distribution_metrics",
    "plot_plddt_distribution_profile_by_class",
    "plot_plddt_distribution_summary",
    "plot_plddt_distribution_vs_d1",
]
