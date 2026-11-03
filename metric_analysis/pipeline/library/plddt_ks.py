"""Per-residue distribution-shift metrics for pLDDT ensembles."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from .descriptor_metrics import ks_residue_results

from metric_analysis.pipeline.library.bootstrap_uncertainty import bootstrap_mean_sd


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving NaN positions."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(p))
    if finite_positions.size == 0:
        return adjusted

    finite_p = p[finite_positions]
    order = np.argsort(finite_p, kind="stable")
    ranked_p = finite_p[order]
    ranks = np.arange(1, ranked_p.size + 1, dtype=float)
    ranked_q = ranked_p * ranked_p.size / ranks
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0.0, 1.0)

    q_in_finite_order = np.empty_like(ranked_q)
    q_in_finite_order[order] = ranked_q
    adjusted[finite_positions] = q_in_finite_order
    return adjusted


def compute_per_residue_plddt_ks(
    wt_matrix: np.ndarray,
    mutant_matrix: np.ndarray,
    *,
    residue_indices: Sequence[object] | None = None,
    alternative: str = "two-sided",
    method: str = "auto",
    min_samples: int = 1,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare WT and mutant pLDDT distributions independently at each residue.

    Inputs are ``residues x structures`` matrices. Non-finite observations are
    removed separately from each residue and ensemble before calling
    :func:`scipy.stats.ks_2samp`. The returned KS statistic is an effect magnitude
    in ``[0, 1]``; p-values are two-sided by default and q-values control the
    Benjamini-Hochberg false-discovery rate across the residues of this variant.

    No residue alignment is inferred here: matrices must already have the same row
    count and ordering. This matches the pLDDT CSV contract used by the structure
    metrics pipeline. The KS statistic itself remains descriptive with tied pLDDT
    values; SciPy's standard p-value calibration assumes continuous distributions,
    so q-values from heavily rounded/tied inputs should be treated as approximate.
    """
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
    if not 0.0 < float(fdr_alpha) < 1.0:
        raise ValueError("fdr_alpha must be between 0 and 1")

    if residue_indices is None:
        indices = np.arange(wt.shape[0])
    else:
        indices = np.asarray(list(residue_indices), dtype=object)
        if indices.shape != (wt.shape[0],):
            raise ValueError(
                "residue_indices length must equal the number of matrix rows"
            )

    statistics, probabilities = ks_residue_results(
        wt, mutant, min_samples=min_samples, alternative=alternative,
        method=method, pvalues=True,
    )
    rows: list[dict[str, object]] = []
    for row_idx, residue_index in enumerate(indices):
        wt_values = wt[row_idx, np.isfinite(wt[row_idx])]
        mutant_values = mutant[row_idx, np.isfinite(mutant[row_idx])]
        n_wt = int(wt_values.size)
        n_mutant = int(mutant_values.size)

        statistic = statistics[row_idx]
        p_value = probabilities[row_idx]

        rows.append(
            {
                "residue_index": residue_index,
                "ks_statistic": statistic,
                "ks_pvalue": p_value,
                "n_wt": n_wt,
                "n_mutant": n_mutant,
            }
        )

    result_df = pd.DataFrame(rows)
    result_df["ks_qvalue"] = _benjamini_hochberg(
        result_df["ks_pvalue"].to_numpy(dtype=float)
    )
    result_df["ks_significant_fdr"] = (
        result_df["ks_qvalue"].le(float(fdr_alpha))
        & result_df["ks_qvalue"].notna()
    )
    return result_df


def summarize_per_residue_plddt_ks(
    residue_results: pd.DataFrame,
    *,
    fdr_alpha: float = 0.05,
) -> dict[str, float | int]:
    """Reduce a KS residue profile to mutation-level, classifier-ready features."""
    required = {"ks_statistic", "ks_qvalue", "n_wt", "n_mutant"}
    missing = sorted(required.difference(residue_results.columns))
    if missing:
        raise ValueError(f"KS residue table is missing required columns: {missing}")

    statistic = pd.to_numeric(residue_results["ks_statistic"], errors="coerce")
    q_value = pd.to_numeric(residue_results["ks_qvalue"], errors="coerce")
    valid = statistic.notna()
    n_tested = int(valid.sum())
    if n_tested == 0:
        return {
            "plddt_ks_mean": np.nan,
            "plddt_ks_median": np.nan,
            "plddt_ks_max": np.nan,
            "plddt_ks_fraction_fdr": np.nan,
            "plddt_ks_n_residues_tested": 0,
            "plddt_ks_min_wt_samples": 0,
            "plddt_ks_min_mutant_samples": 0,
        }

    valid_q = q_value[valid]
    return {
        "plddt_ks_mean": float(statistic[valid].mean()),
        "plddt_ks_median": float(statistic[valid].median()),
        "plddt_ks_max": float(statistic[valid].max()),
        "plddt_ks_fraction_fdr": float(valid_q.le(float(fdr_alpha)).mean()),
        "plddt_ks_n_residues_tested": n_tested,
        "plddt_ks_min_wt_samples": int(
            pd.to_numeric(residue_results.loc[valid, "n_wt"]).min()
        ),
        "plddt_ks_min_mutant_samples": int(
            pd.to_numeric(residue_results.loc[valid, "n_mutant"]).min()
        ),
    }


def plot_plddt_ks_profile_by_class(
    residue_results: pd.DataFrame,
    variant_summary: pd.DataFrame,
    *,
    residue_start: int | None = None,
    hotspot_range: Sequence[int] | None = None,
    out_path: Path | str | None = None,
    bootstrap_resamples: int = 1_000,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    """Plot class-mean KS magnitude with mutation-bootstrap SD ribbons."""
    required_residue = {
        "mutation_code", "residue_index", "ks_statistic", "ks_significant_fdr"
    }
    required_summary = {"mutation_code", "mutation_type"}
    missing = sorted(required_residue.difference(residue_results.columns))
    missing += sorted(required_summary.difference(variant_summary.columns))
    if missing:
        raise ValueError(f"KS profile inputs are missing required columns: {missing}")

    labels = variant_summary[["mutation_code", "mutation_type"]].drop_duplicates(
        "mutation_code"
    )
    work = residue_results.merge(labels, on="mutation_code", how="inner")
    work = work[work["mutation_type"].isin(["Driver", "Passenger"])].copy()
    work["residue_index"] = pd.to_numeric(work["residue_index"], errors="coerce")
    work["ks_statistic"] = pd.to_numeric(work["ks_statistic"], errors="coerce")
    work = work.dropna(subset=["residue_index", "ks_statistic"])
    if work.empty or set(work["mutation_type"]) != {"Driver", "Passenger"}:
        raise ValueError("KS profile needs finite residue results from both classes")

    colors = {"Driver": "#d1495b", "Passenger": "#2e86ab"}
    fig, (ax_stat, ax_fdr) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    class_counts: dict[str, int] = {}
    rng = np.random.default_rng(int(bootstrap_seed))
    for class_name in ("Driver", "Passenger"):
        subset = work[work["mutation_type"] == class_name]
        class_counts[class_name] = int(subset["mutation_code"].nunique())
        grouped = subset.groupby("residue_index", sort=True)
        matrix = subset.pivot_table(
            index="mutation_code", columns="residue_index", values="ks_statistic",
            aggfunc="mean",
        ).sort_index(axis=1)
        values = matrix.to_numpy(dtype=float)
        index = matrix.columns.to_numpy(dtype=float)
        mean = np.nanmean(values, axis=0)
        bootstrap_sd = bootstrap_mean_sd(
            values, n_resamples=bootstrap_resamples, rng=rng,
        )
        x = index + (int(residue_start) if residue_start is not None else 0)
        color = colors[class_name]
        ax_stat.fill_between(
            x, np.clip(mean - bootstrap_sd, 0.0, 1.0),
            np.clip(mean + bootstrap_sd, 0.0, 1.0),
            color=color, alpha=0.18, linewidth=0,
        )
        ax_stat.plot(
            x, mean, color=color, lw=1.5,
            label=(f"{class_name} mean ±bootstrap SD "
                   f"(n={class_counts[class_name]}, B={bootstrap_resamples})"),
        )
        fdr_fraction = grouped["ks_significant_fdr"].mean().to_numpy(dtype=float)
        ax_fdr.plot(x, fdr_fraction, color=color, lw=1.4, label=class_name)

    for axis in (ax_stat, ax_fdr):
        if hotspot_range and len(hotspot_range) == 2:
            axis.axvspan(
                float(hotspot_range[0]), float(hotspot_range[1]),
                color="#f0c419", alpha=0.12, zorder=0,
            )
        axis.grid(axis="y", alpha=0.18)
    ax_stat.set_ylim(0.0, 1.0)
    ax_stat.set_ylabel("KS statistic $D_r$")
    ax_stat.set_title("Per-residue WT vs mutant pLDDT distribution shift")
    ax_stat.legend(loc="upper right", fontsize=9)
    ax_fdr.set_ylim(-0.02, 1.02)
    ax_fdr.set_ylabel("FDR-significant\nvariant fraction")
    ax_fdr.set_xlabel(
        "residue number" if residue_start is not None else "residue array index"
    )
    ax_fdr.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=300, bbox_inches="tight")
    return {
        "figure": fig,
        "n_drivers": class_counts["Driver"],
        "n_passengers": class_counts["Passenger"],
        "n_residues": int(work["residue_index"].nunique()),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "out_path": str(out_path) if out_path is not None else None,
    }


def plot_plddt_ks_vs_d1(
    variant_summary: pd.DataFrame,
    *,
    dataset_name: str = "Dataset",
    out_path: Path | str | None = None,
) -> dict[str, object]:
    """Scatter mutation-level mean KS magnitude against the established D1 score."""
    required = {"mutation_code", "mutation_type", "plddt_ks_mean", "d1_mean"}
    missing = sorted(required.difference(variant_summary.columns))
    if missing:
        raise ValueError(f"KS-vs-D1 table is missing required columns: {missing}")
    work = variant_summary[list(required)].copy()
    work["plddt_ks_mean"] = pd.to_numeric(work["plddt_ks_mean"], errors="coerce")
    work["d1_mean"] = pd.to_numeric(work["d1_mean"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["plddt_ks_mean", "d1_mean"]
    )
    work = work[work["mutation_type"].isin(["Driver", "Passenger"])].copy()
    if len(work) < 2:
        raise ValueError("KS-vs-D1 plot needs at least two labelled finite variants")

    x = work["d1_mean"].to_numpy(dtype=float)
    y = work["plddt_ks_mean"].to_numpy(dtype=float)
    positive = np.all(x > 0)
    use_log_x = bool(positive and float(x.max() / x.min()) >= 100.0)
    x_for_linear = np.log10(x) if use_log_x else x
    pearson_r, pearson_p = stats.pearsonr(x_for_linear, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    colors = {"Driver": "#d1495b", "Passenger": "#2e86ab"}
    for class_name in ("Passenger", "Driver"):
        subset = work[work["mutation_type"] == class_name]
        if subset.empty:
            continue
        ax.scatter(
            subset["d1_mean"], subset["plddt_ks_mean"],
            s=38, alpha=0.72, color=colors[class_name], edgecolor="white",
            linewidth=0.35, label=f"{class_name} (n={len(subset)})",
        )
    if np.unique(x_for_linear).size > 1:
        slope, intercept = np.polyfit(x_for_linear, y, 1)
        grid_linear = np.linspace(x_for_linear.min(), x_for_linear.max(), 200)
        grid_x = 10.0 ** grid_linear if use_log_x else grid_linear
        ax.plot(grid_x, slope * grid_linear + intercept, color="#444444", lw=1.1, ls="--")
    if use_log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Mean D1 per variant" + (" (log scale)" if use_log_x else ""))
    ax.set_ylabel("Mean per-residue KS statistic")
    ax.set_ylim(bottom=0.0, top=min(1.0, max(0.1, float(y.max()) * 1.08)))
    ax.set_title(f"{dataset_name}: pLDDT KS distribution shift vs D1")
    ax.text(
        0.03, 0.97,
        f"Pearson r{' (log D1)' if use_log_x else ''} = {pearson_r:.3f} (p={pearson_p:.2g})\n"
        f"Spearman ρ = {spearman_rho:.3f} (p={spearman_p:.2g})",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#aaaaaa"),
    )
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    if out_path is not None:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=300, bbox_inches="tight")
    return {
        "figure": fig,
        "plot_df": work,
        "n_points": int(len(work)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "log_d1_axis": use_log_x,
        "out_path": str(out_path) if out_path is not None else None,
    }


__all__ = [
    "compute_per_residue_plddt_ks",
    "summarize_per_residue_plddt_ks",
    "plot_plddt_ks_profile_by_class",
    "plot_plddt_ks_vs_d1",
]
