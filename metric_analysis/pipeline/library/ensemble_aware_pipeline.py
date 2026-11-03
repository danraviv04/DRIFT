"""Harmonized "ensemble-aware" layer: coordinate/DNA caches + D1/RMSF/RMSD/DNA
N-size sweeps + unified metric comparisons.

Before this module, this logic was copy-pasted inline across all 5 protein
notebooks (~20-25 cells each). It is ported here config-driven off
protein_configs.json, so run_structure_metrics.py can run it for any protein
without opening a notebook. Two things are deliberately NOT re-implemented here:

- RMSF/RMSD N-size subsampling itself (compute_rmsf_summary_from_coords,
  compute_rmsd_summary_vs_wt, and the prepare/run_chunk_sweep/merge driver) is
  already a correct, protein-agnostic, config-driven implementation in
  run_ensemble_sweep_parallel.py — this module calls it directly (chunk=0,
  total_chunks=1 for a non-parallel single-command run) rather than duplicating
  the subsampling loop a second time.
- Driver/Passenger label assignment is labels.classify_allele_or_artifact, the
  single dispatcher shared with run_ensemble_sweep_parallel.py.

What IS new here (not previously scripted anywhere): the D1 and pLDDT-KS ensemble
sweeps, the DNA per-structure cache + DNA ensemble sweep, the RMSD "N=all
available structures" analysis, and the two unified all-metrics comparison steps.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from metric_analysis.pipeline.library import structure_metrics as sm
from metric_analysis.pipeline.config.protein_config import get_protein_config
from metric_analysis.pipeline.labels import artifact_key, is_wt_artifact, artifact_to_allele, classify_allele_or_artifact
from metric_analysis.pipeline.orchestration import run_ensemble_sweep_parallel as sweep_engine
from metric_analysis.pipeline.orchestration.run_ensemble_sweep_parallel import (
    compare_driver_passenger,
    save_csv,
    get_repeats_for_n,
    compute_rmsf_summary_from_coords,
    _mean_coords_by_residue,
    _rmsd_between_mean_coords,
)

from metric_analysis.pipeline.library.descriptor_metrics import residue_ks_statistics

ENSEMBLE_RANDOM_SEED = 42

REQUIRED_KEYS = [
    "output_prefix", "cache_dir", "sweep_dir", "protein_chains",
    "ensemble_sizes", "ensemble_repeats_by_size", "label_mode",
]

from metric_analysis.pipeline.paths import REPO_ROOT as PROJECT_ROOT


def get_ensemble_aware_config(protein):
    return get_protein_config(protein, REQUIRED_KEYS)


def _prefix(cfg):
    return cfg["output_prefix"]


def _plddt_dir(cfg):
    return cfg.get("plddt_dir") or PROJECT_ROOT / "PLDDTs" / _prefix(cfg)


def _exclude_artifacts(cfg):
    return set(cfg.get("ensemble_aware_exclude_artifacts", []))


def _sweep_dir(cfg):
    d = Path(cfg["sweep_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_dir(cfg):
    d = Path(cfg["cache_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _coord_cache_dir(cfg):
    """Where the aligned C-alpha coordinate cache lives.

    ``coord_cache_dir`` points a protein at a cache in another frame (the WT rigid-core
    fit of cache_builders/subunit_core_fit.py) while its DNA caches and structure index
    stay in ``cache_dir``.
    """
    return Path(cfg.get("coord_cache_dir") or cfg["cache_dir"])


def _plddt_scope(cfg):
    """A subunit dataset's own pLDDT rows and files (no-op for a whole protein).

    The runner already enters this scope around every stage; entering it here as well
    keeps a direct call (run_ensemble_aware_pipeline, the tests) from reading both
    chains of the parent complex and the parent's other subunit's files.
    """
    return sm.plddt_scope(row_block=cfg.get("plddt_row_range"), include=cfg.get("cohort_subunit_token"))


def _aware_dir(cfg):
    # cache_dir and sweep_dir are both "<aware_dir>/cache" and "<aware_dir>/ensemble_sweeps".
    return Path(cfg["sweep_dir"]).parent


def _plot_dir(cfg, *subdirs):
    d = _aware_dir(cfg) / "plots"
    for s in subdirs:
        d = d / s
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sweep_outputs_exist(cfg, family, name):
    d = _sweep_dir(cfg)
    prefix = _prefix(cfg)
    return all(
        (d / f"{prefix}_{name}_ensemble_sweep_{kind}.csv").exists()
        for kind in ("raw", "performance", "summary")
    )


# ── D1 ensemble-size sweep from residue x seed pLDDT matrices ─────────────────
# Cheap: does not parse structures, only reads the per-mutation pLDDT CSVs
# already produced by process_artifacts_to_plddt.

def find_plddt_csvs(plddt_dir, wt_artifact=None):
    """(WT CSV, mutant CSVs) under plddt_dir, honouring the active pLDDT scope.

    A subunit dataset (MSH2) shares its parent's folder with the other subunit's files
    (the 42 superseded MSH6 variants in PLDDTs/MutS_alpha), so the file list goes through
    structure_metrics.list_plddt_csv_files, which keeps only the scope's token. With more
    than one WT CSV (POLE has three) the reference is ``wt_artifact``, matched on
    artifact_key; it used to be whichever sorted first.
    """
    paths = sm.list_plddt_csv_files(plddt_dir)
    wt_paths = [p for p in paths if is_wt_artifact(p.stem)]
    mut_paths = [p for p in paths if p not in wt_paths]
    if not wt_paths:
        raise FileNotFoundError(f"No WT pLDDT CSV found under {plddt_dir}")
    if len(wt_paths) == 1:
        return wt_paths[0], mut_paths
    want = artifact_key(wt_artifact) if wt_artifact else None
    hits = [p for p in wt_paths if artifact_key(p.name.replace("_plddt_values.csv", "")) == want]
    if len(hits) != 1:
        raise ValueError(
            f"{len(wt_paths)} WT pLDDT CSVs under {plddt_dir} ({[p.name for p in wt_paths]}) and "
            f"wt_artifact={wt_artifact!r} picks {len(hits)} of them; set \"wt_artifact\" in the protein config"
        )
    return hits[0], mut_paths


def load_plddt_matrix(path):
    """Residue x sample pLDDT table, sliced to the active scope's rows (a subunit's chain)."""
    df = sm._apply_row_block(pd.read_csv(path, index_col=0), path)
    return df, df.to_numpy(dtype=float)


def compute_d1_for_selected_columns(wt_matrix, mut_matrix, wt_cols, mut_cols, eps=1e-8,
                                    allow_length_truncation=False):
    wt_selected = wt_matrix[:, wt_cols].astype(float)
    mut_selected = mut_matrix[:, mut_cols].astype(float)
    if wt_selected.shape[0] != mut_selected.shape[0]:
        # WT/mutant residue-count mismatch: for a missense variant these must match,
        # so a mismatch signals a malformed artifact. Reject (NaN) by default rather
        # than silently truncating to the shorter length -- mirrors structure_metrics'
        # D1_ALLOW_LENGTH_TRUNCATION policy. Pass allow_length_truncation=True to opt in.
        if not allow_length_truncation:
            warnings.warn(
                f"pLDDT residue-count mismatch (WT={wt_selected.shape[0]}, "
                f"mutant={mut_selected.shape[0]}); D1 set to NaN. Pass "
                f"allow_length_truncation=True to restore legacy min-length truncation.",
                RuntimeWarning,
            )
            return {
                "d1_mean": float("nan"), "d1_median": float("nan"),
                "d1_std": float("nan"), "d1_min": float("nan"), "d1_max": float("nan"),
                "n_residues_used": 0, "d1_length_mismatch": True,
            }
        n = min(wt_selected.shape[0], mut_selected.shape[0])
        wt_selected = wt_selected[:n, :]
        mut_selected = mut_selected[:n, :]
    wt_mean = np.nanmean(wt_selected, axis=1)
    mut_mean = np.nanmean(mut_selected, axis=1)
    d1_per_residue = np.abs(mut_mean / np.maximum(wt_mean, eps) - 1.0)
    return {
        "d1_mean": float(np.nanmean(d1_per_residue)),
        "d1_median": float(np.nanmedian(d1_per_residue)),
        "d1_std": float(np.nanstd(d1_per_residue)),
        "d1_min": float(np.nanmin(d1_per_residue)),
        "d1_max": float(np.nanmax(d1_per_residue)),
        "n_residues_used": int(np.isfinite(d1_per_residue).sum()),
    }


def compute_plddt_ks_for_selected_columns(
    wt_matrix, mut_matrix, wt_cols, mut_cols, *, min_samples=1
):
    """Summarize per-residue WT-vs-mutant KS magnitudes for one subsample.

    The finite, equal-width case is vectorized because an ensemble sweep performs
    this calculation millions of residue-times. Ties are handled by evaluating
    the empirical-CDF difference only after the complete equal-value group, which
    matches ``scipy.stats.ks_2samp``. Inputs containing non-finite values fall back
    to the canonical per-residue implementation.
    """
    wt_selected = np.asarray(wt_matrix, dtype=float)[:, wt_cols]
    mut_selected = np.asarray(mut_matrix, dtype=float)[:, mut_cols]
    if wt_selected.shape[0] != mut_selected.shape[0]:
        return {
            "plddt_ks_mean": float("nan"),
            "plddt_ks_median": float("nan"),
            "plddt_ks_max": float("nan"),
            "n_residues_used": 0,
            "ks_length_mismatch": True,
        }
    if min(wt_selected.shape[1], mut_selected.shape[1]) < int(min_samples):
        return {
            "plddt_ks_mean": float("nan"),
            "plddt_ks_median": float("nan"),
            "plddt_ks_max": float("nan"),
            "n_residues_used": 0,
            "ks_insufficient_samples": True,
        }

    statistic = residue_ks_statistics(wt_selected, mut_selected, min_samples=min_samples)

    finite = statistic[np.isfinite(statistic)]
    if finite.size == 0:
        return {
            "plddt_ks_mean": float("nan"),
            "plddt_ks_median": float("nan"),
            "plddt_ks_max": float("nan"),
            "n_residues_used": 0,
        }
    return {
        "plddt_ks_mean": float(np.mean(finite)),
        "plddt_ks_median": float(np.median(finite)),
        "plddt_ks_max": float(np.max(finite)),
        "n_residues_used": int(finite.size),
    }


def compute_plddt_distribution_for_selected_columns(
    wt_matrix, mutant_matrix, wt_cols, mutant_cols, *, min_samples=1,
    outlier_threshold=sm.DEFAULT_PLDDT_OUTLIER_THRESHOLD,
):
    """Full-distribution pLDDT summaries for one ensemble subsample.

    Unlike KS inference, Wasserstein and the other descriptive effect sizes are
    mathematically defined at N=1.  Retaining N=1 makes their high-variance small-
    ensemble behavior visible rather than silently replacing it with a missing row.
    """
    wt_selected = np.asarray(wt_matrix, dtype=float)[:, np.asarray(wt_cols, dtype=int)]
    mut_selected = np.asarray(mutant_matrix, dtype=float)[:, np.asarray(mutant_cols, dtype=int)]
    if wt_selected.shape[0] != mut_selected.shape[0]:
        return {
            **sm.summarize_per_residue_plddt_distribution_metrics(
                pd.DataFrame(columns=[
                    "wasserstein", "wasserstein_unit", "wasserstein_relative",
                    "shape_excess", "shape_excess_relative", "shape_fraction",
                    "energy_distance", "cvm_effect", "location_shift_relative",
                    "wt_median", "outlier_score", "outlier_fraction",
                    "outlier_mean_deviation", "outlier_count",
                    "outlier_threshold",
                    "n_wt", "n_mutant",
                ]),
                outlier_threshold=outlier_threshold,
            ),
            "distribution_length_mismatch": True,
        }
    residue = sm.compute_per_residue_plddt_distribution_metrics(
        wt_selected, mut_selected, min_samples=min_samples,
        outlier_threshold=outlier_threshold,
    )
    return {
        **sm.summarize_per_residue_plddt_distribution_metrics(residue),
        "distribution_length_mismatch": False,
    }


def _sample_indices(n_available, n, rng):
    n_available, n = int(n_available), int(n)
    if n > n_available:
        return None
    return rng.choice(n_available, size=n, replace=False)


def run_d1_ensemble_sweep(cfg, plddt_dir=None, force=False):
    prefix = _prefix(cfg)
    sweep_dir = _sweep_dir(cfg)
    if not force and _sweep_outputs_exist(cfg, "D1", "D1_true_plddt"):
        print(f"[{prefix}] D1 sweep outputs already exist — skipping. Use force=True to recompute.")
        return {
            "raw": pd.read_csv(sweep_dir / f"{prefix}_D1_true_plddt_ensemble_sweep_raw.csv"),
            "performance": pd.read_csv(sweep_dir / f"{prefix}_D1_true_plddt_ensemble_sweep_performance.csv"),
            "summary": pd.read_csv(sweep_dir / f"{prefix}_D1_true_plddt_ensemble_sweep_summary.csv"),
        }

    plddt_dir = Path(plddt_dir or _plddt_dir(cfg))
    with _plddt_scope(cfg):
        return _run_d1_ensemble_sweep(cfg, plddt_dir)


def _run_d1_ensemble_sweep(cfg, plddt_dir):
    prefix = _prefix(cfg)
    sweep_dir = _sweep_dir(cfg)
    wt_path, mut_paths = find_plddt_csvs(plddt_dir, cfg.get("wt_artifact"))
    _, wt_matrix = load_plddt_matrix(wt_path)
    print(f"[{prefix}] D1 sweep — WT: {wt_path.name}, {len(mut_paths)} mutant pLDDT files, "
          f"{wt_matrix.shape[0]} residues")

    rng = np.random.default_rng(ENSEMBLE_RANDOM_SEED)
    raw_rows = []
    for n in cfg["ensemble_sizes"]:
        reps = get_repeats_for_n(n, cfg["ensemble_repeats_by_size"])
        for mut_path in tqdm(mut_paths, desc=f"[{prefix}] D1 N={n}"):
            artifact = mut_path.stem.replace("_plddt_values", "")
            allele = artifact_to_allele(artifact)
            label_class = classify_allele_or_artifact(artifact, cfg)
            _, mut_matrix = load_plddt_matrix(mut_path)
            if n > wt_matrix.shape[1] or n > mut_matrix.shape[1]:
                continue
            for r in range(reps):
                wt_cols = _sample_indices(wt_matrix.shape[1], n, rng)
                mut_cols = _sample_indices(mut_matrix.shape[1], n, rng)
                if wt_cols is None or mut_cols is None:
                    continue
                vals = compute_d1_for_selected_columns(wt_matrix, mut_matrix, wt_cols, mut_cols)
                raw_rows.append({
                    "metric_family": "D1", "artifact": artifact, "Allele": allele,
                    "label_class": label_class, "ensemble_size": int(n), "repeat": int(r),
                    "n_wt_available": int(wt_matrix.shape[1]), "n_mut_available": int(mut_matrix.shape[1]),
                    **vals,
                })

    raw = pd.DataFrame(raw_rows)
    save_csv(raw, sweep_dir / f"{prefix}_D1_true_plddt_ensemble_sweep_raw.csv")

    perf_rows = [
        {"metric_family": "D1", "ensemble_size": n, "repeat": r, **compare_driver_passenger(g, "d1_mean")}
        for (n, r), g in raw.groupby(["ensemble_size", "repeat"])
    ]
    perf = pd.DataFrame(perf_rows)
    # Descriptive separation magnitude (direction-agnostic, clearly NOT a predictive
    # score): distance of the signed driver-high AUC from chance.
    perf["auc_separation"] = (perf["roc_auc_driver_high"] - 0.5).abs()
    save_csv(perf, sweep_dir / f"{prefix}_D1_true_plddt_ensemble_sweep_performance.csv")

    # PRIMARY reported AUC is the SIGNED driver-high AUC (<0.5 => higher in passengers),
    # so an inverse association is visible rather than flipped to look predictive. Cliff's
    # delta carries the same direction. best_direction / separation are kept only as
    # clearly-labeled descriptive columns.
    summary = perf.groupby(["metric_family", "ensemble_size"], as_index=False).agg(
        n_repeats=("repeat", "nunique"),
        auc_mean=("roc_auc_driver_high", "mean"),
        auc_std=("roc_auc_driver_high", "std"),
        auc_min=("roc_auc_driver_high", "min"),
        auc_max=("roc_auc_driver_high", "max"),
        auc_best_direction_mean=("roc_auc_best_direction", "mean"),
        auc_separation_mean=("auc_separation", "mean"),
        cliffs_mean=("cliffs_delta_driver_minus_passenger", "mean"),
        wasserstein_mean=("wasserstein", "mean"),
        n_driver=("n_driver", "median"),
        n_passenger=("n_passenger", "median"),
    )
    save_csv(summary, sweep_dir / f"{prefix}_D1_true_plddt_ensemble_sweep_summary.csv")

    _plot_d1_auc_vs_n(cfg, summary)
    _write_d1_sem_diagnostics(cfg, raw, perf)
    return {"raw": raw, "performance": perf, "summary": summary}


def _plot_d1_auc_vs_n(cfg, summary):
    import matplotlib.pyplot as plt
    prefix = _prefix(cfg)
    plt.figure(figsize=(8, 5))
    plt.errorbar(summary["ensemble_size"], summary["auc_mean"], yerr=summary["auc_std"], marker="o", capsize=3)
    # No chance line: pinning the axis to 0.5 spent most of the panel on empty
    # range and flattened the very increments the sweep exists to show. The axis
    # is padded around the data + error bars instead, and the label still states
    # where chance lies for a reader who needs it.
    lo = float((summary["auc_mean"] - summary["auc_std"]).min())
    hi = float((summary["auc_mean"] + summary["auc_std"]).max())
    pad = max((hi - lo) * 0.15, 0.005)
    plt.ylim(lo - pad, hi + pad)
    plt.xlabel("Ensemble size N")
    plt.ylabel("ROC-AUC (driver-high, signed; 0.5 = chance)")
    plt.title(f"{prefix} D1 ensemble-size sensitivity")
    plt.grid(alpha=0.3)
    plot_path = _plot_dir(cfg) / f"{prefix}_D1_true_plddt_N_sweep_AUC.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")


def _write_d1_sem_diagnostics(cfg, raw, perf):
    prefix = _prefix(cfg)
    sweep_dir = _sweep_dir(cfg)

    value_summary = perf.groupby(["metric_family", "ensemble_size"], as_index=False).agg(
        n_repeats=("repeat", "nunique"),
        driver_value_mean=("driver_mean", "mean"), driver_value_std=("driver_mean", "std"),
        passenger_value_mean=("passenger_mean", "mean"), passenger_value_std=("passenger_mean", "std"),
        n_driver=("n_driver", "median"), n_passenger=("n_passenger", "median"),
    )
    value_summary["driver_value_sem"] = value_summary["driver_value_std"] / np.sqrt(value_summary["n_repeats"])
    value_summary["passenger_value_sem"] = value_summary["passenger_value_std"] / np.sqrt(value_summary["n_repeats"])
    value_summary["metric_col"] = "d1_mean"
    value_summary["driver_minus_passenger_mean"] = value_summary["driver_value_mean"] - value_summary["passenger_value_mean"]
    save_csv(value_summary, sweep_dir / f"{prefix}_D1_true_plddt_ensemble_sweep_value_summary.csv")

    d1_repeat_means = (
        raw[raw["label_class"].isin(["Driver", "Passenger"])]
        .groupby(["ensemble_size", "repeat"], as_index=False)
        .agg(repeated_mean=("d1_mean", "mean"), n_artifacts=("artifact", "nunique"))
    )
    d1_estimator_sem = d1_repeat_means.groupby(["ensemble_size"], as_index=False).agg(
        n_repeats=("repeat", "nunique"), repeated_mean=("repeated_mean", "mean"),
        repeat_mean_sd=("repeated_mean", "std"), repeated_n_artifacts=("n_artifacts", "median"),
    )
    d1_estimator_sem["sem_of_estimator"] = d1_estimator_sem["repeat_mean_sd"] / np.sqrt(d1_estimator_sem["n_repeats"])
    d1_estimator_sem.loc[d1_estimator_sem["n_repeats"] < 2, "sem_of_estimator"] = np.nan
    d1_estimator_sem.insert(0, "sem_group", "Driver+Passenger")
    save_csv(d1_estimator_sem, sweep_dir / f"{prefix}_D1_true_plddt_ensemble_sweep_estimator_sem.csv")

    d1_variant_sem = raw.groupby(["artifact", "Allele", "label_class", "ensemble_size"], as_index=False).agg(
        n_repeats=("repeat", "nunique"), d1_mean_value=("d1_mean", "mean"), d1_mean_std=("d1_mean", "std"),
    )
    d1_variant_sem["d1_mean_sem"] = d1_variant_sem["d1_mean_std"] / np.sqrt(d1_variant_sem["n_repeats"])
    d1_variant_sem.loc[d1_variant_sem["n_repeats"] < 2, "d1_mean_sem"] = np.nan
    save_csv(d1_variant_sem, sweep_dir / f"{prefix}_D1_true_plddt_ensemble_sweep_variant_level_sem.csv")

    d1_labeled_sem = d1_variant_sem[
        d1_variant_sem["label_class"].isin(["Driver", "Passenger"]) & d1_variant_sem["d1_mean_sem"].notna()
    ].copy()
    d1_variant_sem_summary = d1_labeled_sem.groupby(["ensemble_size"], as_index=False).agg(
        n_artifacts=("artifact", "nunique"),
        median_variant_sem=("d1_mean_sem", "median"), mean_variant_sem=("d1_mean_sem", "mean"),
        q25_variant_sem=("d1_mean_sem", lambda x: float(np.nanquantile(x, 0.25))),
        q75_variant_sem=("d1_mean_sem", lambda x: float(np.nanquantile(x, 0.75))),
        median_n_repeats=("n_repeats", "median"),
    )
    d1_variant_sem_summary.insert(0, "sem_group", "Driver+Passenger")
    d1_variant_sem_summary = d1_variant_sem_summary.merge(
        d1_estimator_sem[["ensemble_size", "repeated_mean", "repeat_mean_sd", "sem_of_estimator"]],
        on="ensemble_size", how="left",
    )
    save_csv(d1_variant_sem_summary, sweep_dir / f"{prefix}_D1_true_plddt_ensemble_sweep_variant_sem_summary.csv")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    x = d1_variant_sem_summary["ensemble_size"].to_numpy(float)
    plt.plot(x, d1_variant_sem_summary["median_variant_sem"].to_numpy(float), marker="o", label="Driver+Passenger median variant SEM")
    plt.fill_between(x, d1_variant_sem_summary["q25_variant_sem"].to_numpy(float), d1_variant_sem_summary["q75_variant_sem"].to_numpy(float), alpha=0.2, linewidth=0, label="IQR across variants")
    plt.plot(x, d1_variant_sem_summary["sem_of_estimator"].to_numpy(float), marker="x", linestyle="--", label="SEM of repeated-mean estimator")
    plt.xlabel("Ensemble size N")
    plt.ylabel("SEM of d1_mean")
    plt.title(f"{prefix} D1 ensemble-size sensitivity: SEM diagnostics")
    plt.yscale("log")
    plt.grid(alpha=0.3, which="both")
    plt.legend(fontsize=9)
    plot_path = _plot_dir(cfg) / f"{prefix}_D1_true_plddt_N_sweep_SEM_diagnostics.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved SEM diagnostics plot: {plot_path}")


# ── pLDDT KS ensemble-size sweep ─────────────────────────────────────────────

def run_plddt_ks_ensemble_sweep(
    cfg, *, plddt_files=None, wt_path=None, force=False, min_samples=1
):
    """Measure Driver/Passenger ROC-AUC of mean per-residue KS versus N (see below)."""
    with _plddt_scope(cfg):
        return _run_plddt_ks_ensemble_sweep(
            cfg, plddt_files=plddt_files, wt_path=wt_path, force=force, min_samples=min_samples
        )


def _run_plddt_ks_ensemble_sweep(
    cfg, *, plddt_files=None, wt_path=None, force=False, min_samples=1
):
    """Measure Driver/Passenger ROC-AUC of mean per-residue KS versus N.

    Sampling and AUC aggregation intentionally match ``run_d1_ensemble_sweep``:
    WT and mutant columns are sampled without replacement using the same fixed
    seed, and the primary performance value is signed driver-high ROC-AUC. The
    KS score requires at least two observations per distribution, so configured
    N=1 rows are retained as unavailable rather than assigned a degenerate score.
    """
    prefix = _prefix(cfg)
    sweep_dir = _sweep_dir(cfg)
    name = "pLDDT_KS"
    paths = {
        kind: sweep_dir / f"{prefix}_{name}_ensemble_sweep_{kind}.csv"
        for kind in ("raw", "performance", "summary")
    }
    if not force and all(path.exists() for path in paths.values()):
        result = {kind: pd.read_csv(path) for kind, path in paths.items()}
        _plot_plddt_ks_auc_vs_n(cfg, result["summary"])
        print(
            f"[{prefix}] pLDDT KS sweep outputs already exist — using cache. "
            "Use force=True to recompute."
        )
        return result

    if plddt_files is None:
        with _plddt_scope(cfg):
            plddt_files = sm.list_plddt_csv_files(_plddt_dir(cfg))
    else:
        plddt_files = sorted(Path(path) for path in plddt_files)
    if wt_path is None:
        with _plddt_scope(cfg):
            discovered_wt, _ = find_plddt_csvs(_plddt_dir(cfg), cfg.get("wt_artifact"))
        wt_path = discovered_wt
    wt_path = Path(wt_path)
    mut_paths = [
        path for path in plddt_files
        if path.resolve() != wt_path.resolve() and not is_wt_artifact(path.stem)
    ]
    if not mut_paths:
        raise ValueError(f"[{prefix}] no mutant pLDDT CSVs available for KS sweep")

    _, wt_matrix = load_plddt_matrix(wt_path)
    print(
        f"[{prefix}] pLDDT KS sweep — WT: {wt_path.name}, "
        f"{len(mut_paths)} mutant pLDDT files"
    )
    rng = np.random.default_rng(ENSEMBLE_RANDOM_SEED)
    raw_rows = []
    for n in cfg["ensemble_sizes"]:
        reps = get_repeats_for_n(n, cfg["ensemble_repeats_by_size"])
        for mut_path in tqdm(mut_paths, desc=f"[{prefix}] pLDDT KS N={n}"):
            artifact = mut_path.stem.replace("_plddt_values", "")
            allele = artifact_to_allele(artifact)
            label_class = classify_allele_or_artifact(artifact, cfg)
            _, mut_matrix = load_plddt_matrix(mut_path)
            if n > wt_matrix.shape[1] or n > mut_matrix.shape[1]:
                continue
            for r in range(reps):
                wt_cols = _sample_indices(wt_matrix.shape[1], n, rng)
                mut_cols = _sample_indices(mut_matrix.shape[1], n, rng)
                vals = compute_plddt_ks_for_selected_columns(
                    wt_matrix, mut_matrix, wt_cols, mut_cols,
                    min_samples=min_samples,
                )
                raw_rows.append({
                    "metric_family": "pLDDT_KS", "artifact": artifact,
                    "Allele": allele, "label_class": label_class,
                    "ensemble_size": int(n), "repeat": int(r),
                    "n_wt_available": int(wt_matrix.shape[1]),
                    "n_mut_available": int(mut_matrix.shape[1]),
                    **vals,
                })

    raw = pd.DataFrame(raw_rows)
    save_csv(raw, paths["raw"])
    perf_rows = [
        {
            "metric_family": "pLDDT_KS", "ensemble_size": n, "repeat": r,
            **compare_driver_passenger(group, "plddt_ks_mean"),
        }
        for (n, r), group in raw.groupby(["ensemble_size", "repeat"])
    ]
    perf = pd.DataFrame(perf_rows)
    perf["auc_separation"] = (perf["roc_auc_driver_high"] - 0.5).abs()
    save_csv(perf, paths["performance"])
    summary = perf.groupby(["metric_family", "ensemble_size"], as_index=False).agg(
        n_repeats=("repeat", "nunique"),
        auc_mean=("roc_auc_driver_high", "mean"),
        auc_std=("roc_auc_driver_high", "std"),
        auc_min=("roc_auc_driver_high", "min"),
        auc_max=("roc_auc_driver_high", "max"),
        auc_best_direction_mean=("roc_auc_best_direction", "mean"),
        auc_separation_mean=("auc_separation", "mean"),
        cliffs_mean=("cliffs_delta_driver_minus_passenger", "mean"),
        wasserstein_mean=("wasserstein", "mean"),
        n_driver=("n_driver", "median"),
        n_passenger=("n_passenger", "median"),
    )
    save_csv(summary, paths["summary"])
    _plot_plddt_ks_auc_vs_n(cfg, summary)
    return {"raw": raw, "performance": perf, "summary": summary}


def _plot_plddt_ks_auc_vs_n(cfg, summary):
    import matplotlib.pyplot as plt

    prefix = _prefix(cfg)
    plot_df = summary.dropna(subset=["auc_mean"]).copy()
    if plot_df.empty:
        warnings.warn(
            f"[{prefix}] no finite pLDDT KS ROC-AUC values to plot",
            RuntimeWarning,
        )
        return None
    error = pd.to_numeric(plot_df["auc_std"], errors="coerce").fillna(0.0)
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        plot_df["ensemble_size"], plot_df["auc_mean"], yerr=error,
        marker="o", capsize=3,
    )
    lo = float((plot_df["auc_mean"] - error).min())
    hi = float((plot_df["auc_mean"] + error).max())
    pad = max((hi - lo) * 0.15, 0.005)
    plt.ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    plt.xlabel("Ensemble size N")
    plt.ylabel("ROC-AUC (driver-high, signed; 0.5 = chance)")
    plt.title(f"{prefix} pLDDT KS ensemble-size sensitivity")
    plt.grid(alpha=0.3)
    plot_path = _plot_dir(cfg) / f"{prefix}_pLDDT_KS_N_sweep_AUC.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")
    return plot_path


# ── Full-distribution pLDDT ensemble-size sweep ──────────────────────────────

def run_plddt_distribution_ensemble_sweep(
    cfg, *, plddt_files=None, wt_path=None, force=False
):
    """Measure stability and driver-high ROC-AUC of distribution metrics versus N."""
    with _plddt_scope(cfg):
        return _run_plddt_distribution_ensemble_sweep(
            cfg, plddt_files=plddt_files, wt_path=wt_path, force=force
        )


def _run_plddt_distribution_ensemble_sweep(
    cfg, *, plddt_files=None, wt_path=None, force=False
):
    prefix = _prefix(cfg)
    sweep_dir = _sweep_dir(cfg)
    name = "pLDDT_Distribution"
    paths = {
        kind: sweep_dir / f"{prefix}_{name}_ensemble_sweep_{kind}.csv"
        for kind in ("raw", "performance", "summary")
    }
    outlier_threshold = float(cfg.get(
        "plddt_outlier_threshold", sm.DEFAULT_PLDDT_OUTLIER_THRESHOLD,
    ))
    if not force and all(path.exists() for path in paths.values()):
        result = {kind: pd.read_csv(path) for kind, path in paths.items()}
        cached_metrics = set(result["summary"].get("metric", pd.Series(dtype=str)))
        cached_threshold = pd.to_numeric(
            result["raw"].get(
                "plddt_outlier_threshold", pd.Series(dtype=float)
            ),
            errors="coerce",
        ).dropna()
        cache_current = (
            "plddt_outlier_score_mean" in cached_metrics
            and not cached_threshold.empty
            and np.allclose(cached_threshold.to_numpy(float), outlier_threshold)
        )
        if cache_current:
            _plot_plddt_distribution_auc_vs_n(cfg, result["summary"])
            print(
                f"[{prefix}] pLDDT distribution sweep outputs already exist — "
                "using cache. Use force=True to recompute."
            )
            return result
        print(
            f"[{prefix}] cached pLDDT distribution sweep predates the thresholded "
            "outlier metric or uses a different threshold — recomputing"
        )

    if plddt_files is None:
        with _plddt_scope(cfg):
            plddt_files = sm.list_plddt_csv_files(_plddt_dir(cfg))
    else:
        plddt_files = sorted(Path(path) for path in plddt_files)
    if wt_path is None:
        with _plddt_scope(cfg):
            discovered_wt, _ = find_plddt_csvs(_plddt_dir(cfg), cfg.get("wt_artifact"))
        wt_path = discovered_wt
    wt_path = Path(wt_path)
    mut_paths = [
        path for path in plddt_files
        if path.resolve() != wt_path.resolve() and not is_wt_artifact(path.stem)
    ]
    if not mut_paths:
        raise ValueError(
            f"[{prefix}] no mutant pLDDT CSVs available for distribution sweep"
        )

    _, wt_matrix = load_plddt_matrix(wt_path)
    print(
        f"[{prefix}] pLDDT distribution sweep — WT: {wt_path.name}, "
        f"{len(mut_paths)} mutant pLDDT files"
    )
    rng = np.random.default_rng(ENSEMBLE_RANDOM_SEED)
    raw_rows = []
    for n in cfg["ensemble_sizes"]:
        reps = get_repeats_for_n(n, cfg["ensemble_repeats_by_size"])
        for mut_path in tqdm(mut_paths, desc=f"[{prefix}] pLDDT distribution N={n}"):
            artifact = mut_path.stem.replace("_plddt_values", "")
            allele = artifact_to_allele(artifact)
            label_class = classify_allele_or_artifact(artifact, cfg)
            _, mut_matrix = load_plddt_matrix(mut_path)
            if n > wt_matrix.shape[1] or n > mut_matrix.shape[1]:
                continue
            for repeat in range(reps):
                wt_cols = _sample_indices(wt_matrix.shape[1], n, rng)
                mut_cols = _sample_indices(mut_matrix.shape[1], n, rng)
                values = compute_plddt_distribution_for_selected_columns(
                    wt_matrix, mut_matrix, wt_cols, mut_cols, min_samples=1,
                    outlier_threshold=outlier_threshold,
                )
                raw_rows.append({
                    "metric_family": "pLDDT_Distribution",
                    "artifact": artifact,
                    "Allele": allele,
                    "label_class": label_class,
                    "ensemble_size": int(n),
                    "repeat": int(repeat),
                    "n_wt_available": int(wt_matrix.shape[1]),
                    "n_mut_available": int(mut_matrix.shape[1]),
                    **values,
                })

    raw = pd.DataFrame(raw_rows)
    save_csv(raw, paths["raw"])
    metric_columns = (
        "plddt_w1_unit_mean",
        "plddt_outlier_score_mean",
        "plddt_w1_shape_excess_relative_mean",
        "plddt_energy_mean",
        "plddt_cvm_mean",
    )
    performance_rows = []
    for metric in metric_columns:
        for (n, repeat), group in raw.groupby(["ensemble_size", "repeat"]):
            performance_rows.append({
                "metric_family": "pLDDT_Distribution",
                "metric": metric,
                "ensemble_size": n,
                "repeat": repeat,
                **compare_driver_passenger(group, metric),
            })
    performance = pd.DataFrame(performance_rows)
    performance["auc_separation"] = (
        performance["roc_auc_driver_high"] - 0.5
    ).abs()
    save_csv(performance, paths["performance"])
    summary = performance.groupby(
        ["metric_family", "metric", "ensemble_size"], as_index=False
    ).agg(
        n_repeats=("repeat", "nunique"),
        auc_mean=("roc_auc_driver_high", "mean"),
        auc_std=("roc_auc_driver_high", "std"),
        auc_min=("roc_auc_driver_high", "min"),
        auc_max=("roc_auc_driver_high", "max"),
        auc_best_direction_mean=("roc_auc_best_direction", "mean"),
        auc_separation_mean=("auc_separation", "mean"),
        cliffs_mean=("cliffs_delta_driver_minus_passenger", "mean"),
        wasserstein_between_classes_mean=("wasserstein", "mean"),
        n_driver=("n_driver", "median"),
        n_passenger=("n_passenger", "median"),
    )
    save_csv(summary, paths["summary"])
    _plot_plddt_distribution_auc_vs_n(cfg, summary)
    return {"raw": raw, "performance": performance, "summary": summary}


def _plot_plddt_distribution_auc_vs_n(cfg, summary):
    import matplotlib.pyplot as plt

    prefix = _prefix(cfg)
    outlier_threshold = float(cfg.get(
        "plddt_outlier_threshold", sm.DEFAULT_PLDDT_OUTLIER_THRESHOLD,
    ))
    plot_df = summary.dropna(subset=["auc_mean"]).copy()
    if plot_df.empty:
        warnings.warn(
            f"[{prefix}] no finite pLDDT distribution ROC-AUC values to plot",
            RuntimeWarning,
        )
        return None
    labels = {
        "plddt_w1_unit_mean": "Wasserstein-1",
        "plddt_outlier_score_mean": f"Outlier score (tau={outlier_threshold:g})",
        "plddt_w1_shape_excess_relative_mean": "W1 shape excess",
        "plddt_energy_mean": "Energy",
        "plddt_cvm_mean": "CvM",
    }
    plt.figure(figsize=(9, 5.5))
    for metric, group in plot_df.groupby("metric", sort=False):
        group = group.sort_values("ensemble_size")
        error = pd.to_numeric(group["auc_std"], errors="coerce").fillna(0.0)
        plt.errorbar(
            group["ensemble_size"], group["auc_mean"], yerr=error,
            marker="o", capsize=2, label=labels.get(metric, metric),
        )
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Ensemble size N")
    plt.ylabel("ROC-AUC (driver-high, signed; 0.5 = chance)")
    plt.title(f"{prefix} distribution-metric ensemble-size sensitivity")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plot_path = _plot_dir(cfg) / f"{prefix}_pLDDT_Distribution_N_sweep_AUC.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")
    return plot_path


# ── RMSF / RMSD N-size sweep: delegate to run_ensemble_sweep_parallel.py ──────
# That script already implements this correctly and config-driven; running it
# with total_chunks=1 gives a non-parallel single-command equivalent.

def run_rmsf_rmsd_ensemble_sweep(protein, force=False, total_chunks=4):
    """RMSF/RMSD N-size sweeps, delegated to run_ensemble_sweep_parallel.py.

    total_chunks MUST match the on-disk coord_chunks split (4, as produced by
    launch_ensemble_sweep.sh). It is not merely a parallelism knob: chunk N holds
    only its own slice of artifacts, so running with total_chunks=1 against an
    existing 4-way split would sweep a quarter of the data and report success.

    The coordinate chunks hold no labels, so they are reused even under force
    (rebuilding them means re-loading a multi-GB cache); only the label-bearing
    raw sweep outputs and the merge are recomputed.
    """
    cfg = get_ensemble_aware_config(protein)
    prefix = _prefix(cfg)
    if not force and (
        _sweep_outputs_exist(cfg, "RMSF", "RMSF_aligned_coord")
        and _sweep_outputs_exist(cfg, "RMSD", "RMSD_aligned_coord")
    ):
        print(f"[{prefix}] RMSF/RMSD sweep outputs already exist — skipping. Use force=True to recompute.")
        return

    chunk_dir = _sweep_dir(cfg) / "coord_chunks"
    existing = sorted(chunk_dir.glob(f"{prefix}_coord_chunk_*.parquet"))
    stale = sweep_engine.stale_coord_chunks_reason(cfg, chunk_dir)
    if existing and stale:
        # The chunks are copies of the coordinate cache and are reused even under force,
        # so chunks from another cache (another frame, an older build) would be swept
        # silently. They are derived data: rebuild them.
        print(f"[{prefix}] discarding {len(existing)} coord chunks: {stale}")
        for p in existing:
            p.unlink()
        existing = []
    if existing:
        found = len(existing)
        if found != total_chunks:
            raise RuntimeError(
                f"[{prefix}] found {found} coord chunks on disk but total_chunks={total_chunks}. "
                f"Sweeping with a mismatched split would silently cover only part of the data. "
                f"Pass total_chunks={found}, or delete {chunk_dir} to rebuild."
            )
        print(f"[{prefix}] reusing {found} existing coordinate chunks")
    else:
        sweep_engine.prepare(protein, total_chunks=total_chunks, force=False)

    _run_chunks_in_parallel(protein, total_chunks, force, prefix)
    sweep_engine.merge(protein, total_chunks=total_chunks, force=force)


def _run_chunks_in_parallel(protein, total_chunks, force, prefix):
    """Run the per-chunk sweeps as concurrent subprocesses, like launch_ensemble_sweep.sh does.

    The chunking exists *for* process-level parallelism: the sweep is CPU-bound
    pandas/numpy, so threads gain nothing under the GIL and the launcher script
    deliberately forks one process per chunk. Looping chunks sequentially in-process
    (the original implementation here) discarded that and ran ~4x slower -- for TP53,
    295 artifacts per chunk, the difference is roughly 20 h versus 5 h.

    Starts are staggered like the launcher: many processes opening archives on one
    external drive at the same instant can fail transiently.
    """
    import os
    import subprocess
    import sys
    import time

    # How many chunk workers may run at once. This is NOT the number of chunks (that
    # must equal the on-disk coord_chunks split) -- it only caps concurrency, so a
    # 4-way split can still be worked through 2-at-a-time on a memory-tight machine.
    # Each worker holds one chunk's coordinates in RAM: for TP53 that is ~46.8M rows
    # per chunk, roughly 2-3 GB resident, so 4 workers peak near 8-10 GB.
    # Override with ENSEMBLE_SWEEP_MAX_PARALLEL=2 (etc.) without touching this file.
    max_parallel = int(os.environ.get("ENSEMBLE_SWEEP_MAX_PARALLEL", "4"))
    max_parallel = max(1, min(max_parallel, total_chunks))

    sweep_module = "metric_analysis.pipeline.orchestration.run_ensemble_sweep_parallel"
    log_dir = _sweep_dir(get_ensemble_aware_config(protein)).parent / "sweep_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{prefix}] {total_chunks} chunks, up to {max_parallel} at a time", flush=True)
    pending = list(range(total_chunks))
    running = []   # (chunk, proc, log_path)
    failed = []

    def _reap(block):
        """Collect finished workers; if block, wait for at least one to finish."""
        while running:
            for i, (chunk, proc, log_path) in enumerate(running):
                rc = proc.poll()
                if rc is not None:
                    running.pop(i)
                    print(f"[{prefix}] chunk {chunk}: {'DONE' if rc == 0 else f'FAILED (rc={rc})'}", flush=True)
                    if rc != 0:
                        failed.append((chunk, log_path))
                    return True
            if not block:
                return False
            time.sleep(5)
        return False

    while pending or running:
        while pending and len(running) < max_parallel:
            chunk = pending.pop(0)
            cmd = [sys.executable, "-m", sweep_module, "--protein", protein,
                   "--chunk", str(chunk), "--total-chunks", str(total_chunks)]
            if force:
                cmd.append("--force")
            log_path = log_dir / f"{prefix}_chunk_{chunk}.log"
            print(f"[{prefix}] launching chunk {chunk}/{total_chunks - 1} -> {log_path.name}", flush=True)
            fh = open(log_path, "w")
            running.append((chunk, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                                     cwd=str(PROJECT_ROOT)), log_path))
            # Stagger starts: many processes opening archives on one external drive at
            # the same instant can fail transiently (same reason launch_*.sh sleeps 15).
            if pending:
                time.sleep(15)
        if running:
            _reap(block=True)

    if failed:
        detail = ", ".join(f"chunk {c} (see {p})" for c, p in failed)
        raise RuntimeError(f"[{prefix}] {len(failed)} chunk sweep(s) failed: {detail}")


def load_rmsf_coord_cache(cfg):
    prefix = _prefix(cfg)
    cache_dir = _coord_cache_dir(cfg)
    parquet_path = cache_dir / f"{prefix}_rmsf_aligned_ca_coordinate_cache.parquet"
    csv_path = cache_dir / f"{prefix}_rmsf_aligned_ca_coordinate_cache.csv.gz"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        return None
    exclude = _exclude_artifacts(cfg)
    if exclude and "artifact" in df.columns:
        df = df.loc[~df["artifact"].astype(str).isin(exclude)].copy()
    return _drop_top_ranked_duplicate_rows(cfg, df)


def _drop_top_ranked_duplicate_rows(cfg, df, quiet=False):
    """Remove AF3's duplicated top-ranked model from an already-built per-structure cache.

    The caches were written before this was noticed, so they hold one extra structure per
    artifact: the archive's top-level `{name}_model.cif`, which is an exact coordinate
    copy of whichever seed-sample AF3 ranked best (verified on MutS_alpha A272V straight
    from the tar -- 125 distinct seed-samples, top-level identical to seed-1_sample-1).
    Leaving it in gives that one member weight 2/126 in every RMSF, RMSD and DNA-distance
    value, and lets the ensemble-size sweep draw the same conformation twice -- which is
    precisely the diversity the sweep exists to measure.

    The caches carry only (artifact, structure_index), so the duplicates are identified
    from the structure file index written alongside them and matched back by that pair.
    Filtering on load rather than rebuilding avoids re-parsing every archive off the
    external drive: the cache stays a faithful record of what was read, and the analyses
    see the ensemble without the double-count.

    Falls through untouched if the index is missing, so a cache without its index still
    loads rather than failing at read time.
    """
    prefix = _prefix(cfg)
    if df is None or df.empty or "structure_index" not in df.columns:
        return df
    if sweep_engine.is_core_fit_cache(cfg):
        return df   # built from the pLDDT samples only: no top-level model to drop

    index_path = _cache_dir(cfg) / f"{prefix}_structure_file_index.csv"
    if not index_path.exists():
        print(f"[{prefix}] no structure file index next to the cache — cannot drop "
              f"AF3's duplicated top-ranked model; ensembles keep the double-count")
        return df

    idx = pd.read_csv(index_path)
    path_cols = [c for c in ("structure_member", "structure_file", "structure_path")
                 if c in idx.columns]
    if not path_cols or "artifact" not in idx.columns:
        return df

    # Match `seed-N_sample-M` against the whole path, not the basename: POLE's truncated
    # 268-471 construct names every member `model.cif` and carries the seed only in its
    # directory. An artifact with no seed-named members at all keeps everything, which is
    # what one TP53 artifact (P53_D57G_with_ZN_and_DNA_20251204_115232) needs -- it has no
    # top-level model, so its structure_index 0 is a real sample.
    joined = idx[path_cols].astype(str).agg(" ".join, axis=1)
    is_seed = joined.str.contains(sm._SEED_SAMPLE_RE)
    has_seeded = idx.assign(_s=is_seed).groupby("artifact")["_s"].transform("any")
    dup = idx.loc[has_seeded & ~is_seed, ["artifact", "structure_index"]]
    if dup.empty:
        return df

    keys = set(zip(dup["artifact"].astype(str), dup["structure_index"].astype("int64")))
    mask = [
        (a, i) not in keys
        for a, i in zip(df["artifact"].astype(str), df["structure_index"].astype("int64"))
    ]
    out = df.loc[mask].copy()
    if not quiet:
        print(f"[{prefix}] dropped AF3's duplicated top-ranked model from {len(dup)} "
              f"artifact(s); ensembles now match the pLDDT pool")
    return out


# ── DNA per-structure cache + DNA-distance N-size sweep ───────────────────────
# Skipped entirely for proteins with no "dna_chains" (KRAS).

def _is_dna_residue(residue):
    name = residue.get_resname().strip().upper()
    return name in {"DA", "DT", "DG", "DC", "A", "T", "G", "C", "DU", "U"}


def _iter_atoms_in_chains(structure, chain_ids, atom_filter=None):
    chain_ids = {str(c) for c in chain_ids}
    for model in structure:
        for chain in model:
            if str(chain.id) not in chain_ids:
                continue
            for residue in chain:
                for atom in residue:
                    if atom_filter is None or atom_filter(atom, residue, chain):
                        yield chain, residue, atom
        break  # first model only


def compute_dna_metrics_for_structure(path, protein_chains, dna_chains, threshold=5.0):
    st = sm._load_structure_any_format(path)
    dna_coords = [
        atom.get_coord().astype(float)
        for _, residue, atom in _iter_atoms_in_chains(st, dna_chains)
        if _is_dna_residue(residue)
    ]
    if not dna_coords:
        raise ValueError("no DNA atoms found")
    dna_coords = np.vstack(dna_coords)

    residue_rows = []
    for chain, residue, atom in _iter_atoms_in_chains(
        st, protein_chains, atom_filter=lambda atom, residue, chain: atom.get_name().strip() == "CA"
    ):
        hetflag, resid, icode = residue.id
        if hetflag.strip():
            continue
        c = atom.get_coord().astype(float)
        d = np.sqrt(np.sum((dna_coords - c[None, :]) ** 2, axis=1))
        min_d = float(np.nanmin(d))
        residue_rows.append({
            "chain": str(chain.id), "resid": int(resid), "icode": str(icode).strip(),
            "resname": residue.get_resname(), "min_distance_to_dna": min_d,
            "near_dna": bool(min_d <= threshold),
        })
    residue_df = pd.DataFrame(residue_rows)
    if residue_df.empty:
        raise ValueError("no protein CA residues found")

    distances = residue_df["min_distance_to_dna"].astype(float).to_numpy()
    summary = {
        "whole_protein_mean_min_distance": float(np.nanmean(distances)),
        "whole_protein_median_min_distance": float(np.nanmedian(distances)),
        "whole_protein_min_min_distance": float(np.nanmin(distances)),
        "whole_protein_fraction_near_dna": float(np.nanmean(distances <= threshold)),
        "n_residues": int(len(distances)),
        "n_near_dna_residues": int(np.sum(distances <= threshold)),
    }
    interface_dist = distances[distances <= threshold]
    if len(interface_dist):
        summary.update({
            "interface_mean_min_distance": float(np.nanmean(interface_dist)),
            "interface_median_min_distance": float(np.nanmedian(interface_dist)),
            "interface_min_min_distance": float(np.nanmin(interface_dist)),
            "n_interface_residues": int(len(interface_dist)),
        })
    else:
        summary.update({
            "interface_mean_min_distance": np.nan, "interface_median_min_distance": np.nan,
            "interface_min_min_distance": np.nan, "n_interface_residues": 0,
        })
    return summary, residue_df


def load_dna_structure_cache(cfg):
    path = _cache_dir(cfg) / f"{_prefix(cfg)}_dna_per_structure_summary_cache.csv.gz"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    exclude = _exclude_artifacts(cfg)
    if exclude and "artifact" in df.columns:
        df = df.loc[~df["artifact"].astype(str).isin(exclude)].copy()
    return _drop_top_ranked_duplicate_rows(cfg, df)


def load_dna_residue_cache(cfg):
    path = _cache_dir(cfg) / f"{_prefix(cfg)}_dna_per_structure_residue_cache.csv.gz"
    if not path.exists():
        return None
    # quiet: the summary cache above already reported the same drop for this protein.
    return _drop_top_ranked_duplicate_rows(cfg, pd.read_csv(path), quiet=True)


def build_dna_structure_cache_for_index(cfg, structure_index_subset, max_attempts=3, retry_sleep_sec=2.0, desc=None):
    protein_chains, dna_chains = cfg["protein_chains"], cfg["dna_chains"]
    threshold = float(cfg.get("dna_distance_threshold", 5.0))
    summary_rows, residue_rows, error_rows = [], [], []
    for row in tqdm(structure_index_subset.itertuples(index=False), total=len(structure_index_subset), desc=desc or f"[{_prefix(cfg)}] Building DNA per-structure cache"):
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                summary, residue_df = compute_dna_metrics_for_structure(row.structure_path, protein_chains, dna_chains, threshold)
                base = {
                    "artifact": row.artifact, "Allele": getattr(row, "Allele", None),
                    "label_class": getattr(row, "label_class", None),
                    "structure_index": int(row.structure_index),
                    "structure_file": getattr(row, "structure_file", None),
                    "structure_path": row.structure_path,
                }
                summary_rows.append({**base, **summary})
                residue_rows.extend({**base, **rr} for rr in residue_df.to_dict("records"))
                break
            except Exception as e:
                last_error = e
                if attempt < max_attempts:
                    time.sleep(retry_sleep_sec)
        else:
            error_rows.append({
                "artifact": row.artifact, "Allele": getattr(row, "Allele", None),
                "label_class": getattr(row, "label_class", None),
                "structure_index": int(row.structure_index),
                "structure_path": row.structure_path, "error": repr(last_error), "n_attempts": max_attempts,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(residue_rows), pd.DataFrame(error_rows)


def save_dna_caches(cfg, summary_df=None, residue_df=None):
    cache_dir = _cache_dir(cfg)
    prefix = _prefix(cfg)
    if summary_df is not None and not summary_df.empty:
        save_csv(summary_df, cache_dir / f"{prefix}_dna_per_structure_summary_cache.csv.gz")
    if residue_df is not None and not residue_df.empty:
        save_csv(residue_df, cache_dir / f"{prefix}_dna_per_structure_residue_cache.csv.gz")


def _aggregate_scalar_cache_for_selected_rows(g, metric_cols):
    out = {}
    for col in metric_cols:
        vals = pd.to_numeric(g[col], errors="coerce")
        out[col] = float(np.nanmean(vals)) if vals.notna().any() else np.nan
    return out


_DNA_METRIC_COLS = [
    "whole_protein_mean_min_distance", "whole_protein_median_min_distance", "whole_protein_min_min_distance",
    "whole_protein_fraction_near_dna", "interface_mean_min_distance", "interface_median_min_distance",
    "interface_min_min_distance", "n_interface_residues",
]


def run_dna_ensemble_sweep(cfg, cache_df=None, force=False):
    prefix = _prefix(cfg)
    sweep_dir = _sweep_dir(cfg)
    if "dna_chains" not in cfg:
        print(f"[{prefix}] No dna_chains in config — DNA sweep not applicable, skipping.")
        return {"raw": pd.DataFrame(), "performance": pd.DataFrame(), "summary": pd.DataFrame()}
    if not force and _sweep_outputs_exist(cfg, "DNA_distance", "DNA_distance"):
        print(f"[{prefix}] DNA sweep outputs already exist — skipping. Use force=True to recompute.")
        return {
            "raw": pd.read_csv(sweep_dir / f"{prefix}_DNA_distance_ensemble_sweep_raw.csv"),
            "performance": pd.read_csv(sweep_dir / f"{prefix}_DNA_distance_ensemble_sweep_performance.csv"),
            "summary": pd.read_csv(sweep_dir / f"{prefix}_DNA_distance_ensemble_sweep_summary.csv"),
        }

    cache_df = cache_df if cache_df is not None else load_dna_structure_cache(cfg)
    if cache_df is None or cache_df.empty:
        print(f"[{prefix}] No DNA per-structure cache available; skipping DNA N-sweep")
        return {"raw": pd.DataFrame(), "performance": pd.DataFrame(), "summary": pd.DataFrame()}

    metric_cols = [c for c in _DNA_METRIC_COLS if c in cache_df.columns]
    rng = np.random.default_rng(ENSEMBLE_RANDOM_SEED)
    raw_rows = []
    grouped = {artifact: g.copy() for artifact, g in cache_df.groupby("artifact", sort=False)}
    available = {a: np.array(sorted(g["structure_index"].dropna().unique()), dtype=int) for a, g in grouped.items()}

    for n in cfg["ensemble_sizes"]:
        reps = get_repeats_for_n(n, cfg["ensemble_repeats_by_size"])
        for artifact, g in tqdm(grouped.items(), desc=f"[{prefix}] DNA N={n}"):
            if is_wt_artifact(artifact):
                continue
            av = available[artifact]
            if len(av) < n:
                continue
            allele = artifact_to_allele(artifact)
            label_class = classify_allele_or_artifact(artifact, cfg)
            for r in range(reps):
                sel = rng.choice(av, size=n, replace=False)
                subset = g[g["structure_index"].isin(sel)]
                vals = _aggregate_scalar_cache_for_selected_rows(subset, metric_cols)
                raw_rows.append({
                    "metric_family": "DNA_distance", "artifact": artifact, "Allele": allele,
                    "label_class": label_class, "ensemble_size": int(n), "repeat": int(r),
                    "n_structures_available": int(len(av)), "n_structures_used": int(n), **vals,
                })

    raw = pd.DataFrame(raw_rows)
    save_csv(raw, sweep_dir / f"{prefix}_DNA_distance_ensemble_sweep_raw.csv")

    perf_rows = [
        {"metric_family": "DNA_distance", "ensemble_size": n, "repeat": r, **compare_driver_passenger(g, metric_col)}
        for metric_col in metric_cols
        for (n, r), g in raw.groupby(["ensemble_size", "repeat"])
    ]
    perf = pd.DataFrame(perf_rows)
    save_csv(perf, sweep_dir / f"{prefix}_DNA_distance_ensemble_sweep_performance.csv")

    if not perf.empty:
        perf["auc_separation"] = (perf["roc_auc_driver_high"] - 0.5).abs()
    # PRIMARY AUC = signed driver-high (see D1 sweep note); best_direction/separation
    # are descriptive-only companions.
    summary = perf.groupby(["metric_family", "metric_col", "ensemble_size"], as_index=False).agg(
        n_repeats=("repeat", "nunique"), auc_mean=("roc_auc_driver_high", "mean"),
        auc_std=("roc_auc_driver_high", "std"),
        auc_best_direction_mean=("roc_auc_best_direction", "mean"),
        auc_separation_mean=("auc_separation", "mean"),
        cliffs_mean=("cliffs_delta_driver_minus_passenger", "mean"),
        wasserstein_mean=("wasserstein", "mean"), n_driver=("n_driver", "median"), n_passenger=("n_passenger", "median"),
    ) if not perf.empty else pd.DataFrame()
    save_csv(summary, sweep_dir / f"{prefix}_DNA_distance_ensemble_sweep_summary.csv")
    return {"raw": raw, "performance": perf, "summary": summary}


# ── RMSD "N=all available structures" analysis ────────────────────────────────
# No subsampling: uses every cached aligned structure per artifact, compared
# against every cached WT structure. Driver/Passenger separation at artifact level.

_RMSD_ALL_METRIC_COLS = [
    "rmsd_ensemble_mean_to_wt_mean", "rmsd_mean_structure_to_wt_mean", "rmsd_median_structure_to_wt_mean",
    "rmsd_max_structure_to_wt_mean", "rmsd_std_structure_to_wt_mean",
]


def _as_int_if_possible(value):
    try:
        if pd.notna(value) and float(value).is_integer():
            return int(value)
    except Exception:
        pass
    return value


def _rmsd_structure_values_to_wt_mean(mut_subset, wt_mean_df):
    if mut_subset is None or mut_subset.empty or wt_mean_df is None or wt_mean_df.empty:
        return pd.DataFrame(columns=["structure_index", "rmsd_structure_to_wt_all_mean", "n_rmsd_common_residues"])
    rows = []
    for structure_index, sg in mut_subset.groupby("structure_index", sort=False):
        sg_mean = _mean_coords_by_residue(sg)
        pair = _rmsd_between_mean_coords(sg_mean, wt_mean_df)
        if pair is None:
            continue
        val, n_common = pair
        if np.isfinite(val):
            rows.append({
                "structure_index": _as_int_if_possible(structure_index),
                "rmsd_structure_to_wt_all_mean": float(val), "n_rmsd_common_residues": int(n_common),
            })
    return pd.DataFrame(rows)


def resolve_wt_artifact(cfg, candidates):
    """Pick the WT artifact every mutant is measured against, explicitly.

    This used to be `[a for a in grouped if is_wt_artifact(a)][0]` over a
    `groupby(sort=False)` -- that is, whichever WT happened to appear first in the
    coordinate cache. For six of the seven proteins there is exactly one WT and the
    choice was never in doubt. POLE has three (`POLE_268-471_WT`, `POLE_WT`,
    `POLE_again_WT`) spanning two different constructs, and the one that won by row
    order was `POLE_268-471_WT` -- the truncated, DNA-bound, 2-Mg construct -- serving
    as reference for all 1,227 POLE artifacts, 1,067 of which are full length. That is
    a real modelling decision (it is where the 204-matched-residue figure comes from)
    and it was being made by cache ordering, silently, with no way to see it and no
    guarantee a rebuild in a different order would make it the same way.

    Resolution order:
      1. `wt_artifact` in the config -- an exact artifact name, always wins;
      2. otherwise the sole WT candidate, if there is exactly one;
      3. otherwise raise, listing what was found, rather than picking one.
    """
    prefix = _prefix(cfg)
    names = [str(c) for c in candidates]
    configured = cfg.get("wt_artifact")
    if configured:
        # artifact_key, as in run_ensemble_sweep_parallel.resolve_wt_artifact: PTEN's
        # config says PTEN_WT and its cache PTEN_WT_data, which an exact match refused.
        hits = [a for a in names if artifact_key(a) == artifact_key(configured)]
        if len(hits) != 1:
            raise RuntimeError(
                f"[{prefix}] configured wt_artifact {configured!r} matches {hits}. "
                f"WT artifacts found: {sorted(a for a in names if is_wt_artifact(a))}"
            )
        return hits[0]

    wt = [a for a in names if is_wt_artifact(a)]
    if not wt:
        raise RuntimeError(f"[{prefix}] No WT artifact found in the aligned coordinate cache")
    if len(wt) > 1:
        raise RuntimeError(
            f"[{prefix}] {len(wt)} WT artifacts present and none chosen: {sorted(wt)}. "
            f"Set \"wt_artifact\" in the protein config -- the reference is a modelling "
            f"decision and must not fall out of cache row order."
        )
    return wt[0]


def run_rmsd_all_instance_analysis(cfg, coord_cache_df):
    prefix = _prefix(cfg)
    if coord_cache_df is None or coord_cache_df.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    grouped = {artifact: g.copy() for artifact, g in coord_cache_df.groupby("artifact", sort=False)}
    wt_artifact = resolve_wt_artifact(cfg, list(grouped))
    wt_subset = grouped[wt_artifact]
    wt_mean_df = _mean_coords_by_residue(wt_subset)
    n_wt_structures = int(wt_subset["structure_index"].dropna().nunique())

    artifact_rows, per_structure_frames = [], []
    for artifact, g in tqdm(grouped.items(), desc=f"[{prefix}] RMSD all instances"):
        if is_wt_artifact(artifact):
            continue
        allele = artifact_to_allele(artifact)
        label_class = classify_allele_or_artifact(artifact, cfg)
        n_structures = int(g["structure_index"].dropna().nunique())
        mut_mean_df = _mean_coords_by_residue(g)
        ensemble_pair = _rmsd_between_mean_coords(mut_mean_df, wt_mean_df)
        if ensemble_pair is None:
            continue
        per_structure_df = _rmsd_structure_values_to_wt_mean(g, wt_mean_df)
        if per_structure_df.empty:
            continue

        per_structure_vals = per_structure_df["rmsd_structure_to_wt_all_mean"].to_numpy(float)
        per_structure_common = int(per_structure_df["n_rmsd_common_residues"].max())
        ensemble_rmsd, ensemble_common = ensemble_pair

        per_structure_df.insert(0, "metric_family", "RMSD_all_instances")
        per_structure_df.insert(1, "rmsd_scope", "N_all_available_structures")
        per_structure_df.insert(2, "artifact", artifact)
        per_structure_df.insert(3, "Allele", allele)
        per_structure_df.insert(4, "label_class", label_class)
        per_structure_df.insert(5, "wt_artifact", wt_artifact)
        per_structure_df.insert(6, "ensemble_size_label", "all_available")
        per_structure_df["n_structures_available"] = n_structures
        per_structure_df["n_wt_structures_available"] = n_wt_structures
        per_structure_frames.append(per_structure_df)

        artifact_rows.append({
            "metric_family": "RMSD_all_instances", "rmsd_scope": "N_all_available_structures",
            "artifact": artifact, "Allele": allele, "label_class": label_class, "wt_artifact": wt_artifact,
            "ensemble_size_label": "all_available", "n_structures_available": n_structures,
            "n_wt_structures_available": n_wt_structures, "n_structures_used": int(per_structure_vals.size),
            "n_wt_structures_used": n_wt_structures,
            "rmsd_ensemble_mean_to_wt_mean": float(ensemble_rmsd),
            "rmsd_mean_structure_to_wt_mean": float(np.nanmean(per_structure_vals)),
            "rmsd_median_structure_to_wt_mean": float(np.nanmedian(per_structure_vals)),
            "rmsd_max_structure_to_wt_mean": float(np.nanmax(per_structure_vals)),
            "rmsd_std_structure_to_wt_mean": float(np.nanstd(per_structure_vals)),
            "n_rmsd_structures_used": int(per_structure_vals.size),
            "n_rmsd_common_residues": int(max(ensemble_common, per_structure_common)),
        })

    artifact_df = pd.DataFrame(artifact_rows)
    per_structure_df = pd.concat(per_structure_frames, ignore_index=True) if per_structure_frames else pd.DataFrame()
    return artifact_df, per_structure_df, wt_artifact


def run_rmsd_all_instances(cfg, force=False):
    prefix = _prefix(cfg)
    sweep_dir = _sweep_dir(cfg)
    out_paths = {
        "artifact_summary": sweep_dir / f"{prefix}_RMSD_all_instances_artifact_summary.csv",
        "performance": sweep_dir / f"{prefix}_RMSD_all_instances_performance.csv",
    }
    if not force and all(p.exists() for p in out_paths.values()):
        print(f"[{prefix}] RMSD all-instances outputs already exist — skipping. Use force=True to recompute.")
        return pd.read_csv(out_paths["artifact_summary"])

    coord_cache_df = load_rmsf_coord_cache(cfg)
    if coord_cache_df is None or coord_cache_df.empty:
        print(f"[{prefix}] No RMSF-aligned coordinate cache; skipping RMSD all-instances analysis")
        return pd.DataFrame()

    artifact_df, structure_df, wt_artifact = run_rmsd_all_instance_analysis(cfg, coord_cache_df)
    if artifact_df.empty:
        print(f"[{prefix}] RMSD all-instances analysis produced no artifact-level rows")
        return artifact_df

    coverage_df = (
        artifact_df.groupby("label_class", dropna=False)
        .agg(n_artifacts=("artifact", "nunique"), n_structures_available_sum=("n_structures_available", "sum"),
             n_structures_used_sum=("n_structures_used", "sum"), median_structures_used=("n_structures_used", "median"),
             min_structures_used=("n_structures_used", "min"), max_structures_used=("n_structures_used", "max"))
        .reset_index().sort_values("label_class")
    )
    structure_count_summary_df = pd.DataFrame([{
        "analysis": "RMSD N=all available structures", "n_artifacts": int(artifact_df["artifact"].nunique()),
        "wt_artifact": wt_artifact, "n_wt_structures_used": int(artifact_df["n_wt_structures_used"].max()),
        "mutant_structures_used_min": int(artifact_df["n_structures_used"].min()),
        "mutant_structures_used_median": float(artifact_df["n_structures_used"].median()),
        "mutant_structures_used_max": int(artifact_df["n_structures_used"].max()),
    }])

    perf_rows = []
    for metric_col in _RMSD_ALL_METRIC_COLS:
        if metric_col not in artifact_df.columns:
            continue
        row = compare_driver_passenger(artifact_df, metric_col)
        row.update({"metric_family": "RMSD_all_instances", "rmsd_scope": "N_all_available_structures"})
        perf_rows.append(row)
    perf_df = pd.DataFrame(perf_rows)
    if "roc_auc_best_direction" not in perf_df.columns:
        perf_df["roc_auc_best_direction"] = np.nan
    if "roc_auc_driver_high" not in perf_df.columns:
        perf_df["roc_auc_driver_high"] = np.nan
    # This is a descriptive SEPARATION leaderboard (which metric best distinguishes the
    # classes), not a predictive-performance ranking. Rank by distance-from-chance of the
    # signed AUC so a strongly inverse metric is not buried; the signed driver-high AUC and
    # Cliff's delta (both directional) travel alongside for interpretation.
    perf_df["auc_separation"] = (perf_df["roc_auc_driver_high"] - 0.5).abs()
    perf_df = perf_df.sort_values(["auc_separation", "metric_col"], ascending=[False, True], na_position="last").reset_index(drop=True)

    ranking_cols = ["metric_family", "rmsd_scope", "metric_col", "n_driver", "n_passenger", "driver_mean",
                    "passenger_mean", "cliffs_delta_driver_minus_passenger", "mannwhitney_p", "ks_stat", "ks_p",
                    "roc_auc_driver_high", "auc_separation", "roc_auc_best_direction", "wasserstein"]
    ranking_df = perf_df[[c for c in ranking_cols if c in perf_df.columns]].copy()
    headline_df = ranking_df.head(1).copy()
    if not headline_df.empty:
        headline_df.insert(0, "analysis", "RMSD N=all available structures")
        headline_df["wt_artifact"] = wt_artifact
        headline_df["n_wt_structures_used"] = int(artifact_df["n_wt_structures_used"].max())
        headline_df["mutant_structures_used_min"] = int(artifact_df["n_structures_used"].min())
        headline_df["mutant_structures_used_median"] = float(artifact_df["n_structures_used"].median())
        headline_df["mutant_structures_used_max"] = int(artifact_df["n_structures_used"].max())

    save_csv(artifact_df, sweep_dir / f"{prefix}_RMSD_all_instances_artifact_summary.csv")
    save_csv(structure_df, sweep_dir / f"{prefix}_RMSD_all_instances_per_structure_values.csv")
    save_csv(coverage_df, sweep_dir / f"{prefix}_RMSD_all_instances_coverage_by_class.csv")
    save_csv(structure_count_summary_df, sweep_dir / f"{prefix}_RMSD_all_instances_structure_count_summary.csv")
    save_csv(perf_df, sweep_dir / f"{prefix}_RMSD_all_instances_performance.csv")
    save_csv(ranking_df, sweep_dir / f"{prefix}_RMSD_all_instances_metric_ranking.csv")
    save_csv(headline_df, sweep_dir / f"{prefix}_RMSD_all_instances_headline_summary.csv")

    _plot_rmsd_all_instances(cfg, artifact_df, ranking_df)
    return artifact_df


def _plot_rmsd_all_instances(cfg, artifact_df, ranking_df):
    import matplotlib.pyplot as plt
    prefix = _prefix(cfg)
    plot_dir = _plot_dir(cfg, "rmsd_diagnostics")

    auc_plot_df = ranking_df.dropna(subset=["roc_auc_driver_high"]).copy()
    if not auc_plot_df.empty:
        # Plot the SIGNED driver-high AUC with a chance line at 0.5, so metrics that
        # separate in the passenger-high direction sit below 0.5 instead of being
        # flipped up to look predictive. Ordering is by distance from chance.
        auc_plot_df = auc_plot_df.sort_values("auc_separation", ascending=True)
        plt.figure(figsize=(8.5, max(3.4, 0.45 * len(auc_plot_df))))
        plt.barh(auc_plot_df["metric_col"], auc_plot_df["roc_auc_driver_high"], color="#4C78A8")
        plt.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        plt.xlabel("Signed driver-high ROC AUC (Driver vs Passenger; 0.5 = chance)")
        plt.ylabel("RMSD metric")
        plt.title(f"{prefix} RMSD N=all artifact-level Driver/Passenger separation")
        # Data-dependent limits that ALWAYS bracket the chance line: a signed AUC below
        # 0.5 (passenger-high) must stay visible, so never clip it off the left edge.
        _lo = float(np.nanmin(auc_plot_df["roc_auc_driver_high"]))
        _hi = float(np.nanmax(auc_plot_df["roc_auc_driver_high"]))
        plt.xlim(max(0.0, min(_lo, 0.5) - 0.05), min(1.0, max(_hi, 0.5) + 0.05))
        plt.grid(True, axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{prefix}_RMSD_all_instances_metric_AUC.png", dpi=180, bbox_inches="tight")
        plt.close()


# ── Unified all-metrics comparisons ───────────────────────────────────────────

def run_unified_comparison_at_max_n(cfg):
    """D1/RMSF/DNA at the largest swept N; RMSD replaced by the true N=all point."""
    prefix = _prefix(cfg)
    aware_dir = _aware_dir(cfg)
    sweep_dir = _sweep_dir(cfg)
    max_n = max(cfg["ensemble_sizes"])

    combined_summary_path = aware_dir / f"{prefix}_all_metrics_ensemble_sweep_summary.csv"
    rmsd_all_perf_path = sweep_dir / f"{prefix}_RMSD_all_instances_performance.csv"
    if not combined_summary_path.exists() or not rmsd_all_perf_path.exists():
        print(f"[{prefix}] Missing combined sweep summary or RMSD-all performance; skipping unified comparison.")
        return pd.DataFrame()

    sweep = pd.read_csv(combined_summary_path)
    sweep["metric_col"] = sweep["metric_col"].fillna("d1_score")
    sweep_at_n = sweep[sweep["ensemble_size"] == max_n].copy()
    sweep_at_n = sweep_at_n.rename(columns={"auc_mean": "roc_auc", "cliffs_mean": "cliffs_delta", "wasserstein_mean": "wasserstein"})
    sweep_at_n["source"] = f"sweep_N{max_n}"
    keep_cols = ["metric_family", "metric_col", "roc_auc", "auc_std", "cliffs_delta", "wasserstein", "n_driver", "n_passenger", "source"]
    sweep_at_n = sweep_at_n[[c for c in keep_cols if c in sweep_at_n.columns]]

    rmsd_all = pd.read_csv(rmsd_all_perf_path)
    # Use the SIGNED driver-high AUC here too, so the unified table stays on one
    # consistent (directional) scale rather than mixing signed and best-direction AUCs.
    rmsd_all = rmsd_all.rename(columns={"roc_auc_driver_high": "roc_auc", "cliffs_delta_driver_minus_passenger": "cliffs_delta"})
    rmsd_all["auc_std"] = np.nan
    rmsd_all["source"] = "N_all_instances"
    rmsd_all = rmsd_all[[c for c in keep_cols if c in rmsd_all.columns]]

    combined = pd.concat([sweep_at_n[sweep_at_n["metric_family"] != "RMSD"], rmsd_all], ignore_index=True)
    combined["roc_auc"] = pd.to_numeric(combined["roc_auc"], errors="coerce")
    # Rank by distance-from-chance so a strongly inverse metric is not pushed to the
    # bottom as if it were uninformative; the plotted value stays signed.
    combined["auc_separation"] = (combined["roc_auc"] - 0.5).abs()
    combined = combined.sort_values("auc_separation", ascending=False).reset_index(drop=True)
    combined["rank"] = combined.index + 1

    out = aware_dir / f"{prefix}_all_metrics_unified_comparison.csv"
    save_csv(combined[keep_cols + ["auc_separation", "rank"]].round(4), out)
    _plot_unified_comparison(cfg, combined, max_n, aware_dir / f"{prefix}_all_metrics_unified_comparison.png")
    return combined


def _plot_unified_comparison(cfg, combined, max_n, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    prefix = _prefix(cfg)

    family_colors = {"D1": "#4C72B0", "RMSF": "#DD8452", "DNA_distance": "#55A868", "RMSD": "#C44E52"}
    default_color = "#8172B2"

    def _label(row):
        col = str(row["metric_col"]) if pd.notna(row["metric_col"]) else ""
        fam = str(row["metric_family"])
        return f"{fam} | {col}" if col and col != fam and col != "d1_score" else fam

    df = combined.copy()
    df["label"] = df.apply(_label, axis=1)
    if "auc_separation" not in df.columns:
        df["auc_separation"] = (df["roc_auc"] - 0.5).abs()
    df = df.sort_values("auc_separation", ascending=True).reset_index(drop=True)
    colors = [family_colors.get(r["metric_family"], default_color) for _, r in df.iterrows()]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.45 * len(df))), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    bars = ax.barh(df["label"], df["roc_auc"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="black", lw=1.0, ls="--", alpha=0.6)
    has_std = df["auc_std"].notna()
    if has_std.any():
        ax.errorbar(df.loc[has_std, "roc_auc"], df.loc[has_std, "label"], xerr=df.loc[has_std, "auc_std"],
                    fmt="none", color="black", capsize=3, lw=1.0, alpha=0.6)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{row['roc_auc']:.3f}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("ROC-AUC (driver-high, signed; 0.5 = chance)", fontsize=10)
    ax.set_title(f"{prefix} — Metric comparison at max ensemble size\n(RMSD: all instances; others: N={max_n})", fontsize=11)
    # Bracket the chance line so passenger-high (signed AUC < 0.5) metrics stay on-panel;
    # the extra right margin leaves room for the value labels drawn above.
    _lo = float(np.nanmin(df["roc_auc"])) if df["roc_auc"].notna().any() else 0.5
    _hi = float(np.nanmax(df["roc_auc"])) if df["roc_auc"].notna().any() else 0.5
    ax.set_xlim(max(0.0, min(_lo, 0.5) - 0.05), min(1.05, max(_hi, 0.5) + 0.12))
    patches = [mpatches.Patch(color=c, label=f) for f, c in family_colors.items() if f in df["metric_family"].values]
    patches.append(plt.Line2D([0], [0], color="black", lw=1, ls="--", label="random (0.5)"))
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    ax2 = axes[1]
    cd = df["cliffs_delta"].fillna(0)
    ax2.barh(df["label"], cd.abs(), color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("|Cliff's delta| (driver - passenger)", fontsize=10)
    ax2.set_xlim(0, min(1.05, cd.abs().max() + 0.1))
    for i, (_, row) in enumerate(df.iterrows()):
        v = row["cliffs_delta"]
        if pd.notna(v):
            ax2.text(abs(v) + 0.01, i, f"{v:+.3f}", va="center", ha="left", fontsize=8)
    ax2.set_title("Effect size", fontsize=11)
    ax2.set_yticklabels([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Full ensemble-aware pipeline for one protein ──────────────────────────────

def run_ensemble_aware_pipeline(protein, force=False):
    """Run every ensemble-aware stage for one protein, in dependency order.

    Assumes the RMSF-aligned coordinate cache (and, for proteins with DNA
    chains, the DNA per-structure cache) already exists under cfg["cache_dir"]
    — building those caches from raw structures is repair_rmsf_cache_chunk.py /
    repair_dna_cache_chunk.py's job, not this pipeline's.
    """
    cfg = get_ensemble_aware_config(protein)
    prefix = _prefix(cfg)
    print(f"=== [{prefix}] Ensemble-aware pipeline ===")

    run_d1_ensemble_sweep(cfg, force=force)
    run_rmsf_rmsd_ensemble_sweep(protein, force=force)
    run_rmsd_all_instances(cfg, force=force)
    if "dna_chains" in cfg:
        run_dna_ensemble_sweep(cfg, force=force)

    # Combine per-family sweep CSVs into one raw/performance/summary set.
    family_names = ["D1_true_plddt", "RMSF_aligned_coord", "RMSD_aligned_coord"]
    if "dna_chains" in cfg:
        family_names.append("DNA_distance")
    sweep_dir = _sweep_dir(cfg)
    aware_dir = _aware_dir(cfg)
    for kind, out_name in [("raw", "raw"), ("performance", "performance"), ("summary", "summary")]:
        parts = []
        for name in family_names:
            p = sweep_dir / f"{prefix}_{name}_ensemble_sweep_{kind}.csv"
            if p.exists():
                parts.append(pd.read_csv(p))
        combined = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
        save_csv(combined, aware_dir / f"{prefix}_all_metrics_ensemble_sweep_{out_name}.csv")

    run_unified_comparison_at_max_n(cfg)
    print(f"=== [{prefix}] Ensemble-aware pipeline complete ===")
