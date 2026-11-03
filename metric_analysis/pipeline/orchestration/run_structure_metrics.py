#!/usr/bin/env python3
"""Single-command driver for the whole per-protein structure-metrics pipeline.

Replaces hand-running {Protein}_structure_metrics.ipynb (one notebook per
protein, ~40-50 cells each) plus the three launch_*.sh bash steps. Everything
is driven off metric_analysis/protein_configs.json, so adding a protein means
adding a config entry, not copying a notebook.

Stages, in dependency order:
  plddt               extract per-residue pLDDT matrices from the raw structures
  plddt_distribution  per-variant pLDDT distribution summaries (the canonical cohort's
                      source paths and counts) plus the residue-effect sweeps
  ensemble_aware      coordinate descriptors (RMSF, RMSD, positional KS) by ensemble size,
                      from the coordinate cache (the WT rigid-core frame by default)

The DRIFT release keeps only these three. The notebooks' other stages (D1 and pLDDT-ratio
profiles, profile KS, LDA discrimination, RMSF/DNA batches) fed no result of the paper.

Usage
-----
    run-structure-metrics --protein POLE,PTEN --stage plddt_distribution
    run-structure-metrics --protein TP53 --stage plddt        # needs the raw archives

Caches are respected by default: every stage skips work whose outputs already
exist, so re-running is cheap. --force recomputes.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: this script only ever writes PNGs, never displays
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from metric_analysis.pipeline.library import structure_metrics as sm
from metric_analysis.pipeline.library import ensemble_aware_pipeline as eap
from metric_analysis.pipeline.config.protein_config import (
    get_protein_config,
    supported_proteins,
    enabled_proteins,
)
from metric_analysis.pipeline.labels import classify_allele_or_artifact

from metric_analysis.pipeline.paths import REPO_ROOT as PROJECT_ROOT

ALL_STAGES = ["plddt", "plddt_distribution", "ensemble_aware"]

BASE_REQUIRED_KEYS = ["output_prefix", "read_root_path", "source_type", "chain_ids", "protein_chains"]


def build_structure_metrics_config(cfg):
    """StructureMetricsConfig for the pLDDT/RMSF/DNA batch drivers in structure_metrics.py."""
    prefix = cfg["output_prefix"]
    return sm.StructureMetricsConfig(
        mut_name=prefix,
        read_root_path=cfg["read_root_path"],
        chain_ids=tuple(cfg["chain_ids"]),
        source_type=cfg["source_type"],
        artifact_glob=cfg.get("artifact_glob", "*.tar.gz"),
        # A subunit dataset reads its PARENT complex's extracted CSVs, so this
        # must be configurable -- it was hardcoded to PLDDTs/<prefix>, which
        # silently sent MSH2 to a directory that does not exist.
        plddt_dir=cfg.get("plddt_dir") or (PROJECT_ROOT / "PLDDTs" / prefix),
        output_dir=PROJECT_ROOT / "metric_analysis" / "structure_metrics" / prefix,
        drop_matching_sample=True,
        skip_existing_csv=True,
        wt_artifact=cfg.get("wt_artifact"),
    )


def _output_dir(cfg):
    return PROJECT_ROOT / "metric_analysis" / "structure_metrics" / cfg["output_prefix"]


# ── Stages ────────────────────────────────────────────────────────────────────

def stage_plddt(cfg, sm_config, state, force=False):
    """Extract pLDDT matrices from raw structures into PLDDTs/<Protein>/*.csv."""
    if force:
        sm_config = sm_config.__class__(**{**sm_config.__dict__, "skip_existing_csv": False})
    summary = sm.process_artifacts_to_plddt(sm_config, show_progress=True)
    print(f"  pLDDT artifacts processed: {len(summary)}")
    return summary


def _match_configured_wt_plddt_file(plddt_files, configured_wt, *, prefix):
    """Map a raw-artifact WT name to its canonical pLDDT CSV.

    pLDDT extraction applies ``canonical_artifact_stem`` before naming CSVs. For
    example, TP53's raw ``P53_WT_with_ZN_and_DNA.tar.gz`` becomes
    ``P53_WT_plddt_values.csv``. Accept the literal name too so older/noncanonical
    caches remain readable, but still require exactly one match.
    """
    configured_wt = str(configured_wt)
    canonical_wt = sm.canonical_artifact_stem(configured_wt)
    wt_bases = {configured_wt, canonical_wt}
    expected_stems = wt_bases | {f"{base}_plddt_values" for base in wt_bases}
    matches = [path for path in plddt_files if path.stem in expected_stems]
    if len(matches) != 1:
        raise ValueError(
            f"[{prefix}] wt_artifact={configured_wt!r} (canonical={canonical_wt!r}) "
            f"matched {len(matches)} pLDDT CSVs: {[path.name for path in matches]}"
        )
    return matches[0]


def stage_plddt_distribution(cfg, sm_config, state, force=False):
    """Export full-distribution pLDDT metrics, comparisons, plots, and N sweep."""
    prefix = cfg["output_prefix"]
    out_dir = _output_dir(cfg)
    summary_path = out_dir / f"{prefix}_plddt_distribution_summary.csv"
    residue_path = out_dir / f"{prefix}_plddt_distribution_per_residue.csv.gz"
    performance_path = out_dir / f"{prefix}_plddt_metric_performance.csv"
    outlier_threshold = float(cfg.get(
        "plddt_outlier_threshold", sm.DEFAULT_PLDDT_OUTLIER_THRESHOLD,
    ))
    plddt_files = sm.list_plddt_csv_files(sm_config.resolved_plddt_dir())
    if not plddt_files:
        raise FileNotFoundError(
            f"No pLDDT CSVs under {sm_config.resolved_plddt_dir()} — "
            "run the plddt stage first"
        )
    configured_wt = cfg.get("wt_artifact")
    wt_file = (
        _match_configured_wt_plddt_file(plddt_files, configured_wt, prefix=prefix)
        if configured_wt else None
    )
    use_cache = not force and summary_path.exists() and residue_path.exists()
    if use_cache:
        summary_df = pd.read_csv(summary_path)
        residue_df = pd.read_csv(residue_path)
        required_summary = {
            "plddt_outlier_score_mean", "plddt_outlier_fraction_mean",
            "plddt_outlier_threshold",
        }
        required_residue = {
            "outlier_score", "outlier_fraction", "outlier_mean_deviation",
            "outlier_count", "outlier_threshold", "wt_median",
        }
        cached_threshold = pd.to_numeric(
            summary_df.get(
                "plddt_outlier_threshold", pd.Series(dtype=float)
            ),
            errors="coerce",
        ).dropna()
        use_cache = (
            required_summary.issubset(summary_df.columns)
            and required_residue.issubset(residue_df.columns)
            and not cached_threshold.empty
            and np.allclose(cached_threshold.to_numpy(float), outlier_threshold)
        )
        if use_cache:
            print(
                f"  pLDDT distribution: using cached {summary_path.name} "
                f"and {residue_path.name}"
            )
        else:
            print(
                "  pLDDT distribution cache predates the thresholded outlier "
                "metric or uses a different threshold — recomputing"
            )
    if not use_cache:
        summary_df, residue_df = sm.build_plddt_distribution_tables(
            plddt_files, wt_file=wt_file,
            outlier_threshold=outlier_threshold,
        )
        if summary_df.empty:
            raise ValueError(f"[{prefix}] pLDDT distribution tables came out empty")

    # Labels remain display/evaluation metadata and enter only after all WT-vs-
    # mutant effects have been computed.
    summary_df["mutation_type"] = summary_df["source_file"].map(
        lambda path: classify_allele_or_artifact(Path(str(path)).stem, cfg)
    )
    summary_df["label"] = summary_df["mutation_type"].map(
        {"Driver": 1, "Passenger": 0}
    )
    consistency = pd.to_numeric(
        summary_df.get("d1_consistency_abs_error"), errors="coerce"
    )
    if consistency.notna().any() and float(consistency.max()) > 1e-10:
        raise ValueError(
            f"[{prefix}] distribution decomposition does not reproduce D1: "
            f"max absolute error={float(consistency.max()):.3g}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    residue_df.to_csv(residue_path, index=False, compression="gzip")

    profile_result = sm.plot_plddt_distribution_profile_by_class(
        residue_df,
        summary_df,
        residue_start=cfg.get("plddt_residue_start"),
        hotspot_range=cfg.get("hotspot_range"),
        out_path=out_dir / f"{prefix}_plddt_distribution_profile_by_class.png",
    )
    distribution_result = sm.plot_plddt_distribution_summary(
        summary_df,
        dataset_name=prefix,
        out_path=out_dir / f"{prefix}_plddt_distribution_by_class.png",
    )
    d1_result = sm.plot_plddt_distribution_vs_d1(
        summary_df,
        dataset_name=prefix,
        out_path=out_dir / f"{prefix}_plddt_distribution_vs_d1.png",
    )

    # Evaluate old and new metrics on one complete-case cohort so AUC changes
    # cannot be caused by a different set of variants.
    comparison = summary_df.copy()
    ks_path = out_dir / f"{prefix}_plddt_ks_summary.csv"
    if ks_path.exists():
        ks_df = pd.read_csv(ks_path)
        if {"mutation_code", "plddt_ks_mean"}.issubset(ks_df.columns):
            comparison = comparison.merge(
                ks_df[["mutation_code", "plddt_ks_mean"]].drop_duplicates(
                    "mutation_code"
                ),
                on="mutation_code",
                how="left",
            )
    metric_specs = [
        ("d1_mean", "D1 (mean shift)"),
        ("plddt_ks_mean", "KS"),
        ("plddt_w1_unit_mean", "Wasserstein-1"),
        (
            "plddt_outlier_score_mean",
            f"Outlier score (tau={outlier_threshold:g})",
        ),
        (
            "plddt_w1_shape_excess_relative_mean",
            "W1 shape excess",
        ),
        ("plddt_energy_mean", "Energy"),
        ("plddt_cvm_mean", "Cramér-von Mises"),
    ]
    available = [(col, label) for col, label in metric_specs if col in comparison]
    common = comparison[
        comparison["mutation_type"].isin(["Driver", "Passenger"])
    ].copy()
    for col, _ in available:
        common[col] = pd.to_numeric(common[col], errors="coerce")
    common = common.dropna(subset=[col for col, _ in available])
    performance_rows = []
    evaluation = common.rename(columns={"mutation_type": "label_class"})
    for col, label in available:
        performance_rows.append({
            "protein": prefix,
            "metric": label,
            "cohort": "common_complete_case",
            **eap.compare_driver_passenger(evaluation, col),
            "spearman_vs_d1": float(
                evaluation[col].corr(evaluation["d1_mean"], method="spearman")
            ),
        })
    performance_df = pd.DataFrame(performance_rows)
    performance_df.to_csv(performance_path, index=False)

    if not performance_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(performance_df))
        auc = pd.to_numeric(
            performance_df["roc_auc_driver_high"], errors="coerce"
        ).to_numpy(float)
        colors = ["#6c757d", "#8d99ae"] + ["#2a9d8f"] * max(len(x) - 2, 0)
        ax.bar(x, auc, color=colors[:len(x)], edgecolor="black", linewidth=0.5)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x, performance_df["metric"], rotation=25, ha="right")
        ax.set_ylabel("ROC-AUC (driver-high, matched cohort)")
        ax.set_title(f"{prefix}: D1, KS, and full-distribution metrics")
        ax.grid(axis="y", alpha=0.25)
        for position, value in zip(x, auc):
            if np.isfinite(value):
                ax.text(position, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"{prefix}_plddt_metric_auc_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    if wt_file is None:
        wt_sources = summary_df.get("wt_source_file", pd.Series(dtype=str)).dropna()
        if wt_sources.empty:
            raise ValueError(
                f"[{prefix}] distribution summary does not identify its WT source"
            )
        wt_file = Path(str(wt_sources.iloc[0]))
    auc_sweep_result = eap.run_plddt_distribution_ensemble_sweep(
        cfg, plddt_files=plddt_files, wt_path=wt_file, force=force
    )
    tested = int((summary_df["status"] == "ok").sum())
    print(
        f"  pLDDT distribution: {tested}/{len(summary_df)} variants tested; "
        f"{len(residue_df)} residue rows; wrote compressed residue table, "
        "summary/performance tables, 4 figures, and ensemble-size sweep"
    )
    state["plddt_distribution_summary_df"] = summary_df
    state["plddt_distribution_residue_df"] = residue_df
    return {
        "summary_df": summary_df,
        "residue_df": residue_df,
        "performance_df": performance_df,
        "profile_result": profile_result,
        "distribution_result": distribution_result,
        "d1_result": d1_result,
        "auc_sweep_result": auc_sweep_result,
    }


def stage_ensemble_aware(cfg, sm_config, state, force=False):
    """Coordinate-cache ensemble-size sweeps (RMSF, RMSD, positional KS; DNA where cached)."""
    eap.run_ensemble_aware_pipeline(cfg["output_prefix"], force=force)


STAGE_FUNCS = {
    "plddt": stage_plddt,
    "plddt_distribution": stage_plddt_distribution,
    "ensemble_aware": stage_ensemble_aware,
}


# ── Driver ────────────────────────────────────────────────────────────────────

def partition_stages(cfg, stages):
    """Apply an optional dataset stage allowlist; legacy configs support all stages."""
    allowed = set(cfg.get("structure_metrics_stages", ALL_STAGES))
    unknown = allowed - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"{cfg['output_prefix']}: unknown configured stages {sorted(unknown)}")
    return ([stage for stage in stages if stage in allowed],
            [stage for stage in stages if stage not in allowed])


def run_protein(protein, stages, force=False):
    cfg = get_protein_config(protein, BASE_REQUIRED_KEYS)
    _, unsupported = partition_stages(cfg, stages)
    if unsupported:
        raise ValueError(f"{protein}: unsupported stages {unsupported}; "
                         "this dataset uses pre-extracted subunit pLDDT only")
    sm_config = build_structure_metrics_config(cfg)
    state = {}
    results = {}

    # A subunit dataset (MSH2, MSH6) reads its parent complex's pLDDT directory,
    # so every read has to be restricted to its own rows and its own artifact
    # files. Entering the scope once here covers all 22 read sites inside
    # structure_metrics without threading a parameter through them. Absent for
    # every whole-protein dataset, where it is a no-op.
    row_block = cfg.get("plddt_row_range")
    include = cfg.get("cohort_subunit_token")
    if row_block is not None or include is not None:
        print(f"[{protein}] pLDDT scope: rows={row_block} files~_{include}_ "
              f"in {sm_config.resolved_plddt_dir()}")

    print(f"\n{'=' * 70}\n{protein}\n{'=' * 70}")
    for stage in stages:
        t0 = time.time()
        print(f"\n[{protein}] --- {stage} ---")
        try:
            with sm.plddt_scope(row_block=row_block, include=include):
                STAGE_FUNCS[stage](cfg, sm_config, state, force=force)
            results[stage] = ("DONE", time.time() - t0)
            print(f"[{protein}] {stage}: DONE ({time.time() - t0:.1f}s)")
        except Exception as e:
            results[stage] = (f"FAILED: {e}", time.time() - t0)
            print(f"[{protein}] {stage}: FAILED after {time.time() - t0:.1f}s: {e!r}", file=sys.stderr)
            traceback.print_exc()
        finally:
            # Several plotting helpers build figures via pyplot without closing them, so
            # figures accumulate across a --protein all run until matplotlib warns and
            # memory climbs. Everything is saved to disk by now (Agg backend, nothing is
            # ever displayed), so dropping them here is free.
            plt.close("all")
    return results


def main():
    # `known` accepts disabled proteins so an explicit --protein PTEN still runs;
    # `default_all` is what bare --protein all fans out to (enabled entries only).
    known = supported_proteins(BASE_REQUIRED_KEYS)
    default_all = enabled_proteins(BASE_REQUIRED_KEYS)
    scaffolded = [p for p in known if p not in default_all]
    parser = argparse.ArgumentParser(
        description="Run the full structure-metrics pipeline for one or more proteins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--protein", default="all",
                        help=f"comma-separated protein names or 'all'. Known: {', '.join(known)}"
                             + (f" (scaffolded, excluded from 'all' — name explicitly to run: "
                                f"{', '.join(scaffolded)})" if scaffolded else ""))
    parser.add_argument("--stage", default="all",
                        help=f"comma-separated stages or 'all' (supported stages per dataset). Order: {', '.join(ALL_STAGES)}")
    parser.add_argument("--force", action="store_true", help="recompute even when cached outputs exist")
    args = parser.parse_args()

    proteins = default_all if args.protein == "all" else [p.strip() for p in args.protein.split(",") if p.strip()]
    if args.protein == "all" and scaffolded:
        print(f"[config] skipping scaffolded protein(s) {scaffolded} "
              f"(\"enabled\": false in protein_configs.json; run them by name to override)")
    unknown = [p for p in proteins if p not in known]
    if unknown:
        parser.error(f"unknown protein(s) {unknown}; known: {known}")

    stages = ALL_STAGES if args.stage == "all" else [s.strip() for s in args.stage.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGE_FUNCS]
    if bad:
        parser.error(f"unknown stage(s) {bad}; known: {ALL_STAGES}")
    stages = [s for s in ALL_STAGES if s in stages]  # keep dependency order

    stage_plan = {}
    for protein in proteins:
        cfg = get_protein_config(protein, BASE_REQUIRED_KEYS)
        selected, unsupported = partition_stages(cfg, stages)
        if unsupported and args.protein != "all" and args.stage != "all":
            parser.error(f"{protein}: unsupported stages {unsupported}; "
                         f"supported stages: {cfg.get('structure_metrics_stages', ALL_STAGES)}. "
                         "Use MutS_alpha for merged raw extraction and coordinate analyses.")
        if unsupported:
            print(f"[{protein}] SKIP unsupported stages: {', '.join(unsupported)}")
        if selected:
            stage_plan[protein] = selected

    print(f"Proteins: {proteins}")
    print(f"Stages:   {stages}")
    print(f"Force:    {args.force}")

    t0 = time.time()
    all_results = {p: run_protein(p, selected, force=args.force)
                   for p, selected in stage_plan.items()}

    print(f"\n{'=' * 70}\nSUMMARY ({time.time() - t0:.1f}s total)\n{'=' * 70}")
    n_failed = 0
    for protein, results in all_results.items():
        for stage, (status, elapsed) in results.items():
            flag = "ok  " if status == "DONE" else "FAIL"
            print(f"  [{flag}] {protein:11s} {stage:15s} {elapsed:7.1f}s  {'' if status == 'DONE' else status}")
            n_failed += status != "DONE"

    if n_failed:
        print(f"\n{n_failed} stage(s) FAILED", file=sys.stderr)
        return 1
    print("\nAll stages completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
