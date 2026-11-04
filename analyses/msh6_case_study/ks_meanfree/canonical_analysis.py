#!/usr/bin/env python
"""Canonical, multi-seed analysis of mean-free per-residue pLDDT marginals.

This supersedes the exploratory ``ks_meanfree.py`` without overwriting its
historical outputs.  Variant identity, labels, mutant files and WT references
come from the structure-metrics distribution summaries through
``canonical_cohort``; filenames are never parsed into mutation keys.

The primary descriptor is ``ks_centred``: each sampled ensemble is centred by
its own per-residue sample mean before the empirical KS distance is calculated.
It is therefore mean-free, but should be described as a marginal distributional
difference rather than as pure variance or physical conformational dispersion.
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


# `metric_analysis` is an EDITABLE install whose finder points at one fixed
# checkout, and sys.path[0] for a script is the script's own directory, not the
# repo root. Without this insert, running from a git worktree silently imports
# the OTHER checkout's canonical_cohort and manifest -- which produced a run
# mixing this tree's macro definitions with the other tree's cohort. Same guard
# as analyses/cohort/build_strict_canonical_manifest.py.
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from metric_analysis.pipeline.library.canonical_cohort import (
    CANONICAL_PROTEINS,
    load_canonical_cohort,
)
from metric_analysis.pipeline.paths import REPO_ROOT
from metric_analysis.pipeline.paths import tagged


# FOXA1 is a positional control, never in a macro.
PRIMARY_PROTEINS = tuple(p for p in CANONICAL_PROTEINS if p != "FOXA1")
# Macro membership is stated, not implied by a count in the name. Whether MSH6
# belongs in the headline macro is a reporting decision -- 38 variants with an
# interval spanning chance -- so both are emitted and neither is privileged.
MACRO_SETS = (
    ("macro_all_evaluation", PRIMARY_PROTEINS),
    ("macro_no_MSH6", tuple(p for p in PRIMARY_PROTEINS if p != "MSH6")),
    ("macro_no_POLE", tuple(p for p in PRIMARY_PROTEINS if p != "POLE")),
    ("macro_no_MSH6_no_POLE",
     tuple(p for p in PRIMARY_PROTEINS if p not in ("MSH6", "POLE"))),
)
# canonical_v2_strict is the SUPERSEDED merged-MutS_alpha run and must not be
# the default output any more -- a bare invocation would overwrite the
# reference run with split-cohort data.
DEFAULT_OUT = REPO_ROOT / "analyses" / "msh6_case_study" / "ks_meanfree" / "results" / tagged("canonical")


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# Compatibility exports for historical experiment imports.
from metric_analysis.pipeline.library.descriptor_metrics import (
    METRICS, REGISTRY, descriptor_values, ks_mean_equal, relative_change,
)
from metric_analysis.pipeline.library.descriptor_sampling import (
    load_matrix, row_block, apply_row_block, stable_rng, analyse_variant,
)


def _analyse_variant_job(args):
    record, sizes, seeds = args
    return analyse_variant(record, sizes=sizes, seeds=seeds)


def score_gene_seeds(per_variant: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["protein", "requested_size", "seed_index", "seed"]
    for key, block in per_variant.groupby(keys, sort=False):
        y = block["label"].to_numpy(int)
        row = dict(zip(keys, key)) | {
            "n": int(len(block)), "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
        }
        for metric in METRICS:
            score = block[metric].to_numpy(float)
            valid = np.isfinite(score)
            row[f"auc_{metric}"] = fast_auc(y[valid], score[valid])
        rows.append(row)
    return pd.DataFrame(rows)


def add_macro_rows(per_gene_seed: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [c for c in per_gene_seed if c.startswith("auc_")]
    rows = []
    available = set(per_gene_seed["protein"])
    for (size, seed_index, seed), block in per_gene_seed.groupby(
        ["requested_size", "seed_index", "seed"], sort=False
    ):
        for name, proteins in MACRO_SETS:
            if not set(proteins).issubset(available):
                continue
            selected = block[block["protein"].isin(proteins)]
            row = {
                "protein": name, "requested_size": size,
                "seed_index": seed_index, "seed": seed,
                "n": int(selected["n"].sum()),
                "n_positive": int(selected["n_positive"].sum()),
                "n_negative": int(selected["n_negative"].sum()),
            }
            row.update({c: float(selected[c].mean()) for c in metric_columns})
            rows.append(row)
    return pd.concat([per_gene_seed, pd.DataFrame(rows)], ignore_index=True, sort=False)


def summarize_seed_scores(scores: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in scores if c.startswith("auc_")]
    rows = []
    for (protein, size), block in scores.groupby(["protein", "requested_size"], sort=False):
        row = {"protein": protein, "requested_size": int(size), "n_seeds": len(block)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(block[metric].mean())
            row[f"{metric}_seed_sd"] = float(block[metric].std(ddof=1))
            row[f"{metric}_min"] = float(block[metric].min())
            row[f"{metric}_max"] = float(block[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _seed_score_matrix(block: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Labels and (seed x variant) scores in a checked, stable mutation order."""
    labels = block[["mutation_code", "label"]].drop_duplicates("mutation_code")
    labels = labels.sort_values("mutation_code")
    pivot = block.pivot(index="mutation_code", columns="seed_index", values=metric)
    pivot = pivot.reindex(labels["mutation_code"])
    if pivot.isna().any().any() and not pivot.isna().all().all():
        raise ValueError(f"missing {metric} seed score in aligned variant matrix")
    return labels["label"].to_numpy(int), pivot.to_numpy(float).T


def mean_seed_auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean([fast_auc(y, score) for score in scores]))


def fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Tie-aware Mann--Whitney AUC, equivalent to sklearn for binary labels."""
    y = np.asarray(y, dtype=int)
    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)
    if not n_positive or not n_negative:
        return float("nan")
    ranks = rankdata(np.asarray(score, dtype=float), method="average")
    return float(
        (ranks[y == 1].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def hierarchical_bootstrap_auc(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stratified variant bootstrap of the reported mean-over-seeds endpoint.

    Variants are resampled once per replicate and applied to every seed, keeping
    the paired design.  Seed columns are then resampled with replacement so the
    interval includes uncertainty in the ten-seed Monte Carlo mean as well as
    cohort uncertainty.
    """
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    n_seeds = scores.shape[0]
    out = np.empty(reps, dtype=float)
    for i in range(reps):
        index = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        seed_index = rng.choice(n_seeds, n_seeds, replace=True)
        out[i] = np.mean([fast_auc(y[index], scores[s, index]) for s in seed_index])
    return out


def bootstrap_summary(
    per_variant: pd.DataFrame,
    *,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    distributions: dict[tuple[str, int, str], np.ndarray] = {}
    available = set(per_variant["protein"])
    for (protein, size), block in per_variant.groupby(["protein", "requested_size"], sort=False):
        for metric in METRICS:
            y, scores = _seed_score_matrix(block, metric)
            observed = mean_seed_auc(y, scores)
            draws = hierarchical_bootstrap_auc(
                y, scores, reps=reps,
                rng=stable_rng(seed, protein, size, metric, "bootstrap"),
            )
            distributions[(protein, int(size), metric)] = draws
            rows.append({
                "protein": protein, "requested_size": int(size), "metric": metric,
                "n": len(y), "n_positive": int(y.sum()), "observed_auc": observed,
                "bootstrap_mean": float(draws.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "bootstrap_reps": reps,
            })

    for size in sorted(per_variant["requested_size"].unique()):
        for name, proteins in MACRO_SETS:
            if not set(proteins).issubset(available):
                continue
            for metric in METRICS:
                draws = np.mean(
                    [distributions[(p, int(size), metric)] for p in proteins], axis=0
                )
                observed = float(np.mean([
                    next(
                        row["observed_auc"] for row in rows
                        if row["protein"] == p and row["requested_size"] == int(size)
                        and row["metric"] == metric
                    )
                    for p in proteins
                ]))
                selected = per_variant[
                    per_variant["protein"].isin(proteins)
                    & per_variant["requested_size"].eq(size)
                    & per_variant["seed_index"].eq(0)
                ]
                rows.append({
                    "protein": name, "requested_size": int(size), "metric": metric,
                    "n": int(len(selected)),
                    "n_positive": int(selected["label"].sum()),
                    "observed_auc": observed, "bootstrap_mean": float(draws.mean()),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "bootstrap_reps": reps,
                })
    return pd.DataFrame(rows)


def permutation_summary(
    per_variant: pd.DataFrame,
    *,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    nulls: dict[tuple[str, int, str], np.ndarray] = {}
    observed_by_key: dict[tuple[str, int, str], float] = {}
    available = set(per_variant["protein"])
    for (protein, size), block in per_variant.groupby(["protein", "requested_size"], sort=False):
        labels = block[["mutation_code", "label"]].drop_duplicates("mutation_code")
        labels = labels.sort_values("mutation_code")
        y = labels["label"].to_numpy(int)
        rng = stable_rng(seed, protein, size, "permutation")
        permuted_y = np.vstack([rng.permutation(y) for _ in range(reps)]).astype(float)
        n_positive = int(y.sum())
        n_negative = int(len(y) - n_positive)
        for metric in METRICS:
            aligned_y, scores = _seed_score_matrix(block, metric)
            if not np.array_equal(y, aligned_y):
                raise ValueError(f"{protein} {size}: label alignment changed across metrics")
            observed = mean_seed_auc(y, scores)
            ranks = np.vstack([rankdata(score) for score in scores])
            rank_sums = permuted_y @ ranks.T
            aucs = (
                rank_sums - n_positive * (n_positive + 1) / 2.0
            ) / (n_positive * n_negative)
            null = aucs.mean(axis=1)
            nulls[(protein, int(size), metric)] = null
            observed_by_key[(protein, int(size), metric)] = observed
            rows.append({
                "protein": protein, "requested_size": int(size), "metric": metric,
                "observed_auc": observed, "null_mean": float(null.mean()),
                "null_p95": float(np.quantile(null, 0.95)),
                "p_one_sided": float((1 + np.sum(null >= observed)) / (reps + 1)) if np.isfinite(observed) else float("nan"),
                "permutation_reps": reps,
            })

    for size in sorted(per_variant["requested_size"].unique()):
        for name, proteins in MACRO_SETS:
            if not set(proteins).issubset(available):
                continue
            for metric in METRICS:
                null = np.mean([nulls[(p, int(size), metric)] for p in proteins], axis=0)
                observed = float(np.mean([
                    observed_by_key[(p, int(size), metric)] for p in proteins
                ]))
                rows.append({
                    "protein": name, "requested_size": int(size), "metric": metric,
                    "observed_auc": observed, "null_mean": float(null.mean()),
                    "null_p95": float(np.quantile(null, 0.95)),
                    "p_one_sided": float((1 + np.sum(null >= observed)) / (reps + 1)) if np.isfinite(observed) else float("nan"),
                    "permutation_reps": reps,
                })
    return pd.DataFrame(rows)


def wt_split_null(
    cohort: pd.DataFrame,
    *,
    sizes: tuple[int, ...],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """Exact disjoint WT-vs-WT baseline where the finite pool permits it."""
    rows = []
    for protein, block in cohort.groupby("protein", sort=False):
        wt_paths = block["wt_source_file"].drop_duplicates().tolist()
        if len(wt_paths) != 1:
            raise ValueError(f"{protein}: canonical cohort has {len(wt_paths)} WT references")
        blocks = {row_block(r) for r in block.to_dict("records")}
        if len(blocks) != 1:
            raise ValueError(f"{protein}: mixed pLDDT row blocks {blocks}")
        wt = apply_row_block(load_matrix(str(wt_paths[0])), blocks.pop(),
                             context=f"{protein} WT split null")
        for size in sizes:
            if 2 * size > wt.shape[1]:
                continue
            for replicate in range(reps):
                order = stable_rng(seed, protein, size, replicate, "wt_split").permutation(wt.shape[1])
                values = descriptor_values(wt[:, order[:size]], wt[:, order[size:2 * size]])
                rows.append({
                    "protein": protein, "requested_size": int(size),
                    "replicate": replicate, "wt_pool_size": int(wt.shape[1]), **values,
                })
    return pd.DataFrame(rows)


def compare_observed_to_wt_null(
    per_variant: pd.DataFrame,
    wt_null: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (protein, size), null_block in wt_null.groupby(["protein", "requested_size"], sort=False):
        observed_block = per_variant[
            (per_variant["protein"] == protein)
            & (per_variant["requested_size"] == size)
        ]
        for metric in METRICS:
            null = null_block[metric].to_numpy(float)
            threshold = float(np.quantile(null, 0.95))
            for seed_index, observed_seed in observed_block.groupby("seed_index", sort=False):
                for variant_group, label in (("all", None), ("driver", 1), ("passenger", 0)):
                    selected = observed_seed if label is None else observed_seed[observed_seed["label"] == label]
                    observed = selected[metric].to_numpy(float)
                    y = np.r_[np.ones(len(observed)), np.zeros(len(null))]
                    score = np.r_[observed, null]
                    rows.append({
                        "protein": protein, "requested_size": int(size),
                        "seed_index": int(seed_index), "metric": metric,
                        "variant_group": variant_group, "n_variants": len(observed),
                        "null_median": float(np.median(null)), "null_p95": threshold,
                        "observed_median": float(np.median(observed)),
                        "fraction_observed_above_null_p95": float(np.mean(observed > threshold)),
                        "auc_real_vs_wt_null": fast_auc(y, score),
                    })
    return pd.DataFrame(rows)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not parsed or min(parsed) < 2:
        raise argparse.ArgumentTypeError("provide comma-separated ensemble sizes >=2")
    return parsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proteins", default=",".join(CANONICAL_PROTEINS))
    parser.add_argument("--sizes", type=parse_csv_ints, default=parse_csv_ints("10,20,40,50,60,90"))
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260829)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--permutation-reps", type=int, default=5000)
    parser.add_argument("--wt-null-reps", type=int, default=200)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--postprocess-only", action="store_true",
        help="reuse an existing per_variant_seed.csv.gz and rebuild uncertainty/null tables",
    )
    args = parser.parse_args(argv)

    proteins = tuple(p for p in args.proteins.split(",") if p)
    unknown = sorted(set(proteins) - set(CANONICAL_PROTEINS))
    if unknown:
        parser.error(f"unknown proteins: {unknown}")
    if args.n_seeds < 2:
        parser.error("--n-seeds must be at least 2")
    if (args.out.exists() and any(args.out.iterdir()) and not args.force
            and not args.postprocess_only):
        parser.error(f"refusing to overwrite non-empty output directory {args.out}; use --force")
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = tuple(args.base_seed + i for i in range(args.n_seeds))
    cohort = pd.concat([load_canonical_cohort(p) for p in proteins], ignore_index=True)
    cohort.to_csv(args.out / "canonical_cohort_manifest.csv", index=False)
    _log(f"canonical cohort: {len(cohort)} variants across {len(proteins)} proteins")

    per_variant_path = args.out / "per_variant_seed.csv.gz"
    if args.postprocess_only:
        if not per_variant_path.is_file():
            parser.error(f"--postprocess-only requires {per_variant_path}")
        _log(f"reusing {per_variant_path.name}")
        per_variant = pd.read_csv(per_variant_path)
        recorded_sizes = tuple(sorted(per_variant["requested_size"].unique()))
        recorded_seeds = tuple(sorted(per_variant["seed"].unique()))
        if recorded_sizes != args.sizes or recorded_seeds != seeds:
            parser.error(
                "requested sizes/seeds do not match saved per-variant data: "
                f"saved sizes={recorded_sizes}, seeds={recorded_seeds}"
            )
        strict_keys = cohort[["protein", "mutation_code"]].drop_duplicates()
        before = len(per_variant)
        per_variant = per_variant.merge(
            strict_keys, on=["protein", "mutation_code"], how="inner",
            validate="many_to_one",
        )
        _log(f"strict-manifest filter: {before:,} -> {len(per_variant):,} descriptor rows")
        per_variant.to_csv(per_variant_path, index=False, compression="gzip")
    else:
        jobs = [
            (record, args.sizes, seeds)
            for record in cohort.to_dict("records")
        ]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, result in enumerate(pool.map(_analyse_variant_job, jobs, chunksize=2), start=1):
                rows.extend(result)
                if i % 100 == 0 or i == len(jobs):
                    _log(f"descriptors: {i}/{len(jobs)} variants")
        per_variant = pd.DataFrame(rows)
        per_variant.to_csv(per_variant_path, index=False, compression="gzip")

    gene_seed = score_gene_seeds(per_variant)
    all_seed_scores = add_macro_rows(gene_seed)
    all_seed_scores.to_csv(args.out / "per_gene_seed_auc.csv", index=False)
    summarize_seed_scores(all_seed_scores).to_csv(args.out / "seed_summary.csv", index=False)

    _log("variant-stratified bootstrap")
    bootstrap_summary(
        per_variant, reps=args.bootstrap_reps, seed=args.base_seed
    ).to_csv(args.out / "bootstrap_summary.csv", index=False)
    _log("within-protein label permutations")
    permutation_summary(
        per_variant, reps=args.permutation_reps, seed=args.base_seed
    ).to_csv(args.out / "permutation_summary.csv", index=False)

    _log("disjoint WT-vs-WT nulls")
    null = wt_split_null(
        cohort, sizes=args.sizes, reps=args.wt_null_reps, seed=args.base_seed
    )
    null.to_csv(args.out / "wt_split_null.csv.gz", index=False, compression="gzip")
    comparison = compare_observed_to_wt_null(per_variant, null)
    comparison.to_csv(args.out / "observed_vs_wt_null_by_class.csv", index=False)
    comparison[comparison["variant_group"] == "all"].drop(
        columns="variant_group"
    ).to_csv(args.out / "observed_vs_wt_null.csv", index=False)

    metadata = {
        "proteins": proteins, "primary_macro_proteins": PRIMARY_PROTEINS,
        "sizes": args.sizes, "seeds": seeds, "workers": args.workers,
        "bootstrap_reps": args.bootstrap_reps,
        "permutation_reps": args.permutation_reps,
        "wt_null_reps": args.wt_null_reps,
        "sampling": "nested within seed; WT and mutant permutations independent; without replacement",
        "cohort_source": "metric_analysis/*_plddt_distribution_summary.csv",
        "descriptor_direction": "signed, predeclared driver-high",
    }
    (args.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    _log(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
