"""Ensemble-shift descriptors computed from the per-structure pLDDT CSVs.

A per-residue mean collapses each conformational ensemble to one number, discarding
the distribution that motivated generating the ensemble in the first place. This
module compares the WT and mutant distributions -- from the residue x structure
pLDDT matrices -- and exposes the comparison as per-variant features:

    plddt_spread_ratio : mean-over-residues[ std_over_structures(MUT) ]
                       / mean-over-residues[ std_over_structures(WT) ]
        Confidence-based ensemble spread. >1 means the variant broadened the
        conformational-confidence cloud (a destabilization signal); ~1 means the
        variant left the ensemble's spread unchanged. This is the confidence
        analogue of RMSF -- true geometric RMSF would need 3D coordinates, which
        are not on disk for these genes (pLDDT CSVs only).

    plddt_wasserstein  : 1-Wasserstein (earth-mover) distance between the WT and
                         MUT per-residue mean-pLDDT profiles.
    (plddt_ks, the PROFILE KS between the two mean profiles, was removed on
    2026-10-13: it never sees a within-residue distribution and is of no use.)

    plddt_ks_residue   : mean over residues of the per-residue two-sample KS
        between the WT and mutant ensembles AT that residue. The RESIDUE
        estimator -- paired per residue, and it does look inside the
        distribution. This is `ks_raw` in the canonical ks_meanfree analysis.

    plddt_ks_centred   : same, after centring each ensemble by its own
        per-residue mean. Mean-free, so it measures shape/spread only.
        `ks_centred` in the canonical analysis.

Structure sampling uses structure_draws' deterministic sampler and sampling keys,
so a given (variant, size, seed) always draws the same structures. Features are keyed by (gene, mutation) to merge onto the labeled
dataset. RMSF is intentionally absent: it is not computable from pLDDT alone.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import residue_ks_statistics

from drift import structure_draws

sample_items_nested = structure_draws.sample_items_nested
select_structure_columns = structure_draws.select_structure_columns
column_confidence_scores = structure_draws.column_confidence_scores
SamplingPolicy = structure_draws.SamplingPolicy
canonical_gene = structure_draws.canonical_gene
normalize_mutation = structure_draws.normalize_mutation

try:  # scipy ships with sklearn; keep a manual fallback so this never hard-fails.
    from scipy.stats import wasserstein_distance as _scipy_w1

    def _w1(a, b):
        return float(_scipy_w1(a, b))

except Exception:  # pragma: no cover - exercised only without scipy
    def _w1(a, b):
        # 1-Wasserstein via the integral of |CDF_a - CDF_b| over a shared grid.
        a = np.sort(np.asarray(a, dtype=np.float64))
        b = np.sort(np.asarray(b, dtype=np.float64))
        grid = np.union1d(a, b)
        ca = np.searchsorted(a, grid, side="right") / a.size
        cb = np.searchsorted(b, grid, side="right") / b.size
        return float(np.sum(np.abs(ca[:-1] - cb[:-1]) * np.diff(grid)))


# profileKS ("plddt_ks": one KS between the WT and mutant MEAN profiles) was
# removed on 2026-10-13 -- of no use to the project; residueKS is the KS.
FEATURE_COLUMNS = ["plddt_spread_ratio", "plddt_wasserstein",
                   "plddt_ks_residue", "plddt_ks_centred"]


def _read_plddt_matrix(csv_path, max_structures, *, seed, key, policy=None,
                       preferred_order=None, selection_out=None, expected_residues=None):
    """Return a residues x structures pLDDT DataFrame (residue index as str).

    Mirrors structure_draws.extract_plddt_from_csv_mean's residue-index
    detection and column (structure) sampling, but keeps the full matrix instead
    of collapsing to a per-residue mean.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    structure_draws.validate_plddt_csv_frame(df, expected_residues, max_structures,
                                      context=csv_path)
    if df.empty:
        return None

    if df.shape[1] > 1:
        idx_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        int_vals = idx_raw.fillna(-1).astype(int).to_numpy()
        is_residue_index = idx_raw.notna().all() and np.array_equal(
            int_vals, np.arange(len(df))
        )
        if is_residue_index:
            idx = idx_raw.fillna(-1).astype(int).astype(str)
            value_df = df.iloc[:, 1:]
        elif idx_raw.notna().any():
            idx = pd.RangeIndex(len(df)).astype(str)
            value_df = df
        else:
            idx = df.iloc[:, 0].astype(str)
            value_df = df.iloc[:, 1:]
    else:
        idx = pd.RangeIndex(len(df)).astype(str)
        value_df = df

    value_df = value_df.apply(pd.to_numeric, errors="coerce")
    policy = SamplingPolicy.coerce(policy)
    if value_df.shape[1] > 0 and (max_structures is not None or policy.selection != "random"):
        cols = select_structure_columns(
            list(value_df.columns.astype(str)),
            None if max_structures is None else int(max_structures),
            seed=seed,
            key=key,
            policy=policy,
            scores=column_confidence_scores(value_df) if policy.uses_confidence else None,
            preferred_order=preferred_order,
        )
        value_df = value_df.loc[:, cols]
    if value_df.shape[1] == 0:
        return None
    if selection_out is not None:
        selection_out.extend(str(c) for c in value_df.columns)

    out = value_df.copy()
    out.index = list(idx)
    return out


def _spread_and_profile(mat):
    """(mean-over-residues of per-residue std across structures, per-residue mean)."""
    arr = mat.to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        per_res_std = np.nanstd(arr, axis=1)      # spread across the ensemble
        per_res_mean = np.nanmean(arr, axis=1)    # confidence profile
    spread = float(np.nanmean(per_res_std)) if per_res_std.size else np.nan
    return spread, pd.Series(per_res_mean, index=mat.index)


def _residue_ks(A, B):
    """Per-residue KS using the shared minimum-sample policy and backend."""
    return residue_ks_statistics(A, B)


def _residue_ks_summaries(wt_mat, mut_mat):
    """(residueKS, centred residueKS), each averaged over usable residues.

    residueKS keeps the mean shift; the centred variant removes each ensemble's
    own per-residue mean first, so it is mean-free and measures shape/spread
    only.

    Residues with any non-finite value are dropped rather than imputed, matching
    how the profile features already filter.
    """
    A = np.asarray(wt_mat, dtype=np.float64)
    B = np.asarray(mut_mat, dtype=np.float64)
    usable = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    # KS is defined from one structure per side; see descriptor_metrics.
    if usable.sum() < 2 or min(A.shape[1], B.shape[1]) < 1:
        return np.nan, np.nan
    A, B = A[usable], B[usable]
    raw = _residue_ks(A, B)
    centred = _residue_ks(
        A - A.mean(axis=1, keepdims=True), B - B.mean(axis=1, keepdims=True),
    )
    return float(np.nanmean(raw)), float(np.nanmean(centred))


def _wt_reference(read_root, wt_glob, mut, max_structures, seed, policy=None,
                  selection_out=None, expected_sources=None, expected_residues=None):
    """Pooled WT residues x structures matrix, sampled for one (size, seed)."""
    policy = SamplingPolicy.coerce(policy)
    wt_csvs = structure_draws.discover_wt_csv_sources(
        read_root, wt_glob, expected_sources=expected_sources,
    )
    if not wt_csvs:
        return None
    wt_csvs = sample_items_nested(
        wt_csvs, max_structures, seed=seed, key=f"wt::{mut}::csv_sources",
        policy=policy,
    )
    parts = []
    for src in wt_csvs:
        m = _read_plddt_matrix(
            src, max_structures, seed=seed, key=f"wt::{mut}::{Path(src).name}",
            policy=policy, selection_out=selection_out,
            expected_residues=expected_residues,
        )
        if m is not None and m.shape[1] > 0:
            parts.append(m)
    if not parts:
        return None
    common = parts[0].index
    for m in parts[1:]:
        common = common.intersection(m.index)
    if len(common) == 0:
        return None
    return pd.concat([m.loc[common] for m in parts], axis=1)


def _apply_row_range(frame, row_range, context):
    """Restrict a residues x structures matrix to a half-open row block.

    MSH2 and MSH6 are two datasets carved out of ONE MutS_alpha prediction set
    by residue range. The variant table honours `plddt_row_range`; these descriptors
    once did not, so every ensemble-shift statistic for both subunits was computed over the
    whole 2,294-residue complex, including the partner subunit. A mutation may
    perturb its partner, but that is outside the defined subunit descriptor.

    Filtering is by NUMERIC INDEX LABEL, matching
    structure_draws._restrict_series_to_row_range, so the variant table and the
    shift descriptors cover identical residues. Positional slicing would silently
    diverge wherever a residue row is missing.
    """
    if row_range is None:
        return frame
    try:
        start, stop = int(row_range[0]), int(row_range[1])
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise ValueError(
            f"plddt_row_range must be a two-element [start, stop); got {row_range!r}"
        ) from exc
    if stop <= start:
        raise ValueError(f"plddt_row_range {row_range!r} is empty ({context})")
    numeric = pd.to_numeric(pd.Series(frame.index, index=frame.index), errors="coerce")
    if numeric.isna().any():
        raise ValueError(
            f"plddt_row_range needs an integer residue-row index; {context} has "
            f"non-numeric labels (e.g. {frame.index[numeric.isna()][:3].tolist()})"
        )
    return frame[(numeric >= start) & (numeric < stop)]


def build_ensemble_shift_features(runtime_configs, *, max_structures_per_variant, seed,
                                  policy=None):
    """One row per (gene, mutation) with the ensemble-shift descriptors.

    `runtime_configs` should be core.STRUCTURE_CONFIGS (each protein's structure
    sources), so mut prefixes, roots, and globs match. Duplicate (gene, mutation) keys
    (e.g. MutSalpha subunits collapsing to a shared residue label) are averaged.
    """
    policy = SamplingPolicy.coerce(policy)
    rows = []
    for gene_key, cfg in runtime_configs.items():
        mut = str(cfg.get("mut", gene_key))
        gene = canonical_gene(gene_key) or canonical_gene(mut) or str(gene_key)
        read_root = Path(cfg.get("read_root", ""))
        wt_glob = str(cfg.get("wt_csv_glob", "*WT*plddt_values.csv"))
        mut_glob = str(cfg.get("mut_csv_glob", "*plddt_values.csv"))
        row_range = cfg.get("plddt_row_range")
        if not read_root.exists():
            if cfg.get("wt_csv_expected_sources") is not None:
                raise ValueError(f"{gene}: pinned WT root does not exist: {read_root}")
            continue

        wt_selected: list[str] = []
        # One WT draw per gene and subsampling seed, shared across its variants.
        wt_mat = _wt_reference(
            read_root, wt_glob, mut, max_structures_per_variant, seed,
            policy=policy, selection_out=wt_selected,
            expected_sources=cfg.get("wt_csv_expected_sources"),
            expected_residues=cfg.get("plddt_expected_residues"),
        )
        if wt_mat is None:
            print(f"[ens-shift] {gene}: no WT reference, skipping gene")
            continue
        wt_mat = _apply_row_range(wt_mat, row_range, f"{gene} WT reference")
        if row_range is not None:
            print(f"[ens-shift] {gene}: restricted to rows "
                  f"[{row_range[0]}, {row_range[1]}) -> {len(wt_mat)} residues")
        wt_spread, wt_profile = _spread_and_profile(wt_mat)

        mut_csvs = structure_draws.discover_mut_csv_sources(
            read_root, mut_glob, cfg.get("mut_csv_replacements"))
        mut_csvs = [p for p in mut_csvs if "WT" not in Path(p).name.upper()]
        # Same cross-dataset-leakage guard as build_variant_table: require the gene's
        # own mutation prefix on the filename.
        mut_csvs = [
            p for p in mut_csvs
            if re.match(rf"^{re.escape(mut)}_", Path(p).name, flags=re.IGNORECASE)
        ]

        for src in mut_csvs:
            variant_name = Path(src).name.replace("_plddt_values.csv", "")
            mutation = normalize_mutation(variant_name)
            if mutation is None:
                continue
            m = _read_plddt_matrix(
                src, max_structures_per_variant, seed=seed, key=structure_draws.source_draw_key("mut", src),
                policy=policy, preferred_order=wt_selected or None,
                expected_residues=cfg.get("plddt_expected_residues"),
            )
            if m is None:
                continue
            m = _apply_row_range(m, row_range, f"{gene} {mutation}")
            common = wt_profile.index.intersection(m.index)
            if len(common) == 0:
                continue
            mut_spread, mut_profile = _spread_and_profile(m.loc[common])
            wt_prof = wt_profile.loc[common].to_numpy(dtype=np.float64)
            mut_prof = mut_profile.to_numpy(dtype=np.float64)
            valid = np.isfinite(wt_prof) & np.isfinite(mut_prof)
            if valid.sum() < 2:
                continue
            wt_prof, mut_prof = wt_prof[valid], mut_prof[valid]

            spread_ratio = (
                mut_spread / wt_spread
                if (np.isfinite(wt_spread) and wt_spread > 0 and np.isfinite(mut_spread))
                else np.nan
            )
            # Computed from the FULL residues x structures matrices, not the
            # mean profiles the two features above use.
            ks_residue, ks_centred = _residue_ks_summaries(
                wt_mat.loc[common], m.loc[common],
            )
            rows.append({
                "gene": gene,
                "mutation": mutation,
                "plddt_spread_ratio": spread_ratio,
                "plddt_wasserstein": _w1(mut_prof, wt_prof),
                "plddt_ks_residue": ks_residue,
                "plddt_ks_centred": ks_centred,
            })

    if not rows:
        return pd.DataFrame(columns=["gene", "mutation", *FEATURE_COLUMNS])
    out = pd.DataFrame(rows)
    # Collapse any duplicate (gene, mutation) to one row, one row per variant
    # (mean is order-independent, unlike keep="first").
    out = (
        out.groupby(["gene", "mutation"], as_index=False)[FEATURE_COLUMNS]
        .mean()
        .reset_index(drop=True)
    )
    return out
