#!/usr/bin/env python3
"""
Standalone parallel RMSF/RMSD/positional-KS ensemble-size sweep — reads the already-built aligned
coordinate cache and produces the same raw/performance/summary CSVs the notebook's
run_rmsf_ensemble_sweep / run_rmsd_ensemble_sweep cells produce, but faster in two
independent ways:

1. Algorithmic fix: the WT reference ensemble's stats depend only on
   (ensemble_size, repeat), not on which mutant artifact is being compared — they're
   computed once per repeat here (matching the notebook's now-fixed sweep functions),
   instead of once per (artifact, repeat) pair. For FOXA1 (822 artifacts) that alone
   removes ~820x redundant work.
2. Process-level parallelism, split by artifact across chunks — CPU-bound pandas/numpy
   work doesn't benefit from threads (GIL), so this uses separate processes instead.

Memory safety: naively loading the full aligned coordinate cache costs ~13.7GB in a
live DataFrame for FOXA1 (61.6M rows; `artifact`/`chain`/`icode` stored as raw
undeduplicated Python string objects) even after column-pruning — running N parallel
workers that each load the whole thing would repeat the exact OOM crisis the DNA/RMSF
cache-repair scripts hit earlier. So this script never lets more than one process hold
the full cache: a one-time `--prepare` step loads it once, casts the string columns to
`category` dtype (13.7GB -> ~2.7GB), splits by artifact into N chunk parquet files
(each chunk's file also includes the WT rows, needed as the shared reference), and
only then do the N chunk workers run — each reading just its own small pre-split file.

Usage
-----
# One-time prep (single process, run this first — and after any coordinate-cache rebuild):
    python -m metric_analysis.pipeline.orchestration.run_ensemble_sweep_parallel --protein FOXA1 --prepare --total-chunks 4

# Run the 4 chunks (can run concurrently):
    python -m metric_analysis.pipeline.orchestration.run_ensemble_sweep_parallel --protein FOXA1 --chunk 0 --total-chunks 4
    python -m metric_analysis.pipeline.orchestration.run_ensemble_sweep_parallel --protein FOXA1 --chunk 1 --total-chunks 4
    python -m metric_analysis.pipeline.orchestration.run_ensemble_sweep_parallel --protein FOXA1 --chunk 2 --total-chunks 4
    python -m metric_analysis.pipeline.orchestration.run_ensemble_sweep_parallel --protein FOXA1 --chunk 3 --total-chunks 4

# Merge (computes performance/summary stats on the *combined* raw data — these need
# every artifact together, not per-chunk subsets, so this step is not parallelized):
    python -m metric_analysis.pipeline.orchestration.run_ensemble_sweep_parallel --protein FOXA1 --merge --total-chunks 4

# Dry-run on a handful of artifacts (correctness check before a full run):
    python -m metric_analysis.pipeline.orchestration.run_ensemble_sweep_parallel --protein FOXA1 --chunk 0 --total-chunks 4 --max-artifacts 5

# Supported proteins: FOXA1, MutS_alpha, KRAS, POLE, TP53
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
except ImportError:
    raise RuntimeError("scipy is required: pip install scipy")

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

from metric_analysis.pipeline.config.protein_config import get_protein_config, supported_proteins
from metric_analysis.pipeline.library.positional_ks import positional_ks, reference_centre
from metric_analysis.pipeline.labels import (
    artifact_key,
    is_wt_artifact,
    extract_mutation_code,
    artifact_to_allele,
    load_clinvar_driver_passenger,
    extract_gene,
    load_csv_label_lookup,
    classify_allele_or_artifact,
)

# ── Per-protein configuration (metric_analysis/protein_configs.json) ───────────
# label_mode-specific fields (hotspot_range / clinvar_csv / clinvar_csv_by_gene /
# labels_csv) aren't in REQUIRED_CONFIG_KEYS since each protein
# only has the ones matching its own label_mode (see labels.classify_allele_or_artifact).

REQUIRED_CONFIG_KEYS = ["cache_dir", "sweep_dir", "ensemble_sizes", "ensemble_repeats_by_size", "label_mode"]

ENSEMBLE_RANDOM_SEED = 42

COORD_COLUMNS = ["artifact", "structure_index", "chain", "resid", "icode", "x_aligned", "y_aligned", "z_aligned"]

# is_wt_artifact, extract_mutation_code, artifact_to_allele, load_clinvar_driver_passenger,
# extract_gene, load_csv_label_lookup, classify_allele_or_artifact
# now live in labels.py (shared with run_structure_metrics.py) — imported above.

_SEED_SAMPLE_RE = re.compile(r"seed-\d+_sample-\d+")


def coord_cache_path(cfg):
    """The aligned C-alpha cache: ``coord_cache_dir`` (another frame) or ``cache_dir``."""
    d = Path(cfg.get("coord_cache_dir") or cfg["cache_dir"])
    prefix = cfg.get("prefix") or cfg["output_prefix"]
    return d / f"{prefix}_rmsf_aligned_ca_coordinate_cache.parquet"


def is_core_fit_cache(cfg):
    """True for a cache written by cache_builders/subunit_core_fit, in either frame (its
    report sits next to it). Those hold exactly the pLDDT samples, so there is no AF3
    top-level model left to drop by structure index."""
    p = coord_cache_path(cfg)
    prefix = cfg.get("prefix") or cfg["output_prefix"]
    return any((p.parent / f"{prefix}_{r}.json").exists() for r in ("core_fit_report", "usalign_frame_report"))


def _chunk_source_stamp(cfg):
    p = coord_cache_path(cfg)
    st = p.stat()
    return {"cache": str(p.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def stale_coord_chunks_reason(cfg, chunk_dir):
    """Why the chunks in chunk_dir are not copies of the current cache, or None if they are."""
    stamp_path = Path(chunk_dir) / "_source.json"
    if not stamp_path.exists():
        return "no _source.json (built before chunks recorded their source)"
    old = json.loads(stamp_path.read_text())
    new = _chunk_source_stamp(cfg)
    return None if old == new else f"built from {old.get('cache')} ({old.get('size')} B), cache is now {new}"


def drop_top_ranked_duplicate_rows(cfg, df, tag=""):
    """Drop AF3's duplicated top-ranked model from a coordinate frame.

    AF3 writes its best prediction twice: as `{name}_model.cif` at the top of the
    archive and as the seed-sample it came from. Verified from the raw tar on
    MutS_alpha A272V -- the 125 seed-samples have 125 distinct CA coordinate sets and
    the top-level file matches seed-1_sample-1 exactly. The coordinate cache was built
    by globbing every .cif, so it carries 126 structures per artifact with one counted
    twice, and a different one per artifact (whichever ranked best).

    This matters more here than anywhere else: the sweep draws N structures without
    replacement to measure how discrimination grows with ensemble size, and a
    duplicated member means a draw can contain the same conformation twice. That
    understates ensemble diversity, which is the quantity being measured.

    The frames carry only (artifact, structure_index), so duplicates are identified
    from the structure file index and matched by that pair. Missing index -> unchanged,
    so a cache without its index still sweeps rather than failing.
    """
    if df is None or df.empty or "structure_index" not in df.columns:
        return df
    if is_core_fit_cache(cfg):
        return df   # built from the pLDDT samples only: no top-level model to drop
    index_path = Path(cfg["cache_dir"]) / f"{cfg['prefix']}_structure_file_index.csv"
    if not index_path.exists():
        print(f"{tag} no structure file index — keeping AF3's duplicated top-ranked model", flush=True)
        return df

    idx = pd.read_csv(index_path)
    path_cols = [c for c in ("structure_member", "structure_file", "structure_path") if c in idx.columns]
    if not path_cols or "artifact" not in idx.columns:
        return df
    # Match the whole path, not the basename: POLE's truncated 268-471 construct names
    # every member `model.cif` and carries the seed only in its directory. An artifact
    # with no seed-named members keeps everything (one TP53 artifact has no top-level).
    joined = idx[path_cols].astype(str).agg(" ".join, axis=1)
    is_seed = joined.str.contains(_SEED_SAMPLE_RE)
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
    print(f"{tag} dropped {len(df) - len(out):,} duplicated top-ranked rows", flush=True)
    return out


def resolve_wt_artifact(cfg, present):
    """Choose the WT reference explicitly instead of taking whichever came first.

    This was `[a for a in artifacts if is_wt_artifact(a)][0]`. With one WT that is
    unambiguous; POLE has three across two constructs and TP53 has two, so the
    reference every mutant is measured against was decided by row order. Worse here
    than elsewhere: each chunk resolved independently, so different chunks could in
    principle align against different wild types.

    Returns None when the configured artifact is not in this frame, which is the
    honest answer for a chunk that does not carry it.
    """
    names = [str(a) for a in present]
    configured = cfg.get("wt_artifact")
    if configured:
        # Match on artifact_key, which ignores the archive extension and a trailing
        # _data: configs, caches and archives disagree on it (PTEN config PTEN_WT vs
        # cache PTEN_WT_data; TP53 config ..._data vs cache without). A miss here
        # silently drops every WT-relative metric, so it must not depend on suffixes.
        want = artifact_key(configured)
        hits = [n for n in names if artifact_key(n) == want]
        if len(hits) > 1:
            raise RuntimeError(f"[{cfg['prefix']}] wt_artifact {configured!r} matches {hits}")
        return hits[0] if hits else None
    wt = [a for a in names if is_wt_artifact(a)]
    if len(wt) > 1:
        raise RuntimeError(
            f"[{cfg['prefix']}] {len(wt)} WT artifacts present and none chosen: {sorted(wt)}. "
            f"Set \"wt_artifact\" in the protein config."
        )
    return wt[0] if wt else None


def get_repeats_for_n(n, repeats_by_size):
    return int(repeats_by_size.get(int(n), 1))


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}  shape={df.shape}")


def compare_driver_passenger(df, score_col, class_col="label_class"):
    tmp = df[[score_col, class_col]].copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
    tmp = tmp.dropna(subset=[score_col])
    drivers = tmp.loc[tmp[class_col].astype(str).eq("Driver"), score_col].astype(float).to_numpy()
    passengers = tmp.loc[tmp[class_col].astype(str).eq("Passenger"), score_col].astype(float).to_numpy()

    out = {
        "metric_col": score_col,
        "n_driver": int(len(drivers)),
        "n_passenger": int(len(passengers)),
        "n_UNK_excluded": int((tmp[class_col].astype(str) == "UNK").sum()),
        "n_WT_excluded": int((tmp[class_col].astype(str) == "WT").sum()),
        "driver_mean": float(np.nanmean(drivers)) if len(drivers) else np.nan,
        "passenger_mean": float(np.nanmean(passengers)) if len(passengers) else np.nan,
        # Always present (even when uncomputable) so every row has the same columns —
        # otherwise a protein/metric with too few labeled samples in every group (e.g.
        # KRAS, which has ~1 usable ClinVar passenger allele) drops these columns from
        # the whole DataFrame and crashes merge()'s .agg() over them.
        "mannwhitney_U": np.nan,
        "mannwhitney_p": np.nan,
        "ks_stat": np.nan,
        "ks_p": np.nan,
        "wasserstein": np.nan,
        "cliffs_delta_driver_minus_passenger": np.nan,
        "roc_auc_driver_high": np.nan,
        "roc_auc_best_direction": np.nan,
    }

    if len(drivers) >= 2 and len(passengers) >= 2:
        try:
            mw = mannwhitneyu(drivers, passengers, alternative="two-sided")
            out["mannwhitney_U"] = float(mw.statistic)
            out["mannwhitney_p"] = float(mw.pvalue)
        except Exception:
            out["mannwhitney_U"] = np.nan
            out["mannwhitney_p"] = np.nan
        try:
            ks = ks_2samp(drivers, passengers)
            out["ks_stat"] = float(ks.statistic)
            out["ks_p"] = float(ks.pvalue)
        except Exception:
            out["ks_stat"] = np.nan
            out["ks_p"] = np.nan
        try:
            out["wasserstein"] = float(wasserstein_distance(drivers, passengers))
        except Exception:
            out["wasserstein"] = np.nan
        try:
            n_greater = sum(d > p for d in drivers for p in passengers)
            n_less = sum(d < p for d in drivers for p in passengers)
            out["cliffs_delta_driver_minus_passenger"] = float((n_greater - n_less) / (len(drivers) * len(passengers)))
        except Exception:
            out["cliffs_delta_driver_minus_passenger"] = np.nan
        if roc_auc_score is not None:
            y = np.r_[np.ones(len(drivers)), np.zeros(len(passengers))]
            scores = np.r_[drivers, passengers]
            try:
                auc = float(roc_auc_score(y, scores))
                out["roc_auc_driver_high"] = auc
                out["roc_auc_best_direction"] = max(auc, 1.0 - auc)
            except Exception:
                out["roc_auc_driver_high"] = np.nan
                out["roc_auc_best_direction"] = np.nan
    return out


# ── RMSF ────────────────────────────────────────────────────────────────────────

def compute_rmsf_summary_from_coords(coord_subset):
    if coord_subset is None or coord_subset.empty:
        return None
    vals = []
    for (chain, resid, icode), g in coord_subset.groupby(["chain", "resid", "icode"], sort=False, observed=True):
        xyz = g[["x_aligned", "y_aligned", "z_aligned"]].to_numpy(float)
        if xyz.shape[0] == 0:
            continue
        center = np.nanmean(xyz, axis=0)
        rmsf = np.sqrt(np.nanmean(np.sum((xyz - center) ** 2, axis=1)))
        vals.append(float(rmsf))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return {
        "rmsf_mean": float(np.nanmean(vals)),
        "rmsf_median": float(np.nanmedian(vals)),
        "rmsf_max": float(np.nanmax(vals)),
        "rmsf_std": float(np.nanstd(vals)),
        "n_residues_used": int(vals.size),
    }


# ── RMSD ────────────────────────────────────────────────────────────────────────

def _mean_coords_by_residue(coord_subset):
    if coord_subset is None or coord_subset.empty:
        return pd.DataFrame(columns=["chain", "resid", "icode", "x_mean", "y_mean", "z_mean"])
    out = (
        coord_subset
        .groupby(["chain", "resid", "icode"], as_index=False, sort=False, observed=True)
        .agg(x_mean=("x_aligned", "mean"), y_mean=("y_aligned", "mean"), z_mean=("z_aligned", "mean"))
    )
    return out


def _rmsd_between_mean_coords(mut_mean_df, wt_mean_df):
    if mut_mean_df is None or wt_mean_df is None or mut_mean_df.empty or wt_mean_df.empty:
        return None
    merged = mut_mean_df.merge(wt_mean_df, on=["chain", "resid", "icode"], suffixes=("_mut", "_wt"), how="inner")
    if merged.empty:
        return None
    diff = merged[["x_mean_mut", "y_mean_mut", "z_mean_mut"]].to_numpy(float) - merged[["x_mean_wt", "y_mean_wt", "z_mean_wt"]].to_numpy(float)
    per_res_sq = np.sum(diff ** 2, axis=1)
    per_res_sq = per_res_sq[np.isfinite(per_res_sq)]
    if per_res_sq.size == 0:
        return None
    return float(np.sqrt(np.nanmean(per_res_sq))), int(per_res_sq.size)


def _per_structure_rmsd_to_wt_mean(mut_subset, wt_mean_df):
    """Per-structure RMSD against the WT ensemble mean, vectorised.

    Previously this looped over each selected structure and, per structure, ran a
    groupby (via _mean_coords_by_residue) plus a merge. Two reasons that was the
    bottleneck of the whole sweep:

      * The groupby was a no-op. The coordinate cache holds exactly one row per
        (artifact, structure_index, chain, resid, icode) -- verified zero duplicates
        over 11.3M rows -- so averaging a single structure's residues returns them
        unchanged. It only paid pandas overhead.
      * It repeated that work ~39x per structure. A chunk selects
        sum(N * repeats) = 3900 structure-slots per artifact from just 101 unique
        structures, and every selection re-derived the same values.

    One merge plus one groupby over the whole subset gives identical results:
    grouping the squared per-residue distances by structure_index and taking the mean
    is the same arithmetic as computing each structure's RMSD separately. With zero
    duplicate residues the two are equal exactly, not approximately.

    Measured effect: TP53's N=1 unit alone had run 26+ minutes and was ~1.3% of a
    chunk, projecting to ~34 h per chunk. This is what made that tractable.
    """
    if mut_subset is None or mut_subset.empty or wt_mean_df is None or wt_mean_df.empty:
        return np.asarray([], dtype=float), 0

    merged = mut_subset.merge(wt_mean_df, on=["chain", "resid", "icode"], how="inner")
    if merged.empty:
        return np.asarray([], dtype=float), 0

    d2 = (
        (merged["x_aligned"].to_numpy(float) - merged["x_mean"].to_numpy(float)) ** 2
        + (merged["y_aligned"].to_numpy(float) - merged["y_mean"].to_numpy(float)) ** 2
        + (merged["z_aligned"].to_numpy(float) - merged["z_mean"].to_numpy(float)) ** 2
    )
    finite = np.isfinite(d2)
    if not finite.any():
        return np.asarray([], dtype=float), 0

    per_structure = (
        pd.DataFrame({"structure_index": merged["structure_index"].to_numpy()[finite], "d2": d2[finite]})
        .groupby("structure_index", sort=False, observed=True)["d2"]
        .agg(["mean", "size"])
    )
    vals = np.sqrt(per_structure["mean"].to_numpy(float))
    vals = vals[np.isfinite(vals)]
    n_common_max = int(per_structure["size"].max()) if len(per_structure) else 0
    return vals, n_common_max


def compute_rmsd_summary_vs_wt(mut_subset, wt_mean):
    if mut_subset is None or mut_subset.empty or wt_mean is None or wt_mean.empty:
        return None
    mut_mean = _mean_coords_by_residue(mut_subset)
    ensemble_pair = _rmsd_between_mean_coords(mut_mean, wt_mean)
    if ensemble_pair is None:
        return None
    ensemble_rmsd, n_common_residues = ensemble_pair

    per_structure_vals, n_common_structure = _per_structure_rmsd_to_wt_mean(mut_subset, wt_mean)
    if per_structure_vals.size == 0:
        return None

    return {
        "rmsd_ensemble_mean_to_wt_mean": float(ensemble_rmsd),
        "rmsd_mean_structure_to_wt_mean": float(np.nanmean(per_structure_vals)),
        "rmsd_median_structure_to_wt_mean": float(np.nanmedian(per_structure_vals)),
        "rmsd_max_structure_to_wt_mean": float(np.nanmax(per_structure_vals)),
        "rmsd_std_structure_to_wt_mean": float(np.nanstd(per_structure_vals)),
        "n_rmsd_structures_used": int(per_structure_vals.size),
        "n_rmsd_common_residues": int(max(n_common_residues, n_common_structure)),
    }


# ── Prepare (one-time, single process): categorize + split by artifact ─────────

def prepare(protein, total_chunks, force=False):
    cfg = get_protein_config(protein, REQUIRED_CONFIG_KEYS)
    chunk_dir = cfg["sweep_dir"] / "coord_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    existing = [chunk_dir / f"{cfg['prefix']}_coord_chunk_{c}.parquet" for c in range(total_chunks)]
    if not force and all(p.exists() for p in existing) and not stale_coord_chunks_reason(cfg, chunk_dir):
        print(f"[{protein}] All {total_chunks} coord chunks already exist — skipping prepare. Use --force to rebuild.")
        return

    coord_cache_parquet = coord_cache_path(cfg)
    print(f"[{protein}] Loading full coordinate cache: {coord_cache_parquet}")
    df = pd.read_parquet(coord_cache_parquet, columns=COORD_COLUMNS)
    print(f"  Loaded {len(df):,} rows")
    df = drop_top_ranked_duplicate_rows(cfg, df, tag=f"[{protein}]")
    for col in ["artifact", "chain", "icode"]:
        df[col] = df[col].astype("category")
    print(f"  Memory after categorize: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    artifacts = sorted(df["artifact"].cat.categories.astype(str))
    wt_artifact = resolve_wt_artifact(cfg, artifacts)
    non_wt = [a for a in artifacts if a != wt_artifact]
    print(f"  {len(artifacts)} artifacts total ({len(non_wt)} non-WT), WT={wt_artifact}")

    for c in range(total_chunks):
        my_artifacts = non_wt[c::total_chunks]
        keep = set(my_artifacts)
        if wt_artifact is not None:
            keep.add(wt_artifact)
        sub = df[df["artifact"].isin(keep)].copy()
        for col in ["artifact", "chain", "icode"]:
            sub[col] = sub[col].astype(str).astype("category")
        out_path = chunk_dir / f"{cfg['prefix']}_coord_chunk_{c}.parquet"
        sub.to_parquet(out_path, index=False)
        print(f"  Chunk {c}: {len(my_artifacts)} artifacts (+WT) -> {len(sub):,} rows -> {out_path.name}")
    del df
    (chunk_dir / "_source.json").write_text(json.dumps(_chunk_source_stamp(cfg)))
    print(f"[{protein}] Prepare complete.")


# ── Chunk sweep (parallel workers) ──────────────────────────────────────────────

def run_chunk_sweep(protein, chunk, total_chunks, max_artifacts=None, force=False):
    cfg = get_protein_config(protein, REQUIRED_CONFIG_KEYS)
    tag = f"[{protein} chunk {chunk}/{total_chunks - 1}]"
    chunk_path = cfg["sweep_dir"] / "coord_chunks" / f"{cfg['prefix']}_coord_chunk_{chunk}.parquet"
    if not chunk_path.exists():
        raise FileNotFoundError(f"{chunk_path} missing — run --prepare first")

    out_dir = cfg["sweep_dir"] / "raw_chunks"
    rmsf_out_path = out_dir / f"{cfg['prefix']}_rmsf_raw_chunk_{chunk}.parquet"
    rmsd_out_path = out_dir / f"{cfg['prefix']}_rmsd_raw_chunk_{chunk}.parquet"
    posks_out_path = out_dir / f"{cfg['prefix']}_posks_raw_chunk_{chunk}.parquet"
    if (not force and max_artifacts is None and rmsf_out_path.exists() and rmsd_out_path.exists()
            and posks_out_path.exists()):
        print(f"{tag} Outputs already exist ({rmsf_out_path.name}, {rmsd_out_path.name}, "
              f"{posks_out_path.name}) — skipping. Use --force to recompute.", flush=True)
        return

    df = pd.read_parquet(chunk_path)
    print(f"{tag} Loaded {len(df):,} rows, {df['artifact'].nunique()} artifacts", flush=True)
    # Chunks built before the duplicate was found still contain it, and they are reused
    # even under --force (rebuilding means re-loading a multi-GB cache), so the filter
    # has to be applied on read rather than only at prepare time.
    df = drop_top_ranked_duplicate_rows(cfg, df, tag=tag)

    grouped = {artifact: g.copy() for artifact, g in df.groupby("artifact", sort=False, observed=True)}
    del df
    wt_artifact = resolve_wt_artifact(cfg, list(grouped))
    wt_g = grouped.get(wt_artifact) if wt_artifact is not None else None
    available = {a: np.array(sorted(g["structure_index"].dropna().unique()), dtype=int) for a, g in grouped.items()}
    wt_available = available.get(wt_artifact, np.asarray([], dtype=int)) if wt_artifact is not None else np.asarray([], dtype=int)

    mutant_artifacts = [a for a in grouped if not is_wt_artifact(a)]
    if max_artifacts is not None:
        mutant_artifacts = mutant_artifacts[:max_artifacts]
        print(f"{tag} Dry-run: capped at {max_artifacts} artifacts", flush=True)

    # Same seed in every chunk process is intentional: the WT precompute block below
    # runs identically (same n, same reps, same rng calls) before any chunk-specific
    # mutant sampling, so every chunk's WT reference for a given (n, repeat) is
    # identical — required for mutant results across different chunks to be
    # comparable against the same reference.
    rng = np.random.default_rng(ENSEMBLE_RANDOM_SEED)

    rmsf_rows, rmsd_rows, posks_rows = [], [], []

    # Positional KS (library/positional_ks.py): per-residue KS on C-alpha distances to
    # the WT centre, and on each axis. The centre is the mean over the WHOLE WT pool,
    # so it is fixed across sizes and repeats. Scope follows protein_chains; a
    # multi-chain parent (MutS_alpha) also gets one column per chain, so MSH2 is read
    # on chain A and MSH6 on chain B, matching the pLDDT scope.
    chains = [str(c) for c in (cfg.get("protein_chains") or [])] or None
    posks_scopes = {"": chains}
    if chains and len(chains) > 1:
        posks_scopes.update({f"_chain_{c}": [c] for c in chains})
    posks_centres = ({sfx: reference_centre(wt_g, ch) for sfx, ch in posks_scopes.items()}
                     if wt_g is not None else {})

    for n in cfg["ensemble_sizes"]:
        reps = get_repeats_for_n(n, cfg["ensemble_repeats_by_size"])
        print(f"{tag} N={n}, repeats={reps}", flush=True)

        wt_vals_by_repeat, wt_mean_by_repeat, wt_subset_by_repeat = {}, {}, {}
        if wt_g is not None and len(wt_available) >= n:
            for r in range(reps):
                wt_sel = rng.choice(wt_available, size=n, replace=False)
                wt_subset = wt_g[wt_g["structure_index"].isin(wt_sel)]
                wt_vals_by_repeat[r] = compute_rmsf_summary_from_coords(wt_subset)
                wt_mean_by_repeat[r] = _mean_coords_by_residue(wt_subset)
                wt_subset_by_repeat[r] = wt_subset

        for artifact in mutant_artifacts:
            g = grouped[artifact]
            av = available[artifact]
            if len(av) < n:
                continue
            allele = artifact_to_allele(artifact)
            label_class = classify_allele_or_artifact(artifact, cfg)
            for r in range(reps):
                sel = rng.choice(av, size=n, replace=False)
                subset = g[g["structure_index"].isin(sel)]

                rmsf_vals = compute_rmsf_summary_from_coords(subset)
                if rmsf_vals is not None:
                    row = {
                        "metric_family": "RMSF", "artifact": artifact, "Allele": allele,
                        "label_class": label_class, "ensemble_size": int(n), "repeat": int(r),
                        "n_structures_available": int(len(av)), **rmsf_vals,
                    }
                    wt_vals = wt_vals_by_repeat.get(r)
                    if wt_vals is not None:
                        for k, v in wt_vals.items():
                            if k.startswith("rmsf_"):
                                row[f"delta_{k}"] = row[k] - v
                    rmsf_rows.append(row)

                wt_mean = wt_mean_by_repeat.get(r)
                rmsd_vals = compute_rmsd_summary_vs_wt(subset, wt_mean) if wt_mean is not None else None
                if rmsd_vals is not None:
                    rmsd_rows.append({
                        "metric_family": "RMSD", "artifact": artifact, "Allele": allele,
                        "label_class": label_class, "ensemble_size": int(n), "repeat": int(r),
                        "n_structures_available": int(len(av)),
                        "n_wt_structures_available": int(len(wt_available)),
                        "n_structures_used": int(n),
                        **rmsd_vals,
                    })

                wt_subset = wt_subset_by_repeat.get(r)
                if wt_subset is not None and posks_centres:
                    row = {
                        "metric_family": "POSKS", "artifact": artifact, "Allele": allele,
                        "label_class": label_class, "ensemble_size": int(n), "repeat": int(r),
                        "n_structures_available": int(len(av)),
                    }
                    for sfx, ch in posks_scopes.items():
                        # distance only: the per-axis variant was dropped (2026-10-16)
                        sc = positional_ks(wt_subset, subset, posks_centres[sfx], ch, modes=("distance",))
                        row[f"posks_distance{sfx}"] = sc["distance"]
                        row[f"n_posks_residues{sfx}"] = sc["n_residues"]
                    posks_rows.append(row)

    rmsf_df = pd.DataFrame(rmsf_rows)
    rmsd_df = pd.DataFrame(rmsd_rows)
    out_dir = cfg["sweep_dir"] / "raw_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    rmsf_df.to_parquet(out_dir / f"{cfg['prefix']}_rmsf_raw_chunk_{chunk}.parquet", index=False)
    rmsd_df.to_parquet(out_dir / f"{cfg['prefix']}_rmsd_raw_chunk_{chunk}.parquet", index=False)
    posks_df = pd.DataFrame(posks_rows)
    posks_df.to_parquet(posks_out_path, index=False)
    print(f"{tag} Done. RMSF rows={len(rmsf_df):,}  RMSD rows={len(rmsd_df):,}  "
          f"POSKS rows={len(posks_df):,}", flush=True)


# ── Merge (single process, once): combine raw + compute performance/summary ────

def merge(protein, total_chunks, force=False):
    cfg = get_protein_config(protein, REQUIRED_CONFIG_KEYS)
    raw_dir = cfg["sweep_dir"] / "raw_chunks"

    for family, name in [("rmsf", "RMSF_aligned_coord"), ("rmsd", "RMSD_aligned_coord"),
                         ("posks", "POSKS_aligned_coord")]:
        raw_out_path = cfg["sweep_dir"] / f"{cfg['prefix']}_{name}_ensemble_sweep_raw.csv"
        perf_out_path = cfg["sweep_dir"] / f"{cfg['prefix']}_{name}_ensemble_sweep_performance.csv"
        summary_out_path = cfg["sweep_dir"] / f"{cfg['prefix']}_{name}_ensemble_sweep_summary.csv"
        if not force and raw_out_path.exists() and perf_out_path.exists() and summary_out_path.exists():
            print(f"[{protein}] {family.upper()} merge outputs already exist — skipping. Use --force to recompute.", flush=True)
            continue

        parts = []
        for c in range(total_chunks):
            p = raw_dir / f"{cfg['prefix']}_{family}_raw_chunk_{c}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p))
                print(f"  Chunk {c} {family}: {parts[-1].shape}", flush=True)
            else:
                print(f"  Chunk {c} {family}: NOT FOUND (skipping)", flush=True)

        if not parts:
            print(f"No {family} chunk outputs found. Skipping.", flush=True)
            continue

        raw = pd.concat(parts, ignore_index=True)
        save_csv(raw, cfg["sweep_dir"] / f"{cfg['prefix']}_{name}_ensemble_sweep_raw.csv")

        metric_cols = [c for c in raw.columns if c.startswith(f"{family}_") or c.startswith(f"delta_{family}_")]
        perf_rows = []
        for metric_col in metric_cols:
            for (n, r), g in raw.groupby(["ensemble_size", "repeat"]):
                perf_rows.append({
                    "metric_family": family.upper(), "metric_col": metric_col,
                    "ensemble_size": n, "repeat": r,
                    **compare_driver_passenger(g, metric_col),
                })
        perf = pd.DataFrame(perf_rows)
        save_csv(perf, cfg["sweep_dir"] / f"{cfg['prefix']}_{name}_ensemble_sweep_performance.csv")

        if not perf.empty:
            perf["auc_separation"] = (perf["roc_auc_driver_high"] - 0.5).abs()
        # PRIMARY AUC = SIGNED driver-high (0.5 = chance), matching the D1 / DNA-distance
        # / unified paths, so an inverse association reads as <0.5 instead of being
        # flipped up to look predictive. best_direction / separation are descriptive only.
        summary = perf.groupby(["metric_family", "metric_col", "ensemble_size"], as_index=False).agg(
            n_repeats=("repeat", "nunique"),
            auc_mean=("roc_auc_driver_high", "mean"),
            auc_std=("roc_auc_driver_high", "std"),
            auc_best_direction_mean=("roc_auc_best_direction", "mean"),
            auc_separation_mean=("auc_separation", "mean"),
            cliffs_mean=("cliffs_delta_driver_minus_passenger", "mean"),
            wasserstein_mean=("wasserstein", "mean"),
            n_driver=("n_driver", "median"),
            n_passenger=("n_passenger", "median"),
        ) if not perf.empty else pd.DataFrame()
        save_csv(summary, cfg["sweep_dir"] / f"{cfg['prefix']}_{name}_ensemble_sweep_summary.csv")

        print(f"[{protein}] {family.upper()} merge complete: raw={raw.shape} perf={perf.shape} summary={summary.shape}", flush=True)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel RMSF/RMSD ensemble-size sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--protein",       required=True, choices=supported_proteins(REQUIRED_CONFIG_KEYS))
    parser.add_argument("--prepare",       action="store_true", help="One-time: categorize + split coordinate cache by artifact")
    parser.add_argument("--chunk",         type=int, help="0-indexed chunk number to process")
    parser.add_argument("--total-chunks",  type=int, default=4)
    parser.add_argument("--merge",         action="store_true", help="Combine chunk outputs and compute performance/summary stats")
    parser.add_argument("--max-artifacts", type=int, default=None, help="Cap mutant artifacts per chunk (dry-run)")
    parser.add_argument("--force",         action="store_true", help="Recompute even if output files already exist")
    args = parser.parse_args()

    if args.prepare:
        prepare(args.protein, args.total_chunks, force=args.force)
    elif args.merge:
        merge(args.protein, args.total_chunks, force=args.force)
    elif args.chunk is not None:
        if not (0 <= args.chunk < args.total_chunks):
            parser.error(f"--chunk must be 0..{args.total_chunks - 1}")
        run_chunk_sweep(args.protein, args.chunk, args.total_chunks, max_artifacts=args.max_artifacts, force=args.force)
    else:
        parser.print_help()
