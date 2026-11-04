"""Build the DRIFT score caches: one labelled table per (ensemble size, seed).

Each cache row is a variant of the strict canonical cohort with its label, its
AlphaMissense score and the ensemble-shift descriptors (residue KS, centred KS,
spread ratio, Wasserstein; ensemble_shift.py) computed from N predictions per side
drawn with one subsampling seed. The caches go to drift/cache and are reused when
present; everything downstream reads them.

Usage:
    python -m drift.run --sizes 1,5,10,20,40,60,90 --seeds 42-51
    python -m drift.run --sizes 90 --seeds 42 --force-recompute
"""
import argparse
from pathlib import Path

import pandas as pd

from . import core
from .core import SAMPLING_MODES, SELECTION_RULES, PROJECT_ROOT, _resolve_run_config
from .ensemble_shift import build_ensemble_shift_features, FEATURE_COLUMNS as ES_FEATURE_COLUMNS
from .protein_config import protein_config_fingerprint

CACHE_DIR = Path(__file__).resolve().parent / "cache"
# Version 7 drops the D1 column; 6 enforced the subunit residue scope.
ES_CACHE_VERSION = 7
STRICT_COHORT_MANIFEST = (
    Path(PROJECT_ROOT) / "metric_analysis" / "manifests" /
    "strict_canonical_manifest.csv"
)

# --------------------------------------------------------------------------- #
# Ground-truth groups. There is ONE binary label route:
#   annotation -- driver = explicit variant annotation / ClinVar class (POLE
#                 binary label table, MSH2/MSH6/PTEN/VHL/PIK3CA ClinVar).
# TP53 and KRAS have no binary label: their assays are continuous targets only
# (analyses/continuous/continuous_functional), never thresholded.
# --------------------------------------------------------------------------- #
GT_GROUPS = {
    "annotation": {"POLE", "MSH2", "MSH6", "VHL", "PTEN", "PIK3CA"},
}


def _gt_subset(df, name):
    """Rows for a GT group; 'all' is every gene."""
    if name == "all":
        return df
    sub = df[df["gene"].isin(GT_GROUPS[name])].copy()
    if sub.empty:
        raise SystemExit(f"[gt-group] no variants for group {name!r} ({sorted(GT_GROUPS[name])})")
    return sub


def _apply_strict_canonical_cohort(df, manifest_path=STRICT_COHORT_MANIFEST):
    """Restrict a score table to the canonical cohort and its labels.

    The versioned strict manifest is the single source of truth for membership and
    labels on every cache and recomputation path; a canonical variant missing from
    the table is an error.
    """
    required = {"gene", "mutation", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"score table missing required columns: {sorted(missing)}")

    manifest = pd.read_csv(manifest_path)
    manifest_required = {"protein", "mutation_code", "label"}
    missing = manifest_required.difference(manifest.columns)
    if missing:
        raise ValueError(f"strict cohort manifest missing columns: {sorted(missing)}")

    canonical = manifest[["protein", "mutation_code", "label"]].rename(columns={
        "protein": "gene",
        "mutation_code": "mutation",
        "label": "strict_label",
    })
    # Legacy alias: MSH2/MSH6 spell the same in both conventions, so this is a
    # no-op for them and only still matters for a pre-split results file.
    canonical["gene"] = canonical["gene"].replace({"MutS_alpha": "MutSalpha"})
    keys = ["gene", "mutation"]
    if canonical.duplicated(keys).any():
        raise ValueError("strict cohort manifest contains duplicate gene/mutation keys")
    if df.duplicated(keys).any():
        raise ValueError("score table contains duplicate gene/mutation keys")

    source_keys = df[keys].drop_duplicates()
    missing_from_dataset = canonical.merge(source_keys, on=keys, how="left", indicator=True)
    missing_from_dataset = missing_from_dataset[missing_from_dataset["_merge"] == "left_only"]
    if not missing_from_dataset.empty:
        examples = missing_from_dataset[keys].head(5).to_dict("records")
        raise ValueError(
            f"score table is missing {len(missing_from_dataset)} canonical variants; "
            f"examples={examples}"
        )

    merged = df.merge(canonical, on=keys, how="inner", validate="one_to_one", sort=False)
    n_excluded = len(df) - len(merged)
    n_relabelled = int((merged["label"].astype(int) != merged["strict_label"].astype(int)).sum())
    merged["label"] = merged.pop("strict_label").astype(int)
    if n_excluded or n_relabelled:
        print(
            f"[cohort] strict canonical manifest: excluded={n_excluded}, "
            f"labels_overridden={n_relabelled}, retained={len(merged)}"
        )
    return merged


def cache_path(size, seed, policy=None):
    """The score cache for one (size, seed, sampling policy) under this protein config.

    The sampling-policy tag is empty for the default nested/random policy; a
    non-default policy gets its own cache rather than colliding with the primary run.
    """
    policy = core.SamplingPolicy.coerce(policy)
    policy_tag = f"_{policy.tag()}" if policy.tag() else ""
    return CACHE_DIR / (
        f"labeled_size{size}_seed{seed}_proteins{protein_config_fingerprint()}"
        f"_esfeat{ES_CACHE_VERSION}{policy_tag}.csv"
    )


def build_or_load_dataset(size, cfg, force=False):
    """Labelled (gene, mutation, label, afm_score) + ensemble-shift descriptors,
    cached per (size, seed, protein config, sampling policy)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    policy = core.sampling_policy_from_config(cfg)
    cache = cache_path(size, cfg["seed"], policy)
    if cache.exists() and not force:
        df = pd.read_csv(cache)
        if all(c in df.columns for c in ES_FEATURE_COLUMNS):
            print(f"[data] loading cached labeled_df: {cache.name}")
            return _apply_strict_canonical_cohort(df)
        print(f"[data] cache missing ensemble-shift columns -> recomputing: {cache.name}")
    print(f"[data] no cache -> computing the descriptors from the pLDDT CSVs (size={size}) ...")
    df = core.build_labeled_dataset_for_config(
        size, seed=cfg["seed"], sampling_policy=policy,
    )[["gene", "mutation", "label", "afm_score"]].copy()

    # Ensemble-shift descriptors from the same pLDDT CSVs and structure sampling.
    feat = build_ensemble_shift_features(
        core.STRUCTURE_CONFIGS, max_structures_per_variant=size, seed=cfg["seed"],
        policy=policy,
    )
    df = df.merge(feat, on=["gene", "mutation"], how="left")
    na = df[ES_FEATURE_COLUMNS].isna().sum()
    undefined = [c for c in ES_FEATURE_COLUMNS if na[c] == len(df)]
    partial = {c: int(na[c]) for c in ES_FEATURE_COLUMNS if 0 < na[c] < len(df)}
    if undefined:
        # e.g. plddt_spread_ratio is undefined at size=1 (no spread from one structure)
        print(f"[data] undefined at size={size} (expected): {undefined}")
    if partial:
        print(f"[data] partially missing: {partial}")

    df = _apply_strict_canonical_cohort(df)
    df.to_csv(cache, index=False)
    print(f"[data] cached -> {cache.name}  ({len(df)} labeled variants, "
          f"+{len(ES_FEATURE_COLUMNS)} ensemble-shift features)")
    return df


def _seed_list(spec):
    """Parse non-negative seeds, accepting either ``42-51`` or ``42,43``."""
    text = str(spec).strip()
    try:
        if "-" in text and "," not in text:
            bounds = [part.strip() for part in text.split("-")]
            if len(bounds) != 2 or not all(bounds):
                raise ValueError
            start, stop = (int(value) for value in bounds)
            if stop < start:
                raise ValueError
            values = list(range(start, stop + 1))
        else:
            values = [int(value) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("seeds must be a comma list or ascending range, e.g. 42-51") from exc
    values = list(dict.fromkeys(values))
    if not values or any(value < 0 for value in values):
        raise ValueError("seeds must be non-negative integers")
    return values


def _positive_int_list(spec, name):
    try:
        values = [int(value) for value in str(spec).split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma list of positive integers") from exc
    values = list(dict.fromkeys(values))
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must be a comma list of positive integers")
    return values


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="1,5,10,20,40,60,90",
                    help="ensemble sizes N (predictions per side), comma list")
    ap.add_argument("--seeds", default="42-51",
                    help="subsampling seeds: a range (42-51) or a comma list")
    ap.add_argument(
        "--sampling-mode", choices=list(SAMPLING_MODES), default="nested",
        help="nested (default): the size-N sample is a prefix of larger samples, "
             "so sizes are paired by common random numbers. independent: a fresh "
             "draw per size -- a supplementary control, not a replacement.",
    )
    ap.add_argument(
        "--selection", choices=list(SELECTION_RULES), default="random",
        help="random (default): uniform draw, so N=1 is a random-single-structure "
             "baseline. top_confidence: keep the highest mean-pLDDT structures, "
             "ranked on wild type and applied to the mutant.",
    )
    ap.add_argument("--force-recompute", action="store_true", help="rebuild caches that exist")
    args = ap.parse_args(argv)
    if args.selection == "top_confidence" and args.sampling_mode == "independent":
        ap.error(
            "--selection top_confidence is deterministic given the pool, so "
            "--sampling-mode independent would change nothing; use nested"
        )
    try:
        seeds = _seed_list(args.seeds)
        sizes = _positive_int_list(args.sizes, "--sizes")
    except ValueError as exc:
        ap.error(str(exc))

    base_cfg = _resolve_run_config({"sampling_mode": args.sampling_mode, "selection": args.selection})
    print(f"[caches] sizes={sizes} seeds={seeds} sampling={args.sampling_mode}/{args.selection} "
          f"-> {CACHE_DIR}")
    for seed in seeds:
        for size in sizes:
            cfg = dict(base_cfg, seed=seed, max_structures_per_variant=size)
            build_or_load_dataset(size, cfg, force=args.force_recompute)
    print("[caches] done")


if __name__ == "__main__":
    main()
