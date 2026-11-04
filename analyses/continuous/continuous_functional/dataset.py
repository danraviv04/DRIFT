"""Label-free descriptor table for the continuous functional analyses.

Functional analyses are continuous only (2026-10-13): no assay is ever
thresholded into driver/passenger. TP53 and KRAS therefore have no binary label
and are disabled in DRIFT's binary pipeline, so their descriptors cannot come from
the labelled fusion caches. This module computes them directly, for the four
assay genes, with the SAME production code the binary pipeline uses:

    variants          drift.structure_draws.build_variant_table + variant_keys
    ensemble shift    drift.ensemble_shift.build_ensemble_shift_features
    AlphaMissense     drift.core.afm_df (KRAS from the P01116-2 / 4B file)

and the same configs (protein_config.load_continuous_configs), sampling policy and
draw keys, so a variant's residueKS here is the number the binary pipeline would
compute for it.

Cohort (cohort()), one row per variant with a pLDDT ensemble AND a measured assay:
    TP53  every modelled variant with a Phenotypic_Selection_Model score
    KRAS  the KRAS4B blind stratum (hotspot_only variants were chosen for their
          assay values and are dropped); no LFC band is excluded any more
    PTEN  the transcript-matched canonical (ClinVar) cohort with a measured
    MSH2  assay score -- their structures were generated for ClinVar variants
Initiator-codon (position 1) substitutions are dropped everywhere: a full-length
residue-1 model does not represent start loss.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from drift import core, structure_draws  # noqa: E402
from drift import run as R  # noqa: E402
from drift.ensemble_shift import (  # noqa: E402
    FEATURE_COLUMNS as ES_FEATURE_COLUMNS,
    build_ensemble_shift_features,
)
from drift.protein_config import (  # noqa: E402
    build_structure_configs,
    load_continuous_configs,
    protein_config_fingerprint,
)
import targets as T  # noqa: E402

CACHE = HERE / "cache"
MANIFEST = REPO / "metric_analysis" / "manifests" / "strict_canonical_manifest.csv"
# Genes whose structures exist only for their ClinVar variants: the continuous
# cohort is that canonical cohort's measured overlap.
CANONICAL_COHORT_GENES = ("PTEN", "MSH2")


def runtime_configs():
    return build_structure_configs(load_continuous_configs(T.CONTINUOUS_GENES))


def _policy(size, seed):
    cfg = R._resolve_run_config({"max_structures_per_variant": size})
    cfg["seed"] = seed
    return cfg, core.sampling_policy_from_config(cfg)


def cache_path(size, seed):
    _, policy = _policy(size, seed)
    tag = f"_{policy.tag()}" if policy.tag() else ""
    return CACHE / (
        f"continuous_size{size}_seed{seed}_"
        f"proteins{protein_config_fingerprint()}_esfeat{R.ES_CACHE_VERSION}{tag}.csv"
    )


def build_descriptors(size, seed, force=False):
    """AM + ensemble-shift descriptors for every modelled variant, no labels."""
    path = cache_path(size, seed)
    if path.exists() and not force:
        return pd.read_csv(path)
    _, policy = _policy(size, seed)
    configs = runtime_configs()
    rows, _ = core.build_variant_table(
        configs, max_structures_per_variant=size, random_seed=seed, policy=policy,
    )
    variants = structure_draws.variant_keys(rows)
    feat = build_ensemble_shift_features(
        configs, max_structures_per_variant=size, seed=seed, policy=policy,
    )
    afm = core.afm_df[["gene", "mutation", "afm_score"]].drop_duplicates(["gene", "mutation"])
    df = (variants.merge(feat, on=["gene", "mutation"], how="outer", validate="one_to_one")
            .merge(afm, on=["gene", "mutation"], how="left", validate="one_to_one"))
    df = df[df["gene"].isin(T.CONTINUOUS_GENES)].sort_values(["gene", "mutation"])
    df = df.reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _position(mutation):
    return mutation.str.extract(r"(\d+)")[0].astype(int)


def cohort(descriptors):
    """Restrict descriptors to the continuous cohort and attach the assay target."""
    # profileKS (plddt_ks) was removed 2026-10-13; caches built before carry it.
    descriptors = descriptors.drop(columns=["plddt_ks"], errors="ignore")
    d = descriptors.merge(T.continuous_targets(), on=["gene", "mutation"],
                          how="inner", validate="one_to_one")
    d = d[_position(d["mutation"]).ne(1)]
    blind = T.kras_blind()
    d = d[~d["gene"].eq("KRAS") | d["mutation"].isin(blind)]
    manifest = pd.read_csv(MANIFEST)
    keep = set(zip(manifest["protein"], manifest["mutation_code"]))
    in_canon = [(g, m) in keep for g, m in zip(d["gene"], d["mutation"])]
    d = d[~d["gene"].isin(CANONICAL_COHORT_GENES) | np.array(in_canon)]
    if d["afm_score"].isna().any():
        bad = d.loc[d["afm_score"].isna(), ["gene", "mutation"]].head().to_dict("records")
        raise ValueError(f"continuous cohort variants without AlphaMissense: {bad}")
    return d.sort_values(["gene", "mutation"]).reset_index(drop=True)


def load(size, seed, force=False):
    """The continuous cohort at one (size, seed): descriptors + assay target."""
    return cohort(build_descriptors(size, seed, force=force))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = load(a.size, a.seed, force=a.force)
    print(out.groupby("gene").agg(n=("y", "size"), ks=("plddt_ks_residue", "mean"),
                                  afm=("afm_score", "mean")))
    print("ES columns missing:", sorted(set(ES_FEATURE_COLUMNS) - set(out.columns)))
