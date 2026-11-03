"""Validated current N=90 article scores; seeds denote subsampling draws."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(42, 52))
SCORE = "plddt_ks_residue"


#: protein-config fingerprint of the caches; d61d29328362 is the v6_ca run
FINGERPRINT = "09ed028c6d1c"


def load_scores(size=90, fingerprint=FINGERPRINT, genes=None):
    """``genes`` restricts both the manifest and the caches to those genes before
    the cohort check, for a panel that draws one gene: the d61d29328362 caches
    still hold POLE P356S, dropped from the manifest on 2026-10-21, and that
    should not block an MSH6 panel. The check itself stays exact."""
    manifest = pd.read_csv(REPO / "metric_analysis/manifests/strict_canonical_manifest.csv")
    expected = manifest.rename(columns={"protein": "gene", "mutation_code": "mutation"})
    if genes is not None:
        expected = expected[expected.gene.isin(genes)]
    expected = expected.set_index(["gene", "mutation"])["label"].sort_index()
    if not expected.index.is_unique:
        raise ValueError("Duplicate manifest variants")
    frames, sources = [], []
    for seed in SEEDS:
        path = REPO / "drift/cache" / (
            f"labeled_size{size}_seed{seed}_ratio_of_means_proteins{fingerprint}_esfeat6.csv")
        frame = pd.read_csv(path, usecols=["gene", "mutation", "label", SCORE])
        if genes is not None:
            frame = frame[frame.gene.isin(genes)]
        indexed = frame.set_index(["gene", "mutation"]).sort_index()
        if not indexed.index.is_unique or not indexed.index.equals(expected.index):
            raise ValueError(f"Cohort mismatch: {path}")
        if not np.array_equal(indexed.label, expected) or not np.isfinite(indexed[SCORE]).all():
            raise ValueError(f"Labels or scores invalid: {path}")
        frames.append(frame.assign(seed=seed))
        sources.append({"path": str(path.relative_to(REPO)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return pd.concat(frames, ignore_index=True), {
        "size": size, "seeds": list(SEEDS), "descriptor": SCORE,
        "sources": sources, "variant_count": len(expected),
        "aggregation": "raw-score mean for violin/assays; mean per-seed ROC with sample SD"}


def mean_scores(scores):
    result = scores.groupby(["gene", "mutation"], sort=True).agg(
        label=("label", "first"), plddt_ks_residue=(SCORE, "mean"),
        n_seeds=("seed", "nunique")).reset_index()
    if not result.n_seeds.eq(len(SEEDS)).all():
        raise ValueError("Incomplete seed coverage")
    return result


def save_provenance(path, provenance):
    Path(path).write_text(json.dumps(provenance, indent=2) + "\n")
