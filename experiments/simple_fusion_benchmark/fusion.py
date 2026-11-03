"""Non-learned fusion scores: normalise each descriptor, then combine it.

Nothing here fits a parameter to labels. Weights are fixed in advance and the
normalisations read feature values only, never labels, so a fusion score is one
vector per variant that does not depend on the CV protocol. The single exception
is `gz`, whose scaler is fitted on each fold's training rows; it exists because
that is exactly the scaler the learned models in the screen use, and it shows
what the per-gene normalisations are worth on their own.

Orientation is fixed a priori: all four inputs are defined so that larger means
more driver-like (AlphaMissense pathogenicity, D1 perturbation, both KS
distances). No direction is ever flipped to fit the data.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

AFM = "afm_score"
DESCRIPTORS = ("d1_score", "plddt_ks_residue", "plddt_ks_centred")

# Same eight input sets as the learned screen, so the two are directly comparable.
ARMS = {
    "AM": [AFM],
    "AM+D1": [AFM, "d1_score"],
    "AM+resKS": [AFM, "plddt_ks_residue"],
    "AM+cKS": [AFM, "plddt_ks_centred"],
    "AM+D1+resKS": [AFM, "d1_score", "plddt_ks_residue"],
    "AM+D1+cKS": [AFM, "d1_score", "plddt_ks_centred"],
    "AM+resKS+cKS": [AFM, "plddt_ks_residue", "plddt_ks_centred"],
    "AM+all": [AFM, *DESCRIPTORS],
}

# Fixed, pre-specified weights on AlphaMissense; the rest is split equally over
# the descriptors. 0.5 is the equal-weight case for a two-column arm.
AM_WEIGHTS = (0.5, 0.667, 0.75, 0.9)


def _per_gene(values, genes, transform):
    out = np.empty(values.shape, dtype=float)
    for gene in np.unique(genes):
        idx = genes == gene
        out[idx] = transform(values[idx])
    return out


def normalise(values, genes, kind, folds=None):
    """Column-wise normalisation of a (rows, columns) block.

    rank  percentile rank in (0, 1) within gene, average ranks for ties
    z     standardised within gene
    vdw   van der Waerden: inverse normal of the within-gene percentile rank
    gz    standardised globally, scaler fitted on each fold's training rows
    """
    if kind == "rank":
        return _per_gene(values, genes, lambda b: rankdata(b, axis=0) / (len(b) + 1.0))
    if kind == "z":
        return _per_gene(values, genes,
                         lambda b: (b - b.mean(0)) / np.where(b.std(0) > 0, b.std(0), 1.0))
    if kind == "vdw":
        return norm.ppf(normalise(values, genes, "rank"))
    if kind == "gz":
        if folds is None:
            raise ValueError("gz normalisation needs folds")
        out = np.empty(values.shape, dtype=float)
        for tr, va in folds:
            mean, sd = values[tr].mean(0), values[tr].std(0)
            out[va] = (values[va] - mean) / np.where(sd > 0, sd, 1.0)
        return out
    raise ValueError(kind)


def combine(block, kind, weights=None):
    """Reduce a normalised (rows, columns) block to one score per row."""
    if kind == "mean":
        return block.mean(1) if weights is None else block @ weights
    if kind == "median":
        return np.median(block, axis=1)
    if kind == "max":
        return block.max(1)
    if kind == "min":
        return block.min(1)
    if kind == "gmean":
        # Rank product, in log space. Only defined for strictly positive input,
        # which is why it is offered for `rank` alone.
        if (block <= 0).any():
            raise ValueError("gmean needs positive input; use rank normalisation")
        return np.exp(np.log(block).mean(1))
    raise ValueError(kind)


def specifications():
    """(method, normalisation, combiner, weights spec) for every fusion score."""
    specs = []
    for norm_kind in ("rank", "z", "vdw"):
        for combiner in ("mean", "median", "max", "min"):
            specs.append((f"{norm_kind}_{combiner}", norm_kind, combiner, None))
        for w in AM_WEIGHTS:
            if w == 0.5:
                continue  # identical to the equal-weight mean for two columns
            specs.append((f"{norm_kind}_wmean{w:g}", norm_kind, "mean", w))
    specs.append(("rank_gmean", "rank", "gmean", None))
    specs.append(("gz_mean", "gz", "mean", None))
    return specs


def build(frame, folds_by_protocol, *, arms=None):
    """Every (arm, fusion method) score.

    Returns a list of records with `score` plus the metadata needed to identify
    the method. `protocol_dependent` marks the one normalisation whose scaler is
    refitted per fold, and therefore differs between LOGO and stratified CV.
    """
    genes = frame.gene.to_numpy()
    records = []
    for arm, columns in (ARMS if arms is None else arms).items():
        values = frame[columns].to_numpy(dtype=float)
        for name, norm_kind, combiner, am_weight in specifications():
            if len(columns) == 1 and (combiner != "mean" or am_weight is not None):
                continue  # every combiner is the identity on a single column
            if am_weight is not None:
                rest = (1.0 - am_weight) / (len(columns) - 1)
                weights = np.array([am_weight] + [rest] * (len(columns) - 1))
            else:
                weights = None
            protocols = (folds_by_protocol if norm_kind == "gz"
                         else {"shared": None})
            for protocol, folds in protocols.items():
                block = normalise(values, genes, norm_kind, folds)
                records.append({
                    "method": name, "arm": arm, "normalisation": norm_kind,
                    "combiner": combiner,
                    "weights": "equal" if weights is None else
                               "/".join(f"{w:.3f}" for w in weights),
                    "protocol_dependent": norm_kind == "gz",
                    "protocol": protocol,
                    "score": combine(block, combiner, weights),
                })
    return records
