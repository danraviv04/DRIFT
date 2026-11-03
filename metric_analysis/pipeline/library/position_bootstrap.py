"""Residue-position clusters as the bootstrap resampling unit (2026-10-20).

Every substitution at one residue position forms one cluster; a bootstrap draws whole
clusters. Since 2026-10-20 (Dan) this is the published resampling unit for every
interval in the paper; the variant-level intervals are kept for the record.

The functions mirror experiments/position_cluster_bootstrap/run.py (the main-text
estimator): AUC intervals stratify clusters as driver-only / passenger-only / mixed
and resample within each stratum; correlations and paired differences resample
clusters without stratification. With singleton clusters (``singleton=True``) every
function consumes the generator exactly as the variant-level code it replaces, so the
variant intervals reproduce bit for bit; each caller checks this.
"""
from __future__ import annotations

import re

import numpy as np

MUTATION = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def position_of(mutation: str) -> int:
    match = MUTATION.match(str(mutation))
    if match is None:
        raise ValueError(f"cannot parse a residue position from {mutation!r}")
    return int(match.group(2))


def clusters_of(mutations, singleton: bool = False) -> list[np.ndarray]:
    """Member index arrays, one per residue position, in first-seen order."""
    mutations = list(mutations)
    if singleton:
        return [np.array([i]) for i in range(len(mutations))]
    order: dict[int, list[int]] = {}
    for i, m in enumerate(mutations):
        order.setdefault(position_of(m), []).append(i)
    return [np.array(v) for v in order.values()]


def strata_of(y, clusters, stratified: bool = True) -> list[np.ndarray]:
    """Cluster ids to resample independently: driver-only, passenger-only, mixed."""
    y = np.asarray(y)
    if not stratified:
        return [np.arange(len(clusters))]
    kinds = {"pos": [], "neg": [], "mixed": []}
    for c, members in enumerate(clusters):
        labels = set(y[members].tolist())
        kinds["pos" if labels == {1} else "neg" if labels == {0} else "mixed"].append(c)
    return [np.array(kinds[k]) for k in ("pos", "neg", "mixed") if kinds[k]]


def stratified_draw(rng, clusters, strata) -> np.ndarray:
    """Variant indices of one class-stratified cluster resample.

    Singleton clusters: ``r_[rng.choice(pos, pos.size), rng.choice(neg, neg.size)]``.
    """
    picked = np.concatenate([rng.choice(s, s.size) for s in strata])
    return np.concatenate([clusters[c] for c in picked])


def unstratified_draw(rng, clusters) -> np.ndarray:
    """Variant indices of one unstratified cluster resample.

    Singleton clusters: ``rng.integers(0, n, n)``.
    """
    picks = rng.integers(0, len(clusters), len(clusters))
    return np.concatenate([clusters[p] for p in picks])


def stratified_weights(rng, clusters, strata, n_variants: int, size: int) -> np.ndarray:
    """(size, n_variants) resample counts, one multinomial per stratum over its clusters.

    Singleton clusters: ``rng.multinomial(len(pos), uniform, size)`` then the same for
    ``neg``, scattered onto the variants (the hierarchical bootstrap of Note S8).
    """
    weights = np.zeros((size, n_variants))
    for s in strata:
        counts = rng.multinomial(len(s), np.full(len(s), 1 / len(s)), size=size).astype(float)
        for j, c in enumerate(s):
            weights[:, clusters[c]] += counts[:, [j]]
    return weights


def describe(y, clusters) -> dict:
    y = np.asarray(y)
    kinds = [set(y[m].tolist()) for m in clusters]
    return dict(variants=int(len(y)), positions=len(clusters),
                driver_positions=sum(k == {1} for k in kinds),
                passenger_positions=sum(k == {0} for k in kinds),
                mixed_positions=sum(len(k) > 1 for k in kinds))
