"""Shared descriptor definitions and interchangeable KS backends.

The residue KS statistic is defined for any ensemble size N >= 1: both empirical
CDFs exist from one observation, and D_r = sup_t |F_WT,r(t) - F_mut,r(t)|. At
n = m = 1 it reduces to 1{x != y}, so the residue mean is the fraction of residues
whose two pLDDTs differ; with n = m it takes values k/n. The statistic is defined
there, but the KS *test* has no power: the smallest two-sided p-value for n = m is
2 / C(2n, n), i.e. 1.0 at N=1, 0.33 at N=2, 0.10 at N=3.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import numpy as np
from scipy import stats


def _inputs(wt, mutant):
    wt, mutant = np.asarray(wt, dtype=float), np.asarray(mutant, dtype=float)
    if wt.ndim != 2 or mutant.ndim != 2 or wt.shape[0] != mutant.shape[0]:
        raise ValueError("KS inputs must be 2-D with matching residue counts")
    return wt, mutant


def ks_residue_results(wt, mutant, *, backend="auto", min_samples=1,
                       alternative="two-sided", method="auto", pvalues=False):
    """Return aligned statistic and p-value arrays, omitting nonfinite samples.

    P-values and one-sided alternatives use SciPy. The fast path computes only
    the two-sided magnitude, evaluating ECDF gaps after complete tie groups.
    DRIFT_KS_BACKEND overrides the selected backend, including explicit requests.
    """
    wt, mutant = _inputs(wt, mutant)
    if int(min_samples) != min_samples or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    minimum = int(min_samples)
    backend = os.environ.get("DRIFT_KS_BACKEND", backend)
    if backend not in ("auto", "fast", "scipy"):
        raise ValueError(f"unknown KS backend: {backend}")
    statistic = np.full(wt.shape[0], np.nan)
    probability = np.full(wt.shape[0], np.nan)
    if min(wt.shape[1], mutant.shape[1]) < minimum:
        return statistic, probability
    finite = np.isfinite(wt).all(axis=1) & np.isfinite(mutant).all(axis=1)
    use_fast = backend != "scipy" and alternative == "two-sided" and not pvalues
    if use_fast and finite.any():
        a, b = wt[finite], mutant[finite]
        na, nb = a.shape[1], b.shape[1]
        pooled = np.concatenate([a, b], axis=1)
        order = np.argsort(pooled, axis=1, kind="stable")
        ordered = np.take_along_axis(pooled, order, axis=1)
        from_a = order < na
        gap = np.abs(np.cumsum(from_a, axis=1) / na
                     - np.cumsum(~from_a, axis=1) / nb)
        group_end = np.ones_like(gap, dtype=bool)
        group_end[:, :-1] = ordered[:, 1:] != ordered[:, :-1]
        statistic[finite] = np.where(group_end, gap, -np.inf).max(axis=1)
    for r in np.flatnonzero(~finite if use_fast else np.ones(len(wt), dtype=bool)):
        a, b = wt[r, np.isfinite(wt[r])], mutant[r, np.isfinite(mutant[r])]
        if min(len(a), len(b)) >= minimum:
            result = stats.ks_2samp(a, b, alternative=alternative, method=method)
            statistic[r], probability[r] = result.statistic, result.pvalue
    return statistic, probability


def residue_ks_statistics(wt, mutant, **kwargs):
    return ks_residue_results(wt, mutant, **kwargs)[0]


def residue_ks(wt, mutant, *, backend="auto", min_samples=1):
    """Mean per-residue KS; missing observations are omitted within each row."""
    values = residue_ks_statistics(wt, mutant, backend=backend, min_samples=min_samples)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def ks_mean_equal(a, b):
    """Compatibility entry point: equal shapes and complete finite rows only."""
    a, b = _inputs(a, b)
    if a.shape != b.shape:
        raise ValueError(f"KS inputs must have equal 2-D shape, got {a.shape}, {b.shape}")
    finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    return residue_ks(a[finite], b[finite])


def relative_change(reference, comparison):
    valid = np.isfinite(reference) & np.isfinite(comparison) & (np.abs(reference) > 0)
    if not valid.any():
        return float("nan")
    return float(np.abs(comparison[valid] / reference[valid] - 1.0).mean())


def _sd_ratio(wt, mutant, *, backend="auto"):
    return relative_change(wt.std(axis=1), mutant.std(axis=1))


def _centred_ks(wt, mutant, *, backend="auto"):
    return residue_ks(wt - wt.mean(axis=1, keepdims=True),
                      mutant - mutant.mean(axis=1, keepdims=True), backend=backend)


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable
    min_samples: int = 1
    needs_equal_n: bool = False
    direction: str = "higher"
    units: str = "dimensionless"
    status: str = "evaluated"
    backends: tuple[str, ...] = ("auto",)

    def __call__(self, wt, mutant, *, backend="auto"):
        wt, mutant = _inputs(wt, mutant)
        if self.needs_equal_n and wt.shape != mutant.shape:
            raise ValueError(f"{self.name} requires equal ensemble sizes")
        if min(wt.shape[1], mutant.shape[1]) < self.min_samples or not len(wt):
            return float("nan")
        valid = np.isfinite(wt).all(axis=1) & np.isfinite(mutant).all(axis=1)
        if not valid.any():
            return float("nan")
        return self.fn(wt[valid], mutant[valid], backend=backend)


REGISTRY = {
    "ks_raw": Metric("ks_raw", residue_ks,
                     backends=("auto", "fast", "scipy")),
    # Defined at N=1 but identically 0 there: centring one observation leaves 0.
    "ks_centred": Metric("ks_centred", _centred_ks,
                         backends=("auto", "fast", "scipy")),
    "sd_ratio": Metric("sd_ratio", _sd_ratio, min_samples=2),
}
METRICS = tuple(REGISTRY)


def descriptor_values(wt, mutant):
    """The canonical descriptors on an equal-size, complete-row pair."""
    wt, mutant = _inputs(wt, mutant)
    if wt.shape != mutant.shape:
        raise ValueError("canonical descriptor inputs must have equal ensemble sizes")
    valid = np.isfinite(wt).all(axis=1) & np.isfinite(mutant).all(axis=1)
    return {name: REGISTRY[name](wt[valid], mutant[valid]) for name in METRICS} | {
        "n_residues": int(valid.sum()),
    }
