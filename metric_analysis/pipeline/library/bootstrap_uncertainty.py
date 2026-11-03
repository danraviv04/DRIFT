"""Reproducible bootstrap uncertainty helpers for per-residue class means."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_sd(
    values: np.ndarray,
    *,
    n_resamples: int = 1_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return the mutation-level bootstrap SD of each column's sample mean.

    Rows are independent mutations and columns are residues. A bootstrap draw
    resamples complete mutation rows, preserving the residue profile within each
    mutation. NaNs are omitted independently in each residue mean.

    Multinomial row counts are equivalent to drawing row indices with replacement,
    while avoiding a potentially very large ``B x mutations x residues`` array.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("bootstrap_mean_sd expects a 2D mutations-by-residues array")
    if array.shape[0] == 0:
        raise ValueError("bootstrap_mean_sd needs at least one mutation")
    if int(n_resamples) < 2:
        raise ValueError("n_resamples must be at least 2")
    if array.shape[0] == 1:
        return np.zeros(array.shape[1], dtype=float)

    generator = rng if rng is not None else np.random.default_rng(42)
    n_rows = array.shape[0]
    counts = generator.multinomial(
        n_rows,
        np.full(n_rows, 1.0 / n_rows),
        size=int(n_resamples),
    ).astype(float, copy=False)
    finite = np.isfinite(array)
    numerators = counts @ np.where(finite, array, 0.0)
    denominators = counts @ finite.astype(float)
    bootstrap_means = np.divide(
        numerators,
        denominators,
        out=np.full_like(numerators, np.nan),
        where=denominators > 0,
    )
    return np.nanstd(bootstrap_means, axis=0, ddof=1)


__all__ = ["bootstrap_mean_sd"]
