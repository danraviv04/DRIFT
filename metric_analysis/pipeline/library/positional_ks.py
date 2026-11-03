"""Positional KS: the KS-score computed on C-alpha positions instead of pLDDT.

The pLDDT KS-score compares, at each residue, the distribution of pLDDT across the
wild-type predictions with its distribution across the mutant predictions, and
averages the per-residue KS statistic over residues. The positional version keeps
that construction and replaces the per-prediction pLDDT value with a per-prediction
position-derived value. Two definitions are provided:

``distance`` (primary)
    d_{n,r} = || x_{n,r} - mu_r ||, the distance of residue r's C-alpha in prediction n
    from the wild-type ensemble centre mu_r (the mean position over the whole wild-type
    pool). A KS on these distances responds both to a shift of the mutant ensemble
    away from the wild-type centre and to a change in its spread, and is invariant
    to rotations of the common frame about mu_r.

``axis``
    The KS statistic on each Cartesian coordinate separately, averaged over x, y and
    z at each residue. It sees direction as well as magnitude, but it depends on the
    orientation of the common frame. Not used in the pipeline since 2026-10-16 (the
    sweep asks for ``modes=("distance",)``); kept for experiments/positional_ks.

Inputs are long-format coordinate frames as stored in the aligned coordinate cache:
one row per (structure_index, chain, resid, icode) with x_aligned, y_aligned and
z_aligned, every artifact already superposed into one common frame: since 2026-10-15
the WT rigid-core fit of cache_builders/subunit_core_fit.py (before, USalign onto the
WT reference, which fitted the complex's first chain only). Residues are matched on (chain, resid, icode). A residue missing from a
prediction is left missing in that prediction's column and omitted from that
residue's KS, as residue_ks does for pLDDT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metric_analysis.pipeline.library.descriptor_metrics import residue_ks_statistics

KEY = ["chain", "resid", "icode"]
XYZ = ["x_aligned", "y_aligned", "z_aligned"]
MODES = ("distance", "axis")


def _restrict(frame, chains):
    if chains is None or frame is None or frame.empty:
        return frame
    return frame[frame["chain"].astype(str).isin([str(c) for c in chains])]


def reference_centre(wt_pool, chains=None):
    """Per-residue mean C-alpha position over the whole wild-type pool."""
    wt_pool = _restrict(wt_pool, chains)
    centre = wt_pool.groupby(KEY, sort=True, observed=True)[XYZ].mean()
    centre.columns = ["cx", "cy", "cz"]
    return centre


def _matrices(subset, centre, names=("distance", "x", "y", "z")):
    """Residues x predictions matrices aligned to the centre's residue order.

    Returns a dict with 'distance' and the three per-axis deviations. The axis
    matrices are deviations from the centre; subtracting a per-residue constant does
    not change a KS statistic, it only keeps the numbers readable.
    """
    sub = subset.merge(centre, left_on=KEY, right_index=True, how="inner")
    dx = sub["x_aligned"].to_numpy(float) - sub["cx"].to_numpy(float)
    dy = sub["y_aligned"].to_numpy(float) - sub["cy"].to_numpy(float)
    dz = sub["z_aligned"].to_numpy(float) - sub["cz"].to_numpy(float)
    sub = sub.assign(dx=dx, dy=dy, dz=dz, dist=np.sqrt(dx * dx + dy * dy + dz * dz))
    out = {}
    for name, col in (("distance", "dist"), ("x", "dx"), ("y", "dy"), ("z", "dz")):
        if name not in names:
            continue
        wide = sub.pivot_table(index=KEY, columns="structure_index", values=col,
                               aggfunc="first", observed=True)
        out[name] = wide.reindex(centre.index).to_numpy(float)
    return out


def positional_ks_per_residue(wt_subset, mutant_subset, centre, chains=None, modes=MODES):
    """Per-residue statistics for the requested definitions, in the centre's residue order.

    Returns a DataFrame indexed like ``centre`` with columns ``distance`` (KS on the
    distances) and/or ``axis`` (mean of the x, y and z KS statistics).
    """
    names = (["distance"] if "distance" in modes else []) + (["x", "y", "z"] if "axis" in modes else [])
    wt_m = _matrices(_restrict(wt_subset, chains), centre, names)
    mu_m = _matrices(_restrict(mutant_subset, chains), centre, names)
    ks = {k: residue_ks_statistics(wt_m[k], mu_m[k]) for k in names}
    out = {}
    if "distance" in modes:
        out["distance"] = ks["distance"]
    if "axis" in modes:
        xyz = np.vstack([ks["x"], ks["y"], ks["z"]])
        n = np.isfinite(xyz).sum(axis=0)
        out["axis"] = np.where(n > 0, np.nansum(xyz, axis=0) / np.maximum(n, 1), np.nan)
    return pd.DataFrame(out, index=centre.index)


def positional_ks(wt_subset, mutant_subset, centre, chains=None, modes=MODES):
    """Variant-level scores: the mean over residues of each per-residue statistic.

    Returns {'distance': float, 'axis': float, 'n_residues': int} (only the requested
    modes; n_residues counts the first of them).
    """
    per_res = positional_ks_per_residue(wt_subset, mutant_subset, centre, chains, modes)
    out = {}
    for mode in modes:
        v = per_res[mode].to_numpy(float)
        v = v[np.isfinite(v)]
        out[mode] = float(v.mean()) if v.size else float("nan")
    out["n_residues"] = int(np.isfinite(per_res[modes[0]].to_numpy(float)).sum())
    return out
