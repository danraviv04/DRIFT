"""Shared loading for item 2 (improving the AM + KS-score combination, 2026-10-16).

Same cells as the paper's headline: N = 90, annotation cohort (all 1,799 variants), v6_ca
fingerprint, ten subsampling seeds, read through the fusion benchmark's loader.
Metrics as in the analysis repo's ks_posks_am/common.py (checked against sklearn), copied
here unchanged when that folder left the release.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import drift.protein_config as pc
from analyses.annotation.simple_fusion_benchmark import size_sweep

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FINGERPRINT = "d61d29328362"
SEEDS = tuple(range(42, 52))
GENES = ("POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA")
N = 90
AM_AMBIGUOUS = (0.34, 0.564)   # AlphaMissense's published likely-benign / likely-pathogenic cut-offs
WT_SPLITS = REPO / "figures/si/lib/results/wt_background_splits.csv"


# ---------------------------------------------------------------- metrics (higher = driver)
def auc(y, s):
    """Mann-Whitney AUC with average ranks for ties."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ap(y, s):
    """Average precision, tied scores grouped into one threshold (as sklearn)."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    o = np.argsort(-s, kind="mergesort")
    ss, tp = s[o], np.cumsum(y[o])
    last = np.r_[ss[1:] != ss[:-1], True]
    tp, k = tp[last], np.flatnonzero(last) + 1
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * tp / k))


def prec_at_r(y, s):
    y = np.asarray(y, bool)
    r = int(y.sum())
    o = np.argsort(-np.asarray(s), kind="mergesort")[:r]
    return y[o].mean()


METRICS = {"auc": auc, "ap": ap, "prec_at_R": prec_at_r}


def load_frames():
    real = pc.protein_config_fingerprint
    pc.protein_config_fingerprint = lambda *a, **k: FINGERPRINT
    try:
        frames = [size_sweep.load_cell(N, s, gt_group="annotation")[0] for s in SEEDS]
    finally:
        pc.protein_config_fingerprint = real
    keys = frames[0][["gene", "mutation", "label"]]
    for f in frames[1:]:
        assert f[["gene", "mutation", "label"]].equals(keys)
    return frames


def wt_null():
    """gene -> (median, robust SD) of the seed-matched WT-vs-WT KS-score (12 vs 13 seeds, 60/side)."""
    bg = pd.read_csv(WT_SPLITS)
    bg = bg[bg.kind == "seed"]
    out = {}
    for g, d in bg.groupby("gene"):
        q1, med, q3 = np.percentile(d.ks, [25, 50, 75])
        out[g] = (float(med), float((q3 - q1) / 1.349))
    return out
