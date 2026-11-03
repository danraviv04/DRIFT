#!/usr/bin/env python
"""Seed-pairing fingerprint: were the WT and the variants run with the same numerics?

AF3 with the same seed, same software and same hardware gives nearly the same
prediction for two inputs that differ by one residue. So the same (seed, sample)
column of two such jobs is strongly correlated, beyond what any two columns are.
For each pair of pools we correlate per-residue deviations (pLDDT minus the pool's
residue mean) of matched columns, and of mismatched columns as the baseline.
  pairing = median r(matched) - median r(mismatched)
High between variants but ~0 between the WT and the variants = the WT was run
with different numerics (software version, GPU type or kernels), not just later.

    PYTHONPATH="." python experiments/wt_variant_offset/seed_pairing.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "si_figures"))
from figlib import cohort  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
GENES = ["MSH2", "PIK3CA", "MSH6", "PTEN", "VHL", "POLE"]
NV = 12


def cols_sorted(df):
    c = [x for x in df.columns if x.startswith("seed-")]
    return df[sorted(c)]


def dev(df, rows):
    x = cols_sorted(df).to_numpy(float)[rows]
    return x - x.mean(1, keepdims=True), list(sorted(c for c in df.columns if c.startswith("seed-")))


def pair_score(a, ca, b, cb, rng):
    common = [c for c in ca if c in set(cb)]
    ia = [ca.index(c) for c in common]
    ib = [cb.index(c) for c in common]
    A, B = a[:, ia], b[:, ib]
    ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    A, B = A[ok], B[ok]
    def r(u, v):
        return float(np.corrcoef(u, v)[0, 1])
    matched = [r(A[:, k], B[:, k]) for k in range(len(common))]
    perm = rng.permutation(len(common))
    perm = np.where(perm == np.arange(len(common)), np.roll(perm, 1), perm)
    mism = [r(A[:, k], B[:, perm[k]]) for k in range(len(common))]
    return float(np.median(matched)), float(np.median(mism))


def main():
    rng = np.random.default_rng(7)
    res = {}
    for g in GENES:
        c = cohort(g)
        r0 = c.iloc[0]
        rows = (slice(int(r0.plddt_row_range_start), int(r0.plddt_row_range_stop))
                if pd.notna(r0.plddt_row_range_start) else slice(None))
        pools = {"WT": pd.read_csv(r0.wt_source_file, index_col=0)}
        if g == "MSH2":
            # The pre-2026-04-02 MSH2 WT (backbone N, superseded) came from the analysis repo's
            # git history (de0501e2^); this release has none, so the comparison is optional.
            old = ROOT / "PLDDTs/MutS_alpha/legacy/MutS_alpha_MSH2_WT_2026-04-02_plddt_values.csv"
            if old.exists():
                pools["WT_old_2026-04-02"] = pd.read_csv(old, index_col=0)
            pools["WT_MSH6_GT"] = pd.read_csv(ROOT / "PLDDTs/MutS_alpha_MSH6_GT/MutS_alpha_MSH6_GT_WT_plddt_values.csv",
                                              index_col=0)
        if g == "POLE":
            for n in ["POLE_WT", "POLE_again_WT"]:
                pools[n] = pd.read_csv(ROOT / f"PLDDTs/POLE/{n}_plddt_values.csv", index_col=0)
        shape = pools["WT"].shape
        sel = c.sample(min(NV, len(c)), random_state=1)
        var = {}
        for m, f in zip(sel.mutation_code, sel.source_file):
            d = pd.read_csv(f, index_col=0)
            if d.shape == shape:
                var[m] = d
        D = {k: dev(v, rows) for k, v in {**pools, **{f"v:{m}": d for m, d in var.items()}}.items()}
        vn = [k for k in D if k.startswith("v:")]
        out = {}
        vv = [pair_score(*D[vn[i]], *D[vn[j]], rng) for i in range(len(vn)) for j in range(i + 1, len(vn))]
        out["variant_vs_variant"] = {"matched": float(np.median([x[0] for x in vv])),
                                     "mismatched": float(np.median([x[1] for x in vv]))}
        for p in pools:
            s = [pair_score(*D[p], *D[v], rng) for v in vn]
            out[f"{p}_vs_variants"] = {"matched": float(np.median([x[0] for x in s])),
                                       "mismatched": float(np.median([x[1] for x in s]))}
        if g == "MSH2":
            out["WT_vs_WT_old"] = dict(zip(["matched", "mismatched"], pair_score(*D["WT"], *D["WT_old_2026-04-02"], rng)))
        if g == "POLE":
            out["WT_vs_POLE_WT"] = dict(zip(["matched", "mismatched"], pair_score(*D["WT"], *D["POLE_WT"], rng)))
            out["WT_vs_POLE_again_WT"] = dict(zip(["matched", "mismatched"], pair_score(*D["WT"], *D["POLE_again_WT"], rng)))
        for k in out:
            out[k]["pairing"] = out[k]["matched"] - out[k]["mismatched"]
        res[g] = out
        print(g)
        for k, v in out.items():
            print(f"   {k:28s} matched r {v['matched']:.3f}  mismatched r {v['mismatched']:.3f}  pairing {v['pairing']:+.3f}")
    (OUT / "seed_pairing.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
