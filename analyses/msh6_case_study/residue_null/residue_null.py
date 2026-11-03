#!/usr/bin/env python
"""Fig. 1e residue test calibrated against a WT-vs-WT null (PTEN).

The draft says the per-residue pLDDT distributions of a mutant differ from WT "even
for passengers". The plain Mann-Whitney test assumes independent predictions, but the
five diffusion samples of one AF3 seed share a trunk pass and are correlated, so its
p-values may be too small. Instead of trusting them, ask the WT ensemble what the
test does when nothing changed.

NULL     200 disjoint splits of the 1,000-prediction PTEN WT pool (seeds 1-25 original +
         seeds 26-200), N per side. Two split modes:
           matched -- the pipeline's own draw: 25 disjoint seeds per side (125
                      predictions, as a WT or mutant pool), then N drawn from each; the
                      primary null
           seed    -- whole seeds to one side, N/5 seeds each (harsher: fewer seeds)
           random  -- predictions split ignoring seed, for comparison
CALIBRATION  share of residue tests with raw P < 0.05 under the null (5% if calibrated)
MUTANTS  WT (from the 125-prediction reference pool, as the pipeline draws it) against
         the mutant, N per side, REPEATS draws. Two readouts:
           residue level  empirical P_r = (1 + #null splits with p_null,r <= p_mut,r) / (1 + B),
                          then Benjamini-Hochberg across residues
           variant level  share of BH-significant residues (raw MWU) vs its null
                          distribution: empirical P = (1 + #null >= observed) / (1 + B)

    python analyses/msh6_case_study/residue_null/residue_null.py

Other genes (2026-10-16): --gene with its own WT pool, mutants and window, e.g. MSH6
(chain B of MutS_alpha; the 75-seed WT, so each matched side is 25 of 75 seeds):

    DRIFT_RUN_TAG=v6_ca python analyses/msh6_case_study/residue_null/residue_null.py \
        --gene MSH6 --mutants E221D E1193K --window 1185 1200 \
        --wt-pool analyses/msh6_case_study/wt_pool/wt_75seeds/MutS_alpha_MSH6_GT_WT_75seeds_plddt_values.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SEED = 20261014
N = 90
B = 200
REPEATS = 10
MUTANTS = ("A79T", "A121E")
WINDOW = slice(112, 128)  # residues 113-128, the Fig. 1e window


def load(path, rows=None):
    d = pd.read_csv(path, index_col=0)
    cols = [c for c in d.columns if c.startswith("seed-")]
    x = d[cols].values
    return (x if rows is None else x[rows]), [int(re.match(r"seed-(\d+)_", c).group(1)) for c in cols]


def mwu_p(a, b):
    return np.nan_to_num(mannwhitneyu(a, b, axis=1).pvalue, nan=1.0)


def bh(p):
    return multipletests(p, method="fdr_bh")[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", default="PTEN")
    ap.add_argument("--mutants", nargs="+", default=list(MUTANTS))
    ap.add_argument("--window", nargs=2, type=int, default=(WINDOW.start + 1, WINDOW.stop),
                    help="first and last residue (1-based, subunit-local) of the Fig. 1e window")
    ap.add_argument("--wt-pool", type=Path, default=None,
                    help="non-PTEN: one WT table holding every seed (the original 25 among them)")
    ap.add_argument("--splits", type=int, default=B,
                    help="WT-WT null splits; the empirical P floor is 1/(1+splits)")
    ap.add_argument("--modes", nargs="+", default=["matched", "seed", "random"])
    args, _ = ap.parse_known_args(argv)
    nb = args.splits   # tolerate --ks (ks_null_matched below)
    window = slice(args.window[0] - 1, args.window[1])
    if args.gene == "PTEN":
        rows, mut_path = None, lambda m: REPO / f"PLDDTs/PTEN/PTEN_{m}_plddt_values.csv"
        wt, wt_seeds = load(REPO / "PLDDTs/PTEN/PTEN_WT_plddt_values.csv")
        ext, ext_seeds = load(REPO / "analyses/msh6_case_study/wt_backgrounds_200/plddt/PTEN_WT_seeds26-200_plddt_values.csv")
        pool = np.concatenate([wt, ext], axis=1)
        seeds = np.array(wt_seeds + ext_seeds)
        stem = "residue_null" + ("" if args.splits == B else f"_B{args.splits}")
    else:
        sys.path.insert(0, str(REPO))
        from metric_analysis.pipeline.library.canonical_cohort import load_canonical_cohort
        coh = load_canonical_cohort(args.gene).set_index("mutation_code")
        rows = slice(int(coh.plddt_row_range_start.iloc[0]), int(coh.plddt_row_range_stop.iloc[0]))
        mut_path = lambda m: coh.loc[m, "source_file"]
        # the pipeline's WT draw: the canonical 25-seed reference pool
        wt, _ = load(coh.wt_source_file.iloc[0], rows)
        pool, pool_seeds = load(REPO / args.wt_pool, rows)
        seeds = np.array(pool_seeds)
        stem = f"residue_null_{args.gene}" + ("" if nb == B else f"_B{nb}")
    uniq = np.unique(seeds)
    per_seed = N // 5

    null_p, null_frac, calib = {}, {}, {}
    for mode in args.modes:
        rng = np.random.default_rng(SEED + zlib.crc32(mode.encode()))
        ps, fr = [], []
        for _ in range(nb):
            if mode == "matched":
                sd = rng.permutation(uniq)
                pa = np.flatnonzero(np.isin(seeds, sd[:25]))
                pb = np.flatnonzero(np.isin(seeds, sd[25:50]))
                a, b = rng.choice(pa, N, replace=False), rng.choice(pb, N, replace=False)
            elif mode == "seed":
                s = rng.permutation(uniq)
                a = np.flatnonzero(np.isin(seeds, s[:per_seed]))[:N]
                b = np.flatnonzero(np.isin(seeds, s[per_seed:2 * per_seed]))[:N]
            else:
                perm = rng.permutation(pool.shape[1])
                a, b = perm[:N], perm[N:2 * N]
            p = mwu_p(pool[:, a], pool[:, b])
            ps.append(p)
            fr.append(bh(p).mean())
        null_p[mode] = np.array(ps)          # B x residues
        null_frac[mode] = np.array(fr)
        calib[mode] = {
            "share_raw_p_below_0.05": float((null_p[mode] < 0.05).mean()),
            "share_raw_p_below_0.001": float((null_p[mode] < 0.001).mean()),
            "bh_frac_median": float(np.median(fr)), "bh_frac_p95": float(np.quantile(fr, 0.95)),
            "bh_frac_max": float(np.max(fr)), "share_splits_zero": float(np.mean(np.array(fr) == 0)),
        }

    out = {"gene": args.gene, "window": list(args.window), "n_per_side": N, "null_splits": nb, "pool": int(pool.shape[1]), "calibration": calib, "mutants": {}}
    table = []
    for mut in args.mutants:
        m, _ = load(mut_path(mut), rows)
        rng = np.random.default_rng(SEED + zlib.crc32(mut.encode()))
        res = {mode: {"raw_bh_frac": [], "emp_bh_frac": [], "emp_window_sig": []} for mode in null_p}
        for rep in range(REPEATS):
            p = mwu_p(wt[:, rng.choice(wt.shape[1], N, replace=False)],
                      m[:, rng.choice(m.shape[1], N, replace=False)])
            raw = bh(p)
            for mode, nulls in null_p.items():
                emp = (1 + (nulls <= p[None, :]).sum(axis=0)) / (1 + nb)
                sig = bh(emp)
                res[mode]["raw_bh_frac"].append(raw.mean())
                res[mode]["emp_bh_frac"].append(sig.mean())
                res[mode]["emp_window_sig"].append(int(sig[window].sum()))
                if rep == 0:
                    table += [{"mutant": mut, "null": mode, "residue": r + 1, "mwu_p": p[r],
                              "empirical_p": emp[r], "bh_significant": bool(sig[r])} for r in range(len(p))]
        out["mutants"][mut] = {}
        for mode in null_p:
            raw_frac = float(np.mean(res[mode]["raw_bh_frac"]))
            out["mutants"][mut][mode] = {
                "raw_mwu_bh_frac_mean": raw_frac,
                "empirical_bh_frac_mean": float(np.mean(res[mode]["emp_bh_frac"])),
                "empirical_bh_frac_range": [float(min(res[mode]["emp_bh_frac"])),
                                            float(max(res[mode]["emp_bh_frac"]))],
                "window_significant": res[mode]["emp_window_sig"],
                "variant_level_empirical_p": float((1 + (null_frac[mode] >= raw_frac).sum()) / (1 + nb)),
            }

    pd.DataFrame(table).to_csv(HERE / f"{stem}_per_residue.csv", index=False)
    (HERE / f"{stem}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


def ks_null_matched():
    """The KS-score WT-vs-WT null of pten_wt_null_panel, redone with the matched seed split.

    pten_wt_null_panel splits WT predictions at random, which puts samples of one seed on
    both sides and narrows the null. Here each side is 25 disjoint seeds, N drawn from
    them; variant scores are read unchanged from that panel's values.csv.
    """
    import sys
    sys.path.insert(0, str(REPO))
    from metric_analysis.pipeline.library.descriptor_metrics import residue_ks
    wt, wt_seeds = load(REPO / "PLDDTs/PTEN/PTEN_WT_plddt_values.csv")
    ext, ext_seeds = load(REPO / "analyses/msh6_case_study/wt_backgrounds_200/plddt/PTEN_WT_seeds26-200_plddt_values.csv")
    pool, seeds = np.concatenate([wt, ext], axis=1), np.array(wt_seeds + ext_seeds)
    uniq = np.unique(seeds)
    rng = np.random.default_rng(SEED + zlib.crc32(b"ks_matched"))
    out = {}
    for n in (50, 100):
        v = pd.read_csv(REPO / f"analyses/msh6_case_study/wt_backgrounds_200/results_wt1000_n{n}/values.csv")
        null = []
        for _ in range(B):
            sd = rng.permutation(uniq)
            pa, pb = np.flatnonzero(np.isin(seeds, sd[:25])), np.flatnonzero(np.isin(seeds, sd[25:50]))
            null.append(float(residue_ks(pool[:, rng.choice(pa, n, replace=False)],
                                         pool[:, rng.choice(pb, n, replace=False)])))
        null = np.array(null)
        rnd = v.loc[v.group == "wt_wt", "ks"].to_numpy()
        p95 = np.quantile(null, 0.95)
        out[f"N{n}"] = {
            "random_split_null": {"median": float(np.median(rnd)), "p95": float(np.quantile(rnd, 0.95)), "max": float(rnd.max())},
            "matched_seed_null": {"median": float(np.median(null)), "p95": float(p95), "max": float(null.max())},
        }
        for grp in ("wt_passenger", "wt_driver"):
            g = v.loc[v.group == grp]
            out[f"N{n}"][grp] = {"n": int(len(g)), "above_matched_p95": int((g.ks > p95).sum()),
                                 "above_matched_max": int((g.ks > null.max()).sum()),
                                 "below_matched_p95": g.loc[g.ks <= p95, "item"].tolist()}
    (HERE / "ks_null_matched.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__" and "--ks" in __import__("sys").argv:
    ks_null_matched()
