#!/usr/bin/env python
"""Score the MSH2 MSA-swap control (README §4; 2026-10-16).

Two WT ensembles, identical inputs except chain A's MSA:
  WT@Q629R-MSA -- MutS_alpha_MSH2_WT_vartmpl, the production WT
                  (PLDDTs/MutS_alpha/MutS_alpha_WT_plddt_values.csv, seeds 1-25 x 5 samples);
  WT@WT-MSA    -- MutS_alpha_MSH2_WT_vartmpl_wtmsa (build_msa_swap_job.py, seeds 1-10 x 5).

The statistic is the production KS-score (descriptor_metrics.residue_ks: mean over residues
of the per-residue two-sample KS on pLDDT), MSH2 rows only (0-933 = residues 1-934).

  same_seed  KS(WT@WT-MSA seeds S, WT@Q629R-MSA seeds S), S = 1-10. Reported, NOT decisive:
             the same seed couples two near-identical runs (MSH2 variants vs this WT score
             ~0.017 lower on shared seeds than on disjoint ones), so it reads low against
             a seed-disjoint background.
  disjoint   KS(WT@WT-MSA seeds S, WT@Q629R-MSA seeds T), T = |S| seeds drawn from the
             vartmpl seeds outside S. DECISIVE: built exactly like the background.
  background KS(WT@Q629R-MSA seeds A, WT@Q629R-MSA seeds B), A and B disjoint, |A| = |B| = |S|
             (prediction noise at the same size).
  variants   context: KS(WT@Q629R-MSA seeds T, variant seeds S) for each MSH2 variant,
             the same seed-disjoint design, so the MSA term can be set against a real effect.

Decision: median(disjoint) <= p95(background) -> the MSA channel is negligible;
          above it -> every variant's KS carries an MSA term; disclose it.
The 615-645 window, where the Q629R search realigned, is tested the same way (window-mean KS
against the background's window-mean p95). The raw per-residue excess over the background mean
(results/msa_swap/*_per_residue.csv) is descriptive only: with one side fixed it is not zero
under the null (the synthetic null shows 0.07 in the window).

Input (the archive from the cluster, or a CSV already extracted from it):
    PYTHONPATH="." python experiments/wt_msa_check/score_msa_swap.py \
        --archive <dir>/MutS_alpha_MSH2_WT_vartmpl_wtmsa_data.tar.gz
    ... score_msa_swap.py --csv <dir>/MutS_alpha_MSH2_WT_vartmpl_wtmsa_plddt_values.csv
The archive is read once with fast_plddt.archive_plddt_table, chains A,B (the extraction
stream_align --plddt-out ran for the WT CSVs); the table is kept as
results/msa_swap/MutS_alpha_MSH2_WT_vartmpl_wtmsa_plddt_values.csv (~0.7 MB).

Synthetic stand-ins, no GPU output needed:
    ... score_msa_swap.py --synthetic null      # vartmpl seeds 18-25 posing as the new run
                                                # (8 seeds: 25 cannot hold three disjoint 10s)
                                                # -> must come out NEGLIGIBLE
    ... score_msa_swap.py --synthetic positive  # the legacy WT (WT MSA but no 2O8B template),
                                                # seeds 1-10 -> must come out ABOVE NOISE
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metric_analysis.pipeline.labels import extract_mutation_code
from metric_analysis.pipeline.library.descriptor_metrics import residue_ks, residue_ks_statistics

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "results/msa_swap"
REF_CSV = REPO / "PLDDTs/MutS_alpha/MutS_alpha_WT_plddt_values.csv"
LEGACY_CSV = REPO / "PLDDTs/_wt_legacy_2026-10-15/MutS_alpha_WT_plddt_values.csv"
VARIANT_DIR = REPO / "PLDDTs/MutS_alpha"
NAME = "MutS_alpha_MSH2_WT_vartmpl_wtmsa"
SAMPLE = re.compile(r"seed-(\d+)_sample-(\d+)")
MSH2_ROWS = 934                 # chain A first: rows 0-933 are MSH2 residues 1-934
TABLE_ROWS = 2294               # MSH2 934 + MSH6 1360
SITE = (615, 645)               # where the Q629R search realigned (README §2), 1-based
N_BG, N_DISJ, N_VAR = 1000, 500, 5
SEED = 20261016


def seed_table(df, rows=MSH2_ROWS):
    """(residues x predictions) array for the seed-sample columns, and each column's seed."""
    cols = [c for c in df.columns if SAMPLE.fullmatch(str(c))]
    if len(df) != TABLE_ROWS:
        raise ValueError(f"expected {TABLE_ROWS} residue rows (MSH2+MSH6), got {len(df)}")
    seeds = np.array([int(SAMPLE.fullmatch(c).group(1)) for c in cols])
    return df[cols].to_numpy(float)[:rows], seeds


def load_csv(path):
    return seed_table(pd.read_csv(path, index_col=0))


def extract(archive):
    from metric_analysis.pipeline.library.fast_plddt import archive_plddt_table
    tab = archive_plddt_table(Path(archive), ["A", "B"])
    extra = [c for c in tab.columns if not SAMPLE.fullmatch(str(c))]
    if extra:
        print(f"dropping non seed-sample columns: {extra}")
        tab = tab.drop(columns=extra)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{NAME}_plddt_values.csv"
    tab.to_csv(path)
    print(f"extracted {tab.shape} -> {path.relative_to(REPO)}")
    return path


def cols(seeds, chosen):
    return np.isin(seeds, list(chosen))


def score(test, tseeds, ref, rseeds, ref_pool, mode):
    rng = np.random.default_rng(SEED)
    S = sorted(set(tseeds.tolist()))
    n = len(S)
    pool = sorted(ref_pool)
    outside = [s for s in pool if s not in S]
    if not set(S) <= set(rseeds.tolist()):
        raise ValueError(f"test seeds {S} are not all in the reference")
    if len(outside) < n or len(pool) < 2 * n:
        raise ValueError(f"reference pool {pool} too small for disjoint draws of {n}")
    t_all = test
    same = residue_ks(ref[:, cols(rseeds, S)], t_all)

    disj, disj_res = [], []
    for _ in range(N_DISJ):
        T = rng.choice(outside, n, replace=False)
        r = residue_ks_statistics(ref[:, cols(rseeds, T)], t_all)
        disj.append(np.nanmean(r))
        disj_res.append(r)
    bg, bg_res = [], []
    for _ in range(N_BG):
        p = rng.permutation(pool)
        r = residue_ks_statistics(ref[:, cols(rseeds, p[:n])], ref[:, cols(rseeds, p[n:2 * n])])
        bg.append(np.nanmean(r))
        bg_res.append(r)
    disj, bg = np.array(disj), np.array(bg)

    # MSH2 variants in the same seed-disjoint design: WT seeds T (outside S), variant seeds S
    var = {}
    for f in sorted(VARIANT_DIR.glob("MutS_alpha_MSH2_*_plddt_values.csv")):
        if "WT" in f.name.upper():
            continue
        v, vseeds = load_csv(f)
        if not set(S) <= set(vseeds.tolist()):
            continue
        vS = v[:, cols(vseeds, S)]
        var[extract_mutation_code(f.name.replace("_plddt_values.csv", ""))] = float(np.mean(
            [residue_ks(ref[:, cols(rseeds, rng.choice(outside, n, replace=False))], vS) for _ in range(N_VAR)]))
    var = pd.Series(var, dtype=float)

    p95, bg_mean = float(np.percentile(bg, 95)), float(bg.mean())
    med = float(np.median(disj))
    excess = np.nanmean(disj_res, axis=0) - np.nanmean(bg_res, axis=0)
    res = np.arange(1, MSH2_ROWS + 1)
    site = (res >= SITE[0]) & (res <= SITE[1])
    var_excess = float(var.median() - bg_mean) if len(var) else float("nan")
    site_med = float(np.median(np.nanmean(np.array(disj_res)[:, site], axis=1)))
    site_p95 = float(np.percentile(np.nanmean(np.array(bg_res)[:, site], axis=1), 95))
    negligible = med <= p95
    rep = {
        "mode": mode, "seeds_test": S, "n_predictions_test": int(test.shape[1]),
        "ensemble_size_per_side": int(test.shape[1]), "reference_pool": pool,
        "same_seed_ks": round(same, 4),
        "disjoint_ks_median": round(med, 4), "disjoint_ks_mean": round(float(disj.mean()), 4),
        "disjoint_ks_p5_p95": [round(float(np.percentile(disj, q)), 4) for q in (5, 95)],
        "background_mean": round(bg_mean, 4), "background_sd": round(float(bg.std()), 4),
        "background_p95": round(p95, 4),
        "disjoint_median_percentile_in_background": round(float((bg < med).mean() * 100), 1),
        "frac_disjoint_draws_above_bg_p95": round(float((disj > p95).mean()), 3),
        "msa_excess_over_background": round(med - bg_mean, 4),
        "variant_n": int(len(var)), "variant_ks_median": round(float(var.median()), 4) if len(var) else None,
        "variant_excess_median": round(var_excess, 4),
        "msa_excess_as_frac_of_variant_excess": (round((med - bg_mean) / var_excess, 3)
                                                 if var_excess and var_excess == var_excess else None),
        "frac_variants_below_disjoint_median": round(float((var < med).mean()), 3) if len(var) else None,
        # the site window on the same footing: its mean KS per draw, against the background's
        "site_615_645_ks_disjoint_median": round(site_med, 4),
        "site_615_645_background_p95": round(site_p95, 4),
        "site_615_645_above_background_p95": bool(site_med > site_p95),
        "per_residue_excess_site_615_645": round(float(excess[site].mean()), 4),
        "per_residue_excess_elsewhere": round(float(excess[~site].mean()), 4),
        "top_excess_residues": [int(r) for r in res[np.argsort(-excess)[:10]]],
        "verdict": "NEGLIGIBLE" if negligible else "ABOVE NOISE",
        "reading": (f"median seed-disjoint KS {med:.3f} <= background p95 {p95:.3f}: the MSA channel is "
                    "negligible; the vartmpl WT stands as a valid WT" if negligible else
                    f"median seed-disjoint KS {med:.3f} > background p95 {p95:.3f}: every variant's KS "
                    "carries an MSA term; disclose it (or put every job on one shared MSA)"),
    }
    pd.DataFrame({"residue": res, "ks_disjoint_mean": np.nanmean(disj_res, axis=0).round(4),
                  "ks_background_mean": np.nanmean(bg_res, axis=0).round(4),
                  "excess": excess.round(4)}).to_csv(OUT / f"{mode}_per_residue.csv", index=False)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--archive", help="MutS_alpha_MSH2_WT_vartmpl_wtmsa_data.tar.gz from the cluster")
    g.add_argument("--csv", help="its pLDDT table, already extracted (chains A,B)")
    g.add_argument("--synthetic", choices=["null", "positive"], help="stand-in test, no GPU output")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    ref, rseeds = load_csv(REF_CSV)
    if sorted(set(rseeds.tolist())) != list(range(1, 26)) or ref.shape[1] != 125:
        raise ValueError(f"{REF_CSV.name}: expected seeds 1-25 x 5 samples, got {ref.shape[1]} columns")
    pool = set(range(1, 26))
    if args.synthetic == "null":
        # vartmpl seeds 18-25 pose as the new run, relabelled 1-8, and leave the reference
        mask = cols(rseeds, range(18, 26))
        test, tseeds = ref[:, mask], rseeds[mask] - 17
        pool = set(range(1, 18))
        ref, rseeds = ref[:, ~mask], rseeds[~mask]
        mode = "synthetic_null"
    elif args.synthetic == "positive":
        leg, lseeds = load_csv(LEGACY_CSV)
        mask = cols(lseeds, range(1, 11))
        test, tseeds, mode = leg[:, mask], lseeds[mask], "synthetic_positive"
    else:
        path = extract(args.archive) if args.archive else Path(args.csv)
        test, tseeds = load_csv(path)
        mode = "msa_swap"
        got = sorted(set(tseeds.tolist()))
        if got != list(range(1, 11)) or test.shape[1] != 50:
            print(f"WARNING: expected seeds 1-10 x 5 = 50 predictions, got {test.shape[1]} over seeds {got}")
    if not np.isfinite(test).all():
        raise ValueError(f"{int((~np.isfinite(test)).sum())} non-finite pLDDTs in the MSH2 rows of the test")

    rep = score(test, tseeds, ref, rseeds, pool, mode)
    (OUT / f"{mode}.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep.items():
        print(f"{k:42s} {v}")
    print(f"-> {(OUT / f'{mode}.json').relative_to(REPO)}")


if __name__ == "__main__":
    sys.exit(main())
