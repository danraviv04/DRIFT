#!/usr/bin/env python
"""Fig. 1f / Note S2 WT-WT backgrounds rebuilt at 2,000 splits (published: 200).

Every run of analyses/msh6_case_study/wt_backgrounds_200 behind the manuscript is replayed with its own
code (pten_wt_null_panel.py, imported unchanged) and its own RNG stream:

    rng = default_rng(20260923)
    [PTEN only] batch_check(..., 200, rng)     # the 2-file PTEN pool
    wt_vs_wt(..., 200, rng)                    # published 200 WT-WT splits
    wt_vs_variant(..., 20, rng) per variant    # published variant scores
    wt_vs_wt(..., 1800, rng)                   # NEW: 1,800 further splits, same generator

The extension is drawn AFTER the variant draws, so the variant RNG stream is untouched, and
the 2,000-split background contains the published 200 splits as its first 200 (split_0 ..
split_199). The script asserts that the replayed 200 splits, every variant score and the
PTEN batch check equal the published values.csv / summary.json exactly.

A stability check re-estimates each p95 under 3 further independent streams of 2,000 splits.

Runs (published dir under analyses/msh6_case_study/wt_backgrounds_200/ -> results/<key>/):
    MSH6 375-prediction pool (DRIFT_RUN_TAG=v6_ca cohort, 143 variants), N=50/100, seed/random
    PTEN 1,000-prediction pool (v3 manifest, 257 variants), N=50/100, seed/random

    cd <repo>; DRIFT_RUN_TAG=v6_ca PYTHONPATH="." python \
        analyses/msh6_case_study/wt_background_2000/run_wt_background_2000.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PUB = REPO / "analyses" / "msh6_case_study" / "wt_backgrounds_200"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PUB))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pten_wt_null_panel as P  # noqa: E402

OUT = HERE / "results"
PUB_SPLITS = 200
TOTAL_SPLITS = 2000
CHECK_STREAMS = 3
MSH6_WT = [Path("analyses/msh6_case_study/wt_pool/wt_75seeds/MutS_alpha_MSH6_GT_WT_75seeds_plddt_values.csv")]
PTEN_WT = [Path("PLDDTs/PTEN/PTEN_WT_plddt_values.csv"),
           Path("analyses/msh6_case_study/wt_backgrounds_200/plddt/PTEN_WT_seeds26-200_plddt_values.csv")]

RUNS = {  # key: (gene, N, split, published dir, WT files)
    "MSH6_n100_seed": ("MSH6", 100, "seed", "results_MSH6_v6ca_wt375_n100_seedsplit", MSH6_WT),
    "MSH6_n50_seed": ("MSH6", 50, "seed", "results_MSH6_v6ca_wt375_n50_seed", MSH6_WT),
    "MSH6_n100_random": ("MSH6", 100, "random", "results_MSH6_v6ca_wt375_n100_random", MSH6_WT),
    "MSH6_n50_random": ("MSH6", 50, "random", "results_MSH6_v6ca_wt375_n50_random", MSH6_WT),
    "PTEN_n100_seed": ("PTEN", 100, "seed", "results_wt1000_n100_seedsplit", PTEN_WT),
    "PTEN_n50_seed": ("PTEN", 50, "seed", "results_wt1000_n50_seedsplit", PTEN_WT),
    "PTEN_n100_random": ("PTEN", 100, "random", "results_wt1000_n100", PTEN_WT),
    "PTEN_n50_random": ("PTEN", 50, "random", "results_wt1000_n50", PTEN_WT),
}


def cohort_of(gene):
    """Exactly pten_wt_null_panel.main's cohort choice."""
    if gene == "PTEN":
        return pd.read_csv(P.MANIFEST).query("protein == 'PTEN'")
    from metric_analysis.pipeline.library.canonical_cohort import load_canonical_cohort
    return load_canonical_cohort(gene)


def bg_stats(null):
    return {"median": float(np.median(null)), "p95": float(np.percentile(null, 95))}


def run(key):
    t0 = time.time()
    gene, n, split, pub_dir, wt_files = RUNS[key]
    pub_vals = pd.read_csv(PUB / pub_dir / "values.csv", float_precision="round_trip")  # exact floats
    pub_sum = json.loads((PUB / pub_dir / "summary.json").read_text())
    assert pub_sum["n_per_side"] == n and pub_sum["wt_splits"] == PUB_SPLITS
    assert pub_sum["variant_draws"] == 20 and pub_sum["seed"] == P.BASE_SEED
    assert pub_sum.get("split", "random") == split, (key, pub_sum.get("split"))
    assert [str(p) for p in wt_files] == [str(Path(p)) for p in pub_sum["wt_csv"]], pub_sum["wt_csv"]

    cohort = cohort_of(gene)
    block = P.row_slice(cohort)
    parts, names = P.load_wt_pool([REPO / p for p in wt_files])
    if block is not None:
        parts = [p[block] for p in parts]
    wt = np.concatenate(parts, axis=1)
    seeds = P.seed_of(names) if split == "seed" else None
    assert wt.shape[1] == pub_sum["wt_pool_structures"]

    # ---- replay of the published stream -------------------------------------------------
    rng = np.random.default_rng(P.BASE_SEED)
    batch = P.batch_check(parts, n, PUB_SPLITS, rng) if len(parts) > 1 else None
    if batch is not None or pub_sum.get("wt_batch_check") is not None:
        assert batch == pub_sum["wt_batch_check"], (key, batch, pub_sum["wt_batch_check"])
    null_pub = P.wt_vs_wt(wt, n, PUB_SPLITS, rng, seeds)
    ref_null = pub_vals[pub_vals.group == "wt_wt"].ks.to_numpy()
    assert np.array_equal(null_pub, ref_null), f"{key}: published WT-WT splits not reproduced"
    rows = []
    for rec in cohort.itertuples():
        mutant = P.load(rec.source_file, block)
        assert mutant.shape[0] == wt.shape[0]
        group = "wt_driver" if rec.label == 1 else "wt_passenger"
        rows.append(dict(group=group, item=rec.mutation_code,
                         ks=P.wt_vs_variant(wt, mutant, n, 20, rng, seeds)))
    var = pd.DataFrame(rows)
    ref_var = pub_vals[pub_vals.group != "wt_wt"].reset_index(drop=True)
    assert var[["group", "item"]].equals(ref_var[["group", "item"]]), f"{key}: variant order differs"
    assert np.array_equal(var.ks.to_numpy(), ref_var.ks.to_numpy()), f"{key}: variant scores differ"

    # ---- extension: 1,800 further splits from the same generator -------------------------
    null_ext = P.wt_vs_wt(wt, n, TOTAL_SPLITS - PUB_SPLITS, rng, seeds)
    null = np.concatenate([null_pub, null_ext])

    # ---- independent stability streams ----------------------------------------------------
    checks = []
    for k in range(CHECK_STREAMS):
        r = np.random.default_rng(zlib.crc32(f"wt_background_2000_check/{key}/{k}".encode()))
        checks.append(bg_stats(P.wt_vs_wt(wt, n, TOTAL_SPLITS, r, seeds)))

    out = OUT / key
    out.mkdir(parents=True, exist_ok=True)
    values = pd.concat([pd.DataFrame(dict(group="wt_wt", item=[f"split_{i}" for i in range(len(null))],
                                          ks=null)), var], ignore_index=True)
    values.to_csv(out / "values.csv", index=False)
    drv = var[var.group == "wt_driver"].ks.to_numpy()
    pas = var[var.group == "wt_passenger"].ks.to_numpy()
    s_new, s_pub = bg_stats(null), bg_stats(null_pub)
    summary = {
        "key": key, "gene": gene, "n_per_side": n, "split": split,
        "wt_pool_structures": int(wt.shape[1]), "wt_csv": [str(p) for p in wt_files],
        "wt_batch_check": batch, "seed": P.BASE_SEED, "variant_draws": 20,
        "wt_splits": TOTAL_SPLITS,
        "published_dir": f"analyses/msh6_case_study/wt_backgrounds_200/{pub_dir}",
        "published_reproduced": "200 WT-WT splits, all variant scores and batch check: exact",
        "extension": f"splits {PUB_SPLITS}..{TOTAL_SPLITS - 1} drawn from the same generator after "
                     "the variant draws (variant RNG stream unchanged)",
        "wt_wt_median": s_new["median"], "wt_wt_p95": s_new["p95"],
        "published_200": {"wt_wt_median": s_pub["median"], "wt_wt_p95": s_pub["p95"],
                          "drivers_above": int((drv > s_pub["p95"]).sum()),
                          "passengers_above": int((pas > s_pub["p95"]).sum())},
        "n_drivers": int(len(drv)), "n_passengers": int(len(pas)),
        "drivers_above": int((drv > s_new["p95"]).sum()),
        "passengers_above": int((pas > s_new["p95"]).sum()),
        "variants_below_or_at_p95": var.loc[var.ks <= s_new["p95"], "item"].tolist()
                                    if gene == "PTEN" else None,
        "driver_vs_passenger_auc": pub_sum["driver_vs_passenger_auc"],
        "driver_vs_passenger_mwu_p": pub_sum.get("driver_vs_passenger_mwu_p"),
        "groups": {g: {"n": int((values.group == g).sum()),
                       "median": float(values.loc[values.group == g, "ks"].median()),
                       "fraction_above_wt_wt_p95": float((values.loc[values.group == g, "ks"]
                                                          > s_new["p95"]).mean())}
                   for g, _, _ in P.GROUPS},
        "stability_streams": [dict(stream=k, **c,
                                   drivers_above=int((drv > c["p95"]).sum()),
                                   passengers_above=int((pas > c["p95"]).sum()))
                              for k, c in enumerate(checks)],
        "runtime_s": round(time.time() - t0, 1),
        "note": "WT-WT dots are overlapping splits of one pool, not independent observations.",
    }
    # AUC / MWU depend only on the (unchanged) variant scores; recompute to confirm
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(var.group == "wt_driver", var.ks))
    assert auc == pub_sum["driver_vs_passenger_auc"], (key, auc)
    if "driver_vs_passenger_mwu_p" in pub_sum:
        assert float(mannwhitneyu(drv, pas).pvalue) == pub_sum["driver_vs_passenger_mwu_p"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{key}: done in {summary['runtime_s']} s; median {s_new['median']:.4f} p95 {s_new['p95']:.4f} "
          f"(published {s_pub['p95']:.4f}); drivers {summary['drivers_above']}/{len(drv)}, "
          f"passengers {summary['passengers_above']}/{len(pas)}", flush=True)
    return key


def main():
    if os.environ.get("DRIFT_RUN_TAG") != "v6_ca":
        raise SystemExit("set DRIFT_RUN_TAG=v6_ca (the MSH6 cohort of the published runs)")
    OUT.mkdir(parents=True, exist_ok=True)
    keys = sys.argv[1:] or list(RUNS)
    with ProcessPoolExecutor(max_workers=4) as ex:
        for k in ex.map(run, keys):
            pass
    print("all runs done", flush=True)


if __name__ == "__main__":
    main()
