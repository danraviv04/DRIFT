#!/usr/bin/env python
"""Note S1 "Sensitivity to the superposition", redone on the matched seed 1-25 draws.

Dan's go-ahead (2026-10-21): the paragraph compared the WT rigid-core fit with US-align
on "an earlier set of N=90 subsamples" (the coordinate sweeps, whose draws for eight POLE
drivers included AF3 seeds 26-32). Here both frames are scored on exactly the predictions
of Supplementary Table S2: the ten production N=90 draws (seeds 42-51), POLE on seeds
1-25, P356S out of the cohort, MSH2 G162R excluded -> 1,797 variants.

Frames (same structures, same rows, different coordinates):
  core_fit  structure_metrics/<gene>/ensemble_aware/cache_corefit (published S1 frame)
  usalign   .../cache_usalign: US-align of the scored chain on a WT prediction; for MSH6
            .../cache_usalign_scored_chain (MSH6 fitted on its own chain, not MSH2's frame)

Each frame runs s1_matched_draws.calculate unchanged except for the coordinate cache
(s1_matched_draws.COORD_FRAME). The core-fit run must reproduce the 2026-10-20 S1 scores
(experiments/annotation_stages/side/s1_matched_draws) exactly; MSH6 positional KS must be
Table S2's 0.762.

Reported per gene and macro: AUC (mean of ten per-seed AUCs, drivers high, never flipped)
of ensemble-mean RMSD, dRMSF and positional KS in both frames; the frame difference with
a paired 95% interval over residue positions (stratified driver-only / passenger-only /
mixed, 2,000 replicates, the S1 resamples); per-variant Spearman rho between frames of
the ten-draw mean scores.

    PYTHONPATH="." python experiments/s1_frame_sensitivity/frame_sensitivity.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT)]
RUN = ROOT / "experiments/annotation_stages"
OUT = HERE / "results"
S1_REFERENCE = RUN / "side/s1_matched_draws/variant_seed_scores.csv.gz"
USALIGN = {"POLE": "cache_usalign", "PTEN": "cache_usalign", "VHL": "cache_usalign",
           "MSH2": "cache_usalign", "MSH6": "cache_usalign_scored_chain", "PIK3CA": "cache_usalign"}
FRAMES = {"core_fit": {}, "usalign": USALIGN}
DESC = {"RMSD": "rmsd", "ΔRMSF": "delta_rmsf", "Positional KS": "posks"}


def load_s1():
    spec = importlib.util.spec_from_file_location("s1_matched_draws", ROOT / "experiments/s1_matched_draws/s1_matched_draws.py")
    s1 = importlib.util.module_from_spec(spec)
    sys.modules["s1_matched_draws"] = s1
    spec.loader.exec_module(s1)
    s1.COHORT = RUN / "side/annotation_cohort_25seed.csv"   # POLE seeds 1-25, P356S out (annotation_stages run)
    s1.CACHE = ROOT / "drift/cache"
    s1.FP = "97f1cdbfda8e"
    return s1


def scores_for(s1, frame):
    s1.COORD_FRAME = FRAMES[frame]
    s1.OUT = OUT / frame
    s1.OUT.mkdir(parents=True, exist_ok=True)
    s1.SCORES = s1.OUT / "variant_seed_scores.csv.gz"
    t0 = time.monotonic()
    scores = s1.calculate()
    print(f"{frame}: {len(scores)} variant x seed rows in {time.monotonic() - t0:.0f}s", flush=True)
    return scores


def main():
    started = time.monotonic()
    s1 = load_s1()
    scores = {f: scores_for(s1, f) for f in FRAMES}
    keys = ["gene", "mutation", "seed"]
    core = scores["core_fit"].set_index(keys).sort_index()
    us = scores["usalign"].set_index(keys).sort_index()
    assert core.index.equals(us.index) and (core.label == us.label).all()
    assert np.array_equal(core.ks.to_numpy(), us.ks.to_numpy()), "pLDDT KS cannot depend on the frame"

    # the core-fit run is the published S1: it must reproduce the 2026-10-20 scores exactly
    ref = pd.read_csv(S1_REFERENCE, float_precision="round_trip").set_index(keys).sort_index()
    assert ref.index.equals(core.index)
    err = {c: float(np.abs(ref[c].to_numpy() - core[c].to_numpy()).max()) for c in ["rmsd", "delta_rmsf", "posks", "rmsf_mut", "ks"]}
    assert max(err.values()) < 1e-12, err
    print("core-fit run reproduces the published S1 scores:", err, flush=True)

    # per-seed AUCs, both frames, and paired position-cluster resamples of the frame difference
    order = s1.variant_order(scores["core_fit"])
    mats = {f: s1.matrices(scores[f], order) for f in FRAMES}
    idx = s1.resamples(mats["core_fit"], order, "position")
    rows, boots = [], {}
    for g in s1.GENES:
        y = mats["core_fit"][g][0]
        for label in DESC:
            A, B = (mats[f][g][1][label] for f in FRAMES)
            auc = {f: float(np.mean([s1.auc(y, M[:, j]) for j in range(M.shape[1])])) for f, M in zip(FRAMES, (A, B))}
            d = np.array([np.mean([s1.auc(y[i], A[i, j]) - s1.auc(y[i], B[i, j]) for j in range(A.shape[1])])
                          if 0 < y[i].sum() < len(i) else np.nan for i in idx[g]])
            boots[(g, label)] = d
            ma, mb = A.mean(1), B.mean(1)
            rows.append({"gene": g, "descriptor": label, "auc_core_fit": auc["core_fit"], "auc_usalign": auc["usalign"],
                         "delta_core_minus_usalign": auc["core_fit"] - auc["usalign"],
                         "delta_lo": float(np.nanpercentile(d, 2.5)), "delta_hi": float(np.nanpercentile(d, 97.5)),
                         "spearman_variant_scores_between_frames": float(spearmanr(ma, mb).statistic),
                         "n_driver": int(y.sum()), "n_passenger": int((y == 0).sum())})
    table = pd.DataFrame(rows)
    for label in DESC:
        t = table[table.descriptor == label]
        mb = np.mean([boots[(g, label)] for g in s1.GENES], axis=0)
        rows.append({"gene": "Macro", "descriptor": label, "auc_core_fit": t.auc_core_fit.mean(),
                     "auc_usalign": t.auc_usalign.mean(),
                     "delta_core_minus_usalign": t.auc_core_fit.mean() - t.auc_usalign.mean(),
                     "delta_lo": float(np.nanpercentile(mb, 2.5)), "delta_hi": float(np.nanpercentile(mb, 97.5))})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "frame_comparison.csv", index=False)

    # Table S2 check (published core-fit values) and the paragraph's summary quantities
    published = pd.read_csv(RUN / "side/s1_matched_draws/auc_matched_draws.csv")
    pub = published.set_index(["gene", "descriptor"]).auc
    s2 = {"RMSD": "RMSD", "ΔRMSF": "ΔRMSF", "Positional KS": "Positional KS"}
    for r in table.itertuples():
        assert abs(r.auc_core_fit - pub.loc[(r.gene, s2[r.descriptor])]) < 1e-12, (r.gene, r.descriptor)
    msh6 = table[(table.gene == "MSH6") & (table.descriptor == "Positional KS")].iloc[0]
    assert round(msh6.auc_core_fit, 3) == 0.762, msh6.auc_core_fit
    genes5 = [g for g in s1.GENES if g != "MSH6"]
    per = table[table.gene.isin(s1.GENES)]
    five = per[per.gene.isin(genes5)]
    summary = {
        "cohort": "1,797 annotation variants (POLE seeds 1-25, P356S out; MSH2 G162R excluded); ten N=90 draws, seeds 42-51",
        "frames": {"core_fit": "cache_corefit (published S1)", "usalign": USALIGN},
        "core_fit_reproduces_published_S1_scores_max_abs_diff": err,
        "spearman_min_rmsd_drmsf_all_genes": float(per[per.descriptor.isin(["RMSD", "ΔRMSF"])].spearman_variant_scores_between_frames.min()),
        "spearman_min_posks_all_genes": float(per[per.descriptor == "Positional KS"].spearman_variant_scores_between_frames.min()),
        "spearman_min_posks_five_genes": float(five[five.descriptor == "Positional KS"].spearman_variant_scores_between_frames.min()),
        "max_abs_dauc_rmsd_drmsf_all_genes": float(per[per.descriptor.isin(["RMSD", "ΔRMSF"])].delta_core_minus_usalign.abs().max()),
        "max_abs_dauc_rmsd_drmsf_five_genes": float(five[five.descriptor.isin(["RMSD", "ΔRMSF"])].delta_core_minus_usalign.abs().max()),
        "max_abs_dauc_posks_five_genes": float(five[five.descriptor == "Positional KS"].delta_core_minus_usalign.abs().max()),
        "msh6_posks_core_fit": float(msh6.auc_core_fit), "msh6_posks_usalign": float(msh6.auc_usalign),
        "msh6_posks_delta": float(msh6.delta_core_minus_usalign),
        "msh6_posks_delta_ci": [float(msh6.delta_lo), float(msh6.delta_hi)],
        "runtime_seconds": round(time.monotonic() - started, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
