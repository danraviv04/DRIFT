"""Thresholded metrics at AlphaMissense's own published operating point (2026-10-17, read-through C6).

Fig. 2h,i transfer a score VALUE chosen on the training genes; on the combined score that cut lands
in different places per gene (MSH6 too high, POLE too low). Here no threshold is learned:
- AM alone: driver if AM > 0.564 (Cheng et al. 2023, likely pathogenic).
- AM + KS-score (gz_mean, the paper's rule): in each gene, the same NUMBER of variants AM calls at
  0.564 are called driver, taken in the combined score's order. So AM's threshold sets how many,
  the combination only sets which; differences are ranking at AM's operating point. No labels used.
N = 90, annotation cohort, ten subsampling seeds (metrics averaged over seeds); paired bootstrap
(seeds + class-stratified variants, 1,000 reps) for the macro deltas.

    PYTHONPATH="." python analyses/annotation/am_operating_point/am_anchored.py
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.annotation.am_combination_v2.common import GENES, load_frames
from analyses.annotation.simple_fusion_benchmark import fusion, size_sweep


def _stable_rng(seed, *parts):
    """Seeded generator per (seed, parts); as drift.canonical_afm._stable_rng (removed 2026-11-03)."""
    token = "|".join(map(str, parts)).encode()
    return np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(token)]))

OUT = Path(__file__).resolve().parent / "results"
CUT = 0.564
REPS = 1000


def calls_topk(score, k):
    order = np.argsort(-score, kind="mergesort")
    c = np.zeros(len(score), bool)
    c[order[:k]] = True
    return c


def m(y, c):
    tp, fp = (c & y).sum(), (c & ~y).sum()
    fn, tn = (~c & y).sum(), (~c & ~y).sum()
    rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) or 1.0
    return {"F1": f1, "Acc": (tp + tn) / len(y), "bal.acc": (rec + spec) / 2,
            "MCC": (tp * tn - fp * fn) / den, "recall": rec, "precision": prec}


def main():
    frames = load_frames()
    folds = size_sweep.make_folds(frames[0])["LOGO"]
    combo = []
    for f in frames:
        rec = [r for r in fusion.build(f, {"LOGO": folds}, arms={"AM+resKS": fusion.ARMS["AM+resKS"]})
               if r["method"] == "gz_mean" and r["protocol"] == "LOGO"][0]
        combo.append(rec["score"])
    combo = np.stack(combo)
    f0 = frames[0]
    am = f0.afm_score.to_numpy(float)
    genes, y_all = f0.gene.to_numpy(), f0.label.to_numpy(bool)
    rows, per = [], {}
    for g in GENES:
        idx = np.flatnonzero(genes == g)
        y = y_all[idx]
        c_am = am[idx] > CUT
        k = int(c_am.sum())
        ma = m(y, c_am)
        mc = pd.DataFrame([m(y, calls_topk(combo[s, idx], k)) for s in range(len(frames))]).mean().to_dict()
        per[g] = (y, idx, k)
        rows.append({"gene": g, "n": len(y), "prevalence": round(y.mean(), 3), "called_rate": round(k / len(y), 3),
                     **{f"AM_{key}": v for key, v in ma.items()}, **{f"combo_{key}": v for key, v in mc.items()}})
    t = pd.DataFrame(rows)
    keys = ["F1", "Acc", "bal.acc", "MCC"]
    macro = {"gene": "MACRO", **{f"AM_{k}": t[f"AM_{k}"].mean() for k in keys},
             **{f"combo_{k}": t[f"combo_{k}"].mean() for k in keys}}
    t = pd.concat([t, pd.DataFrame([macro])], ignore_index=True)
    # paired bootstrap of the macro deltas
    rng = _stable_rng(20261017, "am_operating_point", "bootstrap")
    draws = {k: [] for k in keys}
    for _ in range(REPS):
        sd = rng.choice(len(frames), len(frames), replace=True)
        dg = {k: [] for k in keys}
        for g in GENES:
            y, idx, k_ = per[g]
            pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
            b = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
            yb = y[b]
            ca = am[idx][b] > CUT
            kb = int(ca.sum())
            ma = m(yb, ca)
            mc = pd.DataFrame([m(yb, calls_topk(combo[s, idx][b], kb)) for s in sd]).mean()
            for k in keys:
                dg[k].append(mc[k] - ma[k])
        for k in keys:
            draws[k].append(np.mean(dg[k]))
    ci = {k: np.percentile(draws[k], [2.5, 97.5]) for k in keys}
    OUT.mkdir(exist_ok=True)
    t.to_csv(OUT / "am_anchored_operating_point.csv", index=False)
    pd.DataFrame([{"metric": k, "delta": macro[f"combo_{k}"] - macro[f"AM_{k}"], "ci_lo": ci[k][0], "ci_hi": ci[k][1],
                   "p_one_sided": (1 + sum(d <= 0 for d in draws[k])) / (REPS + 1)} for k in keys]
                 ).to_csv(OUT / "am_anchored_macro_delta.csv", index=False)
    show = ["gene", "prevalence", "called_rate"] + [f"{a}_{k}" for k in keys for a in ("AM", "combo")]
    print(t[show].round(3).to_string(index=False))
    print(pd.read_csv(OUT / "am_anchored_macro_delta.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
