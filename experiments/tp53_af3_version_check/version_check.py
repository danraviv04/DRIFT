#!/usr/bin/env python
"""TP53: do the seeds extended on a second machine (AF3 3.0.3, A40) differ from the cluster ones (3.0.1)?

The two groups keep their machine names in the code and outputs: "athena" is the HPC
cluster (AF3 3.0.1), "golgi" a server without Slurm (AF3 3.0.3, A40 GPUs).

Every TP53 variant has seeds 1-20 from the cluster (3.0.1). Seeds 21-25 came from
golgi (3.0.3/A40) for the 296 variants in golgi_ls_2026-10-15.txt and from
the cluster for the rest. Between 3.0.1 and 3.0.3, v3.0.2 moved flash attention and
SwiGLU to Tokamax kernels and stopped deduplicating user-supplied unpaired MSAs
(the extension JSONs carry the original MSA), so a small numeric offset is
plausible. If one exists it adds to every golgi variant's KS against the WT.

Per variant, residueKS (the canonical per-residue KS, mean over residues) on
equal 25-vs-25 splits, so the KS sample-size floor is the same everywhere:
  cross  = seeds 16-20 vs 21-25      (crosses the extension boundary)
  within = mean of seeds 1-5 vs 6-10 and 11-15 vs 16-20   (same original run)
  excess = cross - within
and the consequence for the paper's quantity, the KS against the WT:
  wt_shift = KS(variant 21-25, WT extension) - KS(variant 16-20, WT 16-20)
plus the plain level shift, mean pLDDT(21-25) - mean pLDDT(1-20).
Then golgi vs cluster on each: medians, Mann-Whitney, bootstrap CI of the
difference in medians. E221Q, E221V and E224A are known cluster extensions;
bare folders in the golgi listing do not override that provenance. Other
folder-only extensions remain excluded.

    PYTHONPATH=. python \
        experiments/tp53_af3_version_check/version_check.py [--plddt-dir PLDDTs/TP53]

--golgi-random N labels N random variants "golgi" instead (smoke test on a gene
with one AF3 version, e.g. --plddt-dir PLDDTs/MutS_alpha_MSH6_GT --prefix
MutS_alpha_MSH6_GT_): both groups should then look the same.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from metric_analysis.pipeline.library.descriptor_metrics import residue_ks

HERE = Path(__file__).resolve().parent
SEED_RE = re.compile(r"seed-(\d+)_sample-(\d+)")
# Extension provenance confirmed by Dan on 2026-10-18.
ATHENA_EXTENSIONS = frozenset({"E221Q", "E221V", "E224A"})


def golgi_variants():
    text = (HERE / "golgi_ls_2026-10-15.txt").read_text()
    tars = set(re.findall(r"P53_([A-Z]\d+[A-Z])_with_ZN_and_DNA_seeds_21-25_data\.tar\.gz", text))
    folders = set(re.findall(r"P53_([A-Z]\d+[A-Z])_with_ZN_and_DNA_seeds_21-25_data(?!\.tar)\b", text))
    return tars, folders


def load(csv):
    df = pd.read_csv(csv, index_col=0)
    seeds = {}
    for col in df.columns:
        m = SEED_RE.search(col)
        if m:
            seeds.setdefault(int(m.group(1)), []).append(col)
    return df, seeds


def block(df, seeds, lo, hi):
    cols = [c for s in range(lo, hi + 1) for c in seeds.get(s, [])]
    return df[cols].to_numpy(dtype=np.float64), len(cols)


def ks(a, b):
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    return residue_ks(a[ok], b[ok])


def compare(frame, col, seed):
    g = frame.loc[frame.group == "golgi", col].dropna().to_numpy()
    a = frame.loc[frame.group == "athena", col].dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boot = [np.median(rng.choice(g, g.size)) - np.median(rng.choice(a, a.size)) for _ in range(2000)]
    return {
        "n_golgi": int(g.size), "n_athena": int(a.size),
        "median_golgi": float(np.median(g)), "median_athena": float(np.median(a)),
        "diff_medians": float(np.median(g) - np.median(a)),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "mannwhitney_p": float(stats.mannwhitneyu(g, a).pvalue),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plddt-dir", default="PLDDTs/TP53")
    ap.add_argument("--prefix", default="P53_")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--golgi-random", type=int, help="smoke test: random 'golgi' group of this size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csvs = sorted(Path(args.plddt_dir).glob(f"{args.prefix}*_plddt_values.csv"))
    name = lambda p: re.match(rf"{re.escape(args.prefix)}(WT|[A-Z]\d+[A-Z])(_|$)", p.name)
    # The primary WT, not an extra pool such as P53_WT_..._again.
    wt_csvs = [p for p in csvs if name(p) and name(p).group(1) == "WT"]
    wt_csv = sorted(wt_csvs, key=lambda p: ("again" in p.name.lower(), len(p.name)))[0]
    print(f"WT: {wt_csv.name} (of {[p.name for p in wt_csvs]})")
    wt, wt_seeds = load(wt_csv)
    wt_ext_lo = 41 if 41 in wt_seeds else 21          # TP53 WT extension = seeds 41-45
    wt_ext, n_wt_ext = block(wt, wt_seeds, wt_ext_lo, wt_ext_lo + 4)
    wt_16_20, _ = block(wt, wt_seeds, 16, 20)
    assert n_wt_ext == 25, f"WT extension has {n_wt_ext} predictions, expected 25 ({wt_csv.name})"

    tars, folders = golgi_variants()
    variants = {name(p).group(1): p for p in csvs if name(p) and name(p).group(1) != "WT"}
    if args.golgi_random:
        rng = np.random.default_rng(args.seed)
        tars, folders = set(rng.choice(sorted(variants), args.golgi_random, replace=False)), set()
    athena_confirmed = set() if args.golgi_random else ATHENA_EXTENSIONS & set(variants)
    if athena_confirmed & tars:
        raise ValueError(f"Conflicting extension provenance: {sorted(athena_confirmed & tars)}")
    excluded_folders = folders - athena_confirmed
    missing = sorted(tars - set(variants))

    rows = []
    for v, csv in sorted(variants.items()):
        if v in excluded_folders:
            continue
        df, seeds = load(csv)
        blocks = {k: block(df, seeds, lo, hi) for k, (lo, hi) in
                  {"1_5": (1, 5), "6_10": (6, 10), "11_15": (11, 15), "16_20": (16, 20), "21_25": (21, 25)}.items()}
        if any(n != 25 for _, n in blocks.values()):
            rows.append({"variant": v, "group": "golgi" if v in tars else "athena",
                         "error": "block sizes " + ",".join(f"{k}={n}" for k, (_, n) in blocks.items())})
            continue
        b = {k: m for k, (m, _) in blocks.items()}
        within = (ks(b["1_5"], b["6_10"]) + ks(b["11_15"], b["16_20"])) / 2
        cross = ks(b["16_20"], b["21_25"])
        old = np.hstack([b["1_5"], b["6_10"], b["11_15"], b["16_20"]])
        rows.append({
            "variant": v, "group": "golgi" if v in tars else "athena",
            "within": within, "cross": cross, "excess": cross - within,
            "wt_ks_old": ks(wt_16_20, b["16_20"]), "wt_ks_new": ks(wt_ext, b["21_25"]),
            "wt_shift": ks(wt_ext, b["21_25"]) - ks(wt_16_20, b["16_20"]),
            "level_shift": float(np.nanmean(b["21_25"]) - np.nanmean(old)),
        })

    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_smoke_{Path(args.plddt_dir).name}" if args.golgi_random else ""
    frame.to_csv(out / f"per_variant{tag}.csv", index=False)
    errors = frame[frame.get("error", pd.Series(dtype=str)).notna()] if "error" in frame else frame.iloc[:0]
    ok = frame.drop(errors.index)
    summary = {
        "plddt_dir": args.plddt_dir, "wt_csv": wt_csv.name, "wt_extension_seeds": [wt_ext_lo, wt_ext_lo + 4],
        "n_variants": int(len(ok)), "n_errors": int(len(errors)),
        "golgi_listed_but_missing": missing, "golgi_folders_excluded": sorted(excluded_folders),
        "athena_extensions_confirmed": sorted(athena_confirmed),
        "athena_extension_confirmation": "Dan, 2026-10-18" if athena_confirmed else None,
        **{col: compare(ok, col, args.seed) for col in ("within", "cross", "excess", "wt_shift", "level_shift")},
    }
    (out / f"summary{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if len(errors):
        print(errors.to_string(index=False))


if __name__ == "__main__":
    main()
