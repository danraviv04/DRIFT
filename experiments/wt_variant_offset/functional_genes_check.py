#!/usr/bin/env python
"""TP53 and KRAS: is their WT offset from the variants like MSH2/PIK3CA's? (2026-10-15)

Two parts:
  1. Effect (local pLDDT CSVs): region_profile.py's test. Per residue, where the WT mean
     pLDDT falls among 150 random variants' means; the fraction of residues where the WT is
     outside the variants' central 95%. MSH2 0.63, PIK3CA 0.49 (offset); VHL/PTEN ~0.10
     (one-template swap, minor); POLE/MSH6 0.00 (identical templates).
  2. Cause (drive, read-only, one archive at a time): each template's query coverage
     (len(queryIndices) / chain length) for the WT and a few variants, from the job-level
     _data.json. A WT-only missing template covering most of the chain is the MSH2 pattern.

    PYTHONPATH="." python experiments/wt_variant_offset/functional_genes_check.py [--drive]
"""
from __future__ import annotations
import os

import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
SAMPLE = re.compile(r"seed-(\d+)_sample-(\d+)")
GENES = {  # pLDDT dir, WT CSV, rows of one copy of the scored chain
    "TP53": ("PLDDTs/TP53", "../_wt_legacy_2026-10-15/P53_WT_plddt_values.csv", slice(0, 393)),
    "KRAS": ("PLDDTs/KRAS", "KRAS_WT_plddt_values.csv", slice(None)),
}
ARCHIVES = {
    "TP53": ["TP53/P53_WT_with_ZN_and_DNA_data", "TP53/P53_A119D_with_ZN_and_DNA_data",
             "TP53/P53_R175H_with_ZN_and_DNA_data"],
    "KRAS": ["KRAS4B_Mutations/KRAS_WT_data", "KRAS4B_Mutations/KRAS_G12D_data",
             "KRAS4B_Mutations/KRAS_G12V_data"],
}


def load(path, rows):
    f = pd.read_csv(path, index_col=0)
    return f[[c for c in f.columns if SAMPLE.fullmatch(c)]].to_numpy(float)[rows]


def effect():
    summ = []
    for g, (d, wt_name, rows) in GENES.items():
        files = sorted(p for p in (REPO / d).glob("*_plddt_values.csv")
                       if p.name != wt_name and "_WT" not in p.name.upper())
        files = [files[i] for i in np.random.default_rng(0).choice(len(files), min(150, len(files)), replace=False)]
        wt = load(REPO / d / wt_name, rows)
        vm = np.stack([x.mean(1) for f in files if (x := load(f, rows)).shape == wt.shape])
        wm = wt.mean(1)
        pct = 100 * (vm < wm).mean(0)
        edge = (pct <= 2.5) | (pct >= 97.5)
        summ.append({"gene": g, "residues": len(wm), "n_variants": len(vm), "wt_samples": wt.shape[1],
                     "frac_wt_outside_variant_95": round(float(edge.mean()), 3),
                     "frac_wt_above_all_variants": round(float((pct == 100).mean()), 3),
                     "frac_wt_below_all_variants": round(float((pct == 0).mean()), 3)})
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "functional_genes_wt_offset.csv", index=False)
    print(s.to_string(index=False))


def coverage():
    rows = []
    for g, names in ARCHIVES.items():
        for n in names:
            src = Path(os.environ.get("DRIFT_ARCHIVE_ROOT", "$DRIFT_ARCHIVE_ROOT")) / f"{n}.tar.gz"
            with tarfile.open(src, "r|gz") as t:
                for m in t:
                    if m.name.endswith("_data.json") and m.name.count("/") <= 2 and "/._" not in m.name:
                        d = json.load(t.extractfile(m))
                        break
                else:
                    raise SystemExit(f"{src}: no job-level _data.json")
            p = next(next(iter(s.values())) for s in d["sequences"] if "protein" in s)
            L = len(p["sequence"])
            for tpl in p.get("templates") or []:
                first = tpl["mmcif"].split("\n", 1)[0]
                rows.append({"gene": g, "job": d["name"], "template": first[5:9].upper(), "chain_len": L,
                             "covered": len(tpl["queryIndices"]),
                             "coverage": round(len(tpl["queryIndices"]) / L, 3)})
            print(f"  read {src.name}", flush=True)
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "functional_genes_template_coverage.csv", index=False)
    print(c.to_string(index=False))


if __name__ == "__main__":
    effect()
    if "--drive" in sys.argv:
        coverage()
