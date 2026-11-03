#!/usr/bin/env python
"""Tabulate the scored chain's templates/MSA per archive from results/provenance/*.json
(written by archive_provenance.py), with the residue offset summary per gene.

    python experiments/wt_variant_offset/template_table.py
"""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "results"
CHAIN = {"PIK3CA": "A", "MutS_alpha_MSH2": "A", "MutS_alpha_WT": "A", "MutS_alpha_MSH6_GT": "B", "POLE": "A", "PTEN": "A", "VHL": "A"}
GENE = [("MutS_alpha_MSH6_GT", "MSH6"), ("MutS_alpha", "MSH2"), ("PIK3CA", "PIK3CA"), ("POLE", "POLE"), ("PTEN", "PTEN"), ("VHL", "VHL")]
rows = []
for f in sorted((R / "provenance").glob("*.json")):
    r = json.loads(f.read_text())
    if "data_json" not in r:
        continue
    gene = next(g for p, g in GENE if f.stem.startswith(p))
    chain = "B" if gene == "MSH6" else "A"
    c = next(c for c in r["data_json"]["chains"] if c["id"] == chain)
    rows.append({"gene": gene, "job": r["data_json"]["name"], "is_wt": "WT" in f.stem, "chain": chain,
                 "templates": " ".join(c.get("template_ids", [])), "templates_sha": c.get("templates_sha"),
                 "unpaired_msa": c["unpaired"]["n_seqs"], "paired_msa": c["paired"].get("n_seqs"),
                 "run_start_utc": r.get("mtime_min", "")[:16]})
t = pd.DataFrame(rows).sort_values(["gene", "is_wt"], ascending=[True, False])
prof = pd.read_csv(R / "wt_vs_variant_profile_summary.csv").set_index("gene")
t["wt_outside_variant95_pct_residues"] = t.gene.map(lambda g: round(100 * prof.loc[g, "frac_wt_outside_variant_95"]))
t.to_csv(R / "template_comparison.csv", index=False)
print(t.to_string(index=False))
