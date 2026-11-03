#!/usr/bin/env python
"""How many of the 1,174 TP53 variants are single-nucleotide changes of TP53 itself?

The TP53 batch was generated with af3_jobs.mutations.min_nt_distance == 1, which takes
the minimum over ALL codons of the wild-type and mutant amino acids. A substitution can
therefore be "one nucleotide away" from some codon of the wild-type residue without being
a single-nucleotide variant of TP53's actual coding sequence. This script counts both, for
the Table S1b wording (Supplementary Note S3).

Inputs:
- NM_000546.6_cds.fa: TP53 CDS from NCBI efetch (db=nuccore, id=NM_000546.6,
  rettype=fasta_cds_na), fetched 2026-10-19. Its translation must equal P04637
  (af3_jobs/seq_cache/P04637.fasta); the script asserts this.
- PLDDTs/TP53/P53_<wt><pos><mut>_plddt_values.csv: the modelled variant set.

Outputs (this folder): variants.csv (one row per variant), summary.json.

    PYTHONPATH=. python \\
        experiments/tp53_snv_reachability/tp53_snv_reachability.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from af3_jobs.mutations import min_nt_distance  # noqa: E402

CDS = HERE / "NM_000546.6_cds.fa"
UNIPROT = ROOT / "af3_jobs" / "seq_cache" / "P04637.fasta"
PLDDTS = ROOT / "PLDDTs" / "TP53"
AAS = "ACDEFGHIKLMNPQRSTVWY"


def fasta(path):
    return "".join(l.strip() for l in path.read_text().splitlines() if not l.startswith(">"))


def snv_aas(codon):
    """Amino acids (and '*') one nucleotide away from this codon."""
    return {str(Seq(codon[:i] + b + codon[i + 1:]).translate())
            for i in range(3) for b in "ACGT" if b != codon[i]}


def main():
    cds = fasta(CDS)
    prot = str(Seq(cds).translate()).rstrip("*")
    if prot != fasta(UNIPROT):
        raise ValueError("NM_000546.6 CDS does not translate to P04637")
    rows = [m.groups() for f in sorted(PLDDTS.iterdir())
            if (m := re.match(r"P53_([A-Z])(\d+)([A-Z])_plddt_values\.csv$", f.name))]
    d = pd.DataFrame(rows, columns=["wt", "pos", "mut"]).astype({"pos": int})
    if not all(prot[p - 1] == w for w, p in zip(d.wt, d.pos)):
        raise ValueError("variant WT residue does not match P04637")
    d["codon"] = [cds[3 * (p - 1):3 * p] for p in d.pos]
    d["any_codon_1nt"] = [min_nt_distance(w, m) == 1 for w, m in zip(d.wt, d.mut)]
    d["actual_codon_snv"] = [m in snv_aas(c) for m, c in zip(d.mut, d.codon)]
    d.to_csv(HERE / "variants.csv", index=False)

    per = d.groupby("pos")
    missing_any = sum(len({a for a in AAS if a != g.wt.iloc[0] and min_nt_distance(g.wt.iloc[0], a) == 1}
                          - set(g.mut)) for _, g in per)
    missing_actual = sum(len({a for a in snv_aas(g.codon.iloc[0]) if a not in ("*", g.wt.iloc[0])}
                             - set(g.mut)) for _, g in per)
    summary = {
        "n_variants": len(d),
        "n_positions": int(d.pos.nunique()),
        "all_within_one_nt_of_some_wt_codon": bool(d.any_codon_1nt.all()),
        "n_actual_codon_snv": int(d.actual_codon_snv.sum()),
        "share_actual_codon_snv": round(float(d.actual_codon_snv.mean()), 4),
        "positions_all_actual_snv": int((per.actual_codon_snv.all()).sum()),
        "substitutions_per_position": {int(k): int(v) for k, v in per.size().value_counts().sort_index().items()},
        "missing_any_codon_1nt_substitutions": int(missing_any),
        "missing_actual_codon_missense_snvs": int(missing_actual),
        "non_snv_by_wt_residue": d[~d.actual_codon_snv].wt.value_counts().to_dict(),
        "cds": {"file": CDS.name, "accession": "NM_000546.6",
                "sha256": hashlib.sha256(CDS.read_bytes()).hexdigest(), "fetched": "2026-10-19"},
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
