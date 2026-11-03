#!/usr/bin/env python
"""Point the canonical cohort's eight 32-seed POLE rows at the 25-seed copies.

The canonical route (metric_analysis.pipeline.library.canonical_cohort, and through
it figlib.cohort, canonical_afm, cohort_size_sweep and the strict-manifest builder)
reads each variant's pLDDT path and counts from
metric_analysis/structure_metrics/POLE/POLE_plddt_distribution_summary.csv, not from
the strict manifest. This rewrites the eight rows of that summary with the pipeline's
own builder (structure_metrics.build_plddt_distribution_tables), after first checking
that the builder reproduces the eight existing 160-prediction rows. Every other row,
and the column order and float text of the file, are left as they are.

    PYTHONPATH="." python analyses/annotation/stages/update_canonical_summary.py
"""
from pathlib import Path
import csv
import io

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MAIN_ROOT = str(ROOT)  # source_file paths in the summary are recorded absolute
SUMMARY = ROOT / "metric_analysis/structure_metrics/POLE/POLE_plddt_distribution_summary.csv"
WT = ROOT / "PLDDTs/POLE/POLE_268-471_WT_plddt_values.csv"
EIGHT = ("D275A", "D275G", "E277G", "P286H", "P286L", "P286R", "P286S", "T278K")


def build(path):
    from metric_analysis.pipeline.library.structure_metrics import build_plddt_distribution_tables
    summary, _ = build_plddt_distribution_tables([WT, path], wt_file=WT, known_mutations=list(EIGHT))
    assert len(summary) == 1, summary
    return summary.iloc[0]


def main():
    text = SUMMARY.read_text()
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    table = pd.read_csv(SUMMARY)
    numeric = [c for c in table.columns if c not in ("mutation_code", "source_file", "wt_source_file",
                                                     "status", "mutation_type", "label")]
    done = 0
    for i, row in enumerate(rows[1:], 1):
        code = row[header.index("mutation_code")]
        if code not in EIGHT:
            continue
        recorded = table.iloc[i - 1]
        assert recorded.mutation_code == code and int(recorded.n_mutant_structures) == 160
        old_file = ROOT / "PLDDTs/POLE" / Path(recorded.source_file).name
        new_file = ROOT / "PLDDTs/POLE_25seed" / old_file.name
        # 1. the builder reproduces the recorded 160-prediction row
        again = build(old_file)
        for c in numeric:
            if c in again.index:
                assert np.isclose(float(again[c]), float(recorded[c]), rtol=1e-12, atol=1e-15), (code, c)
        # 2. the 25-seed row, written in the file's own conventions
        fresh = build(new_file)
        assert int(fresh.n_mutant_structures) == 125 and int(fresh.n_wt_structures) == 125
        for c in header:
            if c == "source_file":
                row[header.index(c)] = f"{MAIN_ROOT}/PLDDTs/POLE_25seed/{new_file.name}"
            elif c in ("mutation_code", "wt_source_file", "mutation_type", "label", "status"):
                continue
            else:
                value = fresh[c]
                row[header.index(c)] = repr(int(value)) if c.startswith("n_") or c.endswith(
                    ("_n_residues", "_samples")) else repr(float(value))
        assert row[header.index("status")] == "ok"
        done += 1
        print(f"{code}: reproduced the 160-prediction row; rewrote it from 125 predictions")
    assert done == len(EIGHT)
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    SUMMARY.write_text(out.getvalue())
    after = pd.read_csv(SUMMARY)
    keep = ~after.mutation_code.isin(EIGHT)
    pd.testing.assert_frame_equal(after[keep], table[keep], check_exact=True)
    assert (after.loc[~keep, "n_mutant_structures"] == 125).all()


if __name__ == "__main__":
    main()
