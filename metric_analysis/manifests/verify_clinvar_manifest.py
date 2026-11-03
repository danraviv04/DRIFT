#!/usr/bin/env python
"""Check clinvar_source_manifest.csv against the ClinVar exports the pipeline reads.

SI Note S3 identifies each ClinVar snapshot by the SHA-256 of its archived file, so
the manifest must name every annotation gene's export and match it byte for byte.
For each gene with a ``clinvar_csv`` in both protein_configs.json files, this
recomputes the SHA-256 and the row count (as read by read_clinvar_csv) and compares
them with the manifest. It exits non-zero on any mismatch, missing gene, or
disagreement between the two configs. Run it before any cohort rebuild or rerun.

    PYTHONPATH=. python \\
        metric_analysis/manifests/verify_clinvar_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from metric_analysis.pipeline.clinvar import read_clinvar_csv  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "clinvar_source_manifest.csv"
CONFIGS = [ROOT / "metric_analysis" / "protein_configs.json",
           ROOT / "drift" / "protein_configs.json"]


def config_paths():
    """{gene: path} from the pipeline config, checked against the DRIFT config."""
    pipeline = {g: c["clinvar_csv"] for g, c in json.loads(CONFIGS[0].read_text()).items()
                if isinstance(c, dict) and c.get("clinvar_csv")}
    drift_text = CONFIGS[1].read_text()
    missing = [p for p in pipeline.values() if p not in drift_text]
    return pipeline, missing


def main():
    problems = []
    paths, not_in_drift = config_paths()
    problems += [f"{p}: in metric_analysis config but not in the DRIFT config" for p in not_in_drift]
    m = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False).set_index("gene")
    for gene, rel in sorted(paths.items()):
        f = ROOT / rel
        if gene not in m.index:
            problems.append(f"{gene}: missing from the manifest ({rel})")
            continue
        row = m.loc[gene]
        if row.source_file != rel:
            problems.append(f"{gene}: manifest names {row.source_file}, config reads {rel}")
        if not f.is_file():
            problems.append(f"{gene}: {rel} does not exist")
            continue
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        rows = len(read_clinvar_csv(f))
        ok = sha == row.sha256 and str(rows) == row.rows
        print(f"  {gene:7s} {'OK ' if ok else 'BAD'} {sha[:12]}  {rows:4d} rows  {rel}")
        if sha != row.sha256:
            problems.append(f"{gene}: sha256 {sha} != manifest {row.sha256}")
        if str(rows) != row.rows:
            problems.append(f"{gene}: {rows} rows != manifest {row.rows}")
    extra = sorted(set(m.index) - set(paths))
    problems += [f"{g}: in the manifest but no config reads a ClinVar file for it" for g in extra]
    if problems:
        print("\n".join(["FAILED:"] + [f"  - {p}" for p in problems]))
        sys.exit(1)
    print(f"clinvar_source_manifest.csv matches all {len(paths)} ClinVar exports.")


if __name__ == "__main__":
    main()
