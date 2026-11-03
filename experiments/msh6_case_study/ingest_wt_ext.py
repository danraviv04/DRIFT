#!/usr/bin/env python
"""Ingest the MSH6 G:T WT seed extension (seeds 26-75) into a 75-seed WT pLDDT table.

The extension was built with af3_jobs.seed_extension from the WT's own MSA-ready
_data.json (run inference-only), so it must be the same job in everything but
modelSeeds. This checks that, archive by archive, before using a single value:

  1. each archive streams cleanly (gzip/tar intact) and holds exactly the 25
     seed-sample predictions of its chunk (seeds a-b x samples 0-4);
  2. the _data.json inside equals the original WT _data.json with modelSeeds
     removed (same sequences, MSAs, templates) -- important since the MSH2/PIK3CA
     WTs turned out to differ from their variants in templates;
  3. no seed overlaps the original 25 (1-25) or another chunk.

Then it extracts C-alpha pLDDT with fast_plddt (validated identical to the
pipeline) for chains A and B (2294 residues), and writes
  wt_75seeds/MutS_alpha_MSH6_GT_WT_75seeds_plddt_values.csv   (2294 x 375)
next to this script. The pipeline's own WT CSV (25 seeds) is NOT touched: every
variant has 25 seeds, so the main comparisons stay matched. This table feeds the
seed-matched WT-vs-WT background only.

    PYTHONPATH=. python experiments/msh6_case_study/ingest_wt_ext.py
"""
from __future__ import annotations
import os

import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from metric_analysis.pipeline.library.fast_plddt import parse_cif_plddt

EXT_DIR = Path(os.environ.get("DRIFT_ARCHIVE_ROOT", "$DRIFT_ARCHIVE_ROOT")) / "MutS_alpha_MSH6_GT_WT_seed_ext_26-75"
ORIG_JSON = Path(os.environ.get("DRIFT_ARCHIVE_ROOT", "$DRIFT_ARCHIVE_ROOT")) / "MutS_alpha_MSH6_GT_WT_data/MutS_alpha_MSH6_GT_WT/MutS_alpha_MSH6_GT_WT_data.json"
ORIG_CSV = REPO / "PLDDTs/MutS_alpha_MSH6_GT/MutS_alpha_MSH6_GT_WT_plddt_values.csv"
OUT = HERE / "wt_75seeds"
CHAINS = ["A", "B"]
SAMPLE = re.compile(r"/seed-(\d+)_sample-(\d+)/[^/]+\.cif$")


def job_hash(obj: dict) -> str:
    """Hash of a job description with modelSeeds removed (key order irrelevant)."""
    obj = {k: v for k, v in obj.items() if k != "modelSeeds"}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    orig = json.loads(ORIG_JSON.read_text())
    ref_hash, orig_seeds = job_hash(orig), set(orig["modelSeeds"])
    print(f"original WT job: seeds {min(orig_seeds)}-{max(orig_seeds)} ({len(orig_seeds)}), hash {ref_hash[:12]}")

    cols, report, seen = {}, [], set(orig_seeds)
    for arc in sorted(EXT_DIR.glob("*_seeds_*_data.tar.gz")):
        a, b = map(int, re.search(r"_seeds_(\d+)-(\d+)_data", arc.name).groups())
        want = {(s, k) for s in range(a, b + 1) for k in range(5)}
        got, data_json = {}, None
        with tarfile.open(arc, "r:gz") as tf:          # streams the whole archive: integrity check
            for m in tf:
                if not m.isfile():
                    continue
                if m.name.endswith("_data.json"):
                    data_json = json.load(tf.extractfile(m))
                elif (hit := SAMPLE.search("/" + m.name)):
                    s, k = int(hit[1]), int(hit[2])
                    text = tf.extractfile(m).read().decode()
                    got[(s, k)] = parse_cif_plddt(text, CHAINS, source=f"{arc.name}:{m.name}")
        seeds = {s for s, _ in got}
        r = {"archive": arc.name, "predictions": len(got),
             "complete": set(got) == want,
             "same_job_as_wt": data_json is not None and job_hash(data_json) == ref_hash,
             "json_seeds": sorted(data_json.get("modelSeeds", [])) if data_json else None,
             "overlap": sorted(seeds & seen)}
        seen |= seeds
        report.append(r)
        print(r, flush=True)
        for (s, k), m in got.items():
            cols[f"seed-{s}_sample-{k}"] = m

    rep = pd.DataFrame(report)
    rep.to_csv(OUT / "ingest_report.csv", index=False)
    ok = rep.complete.all() and rep.same_job_as_wt.all() and not rep.overlap.map(len).any()
    if not ok:
        raise SystemExit("extension FAILED a check -- see ingest_report.csv; nothing written")

    ext = pd.DataFrame(cols)
    base = pd.read_csv(ORIG_CSV, index_col=0)
    assert len(base) == len(ext) == 2294, (base.shape, ext.shape)
    assert not set(base.columns) & set(ext.columns)
    ext.index = base.index
    both = pd.concat([base, ext], axis=1)
    order = sorted(both.columns, key=lambda c: tuple(map(int, re.findall(r"\d+", c))))
    both = both[order]
    both.to_csv(OUT / "MutS_alpha_MSH6_GT_WT_75seeds_plddt_values.csv")
    print(f"written: {both.shape} (original 25 seeds + {ext.shape[1] // 5} new seeds)")


if __name__ == "__main__":
    main()
