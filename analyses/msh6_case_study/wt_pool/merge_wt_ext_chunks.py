#!/usr/bin/env python
"""Chain the 10 MSH6 G:T WT extension chunks (seeds 26-30 ... 71-75) into ONE archive.

    MutS_alpha_MSH6_GT_WT_seeds_26-75_data.tar.gz
      root: MutS_alpha_MSH6_GT_WT_seeds_26-75_data/MutS_alpha_MSH6_GT_WT/seed-*_sample-*/...

Each step is af3_jobs/local_runner/merge_seed_extensions.merge_pair, which refuses a
pair whose data.json differ in anything but modelSeeds or that share a seed/sample,
rebuilds ranking_scores.csv over the union, keeps the top-level files of the best
sample overall, and re-reads its output to count the CIFs. The result keeps the
extension naming (`_seeds_26-75_data`), so it can later be merged with the original
25-seed WT by the same tool, and can never be taken for the original
`MutS_alpha_MSH6_GT_WT_data.tar.gz`. The 10 source chunks are not touched.

    PYTHONPATH=. python analyses/msh6_case_study/wt_pool/merge_wt_ext_chunks.py
"""
from __future__ import annotations
import os

import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "af3_jobs" / "local_runner"))

import merge_seed_extensions as M  # noqa: E402

STEM = "MutS_alpha_MSH6_GT_WT"
SRC = Path(os.environ.get("DRIFT_ARCHIVE_ROOT", "$DRIFT_ARCHIVE_ROOT")) / "MutS_alpha_MSH6_GT_WT_seed_ext_26-75"
OUT_DIR = SRC / "merged"


def main():
    chunks = sorted(SRC.glob(f"{STEM}_seeds_*_data.tar.gz"),
                    key=lambda p: int(re.search(r"_seeds_(\d+)-", p.name)[1]))
    spans = [tuple(map(int, re.search(r"_seeds_(\d+)-(\d+)_", p.name).groups())) for p in chunks]
    assert len(chunks) == 10 and all(b2 == a1 - 1 for (_, b2), (a1, _) in zip(spans, spans[1:])), spans
    lo, hi = spans[0][0], spans[-1][1]
    OUT_DIR.mkdir(exist_ok=True)
    work = OUT_DIR / "_chain_tmp"
    work.mkdir(exist_ok=True)
    gz = M.Gzip()
    acc, a = chunks[0], spans[0][0]
    t0 = time.time()
    for path, (_, b) in zip(chunks[1:], spans[1:]):
        last = b == hi
        name = f"{STEM}_seeds_{a}-{b}_data"
        dest = OUT_DIR if last else work
        n = (b - a + 1) * 5                          # seeds x 5 samples
        msg = M.merge_pair(STEM, acc, path, dest, expect_structures=n, overwrite=True,
                           verify=True, gz=gz, delete_sources=False, out_name=name)
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)
        if acc.parent == work:
            acc.unlink()                             # previous intermediate
        acc = dest / f"{name}.tar.gz"
    shutil.rmtree(work, ignore_errors=True)
    print(f"done: {acc}  ({acc.stat().st_size / 1e9:.2f} GB, seeds {lo}-{hi})")


if __name__ == "__main__":
    main()
