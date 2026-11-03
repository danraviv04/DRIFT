#!/usr/bin/env python
"""Fig. 1e per-residue calibration (MSH6, N=90) at 2,000 WT-WT splits, matched AND random.

experiments/residue_null/residue_null_MSH6_B2000.json has the matched (seed-disjoint) null at
2,000 splits only; the random-split share quoted in Note S2 (5.1%) is still the 200-split
residue_null_MSH6.json. This reruns residue_null/residue_null.py unchanged (imported, output
directory redirected here) with --splits 2000 --modes matched random. Each mode has its own
RNG stream (SEED + crc32(mode)), so the matched arm must equal residue_null_MSH6_B2000.json
exactly -- asserted -- and the random arm is the 2,000-split continuation of the published
200-split random arm's design.

    cd <repo>; DRIFT_RUN_TAG=v6_ca PYTHONPATH="." python \
        experiments/wt_background_2000/residue_null_2000.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DF = REPO / "experiments" / "residue_null"
sys.path.insert(0, str(DF))
import residue_null as rn  # noqa: E402

OUT = HERE / "results" / "fig1e_residue_null"

if __name__ == "__main__":
    if os.environ.get("DRIFT_RUN_TAG") != "v6_ca":
        raise SystemExit("set DRIFT_RUN_TAG=v6_ca")
    OUT.mkdir(parents=True, exist_ok=True)
    rn.HERE = OUT   # residue_null.main writes <stem>.json / <stem>_per_residue.csv to HERE
    rn.main(["--gene", "MSH6", "--mutants", "E221D", "R772L", "--window", "764", "779",
             "--wt-pool", "experiments/msh6_case_study/wt_75seeds/MutS_alpha_MSH6_GT_WT_75seeds_plddt_values.csv",
             "--splits", "2000", "--modes", "matched", "random"])
    new = json.loads((OUT / "residue_null_MSH6_B2000.json").read_text())
    ref = json.loads((DF / "residue_null_MSH6_B2000.json").read_text())
    assert new["calibration"]["matched"] == ref["calibration"]["matched"], "matched calibration differs"
    for m in ref["mutants"]:
        assert new["mutants"][m]["matched"] == ref["mutants"][m]["matched"], f"{m} matched differs"
    print("matched arm reproduces residue_null/residue_null_MSH6_B2000.json exactly")
