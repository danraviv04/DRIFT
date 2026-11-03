#!/usr/bin/env python
"""Fig. 1e per-residue calibration for a driver/passenger pair at NEIGHBOURING sites.

Fig. 1e showed R772L against E221D, 551 residues apart, so the passenger's own
site was never in the window. The replacement pairs put both mutated residues in
one 16-residue window. This reruns residue_null/residue_null.py unchanged
(imported, output directory redirected here), exactly as
wt_background_2000/residue_null_2000.py does for the published pair: 2,000
splits, matched and random arms, the 75-seed WT pool. Each mode and each mutant
has its own RNG stream, so the calibration arms must equal the published
residue_null_MSH6_B2000.json -- asserted -- and only the mutant rows are new.

    cd <repo>; DRIFT_RUN_TAG=v6_ca PYTHONPATH="." python \
        analyses/msh6_case_study/fig1e_near_pair/residue_null_near_pair.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DF = REPO / "analyses" / "msh6_case_study" / "residue_null"
sys.path.insert(0, str(DF))
import residue_null as rn  # noqa: E402

PAIRS = {
    # driver, passenger, panel window (ks_sequence_panel.py centres on both sites)
    "A1055P_I1054F": (("A1055P", "I1054F"), (1047, 1062)),
    "R772L_E760D": (("R772L", "E760D"), (759, 774)),
}
REF = REPO / "analyses" / "msh6_case_study" / "wt_background_2000" / "results" / "fig1e_residue_null" \
    / "residue_null_MSH6_B2000.json"

if __name__ == "__main__":
    if os.environ.get("DRIFT_RUN_TAG") != "v6_ca":
        raise SystemExit("set DRIFT_RUN_TAG=v6_ca")
    ref = json.loads(REF.read_text())
    for name, (mutants, (lo, hi)) in PAIRS.items():
        out = HERE / "results" / name
        out.mkdir(parents=True, exist_ok=True)
        rn.HERE = out   # residue_null.main writes <stem>.json / <stem>_per_residue.csv to HERE
        rn.main(["--gene", "MSH6", "--mutants", *mutants, "--window", str(lo), str(hi),
                 "--wt-pool", "analyses/msh6_case_study/wt_pool/wt_75seeds/MutS_alpha_MSH6_GT_WT_75seeds_plddt_values.csv",
                 "--splits", "2000", "--modes", "matched", "random"])
        new = json.loads((out / "residue_null_MSH6_B2000.json").read_text())
        for mode in ("matched", "random"):
            assert new["calibration"][mode] == ref["calibration"][mode], f"{name}: {mode} calibration differs"
        print(f"{name}: calibration arms reproduce the published 2,000-split run exactly")
