"""A KRAS4B batch chosen blind to both the assay and AlphaMissense.

Every assayed variant has an AM score, so the candidate universe is the screen
itself (3,536 variants, positions 2-188). The structures will be compared
against both the assay value and AM, so the selection may use neither:
picking by assay value inflates every correlation with it, and picking by AM
widens or narrows AM's range and tilts the structure-vs-AM comparison.

Design: every assayed position (the residue is the unit of the
position-grouped CV), k substitutions per position drawn at random from the
ones assayed there. Each position has its own seed (SEED + position), so the
draw at one residue never depends on another.

For each k this reports the batch size and, as a representativeness check
over DRAWS repeated draws, how the sample compares with the full screen.

Writes results/blind_design.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "results"
KRAS_XLSX = REPO / "Ground_Truths" / "KRAS" / "Extended Data Table 1.xlsx"
AM_TSV = REPO / "Ground_Truths" / "AlphaMissense" / "afm_KRAS_predictions.tsv"
KS = (2, 3, 4, 5, 6)
DRAWS = 100
SEED = 42


def load() -> pd.DataFrame:
    dms = pd.read_excel(KRAS_XLSX, sheet_name="KRAS GOF DMS Screen")
    am = pd.read_csv(AM_TSV, sep="\t")
    am["VAR"] = am["a.a.1"] + am["position"].astype(str) + am["a.a.2"]
    merged = dms.merge(am[["VAR", "pathogenicity score"]], on="VAR", how="left")
    assert merged["pathogenicity score"].notna().all(), "assayed variant without an AM score"
    return merged.rename(columns={"HA1E_WT_LFC": "lfc", "pathogenicity score": "am"})


def draw(screen: pd.DataFrame, k: int, replicate: int) -> pd.DataFrame:
    picks = []
    for pos, group in screen.groupby("POS"):
        rng = np.random.default_rng([SEED, replicate, int(pos)])
        picks.append(group.iloc[rng.choice(len(group), size=min(k, len(group)), replace=False)])
    return pd.concat(picks)


def stats(frame: pd.DataFrame) -> dict:
    return {
        "variants": len(frame),
        "residues": frame["POS"].nunique(),
        "gof_lfc_gt_0.68": int((frame.lfc > 0.68).sum()),
        "intermediate_0.1_to_0.68": int(((frame.lfc > 0.1) & (frame.lfc <= 0.68)).sum()),
        "spearman_am_vs_assay": spearmanr(frame.am, frame.lfc).statistic,
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    screen = load()
    rows = [{"design": "full screen", "k": 19, **stats(screen)}]
    for k in KS:
        reps = pd.DataFrame([stats(draw(screen, k, r)) for r in range(DRAWS)])
        row = {"design": f"k={k} per position, mean of {DRAWS} draws", "k": k}
        row.update(reps.mean().round(3).to_dict())
        row["gof_range"] = f"{reps['gof_lfc_gt_0.68'].min()}-{reps['gof_lfc_gt_0.68'].max()}"
        row["spearman_am_vs_assay_sd"] = round(reps["spearman_am_vs_assay"].std(), 3)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["spearman_am_vs_assay"] = table["spearman_am_vs_assay"].round(3)
    table.to_csv(OUT / "blind_design.csv", index=False)
    print(table.to_string(index=False))

    # AM against the signed assay value is U-shaped: highest for strongly
    # depleted variants and for GOF hits, lowest near neutral.
    bins = pd.cut(screen.lfc, [-3, -0.5, -0.25, 0, 0.1, 0.68, 3])
    shape = screen.groupby(bins, observed=True).am.agg(["count", "median", "mean"]).round(3)
    shape.index = shape.index.astype(str)
    shape.to_csv(OUT / "am_by_assay_bin.csv", index_label="lfc_bin")
    print()
    print(shape.to_string())


if __name__ == "__main__":
    main()
