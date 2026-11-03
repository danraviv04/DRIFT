"""Draw the KRAS4B AF3 batch: k=5 random substitutions per assayed position,
plus the 60 hotspot variants carried over from the former cohort.

The random draw is replicate 0 of blind_design.draw (seed per position), so
it is one of the draws whose statistics blind_design.py reports. The hotspot
set is the 60-variant list that KRAS4B.json held before this batch: the 56 of
the former 4A cohort plus G151M, H166W, K178M, M188I, all valid in 4B.

Only the random draw is a sample of the screen. The hotspots were chosen from
the old driver/passenger labels, so statistics over the batch must use the
`blind` rows only; `hotspot_only` rows are for case studies and figures.

Writes:
  batch/kras4b_blind_k5.txt   the random draw, one variant per line (read by
                              af3_jobs/configs/KRAS4B.json)
  results/batch_strata.csv    every batch variant with its stratum, assay
                              value and AM score
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from blind_design import draw, load

HERE = Path(__file__).resolve().parent
K = 5
REPLICATE = 0

HOTSPOTS = [
    "K5E", "K5N", "V9N", "G12A", "G12C", "G12D", "G12E", "G12L", "G12Q", "G12R",
    "G12S", "G12V", "G12W", "G13C", "G13D", "G13P", "G13R", "G13V", "V14I", "A18V",
    "L19F", "Q22E", "Q22K", "Q22R", "P34L", "P34R", "I36M", "D47G", "T58I", "A59S",
    "A59T", "G60R", "G60S", "G60V", "Q61H", "Q61K", "Q61L", "Q61R", "R68S", "Y71H",
    "M72L", "F82T", "E91A", "N116H", "K117N", "K117Y", "Y137K", "A146P", "A146V", "K147E",
    "K147R", "G151M", "D154Y", "F156I", "F156S", "F156V", "H166W", "K178M", "K182F", "M188I",
]


def main() -> None:
    screen = load()
    blind = draw(screen, K, REPLICATE).sort_values(["POS", "VAR"])
    assert set(HOTSPOTS) <= set(screen.VAR), "hotspot without an assay value"

    batch_dir = HERE / "batch"
    batch_dir.mkdir(exist_ok=True)
    (batch_dir / f"kras4b_blind_k{K}.txt").write_text(
        f"# KRAS4B blind draw: k={K} per assayed position, blind_design.draw replicate {REPLICATE}\n"
        + "\n".join(blind.VAR) + "\n"
    )

    blind_set, hot_set = set(blind.VAR), set(HOTSPOTS)
    strata = screen[screen.VAR.isin(blind_set | hot_set)].copy()
    strata["stratum"] = [
        "blind+hotspot" if v in blind_set and v in hot_set else "blind" if v in blind_set else "hotspot_only"
        for v in strata.VAR
    ]
    strata = strata.sort_values(["POS", "VAR"])[["VAR", "POS", "stratum", "lfc", "am"]]
    (HERE / "results").mkdir(exist_ok=True)
    strata.to_csv(HERE / "results" / "batch_strata.csv", index=False)

    print(f"blind draw: {len(blind_set)} variants at {blind.POS.nunique()} positions")
    print(f"hotspots: {len(hot_set)}, of which already in the draw: {len(blind_set & hot_set)}")
    print(f"batch: {len(strata)} variants + WT = {len(strata) + 1} jobs")
    print(strata.stratum.value_counts().to_string())


if __name__ == "__main__":
    main()
