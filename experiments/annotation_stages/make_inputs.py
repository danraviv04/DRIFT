#!/usr/bin/env python
"""Write 25-seed copies of the eight 32-seed POLE pLDDT matrices (2026-10-20).

Eight labelled POLE drivers were predicted with AF3 seeds 1-32 (160 predictions);
the WT reference POLE_268-471_WT and every other POLE variant have seeds 1-25
(125). This keeps seeds 1-25 of the eight, which are exactly the WT's seed
numbers, and drops 26-32. The originals in PLDDTs/POLE/ are tracked and are not
touched; the copies go to PLDDTs/POLE_25seed/ under the SAME basename, so the
mutation parser and artifact keys are unchanged.

Values are copied as text, cell for cell: the kept columns of every row are the
original file's bytes, so no float is re-rendered. Checks: column set of each copy
== WT column set (125 seed-sample tags, seeds 1-25 x samples 0-4); residue index
0-203; values parse identical to the original's kept columns.

    PYTHONPATH="." python experiments/annotation_stages/make_inputs.py
"""
from pathlib import Path
import csv
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "PLDDTs" / "POLE"
DST = ROOT / "PLDDTs" / "POLE_25seed"
WT = SRC / "POLE_268-471_WT_plddt_values.csv"
MAX_SEED = 25
EIGHT = ("D275A", "D275G", "E277G", "P286H", "P286L", "P286R", "P286S", "T278K")
TAG = re.compile(r"^seed-(\d+)_sample-(\d+)$")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    DST.mkdir(exist_ok=True)
    wt_cols = next(csv.reader(WT.open()))[1:]
    assert len(wt_cols) == 125 and all(int(TAG.match(c)[1]) <= MAX_SEED for c in wt_cols)
    records = []
    for code in EIGHT:
        src = SRC / f"POLE_268-471_{code}_plddt_values.csv"
        dst = DST / src.name
        rows = list(csv.reader(src.open(newline="")))
        header = rows[0]
        assert header[0] == "" and len(header) == 161, (src, len(header))
        seeds = [int(TAG.match(c)[1]) for c in header[1:]]
        assert sorted(set(seeds)) == list(range(1, 33)), src
        keep = [0] + [i for i, c in enumerate(header) if i and int(TAG.match(c)[1]) <= MAX_SEED]
        assert set(header[i] for i in keep[1:]) == set(wt_cols), src
        with dst.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            for row in rows:
                writer.writerow([row[i] for i in keep])
        original = pd.read_csv(src, index_col=0)
        copy = pd.read_csv(dst, index_col=0)
        assert copy.index.tolist() == list(range(204)) == original.index.tolist()
        assert copy.shape == (204, 125)
        assert np.array_equal(copy.to_numpy(), original[copy.columns].to_numpy())
        # newline convention matches the source file, so the copy is a strict column subset
        assert src.read_bytes().count(b"\r") == dst.read_bytes().count(b"\r") == 0
        records.append(dict(mutation=code, original=str(src.relative_to(ROOT)),
                            original_sha256=sha(src), original_predictions=160,
                            copy=str(dst.relative_to(ROOT)), copy_sha256=sha(dst),
                            copy_predictions=125, seeds_kept="1-25", seeds_dropped="26-32"))
        print(f"{code}: 160 -> 125 predictions, {dst.relative_to(ROOT)}")
    out = Path(__file__).resolve().parent / "inputs.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    (Path(__file__).resolve().parent / "inputs_wt.json").write_text(json.dumps(
        dict(wt=str(WT.relative_to(ROOT)), wt_sha256=sha(WT), wt_predictions=125), indent=2) + "\n")


if __name__ == "__main__":
    main()
