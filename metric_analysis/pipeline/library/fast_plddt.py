"""Fast per-residue pLDDT extraction, identical in output to the pipeline's stage.

``structure_metrics.process_artifacts_to_plddt`` unpacks each archive in full to
disk (including the 178 MB MSA job file and every confidences JSON) and then
parses the CIFs one by one with Biopython. On TP53 that is ~52 s per archive,
about 6 MB/s from a drive that delivers 20+ MB/s, so the stage is bound by the
unpacking and the parsing rather than by the drive.

This module streams each archive once, reads only the ``.cif`` members into
memory, and pulls the per-residue value straight out of the ``_atom_site`` loop.
The result is the same table the pipeline writes, because it applies the same
rules as ``extract_plddt_from_cif`` with Biopython's default MMCIFParser:

- chains and residues are keyed on ``auth_asym_id`` / ``auth_seq_id`` /
  ``pdbx_PDB_ins_code`` and the HETATM flag (Biopython's ``auth_chains`` and
  ``auth_residues`` defaults), first model only;
- residues are numbered 0, 1, 2, ... across the requested chains in order;
- a residue's value is its CA B-factor (= pLDDT in AF3 output), or the mean
  B-factor of its atoms when it has no CA;
- each CIF is keyed by its parent directory's name, a later CIF with the same
  parent replacing an earlier one, and the sample that echoes the artifact name
  (the job-level ranked model) is dropped (``drop_matching_sample``).

One deliberate difference: columns are written in sorted order. The pipeline
keeps ``glob`` order over a temporary directory, which is filesystem order and
not reproducible; the values per column are unchanged.

``validate_against_pipeline`` checks equality on a real archive.
"""
from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence, Tuple

import numpy as np
import pandas as pd

from metric_analysis.pipeline.library.structure_metrics import (
    canonical_artifact_stem,
    drop_matching_sample,
)

_NEEDED = ("group_PDB", "label_atom_id", "label_comp_id", "auth_asym_id", "auth_seq_id",
           "pdbx_PDB_ins_code", "B_iso_or_equiv", "pdbx_PDB_model_num")


def parse_cif_plddt(text: str, chain_ids: Sequence[str], source: str = "") -> Dict[int, float]:
    """Residue-indexed pLDDT for ``chain_ids`` from one mmCIF text (see module doc)."""
    cols: Dict[str, int] = {}
    in_loop = in_atoms = False
    first_model = None
    # (chain, hetflag, resseq, icode) -> [ca_b or None, sum_b, n_atoms], insertion-ordered
    residues: Dict[tuple, list] = {}
    order: Dict[str, list] = {}
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith("loop_"):
            in_loop, in_atoms, cols = True, False, {}
            continue
        if in_loop and line.startswith("_atom_site."):
            cols[line.split()[0][len("_atom_site."):]] = len(cols)
            in_atoms = True
            continue
        if not in_atoms:
            continue
        if line[0] in "_#" or line.startswith("loop_"):
            if residues:            # the atom_site loop is over
                break
            in_atoms = False
            continue
        f = line.split()
        if first_model is None:
            missing = [k for k in _NEEDED if k not in cols and k != "pdbx_PDB_ins_code"]
            if missing:
                raise ValueError(f"{source}: _atom_site lacks {missing}")
            first_model = f[cols["pdbx_PDB_model_num"]]
        if f[cols["pdbx_PDB_model_num"]] != first_model:
            break
        chain = f[cols["auth_asym_id"]]
        resname = f[cols["label_comp_id"]]
        het = f[cols["group_PDB"]] == "HETATM"
        hetflag = (" " if not het else ("W" if resname in ("HOH", "WAT") else f"H_{resname}"))
        icode = f[cols["pdbx_PDB_ins_code"]] if "pdbx_PDB_ins_code" in cols else "?"
        icode = " " if icode in ("?", ".") else icode
        key = (chain, hetflag, int(f[cols["auth_seq_id"]]), icode)
        b = float(f[cols["B_iso_or_equiv"]])
        rec = residues.get(key)
        if rec is None:
            rec = residues[key] = [None, 0.0, 0]
            order.setdefault(chain, []).append(key)
        if f[cols["label_atom_id"]] == "CA" and rec[0] is None:
            rec[0] = b
        rec[1] += b
        rec[2] += 1
    out: Dict[int, float] = {}
    idx = 0
    for chain_id in chain_ids:
        if chain_id not in order:
            raise KeyError(f"Chain '{chain_id}' not found in {source}. Chains: {list(order)}")
        for key in order[chain_id]:
            ca, s, n = residues[key]
            out[idx] = float(ca) if ca is not None else float(np.mean(s / n)) if n else np.nan
            idx += 1
    return out


def iter_cif_members(archive: Path) -> Iterator[Tuple[str, str]]:
    """Yield (member name, text) for every .cif member, streaming the archive once.

    Decompression runs in a separate ``gzip -dc`` process so it overlaps with the
    parsing here; falls back to Python's tarfile stream if gzip is unavailable.
    """
    archive = Path(archive)
    proc = None
    try:
        if archive.name.endswith((".tar.gz", ".tgz")):
            proc = subprocess.Popen(["gzip", "-dc", str(archive)], stdout=subprocess.PIPE,
                                    bufsize=8 << 20)
            tf = tarfile.open(fileobj=proc.stdout, mode="r|")
        else:
            tf = tarfile.open(archive, mode="r|*")
        with tf:
            for m in tf:
                if not m.isfile() or not m.name.lower().endswith(".cif"):
                    continue
                base = m.name.rsplit("/", 1)[-1]
                if base.startswith("._"):
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                yield m.name, fh.read().decode("utf-8", errors="replace")
    finally:
        if proc is not None:
            proc.stdout.close()
            rc = proc.wait()
            if rc not in (0, -13):        # -13: SIGPIPE if we stopped early
                raise RuntimeError(f"gzip -dc failed ({rc}) on {archive}")


def archive_plddt_table(archive: Path, chain_ids: Sequence[str],
                        drop_matching: bool = True) -> pd.DataFrame:
    """The residue x prediction table the pipeline writes for one archive."""
    maps: Dict[str, Dict[int, float]] = {}
    for name, text in iter_cif_members(archive):
        parent = name.rstrip("/").rsplit("/", 2)
        sample = parent[-2] if len(parent) >= 2 else ""
        maps[sample] = parse_cif_plddt(text, chain_ids, source=f"{archive.name}:{name}")
    if drop_matching:
        drop_matching_sample(maps, canonical_artifact_stem(archive))
    df = pd.DataFrame(maps)
    return df[sorted(df.columns)]


def folder_plddt_table(folder: Path, chain_ids: Sequence[str], drop_matching: bool = True) -> pd.DataFrame:
    """Same as ``archive_plddt_table`` for an unpacked job folder (POLE's source layout)."""
    folder = Path(folder)
    maps: Dict[str, Dict[int, float]] = {}
    for cif in sorted(folder.rglob("*.cif")):
        if cif.name.startswith("._"):
            continue
        maps[cif.parent.name] = parse_cif_plddt(cif.read_text(errors="replace"), chain_ids, source=str(cif))
    if drop_matching:
        drop_matching_sample(maps, canonical_artifact_stem(folder))
    df = pd.DataFrame(maps)
    return df[sorted(df.columns)]


def validate_against_pipeline(archive: Path, chain_ids: Sequence[str]) -> dict:
    """Run the pipeline's own extraction on ``archive`` and compare value for value."""
    import tempfile
    from metric_analysis.pipeline.library.structure_metrics import analyze_plddt_distribution

    fast = archive_plddt_table(archive, chain_ids)
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive, "r:*") as tar_ref:
            tar_ref.extractall(tmp)
        maps = analyze_plddt_distribution(tmp, chain_ids=chain_ids)
    drop_matching_sample(maps, canonical_artifact_stem(archive))
    ref = pd.DataFrame(maps)
    ref = ref[sorted(ref.columns)]
    same_cols = list(fast.columns) == list(ref.columns)
    same_shape = fast.shape == ref.shape
    max_diff = (float(np.nanmax(np.abs(fast.to_numpy() - ref.to_numpy())))
                if same_cols and same_shape else None)
    return {"archive": Path(archive).name, "fast_shape": fast.shape, "pipeline_shape": ref.shape,
            "same_columns": same_cols, "max_abs_diff": max_diff,
            "identical": bool(same_cols and same_shape and fast.equals(ref))}
