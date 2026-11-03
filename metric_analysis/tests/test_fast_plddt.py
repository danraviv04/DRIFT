"""Synthetic checks for fast_plddt: same values as the Biopython extraction.

    PYTHONPATH=. python metric_analysis/tests/test_fast_plddt.py
"""
import io
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from metric_analysis.pipeline.library.fast_plddt import archive_plddt_table, parse_cif_plddt
from metric_analysis.pipeline.library.structure_metrics import extract_plddt_from_cif

HEADER = """data_x
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""


def atom(i, group, name, el, comp, chain, seq, b, x=0.0):
    return (f"{group} {i} {el} {name} . {comp} {chain} 1 {seq} ? {x + i:.3f} {i * .5:.3f} "
            f"{-i:.3f} 1.00 {b:.2f} {seq} {chain} 1")


def make_cif(seed):
    rng = np.random.default_rng(seed)
    lines, i = [], 1
    for chain in "AB":
        for seq, comp in enumerate(["MET", "GLY", "ALA", "SER"], 1):
            for name, el in [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]:
                lines.append(atom(i, "ATOM", name, el, comp, chain, seq, rng.uniform(30, 95))); i += 1
    for seq in (1, 2):                      # DNA chain C: no CA, mean-of-atoms fallback
        for name, el in [("P", "P"), ("C1'", "C"), ("N9", "N")]:
            lines.append(atom(i, "ATOM", name, el, "DA", "C", seq, rng.uniform(30, 95))); i += 1
    lines.append(atom(i, "HETATM", "ZN", "ZN", "ZN", "D", 1, 77.7))
    return HEADER + "\n".join(lines) + "\n#\n_some_other.item 1\n"


def test_single_cif_matches_biopython():
    for chains in (("A",), ("A", "B"), ("A", "B", "C")):
        for seed in range(5):
            text = make_cif(seed)
            with tempfile.NamedTemporaryFile("w", suffix=".cif", delete=False) as fh:
                fh.write(text)
            ref = extract_plddt_from_cif(fh.name, chain_ids=chains)
            got = parse_cif_plddt(text, chains)
            assert list(ref) == list(got), (chains, seed)
            assert np.allclose(list(ref.values()), list(got.values()), rtol=0, atol=1e-12)
            Path(fh.name).unlink()


def test_archive_table_drops_echo_and_matches():
    with tempfile.TemporaryDirectory() as tmp:
        arc = Path(tmp) / "P53_X1Y_with_ZN_and_DNA_data.tar.gz"
        with tarfile.open(arc, "w:gz") as tf:
            names = [f"P53_X1Y_with_ZN_and_DNA_data/P53_X1Y_with_ZN_and_DNA/seed-{s}_sample-{k}/model.cif"
                     for s in (1, 2) for k in (0, 1)]
            names.append("P53_X1Y_with_ZN_and_DNA_data/P53_X1Y_with_ZN_and_DNA/P53_X1Y_with_ZN_and_DNA_model.cif")
            for j, n in enumerate(names):
                data = make_cif(j).encode()
                ti = tarfile.TarInfo(n); ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        df = archive_plddt_table(arc, ("A", "B"))
        assert list(df.columns) == sorted(f"seed-{s}_sample-{k}" for s in (1, 2) for k in (0, 1))
        assert df.shape == (8, 4)
        ref = parse_cif_plddt(make_cif(0), ("A", "B"))
        assert np.allclose(df["seed-1_sample-0"].to_numpy(), list(ref.values()))


if __name__ == "__main__":
    test_single_cif_matches_biopython()
    test_archive_table_drops_echo_and_matches()
    print("fast_plddt: all synthetic tests pass")
