"""Tests for af3_jobs.

The reproduction tests are the point of this file: the ported configs must
regenerate the JSONs that are already sitting in ``muts_jsons/`` byte for byte,
which is what makes it safe to retire the hand-edited generator script.

Run without pytest (no env has it):

    python af3_jobs/tests/test_af3_jobs.py
"""

from __future__ import annotations

import json
import sys
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from af3_jobs.build import build_jobs, build_manifest  # noqa: E402
from af3_jobs.mutations import (  # noqa: E402
    _read_clinvar_table,
    clinvar_row_matches_gene,
    min_nt_distance,
    parse_clinvar_row,
    parse_mutation_name,
)
from af3_jobs.spec import load_spec, spec_from_dict  # noqa: E402

MUTS_JSONS = REPO_ROOT / "muts_jsons"


def _jobs_by_filename(spec_name):
    result = build_jobs(load_spec(spec_name))
    return {job.filename: job for job in result.jobs}, result


def test_parse_mutation_name():
    assert parse_mutation_name("V3L") == ("V", 3, "L")
    assert parse_mutation_name("Val3Leu") == ("V", 3, "L")
    assert parse_mutation_name("NM_000314.8(PTEN):c.139A>G (p.Arg47Gly)") == ("R", 47, "G")
    for bad in ("", "xyz", "3"):
        try:
            parse_mutation_name(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_gene_symbol_with_a_digit_is_not_read_as_a_variant():
    """"PIK3CA" contains the substring K3C; it must not parse as a variant.

    A loose [A-Z]\\d+[A-Z] search over the whole annotation reads the gene symbol
    instead of the protein change, and every row of a PIK3CA export then
    collapses onto the single name "K3C".
    """
    assert parse_mutation_name(
        "NM_006218.4(PIK3CA):c.113G>A (p.Arg38His)"
    ) == ("R", 38, "H")
    assert parse_mutation_name(
        "NM_006218.4(PIK3CA):c.3140A>G (p.His1047Arg)"
    ) == ("H", 1047, "R")
    for bad in ("PIK3CA", "NM_006218.4(PIK3CA)", "PIK3CA|KCNMB3"):
        try:
            parse_mutation_name(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} parsed as a variant: {parse_mutation_name(bad)}")


def test_clinvar_trailing_separator_does_not_shift_columns():
    """ClinVar rows end with a separator, giving one more field than the header.

    Read naively, pandas promotes the first column to the index and shifts every
    column left by one, which silently turns every row into an 'other gene' row.
    """
    import tempfile

    header = "Name\tGene(s)\tProtein change\tCondition(s)\n"
    row = (
        "NM_006218.4(PIK3CA):c.3140A>G (p.His1047Arg)\tPIK3CA\tH1047R"
        "\tNeoplasm\t\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
        fh.write(header + row)
        path = Path(fh.name)
    try:
        frame = _read_clinvar_table(path)
        assert list(frame.columns)[:3] == ["Name", "Gene(s)", "Protein change"]
        first = frame.iloc[0]
        assert first["Gene(s)"] == "PIK3CA", f"columns shifted: {dict(first)}"
        assert clinvar_row_matches_gene(first, "PIK3CA")
        assert parse_clinvar_row(first) == ("H", 1047, "R")
    finally:
        path.unlink()


def test_min_nt_distance():
    assert min_nt_distance("L", "L") == 0
    assert min_nt_distance("G", "D") == 1   # GGT -> GAT
    assert min_nt_distance("M", "W") == 2


def test_pten_regen14_reproduces_existing_directory():
    """The 14 PTEN re-runs must come out identical to the committed JSONs."""
    existing_dir = MUTS_JSONS / "PTEN_Mutations_regen14"
    if not existing_dir.is_dir():
        raise unittest.SkipTest(f"{existing_dir} not present (the committed JSONs are data, not shipped)")
    jobs, result = _jobs_by_filename("PTEN")
    assert not result.skipped, result.skipped

    on_disk = sorted(p.name for p in existing_dir.glob("*.json"))
    assert sorted(jobs) == on_disk, (sorted(jobs), on_disk)
    for filename in on_disk:
        expected = json.loads((existing_dir / filename).read_text())
        assert jobs[filename].payload == expected, filename


def test_muts_alpha_reproduces_existing_archive():
    """Job name, filename and version of the MSH2 batch are all preserved."""
    archive = MUTS_JSONS / "AF3_MSH2_Mutations.zip"
    if not archive.is_file():
        raise unittest.SkipTest(f"{archive} not present (the committed JSONs are data, not shipped)")
    jobs, _ = _jobs_by_filename("MutS_alpha_MSH2")
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        checked = 0
        for name in names:
            if name not in jobs:
                continue
            expected = json.loads(zf.read(name))
            assert jobs[name].payload == expected, name
            checked += 1
    assert checked >= 50, f"only {checked} MSH2 jobs compared"
    assert jobs["AF3_MSH2_T8M.json"].payload["name"] == "MutS_alpha_MSH2_T8M"
    assert jobs["AF3_MSH2_T8M.json"].payload["version"] == 2




def test_fragment_chain_uses_source_numbering():
    """A chain given as residues 322-600 is still addressed in UniProt numbers."""
    spec = spec_from_dict(
        {
            "name": "frag",
            "chains": [
                {
                    "id": "A",
                    "sequence": "ACDEFGHIKL",
                    "first_residue": 322,
                    "mutate": True,
                },
            ],
            "mutations": {"source": "list", "include": ["D324W", "L331M"]},
        }
    )
    result = build_jobs(spec)
    assert not result.skipped, result.skipped
    by_name = {j.name: j for j in result.jobs}
    assert by_name["frag_D324W"].payload["sequences"][0]["protein"]["sequence"] == "ACWEFGHIKL"
    assert by_name["frag_L331M"].payload["sequences"][0]["protein"]["sequence"] == "ACDEFGHIKM"


def test_wt_residue_mismatch_is_skipped_not_applied():
    spec = spec_from_dict(
        {
            "name": "frag",
            "chains": [{"id": "A", "sequence": "ACDEFGHIKL", "mutate": True}],
            "mutations": {"source": "list", "include": ["A1G", "W2G", "A99G"]},
        }
    )
    result = build_jobs(spec)
    assert [j.variant for j in result.jobs if j.variant] == ["A1G"]
    assert len(result.skipped) == 2
    assert "found C" in result.skipped[0]
    assert "outside chain A" in result.skipped[1]


def test_ligand_and_dna_chains_render():
    spec = spec_from_dict(
        {
            "name": "cplx",
            "chains": [
                {"id": "A", "sequence": "ACDEF", "mutate": True},
                {"id": "B", "type": "dna", "sequence": "GATC"},
                {"id": "C", "type": "ligand", "ccd_codes": ["ATP"]},
            ],
            "mutations": {"source": "none"},
        }
    )
    payload = build_jobs(spec).jobs[0].payload
    assert payload["sequences"] == [
        {"protein": {"id": "A", "sequence": "ACDEF"}},
        {"dna": {"id": "B", "sequence": "GATC"}},
        {"ligand": {"id": "C", "ccdCodes": ["ATP"]}},
    ]


def test_spec_rejects_ambiguous_mutation_target():
    for chains in (
        [{"id": "A", "sequence": "AC"}],                                  # none marked
        [{"id": "A", "sequence": "AC", "mutate": True},
         {"id": "B", "sequence": "AC", "mutate": True}],                  # two marked
    ):
        try:
            spec_from_dict({"name": "x", "chains": chains})
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {chains}")


def test_manifest_records_construct():
    """The PIK3CA construct is the niSH2 one, and the manifest says so.

    Built with the mutation source stubbed out, so this does not depend on a
    ClinVar export being present.
    """
    config = json.loads((REPO_ROOT / "af3_jobs/configs/PIK3CA.json").read_text())
    config["mutations"] = {"source": "list", "include": ["H1047R", "E545K"]}
    result = build_jobs(spec_from_dict(config))
    assert not result.skipped, result.skipped

    manifest = build_manifest(result)
    chains = {c["id"]: c for c in manifest["chains"]}
    assert chains["A"]["length"] == 1068 and chains["A"]["mutate"] is True
    assert chains["B"]["length"] == 279 and chains["B"]["residue_range"] == [322, 600]
    assert manifest["structures_per_variant"] == 125

    by_name = {j.name: j for j in result.jobs}
    wt = by_name["PIK3CA_WT"].payload["sequences"][0]["protein"]["sequence"]
    h1047r = by_name["PIK3CA_H1047R"].payload["sequences"][0]["protein"]["sequence"]
    assert wt[1046] == "H" and h1047r[1046] == "R"
    assert sum(a != b for a, b in zip(wt, h1047r)) == 1
    # chain B is untouched by a chain A variant
    assert (
        by_name["PIK3CA_H1047R"].payload["sequences"][1]
        == by_name["PIK3CA_WT"].payload["sequences"][1]
    )


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except unittest.SkipTest as exc:
            print(f"skip {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
