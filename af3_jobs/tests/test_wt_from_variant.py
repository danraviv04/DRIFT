"""Tests for af3_jobs/wt_from_variant.py on a synthetic MSA- and template-ready data.json.

What must hold: only the target chain's sequence and its MSA query rows change (plus
name/modelSeeds); templates, other MSA rows, other chains and every other key are copied
unchanged; and any disagreement with the config is refused before anything is written.

Run without pytest (no env has it):

    PYTHONPATH=. python af3_jobs/tests/test_wt_from_variant.py
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from af3_jobs import wt_from_variant as wfv  # noqa: E402
from af3_jobs.sequences import resolve_all  # noqa: E402
from af3_jobs.spec import spec_from_dict  # noqa: E402

WT_A = "MAVQPKETLQLESAAEVGFVRFFQGMPEKP"   # target, residues 1-30
WT_B = "GSHMLEDPVKRILEQAKRE"              # partner
POS, WT_AA, MUT_AA = 10, "Q", "R"         # WT_A[9] == "Q"
VARIANT = f"{WT_AA}{POS}{MUT_AA}"
MUT_A = WT_A[:POS - 1] + MUT_AA + WT_A[POS:]

CONFIG = {
    "name": "TOY", "job_prefix": "TOY", "model_seeds": 25, "version": 4,
    "chains": [
        {"id": "A", "type": "protein", "sequence": WT_A, "mutate": True},
        {"id": "B", "type": "protein", "sequence": WT_B},
        {"id": "C", "type": "dna", "sequence": "GATTACA"},
        {"id": ["E", "F"], "type": "ligand", "ccd_codes": ["ATP"]},
    ],
    "mutations": {"source": "list", "include": [VARIANT]},
}


def _a3m(query: str, wrap: int | None = None) -> str:
    q = query if not wrap else "\n".join(query[i:i + wrap] for i in range(0, len(query), wrap))
    # hit rows: gaps and lowercase insertions, aligned to the query's columns
    hit1 = "-" * 3 + query[3:12].lower().upper() + "abc" + query[12:]
    hit2 = query[:5] + "--" + query[7:]
    return f">query\n{q}\n>UniRef90_X1/1-40\n{hit1}\n>UniRef90_X2\n{hit2}\n"


def _templates(n: int) -> list[dict]:
    return [{"mmcif": f"data_T{i}\n#\n_entry.id T{i}\n", "queryIndices": list(range(0, 20)),
             "templateIndices": list(range(5, 25))} for i in range(n)]


def make_source(mut_a: str = MUT_A, *, wrap=None) -> dict:
    return {
        "dialect": "alphafold3", "version": 4, "name": f"TOY_{VARIANT}",
        "modelSeeds": list(range(1, 26)),
        "sequences": [
            {"protein": {"id": "A", "sequence": mut_a, "modifications": [],
                         "unpairedMsa": _a3m(mut_a, wrap), "pairedMsa": _a3m(mut_a),
                         "templates": _templates(4)}},
            {"protein": {"id": "B", "sequence": WT_B, "modifications": [],
                         "unpairedMsa": _a3m(WT_B), "pairedMsa": _a3m(WT_B), "templates": _templates(2)}},
            {"dna": {"id": "C", "sequence": "GATTACA", "modifications": [],
                     "unpairedMsa": None, "pairedMsa": None}},
            {"ligand": {"id": ["E", "F"], "ccdCodes": ["ATP"]}},
        ],
        "bondedAtomPairs": None, "userCCD": None,
    }


def _spec(cfg=CONFIG):
    spec = spec_from_dict(copy.deepcopy(cfg))
    return spec, resolve_all(spec.chains)


def _expect(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), f"wrong error: {e}"
        return
    raise AssertionError(f"expected ValueError containing {needle!r}")


def test_only_query_changes():
    src = make_source()
    spec, seqs = _spec()
    wt, check = wfv.build_wt_job(src, spec, seqs, VARIANT)
    a_src, a_wt = src["sequences"][0]["protein"], wt["sequences"][0]["protein"]
    assert a_wt["sequence"] == WT_A
    for key in ("unpairedMsa", "pairedMsa"):
        old, new = a_src[key].split("\n"), a_wt[key].split("\n")
        assert new[1] == WT_A and old[1] == MUT_A
        assert old[:1] + old[2:] == new[:1] + new[2:], "a non-query MSA line changed"
    assert a_wt["templates"] == a_src["templates"]
    assert wt["sequences"][1:] == src["sequences"][1:]
    rest = lambda d: {k: v for k, v in d.items() if k != "sequences"}
    assert rest(wt) == rest(src)
    assert check["reverted"] == "R->Q" and check["chain"] == ["A"] and check["position"] == POS
    assert src["sequences"][0]["protein"]["sequence"] == MUT_A, "source was mutated in place"


def test_wrapped_query_row():
    src = make_source(wrap=7)
    spec, seqs = _spec()
    wt, _ = wfv.build_wt_job(src, spec, seqs, VARIANT)
    msa = wt["sequences"][0]["protein"]["unpairedMsa"].split("\n")
    assert msa[:2] == [">query", WT_A] and msa[2].startswith(">UniRef90_X1")


def test_variant_from_name():
    assert wfv.variant_from_name("TOY_Q10R", "TOY") == "Q10R"
    assert wfv.variant_from_name("MutS_alpha_MSH2_Q629R", "MutS_alpha_MSH2") == "Q629R"
    assert wfv.variant_from_name("P53_R175H_with_ZN_and_DNA", "P53", "_with_ZN_and_DNA") == "R175H"
    _expect(lambda: wfv.variant_from_name("OTHER_Q10R", "TOY"), "does not start with")


def test_refuses_partner_not_wt():
    src = make_source()
    src["sequences"][1]["protein"]["sequence"] = "A" + WT_B[1:]
    spec, seqs = _spec()
    _expect(lambda: wfv.build_wt_job(src, spec, seqs, VARIANT), "is not the config WT")


def test_refuses_wrong_variant():
    src = make_source()
    spec, seqs = _spec()
    # the source carries Q10R, the request names a different substitution at 10
    _expect(lambda: wfv.build_wt_job(src, spec, seqs, "Q10K"), "is not variant Q10K")
    # the config has no Q at 11
    _expect(lambda: wfv.build_wt_job(src, spec, seqs, "Q11R"), "not Q")
    _expect(lambda: wfv.build_wt_job(src, spec, seqs, "Q99R"), "out of range")


def test_refuses_msa_query_mismatch():
    src = make_source()
    src["sequences"][0]["protein"]["pairedMsa"] = _a3m(WT_A)   # stale query row
    spec, seqs = _spec()
    _expect(lambda: wfv.build_wt_job(src, spec, seqs, VARIANT), "MSA query row")


def test_refuses_missing_msa_and_chain():
    src = make_source()
    src["sequences"][0]["protein"]["unpairedMsa"] = ""
    spec, seqs = _spec()
    _expect(lambda: wfv.build_wt_job(src, spec, seqs, VARIANT), "no unpairedMsa")
    src = make_source()
    del src["sequences"][1]
    _expect(lambda: wfv.build_wt_job(src, spec, seqs, VARIANT), "absent from the source")


def test_homomer_target():
    cfg = copy.deepcopy(CONFIG)
    cfg["chains"] = [{"id": ["A", "B", "C", "D"], "type": "protein", "sequence": WT_A, "mutate": True}]
    src = make_source()
    src["sequences"] = [src["sequences"][0]]
    src["sequences"][0]["protein"]["id"] = ["A", "B", "C", "D"]
    spec, seqs = _spec(cfg)
    wt, check = wfv.build_wt_job(src, spec, seqs, VARIANT)
    assert wt["sequences"][0]["protein"]["sequence"] == WT_A and check["chain"] == ["A", "B", "C", "D"]


def test_main_writes_job_zip_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "TOY.json").write_text(json.dumps(CONFIG))
        (tmp / "TOY_Q10R_data.json").write_text(json.dumps(make_source()))
        out = tmp / "jobs" / "TOY_WT_vartmpl"
        wfv.main([str(tmp / "TOY_Q10R_data.json"), "--config", str(tmp / "TOY.json"), "--out", str(out)])
        files = sorted(p.name for p in out.iterdir())
        assert files == ["TOY_WT_vartmpl_data.json"], files
        job = json.loads((out / files[0]).read_text())
        assert job["name"] == "TOY_WT_vartmpl" and job["modelSeeds"] == list(range(1, 26))
        assert job["sequences"][0]["protein"]["sequence"] == WT_A
        assert zipfile.ZipFile(out.with_suffix(".zip")).namelist() == files
        man = json.loads((out.parent / "TOY_WT_vartmpl_manifest.json").read_text())
        assert man["check"]["variant"] == "Q10R" and man["source_name"] == "TOY_Q10R"

        out2 = tmp / "jobs" / "TOY_WT_chunks"
        wfv.main([str(tmp / "TOY_Q10R_data.json"), "--config", str(tmp / "TOY.json"),
                  "--seeds", "1-10", "--chunk", "5", "--out", str(out2)])
        assert sorted(p.name for p in out2.iterdir()) == [
            "TOY_WT_vartmpl_seeds_1-5_data.json", "TOY_WT_vartmpl_seeds_6-10_data.json"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
