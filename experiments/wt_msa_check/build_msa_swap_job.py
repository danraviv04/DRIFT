#!/usr/bin/env python
"""Build the MSH2 MSA-swap control job (README §4; 2026-10-16).

MutS_alpha_MSH2_WT_vartmpl is the Q629R job with the WT residue put back, so it carries
Q629R's templates AND Q629R's MSA. This job differs from it ONLY in chain A's MSA:

  sequences       -- unchanged (WT MSH2 + WT MSH6 + DNA), from the vartmpl WT;
  templates       -- unchanged, byte-identical (Q629R's, including 2O8B);
  chain A MSAs    -- unpairedMsa/pairedMsa replaced by the true-WT MSH2 search
                     (b89d88 / 6b7cb0), from the first JSON of the MSH6 G:T WT
                     seed-extension zip, where MSH2 is unmutated;
  chain B MSAs    -- unchanged; its paired MSA is 1bbf67 in both jobs, so the
                     cross-chain row pairing the WT-MSA search produced is kept;
  name/modelSeeds -- MutS_alpha_MSH2_WT_vartmpl_wtmsa, seeds 1-10 (the same seed values
                     as the first 10 of the vartmpl WT's 1-25).

Writes (small; the 178 MB JSON exists only inside the zip):
  msa_swap_job/MutS_alpha_MSH2_WT_vartmpl_wtmsa.zip        -> <name>_data.json inside
  msa_swap_job/MutS_alpha_MSH2_WT_vartmpl_wtmsa_manifest.json
  results/msa_swap_job_check.json                          -> every check below; exit 1 on failure

The checks run on the JSON read back from the zip, against the vartmpl WT:
  1. diff      -- top level: only name and modelSeeds; entities: only chain A's
                  unpairedMsa/pairedMsa; every other entity byte-identical;
  2. templates -- every chain's templates byte-identical;
  3. query     -- every protein chain's MSA query row equals its sequence;
  4. a3m       -- 0 rows whose match-column length differs from the query's;
  5. hashes    -- chain A unpaired b89d88, paired 6b7cb0; chain B paired 1bbf67 in the
                  new job and in both sources, chain A/B paired row counts as in the source
                  job that produced the pairing.

    python -I experiments/wt_msa_check/build_msa_swap_job.py
"""
import hashlib
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_vartmpl_msa import match_cols, records  # noqa: E402
from msa_response_msh2 import chain, sha_without_query  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VARTMPL = REPO / "muts_jsons/MutS_alpha_MSH2_WT_vartmpl_data.json"
WT_ZIP = REPO / "muts_jsons/MutS_alpha_MSH6_GT_WT_seed_ext_26-75.zip"
NAME = "MutS_alpha_MSH2_WT_vartmpl_wtmsa"
SEEDS = list(range(1, 11))
EXPECT = {("A", "unpairedMsa"): "b89d88", ("A", "pairedMsa"): "6b7cb0", ("B", "pairedMsa"): "1bbf67"}
JOB_DIR = HERE / "msa_swap_job"
REPORT = HERE / "results/msa_swap_job_check.json"


def entity(e):
    (kind, x), = e.items()
    return kind, x


def msa_stats(x, key):
    rows = records(x.get(key) or "")
    if not rows:
        return {"rows": 0}
    L = len(x["sequence"])
    return {"rows": len(rows), "hash": sha_without_query(x[key]),
            "query_equals_sequence": rows[0][1] == x["sequence"],
            "rows_with_wrong_match_length": sum(len(match_cols(r)) != L for _, r in rows)}


def build(var, wt_src):
    job = json.loads(json.dumps(var))      # deep copy; key order kept
    a_new, a_wt = chain(job, "A"), chain(wt_src, "A")
    a_new["unpairedMsa"], a_new["pairedMsa"] = a_wt["unpairedMsa"], a_wt["pairedMsa"]
    job["name"], job["modelSeeds"] = NAME, SEEDS
    return job


def check(new, var, wt_src):
    rep, problems = {}, []
    top = sorted(k for k in set(new) | set(var) if k != "sequences" and new.get(k) != var.get(k))
    rep["top_level_changed"] = top
    if set(top) != {"name", "modelSeeds"}:
        problems.append(f"top-level changes {top}, expected exactly name+modelSeeds")
    if len(new["sequences"]) != len(var["sequences"]):
        problems.append("entity count differs")
    ents = []
    for n_e, v_e in zip(new["sequences"], var["sequences"]):
        (kind, n), (vkind, v) = entity(n_e), entity(v_e)
        changed = sorted(k for k in set(n) | set(v) if n.get(k) != v.get(k))
        allowed = {"unpairedMsa", "pairedMsa"} if (kind == "protein" and n.get("id") == "A") else set()
        e = {"id": n.get("id"), "kind": kind, "keys_changed": changed}
        if vkind != kind or set(changed) - allowed:
            problems.append(f"{kind} {n.get('id')}: unexpected change {changed}")
        if kind == "protein":
            e["templates_identical"] = json.dumps(n.get("templates")) == json.dumps(v.get("templates"))
            e["n_templates"] = len(n.get("templates") or [])
            if not e["templates_identical"]:
                problems.append(f"chain {e['id']}: templates differ")
            for key in ("unpairedMsa", "pairedMsa"):
                m = msa_stats(n, key)
                e[key] = m
                if m["rows"] and not m["query_equals_sequence"]:
                    problems.append(f"chain {e['id']} {key}: query row != sequence")
                if m.get("rows_with_wrong_match_length"):
                    problems.append(f"chain {e['id']} {key}: {m['rows_with_wrong_match_length']} malformed rows")
        ents.append(e)
    rep["entities"] = ents

    # hashes and the cross-chain pairing
    a_new, b_new = chain(new, "A"), chain(new, "B")
    a_src, b_src, b_var = chain(wt_src, "A"), chain(wt_src, "B"), chain(var, "B")
    h = {}
    for (cid, key), want in EXPECT.items():
        got = sha_without_query(chain(new, cid)[key])
        h[f"{cid}.{key}"] = got
        if not got.startswith(want):
            problems.append(f"chain {cid} {key} hash {got}, expected {want}")
    h["B.pairedMsa in wt_src"] = sha_without_query(b_src["pairedMsa"])
    h["B.pairedMsa in vartmpl"] = sha_without_query(b_var["pairedMsa"])
    h["A.unpairedMsa in vartmpl (Q629R search)"] = sha_without_query(chain(var, "A")["unpairedMsa"])
    h["A.pairedMsa in vartmpl (Q629R search)"] = sha_without_query(chain(var, "A")["pairedMsa"])
    rep["hashes"] = h
    for k in ("B.pairedMsa in wt_src", "B.pairedMsa in vartmpl"):
        if not h[k].startswith(EXPECT[("B", "pairedMsa")]):
            problems.append(f"{k} hash {h[k]}, expected 1bbf67")
    pairing = {
        "chain_A_sequence_equal_in_sources": a_src["sequence"] == chain(var, "A")["sequence"],
        "chain_B_sequence_equal_in_sources": b_src["sequence"] == b_var["sequence"],
        "chain_B_pairedMsa_identical_in_sources": b_src["pairedMsa"] == b_var["pairedMsa"],
        "chain_A_msas_identical_to_wt_src": (a_new["unpairedMsa"] == a_src["unpairedMsa"]
                                             and a_new["pairedMsa"] == a_src["pairedMsa"]),
        "paired_rows_new_A_B": [len(records(a_new["pairedMsa"])), len(records(b_new["pairedMsa"]))],
        "paired_rows_wt_src_A_B": [len(records(a_src["pairedMsa"])), len(records(b_src["pairedMsa"]))],
        "paired_rows_vartmpl_A_B": [len(records(chain(var, "A")["pairedMsa"])),
                                    len(records(b_var["pairedMsa"]))],
    }
    rep["pairing"] = pairing
    for k, v in pairing.items():
        if v is False:
            problems.append(f"pairing: {k} is False")
    if pairing["paired_rows_new_A_B"] != pairing["paired_rows_wt_src_A_B"]:
        problems.append("paired row counts differ from the job that produced the pairing")
    rep["problems"] = problems
    return rep


def main():
    var_raw = VARTMPL.read_bytes()
    var = json.loads(var_raw)
    z = zipfile.ZipFile(WT_ZIP)
    src_member = sorted(z.namelist())[0]
    wt_src = json.loads(z.read(src_member))
    job = build(var, wt_src)
    del z

    text = json.dumps(job)                 # same serialization as af3_jobs.wt_from_variant
    member = f"{NAME}_data.json"
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    zpath = JOB_DIR / f"{NAME}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, text)
    del job, text
    with zipfile.ZipFile(zpath) as zf:     # check what was written, not the in-memory copy
        raw = zf.read(member)
    new = json.loads(raw)

    rep = {"generated": date.today().isoformat(), "job": member,
           "vartmpl": str(VARTMPL.relative_to(REPO)), "vartmpl_sha256": hashlib.sha256(var_raw).hexdigest(),
           "wt_msa_source": f"{WT_ZIP.relative_to(REPO)}:{src_member}", "modelSeeds": new["modelSeeds"]}
    rep.update(check(new, var, wt_src))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1))
    manifest = {
        "generated": rep["generated"], "config": "MutS_alpha_MSH2",
        "source_data_json": rep["vartmpl"], "source_sha256": rep["vartmpl_sha256"],
        "msa_source": rep["wt_msa_source"],
        "change": "chain A unpairedMsa/pairedMsa <- true-WT MSH2 search (b89d88/6b7cb0); all else as vartmpl",
        "jobs": [{"file": member, "name": NAME, "modelSeeds": new["modelSeeds"],
                  "sha256": hashlib.sha256(raw).hexdigest()}],
        "run": "inference only (--run_data_pipeline=false); templates are Q629R's, unchanged",
        "check_report": str(REPORT.relative_to(REPO)), "check_ok": not rep["problems"],
    }
    (JOB_DIR / f"{NAME}_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"{member}: seeds {new['modelSeeds'][0]}-{new['modelSeeds'][-1]} -> {zpath.relative_to(REPO)}")
    print("top-level changed:", rep["top_level_changed"])
    for e in rep["entities"]:
        line = f"  {e['kind']} {e['id']}: changed {e['keys_changed']}"
        if e["kind"] == "protein":
            line += f", templates identical {e['templates_identical']} ({e['n_templates']})"
            for k in ("unpairedMsa", "pairedMsa"):
                m = e[k]
                if m["rows"]:
                    line += (f"; {k} {m['rows']} rows {m['hash']} query==seq {m['query_equals_sequence']}"
                             f" bad {m['rows_with_wrong_match_length']}")
        print(line)
    print("hashes:", json.dumps(rep["hashes"]))
    print("pairing:", json.dumps(rep["pairing"]))
    print("OK" if not rep["problems"] else "PROBLEMS:\n  " + "\n  ".join(rep["problems"]))
    sys.exit(1 if rep["problems"] else 0)


if __name__ == "__main__":
    main()
