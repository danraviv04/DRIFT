#!/usr/bin/env python
"""Read-only provenance of AF3 archives: one streaming pass per archive.

Records, per archive: every member's name/size/mtime (run dates, AF3 output-dir
casing: 3.0.1 lowercases the job dir, 3.0.2+ keeps it), the job-level *_data.json
(MSA depth per chain, MSA hash without the query row, template count + ids, seeds,
version/dialect), and the per-residue pLDDT of the seed-1_sample-0 model under
three conventions (CA, N, first atom), to compare with the pLDDT CSV.

Run ONE archive at a time and only when no other drive reader is running:
    PYTHONPATH="." python analyses/provenance/wt_variant_offset/archive_provenance.py \
        $DRIFT_ARCHIVE_ROOT/PIK3CA/PIK3CA_WT_data.tar.gz ...
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "provenance"
MODEL = re.compile(r"seed-1_sample-0/.*model\.cif$")


def msa_summary(s):
    if not s:
        return {"n_seqs": 0}
    recs = s.split(">")[1:]
    body = ">".join(recs[1:])
    return {"n_seqs": len(recs), "sha_without_query": hashlib.sha1(body.encode()).hexdigest()[:12],
            "query_header": recs[0].split("\n", 1)[0][:40]}


def data_json_summary(d):
    out = {k: d.get(k) for k in ("name", "dialect", "version", "modelSeeds")}
    out["chains"] = []
    for s in d["sequences"]:
        (kind, x), = s.items()
        c = {"kind": kind, "id": x.get("id"), "len": len(x.get("sequence", "")),
             "unpaired": msa_summary(x.get("unpairedMsa")), "paired": msa_summary(x.get("pairedMsa"))}
        t = x.get("templates")
        if t is not None:
            c["n_templates"] = len(t)
            c["template_ids"] = [tt["mmcif"].split("\n", 1)[0][5:] for tt in t]
            c["templates_sha"] = hashlib.sha1(json.dumps([tt["mmcif"] for tt in t]).encode()).hexdigest()[:12]
        out["chains"].append(c)
    return out


def atom_conventions(text):
    """{chain: {'CA': [...], 'N': [...], 'first': [...]}} from the first model's atom_site loop."""
    cols, res, in_atoms = {}, {}, False
    for line in text.splitlines():
        if line.startswith("_atom_site."):
            cols[line.split()[0][11:]] = len(cols)
            in_atoms = True
            continue
        if not in_atoms or not line or line[0] in "#_" or line.startswith("loop_"):
            if in_atoms and res and (not line or line[0] in "#_" or line.startswith("loop_")):
                break
            continue
        f = line.split()
        if f[cols["group_PDB"]] != "ATOM":
            continue
        ch, seq, atom, b = f[cols["auth_asym_id"]], int(f[cols["auth_seq_id"]]), f[cols["label_atom_id"]], float(f[cols["B_iso_or_equiv"]])
        r = res.setdefault(ch, {}).setdefault(seq, {})
        r.setdefault("first", b)
        r[atom] = b
    return {ch: {k: [rr.get(k) for _, rr in sorted(v.items())] for k in ("CA", "N", "first")} for ch, v in res.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for a in map(Path, sys.argv[1:]):
        if a.is_dir():                      # POLE_folders layout: read the job JSON directly
            js = [j for j in a.rglob("*_data.json") if not j.name.startswith("._")]
            rec = {"archive": str(a), "data_json_member": str(js[0]), "data_json": data_json_summary(json.loads(js[0].read_text()))}
            (OUT / f"{a.name}.json").write_text(json.dumps(rec))
            print(a.name, "(folder)", flush=True)
            for c in rec["data_json"]["chains"]:
                print("   ", c["kind"], c["id"], c["len"], "unpaired", c["unpaired"], "paired", c["paired"].get("n_seqs"),
                      "templates", c.get("n_templates"), c.get("template_ids", [])[:6], c.get("templates_sha"), flush=True)
            continue
        rec = {"archive": str(a), "bytes": a.stat().st_size, "members": []}
        with tarfile.open(a, "r|gz") as tf:
            for m in tf:
                rec["members"].append([m.name, m.size, m.mtime])
                if not m.isfile():
                    continue
                if m.name.endswith("_data.json") and m.name.count("/") <= 2:
                    d = json.load(tf.extractfile(m))
                    rec["data_json_member"] = m.name
                    rec["data_json"] = data_json_summary(d)
                elif MODEL.search(m.name) and "model_cif" not in rec:
                    rec["model_cif"] = m.name
                    rec["model_plddt"] = atom_conventions(tf.extractfile(m).read().decode())
        mt = [x[2] for x in rec["members"] if x[1] > 0]
        rec["mtime_min"] = datetime.fromtimestamp(min(mt), timezone.utc).isoformat()
        rec["mtime_max"] = datetime.fromtimestamp(max(mt), timezone.utc).isoformat()
        rec["top_dirs"] = sorted({x[0].split("/")[0] for x in rec["members"]})[:5]
        (OUT / f"{a.name.replace('.tar.gz', '')}.json").write_text(json.dumps(rec))
        dj = rec.get("data_json", {})
        print(a.name, rec["mtime_min"], "->", rec["mtime_max"], "top:", rec["top_dirs"][:2], flush=True)
        for c in dj.get("chains", []):
            print("   ", c["kind"], c["id"], c["len"], "unpaired", c["unpaired"], "paired", c["paired"].get("n_seqs"),
                  "templates", c.get("n_templates"), c.get("template_ids", [])[:6], c.get("templates_sha"), flush=True)


if __name__ == "__main__":
    main()
