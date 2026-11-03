#!/usr/bin/env python
"""How much does ONE substitution change AF3's MSA? (MSH2, 2026-10-15)

AF3 searches the MSA from the query sequence, so every variant job got its own MSA.
The provenance hashes (experiments/wt_variant_offset/results/provenance) show that the
search is deterministic: the WT MSH2 chain got the byte-identical MSA (hash b89d88) in
the old MutSa WT job (03-24) and in the MSH6 G:T jobs (09-25), where MSH2 is unmutated.
So the MSA differences between jobs are caused by the substitution, not by search noise.

This compares, for the MSH2 chain (A):
  WT MSA      -- from the MSH6 G:T WT seed-extension job (MSH2 unmutated; checked to be b89d88)
  Q629R MSA   -- the source variant of MutS_alpha_MSH2_WT_vartmpl (= the new WT's MSA)
Reports: row counts, the overlap of the homolog sets (aligned match-column strings), how
many shared homologs are aligned differently, and where in the sequence the differences are.

    python -I experiments/wt_msa_check/msa_response_msh2.py
"""
import os
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_vartmpl_msa import match_cols, records  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
WT_ZIP = REPO / "muts_jsons/MutS_alpha_MSH6_GT_WT_seed_ext_26-75.zip"
Q629R = Path(os.environ.get("DRIFT_ARCHIVE_ROOT", "$DRIFT_ARCHIVE_ROOT")) / "wt_vartmpl_src/MutS_alpha_MSH2_Q629R_data/MutS_alpha_MSH2_Q629R/MutS_alpha_MSH2_Q629R_data.json"


def sha_without_query(a3m):
    recs = a3m.split(">")[1:]
    return hashlib.sha1(">".join(recs[1:]).encode()).hexdigest()[:12]


def chain(d, cid):
    for e in d["sequences"]:
        (_, x), = e.items()
        if x.get("id") == cid:
            return x
    raise KeyError(cid)


def compare(a3m_wt, a3m_var, label):
    rw, rv = records(a3m_wt), records(a3m_var)
    L = len(match_cols(rw[0][1]))
    # key a homolog by its header (database id + hit range); value = aligned match columns
    kw = {h: match_cols(s) for h, s in rw[1:]}
    kv = {h: match_cols(s) for h, s in rv[1:]}
    shared = kw.keys() & kv.keys()
    realigned = [h for h in shared if kw[h] != kv[h]]
    only_wt, only_var = kw.keys() - kv.keys(), kv.keys() - kw.keys()
    # where do the realigned homologs differ?
    pos = Counter()
    for h in realigned:
        a, b = kw[h], kv[h]
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                pos[i + 1] += 1
    # coverage profile difference (fraction of rows with a residue, not a gap, per column)
    def cov(rows):
        n = len(rows)
        c = [0] * L
        for s in rows:
            for i, ch in enumerate(s):
                if ch != "-":
                    c[i] += 1
        return [x / n for x in c]
    cw, cv = cov(list(kw.values())), cov(list(kv.values()))
    dcov = [abs(a - b) for a, b in zip(cw, cv)]
    return {
        "msa": label, "rows_wt": len(rw), "rows_var": len(rv),
        "homologs_shared": len(shared), "only_in_wt": len(only_wt), "only_in_variant": len(only_var),
        "jaccard": round(len(shared) / len(kw.keys() | kv.keys()), 5),
        "shared_but_realigned": len(realigned),
        "realigned_positions_top": [p for p, _ in pos.most_common(10)],
        "coverage_absdiff_max": round(max(dcov), 4),
        "coverage_absdiff_mean": round(sum(dcov) / L, 5),
        "coverage_absdiff_max_at": dcov.index(max(dcov)) + 1,
    }


def main():
    z = zipfile.ZipFile(WT_ZIP)
    wt = json.loads(z.read(sorted(z.namelist())[0]))
    var = json.loads(Q629R.read_text())
    a_wt, a_var = chain(wt, "A"), chain(var, "A")
    print("WT  chain A unpaired hash", sha_without_query(a_wt["unpairedMsa"]), "(old MutSa WT job: b89d88...)")
    print("Q629R chain A unpaired hash", sha_without_query(a_var["unpairedMsa"]), "(b19e16...)")
    diffs = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a_wt["sequence"], a_var["sequence"])) if x != y]
    print("sequence differences WT vs Q629R:", diffs)
    rows = [compare(a_wt[k], a_var[k], k) for k in ("unpairedMsa", "pairedMsa")]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "msa_response_msh2.json").write_text(json.dumps({"diffs": diffs, "rows": rows}, indent=1))
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
