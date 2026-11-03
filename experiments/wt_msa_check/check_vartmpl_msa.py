#!/usr/bin/env python
"""Is the MSA of a regenerated (*_WT_vartmpl) WT job intact? (2026-10-15)

Each *_WT_vartmpl job is a variant's finished AF3 input with the WT residue put back
(af3_jobs.wt_from_variant). Its MSA is therefore the variant's MSA. For each gene this
compares the new WT job with its source variant job and checks:

  1. diff     -- nothing changed except the target chains' sequence, the query row of their
                 unpaired/paired MSA, name and modelSeeds (templates byte-identical);
  2. query    -- every protein chain's MSA query row equals that chain's sequence;
  3. a3m      -- every MSA row has exactly len(query) match columns (uppercase or '-'),
                 so no row is shifted relative to the reverted query;
  4. site     -- at the substituted column: residue counts among the non-query rows
                 (the homologs' view of the site, identical in WT and variant jobs).

Reads only local copies (no drive):
    python -I experiments/wt_msa_check/check_vartmpl_msa.py
"""
import os
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("DRIFT_ARCHIVE_ROOT", "$DRIFT_ARCHIVE_ROOT")) / "wt_vartmpl_src"
OUT = Path(__file__).resolve().parent / "results"
PAIRS = {
    "MSH2": (SRC / "MutS_alpha_MSH2_Q629R_data/MutS_alpha_MSH2_Q629R/MutS_alpha_MSH2_Q629R_data.json",
             REPO / "muts_jsons/MutS_alpha_MSH2_WT_vartmpl_data.json"),
    "PIK3CA": (SRC / "PIK3CA_K532R_data/PIK3CA_K532R/PIK3CA_K532R_data.json",
               "zip:" + str(REPO / "muts_jsons/PIK3CA_WT_vartmpl.zip")),
    "TP53": (SRC / "P53_A119D_with_ZN_and_DNA_data/P53_A119D_with_ZN_and_DNA/P53_A119D_with_ZN_and_DNA_data.json",
             REPO / "muts_jsons/P53_WT_vartmpl_with_ZN_and_DNA_data.json"),
}
MATCH = re.compile(r"[A-Z-]")


def load(p):
    if str(p).startswith("zip:"):
        import zipfile
        z = zipfile.ZipFile(str(p)[4:])
        return json.loads(z.read(z.namelist()[0]))
    return json.loads(Path(p).read_text())


def records(a3m):
    out, name, seq = [], None, []
    for line in a3m.splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(seq)))
            name, seq = line[1:], []
        elif line.strip():
            seq.append(line.strip())
    if name is not None:
        out.append((name, "".join(seq)))
    return out


def match_cols(s):
    return "".join(ch for ch in s if MATCH.fullmatch(ch))


def check_gene(gene, src_path, new_path):
    src, new = load(src_path), load(new_path)
    rep = {"gene": gene, "source": src["name"], "new": new["name"], "problems": []}
    top_changed = sorted(k for k in set(src) | set(new) if k != "sequences" and src.get(k) != new.get(k))
    rep["top_level_changed"] = top_changed
    if set(top_changed) - {"name", "modelSeeds"}:
        rep["problems"].append(f"unexpected top-level change: {top_changed}")
    chains = []
    for s_ent, n_ent in zip(src["sequences"], new["sequences"]):
        (kind, s), = s_ent.items()
        (_, n), = n_ent.items()
        c = {"id": s.get("id"), "kind": kind}
        changed = sorted(k for k in set(s) | set(n) if s.get(k) != n.get(k))
        c["keys_changed"] = changed
        if kind != "protein":
            if changed:
                rep["problems"].append(f"{kind} {c['id']} changed: {changed}")
            chains.append(c)
            continue
        seq_s, seq_n = s["sequence"], n["sequence"]
        diffs = [(i + 1, a, b) for i, (a, b) in enumerate(zip(seq_s, seq_n)) if a != b]
        c["len"], c["seq_diffs"] = len(seq_n), diffs
        if len(seq_s) != len(seq_n) or len(diffs) > 1:
            rep["problems"].append(f"chain {c['id']}: {len(diffs)} sequence differences")
        if set(changed) - {"sequence", "unpairedMsa", "pairedMsa"}:
            rep["problems"].append(f"chain {c['id']}: unexpected keys changed {changed}")
        for msa_key in ("unpairedMsa", "pairedMsa"):
            rs, rn = records(s.get(msa_key) or ""), records(n.get(msa_key) or "")
            m = {"rows": len(rn)}
            if not rn:
                c[msa_key] = m
                continue
            m["query_equals_sequence"] = rn[0][1] == seq_n
            m["src_query_equals_src_sequence"] = rs[0][1] == seq_s
            m["non_query_rows_identical"] = rs[1:] == rn[1:]
            bad = [i for i, (_, r) in enumerate(rn) if len(match_cols(r)) != len(seq_n)]
            m["rows_with_wrong_match_length"] = len(bad)
            m["distinct_non_query_seqs"] = len({match_cols(r) for _, r in rn[1:]})
            if diffs:
                pos = diffs[0][0] - 1
                col = Counter(match_cols(r)[pos] for _, r in rn[1:])
                m["site_column_top"] = col.most_common(6)
                m["site_frac_wt_residue"] = round(col.get(diffs[0][2], 0) / max(1, len(rn) - 1), 4)
                m["site_frac_variant_residue"] = round(col.get(diffs[0][1], 0) / max(1, len(rn) - 1), 4)
                m["site_frac_gap"] = round(col.get("-", 0) / max(1, len(rn) - 1), 4)
            for flag in ("query_equals_sequence", "src_query_equals_src_sequence", "non_query_rows_identical"):
                if not m[flag]:
                    rep["problems"].append(f"chain {c['id']} {msa_key}: {flag} is False")
            if bad:
                rep["problems"].append(f"chain {c['id']} {msa_key}: {len(bad)} rows with wrong match length")
            c[msa_key] = m
        t_same = s.get("templates") == n.get("templates")
        c["templates_identical"] = t_same
        if not t_same:
            rep["problems"].append(f"chain {c['id']}: templates differ")
        chains.append(c)
    rep["chains"] = chains
    return rep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reports = [check_gene(g, *p) for g, p in PAIRS.items()]
    (OUT / "vartmpl_msa_check.json").write_text(json.dumps(reports, indent=1))
    for r in reports:
        print(f"\n== {r['gene']}: {r['new']} from {r['source']}  "
              f"-> {'OK' if not r['problems'] else 'PROBLEMS: ' + '; '.join(r['problems'])}")
        for c in r["chains"]:
            if c["kind"] != "protein":
                continue
            line = f"  chain {c['id']} len {c['len']} diffs {c['seq_diffs']} templates_identical={c['templates_identical']}"
            print(line)
            for k in ("unpairedMsa", "pairedMsa"):
                m = c.get(k, {})
                if m.get("rows"):
                    print(f"    {k}: rows {m['rows']}, distinct {m['distinct_non_query_seqs']}, "
                          f"bad rows {m['rows_with_wrong_match_length']}, "
                          f"query==seq {m['query_equals_sequence']}, rest identical {m['non_query_rows_identical']}"
                          + (f", site WT {m['site_frac_wt_residue']} var {m['site_frac_variant_residue']} "
                             f"gap {m['site_frac_gap']} top {m['site_column_top'][:4]}" if 'site_column_top' in m else ""))


if __name__ == "__main__":
    main()
