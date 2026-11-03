"""A WT job that sees its variants' templates, built from one variant's finished ``*_data.json``.

Why: AF3's template search drops a hit that exactly matches the query over >95% of its
length (``max_subsequence_ratio``). When the PDB holds the exact WT construct, the WT job
loses that template and every variant keeps it (one substitution breaks the match), so
the WT background is offset from all variants alike (MSH2: 2O8B; PIK3CA: 4L1B, 5DXH;
see experiments/wt_variant_offset/).

The fix: take a typical variant's MSA- and template-ready ``_data.json``, put the WT
residue back in the query, and run inference only (``--run_data_pipeline=false``, as for
every ``*_data.json`` the runners see). Exactly three things change:
  - the target chain's ``sequence``,
  - the query row (first record) of its ``unpairedMsa`` and ``pairedMsa``,
  - ``name`` and ``modelSeeds``.
Every other key, the templates included, is copied unchanged. A substitution moves no
alignment column, so the template ``queryIndices``/``templateIndices`` and the other
MSA rows stay valid.

Before writing anything, the source is checked against the batch config: every protein
chain must equal the config's WT sequence except the target chain, which must carry
exactly the named variant, and the MSA query rows must equal that variant sequence.

    python -m af3_jobs.wt_from_variant MutS_alpha_MSH2_Q629R_data.json --config MutS_alpha_MSH2 \\
        --out muts_jsons/MutS_alpha_MSH2_WT_vartmpl

The default seeds are 1-25, the variants' own, so WT and variant draws stay seed-paired.
One job holds all of them unless ``--chunk`` splits it. Chunks are named
``<name>_seeds_<a>-<b>_data.json``, the seed-extension layout.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path

from af3_jobs.mutations import parse_mutation_name
from af3_jobs.seed_extension import make_chunks, parse_seeds
from af3_jobs.sequences import resolve_all
from af3_jobs.spec import load_spec


def _ids(entry: dict) -> tuple[str, ...]:
    i = entry["id"]
    return tuple(i) if isinstance(i, list) else (i,)


def _query_row(a3m: str) -> tuple[str, int, int]:
    """The first record's sequence and the [start, end) span of its sequence lines."""
    lines = a3m.split("\n")
    if not lines or not lines[0].startswith(">"):
        raise ValueError("MSA does not start with a '>' header")
    end = 1
    while end < len(lines) and not lines[end].startswith(">"):
        end += 1
    return "".join(lines[1:end]).strip(), 1, end


def replace_query_row(a3m: str, expected: str, new: str) -> str:
    """Swap the query row of an A3M string, refusing if it is not `expected`."""
    seq, start, end = _query_row(a3m)
    if seq != expected:
        raise ValueError(f"MSA query row ({len(seq)} aa) is not the source chain sequence")
    lines = a3m.split("\n")
    return "\n".join(lines[:start] + [new] + lines[end:])


def variant_from_name(name: str, prefix: str, suffix: str = "") -> str:
    """``MutS_alpha_MSH2_Q629R`` -> ``Q629R`` for job prefix ``MutS_alpha_MSH2``."""
    core = name[len(prefix) + 1:] if name.startswith(prefix + "_") else None
    if core is not None and suffix and core.endswith(suffix):
        core = core[: -len(suffix)]
    if not core:
        raise ValueError(f"job name {name!r} does not start with the config prefix {prefix!r}; pass --variant")
    return core


def build_wt_job(src: dict, spec, sequences: dict, variant: str) -> tuple[dict, dict]:
    """Return (WT payload, check record). Raises ValueError on any mismatch with the config."""
    wt_aa, pos, mut_aa = parse_mutation_name(variant)
    target = spec.target_chain
    expected = {c.ids: sequences[c.id] for c in spec.chains if c.kind == "protein"}
    idx = pos - target.first_residue
    wt_target = expected[target.ids]
    if not 0 <= idx < len(wt_target) or wt_target[idx] != wt_aa:
        found = wt_target[idx] if 0 <= idx < len(wt_target) else "out of range"
        raise ValueError(f"{variant}: config chain {target.id} has {found} at {pos}, not {wt_aa}")
    mut_target = wt_target[:idx] + mut_aa + wt_target[idx + 1:]

    out = copy.deepcopy(src)
    seen, check = set(), {"variant": variant, "chain": list(target.ids), "position": pos,
                          "reverted": f"{mut_aa}->{wt_aa}", "chains": {}}
    for entry in out["sequences"]:
        kind, body = next(iter(entry.items()))
        if kind != "protein":
            continue
        ids = _ids(body)
        if ids not in expected:
            raise ValueError(f"source protein chain {ids} is not in config {spec.name}")
        seen.add(ids)
        is_target = ids == target.ids
        want = mut_target if is_target else expected[ids]
        if body["sequence"] != want:
            what = f"variant {variant} of the config WT" if is_target else "the config WT"
            raise ValueError(f"source chain {ids} is not {what}")
        if not body.get("unpairedMsa"):
            raise ValueError(f"source chain {ids} has no unpairedMsa -- not an MSA-ready data.json")
        rec = {"templates": len(body.get("templates") or [])}
        if is_target:
            body["sequence"] = wt_target
            for key in ("unpairedMsa", "pairedMsa"):
                if body.get(key):
                    body[key] = replace_query_row(body[key], mut_target, wt_target)
                    rec[key] = "query row reverted"
        check["chains"]["/".join(ids)] = rec
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"config protein chains {sorted(missing)} are absent from the source")
    return out, check


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_json", type=Path, help="a finished variant job's <job>_data.json (MSAs + templates)")
    ap.add_argument("--config", required=True, help="af3_jobs config the variant was generated from")
    ap.add_argument("--variant", help="variant to revert, e.g. Q629R (default: parsed from the job name)")
    ap.add_argument("--seeds", default="1-25", help="seed range (default 1-25, the variants' own)")
    ap.add_argument("--chunk", type=int, default=0, help="seeds per job (default: all in one job)")
    ap.add_argument("--name", help="job name (default <prefix>_WT_vartmpl<suffix>)")
    ap.add_argument("--out", type=Path, required=True, help="output directory (a .zip is written next to it)")
    args = ap.parse_args(argv)

    raw = args.data_json.read_bytes()
    src = json.loads(raw)
    spec = load_spec(args.config)
    variant = args.variant or variant_from_name(src["name"], spec.prefix, spec.job_suffix)
    wt, check = build_wt_job(src, spec, resolve_all(spec.chains), variant)

    name = args.name or f"{spec.prefix}_WT_vartmpl{spec.job_suffix}"
    seeds = parse_seeds(args.seeds)
    chunks = make_chunks(seeds, args.chunk) if args.chunk else [seeds]
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for ch in chunks:
        job = dict(wt, name=name, modelSeeds=ch)
        stem = f"{name}_data" if len(chunks) == 1 else f"{name}_seeds_{ch[0]}-{ch[-1]}_data"
        path = args.out / f"{stem}.json"
        text = json.dumps(job)
        path.write_text(text)
        written.append({"file": path.name, "name": name, "modelSeeds": ch,
                        "sha256": hashlib.sha256(text.encode()).hexdigest()})
    zpath = args.out.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for w in written:
            zf.write(args.out / w["file"], w["file"])
    manifest = {
        "generated": date.today().isoformat(),
        "source_data_json": str(args.data_json),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_name": src["name"],
        "config": spec.name,
        "check": check,
        "jobs": written,
        "run": "inference only (--run_data_pipeline=false); templates are the source variant's, unchanged",
    }
    (args.out.parent / f"{args.out.name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{name}: {variant} reverted on chain {'/'.join(check['chain'])}, "
          f"{len(written)} job(s), seeds {seeds[0]}-{seeds[-1]} -> {args.out} and {zpath}")


if __name__ == "__main__":
    main()
