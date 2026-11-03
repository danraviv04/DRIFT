"""Command line front end: ``python -m af3_jobs <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from af3_jobs.build import build_jobs, write_jobs
from af3_jobs.sequences import fetch_uniprot
from af3_jobs.spec import (
    CONFIG_DIR,
    MutationSourceSpec,
    available_configs,
    load_spec,
)


def _apply_overrides(spec, args):
    """Let the CLI override the config's mutation source for a one-off batch."""
    from dataclasses import replace

    if getattr(args, "wt_only", False):
        return replace(spec, mutations=MutationSourceSpec(source="none"))
    variants = getattr(args, "variants", None)
    if variants:
        names = tuple(n.strip() for n in variants.split(",") if n.strip())
        return replace(
            spec,
            mutations=MutationSourceSpec(
                source="list", gene=spec.mutations.gene, include=names
            ),
        )
    return spec


def _cmd_list(args: argparse.Namespace) -> int:
    configs = available_configs()
    if not configs:
        print(f"No configs in {CONFIG_DIR}")
        return 0
    for path in configs:
        spec = load_spec(str(path))
        chains = ", ".join(
            f"{c.id}={c.gene or c.uniprot or '+'.join(c.ccd_codes) or c.kind}"
            f"{'*' if c.mutate else ''}"
            for c in spec.chains
        )
        print(f"{path.stem:16s} {spec.mutations.source:10s} [{chains}]")
        if spec.description:
            print(f"{'':16s} {spec.description}")
    print("\n(* marks the mutated chain)")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    spec = _apply_overrides(load_spec(args.config), args)
    result = build_jobs(spec, refresh_sequences=args.refresh)
    print(f"{spec.name}  ({spec.description or 'no description'})")
    print(f"  config     {spec.source_path}")
    print(f"  seeds      {len(spec.model_seeds)} "
          f"-> {len(spec.model_seeds) * 5} structures per variant")
    total = 0
    for chain in spec.chains:
        seq = result.sequences[chain.id]
        if chain.kind == "ligand":
            print(f"  chain {chain.id}    ligand {'+'.join(chain.ccd_codes)}")
            continue
        total += len(seq)
        rng = (
            f" {chain.residue_range[0]}-{chain.residue_range[1]}"
            if chain.residue_range
            else ""
        )
        mark = "  <-- mutated" if chain.mutate else ""
        print(
            f"  chain {chain.id}    {chain.gene or chain.uniprot or ''}"
            f"{rng}  {len(seq)} aa{mark}"
        )
    print(f"  total      {total} residues")
    for line in result.log:
        print(line)
    print(f"  jobs       {len(result.jobs)} "
          f"({sum(1 for j in result.jobs if j.variant)} variants"
          f"{' + WT' if spec.include_wt else ''})")
    if result.skipped:
        print(f"  skipped    {len(result.skipped)}")
        for line in result.skipped[: args.max_skipped]:
            print(f"    {line}")
        if len(result.skipped) > args.max_skipped:
            print(f"    ... {len(result.skipped) - args.max_skipped} more")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    spec = _apply_overrides(load_spec(args.config), args)
    result = build_jobs(spec, refresh_sequences=args.refresh)

    for line in result.log:
        print(line)
    if result.skipped:
        print(f"Skipped {len(result.skipped)} variant(s):")
        for line in result.skipped[: args.max_skipped]:
            print(f"  {line}")
        if len(result.skipped) > args.max_skipped:
            print(f"  ... {len(result.skipped) - args.max_skipped} more")

    n_variants = sum(1 for j in result.jobs if j.variant)
    if args.dry_run:
        print(f"[dry run] would write {len(result.jobs)} JSONs "
              f"({n_variants} variants) for {spec.name}")
        return 0

    written = write_jobs(
        result,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        write_zip=not args.no_zip,
        write_dir=args.unpacked,
    )
    for kind, path in written.items():
        print(f"{kind:9s} {path}")
    print(f"Wrote {len(result.jobs)} JSONs ({n_variants} variants) for {spec.name}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    sequence = fetch_uniprot(args.accession, refresh=args.refresh)
    if args.range:
        start, _, end = args.range.partition("-")
        sequence = sequence[int(start) - 1 : int(end)]
    print(f"# {args.accession}{' ' + args.range if args.range else ''}  "
          f"{len(sequence)} aa")
    for i in range(0, len(sequence), 60):
        print(sequence[i : i + 60])
    return 0


def _cmd_scaffold(args: argparse.Namespace) -> int:
    path = CONFIG_DIR / f"{args.name}.json"
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    chains = []
    for i, entry in enumerate(args.chain):
        acc, _, rng = entry.partition(":")
        chain = {
            "id": chr(ord("A") + i),
            "type": "protein",
            "uniprot": acc,
            "mutate": i == 0,
        }
        if rng:
            chain["range"] = rng
        chains.append(chain)
    config = {
        "name": args.name,
        "description": args.description or f"{args.name} AF3 mutation ensembles.",
        "model_seeds": 25,
        "dialect": "alphafold3",
        "version": 4,
        "chains": chains,
        "mutations": {
            "source": "clinvar",
            "csv": f"Ground_Truths/ClinVar/clinvar_result_{args.name.lower()}.csv",
            "gene": args.name,
            "include": [],
            "exclude": [],
        },
        "include_wt": True,
        "output": {"dir": "muts_jsons", "tag": ""},
    }
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {path} -- edit it, then: python -m af3_jobs show {args.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m af3_jobs",
        description="Generate AlphaFold 3 input JSONs for a mutation ensemble.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list available configs")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="describe a batch without writing it")
    p_show.add_argument("config", help="config name (PIK3CA) or path to a JSON file")
    p_show.add_argument("--refresh", action="store_true", help="re-fetch UniProt")
    p_show.add_argument("--max-skipped", type=int, default=20)
    p_show.add_argument("--variants", metavar="A1G,B2C",
                       help="ignore the config's mutation source and use exactly "
                            "these variants")
    p_show.add_argument("--wt-only", action="store_true",
                       help="build the WT job alone (useful to sanity-check a new "
                            "complex before committing to a full batch)")
    p_show.set_defaults(func=_cmd_show)

    p_gen = sub.add_parser("generate", help="write the JSONs")
    p_gen.add_argument("config", help="config name (PIK3CA) or path to a JSON file")
    p_gen.add_argument("--out-dir", help="override output.dir from the config")
    p_gen.add_argument("--no-zip", action="store_true", help="skip the zip archive")
    p_gen.add_argument("--unpacked", action="store_true",
                       help="also write the JSONs as loose files in a directory")
    p_gen.add_argument("--dry-run", action="store_true")
    p_gen.add_argument("--refresh", action="store_true", help="re-fetch UniProt")
    p_gen.add_argument("--max-skipped", type=int, default=20)
    p_gen.add_argument("--variants", metavar="A1G,B2C",
                       help="ignore the config's mutation source and use exactly "
                            "these variants")
    p_gen.add_argument("--wt-only", action="store_true",
                       help="build the WT job alone (useful to sanity-check a new "
                            "complex before committing to a full batch)")
    p_gen.set_defaults(func=_cmd_generate)

    p_fetch = sub.add_parser("fetch", help="print a UniProt sequence (and cache it)")
    p_fetch.add_argument("accession", help="e.g. P42336 or P01116-1")
    p_fetch.add_argument("--range", help="1-based inclusive, e.g. 322-600")
    p_fetch.add_argument("--refresh", action="store_true")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_new = sub.add_parser("scaffold", help="write a starter config for a new protein")
    p_new.add_argument("name", help="config name, e.g. BRAF")
    p_new.add_argument("--chain", action="append", default=[], required=True,
                       metavar="ACC[:START-END]",
                       help="repeatable; first chain is the mutated one")
    p_new.add_argument("--description")
    p_new.add_argument("--force", action="store_true")
    p_new.set_defaults(func=_cmd_scaffold)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        # Config and input problems are the user's to fix; a traceback only
        # buries the one line that says what is wrong.
        print(f"error: {exc}", file=sys.stderr)
        return 2
