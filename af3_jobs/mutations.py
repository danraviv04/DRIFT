"""Variant-list construction: parsing, ClinVar filtering, codon-distance enumeration.

The ClinVar gene-matching rules and the codon table are carried over unchanged
from ``generate_af3_ensembles.py``; they are the reason the PTEN batch does not
pick up KLLN rows from a ``KLLN|PTEN`` locus export.
"""

from __future__ import annotations

import re
from pathlib import Path

from af3_jobs.spec import REPO_ROOT, MutationSourceSpec

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

AA_TO_CODONS: dict[str, list[str]] = {}
for _codon, _aa in CODON_TABLE.items():
    if _aa != "*":
        AA_TO_CODONS.setdefault(_aa, []).append(_codon)

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}


def min_nt_distance(aa1: str, aa2: str) -> int:
    """Fewest single-nucleotide substitutions between any codon of aa1 and aa2."""
    c1s = AA_TO_CODONS.get(aa1, [])
    c2s = AA_TO_CODONS.get(aa2, [])
    if not c1s or not c2s:
        return 3
    return min(sum(a != b for a, b in zip(c1, c2)) for c1 in c1s for c2 in c2s)


def parse_mutation_name(mut_name: str) -> tuple[str, int, str]:
    """Parse a variant name into ``(wt_aa, position, mut_aa)``.

    Accepts a bare name (``V3L``, ``Val3Leu``) or a full ClinVar annotation
    (``NM_006218.4(PIK3CA):c.113G>A (p.Arg38His)``).

    The ``p.``-anchored forms are tried first, and the loose one-letter form is
    accepted only when it is the *entire* string. Searching for
    ``[A-Z]\\d+[A-Z]`` anywhere would read the gene symbol PIK3CA as the variant
    "K3C" -- every row of a PIK3CA export then parses to the same name and
    collapses to one variant. PTEN, KRAS, MSH2, TP53, VHL and POLE have no digit
    in their symbols, which is why the looser rule survived this long.
    """
    s = str(mut_name).strip()

    # HGVS protein term, three-letter then one-letter.
    m = re.search(r"p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", s)
    if m:
        orig3, pos, new3 = m.groups()
        if orig3 not in THREE_TO_ONE or new3 not in THREE_TO_ONE:
            raise ValueError(f"Unknown amino acid code in mutation: {s}")
        return THREE_TO_ONE[orig3], int(pos), THREE_TO_ONE[new3]

    m = re.search(r"p\.\(?([A-Z])(\d+)([A-Z])\b", s)
    if m:
        orig_aa, pos, new_aa = m.groups()
        return orig_aa, int(pos), new_aa

    # Bare three-letter notation, e.g. "Val3Leu".
    m = re.fullmatch(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", s)
    if m:
        orig3, pos, new3 = m.groups()
        if orig3 not in THREE_TO_ONE or new3 not in THREE_TO_ONE:
            raise ValueError(f"Unknown amino acid code in mutation: {s}")
        return THREE_TO_ONE[orig3], int(pos), THREE_TO_ONE[new3]

    # Bare one-letter notation, e.g. "V3L" -- whole string only.
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", s)
    if m:
        orig_aa, pos, new_aa = m.groups()
        return orig_aa, int(pos), new_aa

    raise ValueError(f"Unsupported mutation format: {s}")


def _first_row_text(row, columns: tuple[str, ...]) -> str | None:
    import pandas as pd

    for column in columns:
        if column not in row or pd.isna(row[column]):
            continue
        value = str(row[column]).strip()
        if value:
            return value
    return None


def clinvar_row_matches_gene(row, target_gene: str | None) -> bool:
    """Whether a ClinVar row is annotated for the gene whose sequence we mutate.

    The transcript-level annotation in ``Variation``/``Name`` is authoritative.
    This prevents broad locus fields such as ``KLLN|PTEN`` from admitting a KLLN
    variant when the target sequence is PTEN. Older exports without a transcript-
    level gene annotation fall back to exact tokens in ``Genes``/``Gene(s)``.
    """
    if target_gene is None:
        return True

    target = target_gene.strip().upper()
    variation = _first_row_text(row, ("Variation", "Name"))
    if variation is not None:
        annotated_gene = re.search(r"\(([^()]+)\):", variation)
        if annotated_gene is not None:
            return annotated_gene.group(1).strip().upper() == target

    genes = _first_row_text(row, ("Genes", "Gene(s)"))
    if genes is None:
        return False
    gene_tokens = {
        gene.strip().upper() for gene in re.split(r"[|,;]", genes) if gene.strip()
    }
    # Without a transcript-level annotation, multi-gene locus rows are ambiguous.
    return gene_tokens == {target}


CLINVAR_MUTATION_COLUMNS = ("Variation", "Name", "Protein change")


def clinvar_mutation_text(row) -> str | None:
    """Select the most gene-specific mutation annotation available in a row."""
    return _first_row_text(row, CLINVAR_MUTATION_COLUMNS)


def parse_clinvar_row(row) -> tuple[str, int, str]:
    """Parse a row's variant, trying each annotation column in turn.

    The transcript-level ``Variation``/``Name`` is preferred because it is
    gene-specific, but a haplotype or a multi-change row carries no single
    substitution there, and the plain ``Protein change`` column often still does.

    The fallback applies only when the name carries no transcript-level coding
    annotation (``NM_...(GENE):c.``). When it does, the name is authoritative:
    a synonymous (``p.Lys22=``) or splice-site (``c.457+1G>A``) row is not a
    missense on this transcript, and its ``Protein change`` lists substitutions
    on *other* transcripts. Some of those hit the canonical residue by chance
    and pass the WT-residue check (MSH6 export, 2026-09-25).
    """
    name = _first_row_text(row, ("Variation", "Name"))
    columns = CLINVAR_MUTATION_COLUMNS
    if name is not None and re.search(r"\([^()]+\):c\.", name):
        columns = ("Variation", "Name")
    errors = []
    for column in columns:
        text = _first_row_text(row, (column,))
        if text is None:
            continue
        try:
            return parse_mutation_name(text)
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError("; ".join(errors) or "no mutation annotation in row")


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _read_clinvar_table(path: Path):
    """Read a ClinVar export, whether it came down as CSV or TSV.

    The web UI's Download button emits tab-separated files; earlier exports in
    Ground_Truths/ClinVar are comma-separated. The header line decides, since a
    ClinVar row's Condition(s) field is full of commas either way.
    """
    import pandas as pd

    header = path.read_text(errors="replace").split("\n", 1)[0]
    sep = "\t" if header.count("\t") > header.count(",") else ","
    # index_col=False is load-bearing: ClinVar ends every data row with a
    # trailing separator, so rows carry one more field than the header. Left to
    # itself pandas promotes the first column to the index and shifts every
    # column left by one -- silently, and the gene filter then rejects 100% of
    # rows as if the export were for the wrong gene.
    return pd.read_csv(path, sep=sep, index_col=False)


def _from_clinvar(spec: MutationSourceSpec, log: list[str]) -> dict[str, tuple[str, int, str]]:
    csv_path = _resolve_path(spec.csv)
    if not csv_path.exists():
        # A .csv/.tsv mix-up should not read as "you never downloaded it".
        alternatives = [
            p for p in (csv_path.with_suffix(".tsv"), csv_path.with_suffix(".csv"))
            if p.exists()
        ]
        if alternatives:
            raise FileNotFoundError(
                f"ClinVar export not found: {csv_path}\n"
                f"Did you mean {alternatives[0]}? Point mutations.csv at it."
            )
        raise FileNotFoundError(
            f"ClinVar export not found: {csv_path}\n"
            "Export the gene's variants from the ClinVar web UI into that path, "
            "or switch mutations.source to 'list'."
        )
    frame = _read_clinvar_table(csv_path)
    found: dict[str, tuple[str, int, str]] = {}
    unparsed: list[str] = []
    skipped_other_gene = 0
    for _, row in frame.iterrows():
        if not clinvar_row_matches_gene(row, spec.gene):
            skipped_other_gene += 1
            continue
        try:
            orig_aa, pos, new_aa = parse_clinvar_row(row)
        except ValueError as exc:
            unparsed.append(str(exc))
            continue
        found[f"{orig_aa}{pos}{new_aa}"] = (orig_aa, pos, new_aa)
    if not found and len(frame):
        raise ValueError(
            f"{csv_path.name}: none of its {len(frame)} rows parsed as a "
            f"{spec.gene} variant. Check that the export is for {spec.gene} and "
            "that its columns line up -- a shifted header silently turns every "
            "row into an 'other gene' row."
        )
    log.append(
        f"  ClinVar {csv_path.name}: {len(found)} {spec.gene or 'unfiltered'} variants "
        f"from {len(frame)} rows ({skipped_other_gene} annotated to other genes, "
        f"{len(unparsed)} not a single substitution)"
    )
    for message in unparsed:
        log.append(f"    unparsed: {message}")
    return found


def _enumerated(
    sequence: str, first_residue: int, max_nt_subs: int, log: list[str]
) -> dict[str, tuple[str, int, str]]:
    found: dict[str, tuple[str, int, str]] = {}
    for offset, orig in enumerate(sequence):
        if orig not in AA_TO_CODONS:
            continue
        pos = first_residue + offset
        for new in sorted(AA_TO_CODONS):
            if new == orig or min_nt_distance(orig, new) > max_nt_subs:
                continue
            found[f"{orig}{pos}{new}"] = (orig, pos, new)
    log.append(f"  enumerated {len(found)} variants within {max_nt_subs} nt change(s)")
    return found


def collect_mutations(
    spec: MutationSourceSpec,
    sequence: str,
    first_residue: int = 1,
) -> tuple[dict[str, tuple[str, int, str]], list[str]]:
    """Build the variant list for a batch.

    Returns the ``{name: (wt_aa, position, mut_aa)}`` mapping plus a log of what
    each stage contributed, which the CLI prints and the manifest records.
    """
    log: list[str] = []
    found: dict[str, tuple[str, int, str]] = {}

    if spec.source == "clinvar":
        found = _from_clinvar(spec, log)
    elif spec.source == "enumerate":
        found = _enumerated(sequence, first_residue, spec.max_nt_subs, log)
    elif spec.source == "file":
        path = _resolve_path(spec.path)
        names = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        for name in names:
            orig_aa, pos, new_aa = parse_mutation_name(name)
            found[f"{orig_aa}{pos}{new_aa}"] = (orig_aa, pos, new_aa)
        log.append(f"  {path.name}: {len(found)} variants")
    elif spec.source in ("list", "none"):
        pass
    else:
        raise ValueError(f"unknown mutation source '{spec.source}'")

    for name in spec.include:
        orig_aa, pos, new_aa = parse_mutation_name(name)
        key = f"{orig_aa}{pos}{new_aa}"
        if key not in found:
            found[key] = (orig_aa, pos, new_aa)
    if spec.include:
        log.append(f"  include: +{len(spec.include)} forced variant(s)")

    dropped = {n.strip() for n in spec.exclude} | {n.strip() for n in spec.done}
    if dropped:
        before = len(found)
        found = {k: v for k, v in found.items() if k not in dropped}
        log.append(f"  exclude/done: -{before - len(found)} variant(s)")

    return found, log
