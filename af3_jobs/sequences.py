"""Sequence resolution for chain specs: literal, or fetched from UniProt.

Fetched sequences are cached under ``af3_jobs/seq_cache/`` and committed, so a
batch can be regenerated offline and so the exact sequence that went into a run
is auditable without trusting UniProt to be stable.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from af3_jobs.spec import ChainSpec

CACHE_DIR = Path(__file__).resolve().parent / "seq_cache"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


def fetch_uniprot(accession: str, *, refresh: bool = False) -> str:
    """Full sequence for a UniProt accession. Isoforms (``P01116-1``) work too."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{accession}.fasta"
    if refresh or not cached.exists():
        url = UNIPROT_FASTA.format(acc=accession)
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode()
        if not text.startswith(">"):
            raise RuntimeError(f"{accession}: unexpected response from {url}")
        cached.write_text(text)
    text = cached.read_text()
    return "".join(
        line.strip() for line in text.splitlines() if line and not line.startswith(">")
    )


def resolve_chain_sequence(chain: ChainSpec, *, refresh: bool = False) -> str | None:
    """Resolve one chain to its final sequence, applying any residue range.

    Ligand chains return ``None``. A ``residue_range`` is interpreted in the
    source numbering (1-based, inclusive on both ends), which for a UniProt
    chain is UniProt numbering.
    """
    if chain.kind == "ligand":
        return None

    if chain.sequence is not None:
        sequence = chain.sequence
        # A literal sequence is taken as already trimmed unless the config also
        # gives a range whose length disagrees with it.
        if chain.residue_range is not None:
            start, end = chain.residue_range
            if len(sequence) != end - start + 1:
                sequence = sequence[start - 1 : end]
        return sequence

    sequence = fetch_uniprot(chain.uniprot, refresh=refresh)
    if chain.residue_range is not None:
        start, end = chain.residue_range
        if end > len(sequence):
            raise ValueError(
                f"chain {chain.id}: range {start}-{end} exceeds "
                f"{chain.uniprot} length {len(sequence)}"
            )
        sequence = sequence[start - 1 : end]
    return sequence


def resolve_all(chains, *, refresh: bool = False) -> dict[str, str | None]:
    return {c.id: resolve_chain_sequence(c, refresh=refresh) for c in chains}
