"""Extract AlphaMissense scores for KRAS4B from the non-canonical isoform release.

AlphaMissense's canonical file (and the hegelab site, which is where
legacy_4A/afm_KRAS4A_predictions.tsv came from) carries only KRAS4A (ENST00000256078,
UniProt P01116-1, 189 aa). KRAS4B (ENST00000311936, UniProt P01116-2, 188 aa)
is only in AlphaMissense_isoforms_aa_substitutions.tsv.gz (Zenodo 10813168,
2.5 GB). This streams that file, checks its md5 against Zenodo, and keeps
only the KRAS4B rows, so the full file never touches disk.

Outputs, in Ground_Truths/AlphaMissense/:
  AlphaMissense_isoforms_ENST00000311936.tsv   rows exactly as released
  afm_KRAS_predictions.tsv                     same rows in the hegelab layout
                                               the loaders read
  afm_KRAS_predictions.provenance.json         source, checksum, counts

Usage: python analyses/continuous/kras4b_coverage/fetch_afm_kras4b.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "Ground_Truths" / "AlphaMissense"

URL = (
    "https://zenodo.org/api/records/10813168/files/"
    "AlphaMissense_isoforms_aa_substitutions.tsv.gz/content"
)
EXPECTED_MD5 = "112b1921cfcdb3a1047532abe449ad59"
EXPECTED_BYTES = 2461351945

TRANSCRIPT = "ENST00000311936"
UNIPROT_ISOFORM = "P01116-2"
# UniProt P01116-2 (isoform 2B = K-Ras4B).
KRAS4B_SEQUENCE = (
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG"
    "QEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDL"
    "PSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKK"
    "SKTKCVIM"
)

HEGELAB_COLUMNS = [
    "# Uniprot ACC", "Entry name", "Gene name", "Ensemble id", "protein variant",
    "is_snv", "a.a.1", "position", "a.a.2", "pathogenicity score",
    "pathogenicity class", "REVEL pathogenicity score", "REVEL pathogenicity class",
]

CHUNK = 1 << 20
MAX_RETRIES = 8


def stream_rows():
    """Return (header_lines, target_lines, md5_hex, n_bytes)."""
    target = (TRANSCRIPT + ".").encode()
    md5 = hashlib.md5()
    inflater = zlib.decompressobj(31)
    carry = b""
    header: list[bytes] = []
    header_done = False
    rows: list[bytes] = []
    received = 0
    retries = 0
    next_report = 0
    started = time.time()

    while received < EXPECTED_BYTES:
        request = urllib.request.Request(URL)
        if received:
            request.add_header("Range", f"bytes={received}-")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if received and response.status != 206:
                    raise RuntimeError(f"server ignored Range (HTTP {response.status})")
                while chunk := response.read(CHUNK):
                    received += len(chunk)
                    md5.update(chunk)
                    pieces = []
                    data = chunk
                    while data:
                        pieces.append(inflater.decompress(data))
                        if inflater.eof:
                            data = inflater.unused_data
                            inflater = zlib.decompressobj(31)
                        else:
                            data = b""
                    text = carry + b"".join(pieces)
                    cut = text.rfind(b"\n") + 1
                    carry, block = text[cut:], text[:cut]
                    if not header_done:
                        for line in block.split(b"\n"):
                            if line.startswith(b"#") or line.startswith(b"transcript_id"):
                                header.append(line)
                            else:
                                header_done = True
                                break
                    if target in block:
                        rows.extend(
                            line for line in block.split(b"\n") if line.startswith(target)
                        )
                    if received >= next_report:
                        elapsed = time.time() - started
                        print(
                            f"{received / 1e9:5.2f} / {EXPECTED_BYTES / 1e9:.2f} GB  "
                            f"{received / EXPECTED_BYTES:6.1%}  "
                            f"{received / 1e6 / max(elapsed, 1e-9):6.1f} MB/s  "
                            f"{len(rows)} KRAS4B rows",
                            flush=True,
                        )
                        next_report += 100_000_000
        except Exception as error:  # network hiccup: resume from the byte we reached
            retries += 1
            if retries > MAX_RETRIES:
                raise
            print(f"retry {retries} at byte {received}: {error}", flush=True)
            time.sleep(min(60, 5 * retries))

    if carry.startswith(target):
        rows.append(carry)
    return header, rows, md5.hexdigest(), received


def to_hegelab(rows: list[bytes]) -> list[list[str]]:
    out = []
    for raw in rows:
        transcript, variant, score, am_class = raw.decode().split("\t")
        out.append([
            UNIPROT_ISOFORM, "RASK_HUMAN", "KRAS", transcript, f"p.{variant}", "",
            variant[0], variant[1:-1], variant[-1], score, am_class, "", "",
        ])
    return out


def validate(table: list[list[str]]) -> dict:
    """Check that the rows describe KRAS4B and cover it completely."""
    by_position: dict[int, set[str]] = {}
    for row in table:
        wt, position, mut = row[6], int(row[7]), row[8]
        if KRAS4B_SEQUENCE[position - 1] != wt:
            raise ValueError(f"{wt}{position}{mut}: KRAS4B has {KRAS4B_SEQUENCE[position - 1]} there")
        by_position.setdefault(position, set()).add(mut)
    missing = [p for p in range(1, len(KRAS4B_SEQUENCE) + 1) if p not in by_position]
    incomplete = {p: 19 - len(m) for p, m in by_position.items() if len(m) != 19}
    return {
        "positions": len(by_position),
        "max_position": max(by_position),
        "positions_missing": missing,
        "positions_without_19_substitutions": incomplete,
    }


def main() -> int:
    header, rows, md5_hex, n_bytes = stream_rows()
    if md5_hex != EXPECTED_MD5:
        print(f"md5 mismatch: got {md5_hex}, Zenodo says {EXPECTED_MD5}; nothing written")
        return 1
    if not rows:
        print(f"{TRANSCRIPT} is not in the isoform file; nothing written")
        return 1

    table = to_hegelab(rows)
    checks = validate(table)

    raw_path = OUT_DIR / f"AlphaMissense_isoforms_{TRANSCRIPT}.tsv"
    raw_path.write_bytes(b"\n".join(header + rows) + b"\n")

    hegelab_path = OUT_DIR / "afm_KRAS_predictions.tsv"
    with hegelab_path.open("w") as handle:
        handle.write("\t".join(HEGELAB_COLUMNS) + "\n")
        for row in table:
            handle.write("\t".join(row) + "\n")

    provenance = {
        "source": URL,
        "zenodo_record": "10.5281/zenodo.10813168",
        "file": "AlphaMissense_isoforms_aa_substitutions.tsv.gz",
        "md5": md5_hex,
        "bytes": n_bytes,
        "transcript": TRANSCRIPT,
        "uniprot_isoform": UNIPROT_ISOFORM,
        "isoform": "KRAS4B (188 aa)",
        "rows": len(rows),
        "validation": checks,
        "caveat": "AlphaMissense: predictions for non-canonical isoforms were not "
                  "thoroughly evaluated and should be used with caution.",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT_DIR / "afm_KRAS_predictions.provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
