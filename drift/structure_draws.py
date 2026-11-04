"""Structure draws and pLDDT matrix reading for DRIFT.

Every KS value goes through here: which predictions each (variant, size, seed) draw
uses (SamplingPolicy, select_structure_columns, source_draw_key), where the pLDDT
matrices are found (discover_wt_csv_sources, discover_mut_csv_sources) and how they
are read. build_variant_table lists the variants a draw can score, which is the row
set of the score caches.
"""
from __future__ import annotations

import glob
import hashlib
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser

# Token-bounded, as in metric_analysis/pipeline/labels.py: unbounded, the gene
# name in "PIK3CA_H1047R" matched first and every PIK3CA file (WT included)
# parsed as "K3C". No other dataset's names change (checked 2026-10-12 on every
# pLDDT filename and label-source string DRIFT feeds this function).
_MUTATION_RE = re.compile(r"(?<![A-Z0-9])([A-Z])(\d+)([A-Z])(?![A-Z0-9])")
_KNOWN_GENES = {"TP53", "POLE", "MutSalpha"}
_GENE_CANONICAL = {
    "P53": "TP53",
    "TP53": "TP53",
    "POLE": "POLE",
    "MUTSALPHA": "MutSalpha",
    "MUTS_ALPHA": "MutSalpha",
}


def canonical_gene(text: Any) -> str | None:
    if pd.isna(text):
        return None
    key = str(text).upper().strip()
    return _GENE_CANONICAL.get(key, key)


def normalize_mutation(text: Any) -> str | None:
    if pd.isna(text):
        return None
    match = _MUTATION_RE.search(str(text).upper())
    if not match:
        return None
    return f"{match.group(1)}{match.group(2)}{match.group(3)}"


def infer_gene_from_variant(value: Any) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).upper().strip()
    if "_" in s:
        token = canonical_gene(s.split("_", 1)[0])
        if token in _KNOWN_GENES:
            return token
    for token in ["TP53", "P53", "POLE"]:
        if token in s:
            return canonical_gene(token)
    return None


SAMPLING_MODES = ("nested", "independent")
SELECTION_RULES = ("random", "random_pinned", "top_confidence")


@dataclass(frozen=True)
class SamplingPolicy:
    """How the N structures behind one variant's descriptors are chosen from the pool.

    mode
        ``nested``      the default. One seeded permutation per (variant, seed);
                        the size-N sample is its length-N prefix, so a smaller N
                        is contained in every larger N. Comparisons across N are
                        paired by common random numbers, which is what makes the
                        ensemble-size curve a low-variance within-variant
                        contrast. Marginally each N is still a uniform random
                        subset, so no individual size is biased.
        ``independent`` a fresh permutation per (variant, seed, N). Sizes are no
                        longer nested, so cross-N differences are unpaired and
                        noisier. Kept as a supplementary control: it shows the
                        curve is not an artifact of the prefix construction, and
                        it gives the marginal variance of a single draw of size N.

    selection
        ``random``          the default. Uniform over the pool, no confidence
                            information used, so N=1 is a genuine
                            random-single-structure baseline. Wild type and
                            mutant draw independently.
        ``random_pinned``   uniform, but the mutant reuses the structure
                            identities drawn for wild type. This is the control
                            that isolates confidence selection from WT/mutant
                            pairing: ``top_confidence`` changes both at once, so
                            without this arm any gain it shows could be the
                            pairing rather than the confidence.
        ``top_confidence``  rank structures by mean pLDDT and keep the best N.
                            This is the "highest-confidence structure" control.
                            Selection is *pinned to wild type*: the ordering is
                            computed once on the WT ensemble and the same
                            structure identities are then used for the mutant.
                            Ranking WT and mutant independently would select on
                            the quantity being compared, pushing both sides
                            toward their maxima and shrinking the WT-mutant
                            difference by regression-to-the-max. Only defined for the
                            per-structure pLDDT CSV route.

    A default policy (``nested``/``random``) leaves every subseed key untouched,
    so it reproduces prior results byte-for-byte.
    """

    mode: str = "nested"
    selection: str = "random"

    def __post_init__(self) -> None:
        if self.mode not in SAMPLING_MODES:
            raise ValueError(
                f"sampling mode must be one of {SAMPLING_MODES}, got {self.mode!r}"
            )
        if self.selection not in SELECTION_RULES:
            raise ValueError(
                f"selection rule must be one of {SELECTION_RULES}, got {self.selection!r}"
            )

    @property
    def is_default(self) -> bool:
        return self.mode == "nested" and self.selection == "random"

    @property
    def uses_confidence(self) -> bool:
        return self.selection == "top_confidence"

    @property
    def pins_to_wild_type(self) -> bool:
        """Whether the mutant reuses the structures drawn for wild type."""
        return self.selection in ("random_pinned", "top_confidence")

    def tag(self) -> str:
        """Short identifier for cache keys and result columns ('' when default)."""
        return "" if self.is_default else f"{self.mode}-{self.selection}"

    @classmethod
    def coerce(cls, value: "SamplingPolicy | dict | str | None") -> "SamplingPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                mode=str(value.get("mode", "nested")),
                selection=str(value.get("selection", "random")),
            )
        text = str(value).strip()
        if not text or text in {"default", "nested-random"}:
            return cls()
        if "-" in text:
            mode, selection = text.split("-", 1)
            return cls(mode=mode, selection=selection)
        if text in SAMPLING_MODES:
            return cls(mode=text)
        if text in SELECTION_RULES:
            return cls(selection=text)
        raise ValueError(f"cannot interpret sampling policy {value!r}")


DEFAULT_SAMPLING_POLICY = SamplingPolicy()


def stable_subseed(seed: int, key: str) -> int:
    payload = f"{seed}|{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


# Source-path draw keys used to embed the absolute path of the checkout, so the same
# (seed, variant) drew different structures in every clone or worktree. The saved DRIFT
# caches (esfeat6, Figure 2, the simple fusion benchmark) were built in the subunit-split
# worktree, so that root is kept here as a fixed string: every checkout now draws what
# those caches hold. Only the text is used; the directory does not have to exist.
# See experiments/metrics_consolidation/CACHE_DRAW_PROVENANCE.md.
LEGACY_KEY_ROOT = "/Users/danraviv/Thesis/ensemble-driver-classifier.worktrees/subunit-split"
# The repo root, not this file's folder: draw keys are built relative to it, so moving the
# module (it lived at the root until 2026-11-03) must not change a single draw.
CHECKOUT_ROOT = str(Path(__file__).resolve().parents[1])


def source_draw_key(prefix: str, src: Any) -> str:
    """Draw key for one source file, independent of where the checkout lives.

    A path inside this checkout has its root replaced by LEGACY_KEY_ROOT; any other
    path (e.g. an archive on an external drive) is used as is.
    """
    text = str(src)
    if text == CHECKOUT_ROOT or text.startswith(CHECKOUT_ROOT + "/"):
        text = LEGACY_KEY_ROOT + text[len(CHECKOUT_ROOT):]
    return f"{prefix}::{text}"


def _draw_key(key: str, max_items: int | None, policy: SamplingPolicy) -> str:
    """Subseed key for one draw.

    Under ``nested`` the key does not mention N, so every size reuses the same
    permutation and small samples are prefixes of large ones. Under
    ``independent`` N enters the key, giving each size its own permutation.
    """
    if policy.mode == "nested":
        return key
    return f"{key}||N={'all' if max_items is None else int(max_items)}"


def rank_items_deterministically(items: list[str], *, seed: int, key: str) -> list[str]:
    """Reproducible *uniform random* order — not an ordering by confidence.

    Items are sorted first only so that the permutation is independent of
    filesystem enumeration order; the permutation itself is uniform, so the
    length-N prefix is a uniform random subset of size N.
    """
    items = sorted(str(x) for x in items)
    if len(items) <= 1:
        return items
    rng = np.random.default_rng(stable_subseed(seed, key))
    perm = rng.permutation(len(items))
    return [items[i] for i in perm.tolist()]


def rank_items_by_score(items: list[str], scores: dict[str, float]) -> list[str]:
    """Descending by score, ties broken by name so the order is deterministic.

    Items with no finite score sort last (in name order) rather than being
    dropped, so a scored ranking never silently shrinks the available pool.
    """
    items = [str(x) for x in items]

    def sort_key(name: str) -> tuple[int, float, str]:
        value = scores.get(name)
        if value is None or not np.isfinite(value):
            return (1, 0.0, name)
        return (0, -float(value), name)

    return sorted(items, key=sort_key)


def sample_items_nested(
    items: list[str],
    max_items: int | None,
    *,
    seed: int,
    key: str,
    policy: SamplingPolicy | None = None,
) -> list[str]:
    """Choose up to ``max_items`` items under ``policy``.

    ``policy.selection`` is not honoured here: this helper orders opaque item
    names (file paths) with no confidence attached. Confidence-ranked selection
    happens in :func:`select_structure_columns`, which sees the pLDDT values.
    Callers that cannot support it reject ``top_confidence`` up front.
    """
    policy = SamplingPolicy.coerce(policy)
    ranked = rank_items_deterministically(
        items, seed=seed, key=_draw_key(key, max_items, policy)
    )
    if max_items is None or len(ranked) <= max_items:
        return ranked
    return ranked[:max_items]


def select_structure_columns(
    columns: list[str],
    max_items: int | None,
    *,
    seed: int,
    key: str,
    policy: SamplingPolicy | None = None,
    scores: dict[str, float] | None = None,
    preferred_order: list[str] | None = None,
) -> list[str]:
    """Pick which per-structure columns of one pLDDT matrix to average.

    ``preferred_order`` carries the wild-type ranking into a mutant read so that
    ``top_confidence`` selects the same structure identities on both sides of
    the WT-mutant comparison. Columns named in it are taken first (in its order); anything
    still needed is topped up from this matrix's own ranking, so a mutant with
    extra structures, or with column names that do not match WT at all, still
    yields a full sample instead of silently returning fewer than N.
    """
    policy = SamplingPolicy.coerce(policy)
    columns = [str(c) for c in columns]

    if policy.selection == "random":
        return sample_items_nested(
            columns, max_items, seed=seed, key=key, policy=policy
        )

    if policy.uses_confidence:
        own_ranked = rank_items_by_score(columns, scores or {})
    else:
        # random_pinned: the ordering is the same uniform permutation as
        # `random`; only the WT/mutant coupling differs.
        own_ranked = rank_items_deterministically(
            columns, seed=seed, key=_draw_key(key, max_items, policy)
        )
    if preferred_order:
        # Order-preserving dedupe. `preferred_order` can legitimately repeat a
        # name: a gene whose wild type spans several CSVs (POLE has three, two of
        # them the same seed block) contributes one entry per file, and those
        # files reuse `seed-X_sample-Y` labels. Within one mutant matrix a name
        # denotes a single column, so a repeat must select it once -- otherwise
        # the mutant mean silently double-weights those structures and reports
        # more structures than it averaged.
        available = set(columns)
        ordered, seen = [], set()
        for name in preferred_order:
            if name in available and name not in seen:
                ordered.append(name)
                seen.add(name)
        ordered.extend(c for c in own_ranked if c not in seen)
    else:
        ordered = own_ranked

    if max_items is None or len(ordered) <= max_items:
        return ordered
    return ordered[:max_items]


def column_confidence_scores(value_df: pd.DataFrame) -> dict[str, float]:
    """Mean pLDDT per structure — the quantity AlphaFold ranks monomers by."""
    means = value_df.mean(axis=0, skipna=True)
    return {str(name): float(value) for name, value in means.items()}


def extract_archive_to_temp(archive_path: str | Path, tmpdir: str) -> list[str]:
    with tarfile.open(str(archive_path), "r:*") as tf:
        tf.extractall(tmpdir)
    return sorted(glob.glob(str(Path(tmpdir) / "**" / "*.cif"), recursive=True))


def _make_parser() -> MMCIFParser:
    return MMCIFParser(QUIET=True)


def extract_plddt_from_cif(
    cif_path: str | Path, *, chain_id: str, prefer_atom: str = "CA"
) -> dict[tuple[int, str], float]:
    parser = _make_parser()
    structure = parser.get_structure("x", str(cif_path))
    model = next(structure.get_models())
    if chain_id not in model:
        raise KeyError(f"Chain '{chain_id}' not found in {cif_path}")
    chain = model[chain_id]
    out: dict[tuple[int, str], float] = {}
    for residue in chain:
        hetflag, resseq, icode = residue.id
        if hetflag != " ":
            continue
        key = (int(resseq), (str(icode) or "").strip())
        if prefer_atom and (prefer_atom in residue):
            out[key] = float(residue[prefer_atom].get_bfactor())
        else:
            atoms = list(residue.get_atoms())
            if atoms:
                out[key] = float(np.mean([a.get_bfactor() for a in atoms]))
    return out


def extract_plddt_from_cif_multi(
    cif_path: str | Path,
    *,
    chain_ids: list[str],
    prefer_atom: str = "CA",
) -> dict[tuple[str, int, str], float]:
    out: dict[tuple[str, int, str], float] = {}
    for cid in chain_ids:
        try:
            d = extract_plddt_from_cif(cif_path, chain_id=cid, prefer_atom=prefer_atom)
            out.update({(cid, resseq, icode): v for (resseq, icode), v in d.items()})
        except KeyError:
            continue
    if not out:
        raise KeyError(f"None of requested chains {chain_ids} found in {cif_path}")
    return out


def mut_vec_from_dict(mut_dict: dict, common_keys: list) -> np.ndarray:
    return np.fromiter(
        (mut_dict.get(k, np.nan) for k in common_keys),
        dtype=np.float32,
        count=len(common_keys),
    )


def discover_wt_csv_sources(read_root, wt_csv_glob="*WT*plddt_values.csv", *,
                            expected_sources=None):
    """Resolve the configured WT pool before sampling any source files."""
    sources = sorted(glob.glob(str(Path(read_root) / "**" / wt_csv_glob), recursive=True))
    if expected_sources is not None and len(sources) != int(expected_sources):
        raise ValueError(
            f"WT reference {wt_csv_glob!r} under {read_root}: expected "
            f"{expected_sources} CSV source(s), found {len(sources)}: {sources}"
        )
    return sources


def discover_mut_csv_sources(read_root, mut_csv_glob="*plddt_values.csv",
                             replacements=None):
    """Mutant-side CSVs under ``read_root``, with configured file replacements.

    ``replacements`` (config ``mut_csv_replacements``) maps the basename of a file
    found under ``read_root`` to the file that stands in for it, e.g. a 25-seed copy
    of a 32-seed POLE matrix kept outside the root so the recursive glob cannot see
    both. The replacement takes the original's place in the sorted list, so row order
    and everything keyed on the basename are unchanged. A key that matches no file or
    several files, a missing target, or a target with a different basename raises:
    a replacement can never silently fall back to the original.
    """
    sources = sorted(glob.glob(str(Path(read_root) / "**" / mut_csv_glob), recursive=True))
    if not replacements:
        return sources
    for name, target in replacements.items():
        hits = [i for i, p in enumerate(sources) if Path(p).name == name]
        if len(hits) != 1:
            raise ValueError(
                f"mut_csv_replacements: {name!r} matches {len(hits)} file(s) under {read_root}")
        target = Path(target)
        if target.name != name or not target.is_file():
            raise ValueError(f"mut_csv_replacements: bad target for {name!r}: {target}")
        if Path(target).resolve().is_relative_to(Path(read_root).resolve()):
            raise ValueError(f"mut_csv_replacements: target {target} lies under {read_root}")
        sources[hits[0]] = str(target)
    return sources


def validate_plddt_csv_frame(df, expected_residues, max_structures, *, context):
    """Opt-in construct and ensemble-size checks for a pinned CSV reference."""
    if expected_residues is None:
        return
    expected = int(expected_residues)
    index = pd.to_numeric(df.iloc[:, 0], errors="coerce") if df.shape[1] else []
    if len(df) != expected or not np.array_equal(index, np.arange(expected)):
        raise ValueError(f"{context}: expected residue rows 0..{expected - 1}")
    available = df.shape[1] - 1
    if max_structures is not None and available < int(max_structures):
        raise ValueError(
            f"{context}: requested {max_structures} predictions, only {available} available"
        )


def extract_plddt_from_csv_mean(
    csv_path: str | Path,
    max_structures_per_variant: int | None = None,
    *,
    seed: int = 0,
    key: str | None = None,
    policy: SamplingPolicy | None = None,
    preferred_order: list[str] | None = None,
    selection_out: list[str] | None = None,
    expected_residues: int | None = None,
) -> tuple[pd.Series, int]:
    """Mean per-residue pLDDT over the selected structures of one CSV.

    ``selection_out``, when given, is extended with the structure columns that
    were actually averaged, in the order they were chosen. The wild-type read
    uses it to export its confidence ranking to the matching mutant reads.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    validate_plddt_csv_frame(df, expected_residues, max_structures_per_variant,
                            context=csv_path)
    if df.empty:
        return pd.Series(dtype=np.float32), 0

    if df.shape[1] > 1:
        idx_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        _int_vals = idx_raw.fillna(-1).astype(int).to_numpy()
        _is_residue_index = idx_raw.notna().all() and np.array_equal(
            _int_vals, np.arange(len(df))
        )
        if _is_residue_index:
            idx = idx_raw.fillna(-1).astype(int).astype(str)
            value_df = df.iloc[:, 1:]
        elif idx_raw.notna().any():
            # First column has some numeric values but is NOT a residue index
            # (e.g. a structure column named after the reference model).
            # Treat all columns as structure data and use the implicit row index.
            idx = pd.RangeIndex(len(df)).astype(str)
            value_df = df
        else:
            idx = df.iloc[:, 0].astype(str)
            value_df = df.iloc[:, 1:]
    else:
        idx = pd.RangeIndex(len(df)).astype(str)
        value_df = df

    value_df = value_df.apply(pd.to_numeric, errors="coerce")
    policy = SamplingPolicy.coerce(policy)
    if value_df.shape[1] > 0 and (
        max_structures_per_variant is not None or policy.selection != "random"
    ):
        sampled_cols = select_structure_columns(
            list(value_df.columns.astype(str)),
            None if max_structures_per_variant is None else int(max_structures_per_variant),
            seed=seed,
            key=key or f"csv::{Path(csv_path).name}",
            policy=policy,
            scores=column_confidence_scores(value_df) if policy.uses_confidence else None,
            preferred_order=preferred_order,
        )
        value_df = value_df.loc[:, sampled_cols]
    if selection_out is not None:
        selection_out.extend(str(c) for c in value_df.columns)

    n_structures = int(value_df.shape[1]) if value_df.shape[1] > 0 else 0
    series = value_df.mean(axis=1, skipna=True)
    series.index = idx
    series = series.replace([np.inf, -np.inf], np.nan).dropna().astype(np.float32)
    return series, n_structures


def _restrict_series_to_row_range(
    series: pd.Series,
    row_range: Any,
    *,
    context: str = "",
) -> pd.Series:
    """Keep only the rows of a per-residue series inside a half-open row range.

    The pLDDT index is the matrix row number as a string, so a subunit of a
    modelled complex is a contiguous integer slice of it (MSH2 = [0, 934),
    MSH6 = [934, 2294) for MutSalpha). Raises rather than silently returning
    everything when the range is malformed, because a dropped restriction would
    quietly reinstate the whole-complex descriptor.
    """
    try:
        start, stop = (int(row_range[0]), int(row_range[1]))
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise ValueError(
            f"plddt_row_range must be a two-element [start, stop); got {row_range!r}"
        ) from exc
    if stop <= start:
        raise ValueError(f"plddt_row_range {row_range!r} is empty ({context})")

    numeric = pd.to_numeric(pd.Series(series.index, index=series.index), errors="coerce")
    if numeric.isna().any():
        raise ValueError(
            f"plddt_row_range needs an integer residue-row index; "
            f"{context} has non-numeric labels (e.g. {series.index[numeric.isna()][:3].tolist()})"
        )
    return series[(numeric >= start) & (numeric < stop)]


def compute_wt_ref_from_csv_sources(
    csv_sources: list[str],
    max_structures_per_variant: int | None = None,
    *,
    seed: int = 0,
    key_prefix: str = "wt_csv",
    policy: SamplingPolicy | None = None,
    selection_out: list[str] | None = None,
    expected_residues: int | None = None,
) -> tuple[pd.Series, int]:
    policy = SamplingPolicy.coerce(policy)
    wt_parts = []
    wt_structures = 0
    for src in csv_sources:
        s, n = extract_plddt_from_csv_mean(
            src,
            max_structures_per_variant=max_structures_per_variant,
            seed=seed,
            key=f"{key_prefix}::{Path(src).name}",
            policy=policy,
            selection_out=selection_out,
            expected_residues=expected_residues,
        )
        if s.empty:
            continue
        wt_parts.append(s)
        wt_structures += max(int(n), 1)

    if not wt_parts:
        return pd.Series(dtype=np.float32), 0

    common_idx = set(wt_parts[0].index)
    for s in wt_parts[1:]:
        common_idx &= set(s.index)
    common_idx = sorted(common_idx)
    if not common_idx:
        return pd.Series(dtype=np.float32), wt_structures

    wt_mat = np.vstack([s.loc[common_idx].to_numpy(dtype=np.float32) for s in wt_parts])
    wt_ref = pd.Series(wt_mat.mean(axis=0).astype(np.float32), index=common_idx)
    return wt_ref, wt_structures


def _example_source_names(paths: list[str], max_examples: int = 3) -> str:
    if not paths:
        return ""
    return ", ".join(Path(p).name for p in paths[:max_examples])


def scan_plddt_sources(runtime_configs: dict, *, max_examples: int = 3) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gene_key, cfg in runtime_configs.items():
        mut = str(cfg.get("mut", gene_key))
        gene = canonical_gene(gene_key) or canonical_gene(mut) or str(gene_key)
        read_root = Path(cfg.get("read_root", ""))
        wt_glob = str(cfg.get("wt_glob", f"{mut}_*WT*/**/*.cif"))
        mut_glob = str(cfg.get("mut_glob", "**/*.cif"))
        wt_csv_glob = str(cfg.get("wt_csv_glob", "*WT*plddt_values.csv"))
        mut_csv_glob = str(cfg.get("mut_csv_glob", "*plddt_values.csv"))
        source_mode = str(cfg.get("source_mode", "auto"))

        if not read_root.exists():
            rows.append(
                {
                    "gene": gene,
                    "mut": mut,
                    "read_root": str(read_root),
                    "requested_source_mode": source_mode,
                    "n_wt_sources_found": 0,
                    "n_mut_sources_found": 0,
                    "n_wt_cif_fallback_candidates": 0,
                    "n_mut_cif_fallback_candidates": 0,
                    "wt_examples": "",
                    "mut_examples": "",
                    "source_mode_detected": "missing_root",
                }
            )
            continue

        wt_csv_sources = discover_wt_csv_sources(
            read_root, wt_csv_glob, expected_sources=cfg.get("wt_csv_expected_sources"),
        )
        mut_csv_sources = discover_mut_csv_sources(
            read_root, mut_csv_glob, cfg.get("mut_csv_replacements")
        )
        mut_csv_sources = [
            p for p in mut_csv_sources if "WT" not in Path(p).name.upper()
        ]
        wt_cif_sources = sorted(glob.glob(str(read_root / wt_glob), recursive=True))
        mut_cif_sources = sorted(glob.glob(str(read_root / mut_glob), recursive=True))

        detected = "csv" if (wt_csv_sources and mut_csv_sources) else "cif_or_tar"
        rows.append(
            {
                "gene": gene,
                "mut": mut,
                "read_root": str(read_root),
                "requested_source_mode": source_mode,
                "n_wt_sources_found": int(len(wt_csv_sources)),
                "n_mut_sources_found": int(len(mut_csv_sources)),
                "n_wt_cif_fallback_candidates": int(len(wt_cif_sources)),
                "n_mut_cif_fallback_candidates": int(len(mut_cif_sources)),
                "wt_examples": _example_source_names(
                    wt_csv_sources, max_examples=max_examples
                ),
                "mut_examples": _example_source_names(
                    mut_csv_sources, max_examples=max_examples
                ),
                "source_mode_detected": detected,
            }
        )
    return pd.DataFrame(rows).sort_values("gene").reset_index(drop=True)


def build_variant_table(
    runtime_configs: dict,
    *,
    max_structures_per_variant: int | None,
    random_seed: int,
    policy: SamplingPolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per variant the draw can score (some residue with valid mutant and WT
    pLDDT), and a per-dataset audit of what was found and used."""
    if max_structures_per_variant is not None and int(max_structures_per_variant) <= 0:
        raise ValueError(
            "max_structures_per_variant must be None or a positive integer."
        )
    policy = SamplingPolicy.coerce(policy)

    variant_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for gene_key, cfg in runtime_configs.items():
        mut = str(cfg.get("mut", gene_key))
        gene = canonical_gene(gene_key) or canonical_gene(mut) or str(gene_key)
        requested_source_mode = str(cfg.get("source_mode", "auto"))
        read_root = Path(cfg.get("read_root", ""))
        chain_ids = list(cfg.get("chain_ids", ["A"]))
        dirs_tars = bool(cfg.get("dirs_tars", False))
        subunit_tokens = [str(t).upper() for t in cfg.get("subunit_tokens", [])]
        mut_csv_glob = str(cfg.get("mut_csv_glob", "*plddt_values.csv"))
        # Restrict the descriptor to a contiguous block of residue rows, for a
        # dataset that is one subunit of a modelled complex (MSH2 / MSH6). Half
        # open [start, stop) over the pLDDT matrix row index. Absent for every
        # single-subunit dataset, which therefore keeps its whole matrix.
        row_range = cfg.get("plddt_row_range")
        wt_glob = str(cfg.get("wt_glob", f"{mut}_*WT*/**/*.cif"))
        mut_glob = str(cfg.get("mut_glob", "**/*.cif"))

        if not read_root.exists():
            if cfg.get("wt_csv_expected_sources") is not None:
                raise ValueError(f"{gene}: pinned WT root does not exist: {read_root}")
            audit_rows.append(
                {
                    "gene": gene,
                    "mut": mut,
                    "read_root": str(read_root),
                    "used": 0,
                    "reason": "read_root_not_found",
                    "rows_kept": 0,
                    "requested_source_mode": requested_source_mode,
                }
            )
            continue

        if cfg.get("wt_csv_expected_sources") is not None:
            # An explicitly pinned CSV reference must never fall back to another
            # WT construct via a broad CIF/tar pattern.
            wt_sources = discover_wt_csv_sources(
                read_root, cfg.get("wt_csv_glob", "*WT*plddt_values.csv"),
                expected_sources=cfg["wt_csv_expected_sources"],
            )
            wt_mode = "csv"
        elif dirs_tars:
            wt_tar_patterns = [
                str(read_root / f"{mut}_*WT*/**/*.tar.gz"),
                str(read_root / f"**/{mut}_*WT*.tar.gz"),
                str(read_root / "**/*WT*.tar.gz"),
            ]
            wt_sources = sorted(
                {p for pat in wt_tar_patterns for p in glob.glob(pat, recursive=True)}
            )
            wt_mode = "tar"
            if not wt_sources:
                wt_sources = sorted(glob.glob(str(read_root / wt_glob), recursive=True))
                wt_mode = "cif"
        else:
            wt_sources = sorted(glob.glob(str(read_root / wt_glob), recursive=True))
            wt_mode = "cif"

        if wt_mode == "cif":
            wt_sources = sample_items_nested(
                wt_sources,
                max_structures_per_variant,
                seed=random_seed,
                key=f"wt::{mut}::cif_sources",
                policy=policy,
            )

        if not wt_sources:
            csv_wt_sources = discover_wt_csv_sources(
                read_root, cfg.get("wt_csv_glob", "*WT*plddt_values.csv"),
                expected_sources=cfg.get("wt_csv_expected_sources"),
            )
            if csv_wt_sources:
                wt_sources = sample_items_nested(
                    csv_wt_sources,
                    max_structures_per_variant,
                    seed=random_seed,
                    key=f"wt::{mut}::csv_sources",
                    policy=policy,
                )
                wt_mode = "csv"

        if policy.uses_confidence and wt_mode != "csv":
            # Confidence ranking needs per-structure pLDDT values. The CIF/tar
            # routes hand back file paths only, and silently falling back to a
            # random draw would mislabel a random baseline as a top-confidence
            # one, so refuse instead.
            raise NotImplementedError(
                f"{gene}: selection='top_confidence' requires the per-structure "
                f"pLDDT CSV route, but wild-type resolved to '{wt_mode}' sources."
            )

        if not wt_sources:
            audit_rows.append(
                {
                    "gene": gene,
                    "mut": mut,
                    "read_root": str(read_root),
                    "used": 0,
                    "reason": "no_wt_sources",
                    "rows_kept": 0,
                    "requested_source_mode": requested_source_mode,
                }
            )
            continue

        wt_ref_series = None
        wt_selected_columns: list[str] = []
        if wt_mode == "csv":
            wt_ref_series, wt_count = compute_wt_ref_from_csv_sources(
                wt_sources,
                max_structures_per_variant=max_structures_per_variant,
                seed=random_seed,
                key_prefix=f"wt::{mut}",
                policy=policy,
                selection_out=wt_selected_columns,
                expected_residues=cfg.get("plddt_expected_residues"),
            )
            if wt_ref_series.empty:
                audit_rows.append(
                    {
                        "gene": gene,
                        "mut": mut,
                        "read_root": str(read_root),
                        "used": 0,
                        "reason": "no_common_wt_keys",
                        "rows_kept": 0,
                    }
                )
                continue
            # One filter here restricts the whole variant: every mutant row is
            # built by intersecting its own index against this reference, so
            # narrowing the reference narrows its residues without touching the
            # row builder.
            if row_range is not None:
                wt_ref_series = _restrict_series_to_row_range(
                    wt_ref_series, row_range, context=f"{gene} WT reference"
                )
                if wt_ref_series.empty:
                    audit_rows.append(
                        {
                            "gene": gene,
                            "mut": mut,
                            "read_root": str(read_root),
                            "used": 0,
                            "reason": f"no_wt_keys_in_row_range_{row_range}",
                            "rows_kept": 0,
                        }
                    )
                    continue
            common_keys = list(wt_ref_series.index)
            wt_ref = wt_ref_series.to_numpy(dtype=np.float32)
        else:
            common_keys = None
            for src in wt_sources:
                if wt_mode == "tar":
                    with tempfile.TemporaryDirectory() as tmpdir:
                        cif_list = extract_archive_to_temp(src, tmpdir)
                        cif_list = sample_items_nested(
                            cif_list,
                            max_structures_per_variant,
                            seed=random_seed,
                            key=source_draw_key("wt", src),
                            policy=policy,
                        )
                        for p in cif_list:
                            d = extract_plddt_from_cif_multi(p, chain_ids=chain_ids)
                            keys = set(d.keys())
                            common_keys = (
                                keys if common_keys is None else (common_keys & keys)
                            )
                else:
                    d = extract_plddt_from_cif_multi(src, chain_ids=chain_ids)
                    keys = set(d.keys())
                    common_keys = keys if common_keys is None else (common_keys & keys)

            common_keys = sorted(common_keys) if common_keys else []
            if not common_keys:
                audit_rows.append(
                    {
                        "gene": gene,
                        "mut": mut,
                        "read_root": str(read_root),
                        "used": 0,
                        "reason": "no_common_wt_keys",
                        "rows_kept": 0,
                    }
                )
                continue

            wt_sum = np.zeros(len(common_keys), dtype=np.float64)
            wt_count = 0

            for src in wt_sources:
                if wt_mode == "tar":
                    with tempfile.TemporaryDirectory() as tmpdir:
                        cif_list = extract_archive_to_temp(src, tmpdir)
                        cif_list = sample_items_nested(
                            cif_list,
                            max_structures_per_variant,
                            seed=random_seed,
                            key=source_draw_key("wt", src),
                            policy=policy,
                        )
                        for p in cif_list:
                            d = extract_plddt_from_cif_multi(p, chain_ids=chain_ids)
                            v = np.fromiter(
                                (d[k] for k in common_keys),
                                dtype=np.float64,
                                count=len(common_keys),
                            )
                            wt_sum += v
                            wt_count += 1
                else:
                    d = extract_plddt_from_cif_multi(src, chain_ids=chain_ids)
                    v = np.fromiter(
                        (d[k] for k in common_keys),
                        dtype=np.float64,
                        count=len(common_keys),
                    )
                    wt_sum += v
                    wt_count += 1

            if wt_count == 0:
                audit_rows.append(
                    {
                        "gene": gene,
                        "mut": mut,
                        "read_root": str(read_root),
                        "used": 0,
                        "reason": "no_wt_cifs_after_sampling",
                        "rows_kept": 0,
                    }
                )
                continue

            wt_ref = (wt_sum / wt_count).astype(np.float32)

        if wt_mode == "csv":
            # Honour the configured glob. This was hardcoded to
            # "*plddt_values.csv" while scan_plddt_sources already read
            # mut_csv_glob, so the two disagreed; every pre-existing dataset
            # configures exactly that default, which makes reading the config
            # a no-op for them and lets a subunit dataset (MSH2 / MSH6) select
            # only its own files out of a shared complex folder.
            mutant_sources = discover_mut_csv_sources(
                read_root, mut_csv_glob, cfg.get("mut_csv_replacements")
            )
            mutant_sources = [
                p for p in mutant_sources if "WT" not in str(Path(p).name).upper()
            ]
            # Guard against cross-dataset leakage (e.g., stray POLE CSVs copied
            # into another gene's PLDDTs folder) by requiring the filename to
            # start with this gene's own mutation prefix.
            _leaked = [
                p
                for p in mutant_sources
                if not re.match(rf"^{re.escape(mut)}_", Path(p).name, flags=re.IGNORECASE)
            ]
            if _leaked:
                print(
                    f"Ignoring {len(_leaked)} CSV(s) in {read_root} not matching "
                    f"'{mut}_' prefix (cross-dataset leakage): "
                    f"{_example_source_names(_leaked)}"
                )
            mutant_sources = [
                p
                for p in mutant_sources
                if re.match(rf"^{re.escape(mut)}_", Path(p).name, flags=re.IGNORECASE)
            ]
            source_mode = "csv"
        elif dirs_tars:
            mutant_sources = sorted(
                glob.glob(str(read_root / "**" / "*.tar.gz"), recursive=True)
            )
            mutant_sources = [
                p for p in mutant_sources if "WT" not in str(Path(p)).upper()
            ]
            source_mode = "tar"
        else:
            mutant_sources = [
                str(d)
                for d in read_root.iterdir()
                if d.is_dir() and ("WT" not in d.name.upper())
            ]
            source_mode = "dir"

        start_rows = len(variant_rows)
        error_count = 0
        actual_mutant_structures: list[int] = []

        for src in mutant_sources:
            src_path = Path(src)

            if source_mode == "tar":
                variant_name = (
                    src_path.parent.name
                    if src_path.parent != read_root
                    else src_path.name.replace(".tar.gz", "")
                )
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        mut_cifs = extract_archive_to_temp(src, tmpdir)
                        mut_cifs = sample_items_nested(
                            mut_cifs,
                            max_structures_per_variant,
                            seed=random_seed,
                            key=source_draw_key("mut", src),
                            policy=policy,
                        )
                        row = _build_variant_row_from_cifs(
                            variant_name=variant_name,
                            gene=gene,
                            mut=mut,
                            mut_cifs=mut_cifs,
                            common_keys=common_keys,
                            wt_ref=wt_ref,
                            chain_ids=chain_ids,
                        )
                except Exception:
                    error_count += 1
                    continue
            elif source_mode == "csv":
                variant_name = src_path.name.replace("_plddt_values.csv", "")
                try:
                    mut_series, used_structures = extract_plddt_from_csv_mean(
                        src,
                        max_structures_per_variant=max_structures_per_variant,
                        seed=random_seed,
                        key=source_draw_key("mut", src),
                        policy=policy,
                        preferred_order=wt_selected_columns or None,
                        expected_residues=cfg.get("plddt_expected_residues"),
                    )
                    row = _build_variant_row_from_csv(
                        variant_name=variant_name,
                        gene=gene,
                        mut=mut,
                        mut_series=mut_series,
                        wt_ref_series=wt_ref_series,
                        used_structures=used_structures,
                    )
                except Exception:
                    if cfg.get("plddt_expected_residues") is not None:
                        raise
                    error_count += 1
                    continue
            else:
                variant_name = src_path.name
                try:
                    mut_cifs = sorted(
                        glob.glob(str(src_path / mut_glob), recursive=True)
                    )
                    mut_cifs = sample_items_nested(
                        mut_cifs,
                        max_structures_per_variant,
                        seed=random_seed,
                        key=source_draw_key("mut", src),
                        policy=policy,
                    )
                    row = _build_variant_row_from_cifs(
                        variant_name=variant_name,
                        gene=gene,
                        mut=mut,
                        mut_cifs=mut_cifs,
                        common_keys=common_keys,
                        wt_ref=wt_ref,
                        chain_ids=chain_ids,
                    )
                except Exception:
                    error_count += 1
                    continue

            if row is None:
                continue
            if subunit_tokens:
                vname_upper = variant_name.upper()
                for tok in subunit_tokens:
                    if ("_" + tok + "_") in vname_upper:
                        row = dict(row)
                        row["mutation"] = f"{tok}_{row['mutation']}"
                        break
            actual_mutant_structures.append(int(row["Num_Structures"]))
            variant_rows.append(row)

        rows_kept = len(variant_rows) - start_rows
        audit_rows.append(
            {
                "gene": gene,
                "mut": mut,
                "read_root": str(read_root),
                "used": int(rows_kept > 0),
                "reason": "ok" if rows_kept > 0 else "no_mut_rows",
                "wt_sources": int(len(wt_sources)),
                "wt_structures": int(wt_count),
                "rows_kept": int(rows_kept),
                "errors": int(error_count),
                "max_structures_per_variant": max_structures_per_variant,
                "seed": int(random_seed),
                "mean_mut_structures_used": (
                    float(np.mean(actual_mutant_structures))
                    if actual_mutant_structures
                    else np.nan
                ),
                "median_mut_structures_used": (
                    float(np.median(actual_mutant_structures))
                    if actual_mutant_structures
                    else np.nan
                ),
                "n_variants_with_scores": int(len(actual_mutant_structures)),
            }
        )

    if not variant_rows:
        audit_df = pd.DataFrame(audit_rows)
        raise ValueError(
            "No usable variant rows produced from runtime structure pipeline.\n"
            + (audit_df.to_string(index=False) if not audit_df.empty else "")
        )

    out = pd.DataFrame(variant_rows)
    out = out.dropna(subset=["gene", "mutation"]).copy()
    out = (
        out.sort_values(["gene", "mutation", "source_file"])
        .drop_duplicates(["gene", "mutation"], keep="first")
        .reset_index(drop=True)
    )

    audit_df = pd.DataFrame(audit_rows)
    if not audit_df.empty:
        audit_df = audit_df.sort_values(["gene", "mut"]).reset_index(drop=True)

    return out, audit_df


def variant_keys(variant_rows_df: pd.DataFrame) -> pd.DataFrame:
    """The (gene, mutation) keys of build_variant_table's rows, normalised.

    Genes and mutations go through canonical_gene / normalize_mutation, so a
    subunit-tagged mutation ("MSH2_A256G") becomes its plain code; keys are unique
    and sorted.
    """
    if variant_rows_df is None or len(variant_rows_df) == 0:
        return pd.DataFrame(columns=["gene", "mutation"])
    keys = variant_rows_df[["gene", "mutation"]].copy()
    keys["gene"] = keys["gene"].map(canonical_gene)
    keys["mutation"] = keys["mutation"].map(normalize_mutation)
    return (
        keys.dropna()
        .drop_duplicates()
        .sort_values(["gene", "mutation"])
        .reset_index(drop=True)
    )


def _build_variant_row_from_cifs(
    *,
    variant_name: str,
    gene: str,
    mut: str,
    mut_cifs: list[str],
    common_keys: list,
    wt_ref: np.ndarray,
    chain_ids: list[str],
) -> dict[str, Any] | None:
    if not mut_cifs:
        return None

    mut_sum = np.zeros(len(common_keys), dtype=np.float64)
    mut_count = np.zeros(len(common_keys), dtype=np.int32)
    used_structures = 0

    for mut_cif in mut_cifs:
        mut_dict = extract_plddt_from_cif_multi(mut_cif, chain_ids=chain_ids)
        mut_vec = mut_vec_from_dict(mut_dict, common_keys)
        mask = ~np.isnan(mut_vec)
        if not np.any(mask):
            continue
        mut_sum[mask] += mut_vec[mask].astype(np.float64)
        mut_count[mask] += 1
        used_structures += 1

    if used_structures == 0 or not np.any(mut_count > 0):
        return None

    mut_mean_vec = np.full(len(common_keys), np.nan, dtype=np.float32)
    valid_mask = mut_count > 0
    mut_mean_vec[valid_mask] = (mut_sum[valid_mask] / mut_count[valid_mask]).astype(
        np.float32
    )
    valid = ~np.isnan(mut_mean_vec)
    if not np.any(valid) or np.isnan(wt_ref[valid]).any():
        return None

    mutation = normalize_mutation(variant_name)
    if mutation is None:
        return None

    return {
        # The runtime-config key is the authoritative dataset label. Filename
        # prefixes are validated before this point and may be aliases (P53) or
        # complex names (MutS_alpha), so re-inferring the gene from the filename
        # would make configuration-driven proteins fragile.
        "gene": gene,
        "mutation": mutation,
        "Variant": variant_name,
        "Num_Residues": int(valid.sum()),
        "Num_Structures": int(used_structures),
        "label": -1,
        "label_col": "none(-1)",
        "source_file": f"runtime::{mut}",
        "score_priority": 0,
        "n_replicates": int(used_structures),
        "is_labeled": 0,
    }


def _build_variant_row_from_csv(
    *,
    variant_name: str,
    gene: str,
    mut: str,
    mut_series: pd.Series,
    wt_ref_series: pd.Series | None,
    used_structures: int,
) -> dict[str, Any] | None:
    if mut_series.empty or wt_ref_series is None or wt_ref_series.empty:
        return None

    overlap = wt_ref_series.index.intersection(mut_series.index)
    if len(overlap) == 0:
        return None

    wt_vals = wt_ref_series.loc[overlap].to_numpy(dtype=np.float32)
    mut_vals = mut_series.loc[overlap].to_numpy(dtype=np.float32)
    valid = (~np.isnan(wt_vals)) & (~np.isnan(mut_vals))
    if not np.any(valid):
        return None


    mutation = normalize_mutation(variant_name)
    if mutation is None:
        return None

    eff_structures = int(max(used_structures, 1))
    return {
        "gene": gene,
        "mutation": mutation,
        "Variant": variant_name,
        "Num_Residues": int(valid.sum()),
        "Num_Structures": eff_structures,
        "label": -1,
        "label_col": "none(-1)",
        "source_file": f"runtime::{mut}",
        "score_priority": 0,
        "n_replicates": eff_structures,
        "is_labeled": 0,
    }
