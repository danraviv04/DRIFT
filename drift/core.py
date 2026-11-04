"""Data assembly for the DRIFT score caches: AlphaMissense scores, labels, and the
variant table from the structure draws (drift/structure_draws.py). run.py builds the
caches from build_labeled_dataset_for_config; the ensemble-shift descriptors come from
ensemble_shift.py."""

from pathlib import Path
import re

import pandas as pd

from drift import structure_draws
from .protein_config import build_structure_configs, load_protein_configs

# One shared ClinVar reader for the whole repo, so DRIFT and metric_analysis cannot
# drift on export-schema handling (the older Name/Gene(s)/Germline-classification
# layout vs the newer Variation/Genes/Classification one). metric_analysis is
# installed as a package, so this import works from any working directory.
try:
    from metric_analysis.pipeline.clinvar import read_clinvar_csv
except ImportError as _exc:  # pragma: no cover - environment/setup problem, not logic
    raise ImportError(
        "DRIFT needs the `metric_analysis` package for the shared ClinVar reader "
        "(metric_analysis.pipeline.clinvar). Install the repo in editable mode: "
        "`pip install -e .` from the project root."
    ) from _exc

build_variant_table = structure_draws.build_variant_table
SamplingPolicy = structure_draws.SamplingPolicy
SAMPLING_MODES = structure_draws.SAMPLING_MODES
SELECTION_RULES = structure_draws.SELECTION_RULES
canonical_gene = structure_draws.canonical_gene
normalize_mutation = structure_draws.normalize_mutation

# Tree-local: the repo this file sits in.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALPHAMISSENSE_SOURCES = [
    PROJECT_ROOT / "Ground_Truths" / "AlphaMissense",
]

PROTEIN_CONFIGS = load_protein_configs()
STRUCTURE_CONFIGS = build_structure_configs(PROTEIN_CONFIGS)

# TP53 and KRAS have no binary label (2026-10-13): their assays are continuous
# targets only (analyses/continuous/continuous_functional), never thresholded.

SEED = 42

DEFAULT_RUN_CONFIG = {
    "seed": SEED,
    "max_structures_per_variant": 90,
    # Structure-subsampling policy. "nested"/"random" is the primary design;
    # see drift.structure_draws.SamplingPolicy for what the alternatives control.
    "sampling_mode": "nested",
    "selection": "random",
}


def _collect_afm_paths(sources):
    if isinstance(sources, (str, Path)):
        sources = [sources]
    tsv_paths = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            tsv_paths.extend(sorted(pp for pp in p.glob("*.tsv") if pp.is_file()))
        elif p.is_file() and p.suffix.lower() in {".tsv", ".txt"}:
            tsv_paths.append(p)
    dedup = []
    seen = set()
    for p in tsv_paths:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen:
            seen.add(rp)
            dedup.append(p)
    if not dedup:
        raise ValueError(f"No AFM TSV files found in sources: {sources}")
    return dedup

def _normalize_afm_frame(afm):
    required_cols = ["Gene name", "pathogenicity score"]
    missing = [c for c in required_cols if c not in afm.columns]
    if missing:
        raise ValueError(f"Missing AlphaMissense columns: {missing}")

    afm = afm.rename(columns={"Gene name": "gene", "pathogenicity score": "afm_score"}).copy()
    afm["gene"] = afm["gene"].map(canonical_gene)

    if all(c in afm.columns for c in ["a.a.1", "position", "a.a.2"]):
        afm["mutation"] = (
            afm["a.a.1"].astype(str).str.strip().str.upper()
            + afm["position"].astype(str).str.strip()
            + afm["a.a.2"].astype(str).str.strip().str.upper()
        )
    elif "protein variant" in afm.columns:
        afm["mutation"] = afm["protein variant"].map(normalize_mutation)
    else:
        raise ValueError("Could not build mutation key from AlphaMissense table")

    afm["afm_score"] = pd.to_numeric(afm["afm_score"], errors="coerce")
    afm = afm.dropna(subset=["gene", "mutation", "afm_score"]).copy()

    keep_cols = ["gene", "mutation", "afm_score"]
    if "pathogenicity class" in afm.columns:
        afm = afm.rename(columns={"pathogenicity class": "afm_class"})
        keep_cols.append("afm_class")
    return afm[keep_cols].copy()

def build_afm_table(sources):
    afm_paths = _collect_afm_paths(sources)
    parts = []
    audit_rows = []

    for path in afm_paths:
        try:
            afm_raw = pd.read_csv(path, sep="\t", low_memory=False)
            afm_norm = _normalize_afm_frame(afm_raw)
            afm_norm["source_file"] = path.name
            parts.append(afm_norm)
            audit_rows.append({
                "source_file": path.name,
                "used": 1,
                "reason": "ok",
                "rows_read": int(len(afm_raw)),
                "rows_kept": int(len(afm_norm)),
            })
        except Exception as exc:
            audit_rows.append({
                "source_file": path.name,
                "used": 0,
                "reason": f"{type(exc).__name__}: {exc}",
            })

    if not parts:
        raise ValueError("No usable AFM files after normalization.")

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["gene", "mutation", "source_file"])
    out = out.drop_duplicates(subset=["gene", "mutation"], keep="first").reset_index(drop=True)

    audit_df = pd.DataFrame(audit_rows).sort_values("source_file").reset_index(drop=True)
    return out.drop(columns=["source_file"]), audit_df

AFM_5CLASS_ORDER = [
    "Pathogenic",
    "Likely Pathogenic",
    "Ambiguous",
    "Likely Benign",
    "Benign",
    "Unknown",
]
AFM_5CLASS_TO_ID = {name: idx for idx, name in enumerate(AFM_5CLASS_ORDER)}

def normalize_afm_class(text):
    if pd.isna(text):
        return "Unknown"
    s = re.sub(r"\s+", " ", str(text).strip().lower())
    if "ambig" in s or "uncertain" in s or "vus" in s:
        return "Ambiguous"
    if "likely pathogenic" in s:
        return "Likely Pathogenic"
    if "likely benign" in s:
        return "Likely Benign"
    if "pathogenic" in s:
        return "Pathogenic"
    if "benign" in s:
        return "Benign"
    return "Unknown"

def afm_class_to_binary_label(afm_group):
    if afm_group in {"Pathogenic", "Likely Pathogenic"}:
        return 1
    if afm_group in {"Benign", "Likely Benign"}:
        return 0
    return -1

def build_pole_gt(path, gene="POLE", gene_col="Hugo Symbol", mutation_col="HGVSp Short"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"POLE GT file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = [gene_col, mutation_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"POLE GT missing columns: {missing}")

    out = df.copy()
    out["gene"] = out[gene_col].map(canonical_gene)
    out = out[out["gene"] == canonical_gene(gene)].copy()
    out["mutation"] = out[mutation_col].map(normalize_mutation)
    out = out.dropna(subset=["mutation"]).copy()
    out["label"] = 1
    out = out.groupby(["gene", "mutation"], as_index=False).agg(label=("label", "max"))
    out["gt_source"] = f"{gene}_positive_list_present_driver"
    return out


def _clinvar_to_label(text):
    if pd.isna(text):
        return -1
    s = str(text).strip().lower()
    if "pathogenic" in s and "benign" not in s:
        return 1
    if "benign" in s and "pathogenic" not in s:
        return 0
    return -1


_AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
_CLINVAR_NAME_RE = re.compile(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)")

_CLINVAR_TRANSCRIPT_GENE_RE = re.compile(r"\(([A-Za-z0-9_.\-]+)\):c\.")


def _clinvar_row_gene(name):
    """Gene a ClinVar row's variant is actually annotated on, taken from the transcript
    cited in the Name field: 'NM_000551.4(VHL):c.194C>G (p.Ser65Trp)' -> 'VHL'.

    The `Gene(s)` column lists every gene OVERLAPPING the locus, which is not the same
    thing. Filtering on it with exact equality silently drops legitimate rows whose
    Gene(s) is a multi-gene string (VHL: only 94 of 253 rows are bare 'VHL'; the rest
    are 'LOC107303340|VHL' but still cite NM_000551.4(VHL)). Conversely, a naive
    token/split match over Gene(s) would WRONGLY admit rows annotated on the other
    gene -- ClinVar's 'FBXO11|MSH6' rows cite NM_001190274.2(FBXO11), so their p.
    changes (D910G, D889N, S831G, M820V) are FBXO11 residues, not MSH6 ones, and
    'KLLN|PTEN' rows cite KLLN. The cited transcript is the authoritative attribution
    and is the same field `_parse_mutation_from_clinvar_name` reads the p. change from,
    so the gene and the mutation always come from one consistent source.

    Returns the cited gene upper-cased, or None when Name carries no transcript (the
    caller then falls back to the legacy exact `Gene(s)` match)."""
    match = _CLINVAR_TRANSCRIPT_GENE_RE.search(str(name))
    return match.group(1).upper() if match else None


def _select_clinvar_rows_for_gene(df, gene):
    """Rows of a ClinVar export belonging to `gene`, attributed by cited transcript
    (see _clinvar_row_gene) and falling back to the legacy exact `Gene(s)` match for
    rows whose Name has no transcript."""
    cited = df["Name"].map(_clinvar_row_gene)
    exact = df["Gene(s)"].astype(str).str.strip().eq(gene)
    return df[cited.eq(gene.upper()) | (cited.isna() & exact)].copy()


def _parse_mutation_from_clinvar_name(name):
    """Parse canonical missense mutation from ClinVar Name field.
    E.g. 'NM_000179.3(MSH6):c.1A>G (p.Met1Val)' -> 'M1V'.
    Returns None for stop codons and unparseable entries."""
    m = _CLINVAR_NAME_RE.search(str(name))
    if not m:
        return None
    aa1 = _AA3_TO_1.get(m.group(1).capitalize())
    aa2 = _AA3_TO_1.get(m.group(3).capitalize())
    if not aa1 or not aa2:
        return None
    return f"{aa1}{m.group(2)}{aa2}"

def _audit_subunit_collisions_by_gt_source(combined, context=""):
    """Flag (gene, mutation) keys carrying >1 distinct gt_source, i.e. contributed by
    more than one member subunit of a grouped complex (e.g. MutSalpha = MSH2 + MSH6).

    DRIFT keys grouped complexes on (gene, mutation) only, but MSH2 and MSH6 number
    their residues independently, so a shared residue number collapses two genuinely
    different variants into one row -- and conflicting ClinVar labels then resolve to
    driver via max(). Currently LATENT (the modeled MSH2/MSH6 sets share no exact
    codes); we audit rather than refactor so a real collision triggers the eventual
    subunit-aware key change before results are affected. Returns the collision count."""
    if combined is None or combined.empty or "gt_source" not in combined.columns:
        return 0
    per_key = combined.groupby(["gene", "mutation"])["gt_source"].nunique()
    collided = per_key[per_key > 1]
    n = int(len(collided))
    if n:
        examples = ", ".join(str(m) for _, m in list(collided.index)[:5])
        print(
            f"[subunit-collision] {context}: {n} (gene,mutation) key(s) contributed by "
            f">1 member subunit -> collapsed by DRIFT's subunit-blind key (conflicting "
            f"labels resolve to max/driver). Examples: {examples}. A subunit-aware key "
            f"is required to keep these distinct."
        )
    return n


def _audit_subunit_collisions_pre_dedup(combined, key_cols=("gene", "mutation"), context=""):
    """Flag duplicate (gene, mutation) keys BEFORE a keep='first' dedup. For grouped
    AFM tables (member rows are disjoint by subunit), a duplicate key almost always
    means two subunits share a residue number and one score is silently dropped. Same
    latent-collision audit as _audit_subunit_collisions_by_gt_source, for the AFM side."""
    if combined is None or combined.empty:
        return 0
    dup_mask = combined.duplicated(list(key_cols), keep=False)
    if not dup_mask.any():
        return 0
    collided = combined.loc[dup_mask, list(key_cols)].drop_duplicates()
    n = int(len(collided))
    examples = ", ".join(str(m) for m in collided[key_cols[-1]].tolist()[:5])
    print(
        f"[subunit-collision] {context}: {n} duplicate (gene,mutation) key(s) across "
        f"member subunits -> one score dropped by keep='first'. Examples: {examples}. "
        f"A subunit-aware key is required to keep these distinct."
    )
    return n


def build_grouped_afm(gene, member_files):
    """Load member AlphaMissense tables and group them under one configured protein."""
    parts = []
    for member, path in member_files.items():
        path = Path(path)
        if not path.exists():
            print(f"{gene} AFM: {path.name} not found, skipping {member}")
            continue
        try:
            afm_raw = pd.read_csv(path, sep="\t", low_memory=False)
            afm_norm = _normalize_afm_frame(afm_raw)
            afm_norm = afm_norm[afm_norm["gene"] == canonical_gene(member)].copy()
            if afm_norm.empty:
                print(f"{gene} AFM: no {member} rows in {path.name}")
                continue
            afm_norm["gene"] = gene
            parts.append(afm_norm)
        except Exception as exc:
            print(f"{gene} AFM: failed to load {path.name}: {exc}")
    if not parts:
        return pd.DataFrame(columns=["gene", "mutation", "afm_score"])
    combined = pd.concat(parts, ignore_index=True)
    _audit_subunit_collisions_pre_dedup(combined, context=f"{gene} AFM")
    return combined.drop_duplicates(["gene", "mutation"], keep="first").reset_index(drop=True)


def build_clinvar_by_subunit_gt(gene, subunit_files):
    """Build grouped ClinVar labels from explicitly configured member-gene files."""
    parts = []
    for subunit, path in subunit_files.items():
        path = Path(path)
        if not path.exists():
            print(f"{gene} GT: {path.name} not found, skipping {subunit}")
            continue
        # Schema-tolerant read: ClinVar ships two column layouts (older Name/Gene(s)/
        # Germline classification, newer Variation/Genes/Classification with a "G: "
        # value prefix). Normalized here to the older names used below.
        df = read_clinvar_csv(path)
        required = ["Gene(s)", "Name", "Germline classification"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"{gene} GT: {path.name} missing columns {missing}, skipping {subunit}")
            continue
        df = _select_clinvar_rows_for_gene(df, subunit)
        df["label"] = df["Germline classification"].map(_clinvar_to_label)
        df = df[df["label"].isin([0, 1])].copy()
        df["mutation"] = df["Name"].map(_parse_mutation_from_clinvar_name)
        df = df.dropna(subset=["mutation"]).copy()
        df["gene"] = gene
        out = df.groupby(["gene", "mutation"], as_index=False).agg(label=("label", "max"))
        # A single-subunit dataset (VHL, and MSH2/MSH6 since the split) names its
        # own subunit, so collapse "MSH2_MSH2" to "MSH2" rather than emitting a
        # doubled provenance string. Only the audit reads gt_source, and it reads
        # nunique() per key, which is unchanged.
        stem = gene if str(subunit) == str(gene) else f"{gene}_{subunit}"
        out["gt_source"] = f"{stem}_ClinVar_germline"
        parts.append(out)
    if not parts:
        return pd.DataFrame(columns=["gene", "mutation", "label", "gt_source"])
    combined = pd.concat(parts, ignore_index=True)
    _audit_subunit_collisions_by_gt_source(combined, context=f"{gene} ClinVar GT")
    return combined


def build_binary_label_table(
    path, gene, mutation_col, label_col, gene_col=None
):
    """Load an explicit mutation-to-binary-label table for a configured protein."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{gene} label file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_csv(path, low_memory=False)
    required = [mutation_col, label_col] + ([gene_col] if gene_col else [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{gene} label table missing columns: {missing}")
    if gene_col:
        df = df[df[gene_col].map(canonical_gene).eq(canonical_gene(gene))].copy()
    out = pd.DataFrame({
        "gene": gene,
        "mutation": df[mutation_col].map(normalize_mutation),
        "label": pd.to_numeric(df[label_col], errors="coerce"),
    })
    out = out.dropna(subset=["mutation", "label"]).copy()
    invalid = sorted(set(out["label"]) - {0, 1})
    if invalid:
        raise ValueError(f"{gene} label table contains non-binary labels: {invalid[:5]}")
    out["label"] = out["label"].astype(int)
    out = out.groupby(["gene", "mutation"], as_index=False).agg(label=("label", "max"))
    out["gt_source"] = f"{gene}_binary_table_{path.name}"
    return out


def build_configured_external_gt(protein_configs):
    """Load ground truth according to each protein's explicit label strategy."""
    parts = []
    for gene, protein_cfg in protein_configs.items():
        label_cfg = protein_cfg["label"]
        mode = label_cfg["mode"]
        if mode == "continuous_assay":
            # No binary label exists for these genes; they only ever reach the
            # binary pipeline by mistake (load_protein_configs refuses them).
            continue
        elif mode == "positive_list":
            parts.append(build_pole_gt(
                label_cfg["path"],
                gene=gene,
                gene_col=label_cfg.get("gene_col", "Hugo Symbol"),
                mutation_col=label_cfg.get("mutation_col", "HGVSp Short"),
            ))
        elif mode == "clinvar_by_subunit":
            parts.append(build_clinvar_by_subunit_gt(
                gene, label_cfg["subunit_files"]
            ))
        elif mode == "binary_csv":
            parts.append(build_binary_label_table(
                label_cfg["path"],
                gene=gene,
                mutation_col=label_cfg["mutation_col"],
                label_col=label_cfg["label_col"],
                gene_col=label_cfg.get("gene_col"),
            ))
        else:  # Loader validation should make this unreachable.
            raise ValueError(f"Unsupported label mode for {gene}: {mode!r}")
    if not parts:
        return pd.DataFrame(columns=["gene", "mutation", "label", "gt_source"])
    # Per-subunit collision auditing already happens inside the grouped loaders
    # (build_clinvar_by_subunit_gt / build_mutsalpha_gt); no need to repeat it here.
    return pd.concat(parts, ignore_index=True)


def apply_configured_label_policies(merged_df, protein_configs):
    """Apply modes whose negatives are defined by absence (unlisted variants)."""
    out = merged_df.copy()
    for gene, protein_cfg in protein_configs.items():
        label_cfg = protein_cfg["label"]
        mode = label_cfg["mode"]
        gene_mask = out["gene"].eq(gene)
        if mode == "positive_list" and label_cfg.get("unlisted_are_negative", False):
            positive = gene_mask & out["label"].eq(1)
            negative = gene_mask & ~positive
            out.loc[positive, "label"] = 1
            out.loc[positive, "gt_source"] = f"{gene}_positive_list_present_driver"
            out.loc[negative, "label"] = 0
            out.loc[negative, "gt_source"] = f"{gene}_not_in_positive_list_passenger"
    return out

_AFM_TABLE = None


def load_afm_table():
    """(afm_df, afm_audit) from ALPHAMISSENSE_SOURCES, read on first use.

    Reading at import made every module that imports core fail without the
    AlphaMissense files; the table is the same, only read later.
    """
    global _AFM_TABLE
    if _AFM_TABLE is None:
        _AFM_TABLE = build_afm_table(ALPHAMISSENSE_SOURCES)
        print(f"AFM normalized rows: {len(_AFM_TABLE[0]):,}")
    return _AFM_TABLE


def __getattr__(name):  # core.afm_df / core.afm_audit keep working, loaded on first access
    if name == "afm_df":
        return load_afm_table()[0]
    if name == "afm_audit":
        return load_afm_table()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ============ labelled-dataset assembly ============


def _afm_for_merge(afm_df):
    """AlphaMissense rows for every gene, with the grouped subunit tables added."""
    afm = afm_df.copy()
    afm["afm_label_5class"] = afm["afm_class"].map(normalize_afm_class)
    afm["afm_label_5class_id"] = (
        afm["afm_label_5class"].map(AFM_5CLASS_TO_ID).fillna(AFM_5CLASS_TO_ID["Unknown"]).astype(int)
    )
    afm["afm_label_binary"] = afm["afm_label_5class"].map(afm_class_to_binary_label).astype(int)

    for grouped_gene, protein_cfg in PROTEIN_CONFIGS.items():
        member_files = protein_cfg.get("afm", {}).get("member_files")
        if not member_files:
            continue
        grouped_afm = build_grouped_afm(grouped_gene, member_files)
        if grouped_afm.empty:
            continue
        if "afm_class" in grouped_afm.columns:
            grouped_afm["afm_label_5class"] = grouped_afm["afm_class"].map(normalize_afm_class)
            grouped_afm["afm_label_5class_id"] = (
                grouped_afm["afm_label_5class"]
                .map(AFM_5CLASS_TO_ID)
                .fillna(AFM_5CLASS_TO_ID["Unknown"])
                .astype(int)
            )
            grouped_afm["afm_label_binary"] = (
                grouped_afm["afm_label_5class"].map(afm_class_to_binary_label).astype(int)
            )
        afm = (
            pd.concat([afm, grouped_afm], ignore_index=True)
            .drop_duplicates(["gene", "mutation"], keep="first")
            .reset_index(drop=True)
        )
    return afm


def build_labeled_dataset_for_config(max_structures_per_variant, seed=SEED, sampling_policy=None):
    """The labelled variants with AlphaMissense scores, one row per (gene, mutation).

    The variants are those the structure draws can score at this ensemble size and
    seed (build_variant_table), joined with AlphaMissense and the configured labels;
    run.py then restricts them to the strict canonical manifest.
    """
    afm_df = load_afm_table()[0]
    if "afm_class" not in afm_df.columns:
        raise RuntimeError("`afm_class` is missing from AFM table.")

    rows, _ = build_variant_table(
        STRUCTURE_CONFIGS,
        max_structures_per_variant=max_structures_per_variant,
        random_seed=int(seed),
        policy=SamplingPolicy.coerce(sampling_policy),
    )
    merged = structure_draws.variant_keys(rows).merge(
        _afm_for_merge(afm_df), on=["gene", "mutation"], how="inner")

    gt_lookup = (
        build_configured_external_gt(PROTEIN_CONFIGS)
        .sort_values(["gene", "mutation", "label"], ascending=[True, True, False])
        .drop_duplicates(["gene", "mutation"], keep="first")
    )
    merged = merged.merge(
        gt_lookup[["gene", "mutation", "label", "gt_source"]],
        on=["gene", "mutation"],
        how="left",
    )
    merged["gt_source"] = merged["gt_source"].fillna("none")
    merged["label"] = merged["label"].fillna(-1).astype(int)
    merged = apply_configured_label_policies(merged, PROTEIN_CONFIGS)

    labeled_df = (
        merged[merged["label"].isin([0, 1])]
        .sort_values(["gene", "mutation"])
        .drop_duplicates(["gene", "mutation"], keep="first")
        .reset_index(drop=True)
        .copy()
    )
    if labeled_df.empty:
        raise RuntimeError("No labeled rows available after the AlphaMissense + label merge.")
    if labeled_df["label"].nunique() < 2:
        raise RuntimeError("Labeled dataset must contain both classes.")
    return labeled_df


def _resolve_run_config(overrides=None):
    cfg = dict(DEFAULT_RUN_CONFIG)
    if overrides:
        cfg.update(dict(overrides))

    cfg["seed"] = int(cfg["seed"])
    v = cfg.get("max_structures_per_variant", None)
    cfg["max_structures_per_variant"] = None if v is None else int(v)
    if cfg["max_structures_per_variant"] is not None and cfg["max_structures_per_variant"] <= 0:
        raise ValueError("max_structures_per_variant must be a positive integer or None")

    # Validate eagerly so a typo fails at config time rather than after a long
    # rebuild, and normalise to the canonical strings used in cache keys.
    policy = SamplingPolicy(
        mode=str(cfg.get("sampling_mode", "nested")),
        selection=str(cfg.get("selection", "random")),
    )
    cfg["sampling_mode"] = policy.mode
    cfg["selection"] = policy.selection
    return cfg


def sampling_policy_from_config(cfg):
    """The SamplingPolicy a resolved run config describes."""
    return SamplingPolicy(
        mode=str(cfg.get("sampling_mode", "nested")),
        selection=str(cfg.get("selection", "random")),
    )
