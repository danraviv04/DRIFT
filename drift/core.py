"""DRIFT AFM+D1 fusion — science originally extracted from the notebook
(cells 0,2,3,4), with W&B kept optional. No orchestration here; see run.py.

NOTE: this module was seeded by extraction from
`DRIFT/afm_and_d1_head_wandb_folds_ensemble_size.ipynb`, but it has since been
hand-maintained and has diverged from that notebook (e.g. the grouped-protein
loaders build_grouped_afm / build_clinvar_by_subunit_gt / build_binary_label_table
and the leak-free select-then-refit train_one_fold exist only here). Edit this
file directly; the notebook is a stale ancestor, not the source of truth."""

from pathlib import Path
import os
import sys
import re
import glob
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: no GUI backend needed for a script run
import matplotlib.pyplot as plt
import seaborn as sns

from .protein_config import build_d1_runtime_configs, load_protein_configs


def display(*args, **kwargs):  # no-op shim for notebook display() calls
    return None

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    brier_score_loss,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import wandb

    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

# THIS TREE FIRST: the repo root goes on sys.path from __file__, so `import
# metric_analysis` comes from this checkout and not from another tree or worktree.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import importlib
from drift import structure_draws as d1_runtime

d1_runtime = importlib.reload(d1_runtime)

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

build_d1_table = d1_runtime.build_d1_table
SamplingPolicy = d1_runtime.SamplingPolicy
SAMPLING_MODES = d1_runtime.SAMPLING_MODES
SELECTION_RULES = d1_runtime.SELECTION_RULES
aggregate_d1_rows_to_mutation_level = d1_runtime.aggregate_d1_rows_to_mutation_level
canonical_gene = d1_runtime.canonical_gene
normalize_mutation = d1_runtime.normalize_mutation
runtime_scan_plddt_sources = d1_runtime.scan_plddt_sources

SKIP_RERUN = True  # Set to True to skip re-running the D1 calculations and use cached results when available.

# Tree-local, for the same reason as _PROJECT_ROOT above.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALPHAMISSENSE_SOURCES = [
    PROJECT_ROOT / "Ground_Truths" / "AlphaMissense",
]

D1_RATIO_MODE = "ratio_of_means"  # "mean_of_ratios"

# Source policy for D1: prefer CSV ensemble pLDDT sources.
# The runtime can still fall back to CIF/tar paths when CSVs are absent.
D1_SOURCE_POLICY = "csv_preferred_with_runtime_fallback"

PROTEIN_CONFIGS = load_protein_configs()
D1_RUNTIME_CONFIGS = build_d1_runtime_configs(PROTEIN_CONFIGS)


def _example_source_names(paths, max_examples=3):
    if not paths:
        return ""
    return ", ".join(Path(p).name for p in paths[:max_examples])


def scan_plddt_sources(runtime_configs, max_examples=3):
    return runtime_scan_plddt_sources(runtime_configs, max_examples=max_examples)


# TP53 and KRAS have no binary label (2026-10-13): their assays are continuous
# targets only (analyses/continuous/continuous_functional), never thresholded. The
# former TP53 cut (Phenotypic_Selection_Model >= 3.25) is deleted.

FOXA1_HOTSPOT_RANGE = (168, 269)
MUTSALPHA_MSH2_GT_PATH = PROJECT_ROOT / "Ground_Truths" / "ClinVar" / "clinvar_result_msh2.csv"
MUTSALPHA_MSH6_GT_PATH = PROJECT_ROOT / "Ground_Truths" / "ClinVar" / "clinvar_result_msh6.csv"
MUTSALPHA_MSH2_AFM_PATH = PROJECT_ROOT / "Ground_Truths" / "AlphaMissense" / "afm_MSH2_predictions.tsv"
MUTSALPHA_MSH6_AFM_PATH = PROJECT_ROOT / "Ground_Truths" / "AlphaMissense" / "afm_MSH6_predictions.tsv"

SEED = 42
FOLDS = 5
EPOCHS = 250
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 5e-4
PATIENCE = 30

FUSION_MODEL_DEFAULTS = {
    "hidden": 32,
    "dropout": 0.2,
}

FUSION_MODEL_SWEEP_SPACE = {
    "d1_max_structures_per_variant": [1, 5, 10, 20, 40, 60, 90],
}

D1_MAX_STRUCTURES_PER_VARIANT = max(FUSION_MODEL_SWEEP_SPACE["d1_max_structures_per_variant"])

SWEEP_BASE_CONFIG = {
    "foxa1_hotspot_start": FOXA1_HOTSPOT_RANGE[0],
    "foxa1_hotspot_end": FOXA1_HOTSPOT_RANGE[1],
    "seed": SEED,
    "folds": FOLDS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "patience": PATIENCE,
    "hidden": FUSION_MODEL_DEFAULTS["hidden"],
    "dropout": FUSION_MODEL_DEFAULTS["dropout"],
    "d1_max_structures_per_variant": D1_MAX_STRUCTURES_PER_VARIANT,
    # Structure-subsampling policy. "nested"/"random" is the primary design;
    # see drift.structure_draws.SamplingPolicy for what the alternatives control.
    "d1_sampling_mode": "nested",
    "d1_selection": "random",
}

WANDB_PROJECT = "afm_d1_fusion"
WANDB_RUN_NAME = None
ENABLE_WANDB_FOLD_LOGGING = False
WANDB_LOG_FOLD_TABLES = True
WANDB_LOG_OOF_SUMMARY = True


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device():
    forced = os.environ.get("DRIFT_DEVICE", "").strip().lower()
    if forced in ("cpu", "mps", "cuda"):
        return torch.device(forced), f"DRIFT_DEVICE={forced}: device forced by env."
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA available: using GPU."
    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if mps_available:
        return torch.device("mps"), "MPS available: using Apple Silicon GPU via Metal."
    if mps_built and not mps_available:
        return (
            torch.device("cpu"),
            "MPS is built in this PyTorch, but currently unavailable; falling back to CPU.",
        )
    return (
        torch.device("cpu"),
        "MPS backend not available in this PyTorch build; using CPU.",
    )


def maybe_init_wandb_run(
    project=WANDB_PROJECT, name=WANDB_RUN_NAME, config=None, tags=None
):
    if not WANDB_AVAILABLE or wandb is None:
        raise ImportError("wandb is not available. Install wandb and re-run imports.")
    if wandb.run is None:
        return wandb.init(project=project, name=name, config=config, tags=tags)
    if config:
        for key, value in dict(config).items():
            if key not in wandb.config:
                wandb.config[key] = value
    return wandb.run


seed_everything(SEED)
DEVICE, device_msg = pick_device()
print(f"Using device: {DEVICE}")
print(device_msg)
print("Imported the structure-draw runtime from drift/structure_draws.py")
print(f"D1 source policy: {D1_SOURCE_POLICY}")
if not WANDB_AVAILABLE:
    print("wandb import failed; sweep helpers will remain unavailable.")
# ===================== data builders (nb cell 2) =====================


MUTATION_RE = re.compile(r"(?<![A-Z0-9])([A-Z])(\d+)([A-Z])(?![A-Z0-9])")  # token-bounded: PIK3CA is not K3C
KNOWN_GENES = {"TP53", "POLE", "FOXA1", "MutSalpha"}

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


def build_mutsalpha_gt(msh2_path, msh6_path, gene="MutSalpha"):
    """Build MutSalpha GT from MSH2 + MSH6 ClinVar files.
    Parses the canonical mutation from the ClinVar Name field
    (e.g. p.Ala256Gly -> A256G) to avoid multi-alias ambiguity
    in the Protein change column."""
    parts = []
    for path, subunit in [(msh2_path, "MSH2"), (msh6_path, "MSH6")]:
        path = Path(path)
        if not path.exists():
            print(f"MutSalpha GT: {path.name} not found, skipping {subunit}")
            continue
        # Schema-tolerant read: ClinVar ships two column layouts (older Name/Gene(s)/
        # Germline classification, newer Variation/Genes/Classification with a "G: "
        # value prefix). Normalized here to the older names used below.
        df = read_clinvar_csv(path)
        required = ["Gene(s)", "Name", "Germline classification"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"MutSalpha GT: {path.name} missing columns {missing}, skipping {subunit}")
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


def build_mutsalpha_afm(msh2_path, msh6_path, gene="MutSalpha"):
    """Load MSH2 + MSH6 AlphaMissense predictions and relabel as gene='MutSalpha'.
    Mutation keys are plain amino-acid changes (A45V, P522T) to match the D1 aggregated side."""
    parts = []
    for path, subunit in [(msh2_path, "MSH2"), (msh6_path, "MSH6")]:
        path = Path(path)
        if not path.exists():
            print(f"MutSalpha AFM: {path.name} not found, skipping {subunit}")
            continue
        try:
            afm_raw = pd.read_csv(path, sep="	", low_memory=False)
            afm_norm = _normalize_afm_frame(afm_raw)
            afm_norm = afm_norm[afm_norm["gene"] == subunit].copy()
            if afm_norm.empty:
                print(f"MutSalpha AFM: no {subunit} rows in {path.name} (found: {afm_raw.get('Gene name', pd.Series()).unique()[:3].tolist()})")
                continue
            afm_norm["gene"] = gene
            parts.append(afm_norm)
        except Exception as exc:
            print(f"MutSalpha AFM: failed to load {path.name}: {exc}")
    if not parts:
        return pd.DataFrame(columns=["gene", "mutation", "afm_score"])
    combined = pd.concat(parts, ignore_index=True)
    _audit_subunit_collisions_pre_dedup(combined, context=f"{gene} AFM")
    return combined.drop_duplicates(["gene", "mutation"], keep="first").reset_index(drop=True)


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


def build_configured_external_gt(protein_configs, run_config):
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
        elif mode == "hotspot_range":
            continue
        else:  # Loader validation should make this unreachable.
            raise ValueError(f"Unsupported label mode for {gene}: {mode!r}")
    if not parts:
        return pd.DataFrame(columns=["gene", "mutation", "label", "gt_source"])
    # Per-subunit collision auditing already happens inside the grouped loaders
    # (build_clinvar_by_subunit_gt / build_mutsalpha_gt); no need to repeat it here.
    return pd.concat(parts, ignore_index=True)


def apply_configured_label_policies(merged_df, protein_configs, run_config):
    """Apply modes whose negatives are defined by absence or mutation position."""
    out = merged_df.copy()
    mutation_pos = pd.to_numeric(
        out["mutation"].str.extract(r"([0-9]+)", expand=False), errors="coerce"
    )
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
        elif mode == "hotspot_range":
            start = int(run_config[label_cfg["start_config_key"]])
            end = int(run_config[label_cfg["end_config_key"]])
            hotspot = gene_mask & mutation_pos.between(start, end, inclusive="both")
            non_hotspot = gene_mask & ~hotspot
            out.loc[hotspot, "label"] = 1
            out.loc[hotspot, "gt_source"] = f"{gene}_hotspot_{start}_{end}"
            out.loc[non_hotspot, "label"] = 0
            out.loc[non_hotspot, "gt_source"] = f"{gene}_outside_hotspot_passenger"
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
        display(_AFM_TABLE[1].head())
        display(_AFM_TABLE[0].head())
    return _AFM_TABLE


def __getattr__(name):  # core.afm_df / core.afm_audit keep working, loaded on first access
    if name == "afm_df":
        return load_afm_table()[0]
    if name == "afm_audit":
        return load_afm_table()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ============ labeled-dataset assembly (nb cell 3) ============


def summarize_training_overlap(merged_all_df, labeled_df, unknown_df):
    summary_rows = [
        {"metric": "merged_overlap_rows", "value": int(len(merged_all_df))},
        {"metric": "labeled_rows_used_for_training", "value": int(len(labeled_df))},
        {"metric": "unknown_rows", "value": int(len(unknown_df))},
        {"metric": "unique_labeled_gene_mutation", "value": int(labeled_df[["gene", "mutation"]].drop_duplicates().shape[0])},
        {"metric": "duplicate_labeled_gene_mutation_rows", "value": int(labeled_df.duplicated(["gene", "mutation"]).sum())},
    ]
    return pd.DataFrame(summary_rows)

def gene_label_breakdown(labeled_df):
    out = (
        labeled_df.groupby(["gene", "label"], as_index=False)
        .size()
        .pivot(index="gene", columns="label", values="size")
        .fillna(0)
        .astype(int)
        .rename(columns={0: "n_label_0", 1: "n_label_1"})
        .reset_index()
    )
    if "n_label_0" not in out.columns:
        out["n_label_0"] = 0
    if "n_label_1" not in out.columns:
        out["n_label_1"] = 0
    out["n_total"] = out["n_label_0"] + out["n_label_1"]
    return out.sort_values(["n_total", "gene"], ascending=[False, True]).reset_index(drop=True)

def summarize_d1_variant_usage(d1_rows_df):
    cols = [
        "gene", "mutation", "Variant", "d1_score", "structures_used",
        "n_replicate_rows", "source_kind",
    ]
    if d1_rows_df is None or d1_rows_df.empty:
        return pd.DataFrame(columns=cols)

    work = d1_rows_df.copy()
    work["structures_used"] = pd.to_numeric(work.get("Num_Structures", 0), errors="coerce").fillna(0).astype(int)

    if "source_file" not in work.columns:
        work["source_file"] = ""

    if "source_kind" not in work.columns:
        # Backward-compatible fallback for runtime rows produced before source_kind existed.
        default_source_kind = "csv" if str(globals().get("D1_SOURCE_POLICY", "")).startswith("csv") else "unknown"
        work["source_kind"] = default_source_kind

    out = (
        work.groupby(["gene", "mutation"], as_index=False)
        .agg(
            Variant=("Variant", "first"),
            d1_score=("d1_score", "mean"),
            structures_used=("structures_used", "sum"),
            n_replicate_rows=("source_file", "count"),
            source_kind=("source_kind", "first"),
        )
        .sort_values(["gene", "mutation"])
        .reset_index(drop=True)
    )
    return out[cols]

def summarize_d1_gene_diagnostics(plddt_source_scan_df, d1_variant_usage_df):
    source_scan = plddt_source_scan_df.copy()
    if source_scan.empty:
        return source_scan

    if d1_variant_usage_df is None or d1_variant_usage_df.empty:
        gene_usage = pd.DataFrame(columns=[
            "gene",
            "variants_with_d1",
            "structures_used_total",
            "structures_used_mean",
            "structures_used_median",
            "structures_used_min",
            "structures_used_max",
        ])
    else:
        gene_usage = (
            d1_variant_usage_df.groupby("gene", as_index=False)
            .agg(
                variants_with_d1=("mutation", "nunique"),
                structures_used_total=("structures_used", "sum"),
                structures_used_mean=("structures_used", "mean"),
                structures_used_median=("structures_used", "median"),
                structures_used_min=("structures_used", "min"),
                structures_used_max=("structures_used", "max"),
            )
        )

    out = source_scan.merge(gene_usage, on="gene", how="left")

    fill_zero_int = [
        "n_wt_sources_found",
        "n_mut_sources_found",
        "n_wt_cif_fallback_candidates",
        "n_mut_cif_fallback_candidates",
        "variants_with_d1",
        "structures_used_total",
        "structures_used_min",
        "structures_used_max",
    ]
    for col in fill_zero_int:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    for col in ["structures_used_mean", "structures_used_median"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["n_mut_sources_without_d1"] = (
        out["n_mut_sources_found"] - out["variants_with_d1"]
    ).clip(lower=0).astype(int)

    return out.sort_values("gene").reset_index(drop=True)

def _unique_gene_mut_counts(df, count_col_name):
    if df is None or df.empty:
        return pd.DataFrame(columns=["gene", count_col_name])
    return (
        df[["gene", "mutation"]]
        .drop_duplicates()
        .groupby("gene", as_index=False)
        .size()
        .rename(columns={"size": count_col_name})
    )

def summarize_variant_flow(d1_df, merged_all_df, labeled_df, unknown_df):
    d1_counts = _unique_gene_mut_counts(d1_df, "n_d1_variants")
    merged_counts = _unique_gene_mut_counts(merged_all_df, "n_after_afm_merge")
    labeled_counts = _unique_gene_mut_counts(labeled_df, "n_labeled")
    unknown_counts = _unique_gene_mut_counts(unknown_df, "n_unknown")

    out = d1_counts.merge(merged_counts, on="gene", how="outer")
    out = out.merge(labeled_counts, on="gene", how="outer")
    out = out.merge(unknown_counts, on="gene", how="outer")

    for col in ["n_d1_variants", "n_after_afm_merge", "n_labeled", "n_unknown"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    out["n_dropped_at_afm_merge"] = (out["n_d1_variants"] - out["n_after_afm_merge"]).clip(lower=0).astype(int)
    out["n_resolved_after_labeling"] = (out["n_labeled"] + out["n_unknown"]).astype(int)
    out["n_unaccounted_after_labeling"] = (
        out["n_after_afm_merge"] - out["n_resolved_after_labeling"]
    ).clip(lower=0).astype(int)

    out = out.sort_values("gene").reset_index(drop=True)

    totals = {
        "gene": "ALL",
        "n_d1_variants": int(out["n_d1_variants"].sum()),
        "n_after_afm_merge": int(out["n_after_afm_merge"].sum()),
        "n_labeled": int(out["n_labeled"].sum()),
        "n_unknown": int(out["n_unknown"].sum()),
        "n_dropped_at_afm_merge": int(out["n_dropped_at_afm_merge"].sum()),
        "n_resolved_after_labeling": int(out["n_resolved_after_labeling"].sum()),
        "n_unaccounted_after_labeling": int(out["n_unaccounted_after_labeling"].sum()),
    }
    out = pd.concat([out, pd.DataFrame([totals])], ignore_index=True)
    return out


def build_labeled_dataset_for_config(
    foxa1_hotspot_range,
    d1_max_structures_per_variant,
    d1_ratio_mode=D1_RATIO_MODE,
    d1_seed=SEED,
    run_config=None,
    d1_sampling_policy=None,
):
    afm_df = load_afm_table()[0]
    if "afm_class" not in afm_df.columns:
        raise RuntimeError("`afm_class` is missing from AFM table.")

    hotspot_start, hotspot_end = [int(x) for x in foxa1_hotspot_range]
    if hotspot_start > hotspot_end:
        raise ValueError("FOXA1 hotspot start must be <= end")

    label_run_config = dict(run_config or {})
    label_run_config.setdefault("foxa1_hotspot_start", hotspot_start)
    label_run_config.setdefault("foxa1_hotspot_end", hotspot_end)

    d1_sampling_policy = SamplingPolicy.coerce(d1_sampling_policy)
    label_run_config.setdefault("d1_sampling_mode", d1_sampling_policy.mode)
    label_run_config.setdefault("d1_selection", d1_sampling_policy.selection)

    d1_rows_local, d1_audit_local = build_d1_table(
        D1_RUNTIME_CONFIGS,
        max_structures_per_variant=d1_max_structures_per_variant,
        random_seed=int(d1_seed),
        ratio_mode=d1_ratio_mode,
        policy=d1_sampling_policy,
    )
    d1_df_local = aggregate_d1_rows_to_mutation_level(d1_rows_local, ratio_mode=d1_ratio_mode)

    plddt_source_scan = scan_plddt_sources(D1_RUNTIME_CONFIGS)

    afm_for_merge_local = afm_df.copy()
    afm_for_merge_local["afm_label_5class"] = afm_for_merge_local["afm_class"].map(normalize_afm_class)
    afm_for_merge_local["afm_label_5class_id"] = (
        afm_for_merge_local["afm_label_5class"]
        .map(AFM_5CLASS_TO_ID)
        .fillna(AFM_5CLASS_TO_ID["Unknown"])
        .astype(int)
    )
    afm_for_merge_local["afm_label_binary"] = afm_for_merge_local["afm_label_5class"].map(afm_class_to_binary_label).astype(int)

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
        afm_for_merge_local = (
            pd.concat([afm_for_merge_local, grouped_afm], ignore_index=True)
            .drop_duplicates(["gene", "mutation"], keep="first")
            .reset_index(drop=True)
        )
    
    merged_all_local = d1_df_local.merge(afm_for_merge_local, on=["gene", "mutation"], how="inner")
    merged_all_local = merged_all_local.rename(columns={"label": "d1_label"})

    external_gt_df_local = build_configured_external_gt(
        PROTEIN_CONFIGS, label_run_config
    )

    gt_lookup = (
        external_gt_df_local.sort_values(["gene", "mutation", "label"], ascending=[True, True, False])
        .drop_duplicates(["gene", "mutation"], keep="first")
    )
    merged_all_local = merged_all_local.merge(
        gt_lookup[["gene", "mutation", "label", "gt_source"]],
        on=["gene", "mutation"],
        how="left",
    )
    merged_all_local["gt_source"] = merged_all_local["gt_source"].fillna("none")

    merged_all_local["label"] = merged_all_local["label"].fillna(-1).astype(int)
    merged_all_local = apply_configured_label_policies(
        merged_all_local, PROTEIN_CONFIGS, label_run_config
    )

    labeled_df = (
        merged_all_local[merged_all_local["label"].isin([0, 1])]
        .sort_values(["gene", "mutation"])
        .drop_duplicates(["gene", "mutation"], keep="first")
        .reset_index(drop=True)
        .copy()
    )
    unknown_df = merged_all_local[merged_all_local["label"] == -1].copy()

    if labeled_df.empty:
        raise RuntimeError("No labeled rows available after AFM + D1 + GT merge.")
    if labeled_df["label"].nunique() < 2:
        raise RuntimeError("Labeled dataset must contain both classes.")

    overlap_audit = summarize_training_overlap(merged_all_local, labeled_df, unknown_df)
    per_gene_audit = gene_label_breakdown(labeled_df)

    d1_variant_usage_audit = summarize_d1_variant_usage(d1_rows_local)
    d1_gene_audit = summarize_d1_gene_diagnostics(plddt_source_scan, d1_variant_usage_audit)
    variant_flow_audit = summarize_variant_flow(d1_df_local, merged_all_local, labeled_df, unknown_df)

    return {
        "d1_rows": d1_rows_local,
        "d1_df": d1_df_local,
        "d1_audit": d1_audit_local,
        "plddt_source_scan": plddt_source_scan,
        "d1_variant_usage_audit": d1_variant_usage_audit,
        "d1_gene_audit": d1_gene_audit,
        "variant_flow_audit": variant_flow_audit,
        "merged_all": merged_all_local,
        "labeled_df": labeled_df,
        "unknown_df": unknown_df,
        "overlap_audit": overlap_audit,
        "per_gene_audit": per_gene_audit,
        "n_labeled": int(len(labeled_df)),
        "n_unknown": int(len(unknown_df)),
    }

def plot_joint_afm_d1(merged_df):
    plot_all = merged_df.copy()
    genes = sorted(plot_all["gene"].dropna().unique().tolist())
    if not genes:
        print("No rows to plot.")
        return

    nrows = len(genes)
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    shared_handles = None
    shared_labels = None

    for idx, gene in enumerate(genes):
        ax = axes[idx]
        sub = plot_all[plot_all["gene"] == gene].copy()
        sns.scatterplot(
            data=sub,
            x="afm_score",
            y="d1_score",
            hue="afm_label_5class",
            hue_order=AFM_5CLASS_ORDER,
            alpha=0.7,
            ax=ax,
            legend=(idx == 0),
        )
        ax.set_title(f"{gene} (n={len(sub)})")
        ax.set_yscale("log")
        ax.set_ylabel("D1 score (log scale)")
        ax.grid(alpha=0.3)
        if ax.legend_ is not None:
            if idx == 0:
                shared_handles, shared_labels = ax.get_legend_handles_labels()
            ax.legend_.remove()

    if shared_handles is not None and shared_labels is not None:
        fig.legend(shared_handles, shared_labels, loc="center right", bbox_to_anchor=(1.01, 0.5), frameon=True)
        plt.tight_layout(rect=[0, 0, 0.86, 1])
    else:
        plt.tight_layout()
    plt.show()

# ========== model / CV / metrics (nb cell 4) ==========


class ScoreFusionHead(nn.Module):
    """MLP over the 6 engineered AFM/D1 features, plus optional extra features.

    With n_extra=0 this is exactly the original head (in_dim=6): same layers, same
    init order, so base results are unchanged. With n_extra>0 the model expects
    an input of [afm, d1, extra_1..extra_k]; the afm/d1 pair is featurized into
    the usual 6 interaction terms and the k extras are appended raw.
    """

    def __init__(self, hidden=64, dropout=0.2, n_extra=0):
        super().__init__()
        self.n_extra = int(n_extra)
        in_dim = 6 + self.n_extra
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def featurize(self, x):
        afm = x[:, 0:1]
        d1 = x[:, 1:2]
        delta = d1 - afm
        abs_delta = delta.abs()
        prod = afm * d1
        sq_delta = delta.pow(2)
        base = torch.cat([afm, d1, delta, abs_delta, prod, sq_delta], dim=1)
        if self.n_extra > 0:
            return torch.cat([base, x[:, 2:2 + self.n_extra]], dim=1)
        return base

    def forward(self, x):
        return self.net(self.featurize(x))

class PlainMLP(nn.Module):
    """Same architecture as ScoreFusionHead but with NO afm/d1 interaction
    featurization -- it consumes an already-standardized feature matrix as-is.
    Used for the structure-only track (D1 + pLDDT-derived metrics, no AFM),
    where the hardcoded afm*d1 interactions would be meaningless."""

    def __init__(self, in_dim, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


FUSION_REPRESENTATIONS = (
    "raw2", "compact4_v2", "compact4", "engineered6",
)
FUSION_NORMALIZATIONS = ("original", "robust", "rank")


def normalize_fusion_scores_np(raw_features, genes, normalization="original"):
    """Apply one label-free AFM/D1 score representation within each gene.

    ``original`` is inductive and returns an exact numeric copy. ``robust`` and
    ``rank`` use the complete unlabelled score distribution for each gene,
    including a held-out gene, so they are transductive batch normalizations.
    They never inspect labels. A caller predicting robust/rank must therefore
    provide the complete gene batch used to define the transformation.
    """
    raw = np.asarray(raw_features, dtype=np.float32)
    groups = np.asarray(genes, dtype=str)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("AFM+D1 features must have shape (n, >=2)")
    if len(raw) != len(groups):
        raise ValueError("fusion scores and genes must have the same number of rows")
    if normalization == "original":
        return raw.copy()
    if normalization not in FUSION_NORMALIZATIONS:
        raise ValueError(
            f"unknown normalization {normalization!r} for fusion scores; "
            f"choose from {list(FUSION_NORMALIZATIONS)}"
        )
    out = np.empty_like(raw, dtype=np.float32)
    for gene in sorted(np.unique(groups)):
        idx = np.flatnonzero(groups == gene)
        block = raw[idx]
        if normalization == "robust":
            median = np.median(block, axis=0)
            q25, q75 = np.percentile(block, [25, 75], axis=0)
            scale = q75 - q25
            scale = np.where(scale > 1e-12, scale, 1.0)
            out[idx] = (block - median) / scale
        else:  # rank
            for column in range(block.shape[1]):
                out[idx, column] = pd.Series(block[:, column]).rank(
                    method="average", pct=True,
                ).to_numpy(dtype=np.float32)
    return out


def featurize_fusion_np(raw_features):
    """Historical six-feature AFM+D1 basis."""
    x = np.asarray(raw_features, dtype=np.float32)
    afm = x[:, [0]]
    d1 = x[:, [1]]
    delta = d1 - afm
    abs_delta = np.abs(delta)
    prod = afm * d1
    sq_delta = delta ** 2
    return np.concatenate([afm, d1, delta, abs_delta, prod, sq_delta], axis=1).astype(np.float32)


def featurize_compact4_np(raw_features):
    """Legacy compact nonlinear basis: AFM, D1, D1 squared, and AFM-by-D1.

    This is retained so completed ``compact4`` runs remain reproducible. New
    experiments should use ``compact4_v2``.
    """
    x = np.asarray(raw_features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("AFM+D1 features must have shape (n, >=2)")
    afm = x[:, [0]]
    d1 = x[:, [1]]
    return np.concatenate([afm, d1, d1 ** 2, afm * d1], axis=1).astype(np.float32)


def featurize_compact4_v2_np(raw_features):
    """Stable compact basis: AFM, D1, AFM-by-D1, and squared disagreement.

    Compared with the legacy compact basis, this removes standalone ``D1^2``
    (which transferred poorly to FOXA1 and POLE) while preserving one interaction
    and one disagreement term from the successful historical six-feature basis.
    """
    x = np.asarray(raw_features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("AFM+D1 features must have shape (n, >=2)")
    afm = x[:, [0]]
    d1 = x[:, [1]]
    return np.concatenate(
        [afm, d1, afm * d1, (d1 - afm) ** 2], axis=1,
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# Which descriptor pairs with AlphaMissense in the fusion basis.
#
# The basis was hardwired to (afm_score, d1_score), and the representations are
# purely positional -- featurize_* read x[:, 0] and x[:, 1] and never ask what
# column 1 means. So the partner is a free choice, and "AM + KS" was simply
# unreachable: feature variants only APPEND to the basis, they cannot replace
# D1 inside it. cohort_size_sweep.py has an afm_ks_raw candidate; the sweep and
# stratified paths had no equivalent.
#
# Default "d1" reproduces every prior result byte-for-byte.
# --------------------------------------------------------------------------- #
FUSION_PARTNERS = {
    "d1": "d1_score",
    # The profile-KS partner ("ks") was removed on 2026-10-13.
    "ks_residue": "plddt_ks_residue",
    "ks_centred": "plddt_ks_centred",
    "wass": "plddt_wasserstein",
    "spread": "plddt_spread_ratio",
}


def fusion_partner_name(cfg):
    """The configured partner key, defaulting to the historical 'd1'."""
    name = (cfg or {}).get("fusion_partner", "d1")
    if name not in FUSION_PARTNERS:
        raise ValueError(
            f"unknown fusion partner {name!r}; choose from {list(FUSION_PARTNERS)}"
        )
    return name


def fusion_basis_columns(cfg):
    """The two labeled_df columns forming the fusion basis, in order."""
    return ["afm_score", FUSION_PARTNERS[fusion_partner_name(cfg)]]


def featurize_fusion_representation_np(raw_features, representation="engineered6", extras=None):
    """Build one auditable AFM+D1 basis and append optional true descriptors.

    ``extras`` are appended only after the AFM+D1 basis is built. Scaling is
    deliberately left to the caller so it can be fit on training rows only.
    """
    x = np.asarray(raw_features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("AFM+D1 features must have shape (n, >=2)")
    if representation == "raw2":
        base = x[:, :2].astype(np.float32, copy=True)
    elif representation == "compact4_v2":
        base = featurize_compact4_v2_np(x)
    elif representation == "compact4":
        base = featurize_compact4_np(x)
    elif representation == "engineered6":
        base = featurize_fusion_np(x)
    else:
        raise ValueError(
            f"unknown fusion representation {representation!r}; "
            f"choose from {list(FUSION_REPRESENTATIONS)}"
        )
    if extras is None:
        return base
    extras = np.asarray(extras, dtype=np.float32)
    if extras.ndim == 1:
        extras = extras[:, None]
    if extras.ndim != 2 or extras.shape[0] != base.shape[0]:
        raise ValueError("extra fusion features must have shape (n, k)")
    if extras.shape[1] == 0:
        return base
    return np.concatenate([base, extras], axis=1).astype(np.float32)


def fit_fusion_feature_preprocessor_np(
    train_scores, validation_scores, representation="engineered6",
    train_extras=None, validation_extras=None,
):
    """Fit leak-free representation preprocessing on one training fold.

    Every representation is constructed first and then standardized on the
    training fold so MLP and logistic models consume the exact same matrix. The
    returned scaler is sufficient for later artifact prediction.
    """
    train_scores = np.asarray(train_scores, dtype=np.float32)
    validation_scores = np.asarray(validation_scores, dtype=np.float32)
    train_basis_input = train_scores
    validation_basis_input = validation_scores

    train_features = featurize_fusion_representation_np(
        train_basis_input, representation=representation, extras=train_extras,
    )
    validation_features = featurize_fusion_representation_np(
        validation_basis_input, representation=representation,
        extras=validation_extras,
    )
    feature_scaler = StandardScaler()
    x_train = feature_scaler.fit_transform(train_features).astype(np.float32)
    x_validation = feature_scaler.transform(validation_features).astype(np.float32)
    return x_train, x_validation, {
        "input_scaler": None,
        "scaler": feature_scaler,
    }


def transform_fusion_features_np(scores, representation, preprocessing, extras=None):
    """Apply a fitted fold preprocessor to new AFM+D1 scores."""
    basis_input = np.asarray(scores, dtype=np.float32)
    input_scaler = preprocessing.get("input_scaler")
    if input_scaler is not None:
        basis_input = input_scaler.transform(basis_input).astype(np.float32)
    features = featurize_fusion_representation_np(
        basis_input, representation=representation, extras=extras,
    )
    return preprocessing["scaler"].transform(features).astype(np.float32)


def featurize_fusion_np_ext(raw_features, extras=None):
    """The 6 engineered AFM/D1 features, with any extra ensemble-shift features
    appended raw. `extras` is (n, k) or None. Base call reproduces featurize_fusion_np."""
    return featurize_fusion_representation_np(
        raw_features, representation="engineered6", extras=extras,
    )

def predict_proba(model, X_np, batch_size=1024, device=None):
    _device = device if device is not None else DEVICE
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32, device=_device)
            probs = torch.sigmoid(model(xb)).squeeze(1).detach().cpu().numpy()
            out.append(probs)
    return np.concatenate(out) if out else np.array([], dtype=np.float32)

def _safe_metric(metric_fn, *args, default=np.nan, **kwargs):
    try:
        return float(metric_fn(*args, **kwargs))
    except Exception:
        return float(default)

def compute_binary_metrics_at_threshold(y_true, probs, threshold):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float32)
    threshold = float(threshold)
    preds = (probs >= threshold).astype(int)

    return {
        "threshold": threshold,
        "acc": _safe_metric(accuracy_score, y_true, preds),
        "balanced_acc": _safe_metric(balanced_accuracy_score, y_true, preds),
        "f1": _safe_metric(f1_score, y_true, preds, zero_division=0),
        "precision": _safe_metric(precision_score, y_true, preds, zero_division=0),
        "recall": _safe_metric(recall_score, y_true, preds, zero_division=0),
        "mcc": _safe_metric(matthews_corrcoef, y_true, preds),
    }

def build_threshold_sweep(y_true, probs, thresholds=None):
    thr_grid = np.linspace(0.01, 0.99, 199) if thresholds is None else np.asarray(thresholds, dtype=np.float32)
    rows = [compute_binary_metrics_at_threshold(y_true, probs, float(t)) for t in thr_grid]
    return pd.DataFrame(rows)

def summarize_probability_scores(y_true, probs, thresholds=None):
    """OOF probability-based metrics.

    THRESHOLD CAVEAT: `best_t` is chosen to maximize F1 over the SAME labels the
    `*_at_best_t` metrics are then evaluated on. The predictions are out-of-fold,
    but the threshold is selected in-sample, so `*_at_best_t` are retrospectively
    optimized, NOT held-out estimates -- `best_t_is_insample=True` flags this. The
    unbiased, threshold-free results are `oof_auc` / `oof_ap`; `*_at_05` use a fixed
    threshold and are also unbiased. A future nested-threshold variant (pick t on
    training folds, apply to the held-out fold) would make an honest thresholded
    metric; until then treat `*_at_best_t` as descriptive only."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float32)
    thr_grid = np.linspace(0.01, 0.99, 199) if thresholds is None else np.asarray(thresholds, dtype=np.float32)
    threshold_sweep_df = build_threshold_sweep(y_true, probs, thresholds=thr_grid)
    if threshold_sweep_df.empty:
        best_t = 0.5
    else:
        best_t = float(threshold_sweep_df.loc[threshold_sweep_df["f1"].idxmax(), "threshold"])

    m05 = compute_binary_metrics_at_threshold(y_true, probs, 0.5)
    mbt = compute_binary_metrics_at_threshold(y_true, probs, best_t)
    metrics = {
        "oof_ap": _safe_metric(average_precision_score, y_true, probs),
        "oof_auc": _safe_metric(roc_auc_score, y_true, probs),
        "brier": _safe_metric(brier_score_loss, y_true, np.clip(probs, 0.0, 1.0)),
        "best_t": float(best_t),
        "best_t_is_insample": True,
        "f1_at_05": m05["f1"],
        "f1_at_best_t": mbt["f1"],
        "acc_at_05": m05["acc"],
        "acc_at_best_t": mbt["acc"],
        "balanced_acc_at_05": m05["balanced_acc"],
        "balanced_acc_at_best_t": mbt["balanced_acc"],
        "mcc_at_05": m05["mcc"],
        "mcc_at_best_t": mbt["mcc"],
        "precision_at_05": m05["precision"],
        "precision_at_best_t": mbt["precision"],
        "recall_at_05": m05["recall"],
        "recall_at_best_t": mbt["recall"],
    }
    return {
        "metrics": metrics,
        "oof_probs": probs,
        "threshold_sweep_df": threshold_sweep_df,
    }

def build_calibration_df(y_true, probs, n_bins=10, strategy="uniform"):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float32)
    try:
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy=strategy)
        return pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos})
    except Exception:
        return pd.DataFrame(columns=["mean_pred", "frac_pos"])

def make_oof_summary_figure(
    y_true,
    oof_probs,
    history_df=None,
    best_t=None,
    metrics_dict=None,
    baseline_probs=None,
    model_label="MLP",
    baseline_label="LogReg",
    score_context="OOF",
):
    y_true = np.asarray(y_true).astype(int)
    oof_probs = np.asarray(oof_probs, dtype=np.float32)

    oof_ap = _safe_metric(average_precision_score, y_true, oof_probs)
    oof_auc = _safe_metric(roc_auc_score, y_true, oof_probs)
    brier = _safe_metric(brier_score_loss, y_true, np.clip(oof_probs, 0.0, 1.0))

    thr_df = build_threshold_sweep(y_true, oof_probs)
    if best_t is None:
        if thr_df.empty:
            best_t = 0.5
        else:
            best_t = float(thr_df.loc[thr_df["f1"].idxmax(), "threshold"])

    pred_05 = (oof_probs >= 0.5).astype(int)
    pred_best = (oof_probs >= float(best_t)).astype(int)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    cm_05 = confusion_matrix(y_true, pred_05, labels=[0, 1])
    cm_bt = confusion_matrix(y_true, pred_best, labels=[0, 1])

    sns.heatmap(cm_05, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0, 0])
    axes[0, 0].set_title("Confusion Matrix (t=0.5)")
    sns.heatmap(cm_bt, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0, 1])
    axes[0, 1].set_title(f"Confusion Matrix (t={best_t:.2f})")

    fpr, tpr, _ = roc_curve(y_true, oof_probs)
    axes[0, 2].plot(fpr, tpr, lw=2, label=f"{model_label} AUC={oof_auc:.3f}" if np.isfinite(oof_auc) else f"{model_label} AUC=nan")
    if baseline_probs is not None:
        baseline_probs = np.asarray(baseline_probs, dtype=np.float32)
        baseline_auc = _safe_metric(roc_auc_score, y_true, baseline_probs)
        fpr_b, tpr_b, _ = roc_curve(y_true, baseline_probs)
        axes[0, 2].plot(fpr_b, tpr_b, lw=1.5, ls="--", label=f"{baseline_label} AUC={baseline_auc:.3f}" if np.isfinite(baseline_auc) else f"{baseline_label} AUC=nan")
    axes[0, 2].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 2].set_title("ROC")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, oof_probs)
    baseline_rate = y_true.mean()
    axes[1, 0].plot(rec, prec, lw=2, label=f"{model_label} AP={oof_ap:.3f}")
    if baseline_probs is not None:
        baseline_ap = _safe_metric(average_precision_score, y_true, baseline_probs)
        prec_b, rec_b, _ = precision_recall_curve(y_true, baseline_probs)
        axes[1, 0].plot(rec_b, prec_b, lw=1.5, ls="--", label=f"{baseline_label} AP={baseline_ap:.3f}")
    axes[1, 0].axhline(baseline_rate, color="gray", ls="--", lw=1, label=f"baseline={baseline_rate:.3f}")
    axes[1, 0].set_title("Precision-Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].hist(oof_probs[y_true == 0], bins=20, alpha=0.6, label=f"Passenger ({model_label})")
    axes[1, 1].hist(oof_probs[y_true == 1], bins=20, alpha=0.6, label=f"Driver ({model_label})")
    axes[1, 1].axvline(0.5, color="black", ls="--", lw=1, label="t=0.5")
    axes[1, 1].axvline(best_t, color="green", ls="--", lw=1, label=f"t*={best_t:.2f}")
    axes[1, 1].set_title(f"{score_context} Score Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    if not thr_df.empty:
        axes[1, 2].plot(thr_df["threshold"], thr_df["f1"], label="F1", lw=2)
        axes[1, 2].plot(thr_df["threshold"], thr_df["precision"], label="Precision", lw=1.8)
        axes[1, 2].plot(thr_df["threshold"], thr_df["recall"], label="Recall", lw=1.8)
        axes[1, 2].plot(thr_df["threshold"], thr_df["balanced_acc"], label="Balanced Acc", lw=1.8)
        axes[1, 2].axvline(0.5, color="black", ls="--", lw=1)
        axes[1, 2].axvline(best_t, color="green", ls="--", lw=1)
        axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].set_title("Threshold Sweep")
    axes[1, 2].set_xlabel("Threshold")
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].legend()

    cal_df = build_calibration_df(y_true, oof_probs, n_bins=10)
    axes[2, 0].plot([0, 1], [0, 1], "k--", lw=1)
    if not cal_df.empty:
        axes[2, 0].plot(cal_df["mean_pred"], cal_df["frac_pos"], marker="o", lw=2, label=model_label)
    if baseline_probs is not None:
        cal_df_b = build_calibration_df(y_true, baseline_probs, n_bins=10)
        if not cal_df_b.empty:
            axes[2, 0].plot(cal_df_b["mean_pred"], cal_df_b["frac_pos"], marker="o", lw=1.5, ls="--", label=baseline_label)
    axes[2, 0].set_title("Calibration")
    axes[2, 0].set_xlabel("Mean Predicted Probability")
    axes[2, 0].set_ylabel("Fraction of Positives")
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.3)

    if history_df is not None and not history_df.empty:
        curve_df = history_df.groupby("epoch", as_index=False)[["train_loss", "val_ap"]].mean()
        axes[2, 1].plot(curve_df["epoch"], curve_df["train_loss"], label="train_loss")
        axes[2, 1].plot(curve_df["epoch"], curve_df["val_ap"], label="val_ap")
        axes[2, 1].set_title("Average Training Curves")
        axes[2, 1].grid(alpha=0.3)
        axes[2, 1].legend()
    else:
        axes[2, 1].axis("off")

    m = dict(metrics_dict) if metrics_dict else {}
    if not m:
        m05 = compute_binary_metrics_at_threshold(y_true, oof_probs, 0.5)
        mbt = compute_binary_metrics_at_threshold(y_true, oof_probs, best_t)
        m = {
            "oof_ap": oof_ap,
            "oof_auc": oof_auc,
            "brier": brier,
            "best_t": float(best_t),
            "f1_at_05": m05["f1"],
            "f1_at_best_t": mbt["f1"],
            "balanced_acc_at_05": m05["balanced_acc"],
            "balanced_acc_at_best_t": mbt["balanced_acc"],
            "mcc_at_05": m05["mcc"],
            "mcc_at_best_t": mbt["mcc"],
        }

    summary_lines = [
        f"AP: {m.get('oof_ap', np.nan):.4f}",
        f"AUC: {m.get('oof_auc', np.nan):.4f}",
        f"Brier: {m.get('brier', np.nan):.4f}",
        f"Best threshold (in-sample): {m.get('best_t', np.nan):.3f}",
        f"F1@0.5: {m.get('f1_at_05', np.nan):.4f}",
        f"F1@best*: {m.get('f1_at_best_t', np.nan):.4f}",
        f"MCC@0.5: {m.get('mcc_at_05', np.nan):.4f}",
        f"MCC@best*: {m.get('mcc_at_best_t', np.nan):.4f}",
        f"BalAcc@0.5: {m.get('balanced_acc_at_05', np.nan):.4f}",
        f"BalAcc@best*: {m.get('balanced_acc_at_best_t', np.nan):.4f}",
        "* threshold optimized in-sample (descriptive, not held-out)",
    ]
    axes[2, 2].axis("off")
    axes[2, 2].text(0.0, 1.0, "\n".join(summary_lines), va="top", ha="left", fontsize=11)
    axes[2, 2].set_title(f"{score_context} Metric Summary")

    plt.tight_layout()
    return fig

def _stratified_inner_split(y_train, val_frac=0.2):
    """Stratified inner split of the OUTER-training fold, used for epoch SELECTION
    only. Draws from the already-seeded global numpy RNG, so it is reproducible per
    run seed and varies across seeds. Returns (inner_train_idx, inner_val_idx), or
    None when any class is too small to place >=1 example in both halves -- in which
    case the caller falls back to a fixed epoch budget."""
    y_train = np.asarray(y_train).astype(int)
    classes = np.unique(y_train)
    if classes.size < 2:
        return None
    tr_idx, va_idx = [], []
    for c in classes:
        idx = np.where(y_train == c)[0]
        if idx.size < 2:  # need >=1 for inner-train AND >=1 for inner-val
            return None
        perm = np.random.permutation(idx)
        n_val = max(1, int(round(val_frac * idx.size)))
        n_val = min(n_val, idx.size - 1)  # keep >=1 per class in inner-train
        va_idx.extend(perm[:n_val].tolist())
        tr_idx.extend(perm[n_val:].tolist())
    return np.array(sorted(tr_idx), dtype=int), np.array(sorted(va_idx), dtype=int)


def _build_fold_model(model_kind, in_dim, hidden, dropout, n_extra, device):
    if model_kind == "plain":
        return PlainMLP(in_dim=int(in_dim), hidden=hidden, dropout=dropout).to(device)
    return ScoreFusionHead(hidden=hidden, dropout=dropout, n_extra=n_extra).to(device)


def _run_training(model, X_train, y_train, epochs, batch_size, lr, weight_decay,
                  device, X_val=None, y_val=None, patience=None,
                  log_to_wandb=False, fold=None):
    """Train `model` in place.

    With (X_val, y_val): track validation AP each epoch, optionally early-stop on
    `patience`, restore the best epoch's weights, and report best_epoch/best_ap.
    This is the epoch-SELECTION pass and its validation set is the train-only inner
    split -- never the outer-validation fold.

    Without a validation set: train exactly `epochs` epochs with no selection. This
    is the REFIT pass on the full outer-training fold."""
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight_value = neg / max(pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    select = X_val is not None and y_val is not None
    best_ap = -np.inf
    best_state = None
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        train_loss = running / max(len(loader), 1)
        row = {"epoch": epoch, "train_loss": train_loss}

        if select:
            val_probs = predict_proba(model, X_val, device=device)
            val_ap = average_precision_score(y_val, val_probs)
            clipped = np.clip(val_probs, 1e-7, 1.0 - 1e-7)
            val_loss = -float(np.mean(
                pos_weight_value * y_val * np.log(clipped)
                + (1 - y_val) * np.log(1.0 - clipped)
            ))
            val_at_05 = compute_binary_metrics_at_threshold(y_val, val_probs, 0.5)
            row.update({
                "val_loss": val_loss,
                "val_ap": val_ap,
                "val_auc": _safe_metric(roc_auc_score, y_val, val_probs),
                "val_f1_05": val_at_05["f1"],
                "val_precision_05": val_at_05["precision"],
                "val_recall_05": val_at_05["recall"],
                "val_mcc_05": val_at_05["mcc"],
            })
            if log_to_wandb and WANDB_AVAILABLE and wandb is not None and wandb.run is not None:
                wandb.log({
                    "fold": fold,
                    f"fold_{fold}/epoch": epoch,
                    f"fold_{fold}/train_loss": train_loss,
                    f"fold_{fold}/val_loss": val_loss,
                    f"fold_{fold}/val_ap": val_ap,
                    f"fold_{fold}/val_auc": row["val_auc"],
                })
            if val_ap > best_ap:
                best_ap = val_ap
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
        history.append(row)
        if select and patience is not None and wait >= patience:
            break

    if select and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = int(epochs)

    return {
        "best_epoch": int(best_epoch),
        "best_ap": float(best_ap) if np.isfinite(best_ap) else np.nan,
        "history": history,
    }


def train_one_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    hidden=FUSION_MODEL_DEFAULTS["hidden"],
    dropout=FUSION_MODEL_DEFAULTS["dropout"],
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    fold=None,
    log_to_wandb=False,
    device=None,
    n_extra=0,
    model_kind="fusion",
):
    _device = device if device is not None else DEVICE
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train).astype(int)
    in_dim = X_train.shape[1]
    inner_val_frac = 0.2

    # ---- Pass 1: SELECT best_epoch on a train-only inner split. The outer-validation
    #      fold (X_val, y_val) is never touched here, so it cannot leak into model
    #      selection -- the flaw that made the old OOF numbers optimistic. ----
    split = _stratified_inner_split(y_train, val_frac=inner_val_frac)
    if split is not None:
        inner_tr, inner_va = split
        sel_model = _build_fold_model(model_kind, in_dim, hidden, dropout, n_extra, _device)
        sel = _run_training(
            sel_model, X_train[inner_tr], y_train[inner_tr],
            epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            device=_device, X_val=X_train[inner_va], y_val=y_train[inner_va],
            patience=patience, log_to_wandb=log_to_wandb, fold=fold,
        )
        best_epoch = max(1, int(sel["best_epoch"]))
        best_ap = sel["best_ap"]
        sel_history = pd.DataFrame(sel["history"])
        used_fixed_epochs = False
    else:
        # Fold too small for an honest inner split -> leak-free fixed epoch budget.
        best_epoch = int(epochs)
        best_ap = np.nan
        sel_history = pd.DataFrame(columns=[
            "epoch", "train_loss", "val_loss", "val_ap", "val_auc",
            "val_f1_05", "val_precision_05", "val_recall_05", "val_mcc_05",
        ])
        used_fixed_epochs = True

    # ---- Pass 2: REFIT a fresh model on the FULL outer-training fold for exactly
    #      best_epoch epochs (no early stopping). LogReg trains on the same full fold,
    #      so the fusion-vs-baseline comparison is now like-for-like. ----
    model = _build_fold_model(model_kind, in_dim, hidden, dropout, n_extra, _device)
    refit = _run_training(
        model, X_train, y_train,
        epochs=best_epoch, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        device=_device, X_val=None, y_val=None, patience=None,
    )

    # Single, honest prediction of the untouched outer-validation fold.
    final_val_probs = predict_proba(model, X_val, device=_device)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    refit_history = pd.DataFrame(refit["history"]).assign(phase="refit")
    if "val_ap" not in refit_history.columns:
        refit_history["val_ap"] = np.nan  # refit has no inner-val; keep schema stable
    history = (
        pd.concat([sel_history.assign(phase="select"), refit_history], ignore_index=True)
        if len(sel_history) else refit_history
    )
    return {
        "state_dict": best_state,
        "best_epoch": int(best_epoch),
        "best_ap": best_ap,
        "val_probs": final_val_probs,
        "history": history,
        "used_fixed_epochs": bool(used_fixed_epochs),
    }

def _resolve_run_config(overrides=None):
    cfg = dict(SWEEP_BASE_CONFIG)
    if overrides:
        cfg.update(dict(overrides))

    cfg["foxa1_hotspot_start"] = int(cfg["foxa1_hotspot_start"])
    cfg["foxa1_hotspot_end"] = int(cfg["foxa1_hotspot_end"])
    cfg["seed"] = int(cfg["seed"])
    cfg["folds"] = int(cfg["folds"])
    cfg["epochs"] = int(cfg["epochs"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["lr"] = float(cfg["lr"])
    cfg["weight_decay"] = float(cfg["weight_decay"])
    cfg["patience"] = int(cfg["patience"])
    cfg["hidden"] = int(cfg["hidden"])
    cfg["dropout"] = float(cfg["dropout"])

    v = cfg.get("d1_max_structures_per_variant", None)
    cfg["d1_max_structures_per_variant"] = None if v is None else int(v)
    if cfg["d1_max_structures_per_variant"] is not None and cfg["d1_max_structures_per_variant"] <= 0:
        raise ValueError("d1_max_structures_per_variant must be a positive integer or None")

    cfg["d1_ratio_mode"] = D1_RATIO_MODE

    # Validate eagerly so a typo fails at config time rather than after a long
    # D1 rebuild, and normalise to the canonical strings used in cache keys.
    policy = SamplingPolicy(
        mode=str(cfg.get("d1_sampling_mode", "nested")),
        selection=str(cfg.get("d1_selection", "random")),
    )
    cfg["d1_sampling_mode"] = policy.mode
    cfg["d1_selection"] = policy.selection
    return cfg


def sampling_policy_from_config(cfg):
    """The SamplingPolicy a resolved run config describes."""
    return SamplingPolicy(
        mode=str(cfg.get("d1_sampling_mode", "nested")),
        selection=str(cfg.get("d1_selection", "random")),
    )

def run_cross_validated_training(
    labeled_df_in,
    cfg,
    log_to_wandb=False,
    log_fold_tables=False,
    log_fold_figures=True,
    log_oof_summary=True,
    fusion_representation="engineered6",
    fusion_normalization="original",
    progress_callback=None,
):
    seed_everything(cfg["seed"])
    _ = bool(log_fold_figures), bool(log_oof_summary)

    # Stable ordering prevents avoidable fold drift when the same examples are present.
    labeled_df_in = (
        labeled_df_in
        .sort_values(["gene", "mutation"])
        .reset_index(drop=True)
        .copy()
    )

    basis_columns = fusion_basis_columns(cfg)
    raw_features = labeled_df_in[basis_columns].to_numpy(dtype=np.float32)
    groups = labeled_df_in["gene"].astype(str).to_numpy()
    represented_scores = normalize_fusion_scores_np(
        raw_features, groups, normalization=fusion_normalization,
    )
    y = labeled_df_in["label"].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        raise RuntimeError("Cross-validation requires both classes in the labeled set.")

    strata_combo = labeled_df_in["gene"].astype(str) + "__" + labeled_df_in["label"].astype(str)
    split_target = y if strata_combo.value_counts().min() < cfg["folds"] else strata_combo

    skf = StratifiedKFold(n_splits=cfg["folds"], shuffle=True, random_state=cfg["seed"])
    oof_probs = np.zeros(len(labeled_df_in), dtype=np.float32)
    baseline_oof_probs = np.zeros(len(labeled_df_in), dtype=np.float32)

    fold_metrics = []
    baseline_fold_metrics = []
    fold_artifacts = []
    baseline_fold_artifacts = []
    histories = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(len(y)), split_target), start=1):
        X_tr, X_va, preprocessing = fit_fusion_feature_preprocessor_np(
            represented_scores[tr_idx], represented_scores[va_idx],
            representation=fusion_representation,
        )
        scaler = preprocessing["scaler"]
        input_scaler = preprocessing["input_scaler"]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        out = train_one_fold(
            X_tr, y_tr, X_va, y_va,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            hidden=cfg["hidden"],
            dropout=cfg["dropout"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            patience=cfg["patience"],
            fold=fold,
            log_to_wandb=log_to_wandb,
            model_kind="plain",
        )
        probs = out["val_probs"]
        oof_probs[va_idx] = probs

        fold_m = compute_binary_metrics_at_threshold(y_va, probs, 0.5)
        fold_metrics.append({
            "fold": fold,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "best_epoch": int(out["best_epoch"]),
            # True when the fold was too small for an honest inner split, so the epoch
            # count came from the fixed budget rather than validation-based selection.
            # Recorded per fold so an audit can tell which folds used the fallback.
            "used_fixed_epochs": bool(out.get("used_fixed_epochs", False)),
            "ap": _safe_metric(average_precision_score, y_va, probs),
            "auc": _safe_metric(roc_auc_score, y_va, probs),
            "brier": _safe_metric(brier_score_loss, y_va, np.clip(probs, 0.0, 1.0)),
            "f1_05": fold_m["f1"],
            "acc_05": fold_m["acc"],
            "balanced_acc_05": fold_m["balanced_acc"],
            "mcc_05": fold_m["mcc"],
            "precision_05": fold_m["precision"],
            "recall_05": fold_m["recall"],
        })
        histories.append(out["history"].assign(fold=fold))

        fold_artifacts.append({
            "fold": fold,
            "scaler": scaler,
            "state_dict": out["state_dict"],
            "hidden": cfg["hidden"],
            "dropout": cfg["dropout"],
            "model_kind": "plain",
            "representation": fusion_representation,
            "normalization": fusion_normalization,
            "input_scaler": input_scaler,
            "in_dim": int(X_tr.shape[1]),
        })

        if len(np.unique(y_tr)) < 2:
            baseline_probs = np.full(len(va_idx), float(np.mean(y_tr)), dtype=np.float32)
            baseline_model = None
        else:
            baseline_model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=int(cfg["seed"]),
            )
            baseline_model.fit(X_tr, y_tr)
            baseline_probs = baseline_model.predict_proba(X_va)[:, 1].astype(np.float32)

        baseline_oof_probs[va_idx] = baseline_probs
        base_m = compute_binary_metrics_at_threshold(y_va, baseline_probs, 0.5)
        baseline_fold_metrics.append({
            "fold": fold,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "ap": _safe_metric(average_precision_score, y_va, baseline_probs),
            "auc": _safe_metric(roc_auc_score, y_va, baseline_probs),
            "brier": _safe_metric(brier_score_loss, y_va, np.clip(baseline_probs, 0.0, 1.0)),
            "f1_05": base_m["f1"],
            "acc_05": base_m["acc"],
            "balanced_acc_05": base_m["balanced_acc"],
            "mcc_05": base_m["mcc"],
            "precision_05": base_m["precision"],
            "recall_05": base_m["recall"],
        })
        baseline_fold_artifacts.append({
            "fold": fold,
            "scaler": scaler,
            "model": baseline_model,
            "representation": fusion_representation,
            "normalization": fusion_normalization,
            "input_scaler": input_scaler,
        })
        if progress_callback is not None:
            progress_callback(
                fold, int(cfg["folds"]),
                {
                    "held_out_gene": f"stratified-fold-{fold}",
                    "status": "completed",
                    "best_epoch": int(out["best_epoch"]),
                },
            )

    y_true = y

    thr_grid = np.linspace(0.01, 0.99, 199)

    # Route through the single audited summarizer so the in-sample-threshold caveat
    # (best_t_is_insample) travels with these metrics instead of being re-derived here.
    _pack = summarize_probability_scores(y_true, oof_probs, thresholds=thr_grid)
    metrics = _pack["metrics"]
    threshold_sweep_df = _pack["threshold_sweep_df"]

    # Same single-summarizer routing for the logreg baseline, so both tracks carry the
    # identical threshold caveat and cannot drift apart.
    _bpack = summarize_probability_scores(y_true, baseline_oof_probs, thresholds=thr_grid)
    baseline_metrics = _bpack["metrics"]
    baseline_threshold_sweep_df = _bpack["threshold_sweep_df"]

    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    fold_metrics_df = pd.DataFrame(fold_metrics)
    baseline_fold_metrics_df = pd.DataFrame(baseline_fold_metrics)

    if log_to_wandb and WANDB_AVAILABLE and wandb is not None and wandb.run is not None:
        wandb.log(metrics)
        wandb.log({f"logreg_{k}": v for k, v in baseline_metrics.items()})
        if log_fold_tables and not fold_metrics_df.empty:
            wandb.log({"fold_metrics": wandb.Table(dataframe=fold_metrics_df)})
        if log_fold_tables and not baseline_fold_metrics_df.empty:
            wandb.log({"fold_metrics_logreg": wandb.Table(dataframe=baseline_fold_metrics_df)})

    return {
        "metrics": metrics,
        "oof_probs": oof_probs,
        "threshold_sweep_df": threshold_sweep_df,
        "fold_metrics_df": fold_metrics_df,
        "history_df": history_df,
        "fold_artifacts": fold_artifacts,
        "representation": fusion_representation,
        "normalization": fusion_normalization,
        "baseline": {
            "name": f"logreg_{fusion_representation}",
            "metrics": baseline_metrics,
            "oof_probs": baseline_oof_probs,
            "threshold_sweep_df": baseline_threshold_sweep_df,
            "fold_metrics_df": baseline_fold_metrics_df,
            "fold_artifacts": baseline_fold_artifacts,
        },
    }

def _normalize_artifact_scores(raw_features, artifact, genes):
    normalization = artifact.get("normalization", "original")
    raw = np.asarray(raw_features, dtype=np.float32)
    if normalization == "original":
        return raw
    if genes is None:
        raise ValueError(
            f"{normalization} prediction is transductive and requires a gene "
            "identifier for every row in the complete prediction batch"
        )
    return normalize_fusion_scores_np(raw, genes, normalization)


def ensemble_predict(raw_features, artifacts, genes=None):
    if len(raw_features) == 0:
        return np.array([], dtype=np.float32)
    fold_probs = []
    for art in artifacts:
        scaler = art["scaler"]
        if art.get("model_kind") == "plain":
            representation = art.get("representation", "engineered6")
            normalized = _normalize_artifact_scores(raw_features, art, genes)
            Xs = transform_fusion_features_np(
                normalized, representation,
                {"input_scaler": art.get("input_scaler"), "scaler": scaler},
            )
            model = PlainMLP(
                in_dim=int(art.get("in_dim", Xs.shape[1])),
                hidden=int(art["hidden"]), dropout=float(art["dropout"]),
            ).to(DEVICE)
        else:
            # Backward compatibility for artifacts written by the historical
            # standardize-raw-then-featurize ScoreFusionHead path.
            represented = np.asarray(raw_features, dtype=np.float32)
            model = ScoreFusionHead(
                hidden=int(art["hidden"]), dropout=float(art["dropout"]),
            ).to(DEVICE)
            Xs = scaler.transform(represented).astype(np.float32)
        model.load_state_dict(art["state_dict"])
        probs = predict_proba(model, Xs)
        fold_probs.append(probs)
    return np.mean(np.vstack(fold_probs), axis=0)

def ensemble_predict_logreg(raw_features, artifacts, genes=None):
    if len(raw_features) == 0:
        return np.array([], dtype=np.float32)

    fold_probs = []
    for art in artifacts:
        scaler = art.get("scaler")
        model = art.get("model")
        if scaler is None or model is None:
            continue
        normalized = _normalize_artifact_scores(raw_features, art, genes)
        Xs = transform_fusion_features_np(
            normalized, art.get("representation", "engineered6"),
            {"input_scaler": art.get("input_scaler"), "scaler": scaler},
        )
        probs = model.predict_proba(Xs)[:, 1].astype(np.float32)
        fold_probs.append(probs)

    if not fold_probs:
        return np.full(len(raw_features), np.nan, dtype=np.float32)
    return np.mean(np.vstack(fold_probs), axis=0)

def _run_audit_tag(cfg):
    size_val = cfg.get("d1_max_structures_per_variant", None)
    size_tag = "all" if size_val is None else f"n{int(size_val)}"
    seed_tag = f"seed{int(cfg.get('seed', 0))}"
    ratio_tag = str(cfg.get("d1_ratio_mode", "ratio_of_means"))
    policy_tag = sampling_policy_from_config(cfg).tag()
    parts = [seed_tag, size_tag, ratio_tag] + ([policy_tag] if policy_tag else [])
    return "_".join(parts)

def _normalize_for_csv(df):
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(lambda x: str(x) if isinstance(x, (dict, list, tuple, set)) else x)
    return out

def save_run_audit_tables(cfg, data_pack, cv_pack, metrics_dict, output_root=None):
    root = PROJECT_ROOT / "results" / "afm_d1_fusion_run_audits" if output_root is None else Path(output_root)
    run_dir = root / _run_audit_tag(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    table_map = {
        "plddt_source_scan": data_pack.get("plddt_source_scan"),
        "d1_runtime_audit": data_pack.get("d1_audit"),
        "d1_gene_audit": data_pack.get("d1_gene_audit"),
        "d1_variant_usage_audit": data_pack.get("d1_variant_usage_audit"),
        "variant_flow_audit": data_pack.get("variant_flow_audit"),
        "merge_overlap_audit": data_pack.get("overlap_audit"),
        "per_gene_label_audit": data_pack.get("per_gene_audit"),
        "fold_metrics_mlp": cv_pack.get("fold_metrics_df"),
        "fold_metrics_logreg": cv_pack.get("baseline", {}).get("fold_metrics_df"),
        "threshold_sweep_mlp": cv_pack.get("threshold_sweep_df"),
        "threshold_sweep_logreg": cv_pack.get("baseline", {}).get("threshold_sweep_df"),
    }

    saved = {"run_dir": str(run_dir)}
    for name, df in table_map.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            path = run_dir / f"{name}.csv"
            _normalize_for_csv(df).to_csv(path, index=False)
            saved[name] = str(path)

    metrics_path = run_dir / "metrics.csv"
    pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)
    saved["metrics"] = str(metrics_path)

    cfg_path = run_dir / "resolved_config.csv"
    pd.DataFrame([cfg]).to_csv(cfg_path, index=False)
    saved["resolved_config"] = str(cfg_path)

    return saved

def run_fusion_experiment(
    config_overrides=None,
    log_to_wandb=False,
    log_fold_tables=False,
    log_fold_figures=True,
    log_oof_summary=True,
):
    cfg = _resolve_run_config(config_overrides)
    data_pack = build_labeled_dataset_for_config(
        foxa1_hotspot_range=(cfg["foxa1_hotspot_start"], cfg["foxa1_hotspot_end"]),
        d1_max_structures_per_variant=cfg["d1_max_structures_per_variant"],
        d1_ratio_mode=cfg["d1_ratio_mode"],
        d1_seed=cfg["seed"],
        run_config=cfg,
        d1_sampling_policy=sampling_policy_from_config(cfg),
    )
    cv_pack = run_cross_validated_training(
        data_pack["labeled_df"],
        cfg,
        log_to_wandb=log_to_wandb,
        log_fold_tables=log_fold_tables,
        log_fold_figures=log_fold_figures,
        log_oof_summary=log_oof_summary,
    )

    out_metrics = dict(cv_pack["metrics"])

    baseline_metrics = dict(cv_pack.get("baseline", {}).get("metrics", {}))
    for key, value in baseline_metrics.items():
        out_metrics[f"logreg_{key}"] = value

    if "oof_ap" in baseline_metrics:
        out_metrics["delta_oof_ap_vs_logreg"] = float(out_metrics["oof_ap"] - baseline_metrics["oof_ap"])
    if "oof_auc" in baseline_metrics:
        out_metrics["delta_oof_auc_vs_logreg"] = float(out_metrics["oof_auc"] - baseline_metrics["oof_auc"])

    out_metrics.update({
        "foxa1_hotspot_start": cfg["foxa1_hotspot_start"],
        "foxa1_hotspot_end": cfg["foxa1_hotspot_end"],
        "d1_ratio_mode": cfg["d1_ratio_mode"],
        "d1_max_structures_per_variant": cfg["d1_max_structures_per_variant"],
        "n_labeled": data_pack["n_labeled"],
        "n_unknown": data_pack["n_unknown"],
    })

    audit_paths = save_run_audit_tables(cfg, data_pack, cv_pack, out_metrics)

    if log_to_wandb and WANDB_AVAILABLE and wandb is not None and wandb.run is not None:
        wandb.log(out_metrics)
        for table_name in ["plddt_source_scan", "d1_gene_audit", "variant_flow_audit", "overlap_audit", "per_gene_audit"]:
            df = data_pack.get(table_name)
            if isinstance(df, pd.DataFrame) and not df.empty:
                wandb.log({f"audit/{table_name}": wandb.Table(dataframe=df)})

    return {
        "config": cfg,
        "metrics": out_metrics,
        "data": data_pack,
        "cv": cv_pack,
        "audit_paths": audit_paths,
    }
