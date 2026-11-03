"""Single source of truth for the two directory anchors the pipeline relies on.

Before the package refactor every module recomputed these with a brittle
``Path(__file__).resolve().parents[N]`` whose depth changed the moment a file
moved. Import from here instead.

- ``ANALYSIS_DIR`` is the ``metric_analysis/`` directory: it holds the generated
  outputs (``structure_metrics/``, ``results/``), ``protein_configs.json`` and the
  notebooks.
- ``REPO_ROOT`` is its parent, the repo root, against
  which ``protein_configs.json`` resolves its relative paths (``Ground_Truths/``,
  ``PLDDTs/``, ...).

Historically the modules called the repo root ``PROJECT_ROOT``; import
``REPO_ROOT as PROJECT_ROOT`` to keep their downstream
``PROJECT_ROOT / "metric_analysis" / ...`` expressions unchanged.
"""

import os
import subprocess
from pathlib import Path

# pipeline/paths.py -> parents[0] = pipeline/, parents[1] = metric_analysis/
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_DIR.parent

MANUSCRIPT_DIRNAME = "SEPA-Structural-Ensemble-Perturbation-Analysis"


def manuscript_root(*, must_exist: bool = False) -> Path | None:
    """The manuscript repo, which is cloned BESIDE the main checkout.

    REPO_ROOT.parent is wrong in a git worktree: worktrees live under
    ensemble-driver-classifier.worktrees/, so a plain sibling lookup resolves
    to .worktrees/<manuscript repo> and misses. si_panels.py failed loudly on this
    (overnight run 2026-08-30); article_figures.py failed SILENTLY, defaulting
    --copy-to to None and skipping the panel copy while still reporting success.

    git's common dir always points at the MAIN checkout's .git, so its
    grandparent is the directory the manuscript sits in. Returns None when
    must_exist and nothing was found, so callers can skip rather than guess.
    """
    candidates = []
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        if common:
            candidates.append(Path(common).parent.parent / MANUSCRIPT_DIRNAME)
    except (OSError, subprocess.CalledProcessError):
        pass
    candidates.append(REPO_ROOT.parent / MANUSCRIPT_DIRNAME)
    for c in candidates:
        if c.is_dir():
            return c
    return None if must_exist else candidates[0]

# ---------------------------------------------------------------------------
# Run tag.
#
# Every generated results directory is named "<stem>_<tag>", e.g.
# canonical_v3_subunit, benchmark_evaluation_v3_subunit. The tag was previously
# written out as a literal in ~26 places across producers and consumers, so a
# new cohort meant editing all of them and any missed producer silently
# overwrote the PREVIOUS cohort's results under its own name. Derive the
# directory name from tagged() instead.
#
# "v2_strict"  the merged-MutS_alpha cohort (superseded 2026-09-07)
# "v3_subunit" MutS_alpha split into MSH2 + MSH6 (superseded 2026-10-12)
# "v4_msh6gt_pik3ca"  MSH6 from the G:T-mismatch rerun with its own WT (38 -> 143)
#              and PIK3CA added (91, annotation labels): total 3,643 -> 3,839.
#              Never generated; superseded by v5 before its rerun (2026-10-13).
# "v5_kras4b"  v4 plus KRAS switched from 4A (45 labelled) to the KRAS4B batch
#              (all 979 generated mutants; 861 labelled by the same DMS cutoffs).
#              Its results exist only once regenerated.
#
# Override for one invocation without editing code:
#     DRIFT_RUN_TAG=v2_strict python -m drift.canonical_afm
# which is how the merged layout is regenerated, paired with the enabled flags
# in the two protein_configs.json files.
# ---------------------------------------------------------------------------
RUN_TAG = os.environ.get("DRIFT_RUN_TAG", "v6_ca")
PREVIOUS_RUN_TAG = "v3_subunit"


def tagged(stem: str, *, tag: str | None = None) -> str:
    """Directory name for a results stem under the active run tag.

    Pass ``tag`` only to reach a specific historical cohort on purpose, e.g.
    ``tagged("canonical", tag=PREVIOUS_RUN_TAG)`` for an audit trail.
    """
    return f"{stem}_{tag or RUN_TAG}"


# ---------------------------------------------------------------------------
# Raw AF3 archives. Cache indexes (``*_structure_file_index.csv``) record each
# structure's path on the drive the caches were built from; ``on_archive_root``
# moves such a path onto DRIFT_ARCHIVE_ROOT, the folder holding the archives here.
# ---------------------------------------------------------------------------
RECORDED_ARCHIVE_ROOT = "/Volumes/Elements"


def on_archive_root(path) -> Path:
    """A recorded drive path re-rooted onto $DRIFT_ARCHIVE_ROOT (unchanged if unset)."""
    text, root = str(path), os.environ.get("DRIFT_ARCHIVE_ROOT")
    if root and text.startswith(RECORDED_ARCHIVE_ROOT + "/"):
        return Path(root) / text[len(RECORDED_ARCHIVE_ROOT) + 1:]
    return Path(text)


__all__ = ["ANALYSIS_DIR", "REPO_ROOT", "RUN_TAG", "PREVIOUS_RUN_TAG", "tagged",
           "RECORDED_ARCHIVE_ROOT", "on_archive_root"]
