"""Run one stage of the POLE 25-seed rerun with isolated inputs and outputs.

Only module path constants are redirected: model grids, samplers, fitters, threshold
policies and bootstrap definitions stay in the existing modules. Inputs: the score caches
in drift/cache (fingerprint 97f1cdbfda8e, seeds 42-51, written by
`python -m drift.run --sweep`, see run_all.sh). Outputs: publication/ (the stages behind
Figure 2 and the main-text numbers) and side/ (the SI analyses).

    PYTHONPATH="." python experiments/annotation_stages/run_stage.py <stage>

Release copy: only the stages whose outputs the paper uses are kept. The published run
read caches patched from the 10-18 ones (recompute_cache.py, not released); here they are built
directly by the sweep under the same config.
"""
from pathlib import Path
import hashlib
import importlib
import importlib.util
import json
import os
import runpy
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "publication"
SIDE = HERE / "side"
CACHE = ROOT / "drift/cache"
FP = "97f1cdbfda8e"  # the published run: 49028c16cbb3 (with FOXA1, which left on 2026-11-03)
SIZES, SEEDS = (1, 5, 10, 20, 40, 60, 90), range(42, 52)
TAG = "v6_pole25_20261020"
os.environ["DRIFT_RUN_TAG"] = TAG
os.environ["DRIFT_DEVICE"] = "cpu"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".xdg_cache"))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
sys.path[:0] = [str(ROOT)]

PUBLICATION_STAGES = [
    "size", "tables", "am_operating_point", "curves",
    "si_coordinate_vs_plddt", "si_ks_vs_am", "figure2",
]
SIDE_STAGES = ["ks_auc_ci", "wt12v12", "position_cluster", "s8_n90", "s1_matched",
               "fig1cd", "s1a"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invoke(module, args):
    sys.argv = [str(Path(module.__file__).resolve()), *[str(a) for a in args]]
    result = module.main()
    if result not in (None, 0):
        raise RuntimeError(f"{module.__name__} returned {result}")


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_inputs():
    from drift.protein_config import protein_config_fingerprint
    from drift import ensemble_shift
    assert protein_config_fingerprint() == FP, "scientific config changed; revalidate before running"
    assert Path(ensemble_shift.__file__).resolve().is_relative_to(ROOT), \
        f"imported {ensemble_shift.__file__}, not this checkout (PYTHONPATH must be '.')"
    missing = [f"size{n} seed{s}" for n in SIZES for s in SEEDS
               if not (CACHE / f"labeled_size{n}_seed{s}_ratio_of_means_proteins{FP}_esfeat6.csv").exists()]
    assert not missing, (f"{len(missing)} of {len(SIZES) * len(SEEDS)} score caches missing under {CACHE}, "
                         f"e.g. {missing[0]}; build them with `bash run_all.sh B1_cache`")


def run_cohort_file():
    """S8/S1 annotation cohort with the eight rows pointed at their 25-seed sources.

    experiments/s8_representatives/annotation_cohort.csv is the
    10-19 record (eight rows on the 160-prediction files). The copy here takes source
    path and count of every row from the strict manifest, after checking that labels and
    every other row's source basename agree with it.
    """
    import pandas as pd
    repr_dir = ROOT / "experiments/s8_representatives"
    cohort = pd.read_csv(repr_dir / "annotation_cohort.csv")
    manifest = pd.read_csv(ROOT / "metric_analysis/manifests/strict_canonical_manifest.csv")
    m = cohort.merge(manifest[["protein", "mutation_code", "label", "source_file", "n_mutant_structures"]],
                     on=["protein", "mutation_code"], how="left", suffixes=("", "_manifest"),
                     validate="one_to_one")
    # POLE P356S left the cohort on 2026-10-20 (no coordinates); it is the only row the
    # 10-19 record has and the manifest lacks. exclusions.csv still names it, harmlessly.
    gone = m[m.label_manifest.isna()]
    assert list(zip(gone.protein, gone.mutation_code)) == [("POLE", "P356S")], gone[["protein", "mutation_code"]]
    m = m[m.label_manifest.notna()].reset_index(drop=True)
    cohort = cohort.merge(m[["protein", "mutation_code"]], on=["protein", "mutation_code"]).reset_index(drop=True)
    assert len(cohort) == len(m) == 1798 and (m.label == m.label_manifest).all()
    moved = m.source_file != m.source_file_manifest
    assert set(m.loc[moved, "mutation_code"]) == {"D275A", "D275G", "E277G", "P286H", "P286L",
                                                  "P286R", "P286S", "T278K"}, m.loc[moved, "mutation_code"]
    assert (m.loc[moved, "protein"] == "POLE").all()
    assert all(Path(a).name == Path(b).name for a, b in zip(m.source_file, m.source_file_manifest))
    cohort["source_file"] = m.source_file_manifest
    cohort["n_mutant_structures"] = m.n_mutant_structures_manifest
    path = SIDE / "annotation_cohort_25seed.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(path, index=False)
    return path, repr_dir / "exclusions.csv"


def publication_stage(stage, pd):
    from drift import run
    from experiments.simple_fusion_benchmark import size_sweep
    run.CACHE_DIR = CACHE
    size_sweep.CACHE = CACHE
    if stage == "size":
        invoke(size_sweep, ["--gt-group", "annotation", "--out", OUT / "size_sweep"])
    elif stage == "tables":
        from experiments.simple_fusion_benchmark import annotation_tables
        invoke(annotation_tables, ["--gt-group", "annotation", "--size", 90, "--draws", 2000,
                                   "--out", OUT / "annotation_tables"])
    elif stage == "am_operating_point":
        from experiments.am_operating_point import am_anchored
        am_anchored.OUT = OUT / "am_operating_point"
        am_anchored.load_frames = lambda: [size_sweep.load_cell(90, s, "annotation")[0] for s in range(42, 52)]
        invoke(am_anchored, [])
    elif stage == "curves":
        from experiments.figure2 import make_figure2
        make_figure2.CACHE = CACHE
        destination = OUT / "figure2"
        destination.mkdir(exist_ok=True)
        curves = make_figure2.per_gene_curves(make_figure2.annotation_genes())
        assert len(curves) == 420 and set(curves.N) == {1, 5, 10, 20, 40, 60, 90}
        curves.to_csv(destination / "panel_abcd_per_gene_auc.csv", index=False)
    elif stage.startswith("si_"):
        sys.path.insert(0, str(ROOT / "experiments/si_figures"))
        import figlib
        figlib.FUSION_CACHE = CACHE
        figlib.RESULTS = OUT / "si_figures"
        figlib.RESULTS.mkdir(exist_ok=True)
        name = stage.removeprefix("si_")
        module = importlib.import_module(name)
        invoke(module, [])
        metadata = figlib.RESULTS / "run_metadata.json"
        if metadata.exists():
            metadata.rename(figlib.RESULTS / f"run_metadata_{name}.json")
    elif stage == "figure2":
        from experiments.figure2 import make_figure2
        make_figure2.CACHE = CACHE
        make_figure2.AM_ANCHORED = OUT / "am_operating_point/am_anchored_operating_point.csv"
        invoke(make_figure2, ["--gz", OUT / "size_sweep", "--tables", OUT / "annotation_tables",
                              "--out", OUT / "figure2"])
        metadata = OUT / "figure2/panel_cdef_provenance.json"
        data = json.loads(metadata.read_text())
        data["continuous_inputs_reused"] = True
        data["continuous_source_sha256"] = sha(ROOT / data["source"])
        data["continuous_source_provenance"] = json.loads((ROOT / data["provenance"]).read_text())
        metadata.write_text(json.dumps(data, indent=2) + "\n")
    else:
        raise ValueError(f"Unknown publication stage: {stage}")


def side_stage(stage, pd):
    SIDE.mkdir(exist_ok=True)
    if stage == "ks_auc_ci":
        # Main-text hierarchical KS AUC CIs (ks_auc_ci/ks_auc_ci.py). The
        # published run also scored the 10-18 caches as a reproduction baseline.
        out = SIDE / "ks_auc_ci"
        out.mkdir(exist_ok=True)
        module = load_path("ks_auc_ci_pole25_cache", ROOT / "experiments/ks_auc_ci/ks_auc_ci.py")
        module.CACHE, module.FP, module.OUT = CACHE, FP, out / "ks_auc_ci_pole25_cache.csv"
        module.main()
    elif stage == "wt12v12":
        os.environ["WT12_OUT"] = str(SIDE / "wt_split_12v12")
        os.environ["WT12_CACHE"] = str(CACHE)
        runpy.run_path(str(ROOT / "experiments/wt_split_12v12/run_12v12_2000.py"), run_name="__main__")
    elif stage == "position_cluster":
        out = SIDE / "position_cluster_bootstrap"
        out.mkdir(exist_ok=True)
        os.environ.update(PCB_REPAIR=str(HERE), PCB_CACHE=str(CACHE), PCB_FP=FP,
                          PCB_PUBLISHED_KS=str(SIDE / "ks_auc_ci/ks_auc_ci_pole25_cache.csv"),
                          PCB_STATUS=str(out / "STATUS.log"))
        module = load_path("position_cluster_run", ROOT / "experiments/position_cluster_bootstrap/run.py")
        assert module.CACHE == CACHE and module.FP == FP
        sys.argv = ["run.py", "--out", str(out)]
        module.main()
        check = pd.read_csv(out / "check_ks_reproduction.csv")
        assert (check.max_abs_diff < 1e-12).all(), "KS variant scheme must reproduce ks_auc_ci for every gene"
    elif stage == "s8_n90":
        cohort, exclusions = run_cohort_file()
        repr_dir = ROOT / "experiments/s8_representatives"
        sys.path.insert(0, str(repr_dir))
        import n90_comparison as n90
        n90.COHORT, n90.EXCLUSIONS, n90.CACHE, n90.FP = cohort, exclusions, CACHE, FP
        n90.OUT = SIDE / "s8_n90_results"
        n90.main()
    elif stage == "s1_matched":
        cohort, exclusions = run_cohort_file()
        module = load_path("s1_matched_draws_run", ROOT / "experiments/s1_matched_draws/s1_matched_draws.py")
        module.COHORT, module.EXCLUSIONS, module.CACHE, module.FP = cohort, exclusions, CACHE, FP
        module.OUT = SIDE / "s1_matched_draws"
        module.SCORES = module.OUT / "variant_seed_scores.csv.gz"
        module.S9_AUC = SIDE / "s8_n90_results/auc_summary.csv"
        sys.argv = ["s1_matched_draws.py", "--recompute"]
        module.main()
    elif stage == "fig1cd":
        # Fig. 1c,d panels exactly as the manuscript's (POLE_*_row.pdf; Fig. 1d is RMSF since
        # 2026-10-18), on seeds 1-25; and the all-seeds version to reproduce the old numbers.
        args = ["--mutant", "P286R", "--metric", "rmsf", "--panels", "bc", "--panel-width-in", "4.86",
                "--bc-heights", "1.8,1.8", "--shift-xaxis", "--stem-suffix", "_row"]
        script = str(ROOT / "experiments/rmsf_histogram/exemplar_wt_vs_mut.py")
        sys.path.insert(0, str(Path(script).parent))   # it imports its sibling rmsf_histogram
        # The all-seeds check needs the pre-25-seed table (rmsf_histogram/exemplar_P286R_rmsf.csv,
        # superseded data); it is skipped when that file is absent.
        published = ROOT / "experiments/rmsf_histogram/exemplar_P286R_rmsf.csv"
        runs = [(SIDE / "fig1cd", [])] + ([(SIDE / "fig1cd_all_seeds", ["--all-seeds"])] if published.exists() else [])
        for out, extra in runs:
            sys.argv = [script, *args, *extra, "--out-dir", str(out)]
            runpy.run_path(script, run_name="__main__")
        new = pd.read_csv(SIDE / "fig1cd/exemplar_P286R_rmsf.csv")
        assert len(new) == 204
        if published.exists():
            old = pd.read_csv(SIDE / "fig1cd_all_seeds/exemplar_P286R_rmsf.csv")
            published = pd.read_csv(published)
            reproduced = float((old[["wt", "mut", "centre_shift"]] - published[["wt", "mut", "centre_shift"]]).abs().max().max())
            print(f"all-seeds rerun vs the published Fig. 1c,d table: max |diff| {reproduced:.2e}")
            assert reproduced < 1e-9, "the all-seeds rerun does not reproduce the published Fig. 1c,d table"
        else:
            print("skipped: all-seeds check (experiments/rmsf_histogram/exemplar_P286R_rmsf.csv absent)")
    elif stage == "s1a":
        sys.path.insert(0, str(ROOT / "experiments/si_figures"))
        import figlib
        figlib.RESULTS = SIDE / "s1a"
        module = load_path("shift_vs_rmsf_run", ROOT / "experiments/si_figures/shift_vs_rmsf_P286R.py")
        module.OUT = SIDE / "s1a"
        module.FIG1_TABLE = SIDE / "fig1cd/exemplar_P286R_rmsf.csv"
        invoke(module, [])
    else:
        raise ValueError(f"Unknown side stage: {stage}")


def main():
    stage = sys.argv[1]
    if stage not in PUBLICATION_STAGES + SIDE_STAGES:
        raise SystemExit(f"unknown stage {stage!r}; stages: {' '.join(PUBLICATION_STAGES + SIDE_STAGES)}")
    OUT.mkdir(exist_ok=True)
    (OUT / "stages").mkdir(exist_ok=True)
    check_inputs()
    import pandas as pd
    stamp = dict(stage=stage, status="running", run_tag=TAG, fingerprint=FP,
                 cache_directory=str(CACHE),
                 launcher_sha256=sha(__file__), started_utc=datetime.now(timezone.utc).isoformat(),
                 path_policy="Existing analyses, redirected to isolated 25-seed inputs and outputs; scientific defaults retained.")
    state_path = OUT / "stages" / f"{stage}.json"
    state_path.write_text(json.dumps(stamp, indent=2) + "\n")
    try:
        if stage in PUBLICATION_STAGES:
            publication_stage(stage, pd)
        elif stage in SIDE_STAGES:
            side_stage(stage, pd)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        stamp["status"] = "complete"
    except BaseException as exc:
        stamp.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        stamp["finished_utc"] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(stamp, indent=2) + "\n")


if __name__ == "__main__":
    main()
