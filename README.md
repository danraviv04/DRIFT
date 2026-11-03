# DRIFT

**D**river **R**ecognition from **I**nduced **F**luctuations in **T**hree-dimensional ensembles.

Code for *Predicted structural ensembles of proteins can be more informative than the best
single prediction* (Raviv, Dayan et al., in submission). DRIFT predicts an AlphaFold 3
ensemble for every variant and for the wild type (25 seeds × 5 samples), and scores a
variant by how far its per-residue pLDDT distributions move from the wild type's
(the KS-score). This repository rebuilds every figure and number in the paper from the
processed AlphaFold 3 data.

## Contents

| folder | what |
|---|---|
| `af3_jobs/` | AlphaFold 3 job JSONs from one config per construct (`python -m af3_jobs`), seed extensions, WT jobs that see their variants' templates; runners for a Slurm cluster (`slurm/`) and for a machine without Slurm (`local_runner/`) |
| `drift/` | the method: structure draws (`structure_draws.py`), descriptors, score caches, evaluation |
| `metric_analysis/` | pLDDT extraction, coordinate caches (US-align and WT rigid-core frames), labels, the canonical cohort |
| `analyses/` | one folder per analysis behind a figure, table or number (map below), grouped as `cohort/`, `annotation/` (the driver/passenger cohort), `continuous/` (assay genes), `msh6_case_study/` (WT backgrounds, residue nulls), `coordinates/`, `seeds/` and `provenance/` (checks behind Notes S5 and S6 that read the raw archives) |
| `figures/` | the code for each figure (`figure1/`, `figure2/`, `si/`) and the assembler (`make_figures.py`); the hand-made Figure 1 inputs (the BioRender graphical abstract, the ChimeraX render of Fig. 1b) go in `figures/source/` from the data, and a missing panel is drawn as a labelled placeholder |
| `run_all.sh` | rebuilds everything, in order |

## Setup

```bash
conda env create -f environment.yml      # Python 3.11, pinned versions; includes USalign
conda activate drift
pip install -e .
```

No GPU is needed for anything below; AlphaFold 3 itself is only needed to regenerate the
structures (`af3_jobs/`).

## Data

The processed data (about 12 GB, deposited separately; see the paper's Data availability)
unpack into this folder's layout:

- **`PLDDTs/`:** per-residue pLDDT matrices, one CSV per variant and WT, with the 125 predictions as columns.
- **`Ground_Truths/`:** the labels. ClinVar exports, pinned by SHA-256 in `metric_analysis/manifests/clinvar_source_manifest.csv`, and the POLE driver/passenger table (`POLE/combined_pole_labels.csv`, compiled from the POLE/POLD1 driver classification of Dayan et al., Sci. Rep. 2026, doi:10.1038/s41598-026-62150-w; the version used for this study). Also the four assay tables (TP53, KRAS, PTEN, MSH2) and the AlphaMissense scores.
- **`metric_analysis/structure_metrics/<dataset>/ensemble_aware/cache/`:** C-alpha coordinates of every prediction, superposed with US-align.
- **`metric_analysis/manifests/`:** the frozen structural inventory and the strict canonical manifest.
- **A few analysis inputs**, each named where it is read:
  - the 375-prediction MSH6 WT pool;
  - AF3 ranking scores;
  - WT seed maps;
  - KRAS4B batch files;
  - the PTEN WT seed extension.
  - the earlier MSH2 WT pLDDT matrix (`PLDDTs/MutS_alpha/legacy/`, backbone N atoms), one of Table S6's comparisons.

The raw AlphaFold 3 output archives are only needed to redo the extraction
(`run_all.sh extract`) or the Fig. 1b render; point `DRIFT_ARCHIVE_ROOT` at them.

## Running

```bash
bash run_all.sh all            # descriptors -> evaluation -> figures (a few hours on a laptop)
bash run_all.sh list           # every phase and stage
DRY_RUN=1 bash run_all.sh all  # print the commands only
bash run_all.sh S_curves       # one stage
```

| phase | what |
|---|---|
| `extract` (opt-in) | raw archives → pLDDT matrices, coordinate caches, WT pool, ranking scores, seed maps (needs `DRIFT_ARCHIVE_ROOT`) |
| `descriptors` | cohort summaries → strict manifest (1,798 variants) → residue-KS route, score caches (N = 1–90 × seeds 42–51 in `drift/cache`), WT rigid-core coordinate caches and their ensemble-size sweeps |
| `evaluation` | annotation cohort (`analyses/annotation/stages/run_stage.py`), continuous route, then the standalone analyses |
| `figures` | Figure 1 panels and assembly, Figure 2, SI Figs. S1–S6, copied to `figures/` |
| `F_fig1b` (opt-in) | Fig. 1b from the AF3 CIFs; needs ChimeraX (`CHIMERAX`) |

Logs go to `logs/<stage>.log` and `logs/STATUS.log`; the first failing stage stops the run.
`run_all.sh` refuses to start unless `drift/protein_configs.json` has the published
fingerprint (`97f1cdbfda8e`): structure draws are keyed by variant, size and seed, so the
same config gives the same draws on any machine.

## Figures and numbers → code

Stage names are `run_all.sh`'s; `S_<x>` is a stage of `analyses/annotation/stages/run_stage.py`.

| item | script | stage |
|---|---|---|
| Fig. 1a | BioRender, `figures/source/graphical_abstract_2026-10-25.pdf`; its box plot `figures/figure1/ga_boxplot/` | `F1a_ga_boxplot` |
| Fig. 1b | `figures/figure1/fig1b/` (export → RMSF → ChimeraX render → compose) | `F_fig1b` |
| Fig. 1c,d and their text | `analyses/coordinates/rmsf_histogram/exemplar_wt_vs_mut.py` | `F1cd_panels` |
| Fig. 1e–h | `figures/figure1/layout/rerender_efgh.py` (e: `figures/si/lib/ks_sequence_panel.py`; f: `analyses/msh6_case_study/wt_background_2000/`; g: `figures/figure1/ks_combo/`; h: Figure 2's per-gene table) | `F1efgh_panels` |
| Fig. 1e text, Note S2 residue null | `analyses/msh6_case_study/residue_null/`, `analyses/msh6_case_study/wt_background_2000/residue_null_2000.py`, `analyses/msh6_case_study/fig1e_near_pair/` | `N3`–`N6` |
| Fig. 1f and its counts | `analyses/msh6_case_study/wt_backgrounds_200/` (200 splits), `analyses/msh6_case_study/wt_background_2000/` (2,000) | `N1`, `N2` |
| Figure 1 assembly | `figures/make_figures.py Figure1` | `F1_assemble` |
| Fig. 2a,b; KS AUCs | `figures/figure2/make_figure2.py` (per-gene curves) | `S_curves`, `S_figure2` |
| Fig. 2c–f; KS vs assays | `analyses/continuous/continuous_functional/` | `G1`, `G6` |
| Fig. 2g | `analyses/annotation/simple_fusion_benchmark/size_sweep.py` | `S_size` |
| Fig. 2h,i; combination numbers, Table S7 | `analyses/annotation/simple_fusion_benchmark/annotation_tables.py`, `analyses/annotation/am_operating_point/` | `S_tables`, `S_am_operating_point` |
| Main-text CIs | `analyses/annotation/position_cluster_bootstrap/run.py`, `analyses/annotation/ks_auc_ci/` | `S_ks_auc_ci`, `S_position_cluster` |
| AF3 top-1 vs KS (Note S3, Table S3) | `analyses/seeds/af3_top1_vs_ks/` | `N7` |
| ρ(KS, AM), Fig. S5 | `figures/si/lib/ks_vs_am.py`, `figures/si/S5/` | `S_si_ks_vs_am`, `FS5` |
| Figs. S1, S2; Tables S1, S2; Note S1 | `analyses/coordinates/s1_matched_draws/`, `figures/si/lib/shift_vs_rmsf_P286R.py`, `figures/si/lib/coordinate_vs_plddt.py`, `analyses/coordinates/coord_core_fit/validate.py`, `analyses/coordinates/s1_frame_sensitivity/`, `figures/si/S1_S2/` | `S_s1_matched`, `S_s1a`, `K3`, `K4`, `N15`, `FS1_S2` |
| Figs. S3, S4; Note S2 table | `analyses/annotation/wt_split_12v12/`, `figures/si/S3_S4/` | `S_wt12v12`, `FS3_S4` |
| Fig. S6, Note S7 | `analyses/seeds/seed_structure/` (`plateau_run.py` → `plateau_summary.py` → `three_arm_n25.py`) | `N8`–`N10` |
| Table S6 | `analyses/seeds/seed_structure/seed_icc.py`, `analyses/seeds/seed_pairing/`, `analyses/seeds/seed_coupling_ks/` | `N11`–`N13` |
| Note S4 (KRAS) | `analyses/continuous/kras_score_by_assay_bin/` (scores from `analyses/continuous/kras4b_coverage/build_ks.py`) | `N16` (`X5`) |
| Table S4b, TP53 SNVs | `analyses/continuous/tp53_snv_reachability/` | `N17` |
| Note S5 | ClinVar checksums `metric_analysis/manifests/verify_clinvar_manifest.py` (`L0`); transcripts `analyses/provenance/clinvar_transcripts/`; KRAS4B batch design `analyses/continuous/kras4b_coverage/` | |
| Note S6 | AF3 versions `analyses/seeds/tp53_af3_version_check/` (`N18`); templates `analyses/provenance/wt_variant_offset/`; MSA swap `analyses/provenance/wt_msa_check/` (both read the raw archives) | |
| Cohort counts | `analyses/cohort/build_strict_canonical_manifest.py`, `analyses/continuous/continuous_functional/dataset.py` | `L1`, `G1` |

## Known leftovers

- **D1**, an earlier descriptor (mean pLDDT shift), is still computed alongside the KS-score in the score caches and in some comparison tables (AM+D1); no paper result uses it.
- **The FOXA1 hotspot option.** `drift/core.py` still has a residue-interval label option named after FOXA1, the positional control that was removed from this release.
- **Unused functions.** Parts of `drift/` (the learned-fusion models) and of `metric_analysis/pipeline/library/structure_metrics.py` serve no paper result. They will be removed together with D1 in a later revision.
- **Table S6, POLE row.** It was computed before POLE's eight 32-seed drivers were cut to 25 seeds and P356S left the cohort, which changes which POLE variants are sampled. Today's code gives a same-seed r of 0.73 (paper 0.74) and a KS change of −0.006 (paper −0.003); the other five genes reproduce exactly.
- **The tests need the data:** `PYTHONPATH=. pytest drift/tests metric_analysis/tests af3_jobs/tests`.

## Citation

Raviv D, Dayan D, et al. Predicted structural ensembles of proteins can be more
informative than the best single prediction. In submission.
