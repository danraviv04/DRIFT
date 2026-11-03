# DRIFT

**D**river **R**ecognition from **I**nduced **F**luctuations in **T**hree dimenstional structures.

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
| `experiments/` | one folder per analysis behind a figure, table or number (map below) |
| `figures/` | the figure assembler (`make_figures.py`); its hand-made Figure 1 inputs (the BioRender graphical abstract, the ChimeraX render of Fig. 1b) go in `figures/source/` from the data, and a missing panel is drawn as a labelled placeholder |
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
| `descriptors` | cohort summaries → strict manifest (1,798 variants) → residue-KS route, score caches (N = 1–90 × seeds 42–51 in `drift/cache`), WT rigid-core coordinate caches |
| `evaluation` | annotation cohort (`experiments/annotation_stages/run_stage.py`), continuous route, then the standalone analyses |
| `figures` | Figure 1 panels and assembly, Figure 2, SI Figs. S1–S6, copied to `figures/` |
| `F_fig1b` (opt-in) | Fig. 1b from the AF3 CIFs; needs ChimeraX (`CHIMERAX`) |

Logs go to `logs/<stage>.log` and `logs/STATUS.log`; the first failing stage stops the run.
`run_all.sh` refuses to start unless `drift/protein_configs.json` has the published
fingerprint (`97f1cdbfda8e`): structure draws are keyed by variant, size and seed, so the
same config gives the same draws on any machine.

## Figures and numbers → code

Stage names are `run_all.sh`'s; `S_<x>` is a stage of `experiments/annotation_stages/run_stage.py`.

| item | script | stage |
|---|---|---|
| Fig. 1a | BioRender, `figures/source/graphical_abstract_2026-10-25.pdf`; its box plot `experiments/ga_ks_boxplot/` | `F1a_ga_boxplot` |
| Fig. 1b | `experiments/fig1b_corefit_models/` (export → RMSF → ChimeraX render → compose) | `F_fig1b` |
| Fig. 1c,d and their text | `experiments/rmsf_histogram/exemplar_wt_vs_mut.py` | `F1cd_panels` |
| Fig. 1e–h | `experiments/figure1_layout/rerender_efgh.py` (e: `si_figures/ks_sequence_panel.py`; f: `wt_background_2000/`; g: `figure1_ks_combo/`; h: Figure 2's per-gene table) | `F1efgh_panels` |
| Fig. 1e text, Note S2 residue null | `experiments/residue_null/`, `wt_background_2000/residue_null_2000.py`, `fig1e_near_pair/` | `N3`–`N6` |
| Fig. 1f and its counts | `experiments/pten_wt_null_panel/` (200 splits), `wt_background_2000/` (2,000) | `N1`, `N2` |
| Figure 1 assembly | `figures/make_figures.py Figure1` | `F1_assemble` |
| Fig. 2a,b; KS AUCs | `experiments/figure2/make_figure2.py` (per-gene curves) | `S_curves`, `S_figure2` |
| Fig. 2c–f; KS vs assays | `experiments/continuous_functional/` | `G1`, `G6` |
| Fig. 2g | `experiments/simple_fusion_benchmark/size_sweep.py` | `S_size` |
| Fig. 2h,i; combination numbers, Table S7 | `simple_fusion_benchmark/annotation_tables.py`, `am_operating_point/` | `S_tables`, `S_am_operating_point` |
| Main-text CIs | `experiments/position_cluster_bootstrap/run.py`, `ks_auc_ci/` | `S_ks_auc_ci`, `S_position_cluster` |
| AF3 top-1 vs KS (Note S3, Table S3) | `experiments/af3_top1_vs_ks/` | `N7` |
| ρ(KS, AM), Fig. S5 | `experiments/si_figures/ks_vs_am.py`, `si_ks_vs_am_figure/` | `S_si_ks_vs_am`, `FS5` |
| Figs. S1, S2; Tables S1, S2; Note S1 | `s1_matched_draws/`, `si_figures/shift_vs_rmsf_P286R.py`, `si_figures/coordinate_vs_plddt.py`, `coord_core_fit/validate.py`, `s1_frame_sensitivity/`, `si_coordinate_figures/` | `S_s1_matched`, `S_s1a`, `K3`, `N15`, `FS1_S2` |
| Figs. S3, S4; Note S2 table | `wt_split_12v12/`, `si_note_s2_figures/` | `S_wt12v12`, `FS3_S4` |
| Fig. S6, Note S7 | `experiments/seed_structure/` (plateau → `three_arm_n25.py`) | `N8`–`N10` |
| Table S6 | `seed_structure/seed_icc.py`, `wt_variant_offset/seed_pairing.py`, `seed_coupling_ks/` | `N11`–`N13` |
| Note S4 (KRAS) | `experiments/kras_score_by_assay_bin/` (scores from `kras4b_coverage/build_ks.py`) | `N16` (`X5`) |
| Table S4b, TP53 SNVs | `experiments/tp53_snv_reachability/` | `N17` |
| Note S5 | ClinVar checksums `metric_analysis/manifests/verify_clinvar_manifest.py` (`L0`); transcripts `experiments/clinvar_transcripts/`; KRAS4B batch design `kras4b_coverage/` | |
| Note S6 | AF3 versions `tp53_af3_version_check/` (`N18`); templates `wt_variant_offset/`; MSA swap `wt_msa_check/` (both read the raw archives) | |
| Cohort counts | `experiments/build_strict_canonical_manifest.py`, `continuous_functional/dataset.py` | `L1`, `G1` |

## Known leftovers

- **D1**, an earlier descriptor (mean pLDDT shift), is still computed alongside the KS-score in the score caches and in some comparison tables (AM+D1); no paper result uses it.
- **The FOXA1 hotspot option.** `drift/core.py` still has a residue-interval label option named after FOXA1, the positional control that was removed from this release.
- **Unused functions.** Parts of `drift/` (the learned-fusion models) and of `metric_analysis/pipeline/library/structure_metrics.py` serve no paper result. They will be removed together with D1 in a later revision.
- **The tests need the data:** `PYTHONPATH=. pytest drift/tests metric_analysis/tests af3_jobs/tests`.

## Citation

Raviv D, Dayan D, et al. Predicted structural ensembles of proteins can be more
informative than the best single prediction. In submission.
