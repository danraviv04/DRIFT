# POLE on 25 AF3 seeds everywhere (2026-10-20)

**Status: set up and smoke-tested; the overnight run has NOT been launched.**

## Why

Every ensemble should be 25 AF3 seeds × 5 samples = 125 predictions, so that Supplementary Table S2 (prediction counts per system) is unnecessary. Eight POLE drivers were predicted with seeds 1–32 (160 predictions): D275A, D275G, E277G, P286H, P286L, P286R, P286S, T278K. The WT `POLE_268-471_WT` and every other POLE variant have seeds 1–25. Seeds 26–32 have no WT counterpart.

This run keeps seeds 1–25 of the eight, which are exactly the WT's seed numbers, and drops seeds 26–32. The pre-check expected small changes: KS at N=90 moves between −0.017 and +0.006, and POLE AUC by about 0.001.

## POLE P356S leaves the cohort (decided 2026-10-20, same branch)

Dan decided to drop POLE P356S, a passenger with no coordinate structures, from the whole canonical cohort, so POLE is rerun only once.

- **New counts:** POLE is 1,074 variants (38 drivers / 1,036 passengers). The annotation cohort is 1,798 (612 / 1,186). The strict manifest has 2,619 rows.
- **Where it is recorded:** the row moved from `strict_canonical_manifest.csv` to `strict_canonical_exclusions.csv`, with reason `user_decision_2026-10-20_no_coordinate_structures_one_cohort_for_all_descriptors`.
- **Manifest builder:** `build_strict_canonical_manifest.py` has an `EXCLUDED_BY_DECISION` rule, expects POLE 1,074, 2,619 rows and 16 exclusions, and reproduces the edited files. Two differences remain:
  - Run from the worktree, it writes `source_exists=False` for the eight 25-seed paths, because those paths point into the main checkout.
  - PIK3CA rows already differed between the builder and the committed manifest/inventory before this branch. That drift is unrelated to this change.
- **New caches:** every cache drops P356S, so each holds exactly the manifest's variants. `check_cache` compares against the 10-18 cache less P356S and fails if P356S is present.
- **Count guards updated:** `figlib.EXPECTED_COHORT`, the KS-125 extension, `check_representatives` and preflight.
- **Unchanged guards:** the S1/S8 exclusion set {P356S, G162R} still holds. The 1,797 matched-draw cohort is unchanged: P356S was already out of it.
- **The six curated drivers that were never modelled are not modelled now (option b):** D22G, R21M, E830G, E978G and L1096F lie outside residues 268–471, and V411R needs two nucleotide changes. The SI note has to state this.
- **Reading the before/after tables:** POLE rows (and macros) mix two changes, the 25-seed restriction and the P356S removal.

## Binary intervals: position clusters are the published unit (decided 2026-10-20)

Dan decided that every binary interval in the main text resamples **residue positions**: all substitutions at one residue form one cluster, as the continuous endpoints already do. This covers:

- the per-gene KS-score AUC CIs (hierarchical, rng 20261015; clusters stratified as driver-only, passenger-only or mixed);
- the gz_mean AM+KS minus AM paired differences in ROC-AUC, AP and prec@R, macro and per gene (rng 0).

The `position_cluster` stage runs `experiments/position_cluster_bootstrap/run.py` on the new caches. Its built-in check must pass: with singleton clusters it reproduces the variant-level intervals bit for bit (KS for every gene, against `side/ks_auc_ci`; the combinations against `publication/annotation_tables`).

`compare` writes `before_after/binary_intervals.csv` with all 54 intervals: KS for 6 genes and the combination for 7 rows (six genes plus macro) × 3 metrics, each in both units. Rows with `published=True` (position clusters, 2,000 reps) are the numbers for the manuscript; the variant-level rows are kept for the record. Old values, the current main-checkout results, sit alongside, with a flag where a combination interval's zero-crossing changed. The published rows are also appended to `headline.csv`.

## SI intervals: position clusters too (decided 2026-10-20)

Dan extended the published unit to the SI intervals, so Methods' "every bootstrap resampled residue positions" holds everywhere. A shared helper, `metric_analysis/pipeline/library/position_bootstrap.py`, implements the main-text estimator:

- AUC intervals resample positions stratified as driver-only, passenger-only or mixed.
- Correlations and paired differences resample positions unstratified.
- With singleton clusters, the helper consumes the generator exactly like the variant-level code it replaces.

Each script now computes both units, checks that singleton clusters reproduce the variant intervals bit for bit, and treats position clusters as its main output; the variant-level version is kept for the record.

| SI item | Script | Published output (positions) | Record (variants) |
|---|---|---|---|
| Note S1 / Table S2 / Fig. S1b AUC CIs (KS, posKS, RMSD, ΔRMSF); KS+posKS contrast | `s1_matched_draws.py` | `auc_matched_draws.csv`, panels; contrast rows "…, residue positions" | `auc_matched_draws_variant.csv`; contrast rows "Methods paired bootstrap (annotation_tables)" |
| Note S1 interface AUC CIs and KS+interface ΔAUC | `interface_matched_draws/` (copied in from the main checkout, where it was untracked) | `auc_matched_draws.csv`, `ks_plus_interface.csv` | `*_variant.csv` |
| Note S8 / Table S6 hierarchical intervals and paired differences | `n90_comparison.py` (cluster-level multinomial weights) | `auc_summary.csv`, `paired_contrasts.csv` | `*_variant.csv` |
| Fig. S3 ρ KS vs AM per gene and Fisher-z macro | `si_figures/ks_vs_am.py` | `ks_vs_am_correlations.csv`, `ks_vs_am_macro.csv`, panels | `ks_vs_am_correlations_variant.csv` |
| Note S7 AF2 KS / D1 / AM AUC CIs | `si_figures/af2_pole.py` | `af2_pole_auc_intervals.csv`, panels | same file, `unit=variant` |

**Validation (read-only, on the current inputs):**

- Every variant path reproduces the current published intervals: S1, S8 and Fig. S3 to ≤1.1e-16, interface and AF2 exactly.
- Every singleton check passes.
- An S1 → interface integration run on the existing S1 scores passed (`smoke/units_check/run.log`, 154 s).

The interface rerun is a new stage, `interface`, after `s1_matched`; MSH2/MSH6 inputs do not change. `compare` writes `before_after/si_intervals.csv`: every SI interval with the position CI, the variant CI and the value currently in the SI, side by side.

**Two pre-existing blockers found and fixed on the way** (each would have halted the overnight run):

- `si_figures/ks_vs_am.py` keys its reference check by config fingerprint and did not know 49028c16cbb3. Added: non-POLE genes must still match; POLE is unverified until the rerun reports it.
- `s1_matched_draws` re-derives the quoted +0.004 KS+posKS contrast from the v6_ca production caches (d61d29328362), which were not in the worktree. `link_inputs.py` now copies the ten N=90 files.

## What changed (branch `claude/pole-25seed-3c0895`)

| Where | Change |
|---|---|
| `PLDDTs/POLE_25seed/` (new) | 25-seed copies of the eight matrices. They keep the same basenames and use cell-for-cell text copies of seeds 1–25 (`make_inputs.py`; hashes in `inputs.csv`). The 160-prediction originals in `PLDDTs/POLE/` are untouched. |
| `drift/protein_configs.json` | POLE `mut_csv_replacements` maps each original basename to its 25-seed copy. The fingerprint is now **49028c16cbb3** (was ebbbda31758d). |
| `drift.structure_draws.discover_mut_csv_sources` (new) | The single place mutant CSVs are discovered. It is used by `build_d1_table`, `scan_plddt_sources`, `ensemble_shift` and the KS-125 extension. It raises if a replacement key matches 0 or more than 1 files, if a target is missing, if a target has a different basename, or if a target lies under the read root. The recursive glob would otherwise see both copies. |
| `metric_analysis/manifests/strict_canonical_manifest.csv` | For the eight rows: `source_file` → `PLDDTs/POLE_25seed/…` and `n_mutant_structures` 160 → 125. |
| `metric_analysis/structure_metrics/POLE/POLE_plddt_distribution_summary.csv` | This file, not the manifest, is where the **canonical route** (`canonical_cohort`, hence `figlib.cohort`, `canonical_afm`, `cohort_size_sweep`, the manifest builder) gets its paths. The eight rows are recomputed by the pipeline's own `build_plddt_distribution_tables` (`update_canonical_summary.py`). First the script checks that the builder reproduces the recorded 160-prediction rows; every other row is byte-identical. |
| `canonical_cohort._existing_repo_path` | Re-roots another checkout's `PLDDTs/<folder>/<file>` while **keeping the folder**. It falls back to `PLDDTs/<protein>/<name>` only for files recorded in the protein's own folder. Before this change, a `POLE_25seed` path from the main checkout would have resolved silently to the 160-prediction original. All 2,620 cohort paths are otherwise unchanged (checked). |
| `metric_analysis/pipeline/library/af3_seeds.py` (new) | `MAX_AF3_SEED = {"POLE": 25}`, `seed_numbers`, `within_seed_cap`. |
| Coordinate readers | `rmsf_histogram.read_cache` gains `cap_seeds` (default `max_seed=25`; seeds come from the structure file index). `exemplar_wt_vs_mut.py` (Fig. 1c,d) gains `--all-seeds` and `--out-dir`. `shift_vs_rmsf_P286R.py` (S1a) now takes seeds 1–25 instead of the first 125 structures by index. `s1_matched_draws.py` and S8 `n90_comparison.py` cap POLE coordinates, assert that every draw is within the cap, resolve recorded paths in the running checkout (so draw keys match the caches built there), and take overridable `COHORT`/`CACHE`/`FP`. |
| `experiments/wt_split_12v12/run_12v12_2000.py`, `experiments/position_cluster_bootstrap/run.py` | Copied in from the main checkout, where they were untracked. Env overrides were added (`WT12_*`, `PCB_*`); unset, the defaults are as published. |

Fig. 1c,d before this change used all 125 WT predictions but the **first 125 P286R predictions by structure index**, a mix of seeds 1–32. After the change, both sides use seeds 1–25.

## Stages (`run_overnight.py`, in this order, stop at the first failure)

1. `preflight`: checks that this checkout's modules are the ones imported, the fingerprint, input hashes, discovery of the 25-seed files by both routes, the base-cache hashes and the linked inputs.
2. `cache` (`recompute_cache.py`): recomputes all 1,074 labelled POLE variants and drops P356S. That covers 70 default cells (sizes 1, 5, 10, 20, 40, 60, 90 × seeds 42–51) and 30 sampling-policy cells (random_pinned and top_confidence × N 1, 20, 90 × seeds 42–46). Five descriptors use the 10-18 kernels and production draws. Other genes are copied from the 10-18 cache (`experiments/pole_wt_repair/cache`, read-only, main checkout). Output goes to `cache/` here only. Production builders on the real config must agree to 1e-12: all of POLE at (90, 42) and (5, 43), plus three variants at N=1.
3. `check_cache` (`check_cache.py`), a **hard check**. Only the eight may differ. P356S must be absent. Every other POLE row and every other gene must be bit-identical in all 100 cells, and the eight must keep identical gene, mutation, label and AM.
4. The 21 publication stages of the 10-18 run, in `run_stage.py`. They cover Fig. 2, the AM combination and its bootstraps, the learned screens, the article panels and the SI figure stages. Run tag `v6_pole25_20261020`.
5. Side analyses:
   - `ks125`: the N=125 extension.
   - `ks_auc_ci`: main-text hierarchical KS CIs, on both the 10-18 cache and the new cache.
   - `wt12v12`: the 12v12 × 2,000 WT background (S2).
   - `position_cluster`: must reproduce the new `ks_auc_ci` for every gene.
   - `s8_n90`: the S8 matched representatives.
   - `s1_matched`: the S1 coordinate descriptors on the exact KS draws.
   - `fig1cd`: Fig. 1c,d `_row` panels with the manuscript's arguments, plus an all-seeds rerun that must reproduce the published table exactly.
   - `interface`: the MSH2/MSH6 interface control, for its position-cluster intervals.
   - `s1a`.
6. `compare` (`compare.py`) writes `before_after/`. `file_summary.csv` lists every CSV pair compared. `moved_numbers.csv.gz` holds every numeric cell with |change| > 1e-12. `headline.csv` gives KS AUC by N for POLE and macro (including N=125), the gz_mean bootstrap, main-text KS CIs and Fig. 1c,d numbers, with the paper `.tex` lines that contain the old rounded value.
7. `readonly_check` (`verify_readonly.py`) checks that no main-checkout file reached through a link changed, that the base caches are unchanged, and records the main checkout's `git status` difference.

**Expected wall time: about 2 h** (1.5–2.5 h). From the 10-18 stage records, the long stages are `depth` (31 min), `position_cluster` (24 min) and `fusion_intervals` (10 min). The cache stage is about 4 min (smoke timings below); every other stage takes a few minutes or less.

## Launch, follow, resume, stop

```bash
cd <repo>
python experiments/annotation_stages/run_overnight.py --detach
tail -f experiments/annotation_stages/STATUS.log
# after a failure, fix, then resume from that stage (earlier outputs are kept):
python experiments/annotation_stages/run_overnight.py --detach --from <stage>
# stop:
kill -- -$(cat experiments/annotation_stages/logs/runner.pgid)
```

- The runner is started with `Popen(start_new_session=True)`, so its PPID becomes 1. This was verified with a one-stage dry run.
- `caffeinate -i -w <pid>` keeps the Mac awake while the runner lives.
- A fresh start refuses to run if `cache/`, `publication/`, `side/` or `before_after/` exist.
- When resuming `s8_n90`, delete `side/s8_n90_results/` first; that script refuses to overwrite.

## Inputs that git does not carry

`link_inputs.py` was run once. Gitignored inputs that are under 20 MB were **copied** from the main checkout. Larger files, and the coordinate-cache folders `metric_analysis/structure_metrics/<gene>/ensemble_aware`, were **symlinked**. Every target is fingerprinted in `inputs_linked.csv` (local state, gitignored; rerun `link_inputs.py` in a fresh worktree).

The copies and links show up as untracked files in this worktree; do not commit them. Nothing is written into the main checkout, and `readonly_check` verifies this.

## Smoke test (2026-10-20, passed)

| Check | Result | Time |
|---|---|---|
| `recompute_cache.py --smoke` (N=90, seed 42; P286R, A341D, S297F) | Production builders agree, max \|diff\| 0 | 1.7 s |
| `check_cache.py --smoke` | **passed**. A341D and S297F reproduce the 10-18 cache bit for bit from this worktree, so draw keys are path-independent. P286R changed: KS 0.87723 → 0.87386 (−0.0034), D1 +0.00019, centred KS +0.019. | <1 s |
| Full-POLE production builders (one cell, all 1,227 POLE files) | 1,227 rows; replacements picked up | 16 s |
| Fig. 1c,d reader on P286R | **125 WT + 125 P286R** (35 seeds >25 dropped). The all-seeds rerun reproduces the published `exemplar_P286R_rmsf.csv` exactly. New: RMSD 0.210 → 0.212 Å, median RMSF ratio 1.66× → 1.72×, 202/204 residues broader (unchanged). | 5 s |
| S1a reader on P286R | 125 + 125. Pearson r 0.347 → 0.427, Spearman ρ 0.246 → 0.371, RMSD 0.21 → 0.212 Å, site 286 ΔRMSF 0.098 → 0.119 Å | 1 s |
| `preflight.py`, detach dry run (`--from readonly_check`) | passed; runner PPID 1, own session | 2 s |

Smoke outputs are in `smoke/`.

## Not covered by this run (follow-ups)

- **The canonical-route results** (`cohort_size_sweep_v6_ca`, `canonical_annotation_v6_ca`, `ks_meanfree` canonical, canonical LOGO logistic) are not recomputed. The canonical cohort now points at the 25-seed files, but the `article` stage still reads the v6_ca tables, exactly as the 10-18 run did.
- The **coordinate ensemble sweeps** (`structure_metrics/<gene>/ensemble_aware/ensemble_sweeps`, read by the `si_coordinate_vs_plddt` stage) still draw from all 160 predictions of the eight. The published S1 coordinate numbers come from `s1_matched_draws`, which is fixed here.
- The **Fig. 1b renders** (ChimeraX, hand-made) are not rebuilt.
- These metric_analysis POLE per-variant tables still describe the 160-prediction files: `POLE_mutation_features.csv`, `POLE_mutations_with_scores.csv`, `POLE_plddt_ks_summary.csv`, `POLE_plddt_ks_per_residue.csv`, and the structural variant inventories.
- Manuscript text is not touched. Changes are proposed in chat from `before_after/headline.csv`.
