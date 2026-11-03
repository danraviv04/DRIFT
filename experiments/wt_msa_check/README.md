# Is the regenerated WT's MSA broken? (2026-10-15)

## Background
Each `*_WT_vartmpl` job is one variant's finished AF3 input with the WT residue put back:
- MSH2 from Q629R;
- PIK3CA from K532R;
- TP53 from A119D.

The job is run inference-only, so the WT inherits that variant's **MSA and templates**.

## 1. The copy is intact: `check_vartmpl_msa.py` -> `results/vartmpl_msa_check.json`
For all three genes, against the source variant's full `_data.json`:
- The only differences are the reverted residue in the target chain(s), plus `name` and `modelSeeds`.
- Templates are byte-identical.
- Every non-query MSA row is byte-identical, in both the unpaired and the paired MSA.
- The query row equals the WT sequence in every protein chain.
- Every row has exactly len(query) match columns: 0 malformed rows out of 17k/50k (MSH2), 14k/28k (PIK3CA) and 2k/4k (TP53).
- Unmutated partner chains (MSH6 in the MSH2 jobs, p85) are unchanged.

## 2. The MSA is a real input that changes with the mutation
Facts from the provenance hashes (`../wt_variant_offset/results/provenance`, `sha_without_query`):
- **Every job ran its own MSA search** on its own query. The mutated chain's MSA differs between variants (MSH2 A256G 336746, Q629R b19e16), and its depth differs by a few sequences.
- **The search is deterministic.** The same MSH2 query gave byte-identical MSAs months apart: unpaired b89d88 and paired 6b7cb0 in the old MutSα WT job (03-24), the MSH6 G:T variant jobs (09-25) and the MSH6 G:T WT job. Partner chains are identical across all jobs.
- So the MSA difference between two jobs is caused by the substitution, not by search noise.

How much does one substitution change it? `msa_response_msh2.py` compares the true-WT MSH2 MSA with Q629R's, unpaired:
- **Homolog set:** 99.4% shared by sequence ID. 105 hits dropped and 93 added, and 195 of those 198 cover residue 629. These are borderline hits whose score crosses the inclusion threshold.
- **Alignment:** 38% of rows are aligned differently, but only 493 of them differ outside residues 615–645. The realignment is local to the site.
- **Per-column coverage:** max change 6% at 629; mean 0.06%.

## 3. What "broken" would mean, and what is known

Three different things can go wrong. The checks below cover each:

| Possible failure | How to check | Status |
|---|---|---|
| (a) The WT job's MSA is malformed or mismatched to the WT query | §1 | **No.** It is exact. |
| (b) The MSA shared with one variant makes that variant look like the WT, i.e. a source-variant bias | Source variant's KS against the new WT, and its rank | MSH2 Q629R is at noise (0.185 vs WT-WT 0.188; rank 8/109), but it was already low with the old WT (rank 16). PIK3CA K532R: 0.22 (rank 18/91; was 4). TP53 A119D: 0.275, well above noise (rank 751/1174). Sharing the MSA and templates does not collapse a variant to noise. |
| (c) A WT carrying a *variant's* MSA is offset from a WT with the *true* MSA, which would add a small systematic term to every variant's KS | Indirect: in POLE and MSH6 the templates are identical, the WT has its own true MSA, every variant has its own MSA, and the WT sits inside the variant cloud (0% outside the band). MSA-only differences therefore produce no gene-wide offset there. | **No.** Direct test run 10-16 (§4e): swapping in the true-WT MSA moves the WT no more than re-seeding does. |

## 4. The decisive test: the MSA-swap control (run 2026-10-16: NEGLIGIBLE, §4e)
One job that differs from `MutS_alpha_MSH2_WT_vartmpl` **only in chain A's MSA**:
- WT sequence;
- Q629R's templates, byte-identical (including 2O8B);
- chain A's `unpairedMsa`/`pairedMsa` replaced by the true-WT MSH2 search (b89d88/6b7cb0), taken from the first JSON of `muts_jsons/MutS_alpha_MSH6_GT_WT_seed_ext_26-75.zip`;
- chain B (MSH6) unchanged. Its paired MSA is 1bbf67 in both jobs, and both jobs have 50,000 paired rows on each chain, so the A/B row pairing is the one AF3 itself produced for the WT search.

Name `MutS_alpha_MSH2_WT_vartmpl_wtmsa`; seeds 1–10, the same seed values as the first 10 of the vartmpl WT's 1–25.

### 4a. Build: `build_msa_swap_job.py`
```bash
python -I experiments/wt_msa_check/build_msa_swap_job.py
```
- Writes `msa_swap_job/MutS_alpha_MSH2_WT_vartmpl_wtmsa.zip` (30 MB; it holds `MutS_alpha_MSH2_WT_vartmpl_wtmsa_data.json`, 178 MB unzipped). The zip is gitignored; the `_manifest.json` beside it is not.
- Writes `results/msa_swap_job_check.json`, and exits 1 on any failed check. The checks run on the JSON read back from the zip.
- **All checks pass (10-16):**
  - The diff against the vartmpl WT is `name` and `modelSeeds` at the top level, plus chain A's `unpairedMsa`/`pairedMsa`. Nothing else changes: chain B, both DNA strands and both ligands are byte-identical.
  - Templates are byte-identical on both chains (4 each).
  - Every query row equals its chain's sequence.
  - There are 0 malformed rows: chain A has 17,357 unpaired and 50,000 paired; chain B has 17,448 and 50,000.
  - The hashes are A b89d88 / 6b7cb0 and B 1bbf67, and B is 1bbf67 in both sources. The vartmpl's chain A was b19e16 / eeaf11 (the Q629R search).

### 4b. Run it on the cluster (inference only, 10 seeds ≈ 2 h on one GPU)
The JSON is a finished `_data.json`, so both runners call AF3 with `--run_data_pipeline=false`. Use either one:

**Local GPU server** (`af3_jobs/local_runner`, set up as in its README):
```bash
scp experiments/wt_msa_check/msa_swap_job/MutS_alpha_MSH2_WT_vartmpl_wtmsa.zip <user>@<server>:
```
```bash
# on the server, in the runner's install directory
IN=$AF3_WORK_ROOT/inputs/MSH2_wtmsa && mkdir -p $IN && unzip ~/MutS_alpha_MSH2_WT_vartmpl_wtmsa.zip -d $IN
GEN_IN=$IN DRYRUN=1 ./gen_local.sh start MSH2_wtmsa 0      # check: 1 pending, GPU free
GEN_IN=$IN ./gen_local.sh start MSH2_wtmsa 0               # pick a free GPU
./gen_local.sh status MSH2_wtmsa
# -> $AF3_WORK_ROOT/gen/MSH2_wtmsa/MutS_alpha_MSH2_WT_vartmpl_wtmsa_data.tar.gz
```

**A Slurm cluster:** run AlphaFold 3 on the unpacked `_data.json` with `--run_data_pipeline=false`, writing to a fresh output folder so the archive cannot collide with the vartmpl WT.

### 4c. Score: `score_msa_swap.py`
Copy the archive to the Mac's internal disk, not to Elements, and score it. The script reads the archive once (`fast_plddt.archive_plddt_table`, chains A,B), which is the extraction `stream_align --plddt-out` ran for the WT CSVs. It keeps the 2294 × 50 table as `results/msa_swap/MutS_alpha_MSH2_WT_vartmpl_wtmsa_plddt_values.csv`. Afterwards, delete the archive.
```bash
PYTHONPATH="." python experiments/wt_msa_check/score_msa_swap.py \
    --archive ~/Thesis/wt_msa_swap/MutS_alpha_MSH2_WT_vartmpl_wtmsa_data.tar.gz
```
Use `--csv <table>` instead if the table was already extracted. Outputs: `results/msa_swap/msa_swap.json` and `msa_swap_per_residue.csv`.

What it computes. Everything uses the production KS-score (`descriptor_metrics.residue_ks`) over MSH2 rows only (residues 1–934). The reference is `PLDDTs/MutS_alpha/MutS_alpha_WT_plddt_values.csv`, which is the vartmpl WT, i.e. WT@Q629R-MSA, 25 seeds × 5.

| quantity | definition | role |
|---|---|---|
| `disjoint_ks_median` | KS(WT@WT-MSA seeds 1–10, WT@Q629R-MSA on 10 of its seeds 11–25), median of 500 draws | **decisive** |
| `background_p95` | KS(WT@Q629R-MSA seeds A, seeds B), A and B disjoint sets of 10 from 1–25; 95th percentile of 1000 draws | prediction noise at the same size (computed; ~0.21 mean, ~0.27 p95) |
| `same_seed_ks` | KS(WT@WT-MSA seeds 1–10, WT@Q629R-MSA seeds 1–10) | reported only (see below) |
| `variant_ks_median`, `msa_excess_as_frac_of_variant_excess` | the same seed-disjoint design for the 109 MSH2 variants against the WT | sets the MSA term against a real effect |
| `site_615_645_*` | window-mean KS over 615–645 (where the Q629R search realigned) against the background's window p95 | is the term local to the site? |

Why the same-seed number is not the test: one seed couples two runs with near-identical inputs. On shared seeds, MSH2 variants score 0.017 lower against this WT than on disjoint seeds (median of 40 variants; up to 0.11 for single variants). So a same-seed KS reads low against a seed-disjoint background. The disjoint comparison is built exactly like the background.

### 4d. Decision rule
- **`disjoint_ks_median` ≤ `background_p95` → NEGLIGIBLE.** Swapping the MSA moves the WT no more than re-seeding does. The MSA channel is negligible, and the vartmpl WT is a valid WT.
- **Above it → ABOVE NOISE.** Every variant's KS carries a term from the MSA search's response to the substitution. Disclose it (size: `msa_excess_over_background`, and as a fraction of the median variant's excess), or put every job on one shared MSA.
- The site-window test and `top_excess_residues` show whether the term is local to 615–645. The raw per-residue excess is descriptive only: with one side fixed it is not zero under the null.

Synthetic stand-ins (no GPU output) check the script end to end:
- `--synthetic null`: vartmpl seeds 18–25 pose as the new run. That is 8 seeds, because 25 seeds cannot hold three disjoint sets of 10. Result: NEGLIGIBLE, median 0.255 vs p95 0.295 (75th percentile of the background); site window 0.312 vs 0.329.
- `--synthetic positive`: the legacy WT, seeds 1–10. It has the true-WT MSA but lacks the 2O8B template. Result: ABOVE NOISE, 0.436 vs p95 0.270, excess 6× the median variant's.
- A fake `.tar.gz` built from the legacy WT's values went through `--archive`. The job-level top model was dropped, the 50 columns came back value-identical, and the result was identical to the positive run.

### 4e. Result (2026-10-16): NEGLIGIBLE
The job ran on Athena (job 230190): 50/50 predictions, extracted 2294 × 50. Output: `results/msa_swap/msa_swap.json`.

| | KS-score |
|---|---|
| WT@WT-MSA vs WT@Q629R-MSA, seed-disjoint (median of 500 draws) | **0.227** (5–95%: 0.210–0.255) |
| WT-WT background, 10 vs 10 seeds | mean 0.214, **p95 0.270** |
| same-seed (seeds 1–10 on both sides) | 0.239 |
| 615–645 window, seed-disjoint vs background p95 | 0.287 vs 0.307 (not above) |
| MSH2 variants vs WT, same design (median of 109) | 0.251 |

- The swap sits at the 73rd percentile of the re-seeding background, and only 0.8% of its draws exceed the background p95. The synthetic null, built from vartmpl seeds alone, sat at the 75th percentile with an excess of +0.017; the swap's excess is +0.013. So the swap is indistinguishable from a re-seed.
- The largest per-residue excesses are at 843–882, not at the realigned site, and the site window is below its p95. The search's local response to Q629R does not show up in the pLDDT ensemble.
- **Reading:** the borrowed Q629R MSA adds no detectable term. The new MSH2 WT is a valid WT. MSH2's v6 fusion change (+0.004 → −0.008) is therefore not an MSA artifact; it fits KS becoming more redundant with AM (ρ 0.50 → 0.75, see `experiments/v6_ca_run`).

The same construction works for PIK3CA (swap chain A only, from the old PIK3CA WT `_data.json`, 75dd8c, on the drive) and for TP53 (from its old WT `_data.json`).

## 5. Which part of the WT change moved MSH2's fusion? `wt_change_decomposition.py` (2026-10-16)
Between v5 and v6, MSH2's only changed input is the WT CSV; MSH6 is bit-identical in both runs. The CSV changed in two steps. Full-ensemble KS (125 vs 125, MSH2 rows), 109 variants (71 drivers), AM AUC 0.923 throughout; `results/wt_change_decomposition.csv`:

| WT | KS mean | AUC KS | ρ(AM, KS) | z(AM)+z(KS) AUC | fusion Δ |
|---|---|---|---|---|---|
| old job, backbone-N pLDDT (v5) | 0.582 | 0.797 | 0.49 | 0.928 | +0.004 |
| old job, Cα pLDDT | 0.438 | 0.879 | 0.64 | 0.936 | +0.013 |
| regenerated vartmpl WT, Cα (v6) | 0.210 | 0.842 | 0.75 | 0.908 | −0.016 |

- The approximation tracks production: KS AUC 0.797/0.842, against production N=90 values of 0.796/0.841.
- **N → Cα** helped the fusion.
- **The template change** (the old WT lacked 2O8B; §4e rules out the MSA) turned it negative, because KS now agrees more with AM.
- The old-template KS AUC of 0.879 is not a cleaner number. Those WTs put every variant ~0.35 above noise, unevenly (`../wt_variant_offset`).
- **Paired bootstrap** (2000 reps, stratified by label; `results/wt_change_bootstrap.csv`), old Cα minus new:
  - KS AUC: +0.037 [−0.029, +0.102], p=0.28;
  - fusion Δ: +0.028 [−0.001, +0.062], p=0.05.
  The old WT's "better" numbers are not distinguishable from the new ones at n=109.

**Effect on the annotation macro** (`macro_with_old_wt.py`, `results/macro_with_old_wt.csv`). This re-implements the headline's non-learned AM+resKS fusion (gz_mean, N=90, LOGO scalers, 10 descriptor seeds) on the v6 per-variant KS and reproduces the published v6 macro (0.0086 vs 0.0088). Then it swaps in MSH2's KS against each WT:

| MSH2 KS against | MSH2 gain | macro |
|---|---|---|
| production v6 (N=90) | −0.009 | 0.0086 |
| new WT, full ensemble (calibration) | −0.010 | 0.0085 |
| old WT, backbone-N (v5's input) | +0.004 | 0.0108 |
| old WT, Cα | +0.011 | 0.0120 |

- The old Cα WT would raise the macro by +0.0035.
- The other genes barely move (≤0.001), because MSH2 enters their scalers only as training rows.
- The macro bootstrap was not rerun for this scenario.
