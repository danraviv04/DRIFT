# Exploratory structural-representative pLDDT comparison — 2026-10-19

Existing predictions support this comparison. On the same 1,797 variants and
125 predictions per side, full residue KS has equal-gene macro ROC-AUC
**0.851782**, versus **0.647226** for pLDDT D1 between structural medoids and
**0.649475** for their mean absolute pLDDT difference (MAE). Ensemble-mean D1
reaches **0.817157**. This is a useful supplementary comparison of procedures;
it is not evidence that every benefit comes from distribution shape, nor a
comparison against generating only one prediction.

## Inspection and plan

Inspected `experiments/rmsf_histogram/rmsf_histogram.py` and its two saved POLE
representative CIFs; `metric_analysis/pipeline/library/structure_metrics.py`
(`_medoid_frame_index`); `metric_analysis/pipeline/cache_builders/subunit_core_fit.py`;
the six current core-fit reports, per-structure tables and coordinate caches;
the strict canonical cohort loader/manifest and source pLDDT CSVs;
`descriptor_metrics.py`, `descriptor_sampling.py`, `drift/structure_draws.py`;
the POLE WT repair README, provenance and corrected caches;
the saved seed recovery mappings, `rerun_all_genes.py` and `ingest_new_wts.sh`;
and the three supplied manuscript source files. These establish the selection
definition, coordinate frame, source identity, scored subunit, labels and
corrected reference. The manuscript's D1 discussion is currently commented out.

The plan was to reproduce corrected N=1/N=90 performance, audit coverage, time
one medoid calculation, and use the strongest small existing-data comparison
the runtime allowed. The VHL timing showed that exact selection in the cached
frame was inexpensive, so all six genes could be evaluated in one streamed pass.
The successful analysis took **50.48 seconds**, including input hashing and
baseline reproduction. No raw coordinate archive was rescanned or realigned.

## Definitions and comparability

The true medoid here minimizes summed pairwise Cα RMSD:

`argmin_i sum_j sqrt(mean_r sum_xyz (X[i,r] - X[j,r])**2)`.

Distances use the **existing common core-fit frame**, without independent
pairwise superposition. This is a true medoid for that explicitly fixed-frame
distance, not a claim to minimize individually optimized pairwise-fit RMSDs.
`_medoid_frame_index` already implements this definition; the faster distance
calculation was checked against it once per gene.

The `rmsf_histogram.py` function called `medoid` instead selects the structure
nearest the coordinate mean. This minimizes summed **squared** distances and
can differ from the summed-RMSD medoid. Both selections were therefore reported.
Saved assets alone cover only POLE WT and P286R: their Cα pLDDTs match exactly
`seed-15_sample-1` and `seed-27_sample-0`, respectively. The latter is outside
the common pool used here; those two assets cannot supply a cohort-wide baseline.

All exploratory methods use exactly the same **125 seed/sample identities**:
seeds 1–25, samples 0–4. Some POLE mutant pools contain extra seeds; these are
excluded consistently from both selection and scoring. No top-level ranked
copies are included. WT and mutant representatives are selected independently,
with the WT selection shared across variants of a gene. Tie breaking uses the
first lexicographically ordered sample. Labels never enter selection. Every
score remains higher-is-positive, including below-chance VHL results.

Selection and pLDDT scoring use **all residues of the scored chain**, not only
the fitting core. The saved frame was fitted on the existing WT-defined rigid
core; it is unchanged. Residue counts are POLE 204, PTEN 403, VHL 213, MSH2 934,
MSH6 1,360 and PIK3CA 1,068. MSH2 scores chain A; MSH6 scores chain B using its
separate G:T-mismatch system and WT. POLE uses only `POLE_268-471_WT`.

D1 is `mean_r abs(mutant_pLDDT[r] / WT_pLDDT[r] - 1)`; ensemble-mean D1 first
averages each residue across the ensemble. MAE is `mean_r abs(mutant-WT)` in
pLDDT points. All exploratory calculations use float64 and the shared
canonical definitions. The cached production D1 uses float32; it is retained
as a separately identified reference. All exploratory input values were finite
and positive, so no zero-denominator or missing-residue exclusion was needed.
Singleton KS is also saved for completeness, but merely counts unequal rounded
values and is not the primary representative baseline.

## Coverage and results

The annotation cohort was verified as 1,799 variants: POLE 1,075/38 positives,
PTEN 257/248, VHL 124/105, MSH2 109/71, MSH6 143/66 and PIK3CA 91/84.
**Two variants are excluded from every matched comparison**:

- POLE P356S (passenger): no coordinate artifact.
- MSH2 G162R (driver): 125 coordinate structures exist, but eight lack sample
  identities. The 117 identified members were not silently treated as a full
  ensemble, and the eight mappings were not guessed.

MSH2/PIK3CA WT identities were recovered previously from sorted archive members;
this analysis reuses and validates those saved mappings against their recorded
member names and original ingestion convention. It does not independently
re-extract those raw archives. All other coordinate-covered source identities
match their exact canonical source artifact keys (1,798/1,798, including G162R).

**Matched ROC-AUC, N=125 pools.** All representative baselines below use selections
from the full ensemble. Macro is the unweighted mean of six gene AUCs.

| Gene | Variants / positives | Full KS | Mean D1 | Medoid D1 | Medoid MAE | Nearest-mean D1 |
|---|---:|---:|---:|---:|---:|---:|
| POLE | 1,074 / 38 | 0.858819 | 0.853282 | 0.792674 | 0.792496 | 0.792141 |
| PTEN | 257 / 248 | 0.873208 | 0.853943 | 0.874104 | 0.877240 | 0.774642 |
| VHL | 124 / 105 | 0.917794 | 0.813534 | 0.272682 | 0.255138 | 0.731328 |
| MSH2 | 108 / 70 | 0.840977 | 0.802632 | 0.673684 | 0.682707 | 0.676692 |
| MSH6 | 143 / 66 | 0.914109 | 0.861865 | 0.595041 | 0.603896 | 0.607241 |
| PIK3CA | 91 / 84 | 0.705782 | 0.717687 | 0.675170 | 0.685374 | 0.603741 |
| **Equal-gene macro** | **1,797 / 611** | **0.851782** | **0.817157** | **0.647226** | **0.649475** | **0.697631** |

Mean-profile MAE macro AUC is **0.828538**; nearest-mean representative MAE is
**0.713262**. All per-gene values and singleton KS diagnostics are in
`auc_summary.csv`, with one-row-per-variant scores in `variant_scores.csv`.
KS exceeds true-medoid D1/MAE in five genes; PTEN is an exception. Mean D1
exceeds KS in PIK3CA. No reversal or representative was selected using labels.

The corrected cache reference reproduces the supplied numbers to their reported
six decimals, by computing per-gene AUC for each of seeds 42–51, averaging genes
equally and then averaging seeds:

| Cohort | KS N=1 | D1 N=1 | KS N=90 | D1 N=90 |
|---|---:|---:|---:|---:|
| All 1,799 annotation variants | 0.605564 | 0.621482 | 0.850969 | 0.814864 |
| Matched 1,797 variants | 0.606167 | 0.621220 | 0.850801 | 0.814500 |

The complete per-gene/per-seed values are in `corrected_cached_auc_per_seed.csv`.
These cached random-draw results have a different sampling protocol from the
fixed 125-member analysis and must not be described as its identical input pools.

## VHL sensitivity and interpretation

The low VHL medoid AUC is reproducible and was not inverted. Its true-medoid WT
is `seed-12_sample-0`, mean pLDDT 67.717, with mean absolute departure of 1.896
points from the ensemble-mean profile. The nearest-mean WT is
`seed-24_sample-0`, mean pLDDT 69.260, departure 0.334 points.

A post-result diagnostic reports all four fixed crossings of WT and mutant
selection rules. With mutant medoids held fixed, changing only the WT selection
raises D1 AUC from **0.272682 to 0.675188**. Changing both to nearest-mean gives
**0.731328**. With the true-medoid WT and nearest-mean mutants it is **0.252632**.
This implicates the particular WT confidence profile as a substantial source of
sensitivity; structural centrality does not ensure centrality of pLDDT. No one
crossing is adopted as an optimized baseline. See `VHL_crossed_selection_auc.csv`.

This comparison is **worth adding as an explicitly exploratory supplementary
control**, with both selection definitions and a magnitude-sensitive metric.
It supports retaining ensemble information rather than reducing it to one
structurally representative prediction under these procedures. It does **not**
establish superiority for every gene, every centrality definition, or a structure
generated without an ensemble. Selecting a representative already uses all 125
predictions and therefore offers no inference-generation cost saving.

Much of the macro improvement is recovered by averaging: true-medoid D1
0.647226 → mean D1 0.817157. The further difference to KS is only **0.034625**;
against mean MAE it is **0.023244**. These are descriptive differences, not
significance claims. KS versus D1/MAE changes the scoring functional as well as
the retained information. Coordinate-based selection adds another modality,
and all-residue selection may be dominated by disordered regions or domain
motion. Consequently this experiment does not isolate distributional shape.

Other limitations remain: no bootstrap confidence intervals, representative
resampling stability or alternative-frame/core-only sensitivity were run;
PTEN has only nine passengers and PIK3CA seven; MSA/template provenance differs
between some systems; predictions share seeds and within-seed samples; the
fixed-pool analysis is conditional on the available generations. No label
definitions, scores' scientific formulas, fitting cores, data splits or
production code were changed. This was a new exploratory control, not a
production bug fix. The misleading nearest-mean utility name was left intact.

## Validation and outputs

- All six medoid calculations checked against the existing exact utility on
  one complete artifact per gene; full KS checked against SciPy on all scored
  residues for one variant per gene.
- Independent validation recalculated 30 representative/mean/KS scores,
  reproduced all saved per-gene and macro AUCs, checked **3,606 selection origins**,
  verified **1,838 SHA256 input hashes**, and checked exact source identities.
- Exact seed/sample sets, chain/residue coverage, finite positive pLDDTs,
  unique records and consistent canonical/cache labels are asserted.
- The two original POLE representative CIF pLDDT vectors match their source
  columns exactly. Multi-GB coordinate files are recorded by size and mtime;
  their bytes were not separately hashed or raw archives regenerated.
- Two initial guard failures are retained in the logs: the unlabelled POLE
  M294V duplicate, then G162R's incomplete identities. The first is now ignored
  outside the labelled cohort; the second is explicitly excluded across methods.
- No manuscript source, abstract, existing output, user edit or production file
  was modified. No installation, prediction, expensive fit, commit, push or
  external message was performed. The full repository test suite was not run.

All new artifacts live in this directory. `provenance.json`,
`independent_validation.json`, `coordinate_inventory.csv`,
`structure_selections.csv` and `exclusions.csv` provide the audit trail.

To reproduce, copy the three scripts to a **fresh sibling directory under
`experiments/`** and substitute that directory's name in the commands below.
The primary script refuses to replace its output tables. Use the existing environment:

```sh
PYTHONPATH=. python experiments/s8_representatives/check_representatives.py
PYTHONPATH=. python experiments/s8_representatives/validate_exploration.py
PYTHONPATH=. python experiments/s8_representatives/check_vhl_sensitivity.py
```
