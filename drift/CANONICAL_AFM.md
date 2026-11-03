# Canonical AlphaMissense fusion analysis

## Question

Do raw per-residue KS, D1, or centred KS carry information complementary to
AlphaMissense on the exact canonical structural cohort, and can a model trained on other
proteins use that information on a held-out protein?

This is the primary fixed-candidate complementarity analysis. The separate
72-candidate model-selection diagnostic is retained only in its strict, six-evaluation-
protein form at `results/benchmark_evaluation_v2_strict/`. The historical seven-protein
route used eight superseded TP53 labels, included the FOXA1 positional control in the
endpoint and scanned two misleading `*_old.tsv` files; it is not a source of manuscript
numbers. The canonical route loads eight exact current files and joins them to mutation
codes, labels and subunit provenance from `canonical_cohort.py`.

## Candidate freeze and evaluation

The following hierarchy was fixed before canonical fusion performance was inspected on
2026-08-30:

- primary: AlphaMissense + raw per-residue KS;
- pre-existing comparator: AlphaMissense + D1;
- secondary: AlphaMissense + D1 + raw KS, including whether KS adds beyond AM+D1;
- mean-free sensitivity: replace raw KS with centred KS.

Every declared candidate is reported. There is no feature-subset selection, C sweep,
normalisation sweep or best-candidate headline.

Two fusion rules answer different questions:

1. `fixed_rank` is a label-free equal mean of within-protein percentile ranks. It is
   non-learned but transductive because it requires the unlabelled score distribution of
   the evaluated protein.
2. `logo_logistic_original` is linear logistic regression with fixed C=1, equal protein
   and within-protein class mass, outer-training-only standardisation, and one untouched
   held-out protein per fold. `logo_logistic_within_gene_rank` is a declared transductive
   sensitivity analysis.

FOXA1 is excluded from the six-protein headline and evaluated as a control after fitting
on all six evaluation proteins. Ten existing structure-subsampling seeds are retained at
N=10, 20, 40, 50, 60 and 90. No structures are generated.

## Coverage

| dataset | canonical | with AlphaMissense | positive in intersection |
|---|---:|---:|---:|
| POLE | 1,075 | 1,075 | 38 |
| PTEN | 257 | 257 | 248 |
| VHL | 124 | 124 | 105 |
| KRAS | 45 | 45 | 25 |
| MutS_alpha | 147 | 147 | 97 |
| TP53 | 1,174 | 1,174 | 309 |
| FOXA1 | 821 | 821 | 744 |

AlphaMissense covers every variant in the strict cohort. The frozen pre-correction
inventory contains 3,656 labelled artifacts; the live source-supported KRAS rule adds
two variants, giving 3,658. Nine initiator-codon models (five PTEN and four MSH6) and
six transcript-unmatched VHL aliases are excluded before the AlphaMissense join. The
strict cohort therefore contains 3,643 variants, and the primary six-protein AFM
intersection contains 2,822 variants and 822 positives.

## N=90 results

### Six-protein outer-LOGO macro

Intervals are paired, class-stratified evaluation bootstraps that also resample the ten
descriptor seeds. They are conditional on the already fitted held-out predictions; the
models are not refitted inside each bootstrap replicate.

| candidate | macro AUC | delta vs AM | paired 95% CI for delta |
|---|---:|---:|---:|
| AlphaMissense | 0.8735 | — | — |
| AM + D1 | 0.8873 | +0.0138 | -0.0011 to +0.0322 |
| **AM + raw KS** | **0.8807** | **+0.0072** | **-0.0190 to +0.0317** |
| AM + D1 + raw KS | 0.8797 | +0.0062 | -0.0200 to +0.0308 |
| AM + centred KS | 0.8867 | +0.0132 | -0.0019 to +0.0308 |
| AM + D1 + centred KS | 0.8579 | -0.0155 | -0.0353 to +0.0042 |

No candidate establishes a universal cross-protein predictive gain: every positive
six-protein delta interval crosses zero. The triple raw-KS model does not improve on
AM+raw-KS, consistent with the strong redundancy between D1 and raw KS.

The non-learned rule is clearly unsuitable for AFM fusion. AlphaMissense is sufficiently
stronger on several proteins that equal weighting can destroy more ranking information
than it adds; these fixed-rank results remain descriptive rather than a learned fusion
claim.

### Per-protein AM + raw-KS deltas

| held-out dataset | delta vs AM | paired 95% CI |
|---|---:|---:|
| POLE | +0.0421 | +0.0129 to +0.0725 |
| PTEN | +0.0028 | -0.0004 to +0.0097 |
| VHL | +0.0104 | -0.0117 to +0.0461 |
| KRAS | -0.0270 | -0.1736 to +0.1074 |
| MutS_alpha | +0.0103 | -0.0066 to +0.0308 |
| TP53 | +0.0048 | -0.0013 to +0.0106 |

Only POLE's two-sided interval excludes zero. KRAS is too small to estimate its negative
effect precisely, but it is the reason the global result is heterogeneous.

### Label-route stratification

These are descriptive averages of already-held-out per-protein predictions; no subgroup
model is refitted or selected.

| held-out label route | AM AUC | AM + raw-KS AUC | delta | paired 95% CI |
|---|---:|---:|---:|---:|
| annotation: POLE, PTEN, VHL, MutS_alpha | 0.9211 | 0.9375 | +0.0164 | +0.0058 to +0.0283 |
| functional: KRAS, TP53 | 0.7783 | 0.7672 | -0.0111 | -0.0853 to +0.0562 |
| annotation excluding POLE | 0.9535 | 0.9613 | +0.0078 | -0.0022 to +0.0212 |

The no-POLE row is a post-result fragility sensitivity and must not become the headline.
Its positive point estimate shows that the annotation-side result is not numerically
reversed without POLE, but its interval crosses zero and therefore does not establish an
independent no-POLE gain.

## Conditional association beyond AlphaMissense

Within each protein, likelihood-ratio tests use the mean N=90 descriptor across the ten
sampling seeds. They are deliberately in-sample association tests, not predictive
validation.

- D1 beyond AM: Fisher p = 2.07e-6; Holm-adjusted across all six declared tests
  = 6.20e-6;
- raw KS beyond AM: Fisher p = 7.22e-12; Holm-adjusted across all six declared tests
  = 4.33e-11;
- raw KS beyond AM+D1: Fisher p = 5.21e-5; adjusted = 1.04e-4;
- centred KS beyond AM: Fisher p = 6.09e-8; adjusted = 2.43e-7;
- centred KS beyond AM+D1: Fisher p = 1.15e-3 after adjustment.

For raw KS beyond AM, individual p-values are POLE 7.47e-10, TP53 3.23e-5, PTEN
0.0481, MutS_alpha 0.0666, VHL 0.140 and KRAS 0.743. FOXA1 is null (p=0.202).

This supports the sentence that KS contains label-associated information not reducible
to AlphaMissense or D1. It does **not** support the stronger sentence that adding KS
reliably improves prediction on an arbitrary unseen protein.

## Manuscript interpretation

The strongest defensible case is:

> Per-residue KS contains information conditionally independent of AlphaMissense and D1.
> A fixed linear fusion improves the held-out macro across annotation-labelled datasets,
> but the no-POLE sensitivity is inconclusive and the benefit does not transfer uniformly
> to functional-screen endpoints.

Do not report “DRIFT improves AlphaMissense across genes.” The six-protein interval does
not support it. Report the heterogeneity, the annotation-route result, the null FOXA1
association test, and the failure of equal weighting. That combination is methodologically
stronger than selecting the most favorable model from another grid.

## Reproduction

```bash
PYTHONPATH=DRIFT python \
  -m drift.canonical_afm --force
```

Primary files:

- `results/afm_fusion/canonical_v2_strict/auc_seed_summary.csv`
- `results/afm_fusion/canonical_v2_strict/n90_bootstrap_summary.csv`
- `results/afm_fusion/canonical_v2_strict/incremental_lrt_per_gene.csv`
- `results/afm_fusion/canonical_v2_strict/incremental_lrt_combined.csv`
- `results/afm_fusion/canonical_v2_strict/label_route_summary.csv`
- `results/afm_fusion/canonical_v2_strict/afm_coverage_and_inputs.csv`
- `results/afm_fusion/canonical_v2_strict/run_metadata.json`

## Exploratory control-inclusive size sweep

`cohort_size_sweep.py` evaluates the fixed raw-score logistic candidates AM, AM+D1,
AM+raw KS and AM+D1+raw KS at N=1,5,10,20,40,60,90. It extends the canonical ten
paired seed streams only at N=1 and N=5 and reuses the audited descriptor values at the
larger sizes. No descriptor, C, weighting, model family or normalisation is selected.

The two requested cohorts both contain the FOXA1 positional control:

- `all_7`: the six evaluation datasets plus FOXA1;
- `annotation_5_control_inclusive`: POLE, PTEN, VHL and MutS_alpha plus FOXA1.

They are therefore exploratory control-inclusive analyses, not replacements for the
canonical six-dataset and annotation-four results above. At N=90:

| cohort and scoring scope | AM | AM+D1 Δ | AM+raw KS Δ | AM+D1+raw KS Δ |
|---|---:|---:|---:|---:|
| all 7, as requested | 0.8789 | -0.0044 | -0.0003 | -0.0028 |
| same all-7 fits, score six without FOXA1 | 0.8735 | +0.0036 | +0.0082 | +0.0068 |
| annotation 5, as requested | 0.9192 | -0.0282 | -0.0242 | -0.0035 |
| same annotation-5 fits, score four without FOXA1 | 0.9211 | -0.0159 | -0.0186 | +0.0147 |

The paired variant+seed 95% intervals for the requested control-inclusive raw-KS deltas
are -0.0184 to +0.0182 (all seven) and -0.0339 to -0.0153 (annotation five). Neither
supports a gain over AlphaMissense, and the annotation-five fit is significantly worse.
FOXA1 itself is strongly negative at N=90 (-0.0508 and -0.0465 in the respective fits),
which contributes to the sign reversal when it is included in the macro.

The control-excluded annotation-four row is heterogeneous: raw KS is -0.0186 while the
triple is +0.0147 (95% interval +0.0044 to +0.0254), but it is not a clean annotation-four training
experiment: FOXA1 entered the outer-training set whenever one of those four proteins was
held out. It is a sensitivity describing the fitted annotation-five model, not evidence
that should supersede the canonical annotation-route result.

N=1 raw KS is retained only as a diagnostic. Singleton-vs-singleton KS is almost
degenerate (gene means 0.9768-0.9994), so the interpretable KS size curve begins at N=5.

```bash
PYTHONPATH=DRIFT python \
  -m drift.cohort_size_sweep \
  --sizes 1,5,10,20,40,60,90 --workers 4 --bootstrap-reps 1000 \
  --wandb --wandb-project afm_d1_ks_cohort_size_sweep
```

Primary files are under `results/afm_fusion/cohort_size_sweep_v2_strict/`: `auc_seed_summary.csv`,
`n90_bootstrap_summary.csv`, `held_out_predictions.csv.gz`, `raw_ks_resolution.csv`,
`fusion_delta_by_size.png`, and `run_metadata.json`.

### Post-hoc log-D1 ablation

A deliberately isolated follow-up replaces D1 by `log10(D1)` before the same
outer-training standardisation. The transform is monotone and was already used in the
conditional-association analysis, but its fusion performance was inspected after the
original candidate freeze; it is therefore a sensitivity, not a new primary result.

On the clean canonical cohorts at N=90:

| cohort | original AM+D1 Δ | AM+log-D1 Δ | AM+KS Δ | AM+log-D1+KS Δ |
|---|---:|---:|---:|---:|
| six evaluation datasets | +0.0138 | **+0.0151** | +0.0072 | +0.0076 |
| four annotation datasets | +0.0088 | **+0.0101** | **+0.0159** | +0.0141 |

The log transform improves AM+D1 by only +0.0013 macro AUC on the six datasets and
+0.0013 on annotation four. Those differences are stable across descriptor seeds but
do not include cohort uncertainty. The N=90 paired variant+seed interval for
AM+log-D1 versus AM is -0.0036 to +0.0371 on the six-dataset macro and -0.0011 to
+0.0211 on annotation four.

Adding KS to log-D1 does not help: AM+log-D1+KS remains far below AM+log-D1 on the
six-dataset macro and below AM+KS on annotation four. The appropriate reduced models
remain AM+log-D1 and AM+raw KS, reported separately.

The control-inclusive ablation further demonstrates why FOXA1 cannot be pooled casually.
AM+log-D1 is +0.0120 on the all-seven macro, but -0.0351 on annotation five; in the
latter fit the held-out POLE delta is -0.1567. The sign therefore depends strongly on
which label routes enter training.

Canonical reproduction:

```bash
PYTHONPATH=DRIFT python \
  -m drift.cohort_size_sweep \
  --base-scores drift/results/afm_fusion/cohort_size_sweep_v2_strict/per_variant_seed_requested_sizes.csv.gz \
  --sizes 1,5,10,20,40,60,90 \
  --out drift/results/afm_fusion/cohort_size_sweep_logd1_canonical_v2_strict \
  --log-d1-ablation --cohort-panel canonical --bootstrap-reps 1000
```
