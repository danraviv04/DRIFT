# Simple fusion, and a wider metric battery

Two questions in one run:

1. **Does the fusion of AlphaMissense with the structural descriptors have to be
   learned?** `am_descriptor_model_screen` fitted twelve model families to that
   combination. This adds the fusions that fit nothing at all — normalise each
   descriptor within its gene, then average the normalised values.
2. **Does the answer depend on the metric?** Everything so far has been ranked by
   macro ROC-AUC. Every score here is also scored on average precision, precision
   at R, partial AUC, accuracy, balanced accuracy, F1, precision, recall,
   specificity, MCC, Cohen's kappa, Brier and ECE.

Raw AlphaMissense, each descriptor alone, a random ranking, and the two constant
predictors are carried through the same battery, so every number has a floor to
be read against.

Nothing is refitted. The 752 learned evaluations are read from
`experiments/am_descriptor_model_screen/results_v3_subunit_scope` exactly as
saved, and the run asserts that it reproduces every one of their macro AUCs to
1e-12 from the stored out-of-fold predictions before doing anything else.

## Running it

```bash
cd <repo>
export PYTHONPATH="$PWD"
python -u \
  experiments/simple_fusion_benchmark/selfcheck.py
python -u \
  experiments/simple_fusion_benchmark/run_benchmark.py \
  --out experiments/simple_fusion_benchmark/results_v1
```

About a minute. `run_benchmark.py` refuses to overwrite an existing `--out`.
`selfcheck.py` is a plain script, not pytest — no environment here has pytest —
and covers the metric implementations against sklearn, the degenerate cases, the
two Youden searches against planted cuts, and the fusion builders' invariants.

## Cohort, folds and weighting

The same 2,822 variants across seven evaluation datasets, at N=90 and descriptor
seed 42, with the same LOGO (MSH2/MSH6 held out together) and stratified 5-fold
splits, re-derived and checked index-by-index against the screen's saved
`fold_indices.npz`. Macro values are the equally weighted mean over the seven
genes, as everywhere else in the project.

Prevalence is wildly unequal and this is the single most important thing to hold
in mind when reading accuracy and F1:

| gene | n | drivers | prevalence |
|---|---|---|---|
| KRAS | 45 | 25 | 55.6% |
| MSH2 | 109 | 71 | 65.1% |
| MSH6 | 38 | 26 | 68.4% |
| POLE | 1075 | 38 | **3.5%** |
| PTEN | 257 | 248 | **96.5%** |
| TP53 | 1174 | 309 | 26.3% |
| VHL | 124 | 105 | 84.7% |

## The non-learned fusions (`fusion.py`)

Three label-free normalisations applied **within each gene** — percentile `rank`,
standardised `z`, and `vdw` (inverse normal of the percentile rank) — plus `gz`,
a single global scaler fitted on each fold's training rows, which is exactly the
scaler the learned models use and is the only protocol-dependent variant.

Normalisation is not optional. `d1_score` spans 0.0001–0.076 and `afm_score`
spans 0.04–1.0, so averaging the raw columns would be an AlphaMissense score with
rounding noise added.

Combiners: `mean`, `median`, `max` (OR-like), `min` (AND-like), `gmean` (the rank
product), and `wmean` at fixed AlphaMissense weights 0.667, 0.75 and 0.9 with the
remainder split equally over the descriptors. The weights are pre-specified, not
tuned; the whole grid is reported and no configuration is selected.

Orientation is fixed a priori — every input is defined so that larger means more
driver-like — and no direction is ever flipped to fit the data.

330 evaluations over the same eight input combinations as the learned screen.

## Threshold policies (`metrics.py`)

F1 and accuracy need an operating point, and the scores are not comparable out of
the box: probabilities, SVM decision values and mean ranks live on different
scales. Five policies, none of which may look at the held-out gene's labels
except where stated:

| policy | what it transfers | notes |
|---|---|---|
| `train_prevalence` | a call **rate** | the same gene's training prevalence when the protocol supplies it (stratified), otherwise the equally weighted mean prevalence of the training genes (LOGO). Which one was used is recorded per cell in `operating_points.csv`. |
| `train_youden_rate` | a call **rate** | the rate maximising macro-mean Youden J on the training genes. Scale-free: a shift in one gene's scores cannot change it. |
| `train_youden_value` | a score **threshold** | the value maximising macro-mean Youden J on the training genes. Keeps exactly the absolute-scale information the rate policies throw away. |
| `native_half` | nothing | score ≥ 0.5, for probability outputs only. Blank for the two SVMs, which emit decision values. |
| `oracle_rate` | the held-out gene's true driver **count** | **an upper bound, not an operating point.** It is told how many drivers the gene has, though not which. At this rate precision = recall = F1 = precision@R. |

Rate policies take the top fraction by the k-th largest score and predict
`score >= t`, so ties at the cut are never split arbitrarily; the realised rate is
reported as `ppr` in every table.

Two dependencies are worth naming rather than hiding. First, under stratified CV
the training rows' scores come from other folds' models, so a threshold fitted on
them carries a mild cross-fold dependency; under LOGO it does not. Second,
`native_half` is not AlphaMissense's intended operating point — its published
boundaries are 0.564 and 0.34, and both are in
`fixed_prediction_reference.csv`.

## What came out

Best observed per kind, LOGO, from `tables/headline.md`. These are maxima over
the whole screen and are therefore optimistic in exactly the way the parent
experiment's are; what matters here is the **gap between columns**, not the
absolute values.

| metric | best learned | best non-learned | raw AlphaMissense |
|---|---|---|---|
| ROC-AUC | 0.8998 (MLP · AM+D1+resKS) | 0.8982 (z_wmean0.9 · AM+D1) | 0.8873 |
| average precision | 0.8002 (SVM_linear · AM+resKS) | **0.8187** (z_wmean0.9 · AM+resKS+cKS) | 0.7702 |
| F1 @oracle | 0.7945 (MLP · AM+D1+resKS) | **0.7972** (z_wmean0.9 · AM+resKS+cKS) | 0.7551 |
| MCC @youden-value | **0.5851** (SplineLogreg · AM+D1) | 0.5594 (z_wmean0.9 · AM+D1+cKS) | 0.4849 |
| MCC @youden-rate | 0.4829 (SVM_RBF · AM+cKS) | 0.4731 (vdw_wmean0.75 · AM+D1) | 0.4324 |
| Brier | 0.1312 (SplineLogreg · AM+D1) | not defined | 0.1821 |

**1. Learning the combination buys almost nothing in ranking.** A weighted mean
of two within-gene z-scores, with a weight fixed in advance, lands 0.0016 macro
AUC below the best of 752 fitted models under LOGO, and ahead of all of them on
average precision and on F1 at the oracle rate. Both beat raw AlphaMissense by
0.011–0.049 depending on the metric. This supports the manuscript's existing
decision to report a non-learned combination, and it now has a like-for-like
learned comparison behind it.

**2. Where the learned models do win is calibration, not ranking.** The gap is
0.002 on AUC and 0.026 (LOGO) to 0.082 (stratified) on MCC at a transferred score
threshold, and only the learned probability outputs have a Brier score at all.
`train_youden_value` beats `train_youden_rate` for 73% of scores under LOGO
(median +0.036 MCC) and for 94% under stratified CV (median +0.111), so the
absolute scale carries real information on these genes — and the models that
produce a well-behaved absolute scale are the fitted ones. The LOGO figure is the
honest one: transferring a threshold to a gene the model has never seen is
harder, and a quarter of the time it is worse than transferring a rate.

**3. ROC-AUC is not a proxy for the threshold metrics.** Across all evaluations
its Spearman correlation with the others is 0.80–0.81 for average precision and
F1@oracle, but only 0.46–0.58 for accuracy, balanced accuracy and MCC
(`figures/metric_vs_auc_*.png`). Ranking by AUC and ranking by MCC give
materially different orders.

**4. Accuracy and F1 are close to meaningless without their floor.** "Always
driver" scores F1 = 0.982 and accuracy = 0.965 on PTEN and macro F1 = 0.671
overall, with no input whatsoever. On POLE, "always passenger" scores 96.5%
accuracy. Read `figures/prevalence_reference.png` before quoting either metric,
and quote MCC or balanced accuracy in preference to both.

**5. AlphaMissense's own published cutoff is a strong baseline at a fixed
operating point** — macro MCC 0.583, above raw AlphaMissense under any threshold
this experiment transfers (0.485 at `train_youden_value`) and above the best
non-learned fusion's `train_youden_rate` (0.473). A macro-mean Youden threshold
fitted on six genes transfers worse to the seventh than a boundary calibrated on
a much larger corpus. This is a statement about threshold transfer, not about
ranking quality: the same fusions beat it comfortably on AUC and AP.

## Outputs

| file | contents |
|---|---|
| `tables/headline.md` | the comparison above, both protocols |
| `macro_metrics.csv` | 1,092 evaluations × 48 macro metrics |
| `per_gene_metrics.csv` | the same, per gene, with n / positives / prevalence |
| `operating_points.csv` | the cut used in every (evaluation, fold, gene) cell and where it came from |
| `fixed_prediction_reference.csv` | constant predictors and AlphaMissense's published cutoffs |
| `evaluations.csv` | the registry: kind, method, family, arm, params, protocol |
| `scores.npz` | every score vector, including the learned ones as read |
| `index.html` | filter by protocol / kind / inputs, sort by any metric |
| `figures/` | metric-vs-AUC scatters, metric agreement heatmaps, leaderboards, the prevalence reference |
| `metadata.json` | cohort, source and input hashes, policy and weight settings |

## Per label source (`--gt-group`)

`run_benchmark.py --gt-group annotation|functional` runs the whole battery on
that label source. The learned grid is **refitted** on the cohort rather than
sliced out of the seven-gene fit: a model trained on six genes including the
functional ones is not the model you would deploy on annotated genes. Refitting
the full 752-evaluation grid costs about 100 s, so there is no reason to
approximate it. `--gt-group evaluation` still reuses the screen's saved
predictions and reproduces them exactly (max \|diff\| = 0 across 1,092 × 58).

Each run writes an `ANALYSIS.md` computed from its own table — not left for the
reader to derive — and the report gained a family-comparison view. Best observed
under LOGO:

| cohort | metric | AM alone | best learned | best non-learned | strongest input |
|---|---|---|---|---|---|
| annotation | ROC-AUC | 0.9309 | 0.9442 (QDA) | 0.9435 (gz_mean) | **AM+resKS** +0.0133 |
| annotation | AP | 0.8110 | 0.8641 (LDA) | **0.8657** (z_wmean0.9) | AM+resKS+cKS +0.0546 |
| functional | ROC-AUC | 0.7783 | 0.8189 (LDA) | 0.8183 (z_wmean0.9) | **AM+cKS** +0.0406 |
| functional | AP | 0.6680 | **0.7267** (SVM_linear) | 0.7127 (z_wmean0.9) | AM+D1+cKS +0.0587 |

**13. The learned/non-learned gap stays inside 0.002 on ranking, on both label
sources.** Annotation ROC-AUC: 0.9442 against 0.9435. Functional: 0.8189 against
0.8183. The refit does not change the conclusion the pooled run reached; it
confirms it holds separately.

**14. Non-learned wins on how *often* it helps, learned on how *well* the best
one does.** On annotation ROC-AUC, 33% of non-learned configurations beat
AlphaMissense against 18% of learned; on AP, 78% against 52%. The best cell is
a maximum over 541 configurations and is optimistic; the share is not, and it
favours the non-learned side. The exception is MCC at a transferred threshold,
where under stratified CV 26% of learned configurations beat AlphaMissense
against 1% of non-learned — the calibration finding again, now separated by
label source.

**15. AM+resKS is the annotation winner under the learned models too.** The
strongest input set for annotation ROC-AUC is AM+resKS whether the combination
is fitted (QDA, +0.0133) or not (gz_mean, +0.0126). The non-learned sweep did
not simply pick an arm that suits rank averaging.

## The ensemble-size sweep (`size_sweep.py`)

The non-learned half on its own, swept over the seven ensemble sizes and ten
descriptor seeds the `esfeat6` caches provide — 70 cells, 11,830 evaluations, five
minutes, because nothing is fitted. A fitted model cannot be swept this cheaply,
which is why this is a separate run and a separate page rather than another
column on the table above.

```bash
for g in annotation functional evaluation; do
  python -u \
    experiments/simple_fusion_benchmark/size_sweep.py --gt-group "$g"
  python -u \
    experiments/simple_fusion_benchmark/size_sweep_figures.py \
    --results experiments/simple_fusion_benchmark/results_size_sweep_"$g"_v1
done
python -u \
  experiments/simple_fusion_benchmark/size_sweep_figures.py \
  --results experiments/simple_fusion_benchmark/results_size_sweep_evaluation_v1 \
  --compare experiments/simple_fusion_benchmark/results_size_sweep_{annotation,functional,evaluation}_v1
```

The cohort is identical in every cell — same 2,822 variants in the same order,
asserted, with AlphaMissense's column checked to be bit-identical across sizes —
so a curve over sizes is a curve over ensemble depth and nothing else. LOGO only
by default; the fixed-size benchmark already reports both protocols.

Best non-learned macro ROC-AUC per size, against AlphaMissense alone at 0.8873:

| N | best non-learned | Δ vs AM | seed SD | equal-rank AM+D1 | D1 alone |
|---|---|---|---|---|---|
| 1 | 0.8936 | +0.0062 | 0.0059 | 0.8145 | 0.6191 |
| 5 | 0.8955 | +0.0082 | 0.0030 | 0.8360 | 0.6745 |
| 10 | 0.8969 | +0.0095 | 0.0056 | 0.8484 | 0.7013 |
| 20 | 0.8977 | +0.0103 | 0.0013 | 0.8606 | 0.7301 |
| 40 | 0.8975 | +0.0101 | 0.0010 | 0.8629 | 0.7394 |
| 60 | 0.8982 | +0.0109 | 0.0010 | 0.8655 | 0.7481 |
| 90 | 0.8977 | +0.0104 | 0.0004 | 0.8674 | 0.7533 |

### Split by label source

The sweep runs per GT group, because the pooled macro over all seven genes turns
out to be close to the average of two very different answers. `--gt-group` takes
`annotation` (POLE, MSH2, MSH6, VHL, PTEN — clinical annotation), `functional`
(TP53, KRAS — functional screens), or `evaluation` (both pooled), and writes
`results_size_sweep_<group>_v1`.

Change in macro ROC-AUC against that group's own AlphaMissense baseline, for the
**pre-specified** `z_wmean0.9 · AM+D1` (not the post-hoc best):

| N | annotation (AM 0.9309) | functional (AM 0.7783) | pooled (AM 0.8873) |
|---|---|---|---|
| 1 | −0.0027 | +0.0208 | +0.0040 |
| 5 | −0.0008 | +0.0307 | +0.0082 |
| 10 | +0.0008 | +0.0294 | +0.0090 |
| 20 | +0.0013 | +0.0328 | +0.0103 |
| 40 | +0.0012 | +0.0324 | +0.0101 |
| 60 | +0.0021 | +0.0328 | +0.0109 |
| 90 | +0.0019 | +0.0315 | +0.0104 |

**5b. Essentially all of the non-learned gain is on functional-screen labels.**
On the two screened genes the fixed method adds +0.021 to +0.033 at every depth.
On the five annotated genes it adds **nothing worth quoting** — negative at N=1
and N=5, and +0.0019 at N=90. The pooled +0.0104 is these two averaged, not a
result about seven genes. This is the same pooling caveat the project already
recorded for D1 on annotation vs functional labels, now reproduced for the
fusion.

The likely mechanism is headroom, not descriptor quality: AlphaMissense is
already at 0.931 on the annotated genes and only 0.778 on the screened ones, and
the descriptors are *better* on the annotated set (D1 0.790, residue KS 0.812 at
N=90) than on the screened one (0.660, 0.616). There is simply less left to add
where AlphaMissense is strong.

Two caveats that bound this hard. The functional group is **two genes**, so its
LOGO trains on one gene and tests on the other, and KRAS brings 45 variants
against TP53's 1,174 — its seed SD is 0.018 at N=1 against 0.001 for annotation.
And the best *observed* annotation method does reach +0.0111 (`gz_mean · AM+resKS`),
but that is a maximum over the whole grid, not the pre-specified method.

### How much of this survives a variant bootstrap

`annotation_tables.py` rebuilds the full battery per gene for AM+resKS under
`z_mean`, `z_wmean0.9` and `gz_mean`, and attaches a 95% interval from a
**paired bootstrap over variants**: 2,000 draws, each resampling a gene's
variants *and* picking one descriptor seed, with AlphaMissense scored on the
same resample and the operating point held at the value fitted on the training
genes. Outputs are `TABLES.md`, a styled `tables.html`, and CSVs.

This changes the reading materially, so it is recorded here rather than left in
a results directory.

**Of 288 measured changes, 93 have an interval excluding zero.** The rest are
inconclusive — not small, *inconclusive*. Two consequences:

**16. The macro ROC-AUC gain on annotation is not conclusive.** `gz_mean` on
AM+resKS is +0.0111 with interval **[−0.0012, +0.0237]**, P(Δ>0)=0.96;
`z_wmean0.9` is +0.0072 [−0.0031, +0.0154]. Per gene, only POLE's ROC-AUC clears
zero (+0.040 [+0.021, +0.059]); MSH2, VHL, PTEN and MSH6 all straddle it. Every
macro ROC-AUC figure quoted earlier in this file is one gene carrying four that
are indistinguishable from noise, and should be read that way.

The reason is in the counts, which the first version of these tables omitted:
**PTEN has 9 negatives, MSH6 12, VHL 19.** An AUC estimated from nine negatives
cannot resolve a 0.003 difference.

**16b. One column was dead and has been replaced.** `top-10% lift` is bounded by
`min(1, R/decile)/prevalence`, which is **1.036** on a gene that is 96.5% driver.
Every score in these tables attains that ceiling, so the observed change was
exactly 0.0000 on four of the five genes and the column was conclusive nowhere.
(It is not mathematically constant — a random score does fall ~0.04 below — but
it cannot separate scores that are all reasonable.) It is replaced by
`prec@20`, a fixed follow-up budget that means the same thing on a 38-variant
gene and a 1,075-variant one, and `rare-class prec@R`, which ranks towards
whichever class is the *minority* in that gene — positives on POLE (38 of 1,075),
negatives on PTEN (9 of 257). Conclusive cells: 6/18 and 5/18, against 0/18 for
the column they replace.

Partial credit only, though: `prec@20` is still 1.000 on four of five genes,
because the top 20 of a mostly-positive gene are all positive whatever the score
does. That saturation is a property of this cohort, not of the metric, and no
top-k column escapes it. `rare-class prec@R` is the one that does move on more
than POLE alone.

**17. Average precision and precision@R do survive, for all three methods.**
Macro AP: +0.025 [+0.007, +0.049], +0.032 [+0.013, +0.057], +0.027 [+0.010,
+0.051]. Macro prec@R likewise. AP is prevalence-sensitive and POLE is 3.5%
positive, so it is the metric that registers what the fusion actually does —
re-rank a small positive class — while ROC-AUC averages it away against four
saturated genes. Across all rows, ROC-AUC is conclusive in 3 of 18 cells;
recall, F1 and accuracy at the transferred threshold in 13, 10 and 9.
**Brier, ECE and top-decile lift are conclusive nowhere.**

Both points argue for quoting AP, not ROC-AUC, as the headline on this cohort.

### Which descriptor, and on which labels

The first pass fixed most panels on AM+D1 and AM+all, which hid that **the best
input set flips between label sources**. `arm_method_grid.png` shows all 23
methods against all 8 input sets with nothing else fixed; `descriptor_marginal.png`
asks the sharper question — what does each descriptor add *on top of what is
already there* — from four different starting points.

Change vs AlphaMissense at N=90, LOGO, best observed method per cell:

(The per-arm deltas below are point estimates over the grid; §16 shows that
macro ROC-AUC differences of this size on annotation sit inside the variant
bootstrap's interval. The ordering between arms is stable; the absolute gain is
not resolved.)

| input set | annotation (AM 0.9309) | functional (AM 0.7783) |
|---|---|---|
| AM+D1 | +0.0029 | +0.0315 |
| **AM+resKS** | **+0.0111** | +0.0078 |
| AM+cKS | −0.0005 | +0.0286 |
| AM+D1+resKS | +0.0059 | +0.0186 |
| **AM+D1+cKS** | +0.0021 | **+0.0349** |
| AM+resKS+cKS | +0.0052 | +0.0162 |
| AM+all | +0.0040 | +0.0230 |

**10. resKS is the annotation descriptor; D1 and cKS are the functional ones.**
On annotated genes `AM+resKS` is the best arm by ~4x over `AM+D1`, and its
marginal contribution is positive from N≈5 on every one of the four starting
points (+0.011 to +0.022). On screened genes the ordering inverts: `AM+D1+cKS`
wins, adding resKS on top of D1 is *negative* at every size, and adding resKS to
AlphaMissense alone decays with depth rather than growing. Reporting one
"best arm" pooled across both label sources would have been wrong in both
directions.

**11. cKS is not a weak descriptor so much as a label-specific one.** On
annotation it is negative from every starting point at every size — `AM+cKS` is
the only arm that fails to beat AlphaMissense at all. On functional it adds
+0.028 on top of AlphaMissense alone and turns `AM+D1` into the best arm in the
whole grid. Its N=1 marginal is exactly zero by construction: a mean-free
statistic on one structure has nothing to be free of, so the column is constant
and cannot change any within-gene ordering.

**12. The winning method is group-specific too, and for a structural reason.**
`gz_mean` (the fold-fitted global scaler) is the best method on annotation but
one of the worst on functional (−0.04 to −0.06). Functional has only two genes,
so under LOGO its "global" scaler is fitted on a single training gene and
transfers badly; annotation has four fold groups to average over. That is a
property of the protocol, not of the descriptor.

**6. The fusion gain is bought by N=20, even though the descriptors are not.**
D1 alone keeps climbing from 0.730 at N=20 to 0.753 at N=90 and residue KS from
0.738 to 0.756, but the fused advantage over AlphaMissense is +0.0103 at N=20 and
+0.0104 at N=90 — flat within the seed spread. A descriptor that keeps sharpening
is not the same thing as a fusion that keeps improving, because the extra
sharpness lands where AlphaMissense was already right. The one thing N does keep
buying is stability: the seed SD falls from 0.0059 at N=1 to 0.0004 at N=90.

**7. Equal weighting is a depth problem, not just a weight problem.** At N=1 the
equal-weight rank fusion on AM+D1 is 0.8145, a full 0.073 below AlphaMissense; it
climbs the whole way to 0.8674 at N=90 and still ends 0.020 short. The weaker the
descriptor, the more an equal vote costs — and at small N every descriptor is
weak. `z_wmean0.9` is above AlphaMissense at every size, N=1 included (+0.0040
there). `rank_wmean0.9` is not: it sits marginally *below* at N=1 and N=5
(−0.0006, −0.0003) and only reaches +0.0009 by N=90, which is the depth cost of
throwing away magnitude on top of the weight.

**8. The pooled macro gain is two genes.** Per gene at N=90, `z_wmean0.9 · AM+D1`
against AlphaMissense alone: KRAS +0.064, POLE +0.023, PTEN and VHL +0.000,
TP53 −0.001, MSH2 −0.005, MSH6 −0.008. The equally weighted macro mean turns two
real gains into a modest average; it does not mean seven genes each improved.
Note the two are one from each label source, which is why the split above matters
and why the pooled figure alone would have been misleading.

**9. Even a single structure is not nothing.** At N=1 there is no ensemble and D1
is a one-sample statistic, yet it is already at 0.619 macro AUC and the fusion is
+0.0062 over AlphaMissense. Centred KS is the exception: exactly 0.500 at N=1, by
construction, since a mean-free statistic on one structure has nothing to be free
of.

## Where it shows up

The results gallery cards these under **Cross-protein analyses** as two separate
subpages — `metric_analysis/results/NONLEARNED_FUSION/index.html` for the size
sweep, which shows no fitted model at all and is the one to read first, and
`metric_analysis/results/FUSION_BENCHMARK/index.html` for the fixed-size
comparison against the learned models. The sweep page leads with the
annotation/functional split and keeps the pooled run as a subsection:

```bash
python -m \
  metric_analysis.pipeline.orchestration.collect_results
```

The subpage reads `tables/headline.csv` rather than re-deriving "best of each
kind" from the 48-metric grid, so it cannot drift from the table above; regenerate
this experiment first and the gallery second. `results_v1/` is mirrored whole
under `FUSION_BENCHMARK/report/` (minus `scores.npz`, which nothing links), so the
interactive report works from the gallery with its figures and downloads intact.
`--fusion-benchmark-dir` and `--nonlearned-sweep-base` point the collector at
different results directories, and both are silently skipped when absent, like
every other optional input.

## Scope and caveats

- One descriptor seed and one ensemble size (N=90, seed 42). No sampling
  uncertainty is quantified here.
- Sorted tops are post-hoc maxima over the screen, not nested-validation
  estimates of a selected method. The comparison between *kinds* is the result;
  the individual winner is not.
- `gz` under stratified CV refits its scaler per fold, so a gene split across
  five folds gets five affine maps and its rows are not strictly comparable
  within the gene. The learned models have exactly this property, which is why
  `gz` is kept as their non-learned counterpart.
- The random reference is a single draw (seed 42), so its macro AUC is 0.479
  rather than 0.500. It illustrates the floor; it does not estimate it.
- Brier and ECE are reported only for scores that claim to be probabilities and
  actually lie in [0, 1]; both SVM families are blank by construction.
