# Plan: ensemble precision, variant ordering, seed diversity and AF3 ranking

Status: execution authorized and started locally on 2026-10-18. The initial plan is frozen in plateau_20261018/plan_at_start.md.

## Questions

1. Does AUC flatten because KS scores are already precise, or because further precision no longer changes driver–passenger ordering?
2. At fixed conformation count, does spreading predictions across more seeds improve score precision, ranking stability or AUC?
3. Does selecting conformations using AF3's own ranking_score affect those outcomes?

The third question concerns conformation selection. Merely reordering an unchanged set cannot affect KS. AF3 ranking_score must not be replaced with mean pLDDT. Variant ranking by KS and conformation ranking by AF3 are separate quantities.

## 0. Freeze inputs and reconcile the existing analyses

- Six annotation genes: POLE, PTEN, VHL, MSH2, MSH6, PIK3CA; 1,799 labelled variants. Use the strict canonical labels, without label-dependent exclusions or changing score direction.
- Verify the POLE repair against actual configuration and source checks. The documented repair pins POLE_268-471_WT; the historical Figure 2 route pooled three WT sources. The canonical descriptor route already used a single WT. Historical pooled outputs must not enter the new curve.
- Inventory source files, hashes, WT identity, residue indices, seed/sample identities, duplicate sources, effective N on each side, AF3 ranking coverage and coordinate coverage.
- Reproduce representative corrected Figure 2 cells before extending the curve. Document the additional sampler difference: the repaired Figure 2 route shares one WT draw across variants per gene/repeat; the canonical/pilot route draws WT separately for each variant. Use the corrected Figure 2 sampling contract for the primary plateau analysis; keep other contracts explicitly separate.
- Freeze the same variant set, labels, WT reference, residue scope and descriptor across every size/policy within an analysis. Use the same complete-case coordinate cohort at all sizes and policies; list exclusions. The pilot excluded POLE P356S and MSH2 G162R from coordinates.
- For the controlled equal-count seed analysis, use the common 25 seeds × 5 samples. Eight POLE variants have additional predictions; do not silently give those variants a larger candidate pool. Reconcile this restriction with the reproduced curve and present any changed-pool curve separately.

## A. Score precision and variant-ranking stability versus N

Use existing size grid N = 1, 5, 10, 20, 40, 60, 90, plus N=25 to connect to the pilot and N=125 as the full-pool endpoint where feasible. Use 200 reproducible nested draws without replacement; pairing across sizes follows the frozen sampling contract.

Primary descriptor: existing raw per-residue pLDDT KS averaged over the fixed residue scope. Parallel secondary analysis: existing radial-distance coordinate KS over the fixed core residues in the core-fit frame. Do not change coordinate alignment, distance centre or residue masks with N or policy.

For every gene and size, retain per-variant/per-draw scores and calculate:

- Score SD per variant; summarize its distribution, also separating drivers and passengers.
- Tie-aware variant-ranking agreement between draws (Spearman correlation), plus agreement with the full-pool ordering. Mark correlations undefined for constant score vectors; particularly audit the heavily tied N=1 case.
- For every driver–passenger pair, the frequency of correct ordering, ties and reversed ordering. Correct ordering receives 1, ties 0.5, reversed ordering 0. Report continuous probabilities first; any categories of stable/unstable pairs must have thresholds fixed before examining outcomes.
- Mean per-draw AUC and SD. Verify that mean pair-order correctness equals mean per-draw AUC. Do not replace this with AUC of draw-averaged scores.
- Paired AUC difference from the largest evaluated size. Give equal-gene macro results in addition to per-gene results, without pooling raw scores across genes.

Interpret the full-pool result as a conditional stability reference. At N=125, drawing the same full pool makes computational draw variance zero mechanically. Smaller draws overlap the reference; this is not independent validation or evidence that additional AF3 seeds cannot help.

## B. Direct equal-count seed-diversity experiment

At fixed N, compare:

1. Spread: allocate samples across as many seeds as possible, as evenly as possible; randomize which seeds receive the extra samples and sample identities within seeds.
2. Concentrated: fill complete five-sample seeds first, with one randomly chosen partial seed if N is not divisible by five.
3. Uniform-random conformations: retain the pilot policy as a bridge, recording actual seed counts and overlap.

Examples: N=10 is 10×1 versus 2×5; N=25 is 25×1 versus 5×5; N=40 is 15 seeds×2 plus 10 seeds×1 versus 8×5. Spread and concentrated converge to the same full set at N=125.

Evaluate the policies under two explicit WT–mutant identity regimes:

- Independently selected WT and mutant samples, matching the primary sampling intent; record actual shared seeds and shared seed/sample identities.
- Exactly matched WT–mutant seed/sample identities, applying the same selection on both sides. This controls the fraction of shared identities while varying diversity.

Keep the original disjoint-whole-seed result as a sensitivity analysis only at feasible N. Do not claim disjoint sets at N=125 from a shared 25-seed pool. Neither independence nor matched identities eliminates every interaction with seed coupling; report the policy-by-pairing interaction instead of attributing all differences to variance.

Measure all outcomes from A, using paired comparisons on identical variant cohorts. Reproduce the N=25 pilot under its original sampler before comparing it with the new spread policy or the Figure 2 sampler.

## C. Added question: selection using AF3's own ranking_score

- Recover the recorded ranking_score by exact job/seed/sample identity from ranking_scores.csv; use per-sample summary_confidences.json only as a documented fallback. Reuse the archive parsing/provenance conventions in af3_jobs/local_runner/merge_seed_extensions.py. Do not infer ranking_score from pLDDT or silently mix rounded fallback values with full-precision scores.
- Audit availability, score precision and ties; use a fixed seed/sample tie-break. Freeze a common complete-ranking cohort for every comparison; report any difference from the full cohort.
- Compare uniform random selection with AF3 top-N selection. Separate independently ranked WT/mutant selection from WT-ranked identities applied to both sides; include a random selection with the same identity-pairing rule as each ranked arm.
- Top-N from a fixed complete pool is deterministic: its across-draw SD is not evidence of improved precision. Report its AUC separately.
- To test variability under ranked selection, repeatedly draw a candidate pool of M=min(2N,125), then compare AF3 top-N with uniform N from the exact same candidate pool. Report M and N separately. At M=N=125, the two methods coincide.
- Record selected seed counts/overlap; add a seed-allocation-matched ranking control where feasible to distinguish confidence selection from seed-diversity changes. Do not treat changes in independent WT/mutant selection and WT-pinned selection as a pure ranking effect.
- No ranking cutoffs, descriptors or residue selections will be tuned to labels.

## Uncertainty and interpretation

- Computational draw SD measures conditional resampling sensitivity, not biological replication.
- For AUC differences, use paired position-cluster bootstrap within each gene: resample mutation positions, retain substitutions at each selected position together, and apply identical cluster multiplicities across sizes/policies. Keep mixed-label positions intact; report and exclude bootstrap replicates with only one class. Proposed default: 2,000 bootstrap resamples, with Monte Carlo adequacy checked.
- Report position-cluster uncertainty separately from draw variability. Preserve the shared-WT dependence and paired draw identities when computing contrasts.
- Do not declare a plateau from overlapping adjacent error bars. Proposed practical tolerance is 0.005 AUC, matching the existing convergence utility; freeze that tolerance before results and assess paired differences from the largest size. Failure to rule out a material gain means unresolved convergence, not proof of no gain. Endpoint evidence remains conditional on stored predictions.
- Investigate conservation, substitution class and structural region only after precision and ordering analyses. Treat explanations of consistently reversed pairs as within-gene exploratory analyses; do not retrospectively optimize the descriptor.

## Implementation and validation

Reuse:

- metric_analysis/pipeline/library/canonical_cohort.py: cohort and recorded sources.
- metric_analysis/pipeline/library/descriptor_metrics.py: KS definitions/backends.
- metric_analysis/pipeline/library/descriptor_sampling.py: reproducible sampling utilities and residue blocks.
- experiments/seed_structure/sampling_auc.py: fixed-N pilot, paired score exports and AUC summaries.
- experiments/pole_wt_repair/{README.md,publication/README.md}: corrected reference/sampling provenance and reproduction targets.
- drift/convergence.py: inspect existing summaries; its draw-based intervals do not substitute for position-cluster uncertainty.

Add isolated runners and outputs under experiments/seed_structure/ for size/ordering analysis, balanced seed policies, AF3-ranking extraction, clustered contrasts and figure assembly. Prefer shared existing metric and source readers; do not modify labels, scientific definitions or publication outputs.

Validation: exact N per side; no replacement; seed-allocation constraints; explicit identity matching; WT source and residue alignment checks; scalar-SciPy KS parity including ties; AUC/pair-order equality; score and AUC invariance to column reordering; position-cluster resampling integrity; identical cohorts across paired policies; reproducible RNGs; reproduction of selected repaired curve cells and pilot cells. Validate the full-pool policy equivalence and deterministic top-N interpretation.

## Deliverables and order

### W&B tracking

W&B is disabled at the user's request. All execution and outputs remain local. The prepared optional logger targets workspace `danraviv-research`, project `ensemble_plateau_analysis`; it has not uploaded results. Local CSVs remain authoritative. The completed N=25 pilot is a separate `seed_sampling_pilot_20261018` group, uploaded by `log_wandb.py`. It is not the planned multi-size plateau experiment.

For new analyses use separate groups for the input audit, size/ordering analysis, balanced seed policies, and AF3-ranking selection. Run configurations must record gene, descriptor, residue scope, ensemble size, AF3 seed allocation, WT–mutant identity policy, computational RNG seeds, repeat count, input hashes and cohort exclusions. Define ensemble size as the curve axis; log AUC mean/SD, per-variant score variability, ranking stability and pair-order summaries. Keep position-cluster intervals distinct from computational draw variability. Upload summary tables, figures, validation and provenance artifacts; do not upload raw structural archives. Distinguish progress from final scientific results and never mark a planned stage complete merely because tracking is initialized.

### Execution sequence

1. Input/sampler/ranking-coverage audit and frozen provenance manifest.
2. Analysis A: per-gene precision, ranking stability, pair ordering and AUC curves.
3. Analysis B: equal-count seed-policy comparisons and pairing sensitivity.
4. Analysis C: AF3-ranked selection comparisons, with deterministic and stochastic arms distinguished.
5. Compact supplementary figure: precision, ranking stability and AUC versus N, followed by seed-policy and AF3-ranking contrasts. Put per-gene details, pair probabilities, exclusions and uncertainty procedures in supporting tables.

This plan does not schedule new AF3 inference or manuscript changes. The archive-coverage audit determines whether ranking analysis is feasible for all six genes.

## Frozen execution details

- A/B use 200 draws, all nine sizes and both independent/matched identities. Their common 25-seed candidate pool intentionally differs from the historical curve for eight POLE mutant pools with extra predictions; representative original corrected cells are reproduced separately.
- C uses N=10,25,40,90,125, with the corresponding candidate pool M=min(2N,125). It includes deterministic full-pool top-N, stochastic candidate-pool selection, and seed-allocation-controlled ranking arms.
- Pair-consistency summaries use a predeclared 0.95 probability threshold; full pair probabilities are saved.
- Four methodological tests initially passed; a fifth tests AF3-ranked selection and seed-allocation controls.
