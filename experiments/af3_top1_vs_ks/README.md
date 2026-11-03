# AF3's best structure against the pLDDT KS-score (2026-10-21)

A redo of Supplementary Note S8 with **one selection rule and one metric**, both fixed before any result was seen. It replaces the medoid / nearest-to-mean / ensemble-mean comparison, which tried several representatives and two metrics.

**Question.** If each sequence is represented by the single structure AF3 itself ranks best, does comparing the WT and mutant structures separate drivers from passengers as well as the KS-score does on the same predictions?

## Design

- **Selection: AF3's top-ranked prediction.** Within each production N=90 draw, the WT and mutant predictions with the highest recorded AF3 `ranking_score`. Each side is chosen independently, and the WT pick is shared by all variants of a gene. This is AF3's own "best structure". It needs no coordinates, superposition or medoid definition. No ranking score was tied at the top (0 of 18,040 picks).
- **Metric: mean absolute pLDDT difference (MAE)** between the two picks, over the residues the KS-score uses. This keeps the signal source fixed, so the only difference from the KS-score is one prediction per side against the whole distribution.
- RMSD was rejected for two reasons. It would also switch from confidence to coordinates. And Note S1 already shows that coordinate signals are weak even with full ensembles.
- **Same predictions as the KS-score.** The ten published N=90 draws are those of the POLE 25-seed rerun, cache fingerprint `49028c16cbb3`. Both scores use the same 90 predictions per side, so the comparison holds compute fixed.
- **Cohort:** all 1,798 labelled variants (612 drivers / 1,186 passengers). Coordinates are not needed, so MSH2 G162R is included, unlike in the old S8.
- **Statistics:** the Note S8 estimator, unchanged (`n90_comparison.uncertainty`). Per-gene AUC is averaged over the ten subsampling seeds, and the macro is the equal mean of the six genes. Intervals use the paired hierarchical bootstrap over residue positions: 2,000 replicates, rng 20261019, with the seed indices resampled jointly.

## Results

ROC-AUC at N=90, with 95% intervals:

| Gene | Drivers / passengers | KS-score | AF3 top-1 MAE | KS − top-1 |
|---|---|---|---|---|
| POLE | 38 / 1,036 | 0.858 (0.795–0.910) | 0.806 (0.734–0.866) | 0.053 (0.009–0.096) |
| PTEN | 248 / 9 | 0.872 (0.745–0.955) | 0.719 (0.608–0.836) | 0.153 (0.011–0.279) |
| VHL | 105 / 19 | 0.916 (0.838–0.974) | 0.752 (0.648–0.847) | 0.164 (0.092–0.246) |
| MSH2 | 71 / 38 | 0.848 (0.743–0.932) | 0.614 (0.529–0.697) | 0.234 (0.127–0.335) |
| MSH6 | 66 / 77 | 0.903 (0.848–0.950) | 0.539 (0.486–0.592) | 0.363 (0.292–0.432) |
| PIK3CA | 84 / 7 | 0.707 (0.521–0.874) | 0.562 (0.369–0.738) | 0.145 (−0.078 to 0.385) |
| **Macro** | 612 / 1,186 | **0.851 (0.806–0.889)** | **0.665 (0.621–0.709)** | **0.186 (0.133–0.236)** |

- The KS-score was higher in all six genes, and the interval excludes zero in five. PIK3CA, with only 7 passengers, is the exception.
- The top-1 score was also less stable across subsampling seeds: s.d. of the macro AUC 0.019 against 0.004.
- **POLE** is where the single structure did best (0.81) and the gap is smallest. It is a small, rigid construct: its rigid core is the whole chain (Note S1).
- **MSH6** is where top-1 is at chance (0.54) while the KS-score reaches 0.90. MSH6 is the largest system, with blocks placed differently across predictions (Supplementary Fig. S1d). One structure per side there mostly reports which placement was drawn.

## Conclusions

1. AF3's own best structure per sequence does not substitute for the ensemble. On the same 90 predictions per side, comparing the top-ranked WT and mutant structures reached a macro AUC of 0.67, against 0.85 for the KS-score.
2. The ensemble's value is therefore not that it helps find a better single structure. Even the structure AF3 ranks best among 90 loses most of the signal that the distribution of those 90 carries.
3. The gap is largest where the predictions vary most between runs (MSH6, MSH2) and smallest in the rigid POLE domain. This is consistent with a single prediction's pLDDT difference being dominated by run-to-run variation, which the distribution averages out.

**Context, not part of this test.** The existing N=1 results give a random single prediction per side a macro AUC of about 0.62 (D1) or 0.61 (KS). AF3's top pick (0.67, MAE) is somewhat better than a random pick. Those numbers use a different metric and draws, so they are a rough reference only.

**Caveats**
- `ranking_score` favours confident structures. Choosing the most confident prediction may itself compress WT–mutant pLDDT differences, so the result describes AF3's pick as it is, not every possible single structure.
- MAE and the KS-score differ in scoring function as well as in using one prediction against many. Holding the signal source fixed removes the biggest confound, but not that one.
- Intervals are conditional on these six datasets and prediction pools, with no multiplicity adjustment.

## Reproduction checks (all pass, `run.log`)

- KS was recomputed directly on every variant × draw (17,980) and matches the published cache to ≤1.1e-16. The cached value is the one used, because ~1e-16 differences flip exact ties: POLE has KS-scores of 1.0.
- KS point AUCs equal the published main-text per-gene AUCs (all six genes) and Note S8 (the five genes without the extra MSH2 variant).
- KS intervals equal Note S8 for VHL, POLE and PTEN. For MSH6 and PIK3CA they cannot match: the extra MSH2 variant changes MSH2's position clusters, which shifts the bootstrap random stream for the genes drawn after it.
- Singleton clusters reproduce the variant-level intervals exactly (built-in check).
- Every ranking file is `complete`, from `ranking_scores.csv`, and matched by SHA-256 to the pLDDT source it was extracted against. For the eight POLE 25-seed copies, the match is to their 160-prediction originals.

## Files

- `af3_top1_vs_ks.py`: the whole analysis, about 60 s. Run from the repo root with `PYTHONPATH="." python experiments/af3_top1_vs_ks/af3_top1_vs_ks.py`. It refuses to overwrite `results/`.
- `results/auc_summary.csv` and `results/paired_contrasts.csv`: the published unit (positions). The `*_variant.csv` files are variant-level, kept for the record.
- `results/variant_scores.csv.gz`: per variant × draw, the KS-score and the top-1 MAE.
- `results/selections.csv`: every pick with its ranking score.
- `results/provenance.json`: design and input hashes.
- Inputs are read, never written: the POLE 25-seed cache in `experiments/annotation_stages/cache` (gitignored, so it exists only there), and the AF3 ranking scores in `experiments/seed_structure/plateau_20261018/af3_ranking`.
