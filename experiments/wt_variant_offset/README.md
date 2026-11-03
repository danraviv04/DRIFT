# WT-vs-variant offset in MSH2 and PIK3CA (2026-10-15)

**Question.** In MSH2 and PIK3CA every variant, driver or passenger, has a residue
KS-score of ~0.55-0.60 at N = 60 against its WT. The WT-vs-WT background is ~0.19
(SI Figure S4, `experiments/si_figures/wt_backgrounds.py`). Why, and why only these
two genes?

## TL;DR

AF3 throws out any template that is an exact, near-full-length copy of the query.
For MSH2 and PIK3CA the PDB has such a structure of the wild type, so the **WT jobs lost
it and every variant kept it**. The WT therefore sits apart from all variants, and every
variant scores ~0.57 regardless of class. MSH2's WT file also stores the wrong atom
(N instead of Cα). Fix: regenerate both WTs using the variants' templates.

| gene | WT vs variant templates | WT off at % of residues | AUC now | AUC with fix (estimate) | regenerate WT? |
|---|---|---|---|---|---|
| MSH2 | differ (2O8B missing from WT) + wrong atom | 63% | 0.80 | 0.88 (Cα fix alone) | **yes** |
| PIK3CA | differ (4L1B, 5DXH missing from WT) | 49% | 0.66 | ~0.74 | **yes** |
| VHL | 1 of 4 differs | 10% | 0.91 | — | optional |
| PTEN | 1 of 4 differs | 10% | 0.87 | — | optional |
| POLE | identical | 0% | 0.86 | — | no |
| MSH6 | identical | 0% | 0.90 | — | no |

## Answer

**The WT jobs were conditioned on different structural templates than their variants.**
AF3's template search keeps the top 4 hits. Before that it drops any hit whose aligned
sequence is an exact substring of the query covering more than 95% of it, a
"duplicate" of the query. This is AF3's `Hit.keep`, with `max_subsequence_ratio = 0.95`
and `max_hits = 4` in `DataPipeline`. When the PDB holds a near-full-length structure
whose sequence is exactly the WT construct, three things follow:

- That structure is removed from the WT job only.
- Every variant keeps it, because one substitution breaks the exact match.
- The WT gets a different fourth (or third and fourth) template.

The input JSONs are identical apart from the coded residue, as the 10-12 drive audit
found. Templates are picked at run time, so that audit could not see this.

Scored chain, read from each archive's job-level `_data.json`
(`results/template_comparison.csv`, `results/provenance/`):

| gene | WT templates | variant templates (same set and hash in every variant read) | WT outside variant 95% band |
|---|---|---|---|
| PIK3CA (p110α) | 3HHM 4JPS **4A55 4WAF** | **4L1B** 3HHM 4JPS **5DXH** (A1035T, K532R) | 49% of residues |
| MSH2 (MutSα chain A) | 2O8E 6I5F 7AI5 **7AI6** | **2O8B** 2O8E 6I5F 7AI5 (A256G, Q629R) | 63% |
| VHL | 4B9K 3ZRC 1LQB **1LM8** | **4WQO** 4B9K 3ZRC 1LQB (D121V) | 10% |
| PTEN | 7JVX 7JTX 7JUL **1D5R** | 7JVX 7JTX 7JTX 7JUL (A121E) | 10% |
| POLE 268-471 | 4PTF 6I8A 5OKI 3A2F | identical (P286R) | 0% |
| MSH6 (G:T, chain B) | 2O8B 3THW 6I5F 6I5F | identical (A1055P) | 0% |

- 2O8B is the full MutSα complex.
- 4L1B and 5DXH are full-length p110α/p85.
- The MSA is not the difference. Within each gene, WT and variants have the same
  depth (±0.1%, e.g. PIK3CA 14,303 vs 14,291-14,303 sequences), and the variants
  differ from each other as much as from the WT.
- Run date is not the difference either. The MutSα WT ran on 03-24, between variants
  run on 03-23 and 04-20. The PIK3CA WT ran on 09-22, after its variants (09-20 and 09-21).
- Software and hardware numerics are not the difference. The WT's seed-k predictions
  are still correlated with the variants' seed-k predictions (matched-column r: MSH2
  0.38, PIK3CA 0.46), as they are in the clean genes (0.46-0.74). A different AF3
  version or GPU kernel would break that pairing (`seed_pairing.py`).

**Internal control (MSH2).** In the MSH2 jobs the MSH6 chain is unmutated. It got the
same templates (hash 1a55d3337ddc) and the same MSA as in the WT job, so only the MSH2
chain's templates differ. On the MSH2 rows variant KS is 0.44 and 109/109 variants are
above the WT-WT p95. On the MSH6 rows of the same jobs it is 0.227 (WT-WT median 0.186)
and 34/109 are above (`msh6_rows_control.py`). The offset follows the template swap,
chain by chain.

### A second, MSH2-only defect: the WT CSV holds the wrong atom

`PLDDTs/MutS_alpha/MutS_alpha_WT_plddt_values.csv` (committed 06-02, de0501e2) holds
the **backbone N** pLDDT of each residue. Every other WT CSV, and every variant CSV,
holds the **Cα** pLDDT.

- Matched against the cached WT model `_rmsf_wt_reference.cif`, the N convention has
  max error 0.000 and the Cα convention 5.3 (`wt_csv_convention.py`; all other genes
  match on Cα).
- The CSV it replaced, `MutS_alpha_MSH2_WT_plddt_values.csv` (git de0501e2^), holds the
  same 125 predictions as Cα: matched-seed r = 0.993, Cα max error 0.000.
- N and Cα differ by 0.83 pLDDT per residue on average, so this is the smaller of the
  two effects. On its own it moves MSH2 variant KS from 0.59 to 0.44.

## Where the offset lives (question 1)

`region_profile.py` → `results/wt_vs_variant_profile.png` and `.csv`, plus `locate_offset.py`.

- It is not uniform, and it is not one locus. It runs in domain-sized blocks of both
  signs.
  - MSH2, N-terminal mismatch-binding domain (1-100): the WT is 3-10 pLDDT below every
    variant.
  - MSH2, most of the remaining chain: the WT is 0.5-2.5 pLDDT above. 96% of variants
    have a lower mean than the WT.
  - PIK3CA: the C-terminal tail (1051-1068) is 11 pLDDT lower in the WT, with smaller
    blocks around 201-250 and 351-400. 99% of variants have a higher mean than the WT.
  - PIK3CA: the WT ensemble is also wider (per-residue SD 0.65 vs 0.41).
- The WT lies outside the central 95% of variant means at 63% (MSH2) and 49% (PIK3CA)
  of residues. The figure is 10% in PTEN and VHL, and 0% in MSH6 and POLE.
- The variants agree with each other. Variant-vs-variant residue KS (125 v 125) has a
  median of 0.26 (MSH2) and 0.30 (PIK3CA), and the most typical variants are 0.18-0.23
  apart. Every WT pool lies outside that cloud: the nearest variant is 0.53 from the
  MSH2 WT and 0.52 from the PIK3CA WT (`batch_structure.py`). The WT is the outlier,
  not the variants.

## Effect on per-gene AUC (residue KS, N = 60)

My sampler reproduces production: MSH2 0.80 and PIK3CA 0.66, against `wt_background_variant_scores.csv`
values of 0.803 and 0.665.

| | MSH2 | PIK3CA |
|---|---|---|
| official WT (production) | 0.80 | 0.66 |
| same WT predictions, Cα extraction (fixes defect 2 only) | **0.88**, Δ CI [+0.02, +0.14] | n/a (already Cα) |
| in-batch reference: the 10 most typical other variants, label-blind | 0.84, Δ CI [−0.05, +0.14] | 0.74, Δ CI [−0.04, +0.22] |
| rank agreement of variant scores, official vs in-batch (Spearman) | 0.60 | 0.59 |
| spread of variant KS (SD) | 0.035 | 0.020 |

Controls for the in-batch reference: in clean genes it reproduces the WT ranking
(Spearman MSH6 0.93, POLE 0.89, PTEN 0.84, VHL 0.79). Because its pool contains
mutants, it costs AUC there, from MSH6 −0.01 to VHL −0.20. So the in-batch numbers
for MSH2 and PIK3CA are conservative (`auc_effect.py`, `results/auc_effect_summary.json`).
Clean genes have a KS spread (SD) of 0.06-0.21.

**This is not a uniform shift.** A per-gene additive offset would leave AUC and ranks
unchanged. Instead:

- Per residue, the KS saturates wherever the WT is displaced, which compresses the
  score range two- to five-fold.
- A mutation that moves a displaced residue *toward* the WT lowers its score.
- The variant ranking is reshuffled: Spearman 0.6, against ≥0.79 for the same check in
  clean genes.

The AUC cost is about 0.05-0.08:

- It is significant for MSH2's extraction fix alone (+0.08).
- It is not significant per gene for the template swap. PIK3CA has only 7 passengers.

The direction is consistent: the offset WT under-reports MSH2 and PIK3CA.

## Do these WTs need regenerating? (question 3)

**Yes, MSH2 and PIK3CA.** No existing pool can stand in:

- The 04-02 MutSα WT CSV is the same 125 predictions, extracted as Cα. Its 0.47 KS from
  the production CSV is purely the extraction convention, and it is still 0.42-0.45 from
  the variants.
- The MSH6_GT jobs' MSH2 chain has the WT-only template set (hash 77383707314c, the same
  as the MutSα WT), and it uses a different DNA. It is 0.57 from the MSH2 variants.
- No variant is effectively WT: every variant keeps the duplicate template.

How to regenerate:

1. Build the WT job **MSA- and template-ready**, from a typical variant's finished
   `_data.json` (e.g. MSH2 Q629R, PIK3CA K532R):
   - Revert the query sequence and the MSA query rows to WT.
   - Keep that variant's `templates` block unchanged. A single substitution does not
     change the template query/template index maps.
   - Run it with `--run_data_pipeline=false`. This is the same route as
     `python -m af3_jobs.seed_extension`.

   The WT then sees exactly the variants' templates. 25 seeds × 5 samples, one job per
   gene. MutSα took ~5 h per job on Athena.
2. Do **not** turn the duplicate filter off for the WT only. Variants would still differ
   wherever the filter removes a mutant PDB entry that matches them exactly.
3. MSH2, right now and at no cost: re-extract the MutSα WT CSV as Cα, for example by
   restoring the de0501e2^ CSV, which is the same predictions. This fixes defect 2
   before any regeneration. I did not change the tracked file; that is the user's call.

**VHL and PTEN** carry a one-template asymmetry (4WQO; 1D5R vs a second 7JTX chain). It
shifts the WT at ~10% of residues, against 49-63% in MSH2 and PIK3CA. Their drivers sit
far above it (AUC 0.87-0.92), so it is minor. It should still be noted, or removed the
same way when convenient.

**POLE and MSH6** are clean.

**Not checked:** TP53 and KRAS (functional genes). Any gene whose exact WT construct
exists in the PDB at ≥95% length is exposed. Check its WT `_data.json` templates
against one variant's with `archive_provenance.py` before trusting its WT background.

## Recommended SI Figure S4 caption paragraph

> In MSH2 and PIK3CA every variant, driver or passenger, lies above the WT-WT background.
> This is a property of the wild-type reference, not detection. AlphaFold 3's template
> search discards any template that is an exact, near-full-length copy of the query
> (`max_subsequence_ratio` 0.95). For these two genes the PDB holds such structures of
> the wild-type construct: full MutSα (2O8B) and full p110α/p85 (4L1B, 5DXH). They were
> therefore excluded from the wild-type predictions but used for every variant, whose
> single substitution breaks the exact match. The wild-type ensemble is shifted at about
> half of all residues relative to the variants, which agree closely with one another.
> All variant scores in these genes therefore carry a shared, non-uniform offset. Their
> position against the background cannot be read as detection above prediction noise,
> and their within-gene AUCs (0.80, 0.66) are, if anything, underestimates. In the other
> genes the wild-type and variant jobs used identical (POLE, MSH6) or near-identical
> (PTEN, VHL; one of four templates differs) templates.

If the WTs are regenerated before submission, replace this paragraph with one Methods
sentence: "wild-type jobs were run with the variants' template set, so that AF3's
duplicate-template filter treats both alike."

## Files

Every script runs as `PYTHONPATH="." python experiments/wt_variant_offset/<script>`.

| script | output in `results/` |
|---|---|
| `locate_offset.py` | `locate_summary.json`, `per_residue_profile.csv` (MSH2, PIK3CA + 4 controls) |
| `region_profile.py` | `wt_vs_variant_profile.{png,csv}`, `_summary.csv` |
| `batch_structure.py` | `pairwise_ks_{MSH2,PIK3CA}.csv`: variant-variant KS matrix + the WT pools |
| `seed_pairing.py` | `rerun_2026-10-19/seed_pairing.json`: same-seed correlation, WT vs variants vs variant pairs, on the current WTs (source of SI Note S6). The top-level `seed_pairing.json` (10-15 10:31) is SUPERSEDED: it predates the MSH2 Cα WT (14e9d0c6) and the MSH2/PIK3CA template-matched WTs (5811e9e5); do not quote its WT–variant 0.38–0.74 |
| `wt_csv_convention.py` | `wt_csv_convention.json`: which atom each WT CSV holds |
| `msh2_convention_auc.py` | `msh2_convention_{auc.json,scores.csv}` |
| `msh6_rows_control.py` | `msh6_rows_control.json` |
| `auc_effect.py` | `auc_effect_{summary.json,variant_scores.csv}`: official vs leave-one-out vs typical-variant reference, 6 genes |
| `archive_provenance.py` | `provenance/*.json`: **reads the Elements drive**; run one archive at a time with no other reader |
| `read_drive_when_free.sh`, `read_drive_controls.sh` | the waits used on 10-15 |
| `template_table.py` | `template_comparison.csv` |

The MSH6_GT WT templates come from the local copy
`~/Downloads/MutS_alpha_MSH6_GT_WT_data/.../MutS_alpha_MSH6_GT_WT_data.json`
(chain B hash 1a55d3337ddc, identical to its variant).

## TP53 and KRAS (checked 2026-10-15, `functional_genes_check.py --drive`)

| gene | WT vs variant templates | WT outside variant 95% | regenerate WT? |
|---|---|---|---|
| TP53 | WT lacks **6XRE** (100% chain coverage), which the variants keep (A119D, R175H) | 32% of residues | **yes**: `muts_jsons/P53_WT_vartmpl.zip`, from A119D |
| KRAS | sets differ per variant, but the WT keeps full-length templates (7ROV 100%, 4DSN 99.5%) | 9% | no (VHL/PTEN level) |

Outputs: `results/functional_genes_wt_offset.csv`, `results/functional_genes_template_coverage.csv`.
