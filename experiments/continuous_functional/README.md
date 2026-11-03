# Continuous functional analyses

**Functional analyses are continuous only (2026-10-13).** No assay is ever
thresholded into a driver/passenger label anywhere in the project:

- **TP53 and KRAS have no binary label.** The former cuts (TP53
  `Phenotypic_Selection_Model >= 3.25` / 75th percentile; KRAS `HA1E_WT_LFC > 0.68`
  driver, `<= 0.1` passenger) are deleted from the code, the configs and the
  manifest. Both genes are disabled in the binary pipeline
  (`label.mode = "continuous_assay"`).
- **PTEN and MSH2** keep their ClinVar label for the annotation (binary) analyses;
  their measured assay is used here as a continuous target.
- The binary `functional` GT group, `--gt-group functional`, and the mixed
  `evaluation` group are retired. Binary results are annotation-only.

## The four targets

| gene | source | target (higher = stronger cancer-relevant effect) | cohort |
|---|---|---|---|
| TP53 | Giacomelli 2018, `Ground_Truths/Supplementary table 3.xlsx` | log10 `Phenotypic_Selection_Model` | every modelled variant with a score (1,174) |
| KRAS | Kwon 2024, `Ground_Truths/KRAS/Extended Data Table 1.xlsx`, "KRAS GOF DMS Screen" | signed `HA1E_WT_LFC` | KRAS4B blind stratum, 5 random substitutions per position (935) |
| PTEN | Mighell 2018, Table S2 → `Ground_Truths/PTEN/functional_scores.csv` | `-Cum_score` | canonical ClinVar cohort with a measured score (253) |
| MSH2 | Jia 2021, TableS5_AllVariants → `Ground_Truths/MSH2/functional_scores.csv` | `LOF score` | canonical ClinVar cohort with a measured score (106) |

KRAS's 44 `hotspot_only` variants (`experiments/kras4b_coverage/results/batch_strata.csv`)
were generated because of their assay values, so they are not a sample of the
screen and are excluded. Initiator-codon (position 1) variants are excluded for
every gene.

## Descriptors: `dataset.py`

TP53 and KRAS are not in the binary pipeline, so their descriptors cannot come
from the labelled fusion caches. `dataset.py` computes AlphaMissense, D1 and the
ensemble-shift descriptors (residueKS, profileKS, ...) for the four genes with
the **same production functions, configs, sampling policy and draw keys** as the
binary pipeline (`core.build_d1_table`, `ensemble_shift.build_ensemble_shift_features`,
`protein_config.load_continuous_configs`). Checked on 2026-10-13: TP53, PTEN and
MSH2 descriptors are bit-identical to the old production caches. Caches go to
`cache/continuous_size{N}_seed{S}_..._proteins{fingerprint}_esfeat6.csv`
(~70 s per cell).

## Descriptive associations

- `plot_four_gene_ks.py` → `four_gene_ks/`: residueKS (N=90, mean over seeds
  42–51) vs each assay, Pearson/Spearman/OLS R² with 2,000 residue-cluster
  bootstrap CIs, plus PTEN high-confidence, MSH2 non-splice and TP53 raw-score
  sensitivity rows.
- `plot_four_gene_am.py` → `four_gene_am/`: AlphaMissense vs each assay on the
  same variants (and AM vs residueKS side by side), and across every
  assay-measured missense variant (whole screens). KRAS uses the P01116-2 (4B)
  AM file.

## Ridge head: `run_continuous.py`

Regresses each gene's assay on AlphaMissense plus one structural descriptor
(`afm_d1`, `afm_ks` = profileKS, `afm_resks` = residueKS) across sizes
`1,5,10,20,40,60,90` × seeds `42-51`.

| model | features | R² meaningful |
|---|---|---|
| `ridge` | engineered6 AFM+partner | yes |
| `ridge_raw2` | AFM + partner | yes |
| `ridge_afm` | AFM alone, rescaled | yes |
| `ridge_partner` | partner alone, rescaled | yes |
| `afm`, `partner` | raw score, unfitted | no — wrong scale |

Schemes: `within_gene` (KFold 5), **`within_gene_pos`** (GroupKFold on residue
position — the one to quote), `logo` (leave-one-gene-out on a within-gene
percentile-rank target; four folds, directional only).

`spearman` is on the pooled out-of-fold vector; **quote `spearman_fold`** (mean of
per-fold values). Pooling mixes separately fitted models and can flip a sign;
within one fold there is one model. `r2` stays on the pooled vector.

Results go to `results_<RUN_TAG>/` (currently `results_v5_kras4b/`). The older
`results/` and `results_resks/` hold the superseded two-gene runs with the binary
`logreg` head and 4A KRAS; they are kept as history and nothing reads them.

## Reproduce

```bash
python experiments/continuous_functional/run_continuous.py
python experiments/continuous_functional/predictions.py
python experiments/continuous_functional/make_figures.py
python experiments/continuous_functional/plot_four_gene_ks.py
python experiments/continuous_functional/plot_four_gene_am.py
python experiments/continuous_functional/make_page.py
```

The first run builds any missing descriptor caches.

## The gallery page

`make_page.py` writes `metric_analysis/results/continuous_functional/index.html`
with collect_results' own `_page` / `_table_html`. The card on the gallery index
is emitted by `collect_results.render_index` and is gated on this page existing;
`metric_analysis/results/` is gitignored. To refresh only the index:

```python
import pandas as pd, sys
from pathlib import Path
REPO = Path("<repo>")
sys.path.insert(0, str(REPO / "metric_analysis" / "pipeline" / "orchestration"))
import collect_results as CR
d = REPO / "metric_analysis" / "results"
CR.render_index(d, pd.read_csv(d / "manifest.csv"))
```

## Files

- `targets.py` — the four continuous targets and the KRAS blind stratum.
- `dataset.py` — label-free descriptors and the continuous cohort.
- `run_continuous.py` — the ridge sweep.
- `tables.py` — the one place the aggregation from `continuous_runs.csv` is defined.
- `predictions.py` — per-variant out-of-fold predictions at N=90, seed 42.
- `make_figures.py` — per-gene Spearman and R² vs ensemble size.
- `plot_four_gene_ks.py`, `plot_four_gene_am.py` — descriptive associations.
- `make_page.py` — the gallery subpage.
