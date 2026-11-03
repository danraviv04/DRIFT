# metric_analysis

Driver vs passenger structural analysis for POLE, TP53, FOXA1, MutS_alpha and KRAS.

Everything runs from a Python package, `metric_analysis.pipeline` (see [Layout](#layout)).
The `notebooks/*_structure_metrics.ipynb` notebooks are **viewers only** — they load cached
results and do no computation.

## Quick start

One-time setup (from the repo root — the parent of `metric_analysis/`):

```bash
conda env create -f environment.yml
pip install -e .        # installs the `metric_analysis` package + short console commands
```

Then:

```bash
run-structure-metrics --protein all     # the whole pipeline
collect-results                         # -> metric_analysis/results/index.html
open metric_analysis/results/index.html
```

Every console command below has an equivalent `python -m` form that works **without**
`pip install -e .`, as long as you run it from the repo root (so the package resolves):

```bash
python -m metric_analysis.pipeline.orchestration.run_structure_metrics --protein all
python -m metric_analysis.pipeline.orchestration.collect_results
```

| console command | module |
|---|---|
| `run-structure-metrics` | `metric_analysis.pipeline.orchestration.run_structure_metrics` |
| `collect-results` | `metric_analysis.pipeline.orchestration.collect_results` |
| `run-rmsf-sharded` | `metric_analysis.pipeline.orchestration.run_rmsf_sharded` |
| `check-label-consistency` | `metric_analysis.pipeline.checks.check_label_consistency` |
| `make-viewer-notebooks` | `metric_analysis.pipeline.notebooks_gen.make_viewer_notebooks` |

Everything is cache-aware: rerunning skips work whose outputs already exist, so the
second run costs seconds. Use `--force` to recompute.

## run-structure-metrics

Flags (`run-structure-metrics <flags>`):

```bash
--protein FOXA1,TP53   # or 'all' (default)
--stage discrimination # comma-separated subset of the stages below, or 'all' (default)
--force                # ignore caches and recompute
--report               # also rebuild results/ + the HTML gallery when finished
```

Stages, in dependency order:

| stage | what it does | cost |
|---|---|---|
| `plddt` | extract per-residue pLDDT from the raw structures | slow, needs the external drive |
| `plddt_profile` | WT, configured driver, and configured passenger profiles; residue-wise ensemble-distribution panels when both exemplars are explicit | seconds |
| `features` | build the mutation feature table + assign labels | ~10 s |
| `plddt_ks` | per-residue WT-vs-mutant pLDDT KS tests + KS ROC-AUC ensemble-size sweep | seconds – ~10 min |
| `plddt_distribution` | per-residue Wasserstein-1, thresholded outlier, shape-excess, energy and CvM effects + ROC-AUC ensemble-size sweep | minutes – ~1 h |
| `discrimination` | Cohen's d ranking, cross-validated LDA, report, plots | ~2 s |
| `rmsf` | RMSF vs WT (+ interface-only), class dominance, RMSF-vs-D1 grid | slow, needs USalign |
| `dna` | protein–DNA minimum distances + class dominance + metric ranks (skipped when no `dna_chains`) | slow |
| `ensemble_aware` | D1/RMSF/RMSD/DNA ensemble-size sweeps + unified comparison | ~10 min – 1 h per protein |

Everyday use is the two fast stages:

```bash
run-structure-metrics --protein all --stage features,discrimination
```

### Full-distribution pLDDT metrics

`plddt_distribution` complements the mean-only D1 score and the bounded KS
statistic.  At each residue, it computes the empirical Wasserstein-1 distance

\[
W_1(F,G)=\int_{-\infty}^{\infty}|F(t)-G(t)|\,dt
=\int_0^1|F^{-1}(u)-G^{-1}(u)|\,du,
\]

and reports it both in pLDDT points and on the fixed pLDDT/100 scale.  Its exact
decomposition

\[
W_1 = |\mu_{mut}-\mu_{WT}| + W_{1,shape}
\]

separates the location change already visible to D1 from distributional change
beyond the means.  `plddt_w1_shape_fraction` is the residue-summed shape excess
divided by residue-summed W1; it is zero for a pure translation.

The supervisor-defined conformation-outlier score uses the WT residue median as
a robust reference. For mutant conformation (i) at residue (r),

\[
L_{r,i}=\left|\frac{m_{r,i}}{\operatorname{median}_k(w_{r,k})}-1\right|,
\qquad
S_r(\tau)=\frac{1}{N_{mut}}\sum_{i=1}^{N_{mut}}
L_{r,i}\,\mathbf 1\{L_{r,i}>\tau\},
\]

with the predeclared threshold \(\tau=0.01\) stored explicitly in every protein
configuration and output table. The mutation-level predictor
`plddt_outlier_score_mean` is the mean of \(S_r\) over valid residues. The mean
outlier fraction and selected-deviation magnitude are exported as interpretation
diagnostics but are not evaluated as additional predictors. WT and mutant
conformations are not paired.

Two secondary whole-distribution effects are exported for robustness:

\[
E=\sqrt{2\,\mathbb E|X-Y|-\mathbb E|X-X'|-\mathbb E|Y-Y'|},
\qquad
C=\int(F-G)^2\,dH.
\]

Energy is evaluated after scaling pLDDT to [0,1], and CvM uses the pooled
empirical distribution H with exact tie grouping.  These are effect sizes, not
p-values.  Labels are attached only after WT-vs-mutant calculation; the summary
stage compares D1, KS, and all new metrics on the same complete-case cohort.
Per-residue output is gzip-compressed because the seven-protein table contains
millions of rows.

The uncertainty ribbons on the per-residue class-mean D1, pLDDT-ratio, KS, and
full-distribution profiles are mutation-level bootstrap SDs.  Each bootstrap
draw resamples complete mutation profiles within driver and passenger classes,
so the residue-to-residue dependence within a mutation is preserved.  The
default is 1,000 resamples with seed 42.  These ribbons estimate the sampling
SD of the class mean; they are not raw between-mutation SDs or 95% confidence
intervals.

`plddt_profile` also writes a residue-faceted KDE PDF for every protein using the
explicit driver/passenger exemplars in `protein_configs.json`.  Each page contains
ten enlarged panels (two columns); every panel compares the raw WT, driver, and
passenger ensemble samples with a fixed 0.35-pLDDT Gaussian bandwidth.  Both axes
scale within a residue so narrow distributions remain readable, and should therefore
be compared between the three series inside a panel rather than by peak height or
horizontal position across different panels.  The exemplar mutation panels are
explicitly labelled `Site D` and `Site P`; coincident sites are labelled
`Sites D + P`.  Multi-chain matrices use the verified chain segments in
`protein_configs.json`, so MutS_alpha resets numbering between MSH2/MSH6 and TP53
marks residue 119 in all four mutated copies.

`plddt`, `rmsf` and `dna` read raw structures from `$DRIFT_ARCHIVE_ROOT/...`, so they need
that drive mounted. `ensemble_aware` only reads the caches under
`structure_metrics/<Protein>/ensemble_aware/cache/`, so it works without the drive.

`rmsf` and `dna` are cache-aware in a useful way: rerunning them **without** `--force`
reloads the summary CSVs and redoes only the cheap downstream work (dominance, plots,
interface filtering) in a few seconds, no drive needed. That is the way to pick up a
plotting or statistics change without paying for the alignments again.

### Parallel RMSF

`rmsf` is one USalign subprocess per structure, run sequentially — for MutS_alpha that is
~5 min × 152 artifacts ≈ 13 h. Every structure aligns against the same shared WT reference,
so the work is embarrassingly parallel:

```bash
run-rmsf-sharded --protein MutS_alpha --shards 4   # ~3.5 h
```

It slices the artifact list N ways, runs the *same* batch function in N processes, merges,
then reruns the ordinary `rmsf` stage over the merged CSVs for the plots. Verified against
the single-process path by a test that asserts the residue tables and delta summaries come
out frame-equal.

Do not raise `--shards` to the core count: the machine still has to be usable, and this
repo has already had one out-of-memory restart under heavier load. Shard CSVs are written
at the end of each shard, so a crash mid-run loses that shard's work — rerun a single shard
with `--shard <i>`, then `--merge-only`.

## Class dominance, and why it is separate from `discrimination`

Two independent tests of the same question, on different inputs:

| | `discrimination` stage | class dominance (`rmsf` / `dna` stages) |
|---|---|---|
| input | pLDDT-derived feature table | RMSF and DNA-distance metrics directly |
| method | Cohen's d ranking + cross-validated LDA | Mann-Whitney U |
| assumes normality | yes | no |

`p_driver_gt_passenger` is P(random driver > random passenger) — identical to that metric's
ROC-AUC used alone — and `rank_biserial` is 2p−1, so 0 is no separation and ±1 is perfect.

**Normality is violated for all five proteins** (see `*_lda_assumption_tests.csv`), which is
why the distribution-free view earns its place rather than merely duplicating the LDA.

Both of these were **lost for several weeks** and restored on 2026-07-30. They lived inline
in the five per-protein notebooks; replacing those with `run_structure_metrics.py` ported the
compute but not these analyses, and nothing failed — a dropped analysis leaves no error
behind, only an absent file. `lda_assumption_tests`, `plot_rmsf_metric_scatter_grid` and
`plot_metric_rank_with_highlights` had it worse: the functions were sitting in
`structure_metrics.py` the whole time with no caller.

If you add an analysis, add it to `ARTIFACT_GLOBS` in `collect_results.py` and to the
template in `make_viewer_notebooks.py`, or it will exist on disk and be invisible.

## check_label_consistency.py

Run this whenever labels or ground-truth files change.

```bash
check-label-consistency            # all proteins
check-label-consistency --strict   # exit 1 on any failure
```

It checks two independent things, because they fail independently:

- **agreement** — every metric family gives the same allele the same class, *and* the
  families cover the same variants. KRAS currently fails the second: its D1 and RMSD
  sweeps share only 7% of their variants, so its unified comparison is not like-for-like.
- **plausibility** — the absolute counts are believable: ≥30% of variants labelled,
  neither class under 5 members, and for `phenotypic_percentile` the observed driver
  rate matches the percentile.

The plausibility half exists because agreement alone once passed cleanly while every
family was consistently wrong (a ClinVar parsing bug had left MutS_alpha with 15 of 151
variants labelled and MSH2 with zero drivers). Only the absolute count looked wrong.

## make-viewer-notebooks

```bash
make-viewer-notebooks           # regenerate all 5 viewers into notebooks/
make-viewer-notebooks --check   # fail if any has drifted
```

All five notebooks come from one template. Edit the template in this script, not the
`.ipynb` files — direct edits are overwritten on the next regeneration.

## Where labels come from

One dispatcher, `labels.py::classify_allele_or_artifact`, selected by `label_mode` in
`protein_configs.json`. Each protein's ground truth is genuinely different:

| protein | `label_mode` | source |
|---|---|---|
| POLE | `csv_lookup` | `Ground_Truths/POLE/combined_pole_labels.csv` |
| FOXA1 | `hotspot_range` | residue index within [168, 269] |
| TP53 | `phenotypic_percentile` | `Supplementary table 3.xlsx`, 75th percentile over **all** scored alleles |
| MutS_alpha | `clinvar_by_gene` | ClinVar per gene — MSH2 and MSH6 number residues independently |
| KRAS | `dms_lfc` | HA1E DMS screen. **Not ClinVar**: it covers only 4 of KRAS's 21 variants and contains no benign ones, so no two-class split exists |

Never label inline. Call the dispatcher with the **full artifact/file name**, not a bare
mutation code — `clinvar_by_gene` needs the name to tell MSH2 from MSH6.

## Layout

The Python code is the `metric_analysis.pipeline` package; the analysis directory also
holds config, notebooks, scripts and the generated output trees.

```
metric_analysis/
  protein_configs.json                per-protein config: paths, chains, ensemble sizes, label source
  pipeline/                           the importable package
    paths.py                          ANALYSIS_DIR / REPO_ROOT — the two directory anchors
    labels.py                         the one Driver/Passenger dispatcher
    config/protein_config.py          config loader (resolves relative paths against the repo root)
    library/
      structure_metrics.py            shared library: pLDDT, RMSF, DNA distance, LDA, plotting
      ensemble_aware_pipeline.py      ensemble-size sweeps + RMSD-all-instances + unified comparison
    orchestration/
      run_structure_metrics.py        the orchestrator (start here)
      run_ensemble_sweep_parallel.py  per-chunk sweep worker
      run_rmsf_sharded.py             parallel RMSF driver
      collect_results.py              gather outputs -> results/ + manifest.csv + HTML gallery
    cache_builders/
      repair_rmsf_cache_chunk.py      build the coordinate cache, one chunk per process
      repair_dna_cache_chunk.py       build the DNA-distance cache, one chunk per process
    checks/check_label_consistency.py label sanity checks
    notebooks_gen/make_viewer_notebooks.py  generates the 5 viewer notebooks from one template
  conservation/                       standalone conservation-scoring line (no package imports)
  scripts/                            launch_*.sh, progress.sh, validate_and_regen.sh
  notebooks/                          5 viewers (generated); archive/ holds old exploratory notebooks
  tests/                              fast tests (no drive, no multi-GB caches)
  results/                            generated gallery (gitignored, rebuild any time)
  structure_metrics/<Protein>/        outputs; ensemble_aware/{cache,ensemble_sweeps,plots}
```

The packaging lives one level up: `../pyproject.toml` (at the repo root) declares the
`metric_analysis` package and the console-script commands.

The chunked helpers (`cache_builders/repair_rmsf_cache_chunk.py`,
`cache_builders/repair_dna_cache_chunk.py`, `orchestration/run_ensemble_sweep_parallel.py`)
and their `scripts/launch_*.sh` wrappers are still used to **build** the multi-GB
coordinate/DNA caches in parallel across chunks. `run-structure-metrics` consumes those
caches; it does not rebuild them.

## Install

    conda env create -f environment.yml
    pip install -e .        # from the repo root: installs the package + console commands

The editable install is optional — every command has a `python -m metric_analysis.pipeline.…`
form that works from the repo root without it (see [Quick start](#quick-start)).

Plus two things pip cannot provide: the **USalign** binary on `PATH` (needed by the
`rmsf` stage and by cache building), and the **`$DRIFT_ARCHIVE_ROOT`** drive (needed by
`plddt`, `rmsf`, `dna` — not by `features`, `discrimination` or `ensemble_aware`).

`torch`/`esm`/`wandb` are deliberately absent: those belong to the DRIFT classifier and
the embedding line, and folding them in would force a multi-GB install on anyone who
only wants the metrics.

## Same-named functions that are NOT duplicates

Three functions share the name `extract_plddt_from_cif`, and two more names appear twice.
They look like copy-paste and are not. **Do not merge them** — they have different
contracts and different callers:

| name | location | takes | returns |
|---|---|---|---|
| `extract_plddt_from_cif` | `pipeline/library/structure_metrics.py` | cif *path* | `{index: float}`, multi-chain |
| `extract_plddt_from_cif` | `../drift/structure_draws.py` | cif *path* | `{(resseq, icode): float}`, single chain |
| `extract_plddt_from_cif` | `../vector_output_mains/preprocess_tars.py` | cif *content string* | `List[float]` in sequence order, hand-parses `_atom_site` |
| `load_plddt_matrix` | `pipeline/library/structure_metrics.py` | csv path | `ndarray` |
| `load_plddt_matrix` | `pipeline/library/ensemble_aware_pipeline.py` | csv path | `(DataFrame, ndarray)` |
| `extract_mutation_code` | `pipeline/library/structure_metrics.py` | filename | supports `known_mutations` + custom `pattern`, no uppercasing |
| `extract_mutation_code` | `pipeline/labels.py` | any string | plain regex on `.upper()` |

A parsing fix in one does **not** belong in the others. `extract_mutation_code`
especially: it sits directly on the label path where several bugs were already found, and
the two variants differ in uppercasing — swapping them would silently change which
variants get labelled.

## Known data limitations (carry these into the write-up)

Two findings that no rerun fixes. Both were surfaced by `check_label_consistency.py`
and the cache error logs, and both affect how results should be reported.

### KRAS: coordinate and confidence descriptors now have matched coverage

The earlier 43-variant RMSD cache was incomplete and made cross-family comparisons use
different variants. That limitation has been repaired. `check_label_consistency.py
--protein KRAS` now reports 56 variants, 25 drivers and 20 passengers under the
source-supported HA1E WT LFC rule, 11 deliberately unlabelled intermediate variants,
and 100% variant-set overlap among D1, RMSD and RMSF. The canonical 11- and
22-descriptor panels therefore compare all descriptor families on the same 45 labelled
KRAS variants.

The 11 intermediate variants have $0.1<$ HA1E WT LFC $\leq0.68$ and are retained for
continuous-score sensitivity analyses but excluded from binary discrimination. L19F
and N116H, which lie above the source article's GOF cutoff but below the former local
$\geq1.0$ cutoff, are now drivers. No KRAS structures had to be regenerated for this
label correction; their pLDDT and coordinate descriptors were already present.

### 346 structures are permanently missing from the DNA caches

The pipeline treats the DNA per-structure caches as complete. They are not:

| protein | structures missing | recorded in |
|---|---|---|
| MutS_alpha | **329** (78 + 74 + 75 + 102) | `structure_metrics/MutS_alpha/ensemble_aware/cache/MutS_alpha_dna_cache_chunk_{0..3}_errors.csv` |
| TP53 | **17** | `structure_metrics/TP53/ensemble_aware/cache/TP53_dna_cache_chunk_{0..3}_errors.csv` |

(Counts are data rows, excluding the CSV header in each of the four files — an earlier
audit reported 333 and 21 by counting headers too.)

Each row records the artifact, structure index, source `tar://` path and the error. These
failed during cache building and were never recovered, so every DNA-distance number for
those two proteins is computed over a slightly reduced ensemble. RMSF logs by contrast all
end `errors: 0`.

**Never delete the `*_errors.csv` files.** They are ~180 KB and the only record of which
structures are absent — without them the caches look complete. Two further historical sets
are kept for the same reason: a 1,098-row pre-rebuild set for MutS_alpha and a 335-row set
for KRAS.

## Gotchas

- `structure_metrics/` is gitignored (pre-existing), yet ~128 files under it are tracked
  from before that rule. Newly generated outputs there are **not** picked up by git.
- The shared library is `pipeline/library/structure_metrics.py`, imported as
  `metric_analysis.pipeline.library.structure_metrics`. It used to sit at the analysis root
  as `structure_metrics.py`, sharing a name with the `structure_metrics/` output directory —
  the package refactor removed that latent shadowing, so the two can no longer be confused.
- The filesystem is case-insensitive: `pole_x.csv` and `POLE_x.csv` are the same file.
- `total_chunks` must match the number of `coord_chunks` on disk (4). Chunk N holds only
  its slice of artifacts, so a mismatch would sweep part of the data and report success.
  There is an explicit guard, but don't override it.
- `d1_outputs/d1_<Protein>_summary.csv` is **not** produced by this pipeline (the
  notebooks made it via `drift/structure_draws.py`). The `d1_mean` feature is unaffected;
  those standalone files are stale, and KRAS never had one.
