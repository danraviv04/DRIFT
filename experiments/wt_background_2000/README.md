# WT–WT backgrounds at 2,000 splits (2026-10-20)

Supplementary Fig. S2 already uses 2,000 WT–WT splits (`experiments/wt_split_12v12/results_12v12_2000`). Fig. 1f and the rest of Supplementary Note S2 still used 200. This directory rebuilds those remaining backgrounds at 2,000 splits.

It also re-renders the two S2 panels with "seed-disjoint" in place of "seed-matched". The paper repo was not touched, and no published result directory was overwritten.

## Method

Each published run in `experiments/pten_wt_null_panel/` is replayed by its own code. The script imports `pten_wt_null_panel.py` unchanged and uses its own RNG stream, `default_rng(20260923)`:

1. PTEN only: the batch check, 200 reps.
2. The published 200 WT–WT splits.
3. The 20 draws per variant.
4. **New:** 1,800 further WT–WT splits drawn from the same generator.

The extension is drawn after the variant draws, so the variant RNG stream is untouched.

**What is asserted, for all 8 runs:**

- The replayed 200 splits are bit-identical to the published `values.csv`.
- Every variant score is bit-identical to the published scores.
- The PTEN batch check is bit-identical.
- AUC and Mann–Whitney P are recomputed and equal the published values.

**Reading note.** Read the published `values.csv` with `float_precision="round_trip"`. The default pandas parser is off by about 1e-16.

Two consequences of this design:

- The 2,000-split background contains the published 200 as `split_0`–`split_199`.
- Only the background changes. Variant scores, AUC and Mann–Whitney P are the published ones.

**Stability.** Every p95 was re-estimated under 3 further independent streams of 2,000 splits. These are recorded under `stability_streams` in each `summary.json`.

## Scripts

Run from the repo root with `PYTHONPATH="." python`. The first two also need `DRIFT_RUN_TAG=v6_ca`, which selects the MSH6 cohort of the published runs.

| Script | What it does | Wall time |
|---|---|---|
| `run_wt_background_2000.py` | 8 runs: MSH6 375-prediction pool and PTEN 1,000-prediction pool; N = 50 and 100; seed-disjoint and random. Writes `results/<GENE>_n<N>_<split>/values.csv` and `summary.json` | 2 min 6 s (4 processes) |
| `residue_null_2000.py` | Fig. 1e per-residue calibration (MSH6, N = 90), matched and random arms, 2,000 splits. Runs `residue_null/residue_null.py` unchanged, with output redirected to `results/fig1e_residue_null/`. Each mode has its own seed, so the matched arm must equal `residue_null/residue_null_MSH6_B2000.json`; this is asserted and holds exactly | 55 s |
| `make_fig1f_panel.py` | Fig. 1f at 2,000 splits, drawn with `pten_wt_null_panel.draw_panel` unchanged. Style check: the same call on the published values reproduces the published PNG pixel for pixel (asserted) | seconds |
| `s2_panels_seed_disjoint.py` | S2a/S2b from the published `results_12v12_2000` CSVs. Changes only the legend strings. Asserts that the values table equals the published one with the label swapped | seconds |
| `summarise.py` | Writes `results/before_after.csv` and `results/counterfactual_random_background.csv` | seconds |

Logs are in `logs/`.

## Published (200) → new (2,000)

### Main text

| Quantity | Published | New |
|---|---|---|
| MSH6 N = 100 drivers above the WT–WT p95 ("49 of the 66") | 49/66 | **48/66** |
| MSH6 N = 100 passengers above the WT–WT p95 ("11 of the 77") | 11/77 | **8/77** |
| Mann–Whitney P (N = 100); AUC | 1.4e-16; 0.90 | unchanged |
| R772L residues differing (Fig. 1e), mean [range] | 24.5% [22.6–30.6] | 25.3% [22.6–31.2], so "25%" holds |
| E221D residues differing | 0 | 0 |

### Fig. 1f legend

| Quantity | Published | New |
|---|---|---|
| Number of WT–WT comparisons | 200 | **2,000** |
| p95 line | 0.161 | 0.163 |
| Panel | — | `results/fig1f_panel/MSH6_wt_null_panel.pdf` |

### SI Note S2

**Seed-structure paragraph**

| Quantity | Published | New |
|---|---|---|
| Random split, N = 100: drivers / passengers above the random p95 (random-run variant scores) | all 66 / 61 | **66 / 65** |
| Seed-disjoint split, N = 100, comparison numbers | 49 / 11 | **48 / 8** |

**Per-residue test (Fig. 1e)**

| Quantity | Published | New |
|---|---|---|
| Share of WT–WT residue tests with raw P < 0.05, seed-disjoint | 17.4% | **17.6%** |
| Same, random split | 5.1% | **5.0%** |
| R772L residues differing, mean (range) | 25% (23–31%) | 25% (23–31%), unchanged when rounded |
| R772L, raw P | 43% | 43% |
| E221D residues differing | none | none |
| E221D, raw P | 1.3% | 1.3% |
| Window 764–779 | 16 and 0 | 16 and 0 |
| R772L variant-level empirical P | < 0.005 (= floor 1/201) | **0.003**. Not at the floor (1/2001 = 0.0005): 5 of the 2,000 WT–WT splits reached R772L's fraction |
| E221D variant-level empirical P | 0.50 | **0.49** |
| Number of WT–WT values per residue (also in Methods) | 200 | **2,000** |

**KS-score background (Fig. 1f)**

| Quantity | Published | New |
|---|---|---|
| WT–WT median, N = 50 / 100 | 0.175 / 0.137 | **0.176 / 0.136** |
| WT–WT p95, N = 50 / 100 | 0.197 / 0.161 | **0.206 / 0.163** |
| Random-split p95, N = 50 / 100 | 0.181 / 0.129 | **0.182 / 0.128** |
| Drivers / passengers above p95, N = 100 | 49 / 11 | **48 / 8** |
| Drivers / passengers above p95, N = 50 | 51 / 12 | **45 / 7** |
| Mann–Whitney P; AUC | 1.4e-16; 0.90 | unchanged |

**PTEN**

| Quantity | Published | New |
|---|---|---|
| Seed sharing, MSH6 / PTEN | 67% / 25% | unchanged. This is a design property (50/75 and 50/200 seeds) and does not depend on the split count |
| PTEN p95, N = 50 / 100 | 0.204 / 0.156 | **0.200 / 0.159** |
| PTEN random-split p95, N = 50 / 100 | 0.177 / 0.124 | **0.176 / 0.126** |
| N = 50 above p95 | 256/257, S294R below | 256/257, S294R below (unchanged) |
| N = 100 above p95 | all 257 (S294R "slightly above") | **256/257: S294R (0.1561) now below p95 0.1592** |

**Other genes (S2):** already at 2,000 splits, so nothing changes. Only the legend wording changes.

### Random-split counterfactual (background only)

The published seed-disjoint variant scores are held fixed and counted against the random-split p95 (`results/counterfactual_random_background.csv`):

| Background | N = 100 drivers / passengers | N = 50 drivers / passengers |
|---|---|---|
| Random, 200 splits (published background) | 66 / 77 | 66 / 44 |
| Random, 2,000 splits | **66 / 77** | **65 / 43** |
| Seed-disjoint, 2,000 splits | 48 / 8 | 45 / 7 |

### Stability of the 2,000-split p95

Main stream plus 3 independent streams:

| Run | p95 range | Drivers above | Passengers above |
|---|---|---|---|
| MSH6 N = 100, seed-disjoint | 0.1626–0.1655 | 47–48 | 6–8 |
| MSH6 N = 50, seed-disjoint | 0.2061–0.2087 | 42–45 | 6–7 |
| PTEN N = 100 | 0.1576–0.1592 | — | — |
| PTEN N = 50 | 0.1998–0.2015 | — | — |

For PTEN, S294R is below the p95 in every stream at both sizes, and all 248 drivers are above it. The MSH6 counts still move by a few variants between streams, because many variants sit near the line.

For comparison, the earlier independent check (`manuscript_review/si_rebuild_2026-10-20/checks/note_s2/fig1f_p95_stability.csv`, stream 0) gave:

- N = 100: p95 0.168, 45/6.
- N = 50: p95 0.206, 44/7.

## Supplementary Fig. S2 panels (`results/s2_panels/`)

The panels are re-rendered from the published S2 data:

- `wt_background_splits.csv`, sha256 `d48adba3…`.
- `variant_scores`, sha256 `949c42ef…`. Both hashes are in `provenance.json`.

The text changes are:

- Panel a legend: "seed-matched" → "seed-disjoint".
- Panel b legend: "Seed-matched split" → "Seed-disjoint split".

The values table equals the published one apart from that word.

A pixel diff against the published PNGs shows that only the legend rows (y 40–63 px) differ. Panel b's PNG is 5 px narrower, because the shorter legend tightens the bounding box. Below the legend it is pixel-identical when left-aligned.

The S2 caption and Note S2 text still say "seed-matched" in the paper (SI lines 331, 359, 394, 820, 827). That text was not edited.

## Things to know

- **Fig. 1f with 2,000 dots.** In the published style (dots s = 7, alpha 0.55, drawn over the box), the 2,000 WT–WT dots hide the WT–WT box and median. `MSH6_wt_null_panel.pdf` is the requested same-style panel.

  `MSH6_wt_null_panel_lightdots.pdf` is an alternative. Only the WT–WT group differs: dots s = 1.5, alpha 0.2, and the box drawn over them. Choose one.

  The x-tick now reads "WT vs WT (n=2000)". The jitter of the variant dots differs from the published panel, because the jitter RNG is shared across groups; the data are the same.
- **Published 200 splits are the first 200.** The new main-stream background is a superset of the published one, not an independent draw.
