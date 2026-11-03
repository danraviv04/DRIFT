# WT–WT background for Supplementary Fig. S2: 12-vs-12 seeds, 2,000 splits (2026-10-20)

The question was whether the S2a background should split the 25 WT seeds 12 vs 12 rather than 12 vs 13. In the 12-vs-13 split, the 13-seed side draws 60 of its 65 predictions.

## Scripts

All scripts run with `PYTHONPATH="." python experiments/wt_split_12v12/<script>`.

- **`compare.py`** reruns the published 12-vs-13 split. It reproduces the published `wt_background_splits.csv` exactly, and the script asserts this. It then runs 12 vs 12 with the same 200 splits.
  Outputs: `summary.csv`, `splits.csv`.
- **`rng_noise.py`** runs 10 independent RNG streams × 200 splits per design. This showed that the design makes no real difference: 12 vs 12 raises p95 by at most about 0.01. The count of variants above p95 does depend on the stream:
  - VHL passengers: 7–17 of 19 (published 17).
  - POLE passengers: 519–619 of 1,037 (published 612).
  - PIK3CA drivers: 26–42 of 84 (published 38).

  Outputs: `rng_noise.csv`, `rng_noise_summary.csv`.
- **`run_12v12_2000.py`** rebuilds S2 with 12 vs 12 seeds and 2,000 splits. Everything else is unchanged from `si_figures/wt_backgrounds.py`:
  - It uses the same N = 60.
  - Variant scores come from the POLE single-WT caches in `pole_wt_repair/cache`, as in the published S2. They are identical to the published scores, all 1,799 of them.
  - It adds a stability check: three further independent streams of 2,000 splits each.

  Outputs go to `results_12v12_2000/`. The script does not touch `si_figures/results/` or the paper repo.

## Result (`results_12v12_2000/`, main stream; range over the 3 check streams in brackets)

| Gene | p95 | Drivers above | Passengers above | Published (12v13, 200) |
|---|---|---|---|---|
| POLE | 0.274 | 37/38 | 590/1,037 [564–588] | 37/38, 612/1,037 |
| PTEN | 0.217 | 243/248 | 7/9 [7] | 241/248, 6/9 |
| VHL | 0.247 | 101/105 | 9/19 [7–9] | 103/105, 17/19 |
| MSH2 | 0.245 | 42/71 | 3/38 [2–3] | 42/71, 3/38 |
| MSH6 | 0.231 | 26/66 | 3/77 [3] | 26/66, 3/77 |
| PIK3CA | 0.280 | 35/84 | 1/7 [0–1] | 38/84, 1/7 |

The per-residue panel (b) gives these median rates of raw P < 0.05 in WT–WT comparisons:

- Seed-matched splits: 12.3–23.9%.
- Random splits: 0.5–3.7%.
- Published: 10–23% and 0.5–3.8%.
