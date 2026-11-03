# KRAS4B: KS-score and AlphaMissense across the HA1E assay range (2026-10-20)

For SI Note S3 (KRAS/PIK3CA interpretation). Question: is DRIFT's weak KRAS
correlation the same U-shape that makes AlphaMissense's signed correlation weak?

`kras_score_by_assay_bin.py` → `by_bin.csv`, `contrasts.csv`. 935 blind variants
(hotspot_only excluded); KS = `kras4b_coverage/results/ks/mean_scores.csv`
(ρ with LFC −0.068, the main-text −0.07).

**Answer: no.** AM is U-shaped (median 0.995 depleted, 0.92 neutral, 0.999 GOF;
ρ with |LFC| 0.29). The KS-score is not: slightly higher for variants depleted in
the screen (median 0.636 vs 0.591 neutral, AUC 0.59) and not higher for strong
activators (0.545, n=23, AUC 0.42, P=0.22); ρ with |LFC| 0.035.
P values are naive (five substitutions per position are not independent): descriptive only.
