# Does sharing AF3 seed numbers lower the WT–mutant KS-score? (2026-10-20)

Cited in SI Note S6 (restructured SI). Moved here from
`manuscript_review/si_rebuild_2026-10-20/checks/note_s6_figs3/` (where it was first run
by the Note S6 audit); outputs copied unchanged.

Design (`seed_coupling_ks.py`): per gene, 30 random variants with 125-prediction pools,
20 repetitions, whole seeds, N = 60 per side (12 seeds × 5 samples, the Note S2 design).
`coupled` = WT seeds S1 vs mutant seeds S1 (same seed numbers, as in production);
`disjoint` = WT seeds S1 vs mutant seeds S2; `wt_wt` = the Note S2 background.

Result (`seed_coupling_ks_summary.csv`, median paired coupled − disjoint):
MSH6 −0.015, PIK3CA −0.012, MSH2 −0.011, VHL −0.011, POLE −0.003, PTEN +0.001.
Sharing seed numbers slightly lowers the WT–mutant KS-score, so the seed-disjoint
WT–WT background is slightly conservative.
