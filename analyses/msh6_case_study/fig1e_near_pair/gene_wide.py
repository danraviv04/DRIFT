#!/usr/bin/env python
"""Fig. 1e across the whole of MSH6: A1055P and I1054F at every residue.

Fig. 1e shows 16 residues around the two sites. This shows all 1,360, so the
reader can see whether the driver's shift stays near its site or runs through
the protein, and whether the passenger's stays at its own residue.

Tracks, top to bottom, on one residue axis:

  <pLDDT_i^WT>   mean over the 125-prediction WT reference pool: where the
                 prediction is confident at all
  KS_i           per-residue two-sample KS, each variant against its own
                 canonical WT draw (N=90, canonical seed). The mean of each
                 trace IS the variant's KS-score -- asserted against
                 per_variant_seed.csv.gz
  differs        the calibrated residue test of Supplementary Note S2
                 (residue_null/residue_null.py: 2,000 seed-disjoint WT-WT
                 splits, empirical P, BH across residues), as the share of its
                 ten draws in which each residue was called. residue_null.py
                 saves draw 0 only, so this replays its matched arm and mutant
                 draws on the same RNG streams; draw 0 must equal the saved
                 table and the mean share the saved JSON -- both asserted
  shift          <pLDDT_i^MUT> - <pLDDT_i^WT> on the canonical draws, against
                 the WT spread +/- sigma_i^WT at that residue

    cd <repo>; DRIFT_RUN_TAG=v6_ca PYTHONPATH="." python \
        analyses/msh6_case_study/fig1e_near_pair/gene_wide.py
"""
import json
import os
import sys
import zlib
from pathlib import Path

# EXP, not HERE: figlib exports its own HERE and REPO, and the star import
# below would otherwise point every output path at si_figures/.
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP.parents[2] / "figures" / "si" / "lib"))
sys.path.insert(0, str(EXP.parents[2] / "analyses" / "msh6_case_study" / "residue_null"))
from figlib import *  # noqa: F403,E402
import residue_null as rn  # noqa: E402
from ks_sequence_panel import DRIVER_C, PASSENGER_C, WT_C, residue_ks  # noqa: E402
from exemplar_sampling import CANONICAL_SEED, PER_VARIANT, sampled_pair  # noqa: E402
from metric_analysis.pipeline.library.canonical_cohort import load_canonical_cohort  # noqa: E402

GENE, DRIVER, PASSENGER = "MSH6", "A1055P", "I1054F"
WINDOW = (1047, 1062)          # the Fig. 1e window
SIZE, SPLITS = 90, 2000
WT_POOL = REPO / "analyses/msh6_case_study/wt_pool/wt_75seeds/MutS_alpha_MSH6_GT_WT_75seeds_plddt_values.csv"
SAVED = EXP / "results" / "A1055P_I1054F" / "residue_null_MSH6_B2000"
OUT = EXP / "results" / "gene_wide"


def calibrated_calls():
    """Residue calls of the calibrated test, all ten draws (draws x residues)."""
    coh = load_canonical_cohort(GENE).set_index("mutation_code")
    rows = slice(int(coh.plddt_row_range_start.iloc[0]), int(coh.plddt_row_range_stop.iloc[0]))
    wt, _ = rn.load(coh.wt_source_file.iloc[0], rows)
    pool, pool_seeds = rn.load(WT_POOL, rows)
    seeds = np.array(pool_seeds)
    uniq = np.unique(seeds)
    rng = np.random.default_rng(rn.SEED + zlib.crc32(b"matched"))
    null = np.empty((SPLITS, pool.shape[0]))
    for k in range(SPLITS):
        sd = rng.permutation(uniq)
        pa = np.flatnonzero(np.isin(seeds, sd[:25]))
        pb = np.flatnonzero(np.isin(seeds, sd[25:50]))
        a, b = rng.choice(pa, rn.N, replace=False), rng.choice(pb, rn.N, replace=False)
        null[k] = rn.mwu_p(pool[:, a], pool[:, b])
    calls = {}
    for mut in (DRIVER, PASSENGER):
        m, _ = rn.load(coh.loc[mut, "source_file"], rows)
        rng = np.random.default_rng(rn.SEED + zlib.crc32(mut.encode()))
        sig = []
        for _ in range(rn.REPEATS):
            p = rn.mwu_p(wt[:, rng.choice(wt.shape[1], rn.N, replace=False)],
                         m[:, rng.choice(m.shape[1], rn.N, replace=False)])
            sig.append(rn.bh((1 + (null <= p[None, :]).sum(axis=0)) / (1 + SPLITS)))
        calls[mut] = np.array(sig)

    saved = pd.read_csv(f"{SAVED}_per_residue.csv")
    summary = json.loads(Path(f"{SAVED}.json").read_text())
    for mut, c in calls.items():
        s = saved[(saved.mutant == mut) & (saved.null == "matched")].sort_values("residue")
        assert np.array_equal(c[0], s.bh_significant.to_numpy()), f"{mut}: draw 0 differs"
        assert c.mean(axis=1).mean() == summary["mutants"][mut]["matched"]["empirical_bh_frac_mean"]
    print("[gene_wide] calibrated calls reproduce residue_null_MSH6_B2000 exactly")
    return calls, wt


def canonical_tracks():
    """KS_i and mean shift per variant, each against its own canonical WT draw."""
    recs = cohort(GENE).set_index("mutation_code")
    ref = pd.read_csv(PER_VARIANT)
    out = {}
    for code in (DRIVER, PASSENGER):
        rec = recs.loc[code].to_dict() | {"protein": GENE, "mutation_code": code}
        resid, wt, mut, _ = sampled_pair(rec, SIZE, CANONICAL_SEED)
        ks = residue_ks(wt, mut)
        row = ref[(ref.protein == GENE) & (ref.mutation_code == code)
                  & (ref.requested_size == SIZE) & (ref.seed == CANONICAL_SEED)]
        np.testing.assert_allclose(ks.mean(), row.ks_raw.iloc[0], atol=1e-12, rtol=0)
        out[code] = dict(resid=resid.astype(int), ks=ks, shift=mut.mean(1) - wt.mean(1))
    assert np.array_equal(out[DRIVER]["resid"], out[PASSENGER]["resid"])
    return out


def draw(tr, calls, wt_pool):
    resid = tr[DRIVER]["resid"]
    wt_mean, wt_sd = wt_pool.mean(1), wt_pool.std(1, ddof=1)
    fig = plt.figure(figsize=(W2, W2 * 0.60))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.75, 1.7, 0.42, 1.5], hspace=0.14,
                          left=0.115, right=0.985, top=0.90, bottom=0.085)
    ax_wt, ax_ks, ax_c, ax_sh = (fig.add_subplot(gs[0]),) + tuple(
        fig.add_subplot(gs[i]) for i in (1, 2, 3))
    for a in (ax_wt, ax_ks, ax_c):
        a.sharex(ax_sh)
        a.tick_params(axis="x", labelbottom=False)
    for a in (ax_wt, ax_ks, ax_c, ax_sh):
        a.axvspan(WINDOW[0] - 0.5, WINDOW[1] + 0.5, color="#f0c808", alpha=0.30, lw=0, zorder=0)

    ax_wt.fill_between(resid, 0, wt_mean, color=WT_C, alpha=0.35, lw=0)
    ax_wt.plot(resid, wt_mean, color=WT_C, lw=0.5)
    ax_wt.axhline(70, color="#666", ls=":", lw=0.6)
    ax_wt.set(ylim=(0, 100), yticks=[0, 50, 100])
    ax_wt.set_ylabel(r"$\langle \mathrm{pLDDT}_i^{\,\mathrm{WT}} \rangle$")
    ax_wt.annotate("Fig. 1e", xy=((WINDOW[0] + WINDOW[1]) / 2, 1.0),
                   xycoords=("data", "axes fraction"), xytext=(0, 3),
                   textcoords="offset points", ha="center", va="bottom",
                   fontsize=6, color="#8a7500")

    # The KS-score goes in the legend: printed beside its dashed line it sat on
    # top of the traces, which fill the whole width at that height.
    for code, c, role in ((DRIVER, DRIVER_C, "Driver"), (PASSENGER, PASSENGER_C, "Passenger")):
        k = tr[code]["ks"]
        ax_ks.plot(resid, k, color=c, lw=0.45, alpha=0.9,
                   label=f"{role} {code}  (KS-score {k.mean():.3f}, dashed)")
        ax_ks.axhline(k.mean(), color=c, ls="--", lw=0.8)
    ax_ks.set(ylim=(0, 1.04), yticks=[0, 0.5, 1])
    ax_ks.set_ylabel(r"$\mathrm{KS}_i$")
    ax_ks.legend(loc="upper left", fontsize=6.5, handlelength=1.4, borderpad=0.3,
                 frameon=False)

    # One tick per residue, opacity = share of the ten draws that called it.
    for y, code, c in ((1, DRIVER, DRIVER_C), (0, PASSENGER, PASSENGER_C)):
        share = calls[code].mean(axis=0)
        hit = share > 0
        rgba = np.array([matplotlib.colors.to_rgba(c)] * hit.sum())
        rgba[:, 3] = share[hit]
        ax_c.vlines(resid[hit], y - 0.38, y + 0.38, colors=rgba, lw=0.6)
    ax_c.set(ylim=(-0.6, 1.6), yticks=[0, 1])
    ax_c.set_yticklabels([PASSENGER, DRIVER], fontsize=6)
    for lab, c in zip(ax_c.get_yticklabels(), (PASSENGER_C, DRIVER_C)):
        lab.set_color(c)
    ax_c.tick_params(axis="both", length=0)
    ax_c.set_ylabel("differs", labelpad=4)
    for s in ("left", "bottom"):
        ax_c.spines[s].set_visible(False)

    # The limits hold every shift; only the WT band is clipped, and only where the
    # prediction is disordered (sigma_i^WT reaches 13 pLDDT points there).
    ax_sh.fill_between(resid, -wt_sd, wt_sd, color=WT_C, alpha=0.20, lw=0,
                       label=r"WT $\pm\,\sigma_i^{\mathrm{WT}}$")
    ax_sh.axhline(0, color="#444", ls="--", lw=0.5)
    for code, c in ((PASSENGER, PASSENGER_C), (DRIVER, DRIVER_C)):
        ax_sh.plot(resid, tr[code]["shift"], color=c, lw=0.45, alpha=0.9)
    lo = min(tr[c]["shift"].min() for c in tr)
    hi = max(tr[c]["shift"].max() for c in tr)
    ax_sh.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax_sh.set_ylabel(r"$\langle \mathrm{pLDDT}_i^{\,\mathrm{MUT}} \rangle - "
                     r"\langle \mathrm{pLDDT}_i^{\,\mathrm{WT}} \rangle$")
    ax_sh.legend(loc="lower left", fontsize=6, handlelength=1.2, borderpad=0.2, frameon=False)
    ax_sh.set(xlim=(resid.min() - 0.5, resid.max() + 0.5), xlabel=f"{pretty(GENE)} residue")
    ax_sh.set_xticks(np.arange(100, resid.max() + 1, 100))
    ax_sh.tick_params(axis="x", labelsize=6)

    fig.align_ylabels([ax_wt, ax_ks, ax_c, ax_sh])
    # Supplementary Fig. S3 is one figure, not assembled from panels, so the
    # letters the caption refers to are drawn here.
    for a, letter in zip((ax_wt, ax_ks, ax_c, ax_sh), "abcd"):
        fig.text(0.006, a.get_position().y1, letter, ha="left", va="top",
                 fontsize=9, weight="bold")
    return fig, wt_mean, wt_sd


def main():
    if os.environ.get("DRIFT_RUN_TAG") != "v6_ca":
        raise SystemExit("set DRIFT_RUN_TAG=v6_ca")
    tr = canonical_tracks()
    calls, wt_pool = calibrated_calls()
    fig, wt_mean, wt_sd = draw(tr, calls, wt_pool)
    save(fig, f"{GENE}_gene_wide_{DRIVER}_vs_{PASSENGER}", OUT)

    resid = tr[DRIVER]["resid"]
    table = pd.DataFrame({"residue": resid, "wt_mean_plddt_pool": wt_mean, "wt_sd_plddt_pool": wt_sd})
    for code in (DRIVER, PASSENGER):
        table[f"ks_{code}"] = tr[code]["ks"]
        table[f"shift_{code}"] = tr[code]["shift"]
        table[f"called_share_{code}"] = calls[code].mean(axis=0)
    table.to_csv(OUT / f"{GENE}_gene_wide_per_residue.csv", index=False)

    tag = f"{GENE}_gene_wide"
    called = {c: calls[c].mean(axis=0) >= 0.5 for c in calls}
    confident = wt_mean >= 70
    for code in (DRIVER, PASSENGER):
        record(tag, f"{code}_ks_score", float(tr[code]["ks"].mean()), "mean of the KS_i trace")
        record(tag, f"{code}_residues_called_half_draws", int(called[code].sum()),
               "called in >= 5 of 10 draws")
        record(tag, f"{code}_called_outside_window", int((called[code] & ((resid < WINDOW[0]) | (resid > WINDOW[1]))).sum()))
        record(tag, f"{code}_called_in_low_confidence", int((called[code] & ~confident).sum()),
               "WT mean pLDDT < 70")
    record(tag, "low_confidence_residues", int((~confident).sum()), "WT mean pLDDT < 70")
    write_values(f"{tag}_values.csv", OUT)
    provenance(OUT, figure=tag, size=SIZE, seed=CANONICAL_SEED, driver=DRIVER, passenger=PASSENGER,
               window=list(WINDOW), splits=SPLITS,
               ks_tracks="canonical independent per-variant WT draws",
               calls="residue_null.py matched arm replayed, all ten draws")
    print(pd.read_csv(OUT / f"{tag}_values.csv").to_string(index=False))


if __name__ == "__main__":
    main()
