#!/usr/bin/env python
"""Exemplar figure: a missense leaves the predicted structure but widens the pool.

    conda activate thesis_core
    python analyses/coordinates/rmsf_histogram/exemplar_wt_vs_mut.py --mutant P286R

THE STATEMENT THIS FIGURE MAKES
A single substitution does not change the predicted conformation -- the two
ensemble-mean structures superimpose to a fraction of an Angstrom -- yet the
mutant ensemble is broader at essentially every residue. That is the premise of
the whole approach: if the mean structure carried the signal, a single predicted
model would already separate drivers from passengers, and it does not.

It is the visual form of the main-text sentence at L200-203, "a point
substitution rarely changes a predicted fold: both return high-confidence
structures nearly superimposable on the wild type".

WHY THESE TWO PANELS
a  Per-residue spread, WT against mutant, one point per residue, with the line
   of equality. The claim "the ensemble is broader" becomes a count of points
   above a diagonal, which needs no scale-reading and no binning choice. The
   annotation carries the other half of the statement as two numbers, because
   "the structure did not move" is a magnitude and not a shape.
b  The same two quantities along the sequence. This is what tells the reader the
   broadening is distributed rather than a local unfolding at the substituted
   residue, which a scatter cannot show.
c  The centre shift along the same axis, on ITS OWN scale. It cannot share b's
   axis: in Angstrom the centre shift (median 0.092) is larger than the WT
   spread (0.073), so an overlay would say the means are further apart than
   either ensemble is wide -- the opposite of the claim. The claim holds on a
   STRUCTURAL scale, which is what the shaded band supplies: 1-3 A separates
   genuine alternative conformations and a Ca-Ca bond is 3.8 A, so 0.21 A RMSD
   is superimposable. Two yardsticks, therefore two panels.

Deliberately NOT a histogram over residues: a histogram spends its resolution on
the shape of the dispersion distribution, which is not the claim, and it cannot
pair a residue in one ensemble with the same residue in the other. The pairing
is the entire content here.

DISPERSION ESTIMATOR. --metric selects it, mad_mean by default:
  mad_mean    mean_i || p_i,r - mean_i(p_i,r) ||        (L1, outlier-tolerant)
  mad_median  median_i || p_i,r - median_i(p_i,r) ||    (fully robust)
  rmsf        sqrt( mean_i || p_i,r - mean_i(p_i,r) ||^2 )   (Methods eq. 4)
The centre shift, || mean_WT(r) - mean_mut(r) ||, is a distance between two
points and does not depend on this choice.

HOW TYPICAL IS THE VARIANT. P286R is the strongest driver in POLE on this axis:
+62.7% RMSF against a driver median of +8.6% (IQR 2.1-17.8), rank 38 of 38. It
is the canonical hotspot and it is the clearest illustration, but a caption must
say it was chosen for clarity or a reader who computes the median driver will
read the number as a general claim. --mutant V411L gives a driver at the median
(+8.8%) for a side-by-side. See experiments/mean_vs_spread/ for the population.

The effect is also POLE-specific: across the six evaluation datasets dRMSF has
macro AUC 0.502 signed, and drivers are less mobile in five of seven. This
figure illustrates a mechanism in the demonstration gene; it does not generalise.

Pools are truncated to a common depth because every dispersion estimator grows
with N. WT is POLE_268-471_WT, the pipeline's reference; see rmsf_histogram.py
for why that artifact and not the other two.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42

import rmsf_histogram as R          # residue_table, load, CACHE, XYZ, WT_PRIMARY

HERE = Path(__file__).resolve().parent
WT_C, MU_C = "#2980b9", "#c0392b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", default="P286R")
    ap.add_argument("--metric", default="mad_mean",
                    choices=["rmsf", "mad_mean", "mad_median"])
    ap.add_argument("--keep-duplicate", action="store_true",
                    help="keep AF3's duplicated top-ranked model; see "
                         "rmsf_histogram.read_cache for what it is and why it goes")
    # A panel's on-paper text size is set by the width it is AUTHORED at, not
    # the width it is placed at: final_pt = source_pt * (placed/native) * (180/260).
    # At the default 7.4 in the 8.5 pt annotations land at 3.8 pt in a 241 pt
    # slot -- the header comment below computes 6 pt because it omits the
    # 260->180 mm scaling the figure takes at submission. Authoring near the
    # placed width is what puts them back in the 5-7 pt the checklist asks for.
    ap.add_argument("--panel-width-in", type=float, default=7.4,
                    help="authored width of the bc panels; heights scale with it")
    ap.add_argument("--shift-xaxis", action="store_true",
                    help="give the centre-shift panel its OWN residue axis. It "
                         "is suppressed by default because the panel sits "
                         "directly above the spread panel and borrows its axis; "
                         "side by side it needs its own or it has none at all.")
    ap.add_argument("--bc-heights", default="",
                    help="absolute authored heights in inches for the shift and "
                         "spread panels, e.g. '1.31,1.31'. Default keeps the "
                         "2.2/3.7 ratio, which places the shift panel 1.7x "
                         "shorter than the spread panel -- at a 241 pt slot "
                         "that is 17.5 mm and its y axis fits two ticks.")
    ap.add_argument("--stem-suffix", default="",
                    help="appended to the output stems, to render a variant "
                         "without overwriting the panels in use")
    ap.add_argument("--panels", default="abc", choices=["abc", "bc"],
                    help="abc: the standalone figure. bc: the two sequence "
                         "profiles alone, stacked, no suptitle and no baked "
                         "letters -- a panel for Figure 1 of the manuscript, "
                         "which supplies its own letter in its assembly "
                         "overlay. The scatter (a) is dropped there because "
                         "Figure 1 already carries the ensemble renders that "
                         "make the same point pictorially.")
    ap.add_argument("--all-seeds", action="store_true",
                    help="keep P286R's AF3 seeds 26-32 (160 predictions). Default: "
                         "seeds 1-25 on both sides, 125 each, as the pLDDT analyses "
                         "use since 2026-10-20 (analyses/annotation/stages)")
    ap.add_argument("--shift-note", default="full", choices=["full", "rmsd", "none"],
                    help="annotation in the centre-shift panel: full (RMSD, %% of "
                         "residues < 0.5 A, the 1-3 A yardstick), rmsd only, or none")
    ap.add_argument("--wt-colour", default=WT_C,
                    help="WT colour in the spread panel; Figure 1 uses #8d8d8d, the WT "
                         "grey of panel e (ks_sequence_panel.WT_C), since 2026-10-28")
    ap.add_argument("--wt-on-top", action="store_true",
                    help="draw the WT line over the mutant fill, slightly heavier, so a "
                         "grey WT stays visible under the red (Figure 1d, 2026-10-28)")
    ap.add_argument("--out-dir", type=Path, default=HERE,
                    help="where tables and panels are written (default: this folder)")
    a = ap.parse_args()
    OUT = a.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    # Axis wording follows the estimator: Fig. 1d is RMSF since 2026-10-18, and the
    # text calls it RMSF, so the panel must not say the generic 'spread'.
    SPREAD_LABEL = r"C$\alpha$ RMSF" if a.metric == "rmsf" else "spread"
    art = f"POLE_268-471_{a.mutant}"

    d = R.read_cache([R.WT_PRIMARY, art], a.keep_duplicate,
                     max_seed=None if a.all_seeds else 25)
    counts = d.groupby("artifact").structure_index.nunique()
    n = int(counts.min())
    print("structures per side: " + ", ".join(f"{k} {int(v)}" for k, v in counts.items()))
    wt = R.residue_table(R.load(R.WT_PRIMARY, d, n), a.metric)
    mu = R.residue_table(R.load(art, d, n), a.metric)
    w, m = wt.disp.to_numpy(), mu.disp.to_numpy()
    shift = np.linalg.norm(wt[R.XYZ].to_numpy() - mu[R.XYZ].to_numpy(), axis=1)
    pos = wt.resid.to_numpy() + 267
    site = int("".join(c for c in a.mutant[1:-1] if c.isdigit()))

    wider, ratio = int((m > w).sum()), float(np.median(m / w))
    rmsd = float(np.sqrt((shift ** 2).mean()))
    print(f"n={n} structures per side, {len(w)} residues")
    print(f"  broader at {wider}/{len(w)} residues ({100*wider/len(w):.0f}%), "
          f"median ratio {ratio:.2f}x")
    print(f"  centre shift RMSD {rmsd:.3f} A; {100*(shift<0.5).mean():.0f}% of "
          f"residues move < 0.5 A")
    pd.DataFrame({"pole_resid": pos, "wt": w, "mut": m, "centre_shift": shift,
                  "wt_resname": wt.resname, "mut_resname": mu.resname}
                 ).to_csv(OUT / f"exemplar_{a.mutant}_{a.metric}.csv", index=False)

    # ------------------------------------------------------------------
    # Each panel as a function, so the standalone figure and the two files
    # Figure 1 places draw from ONE definition of each plot. They diverge
    # only in what a panel standing alone has to carry: its own letter, its
    # own x axis, its own scale bar.
    # ------------------------------------------------------------------
    def scatter_panel(ax, letter=None):
        hi = max(w.max(), m.max()) * 1.06
        ax.plot([0, hi], [0, hi], color="0.45", lw=1, ls="--", zorder=1)
        ax.scatter(w, m, s=16, c=MU_C, alpha=.55, edgecolor="none", zorder=2)
        j = int(np.where(pos == site)[0][0]) if site in pos else None
        if j is not None:
            ax.scatter([w[j]], [m[j]], s=70, facecolor="none", edgecolor="k",
                       lw=1.3, zorder=3)
            ax.annotate(f"substituted\nresidue {site}", (w[j], m[j]),
                        textcoords="offset points", xytext=(10, -2), fontsize=8,
                        color="0.25", linespacing=1.15)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_xlabel(rf"WT per-residue {SPREAD_LABEL} ($\mathrm{{\AA}}$)")
        ax.set_ylabel(rf"{a.mutant} per-residue {SPREAD_LABEL} ($\mathrm{{\AA}}$)")
        ax.text(.04, .96,
                f"broader at {wider} of {len(w)} residues\n"
                rf"median {ratio:.2f}$\times$ wider" "\n\n"
                "mean structures superimpose\n"
                rf"to {rmsd:.2f} $\mathrm{{\AA}}$ RMSD",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.45", fc="#f5f5f5", ec="0.75", lw=.8))
        if letter:
            ax.set_title(letter, loc="left", fontweight="bold", fontsize=11)
        ax.grid(alpha=.18)

    def spread_panel(bx, xaxis, letter=None):
        """Per-residue spread of both ensembles along the sequence."""
        bx.fill_between(pos, w, color=a.wt_colour, alpha=.42, lw=0, label="WT")
        bx.fill_between(pos, m, color=MU_C, alpha=.42, lw=0, label=a.mutant)
        bx.plot(pos, m, color=MU_C, lw=.9)
        bx.plot(pos, w, color=a.wt_colour, lw=1.3 if a.wt_on_top else .9,
                zorder=3 if a.wt_on_top else None)
        bx.axvline(site, color="k", lw=.9, ls=":")
        bx.annotate(f"{a.mutant}", (site, bx.get_ylim()[1]),
                    textcoords="offset points", xytext=(4, -22), fontsize=8.5)
        bx.set_xlim(pos.min(), pos.max())
        bx.set_ylabel(rf"{SPREAD_LABEL} ($\mathrm{{\AA}}$)")
        bx.legend(fontsize=8.5, frameon=False, loc="upper right", ncol=2)
        if letter:
            bx.set_title(letter, loc="left", fontweight="bold", fontsize=11)
        bx.grid(alpha=.18)
        if xaxis:
            bx.set_xlabel("POLE residue")
        else:
            bx.tick_params(labelbottom=False)

    def shift_panel(cx, xaxis, band, letter=None):
        """Distance between the two ensemble means, on its own scale."""
        if band:
            # The structural yardstick, and the reason this is a second panel
            # rather than a second curve on the one above: see the docstring.
            # It comes OFF in Figure 1, where the same yardstick is in the
            # corner annotation in words and the band's label lands on the
            # residue-290 spike at that scale.
            cx.axhspan(0, 0.5, color="#dfe9f2", zorder=0)
            cx.text(pos.min() + 3, 0.46, "superimposable", fontsize=7.5,
                    va="top", color="#3b6ea5")
        cx.plot(pos, shift, color="#2c3e50", lw=.9)
        cx.axvline(site, color="k", lw=.9, ls=":")
        cx.set_xlim(pos.min(), pos.max())
        cx.set_ylim(0, max(1.45, shift.max() * 1.08))
        # Says what is computed (2026-10-23, author's request): the distance
        # between the two ensembles' mean C-alpha positions, not a "shift" of
        # anything the reader has to look up.
        cx.set_ylabel(r"$\Vert\overline{\mathrm{Pos}}_{\mathrm{WT}}"
                      r"-\overline{\mathrm{Pos}}_{\mathrm{MUT}}\Vert$" "\n"
                      r"($\mathrm{\AA}$)")
        if letter:
            cx.set_title(letter, loc="left", fontweight="bold", fontsize=11)
        cx.grid(alpha=.18)
        # --shift-note rmsd (2026-10-28, author's request): the RMSD alone; the
        # %<0.5 A count and the 1-3 A yardstick crowded the panel.
        note = {"full": rf"RMSD {rmsd:.2f} $\mathrm{{\AA}}$   "
                        f"{100*(shift<0.5).mean():.0f}% of residues "
                        rf"$<$ 0.5 $\mathrm{{\AA}}$" "\n"
                        r"(1$-$3 $\mathrm{\AA}$ separates alternative conformations)",
                "rmsd": rf"RMSD {rmsd:.2f} $\mathrm{{\AA}}$",
                "none": ""}[a.shift_note]
        if note:   # alone it can be read at tick-label size (2026-10-28)
            cx.text(.985, .92, note, transform=cx.transAxes, ha="right", va="top",
                    fontsize=7.5 if a.shift_note == "full" else 10,
                    color="0.25", linespacing=1.3)
        if xaxis:
            cx.set_xlabel("POLE residue")
        else:
            cx.tick_params(labelbottom=False)

    if a.panels == "bc":
        # TWO FILES, not one two-axes panel. Each is a lettered panel of
        # Figure 1 (c and d), and make_figures.py draws letters in its own
        # overlay -- one per placed file -- so a panel that needs two letters
        # has to be two files. Both are 7.4 in wide with the same left/right
        # margins, so the two residue axes align exactly once the assembly
        # scales them to a common width.
        #
        # ORDER IS REVERSED against the standalone figure: the centre shift
        # goes on TOP. The claim reads "the substitution left the structure
        # where it was, and changed how tightly it is predicted", and Figure 1
        # now reads in that order. 2.2 in against 3.7 in because the shift is
        # one flat line and the spread is two curves and a filled difference.
        #
        # 7.4 in against the ~362 pt slot is scale 0.68, which holds the 8.5 pt
        # annotations near 6 pt in print; the 10.6 in standalone width would
        # land them at 4 pt.
        # Heights scale with the width so the panels keep the proportions
        # they have today; only the text-to-panel ratio changes.
        wscale = a.panel_width_in / 7.4
        if a.bc_heights:
            hs = [float(v) for v in a.bc_heights.split(",")]
            if len(hs) != 2:
                raise SystemExit("--bc-heights needs exactly two values")
        else:
            hs = [2.2 * wscale, 3.7 * wscale]
        # has_xaxis drives the bottom margin. It used to be inferred from the
        # panel's height, which breaks the moment the two are given the SAME
        # height: the spread panel would take the short-panel margin and clip
        # its residue axis.
        for stem, h, draw, has_xaxis in (
                (f"POLE_exemplar_shift_WT_vs_{a.mutant}{a.stem_suffix}", hs[0],
                 lambda cx: shift_panel(cx, xaxis=a.shift_xaxis, band=False),
                 a.shift_xaxis),
                (f"POLE_exemplar_spread_WT_vs_{a.mutant}{a.stem_suffix}", hs[1],
                 lambda bx: spread_panel(bx, xaxis=True), True)):
            fig = plt.figure(figsize=(a.panel_width_in, h))
            draw(fig.add_subplot(111))
            # No suptitle and tight margins: whitespace inside a panel file is
            # whitespace on the assembled page, and a title inside one panel of
            # an assembled figure reads as a title for the whole figure.
            # Margins are FRACTIONS, so a narrower authored width shrinks them
            # in points while the fonts stay the same size -- at 3.28 in the
            # 0.088 left margin is 21 pt against 47 pt at 7.4 in, and the
            # y-axis label is clipped off. Dividing by wscale keeps every
            # margin the same ABSOLUTE size, which is what the axis furniture
            # actually needs. Heights scaled by wscale too, so one factor does
            # both directions.
            top0 = .98
            bot0 = .155 if has_xaxis else .055
            # .115, not .088 (2026-10-23): c's y label is two lines since it
            # spells out the distance of means; d takes the same margin so the
            # two residue axes still line up.
            fig.subplots_adjust(
                left=min(.115 / wscale, .40),
                right=max(1 - (1 - .985) / wscale, .60),
                top=max(1 - (1 - top0) / wscale, .60),
                bottom=min(bot0 / wscale, .40))
            for ext in ("png", "pdf"):
                fig.savefig(OUT / f"{stem}.{ext}", dpi=200)
            plt.close(fig)
            print(f"wrote {OUT}/{stem}.pdf")
        return

    fig = plt.figure(figsize=(10.6, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.45], height_ratios=[1.9, 1],
                          hspace=.16, wspace=.28)
    scatter_panel(fig.add_subplot(gs[:, 0]), letter="a")
    bx = fig.add_subplot(gs[0, 1])
    spread_panel(bx, xaxis=False, letter="b")
    shift_panel(fig.add_subplot(gs[1, 1], sharex=bx), xaxis=True, band=True,
                letter="c")
    fig.suptitle(f"{a.mutant} leaves the predicted POLE structure unchanged "
                 f"and broadens the predicted ensemble", fontsize=11, y=.99)
    fig.subplots_adjust(top=.90, left=.075, right=.985, bottom=.10)
    stem = f"POLE_exemplar_WT_vs_{a.mutant}"
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200)
    print(f"wrote {OUT}/{stem}.png")


if __name__ == "__main__":
    main()
