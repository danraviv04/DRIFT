#!/usr/bin/env python
"""Residue-KS explainer, rebuilt as two panels.

WHAT THEY ESTABLISH
-------------------
(a) At a single residue the WT, driver and passenger ensembles are three tight
    pLDDT distributions on the same scale. The driver's sits displaced by a few
    tenths of a pLDDT point -- small in absolute terms, but several WT standard
    deviations, which is why a two-sample KS at that residue reaches 1.0 while
    the passenger's stays near the WT. Shown for a contiguous stretch of
    sequence so the reader sees the residue-to-residue variation, not one
    hand-picked site.

(b) Those per-residue KS values are a 403-long profile. Its mean IS the variant
    score, and that score is one point in the PTEN cohort. Panels b1 -> b3 share
    one y axis, so "per-residue KS", "distribution of per-residue KS" and
    "variant score" are literally the same quantity read three ways.

WHY THE y AXIS OF (a) IS A DIFFERENCE, NOT RAW pLDDT
----------------------------------------------------
Per-residue WT spread is 0.03-0.25 pLDDT points while the absolute level moves
over 6 points across a 16-residue window. On a shared absolute axis every
distribution is a flat line. Panel (a) therefore plots

    pLDDT  -  mean WT pLDDT at that residue

and prints the absolute WT mean under each residue label. Nothing is rescaled
per residue beyond that shift: one pLDDT point is the same height everywhere,
so the driver's displacement can be compared across residues by eye.

VIOLIN WIDTH
------------
Gaussian KDE, display only -- every KS in this figure is computed on the raw
values at the exact 0.01 pLDDT reporting grid, unsmoothed. Within one residue
the three violins share one width scale, so their relative widths are
comparable; between residues only position is comparable.

WT DRAW
-------
The canonical pipeline draws WT independently per variant from the same 125
structure pool. Panel (a) needs one WT to show, so it uses the driver's
canonical draw for both comparisons and its per-residue KS strip is computed
against that shared draw. Panel (b) is canonical throughout -- independent
per-variant draws, verified against per_variant_seed.csv.gz. The only quantity
this moves is the passenger's mean KS (shared vs canonical); both are printed
in the footnote and recorded in the values CSV.

    python figures/si/lib/ks_sequence_panel.py
    ... --gene PTEN --driver A121E --passenger A79T --window 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figlib import *  # noqa: F403,E402

from ratio_exemplar import CANONICAL_SEED, PER_VARIANT, sampled_pair  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402
from scipy.stats import gaussian_kde, ks_2samp  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

WT_C = "#8d8d8d"
DRIVER_C, PASSENGER_C = DRV, "#2e86ab"

THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
             "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
             "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
             "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def sequence(gene, cohort_frame):
    """One-letter sequence from the WT coordinate cache, checked against every
    cohort mutation code. A silent off-by-one here would mislabel the x axis."""
    # The WT artifact is named after the modelled dataset: MSH6 is chain B of
    # MutS_alpha_MSH6_GT, so its WT is MutS_alpha_MSH6_GT_WT_data, not MSH6_WT_data.
    parent = (cohort_frame.parent_dataset.iloc[0] if "parent_dataset" in cohort_frame
              else to_dir_key(gene))
    d = coords(gene, artifact=f"{parent}_WT_data",
               columns=["structure_index", "resid", "resname"])
    if d.empty:
        raise SystemExit(f"no WT entry in the {gene} coordinate cache")
    one = d[d.structure_index == d.structure_index.min()].sort_values("resid")
    seq = "".join(THREE2ONE.get(r, "X") for r in one.resname)
    resid = one.resid.to_numpy(int)
    bad = [c for c in cohort_frame.mutation_code
           if seq[int(c[1:-1]) - 1] != c[0]]
    if bad:
        raise SystemExit(f"sequence disagrees with {len(bad)} mutation codes, "
                         f"e.g. {bad[:5]}")
    print(f"[ks_sequence_panel] sequence {len(seq)} aa, "
          f"{len(cohort_frame)} mutation codes agree")
    return seq, resid


def residue_ks(w, m):
    """Per-residue two-sample KS on exact 0.01 pLDDT grid ticks (no smoothing)."""
    return np.array([ks_2samp(np.rint(a * 100).astype(int),
                              np.rint(b * 100).astype(int)).statistic
                     for a, b in zip(w, m)])


def vkde(ax, values, centre, half, grid, scale, colour, lw=0.6, alpha=0.5):
    """One right-facing HALF violin with its flat edge on x=centre.

    Half rather than mirrored: a mirrored violin spends half its ink restating
    the same density, and at three ensembles per residue that is what crowds the
    slot. The flat edge also gives each ensemble an unambiguous baseline to read
    its position against. No centre bar -- the density is the statement.
    """
    d = density(values, grid)
    ax.fill_betweenx(grid, centre, centre + d * scale,
                     color=colour, alpha=alpha, lw=0)
    # Outline only where the KDE has support, otherwise it draws a full-height
    # vertical line along the zero-density baseline.
    support = d > 0.004 * d.max()
    edge = np.where(support, d, np.nan)
    ax.plot(centre + edge * scale, grid, color=colour, lw=lw)
    # Flat edge only across the support, so it does not rule the whole panel.
    lo, hi = grid[support][0], grid[support][-1]
    ax.plot([centre, centre], [lo, hi], color=colour, lw=lw)
    return d


def density(values, grid):
    """KDE with a bandwidth floor at the 0.01 pLDDT reporting grid."""
    sd = float(values.std(ddof=1))
    if sd <= 0:
        out = np.zeros_like(grid)
        out[np.argmin(np.abs(grid - values[0]))] = 1.0
        return out
    scott = len(values) ** (-1.0 / 5.0)
    return gaussian_kde(values, bw_method=max(scott, 0.012 / sd))(grid)


# ── panel a ───────────────────────────────────────────────────────────────────

def site_of(code):
    return int("".join(c for c in code[1:-1] if c.isdigit()))


def panel_a(gene, driver, passenger, seq, resid, wt, mut_d, mut_p, ks_d, ks_p,
            lo, hi, args):
    idx = np.where((resid >= lo) & (resid <= hi))[0]
    n = len(idx)
    site, p_site = site_of(driver), site_of(passenger)
    # Each mutated residue is shaded in its class colour, so a window holding
    # both sites says which column belongs to which variant without a key.
    shade = {p_site: PASSENGER_C, site: DRIVER_C} if lo <= p_site <= hi else {site: "#f0c808"}

    delta = {"WT": [], driver: [], passenger: []}
    for i in idx:
        base = wt[i].mean()
        # z mode divides by the WT spread at that residue, which is what a KS
        # actually responds to; delta mode keeps real pLDDT points.
        scale = float(wt[i].std(ddof=1)) if args.y_mode == "z" else 1.0
        delta["WT"].append((wt[i] - base) / scale)
        delta[driver].append((mut_d[i] - base) / scale)
        delta[passenger].append((mut_p[i] - base) / scale)

    flat = np.concatenate([v for g in delta.values() for v in g])
    pad = 0.08 * (flat.max() - flat.min())
    ylim = (flat.min() - pad, flat.max() + pad)
    grid = np.linspace(*ylim, 512)

    fig = plt.figure(figsize=(W2, W2 * (0.42 if args.panel else 0.50)))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3.1], hspace=0.10,
                          left=0.085, right=0.995,
                          top=0.90 if args.panel else 0.865,
                          bottom=0.205 if args.panel else 0.185)
    ax_ks = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=ax_ks)

    # Half violins occupy one side only, so the three fit in a tighter slot.
    half, off = 0.245, 0.30
    for col, i in enumerate(idx):
        if int(resid[i]) in shade:
            for a in (ax, ax_ks):
                a.axvspan(col - 0.5, col + 0.5, color=shade[int(resid[i])],
                          alpha=0.16 if len(shade) == 1 else 0.10, lw=0, zorder=0)
        groups = [("WT", delta["WT"][col], -0.42, WT_C),
                  (driver, delta[driver][col], -0.42 + off, DRIVER_C),
                  (passenger, delta[passenger][col], -0.42 + 2 * off, PASSENGER_C)]
        peak = max(density(v, grid).max() for _, v, _, _ in groups)
        for _, v, dx, c in groups:
            vkde(ax, v, col + dx, half, grid, half / peak, c)
        ax_ks.bar([col - 0.16, col + 0.16], [ks_d[i], ks_p[i]], width=0.30,
                  color=[DRIVER_C, PASSENGER_C], alpha=0.85, lw=0)

    ax.axhline(0, color="#444", ls="--", lw=0.6, zorder=1)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(ylim)
    # i is the residue on the x axis; <.> is the mean over the WT ensemble. Each
    # violin is one ensemble's per-structure pLDDT at residue i, so the label
    # subtracts the WT MEAN from single structures -- not a mean difference,
    # which would be one number per residue rather than a distribution.
    ax.set_ylabel(r"$\mathrm{pLDDT}_i - \langle \mathrm{pLDDT}_i^{\,\mathrm{WT}} \rangle$"
                  if args.y_mode == "delta" else
                  r"$(\mathrm{pLDDT}_i - \langle \mathrm{pLDDT}_i^{\,\mathrm{WT}} \rangle)"
                  r"\,/\,\sigma_i^{\mathrm{WT}}$")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"$\\bf{{{seq[int(resid[i]) - 1]}}}$\n{int(resid[i])}"
                        for i in idx], linespacing=1.35)
    ax.tick_params(axis="x", length=0, pad=2)
    if len(shade) > 1:
        for col, i in enumerate(idx):
            if int(resid[i]) in shade:
                ax.get_xticklabels()[col].set_color(shade[int(resid[i])])
    for col, i in enumerate(idx):
        ax.annotate(f"{wt[i].mean():.1f}\n{wt[i].std(ddof=1):.2f}", xy=(col, -0.185),
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=5, color="#999", linespacing=1.4)
    ax.annotate("WT mean pLDDT\nWT SD", xy=(-0.012, -0.185), xycoords="axes fraction",
                ha="right", va="top", fontsize=5, color="#999", linespacing=1.4)

    ax_ks.set_ylim(0, 1.06)
    ax_ks.set_yticks([0, 0.5, 1])
    ax_ks.set_ylabel(r"$\mathrm{KS}_i$", labelpad=8)
    ax_ks.tick_params(axis="x", length=0, labelbottom=False)
    ax_ks.spines["bottom"].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.5, ec=c, lw=0.6)
               for c in (WT_C, DRIVER_C, PASSENGER_C)]
    fig.legend(handles, [f"WT  (n={wt.shape[1]})",
                         f"Driver {driver}  (n={mut_d.shape[1]})",
                         f"Passenger {passenger}  (n={mut_p.shape[1]})"],
               loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=3,
               handlelength=1.1, columnspacing=1.6, handletextpad=0.5,
               borderpad=0.2, fontsize=6.5)

    if args.panel:
        # Layout-ready: the figure's own title and footnote become the
        # manuscript caption, and the panel letter is drawn by the assembler.
        # Short on purpose. The legend already names all three ensembles and
        # their n, so spelling them out again both duplicated it and ran the
        # title straight into it in the assembled figure.
        ax_ks.set_title(f"{pretty(gene)} residues {lo}–{hi}",
                        pad=6, loc="left", fontsize=8)
    else:
        where = (f"shaded columns = the mutated residues, {driver} and {passenger}"
                 if len(shade) > 1 else
                 f"shaded column = the mutated residue {driver}; "
                 f"{passenger} is mutated at {p_site}, outside this window")
        ax_ks.set_title(
            f"{pretty(gene)} residues {lo}–{hi} — per-structure pLDDT for the WT, "
            f"driver and passenger ensembles\n({where})",
            pad=8, linespacing=1.5, loc="left")
        fig.text(0.085, 0.030,
                 "Half-violins are Gaussian KDEs for display only; every KS is computed on the raw "
                 "0.01-pLDDT-grid values. Within a residue the three share one width scale; the flat "
                 "edge is each ensemble's baseline.",
                 fontsize=5.4, color="#777")
    return fig, idx


# ── panel b ───────────────────────────────────────────────────────────────────

def panel_b(gene, driver, passenger, resid, ks_d, ks_p, lo, hi, args):
    pv = pd.read_csv(PER_VARIANT)
    coh = pv[(pv.protein == to_dir_key(gene)) & (pv.requested_size == args.size)
             & (pv.seed == args.seed)].copy()
    if coh.empty:
        raise SystemExit("canonical per-variant table has no rows for this cell")
    auc = roc_auc_score(coh.label, coh.ks_raw)

    fig = plt.figure(figsize=(W2, W2 * 0.33))
    gs = fig.add_gridspec(1, 3, width_ratios=[3.1, 1.0, 1.0], wspace=0.13,
                          left=0.065, right=0.995, top=0.775, bottom=0.235)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0]),) + tuple(
        fig.add_subplot(gs[i]) for i in (1, 2))

    series = [(driver, ks_d, DRIVER_C, "Driver"),
              (passenger, ks_p, PASSENGER_C, "Passenger")]

    ax1.axvspan(lo, hi, color="#f0c808", alpha=0.20, lw=0, zorder=0)
    ax1.annotate(f"panel a  {lo}–{hi}", xy=(hi + 6, 0.025), ha="left",
                 va="bottom", fontsize=5.5, color="#8a7500")
    for code, k, c, role in series:
        ax1.plot(resid, k, color=c, lw=0.55, alpha=0.9, zorder=3,
                 label=f"{role} {code}")
        ax1.axhline(k.mean(), color=c, ls="--", lw=0.8, zorder=4)
    ax1.legend(loc="lower left", bbox_to_anchor=(0.30, 1.005), ncol=2,
               handlelength=1.4, columnspacing=1.6, fontsize=6)
    ax1.set(xlim=(resid.min(), resid.max()), ylim=(0, 1.04),
            xlabel=f"{pretty(gene)} residue number", ylabel="Per-residue KS")
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax1.set_title(f"b1  one KS per residue  ({len(resid)} residues)",
                  loc="left", pad=16)

    bins = np.linspace(0, 1, 41)
    for code, k, c, role in series:
        ax2.hist(k, bins=bins, orientation="horizontal", density=True,
                 color=c, alpha=0.35, lw=0)
        ax2.hist(k, bins=bins, orientation="horizontal", density=True,
                 histtype="step", color=c, lw=0.8)
        ax2.axhline(k.mean(), color=c, ls="--", lw=0.8)
        ax2.annotate(f"{k.mean():.3f}", xy=(0.97, k.mean()), xycoords=("axes fraction", "data"),
                     ha="right", va="bottom" if code == driver else "top",
                     fontsize=6, color=c, weight="bold")
    ax2.set(ylim=(0, 1.04), xlabel="density")
    ax2.tick_params(labelleft=False)
    ax2.set_title("b2  their distribution;\n      the mean is the score",
                  loc="left", pad=16, linespacing=1.4)

    rng = np.random.default_rng(0)
    for label, x, c, name in ((0, 0.0, PASSENGER_C, "Passenger"),
                              (1, 1.0, DRIVER_C, "Driver")):
        v = coh[coh.label == label].ks_raw.to_numpy()
        ax3.scatter(x + rng.uniform(-0.17, 0.17, len(v)), v, s=3.2, color=c,
                    alpha=0.45, lw=0, zorder=2)
        ax3.plot([x - 0.3, x + 0.3], [np.median(v)] * 2, color=c, lw=1.2, zorder=4)
        ax3.annotate(f"{name}\nn={len(v)}", xy=(x, -0.075), xycoords=("data", "axes fraction"),
                     ha="center", va="top", fontsize=5.8, color=c, linespacing=1.3)
    for code, k, c, role in series:
        y = float(coh.loc[coh.mutation_code == code, "ks_raw"].iloc[0])
        np.testing.assert_allclose(y, k.mean(), atol=1e-12, rtol=0)
        x = 1.0 if role == "Driver" else 0.0
        ax3.scatter([x], [y], s=26, facecolor="none", edgecolor="#172c45",
                    lw=0.9, zorder=6)
        ax3.add_artist(ConnectionPatch(
            xyA=(1.0, k.mean()), coordsA=ax2.get_yaxis_transform(),
            xyB=(x, y), coordsB=ax3.transData, axesA=ax2, axesB=ax3,
            color=c, lw=0.7, ls=(0, (3, 2)), alpha=0.9, zorder=1,
            arrowstyle="-|>", mutation_scale=6, shrinkB=5))
        ax3.annotate(code, xy=(x, y), xytext=(0, 9), textcoords="offset points",
                     ha="center", va="bottom", fontsize=6, color="#172c45",
                     weight="bold")
    ax3.set(xlim=(-0.55, 1.55), ylim=(0, 1.04))
    ax3.set_xticks([])
    ax3.tick_params(labelleft=False)
    ax3.annotate(f"ROC-AUC = {auc:.3f}", xy=(0.5, 0.035), xycoords="axes fraction",
                 ha="center", fontsize=6, color="#444")
    ax3.set_title(f"b3  every {pretty(gene)} variant's score\n      "
                  f"(N={args.size}, canonical seed)", loc="left", pad=16,
                  linespacing=1.4)
    for a in (ax2, ax3):
        a.set_ylim(0, 1.04)

    fig.text(0.065, 0.022,
             f"DRIFT KS(variant) = mean over the {len(resid)} residues of the per-residue "
             f"two-sample KS between the WT and mutant pLDDT ensembles. "
             f"Panel b is canonical: independent per-variant WT draws, N={args.size}, "
             f"seed {args.seed}.", fontsize=5.4, color="#777")
    return fig, coh, auc


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene", default="PTEN")
    ap.add_argument("--driver", default="A121E")
    ap.add_argument("--passenger", default="A79T")
    ap.add_argument("--size", type=int, default=90)
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED)
    ap.add_argument("--window", type=int, default=16,
                    help="residues shown in panel a, centred on both mutated sites "
                         "when they fit, otherwise on the driver site")
    ap.add_argument("--window-start", type=int, default=None,
                    help="override: first residue of the panel-a window")
    ap.add_argument("--y-mode", choices=("delta", "z"), default="delta",
                    help="panel a y axis: pLDDT points, or WT standard deviations")
    ap.add_argument("--panel", action="store_true",
                    help="emit the window as a LAYOUT-ready manuscript panel: "
                         "no figure title block or footnote (those become the "
                         "caption), tighter margins, stem PTEN_residue_distributions. "
                         "Figure1_resKS consumes it in place of "
                         "PTEN_profiles_and_ratio_resKS.")
    ap.add_argument("--out-dir", type=Path, default=RESULTS,
                    help="where panels, CSVs and run_metadata.json go (default: results/)")
    args = ap.parse_args(argv)
    gene = to_dir_key(args.gene)
    out = args.out_dir

    coh = cohort(gene)
    seq, _ = sequence(gene, coh)
    recs = coh.set_index("mutation_code")
    for code in (args.driver, args.passenger):
        if code not in recs.index:
            raise SystemExit(f"{code} is not in the {gene} canonical cohort")

    pair = {}
    for code in (args.driver, args.passenger):
        rec = recs.loc[code].to_dict() | {"protein": gene, "mutation_code": code}
        pair[code] = sampled_pair(rec, args.size, args.seed)
    resid = pair[args.driver][0]
    if not np.array_equal(resid, pair[args.passenger][0]):
        raise SystemExit("the two variants do not share a residue index")
    assert int(recs.loc[args.driver, "label"]) == 1
    assert int(recs.loc[args.passenger, "label"]) == 0

    _, wt_shared, mut_d, n_d = pair[args.driver]
    _, wt_p, mut_p, n_p = pair[args.passenger]

    # canonical: each variant against its own WT draw -- must reproduce the table
    ks_can = {c: residue_ks(pair[c][1], pair[c][2]) for c in pair}
    ref = pd.read_csv(PER_VARIANT)
    for code, k in ks_can.items():
        row = ref[(ref.protein == gene) & (ref.mutation_code == code)
                  & (ref.requested_size == args.size) & (ref.seed == args.seed)]
        if len(row) != 1:
            raise SystemExit(f"no unique canonical row for {code}")
        np.testing.assert_allclose(k.mean(), row.ks_raw.iloc[0], atol=1e-12, rtol=0)
        print(f"[ks_sequence_panel] {code} canonical KS {k.mean():.6f}  OK")

    # panel a: one displayed WT, so its KS strip uses the shared draw
    ks_shared = {c: (ks_can[c] if c == args.driver
                     else residue_ks(wt_shared, mut_p)) for c in pair}
    drift = abs(ks_shared[args.passenger].mean() - ks_can[args.passenger].mean())
    print(f"[ks_sequence_panel] {args.passenger} shared-WT KS "
          f"{ks_shared[args.passenger].mean():.6f}  (drift {drift:.6f})")
    if drift > 0.02:
        raise SystemExit("shared WT draw moves the passenger score too far")

    # Centre on both mutated residues when they fit in one window, so the
    # passenger's own site is on the page beside the driver's; otherwise on the
    # driver's.
    site, p_site = site_of(args.driver), site_of(args.passenger)
    first, last = sorted((site, p_site))
    if args.window_start is not None:
        lo = args.window_start
    elif last - first < args.window:
        lo = first - (args.window - 1 - (last - first)) // 2
    else:
        lo = site - args.window // 2
    lo = int(np.clip(lo, resid.min(), resid.max() - args.window + 1))
    hi = lo + args.window - 1

    fig, idx = panel_a(gene, args.driver, args.passenger, seq, resid, wt_shared,
                       mut_d, mut_p, ks_shared[args.driver],
                       ks_shared[args.passenger], lo, hi, args)
    suffix = "" if args.y_mode == "delta" else f"_{args.y_mode}"
    if args.panel:
        save(fig, f"{gene}_residue_distributions", out)
    else:
        save(fig, f"{gene}_ks_sequence_window_{args.driver}_vs_{args.passenger}{suffix}", out)

    fig, coh_scores, auc = panel_b(gene, args.driver, args.passenger, resid,
                                   ks_can[args.driver], ks_can[args.passenger],
                                   lo, hi, args)
    save(fig, f"{gene}_ks_aggregation_{args.driver}_vs_{args.passenger}", out)

    # ── quotable numbers ──
    tag = f"{gene}_ks_panels"
    record(tag, "window", f"{lo}-{hi}", "panel a residues")
    record(tag, "window_sequence", seq[lo - 1:hi])
    record(tag, "cohort_auc_ks", float(auc),
           f"{len(coh_scores)} variants, N={args.size}, seed={args.seed}")
    for code, role in ((args.driver, "driver"), (args.passenger, "passenger")):
        k = ks_can[code]
        record(tag, f"{code}_sepa_ks_canonical", float(k.mean()), role)
        record(tag, f"{code}_sepa_ks_shared_wt", float(ks_shared[code].mean()),
               "panel a; shared driver WT draw")
        record(tag, f"{code}_window_mean_ks", float(k[(resid >= lo) & (resid <= hi)].mean()))
        record(tag, f"{code}_residues_ks_above_0.9", int((k > 0.9).sum()))
    per_res = pd.DataFrame({"residue": resid.astype(int),
                            "aa": [seq[int(r) - 1] for r in resid],
                            f"ks_{args.driver}": ks_can[args.driver],
                            f"ks_{args.passenger}": ks_can[args.passenger],
                            "wt_mean_plddt": wt_shared.mean(1),
                            "wt_sd_plddt": wt_shared.std(1),
                            f"{args.driver}_mean_plddt": mut_d.mean(1),
                            f"{args.passenger}_mean_plddt": mut_p.mean(1)})
    per_res.to_csv(out / f"{gene}_ks_panels_per_residue.csv", index=False)
    window = per_res[(per_res.residue >= lo) & (per_res.residue <= hi)]
    rows = []
    for code, mat in (("WT", wt_shared), (args.driver, mut_d), (args.passenger, mut_p)):
        for i in idx:
            for j, v in enumerate(mat[i]):
                rows.append(dict(ensemble=code, residue=int(resid[i]),
                                 aa=seq[int(resid[i]) - 1], sampled_column=j, plddt=v))
    pd.DataFrame(rows).to_csv(out / f"{gene}_ks_panels_window_samples.csv",
                              index=False)
    print(window.to_string(index=False))
    write_values(f"ks_sequence_panel_values_{gene}.csv", out)
    provenance(out, figure=f"ks_sequence_panel_{gene}", size=args.size, seed=args.seed,
               driver=args.driver, passenger=args.passenger,
               window=[lo, hi], canonical_parity=True,
               panel_a_wt="shared driver WT draw (display)",
               panel_b_wt="canonical independent per-variant WT draws",
               ks_definition="mean over residues of two-sample KS on exact 0.01 pLDDT ticks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
