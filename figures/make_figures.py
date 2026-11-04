#!/usr/bin/env python
"""Assemble Figure 1 from its individual panels. Vector in, vector out.

    python make_figures.py Figure1    # build it (also the default: every layout)
    python make_figures.py --list     # show panels available and what is missing

Design notes
------------
Self-contained on purpose. This script imports nothing else from the repository,
so a figure can be rebuilt from its panels without the score caches or the
external drive. The only dependencies are pypdf and matplotlib.

Panels stay vector. Each panel PDF is placed into the output page with pypdf's
page merging, so text in a plot remains text at any magnification -- assembling
from PNGs would silently rasterise the whole figure, which is the most common way
a good figure set ends up looking soft in print. Raster panels (a ChimeraX render
has no vector equivalent) are placed as images, which is correct for them.

Panel letters are drawn in a separate matplotlib overlay and merged on top,
rather than baked into the panels. That way re-lettering after a re-cut is a
one-line edit to LAYOUTS and never requires regenerating a panel.

Editing a figure means editing LAYOUTS. Coordinates are in points from the
bottom-left of the page (PDF convention), which is also what a journal's figure
guidelines are usually written in.

    (panel, x, y, w, letter)
        panel   file stem, resolved from figures/panels/ then figures/source/
        x, y    bottom-left corner, points
        w       target width, points; height follows the panel's aspect ratio
        letter  panel label, or None for no label

A layout may also carry

    "notes": [(text, x, y, fontsize, ha), ...]

for the rare label that belongs to the assembly rather than to any one panel --
a magnification saying which panel it magnifies, say. It is drawn in the same
overlay as the letters. Keep these few: a label that describes ONE panel's
content belongs in that panel, where the generator can keep it in step with the
data, or in the caption.

Missing panels are not an error: a dashed placeholder box is drawn with the
expected filename, so the figure can be assembled and reviewed before the
hand-made assets exist.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent          # this script lives in figures/ (tools/ in the paper repo)
FIGDIR = REPO / "figures"
PANELS = FIGDIR / "panels"        # drawn by the figure scripts (vector)
SOURCE = FIGDIR / "source"        # hand-made: renders, schematics
SEARCH = (PANELS, SOURCE)

MM = 72.0 / 25.4          # points per millimetre
WIDE = 260 * MM           # wider working width. Three panels across a row
                          # need more than a 183 mm column to stay legible;
                          # scale to the journal's column at submission.


def find_panel(stem):
    """panels/ (generated) then source/ (hand-made) then si_panels/ (exported)."""
    for d in SEARCH:
        for ext in (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svg"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def _aspect(stem, default=1.4):
    """width/height of a panel, measured from the file. Default if absent.

    Measured rather than assumed because the supplementary grids mix panels
    whose aspect ratios run from 0.97 to 3.6; a single assumed ratio would
    overlap rows in some grids and leave gaps in others. A missing panel falls
    back to `default` so a layout still builds (with a placeholder box).
    """
    p = find_panel(stem)
    if p is None:
        return default
    try:
        if p.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            box = PdfReader(str(p)).pages[0].mediabox
            return float(box.width) / float(box.height)
        from PIL import Image
        with Image.open(p) as im:
            return im.size[0] / im.size[1]
    except Exception:
        return default


def _size(stem, default=(368.4, 296.4)):
    """(width, height) of a panel in points, measured from the file.

    Same measurement as _aspect, kept separate because scaled_row needs the
    absolute native width and not just the ratio.

    RASTER PANELS ARE CONVERTED THROUGH THEIR DPI, which matters as soon as a
    row mixes raster with vector: the pipeline PNGs are 300 dpi, so a 2400 px
    panel is 576 pt of paper and not 2400. Returning pixels made scaled_row
    hand that panel four times the width of a vector neighbour and squeeze the
    rest of the row to nothing. A PNG with no dpi tag falls back to 72 dpi
    (pixels = points), which is the old behaviour.
    """
    p = find_panel(stem)
    if p is None:
        return default
    try:
        if p.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            box = PdfReader(str(p)).pages[0].mediabox
            return float(box.width), float(box.height)
        from PIL import Image
        with Image.open(p) as im:
            w, h = im.size
            dx, dy = im.info.get("dpi", (72.0, 72.0))
        return float(w) * 72.0 / float(dx or 72.0), float(h) * 72.0 / float(dy or 72.0)
    except Exception:
        return default


def matched_row(stems, y, letters, page_w=None, x0=10, gap=10):
    """One row placed at a COMMON HEIGHT, filling the width.

    The alternative to scaled_row, and the trade is explicit: equal heights
    line the panels up top and bottom, which is what a row of three plots
    should look like, but they can only be equal if the panels are scaled
    unequally. Here f and g are 296.4 pt tall natively and h is 360, so
    matching heights puts h at scale 0.50 against their 0.61 -- its text is
    about a fifth smaller than theirs. Use scaled_row when one text size across
    the row matters more, this when the alignment does.

    Panels sit on `y`; every one has the same height by construction.
    """
    page_w = page_w or WIDE
    nat = [_size(t) for t in stems]
    avail = page_w - 2 * x0 - (len(stems) - 1) * gap
    h = avail / sum(w / hh for w, hh in nat)      # common height, widths fill
    out, x = [], x0
    for stem, (w, hh), letter in zip(stems, nat, letters):
        out.append((stem, round(x, 1), y, round(h * w / hh, 1), letter))
        x += h * w / hh + gap
    return out


# The live figure layouts; run_all.sh assembles Figure 1 (Figure 2 and the SI
# figures are drawn whole by their own scripts).
LAYOUTS = {}


# =============================================================================
# 2026-10-28. FIGURE 1 -- the live figure, figures/Figure1.pdf (as Figure2.pdf).
# Every earlier Figure 1 layout above is a superseded variant; their built PDFs are
# in figures/archive/fig1_variants_2026-10-28 (a rebuild still writes to figures/).
#
# Rows, bottom up: f g h | e | c d | a beside b. Built to print as large as the
# manuscript page allows (the page holds it by height):
#   b   the stacked render pairs with 96 pt magnifier circles, text x1.2 and the
#       RMSF key as a bar underneath (analysis repo figures/figure1/fig1b/
#       render_panel_b.py --style tubes_rmsf --compose-only --stack-bubble-d 96
#       --font-scale 1.2 --key-below); without a key column b is narrower, so a and
#       b share a taller row.
#   c,d exemplar_wt_vs_mut.py ... --panel-width-in 4.4 --bc-heights 1.8,1.8
#       --shift-note rmsd (RMSD only, at 10 pt); text ~10% larger once placed.
#   e-h the same plots and data drawn at 0.87 (e, h) or 0.93 (f, g) of their usual
#       size, so their text prints 15% / 8% larger (analysis repo
#       figures/figure1/layout/rerender_efgh.py; checked pixel-identical to the
#       panels they replace at 1.0).
# Gaps are tight (12 pt between rows, 22 around c/d), d draws WT in grey (panel e uses grey; d is
# darker, #5a5a5a, line on top: --wt-colour "#5a5a5a" --wt-on-top), and f-h sit at 0.9 of their B-layout width,
# because every point of height saved here prints the whole figure larger.
# =============================================================================
def stacked_rows(rows, row_ab, page_w=None, x0=10, gap_ab=14, row_gap=12, top=12, y0=14,
                 gaps=None):
    """Rows of panels bottom-up, each spread over the full width, then a and b on top
    at one common height. ``rows`` is a list of [(stem, width, letter), ...]; a row of
    several panels is spread edge to edge, a single panel is centred. ``gaps`` gives
    the gap above each row (default row_gap for all)."""
    W = (page_w or WIDE) - 2 * x0
    panels, y = [], y0
    for row, row_gap in zip(rows, gaps or [row_gap] * len(rows)):
        gap = (W - sum(w for _, w, _ in row)) / (len(row) - 1) if len(row) > 1 else 0
        x = x0 if len(row) > 1 else x0 + (W - row[0][1]) / 2
        for stem, w, letter in row:
            panels.append((stem, round(x, 1), round(y, 1), round(w, 1), letter))
            x += w + gap
        y += max(w / _aspect(st) for st, w, _ in row) + row_gap
    (a, _, la), (b, _, lb) = row_ab
    h = (W - gap_ab) / (_aspect(a) + _aspect(b))
    panels += [(a, x0, round(y, 1), round(h * _aspect(a), 1), la),
               (b, round(x0 + h * _aspect(a) + gap_ab, 1), round(y, 1), round(h * _aspect(b), 1), lb)]
    return {"size": (page_w or WIDE, round(y + h + top, 1)), "panels": panels}


_FGH = ["MSH6_wt_null_panel_big", "MSH6_ROC_resKS_big", "MSH6_Nsweep_resKS_big"]
LAYOUTS["Figure1"] = {
    **stacked_rows(
        # f-h at one height filling the row (matched_row), then 0.9 of that width
        [[(st, w * .9, L) for st, _, _, w, L in matched_row(_FGH, y=0, letters="fgh")],
         [("MSH6_residue_distributions_flat_big", 717.0, "e")],
         [("POLE_centre_shift_row_big", 353.5, "c"), ("POLE_spread_profile_row_big", 353.5, "d")]],
        [("graphical_abstract_2026-10-25", None, "a"),
         ("fig1b_tubes_rmsf_stacked_keybelow", None, "b")],
        # 2026-10-28: 22 pt, not 12, above and below c/d, so the strip reads as its own row
        gaps=[12, 22, 22]),
    "caption": "A missense mutation perturbs the predicted ensemble beyond its mean structure",
    "boxes": (),
}


RASTER = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def overlay(size, items, missing, notes=(), boxes=()):
    """One-page PDF holding the panel letters, notes, frames and placeholders."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    w, h = size
    fig = plt.figure(figsize=(w / 72, h / 72))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")

    for letter, x, y in items:
        ax.text(x, y, letter, fontsize=11, fontweight="bold", ha="left", va="top")
    # Frames requested by spec["boxes"]. Drawn in the overlay rather than in
    # the panel itself because the graphical abstract is a cropped raster: it
    # has no margin of its own to put a border in, and a rule drawn here sits
    # exactly on the placed edges the assembly computed.
    for x, y, pw, ph in boxes:
        ax.add_patch(plt.Rectangle((x, y), pw, ph, fill=False,
                                   ec="black", lw=0.9, joinstyle="miter"))
    for text, x, y, size, ha in notes:
        ax.text(x, y, text, fontsize=size, ha=ha, va="top", color="0.25")
    for stem, x, y, pw, ph in missing:
        ax.add_patch(plt.Rectangle((x, y), pw, ph, fill=False, ls=(0, (5, 4)),
                                   ec="#999999", lw=1.0))
        ax.text(x + pw / 2, y + ph / 2 + 6, "panel not yet created",
                ha="center", va="center", fontsize=8, color="#777777")
        ax.text(x + pw / 2, y + ph / 2 - 7, f"figures/source/{stem}.pdf|.png",
                ha="center", va="center", fontsize=7, color="#999999", family="monospace")

    buf = io.BytesIO()
    # transparent=True is REQUIRED. matplotlib writes an opaque white page
    # background by default, and this overlay is merged ON TOP of the panels --
    # an opaque background paints over every panel and leaves a figure showing
    # nothing but the letters.
    fig.savefig(buf, format="pdf", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def _carry_patterns(src, scale, x, y):
    """Move a source page's pattern spaces along with its content.

    merge_transformed_page applies the scale and offset to the drawing but not
    to /Pattern resources, whose /Matrix maps pattern space onto the page's
    DEFAULT coordinates. A shading (e.g. a gradient colour bar) then lands where
    it sat on the source page, and the shape it fills shows only one end of the
    gradient. Composing each matrix with the same transform keeps it in place.
    Only the page's own patterns are handled; patterns inside form XObjects
    live in the XObject's space and already move with it.
    """
    from pypdf.generic import ArrayObject, FloatObject, NameObject
    res = src.get("/Resources")
    pats = res.get_object().get("/Pattern") if res is not None else None
    if pats is None:
        return
    for key in list(pats.get_object().keys()):
        pat = pats.get_object()[key].get_object()
        a, b, c, d, e, f = [float(v) for v in pat.get("/Matrix", [1, 0, 0, 1, 0, 0])]
        pat[NameObject("/Matrix")] = ArrayObject(FloatObject(v) for v in (
            a * scale, b * scale, c * scale, d * scale, e * scale + x, f * scale + y))


def build(name, spec, verbose=True):
    from pypdf import PdfReader, PdfWriter, PageObject, Transformation

    w, h = spec["size"]
    page = PageObject.create_blank_page(width=w, height=h)
    letters, missing, boxes = [], [], []
    framed = tuple(spec.get("boxes", ()))
    # The frame is INSET FROM NOTHING by default, which puts the rule straight
    # onto whatever sits at the edge of the panel -- for the graphical abstract
    # that is its own column headings. The pad holds the rule off the artwork,
    # per side: the artwork's top and bottom do not carry the same margin, so
    # one number cannot sit the rule evenly against both.
    pad = spec.get("box_pad", 8)
    pad = pad if isinstance(pad, dict) else dict.fromkeys("ltrb", float(pad))

    for stem, x, y, target_w, letter in spec["panels"]:
        path = find_panel(stem)
        if path is None:
            ph = target_w * 0.75
            missing.append((stem, x, y, target_w, ph))
            if letter:
                letters.append((letter, x, y + ph + 12))
            if verbose:
                print(f"    MISSING  {stem}")
            continue

        if path.suffix.lower() == ".pdf":
            src = PdfReader(str(path)).pages[0]
            sw = float(src.mediabox.width); sh = float(src.mediabox.height)
            scale = target_w / sw
            _carry_patterns(src, scale, x, y)
            page.merge_transformed_page(
                src, Transformation().scale(scale).translate(x, y))
            drawn_h = sh * scale
        elif path.suffix.lower() in RASTER:
            # Raster panels (ChimeraX renders) are wrapped in a one-page PDF at
            # their native pixel size, then scaled -- keeps full resolution
            # rather than resampling.
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            img = mpimg.imread(str(path))
            ih, iw = img.shape[0], img.shape[1]
            drawn_h = target_w * ih / iw
            f = plt.figure(figsize=(target_w / 72, drawn_h / 72))
            a = f.add_axes([0, 0, 1, 1]); a.imshow(img); a.axis("off")
            b = io.BytesIO()
            f.savefig(b, format="pdf", dpi=600, transparent=True)
            plt.close(f); b.seek(0)
            page.merge_transformed_page(
                PdfReader(b).pages[0], Transformation().translate(x, y))
        else:
            print(f"    SKIP  {path.name}: convert .svg to .pdf first "
                  f"(Inkscape: inkscape --export-type=pdf {path.name})")
            continue

        top = y + drawn_h
        if stem in framed:
            boxes.append((x - pad["l"], y - pad["b"],
                          target_w + pad["l"] + pad["r"],
                          drawn_h + pad["b"] + pad["t"]))
            top += pad["t"]                 # letter clears the rule, not the art
        if letter:
            letters.append((letter, x, top + 12))
        if verbose:
            print(f"    {path.parent.name}/{path.name}")

    ov = PdfReader(overlay((w, h), letters, missing,
                           spec.get("notes", ()), boxes)).pages[0]
    page.merge_page(ov)

    out = FIGDIR / f"{name}.pdf"
    wr = PdfWriter(); wr.add_page(page)
    with out.open("wb") as fh:
        wr.write(fh)
    return out, len(missing)


def list_panels():
    print(f"{'panel':34s} {'found in':14s} {'used by'}")
    print("-" * 66)
    used = {}
    for fig, spec in LAYOUTS.items():
        for stem, *_ in spec["panels"]:
            used.setdefault(stem, []).append(fig)
    for stem in sorted(used):
        p = find_panel(stem)
        where = p.parent.name if p else "-- MISSING --"
        print(f"{stem:34s} {where:14s} {', '.join(used[stem])}")
    extra = sorted({p.stem for p in PANELS.glob("*.pdf")} - set(used))
    if extra:
        print("\navailable but not placed in any figure:")
        for s in extra:
            print(f"  {s}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", help="figure names; default is all")
    ap.add_argument("--list", action="store_true", help="show panel inventory and exit")
    args = ap.parse_args(argv)

    if args.list:
        list_panels()
        return 0

    names = args.figures or list(LAYOUTS)
    unknown = [n for n in names if n not in LAYOUTS]
    if unknown:
        ap.error(f"unknown figure(s) {unknown}; known: {list(LAYOUTS)}")

    total_missing = 0
    for n in names:
        print(f"{n}:")
        out, miss = build(n, LAYOUTS[n])
        total_missing += miss
        print(f"  -> {out.relative_to(REPO)}"
              + (f"   ({miss} placeholder{'s' if miss != 1 else ''})" if miss else ""))
    if total_missing:
        print(f"\n{total_missing} panel(s) still to be made -- see README.md (figures/) "
              "and `python make_figures.py --list`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
