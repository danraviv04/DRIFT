"""Figure 1b redrawn from the per-residue RMSF, in three styles.

The current panel b (paper repo figures/source/wt.png, mutant.png and the two helix zooms)
overlays each 125-model ensemble, which shows that the fold is unchanged but not that P286R
is wider: the widening is a tenth of an Angstrom. Every style draws the per-residue C-alpha
RMSF of each 125-model ensemble (rmsf_bfactor.py, = Fig. 1d) in the core-fit frame, on one
scale for both sides, as a whole-domain render and a zoom per side.

  worm_bluered  Figure1_resKS_v2. Each side's medoid as a tube of radius RADIUS_PER_A x RMSF,
                steel blue / firebrick (the current colours), in the current panel's four
                files and aspect ratios: the default ChimeraX orientation, and a tall zoom on
                POLE 365-382 (the current zooms' window) turned so the 368-378 helix is vertical.
  worm_rmsf     Figure1_resKS_v3. The same worms, coloured by RMSF too.
  tubes_rmsf    Figure1_resKS_v4. All 125 models per side as thin tubes of one width,
                coloured by RMSF. At this spread the 125 traces coincide, so only the
                colour shows the widening.

worm_rmsf and tubes_rmsf are composed into ONE vector panel (fig1b_<style>.pdf): each side
labelled above, the render with a MAGNIFIER on BUBBLE_ZOOM -- a circle on the render joined
by two tangent lines to a round inset of that region in the same orientation, its first and
last residue numbered at the chain ends -- and the shared RMSF key and P286 marker between
the sides. They use the orientation that puts P286 towards the viewer.

BUBBLE_ZOOM = POLE 411-428 (helix, loop, short helix, loop, helix): of the 15-21-residue
windows at least 15 A from P286, the one P286R widens most (mean RMSF +0.082 A against
+0.056 A for the median residue and +0.054 A for the old 365-382 window; it holds the
416-420 peak of Fig. 1d). The site loop 278-292 widens slightly more (+0.099 A) but sits on
the mutation; the distal window shows the effect reaching beyond it.

Positions come from ChimeraX itself: chimerax_project.py reports where the region's C-alpha
atoms land in each saved view, so the circle and the residue numbers follow the renders
exactly. Each pair (renders, zooms) shares one camera and is cropped to one shared box, so
both sides keep one magnification.

Writes panel_b/<style>/panel_b.cxc (open it in ChimeraX to inspect or adjust), renders it
in the ChimeraX GUI (a window opens briefly), and writes the finished files to
panel_b/<style>/. --copy-to puts them in the paper repo's figures/source/.

    python experiments/fig1b_corefit_models/render_panel_b.py \
        [--style worm_rmsf ...] [--copy-to <paper repo>/figures/source]
"""
from pathlib import Path
import argparse
import os
import json
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export_models import CHAIN, read_atom_site, tokens  # noqa: E402

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'      # as the other Figure 1 panels

CHIMERAX = Path(os.environ.get('CHIMERAX') or shutil.which('chimerax') or '/Applications/ChimeraX-1.11.1.app/Contents/MacOS/ChimeraX')
PROJECT = HERE / 'chimerax_project.py'
OUT = HERE / 'panel_b'
MEDOID = {'wt': 'POLE_WT_medoid_bRMSF.cif', 'mutant': 'POLE_P286R_medoid_bRMSF.cif'}
RADIUS_PER_A = 6.0             # worm radius (A) per A of RMSF; no cap
TUBE_A = 0.5                   # tubes_rmsf width and thickness (A)
HELIX_ZOOM = (365, 382)        # worm_bluered: the current zooms' window (POLE numbering)
HELIX_AXIS = (368, 378)        # ... and the helix stood upright in it
BUBBLE_ZOOM = (411, 428)       # worm_rmsf, tubes_rmsf: the magnifier's region
FIRST_RESIDUE = 268
SITE = 286
P286_TURNS = (('y', -18), ('x', 52))   # P286 towards the viewer (fig1b_spread_*.cxc)
PALETTE = [(0.04, '#fff5eb'), (0.15, '#fd8d3c'), (0.30, '#7f2704')]
MARKER = '#00a0b0'
LABEL = {'wt': ('WT ensemble', '#5aae42'),          # the colours of the DRIFT GA redesign's
         'mutant': ('P286R ensemble', '#eac61e')}   # "WT Ensemble" / "MUT Ensemble"
LENS = '#3a3a3a'
ASPECT = {'render': 1897 / 1775, 'zoom': 341 / 851}   # the current files' width/height
MARGIN = 0.04
# The composed panel, pt: WT render, its bubble, the key, P286R's bubble, P286R render; a
# label band on top. It spans Figure 1's full 717 pt (with_row_b full_width), and the width
# beyond the old 600 pt row-b slot goes to the bubbles (96 -> 140 pt, 2026-10-24).
BLOCK_W = 717.0
RENDER_W = 160.0
BUBBLE_D = 140.0
BUBBLE_GAP = 10.0
LABEL_BAND = 14.0
# The stacked layout (Figure1_resKS_v*_wide, beside panel a): two rows, the key column at right,
# drawn near its placed size (~225 pt) so its text keeps the row layout's point sizes.
STACK_RENDER_W = 105.0
STACK_BUBBLE_D = 72.0
STACK_GAP = 6.0
KEY_COL = 48.0

STYLES = {
    'worm_bluered': dict(ensemble=False, worm=True, colour='side', turns=(), bubble=False),
    'worm_rmsf': dict(ensemble=False, worm=True, colour='rmsf', turns=P286_TURNS, bubble=True),
    'tubes_rmsf': dict(ensemble=True, worm=False, colour='rmsf', turns=P286_TURNS, bubble=True),
}


def ca_coords(cif):
    lines = Path(cif).read_text().splitlines()
    cols, rows = read_atom_site(lines)
    out = {}
    for i, sp in rows:
        ch, atom, seq = tokens(lines[i], sp, cols, ('label_asym_id', 'label_atom_id', 'label_seq_id'))
        if ch == CHAIN and atom == 'CA':
            out[int(seq) + FIRST_RESIDUE - 1] = np.array(
                [float(t) for t in tokens(lines[i], sp, cols, ('Cartn_x', 'Cartn_y', 'Cartn_z'))])
    return out


def rot(axis, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return {'x': np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            'y': np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])}[axis]


def camera(axes, centre):
    """ChimeraX 'view matrix camera' (3x4 row-major): columns = camera x, y, z, origin.
    The camera looks down its -z; `view <spec>` afterwards re-centres and zooms."""
    m = np.column_stack([axes, centre + 100 * axes[:, 2]])
    return ','.join(f'{v:.6f}' for v in m.ravel())


def cameras(style, ca):
    """Render and zoom cameras. `turns` are ChimeraX `turn` steps from the default view
    (screen axes, right-handed), i.e. a model rotation M; the camera then has axes M^T.
    A bubble keeps the render's orientation (a magnifier); the tall helix zoom stands the
    helix upright, looking from as close to the render's direction as that allows."""
    M = np.eye(3)
    for ax, deg in STYLES[style]['turns']:
        M = rot(ax, deg) @ M
    C = M.T
    render = camera(C, np.array(list(ca.values())).mean(0))
    lo, hi = BUBBLE_ZOOM if STYLES[style]['bubble'] else HELIX_ZOOM
    win = np.array([ca[r] for r in range(lo, hi + 1)])
    if STYLES[style]['bubble']:
        return render, camera(C, win.mean(0))
    h = np.array([ca[r] for r in range(HELIX_AXIS[0], HELIX_AXIS[1] + 1)])
    up = np.linalg.svd(h - h.mean(0))[2][0]
    if up @ (h[-1] - h[0]) < 0:
        up = -up                                       # N to C points up
    z = C[:, 2] - (C[:, 2] @ up) * up
    if np.linalg.norm(z) < 0.3:
        raise ValueError('helix axis is nearly along the view direction; choose other turns')
    z /= np.linalg.norm(z)
    return render, camera(np.column_stack([np.cross(up, z), up, z]), h.mean(0))


def write_cxc(path, style, render_cam, zoom_cam):
    s = STYLES[style]
    lo, hi = BUBBLE_ZOOM if s['bubble'] else HELIX_ZOOM
    if s['ensemble']:
        side = {'wt': '#1-125', 'mutant': '#126-250'}
        mark = {'wt': '#1', 'mutant': '#126'}
        opens = ['open ../../models/WT/*.cif', 'open ../../models/P286R/*.cif']
    else:
        side = {'wt': '#1', 'mutant': '#2'}
        mark = side
        opens = [f'open ../../models/medoid/{MEDOID[k]}' for k in ('wt', 'mutant')]
    both = f"{side['wt']},{side['mutant'][1:]}"
    marks = f"{mark['wt']},{mark['mutant'][1:]}"
    pal = ':'.join(f'{v:g},{c}' for v, c in PALETTE)
    project = f'runscript {PROJECT}'
    region = {k: f'{mark[k]}/A:{lo}-{hi}@CA' for k in mark}
    L = [f'# Figure 1b, style {style} (render_panel_b.py). {side["wt"]} = WT, {side["mutant"]} = P286R,',
         '# core-fit frame. Saves raw/*.png (and, for the bubble, raw/*_points.json) next to this file.',
         'windowsize 1000 936', *opens, f'renumber {both}/A start {FIRST_RESIDUE}']
    if s['ensemble']:
        L += [f'open ../../results/POLE_WT_rmsf.defattr models {side["wt"]}',
              f'open ../../results/POLE_P286R_rmsf.defattr models {side["mutant"]}']
    L += [f'hide {both} atoms', f'hide {both} cartoons', f'show {both}/A cartoons']
    if s['worm']:
        L += [f'cartoon byattribute bfactor {both}/A 0.0125:{0.0125 * RADIUS_PER_A:g} 0.5:{0.5 * RADIUS_PER_A:g}']
    else:
        L += [f'cartoon style {both} modeHelix default arrows false xsection oval width {TUBE_A:g} thickness {TUBE_A:g}']
    if s['colour'] == 'side':
        L += [f'color {side["wt"]}/A steelblue target c', f'color {side["mutant"]}/A firebrick target c']
    elif s['ensemble']:
        L += [f'color byattribute r:rmsf {both}/A palette {pal} target c']
    else:
        L += [f'color bfactor {both}/A palette {pal} target c']
    if s['bubble']:   # the P286 C-alpha marker (the cartoon hides backbone atoms otherwise)
        L += [f'show {marks}/A:{SITE}@CA atoms', f'style {marks}/A:{SITE}@CA sphere',
              f'size {marks}/A:{SITE}@CA atomRadius 1.8', f'color {marks}/A:{SITE}@CA {MARKER} target a',
              f'cartoon {marks}/A:{SITE} suppressBackboneDisplay false']
    L += ['lighting soft', 'graphics silhouettes true width 1.5', 'set bgColor white', '',
          '# whole domain: one camera for both',
          f'view matrix camera {render_cam}', f'view {both}']
    if s['bubble']:
        L += [f'{project} raw/render_points.json wt={region["wt"]} mutant={region["mutant"]}']
    L += [f'hide {side["mutant"]} models', 'save raw/wt.png width 2400 supersample 3',
          f'show {side["mutant"]} models', f'hide {side["wt"]} models', 'save raw/mutant.png width 2400 supersample 3',
          f'show {side["wt"]} models', '',
          f'# zoom, POLE {lo}-{hi}: one camera for both',
          f'hide {both} atoms', f'hide {both}/A cartoons', f'show {both}/A:{lo}-{hi} cartoons',
          f'view matrix camera {zoom_cam}', f'view {both}/A:{lo}-{hi}']
    if s['bubble']:
        ends = ' '.join(f'{k}_{e}={mark[k]}/A:{r}@CA' for k in mark for e, r in (('first', lo), ('last', hi)))
        L += [f'{project} raw/zoom_points.json wt={region["wt"]} mutant={region["mutant"]} {ends}']
    L += [f'hide {side["mutant"]} models', 'save raw/wt_zoom.png width 2400 supersample 3',
          f'show {side["mutant"]} models', f'hide {side["wt"]} models', 'save raw/mutant_zoom.png width 2400 supersample 3',
          f'show {side["wt"]} models']
    path.write_text('\n'.join(L) + '\n')


def shared_crop(paths, aspect, circle=False):
    """Crop every image to one box: the union of their non-white content, plus a margin,
    widened to `aspect` (width/height) and padded with white where it leaves the image.
    circle=True sizes a square so its inscribed circle holds all the content.
    Returns the crops and the box's top-left corner in the original pixels."""
    ims = [Image.open(p).convert('RGB') for p in paths]
    boxes = []
    for im in ims:
        ys, xs = np.nonzero((np.asarray(im) < 245).any(2))
        boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    if circle:
        w = h = np.hypot(x1 - x0, y1 - y0) * (1 + MARGIN)
    else:
        w, h = (x1 - x0) * (1 + 2 * MARGIN), (y1 - y0) * (1 + 2 * MARGIN)
        w, h = max(w, h * aspect), max(h, w / aspect)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (round(cx - w / 2), round(cy - h / 2), round(cx - w / 2) + round(w), round(cy - h / 2) + round(h))
    out = []
    for im in ims:
        canvas = Image.new('RGB', (box[2] - box[0], box[3] - box[1]), 'white')
        canvas.paste(im, (-box[0], -box[1]))
        out.append(canvas)
    return out, box[:2]


def to_image(points_json, raw_png, origin):
    """Window positions from chimerax_project.py -> pixels of the cropped image."""
    p = json.loads(Path(points_json).read_text())
    (ww, wh), (iw, ih) = p.pop('window'), Image.open(raw_png).size
    return {k: np.array(v) * [iw / ww, ih / wh] - origin for k, v in p.items()}


def tangents(c1, r1, c2, r2):
    """The two outer common tangents of two circles, as ((p1, p2), (q1, q2))."""
    d = c2 - c1
    dist = np.linalg.norm(d)
    u = d / dist
    cos = (r1 - r2) / dist
    sin = np.sqrt(max(0.0, 1 - cos ** 2))
    out = []
    for sgn in (1, -1):
        n = np.array([u[0] * cos - sgn * u[1] * sin, sgn * u[0] * sin + u[1] * cos])
        out.append((c1 + r1 * n, c2 + r2 * n))
    return out


def compose(style, img, pts, path, layout='row', bubble_d=None, font_scale=1.0,
            key_below=False):
    """One vector panel: per side a label, the render with a circle on the zoom region, two
    tangent lines to a round inset of that region with its end residues numbered, and the
    shared RMSF key and P286 marker.

    layout 'row': WT | bubble | key | bubble | P286R, one row (fills Figure 1's row b).
    layout 'stacked': the WT row over the P286R row, each render | bubble, the key in a
    column at the right -- a near-square block to sit beside panel a."""
    lo, hi = BUBBLE_ZOOM
    stacked = layout == 'stacked'
    rw = STACK_RENDER_W if stacked else RENDER_W
    d = bubble_d or (STACK_BUBBLE_D if stacked else BUBBLE_D)
    fs = font_scale
    band = LABEL_BAND * fs
    rh, r2 = rw / ASPECT['render'], d / 2

    def pt(xy_px, im, x0, y0, w, h):           # image pixels -> panel pt
        xy = np.atleast_2d(xy_px)
        return np.column_stack([x0 + xy[:, 0] * w / im.width, y0 + h - xy[:, 1] * h / im.height])

    # The circle on each render, relative to the render's bottom-left corner, and the room it
    # needs below the render. Each bubble sits level with its circle (so the connectors cross
    # as little of the render as possible), inside its row and under the region label.
    rel = {}
    for s in ('wt', 'mutant'):
        src = pt(pts['render'][s], img[s], 0, 0, rw, rh)
        c = src.mean(0)
        rel[s] = (c, np.linalg.norm(src - c, axis=1).max() + 4)
    pad = {s: max(0.0, 4 - (c[1] - r)) for s, (c, r) in rel.items()}
    if stacked:
        # key_below (2026-10-28): the key as a horizontal bar under the P286R row, which
        # gives its column's width back to the renders when b sits beside panel a.
        kb = 24 * fs if key_below else 0.0
        W = rw + BUBBLE_GAP + d + (0 if key_below else KEY_COL * fs)
        y0 = {'mutant': pad['mutant'] + kb}
        y0['wt'] = y0['mutant'] + rh + band + STACK_GAP + pad['wt']
        H = y0['wt'] + rh + band
        x0 = {'wt': 0.0, 'mutant': 0.0}
        bxc = {s: rw + BUBBLE_GAP + r2 for s in rel}
        key_c = np.array([W - KEY_COL * fs / 2, H / 2])
        top = rh + 2                  # the bubble may rise into the label band (other x)
    else:
        W = BLOCK_W
        pad = dict.fromkeys(pad, max(pad.values()))
        y0 = dict(pad)
        H = y0['wt'] + rh + band
        x0 = {'wt': 0.0, 'mutant': W - rw}
        bxc = {'wt': rw + BUBBLE_GAP + r2, 'mutant': W - rw - BUBBLE_GAP - r2}
        key_c = np.array([(bxc['wt'] + bxc['mutant']) / 2, y0['wt'] + rh / 2])
        top = rh - 2                  # its region label then sits in the label band (other x)
    c1 = {s: rel[s][0] + [x0[s], y0[s]] for s in rel}
    r1 = {s: rel[s][1] for s in rel}
    c2 = {s: np.array([bxc[s], y0[s] + np.clip(rel[s][0][1], r2 + 2 - pad[s], top - r2)]) for s in rel}
    for s in rel:
        if c1[s][0] - r1[s] < 0 or c1[s][0] + r1[s] > W:
            print(f'  warning: the {s} circle runs off the panel sideways')

    fig = plt.figure(figsize=(W / 72, H / 72))
    ov = fig.add_axes([0, 0, 1, 1], facecolor='none', zorder=5)
    ov.set_xlim(0, W)
    ov.set_ylim(0, H)
    ov.axis('off')

    for side in ('wt', 'mutant'):
        im = img[side]
        ax = fig.add_axes([x0[side] / W, y0[side] / H, rw / W, rh / H])
        ax.imshow(np.asarray(im), interpolation='none', extent=(0, 1, 0, 1), aspect='auto')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        zim = img[side + '_zoom']
        a, b = c1[side], c2[side]
        zax = fig.add_axes([(b[0] - r2) / W, (b[1] - r2) / H, d / W, d / H])
        art = zax.imshow(np.asarray(zim), interpolation='none', extent=(0, 1, 0, 1))
        art.set_clip_path(Circle((0.5, 0.5), 0.5, transform=zax.transData))
        zax.set_xlim(0, 1)
        zax.set_ylim(0, 1)
        zax.axis('off')

        for (p1, p2) in tangents(a, r1[side], b, r2):
            ov.plot([p1[0], p2[0]], [p1[1], p2[1]], color=LENS, lw=0.6, solid_capstyle='round')
        ov.add_patch(Circle(a, r1[side], fill=False, ec=LENS, lw=0.8))
        ov.add_patch(Circle(b, r2, fill=False, ec=LENS, lw=0.9))
        for end, num in (('first', lo), ('last', hi)):
            p = pt(pts['zoom'][f'{side}_{end}'], zim, b[0] - r2, b[1] - r2, d, d)[0]
            out = (p - b) / max(np.linalg.norm(p - b), 1e-6)
            ov.text(*(p + 7 * out), str(num), fontsize=7.5 * fs, ha='center', va='center',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        ov.text(b[0], b[1] + r2 + 3, f'residues {lo}–{hi}', fontsize=7.5 * fs, ha='center', va='bottom',
                color=LENS)
        text, colour = LABEL[side]
        ov.text(x0[side] + rw / 2, y0[side] + rh + band / 2, text, ha='center', va='center',
                fontsize=10 * fs, color=colour)

    if stacked and key_below:
        vmin, vmax = PALETTE[0][0], PALETTE[-1][0]
        cmap = LinearSegmentedColormap.from_list(
            'rmsf', [((v - vmin) / (vmax - vmin), c) for v, c in PALETTE])
        bw, bh, by = 0.36 * W, 5.0 * fs, 11 * fs
        bx = (W - bw) / 2
        cax = fig.add_axes([bx / W, by / H, bw / W, bh / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(vmin, vmax), cmap), cax=cax,
                          orientation='horizontal', ticks=[v for v, _ in PALETTE])
        cb.ax.tick_params(labelsize=7.5 * fs, length=2, width=0.5, pad=1.5)
        cb.outline.set_linewidth(0.5)
        cb.ax.set_xticklabels([f'{v:.2f}' for v, _ in PALETTE])
        ov.text(bx - 6, by + bh / 2, 'Cα RMSF (Å)', ha='right', va='center', fontsize=8 * fs)
        ov.scatter([bx + bw + 12], [by + bh / 2], s=22 * fs, color=MARKER, edgecolor='black',
                   linewidth=0.4, zorder=3)
        ov.text(bx + bw + 18, by + bh / 2, 'P286', fontsize=8 * fs, va='center', ha='left')
        fig.savefig(path, dpi=600)
        plt.close(fig)
        return

    cx = key_c[0]
    kw, kh = 6.0, 58.0
    ky = key_c[1] - kh / 2 + 6
    vmin, vmax = PALETTE[0][0], PALETTE[-1][0]
    cmap = LinearSegmentedColormap.from_list(
        'rmsf', [((v - vmin) / (vmax - vmin), c) for v, c in PALETTE])
    cax = fig.add_axes([(cx - 12) / W, ky / H, kw / W, kh / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(vmin, vmax), cmap), cax=cax,
                      orientation='vertical', ticks=[v for v, _ in PALETTE])
    cb.ax.tick_params(labelsize=7.5 * fs, length=2, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.ax.set_yticklabels([f'{v:.2f}' for v, _ in PALETTE])
    ov.text(cx, ky + kh + 5, 'Cα RMSF\n(Å)' if stacked else 'Cα RMSF (Å)', ha='center', va='bottom',
            fontsize=8 * fs, linespacing=1.2)
    if STYLES[style]['worm']:
        ov.text(cx, ky - 6, 'colour and\ntube width', ha='center', va='top', fontsize=7 * fs, linespacing=1.2)
    my = ky - (30 if STYLES[style]['worm'] else 12)
    ov.scatter([cx - 14], [my], s=22, color=MARKER, edgecolor='black', linewidth=0.4, zorder=3)
    ov.text(cx - 8, my, 'P286', fontsize=8 * fs, va='center', ha='left')
    fig.savefig(path, dpi=600)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--style', nargs='+', choices=list(STYLES), default=list(STYLES))
    ap.add_argument('--copy-to', type=Path, help="paper repo figures/source/")
    ap.add_argument('--compose-only', action='store_true',
                    help='reuse panel_b/<style>/raw/ instead of rendering again')
    # 2026-10-28: the wide Figure 1 draft wants larger magnified circles and text in the
    # stacked panel; written only to --stacked-to, the panels in use are not touched.
    ap.add_argument('--stack-bubble-d', type=float, help='stacked bubble diameter, pt')
    ap.add_argument('--font-scale', type=float, default=1.0, help='stacked text size factor')
    ap.add_argument('--stacked-to', type=Path, help='write ONLY the stacked panel, here')
    ap.add_argument('--key-below', action='store_true',
                    help='stacked: horizontal key under the panel instead of a key column')
    args = ap.parse_args()

    ca = ca_coords(HERE / 'models/medoid' / MEDOID['wt'])
    for style in args.style:
        bubble = STYLES[style]['bubble']
        out = OUT / style
        if not args.compose_only:
            (out / 'raw').mkdir(parents=True, exist_ok=True)
            for old in (out / 'raw').glob('*'):
                old.unlink()
            cxc = out / 'panel_b.cxc'
            write_cxc(cxc, style, *cameras(style, ca))
            with open(out / 'chimerax.log', 'w') as log:
                subprocess.run([str(CHIMERAX), '--cmd', f'open {cxc}; exit'], check=True,
                               stdout=log, stderr=subprocess.STDOUT)

        img, pts = {}, {}
        for kind, keys in {'render': ('wt', 'mutant'), 'zoom': ('wt_zoom', 'mutant_zoom')}.items():
            raw = [out / 'raw' / f'{k}.png' for k in keys]
            missing = [p for p in raw if not p.exists()]
            if missing:
                raise SystemExit(f'ChimeraX did not write {missing}')
            crops, origin = shared_crop(raw, 1.0 if bubble and kind == 'zoom' else ASPECT[kind],
                                        circle=bubble and kind == 'zoom')
            img.update(zip(keys, crops))
            if bubble:
                pts[kind] = to_image(out / 'raw' / f'{kind}_points.json', raw[0], origin)

        if bubble and args.stacked_to:
            compose(style, img, pts, args.stacked_to, layout='stacked',
                    bubble_d=args.stack_bubble_d, font_scale=args.font_scale,
                    key_below=args.key_below)
            print(style, '->', args.stacked_to)
            continue
        if bubble:
            files = [out / f'fig1b_{style}.pdf', out / f'fig1b_{style}_stacked.pdf']
            compose(style, img, pts, files[0], layout='row')
            compose(style, img, pts, files[1], layout='stacked')
        else:
            files = []
            for key, im in img.items():
                side, _, zoom = key.partition('_')
                files.append(out / f'{side}_rmsf_worm{"_helix_zoom" if zoom else ""}.png')
                im.save(files[-1], dpi=(600, 600))
        print(style, '->', ', '.join(f.name for f in files))
        if args.copy_to:
            for f in files:
                shutil.copy2(f, args.copy_to / f.name)
            print(f'  copied to {args.copy_to}')


if __name__ == '__main__':
    main()
