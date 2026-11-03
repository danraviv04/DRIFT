# fig1b_corefit_models: full-atom POLE WT and P286R ensembles for Fig. 1b (2026-10-22)

Fig. 1b superposes the WT and p.P286R POLE ensembles, 125 predictions each. This folder
gives ChimeraX the full-atom AF3 models already placed in the core-fit frame used by the
coordinate metrics, with AF3's per-atom pLDDT still in the B-factor column.

- `export_models.py` reads each AF3 CIF from `$DRIFT_ARCHIVE_ROOT/POLE_folders` and recovers
  its core-fit superposition, then rewrites only the `atom_site` coordinates. The
  transform is the Kabsch fit of the structure's own Cα atoms onto its cached core-fit Cα
  positions (`subunit_core_fit.py`; for POLE the rigid core is all 204 residues). It is
  applied to every atom: protein, DNA and Mg.
- `models/WT/` and `models/P286R/` hold 125 CIFs each (~170 MB, gitignored). The
  ensembles are the columns of the pLDDT CSVs the KS-score reads: the pinned
  `POLE_268-471_WT` CSV and the 25-seed P286R copy, seeds 1-25 on both sides.
- `fig1b_ensembles.cxc` opens them in ChimeraX. WT is models #1-125 (blue) and P286R is
  #126-250 (red). Chain A is renumbered to POLE numbering (268-471), so `:286` is the site.
  Commented lines switch to pLDDT colouring and zoom on P286. Run it from ChimeraX with
  `open <path>/fig1b_ensembles.cxc`. Relative paths resolve from the script's folder.
- `results/manifest.csv` has one row per model: its source path, fit residual, core-fit
  RMSD, and the pLDDT check. `results/summary.json` holds the per-side maxima.

The files on disk are not renumbered: residue 19 is p.286 (index 0 is residue 268).

## Spread views: RMSF in the B-factor (`rmsf_bfactor.py`)

An overlay of 250 traces shows that the fold is unchanged, but it hardly shows the
widening, which is about a tenth of an Ångström. `rmsf_bfactor.py` computes the
per-residue Cα RMSF of each 125-model ensemble (Methods eq. 4) in this frame. It writes
the RMSF into the B-factor of each side's medoid, the model closest to its own ensemble
centre: WT seed-15_sample-1 and P286R seed-21_sample-3. Outputs go to `models/medoid/`
and `results/per_residue_rmsf.csv`.

- It matches Fig. 1d, which uses the US-align cache: the largest difference is 0.0001 Å
  and r = 0.99999998. The WT median is 0.082 Å and the P286R median is 0.137 Å. 202 of
  204 residues are wider in P286R, with a median ratio of 1.72.
- `fig1b_spread_worm.cxc` puts WT and P286R side by side as worms. Tube radius and
  colour both show the residue's RMSF, on the same scale for both sides. The radius is
  8 × RMSF, capped at 0.30 Å, so **the caption must say the radius is exaggerated**.
- `fig1b_spread_ensemble.cxc` shows all 125 models per side as thin tubes of one width,
  in the same colours as the worm view (`results/POLE_<side>_rmsf.defattr`, a residue
  attribute `rmsf`). The spread is far thinner than a tube, so the 125 traces look like
  one, and only the colour shows the widening.
- `fig1b_spread_ratio.cxc` shows one structure coloured by log2(RMSF P286R / RMSF WT).
  It shows where the widening is, not how large it is.
- Both scripts mark P286 with a Cα sphere and turn the view so the site faces the
  camera. Previews are in `results/previews/`; they were rendered in the ChimeraX GUI.

## Figure 1b v2 (`render_panel_b.py`)

This is the worm idea fitted to the paper's panel b. Each side's medoid is drawn as a
tube of radius 6 × its RMSF, with no cap. A 0.30 Å cap would have hidden the largest
P286R peaks, at 283–285 and 416–420. The script writes four images in the current
panel's colours (steelblue/firebrick) and at its two aspect ratios: two whole-domain
renders sharing one camera, and two zooms on POLE 365–382 with the helix vertical. It
writes `panel_b/panel_b.cxc`, renders it in the ChimeraX GUI, crops each pair to one
shared box, and with `--copy-to` puts the files in the paper repo's `figures/source/`.
There, `LAYOUTS["Figure1_resKS_v2"]` (tools/make_figures.py) swaps them into
Figure1_resKS's four slots. The page size and every other panel are unchanged, and
Figure1_resKS rebuilds byte-identical.

The script has three `--style`s, and each writes to `panel_b/<style>/`:
- `worm_bluered` makes v2.
- `worm_rmsf` makes v3: the same worms, coloured by RMSF as well.
- `tubes_rmsf` makes v4: all 125 models per side as thin tubes of one width, coloured
  by RMSF.

The last two are composed into one vector panel, `fig1b_<style>.pdf`. It has a label
above each pair ("WT ensemble" and "P286R ensemble", in the DRIFT GA redesign's
green and yellow), the shared colour key between the pairs, and a P286 marker. The
orientation puts P286 towards the viewer. Text is 7.5-10 pt at working size, which
is ≥ 5 pt after scaling to 183 mm. `with_row_b` in make_figures.py puts that panel in
row b's span and lifts panel a by the 14 pt label band. v3 and v4 also use the
redesign as panel a.

**Magnifier (v3/v4).** Each render has a circle on residues 411–428, joined by two
tangent lines to a round inset of that region. The inset keeps the render's
orientation, numbers the region's first and last residue at the chain ends, and is
headed "residues 411–428". Positions come from ChimeraX itself:
`chimerax_project.py`, run inside the .cxc, reports where those Cα atoms land in each
saved view. The overlay check put every projected Cα on the chain. 411–428 (helix,
loop, short helix, loop, helix) is the 15–21-residue window at least 15 Å from P286
that P286R widens most: mean RMSF +0.082 Å, against +0.054 Å for the old 365–382
helix and +0.056 Å for the median residue. It holds the 416–420 peak of Fig. 1c,d.
The site loop 278–292 widens slightly more (+0.099 Å) but sits on the mutation.
`BUBBLE_ZOOM` in render_panel_b.py changes the region. Since 10-24 the row panel spans the figure's full 717 pt and the bubbles are 140 pt across (were 96). The bubbles sit level with
their circles, and the panel adds the bottom margin the circles need.

**Wide layouts (10-23).** At Figure 1's ~0.66 page height, v4 (w/h 0.58) printed about
4 in wide, with every panel at ~0.37 of its size. Each bubble style is therefore also
composed as `fig1b_<style>_stacked.pdf`: the WT row over the P286R row, with the key
in a right-hand column. It is drawn near its placed size, so its text keeps the row
version's point sizes. `--compose-only` rebuilds both layouts from `raw/` without
ChimeraX. The paper's `Figure1_resKS_v{3,4}_wide` puts this panel beside a and uses
the flatter panel e (`experiments/figure1_wide/flat_panel_e.py 0.32 --gene MSH6
--driver A1055P --passenger I1054F`). That gives w/h 0.83, about 6 in wide at 0.72.
`v4_wide_noCD` also drops c and d and reletters the rest: w/h 1.00, about 7.2 in.
The manuscript kept v4, so the paper repo archives these layouts' PDFs and panels in
`figures/archive/fig1_wide_2026-10-23/` (still buildable; paper commit 369b9b1).

ChimeraX does not always get the requested window size (the first v2 render came out
1000 × 902, not 936). That changes the pixel scale, not the view, and the shared crop
and slot scaling remove it.

## Checks (all passed, 250/250)

| check | WT | P286R |
|---|---:|---:|
| fit residual onto the cached Cα, max | 3e-6 Å | 5e-6 Å |
| written Cα vs cache, max | 0.0005 Å | 0.0005 Å |
| Cα B-factor vs pLDDT CSV, max | 0 | 0 |
| core-fit RMSD to the WT mean core, median | 0.10 Å | 0.25 Å |
| residue 19 | PRO | ARG |

Also checked: the drive path's `seed-N_sample-M` matches the cache sample, and every
transform is a proper rotation. Nothing in a CIF other than the coordinates differs from
its AF3 original. Without any further fit, the two ensemble-mean Cα structures are
0.212 Å apart, which matches the Fig. 1c caption's 0.21 Å. ChimeraX 1.11.1 loads all 250
models from any working directory. The image itself was not rendered here.
