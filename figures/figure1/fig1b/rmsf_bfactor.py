"""Per-residue RMSF in the B-factor, for a Fig. 1b that shows the spread, not just the overlay.

Fig. 1 says a substitution leaves the mean structure in place (the ensemble centres are
0.21 A apart) and widens the ensemble (Fig. 1d). An overlay of 250 thin traces shows the
first half well and the second half poorly, because the widening is a tenth of an Angstrom.
This writes the per-residue spread into the B-factor of one representative model per
side, so ChimeraX can draw it as tube radius and colour (a "worm"/putty view).

- RMSF per residue = sqrt(mean_i |p_i - mean(p)|^2) over the 125 C-alpha positions
  (Methods eq. 4), in the core-fit frame of export_models.py, on the same 125 structures.
- Representative = the medoid: the model whose C-alpha atoms are closest (RMSD, no fit) to
  its own ensemble centre.
- Fig. 1d is computed by rmsf_histogram/exemplar_wt_vs_mut.py from the older US-align cache;
  its RMSF is recomputed here on the same structures and compared, so the colour scale can
  be read against panel d.

Outputs (models/medoid/, gitignored; results/per_residue_rmsf.csv):
  POLE_WT_medoid_bRMSF.cif          B = WT RMSF (A), chain A; other chains B = 0
  POLE_P286R_medoid_bRMSF.cif       B = P286R RMSF (A)
  POLE_P286R_medoid_bLog2Ratio.cif  B = log2(RMSF P286R / RMSF WT)

Run from the repo root, after export_models.py:
    python figures/figure1/fig1b/rmsf_bfactor.py
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'analyses/coordinates/rmsf_histogram'))
from export_models import CHAIN, CORE_FIT, L, SIDES, read_atom_site, tokens  # noqa: E402
import rmsf_histogram as fig1d  # noqa: E402

XYZ = ['x_aligned', 'y_aligned', 'z_aligned']
FIRST_RESIDUE = 268


def ensemble(cache, art, samples):
    """(n, L, 3) C-alpha array ordered by sample, then residue."""
    d = cache[(cache.artifact == art) & cache['sample'].isin(samples)].sort_values(['sample', 'resid'])
    X = d[XYZ].to_numpy().reshape(-1, L, 3)
    order = sorted(samples)
    if len(X) != len(samples) or d['sample'].unique().tolist() != order:
        raise ValueError(f'{art}: cache does not hold every manifest sample')
    return X, order


def rmsf(X):
    return np.sqrt(((X - X.mean(0)) ** 2).sum(2).mean(0))


def rmsf_fig1d(art):
    """RMSF and ensemble centre as Fig. 1c,d compute them: the US-align cache (not the
    core-fit frame), seeds 1-25, AF3's top-model copy dropped."""
    d = fig1d.read_cache([art], quiet=True).sort_values(['structure_index', 'resid'])
    X = d[XYZ].to_numpy().reshape(-1, L, 3)
    return rmsf(X), len(X), X.mean(0)


def write_bfactor(src, out, values):
    """Copy src with every chain-A atom's B-factor set to its residue's value, others 0."""
    lines = src.read_text().splitlines(keepends=True)
    cols, rows = read_atom_site(lines)
    for i, sp in rows:
        ch, seq = tokens(lines[i], sp, cols, ('label_asym_id', 'label_seq_id'))
        v = values[int(seq) - 1] if ch == CHAIN else 0.0
        a, b = sp[cols.index('B_iso_or_equiv')]
        lines[i] = lines[i][:a] + f'{v:.3f}'.rjust(b - a) + lines[i][b:]
    out.write_text(''.join(lines))


def main():
    manifest = pd.read_csv(HERE / 'results/manifest.csv')
    arts = [a for a, _, _ in SIDES.values()]
    cache = pq.read_table(CORE_FIT, filters=[('artifact', 'in', arts)]).to_pandas()
    cache['artifact'] = cache['artifact'].astype(str)
    cache['sample'] = cache['sample'].astype(str)

    per, medoid, check = {}, {}, {}
    for side, (art, _, _) in SIDES.items():
        m = manifest[manifest.side == side]
        X, order = ensemble(cache, art, m['sample'].tolist())
        per[side] = rmsf(X)
        dev = np.sqrt(((X - X.mean(0)) ** 2).sum(2).mean(1))
        k = int(dev.argmin())
        medoid[side] = dict(sample=order[k], rmsd_to_centre_A=float(dev[k]),
                            output=m.set_index('sample').output[order[k]])
        ref, n, per[side + '_centre_fig1c'] = rmsf_fig1d(art)
        check[side] = dict(fig1d_structures=n, max_abs_diff_A=float(np.abs(ref - per[side]).max()),
                           pearson_r=float(np.corrcoef(ref, per[side])[0, 1]),
                           median_core_fit_A=float(np.median(per[side])), median_fig1d_A=float(np.median(ref)))
        per[side + '_fig1d'] = ref
        per[side + '_centre'] = X.mean(0)

    ratio = per['P286R'] / per['WT']
    table = pd.DataFrame({
        'resid_af3': np.arange(1, L + 1), 'resid_pole': np.arange(FIRST_RESIDUE, FIRST_RESIDUE + L),
        'rmsf_wt_A': per['WT'], 'rmsf_p286r_A': per['P286R'], 'ratio': ratio, 'log2_ratio': np.log2(ratio),
        'centre_shift_A': np.linalg.norm(per['P286R_centre'] - per['WT_centre'], axis=1),
        'rmsf_wt_fig1d_A': per['WT_fig1d'], 'rmsf_p286r_fig1d_A': per['P286R_fig1d'],
        'centre_shift_fig1c_A': np.linalg.norm(per['P286R_centre_fig1c'] - per['WT_centre_fig1c'], axis=1)})
    table.to_csv(HERE / 'results/per_residue_rmsf.csv', index=False, float_format='%.5f')
    # Fig. 1c's numbers in both frames (exemplar_wt_vs_mut.py: RMSD = root-mean-square shift)
    frames = {name: dict(rmsd_between_means_A=float(np.sqrt((s ** 2).mean())),
                         share_below_0p5_A=float((s < 0.5).mean()))
              for name, s in (('core_fit', table.centre_shift_A), ('us_align_fig1c', table.centre_shift_fig1c_A))}
    frames['centre_shift_max_abs_diff_A'] = float((table.centre_shift_A - table.centre_shift_fig1c_A).abs().max())

    out = HERE / 'models/medoid'
    out.mkdir(parents=True, exist_ok=True)
    jobs = [('WT', 'bRMSF', table.rmsf_wt_A), ('P286R', 'bRMSF', table.rmsf_p286r_A),
            ('P286R', 'bLog2Ratio', table.log2_ratio)]
    for side, tag, values in jobs:
        write_bfactor(HERE / medoid[side]['output'], out / f'POLE_{side}_medoid_{tag}.cif', values.to_numpy())

    # The same per-residue RMSF as a ChimeraX residue attribute "rmsf", for colouring all 125
    # models of a side at once (fig1b_spread_ensemble.cxc). POLE numbering: open after renumber.
    for side, col in (('WT', 'rmsf_wt_A'), ('P286R', 'rmsf_p286r_A')):
        body = ''.join(f'\t/A:{r}\t{v:.5f}\n' for r, v in zip(table.resid_pole, table[col]))
        (HERE / f'results/POLE_{side}_rmsf.defattr').write_text(
            'attribute: rmsf\nmatch mode: any\nrecipient: residues\n' + body)

    summary = dict(
        medoid=medoid, fig1d_check=check, fig1c_frames=frames,
        rmsf_A={s: dict(zip(['min', 'p25', 'median', 'p75', 'p95', 'max'],
                            np.percentile(per[s], [0, 25, 50, 75, 95, 100]).round(4).tolist()))
                for s in SIDES},
        ratio=dict(median=float(np.median(ratio)), residues_wider=int((ratio > 1).sum()),
                   min=float(ratio.min()), max=float(ratio.max())),
        top10_ratio=table.nlargest(10, 'ratio')[['resid_pole', 'rmsf_wt_A', 'rmsf_p286r_A', 'ratio']]
        .round(3).to_dict('records'))
    (HERE / 'results/rmsf_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
