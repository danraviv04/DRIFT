"""Full-atom POLE WT and P286R models in the core-fit frame, for the Fig. 1b render.

Fig. 1b superposes the WT and p.P286R POLE ensembles (125 predictions each). The
coordinate cache stores only C-alpha positions in the core-fit frame
(cache_builders/subunit_core_fit.py: Kabsch on the WT rigid core, which for POLE is all
204 residues, onto the mean WT core). The per-structure transforms were not saved, but a
superposition is a rigid motion, so each one is recovered exactly: Kabsch of the
structure's own C-alpha atoms (from the AF3 CIF on the drive) onto its cached core-fit
C-alpha positions, then applied to every atom (protein, DNA, Mg).

The ensembles are exactly the pLDDT ensembles the KS-score reads: the columns of the WT
CSV the POLE config pins and of the 25-seed P286R copy (seeds 1-25, 125 each). Only the
atom_site coordinates are rewritten; every other byte of the AF3 CIF is kept, so the
B-factor column still holds AF3's per-atom pLDDT. Residue numbers stay AF3's 1-204
(POLE 268-471); fig1b_ensembles.cxc renumbers them in ChimeraX.

Checks (any failure stops the run): the fit residual onto the cache, the written C-alpha
positions against the cache, the C-alpha B-factor against the pLDDT CSV, the sample tag
of the drive path against the cache, and residue 19 (p.286) being PRO / ARG.

Reads the Elements drive (one sequential reader, ~175 MB); writes ~175 MB to models/.

Run from the repo root:
    python figures/figure1/fig1b/export_models.py
"""
from pathlib import Path
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from metric_analysis.pipeline.cache_builders.subunit_core_fit import kabsch  # noqa: E402
from metric_analysis.pipeline.paths import on_archive_root  # noqa: E402

CACHE_DIR = ROOT / 'metric_analysis/structure_metrics/POLE/ensemble_aware'
CORE_FIT = CACHE_DIR / 'cache_corefit/POLE_rmsf_aligned_ca_coordinate_cache.parquet'
CORE_FIT_RMSD = CACHE_DIR / 'cache_corefit/POLE_core_fit_per_structure.csv'
INDEX = CACHE_DIR / 'cache/POLE_structure_file_index.csv'
SIDES = {
    'WT': ('POLE_268-471_WT', ROOT / 'PLDDTs/POLE/POLE_268-471_WT_plddt_values.csv', 'PRO'),
    'P286R': ('POLE_268-471_P286R', ROOT / 'PLDDTs/POLE_25seed/POLE_268-471_P286R_plddt_values.csv', 'ARG'),
}
CHAIN = 'A'
L = 204
SITE = 19                 # label_seq_id of p.286 (index 0 is residue 268)
MAX_FIT_RESIDUAL = 0.01   # A; the CIF has 3 decimals, so a true rigid motion leaves ~0.001
MAX_PLDDT_DIFF = 0.01
SAMPLE = re.compile(r'seed-\d+_sample-\d+')


def read_atom_site(lines):
    """Column names, data line numbers and their token spans for the atom_site loop."""
    head = next(i for i, s in enumerate(lines) if s.startswith('_atom_site.'))
    cols = []
    i = head
    while lines[i].startswith('_atom_site.'):
        cols.append(lines[i].strip().split('.', 1)[1])
        i += 1
    rows = []
    while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', 'loop_', '_')):
        rows.append((i, [m.span() for m in re.finditer(r'\S+', lines[i])]))
        i += 1
    return cols, rows


def tokens(line, spans, cols, names):
    return [line[slice(*spans[cols.index(n)])] for n in names]


def export(src, cached_ca, out):
    lines = src.read_text().splitlines(keepends=True)
    cols, rows = read_atom_site(lines)
    xyz = np.array([[float(t) for t in tokens(lines[i], sp, cols, ('Cartn_x', 'Cartn_y', 'Cartn_z'))]
                    for i, sp in rows])
    meta = [tokens(lines[i], sp, cols, ('label_asym_id', 'label_atom_id', 'label_seq_id', 'label_comp_id',
                                        'B_iso_or_equiv')) for i, sp in rows]
    ca = [k for k, (ch, atom, *_) in enumerate(meta) if ch == CHAIN and atom == 'CA']
    seq = [int(meta[k][2]) for k in ca]
    if seq != list(range(1, L + 1)):
        raise ValueError(f'{src}: chain {CHAIN} C-alpha residues are not 1-{L}')

    R, t = kabsch(xyz[ca], cached_ca)
    if not (np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)):
        raise ValueError(f'{src}: transform is not a proper rotation')
    fit_residual = float(np.sqrt((((xyz[ca] @ R + t) - cached_ca) ** 2).sum(1).mean()))
    if fit_residual > MAX_FIT_RESIDUAL:
        raise ValueError(f'{src}: fit residual {fit_residual:.4f} A onto the cache; not the cached structure')

    new = xyz @ R + t
    for (i, sp), v in zip(rows, new):
        s = lines[i]
        for n, val in zip(('Cartn_z', 'Cartn_y', 'Cartn_x'), v[::-1]):   # right to left keeps spans valid
            a, b = sp[cols.index(n)]
            s = s[:a] + f'{val:.3f}'.rjust(b - a) + s[b:]
        lines[i] = s
    out.write_text(''.join(lines))
    return dict(fit_residual_A=fit_residual,
                resname_site=meta[ca[SITE - 1]][3],
                ca_bfactor=np.array([float(meta[k][4]) for k in ca]),
                atoms=len(rows))


def check_written(path, cached_ca):
    lines = path.read_text().splitlines()
    cols, rows = read_atom_site(lines)
    ca = [[float(t) for t in tokens(lines[i], sp, cols, ('Cartn_x', 'Cartn_y', 'Cartn_z'))]
          for i, sp in rows if tokens(lines[i], sp, cols, ('label_asym_id', 'label_atom_id')) == [CHAIN, 'CA']]
    return float(np.abs(np.array(ca) - cached_ca).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='first N samples per side (smoke test)')
    args = ap.parse_args()

    arts = [a for a, _, _ in SIDES.values()]
    cache = pq.read_table(CORE_FIT, filters=[('artifact', 'in', arts)]).to_pandas()
    cache['artifact'] = cache['artifact'].astype(str)
    cache['sample'] = cache['sample'].astype(str)
    index = pd.read_csv(INDEX)
    index = index[index.artifact.isin(arts)]
    fit_rmsd = pd.read_csv(CORE_FIT_RMSD)
    fit_rmsd = fit_rmsd[fit_rmsd.artifact.isin(arts)].set_index(['artifact', 'sample']).fit_rmsd

    records = []
    mean_ca = {}
    for side, (art, csv, site_res) in SIDES.items():
        plddt = pd.read_csv(csv, index_col=0)
        if plddt.shape != (L, 125):
            raise ValueError(f'{csv}: shape {plddt.shape}, expected ({L}, 125)')
        out_dir = HERE / 'models' / side
        out_dir.mkdir(parents=True, exist_ok=True)
        c = cache[cache.artifact == art]
        paths = index[index.artifact == art].set_index('structure_index').structure_path
        mean_ca[side] = 0.0
        for sample in list(plddt.columns)[:args.limit]:
            s = c[c['sample'] == sample].sort_values('resid')
            if s.resid.tolist() != list(range(1, L + 1)):
                raise ValueError(f'{art} {sample}: cache does not hold residues 1-{L}')
            sidx = int(s.structure_index.iloc[0])
            src = on_archive_root(paths.loc[sidx])
            if src.parent.name != sample:
                raise ValueError(f'{art} {sample}: structure_index {sidx} points at {src}')
            cached_ca = s[['x_aligned', 'y_aligned', 'z_aligned']].to_numpy()
            mean_ca[side] = mean_ca[side] + cached_ca
            out = out_dir / f'{art}_{sample}.cif'
            r = export(src, cached_ca, out)
            if r['resname_site'] != site_res:
                raise ValueError(f'{src}: residue {SITE} is {r["resname_site"]}, expected {site_res}')
            plddt_diff = float(np.abs(r.pop('ca_bfactor') - plddt[sample].to_numpy()).max())
            if plddt_diff > MAX_PLDDT_DIFF:
                raise ValueError(f'{src}: C-alpha B-factor differs from {csv.name} by {plddt_diff:.3f}')
            records.append(dict(side=side, artifact=art, sample=sample, structure_index=sidx,
                                source=str(src), output=str(out.relative_to(HERE)),
                                core_fit_rmsd_A=float(fit_rmsd.loc[(art, sample)]),
                                written_ca_max_dev_A=check_written(out, cached_ca),
                                ca_plddt_max_diff=plddt_diff, mean_ca_plddt=float(plddt[sample].mean()),
                                **r))
            print(f'{side:5s} {sample:18s} residual {r["fit_residual_A"]:.4f} A', flush=True)

    m = pd.DataFrame(records)
    (HERE / 'results').mkdir(exist_ok=True)
    m.to_csv(HERE / 'results/manifest.csv', index=False)
    summary = {side: dict(models=int(len(g)),
                          fit_residual_A_max=float(g.fit_residual_A.max()),
                          written_ca_max_dev_A=float(g.written_ca_max_dev_A.max()),
                          ca_plddt_max_diff=float(g.ca_plddt_max_diff.max()),
                          core_fit_rmsd_A_median=float(g.core_fit_rmsd_A.median()),
                          mean_ca_plddt=float(g.mean_ca_plddt.mean()),
                          atoms_per_model=sorted(g.atoms.unique().tolist()))
               for side, g in m.groupby('side', sort=False)}
    # WT and P286R ensemble-mean C-alpha structures as they sit in the frame (no further fit)
    wt, mut = (mean_ca[s] / (m.side == s).sum() for s in SIDES)
    summary['mean_structure_ca_rmsd_A'] = float(np.sqrt(((wt - mut) ** 2).sum(1).mean()))
    (HERE / 'results/summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
