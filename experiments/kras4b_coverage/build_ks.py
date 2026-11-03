"""Extract validated KRAS4B pLDDTs and compute N=90 KS without binary selection."""
import os
import gzip
import hashlib
import io
import json
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from drift.structure_draws import select_structure_columns, source_draw_key
from metric_analysis.pipeline.library.descriptor_metrics import residue_ks
from experiments.elements_drive_audit.validate_batch import SEED_SAMPLE, _Hashing

DRIVE = Path(os.environ.get("DRIFT_ARCHIVE_ROOT", "$DRIFT_ARCHIVE_ROOT")) / 'KRAS4B_Mutations'
PLDDT = ROOT / 'PLDDTs/KRAS'  # PLDDTs/KRAS4B until 2026-10-13; draw keys hash this path
OUT = Path(__file__).resolve().parent / 'results/ks'
VALIDATION = ROOT / 'experiments/elements_drive_audit/results/KRAS4B_Mutations_validation.csv'


def ca_values(text):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('_atom_site.'))
    cols, j = [], start
    while lines[j].startswith('_atom_site.'):
        cols.append(lines[j].split('.', 1)[1].strip())
        j += 1
    ix = {c: i for i, c in enumerate(cols)}
    values = {}
    for line in lines[j:]:
        if not line or line.startswith(('#', '_', 'loop_')):
            break
        row = line.split()
        if row[ix['group_PDB']] == 'ATOM' and row[ix['label_asym_id']] == 'A' and row[ix['label_atom_id']] == 'CA':
            pos = int(row[ix['label_seq_id']])
            if pos in values:
                raise ValueError('Duplicate CA residue')
            values[pos] = float(row[ix['B_iso_or_equiv']])
    if set(values) != set(range(1, 189)):
        raise ValueError('Expected all 188 KRAS4B residues')
    a = np.array([values[i] for i in range(1, 189)])
    if not np.isfinite(a).all() or not ((a >= 0) & (a <= 100)).all():
        raise ValueError('Invalid pLDDT')
    return a


def extract(archive):
    path = DRIVE / archive
    job = archive.removesuffix('_data.tar.gz')
    output = PLDDT / f'{job}_plddt_values.csv'
    # Resume only after confirming both source identity and complete matrix.
    metadata = output.with_suffix('.json')
    identity = {'archive': archive, 'bytes': path.stat().st_size, 'mtime_ns':path.stat().st_mtime_ns}
    if output.exists() and metadata.exists():
        old = json.loads(metadata.read_text())
        if 'archive_md5' in old and all(old.get(k) == v for k,v in identity.items()):
            matrix = pd.read_csv(output,index_col=0)
            if matrix.shape == (188,125) and np.isfinite(matrix.to_numpy()).all() and hashlib.sha256(output.read_bytes()).hexdigest() == old['csv_sha256']:
                return old
    samples = {}
    with open(path, 'rb', buffering=0) as raw:
      hashing = _Hashing(raw)
      with gzip.GzipFile(fileobj=io.BufferedReader(hashing, 1 << 20)) as gz:
       with tarfile.open(fileobj=gz, mode='r|') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('_model.cif') or not SEED_SAMPLE.search(member.name):
                continue
            name = Path(member.name).parent.name
            text = tar.extractfile(member).read().decode()
            a = ca_values(text)
            if not samples:
                # Independently check the fast reader against Bio.PDB once per archive.
                model = next(MMCIFParser(QUIET=True).get_structure('x', io.StringIO(text)).get_models())
                reference = np.array([r['CA'].bfactor for r in model['A'] if r.id[0] == ' '])
                np.testing.assert_array_equal(a, reference)
            if name in samples:
                raise ValueError(f'{archive}: duplicate sample {name}')
            samples[name] = a
       while gz.read(1 << 22):
           pass
      identity['archive_md5'] = hashing.md5.hexdigest()
    expected = {f'seed-{s}_sample-{i}' for s in range(1,26) for i in range(5)}
    if set(samples) != expected:
        raise ValueError(f'{archive}: incomplete or unexpected seed/sample set')
    pd.DataFrame({k:samples[k] for k in sorted(samples)}).to_csv(output)
    identity.update(csv_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), n_residues=188, n_samples=125)
    metadata.write_text(json.dumps(identity,indent=2)+'\n')
    return identity


def main():
    audit = pd.read_csv(VALIDATION)
    flags = ['readable','gzip_ok','success','name_ok','entities_ok','seeds_ok','ensemble_ok','models_ok']
    assert len(audit)==980 and audit[flags].eq(True).all().all()
    assert audit.vs_wt_ok.fillna(audit.job.str.endswith('_WT')).eq(True).all()
    assert not audit.failed_marker.any() and audit.n_seed_cifs.eq(125).all()
    strata_path = Path(__file__).resolve().parent / 'results/batch_strata.csv'
    strata = pd.read_csv(strata_path)
    assert strata.VAR.is_unique and len(strata)==979
    assert set(audit.job)=={'KRAS_'+v for v in strata.VAR} | {'KRAS_WT'}
    PLDDT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    extracts=[]
    with ProcessPoolExecutor(3) as pool:
        futures=[pool.submit(extract,a) for a in audit.archive]
        for i, f in enumerate(as_completed(futures),1):
            extracts.append(f.result())
            if i % 50 == 0 or i==980: print(f'Extracted {i}/980',flush=True)
    audited_hashes = audit.set_index('archive').md5.to_dict()
    assert all(rec['archive_md5'] == audited_hashes[rec['archive']] for rec in extracts)
    wt_path=PLDDT/'KRAS_WT_plddt_values.csv'
    wt=pd.read_csv(wt_path,index_col=0)
    wt_draws={seed:wt[select_structure_columns(list(wt.columns),90,seed=seed,
                key=f'wt::KRAS::{wt_path.name}')].to_numpy() for seed in range(42,52)}
    rows=[]
    for i, rec in enumerate(strata.itertuples(),1):
        path=PLDDT/f'KRAS_{rec.VAR}_plddt_values.csv'
        mutant=pd.read_csv(path,index_col=0)
        for seed in range(42,52):
            cols=select_structure_columns(list(mutant.columns),90,seed=seed,key=source_draw_key('mut',str(path)))
            a,b=wt_draws[seed],mutant[cols].to_numpy()
            ks=residue_ks(a,b)
            if i==1 and seed==42:
                assert np.isclose(ks,residue_ks(a,b,backend='scipy'))
            rows.append(dict(gene='KRAS',mutation=rec.VAR,seed=seed,plddt_ks_residue=ks,stratum=rec.stratum))
        if i%200==0:print(f'KS {i}/979',flush=True)
    runs=pd.DataFrame(rows)
    runs.to_csv(OUT/'per_seed.csv',index=False)
    means=runs.groupby(['gene','mutation','stratum']).agg(plddt_ks_residue=('plddt_ks_residue','mean'),n_seeds=('seed','nunique')).reset_index()
    assert means.n_seeds.eq(10).all()
    means.to_csv(OUT/'mean_scores.csv',index=False)
    metadata={'N':90,'seeds':list(range(42,52)),'sampling':'drift.structure_draws.select_structure_columns, default nested random; ensemble_shift WT/mutant draw keys',
              'metric':'shared descriptor_metrics.residue_ks; mean over 188 protein residues',
              'validation_sha256':hashlib.sha256(VALIDATION.read_bytes()).hexdigest(),
              'strata_sha256':hashlib.sha256(strata_path.read_bytes()).hexdigest(),
              'primary_strata':['blind','blind+hotspot'],'extracts':sorted(extracts,key=lambda x:x['archive'])}
    (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(means.groupby('stratum').size().to_string())


if __name__=='__main__':
    main()
