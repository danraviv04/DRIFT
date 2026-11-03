"""Read-only, bounded comparison of structural representatives and pLDDT ensembles.

Run with the existing thesis_core Python. No raw structures are regenerated or
realigned. Outputs are local to this new exploratory directory and never replaced.
"""
from pathlib import Path
import ast
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from metric_analysis.pipeline.labels import artifact_key, extract_mutation_code
from metric_analysis.pipeline.library.canonical_cohort import load_all_canonical_cohorts
from metric_analysis.pipeline.library.descriptor_metrics import relative_change, residue_ks
from metric_analysis.pipeline.library.descriptor_sampling import apply_row_block, row_block

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
GENES = ['VHL', 'POLE', 'PTEN', 'MSH2', 'MSH6', 'PIK3CA']
EXPECTED = {'POLE': (1074, 38),  # P356S out of the cohort since 2026-10-20
            'PTEN': (257, 248), 'VHL': (124, 105),
            'MSH2': (109, 71), 'MSH6': (143, 66), 'PIK3CA': (91, 84)}
TAGS = sorted(f'seed-{s}_sample-{k}' for s in range(1, 26) for k in range(5))
XYZ = ['x_aligned', 'y_aligned', 'z_aligned']
SOURCES = {}


def digest(path):
    path = Path(path)
    if str(path) not in SOURCES:
        SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return SOURCES[str(path)]


def save(frame, name):
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False)


def matrix(record, wt=False):
    path = record['wt_source_file' if wt else 'source_file']
    frame = pd.read_csv(path, index_col=0)
    assert frame.index.tolist() == list(range(len(frame))), path
    assert set(TAGS).issubset(frame.columns), path
    assert not frame.columns.duplicated().any(), path
    x = apply_row_block(frame[TAGS].to_numpy(float), row_block(record))
    assert np.isfinite(x).all() and (x > 0).all(), path
    digest(path)
    return x


def selections(frame, recovered, length):
    frame = frame.copy()
    if recovered is not None:
        assert frame['sample'].isna().all() or frame['sample'].astype(str).eq('').all()
        frame['sample'] = frame.structure_index.map(recovered)
    frame = frame[frame['sample'].isin(TAGS)]
    assert set(frame['sample']) == set(TAGS)
    assert frame.chain.nunique() == 1
    assert not frame.duplicated(['sample', 'resid']).any()
    assert set(frame.resid.unique()) == set(range(1, length + 1))
    frame = frame.sort_values(['sample', 'resid'])
    assert len(frame) == 125 * length
    xyz = frame[XYZ].to_numpy().reshape(125, length, 3)
    assert np.isfinite(xyz).all()
    distances = squareform(pdist(xyz.reshape(125, -1))) / np.sqrt(length)
    medoid = int(distances.sum(1).argmin())
    nearest = int(np.square(xyz - xyz.mean(0)).sum((1, 2)).argmin())
    sample_indices = frame.groupby('sample', observed=True).structure_index.first()
    return {'medoid': medoid, 'nearest_mean': nearest,
            'medoid_structure_index': int(sample_indices[TAGS[medoid]]),
            'nearest_mean_structure_index': int(sample_indices[TAGS[nearest]]),
            'mean_pairwise_rmsd_at_medoid': float(distances[medoid].mean())}, xyz


def artifact_frames(path, artifacts):
    """Stream an artifact-sorted parquet once, keeping at most one carry group."""
    dataset = ds.dataset(path)
    columns = ['artifact', 'structure_index', 'sample', 'chain', 'resid'] + XYZ
    scanner = dataset.scanner(columns=columns,
                              filter=ds.field('artifact').isin(artifacts), batch_size=250000)
    carry = None
    seen = set()
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        names = frame.artifact.astype(str)
        last = names.iloc[-1]
        carry = frame[names.eq(last)]
        for artifact, group in frame[~names.eq(last)].groupby('artifact', observed=True, sort=False):
            assert artifact not in seen, 'cache must be artifact-contiguous'
            seen.add(artifact)
            yield str(artifact), group
    if carry is not None:
        artifact = str(carry.artifact.iloc[0])
        assert artifact not in seen
        yield artifact, carry


def cached_baselines(cohort, excluded):
    rows = []
    for n in [1, 90]:
        for seed in range(42, 52):
            path = ROOT / 'experiments/pole_wt_repair/cache' / (
                f'labeled_size{n}_seed{seed}_ratio_of_means_proteinsebbbda31758d_esfeat6.csv')
            digest(path)
            frame = pd.read_csv(path).query('gene in @GENES')
            canonical = cohort.rename(columns={'protein': 'gene', 'mutation_code': 'mutation'})
            frame = frame.merge(canonical[['gene', 'mutation', 'label']],
                                on=['gene', 'mutation'], validate='one_to_one', suffixes=('', '_canonical'))
            assert len(frame) == 1798 and frame.label.eq(frame.label_canonical).all()
            for scope in ['all_annotation', 'coordinate_matched']:
                d = frame if scope == 'all_annotation' else frame[
                    ~pd.MultiIndex.from_frame(frame[['gene', 'mutation']]).isin(excluded)]
                for gene, block in d.groupby('gene'):
                    for name, col in [('D1', 'd1_score'), ('KS', 'plddt_ks_residue')]:
                        assert np.isfinite(block[col]).all()
                        rows.append(dict(gene=gene, method=name, N=n, seed=seed, cohort=scope,
                                         n=len(block), positives=int(block.label.sum()),
                                         auc=roc_auc_score(block.label, block[col])))
    result = pd.DataFrame(rows)
    macro = result.groupby(['method', 'N', 'seed', 'cohort'], as_index=False).auc.mean().assign(gene='MACRO')
    result = pd.concat([result, macro], ignore_index=True)
    save(result, 'corrected_cached_auc_per_seed.csv')
    summary = result.groupby(['gene', 'method', 'N', 'cohort'], as_index=False).agg(
        auc=('auc', 'mean'), sampling_sd=('auc', 'std'))
    save(summary, 'corrected_cached_auc_summary.csv')
    reference = {('KS', 1): .605564, ('KS', 90): .850969,
                 ('D1', 1): .621482, ('D1', 90): .814864}
    for (name, n), value in reference.items():
        actual = summary.query('gene == "MACRO" and cohort == "all_annotation" and method == @name and N == @n').auc.item()
        assert abs(actual - value) < .0000005, (name, n, actual)
    return summary


def main():
    start = time.monotonic()
    assert not (OUT / 'variant_scores.csv').exists()
    cohort = load_all_canonical_cohorts()
    cohort = cohort[cohort.protein.isin(GENES)].copy()
    for gene, (count, positives) in EXPECTED.items():
        d = cohort[cohort.protein == gene]
        assert len(d) == count and d.label.sum() == positives
    digest(ROOT / 'metric_analysis/manifests/strict_canonical_manifest.csv')
    score_rows, selection_rows, exclusions, validation, inventory = [], [], [], [], []
    # Extract just this existing pure function, avoiding unrelated plotting imports.
    utility = ROOT / 'metric_analysis/pipeline/library/structure_metrics.py'
    node = next(n for n in ast.parse(utility.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_medoid_frame_index')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(utility), 'exec'), namespace)
    for gene in GENES:
        if time.monotonic() - start > 180:
            raise RuntimeError('Lightweight runtime budget reached; do not expand the run.')
        gene_start = time.monotonic()
        records = cohort[cohort.protein == gene].to_dict('records')
        folder = ROOT / 'metric_analysis/structure_metrics' / gene / 'ensemble_aware/cache_corefit'
        report_path = folder / f'{gene}_core_fit_report.json'
        report = json.loads(report_path.read_text())
        digest(report_path)
        path = folder / f'{gene}_rmsf_aligned_ca_coordinate_cache.parquet'
        inventory.append({'gene': gene, 'path': str(path), 'bytes': path.stat().st_size,
                          'mtime_ns': path.stat().st_mtime_ns, 'wt': report['wt_artifact'],
                          'fit_chain': report['chain'], 'fit_residues': report['core_residues']})
        index_path = folder / f'{gene}_core_fit_per_structure.csv'
        index = pd.read_csv(index_path)
        digest(index_path)
        by_mutation = {}
        labelled_mutations = {r['mutation_code'] for r in records}
        for artifact in index.artifact.unique():
            mutation = extract_mutation_code(artifact)
            if mutation in labelled_mutations:
                assert mutation not in by_mutation, (gene, mutation)
                by_mutation[mutation] = artifact
        recovered = None
        if gene in ['MSH2', 'PIK3CA']:
            mapping_path = ROOT / 'experiments/seed_structure/results_rerun_20261018' / f'{gene}_recovered_seed_mapping.csv'
            mapping = pd.read_csv(mapping_path)
            digest(mapping_path)
            assert len(mapping) == 125 and set(mapping['sample']) == set(TAGS)
            assert mapping.archive_member.str.extract(r'(seed-\d+_sample-\d+)', expand=False).eq(mapping['sample']).all()
            recovered = dict(zip(mapping.structure_index, mapping['sample']))
        assert len({r['wt_source_file'] for r in records}) == 1
        assert Path(records[0]['wt_source_file']).resolve() == Path(report['wt_plddt_csv']).resolve()
        if gene == 'POLE':
            assert report['wt_artifact'] == 'POLE_268-471_WT'
        wt = matrix(records[0], wt=True)
        assert len(wt) == report['residues']
        wanted = [report['wt_artifact']]
        for record in records:
            mutation = record['mutation_code']
            if mutation not in by_mutation:
                exclusions.append({'gene': gene, 'mutation': mutation, 'label': record['label'],
                                   'reason': 'coordinate artifact missing'})
            else:
                artifact = by_mutation[mutation]
                missing = set(TAGS) - set(index.loc[index.artifact.eq(artifact), 'sample'].dropna())
                if missing:
                    exclusions.append({'gene': gene, 'mutation': mutation, 'label': record['label'],
                                       'reason': f'{len(missing)} coordinate seed/sample identities missing'})
                    del by_mutation[mutation]
                    continue
                wanted.append(by_mutation[mutation])
        selected = {}
        for artifact, frame in artifact_frames(path, wanted):
            is_wt = artifact == report['wt_artifact']
            choice, xyz = selections(frame, recovered if is_wt else None, len(wt))
            # Compare optimized distance computation to the existing exact utility
            # once per gene, on all 125 members and all scored residues.
            if not selected:
                existing = namespace['_medoid_frame_index'](xyz)
                assert choice['medoid'] == existing
                validation.append({'gene': gene, 'check': 'medoid_matches_existing_exact_utility',
                                   'artifact': artifact, 'passed': True})
            selected[artifact] = choice
            for method in ['medoid', 'nearest_mean']:
                selection_rows.append({'gene': gene, 'artifact': artifact, 'is_wt': is_wt,
                                       'selection': method, 'sample': TAGS[choice[method]],
                                       'structure_index': choice[f'{method}_structure_index'],
                                       'residues': len(wt), 'pool_n': 125})
        assert set(selected) == set(wanted)
        wc = selected[report['wt_artifact']]
        for record in records:
            mutation = record['mutation_code']
            if mutation not in by_mutation:
                continue
            mutant = matrix(record)
            assert mutant.shape == wt.shape
            mc = selected[by_mutation[mutation]]
            values = {'full_KS_125': residue_ks(wt, mutant),
                      'ensemble_mean_D1_125': relative_change(wt.mean(1), mutant.mean(1)),
                      'ensemble_mean_MAE_125': float(np.abs(wt.mean(1) - mutant.mean(1)).mean())}
            for method in ['medoid', 'nearest_mean']:
                w, m = wt[:, wc[method]], mutant[:, mc[method]]
                values[f'{method}_D1'] = relative_change(w, m)
                values[f'{method}_MAE'] = float(np.abs(w - m).mean())
                values[f'{method}_singleton_KS'] = float((w != m).mean())
            if len(score_rows) == 0 or score_rows[-1]['gene'] != gene:
                check = np.mean([ks_2samp(a, b).statistic for a, b in zip(wt, mutant)])
                assert abs(check - values['full_KS_125']) < 1e-12
                validation.append({'gene': gene, 'check': 'full_KS_matches_scipy_all_residues',
                                   'artifact': by_mutation[mutation], 'passed': True})
            score_rows.append({'gene': gene, 'mutation': mutation, 'label': record['label'],
                               'residues': len(wt), 'pool_n': 125, **values})
        print(f'{gene}: {len(records)} labelled; {len(wanted)-1} coordinate-covered; '
              f'{time.monotonic()-gene_start:.2f}s', flush=True)
    scores = pd.DataFrame(score_rows)
    assert [(r['gene'], r['mutation']) for r in exclusions] == [('POLE', 'P356S'), ('MSH2', 'G162R')]
    save(scores, 'variant_scores.csv')
    save(pd.DataFrame(selection_rows), 'structure_selections.csv')
    save(pd.DataFrame(exclusions), 'exclusions.csv')
    methods = [c for c in scores if c not in ['gene', 'mutation', 'label', 'residues', 'pool_n']]
    auc_rows = []
    for gene, block in scores.groupby('gene'):
        for method in methods:
            assert np.isfinite(block[method]).all()
            auc_rows.append({'gene': gene, 'method': method, 'n': len(block),
                             'positives': int(block.label.sum()),
                             'auc': roc_auc_score(block.label, block[method])})
    auc = pd.DataFrame(auc_rows)
    macro = auc.groupby('method', as_index=False).auc.mean().assign(gene='MACRO', n=len(scores), positives=int(scores.label.sum()))
    auc = pd.concat([auc, macro], ignore_index=True)
    save(auc, 'auc_summary.csv')
    baseline = cached_baselines(cohort, [(r['gene'], r['mutation']) for r in exclusions])
    save(pd.DataFrame(validation), 'validation.csv')
    save(pd.DataFrame(inventory), 'coordinate_inventory.csv')
    save(cohort, 'annotation_cohort.csv')
    provenance = {'selection': 'argmin_i sum_j sqrt(mean_r sum_xyz (X_i-X_j)^2); no pairwise refitting',
                  'frame': 'existing core_fit frame; selection and scores use every scored-chain residue',
                  'pools': '125 unique seed/sample tags, seeds 1..25, samples 0..4; no top-level copies',
                  'ties': 'first lexicographic seed/sample order; labels never used in selection',
                  'd1': 'float64 canonical relative_change, no zero or nonfinite pLDDT in input',
                  'cache_baselines': 'corrected fingerprint ebbbda31758d, ten seeds 42..51; mean per-seed AUC',
                  'coordinate_hash_policy': 'size and mtime only for multi-GB read-only coordinates; small inputs SHA256',
                  'runtime_seconds': time.monotonic()-start, 'source_sha256': SOURCES,
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(auc.pivot(index='gene', columns='method', values='auc').round(6).to_string())
    print(baseline.query('gene == "MACRO"').to_string(index=False))
    print(f'Total {time.monotonic()-start:.2f}s', flush=True)


if __name__ == '__main__':
    main()
