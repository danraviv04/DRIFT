"""Matched production-draw representative comparison and paired uncertainty.

Uses existing structures only. Run with PYTHONPATH=. in thesis_core.
Never replaces the earlier N=125 exploration or its outputs.
"""
from pathlib import Path
import hashlib
import json
import re
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from drift import structure_draws as runtime
from drift.protein_config import load_protein_configs
from metric_analysis.pipeline.library.descriptor_metrics import residue_ks
from metric_analysis.pipeline.library.descriptor_sampling import apply_row_block, row_block
from metric_analysis.pipeline.labels import artifact_key
from metric_analysis.pipeline.library.af3_seeds import within_seed_cap
from metric_analysis.pipeline.library.canonical_cohort import _existing_repo_path
from metric_analysis.pipeline.library.position_bootstrap import clusters_of, strata_of, stratified_weights
from check_representatives import ROOT, OUT as HERE, GENES, artifact_frames

OUT = HERE / 'n90_results'
# Inputs, overridable by a caller (analyses/annotation/stages redirects them; defaults unchanged).
COHORT = HERE / 'annotation_cohort.csv'
EXCLUSIONS = HERE / 'exclusions.csv'
CACHE = ROOT / 'experiments/pole_wt_repair/cache'
FP = 'ebbbda31758d'
SEEDS = list(range(42, 52))
METHODS = ['KS', 'mean_D1', 'mean_MAE', 'medoid_D1', 'medoid_MAE',
           'nearest_mean_D1', 'nearest_mean_MAE']
BOOT = 2000
HASHES = {}


def sha(path):
    path = Path(path)
    HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()


def load_matrix(record, wt=False):
    path = record['wt_source_file' if wt else 'source_file']
    frame = pd.read_csv(path, index_col=0)
    assert frame.index.tolist() == list(range(len(frame)))
    assert not frame.columns.duplicated().any()
    assert all(re.fullmatch(r'seed-\d+_sample-\d+', c) for c in frame.columns)
    sha(path)
    matrix = apply_row_block(frame.to_numpy(float), row_block(record))
    assert np.isfinite(matrix).all() and (matrix > 0).all()
    return matrix, list(frame.columns)


def orders(columns, cfg, source, wt=False):
    key = f"wt::{cfg['mut']}::{Path(source).name}" if wt else runtime.source_draw_key('mut', source)
    tags = [runtime.select_structure_columns(columns, 90, seed=seed, key=key) for seed in SEEDS]
    # every draw within the gene's AF3 seed cap (POLE: seeds 1-25, 2026-10-20)
    assert within_seed_cap([t for draw in tags for t in draw], cfg['mut']).all(), source
    return np.array([[columns.index(t) for t in draw] for draw in tags]), tags


def select(frame, draw_tags, recovered=None):
    frame = frame.copy()
    if recovered is not None:
        frame['sample'] = frame.structure_index.map(recovered)
    assert not frame.duplicated(['sample', 'resid']).any()
    tags = sorted(set(t for draw in draw_tags for t in draw))
    assert set(tags).issubset(frame['sample'].dropna())
    frame = frame[frame['sample'].isin(tags)].sort_values(['sample', 'resid'])
    length = frame.resid.nunique()
    assert set(frame.resid) == set(range(1, length + 1)) and frame.chain.nunique() == 1
    assert len(frame) == len(tags) * length
    xyz = frame[['x_aligned', 'y_aligned', 'z_aligned']].to_numpy().reshape(len(tags), length, 3)
    assert np.isfinite(xyz).all()
    distances = squareform(pdist(xyz.reshape(len(tags), -1))) / np.sqrt(length)
    choices = []
    for draw in draw_tags:
        candidates = sorted(draw)
        indices = np.array([tags.index(t) for t in candidates])
        selected = indices[distances[np.ix_(indices, indices)].sum(1).argmin()]
        tensor = xyz[indices]
        near = indices[np.square(tensor - tensor.mean(0)).sum((1, 2)).argmin()]
        choices.append((tags[selected], tags[near]))
    # Direct pairwise subtraction independently checks one actual N=90 medoid.
    ids = np.array([tags.index(t) for t in sorted(draw_tags[0])])
    tensor = xyz[ids]
    totals = [np.sqrt(np.square(tensor - x).sum(2).mean(1)).sum() for x in tensor]
    assert tags[ids[int(np.argmin(totals))]] == choices[0][0]
    return choices


def d1(a, b):
    # Match the production ratio/aggregation precision for all D1 baselines.
    return float(np.abs(b.astype(np.float32) / a.astype(np.float32) - 1).mean())


def calculate():
    cohort = pd.read_csv(COHORT)
    excluded = pd.read_csv(EXCLUSIONS)
    # Recorded paths name the main checkout; resolve them in this checkout, as the
    # canonical cohort does, so the source draw keys match the caches built here.
    for column in ('source_file', 'wt_source_file'):
        cohort[column] = [str(_existing_repo_path(v, protein=p))
                          for v, p in zip(cohort[column], cohort.parent_dataset)]
    keys = pd.MultiIndex.from_frame(excluded[['gene', 'mutation']])
    keep = ~pd.MultiIndex.from_frame(cohort[['protein', 'mutation_code']]).isin(keys)
    cohort = cohort[keep].copy()
    assert len(cohort) == 1797
    configs = load_protein_configs()
    reference = []
    for seed in SEEDS:
        path = CACHE / f'labeled_size90_seed{seed}_ratio_of_means_proteins{FP}_esfeat6.csv'
        sha(path)
        reference.append(pd.read_csv(path).assign(seed=seed))
    reference = pd.concat(reference).set_index(['gene', 'mutation', 'seed'])
    rows, selections, audits = [], [], []
    for gene in GENES:
        started = time.monotonic()
        records = cohort[cohort.protein.eq(gene)].to_dict('records')
        cfg = configs[gene]['d1']
        source = records[0]['wt_source_file']
        assert {r['wt_source_file'] for r in records} == {source}
        assert [p.resolve() for p in cfg['read_root'].glob(cfg['wt_csv_glob'])] == [Path(source).resolve()]
        folder = ROOT / 'metric_analysis/structure_metrics' / gene / 'ensemble_aware/cache_corefit'
        report_path = folder / f'{gene}_core_fit_report.json'
        report = json.loads(report_path.read_text())
        sha(report_path)
        if gene == 'POLE':
            assert report['wt_artifact'] == 'POLE_268-471_WT'
        index_path = folder / f'{gene}_core_fit_per_structure.csv'
        sha(index_path)
        index = pd.read_csv(index_path)
        by_key = {artifact_key(a): a for a in index.artifact.unique()}
        wt, wt_cols = load_matrix(records[0], wt=True)
        assert len(wt) == report['residues']
        wo, wt_tags = orders(wt_cols, cfg, source, wt=True)
        matrices, tags, mutant_orders, sources = {}, {}, {}, {}
        for record in records:
            mutation = record['mutation_code']
            key = artifact_key(Path(record['source_file']).name.removesuffix('_plddt_values.csv'))
            artifact = by_key[key]
            m, columns = load_matrix(record)
            assert len(m) == len(wt)
            mo, mtags = orders(columns, cfg, record['source_file'])
            matrices[artifact], tags[artifact], mutant_orders[artifact] = (m, columns), mtags, mo
            sources[mutation] = artifact
        recovered = None
        if gene in ['MSH2', 'PIK3CA']:
            mp = ROOT / 'analyses/seeds/seed_structure/results_rerun_20261018' / f'{gene}_recovered_seed_mapping.csv'
            sha(mp)
            mapping = pd.read_csv(mp)
            recovered = dict(zip(mapping.structure_index, mapping['sample']))
        wanted = [report['wt_artifact'], *matrices]
        coordinate_path = folder / f'{gene}_rmsf_aligned_ca_coordinate_cache.parquet'
        chosen = {}
        for artifact, frame in artifact_frames(coordinate_path, wanted):
            is_wt = artifact == report['wt_artifact']
            draws = wt_tags if is_wt else tags[artifact]
            chosen[artifact] = select(frame, draws, recovered if is_wt else None)
            # Save each selected sample and its entire candidate pool for audit.
            for seed, draw, pair in zip(SEEDS, draws, chosen[artifact]):
                selections.append({'gene': gene, 'artifact': artifact, 'seed': seed,
                                   'is_wt': is_wt, 'medoid': pair[0], 'nearest_mean': pair[1],
                                   'candidate_tags': '|'.join(draw)})
        assert set(chosen) == set(wanted)
        wc = chosen[report['wt_artifact']]
        max_d1_error = 0.0
        for number, record in enumerate(records):
            mutation = record['mutation_code']
            artifact = sources[mutation]
            mutant, columns = matrices[artifact]
            for j, seed in enumerate(SEEDS):
                a, b = wt[:, wo[j]], mutant[:, mutant_orders[artifact][j]]
                ref = reference.loc[(gene, mutation, seed)]
                assert int(ref.label) == int(record['label'])
                computed_d1 = d1(a.mean(1), b.mean(1))
                error = abs(computed_d1 - ref.d1_score)
                max_d1_error = max(max_d1_error, error)
                assert error < 1e-7, (gene, mutation, seed, error)
                # Cache KS is reused only after direct all-residue checks per gene.
                if number == 0:
                    assert abs(residue_ks(a, b) - ref.plddt_ks_residue) < 1e-12
                values = {'KS': float(ref.plddt_ks_residue), 'mean_D1': float(ref.d1_score),
                          'mean_MAE': float(np.abs(b.mean(1) - a.mean(1)).mean())}
                for k, name in enumerate(['medoid', 'nearest_mean']):
                    left = wt[:, wt_cols.index(wc[j][k])]
                    right = mutant[:, columns.index(chosen[artifact][j][k])]
                    values[name + '_D1'] = d1(left, right)
                    values[name + '_MAE'] = float(np.abs(right - left).mean())
                rows.append({'gene': gene, 'mutation': mutation, 'label': int(record['label']),
                             'seed': seed, 'n_residues': len(wt), **values})
        audits.append({'gene': gene, 'variants': len(records), 'max_mean_D1_cache_error': max_d1_error,
                       'KS_direct_cache_checks': 10, 'medoids_direct_checked': len(chosen),
                       'coordinate_path': str(coordinate_path), 'coordinate_bytes': coordinate_path.stat().st_size,
                       'coordinate_mtime_ns': coordinate_path.stat().st_mtime_ns})
        pd.DataFrame(rows).query('gene == @gene').to_csv(OUT / f'{gene}_scores.csv', index=False)
        print(f'{gene}: {len(records)} variants, 10 matched draws; {time.monotonic()-started:.1f}s', flush=True)
    pd.DataFrame(selections).to_csv(OUT / 'selections.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'reproduction_checks.csv', index=False)
    cohort.to_csv(OUT / 'cohort.csv', index=False)
    excluded.to_csv(OUT / 'exclusions.csv', index=False)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'variant_scores.csv.gz', index=False)
    return result


def uncertainty(scores, unit='variant', suffix=''):
    """Hierarchical paired bootstrap; ``unit`` is the variant resampling unit.

    unit='variant': the 10-19 procedure (class-stratified multinomial variant counts).
    unit='position': whole residue positions, strata driver-only / passenger-only /
    mixed (published unit since 2026-10-20). unit='singleton': that code with one
    variant per cluster; it must reproduce 'variant' exactly (checked in main).
    Outputs carry ``suffix`` (e.g. '_variant').
    """
    rng = np.random.default_rng(20261019)
    # Common seed-index resampling preserves pairing across methods and genes.
    seed_draws = rng.integers(0, 10, (BOOT, 10))
    seed_counts = np.stack([(seed_draws == j).sum(1) for j in range(10)], 1)
    boot_results, points, checks = [], [], []
    for gene in GENES:
        block = scores[scores.gene.eq(gene)]
        labels = block.groupby('mutation').label.first().sort_index()
        y = labels.to_numpy()
        tensors = np.stack([block.pivot(index='mutation', columns='seed', values=m).loc[labels.index, SEEDS].to_numpy()
                            for m in METHODS], -1)
        assert np.isfinite(tensors).all()
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if unit == 'variant':
            pc = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)), size=BOOT).astype(float)
            nc = rng.multinomial(len(neg), np.full(len(neg), 1 / len(neg)), size=BOOT).astype(float)
        else:
            clusters = clusters_of(labels.index, singleton=unit == 'singleton')
            w = stratified_weights(rng, clusters, strata_of(y, clusters), len(y), BOOT)
            pc, nc = w[:, pos], w[:, neg]
        norm = pc.sum(1) * nc.sum(1)
        draws = np.zeros((BOOT, len(METHODS)))
        for m, method in enumerate(METHODS):
            point = []
            for j, seed in enumerate(SEEDS):
                a, b = tensors[pos, j, m], tensors[neg, j, m]
                kernel = (a[:, None] > b).astype(float) + .5 * (a[:, None] == b)
                point.append(float(kernel.mean()))
                assert abs(point[-1] - roc_auc_score(y, tensors[:, j, m])) < 1e-12
                boot_auc = np.sum((pc @ kernel) * nc, axis=1) / norm
                # Independently check weighted AUC against sklearn, including ties.
                for repeat in range(2):
                    counts = np.zeros(len(y)); counts[pos], counts[neg] = pc[repeat], nc[repeat]
                    expected = roc_auc_score(y, tensors[:, j, m], sample_weight=counts)
                    assert abs(expected - boot_auc[repeat]) < 1e-12
                draws[:, m] += boot_auc * seed_counts[:, j] / 10
                points.append({'gene': gene, 'method': method, 'seed': seed, 'auc': point[-1]})
            checks.append({'gene': gene, 'method': method, 'sklearn_point_checks': 10,
                           'sklearn_weighted_bootstrap_checks': 20})
        boot_results.append(draws)
        print(f'{gene}: paired bootstrap complete', flush=True)
    per_seed = pd.DataFrame(points)
    macro = per_seed.groupby(['method', 'seed'], as_index=False).auc.mean().assign(gene='MACRO')
    per_seed = pd.concat([per_seed, macro], ignore_index=True)
    if suffix is not None:
        per_seed.to_csv(OUT / f'auc_per_seed{suffix}.csv', index=False)
    all_boot = np.stack(boot_results)
    all_boot = np.concatenate([all_boot, all_boot.mean(0, keepdims=True)])
    if suffix is not None:
        np.savez_compressed(OUT / f'bootstrap_auc_draws{suffix}.npz', auc=all_boot, genes=GENES + ['MACRO'], methods=METHODS)
    summary, contrasts = [], []
    for gi, gene in enumerate(GENES + ['MACRO']):
        for mi, method in enumerate(METHODS):
            point = per_seed.query('gene == @gene and method == @method').auc.mean()
            lo, hi = np.quantile(all_boot[gi, :, mi], [.025, .975])
            summary.append({'gene': gene, 'method': method, 'auc': point, 'lo': lo, 'hi': hi})
            if method != 'KS':
                delta = all_boot[gi, :, 0] - all_boot[gi, :, mi]
                lo, hi = np.quantile(delta, [.025, .975])
                ks_point = per_seed.query('gene == @gene and method == "KS"').auc.mean()
                contrasts.append({'gene': gene, 'comparison': 'KS minus ' + method,
                                  'difference': ks_point - point, 'lo': lo, 'hi': hi})
    summary, contrasts = pd.DataFrame(summary).assign(unit=unit), pd.DataFrame(contrasts).assign(unit=unit)
    if suffix is not None:
        summary.to_csv(OUT / f'auc_summary{suffix}.csv', index=False)
        contrasts.to_csv(OUT / f'paired_contrasts{suffix}.csv', index=False)
        pd.DataFrame(checks).to_csv(OUT / f'bootstrap_validation{suffix}.csv', index=False)
    print(summary.query('gene == "MACRO"').to_string(index=False), flush=True)
    return summary, contrasts


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    for path in [Path(__file__), HERE / 'check_representatives.py', ROOT / 'drift/structure_draws.py',
                 ROOT / 'drift/ensemble_shift.py', ROOT / 'drift/protein_configs.json',
                 COHORT, EXCLUSIONS]:
        sha(path)
    scores = calculate()
    with threadpool_limits(limits=2):
        # Residue positions are the published unit since 2026-10-20 (Dan); the
        # variant-level intervals are kept, suffixed _variant, for the record.
        variant = uncertainty(scores, 'variant', '_variant')
        singleton = uncertainty(scores, 'singleton', None)
        for a, b in zip(variant, singleton):
            pd.testing.assert_frame_equal(a.drop(columns='unit'), b.drop(columns='unit'), check_exact=True)
        print('singleton position clusters reproduce the variant intervals exactly', flush=True)
        uncertainty(scores, 'position', '')
    (OUT / 'provenance.json').write_text(json.dumps({
        'N': 90, 'seeds': SEEDS, 'bootstrap_replicates': BOOT, 'bootstrap_rng': 20261019,
        'bootstrap': 'class-stratified variants within each gene; same variant counts across all methods and seeds; ten seed indices resampled jointly across methods and genes; percentile 95% intervals',
        'estimand': 'mean per-seed gene AUC; macro is equal mean of six gene AUCs',
        'selection': 'within each actual production N90 draw; summed fixed-frame CA RMSD medoid or nearest coordinate mean; lexicographic tie breaking',
        'source_pools': 'all source CSV columns, including extra POLE seeds where present; exact production draw keys',
        'scope': 'all scored-chain residues in existing core_fit frame; separate MSH2/MSH6 systems; corrected single POLE WT',
        'D1_precision': 'production float32 ratio and aggregation for all D1 comparisons',
        'cached_values': 'corrected ebbbda31758d KS reused after 60 direct checks; mean D1 checked for every matched variant/draw',
        'source_sha256': HASHES, 'runtime_seconds': time.monotonic() - started,
        'limitations': 'conditional on existing generations and fitting frame; no multiple-comparison adjustment or inference for untested selection rules',
    }, indent=2) + '\n')
    print(f'Total: {time.monotonic()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
