"""AF3's own top-ranked prediction against the pLDDT KS-score (Supplementary Note S8 redo).

One selection rule and one metric, fixed before looking at any result (Dan, 2026-10-21):

- selection: within each production N=90 draw, the WT prediction and the mutant prediction
  with the highest AF3 ranking_score (AF3's "best structure"), chosen independently per side;
- metric: mean absolute per-residue pLDDT difference (MAE) between the two selected
  predictions, over the same scored residues as the KS-score.

The KS-score is recomputed directly on the same draws for every variant and seed and must
equal the published POLE 25-seed cache (fingerprint 49028c16cbb3). Uncertainty is the
Note S8 estimator (n90_comparison.uncertainty): paired hierarchical bootstrap over residue
positions, 2,000 replicates, ten subsampling seeds resampled jointly.

Run from the repo root:
    PYTHONPATH="." python analyses/seeds/af3_top1_vs_ks/af3_top1_vs_ks.py
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'analyses/coordinates/s8_representatives'))
import n90_comparison as n90  # noqa: E402
from drift.protein_config import load_protein_configs, protein_config_fingerprint  # noqa: E402
from metric_analysis.pipeline.library.descriptor_metrics import residue_ks  # noqa: E402
from metric_analysis.pipeline.library.canonical_cohort import _existing_repo_path  # noqa: E402

OUT = HERE / 'results'
# The published N=90 draws: POLE 25-seed config, seeds 42-51 (run_all.sh cache); read, never
# written. Cohort and comparison tables come from the 25-seed runner's side stages.
RUN = ROOT / 'analyses/annotation/stages'
CACHE = ROOT / 'drift/cache'
FP = protein_config_fingerprint()  # published: 49028c16cbb3, which also held FOXA1
COHORT = ROOT / 'analyses/annotation/stages/side/annotation_cohort_25seed.csv'
PUBLISHED_S8 = ROOT / 'analyses/annotation/stages/side/s8_n90_results/auc_summary.csv'
MAIN_TEXT_KS = ROOT / 'analyses/annotation/stages/side/ks_auc_ci/ks_auc_ci_pole25_cache.csv'
RANKING = ROOT / 'analyses/seeds/seed_structure/plateau_20261018/af3_ranking'
METHODS = ['KS', 'af3_top1_MAE']
SEEDS = n90.SEEDS
HASHES = n90.HASHES


def sha(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    HASHES[str(path)] = digest
    return digest


def ranking(gene, mutation, source):
    """Recorded AF3 ranking_score per seed/sample tag, checked against its pLDDT source."""
    path = RANKING / gene / f'{mutation}.json'
    record = json.loads(path.read_text())
    sha(path)
    assert record['status'] == 'complete' and record['precision'] == 'ranking_scores.csv', path
    source = Path(source)
    if source.parent.name == 'POLE_25seed':
        # Cell-for-cell copies of seeds 1-25 of the 160-prediction originals, whose scores
        # were recorded; the original must be the file the scores were matched to.
        original = ROOT / 'PLDDTs/POLE' / source.name
        assert sha(original) == record['source_sha256'], original
    else:
        assert sha(source) == record['source_sha256'], source
    return record['scores']


def top1(draw, scores):
    """Highest ranking_score in the draw; ties go to the first tag in lexicographic order."""
    candidates = sorted(draw)
    values = np.array([scores[t] for t in candidates])
    assert np.isfinite(values).all()
    return candidates[int(np.argmax(values))], int((values == values.max()).sum())


def calculate():
    cohort = pd.read_csv(COHORT)
    for column in ('source_file', 'wt_source_file'):
        cohort[column] = [str(_existing_repo_path(v, protein=p))
                          for v, p in zip(cohort[column], cohort.parent_dataset)]
    assert len(cohort) == 1798 and cohort.label.sum() == 612
    configs = load_protein_configs()
    reference = []
    for seed in SEEDS:
        path = CACHE / f'labeled_size90_seed{seed}_proteins{FP}_esfeat7.csv'
        sha(path)
        reference.append(pd.read_csv(path).assign(seed=seed))
    reference = pd.concat(reference).set_index(['gene', 'mutation', 'seed'])
    rows, selections, audits = [], [], []
    for gene in n90.GENES:
        started = time.monotonic()
        records = cohort[cohort.protein.eq(gene)].to_dict('records')
        cfg = configs[gene]['structures']
        source = records[0]['wt_source_file']
        assert {r['wt_source_file'] for r in records} == {source}
        wt, wt_cols = n90.load_matrix(records[0], wt=True)
        wo, wt_tags = n90.orders(wt_cols, cfg, source, wt=True)
        wt_scores = ranking(gene, 'WT', source)
        wt_pick = [top1(draw, wt_scores) for draw in wt_tags]
        for seed, draw, (tag, ties) in zip(SEEDS, wt_tags, wt_pick):
            selections.append({'gene': gene, 'mutation': 'WT', 'seed': seed, 'top1': tag,
                               'ranking_score': wt_scores[tag], 'tied_at_top': ties})
        max_ks_error = 0.0
        for record in records:
            mutation = record['mutation_code']
            mutant, columns = n90.load_matrix(record)
            assert mutant.shape[0] == wt.shape[0]
            mo, mtags = n90.orders(columns, cfg, record['source_file'])
            scores = ranking(gene, mutation, record['source_file'])
            for j, seed in enumerate(SEEDS):
                a, b = wt[:, wo[j]], mutant[:, mo[j]]
                ref = reference.loc[(gene, mutation, seed)]
                assert int(ref.label) == int(record['label'])
                error = abs(residue_ks(a, b) - ref.plddt_ks_residue)
                max_ks_error = max(max_ks_error, error)
                assert error < 1e-12, (gene, mutation, seed, error)
                # The cached value, not the recomputation: ~1e-16 differences flip exact
                # ties (POLE has KS-scores of 1.0) and would move AUCs in the 6th decimal.
                ks = float(ref.plddt_ks_residue)
                tag, ties = top1(mtags[j], scores)
                left = wt[:, wt_cols.index(wt_pick[j][0])]
                right = mutant[:, columns.index(tag)]
                selections.append({'gene': gene, 'mutation': mutation, 'seed': seed, 'top1': tag,
                                   'ranking_score': scores[tag], 'tied_at_top': ties})
                rows.append({'gene': gene, 'mutation': mutation, 'label': int(record['label']),
                             'seed': seed, 'n_residues': len(wt), 'KS': ks,
                             'af3_top1_MAE': float(np.abs(right - left).mean())})
        audits.append({'gene': gene, 'variants': len(records), 'ks_recomputed': len(records) * 10,
                       'max_ks_cache_error': max_ks_error})
        print(f'{gene}: {len(records)} variants, KS recomputed and matched '
              f'(max error {max_ks_error:.1e}); {time.monotonic() - started:.1f}s', flush=True)
    selections = pd.DataFrame(selections)
    selections.to_csv(OUT / 'selections.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'reproduction_checks.csv', index=False)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'variant_scores.csv.gz', index=False)
    return result, selections


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    for path in [Path(__file__), ROOT / 'analyses/coordinates/s8_representatives/n90_comparison.py',
                 ROOT / 'drift/structure_draws.py', ROOT / 'drift/protein_configs.json', COHORT]:
        sha(path)
    scores, selections = calculate()
    # The Note S8 estimator, unchanged, on the two methods.
    n90.METHODS, n90.OUT = METHODS, OUT
    with threadpool_limits(limits=2):
        variant = n90.uncertainty(scores, 'variant', '_variant')
        singleton = n90.uncertainty(scores, 'singleton', None)
        for a, b in zip(variant, singleton):
            pd.testing.assert_frame_equal(a.drop(columns='unit'), b.drop(columns='unit'), check_exact=True)
        print('singleton position clusters reproduce the variant intervals exactly', flush=True)
        summary, _ = n90.uncertainty(scores, 'position', '')
    # KS must reproduce the published numbers. Note S8 used 1,797 variants (MSH2 G162R has no
    # coordinates), so it is the reference for the other five genes; MSH2 is checked against
    # the main-text per-gene AUC on all 1,798 (its interval comes from a different bootstrap).
    mine = summary.query('method == "KS"').set_index('gene')
    published = pd.read_csv(PUBLISHED_S8).query('method == "KS"').set_index('gene')
    sha(PUBLISHED_S8)
    five = [g for g in n90.GENES if g != 'MSH2']
    assert np.allclose(mine.loc[five, 'auc'], published.loc[five, 'auc'], atol=1e-12), 'KS AUC vs S8'
    # The extra MSH2 variant changes MSH2's clusters and so the random stream for every gene
    # drawn after it; intervals can match only for the genes drawn before MSH2.
    before = n90.GENES[:n90.GENES.index('MSH2')]
    for column in ('lo', 'hi'):
        assert np.allclose(mine.loc[before, column], published.loc[before, column], atol=1e-12), f'KS {column} vs S8'
    main_text = pd.read_csv(MAIN_TEXT_KS).set_index('gene')
    sha(MAIN_TEXT_KS)
    assert np.allclose(mine.loc[n90.GENES, 'auc'], main_text.loc[n90.GENES, 'auc'], atol=1e-12), 'KS AUC vs main text'
    print('KS reproduces Note S8 (five genes, with intervals) and the main-text AUCs (six genes)', flush=True)
    (OUT / 'provenance.json').write_text(json.dumps({
        'question': 'Does AF3\'s own best structure per sequence match the KS-score on the same predictions?',
        'N': 90, 'seeds': len(SEEDS), 'bootstrap_replicates': n90.BOOT, 'bootstrap_rng': 20261019,
        'selection': 'max AF3 ranking_score within each production N=90 draw, WT and mutant independently; '
                     'WT pick shared by all variants of a gene; ties to the first tag in lexicographic order',
        'metric': 'mean absolute per-residue pLDDT difference between the two picks, scored residues as KS',
        'draws': f'POLE 25-seed production N=90 draws, cache fingerprint {FP}; KS recomputed and matched for every variant and seed',
        'cohort': '1,798 labelled variants (612 drivers / 1,186 passengers); no coordinates needed',
        'ties_at_top': int((selections.tied_at_top > 1).sum()),
        'source_sha256': HASHES, 'runtime_seconds': time.monotonic() - started,
    }, indent=2) + '\n')
    print(f'Total: {time.monotonic() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
