#!/usr/bin/env python
"""Panel-e twin, as a ratio: per-residue mutant/WT pLDDT for one driver and one passenger.

WHAT IT ESTABLISHES
-------------------
The driver's perturbation is localised and BIDIRECTIONAL -- it raises predicted
confidence over some regions and lowers it over others -- while the passenger
stays near 1 throughout. That is the visual case for the absolute value in D1:
a signed mean would cancel the two directions against each other (median
within-dataset ROC-AUC 0.707 -> 0.634; POLE 0.854 -> 0.740, Supplementary
Note 1).

Panel b plots |ratio - 1| directly. D1 is the mean of that curve, so the panel
shows the quantity being averaged rather than only its result.

ORIENTATION -- READ THIS BEFORE COMPARING TO THE EXISTING FIGURE
---------------------------------------------------------------
This script plots **mutant / WT**, matching D1's definition
    D1(v) = mean_r | mean_pLDDT_mut(r) / mean_pLDDT_WT(r) - 1 | .
The pipeline's existing `<GENE>_plddt_ratio_exemplars.png` plots **WT / mutant**
-- the reciprocal (flipped deliberately in commit 96cd850c so that drivers rise
above 1). The two are not interchangeable: |x - 1| and |1/x - 1| differ, and
only the orientation here is the one D1 averages.

SAMPLING
--------
The N=90 subsample and seed are taken from ks_meanfree/canonical_analysis.py via
its own `stable_rng`, at seed_index 0, so the D1 annotated on panel b equals the
value in per_variant_seed.csv.gz for that variant and seed. Verified at runtime.

SOURCE
------
Exact mutant and WT pLDDT files from the canonical cohort (strict manifest).
POST the strict canonical cutover (2026-08-31 14:04).

    python experiments/si_figures/ratio_exemplar.py
    ... --gene POLE --driver P286R --passenger <code>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figlib import *  # noqa: F403,E402

sys.path.insert(0, str(REPO / "experiments" / "ks_meanfree"))
from canonical_analysis import apply_row_block, row_block, stable_rng  # noqa: E402
from metric_analysis.pipeline.paths import tagged  # noqa: E402
from drift import ensemble_shift  # noqa: E402
from drift.protein_config import load_continuous_configs  # noqa: E402
from drift.structure_draws import discover_wt_csv_sources  # noqa: E402

PER_VARIANT = (
    REPO / "experiments" / "ks_meanfree" / "results" / tagged("canonical")
    / "per_variant_seed.csv.gz"
)
#: seed_index 0 of the canonical ten
CANONICAL_SEED = 20260829

DRIVER_C, PASSENGER_C, WT_C = DRV, "#2e86ab", "#9a9a9a"


def sampled_pair(record, size, seed):
    """WT and mutant pLDDT matrices under the canonical subsample, residues aligned."""
    gene = record["protein"]
    cfg = load_continuous_configs((gene,))[gene]["d1"]
    if cfg.get("wt_csv_expected_sources") is not None:
        sources = discover_wt_csv_sources(
            cfg["read_root"], cfg["wt_csv_glob"],
            expected_sources=cfg["wt_csv_expected_sources"])
        if Path(record["wt_source_file"]).resolve() != Path(sources[0]).resolve():
            raise ValueError(f"{gene}: manifest WT differs from the configured reference")
    wt, mut = [ensemble_shift._read_plddt_matrix(
        record[key], None, seed=seed, key="ratio_exemplar",
        expected_residues=cfg.get("plddt_expected_residues"))
        for key in ("wt_source_file", "source_file")]
    if not wt.index.equals(mut.index):
        raise ValueError(f"{gene}: WT and mutant residue row labels differ")
    block = row_block(record)
    context = f"{record['protein']} {record['mutation_code']}"
    residues = (np.arange(block[1] - block[0], dtype=float) if block is not None
                else np.asarray(wt.index, dtype=float)) + 1.0
    wt = apply_row_block(wt.to_numpy(float), block, context=context)
    mut = apply_row_block(mut.to_numpy(float), block, context=context)
    if wt.shape[0] != mut.shape[0]:
        raise SystemExit(f"residue mismatch: {wt.shape[0]} != {mut.shape[0]}")

    code = record["mutation_code"]
    wt_order = stable_rng(seed, record["protein"], code, "wt").permutation(wt.shape[1])
    mut_order = stable_rng(seed, record["protein"], code, "mutant").permutation(mut.shape[1])
    n = min(size, wt.shape[1], mut.shape[1])

    w = wt[:, wt_order[:n]]
    m = mut[:, mut_order[:n]]
    valid = np.isfinite(w).all(axis=1) & np.isfinite(m).all(axis=1)
    resid = residues[valid]   # 1-based, subunit-local when a row block is present
    return resid, w[valid], m[valid], int(n)


def ratio_profile(w, m):
    """Per-residue mean-pLDDT ratio, and the |ratio - 1| that D1 averages."""
    wt_mean, mut_mean = w.mean(axis=1), m.mean(axis=1)
    ok = np.abs(wt_mean) > 0
    ratio = np.full_like(wt_mean, np.nan)
    ratio[ok] = mut_mean[ok] / wt_mean[ok]
    return ratio, np.abs(ratio - 1.0), wt_mean


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene", default="PTEN")
    ap.add_argument("--driver", default="A121E")
    ap.add_argument("--passenger", default="A79T")
    ap.add_argument("--size", type=int, default=90)
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED)
    args = ap.parse_args(argv)
    gene = to_dir_key(args.gene)

    coh = cohort(gene).set_index("mutation_code")
    for code in (args.driver, args.passenger):
        if code not in coh.index:
            raise SystemExit(f"{code} is not in the {gene} canonical cohort")

    series, wt_ref = {}, None
    for code, role in ((args.driver, "driver"), (args.passenger, "passenger")):
        rec = coh.loc[code].to_dict() | {"mutation_code": code, "protein": gene}
        resid, w, m, n = sampled_pair(rec, args.size, args.seed)
        ratio, absdev, wt_mean = ratio_profile(w, m)
        d1 = float(np.nanmean(absdev))
        series[role] = {"code": code, "resid": resid, "ratio": ratio,
                        "absdev": absdev, "d1": d1, "n": n,
                        "pos": int("".join(c for c in code[1:-1] if c.isdigit()))}
        wt_ref = (resid, wt_mean)
        print(f"[ratio_exemplar] {role:9s} {code}  N={n}  D1={d1:.6f}")

    # The point of pinning the canonical RNG: this must reproduce the table.
    if PER_VARIANT.is_file():
        pv = pd.read_csv(PER_VARIANT)
        for role, s in series.items():
            row = pv[(pv.protein == gene) & (pv.mutation_code == s["code"])
                     & (pv.requested_size == args.size) & (pv.seed == args.seed)]
            if len(row) == 1:
                ref = float(row.d1.iloc[0])
                agree = abs(ref - s["d1"]) < 1e-9
                print(f"[ratio_exemplar] {s['code']} D1 vs canonical table: "
                      f"{s['d1']:.6f} vs {ref:.6f}  {'OK' if agree else 'MISMATCH'}")
                if not agree:
                    raise SystemExit("D1 does not reproduce the canonical table")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W2, W2 * 0.52), sharex=True,
                                   gridspec_kw={"height_ratios": [1.25, 1]})

    # WT confidence behind the ratio: the wild regions of panel a are exactly
    # where WT pLDDT collapses, and a reader should be able to see that.
    bg = ax1.twinx()
    bg.fill_between(wt_ref[0], wt_ref[1], 0, color=WT_C, alpha=0.12, lw=0, zorder=0)
    bg.set_ylim(0, 100)
    bg.set_ylabel("WT mean pLDDT", color="#999")
    bg.tick_params(axis="y", colors="#999")
    bg.spines["right"].set_visible(True)
    bg.spines["right"].set_color("#ccc")

    for role, colour in (("driver", DRIVER_C), ("passenger", PASSENGER_C)):
        s = series[role]
        ax1.plot(s["resid"], s["ratio"], color=colour, lw=0.8,
                 label=f"{role.capitalize()} {s['code']}", zorder=3)
        ax2.plot(s["resid"], s["absdev"], color=colour, lw=0.8,
                 label=f"{role.capitalize()} {s['code']}   D1 = {s['d1']:.4f}", zorder=3)
        for ax in (ax1, ax2):
            ax.axvline(s["pos"], color=colour, ls=":", lw=0.6, alpha=0.8, zorder=1)
        record("ratio_exemplar", f"{gene}_{s['code']}_D1", s["d1"], f"N={s['n']}")
        record("ratio_exemplar", f"{gene}_{s['code']}_max_absdev",
               float(np.nanmax(s["absdev"])))
        record("ratio_exemplar", f"{gene}_{s['code']}_frac_above_1",
               float(np.nanmean(s["ratio"] > 1)), "fraction of residues raised vs WT")

    ax1.axhline(1.0, color="#444", ls="--", lw=0.7, zorder=2)
    ax1.set_ylabel("mutant / WT\nmean pLDDT")
    ax1.legend(loc="lower left", ncol=2)
    ax1.set_zorder(bg.get_zorder() + 1)
    ax1.patch.set_visible(False)

    ax2.axhline(0.0, color="#444", ls="--", lw=0.7, zorder=2)
    ax2.set_ylabel("| mutant/WT $-$ 1 |\n(the quantity D1 averages)")
    ax2.set_xlabel(f"{pretty(gene)} residue number")
    ax2.legend(loc="upper left", ncol=2)

    ax1.set_title(f"{pretty(gene)} per-residue pLDDT ratio — "
                  f"{args.driver} (driver) vs {args.passenger} (passenger), N={args.size}")
    fig.align_ylabels([ax1, ax2])
    save(fig, f"{gene}_ratio_exemplar_{args.driver}_vs_{args.passenger}")

    write_values(f"ratio_exemplar_values_{gene}.csv")
    provenance(figure=f"ratio_exemplar_{gene}", size=args.size, seed=args.seed,
               driver=args.driver, passenger=args.passenger,
               orientation="mutant/WT (matches D1; the pipeline figure is WT/mutant)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
