"""Synthetic checks for metric_analysis.pipeline.library.positional_ks.

Runs under pytest, or directly (no env here has pytest):
    PYTHONPATH=. python metric_analysis/tests/test_positional_ks.py
"""
import numpy as np
import pandas as pd

from metric_analysis.pipeline.library.positional_ks import positional_ks, positional_ks_per_residue, reference_centre

R, N = 30, 60  # residues, predictions per ensemble


def ensemble(rng, centres, sd=0.2, n=N, chain="A", offset=0):
    """Long-format frame: n predictions of R residues scattered around `centres`."""
    rows = []
    for s in range(n):
        xyz = centres + rng.normal(0, sd, centres.shape)
        for r in range(len(centres)):
            rows.append((s + offset, chain, r + 1, "", *xyz[r]))
    return pd.DataFrame(rows, columns=["structure_index", "chain", "resid", "icode",
                                       "x_aligned", "y_aligned", "z_aligned"])


def rotate(frame, centre_xyz, angle):
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    xyz = frame[["x_aligned", "y_aligned", "z_aligned"]].to_numpy() - centre_xyz
    out = frame.copy()
    out[["x_aligned", "y_aligned", "z_aligned"]] = xyz @ rot.T + centre_xyz
    return out


def base():
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 10, (R, 3))
    return rng, centres


def test_identical_is_zero():
    rng, c = base()
    wt = ensemble(rng, c)
    centre = reference_centre(wt)
    s = positional_ks(wt, wt, centre)
    assert s["distance"] == 0 and s["axis"] == 0 and s["n_residues"] == R


def test_wt_vs_wt_is_small():
    rng, c = base()
    pool = ensemble(rng, c, n=2 * N)
    a, b = pool[pool.structure_index < N], pool[pool.structure_index >= N]
    s = positional_ks(a, b, reference_centre(pool))
    assert s["distance"] < 0.2 and s["axis"] < 0.2, s


def test_local_shift_detected_only_where_it_happens():
    rng, c = base()
    wt = ensemble(rng, c)
    shifted = c.copy(); shifted[5] += [2.0, 0, 0]            # move residue 6 by 2 A
    mut = ensemble(rng, shifted)
    per = positional_ks_per_residue(wt, mut, reference_centre(wt))
    assert per["distance"].iloc[5] > 0.9 and per["axis"].iloc[5] > 0.3, per.iloc[5]
    others = per.drop(per.index[5])
    assert others["distance"].mean() < 0.25, others["distance"].mean()


def test_spread_change_detected_by_distance():
    rng, c = base()
    wt = ensemble(rng, c, sd=0.2)
    mut = ensemble(rng, c, sd=0.6)                            # same centre, 3x spread
    s = positional_ks(wt, mut, reference_centre(wt))
    assert s["distance"] > 0.7, s


def test_distance_is_rotation_invariant_axis_is_not():
    rng, c = base()
    wt = ensemble(rng, c)
    shifted = c.copy(); shifted[:10] += [1.0, 0.5, 0]
    mut = ensemble(rng, shifted)
    centre = reference_centre(wt)
    s0 = positional_ks(wt, mut, centre)
    # rotate both ensembles about one residue-independent point: per-residue distances
    # to their (equally rotated) centres are unchanged; the per-axis values are not.
    pivot = c.mean(axis=0)
    wt_r, mut_r = rotate(wt, pivot, .7), rotate(mut, pivot, .7)
    s1 = positional_ks(wt_r, mut_r, reference_centre(wt_r))
    assert abs(s0["distance"] - s1["distance"]) < 1e-9, (s0, s1)
    assert abs(s0["axis"] - s1["axis"]) > 1e-3, (s0, s1)


def test_missing_residues_and_chain_filter():
    rng, c = base()
    wt = ensemble(rng, c)
    mut = ensemble(rng, c)
    mut = mut[~((mut.resid == 3) & (mut.structure_index < 20))]   # residue 3 absent in 20 predictions
    other = ensemble(rng, c + 50, chain="B")                         # a partner chain, far away
    centre = reference_centre(pd.concat([wt, other]), chains=["A"])
    s = positional_ks(pd.concat([wt, other]), pd.concat([mut, other]), centre, chains=["A"])
    assert s["n_residues"] == R and np.isfinite(s["distance"]), s


def test_distance_only_mode_matches_and_skips_axis():
    rng, c = base()
    wt = ensemble(rng, c)
    shifted = c.copy(); shifted[:10] += [1.0, 0.5, 0]
    mut = ensemble(rng, shifted)
    centre = reference_centre(wt)
    both = positional_ks(wt, mut, centre)
    only = positional_ks(wt, mut, centre, modes=("distance",))
    assert "axis" not in only and only["distance"] == both["distance"], (both, only)
    assert only["n_residues"] == both["n_residues"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
