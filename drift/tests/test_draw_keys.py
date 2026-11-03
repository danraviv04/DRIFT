"""Structure draws must not depend on where the checkout lives.

The draw keys name the source file. They used to embed the absolute checkout path, so
every clone and worktree drew different structures and silently rebuilt the DRIFT caches
with a different sample under the same filename. See
experiments/metrics_consolidation/CACHE_DRAW_PROVENANCE.md.
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from drift import structure_draws as d1_runtime
from drift import core, run


class TestSourceDrawKey(unittest.TestCase):
    def test_checkout_root_is_the_repo_root(self):
        # draw keys are relative to it; the module's own folder would change every draw
        self.assertEqual(Path(d1_runtime.CHECKOUT_ROOT), Path(__file__).resolve().parents[2])

    def test_path_in_checkout_uses_the_legacy_root(self):
        src = Path(d1_runtime.CHECKOUT_ROOT) / "PLDDTs" / "KRAS" / "KRAS_G12D_plddt_values.csv"
        self.assertEqual(
            d1_runtime.source_draw_key("mut", src),
            f"mut::{d1_runtime.LEGACY_KEY_ROOT}/PLDDTs/KRAS/KRAS_G12D_plddt_values.csv",
        )

    def test_path_outside_checkout_is_unchanged(self):
        src = "/Volumes/Elements/PTEN/PTEN_WT_data.tar.gz"
        self.assertEqual(d1_runtime.source_draw_key("wt", src), f"wt::{src}")

    def test_sibling_directory_sharing_the_prefix_is_not_rewritten(self):
        src = d1_runtime.CHECKOUT_ROOT + "_copy/PLDDTs/x.csv"
        self.assertEqual(d1_runtime.source_draw_key("mut", src), f"mut::{src}")


SIZE, SEED = 5, 42
SAVED = run.CACHE_DIR / (
    f"labeled_size{SIZE}_seed{SEED}_ratio_of_means_"
    f"proteins{run.protein_config_fingerprint()}_esfeat{run.ES_CACHE_VERSION}.csv"
)


@unittest.skipUnless(
    SAVED.exists(),
    f"needs {SAVED.name} (gitignored cache) -- build it with drift/run.py",
)
class TestSavedCacheReproduces(unittest.TestCase):
    """Rebuilding two genes from this checkout gives the saved cache exactly."""

    # POLE exercises the 25-seed replacements (mut_csv_replacements); MSH6 a subunit block.
    # (KRAS, here until 2026-10-13, is continuous-only and has no binary config.)
    GENES = ("POLE", "MSH6")

    def test_d1_and_ensemble_shift_match_the_saved_cache(self):
        cfgs = {g: core.D1_RUNTIME_CONFIGS[g] for g in self.GENES}
        policy = d1_runtime.SamplingPolicy()
        d1_df, _ = d1_runtime.build_d1_table(
            cfgs, max_structures_per_variant=SIZE, random_seed=SEED, policy=policy)
        feat = run.build_ensemble_shift_features(
            cfgs, max_structures_per_variant=SIZE, seed=SEED, policy=policy)
        rebuilt = (d1_df[["gene", "mutation", "d1_score"]]
                   .drop_duplicates(["gene", "mutation"])
                   .merge(feat, on=["gene", "mutation"], how="left"))

        cols = ["d1_score", *run.ES_FEATURE_COLUMNS]
        saved = pd.read_csv(SAVED, usecols=["gene", "mutation", *cols])
        saved = saved[saved["gene"].isin(self.GENES)]
        self.assertGreater(len(saved), 0)
        m = saved.merge(rebuilt, on=["gene", "mutation"], suffixes=("_saved", "_rebuilt"))
        self.assertEqual(len(m), len(saved), "saved rows missing from the rebuild")
        for c in cols:
            a = m[f"{c}_saved"].to_numpy(float)
            b = m[f"{c}_rebuilt"].to_numpy(float)
            same = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=0, atol=1e-9)
            self.assertTrue(same.all(), f"{c}: {int((~same).sum())} of {len(m)} rows differ")


if __name__ == "__main__":
    unittest.main()
