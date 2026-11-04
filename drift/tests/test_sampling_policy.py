"""Regressions for the structure-subsampling policy.

The single most important test here is
`TestSamplingPolicy.test_default_policy_matches_pre_policy_sampler`: the default
nested/random policy must keep selecting exactly what the pre-policy code
selected, or every published KS number silently changes.
"""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from drift import structure_draws as sdraws


def _legacy_sample(items, max_items, *, seed, key):
    """Verbatim copy of the sampler as it stood before SamplingPolicy existed."""
    items = sorted(str(x) for x in items)
    if len(items) <= 1:
        ranked = items
    else:
        payload = f"{seed}|{key}".encode("utf-8")
        sub = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)
        rng = np.random.default_rng(sub)
        ranked = [items[i] for i in rng.permutation(len(items)).tolist()]
    if max_items is None or len(ranked) <= max_items:
        return ranked
    return ranked[:max_items]


ITEMS = [f"seed-{i}_sample-{j}" for i in range(8) for j in range(5)]


SIZES = [1, 5, 10, 20, 40]


class TestSamplingPolicy(unittest.TestCase):
    def test_default_policy_matches_pre_policy_sampler(self):
        for seed in (42, 51, 2026):
            for size in SIZES:
                key = f"mut::{seed}"
                self.assertEqual(
                    sdraws.select_structure_columns(ITEMS, size, seed=seed, key=key),
                    _legacy_sample(ITEMS, size, seed=seed, key=key),
                    msg=f"default policy diverged at seed={seed} size={size}",
                )

    def test_nested_sizes_are_prefixes(self):
        order = sdraws.rank_items_deterministically(ITEMS, seed=42, key="k")
        for size in SIZES:
            self.assertEqual(
                sdraws.sample_items_nested(ITEMS, size, seed=42, key="k"), order[:size]
            )

    def test_independent_sizes_are_not_prefixes(self):
        policy = sdraws.SamplingPolicy(mode="independent")
        draws = {
            size: sdraws.sample_items_nested(ITEMS, size, seed=42, key="k", policy=policy)
            for size in SIZES
        }
        for size, drawn in draws.items():
            self.assertEqual(len(drawn), size)
            self.assertEqual(len(set(drawn)), size, "draws must be without replacement")
        # At least one size must break the prefix relation, otherwise the mode
        # is silently behaving like nested.
        self.assertTrue(
            any(draws[b][: len(draws[a])] != draws[a]
                for a, b in zip(SIZES[:-1], SIZES[1:]))
        )

    def test_independent_mode_still_uniform_at_each_size(self):
        """Each size is a uniform subset; only the pairing across sizes changes."""
        policy = sdraws.SamplingPolicy(mode="independent")
        counts = {}
        for seed in range(200):
            picked = sdraws.sample_items_nested(ITEMS, 1, seed=seed, key="k", policy=policy)[0]
            counts[picked] = counts.get(picked, 0) + 1
        # With 40 items and 200 draws a uniform sampler touches many of them;
        # a degenerate sampler would concentrate on one or two.
        self.assertGreater(len(counts), 15)

    def test_top_confidence_ranks_by_score_then_name(self):
        scores = {c: float(i % 4) for i, c in enumerate(ITEMS)}
        policy = sdraws.SamplingPolicy(selection="top_confidence")
        picked = sdraws.select_structure_columns(
            ITEMS, 5, seed=42, key="k", policy=policy, scores=scores
        )
        self.assertTrue(all(scores[c] == 3.0 for c in picked))
        self.assertEqual(picked, sorted(picked), "ties must break deterministically")

    def test_top_confidence_is_pinned_to_wild_type_order(self):
        """The mutant must reuse WT's chosen structures, not re-rank its own."""
        scores = {c: float(i) for i, c in enumerate(ITEMS)}
        policy = sdraws.SamplingPolicy(selection="top_confidence")
        wt_order = sdraws.select_structure_columns(
            ITEMS, 3, seed=42, key="wt", policy=policy, scores=scores
        )
        mutant_scores = {c: -float(i) for i, c in enumerate(ITEMS)}  # opposite ranking
        picked = sdraws.select_structure_columns(
            ITEMS, 3, seed=42, key="mut", policy=policy,
            scores=mutant_scores, preferred_order=wt_order,
        )
        self.assertEqual(picked, wt_order)

    def test_wt_pinning_tops_up_when_mutant_lacks_columns(self):
        scores = {c: float(i) for i, c in enumerate(ITEMS)}
        policy = sdraws.SamplingPolicy(selection="top_confidence")
        available = ITEMS[5:]
        picked = sdraws.select_structure_columns(
            available, 4, seed=42, key="mut", policy=policy,
            scores={c: scores[c] for c in available},
            preferred_order=ITEMS[:3],  # none of these exist in the mutant
        )
        self.assertEqual(len(picked), 4)
        self.assertTrue(set(picked).issubset(available))

    def test_repeated_preferred_names_select_each_column_once(self):
        """A gene whose WT spans several CSVs repeats names in `preferred_order`.

        POLE has three WT files, two of them the same seed block, so the WT
        selection legitimately lists `seed-X_sample-Y` more than once. Selecting
        the column twice would double-weight it in the mutant mean and overstate
        the number of structures averaged.
        """
        policy = sdraws.SamplingPolicy(selection="random_pinned")
        repeated = [ITEMS[0], ITEMS[1], ITEMS[0], ITEMS[2], ITEMS[1]]
        picked = sdraws.select_structure_columns(
            ITEMS, 4, seed=42, key="mut", policy=policy, preferred_order=repeated
        )
        self.assertEqual(len(picked), len(set(picked)))
        self.assertEqual(picked[:3], [ITEMS[0], ITEMS[1], ITEMS[2]])
        self.assertEqual(len(picked), 4)

    def test_duplicate_preferred_names_do_not_inflate_structure_count(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "v_plddt_values.csv"
            frame = pd.DataFrame({f"s{i}": np.full(4, 60.0 + i) for i in range(6)})
            frame.insert(0, "residue", range(4))
            frame.to_csv(path, index=False)

            chosen = []
            _, n = sdraws.extract_plddt_from_csv_mean(
                path, 3, seed=42, key="k",
                policy=sdraws.SamplingPolicy(selection="random_pinned"),
                preferred_order=["s0", "s1", "s0", "s1", "s2"],
                selection_out=chosen,
            )
            self.assertEqual(n, 3)
            self.assertEqual(chosen, ["s0", "s1", "s2"])

    def test_unscored_columns_sort_last_but_are_not_dropped(self):
        policy = sdraws.SamplingPolicy(selection="top_confidence")
        scores = {ITEMS[0]: 10.0, ITEMS[1]: float("nan")}
        picked = sdraws.select_structure_columns(
            ITEMS, len(ITEMS), seed=42, key="k", policy=policy, scores=scores
        )
        self.assertEqual(len(picked), len(ITEMS))
        self.assertEqual(picked[0], ITEMS[0])

    def test_random_pinned_keeps_the_random_order_but_couples_wt_and_mutant(self):
        """The control that separates confidence selection from WT/mutant pairing.

        `top_confidence` changes two things at once relative to the default: it
        ranks by confidence *and* it makes the mutant reuse WT's structures.
        `random_pinned` changes only the second, so the pair of arms attributes
        any gain to the right cause.
        """
        policy = sdraws.SamplingPolicy(selection="random_pinned")
        wt = sdraws.select_structure_columns(ITEMS, 5, seed=42, key="wt", policy=policy)
        # The WT side is drawn exactly as plain random does.
        self.assertEqual(wt, sdraws.select_structure_columns(ITEMS, 5, seed=42, key="wt"))
        # The mutant reuses WT identities despite a different key.
        pinned = sdraws.select_structure_columns(
            ITEMS, 5, seed=42, key="mut", policy=policy, preferred_order=wt
        )
        self.assertEqual(pinned, wt)
        # ...whereas the default draws the mutant independently.
        self.assertNotEqual(
            sdraws.select_structure_columns(ITEMS, 5, seed=42, key="mut"), wt
        )

    def test_random_pinned_does_not_consult_confidence(self):
        policy = sdraws.SamplingPolicy(selection="random_pinned")
        self.assertFalse(policy.uses_confidence)
        self.assertTrue(policy.pins_to_wild_type)
        scores = {c: float(i) for i, c in enumerate(ITEMS)}
        with_scores = sdraws.select_structure_columns(
            ITEMS, 4, seed=42, key="k", policy=policy, scores=scores
        )
        without = sdraws.select_structure_columns(ITEMS, 4, seed=42, key="k", policy=policy)
        self.assertEqual(with_scores, without)

    def test_invalid_policies_rejected(self):
        for kwargs in ({"mode": "ranked"}, {"selection": "best"}, {"mode": ""}):
            with self.assertRaises(ValueError):
                sdraws.SamplingPolicy(**kwargs)

    def test_tag_is_empty_for_default_so_caches_stay_valid(self):
        self.assertEqual(sdraws.SamplingPolicy().tag(), "")
        self.assertEqual(sdraws.SamplingPolicy(mode="independent").tag(), "independent-random")
        self.assertEqual(
            sdraws.SamplingPolicy.coerce("nested-top_confidence"),
            sdraws.SamplingPolicy(selection="top_confidence"),
        )
        self.assertEqual(sdraws.SamplingPolicy.coerce(None), sdraws.SamplingPolicy())

    def test_csv_reader_honours_policy_and_reports_selection(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "v_plddt_values.csv"
            frame = pd.DataFrame(
                {f"s{i}": np.full(6, 50.0 + i) for i in range(10)}
            )
            frame.insert(0, "residue", range(6))
            frame.to_csv(path, index=False)

            chosen = []
            series, n = sdraws.extract_plddt_from_csv_mean(
                path, 3, seed=42, key="k",
                policy=sdraws.SamplingPolicy(selection="top_confidence"),
                selection_out=chosen,
            )
            self.assertEqual(n, 3)
            self.assertEqual(chosen, ["s9", "s8", "s7"])
            self.assertAlmostEqual(float(series.iloc[0]), 58.0)


if __name__ == "__main__":
    unittest.main()
