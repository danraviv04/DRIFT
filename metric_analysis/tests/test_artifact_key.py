"""WT resolution must not depend on '.tar.gz' or a trailing '_data'.

    PYTHONPATH=. python metric_analysis/tests/test_artifact_key.py
"""
from pathlib import Path
from types import SimpleNamespace

from metric_analysis.pipeline.labels import artifact_key
from metric_analysis.pipeline.library.structure_metrics import _resolve_wt_artifact_path
from metric_analysis.pipeline.orchestration.run_ensemble_sweep_parallel import resolve_wt_artifact


def test_key():
    same = ["P53_WT_with_ZN_and_DNA", "P53_WT_with_ZN_and_DNA_data", "P53_WT_with_ZN_and_DNA.tar.gz",
            "P53_WT_with_ZN_and_DNA_data.tar.gz", "/Volumes/Elements/TP53/P53_WT_with_ZN_and_DNA_data.tar.gz",
            "p53_wt_with_zn_and_dna_DATA"]
    assert len({artifact_key(s) for s in same}) == 1
    # different ensembles stay different
    assert artifact_key("P53_WT_with_ZN_and_DNA") != artifact_key("P53_WT_with_ZN_and_DNA_again")
    assert artifact_key("POLE_268-471_WT") != artifact_key("POLE_WT")
    # only ONE trailing _data, and not a _data in the middle
    assert artifact_key("X_data_data") == "X_DATA"
    assert artifact_key("X_data_extra") == "X_DATA_EXTRA"


def test_sweep_resolver_all_naming_cases():
    cases = [  # (config, names present in the cache, expected)
        ("PTEN_WT", ["PTEN_WT_data", "PTEN_A121E_data"], "PTEN_WT_data"),
        ("P53_WT_with_ZN_and_DNA_data", ["P53_WT_with_ZN_and_DNA", "P53_WT_with_ZN_and_DNA_again",
                                         "P53_A119D_with_ZN_and_DNA"], "P53_WT_with_ZN_and_DNA"),
        ("MutS_alpha_MSH6_GT_WT_data", ["MutS_alpha_MSH6_GT_WT_data", "MutS_alpha_MSH6_GT_A1055P_data"],
         "MutS_alpha_MSH6_GT_WT_data"),
        ("POLE_268-471_WT", ["POLE_268-471_WT", "POLE_WT", "POLE_again_WT"], "POLE_268-471_WT"),
    ]
    for configured, present, expected in cases:
        got = resolve_wt_artifact({"wt_artifact": configured, "prefix": "X"}, present)
        assert got == expected, (configured, got)
    assert resolve_wt_artifact({"wt_artifact": "NOPE_WT", "prefix": "X"}, ["PTEN_WT"]) is None


def test_archive_resolver():
    arcs = [Path(f"/x/{n}.tar.gz") for n in
            ["P53_WT_with_ZN_and_DNA_data", "P53_WT_with_ZN_and_DNA_again", "P53_A119D_with_ZN_and_DNA_data"]]
    for configured in ("P53_WT_with_ZN_and_DNA_data", "P53_WT_with_ZN_and_DNA"):
        cfg = SimpleNamespace(wt_artifact=configured, mut_name="TP53")
        got = _resolve_wt_artifact_path(cfg, arcs, "WT")
        assert got[0].name == "P53_WT_with_ZN_and_DNA_data.tar.gz", (configured, got)


if __name__ == "__main__":
    test_key(); test_sweep_resolver_all_naming_cases(); test_archive_resolver()
    print("artifact_key: all tests pass")
