"""Figure 1e-h re-drawn with larger type for the wide draft (2026-10-28).

Each panel is drawn by its own generator's plotting function, from the same inputs,
with every figure authored K times its usual size. The assembler scales a panel to
its slot, so the text prints 1/K as large and nothing else changes. K = 1 reproduces
the panels in use (the check this script was validated with).

    e  experiments/figure1_wide/flat_panel_e.py's recipe (ks_sequence_panel.panel_a at
       W2 x 0.32), MSH6 A1055P vs I1054F; recomputed from the pLDDT matrices
    f  pten_wt_null_panel.draw_panel from wt_background_2000/results/MSH6_n100_seed
       (the 2,000-split background the paper's panel uses, make_fig1f_panel.py)
    g  make_panels_gh.panel_g from the score caches (published: v6_ca, fingerprint
       d61d29328362; release: the 25-seed config's, whose MSH6 rows are identical)
    h  nsweep_panel.panel_nsweep_metrics from Figure 2's per-gene AUC table
       (annotation_stages stage `curves`; MSH6 identical to the v6 table the paper drew)

Writes to panels/ under the stems the assembler looks for. Nothing else is touched.
Each panel runs in its own process, as its generator does: draw_panel and
make_panels_gh both change global rcParams, which would leak into the next panel.

    python figures/figure1/layout/rerender_efgh.py [K [OUT [efgh]]]
"""
import inspect
import json
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, REPO / "figures/si/lib",
          REPO / "analyses/msh6_case_study/wt_backgrounds_200", REPO / "figures/figure1/ks_combo"):
    sys.path.insert(0, str(p))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# K is one factor, or one per panel ("e=0.87,f=0.93,..."). 0.87 (text x1.15) clips f's
# title and pushes g's legend over the curve; those two take 0.93 (x1.08).
KARG = sys.argv[1] if len(sys.argv) > 1 else "e=0.87,f=0.93,g=0.93,h=0.87"
K = ({p: float(v) for p, v in (kv.split("=") for kv in KARG.split(","))} if "=" in KARG
     else dict.fromkeys("efgh", float(KARG)))
OUT = HERE / (sys.argv[2] if len(sys.argv) > 2 else "panels")   # [K] [out folder]
OUT.mkdir(exist_ok=True)
GENE = "MSH6"


@contextmanager
def authored_at(k):
    """Every new figure at k x its requested size; fonts stay in points."""
    sub, fig = plt.subplots, plt.figure

    def scaled(kw):
        if "figsize" in kw:
            kw["figsize"] = tuple(v * k for v in kw["figsize"])
        return kw
    plt.subplots = lambda *a, **kw: sub(*a, **scaled(kw))
    plt.figure = lambda *a, **kw: fig(*a, **scaled(kw))
    try:
        yield
    finally:
        plt.subplots, plt.figure = sub, fig


def panel_e():
    """flat_panel_e.py 0.32 --gene MSH6 --driver A1055P --passenger I1054F, into OUT."""
    import figlib
    import ks_sequence_panel as KS
    hfrac, old, top0, bot0, ann0 = 0.32, 0.42, 0.90, 0.205, -0.185
    top = 1 - (1 - top0) * old / hfrac
    bot = bot0 * old / hfrac
    ann = ann0 / ((top - bot) * hfrac / ((top0 - bot0) * old))
    src = textwrap.dedent(inspect.getsource(KS.panel_a))
    for a, b in [("W2 * (0.42 if args.panel else 0.50)", f"W2 * {hfrac}"),
                 ("top=0.90 if args.panel else 0.865", f"top={top:.4f}"),
                 ("bottom=0.205 if args.panel else 0.185", f"bottom={bot:.4f}"),
                 ("xy=(col, -0.185)", f"xy=(col, {ann:.4f})"),
                 ("xy=(-0.012, -0.185)", f"xy=(-0.012, {ann:.4f})"),
                 ("left=0.085", "left=0.075")]:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    exec(compile(src, KS.__file__, "exec"), KS.__dict__)
    save = figlib.save
    KS.save = lambda fig, name, out_dir=None: save(
        fig, f"{name}_flat" if name.endswith("_residue_distributions") else name, out_dir=OUT)
    KS.RESULTS = OUT
    KS.record = KS.write_values = KS.provenance = lambda *a, **k: None
    KS.main(["--panel", "--gene", GENE, "--driver", "A1055P", "--passenger", "I1054F",
             "--out-dir", str(OUT)])


def panel_f():
    """The paper's 1f is make_fig1f_panel.draw_panel_lightdots, not draw_panel."""
    sys.path.insert(0, str(REPO / "analyses/msh6_case_study/wt_background_2000"))
    import make_fig1f_panel as F
    src = REPO / "analyses/msh6_case_study/wt_background_2000/results/MSH6_n100_seed"
    values = pd.read_csv(src / "values.csv")
    summary = json.loads((src / "summary.json").read_text())
    F.draw_panel_lightdots(values, summary, OUT, GENE)
    for ext in ("pdf", "png"):   # the stem the assembler looks for
        (OUT / f"{GENE}_wt_null_panel_lightdots.{ext}").replace(OUT / f"{GENE}_wt_null_panel.{ext}")


def panel_g():
    import make_panels_gh as G
    from figures.figure1.seed_panel_scores import load_scores
    G.set_gene(GENE, OUT)
    from drift.protein_config import protein_config_fingerprint
    scores, prov = load_scores(fingerprint=protein_config_fingerprint(), genes=[GENE])
    fig, _ = G.panel_g(scores[scores.gene.eq(GENE)].copy(), prov["size"])
    fig.savefig(OUT / f"{GENE}_ROC_resKS.pdf", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_h():
    from figures.figure1.layout import nsweep_panel as A
    A.VALUES.clear()
    A.panel_nsweep_metrics(OUT, GENE, f"{GENE}_Nsweep_resKS", "F1h_resks",
                           columns=(("ks_raw", "KS-score", A.ACC),), row4_style=True,
                           auc_table=REPO / "analyses/annotation/stages/publication/figure2/panel_abcd_per_gene_auc.csv")
    pd.DataFrame(A.VALUES).to_csv(OUT / f"{GENE}_Nsweep_resKS_values.csv", index=False)


if __name__ == "__main__":
    import subprocess
    which = sys.argv[3] if len(sys.argv) > 3 else "efgh"
    if len(which) > 1:
        for name in which:
            subprocess.run([sys.executable, __file__, KARG, str(OUT.name), name], check=True)
    else:
        print(f"--- panel {which} (K = {K[which]})")
        with authored_at(K[which]):
            {"e": panel_e, "f": panel_f, "g": panel_g, "h": panel_h}[which]()
