"""Read-only check for SI Note S3: which NM transcripts the retained ClinVar codes
come from, and which germline classifications occur in the archived exports."""
import re, sys
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from metric_analysis.pipeline.clinvar import read_clinvar_csv  # noqa
AA = dict(Ala="A",Arg="R",Asn="N",Asp="D",Cys="C",Gln="Q",Glu="E",Gly="G",His="H",Ile="I",
          Leu="L",Lys="K",Met="M",Phe="F",Pro="P",Ser="S",Thr="T",Trp="W",Tyr="Y",Val="V")
man = pd.read_csv(REPO/"metric_analysis/manifests/strict_canonical_manifest.csv")
files = {"PTEN":"clinvar_result_pten.csv","VHL":"clinvar_result_vhl.csv","MSH2":"clinvar_result_msh2.csv",
         "MSH6":"clinvar_result_msh6.csv","PIK3CA":"clinvar_result_pik3ca.tsv"}
for g,f in files.items():
    fr = read_clinvar_csv(REPO/"Ground_Truths/ClinVar"/f)
    pat = re.compile(rf"^(NM_\d+\.\d+)\({g}\):.*?\(p\.([A-Z][a-z]{{2}})(\d+)([A-Z][a-z]{{2}})\)")
    code_tx = {}
    for name in fr["Name"].astype(str):
        m = pat.search(name)
        if m and m.group(2) in AA and m.group(4) in AA:
            code_tx.setdefault(f"{AA[m.group(2)]}{m.group(3)}{AA[m.group(4)]}", set()).add(m.group(1))
    kept = set(man.loc[man.protein.eq(g), "mutation_code"])
    tx = {}
    for c in kept:
        for t in code_tx.get(c, {"<none>"}):
            tx[t] = tx.get(t, 0) + 1
    allnm = pd.Series([re.match(r"^(NM_\d+\.\d+)", n).group(1) if re.match(r"^(NM_\d+\.\d+)", n) else "other"
                       for n in fr["Name"].astype(str)]).value_counts().to_dict()
    col = [c for c in fr.columns if "lassification" in c]
    print(g, "kept", len(kept), "retained-code transcripts", tx, "| all export NM", allnm)
    for c in col:
        print("   ", c, fr[c].value_counts(dropna=False).to_dict())
