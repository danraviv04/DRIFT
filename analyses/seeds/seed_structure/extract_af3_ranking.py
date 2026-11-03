"""Recover recorded AF3 ranking scores from the indexed job, never pLDDT."""
import os
import io
import gzip
import math
import json
from pathlib import Path
import re
import tarfile

import pandas as pd

from metric_analysis.pipeline.labels import artifact_key
from metric_analysis.pipeline.paths import on_archive_root
from metric_analysis.pipeline.library.canonical_cohort import load_all_canonical_cohorts
from plateau_run import OUT, ROOT, GENES, TAGS, sha


def parse_csv(data):
    frame = pd.read_csv(io.BytesIO(data))
    assert {"seed", "sample", "ranking_score"}.issubset(frame.columns)
    assert not frame.duplicated(["seed", "sample"]).any()
    scores = {f"seed-{int(r.seed)}_sample-{int(r.sample)}": float(r.ranking_score)
              for r in frame.itertuples()}
    assert all(math.isfinite(value) for value in scores.values()), "nonfinite AF3 ranking score"
    return scores


def read_ranking(row, archive_override=None):
    if archive_override:
        archive, parent = archive_override, None
    elif str(row.get("structure_path", "")).startswith("tar://"):
        archive = str(on_archive_root(row["structure_archive"]))
        parent = re.split(r"/seed-\d+_sample-\d+", row["structure_member"])[0]
    else:
        path = on_archive_root(row["structure_path"])
        parent = path.parent.parent if re.fullmatch(r"seed-\d+_sample-\d+", path.parent.name) else path.parent
        csvs = sorted(parent.glob("*ranking_scores.csv"))
        if len(csvs) == 1:
            data = csvs[0].read_bytes()
            return parse_csv(data), str(csvs[0]), "ranking_scores.csv"
        assert not csvs, "ambiguous ranking CSV"
        scores = {}
        for tag in TAGS:
            files = list((parent / tag).glob("*summary_confidences.json"))
            if len(files) == 1:
                scores[tag] = float(json.loads(files[0].read_text())["ranking_score"])
        return scores, str(parent), "summary_confidences.json (rounded fallback)"
    summaries = {}
    # Buffer compressed I/O, but bound decompressed blocks for huge MSA members.
    with open(archive, "rb", buffering=4*1024*1024) as source, gzip.GzipFile(fileobj=source) as decoded, \
            tarfile.open(fileobj=decoded, mode="r|", bufsize=1024*1024) as handle:
        for member in handle:
            if not member.isfile():
                continue
            if member.name.endswith("ranking_scores.csv") and (parent is None or str(Path(member.name).parent) == parent):
                return parse_csv(handle.extractfile(member).read()), f"{archive}::{member.name}", "ranking_scores.csv"
            tag = re.search(r"seed-\d+_sample-\d+", member.name)
            if tag and member.name.endswith("summary_confidences.json") and (parent is None or member.name.startswith(parent + "/")):
                summaries[tag[0]] = float(json.loads(handle.extractfile(member).read())["ranking_score"])
    return summaries, archive, "summary_confidences.json (rounded fallback)"


def main():
    cohort = load_all_canonical_cohorts()
    destination = OUT / "af3_ranking"
    destination.mkdir(parents=True, exist_ok=True)
    audit = []
    for gene in GENES:
        folder = destination / gene
        folder.mkdir(exist_ok=True)
        cache = ROOT / "metric_analysis/structure_metrics" / gene / "ensemble_aware/cache_corefit"
        report = json.loads((cache / f"{gene}_core_fit_report.json").read_text())
        index = pd.read_csv(report["structure_index_file"])
        index["key"] = index.artifact.map(artifact_key)
        # Prefer a sample's path: it pins the exact job/attempt, not a sibling rerun.
        sample_rows = index[index.structure_path.astype(str).str.contains(r"seed-\d+_sample-\d+", regex=True)]
        by_key = {key: b.iloc[0].to_dict() for key, b in sample_rows.groupby("key")}
        records = cohort[cohort.protein == gene].to_dict("records")
        jobs = [("WT", records[0]["wt_source_file"], artifact_key(report["wt_artifact"]))]
        jobs += [(r["mutation_code"], r["source_file"], artifact_key(Path(r["source_file"]).name.replace("_plddt_values.csv", ""))) for r in records]
        for number, (mutation, source, key) in enumerate(jobs, 1):
            target = folder / f"{mutation}.json"
            if target.exists():
                result = json.loads(target.read_text())
            else:
                result = {"gene": gene, "mutation": mutation, "source": source,
                          "source_sha256": sha(source), "status": "missing", "scores": {}}
                try:
                    override = None
                    if mutation == "WT" and gene in ["MSH2", "PIK3CA"]:
                        override = f"{os.environ.get('DRIFT_ARCHIVE_ROOT', '$DRIFT_ARCHIVE_ROOT')}/EXTRA_WT/{report['wt_artifact']}.tar.gz"
                    if key not in by_key and override is None:
                        raise ValueError("no indexed source job")
                    scores, origin, precision = read_ranking(by_key.get(key), override)
                    result.update(scores={t: scores[t] for t in TAGS if t in scores},
                                  origin=origin, precision=precision)
                    result["status"] = "complete" if len(result["scores"]) == 125 else "incomplete"
                except (OSError, ValueError, KeyError, AssertionError, tarfile.TarError) as error:
                    result["error"] = f"{type(error).__name__}: {error}"
                target.write_text(json.dumps(result, indent=2) + "\n")
            audit.append({k: v for k, v in result.items() if k != "scores"} | {"n_ranked_samples": len(result["scores"])})
            if number % 100 == 0:
                print(f"Ranking {gene}: {number}/{len(jobs)}", flush=True)
        pd.DataFrame(audit).to_csv(destination / "coverage.csv", index=False)
        print(f"Ranking {gene} completed", flush=True)


if __name__ == "__main__":
    main()
