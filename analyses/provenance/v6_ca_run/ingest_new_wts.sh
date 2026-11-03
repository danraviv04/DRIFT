#!/usr/bin/env bash
# Ingest the three regenerated WTs (af3_jobs.wt_from_variant; landed 2026-10-15 in
# $DRIFT_ARCHIVE_ROOT/EXTRA_WT). One drive reader at a time; resumable.
#
#   bash analyses/provenance/v6_ca_run/ingest_new_wts.sh
#
# Per gene:
#   1. one stream_align pass: C-alpha pLDDT CSV (to a staging dir) + coordinates of the
#      125 seed-samples aligned onto the existing cache's WT reference;
#   2. merge the coordinates into the gene's cache (stream_align backs it up first);
#   3. check the CSV shape, move the old WT CSV to PLDDTs/_wt_legacy_2026-10-15/, and put
#      the new one in place (final names: see the end of this script).
# The configs (wt_artifact pins for the coordinate caches, DRIFT data_revision) are a
# separate, reviewed edit.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=.
SRC="${DRIFT_ARCHIVE_ROOT:?set DRIFT_ARCHIVE_ROOT to the AlphaFold 3 archive folder}/EXTRA_WT"
STAGE=analyses/provenance/v6_ca_run/wt_ingest
LEGACY=PLDDTs/_wt_legacy_2026-10-15
mkdir -p "$STAGE" "$LEGACY"
[ -d "$SRC" ] || { echo "$SRC not mounted"; exit 2; }

ingest () {  # ingest <cache name> <archive stem> <chains> <rename or ''> <pLDDT dir> <WT csv> <rows>
  local G=$1 A=$2 CH=$3 RN=$4 PD=$5 WT=$6 ROWS=$7
  local C=metric_analysis/structure_metrics/$G/ensemble_aware/cache
  local X=$C/../cache_newwt
  local ROOT=$STAGE/$G/archive && mkdir -p "$ROOT" "$STAGE/$G/plddt"
  ln -sf "$SRC/$A.tar.gz" "$ROOT/"
  echo "== $G: stream_align $A"
  python -m metric_analysis.pipeline.cache_builders.stream_align --archive-root "$ROOT" \
      --cache-dir "$X" --ref "$C/_rmsf_wt_reference.cif" --chains "$CH" \
      ${RN:+--rename "$RN"} --member-regex 'seed-\d+_sample-\d+' --plddt-out "$STAGE/$G/plddt" --workers 12
  if [ ! -f "$X/.merged" ]; then
    python -m metric_analysis.pipeline.cache_builders.stream_align --merge --cache-dir "$X" \
        --main "$C/${G}_rmsf_aligned_ca_coordinate_cache.parquet"
    touch "$X/.merged"
  fi
  local NEW
  NEW=$(ls "$STAGE/$G/plddt"/*_plddt_values.csv)
  python - "$NEW" "$ROWS" <<'PYEOF'
import re, sys
import pandas as pd
path, rows = sys.argv[1], int(sys.argv[2])
d = pd.read_csv(path, index_col=0)
extra = [c for c in d.columns if not re.fullmatch(r"seed-\d+_sample-\d+", c)]
if extra:   # the job-level top model is named after the job, not the archive
    print(f"dropping non-seed columns {extra}")
    d = d.drop(columns=extra)
    d.to_csv(path)
if d.shape != (rows, 125):
    raise SystemExit(f"{path}: shape {d.shape}, expected ({rows}, 125)")
print(f"{path}: {d.shape} ok")
PYEOF
  if [ ! -f "$LEGACY/$WT" ]; then git mv "$PD/$WT" "$LEGACY/$WT"; fi
  cp "$NEW" "$PD/$WT"
  echo "== $G: $PD/$WT replaced (old in $LEGACY)"
}

ingest PIK3CA     PIK3CA_WT_vartmpl_data              A       ''        PLDDTs/PIK3CA     PIK3CA_WT_plddt_values.csv     1068
ingest TP53       P53_WT_vartmpl_with_ZN_and_DNA_data A,B,C,D '_data$'  PLDDTs/TP53       P53_WT_plddt_values.csv        1572
ingest MutS_alpha MutS_alpha_MSH2_WT_vartmpl_data     A,B     ''        PLDDTs/MutS_alpha MutS_alpha_WT_plddt_values.csv 2294
# Final names (decided 2026-10-15 after the run above): PIK3CA and TP53 carry wt_artifact
# pins (their coordinate caches hold two WTs), and the pLDDT stages map a pin to the CSV's
# canonical stem, so their new CSVs are named after the new archive. MSH2 has no pin and
# keeps the old filename. drift/si_extra.py WT_FILE follows these names.
for pair in "PIK3CA/PIK3CA_WT:PIK3CA/PIK3CA_WT_vartmpl" "TP53/P53_WT:TP53/P53_WT_vartmpl"; do
  from=PLDDTs/${pair%%:*}_plddt_values.csv; to=PLDDTs/${pair##*:}_plddt_values.csv

  if [ -f "$from" ] && [ ! -f "$to" ]; then mv "$from" "$to"; fi
done
echo "all three ingested"
