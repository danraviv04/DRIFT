#!/usr/bin/env bash
# DRIFT: rebuild the paper's figures and numbers from the processed data.
#
#   bash run_all.sh all                    # descriptors -> evaluation -> figures
#   bash run_all.sh evaluation figures     # some phases, in the order given
#   bash run_all.sh B1_cache               # one stage (any name `list` prints)
#   bash run_all.sh list                   # every phase and stage, nothing run
#   DRY_RUN=1 bash run_all.sh all          # print each command instead of running it
#
# Run from anywhere, inside the `drift` conda env (environment.yml); USalign must be on PATH.
# Each phase needs the outputs of the ones before it. A stage's log is logs/<stage>.log,
# and logs/STATUS.log has one line per stage. The first failing stage stops the run.
#
# Phases
#   extract      raw AlphaFold 3 archives -> pLDDT matrices (PLDDTs/), US-align coordinate
#                caches, the MSH6 WT extension pool, AF3 ranking scores, KRAS4B scores and
#                the WT seed maps. Needs the archives under DRIFT_ARCHIVE_ROOT; not part of
#                `all` (the processed data ship separately). AF3 jobs: `python -m af3_jobs`.
#   descriptors  ClinVar export checksums (L0); per-protein cohort summaries (A1, A2); the strict
#                canonical manifest (L1, 1,798 variants); then together the residue-KS route
#                (C1), the score caches for N in {1,5,10,20,40,60,90} x seeds 42-51 (B1,
#                drift/cache) and the WT rigid-core coordinate caches (K1-K3)
#   evaluation   annotation cohort (POLE 25-seed config, run_stage.py): size sweep, tables, AM
#                operating point, per-gene curves, KS CIs, position-cluster intervals, WT
#                12v12 backgrounds, S8 and S1 comparisons; the continuous route (TP53, KRAS4B,
#                PTEN, MSH2 assays); then Fig. 1e,f backgrounds and residue nulls, AF3 top-1,
#                seed allocation (Fig. S6), Table S6, and the Note S1/S4-S6 checks
#   figures      Figure 1 panels and assembly, Figure 2, Supplementary Figures S1-S6, each
#                copied to figures/ under the paper's file name
#   F_fig1b      (stage, not in `figures`) Fig. 1b from the AF3 CIFs; needs ChimeraX. Without
#                it Figure 1 uses figures/source/fig1b_tubes_rmsf_stacked_keybelow.pdf
#
# Not scripted (done by hand for the paper): the AF3 runs themselves (cluster), the
# graphical abstract (BioRender; figures/source/graphical_abstract_2026-10-25.pdf), the
# hand-typed SI tables (S1-S5; rebuild.py asserts S2 and the Note S2 table), and the
# template/MSA provenance checks of Note S6 (wt_variant_offset/, wt_msa_check/; drive).
#
# Settings
#   DRIFT_ARCHIVE_ROOT   folder holding the raw AF3 archives (extract, F_fig1b)
#   CHIMERAX             ChimeraX binary for Fig. 1b (else `chimerax` on PATH)
#   DRY_RUN=1            print commands, run nothing
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$W" || exit 1
PY=python
export PYTHONPATH="$W"
export DRIFT_DEVICE=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Output folders of the v6_ca run (per-protein stages, continuous route, coordinates) carry
# this tag; the 25-seed runner (analyses/annotation/stages/run_stage.py) sets its own.
export DRIFT_RUN_TAG=v6_ca
FINGERPRINT=22e178adaa91      # drift/protein_configs.json (v1.0: 97f1cdbfda8e; the paper's run: 49028c16cbb3)
LOGS="$W/logs"
STATUS="$LOGS/STATUS.log"
BINARY="POLE,PTEN,VHL,MSH2,MSH6,PIK3CA"
# Coordinate datasets, re-fitted on each WT rigid core. POLE P356S has no coordinates (left
# the cohort).
COORD="POLE,PTEN,VHL,PIK3CA,MSH2,MSH6"
SEEDS="42 43 44 45 46 47 48 49 50 51"
RS="analyses/annotation/stages/run_stage.py"

say () {
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "$*"; else echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; fi
}

stage () {          # stage <name> <command...>: run it, log it, return its status
  local name="$1"; shift
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "[$name] $(printf '%q ' "$@")"; return 0; fi
  say "START $name"
  local t0=$SECONDS rc=0
  "$@" > "$LOGS/$name.log" 2>&1 || rc=$?
  if [ $rc -eq 0 ]; then
    say "OK    $name  ($((SECONDS-t0))s)"
  else
    say "FAIL  $name  ($((SECONDS-t0))s, exit $rc)  -> logs/$name.log"
    tail -4 "$LOGS/$name.log" | sed 's/^/        /' | tee -a "$STATUS"
  fi
  return $rc
}

together () {       # together <stage-fn>...: run stage functions in parallel, fail if any fails
  local pids=() rc=0 p f
  if [ "${DRY_RUN:-0}" = 1 ]; then for f in "$@"; do "$f"; done; return 0; fi
  for f in "$@"; do "$f" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

check_config () {
  local fp
  fp=$($PY -c 'from drift.protein_config import protein_config_fingerprint as f; print(f())') || exit 3
  if [ "$fp" != "$FINGERPRINT" ]; then
    echo "drift/protein_configs.json fingerprint is $fp, not the published $FINGERPRINT."
    echo "Results would not be the paper's, and the 25-seed stages refuse to run. Restore the file."
    exit 3
  fi
  [ "${DRY_RUN:-0}" = 1 ] || say "=== run_all $* | fingerprint $fp | commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
}

# ---------------------------------------------------------------------------- extract
# Raw archives -> processed data. Paths come from metric_analysis/protein_configs.json
# ($DRIFT_ARCHIVE_ROOT/<batch>).
need_archives () {
  if [ -z "${DRIFT_ARCHIVE_ROOT:-}" ] && [ "${DRY_RUN:-0}" != 1 ]; then
    echo "extract needs DRIFT_ARCHIVE_ROOT (the folder holding the raw AF3 archives)"; exit 2
  fi
}
# Datasets read from metric_analysis/protein_configs.json (archive roots, chains, cache dirs).
PLDDT_SETS="POLE,PTEN,VHL,PIK3CA,MutS_alpha,MutS_alpha_MSH6_GT"
COORD_SETS="POLE PTEN VHL PIK3CA MutS_alpha MutS_alpha_MSH6_GT"
X1_plddt () {         # per-structure pLDDT matrices -> PLDDTs/<dataset>/
  stage X1_plddt $PY -m metric_analysis.pipeline.orchestration.run_structure_metrics \
        --protein "$PLDDT_SETS" --stage plddt --force \
  && stage X1b_pole_inputs $PY analyses/annotation/stages/make_inputs.py   # POLE's eight 32-seed drivers cut to seeds 1-25
}
X2_coordinates () {   # US-align Calpha coordinate caches (resumes; for a clean rebuild delete the
  local d c          # dataset's cache parquet, structure index, errors CSV and WT reference first)
  for d in $COORD_SETS; do
    for c in 0 1 2 3; do
      stage "X2_coords_${d}_$c" $PY -m metric_analysis.pipeline.cache_builders.repair_rmsf_cache_chunk \
            --protein "$d" --chunk "$c" --total-chunks 4 || return 1
    done
    stage "X2_coords_${d}_merge" $PY -m metric_analysis.pipeline.cache_builders.repair_rmsf_cache_chunk \
          --protein "$d" --merge || return 1
  done
}
X3_msh6_wt_pool () {  # the 375-prediction MSH6 WT pool (Fig. 1e,f, Note S2)
  stage X3_msh6_wt_pool $PY analyses/msh6_case_study/wt_pool/ingest_wt_ext.py
}
X4_af3_ranking () {   # AF3 ranking_score per prediction (Note S3, Table S3); ~3 h
  stage X4_af3_ranking $PY analyses/seeds/seed_structure/extract_af3_ranking.py
}
X5_kras4b_ks () {     # KRAS4B pLDDT and KS from the audited batch (Note S4)
  stage X5_kras4b_ks $PY analyses/continuous/kras4b_coverage/build_ks.py
}
X6_seed_maps () {     # AF3 seed of every WT structure, recovered from the archives (Fig. S6, Table S6)
  stage X6_seed_maps $PY analyses/seeds/seed_structure/rerun_all_genes.py
}
extract () {
  need_archives
  say "=== extract"
  X1_plddt && X2_coordinates && X3_msh6_wt_pool && X4_af3_ranking && X5_kras4b_ks && X6_seed_maps
}

# ---------------------------------------------------------------------------- labels
# The manifest builder reads A1's per-protein *_plddt_distribution_summary.csv (variant
# paths and counts), and B1, C1 and every cohort reader read the manifest, so the cohort
# is rebuilt between A1 and the rest. A1 writes POLE's eight 32-seed drivers with their
# 160-prediction files; update_canonical_summary.py points those rows at the 25-seed copies.
L0_clinvar_sha () {   # the ClinVar exports are the archived ones (run before any cohort rebuild)
  stage L0_clinvar_sha $PY metric_analysis/manifests/verify_clinvar_manifest.py
}
A1_structure_metrics () {   # the per-variant cohort summaries; MutS_alpha and MutS_alpha_MSH6_GT
  # are the modelled complexes the manifest builder reads (MSH2 and MSH6 are their subunits)
  stage A1_structure_metrics $PY -m metric_analysis.pipeline.orchestration.run_structure_metrics \
        --protein "$BINARY,MutS_alpha,MutS_alpha_MSH6_GT" --stage plddt_distribution --force \
  && stage A2_pole_summary_rows $PY analyses/annotation/stages/update_canonical_summary.py
}
L1_manifest () { stage L1_manifest $PY analyses/cohort/build_strict_canonical_manifest.py; }

# ---------------------------------------------------------------------------- descriptors
C1_ks_meanfree () { stage C1_ks_meanfree $PY analyses/msh6_case_study/ks_meanfree/canonical_analysis.py --force; }
b1_seed () {   # the score caches for one subsampling seed, every size; they go to drift/cache
  # and are reused when present
  stage "B1_cache_seed$1" $PY -u -m drift.run --sizes 1,5,10,20,40,60,90 --seeds "$1"
}
B1_cache () {        # ten seeds in parallel, ~7 min
  local pids=() rc=0 s p
  if [ "${DRY_RUN:-0}" = 1 ]; then for s in $SEEDS; do b1_seed "$s"; done; return 0; fi
  for s in $SEEDS; do b1_seed "$s" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
K_coordinates () {
  stage K1_core_fit_caches $PY -m metric_analysis.pipeline.cache_builders.subunit_core_fit --dataset "$COORD" \
  && stage K2_usalign_frame_caches $PY -m metric_analysis.pipeline.cache_builders.subunit_core_fit \
        --dataset "$COORD" --frame usalign \
  && stage K3_validate_caches $PY analyses/coordinates/coord_core_fit/validate.py --dataset "$COORD" \
  && stage K4_coordinate_sweeps $PY -m metric_analysis.pipeline.orchestration.run_structure_metrics \
        --protein "$COORD" --stage ensemble_aware --force
}
descriptors () {
  say "=== descriptors"
  L0_clinvar_sha && A1_structure_metrics && L1_manifest \
  && together C1_ks_meanfree B1_cache K_coordinates
}

# ---------------------------------------------------------------------------- evaluation
rs () { stage "S_$1" $PY "$RS" "$1"; }    # a 25-seed runner stage
E_annotation () {
  rs size && rs tables && rs am_operating_point && rs curves \
  && rs ks_auc_ci && rs position_cluster && rs wt12v12 \
  && rs s8_n90 && rs s1_matched && rs fig1cd && rs s1a
}
E_continuous () {    # Fig. 2c-f; builds the N=90 continuous caches it reads
  stage G6_four_gene_ks $PY analyses/continuous/continuous_functional/plot_four_gene_ks.py
}
POOL=analyses/msh6_case_study/wt_pool/wt_75seeds/MutS_alpha_MSH6_GT_WT_75seeds_plddt_values.csv
PTEN_POOL="PLDDTs/PTEN/PTEN_WT_plddt_values.csv analyses/msh6_case_study/wt_backgrounds_200/plddt/PTEN_WT_seeds26-200_plddt_values.csv"
wt_null () {          # wt_null <name> <args...>: one 200-split WT-vs-WT background
  local name="$1"; shift
  stage "N1_$name" $PY analyses/msh6_case_study/wt_backgrounds_200/pten_wt_null_panel.py "$@" \
        --out "analyses/msh6_case_study/wt_backgrounds_200/results_$name"
}
E_backgrounds () {    # Fig. 1f and its 48/66, 8/77: 200-split backgrounds, replayed and extended to 2,000
  wt_null MSH6_v6ca_wt375_n100_seedsplit --gene MSH6 --wt-csv $POOL --n 100 --split seed --panel \
  && wt_null MSH6_v6ca_wt375_n50_seed --gene MSH6 --wt-csv $POOL --n 50 --split seed \
  && wt_null MSH6_v6ca_wt375_n100_random --gene MSH6 --wt-csv $POOL --n 100 --split random \
  && wt_null MSH6_v6ca_wt375_n50_random --gene MSH6 --wt-csv $POOL --n 50 --split random \
  && wt_null wt1000_n100_seedsplit --wt-csv $PTEN_POOL --n 100 --split seed \
  && wt_null wt1000_n50_seedsplit --wt-csv $PTEN_POOL --n 50 --split seed \
  && wt_null wt1000_n100 --wt-csv $PTEN_POOL --n 100 \
  && wt_null wt1000_n50 --wt-csv $PTEN_POOL --n 50 \
  && stage N2_wt_background_2000 $PY analyses/msh6_case_study/wt_background_2000/run_wt_background_2000.py
}
E_residue_null () {   # Fig. 1e text (20.13%, 0.78%), Note S2 (17.6%), Fig. S3b-e
  stage N3_residue_null $PY analyses/msh6_case_study/residue_null/residue_null.py --gene MSH6 --mutants E221D R772L \
        --window 764 779 --wt-pool $POOL --splits 2000 --modes matched \
  && stage N4_residue_null_2000 $PY analyses/msh6_case_study/wt_background_2000/residue_null_2000.py \
  && stage N5_near_pair $PY analyses/msh6_case_study/fig1e_near_pair/residue_null_near_pair.py \
  && stage N6_gene_wide $PY analyses/msh6_case_study/fig1e_near_pair/gene_wide.py
}
E_seeds () {          # Note S3 + Table S3 (AF3 top-1); Fig. S6 + Note S7 (seed allocation); Table S6
  stage N7_af3_top1_vs_ks $PY analyses/seeds/af3_top1_vs_ks/af3_top1_vs_ks.py \
  && stage N8_plateau_run $PY analyses/seeds/seed_structure/plateau_run.py --workers 3 \
  && stage N9_plateau_summary $PY analyses/seeds/seed_structure/plateau_summary.py \
  && stage N10_three_arm_n25 $PY analyses/seeds/seed_structure/three_arm_n25.py \
  && stage N11_seed_icc $PY analyses/seeds/seed_structure/seed_icc.py \
  && stage N12_seed_pairing $PY analyses/seeds/seed_pairing/seed_pairing.py \
  && stage N13_seed_coupling_ks $PY analyses/seeds/seed_coupling_ks/seed_coupling_ks.py
}
E_checks () {         # Note S1 frame sensitivity; Notes S4-S6 and Table S4b checks
  stage N14_usalign_scored_chain $PY analyses/coordinates/coord_core_fit/usalign_scored_chain.py --steps align \
  && stage N15_frame_sensitivity $PY analyses/coordinates/s1_frame_sensitivity/frame_sensitivity.py \
  && stage N16_kras_by_assay_bin $PY analyses/continuous/kras_score_by_assay_bin/kras_score_by_assay_bin.py \
  && stage N17_tp53_snv_reachability $PY analyses/continuous/tp53_snv_reachability/tp53_snv_reachability.py \
  && stage N18_af3_version_check $PY analyses/seeds/tp53_af3_version_check/version_check.py \
        --out analyses/seeds/tp53_af3_version_check/results_v6
}
E_standalone () { E_backgrounds && E_residue_null && E_seeds && E_checks; }
evaluation () {
  say "=== evaluation"
  together E_annotation E_continuous && E_standalone
}

# ---------------------------------------------------------------------------- figures
F1P=figures/figure1/layout/panels   # Figure 1 panels are drawn here, then collected
F_figure2 () {
  rs si_coordinate_vs_plddt && rs si_ks_vs_am && rs figure2 \
  && stage F2_collect cp analyses/annotation/stages/publication/figure2/Figure2.pdf figures/Figure2.pdf
}
F_fig1b () {   # Fig. 1b: needs the AF3 CIFs (DRIFT_ARCHIVE_ROOT) and ChimeraX; not in `figures`
  need_archives
  local B=figures/figure1/fig1b
  stage F1b_export_models $PY $B/export_models.py \
  && stage F1b_rmsf_bfactor $PY $B/rmsf_bfactor.py \
  && stage F1b_render $PY $B/render_panel_b.py --style tubes_rmsf \
  && stage F1b_compose $PY $B/render_panel_b.py --style tubes_rmsf --compose-only --stack-bubble-d 96 \
        --font-scale 1.2 --key-below --stacked-to "$F1P/fig1b_tubes_rmsf_stacked_keybelow.pdf"
}
collect_figure1 () {   # the assembler reads figures/panels/<stem>_big.pdf, the paper's names
  local s
  mkdir -p figures/panels
  cp "$F1P/POLE_exemplar_shift_WT_vs_P286R_row_rmsf_big.pdf" figures/panels/POLE_centre_shift_row_big.pdf \
  && cp "$F1P/POLE_exemplar_spread_WT_vs_P286R_row_rmsf_big.pdf" figures/panels/POLE_spread_profile_row_big.pdf \
  || return 1
  for s in MSH6_residue_distributions_flat MSH6_wt_null_panel MSH6_ROC_resKS MSH6_Nsweep_resKS; do
    cp "$F1P/$s.pdf" "figures/panels/${s}_big.pdf" || return 1
  done
  # 1a (BioRender) and 1b (ChimeraX) are read from figures/source/ unless F_fig1b ran
  [ ! -e "$F1P/fig1b_tubes_rmsf_stacked_keybelow.pdf" ] \
    || cp "$F1P/fig1b_tubes_rmsf_stacked_keybelow.pdf" figures/panels/
}
F_figure1 () {
  stage F1a_ga_boxplot $PY figures/figure1/ga_boxplot/make_ga_ks_boxplot.py \
  && stage F1cd_panels $PY analyses/coordinates/rmsf_histogram/exemplar_wt_vs_mut.py --mutant P286R --metric rmsf \
        --panels bc --panel-width-in 4.4 --bc-heights 1.8,1.8 --shift-xaxis --stem-suffix _row_rmsf_big \
        --shift-note rmsd --wt-colour "#5a5a5a" --wt-on-top --out-dir "$F1P" \
  && stage F1efgh_panels $PY figures/figure1/layout/rerender_efgh.py \
  && stage F1_collect collect_figure1 \
  && stage F1_assemble $PY figures/make_figures.py Figure1
}
collect_si () {        # each SI figure under the paper's file name
  cp figures/si/S1_S2/results/FigureS_coordinate_descriptors.pdf \
     figures/si/S1_S2/results/FigureS_rigid_core.pdf \
     figures/si/S3_S4/results/FigureS_per_residue_test.pdf \
     figures/si/S3_S4/results/FigureS_wt_backgrounds.pdf \
     figures/si/S5/results/FigureS_ks_vs_alphamissense.pdf figures/ \
  && cp analyses/seeds/seed_structure/three_arm_n25/three_arm_n25_plddt.pdf figures/FigureS_seed_spread.pdf
}
gather_si_inputs () {  # the SI rebuilds read inputs/, copies of the runs' outputs
  local S=analyses/annotation/stages/side P=analyses/annotation/stages/publication/si_figures
  local I1=figures/si/S1_S2/inputs I2=figures/si/S3_S4/inputs
  local I3=figures/si/S5/inputs G=analyses/msh6_case_study/fig1e_near_pair/results/gene_wide
  mkdir -p $I1 $I2 $I3
  cp $S/s1_matched_draws/auc_matched_draws.csv $I1/auc_matched_draws_25seed.csv \
  && cp $S/s1_matched_draws/variant_seed_scores.csv.gz $I1/variant_seed_scores_25seed.csv.gz \
  && cp $S/s1a/shift_vs_rmsf_P286R.csv $I1/shift_vs_rmsf_P286R_25seed.csv \
  && cp $S/s1a/shift_vs_rmsf_P286R_values.csv $I1/shift_vs_rmsf_P286R_values_25seed.csv \
  && cp $P/wt_rmsf_vs_plddt_residues.csv $I1/ \
  && cp $S/wt_split_12v12/wt_background_splits.csv $S/wt_split_12v12/wt_background_variant_scores.csv \
        $S/wt_split_12v12/wt_backgrounds_values.csv $G/MSH6_gene_wide_per_residue.csv \
        $G/MSH6_gene_wide_values.csv $I2/ \
  && cp $P/ks_vs_am_correlations.csv $P/ks_vs_am_macro.csv $I3/ \
  && cp $P/ks_vs_am_variants.csv $I3/ks_vs_am_variants_25seed.csv
}
F_si () {
  stage FS_inputs gather_si_inputs \
  && stage FS1_S2_coordinates $PY figures/si/S1_S2/rebuild.py \
  && stage FS3_S4_note_s2 $PY figures/si/S3_S4/rebuild.py \
  && stage FS5_ks_vs_am $PY figures/si/S5/rebuild.py \
  && stage FS_collect collect_si
}
figures () {
  say "=== figures"
  F_figure2 && F_figure1 && F_si
}

all () { descriptors && evaluation && figures && say "=== all phases done"; }

list () {
  sed -n '/^# Phases/,/^# Settings/p' "$0" | sed '$d'
  echo "Stages (bash run_all.sh <name>):"
  grep -oE '^[A-Z][A-Za-z0-9_]* \(\)' "$0" | sed 's/ ()//; s/^/  /'
  echo "  S_<stage>   one stage of analyses/annotation/stages/run_stage.py (e.g. S_curves)"
}

[ $# -gt 0 ] && [ "$*" != list ] || { list; exit 0; }
runs () {            # runs <target>: a phase or stage function, or S_<run_stage.py stage>
  if declare -F "$1" >/dev/null; then "$1"; else rs "${1#S_}"; fi
}
for target in "$@"; do
  declare -F "$target" >/dev/null || [[ "$target" == S_* ]] \
    || { echo "unknown phase or stage: $target"; list; exit 2; }
done
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$LOGS"
check_config "$@"
for target in "$@"; do
  runs "$target" || { say "=== stopped at $target"; exit 1; }
done
