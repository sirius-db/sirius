#!/usr/bin/env bash
# Three-arm late-materialization benchmark on GB300.
#
# Arms:
#   1. gate-off identity check: dev | decompression-pushdown | late-mat (no LATE_MAT env)
#      All three should be byte-identical and timing-identical (within noise).
#      Establishes that the late-mat gating code costs nothing.
#
#   2. late-mat gate-on: late-mat branch + SIRIUS_EXP_LATE_MAT=1
#      Measures stamping overhead only — nothing consumes origins yet.
#      Expected: indistinguishable from gate-off. Any delta > noise is a finding.
#
#   3. fused-scan trace: q17/18/19, gate-on (FUSED_SCAN_FILTER=1 + LATE_MAT=1)
#      Captures simpatico decision/enumeration trace to stderr.
#      Check blocks=T/C ratios: T << C means decompression pushdown is compacting.
#
# Usage (from repo root):
#   pixi run bash bench/late-mat-runs.sh [arm1|arm2|arm3]
#   (no arg = print menu)
#
# Prereqs:
#   - dev worktree built:   /home/nvidia/joost/sirius-dev-bench/build/...
#   - decomp worktree built: /home/nvidia/joost/sirius-decomp-bench/build/...
#   - SF1000 data at /raid/datasets/tpch_sf1000

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
REPRO="$HERE/sf1000-repro"

DATA="${DATA:-/raid/datasets/tpch_sf1000}"
PLANS="$REPRO/plans"
CFG="$REPRO/sirius-sf1000.yaml"

DEV_REPO=/home/nvidia/joost/sirius-dev-bench
DECOMP_REPO=/home/nvidia/joost/sirius-decomp-bench

_check_data() {
  [ -d "$DATA" ] || { echo "ERROR: no SF1000 parquet at $DATA (set DATA=)"; exit 1; }
}

_common_env() {
  export SIRIUS_PRE_SQL="SET pin_table_compression = true; \
SET pin_table_input_compression_plan_dir = '$PLANS'; \
SET expression_evaluator_strategy = 'ast_jit'"
  for t in LINEITEM ORDERS PART CUSTOMER SUPPLIER NATION REGION PARTSUPP; do
    export "SIRIUS_PIN_TIER_$t=gpu"
  done
}

_run_suite() {
  local repo="$1" name="$2"
  python3 "$repo/test/tpch_performance/performance_test.py" \
    --input "$DATA" \
    --mode grouped --iterations 3 --engine gpu --pin host \
    --queries 1-22 --config "$CFG" --name "$name"
}

_run_queries() {
  local repo="$1" name="$2" queries="$3"
  python3 "$repo/test/tpch_performance/performance_test.py" \
    --input "$DATA" \
    --mode grouped --iterations 3 --engine gpu --pin host \
    --queries "$queries" --config "$CFG" --name "$name"
}

arm1() {
  _check_data
  _common_env

  echo "=== ARM 1: gate-off identity check ==="
  echo "--- dev ---"
  (cd "$DEV_REPO" && _run_suite "$DEV_REPO" "late_mat_arm1_dev")

  echo "--- decompression-pushdown (gate-off) ---"
  (cd "$DECOMP_REPO" && _run_suite "$DECOMP_REPO" "late_mat_arm1_decomp_off")

  echo "--- late-materialization (LATE_MAT unset) ---"
  (cd "$REPO" && _run_suite "$REPO" "late_mat_arm1_latemat_off")

  echo "=== ARM 1 done. All three should be within ±1.2% suite noise. ==="
}

arm2() {
  _check_data
  _common_env
  export SIRIUS_EXP_LATE_MAT=1

  echo "=== ARM 2: SIRIUS_EXP_LATE_MAT=1 (stamping overhead only) ==="
  (cd "$REPO" && _run_suite "$REPO" "late_mat_arm2_latemat_on")

  echo "=== ARM 2 done. Delta vs arm1 late-mat should be noise (< 1.2%). ==="
}

arm3() {
  _check_data
  _common_env
  export SIRIUS_EXP_FUSED_SCAN_FILTER=1
  export SIRIUS_EXP_LATE_MAT=1
  export SIRIUS_EXP_FUSED_SCAN_DIAG=1
  export SIRIUS_PRE_SQL="${SIRIUS_PRE_SQL}; SET sirius_log_backend='spdlog'"

  local logfile="$REPO/test/tpch_performance/output/late_mat_arm3_diag.log"
  echo "=== ARM 3: fused-scan trace on q17/18/19 ==="
  echo "Trace goes to stderr -> $logfile"
  echo "(grep 'simpatico:' for decision lines; blocks=T/C shows compaction)"

  (cd "$REPO" && _run_queries "$REPO" "late_mat_arm3_diag" "17,18,19") 2>&1 | tee "$logfile"

  echo ""
  echo "=== simpatico trace summary ==="
  grep "simpatico:" "$logfile" | grep -v "refused\|capped" || echo "(no trace lines — check SIRIUS_EXP_FUSED_SCAN_FILTER is reaching the right build)"
}

case "${1:-}" in
  arm1) arm1 ;;
  arm2) arm2 ;;
  arm3) arm3 ;;
  *)
    echo "Usage: pixi run bash bench/late-mat-runs.sh [arm1|arm2|arm3]"
    echo ""
    echo "  arm1  gate-off identity: dev | decomp | late-mat (no LATE_MAT)"
    echo "  arm2  gate-on overhead:  late-mat + SIRIUS_EXP_LATE_MAT=1"
    echo "  arm3  fused-scan trace:  q17/18/19 + FUSED_SCAN_DIAG=1"
    echo ""
    echo "Prereqs:"
    echo "  dev build:   $DEV_REPO/build/release/extension/sirius/sirius.duckdb_extension"
    echo "  decomp build: $DECOMP_REPO/build/release/extension/sirius/sirius.duckdb_extension"
    echo "  SF1000 data:  $DATA"
    for f in \
      "$DEV_REPO/build/release/extension/sirius/sirius.duckdb_extension" \
      "$DECOMP_REPO/build/release/extension/sirius/sirius.duckdb_extension"; do
      [ -f "$f" ] && echo "  OK  $f" || echo "  MISSING $f"
    done
    [ -d "$DATA" ] && echo "  OK  $DATA" || echo "  MISSING $DATA"
    ;;
esac
