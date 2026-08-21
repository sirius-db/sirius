#!/usr/bin/env bash
# Reproduce the TPC-H SF1000 result: 7.00 s suite best-of-3, 22/22 byte-identical (unpatched
# libcudf, late-materialization + fused decode-time filtering on).
# Run from the repo root:  pixi run bash bench/sf1000-repro/run.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

DATA="${DATA:-$HOME/tpch_parquet_sf1000}"          # SF1000 parquet, one dir per table
CUDF_SO="${CUDF_SO:-}"                             # leave unset to use pixi-provided libcudf
PLANS="${PLANS:-$HERE/plans}"
CFG="${CFG:-$HERE/sirius-sf1000.yaml}"
NAME="${NAME:-sf1000_repro}"

# Late materialization (group-by-rowid ride) and fused decode-time filtering. Both off by
# default; both are worth real suite time here and are safe to leave on for a repro run.
# PIN_UNIQUE_COLS must cover every rider's key, not just the ride's own — dropping n_name/
# n_nationkey still gets a correct answer (nation just stops riding) but costs real time.
# See docs/super-sirius/late-materialization.md.
export SIRIUS_EXP_LATE_MAT="${SIRIUS_EXP_LATE_MAT:-1}"
export SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS="${SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS:-c_custkey,n_name,n_nationkey}"
export SIRIUS_EXP_FUSED_SCAN_FILTER="${SIRIUS_EXP_FUSED_SCAN_FILTER:-1}"

[ -d "$DATA" ] || { echo "ERROR: no SF1000 parquet at $DATA (set DATA=)"; exit 1; }

# LD_PRELOAD the patched libcudf only when explicitly provided. When unset the loader finds
# libcudf via the extension's DT_RPATH (pixi env). Do NOT use LD_LIBRARY_PATH — DT_RPATH
# wins over it, so it silently loads the wrong lib. Verify with LD_DEBUG=libs if unsure.
if [ -n "$CUDF_SO" ]; then
  [ -f "$CUDF_SO" ] || { echo "ERROR: CUDF_SO set but not found at $CUDF_SO"; exit 1; }
  export LD_PRELOAD="$CUDF_SO"
fi

# Two settings do the work here:
#   pin_table_input_compression_plan_dir -> simpatico plans; l_shipinstruct MUST stay `dictionary`
#     or the decode-time predicate pushdown has no dictionary to answer from and silently no-ops.
#   expression_evaluator_strategy=ast_jit -> cuDF's JIT expression evaluator instead of the
#     interpreted AST walker. Worth -4.17% suite on its own. Default is ast_interpret.
export SIRIUS_PRE_SQL="SET pin_table_compression = true; \
SET pin_table_input_compression_plan_dir = '$PLANS'; \
SET expression_evaluator_strategy = 'ast_jit'"

# All eight tables pinned GPU-tier compressed. partsupp included: no TPC-H query ever
# materialises ps_comment, so the union across q2/q9/q11/q16/q20 is 4 narrow columns (~20-25 GB).
for t in LINEITEM ORDERS PART CUSTOMER SUPPLIER NATION REGION PARTSUPP; do
  export "SIRIUS_PIN_TIER_$t=gpu"
done

# The first run compiles JIT kernels (~19 s across the suite) into $HOME/.cudf/$VERSION/$ARCH.
# That cache persists across processes, so run twice if you want warm numbers; steady-state
# per-iteration timings are unaffected either way because we report best-of-3.
echo "data      : $DATA"
echo "libcudf   : ${CUDF_SO:-<pixi-provided>}"
echo "plans     : $PLANS"
echo "config    : $CFG"
echo "late-mat  : SIRIUS_EXP_LATE_MAT=$SIRIUS_EXP_LATE_MAT SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS=$SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS"
echo "fused scan: SIRIUS_EXP_FUSED_SCAN_FILTER=$SIRIUS_EXP_FUSED_SCAN_FILTER"
echo

cd "$REPO"
python3 test/tpch_performance/performance_test.py \
  --input "$DATA" \
  --mode grouped --iterations 3 --engine gpu --pin host \
  --queries 1-22 --config "$CFG" --name "$NAME"
