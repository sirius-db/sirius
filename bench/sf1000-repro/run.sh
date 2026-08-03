#!/usr/bin/env bash
# Reproduce the TPC-H SF1000 result: 6.918 s suite, 22/22 byte-identical.
# Run from the repo root:  pixi run bash bench/sf1000-repro/run.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

DATA="${DATA:-$HOME/tpch_parquet_sf1000}"          # SF1000 parquet, one dir per table
CUDF_SO="${CUDF_SO:-$HOME/cudf-src/cpp/build/libcudf.so}"
PLANS="${PLANS:-$HERE/plans}"
CFG="${CFG:-$HERE/sirius-sf1000.yaml}"
NAME="${NAME:-sf1000_repro}"

[ -d "$DATA" ]     || { echo "ERROR: no SF1000 parquet at $DATA (set DATA=)"; exit 1; }
[ -f "$CUDF_SO" ]  || { echo "ERROR: no patched libcudf at $CUDF_SO -- run build-libcudf.sh first"; exit 1; }

# LD_PRELOAD, not LD_LIBRARY_PATH. The Sirius extension .so carries DT_RPATH, which the loader
# searches BEFORE LD_LIBRARY_PATH, so LD_LIBRARY_PATH silently loses to the pixi-provided libcudf.
# Verify with LD_DEBUG=libs that only $CUDF_SO is initialised.
export LD_PRELOAD="$CUDF_SO"

# Two settings do the work here:
#   pin_table_input_compression_plan_dir -> simpatico plans; l_shipinstruct MUST stay `dictionary`
#     or the decode-time predicate pushdown has no dictionary to answer from and silently no-ops.
#   expression_evaluator_strategy=ast_jit -> cuDF's JIT expression evaluator instead of the
#     interpreted AST walker. Worth -4.17% suite on its own. Default is ast_interpret.
export SIRIUS_PRE_SQL="SET pin_table_compression = true; \
SET pin_table_input_compression_plan_dir = '$PLANS'; \
SET expression_evaluator_strategy = 'ast_jit'"

# Fused scan-filter (decode-time filtering + selection-consuming decompression):
# the -15.4% on top of the 8.180 s stack. Off by default in the engine; this kit
# turns it on. Gate off => byte-identical classic behavior. Tuning knobs (defaults
# shipped, all measured — see bench/sf1000-repro/NEXT-STEPS.md):
#   SIRIUS_EXP_FUSED_SCAN_MAX_SEL=0.35        write-skip bail threshold (RULE 2)
#   SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL=0.10  tier-B re-admission threshold
#   SIRIUS_EXP_FUSED_SCAN_K4_MAX_SEL=0.15     K3-vs-K4 payload pick crossover
#   SIRIUS_EXP_FUSED_SCAN_MAX_MEMBER=1        dynamic membership sources kept per scan
#   SIRIUS_EXP_FUSED_SCAN_DIAG=1              deterministic decision tracing (debug)
export SIRIUS_EXP_FUSED_SCAN_FILTER=1

# All eight tables pinned GPU-tier compressed. partsupp included: no TPC-H query ever
# materialises ps_comment, so the union across q2/q9/q11/q16/q20 is 4 narrow columns (~20-25 GB).
for t in LINEITEM ORDERS PART CUSTOMER SUPPLIER NATION REGION PARTSUPP; do
  export "SIRIUS_PIN_TIER_$t=gpu"
done

# The first run compiles JIT kernels (~19 s across the suite) into $HOME/.cudf/$VERSION/$ARCH.
# That cache persists across processes, so run twice if you want warm numbers; steady-state
# per-iteration timings are unaffected either way because we report best-of-3.
echo "data      : $DATA"
echo "libcudf   : $CUDF_SO"
echo "plans     : $PLANS"
echo "config    : $CFG"
echo

cd "$REPO"
python3 test/tpch_performance/performance_test.py \
  --input "$DATA" \
  --mode grouped --iterations 3 --engine gpu --pin host \
  --queries 1-22 --config "$CFG" --name "$NAME"
