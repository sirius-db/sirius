#!/usr/bin/env bash
# =============================================================================
# Profile ONE pinned TPC-H query under nsys, capturing ONLY query execution.
#
# Mirrors the multigpu_1k benchmark setup (views + per-query pin via
# tpch_pin_columns.py) but wraps the query in CALL profiler_start()/
# profiler_stop() and runs nsys with --capture-range=cudaProfilerApi so the
# trace excludes pin population and CUDA-context init — you get a clean profile
# of query execution only.
#
# Usage:
#   ./profile_query.sh <query_num> <tier> [config_yaml]
#     tier: none | host | gpu
#   QUERY_TIMEOUT=900 ./profile_query.sh 9 host
#   ./profile_query.sh 9 gpu configs/sirius_4gpu.yaml
#
# Output: results_profile/<tier>_q<N>_<ts>/q<N>.{nsys-rep,sqlite}, query.sql, run.log
# Then runs nsys_analyze.sh on the sqlite for a quick breakdown.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

Q="${1:?usage: profile_query.sh <query_num> <tier:none|host|gpu> [config_yaml]}"
TIER="${2:?tier required: none|host|gpu}"
CONFIG="${3:-$HERE/configs/sirius_4gpu.yaml}"
[[ "$CONFIG" = /* ]] || CONFIG="$HERE/$CONFIG"

DATA="${DATA:-/scratch/prestouser/tpch-rs-float/scale-1000}"
DUCKDB="${DUCKDB:-$REPO/build/release/duckdb}"
PYTHON="${PYTHON:-$REPO/.pixi/envs/default/bin/python}"
GEN="$HERE/gen_query_sql.py"
QUERY_TIMEOUT="${QUERY_TIMEOUT:-1800}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$HERE/results_profile/${TIER}_q${Q}_${TS}}"

command -v nsys >/dev/null || { echo "ERROR: nsys not on PATH" >&2; exit 1; }
[[ -x "$DUCKDB" ]] || { echo "ERROR: duckdb not found: $DUCKDB" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }

mkdir -p "$OUT"
SQL="$OUT/q${Q}.sql"
"$PYTHON" "$GEN" --data "$DATA" --query "$Q" --tier "$TIER" --profile > "$SQL"

echo "=== profiling q$Q tier=$TIER config=$(basename "$CONFIG") ==="
echo "  data:    $DATA"
echo "  out:     $OUT"
echo "  capture: cudaProfilerApi (query execution only; pin/init excluded)"

# --capture-range=cudaProfilerApi: nsys starts recording at cudaProfilerStart()
#   (CALL profiler_start, after pin) and stops at cudaProfilerStop().
# Same low-overhead trace flags as profile_tpch_nsys.sh.
START=$(date +%s)
SIRIUS_CONFIG_FILE="$CONFIG" SIRIUS_LOG_DIR="$OUT/log" \
  timeout "$QUERY_TIMEOUT" \
  nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cudabacktrace=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output="$OUT/q${Q}" \
    --force-overwrite=true \
    --stats=false \
    --export=sqlite \
    "$DUCKDB" -unsigned -init /dev/null -f "$SQL" \
    > "$OUT/run.log" 2>&1
RC=$?
END=$(date +%s)
echo "  nsys exit=$RC, wall=$((END-START))s"

if [[ $RC -eq 124 ]]; then echo "  TIMEOUT after ${QUERY_TIMEOUT}s" >&2; fi
echo "  query Run Time line:"; grep -E 'Run Time \(s\)' "$OUT/run.log" | tail -1 | sed 's/^/    /'

SQLITE="$OUT/q${Q}.sqlite"
if [[ -f "$SQLITE" ]]; then
  echo ""
  echo "=== nsys_analyze.sh $SQLITE ==="
  bash "$REPO/test/tpch_performance/nsys_analyze.sh" "$SQLITE" || true
else
  echo "WARN: no sqlite produced ($SQLITE) — check $OUT/run.log" >&2
fi
echo ""
echo "artifacts: $OUT"
