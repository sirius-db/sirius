#!/usr/bin/env bash
# Study 1 (scale-out) driver: run the TPC-H set at one CN count with that arm's VALIDATED
# memory configuration, then leave the cluster down.
#
# The per-arm arena/pool split is NOT a preference -- it is measured, and it differs per arm
# because the staging arena requirement scales as 1/N. See ../../../bench/a100x8/CONFIGURATIONS.md.
#
# Usage:  ./study1-run.sh <num_cns> [runs] [query ...]
#         ./study1-run.sh 8 3            # the full set
#         ./study1-run.sh 8 3 q02        # backfill one query into an existing arm
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SR_DIR=$(cd "$HERE/../.." && pwd)
cd "$SR_DIR"

N=${1:?usage: study1-run.sh <num_cns> [runs] [query ...]}
RUNS=${2:-3}
shift $(( $# >= 2 ? 2 : 1 ))
SUBSET=("$@")
DATA=${DATA:-/home/ubuntu/tpch_parquet_sf500}
OUT=${OUT:-/tmp/bench/study1-${N}cn}

# Measured minima (arena) and the pool that fits alongside on an 80 GiB card.
case "$N" in
  8) STAGING=12GiB; GPU_MEM=66GiB; HOST_MEM=100GiB ;;
  4) STAGING=24GiB; GPU_MEM=54GiB; HOST_MEM=200GiB ;;
  # No 2-CN config passes q03+q07+q17 simultaneously (arena and pool both need 1/N and the
  # sum exceeds the card). 32/46 is the least-bad midpoint; expect failures and label the
  # arm as hardware-limited rather than as an engine scaling result.
  2) STAGING=32GiB; GPU_MEM=46GiB; HOST_MEM=400GiB ;;
  1) STAGING=8GiB;  GPU_MEM=70GiB; HOST_MEM=400GiB ;;
  *) echo "no validated config for ${N} CNs" >&2; exit 2 ;;
esac

# Clean queries first so that a restart triggered by a late failure cannot contaminate the
# numbers that matter. q02/q17/q11 are the known-risky tail: q02 wedged at 8 CN, q17 needs the
# arena headroom this config only just provides, and q11 wedges at every arm (its FRACTION is
# the SF1 literal -- see CONFIGURATIONS.md).
QUERIES=(q01 q03 q04 q06 q07 q12 q13 q14 q16 q19 q20 q22 q02 q17 q11)
# A subset backfills one arm without re-running the whole set (e.g. re-measuring a query that
# failed on a since-fixed configuration limit). Results append to the same CSV.
[ ${#SUBSET[@]} -gt 0 ] && QUERIES=("${SUBSET[@]}")

mkdir -p "$OUT"

teardown() {
    pkill -f '[s]irius-starrocks-cn' 2>/dev/null || true
    pkill -f '[S]tarRocksFE' 2>/dev/null || true
    sleep 5
    ps ax -o pid,cmd 2>/dev/null | grep -E '[s]irius-starrocks-cn|[P]rocBasedMain' \
        | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
    for _ in $(seq 1 20); do
        sleep 2
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
               | awk '$1 > 1000' | wc -l)
        [ "${busy:-0}" -eq 0 ] && break
    done
    # Clean bootstrap: same ports every arm, so stale registrations would read Alive and a
    # readiness check would pass against a still-booting cluster. Safe -- no persistent tables.
    rm -rf "$SR_DIR/starrocks/output/fe/meta" 2>/dev/null || true
}

launch() {
    NUM_CNS="$N" GPU_MEM="$GPU_MEM" STAGING="$STAGING" HOST_MEM="$HOST_MEM" \
        ./benchmarks/nixl-nvlink/script-box.sh >>"/tmp/study1-${N}cn-cluster.log" 2>&1 &
}

echo "=== Study 1 | ${N} CN | pool=${GPU_MEM} arena=${STAGING} host=${HOST_MEM} ==="
echo "=== queries: ${QUERIES[*]}"

teardown
launch

MYSQL="mysql --host 127.0.0.1 --port 9030 --user root --batch --connect-timeout=5"
for _ in $(seq 1 36); do
    sleep 5
    alive=$(pixi run $MYSQL -e "SHOW COMPUTE NODES;" 2>/dev/null | grep -c "true") || alive=0
    procs=$(pgrep -fc '[s]irius-starrocks-cn' 2>/dev/null || echo 0)
    [ "$alive" -eq "$N" ] && [ "$procs" -eq "$N" ] && break
done
echo "=== cluster up: $alive/$N CNs ==="
[ "$alive" -eq "$N" ] || { echo "BRING-UP FAILED"; teardown; exit 1; }
sleep 5

# Raise the FE's own query timeout. This is SEPARATE from bench.sh's client-side
# $QUERY_TIMEOUT: the FE aborts server-side at `query_timeout` (default 300 s) with
# ERROR 5024 "Query reached its timeout of 300 seconds", which the sweep records as a
# refusal indistinguishable from an engine failure. q02 hit exactly this at 8 CN while
# completing in ~1.5 s at 4 and 2 CN, so leaving it at the default would report a
# CONFIGURATION limit as an engine result. GLOBAL because bench.sh opens a new session
# per query, so a session-scoped SET would not survive.
pixi run $MYSQL -e "SET GLOBAL query_timeout = ${FE_QUERY_TIMEOUT:-900};" 2>/dev/null \
    && echo "=== FE query_timeout set to ${FE_QUERY_TIMEOUT:-900}s ==="

# bench.sh restarts the cluster after a wedge -- the CN has no cancel_plan_fragment, so a hung
# query strands its fragments and starves every query after it.
TPCH_DATA="$DATA" \
QUERY_TIMEOUT=${QUERY_TIMEOUT:-300} \
COLD_TIMEOUT=${COLD_TIMEOUT:-420} \
MIN_BACKENDS="$N" \
RESTART_CMD="pkill -f '[s]irius-starrocks-cn' 2>/dev/null || true; pkill -f '[S]tarRocksFE' 2>/dev/null || true; sleep 12; rm -rf '$SR_DIR/starrocks/output/fe/meta'; cd '$SR_DIR' && NUM_CNS=$N GPU_MEM=$GPU_MEM STAGING=$STAGING HOST_MEM=$HOST_MEM ./benchmarks/nixl-nvlink/script-box.sh >>/tmp/study1-${N}cn-cluster.log 2>&1 & sleep 45" \
    pixi run bash benchmarks/tpch/bench.sh "$OUT/timings.csv" "$RUNS" "${QUERIES[@]}"

teardown
echo "=== ${N}-CN arm complete -> $OUT/timings.csv ==="
