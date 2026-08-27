#!/usr/bin/env bash
# Study 3 (cost efficiency), Engine A arm: TPC-H at SF500 and SF1000 on all 8 GPUs.
#
# The cost metric is wall time -- cost_per_run = (wall_seconds / 3600) * $/hr -- so the only
# thing this has to produce is a defensible per-scale-factor runtime for the query set.
#
# BOTH SCALE FACTORS RUN BACK TO BACK IN ONE INVOCATION, DELIBERATELY. Repeat measurement of
# identical cells on this box has differed by 2.0-2.2x between campaigns hours apart (see
# ../../../bench/a100x8/SCALE-OUT-SUMMARY.md), which is larger than most effects being
# reported. An SF500 number from yesterday divided into an SF1000 number from today would be
# dominated by that drift rather than by the scale factor.
#
# Arena sizing per scale factor: the arena holds packed exchange batches, so its requirement
# tracks the exchange volume, which tracks the scale factor. SF500 measured 12 GiB at 8 CNs
# (fails at 8, passes at 12); SF1000 is 2x the data, so it starts at 24 GiB. That is a scaled
# first guess, not a measurement -- if a query dies with `exchange staging arena exhausted`,
# raise STAGING_1000 and lower GPU_MEM_1000 by the same amount.
#
# Usage:  ./study3-cost.sh [runs]          # default 3 timed runs after a cold warm-up
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SR_DIR=$(cd "$HERE/../.." && pwd)
cd "$SR_DIR"

RUNS=${1:-3}
NUM_CNS=${NUM_CNS:-8}
OUT_ROOT=${OUT_ROOT:-/tmp/bench/study3}

DATA_500=${DATA_500:-/home/ubuntu/tpch_parquet_sf500}
DATA_1000=${DATA_1000:-/home/ubuntu/tpch_parquet_sf1000}

# 80 GiB card: GPU_MEM = 80 - arena - ~2 GiB of context/fragmentation.
STAGING_500=${STAGING_500:-12GiB};  GPU_MEM_500=${GPU_MEM_500:-66GiB}
STAGING_1000=${STAGING_1000:-24GiB}; GPU_MEM_1000=${GPU_MEM_1000:-54GiB}
# 1771 GiB of host RAM / 8 CNs = 221 GiB each. 100 GiB leaves ~597 GiB of page cache, which is
# 2.3x the SF1000 dataset -- the parquet is re-read every query, so the cache is load-bearing.
HOST_MEM=${HOST_MEM:-100GiB}

# Same 15 queries as Study 1, clean ones first so a restart from a late failure cannot
# contaminate the numbers that matter. q02/q17/q11 are the known-risky tail.
QUERIES=(q01 q03 q04 q06 q07 q12 q13 q14 q16 q19 q20 q22 q02 q17 q11)

MYSQL="mysql --host 127.0.0.1 --port 9030 --user root --batch --connect-timeout=5"

teardown() {
    pkill -f '[s]irius-starrocks-cn' 2>/dev/null || true
    pkill -f '[S]tarRocksFE' 2>/dev/null || true
    sleep 5
    ps ax -o pid,cmd 2>/dev/null | grep -E '[s]irius-starrocks-cn|[P]rocBasedMain' \
        | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
    # The CN reserves pool+arena up front, so a process that exited but has not torn down its
    # CUDA context still owns the card and the next bring-up fails its allocation.
    for _ in $(seq 1 25); do
        sleep 2
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
               | awk '$1 > 1000' | wc -l)
        [ "${busy:-0}" -eq 0 ] && break
    done
    # Clean bootstrap: both scale factors reuse the same ports, so stale registrations would
    # read Alive and a readiness check would pass against a still-booting cluster.
    rm -rf "$SR_DIR/starrocks/output/fe/meta" 2>/dev/null || true
}

run_sf() {
    local label=$1 data=$2 staging=$3 gpu_mem=$4 qto=$5 cto=$6
    local out="$OUT_ROOT/$label"
    mkdir -p "$out"

    echo ""
    echo "############################################################"
    echo "# Study 3 · Engine A · $label"
    echo "#   data=$data"
    echo "#   ${NUM_CNS} CNs · pool=$gpu_mem · arena=$staging · host=$HOST_MEM"
    echo "############################################################"

    teardown
    NUM_CNS="$NUM_CNS" GPU_MEM="$gpu_mem" STAGING="$staging" HOST_MEM="$HOST_MEM" \
        ./benchmarks/nixl-nvlink/script-box.sh >>"/tmp/study3-${label}-cluster.log" 2>&1 &

    local alive=0 procs=0
    for _ in $(seq 1 40); do
        sleep 5
        alive=$(pixi run $MYSQL -e "SHOW COMPUTE NODES;" 2>/dev/null | grep -c "true") || alive=0
        procs=$(pgrep -c -f 'sirius-starrocks-cn' 2>/dev/null || echo 0)
        [ "$alive" -eq "$NUM_CNS" ] && [ "$procs" -eq "$NUM_CNS" ] && break
    done
    echo "=== cluster up: $alive/$NUM_CNS CNs ==="
    [ "$alive" -eq "$NUM_CNS" ] || { echo "BRING-UP FAILED for $label"; return 1; }
    sleep 5

    # Server-side cap, separate from bench.sh's client-side timeout. Left at its 300 s default
    # the FE aborts with ERROR 5024, which the sweep records as a refusal indistinguishable
    # from an engine failure -- and SF1000 queries legitimately run past 300 s.
    #
    # This ALSO has to be re-applied inside RESTART_CMD below. `SET GLOBAL` persists in the FE
    # metadata, and the restart wipes that metadata for a clean bootstrap, so the setting is
    # silently lost the first time a query fails. In the 2026-08-12 SF1000 run q02 ran after
    # two restarts and hit the stock 300 s ceiling despite this line setting 1800 s -- and the
    # resulting ERROR 5024 reads exactly like an engine failure.
    pixi run $MYSQL -e "SET GLOBAL query_timeout = ${cto};" 2>/dev/null \
        && echo "=== FE query_timeout = ${cto}s ==="

    # Wall-clock the whole sweep: the cost metric is time, so the elapsed number is itself a
    # result, not just bookkeeping.
    local t0 t1
    t0=$(date +%s)
    TPCH_DATA="$data" \
    QUERY_TIMEOUT="$qto" \
    COLD_TIMEOUT="$cto" \
    MIN_BACKENDS="$NUM_CNS" \
    RESTART_CMD="pkill -f '[s]irius-starrocks-cn' 2>/dev/null || true; pkill -f '[S]tarRocksFE' 2>/dev/null || true; sleep 12; rm -rf '$SR_DIR/starrocks/output/fe/meta'; cd '$SR_DIR' && NUM_CNS=$NUM_CNS GPU_MEM=$gpu_mem STAGING=$staging HOST_MEM=$HOST_MEM ./benchmarks/nixl-nvlink/script-box.sh >>/tmp/study3-${label}-cluster.log 2>&1 & sleep 45; pixi run $MYSQL -e 'SET GLOBAL query_timeout = ${cto};' 2>/dev/null" \
        pixi run bash benchmarks/tpch/bench.sh "$out/timings.csv" "$RUNS" "${QUERIES[@]}"
    t1=$(date +%s)

    echo "=== $label sweep wall time: $((t1 - t0)) s ==="
    echo "$label,$((t1 - t0))" >> "$OUT_ROOT/sweep-wall-seconds.csv"
}

mkdir -p "$OUT_ROOT"
[ -f "$OUT_ROOT/sweep-wall-seconds.csv" ] || echo "scale,sweep_wall_s" > "$OUT_ROOT/sweep-wall-seconds.csv"

# SF500 first: it is the cheaper failure. If the config or the query set is wrong, finding out
# here costs minutes rather than the hour SF1000 would.
run_sf sf500  "$DATA_500"  "$STAGING_500"  "$GPU_MEM_500"  300 600
run_sf sf1000 "$DATA_1000" "$STAGING_1000" "$GPU_MEM_1000" 900 1800

teardown
echo ""
echo "=== Study 3 Engine A complete -> $OUT_ROOT ==="
