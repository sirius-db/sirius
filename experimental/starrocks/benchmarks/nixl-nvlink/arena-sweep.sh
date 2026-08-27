#!/usr/bin/env bash
# Find the minimum SIRIUS_EXCHANGE_STAGING_BYTES that lets the arena-sensitive TPC-H
# queries pass at a given CN count.
#
# WHY: the exchange staging arena is a bare cudaMalloc OUTSIDE the RMM pool. Every
# send-side packed batch is gathered into it and every receive-side transfer lands in it.
# With N CNs each node carries 1/N of the fan-out, so the per-CN peak lease grows as the
# CN count FALLS. A single arena size therefore cannot serve every arm of a scale-out
# study -- each arm needs its own, and this script measures which.
#
# The card is fixed at 80 GiB, and the arena is outside the pool, so buying arena costs
# pool 1:1:  GPU_MEM = CARD_GIB - STAGING - RESERVE.
#
# Usage:
#   ./arena-sweep.sh <num_cns> <staging_gib>[,<staging_gib>...] [query ...]
#   ./arena-sweep.sh 4 24,32,40
#   ./arena-sweep.sh 2 32,48 q03 q17
#
# Emits one CSV row per (cns, staging, query) to $OUT_CSV plus a human log on stdout.
# On an arena failure it records the FULL diagnostic (requested/free/capacity/leases),
# which the bench.sh sweep truncates.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SR_DIR=$(cd "$HERE/../.." && pwd)
cd "$SR_DIR"

NUM_CNS=${1:?usage: arena-sweep.sh <num_cns> <staging_gib>[,...] [query ...]}
STAGING_LIST=${2:?usage: arena-sweep.sh <num_cns> <staging_gib>[,...] [query ...]}
shift 2
QUERIES=("$@")
[ ${#QUERIES[@]} -eq 0 ] && QUERIES=(q03 q07 q17)

DATA=${DATA:-/home/ubuntu/tpch_parquet_sf500}
OUT_CSV=${OUT_CSV:-/tmp/bench/arena-sweep.csv}
# 80 GiB card. Reserve covers the CUDA context, cudf's own allocations and fragmentation.
CARD_GIB=${CARD_GIB:-80}
RESERVE_GIB=${RESERVE_GIB:-2}
# Host memory scales with the CN count: the box has ~1650 GiB usable.
HOST_GIB=$(( 800 / NUM_CNS ))
TIMEOUT=${TIMEOUT:-420}

mkdir -p "$(dirname "$OUT_CSV")"
[ -f "$OUT_CSV" ] || echo "cns,staging_gib,gpu_mem_gib,query,status,ms,rows,detail" > "$OUT_CSV"

MYSQL="mysql --host 127.0.0.1 --port 9030 --user root --batch --connect-timeout=5"

teardown() {
    pkill -f '[s]irius-starrocks-cn' 2>/dev/null || true
    pkill -f '[S]tarRocksFE' 2>/dev/null || true
    sleep 4
    # SIGKILL whatever ignored SIGTERM, else the next bring-up hits a busy port / busy GPU.
    ps ax -o pid,cmd 2>/dev/null | grep -E '[s]irius-starrocks-cn|[P]rocBasedMain' \
        | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
    # Block until the GPUs are actually released. The CN reserves pool+arena up front, so a
    # process that has exited but not yet torn down its CUDA context still holds the card and
    # the next bring-up fails its allocation.
    for _ in $(seq 1 20); do
        sleep 2
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
               | awk '$1 > 1000' | wc -l)
        [ "${busy:-0}" -eq 0 ] && break
    done

    # Wipe the FE's persisted metadata so every config bootstraps a CLEAN cluster.
    #
    # THIS IS LOAD-BEARING, not hygiene. The FE keys a compute node by
    # (advertise_host, heartbeat_port) and persists it. Every config here reuses the SAME
    # ports, so after a restart the previous config's 8 registrations are still in meta and
    # are still marked Alive -- the FE only flips them to dead after a heartbeat timeout that
    # is longer than our bring-up. A readiness check that counts Alive rows therefore returns
    # the STALE count immediately, the sweep starts querying while the real CNs are still
    # booting, and the FE answers "No alive backend ... in warehouse default_warehouse" or
    # fails in its CN channel pool with "Unable to validate object". Both look like query
    # failures and would be misread as arena failures -- exactly the confusion this script
    # exists to remove.
    #
    # Safe to delete: engine A creates no persistent tables (every query reads parquet
    # through FILES()), so meta holds nothing but node registrations and session defaults.
    rm -rf "$SR_DIR/starrocks/output/fe/meta" 2>/dev/null || true
}

# Bring up NUM_CNS nodes and block until exactly that many are alive. Returns the final
# count on stdout (callers read the LAST line; progress goes to stderr).
bringup() {
    local gpu_mem=$1 staging=$2
    GPU_MEM="${gpu_mem}GiB" \
    STAGING="${staging}GiB" \
    SIRIUS_EXCHANGE_STAGING_BYTES="${staging}GiB" \
    HOST_MEM="${HOST_GIB}GiB" \
    NUM_CNS="$NUM_CNS" \
        ./benchmarks/nixl-nvlink/script-box.sh >>"/tmp/arena-sweep-cluster.log" 2>&1 &

    local alive=0 procs=0
    for _ in $(seq 1 36); do
        sleep 5
        # Require BOTH: the FE reports N alive, and N CN processes actually exist. After the
        # meta wipe the registration count cannot be stale, and the process check catches a
        # CN that registered and then died during engine bring-up (e.g. pool+arena > card).
        alive=$(pixi run $MYSQL -e "SHOW COMPUTE NODES;" 2>/dev/null | grep -c "true") || alive=0
        procs=$(pgrep -fc '[s]irius-starrocks-cn' 2>/dev/null || echo 0)
        echo "    waiting: $alive registered / $procs processes (want $NUM_CNS)" >&2
        [ "$alive" -eq "$NUM_CNS" ] && [ "$procs" -eq "$NUM_CNS" ] && break
    done

    # Settle, then prove the cluster can actually serve a scan before we trust a timing.
    sleep 5
    if ! pixi run $MYSQL -e \
        "SELECT count(*) FROM FILES('path'='file://$DATA/region/*.parquet','format'='parquet');" \
        >/dev/null 2>&1; then
        echo "    probe query FAILED -- cluster not serving" >&2
        echo 0
        return
    fi
    echo "$alive"
}

echo "=== arena sweep: ${NUM_CNS} CNs, staging {$STAGING_LIST} GiB, queries ${QUERIES[*]} ==="

for staging in ${STAGING_LIST//,/ }; do
    gpu_mem=$(( CARD_GIB - staging - RESERVE_GIB ))
    if [ "$gpu_mem" -lt 8 ]; then
        echo "SKIP staging=${staging}GiB -> pool would be ${gpu_mem}GiB (<8GiB, unusable)"
        for q in "${QUERIES[@]}"; do
            echo "$NUM_CNS,$staging,$gpu_mem,$q,infeasible,,,pool below 8GiB" >> "$OUT_CSV"
        done
        continue
    fi

    echo ""
    echo "--- ${NUM_CNS} CN | staging=${staging}GiB | pool=${gpu_mem}GiB | host=${HOST_GIB}GiB ---"
    teardown
    alive=$(bringup "$gpu_mem" "$staging" | tail -1)

    if [ "$alive" != "$NUM_CNS" ]; then
        echo "  BRING-UP FAILED: $alive/$NUM_CNS alive (pool+arena likely exceeds the card)"
        for q in "${QUERIES[@]}"; do
            echo "$NUM_CNS,$staging,$gpu_mem,$q,bringup_failed,,,only $alive/$NUM_CNS CNs alive" >> "$OUT_CSV"
        done
        continue
    fi
    echo "  cluster up: $alive/$NUM_CNS CNs"

    for q in "${QUERIES[@]}"; do
        sql=$(sed "s|__TPCH_DATA__|$DATA|g" "$HERE/../tpch/queries/$q.sql")
        out=$(mktemp)
        t0=$(date +%s%3N)
        timeout "$TIMEOUT" pixi run $MYSQL -e "$sql" > "$out" 2>&1
        rc=$?
        t1=$(date +%s%3N)
        ms=$(( t1 - t0 ))

        if [ $rc -eq 124 ]; then
            status=wedge; rows=; detail="timeout at ${TIMEOUT}s"
        elif head -1 "$out" | grep -qi '^ERROR'; then
            status=refused
            rows=
            # Keep the whole arena diagnostic on one CSV-safe line: it carries
            # requested/free/capacity/leases, which is what sizes the arena.
            detail=$(tr '\n' ' ' < "$out" | sed 's/,/;/g' | cut -c1-400)
        else
            rows=$(( $(wc -l < "$out") - 1 ))
            [ "$rows" -lt 0 ] && rows=0
            if [ "$rows" -eq 0 ]; then status=empty; else status=pass; fi
            detail=
        fi

        printf "  %-4s %-8s %7sms rows=%-7s %s\n" "$q" "$status" "$ms" "${rows:-–}" "${detail:0:110}"
        echo "$NUM_CNS,$staging,$gpu_mem,$q,$status,$ms,${rows:-},\"$detail\"" >> "$OUT_CSV"
        rm -f "$out"
    done
done

teardown
echo ""
echo "=== sweep complete -> $OUT_CSV ==="
