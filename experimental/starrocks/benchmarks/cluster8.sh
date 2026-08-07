#!/usr/bin/env bash
# Bring up 1 FE + N Sirius GPU compute nodes, one CN per GPU, cross-node exchange over nixl.
#
# The `cluster2` pixi task generalized to a loop. Two CNs could offset their ports by +2; at
# eight the heartbeat range would collide with the thrift range, so each CN instead gets a
# contiguous 10-port block based at $PORT_BASE -- clear of the FE's ports (8030/9010/9020/9030)
# and of the CN defaults (9050/9060/8040/8060/9070).
#
# Two identities must stay unique across CNs: the FE keys a node by
# (advertise_host, heartbeat_port), and the nixl agent is named {advertise_host}:{brpc_port}.
#
# Usage:  ./benchmarks/cluster8.sh
#         NUM_CNS=4 GPU_MEM=48GiB ./benchmarks/cluster8.sh
#
# Run it in its own terminal or as its own background task -- never chained behind `&` inside
# another shell command, or the cluster dies with that shell.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SR_DIR=$(cd "$HERE/.." && pwd)              # experimental/starrocks
REPO_ROOT=$(cd "$SR_DIR/../.." && pwd)
TOOLS_DIR=${TOOLS_DIR:-$(cd "$REPO_ROOT/.." && pwd)/tools}

NUM_CNS=${NUM_CNS:-8}
PORT_BASE=${PORT_BASE:-9100}
PORT_STRIDE=${PORT_STRIDE:-10}
# The staging arena sits OUTSIDE --gpu-memory-limit, so a CN really occupies
# GPU_MEM + STAGING + CUDA context. On an 80GiB A100 that is ~75 of 80 GiB.
GPU_MEM=${GPU_MEM:-64GiB}
HOST_MEM=${HOST_MEM:-128GiB}
STAGING=${STAGING:-8GiB}

CN_BIN=$SR_DIR/target/release/sirius-starrocks-cn
FE_BIN=$SR_DIR/starrocks/output/fe/bin/start_fe.sh

[ -x "$CN_BIN" ] || { echo "no CN binary at $CN_BIN -- run: pixi run cn-build" >&2; exit 1; }
[ -x "$FE_BIN" ] || { echo "no packaged FE at $FE_BIN -- run: pixi run fe-check" >&2; exit 1; }

# NIXL_PREFIX / NIXL_PLUGIN_DIR / LD_LIBRARY_PATH (engine .so + pixi env lib + nixl + UCX) /
# UCX_TLS, all derived from the repo and $TOOLS_DIR locations; fails loudly when nixl is
# absent rather than continuing misconfigured.
# shellcheck source=../scripts/cn-env.sh
source "$SR_DIR/scripts/cn-env.sh"

export SIRIUS_EXCHANGE_STAGING_BYTES=${SIRIUS_EXCHANGE_STAGING_BYTES:-$STAGING}

avail=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$avail" -ge "$NUM_CNS" ] || {
    echo "asked for $NUM_CNS CNs but only $avail GPUs are visible" >&2; exit 1; }

pids=()
cleanup() {
    status=$?
    trap - EXIT INT TERM
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

cd "$SR_DIR"
"$FE_BIN" --logconsole &
pids+=("$!")

for i in $(seq 0 $((NUM_CNS - 1))); do
    base=$((PORT_BASE + i * PORT_STRIDE))
    "$CN_BIN" \
        --gpu-device "$i" \
        --heartbeat-port "$base" \
        --thrift-port    "$((base + 1))" \
        --brpc-port      "$((base + 2))" \
        --http-port      "$((base + 3))" \
        --starlet-port   "$((base + 4))" \
        --gpu-memory-limit "$GPU_MEM" \
        --host-memory-limit "$HOST_MEM" \
        --engine-dir ".cn$i" &
    pids+=("$!")
    echo "CN$i gpu=$i heartbeat=$base brpc=$((base + 2)) pid=${pids[-1]}"
done

echo "FE + $NUM_CNS CNs launched; each CN self-registers with the FE on :9030"
# Any child exiting means the cluster is broken -- fall through to cleanup rather than
# leaving a half-cluster that the benchmark would silently measure.
wait -n "${pids[@]}"
