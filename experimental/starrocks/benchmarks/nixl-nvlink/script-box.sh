#!/usr/bin/env bash
# Bring up 1 FE + N Sirius GPU compute nodes, one CN per GPU, cross-node exchange over nixl.
#
# Defaults target this box: 8x A100-SXM4-80GB, NV12 between every pair, 240 vCPU / 1771 GB.
# Memory budget per GPU: 68 GiB pool (0.85 x 80 GiB) + 8 GiB staging arena + ~1 GiB context
# = 77 GiB of the 80 GiB device. DO NOT copy GPU_MEM/STAGING from the H100 notes (40/32 GiB)
# -- that leaves only a 40 GiB pool, halving throughput.
#
# The `cluster2` pixi task generalized to a loop. Two CNs could offset their ports by +2; past
# a handful the heartbeat range would collide with the thrift range, so each CN instead gets a
# contiguous 10-port block based at $PORT_BASE -- clear of the FE's ports (8030/9010/9020/9030)
# and of the CN defaults (9050/9060/8040/8060/9070).
#
# Two identities must stay unique across CNs: the FE keys a node by
# (advertise_host, heartbeat_port), and the nixl agent is named {advertise_host}:{brpc_port}.
#
# Usage:  ./benchmarks/nixl-nvlink/script-box.sh
#         NUM_CNS=4 ./benchmarks/nixl-nvlink/script-box.sh   # scale-out study, 4-GPU arm
#
# Run it in its own terminal or as its own background task -- never chained behind `&` inside
# another shell command, or the cluster dies with that shell.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # benchmarks/nixl-nvlink
SR_DIR=$(cd "$HERE/../.." && pwd)           # experimental/starrocks
REPO_ROOT=$(cd "$SR_DIR/../.." && pwd)      # repo root
TOOLS_DIR=${TOOLS_DIR:-$(cd "$REPO_ROOT/.." && pwd)/tools}

# One CN per visible GPU by default; override to use a subset (the first NUM_CNS ordinals).
avail=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
NUM_CNS=${NUM_CNS:-$avail}
PORT_BASE=${PORT_BASE:-9100}
PORT_STRIDE=${PORT_STRIDE:-10}
# The staging arena sits OUTSIDE --gpu-memory-limit, so a CN really occupies
# GPU_MEM + STAGING + CUDA context. On an 80 GiB A100 the budget is tight:
#   62 GiB pool + 16 GiB arena + ~1 GiB context = 79 GiB < 80 GiB.
# Raising STAGING therefore REQUIRES lowering GPU_MEM by the same amount.
#
# STAGING=16GiB derived from GB200 SF500 learning: q17 failed at 16 GiB and passed
# at 32 GiB on a 4-CN GB200 cluster. With 8 CNs each handles half the fan-out, so
# ~16 GiB/CN should suffice. q21 is a lease-lifecycle bug -- NOT a sizing issue --
# and will fail regardless of arena size. q15 is non-deterministic (FP64 decimal
# lowering / float-equality predicate); expect ~50% pass rate.
# q07 and q13 passed comfortably on GB200; include them in the sweep.
GPU_MEM=${GPU_MEM:-62GiB}
# 120 GiB x 8 CNs = 960 GiB of ~1649 GiB, leaving ~437 GiB page cache = 3.4x SF500.
# [SF1000 arm]: override with HOST_MEM=107GiB (800 GiB total, ~597 GiB cache = 2.3x)
# [4-CN arm]:   override with HOST_MEM=200GiB
# [2-CN arm]:   override with HOST_MEM=300GiB
HOST_MEM=${HOST_MEM:-120GiB}
STAGING=${STAGING:-16GiB}

CN_BIN=$SR_DIR/target/release/sirius-starrocks-cn
FE_BIN=$SR_DIR/starrocks/output/fe/bin/start_fe.sh

# The FE is Java. Without this the launcher works from a pixi shell but fails when bench.sh's
# RESTART_CMD relaunches it from a bare shell -- and a restart that silently brings up no FE
# turns every subsequent measurement into a phantom wedge.
if [ -z "${JAVA_HOME:-}" ] && [ -x "$SR_DIR/.pixi/envs/default/lib/jvm/bin/java" ]; then
    export JAVA_HOME="$SR_DIR/.pixi/envs/default/lib/jvm"
    export PATH="$JAVA_HOME/bin:$PATH"
fi

[ -x "$CN_BIN" ] || { echo "no CN binary at $CN_BIN -- run: pixi run cn-build" >&2; exit 1; }
[ -x "$FE_BIN" ] || { echo "no packaged FE at $FE_BIN -- run: pixi run fe-check" >&2; exit 1; }

# NIXL_PREFIX / NIXL_PLUGIN_DIR / LD_LIBRARY_PATH / UCX_TLS, all derived from the repo and
# $TOOLS_DIR locations. Not optional: without NIXL_PLUGIN_DIR the agent comes up with no UCX
# plugin. cn-env.sh fails loudly (pointing at notes-setup.md section 3) when nixl is absent.
# shellcheck source=../../scripts/cn-env.sh
source "$SR_DIR/scripts/cn-env.sh"

# CUDA forward compatibility. The engine is built against CUDA 13 (the pixi platforms pin
# __cuda=13; libcudart.so.13, librmm/libcudf cuda13), which needs an r580+ driver. This box runs
# the r570 kernel driver (max CUDA 12.8), so rmm's first cudaGetDeviceCount fails the CN with
# `cudaErrorInsufficientDriver`. /usr/local/cuda/compat ships the forward-compat user-mode
# driver (libcuda.so.580.x) that pairs a CUDA 13 runtime with the older kernel driver -- it must
# come FIRST so its libcuda.so.1 wins over the system one. Supported because these are
# data-center GPUs (A100-SXM4); on a box whose driver is already r580+, this directory is
# absent and the guard below simply skips it.
CUDA_COMPAT=${CUDA_COMPAT:-/usr/local/cuda/compat}
COMPAT_PREFIX=""
if [ -e "$CUDA_COMPAT/libcuda.so.1" ]; then
    COMPAT_PREFIX="$CUDA_COMPAT:"
    echo "using CUDA forward-compat driver from $CUDA_COMPAT"
fi

# cn-env.sh already put the engine .so, the pixi env's lib/ (GLIBCXX_3.4.31 for libcudf and
# the extension) and the nixl/UCX libdirs on LD_LIBRARY_PATH; only the compat driver must be
# prepended here, after the fact, so its libcuda.so.1 wins over the system one.
[ -n "$COMPAT_PREFIX" ] && export LD_LIBRARY_PATH="${COMPAT_PREFIX}${LD_LIBRARY_PATH}"
export SIRIUS_EXCHANGE_STAGING_BYTES=${SIRIUS_EXCHANGE_STAGING_BYTES:-$STAGING}

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
