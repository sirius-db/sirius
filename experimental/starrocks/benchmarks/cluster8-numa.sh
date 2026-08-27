#!/usr/bin/env bash
# cluster8.sh + NUMA pinning: each CN is launched under `numactl`, confined to the socket that
# owns its GPU.
#
# Why a separate script: cluster8.sh is the known-good launcher and stays byte-identical, so an
# A/B is just a change of script name.
#
# What the engine already does on its own: it binds each CN's pinned host arena to its GPU's
# NUMA node (`numa_alloc_onnode`) and pins the 4 GPU pipeline threads to that socket's cores.
# What it does NOT reach: the CN's own tokio/brpc/thrift/nixl/UCX threads and every allocation
# the Rust side makes. `numactl` covers all of it, including anything a future engine change
# spawns. (The engine's scan_manager/task_creator/downgrade pools are pinned separately, from
# the `cpu_affinity` keys the CN now writes into its derived YAML — that path works without this
# script, and this script works without that path. Both agree on the same socket.)
#
# Usage:  ./benchmarks/cluster8-numa.sh
#         NUM_CNS=4 GPU_MEM=140GiB ./benchmarks/cluster8-numa.sh
#         NUMA_MEM_POLICY=membind ./benchmarks/cluster8-numa.sh   # strict; see below
#         DRY_RUN=1 NUM_CNS=4 ./benchmarks/cluster8-numa.sh        # print the mapping, start nothing
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

# `preferred` (default) allocates on the CN's own node and silently falls back to the other
# socket when it runs out. `membind` turns that overshoot into an allocation failure instead --
# use it only to prove a result is really NUMA-local, never for a capacity run. On this box
# HOST_MEM x 2 CNs must fit in one socket's ~490 GB for `membind` to be safe.
NUMA_MEM_POLICY=${NUMA_MEM_POLICY:-preferred}
case "$NUMA_MEM_POLICY" in
    preferred|membind) ;;
    *) echo "NUMA_MEM_POLICY must be 'preferred' or 'membind', got '$NUMA_MEM_POLICY'" >&2
       exit 1 ;;
esac

CN_BIN=$SR_DIR/target/release/sirius-starrocks-cn
FE_BIN=$SR_DIR/starrocks/output/fe/bin/start_fe.sh

[ -x "$CN_BIN" ] || { echo "no CN binary at $CN_BIN -- run: pixi run cn-build" >&2; exit 1; }
[ -x "$FE_BIN" ] || { echo "no packaged FE at $FE_BIN -- run: pixi run fe-check" >&2; exit 1; }
command -v numactl >/dev/null || {
    echo "numactl not found -- install it, or use ./benchmarks/cluster8.sh unpinned" >&2; exit 1; }

# NIXL_PREFIX / NIXL_PLUGIN_DIR / LD_LIBRARY_PATH (engine .so + pixi env lib + nixl + UCX) /
# UCX_TLS, all derived from the repo and $TOOLS_DIR locations; fails loudly when nixl is
# absent rather than continuing misconfigured.
# shellcheck source=../scripts/cn-env.sh
source "$SR_DIR/scripts/cn-env.sh"

export SIRIUS_EXCHANGE_STAGING_BYTES=${SIRIUS_EXCHANGE_STAGING_BYTES:-$STAGING}

avail=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$avail" -ge "$NUM_CNS" ] || {
    echo "asked for $NUM_CNS CNs but only $avail GPUs are visible" >&2; exit 1; }

# ---------------------------------------------------------------------------------------------
# GPU -> NUMA node, straight from sysfs.
#
# The GPU set is vendor 0x10de with a display/3D class (0x0300xx VGA, 0x0302xx 3D controller);
# the NVIDIA-vendored host bridges on this platform are 0x0604xx and must not be counted, or
# every ordinal would shift. Sorting the PCI addresses as strings sorts them numerically (sysfs
# renders them fixed-width), which is the order CUDA enumerates devices in: CUDA_DEVICE_ORDER
# defaults to FASTEST_FIRST, and on a homogeneous box that ties back to PCI bus order.
#
# This mirrors src/gpu_affinity.rs, which resolves the same mapping for the engine's YAML.
# ---------------------------------------------------------------------------------------------
gpu_bdfs=()
for dev in /sys/bus/pci/devices/*; do
    [ "$(cat "$dev/vendor" 2>/dev/null)" = "0x10de" ] || continue
    case "$(cat "$dev/class" 2>/dev/null)" in
        0x0300*|0x0302*) gpu_bdfs+=("$(basename "$dev")") ;;
    esac
done
IFS=$'\n' read -r -d '' -a gpu_bdfs < <(printf '%s\n' "${gpu_bdfs[@]}" | sort && printf '\0')

# The only nodes that may ever be handed to numactl. On this box that is `0-1`: the four
# 184 GiB GPU-HBM domains (2/10/18/26) and the 28 empty ones are cpuless and MUST NOT be bound,
# for cpus or for memory. Reading has_cpu rather than hardcoding keeps that true elsewhere too.
cpu_nodes=$(cat /sys/devices/system/node/has_cpu)
node_has_cpus() {  # $1 = node id
    local n
    for n in $(echo "$cpu_nodes" | tr ',' ' '); do
        case "$n" in
            *-*) [ "$1" -ge "${n%-*}" ] && [ "$1" -le "${n#*-}" ] && return 0 ;;
            *)   [ "$1" -eq "$n" ] && return 0 ;;
        esac
    done
    return 1
}

# Resolve every node up front: a half-pinned cluster would measure as an imbalance.
nodes=()
for i in $(seq 0 $((NUM_CNS - 1))); do
    bdf=${gpu_bdfs[$i]:-}
    [ -n "$bdf" ] || { echo "no sysfs PCI entry for GPU $i (found ${#gpu_bdfs[@]} GPUs)" >&2; exit 1; }
    node=$(cat "/sys/bus/pci/devices/$bdf/numa_node")
    [ "$node" -ge 0 ] || { echo "GPU $i ($bdf) reports numa_node=$node" >&2; exit 1; }
    node_has_cpus "$node" || {
        echo "GPU $i ($bdf) maps to NUMA node $node, which has no CPUs (has_cpu=$cpu_nodes)." >&2
        echo "Refusing to bind a cpuless/HBM domain." >&2
        exit 1; }
    nodes+=("$node")
    echo "GPU$i $bdf -> NUMA node $node (cpus $(cat "/sys/bus/pci/devices/$bdf/local_cpulist"))"
done

if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY_RUN: would launch the FE unpinned and each CN under"
    for i in $(seq 0 $((NUM_CNS - 1))); do
        echo "  numactl --cpunodebind=${nodes[$i]} --$NUMA_MEM_POLICY=${nodes[$i]} $CN_BIN --gpu-device $i ..."
    done
    exit 0
fi

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
# The FE is deliberately left unpinned: it talks to every CN, so either socket is arbitrary.
"$FE_BIN" --logconsole &
pids+=("$!")

for i in $(seq 0 $((NUM_CNS - 1))); do
    base=$((PORT_BASE + i * PORT_STRIDE))
    node=${nodes[$i]}
    # numactl execvp()s the target, so $! is still the CN's pid -- pids[]/cleanup() and
    # `pkill -f '[s]irius-starrocks-cn'` keep working exactly as with cluster8.sh.
    numactl --cpunodebind="$node" "--$NUMA_MEM_POLICY=$node" \
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
    echo "CN$i gpu=$i numa=$node($NUMA_MEM_POLICY) heartbeat=$base brpc=$((base + 2)) pid=${pids[-1]}"
done

echo "FE + $NUM_CNS CNs launched (NUMA-pinned); each CN self-registers with the FE on :9030"
echo "verify:  grep Cpus_allowed_list /proc/<cn-pid>/task/*/status | sort -u"
echo "         numastat -p <cn-pid>"
# Any child exiting means the cluster is broken -- fall through to cleanup rather than
# leaving a half-cluster that the benchmark would silently measure.
wait -n "${pids[@]}"
