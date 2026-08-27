#!/usr/bin/env bash
# Launch this host's share of a two-machine engine A cluster.
#
#   ./benchmarks/cn-2host.sh 10.87.140.52 10.87.140.52          # gcn-17, also starts the FE
#   ./benchmarks/cn-2host.sh 10.87.140.53 10.87.140.52 --no-fe  # gcn-18, CNs only
#
# Refuses to start rather than starting degraded: a half-configured cluster still answers
# queries, and the benchmark silently measures it.
#
# --engine-dir is relative and resolved against this checkout, which lives on NFS (`master:/home`,
# see /proc/mounts) and is therefore THE SAME DIRECTORY on both hosts. So the per-CN engine dir is
# suffixed with the advertise host's last octet (.cn0-52 / .cn0-53): unsuffixed, both hosts' CN0
# would race on the same derived-sirius-config.yaml and the same log/ + telemetry/ trees.
set -euo pipefail

ADVERTISE=${1:?usage: cn-2host.sh <advertise-host> <fe-host> [--no-fe]}
FE_HOST=${2:?usage: cn-2host.sh <advertise-host> <fe-host> [--no-fe]}
START_FE=1; [ "${3:-}" = "--no-fe" ] && START_FE=0

SR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SR_DIR"

PER_HOST=${NUM_CNS_PER_HOST:-4}
export NUM_CNS=${NUM_CNS:-$((PER_HOST * 2))}       # total across BOTH hosts -- drives warmup
GPU_MEM=${GPU_MEM:-140GiB}
HOST_MEM=${HOST_MEM:-160GiB}

# Index-aligned with GPU ordinal. ONLY 0 AND 1 ARE EVER VALID for --membind: nodes 2/10/18/26
# are GPU HBM with zero CPUs, and binding host pages there eats the HBM of a GPU a CN is using.
read -r -a NODES <<< "${CN_NODE:-0 0 1 1}"
read -r -a CPUS  <<< "${CN_CPUS:-0-35 36-71 72-107 108-143}"

[ "${#NODES[@]}" -ge "$PER_HOST" ] || { echo "CN_NODE needs $PER_HOST entries" >&2; exit 1; }
[ "${#CPUS[@]}"  -ge "$PER_HOST" ] || { echo "CN_CPUS needs $PER_HOST entries" >&2; exit 1; }

for i in $(seq 0 $((PER_HOST - 1))); do
    case "${NODES[$i]}" in 0|1) ;; *)
        echo "CN$i: --membind ${NODES[$i]} is not a CPU-bearing node (HBM interlock)" >&2
        exit 1 ;;
    esac
done

. configs/gb200-4gpu/engine-a-2host.env            # UCX_*, staging, warmup, datasource pin
source scripts/cn-env.sh                            # LD_LIBRARY_PATH, nixl plugins

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "cn-2host: unsetting inherited CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'" \
         "(it would override --gpu-device and collapse all CNs onto one GPU)" >&2
    unset CUDA_VISIBLE_DEVICES
fi

# --- preflight ---------------------------------------------------------------------------------
# The header promises "refuses to start rather than starting degraded". That is only true if we
# actually check; every failure below otherwise produces a HALF cluster, which still answers
# queries and which the benchmark silently measures.
CN_BIN=target/release/sirius-starrocks-cn
FE_BIN=starrocks/output/fe/bin/start_fe.sh
[ -x "$CN_BIN" ] || { echo "cn-2host: no CN binary at $SR_DIR/$CN_BIN" >&2; exit 1; }
if [ "$START_FE" = 1 ]; then
    [ -x "$FE_BIN" ] || { echo "cn-2host: no packaged FE at $SR_DIR/$FE_BIN" >&2; exit 1; }
fi
command -v numactl >/dev/null 2>&1 || { echo "cn-2host: numactl not found" >&2; exit 1; }

# FE node identity is (advertise_host, heartbeat_port) and the nixl agent name is
# {advertise_host}:{brpc_port}, so an overlapping port block is an IDENTITY collision that corrupts
# both registries -- not a clean bind failure. Read /proc/net/tcp{,6} directly (st 0A == TCP_LISTEN):
# no iproute2 dependency, and it sees listeners owned by other users too. This is a pure read.
declare -A BOUND=()
for f in /proc/net/tcp /proc/net/tcp6; do
    [ -r "$f" ] || continue
    while read -r _sl laddr _rem st _rest; do
        [ "$st" = "0A" ] || continue
        hex=${laddr##*:}
        [[ $hex =~ ^[0-9A-Fa-f]+$ ]] || continue
        BOUND[$((16#$hex))]=1
    done < <(tail -n +2 "$f")
done

want=()
if [ "$START_FE" = 1 ]; then want+=(6090 8030 9010 9020 9030); fi   # 6090 = shared_data StarMgr
for i in $(seq 0 $((PER_HOST - 1))); do
    base=$((9100 + i * 10))
    for off in 0 1 2 3 4; do want+=("$((base + off))"); done
done
busy=()
for p in "${want[@]}"; do
    if [ -n "${BOUND[$p]:-}" ]; then busy+=("$p"); fi
done
if [ "${#busy[@]}" -gt 0 ]; then
    echo "cn-2host: required ports already bound: ${busy[*]}" >&2
    echo "  A cluster is very likely already running (G1). Shut it down first -- do NOT launch a" >&2
    echo "  second one on top of it." >&2
    exit 1
fi

# The CN's own ensure_gpu_unclaimed preflight is SKIPPED whenever --gpu-memory-limit is set, which
# this script always does -- so it cannot protect us. The RMM pool is reserved in full at startup,
# so a second CN on a claimed GPU is an allocation failure or a zero-headroom cluster, never just
# a slowdown.
claimed=()
for i in $(seq 0 $((PER_HOST - 1))); do
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$i" 2>/dev/null |
            tr -d ' ' | tr '\n' ',' | sed 's/,$//') || procs="<nvidia-smi query failed>"
    if [ -n "$procs" ]; then claimed+=("gpu$i(pids: $procs)"); fi
done
if [ "${#claimed[@]}" -gt 0 ] && [ "${ALLOW_SHARED_GPUS:-0}" != 1 ]; then
    echo "cn-2host: these GPUs already have compute processes: ${claimed[*]}" >&2
    echo "  Set ALLOW_SHARED_GPUS=1 to override." >&2
    exit 1
fi

# --- launch ------------------------------------------------------------------------------------
# The trap is armed BEFORE the first fork: an interrupt during the launch window would otherwise
# orphan the CNs already started, leaving them holding GPUs, ports 9100-9134 and FE registry
# entries -- exactly the half-cluster the preflight above exists to prevent.
pids=()
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [ "${#pids[@]}" -gt 0 ]; then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

if [ "$START_FE" = 1 ]; then
    # Membound to every CPU-bearing node, DERIVED from the hardware rather than hardcoded, so the
    # ~10-20 GiB JVM cannot allocate into GPU HBM -- the one exposure the CN membind exists to
    # close. Deliberately NOT cpubound: the FE's cross-socket float is what absorbs error once all
    # CNs are hard-pinned (cluster4-numa.sh uses PIN_FE=1 to opt out of that float).
    FE_NODES=$(numactl --hardware |
        awk '/^node [0-9]+ cpus:/ && NF > 3 { n = (n == "" ? $2 : n "," $2) } END { print n }')
    [ -n "$FE_NODES" ] ||
        { echo "cn-2host: no NUMA node reports CPUs -- refusing to membind the FE" >&2; exit 1; }
    numactl --membind="$FE_NODES" -- "$FE_BIN" --logconsole > /tmp/fe.log 2>&1 &
    pids+=($!)
    echo "FE started (membind=$FE_NODES, no cpubind) -> /tmp/fe.log"
fi

for i in $(seq 0 $((PER_HOST - 1))); do
    base=$((9100 + i * 10))
    numactl --physcpubind="${CPUS[$i]}" --membind="${NODES[$i]}" -- \
        "$CN_BIN" \
            --fe-host           "$FE_HOST" \
            --advertise-host    "$ADVERTISE" \
            --bind-host         0.0.0.0 \
            --gpu-device        "$i" \
            --heartbeat-port    "$base" \
            --thrift-port       "$((base + 1))" \
            --brpc-port         "$((base + 2))" \
            --http-port         "$((base + 3))" \
            --starlet-port      "$((base + 4))" \
            --gpu-memory-limit  "$GPU_MEM" \
            --host-memory-limit "$HOST_MEM" \
            --engine-dir        ".cn$i-${ADVERTISE##*.}" \
            > "/tmp/cn-${ADVERTISE##*.}-$i.log" 2>&1 &
    pids+=($!)
    echo "CN$i gpu=$i node=${NODES[$i]} cpus=${CPUS[$i]} ports=$base-$((base+4))" \
         "engine-dir=.cn$i-${ADVERTISE##*.} -> /tmp/cn-${ADVERTISE##*.}-$i.log"
done

wait -n "${pids[@]}"
