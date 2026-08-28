#!/usr/bin/env bash
# NUMA-pinned bring-up of 1 FE + 4 Sirius GPU compute nodes on the 4x GB200 box.
#
# This is benchmarks/cluster8.sh with three additions, all of which exist because this box has
# four GPUs whose HBM is exposed as CPU-less NUMA nodes:
#
#   1. every CN is wrapped in `numactl --physcpubind=<its socket's cpus> --membind=<its socket>`,
#      so a CN's host allocations can never land on a GPU HBM node (2/10/18/26). Unpinned CNs run
#      with Mems_allowed_list = "0-2,10,18,26" -- GPU0's HBM is in their allowed set today.
#      The FE gets the same HBM exclusion via `--membind=<all CPU-bearing nodes>` with NO cpubind,
#      so it keeps its float across both sockets while still being unable to allocate into HBM.
#   2. a real preflight: GPU count, both binaries, numactl, every port (including the shared-data
#      FE's StarMgr port 6090), the GPU->socket mapping, and -- most importantly -- an assertion
#      that each configured membind node actually has CPUs, which is what makes "never membind an
#      HBM node" a mechanical guarantee instead of a comment.
#   3. all tunables live in ./engine-a.env, which documents the arithmetic behind each one.
#
# Usage:  ./configs/gb200-4gpu/cluster4-numa.sh
#         SCALE_FACTOR=1000 ./configs/gb200-4gpu/cluster4-numa.sh
#         CPU_SPLIT=disjoint ./configs/gb200-4gpu/cluster4-numa.sh
#         HOST_MEM=112GiB GPU_MEM=128GiB STAGING=32GiB ./configs/gb200-4gpu/cluster4-numa.sh
#
# SCALE_FACTOR picks the memory/timeout preset in engine-a.env (100, 500, 1000, 3000, 10000).
# Explicit GPU_MEM / STAGING / HOST_MEM still win. Switching scale factor is a relaunch, not a
# rebuild — the CN binary does not bake in the dataset.
#
# Run it in its own terminal or as its own background task -- never chained behind `&` inside
# another shell command, or the cluster dies with that shell.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)      # configs/gb200-4gpu
SR_DIR=$(cd "$HERE/../.." && pwd)                       # experimental/starrocks

die() { echo "cluster4-numa: $*" >&2; exit 1; }

# --- configuration ----------------------------------------------------------------------------
# engine-a.env FIRST: it sets TOOLS_DIR and UCX_TLS, and cn-env.sh only fills those in when they
# are unset (${X:-default}), so an operator/config value wins. Reversing this order would silently
# discard the UCX_TLS choice.
# shellcheck source=./engine-a.env
source "$HERE/engine-a.env"
# NIXL_PREFIX / NIXL_PLUGIN_DIR / NIXL_NO_STUBS_FALLBACK / LD_LIBRARY_PATH (engine .so + pixi env
# lib + nixl + UCX). Fails loudly when nixl is absent rather than continuing misconfigured.
# shellcheck source=../../scripts/cn-env.sh
source "$SR_DIR/scripts/cn-env.sh"

# Same precedence rule as cluster8.sh: an explicit SIRIUS_EXCHANGE_STAGING_BYTES beats STAGING.
# Unset means NO arena is built at all and every remote exchange fails, so it is never left unset.
export SIRIUS_EXCHANGE_STAGING_BYTES=${SIRIUS_EXCHANGE_STAGING_BYTES:-$STAGING}

# --gpu-device is turned into CUDA_VISIBLE_DEVICES by the CN, but an ALREADY-EXPORTED
# CUDA_VISIBLE_DEVICES wins and --gpu-device is merely warned about -- which would silently land
# all four CNs on the same GPU. Clear it so --gpu-device is authoritative.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "cluster4-numa: unsetting inherited CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'" \
         "(it would override --gpu-device and collapse all CNs onto one GPU)" >&2
    unset CUDA_VISIBLE_DEVICES
fi

CN_BIN=$SR_DIR/target/release/sirius-starrocks-cn
FE_BIN=$SR_DIR/starrocks/output/fe/bin/start_fe.sh
ENGINE_DIR_PREFIX=${ENGINE_DIR_PREFIX:-.cn}

read -r -a GPUS  <<< "$CN_GPU"
read -r -a NODES <<< "$CN_NODE"
read -r -a CPUS  <<< "${CN_CPUS:-}"

# --- preflight --------------------------------------------------------------------------------
# Everything below refuses to start rather than starting degraded. A half-configured cluster is
# worse than no cluster: it still answers queries, and the benchmark silently measures it.

[ -x "$CN_BIN" ] || die "no CN binary at $CN_BIN -- run: pixi run cn-build"
[ -x "$FE_BIN" ] || die "no packaged FE at $FE_BIN -- run: pixi run fe-check"

[ "${#GPUS[@]}"  -ge "$NUM_CNS" ] || die "CN_GPU lists ${#GPUS[@]} entries, need $NUM_CNS"
[ "${#NODES[@]}" -ge "$NUM_CNS" ] || die "CN_NODE lists ${#NODES[@]} entries, need $NUM_CNS"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found -- cannot verify GPU count"
# `set -o pipefail` is on: if nvidia-smi exits non-zero the whole pipeline does, and a bare
# assignment would take `set -e` down with stderr already swallowed and no diagnostic at all.
# Force the failure into avail=0 so the check below reports it properly.
avail=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l) || avail=0
[[ $avail =~ ^[0-9]+$ ]] || avail=0
[ "$avail" -ge "$NUM_CNS" ] ||
    die "asked for $NUM_CNS CNs but only $avail GPUs are visible (0 means nvidia-smi itself failed)"

# CN_GPU entries are used unquoted as `-i "$gpu"` and as --gpu-device; a non-numeric or
# out-of-range entry otherwise surfaces as an opaque nvidia-smi failure much later.
for i in $(seq 0 $((NUM_CNS - 1))); do
    [[ ${GPUS[$i]} =~ ^[0-9]+$ ]] || die "CN$i: CN_GPU entry '${GPUS[$i]}' is not a GPU index"
    [ "${GPUS[$i]}" -lt "$avail" ] ||
        die "CN$i: CN_GPU entry '${GPUS[$i]}' is out of range -- only $avail GPUs are visible"
done

PIN=1
if [ "$CPU_SPLIT" = none ]; then
    PIN=0
    echo "cluster4-numa: WARNING CPU_SPLIT=none -- no numactl. Host pages may land on a GPU HBM" \
         "NUMA node (2/10/18/26) and eat the HBM of a GPU a CN is computing on." >&2
else
    command -v numactl >/dev/null 2>&1 ||
        die "numactl not found (expected /usr/bin/numactl) -- required for CPU_SPLIT=$CPU_SPLIT.
     Re-run with CPU_SPLIT=none to launch unpinned, but read the HBM warning in engine-a.env first."
    [ "${#CPUS[@]}" -ge "$NUM_CNS" ] || die "CN_CPUS lists ${#CPUS[@]} entries, need $NUM_CNS"
fi

# "0-3,8,10-11" -> "0 1 2 3 8 10 11"
expand_cpulist() {
    local spec=$1 part lo hi
    for part in ${spec//,/ }; do
        if [[ $part == *-* ]]; then
            lo=${part%%-*}; hi=${part##*-}
            [[ $lo =~ ^[0-9]+$ && $hi =~ ^[0-9]+$ ]] || return 1
            seq "$lo" "$hi"
        else
            [[ $part =~ ^[0-9]+$ ]] || return 1
            echo "$part"
        fi
    done
}

# Comma list of every NUMA node that actually has CPUs -- i.e. every node that is real system
# memory rather than GPU HBM. Used for the FE's membind, so that list is derived from the hardware
# instead of hardcoded as "0,1". `node 0 cpus: 0 1 2 ...` has NF > 3; `node 2 cpus:` has NF == 3.
cpu_bearing_nodes() {
    numactl --hardware |
        awk '/^node [0-9]+ cpus:/ && NF > 3 { n = (n == "" ? $2 : n "," $2) } END { print n }'
}

# The "CPU Affinity" column of `nvidia-smi topo -m` for one GPU, e.g. "0-71". Index-free on
# purpose: the matrix cells are X/NV18/SYS/PIX and never numeric, so the FIRST numeric-looking
# field in the GPU's row is its CPU Affinity. (The header row carries an extra leading empty
# field and the data rows carry a stray empty field before "GPU NUMA ID", so column-index
# arithmetic against the header is not safe here.) Prints nothing if it cannot parse.
gpu_cpu_affinity() {
    nvidia-smi topo -m 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' |
        awk -F'\t' -v gg="GPU$1" '
            $1 == gg {
                for (i = 2; i <= NF; i++)
                    if ($i ~ /^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$/) { print $i; exit }
            }'
}

# THE HBM INTERLOCK. NUMA nodes 2, 10, 18 and 26 on this box have zero CPUs and 188,416 MB each --
# they are the four GPUs' HBM. A --membind onto one of them consumes the HBM of a GPU that a CN is
# computing on. Rather than hardcoding the forbidden list (which would rot if the box changed), we
# assert the positive property that actually matters: a valid membind target is a node that HAS
# CPUs. Every HBM node fails that test by construction.
if [ "$PIN" = 1 ]; then
    for i in $(seq 0 $((NUM_CNS - 1))); do
        node=${NODES[$i]}
        [[ $node =~ ^[0-9]+$ ]] || die "CN$i: CN_NODE entry '$node' is not a NUMA node number"

        node_cpus=$(numactl --hardware | sed -n "s/^node $node cpus: *//p")
        [ -n "$node_cpus" ] ||
            die "CN$i: NUMA node $node has NO CPUs -- it is GPU HBM, not system memory.
     --membind onto it would consume a GPU's HBM. Valid membind targets on this box are 0 and 1."

        # cpubind and membind must agree, or the CN runs its threads on one socket while all its
        # memory is allocated on the other -- every access becomes a cross-socket hop.
        want=$(expand_cpulist "${CPUS[$i]}") || die "CN$i: unparseable CN_CPUS entry '${CPUS[$i]}'"
        for c in $want; do
            grep -qw "$c" <<< "$node_cpus" ||
                die "CN$i: cpu $c (from CN_CPUS='${CPUS[$i]}') is not on NUMA node $node.
     cpubind and membind disagree; every memory access would be a cross-socket hop.
     Node $node has cpus: $node_cpus"
        done

        # ...and the pair (cpubind, membind) must also be the socket the GPU is actually attached
        # to. Without this, a transposed CN_NODE/CN_CPUS (e.g. "1 1 0 0") passes every check above
        # and silently runs all four CNs on the socket FARTHEST from their GPU, turning every
        # host-staging copy into a cross-socket hop. Parse failure only warns -- a topology parser
        # that refuses to launch the cluster would be worse than the gap it closes.
        gpu_cpus=$(gpu_cpu_affinity "${GPUS[$i]}") || gpu_cpus=""
        if [ -z "$gpu_cpus" ]; then
            echo "cluster4-numa: WARNING could not read GPU ${GPUS[$i]}'s CPU-affinity column from" \
                 "'nvidia-smi topo -m'; skipping the GPU<->socket check for CN$i." >&2
        elif ! gpu_cpu_list=$(expand_cpulist "$gpu_cpus"); then
            echo "cluster4-numa: WARNING unparseable CPU affinity '$gpu_cpus' for GPU ${GPUS[$i]};" \
                 "skipping the GPU<->socket check for CN$i." >&2
        else
            for c in $want; do
                grep -qw "$c" <<< "$gpu_cpu_list" ||
                    die "CN$i: cpu $c (from CN_CPUS='${CPUS[$i]}') is NOT in GPU ${GPUS[$i]}'s CPU
     affinity '$gpu_cpus'. This CN would run on the socket farthest from its own GPU and every
     host-staging copy would become a cross-socket hop. Check that CN_GPU, CN_NODE and CN_CPUS
     are index-aligned (GPU0/GPU1 -> node 0 -> cpus 0-71; GPU2/GPU3 -> node 1 -> cpus 72-143)."
            done
        fi
    done
fi

# Refuse to start if any port we need is already listening. This is what stops a second cluster
# from being launched on top of a running one -- the FE keys a node by (advertise_host,
# heartbeat_port) and the nixl agent is named {advertise_host}:{brpc_port}, so a port collision is
# an IDENTITY collision, not just a bind failure.
#
# Read /proc/net/tcp{,6} directly rather than shelling out to `ss`: no iproute2 dependency, no
# output-format drift, and it sees sockets owned by every user (which matters -- the process we
# are trying not to collide with may not be ours). st == 0A is TCP_LISTEN; the port is the hex
# suffix of the local_address field. This is a pure read; it never binds anything.
scan_listening_ports() {
    local f sl local_addr rem st rest hex
    for f in /proc/net/tcp /proc/net/tcp6; do
        [ -r "$f" ] || continue
        while read -r sl local_addr rem st rest; do
            [ "$st" = "0A" ] || continue
            hex=${local_addr##*:}
            [[ $hex =~ ^[0-9A-Fa-f]+$ ]] || continue
            printf '%d\n' "$((16#$hex))"
        done < <(tail -n +2 "$f")
    done
}

declare -A BOUND=()
while read -r p; do
    [ -n "$p" ] && BOUND["$p"]=1
done < <(scan_listening_ports)

# PORT_STRIDE is operator-overridable (engine-a.env uses ${PORT_STRIDE:-10}). Each CN block
# consumes 5 consecutive ports, so a stride below 5 aliases CN i's thrift/brpc/http/starlet onto
# CN i+1's heartbeat -- and the scan below only compares against ports that are ALREADY bound, so
# it would never notice the blocks colliding with each other.
[[ $PORT_STRIDE =~ ^[0-9]+$ ]] && [ "$PORT_STRIDE" -ge 5 ] ||
    die "PORT_STRIDE=$PORT_STRIDE must be an integer >= 5: each CN uses 5 consecutive ports, and
     heartbeat/brpc are the only CN identity levers (FE node key, nixl agent name), so overlapping
     blocks corrupt both registries rather than merely failing to bind."

want_ports=()
# The FE's own ports -- this script launches the FE, so they must be free too.
# 6090 is cloud_native_meta_port: engine A's packaged FE conf sets `run_mode = shared_data`
# (starrocks/output/fe/conf/fe.conf:80), which starts StarMgrServer on it. A stale StarMgr on 6090
# would otherwise pass this preflight, the launcher would fork the FE and all four CNs, the FE
# would abort on the bind, and we would be left with exactly the half-cluster this check exists to
# prevent.
for p in 6090 8030 9010 9020 9030; do want_ports+=("$p"); done
for i in $(seq 0 $((NUM_CNS - 1))); do
    base=$((PORT_BASE + i * PORT_STRIDE))
    for off in 0 1 2 3 4; do want_ports+=("$((base + off))"); done
done

busy=()
for p in "${want_ports[@]}"; do
    [ -n "${BOUND[$p]:-}" ] && busy+=("$p")
done
if [ "${#busy[@]}" -gt 0 ]; then
    die "these required ports are already bound: ${busy[*]}
     A cluster is very likely already running. Shut it down first -- do NOT launch a second one
     on top of it: FE node identity is (advertise_host, heartbeat_port) and the nixl agent name is
     {advertise_host}:{brpc_port}, so overlapping ports corrupt both registries."
fi

# The CN's own ensure_gpu_unclaimed preflight is SKIPPED whenever --gpu-memory-limit is set (which
# it always is here), so it cannot protect us. Do the check ourselves: the RMM pool is reserved in
# full at startup, so a second CN on an already-claimed GPU does not degrade -- it fails to
# allocate, or worse, succeeds and leaves no headroom.
claimed=()
for i in $(seq 0 $((NUM_CNS - 1))); do
    gpu=${GPUS[$i]}
    # `|| procs=""`: pipefail would otherwise turn an nvidia-smi hiccup into a silent `set -e`
    # exit with stderr already discarded -- zero diagnostic on a preflight whose whole job is
    # to produce diagnostics. A failed query means "cannot prove the GPU is free", which is
    # reported below rather than treated as free.
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | tr -d ' ') ||
        procs="<nvidia-smi query failed>"
    [ -n "$procs" ] && claimed+=("gpu$gpu(pids: $(tr '\n' ',' <<< "$procs" | sed 's/,$//'))")
done
if [ "${#claimed[@]}" -gt 0 ] && [ "${ALLOW_SHARED_GPUS:-0}" != 1 ]; then
    die "these GPUs already have compute processes: ${claimed[*]}
     The RMM pool is reserved in full at startup, so sharing a GPU is not a slowdown -- it is an
     allocation failure or a zero-headroom cluster. Set ALLOW_SHARED_GPUS=1 to override."
fi

# --- report -----------------------------------------------------------------------------------
echo "cluster4-numa: $NUM_CNS CNs, CPU_SPLIT=$CPU_SPLIT  SCALE_FACTOR=${SCALE_FACTOR:-100}"
echo "  GPU_MEM=$GPU_MEM  STAGING=$SIRIUS_EXCHANGE_STAGING_BYTES  HOST_MEM=$HOST_MEM" \
     " watchdog=${SIRIUS_QUERY_WATCHDOG_SECS}s  rpc=${SIRIUS_CN_RPC_TIMEOUT_SECS:-60}s"
echo "  UCX_TLS=$UCX_TLS"
printf '  %-4s %-4s %-9s %-8s %-10s %-7s %-7s %-7s %s\n' \
       CN gpu cpubind membind heartbeat thrift brpc http starlet
for i in $(seq 0 $((NUM_CNS - 1))); do
    base=$((PORT_BASE + i * PORT_STRIDE))
    printf '  %-4s %-4s %-9s %-8s %-10s %-7s %-7s %-7s %s\n' \
        "CN$i" "${GPUS[$i]}" "${CPUS[$i]:-<none>}" "${NODES[$i]}" \
        "$base" "$((base + 1))" "$((base + 2))" "$((base + 3))" "$((base + 4))"
done

# --- launch -----------------------------------------------------------------------------------
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

# --engine-dir is relative and resolved against the CWD, and the derived config's telemetry path
# is relative too, so the cd fixes where all per-CN artifacts land -- matching cluster8.sh exactly
# so the bench harness finds .cn0/log etc. where it expects.
cd "$SR_DIR"

# The FE is deliberately NOT cpu-pinned: it is the only process that can still float across both
# sockets once all four CNs are hard-membound, and that float is what absorbs error in the
# fixed-tenant budget.
#
# It IS membound, to every CPU-bearing node. Leaving it bare would leave a ~10-20 GiB JVM with
# Mems_allowed_list = "0-2,10,18,26" -- i.e. still able to allocate into GPU0's HBM, which is the
# single exposure the CN membind exists to eliminate. `--membind=<all CPU nodes>` (no cpubind)
# excludes the HBM nodes WITHOUT costing the float: both sockets stay allowed. The node list is
# derived from the hardware, so an HBM node can never appear in it.
#
# PIN_FE=1 additionally cpubinds the FE to CN0's socket. That is strictly more restrictive -- it
# destroys the cross-socket float this section is protecting -- so set it only for a specific
# experiment.
if [ "$PIN" = 1 ]; then
    if [ "${PIN_FE:-0}" = 1 ]; then
        numactl --physcpubind="${CPUS[0]}" --membind="${NODES[0]}" -- "$FE_BIN" --logconsole &
    else
        FE_NODES=$(cpu_bearing_nodes)
        [ -n "$FE_NODES" ] ||
            die "no NUMA node reports any CPUs -- refusing to membind the FE (see the HBM interlock)"
        numactl --membind="$FE_NODES" -- "$FE_BIN" --logconsole &
    fi
else
    "$FE_BIN" --logconsole &
fi
pids+=("$!")

for i in $(seq 0 $((NUM_CNS - 1))); do
    base=$((PORT_BASE + i * PORT_STRIDE))
    cn_args=(
        --gpu-device        "${GPUS[$i]}"
        --heartbeat-port    "$base"
        --thrift-port       "$((base + 1))"
        --brpc-port         "$((base + 2))"
        --http-port         "$((base + 3))"
        --starlet-port      "$((base + 4))"
        --gpu-memory-limit  "$GPU_MEM"
        --host-memory-limit "$HOST_MEM"
        --engine-dir        "$ENGINE_DIR_PREFIX$i"
    )
    if [ "$PIN" = 1 ]; then
        # numactl execs the CN rather than forking, so $! is the CN's own pid and the cleanup
        # trap kills the CN itself, not a wrapper.
        numactl --physcpubind="${CPUS[$i]}" --membind="${NODES[$i]}" -- "$CN_BIN" "${cn_args[@]}" &
    else
        "$CN_BIN" "${cn_args[@]}" &
    fi
    pids+=("$!")
    echo "CN$i gpu=${GPUS[$i]} node=${NODES[$i]} cpus=${CPUS[$i]:-<none>}" \
         "heartbeat=$base brpc=$((base + 2)) pid=${pids[-1]}"
done

echo "FE + $NUM_CNS CNs launched; each CN self-registers with the FE on :9030"
echo "verify pinning took:  grep Mems_allowed_list /proc/<cn-pid>/status   ->  must read 0 or 1,"
echo "                      NOT '0-2,10,18,26' (that means the GPU HBM nodes are still allowed)"
echo "                      the FE must read the CPU-node list (e.g. '0-1'), also never 0-2,10,18,26"
# Any child exiting means the cluster is broken -- fall through to cleanup rather than leaving a
# half-cluster that the benchmark would silently measure.
wait -n "${pids[@]}"
