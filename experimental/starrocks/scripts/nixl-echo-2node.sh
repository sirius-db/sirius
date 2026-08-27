#!/usr/bin/env bash
# Drives the two-host GPU buffer echo test (`nixl_transport::nixl_echo::two_node_gpu_echo`):
# a GPU buffer goes from one machine to another over nixl and comes back, byte-verified on both.
#
#   ./scripts/nixl-echo-2node.sh
#   ECHO_HOST=... ORIGIN_HOST=... SIZES=1048576 ./scripts/nixl-echo-2node.sh
#
# Assumes the repo is on a filesystem both hosts see (here: NFS /home), so one build serves both.
# Each side gets one GPU via CUDA_VISIBLE_DEVICES, the invariant `ArenaRegion::device_id() == 0`
# and production's one-CN-per-GPU rely on.
#
# UCX_NET_DEVICES is not optional on these boxes: left to itself UCX advertises the DPU interface
# (100.127.x), which is not routable between nodes, and every connection attempt stalls until the
# TCP timeout. Pin it to a fabric that both hosts can reach.
set -euo pipefail

ORIGIN_HOST=${ORIGIN_HOST:-presto-gb200-gcn-17}
ECHO_HOST=${ECHO_HOST:-presto-gb200-gcn-18}
# Interface carrying the UCX wireup and the control socket. The 400G RoCE planes (enp3s0np0,
# enP2p3s0np0, enP16p3s0np0, enP18p3s0np0) are point-to-point /31s but routed host-to-host.
IFACE=${IFACE:-enp3s0np0}
# What UCX may use, if that differs from the interface the control socket is resolved on — an
# RDMA run wants the HCA (`mlx5_0:1`) alongside the Ethernet device that carries UCP wireup.
NET_DEVICES=${NET_DEVICES:-$IFACE}
ORIGIN_GPU=${ORIGIN_GPU:-0}
ECHO_GPU=${ECHO_GPU:-0}
PORT=${PORT:-18090}

SIZES=${SIZES:-1048576,16777216,268435456}
ITERATIONS=${ITERATIONS:-10}
WARMUP=${WARMUP:-3}
STAGING=${STAGING:-2GiB}
GPU_MEM=${GPU_MEM:-8GiB}
# cuda_copy lets UCX recognise VRAM pointers, cuda_ipc carries the payload (over NVLink, and on
# an NVL72 that domain spans hosts), tcp carries the UCP wireup: cuda_ipc has no active-message
# capability, so without an AM-capable transport the endpoint cannot be created at all.
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}
# Extra `export K=V` lines injected into both roles, for one-off UCX knobs.
EXTRA_ENV=${EXTRA_ENV:-}

SR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR=${LOG_DIR:-/tmp/nixl-echo-2node}
mkdir -p "$LOG_DIR"

cd "$SR_DIR"

# --- build once ------------------------------------------------------------------------------
# /usr/bin ahead of the pixi env: nixl-sys's build.rs hardcodes `.compiler("g++")`, ignoring CXX,
# and the conda g++ lacks the multiarch include path the host glibc headers need.
echo "==> building the test binary"
build_output=$(pixi run bash -c '
    set -euo pipefail
    source scripts/cn-env.sh
    export PATH=/usr/bin:$PATH
    cargo test --release -p sirius-starrocks-cn --no-run 2>&1' | tee /dev/stderr)

TEST_BIN=$(sed -n 's|.*Executable unittests src/lib\.rs (\(.*\))|\1|p' <<<"$build_output" | tail -1)
[ -n "$TEST_BIN" ] || { echo "could not find the test binary in cargo's output" >&2; exit 1; }
TEST_BIN=$SR_DIR/$TEST_BIN
echo "==> test binary: $TEST_BIN"

# --- resolve the echo host's address on the chosen fabric ------------------------------------
ECHO_IP=$(ssh -o BatchMode=yes "$ECHO_HOST" "ip -4 -br addr show dev $IFACE" \
          | awk '{print $3}' | cut -d/ -f1)
[ -n "$ECHO_IP" ] || { echo "no IPv4 address on $ECHO_HOST:$IFACE" >&2; exit 1; }
echo "==> echo host $ECHO_HOST reachable at $ECHO_IP on $IFACE"

# Common environment for both roles. The test reads all of its knobs from here.
common_env() {
    cat <<EOF
cd $SR_DIR
source scripts/cn-env.sh
export UCX_TLS='$UCX_TLS'
export UCX_NET_DEVICES=$NET_DEVICES
export NIXL_ECHO_CONTROL=$ECHO_IP:$PORT
export NIXL_ECHO_SIZES=$SIZES
export NIXL_ECHO_ITERATIONS=$ITERATIONS
export NIXL_ECHO_WARMUP=$WARMUP
export NIXL_ECHO_GPU_MEMORY_LIMIT=$GPU_MEM
export SIRIUS_EXCHANGE_STAGING_BYTES=$STAGING
export RUST_BACKTRACE=1
# UCX_PROTO_INFO=y prints the transport UCP picked per operation — the only direct way to see
# whether the payload rode cuda_ipc (NVLink) or fell back to a host-staged path.
export UCX_PROTO_INFO=${UCX_PROTO_INFO:-n}
export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-warn}
$EXTRA_ENV
EOF
}

RUN_TEST="$TEST_BIN two_node_gpu_echo --ignored --nocapture --test-threads=1"

cleanup() {
    # Bracketed so the pattern cannot match the ssh session's own command line.
    ssh -o BatchMode=yes "$ECHO_HOST" "pkill -f '[s]irius_starrocks_cn.*two_node_gpu_echo' || true" 2>/dev/null || true
    ssh -o BatchMode=yes "$ORIGIN_HOST" "pkill -f '[s]irius_starrocks_cn.*two_node_gpu_echo' || true" 2>/dev/null || true
}
trap cleanup EXIT

# --- echo side first: it listens, so it must be up before the origin dials --------------------
echo "==> starting echo   on $ECHO_HOST   GPU $ECHO_GPU"
ssh -o BatchMode=yes "$ECHO_HOST" "$(common_env)
export NIXL_ECHO_ROLE=echo
export CUDA_VISIBLE_DEVICES=$ECHO_GPU
export NIXL_ECHO_NVLINK_GPU=$ECHO_GPU
nohup $RUN_TEST > /tmp/nixl-echo-echo.log 2>&1 < /dev/null &
echo started" >/dev/null

# --- origin side: dials, drives every phase, owns the verdict ---------------------------------
echo "==> starting origin on $ORIGIN_HOST GPU $ORIGIN_GPU"
set +e
ssh -o BatchMode=yes "$ORIGIN_HOST" "$(common_env)
export NIXL_ECHO_ROLE=origin
export CUDA_VISIBLE_DEVICES=$ORIGIN_GPU
export NIXL_ECHO_NVLINK_GPU=$ORIGIN_GPU
$RUN_TEST" 2>&1 | tee "$LOG_DIR/origin.log"
origin_rc=${PIPESTATUS[0]}
set -e

echo
echo "==== echo host log ($ECHO_HOST) ===="
ssh -o BatchMode=yes "$ECHO_HOST" "cat /tmp/nixl-echo-echo.log" | tee "$LOG_DIR/echo.log"

echo
echo "==== origin exit=$origin_rc — logs in $LOG_DIR ===="
exit "$origin_rc"
