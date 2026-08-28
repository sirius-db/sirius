#!/usr/bin/env bash
# cluster8.sh variant for the pinned-table benchmark: launches 1 FE + NUM_CNS CNs,
# one per GPU, each with a full --sirius-config (from gen-config.sh) instead of the
# memory carve-out flags — that is the only way to reach the pin-compression keys.
#
# BLOCKS for the cluster's lifetime, and its EXIT/INT trap tears the cluster down
# with the shell — give it its own terminal or background task.
#
# Env: NUM_CNS (default 2), STAGING (default 32GiB), CONFIG_DIR (default ./generated)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SR_DIR="$(cd "$HERE/../.." && pwd)"
FE_BIN=$SR_DIR/starrocks/output/fe/bin/start_fe.sh
CN_BIN=$SR_DIR/target/release/sirius-starrocks-cn
NUM_CNS=${NUM_CNS:-2}
CONFIG_DIR=${CONFIG_DIR:-$HERE/generated}
PORT_BASE=9100 PORT_STRIDE=10

# An exported CUDA_VISIBLE_DEVICES beats --gpu-device and collapses every CN
# onto one GPU (only warn!-ed about) — clear it unconditionally.
unset CUDA_VISIBLE_DEVICES
export SIRIUS_QUERY_WATCHDOG_SECS=${SIRIUS_QUERY_WATCHDOG_SECS:-300}
export SIRIUS_EXCHANGE_STAGING_BYTES=${STAGING:-32GiB}

[ -x "$FE_BIN" ] || { echo "no packaged FE at $FE_BIN (pixi run -e fe fe-build)" >&2; exit 1; }
[ -x "$CN_BIN" ] || { echo "no CN at $CN_BIN (pixi run cn-build)" >&2; exit 1; }
for i in $(seq 0 $((NUM_CNS - 1))); do
  [ -f "$CONFIG_DIR/cn$i.yaml" ] || { echo "missing $CONFIG_DIR/cn$i.yaml (run gen-config.sh)" >&2; exit 1; }
done

pids=()
cleanup() { status=$?; kill "${pids[@]}" 2>/dev/null || true; wait "${pids[@]}" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT INT TERM

# shellcheck source=../../scripts/cn-env.sh
source "$SR_DIR/scripts/cn-env.sh"
cd "$SR_DIR"
"$FE_BIN" --logconsole & pids+=("$!")
for i in $(seq 0 $((NUM_CNS - 1))); do
  base=$((PORT_BASE + i * PORT_STRIDE))
  "$CN_BIN" \
    --gpu-device "$i" \
    --heartbeat-port "$base" \
    --thrift-port "$((base + 1))" \
    --brpc-port "$((base + 2))" \
    --http-port "$((base + 3))" \
    --starlet-port "$((base + 4))" \
    --sirius-config "$CONFIG_DIR/cn$i.yaml" \
    --engine-dir ".cn$i" &
  pids+=("$!")
  echo "CN$i gpu=$i heartbeat=$base brpc=$((base + 2)) pid=${pids[-1]}"
done
echo "FE + $NUM_CNS CNs launched; each CN self-registers with the FE on :9030"
wait -n "${pids[@]}"
cleanup
