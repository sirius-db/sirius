#!/usr/bin/env bash
# hwsim WS5: probe MPS CUDA_MPS_ACTIVE_THREAD_PERCENTAGE as a compute throttle.
#
# Starts a PRIVATE MPS control daemon (own pipe/log dirs, no root needed on a
# GPU in Default compute mode), measures victim_bench under several active-thread
# percentages, then shuts the daemon down. Other processes on the GPU are NOT
# affected: only clients that point CUDA_MPS_PIPE_DIRECTORY at our daemon go
# through MPS.
set -u
cd "$(dirname "$0")"

MPS_DIR=${MPS_DIR:-/tmp/hwsim-mps-$USER}
export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe"
export CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

cleanup() {
  echo quit | nvidia-cuda-mps-control 2>/dev/null
  rm -rf "$MPS_DIR"
}
trap cleanup EXIT

nvidia-cuda-mps-control -d || { echo "failed to start MPS daemon"; exit 1; }
echo "MPS daemon started (pipe: $CUDA_MPS_PIPE_DIRECTORY)"

SECONDS_PER_RUN=${SECONDS_PER_RUN:-1.2}

for pct in 100 75 50 25; do
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct \
  CUDA_MODULE_LOADING=EAGER \
  timeout -s KILL 40 ./victim_bench --victim fma --seconds "$SECONDS_PER_RUN" \
    | grep RESULT | sed "s/^/[mps pct=$pct] /"
done
# Cross-talk: memory-bound victim under 25% active threads.
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 \
CUDA_MODULE_LOADING=EAGER \
timeout -s KILL 40 ./victim_bench --victim saxpy --seconds "$SECONDS_PER_RUN" \
  | grep RESULT | sed "s/^/[mps pct=25] /"

# Bonus: does cross-process smsteal behave spatially under MPS?
./throttle_compute --mode smsteal --fraction 0.50 --duration 20 >/dev/null &
TPID=$!
sleep 1
CUDA_MODULE_LOADING=EAGER timeout -s KILL 40 ./victim_bench --victim fma \
  --seconds "$SECONDS_PER_RUN" | grep RESULT | sed "s/^/[mps xproc-smsteal:0.50] /"
kill "$TPID" 2>/dev/null; wait "$TPID" 2>/dev/null

echo "quitting MPS daemon"
