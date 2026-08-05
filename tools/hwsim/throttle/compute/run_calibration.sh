#!/usr/bin/env bash
# hwsim WS5: calibration matrix for the GPU compute throttlers.
# Short runs only (~1.5 s per cell, ~60 s total GPU-busy). Checks GPU idleness
# before each section; aborts a section if someone else is on the GPU.
set -u
cd "$(dirname "$0")"

SECONDS_PER_RUN=${SECONDS_PER_RUN:-1.2}
RUN_TIMEOUT=${RUN_TIMEOUT:-40}

others_on_gpu() {
  # Any compute app that is not one of our children?
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
  for p in $pids; do
    case " ${OUR_PIDS:-} " in
      *" $p "*) ;;
      *) return 0 ;;
    esac
  done
  return 1
}

require_idle() {
  # Retry for up to ~30 s: short-lived microbenchmarks (e.g. WS4's membw
  # throttler) come and go; only skip if the GPU stays busy.
  for _ in $(seq 1 10); do
    others_on_gpu || return 0
    sleep 3
  done
  echo "SKIP section '$1': GPU busy with another user's process"
  return 1
}

run() { # run <label> <cmd...>
  local label=$1; shift
  timeout -s KILL "$RUN_TIMEOUT" "$@" | grep -E "RESULT|smsteal:|greenctx:" | sed "s/^/[$label] /"
}

echo "=== baselines ==="
require_idle baselines && {
  run base ./victim_bench --victim fma --seconds "$SECONDS_PER_RUN"
  run base ./victim_bench --victim saxpy --seconds "$SECONDS_PER_RUN"
}

echo "=== in-process smsteal (SM co-residency semantics) ==="
require_idle smsteal && {
  for v in fma saxpy; do
    for f in 0.25 0.50 0.75; do
      run "inproc" ./victim_bench --victim "$v" --seconds "$SECONDS_PER_RUN" --co "smsteal:$f"
    done
  done
}

echo "=== in-process duty cycle (10 ms period) ==="
require_idle duty && {
  for v in fma saxpy; do
    for f in 0.25 0.50 0.75; do
      run "inproc" ./victim_bench --victim "$v" --seconds "$SECONDS_PER_RUN" --co "duty:$f:10"
    done
  done
}

echo "=== green contexts (victim restricted to N SMs) ==="
require_idle greenctx && {
  # fractions of 152 SMs: keep 75% -> 114, 50% -> 76, 25% -> 38
  for n in 114 76 38; do
    run "greenctx" ./victim_bench --victim fma --seconds "$SECONDS_PER_RUN" --greenctx "$n"
  done
  run "greenctx" ./victim_bench --victim saxpy --seconds "$SECONDS_PER_RUN" --greenctx 38
}

echo "=== cross-process (NO MPS: expect time-slicing, not SM sharing) ==="
require_idle crossproc && {
  for f in 0.25 0.50 0.75; do
    ./throttle_compute --mode smsteal --fraction "$f" --duration 20 >/dev/null &
    TPID=$!
    OUR_PIDS="$TPID"
    sleep 1
    run "xproc-smsteal:$f" ./victim_bench --victim fma --seconds "$SECONDS_PER_RUN"
    kill "$TPID" 2>/dev/null; wait "$TPID" 2>/dev/null
    OUR_PIDS=""
  done
  ./throttle_compute --mode duty --fraction 0.50 --duration 20 >/dev/null &
  TPID=$!
  OUR_PIDS="$TPID"
  sleep 1
  run "xproc-duty:0.50" ./victim_bench --victim fma --seconds "$SECONDS_PER_RUN"
  kill "$TPID" 2>/dev/null; wait "$TPID" 2>/dev/null
  OUR_PIDS=""
}

echo "=== done ==="
