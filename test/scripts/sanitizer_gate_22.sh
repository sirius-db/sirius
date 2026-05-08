#!/usr/bin/env bash
# Phase 22 D-12 Cluster B sanitizer gate.
#
# Runs SF1 Q11 num_gpus=2 under compute-sanitizer memcheck +
# track-stream-ordered-races=all and gates exit 0 on:
#   Cluster B (`alloc_and_peer_copy_async`) race count == 0.
#
# Cluster A (cudf+kvikio internal cross-stream gap at
# `read_column_chunks_async` / `posix_device_io`) is recorded on stdout
# as advisory only and does NOT affect the exit code (CONTEXT.md D-09).
#
# Per project memory `feedback_sanitizer_via_bash_not_mcp`,
# compute-sanitizer MUST be invoked via Bash + `timeout` (NOT MCP) on
# this host because the MCP-routed sanitizer hangs.
#
# Per project memory `feedback_test_runtime_caps`, the sanitizer-budget
# is bounded by `timeout 600` (SF1 Q11 baseline ~30s; under sanitizer
# ~10x; safety margin 2-3x). The wall-clock budget is parameterized via
# P22_TIMEOUT_SEC (default 600 seconds) so callers can dial up for
# slower hardware without editing the script.
#
# Pitfall 5 (literal filter shape): the gate is `grep -cE` matching the
# Cluster B stack frame; never `grep -v`. Cluster B count must be 0 ->
# `grep -cE <pattern>` returns 0; NOT `grep -cv <pattern>` returns N.
#
# Exit codes:
#   0 - PASS: Cluster B = 0
#   1 - FAIL: Cluster B > 0 (the Phase 22 same-stream invariant fix
#       did not land or has regressed)
#   2 - environment error: compute-sanitizer not found or unittest
#       binary missing/non-executable
#   3 - sanitizer crashed before producing log output (distinguishes
#       from "0 races detected")
# 124 - timeout fired (compute-sanitizer hung; per project memory)
#
# Environment overrides (all optional):
#   P22_SANITIZER_LOG  - log output path (default: /tmp/p22_sanitizer.log)
#   P22_UNITTEST_BIN   - path to sirius_unittest
#                        (default: build/release/extension/sirius/test/cpp/sirius_unittest)
#   P22_QUERY          - Catch2 test name to run
#                        (default: 'gpu_execution - TPC-H Query 11 parquet')
#   P22_TIMEOUT_SEC    - sanitizer wall-clock budget seconds (default: 600)

set -euo pipefail

CUDA_BIN=$(ls -1 /usr/local/cuda*/bin/compute-sanitizer 2>/dev/null | head -1)
if [[ -z "${CUDA_BIN}" ]]; then
  echo "[p22-sanitizer-gate] ERROR: compute-sanitizer not found under /usr/local/cuda*/bin/"
  echo "[p22-sanitizer-gate] HINT: install CUDA toolkit, or set CUDA_BIN env override"
  exit 2
fi

LOG="${P22_SANITIZER_LOG:-/tmp/p22_sanitizer.log}"
UNIT="${P22_UNITTEST_BIN:-build/release/extension/sirius/test/cpp/sirius_unittest}"
SF1_QUERY="${P22_QUERY:-gpu_execution - TPC-H Query 11 parquet}"
TIMEOUT_SEC="${P22_TIMEOUT_SEC:-600}"

if [[ ! -x "${UNIT}" ]]; then
  echo "[p22-sanitizer-gate] ERROR: unittest binary not found or not executable at ${UNIT}"
  echo "[p22-sanitizer-gate] HINT: build first (mcp__project-commands__run_command build)"
  echo "[p22-sanitizer-gate] HINT: or set P22_UNITTEST_BIN env override"
  exit 2
fi

# Skip running the sanitizer entirely if a pre-recorded log was supplied
# (used by negative-tests: inject a fake Cluster B frame into a copy of
# the log and re-invoke the script with P22_SANITIZER_LOG pointed at the
# tampered file). Detection is the existence of a non-empty file already
# at $LOG combined with P22_SKIP_RUN=1.
if [[ "${P22_SKIP_RUN:-0}" == "1" && -s "${LOG}" ]]; then
  echo "[p22-sanitizer-gate] P22_SKIP_RUN=1; using pre-recorded log at ${LOG}"
else
  echo "[p22-sanitizer-gate] starting compute-sanitizer on '${SF1_QUERY}'"
  echo "[p22-sanitizer-gate] cuda_bin=${CUDA_BIN}"
  echo "[p22-sanitizer-gate] unit=${UNIT}"
  echo "[p22-sanitizer-gate] log=${LOG}"
  echo "[p22-sanitizer-gate] timeout=${TIMEOUT_SEC}s"

  # Phase 21 21-VERDICT.md Section F verbatim shape (memcheck +
  # track-stream-ordered-races=all + show-backtrace + log-file +
  # print-limit 100). `|| true` because compute-sanitizer's own non-zero
  # exit signals "races / API errors found" — we gate on parsed log
  # content, not on its exit status. The only sanitizer exit we treat
  # specially is the timeout-fired status (124).
  set +e
  timeout "${TIMEOUT_SEC}" "${CUDA_BIN}" \
    --tool memcheck \
    --track-stream-ordered-races=all \
    --show-backtrace=yes \
    --launch-timeout="${TIMEOUT_SEC}" \
    --log-file "${LOG}" \
    --print-limit 100 \
    "${UNIT}" "${SF1_QUERY}"
  SAN_EXIT=$?
  set -e

  if [[ "${SAN_EXIT}" -eq 124 ]]; then
    echo "[p22-sanitizer-gate] FAIL: compute-sanitizer hung — killed at ${TIMEOUT_SEC}s timeout"
    exit 124
  fi
fi

# Distinguish "sanitizer crashed before writing log" from "0 races
# detected" — per the spec, an empty/missing log is exit 3 (NOT 0).
if [[ ! -s "${LOG}" ]]; then
  echo "[p22-sanitizer-gate] FAIL: log not produced at ${LOG} — sanitizer likely crashed pre-output"
  exit 3
fi

# Pitfall 5 literal filters. Each `grep -cE` is wrapped with `|| true`
# because grep returns exit 1 on zero matches (which is the desired
# pass state for Cluster B), and `set -e` would otherwise abort.
CLUSTER_B=$(grep -cE 'Host Frame:.*alloc_and_peer_copy_async' "${LOG}" || true)
CLUSTER_A=$(grep -cE 'Host Frame:.*(read_column_chunks_async|posix_device_io)' "${LOG}" || true)
TOTAL_RACES=$(grep -cE 'Use-before-alloc on allocation' "${LOG}" || true)

echo "[p22-sanitizer-gate] cluster_B=${CLUSTER_B} (gate: must be 0)"
echo "[p22-sanitizer-gate] cluster_A=${CLUSTER_A} (advisory; D-09)"
echo "[p22-sanitizer-gate] total_races=${TOTAL_RACES}"
echo "[p22-sanitizer-gate] log=${LOG}"

if [[ "${CLUSTER_B}" -ne 0 ]]; then
  echo "[p22-sanitizer-gate] FAIL: ${CLUSTER_B} Cluster B race-frame mention(s) found"
  echo "[p22-sanitizer-gate] First 200 lines of context around the offending Host Frame:.*alloc_and_peer_copy_async:"
  grep -B 1 -A 30 -E 'Host Frame:.*alloc_and_peer_copy_async' "${LOG}" | head -200 || true
  exit 1
fi

echo "[p22-sanitizer-gate] PASS: Cluster B = 0"
exit 0
