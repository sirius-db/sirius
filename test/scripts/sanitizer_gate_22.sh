#!/usr/bin/env bash
# Phase 22 D-12 / Phase 22.1 GATE-22.1-B sanitizer gate.
#
# Runs SF1 Q11 num_gpus=2 under compute-sanitizer memcheck +
# track-stream-ordered-races=all and gates exit 0 on:
#   Cluster B (`alloc_and_peer_copy_async`) race count == 0
#   AND
#   Cluster A (`read_column_chunks_async` / `posix_device_io`) frame
#                 mention count == 0 (Phase 22.1 GATE-22.1-B per K.1
#                 closure trajectory in 22-VERDICT.md K.1; was advisory
#                 only in Phase 22).
#
# Phase 22 baseline: Cluster A was advisory; SF1 Q11 num_gpus=2 reported
#   ~6 cudf+kvikio cross-stream race blocks per the Plan 22-04 micro-
#   validation log. Phase 22.1 removed all kvikio-routed reads from
#   src/ (sites #1-#7 per 22.1-CONTEXT.md D-01) so the K.1 source frame
#   no longer exists in tree; Cluster A is now expected to be 0.
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
# Pitfall 7 (Phase 23 gap-closure): the cluster_B grep must distinguish
# race-check findings from API-error backtraces. compute-sanitizer surfaces
# `Host Frame: ... alloc_and_peer_copy_async ...` lines in BOTH contexts:
#   1) Race findings (headed by `Use-before-alloc on allocation`) — the
#      ACTUAL gate target; cluster_B must count these.
#   2) API-error backtraces (headed by `Program hit cuda...`) — emitted
#      by probe_peer_dma_works during init when cudaMemcpyPeer returns
#      cudaErrorPeerAccessAlreadyEnabled. NOT a Phase 22 D-07 regression.
# The cluster_B count uses an awk windowed match (in_race state) instead
# of a flat grep. See 23-VERDICT.md Section J for the false-positive trail.
#
# Exit codes:
#   0 - PASS: Cluster B = 0 AND Cluster A = 0
#   1 - FAIL: Cluster B > 0 OR Cluster A > 0
#       Cluster B > 0: Phase 22 same-stream invariant fix regressed
#       Cluster A > 0: Phase 22.1 GATE-22.1-B regressed (a kvikio-routed
#                      read crept back into src/; check GATE-22.1-A
#                      bypass-grep)
#   2 - environment error: compute-sanitizer not found or unittest
#       binary missing/non-executable
#   3 - sanitizer crashed before producing log output (distinguishes
#       from "0 races detected")
#   4 - selftest failed (P22_SELFTEST=1 mode only)
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

# Phase 23 self-test (P22_SELFTEST=1 only): inject a fake log with one
# race finding + one API-error backtrace, both mentioning
# alloc_and_peer_copy_async, and verify cluster_B=1 (race) and not 2.
if [[ "${P22_SELFTEST:-0}" == "1" ]]; then
  tmp=$(mktemp)
  cat >"$tmp" <<'SELFTEST_LOG'
========= COMPUTE-SANITIZER
========= Use-before-alloc on allocation 0xdead of size 16
=========     at 0x100 in kernel
=========     Host Frame: cucascade::data::alloc_and_peer_copy_async + 0x80
=========
========= Program hit cudaErrorPeerAccessAlreadyEnabled (error 704)
=========     Saved host backtrace
=========     Host Frame: cucascade::data::alloc_and_peer_copy_async + 0x40
=========
SELFTEST_LOG
  CB_TEST=$(awk '
    /^=+ Use-before-alloc on allocation/ { in_race=1; in_api=0; next }
    /^=+ Program hit cuda/               { in_race=0; in_api=1; next }
    /^=+ ERROR:/                         { in_race=0; in_api=1; next }
    /^=+ COMPUTE-SANITIZER/              { in_race=0; in_api=0; next }
    /^$/                                 { in_race=0; in_api=0; next }
    in_race && /Host Frame:.*alloc_and_peer_copy_async/ { count++ }
    END { print count + 0 }
  ' "$tmp")
  rm -f "$tmp"
  if [[ "${CB_TEST}" != "1" ]]; then
    echo "[p22-sanitizer-gate] SELFTEST FAIL: expected cluster_B=1 (race), got cluster_B=${CB_TEST}"
    exit 4
  fi
  echo "[p22-sanitizer-gate] SELFTEST PASS: windowed cluster_B counter is correct"
  exit 0
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
#
# Phase 23 gap-closure: cluster_B was a false-positive when total_races=0
# because alloc_and_peer_copy_async also appears in cudaErrorPeerAccessAlreadyEnabled
# API-error backtraces emitted by probe_peer_dma_works at init. We now
# count host-frame mentions ONLY inside race-check sections (headed by
# 'Use-before-alloc on allocation' — same header TOTAL_RACES uses).
# An API-error section is headed by 'Program hit cuda' (compute-sanitizer
# boilerplate for cudaError* surfaced via memcheck).
CLUSTER_B=$(awk '
  /^=+ Use-before-alloc on allocation/ { in_race=1; in_api=0; next }
  /^=+ Program hit cuda/               { in_race=0; in_api=1; next }
  /^=+ ERROR:/                         { in_race=0; in_api=1; next }
  /^=+ COMPUTE-SANITIZER/              { in_race=0; in_api=0; next }
  /^$/                                 { in_race=0; in_api=0; next }
  in_race && /Host Frame:.*alloc_and_peer_copy_async/ { count++ }
  END { print count + 0 }
' "${LOG}" || true)
CLUSTER_A=$(grep -cE 'Host Frame:.*(read_column_chunks_async|posix_device_io)' "${LOG}" || true)
TOTAL_RACES=$(grep -cE 'Use-before-alloc on allocation' "${LOG}" || true)

echo "[p22-sanitizer-gate] cluster_B=${CLUSTER_B} (gate: must be 0)"
echo "[p22-sanitizer-gate] cluster_A=${CLUSTER_A} (gate: must be 0; Phase 22.1 GATE-22.1-B)"
echo "[p22-sanitizer-gate] total_races=${TOTAL_RACES}"
echo "[p22-sanitizer-gate] log=${LOG}"

if [[ "${CLUSTER_B}" -ne 0 ]]; then
  echo "[p22-sanitizer-gate] FAIL: ${CLUSTER_B} Cluster B race-frame mention(s) found"
  echo "[p22-sanitizer-gate] First 200 lines of context around the offending Host Frame:.*alloc_and_peer_copy_async:"
  grep -B 1 -A 30 -E 'Host Frame:.*alloc_and_peer_copy_async' "${LOG}" | head -200 || true
  exit 1
fi

if [[ "${CLUSTER_A}" -ne 0 ]]; then
  echo "[p22-sanitizer-gate] FAIL: ${CLUSTER_A} Cluster A race-frame mention(s) found (Phase 22.1 GATE-22.1-B)"
  echo "[p22-sanitizer-gate] First 200 lines of context around the offending Host Frame:.*(read_column_chunks_async|posix_device_io):"
  grep -B 1 -A 30 -E 'Host Frame:.*(read_column_chunks_async|posix_device_io)' "${LOG}" | head -200 || true
  echo "[p22-sanitizer-gate] HINT: Phase 22.1 removed all kvikio-routed reads from src/. A non-zero Cluster A means a regression — verify GATE-22.1-A bypass-grep is also 0:"
  echo "[p22-sanitizer-gate] HINT:   grep -rn 'cudf::io::datasource::create\\|cudf::io::source_info{' src/ | grep -v 'data_source\\.get()\\|datasource\\.get()'"
  exit 1
fi

echo "[p22-sanitizer-gate] PASS: Cluster B = 0 AND Cluster A = 0"
exit 0
