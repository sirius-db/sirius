---
phase: 07-p2p-direct-transfer-adaptive-scan-partitioning
plan: 01
subsystem: infrastructure
tags: [mgpu-06, p2p, cuda, peer-access, sirius-context]

# Dependency graph
requires: []
provides:
  - cudaDeviceEnablePeerAccess loop in SiriusContext::initialize() for every GPU pair
  - peer_access_enabled_pairs_ cache + is_peer_access_enabled(src, dst) accessor
  - Audit log line per pair (enabled / no-P2P / probe-error / enable-error)
  - Clean teardown (clear cache; no explicit disable — CUDA cleans at exit)
affects: [07-02, 07-03, MGPU-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Always consume cudaGetLastError() after cudaDeviceEnablePeerAccess to prevent sticky error state poisoning unrelated CUDA calls"

key-files:
  created: []
  modified:
    - src/include/sirius_context.hpp (declarations at 2d2aecd, prior session)
    - src/sirius_context.cpp (enable loop + teardown + sticky-error fix)

key-decisions:
  - "Use rmm::cuda_set_device_raii (not cucascade::cuda_set_device_raii) — rmm's variant is already consumed elsewhere in SiriusContext and guarantees device restoration on every iteration's scope exit."
  - "Consume cudaGetLastError() after every cudaDeviceEnablePeerAccess call, regardless of return value. CUDA leaves the return code in the thread-local error slot even when the caller handles it; without the consume the next unrelated CUDA API (e.g., thrust::exclusive_scan) surfaces the same error as 'cudaErrorInvalidDevice: invalid device ordinal'. This is the bug that surfaced as the initial unit-tests regression (test case 27)."
  - "Treat cudaErrorPeerAccessAlreadyEnabled as success (idempotent) — cucascade or a prior sirius init may have already enabled the pair. Still consume the sticky state."
  - "Skip loop on single-GPU hosts with an info-level log line so the audit trail is explicit rather than silent."

patterns-established:
  - "Post-CUDA-call sticky state consume — `(void)cudaGetLastError();` after any cuda*EnablePeerAccess* family call that can return a benign error."

requirements-completed: [MGPU-06-infra]

# Metrics
duration: ~25min (spread: initial code attempt + MCP build failures during sccache-contention period + Task 2 commit + sticky-error regression diagnosis + fix commit + unit-tests verification)
completed: 2026-04-21
---

# Phase 07 Plan 01: MGPU-06 Peer-Access Enable Loop

Adds the `cudaDeviceEnablePeerAccess` once-per-pair enable loop to `SiriusContext::initialize()`, the net-new Sirius code RESEARCH.md Finding 1 identified as the core infrastructure for MGPU-06. cuCascade's `convert_gpu_to_gpu` at `cucascade/src/data/representation_converter.cpp:173` already calls `cudaMemcpyPeerAsync`; what was missing is the `cudaDeviceEnablePeerAccess` driver-state registration that lets that call bypass host staging and dodge the Phase-4-deferred GPU1→GPU0 return-leg bug.

## Performance

- **Duration:** ~25 min total (code + build + diagnostics + fix + verification). Broken down:
  - Task 1 (header declarations): committed in prior session as `2d2aecd` (~5 min)
  - Task 2 (enable loop implementation): 62-line diff in `initialize()` + 5-line teardown, committed as `8e673d7`
  - Environmental blocker: sccache port 4226 held by another user's stale server; required restart (unblocked the build)
  - First unit-tests run: FAIL at case 27 `cudaErrorInvalidDevice: invalid device ordinal` — sticky-error regression
  - Diagnostic revert: `aff97b2` proved the regression was my change (revert → tests pass)
  - Fix: `(void)cudaGetLastError();` after every `cudaDeviceEnablePeerAccess`; committed as `752a644`
  - Final unit-tests: 974/974 PASS, 78,789,809 assertions, 222s runtime
- **MCP Build:** exit 0 (137.4s cold, 10.6s incremental)
- **MCP Unit-tests:** exit 0, 974/974 cases

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | `2d2aecd` *(prior session)* | declare MGPU-06 peer-access cache + is_peer_access_enabled accessor |
| 2 | `8e673d7` | feat(07-01): enable P2P peer access for every GPU pair in SiriusContext::initialize() |
| — | `aff97b2` | Revert "feat(07-01): ..." *(diagnostic; immediately re-applied)* |
| — | `f510f38` | Reapply "feat(07-01): ..." |
| 2-fix | `752a644` | fix(07-01): consume sticky cudaGetLastError after cudaDeviceEnablePeerAccess |

## Accomplishments

### Task 1 — Declarations (prior session)
- `peer_pair_hash` struct for `std::pair<int,int>` hashing
- `std::unordered_set<std::pair<int,int>, peer_pair_hash> peer_access_enabled_pairs_` member
- `[[nodiscard]] bool is_peer_access_enabled(int src, int dst) const` public accessor

### Task 2 — Enable loop + teardown
Inserted the MGPU-06 enable block in `SiriusContext::initialize()` **between** the per-GPU `gpu_io_backends_` cache construction and the `pipeline_executor_` construction — same seam as Research §Recommendation. On single-GPU hosts, the loop short-circuits with an info log. On multi-GPU hosts, for every (i, j) pair where `cudaDeviceCanAccessPeer` returns true, `cudaDeviceEnablePeerAccess(j, 0)` is called from GPU `i`'s context via `rmm::cuda_set_device_raii guard_i`. Outcomes (enabled / already-enabled / no-P2P / probe-error / enable-error) each produce exactly one audit log line. Non-fatal — host-staged fallback remains correct.

In `SiriusContext::terminate()`: `peer_access_enabled_pairs_.clear()` runs between `gpu_io_backends_.clear()` and `memory_manager_->shutdown()`. No `cudaDeviceDisablePeerAccess` call — CUDA cleans up at process exit, and explicit disable during shutdown risks tearing down mappings the memory-manager teardown may still traverse for GPU→HOST drains.

### Fix — sticky-error consume (`752a644`)
**Bug surfaced during unit-tests:** test case 27 (`host_parquet_representation converts to gpu_table_representation`) failed with `cudaErrorInvalidDevice: invalid device ordinal` in `thrust::exclusive_scan`'s tmp-storage lookup — even though no invalid device ordinal was passed. Diagnostic revert proved the failure was introduced by Plan 07-01's enable loop.

**Root cause:** `cudaDeviceEnablePeerAccess` populates the CUDA runtime's thread-local last-error slot with its return value, regardless of whether the caller handled the return code directly. Subsequent unrelated CUDA calls (e.g., anything that internally queries `cudaGetLastError` as part of its error-propagation path) would observe this stale state and report the peer-access error as theirs.

**Fix:** `(void)cudaGetLastError();` immediately after every `cudaDeviceEnablePeerAccess` call, regardless of outcome. Consumes the sticky slot before any other CUDA call can inherit it.

## Structural Invariants (all green)

| Gate | Expected | Actual |
|------|----------|--------|
| `cudaDeviceEnablePeerAccess` in `src/sirius_context.cpp` | 1 call site | 1 (at line 291) |
| `cudaDeviceCanAccessPeer` in `src/sirius_context.cpp` | 1 call site | 1 (at line 276) |
| `cudaErrorPeerAccessAlreadyEnabled` | present | present (line 297) |
| `(void)cudaGetLastError();` after enable call | present | present (line 295) |
| `peer_access_enabled_pairs_.emplace` | present | present (line 299) |
| `peer_access_enabled_pairs_.clear` | present | present (line 419 in terminate) |
| MGPU-06 audit log lines | 4 branches | 4 lines (enabled / no-P2P / probe-error / enable-error) |
| `rmm::cuda_stream_default` in changes | 0 | 0 (HYG-02) |
| `CUCASCADE_CUDA_TRY` adjacent to peer-access calls | 0 | 0 |
| Teardown order | `gpu_io_backends_.clear()` → `peer_access_enabled_pairs_.clear()` → `memory_manager_->shutdown()` | preserved |
| MCP build exit code | 0 | 0 (10.6s incremental after fix) |
| MCP unit-tests exit code | 0 | 0 (974/974, 78,789,809 assertions) |

## Requirements Progress

| REQ-ID | Status | Notes |
|--------|--------|-------|
| **MGPU-06** *(infra)* | ✅ Infrastructure in place | Driver state enabled for every supported pair. Plan 07-02 un-hides GPU↔GPU round-trip tests + adds return-leg + checksum to close MGPU-06 end-to-end. |

## Carries Over

- **Plan 07-02** will flip the hidden `[.][multi_gpu_transfer]` + `[.][mem_04_p2p_transfer]` tests to visible, add checksum-based data integrity on round-trip payloads, and — only if the return leg still fails — register a Sirius-side converter override (RESEARCH.md Pattern 2). With P2P now enabled at init, cucascade's `convert_gpu_to_gpu` should see `cudaMemcpyPeerAsync` succeed on both legs.
- **Plan 07-03** will consume `is_peer_access_enabled(src, dst)` in the adaptive scan integration test to gate the P2P-path assertion vs the host-staged-fallback assertion.

## Issues Encountered

- **sccache port contention** (external): Plan 07-01's MCP build was blocked by a stale sccache server on port 4226 owned by another user. User restarted the server; Sirius builds now work on the default port. A transient workaround with `SCCACHE_SERVER_PORT=4227` was tested but ultimately reverted in `.ai-helper/commands.yaml`.
- **Sticky CUDA error state** (internal, fixed): documented above. Established the pattern "always consume `cudaGetLastError()` after `cudaDeviceEnablePeerAccess`" for future CUDA-state-mutation code.

---
*Plan 07-01 completed: 2026-04-21*
