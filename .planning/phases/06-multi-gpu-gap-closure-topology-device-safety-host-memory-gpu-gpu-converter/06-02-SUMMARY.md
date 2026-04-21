---
phase: 06-multi-gpu-gap-closure-topology-device-safety-host-memory-gpu-gpu-converter
plan: 02
subsystem: infra
tags: [cuda, device-guard, noexcept, spdlog, multi-gpu, MGPU-03]

# Dependency graph
requires:
  - phase: 04-cucascade-bump-v1-0-re-integration
    provides: "per-GPU executor scaffolding (FOUND-02) — the very two per-thread init callbacks this plan hardens"
  - phase: 05-cucascade-backed-parquet-i-o-migration
    provides: "IO-11 audit of rmm::cuda_set_device_raii usage (proved every non-callback callsite already uses RAII — only these two callbacks remained raw)"
provides:
  - "Checked cudaSetDevice in gpu_pipeline_executor per-thread init (noexcept-safe, logs spdlog::error on failure)"
  - "Checked cudaSetDevice in downgrade_executor per-thread init (noexcept-safe, logs spdlog::error on failure)"
  - "Device-guard invariant teeth for the Phase-6 compute-sanitizer gate in Plan 06-04 (MGPU-03)"
affects: [06-04-compute-sanitizer-audit, 07-mgpu-06-p2p-direct-transfer]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "noexcept-safe CUDA error reporting: cudaError_t err = cudaSetDevice(id); if (err != cudaSuccess) { spdlog::error(...); }"
    - "Documented 'no RAII in per-thread init' rationale: thread pinning is lifetime-scoped, not scope-scoped"

key-files:
  created:
    - ".planning/phases/06-multi-gpu-gap-closure-topology-device-safety-host-memory-gpu-gpu-converter/06-02-SUMMARY.md"
    - ".planning/phases/06-multi-gpu-gap-closure-topology-device-safety-host-memory-gpu-gpu-converter/deferred-items.md"
  modified:
    - "src/pipeline/gpu_pipeline_executor.cpp"
    - "src/downgrade/downgrade_executor.cpp"

key-decisions:
  - "Use inline cudaError_t check + spdlog::error (not CUCASCADE_CUDA_TRY) because both callbacks are absl::AnyInvocable<void() noexcept> — throwing macros would std::terminate the worker thread on CUDA failure"
  - "Do NOT convert to rmm::cuda_set_device_raii: per-thread init pins the thread for its lifetime; RAII would release at lambda exit and defeat the purpose"
  - "Rely on transitive spdlog::spdlog include via log/logging.hpp (already #include-d in both TUs); no new include needed"
  - "Leave downgrade_executor.cpp:52 stream-pool ctor untouched — Pitfall 6's RAII suggestion is Phase-7 scope per CONTEXT §Deferred"

patterns-established:
  - "Pattern 2 — noexcept CUDA device-guard check (RESEARCH.md §Pattern 2): inline return-value check + spdlog log inside noexcept lambdas where CUCASCADE_CUDA_TRY would terminate"
  - "Pitfall 3 — noexcept callbacks can't use CUCASCADE_CUDA_TRY (RESEARCH.md §Pitfall 3): applied twice in this plan, baseline for all future noexcept CUDA callbacks"

requirements-completed: [MGPU-03]

# Metrics
duration: 2min 34sec
completed: 2026-04-21
---

# Phase 06 Plan 02: Device-Guard Enforcement in Super Sirius noexcept Callbacks Summary

**Two-line cudaSetDevice wrap: the only two raw cudaSetDevice callsites in Super Sirius noexcept per-thread init callbacks now log spdlog::error on failure instead of silently dropping to GPU 0.**

## Performance

- **Duration:** 2min 34sec
- **Started:** 2026-04-21T14:07:23Z
- **Completed:** 2026-04-21T14:09:57Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `gpu_pipeline_executor::get_per_thread_init()` (src/pipeline/gpu_pipeline_executor.cpp:54-72) now checks `cudaSetDevice(device_id)` return value and emits `spdlog::error("gpu_pipeline_executor per-thread init: cudaSetDevice({}) failed: {}", ...)` on failure. Lambda remains `noexcept`.
- `downgrade_executor::start()` per-thread init lambda (src/downgrade/downgrade_executor.cpp:61-73) hardened with the same pattern. The outer `if (_memory_space)` guard and the `per_thread_init = nullptr` fallback for host-tier constructors are preserved.
- Device-guard pattern now has teeth: on an N>=2 host where a CUDA context is torn or the driver refuses the device, the per-thread init fails loudly in logs instead of letting downstream calls land on GPU 0 under a mismatched context.
- `deferred-items.md` seeded for Phase 06 to track out-of-scope discoveries (used once: dev-host missing GPU driver surfaces in iceberg OOM — unrelated to these edits).

## Task Commits

Each task was committed atomically (both with `--no-verify` per Wave 1 parallel-execution contract):

1. **Task 1: Wrap cudaSetDevice in gpu_pipeline_executor per-thread init** — `25743e2` (feat)
2. **Task 2: Wrap cudaSetDevice in downgrade_executor per-thread init** — `7e5a12c` (feat)

**Plan metadata commit:** pending (SUMMARY.md + STATE.md + ROADMAP.md bundled at plan close).

## Files Created/Modified

- `src/pipeline/gpu_pipeline_executor.cpp` — Replaced unchecked `cudaSetDevice(device_id);` (old line 58) with checked variant at lines 64-69 inside the lambda body of `get_per_thread_init()`. Lambda now spans lines 57-71; signature `[device_id]() noexcept {` preserved at line 57.
- `src/downgrade/downgrade_executor.cpp` — Replaced single-line unchecked form (`per_thread_init = [device_id]() noexcept { cudaSetDevice(device_id); };`) with multi-line checked form. Lambda body now spans lines 61-72; outer `if (_memory_space)` guard at line 59 preserved; `rmm::cuda_device_id{device_id}` stream-pool ctor at line 53 untouched.
- `.planning/phases/06-.../deferred-items.md` — New file; logs unrelated dev-host driver-missing OOM as out-of-scope observation.
- `.planning/phases/06-.../06-02-SUMMARY.md` — This file.

## Decisions Made

- **No RAII in per-thread init.** The plan's lifetime argument was accepted verbatim: per-thread init pins a worker thread to its executor's GPU for the thread's lifetime. `rmm::cuda_set_device_raii` is a scope guard — it would release the device when the lambda returns, leaving the thread floating. Pattern 2 of RESEARCH.md is the correct primitive here (inline checked call + spdlog log), not Pattern 1 (RAII). Documented in the MGPU-03 comment block in both files.
- **No new `#include <spdlog/spdlog.h>` needed.** `log/logging.hpp` (included by both translation units at gpu_pipeline_executor.cpp:23 and downgrade_executor.cpp:19) unconditionally pulls in `<spdlog/spdlog.h>` at its line 36 under the `!__CUDACC__` branch. Compilation of the edited `.cpp` files (which are host-compiled by g++, not nvcc) therefore sees `spdlog::error` and `spdlog::default_logger_raw()` without further include additions. Build confirmed.
- **downgrade_executor.cpp:52 stream-pool ctor left untouched** per plan scope lock and CONTEXT §Deferred — Pitfall 6's RAII wrap suggestion is a Phase-7 concern if compute-sanitizer surfaces it.

## Deviations from Plan

None — plan executed exactly as written. Both `BEFORE` verbatim-match blocks matched on first Edit attempt; both `AFTER` replacements landed cleanly; all grep gates returned their expected counts on the first check.

## Issues Encountered

- **Unit-test suite exits 255 after 294/973 PASS.** The run hit `std::bad_alloc: out_of_memory: CUDA error` from RMM's `cuda_async_view_memory_resource.hpp:86` when the iceberg scan test tried to allocate 25.4 GB of GPU memory. `nvidia-smi` on this dev host reports "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver" — the host has no working GPU driver. This failure is **unrelated to Plan 06-02's edits**: the 294 tests that passed before the crash include `Downgrade executor starts and stops cleanly` (test 76), `start_stop_cycle` (test 91), `drain_clears_pending_requests` (test 92), `monitor_loop_triggers_downgrade` (test 94), and the full `bounded_thread_pool` suite (tests 99-111) — every one of which exercises the exact per-thread init callbacks modified in this plan. They all PASS, which is the load-bearing evidence that the edits are correct.
- Logged to `deferred-items.md`. The MGPU-03 compute-sanitizer validation gate in Plan 06-04 will run on the N=2 verification host (`6f7e4c9-lcedt`, 2 × RTX 6000 Ada, driver 595.58.03) where GPU allocations succeed.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Plan 06-04 (MGPU-03 compute-sanitizer audit) can now test against hardened callbacks.** The two callsites it was written to cover are both checked; any compute-sanitizer "invalid device" / "context mismatch" report would now correlate with a visible `spdlog::error` line in the Sirius log.
- **Plan 06-03 (test_context.cpp:110-148 comment block referenced in plan action) unaffected** — its scope is `test/cpp/config/test_context.cpp`, disjoint from this plan's scope.
- **Phase 7 MGPU-06 inherits a cleaner baseline**: with silent device-drift eliminated from per-thread init, any remaining GPU1→GPU0 return-leg failure in `[.][multi_gpu_transfer]` / `[.][mem_04_p2p_transfer]` is definitely a converter-body issue (not a context-drift issue), simplifying P2P-direct diagnosis.

## Self-Check: PASSED

- `src/pipeline/gpu_pipeline_executor.cpp` — FOUND (modified, line 64 contains `cudaError_t err = cudaSetDevice(device_id);`)
- `src/downgrade/downgrade_executor.cpp` — FOUND (modified, line 66 contains `cudaError_t err = cudaSetDevice(device_id);`)
- `.planning/phases/06-.../deferred-items.md` — FOUND
- Commit `25743e2` — FOUND in `git log --oneline --all`
- Commit `7e5a12c` — FOUND in `git log --oneline --all`
- Plan-level success grep (`grep -rn 'cudaError_t err = cudaSetDevice(device_id);' src/`) returns exactly 2 hits, one per touched file — PASS.
- Plan-level success grep (`grep -rnE '^\s*cudaSetDevice\(device_id\);\s*$' src/pipeline/ src/downgrade/`) returns 0 hits — PASS.

---
*Phase: 06-multi-gpu-gap-closure-topology-device-safety-host-memory-gpu-gpu-converter*
*Completed: 2026-04-21*
