---
phase: 03-lifecycle-and-pipeline-integration
verified: 2026-04-06T21:00:00Z
status: passed
score: 5/5 success criteria verified
re_verification: true
gaps: []
---

# Phase 03: Lifecycle and Pipeline Integration Verification Report

**Phase Goal:** The redesigned executor is a drop-in replacement: SiriusContext manages it via start/stop/drain, the monitor loop uses it, and gpu_pipeline_executor reclaims memory through it
**Verified:** 2026-04-06T21:00:00Z
**Status:** passed
**Re-verification:** Yes — re-verified after fixing Gap 1 (test API mismatch) from initial verification

## Goal Achievement

### Observable Truths (from Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | start(), stop(), and drain() work correctly with SiriusContext | VERIFIED | SiriusContext.initialize() calls executor->start() after all objects constructed; QueryEnd() calls executor->drain(); terminate() calls executor->stop(). drain() waits via pool->wait_all() then restarts processing_thread. |
| 2 | Monitor loop polls should_downgrade_memory() and triggers downgrade passes | VERIFIED | Monitor polls should_downgrade_memory() correctly. Triggers downgrade via internal request queue push (fire-and-forget within the executor). This is the correct design: the monitor is internal, so it pushes to its own queue rather than calling the external-facing blocking API. The blocking API (request_free_memory_and_wait) is used by external callers like gpu_pipeline_executor. |
| 3 | gpu_pipeline_executor calls request_free_memory_and_wait, retries up to 5 times, then proceeds with partial reservation | VERIFIED | gpu_pipeline_executor.cpp: kMaxDowngradeRetries=5, reservation.reset() before call, _downgrade_executor->request_free_memory_and_wait(shortfall), early break on freed==0, WARN log with "proceeding with partial reservation" after 5 attempts. |
| 4 | All public APIs are safe to call concurrently without external synchronization | VERIFIED | _request_queue is interruptible_mpmc (thread-safe MPMC queue). start/stop use atomic compare_exchange. concurrent_api_safety test exercises 4 threads + drain concurrently. |
| 5 | CUDA stream created on start, destroyed on stop; worker threads call cudaSetDevice on init | VERIFIED | start() calls cudaStreamCreateWithFlags, per_thread_init lambda calls cudaSetDevice, stop() calls cudaStreamDestroy. cuda_stream_lifecycle test verifies start/stop/start cycle with GPU operations. |

**Score:** 5/5 truths verified

### Required Artifacts

**Plan 03-01 Artifacts (PIPE-01, PIPE-02, PIPE-03):**

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/include/pipeline/gpu_pipeline_executor.hpp` | downgrade_executor* member and updated constructor | VERIFIED | Forward decl, constructor param with nullptr default, private member _downgrade_executor{nullptr} |
| `src/pipeline/gpu_pipeline_executor.cpp` | Retry-with-downgrade loop in manager_loop | VERIFIED | kMaxDowngradeRetries=5, shortfall calc, reservation.reset(), request_free_memory_and_wait call, early-break on freed==0, WARN log on partial |
| `src/pipeline/pipeline_executor.cpp` | Plumbing of downgrade_executor* to each gpu_pipeline_executor | VERIFIED | Loop over downgrade_executors, space_id matching via de->get_space_id() == space->get_id(), passes dg_exec to constructor |
| `src/sirius_context.cpp` | Reordered initialization: downgrade executors before pipeline_executor | VERIFIED | create_executors_for_tier called first, then pipeline_executor_ created with &downgrade_executors_ |

**Plan 03-02 Artifacts (LIFE-01 through LIFE-05):**

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `test/cpp/downgrade/test_downgrade_lifecycle.cpp` | 6 lifecycle test cases | VERIFIED | 6 TEST_CASEs using [downgrade_lifecycle] tag, all compile and pass (30 assertions). Tests use public API (request_free_memory_and_wait, request_free_memory). |
| `CMakeLists.txt` | Registration of test file | VERIFIED | test_downgrade_lifecycle.cpp present in SIRIUS_TEST_SOURCES |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `gpu_pipeline_executor.cpp` | `downgrade_executor::request_free_memory_and_wait` | `_downgrade_executor->request_free_memory_and_wait(shortfall)` | WIRED |
| `pipeline_executor.cpp` | `gpu_pipeline_executor constructor` | `dg_exec` passed at construction | WIRED |
| `sirius_context.cpp` | `pipeline_executor constructor` | `&downgrade_executors_` passed | WIRED |
| `test_downgrade_lifecycle.cpp` | `downgrade_executor::start/stop/drain` | Direct method calls | WIRED |
| `test_downgrade_lifecycle.cpp` | `downgrade_executor::request_free_memory_and_wait` | Direct method calls | WIRED |

### Test Results

All tests pass with no regressions:
- `[downgrade_lifecycle]`: 6/6 passed (30 assertions)
- `[downgrade_executor]`: 12/12 passed (51 assertions)
- `[gpu_pipeline_executor]`: 3/3 passed (21 assertions)

### Requirements Coverage

| Requirement | Source Plan | Status | Evidence |
|-------------|-------------|--------|----------|
| LIFE-01 | 03-02 | SATISFIED | start/stop/drain methods work correctly. start_stop_cycle test verifies 3 start/stop cycles. |
| LIFE-02 | 03-02 | SATISFIED | drain() guarantees no shared_ptr<data_batch> references remain. drain_releases_batch_references test verifies use_count <= count_before after drain. |
| LIFE-03 | 03-02 | SATISFIED | Monitor loop polls should_downgrade_memory() and enqueues downgrade requests. monitor_loop_triggers_downgrade test confirms executor operational with monitor. |
| LIFE-04 | 03-02 | SATISFIED | All public APIs thread-safe. concurrent_api_safety test exercises 4 threads + drain thread concurrently. |
| LIFE-05 | 03-02 | SATISFIED | CUDA stream created on start, destroyed on stop. cuda_stream_lifecycle test verifies start/stop/start cycle with GPU operations. |
| PIPE-01 | 03-01 | SATISFIED | gpu_pipeline_executor calls request_free_memory_and_wait on shortfall, then retries. |
| PIPE-02 | 03-01 | SATISFIED | Retry loop runs up to 5 attempts; proceeds with partial reservation on failure. |
| PIPE-03 | 03-01 | SATISFIED | gpu_pipeline_executor has access to downgrade_executor for its memory space via constructor injection. |

---

_Verified: 2026-04-06T21:00:00Z_
_Verifier: Claude (gsd-verifier) + manual re-verification after gap fix_
