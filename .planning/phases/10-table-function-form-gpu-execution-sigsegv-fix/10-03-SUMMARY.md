---
phase: 10-table-function-form-gpu-execution-sigsegv-fix
plan: "03"
subsystem: gpu-execution
tags: [sigsegv, parquet-scan, cuda-stream, rmm, use-after-destroy, fix]

# Dependency graph
requires:
  - phase: 10-table-function-form-gpu-execution-sigsegv-fix
    provides: 10-02-GDB.md (confirmed_hypothesis=H1)
provides:
  - 10-03-FIX.md with applied_hypothesis=H1, fix_commit=71fd623
  - SIGSEGV closed: gpu_execution - filter equality parquet exits 0
  - translated_expression::owned_stream field for stream lifetime ownership
affects:
  - 10-04-VALIDATION (ship-gate SF100 re-run and full VALIDATION.md)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Stream lifetime ownership: move temporary CUDA stream into translated_expression::owned_stream so cudaFreeAsync uses a valid stream handle during scalar destruction"
    - "C++ destruction order: declare owned_stream BEFORE owned_literals in struct so stream outlives scalars (reverse-declaration-order destruction)"
    - "std::optional<rmm::cuda_stream>: defer stream creation to avoid allocating a real CUDA stream for every translated_expression construction"

key-files:
  created:
    - .planning/phases/10-table-function-form-gpu-execution-sigsegv-fix/10-03-FIX.md
  modified:
    - src/include/expression_executor/gpu_expression_translator.hpp
    - src/op/sirius_physical_parquet_scan.cpp

key-decisions:
  - "Root cause refined from H1: not a stream-ordered allocation race but a use-after-destroy — translation_stream goes out of scope at end of for-loop body, scalars retain stale cudaStream_t handle, cudaFreeAsync fires on stale handle at next QueryBegin"
  - "Fix: move translation_stream into translated_expression::owned_stream declared before owned_literals to ensure C++ reverse-destruction order keeps stream valid for cudaFreeAsync"
  - "std::optional<rmm::cuda_stream> chosen over rmm::cuda_stream to avoid eager stream allocation for every translated_expression"
  - "[mgpu-audit] pre-existing SIGSEGV confirmed: fails identically on base branch before fix; not introduced by this plan"

requirements-completed: [CRIT-1, CRIT-2, CRIT-6]

# Metrics
duration: 55min
completed: 2026-04-27
---

# Phase 10 Plan 03: FIX Summary

**Stream use-after-destroy SIGSEGV in parquet filter translation — fixed by moving `translation_stream` into `translated_expression::owned_stream` with correct C++ destruction order**

## Performance

- **Duration:** ~55 min (including context recovery from prior session)
- **Completed:** 2026-04-27
- **Tasks:** 2 (apply fix, verify)
- **Files modified:** 2 source files, 36 LOC added

## Accomplishments

- Refined H1 root cause: the SIGSEGV is a use-after-destroy (not a stream-ordered allocation race). `translation_stream` (local `rmm::cuda_stream`) is destroyed at end of the for-loop body while the scalars it allocated retain its `cudaStream_t` handle; `cudaFreeAsync` on the stale handle fires at the next `QueryBegin` → SIGSEGV
- Applied two-part fix in 36 LOC:
  - Added `std::optional<rmm::cuda_stream> owned_stream{}` to `translated_expression`, declared BEFORE `owned_literals` to ensure stream outlives scalars (C++ reverse-declaration-order destruction)
  - Moved `translation_stream` into `translated->owned_stream` before emplacing into `translated_filter_by_device`
- Verified: `gpu_execution - filter equality parquet` passes 31 assertions (was: SIGSEGV at assertion 19); `tpch_q1_sf10_2gpu` continues to pass 16 assertions
- Full suite: only pre-existing `[mgpu-audit]` SIGSEGV remains (confirmed pre-existing on base branch before fix)
- HYG-02 invariant: `rmm::cuda_stream_default` count 41 → 40

## Task Commits

1. **Fix commit** — `71fd623` — `feat(10-03): fix stream use-after-destroy SIGSEGV in parquet filter translation`
2. **Plan metadata** — (this SUMMARY commit)

## Files Created/Modified

- `src/include/expression_executor/gpu_expression_translator.hpp` — added `owned_stream` field to `translated_expression` struct
- `src/op/sirius_physical_parquet_scan.cpp` — move `translation_stream` into `translated->owned_stream`
- `.planning/phases/10-table-function-form-gpu-execution-sigsegv-fix/10-03-FIX.md` — fix description with applied_hypothesis, diff summary, test results

## Decisions Made

1. **Root cause is use-after-destroy, not allocation race**: Initial H1 hypothesis described a stream-ordered allocation race (scalar allocated but not yet visible when planning_stream reads it). Static analysis of the code path showed the actual failure is later: `translation_stream` is destroyed before scalars are freed, so `cudaFreeAsync` fires on a stale stream handle during the second query's `QueryBegin` → `task_creator_->reset(false)` → `_parquet_scan_operator_global_state_map` cleared → `translated_expression` destroyed.

2. **`std::optional<rmm::cuda_stream>` over plain `rmm::cuda_stream`**: Plain `rmm::cuda_stream{}` default-constructs a real CUDA stream on every `translated_expression` construction, even in the common case where no translation_stream needs to be owned (e.g., when the translation uses a persistent external stream). `std::optional` avoids this overhead.

3. **Declaration order is the critical invariant**: C++ destroys struct members in reverse declaration order. `owned_stream` must be declared before `owned_literals`; if reversed, the stream would be destroyed first (before scalars call `cudaFreeAsync`), reproducing the bug.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Root cause refined from H1 (allocation race) to use-after-destroy**
- **Found during:** Task 1 (applying the H1 fix branch)
- **Issue:** The initial fix (explicit translation_stream + synchronize) did not eliminate the SIGSEGV. Investigation revealed the root cause is not a use-before-alloc race during the planning phase, but a use-after-destroy during the teardown phase. The `translation_stream` goes out of scope at the end of the for-loop body, invalidating the `cudaStream_t` handle that the scalars will later use for `cudaFreeAsync`.
- **Fix:** Added `owned_stream` field to `translated_expression` and moved `translation_stream` into it, extending the stream's lifetime to match the scalar lifetimes.
- **Files modified:** `src/include/expression_executor/gpu_expression_translator.hpp`, `src/op/sirius_physical_parquet_scan.cpp`
- **Commit:** `71fd623`

### Out-of-Scope Discoveries (deferred)

- `gpu_execution - [mgpu-audit] per-GPU distribution on TPC-H Q1` SIGSEGV: pre-existing on base branch, not introduced by this fix. Deferred to future investigation.

## Test Results

| Test | Before | After |
|------|--------|-------|
| `gpu_execution - filter equality parquet` | SIGSEGV (assertion 19/31) | PASS (31/31, exit 0) |
| `gpu_execution - tpch_q1_sf10_2gpu` | PASS (16 assertions) | PASS (16 assertions) |
| Full suite | exit -1, 2 SIGSEGV | exit -1, 1 SIGSEGV (pre-existing [mgpu-audit]) |

## Next Phase Readiness

Plan 10-04 (VALIDATION) is fully unblocked:
- Fix commit: `71fd623` on `feature/single-node-multi-gpu2`
- Target tests are GREEN
- Remaining `[mgpu-audit]` failure is pre-existing (documented)
- Ship-gate: SF100 Q11 2-GPU run + full unit-test re-run on clean session

---
*Phase: 10-table-function-form-gpu-execution-sigsegv-fix*
*Completed: 2026-04-27*
