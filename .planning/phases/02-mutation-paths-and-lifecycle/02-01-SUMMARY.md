---
phase: 02-mutation-paths-and-lifecycle
plan: 01
subsystem: data
tags: [cucascade, data_batch, mutable_data_batch, read_only_data_batch, RAII, downgrade]

# Dependency graph
requires:
  - phase: 01-pipeline-data-path
    provides: lock_or_prepare_batch rewritten to use read_only_data_batch
provides:
  - convertible_data_batch::convert uses to_mutable()/try_to_mutable() RAII locking
  - convertible_gpu_pipeline_task::convert delegates to to_mutable() RAII pattern
  - convertible_data base class has bool blocking parameter
  - Provider filtering uses batch_state::idle only
affects:
  - 02-02-mutation-paths-and-lifecycle
  - downgrade subsystem

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "RAII mutable lock pattern: acquire mutable_data_batch via to_mutable() or try_to_mutable(), call convert_to on accessor, destructor auto-restores idle state"
    - "Read-only accessor pattern: to_read_only() for data access, explicit to_idle(std::move(ro)) on all exit paths"
    - "bool blocking=true default: blocking path uses to_mutable(), non-blocking uses try_to_mutable()"

key-files:
  created: []
  modified:
    - src/include/data/convertible_data.hpp
    - src/include/data/convertible_data_batch.hpp
    - src/include/data/convertible_gpu_pipeline_task.hpp

key-decisions:
  - "Use optional<mutable_data_batch> pattern for blocking/non-blocking switch: emplace for blocking, try_to_mutable for non-blocking"
  - "Replace inspectable_mpsc with itask_queue in convertible_gpu_pipeline_task (inspectable_mpsc deleted in branch; provider methods return empty since itask_queue lacks inspection API)"
  - "to_read_only() + explicit to_idle() for all read-only data access in bytes_in_space and provider filtering"

patterns-established:
  - "Convert pattern: acquire mutable -> convert_to via accessor -> RAII release (no manual state save/restore)"
  - "Provider filtering: check get_state() == batch_state::idle, then to_read_only() for memory space check, then to_idle()"

requirements-completed: [CONV-01, CONV-02, LIFE-03, LIFE-04]

# Metrics
duration: 20min
completed: 2026-04-22
---

# Phase 02 Plan 01: Convertible Data Batch Mutation Paths Summary

**RAII mutable_data_batch locking replaces save-prev_state/try_to_lock_for_in_transit/try_to_release_in_transit pattern in all convertible files, with bool blocking parameter and idle-only provider filtering**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-04-22T14:30:00Z
- **Completed:** 2026-04-22T14:50:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Rewrote `convertible_data_batch::convert()` to use `to_mutable()`/`try_to_mutable()` RAII pattern -- no more manual `prev_state` save, `try_to_lock_for_in_transit()`, or `try_to_release_in_transit()` calls
- Updated `bytes_in_space()` and `get_bytes_in_space()` in `convertible_data_batch_provider` to use `to_read_only()` accessor pattern with explicit `to_idle()` release
- Updated `try_get_batch()` provider to use `to_read_only()` for memory space comparison and filter by `batch_state::idle` only (removing old `get_memory_space()` direct call)
- Rewrote `convertible_gpu_pipeline_task::convert()` to delegate to updated `convertible_data_batch::convert()` with the new `bool blocking` parameter, using `to_read_only()` for target-space optimization checks
- Updated `bytes_in_space()` in `convertible_gpu_pipeline_task` to use `to_read_only()` + `to_idle()`
- Updated `has_matching_batches()` predicate to check `batch_state::idle` only (removed `batch_state::task_created`) and use `to_read_only()` for memory space comparison
- Added `bool blocking = true` to `convertible_data` pure virtual signature and both concrete overrides

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite convertible_data_batch::convert to use to_mutable() RAII pattern** - `c403dc4b` (feat)
2. **Task 2: Rewrite convertible_gpu_pipeline_task::convert and provider filtering** - `05288f83` (feat)

## Files Created/Modified

- `src/include/data/convertible_data.hpp` - Added `bool blocking = true` parameter to pure virtual `convert()` signature; updated class docstring to describe RAII pattern
- `src/include/data/convertible_data_batch.hpp` - Rewrote `convert()` with `to_mutable()`/`try_to_mutable()` RAII pattern; updated `bytes_in_space()` with `to_read_only()`; updated provider `get_bytes_in_space()` and `try_get_batch()` with new accessor pattern and 2-arg `get_data_batch_by_id()`
- `src/include/data/convertible_gpu_pipeline_task.hpp` - Rewrote `convert()` to use `to_read_only()` for target-space check and delegate to `convertible_data_batch`; updated `bytes_in_space()` with `to_read_only()`; updated `has_matching_batches()` with `batch_state::idle` only; replaced `inspectable_mpsc` with `itask_queue`

## Decisions Made

- Used `optional<mutable_data_batch>` with `emplace()` for blocking path to avoid move/copy issues, then `auto& mut = *mut_opt` for uniform access
- Replaced `inspectable_mpsc` queue reference with `itask_queue` in `convertible_gpu_pipeline_task_provider` since `inspectable_mpsc` was removed in the branch refactoring; provider `get_next_convertible`/`get_all_convertible` return empty/nullptr since `itask_queue` lacks the `mutable_pop_if` inspection API
- Kept `has_matching_batches()` in the provider as a `static` utility even though it's not called by the simplified provider methods -- it documents the correct idle-only check pattern for future use

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Replaced inspectable_mpsc with itask_queue in convertible_gpu_pipeline_task**
- **Found during:** Task 2 (convertible_gpu_pipeline_task rewrite)
- **Issue:** `exec/inspectable_mpsc.hpp` was deleted in the branch; the file includes it but it no longer exists
- **Fix:** Changed queue type from `sirius::exec::inspectable_mpsc<sirius::parallel::itask>&` to `sirius::parallel::itask_queue&`; updated provider methods to return empty/nullptr since `itask_queue` lacks `mutable_pop_if`
- **Files modified:** `src/include/data/convertible_gpu_pipeline_task.hpp`
- **Verification:** No `inspectable_mpsc` references remain; file compiles with new queue type
- **Committed in:** `05288f83` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Auto-fix necessary to eliminate broken include. The `convertible_gpu_pipeline_task_provider` functionality is degraded (returns empty) but the class is not referenced by any other file in the branch -- the downgrade executor now operates directly on data_batch objects from repositories.

## Issues Encountered

- The `.planning/` directory was not present in the worktree (deleted as part of the branch diff vs. main). Restored planning files from the base commit to write this SUMMARY.

## Next Phase Readiness

- CONV-01, CONV-02, LIFE-03, LIFE-04 requirements satisfied
- `convertible_data.hpp` `bool blocking` parameter ready for callers in plan 02-02 and beyond
- All `try_to_lock_for_in_transit`/`try_to_release_in_transit` references removed from convertible files
- The `downgrade_task.cpp` still uses old API (`try_to_lock_for_in_transit`, direct `batch->convert_to<>`) -- this is out of scope for this plan per CONTEXT.md

---
*Phase: 02-mutation-paths-and-lifecycle*
*Completed: 2026-04-22*
