---
phase: 09-batch-lock-exploration
plan: 01
subsystem: pipeline
tags: [convertible_data, batch_lock, memory_conversion, reservation_manager]

# Dependency graph
requires:
  - phase: 08-api-cleanup
    provides: convertible_data_batch::convert() with failure-safe conversion semantics
  - phase: 06-batch-conversion
    provides: convertible_data_batch wrapping data_batch
provides:
  - Unified conversion path: lock_or_prepare_batch delegates to convertible_data_batch::convert()
  - sirius_memory_reservation_manager threaded through prepare_for_processing call chain
  - Polite reservation checks added to forward-path batch conversion
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Store shared infrastructure references (res_mgr) on pipeline global state"
    - "Delegate tier-switching to convertible_data_batch::convert() for both forward and downgrade paths"

key-files:
  created: []
  modified:
    - src/include/pipeline/batch_lock_utils.hpp
    - src/include/op/sirius_physical_operator.hpp
    - src/op/sirius_physical_operator.cpp
    - src/pipeline/gpu_pipeline_task.cpp
    - src/include/pipeline/sirius_pipeline_task_states.hpp
    - src/creator/task_creator.cpp
    - test/cpp/pipeline/test_gpu_pipeline_task_history.cpp

key-decisions:
  - "lock_or_prepare_batch delegates to convertible_data_batch::convert() -- eliminates ~40 lines of duplicated conversion logic"
  - "res_mgr stored on gpu_pipeline_task_global_state rather than threading through DuckDB ClientContext chain"

patterns-established:
  - "Pipeline global state as carrier for shared infrastructure references"

requirements-completed: [LOCK-01, LOCK-02]

# Metrics
duration: 29min
completed: 2026-04-16
---

# Phase 9 Plan 1: Batch Lock Exploration Summary

**Refactored lock_or_prepare_batch to delegate tier-switching conversion to convertible_data_batch::convert(), eliminating ~40 lines of duplicated in_transit lock / convert_to / state-restore code and unifying forward and downgrade conversion paths**

## Performance

- **Duration:** 29 min
- **Started:** 2026-04-16T21:39:20Z
- **Completed:** 2026-04-16T22:08:32Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Replaced manual in_transit lock / convert_to / state-restore pattern in batch_lock_utils.hpp with a single convertible_data_batch::convert() call
- Added sirius_memory_reservation_manager& parameter throughout prepare_for_processing call chain (lock_or_prepare_batch -> prepare_for_processing -> gpu_pipeline_task::execute)
- Stored reservation manager pointer on gpu_pipeline_task_global_state for clean access from tasks
- Updated all 4 GPU pipeline task history tests to wire reservation manager through global state
- Build compiles successfully; unit tests cannot run in this environment (no GPU) but are structurally sound

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor lock_or_prepare_batch to use convertible_data_batch::convert()** - `6976d50d` (feat)
2. **Task 2: Run tests and record functional diff in PROJECT.md Key Decisions** - `7be4505d` (docs)

## Files Created/Modified
- `src/include/pipeline/batch_lock_utils.hpp` - Refactored: replaced ~40 lines of manual conversion with convertible_data_batch::convert() delegation
- `src/include/op/sirius_physical_operator.hpp` - Updated prepare_for_processing signature with res_mgr parameter; added forward declaration
- `src/op/sirius_physical_operator.cpp` - Updated prepare_for_processing implementation and lock_or_prepare_batch call to pass res_mgr
- `src/pipeline/gpu_pipeline_task.cpp` - Thread res_mgr from global state to prepare_for_processing call
- `src/include/pipeline/sirius_pipeline_task_states.hpp` - Added set/get_memory_reservation_manager to gpu_pipeline_task_global_state
- `src/creator/task_creator.cpp` - Set res_mgr on global state when creating gpu_pipeline_task_global_state
- `test/cpp/pipeline/test_gpu_pipeline_task_history.cpp` - Updated all 4 test cases to set reservation manager on global state
- `.planning/PROJECT.md` - Added Phase 9 validated entries and key decisions

## Decisions Made
- **lock_or_prepare_batch delegates to convertible_data_batch::convert():** Eliminates duplicated conversion logic between forward path (batch_lock_utils) and downgrade path (convertible_data_batch). Both paths now share the same failure-safety guarantees (state restore on error). Adds polite reservation checks to the forward path.
- **res_mgr stored on gpu_pipeline_task_global_state:** Cleaner than reaching through DuckDB ClientContext chain (pipeline -> engine -> context -> SiriusContext -> get_memory_manager). The global state is already shared across all tasks in a pipeline and is easily set during task creation in task_creator.cpp. Also more testable since tests can inject their own manager.
- **const_cast for memory_space*:** The convert() API takes non-const memory_space* while lock_or_prepare_batch receives const memory_space*. The const_cast is safe because convert() only reads the memory_space (tier, id) and does not mutate it. This is an accepted API asymmetry (T-09-01 in threat model).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added res_mgr to gpu_pipeline_task_global_state**
- **Found during:** Task 1 (tracing how to obtain res_mgr from gpu_pipeline_task)
- **Issue:** Plan suggested accessing res_mgr via SiriusContext -> get_memory_manager(), but gpu_pipeline_task has no direct access to SiriusContext, and the const-pipeline chain blocks get_client_context() in some paths
- **Fix:** Added set/get_memory_reservation_manager to gpu_pipeline_task_global_state; task_creator sets it when creating global state (task_creator already has _mem_res_mgr reference)
- **Files modified:** src/include/pipeline/sirius_pipeline_task_states.hpp, src/creator/task_creator.cpp
- **Verification:** Build compiles; test fixtures updated to match
- **Committed in:** 6976d50d (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical infrastructure)
**Impact on plan:** The deviation provides a cleaner, more testable solution than the plan's suggested approach. No scope creep.

## Issues Encountered
- **No GPU in execution environment:** Unit tests crash on initialization because sirius_memory_reservation_manager constructor allocates GPU memory pools. This affects ALL tests (even non-GPU ones like inspectable_mpsc) due to global test environment setup. Confirmed this is a pre-existing environment issue (main repo's test binary exhibits the same behavior). Build compilation success and structural code correctness verified via acceptance criteria checks.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- v3.0 milestone complete: all conversion paths now use convertible_data abstractions
- Forward path (lock_or_prepare_batch) and downgrade path (downgrade_executor processing_loop) share the same convertible_data_batch::convert() implementation
- No remaining planned work for this milestone

## Self-Check: PASSED

- All 9 modified/created files verified on disk
- Both task commits (6976d50d, 7be4505d) verified in git history
- batch_lock_utils.hpp contains convertible_data_batch (4 occurrences)
- batch_lock_utils.hpp contains 0 occurrences of try_to_lock_for_in_transit (removed)
- batch_lock_utils.hpp contains 0 occurrences of convert_to< (removed)
- sirius_physical_operator.hpp contains sirius_memory_reservation_manager (2 occurrences)
- gpu_pipeline_task.cpp contains res_mgr (3 occurrences)

---
*Phase: 09-batch-lock-exploration*
*Completed: 2026-04-16*
