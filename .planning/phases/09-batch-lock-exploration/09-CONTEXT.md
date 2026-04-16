# Phase 9: Batch Lock Exploration - Context

**Gathered:** 2026-04-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Analyze `batch_lock_utils::lock_or_prepare_batch` and conditionally refactor it to use `convertible_data_batch::convert()` internally for the shared conversion logic. The function's outer structure (wait-to-lock retry loop, processing handle acquisition) stays; only the tier-switching conversion step is delegated.

**Requirements covered:** LOCK-01, LOCK-02

</domain>

<decisions>
## Implementation Decisions

### Analysis approach (go/no-go)
- **D-01:** The go/no-go analysis is pre-decided: **go**. The user chose to use `convertible_data_batch::convert()` inside `lock_or_prepare_batch` rather than replacing the function entirely. The functional diff analysis (LOCK-01) documents the behavioral differences and how they are resolved.

### In-transit lock restructuring
- **D-02:** Restructure `lock_or_prepare_batch` so it does NOT acquire the in_transit lock itself. Let `convertible_data_batch::convert()` handle the full in_transit locking, conversion, and state restore internally. The outer function only handles the `wait_to_lock_for_processing` retry loop and processing handle acquisition.

### Reservation manager threading
- **D-03:** Add `sirius_memory_reservation_manager&` parameter to `lock_or_prepare_batch`. The caller (`pipelineable_operator_data::prepare_for_processing`) passes it from `sirius_context`. This gives the forward path polite reservation checks before converting, matching the `convert()` contract.

### Refactor boundary
- **D-04:** Minimal refactor — replace only the tier-switching conversion logic inside the while loop (lines 69-117 of `batch_lock_utils.hpp`) with a `convertible_data_batch::convert()` call. Keep the `wait_to_lock_for_processing` retry loop, the `memory_space_mismatch` handling, and the processing handle acquisition exactly as-is. Smallest diff, lowest risk.

### Claude's Discretion
- How to handle the contention case (line 77-81: another thread has in_transit lock → wait for processing lock) after restructuring — may simplify naturally since `convert()` returns false on contention
- Whether `pipelineable_operator_data::prepare_for_processing` needs a signature change to accept `res_mgr`, or whether it can retrieve it internally
- Exact structure of the functional diff documentation for LOCK-01

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### batch_lock_utils (primary target)
- `src/include/pipeline/batch_lock_utils.hpp` — `lock_or_prepare_batch` function to be refactored
- `src/op/sirius_physical_operator.cpp` — Primary call site: `pipelineable_operator_data::prepare_for_processing()` (line 51)

### convertible_data abstractions (replacement infrastructure)
- `src/include/data/convertible_data.hpp` — `convertible_data` interface with `convert()` signature
- `src/include/data/convertible_data_batch.hpp` — `convertible_data_batch::convert()` implementation that will be called inside `lock_or_prepare_batch`

### Memory reservation (new dependency for forward path)
- `src/include/memory/sirius_memory_reservation_manager.hpp` — Reservation manager to be threaded through to `lock_or_prepare_batch`

### Existing tests
- `test/cpp/pipeline/test_gpu_pipeline_task_history.cpp` — Tests that exercise `lock_or_prepare_batch` indirectly

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `convertible_data_batch::convert()` — Already implements the save-state / lock-in-transit / tier-switch / convert_to / restore pattern with failure safety
- `sirius::converter_registry::get()` — Singleton converter registry used by both functions
- `sirius_memory_reservation_manager` — Available in `sirius_context`, needs to be threaded to call site

### Established Patterns
- `convertible_data_batch` wraps `shared_ptr<data_batch>` — same type `lock_or_prepare_batch` receives
- `convert()` takes `vector<memory_space*>` — `lock_or_prepare_batch` has a single target, will need to wrap in a single-element vector
- Forward path throws `rmm::out_of_memory` on failure — with reservation checks this may become a graceful `false` return instead

### Integration Points
- `pipelineable_operator_data::prepare_for_processing()` in `sirius_physical_operator.cpp` — only call site of `lock_or_prepare_batch`, needs `res_mgr` parameter
- `gpu_pipeline_executor` — calls `prepare_for_processing`, has access to `sirius_context` for `res_mgr`

</code_context>

<specifics>
## Specific Ideas

- The refactor makes `lock_or_prepare_batch` a thin orchestrator: retry loop + processing handle, with `convert()` doing the heavy lifting
- This eliminates ~40 lines of duplicated conversion logic (the tier-switch + in_transit + state restore pattern)
- The `convertible_data.hpp` docstring already mentions `lock_or_prepare_batch` as a generalization target (line 44)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 09-batch-lock-exploration*
*Context gathered: 2026-04-16*
