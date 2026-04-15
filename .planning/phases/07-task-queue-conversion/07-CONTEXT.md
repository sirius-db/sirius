# Phase 7: Task Queue Conversion - Context

**Gathered:** 2026-04-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Concrete `convertible_data` implementations wrapping `gpu_pipeline_task` and `inspectable_mpsc<itask>`. The task wrapper takes temporary RAII ownership of a task extracted from the queue, converts its data batches between memory tiers, and returns the task to the queue via destructor. No batch conversion changes — that's Phase 6.

</domain>

<decisions>
## Implementation Decisions

### RAII Ownership & Queue Return
- **D-01:** `convertible_gpu_pipeline_task` constructor takes `(unique_ptr<itask>, inspectable_mpsc<itask>&)` — queue passed as reference, not raw pointer
- **D-02:** Move-only wrapper: delete copy constructor/assignment, enable move semantics. The `unique_ptr<itask>` member naturally enforces this
- **D-03:** Destructor pushes the task back to the queue via `push()`. If the queue is interrupted (`push()` returns false), log a warning via SIRIUS_LOG_WARN and let the task be destroyed — queue interruption means shutdown, the task is no longer needed

### Task Data Batch Access
- **D-04:** The `mutable_pop_if` predicate uses `dynamic_cast<gpu_pipeline_task*>` to reach the task, then `task.local_state()` (public accessor on `itask`) → `dynamic_cast<gpu_pipeline_task_local_state*>` → `dynamic_cast<pipelineable_operator_data*>(_input_data.get())` → `get_data_batches()`. If any cast fails, the predicate returns false (skip the task)
- **D-05:** `partitioned_operator_data` extends `pipelineable_operator_data`, so a single `dynamic_cast<pipelineable_operator_data*>` catches both types

### Convert Scope Per Task
- **D-06:** `convert()` iterates only data_batches matching the target memory_space within the task's operator_data — batches already in the target space are skipped
- **D-07:** Per-batch independence on failure: successful batch conversions keep data in the new tier, failed batches retain original `idata_representation`. Batch_state is always restored to its pre-conversion value (e.g., `task_created`) for all batches regardless of conversion outcome
- **D-08:** Each batch follows the save-prev_state / lock-for-in_transit / convert / restore-state pattern from Phase 6's `convertible_data_batch::convert()`

### File Organization
- **D-09:** Single header file: `src/include/data/convertible_gpu_pipeline_task.hpp` containing both `convertible_gpu_pipeline_task` and `convertible_gpu_pipeline_task_provider`. Matches Phase 5 (D-02) and Phase 6 (D-05) pattern

### Testing Strategy
- **D-10:** GPU integration tests reusing existing test utilities (Phase 6 pattern: D-06/D-07) — create real tasks with data batches, convert through the converter registry, validate results
- **D-11:** Test RAII semantics: task returned to queue on normal destruction, on conversion failure, and on exception
- **D-12:** Test predicate filtering: non-gpu_pipeline_tasks skipped, tasks without pipelineable_operator_data skipped, only tasks with matching memory_space and batch_state::task_created selected

### Claude's Discretion
- Exact test case structure and Catch2 tag naming
- How to construct mock/real `gpu_pipeline_task` instances for testing (may need minimal pipeline setup)
- Internal helper methods within the implementation classes
- Whether `bytes_in_space()` sums across all batches in the task or only matching-state batches
- Return value semantics for `convert()` when some batches succeed and some fail

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Abstract Interfaces (Phase 5 output)
- `src/include/data/convertible_data.hpp` — `convertible_data` and `convertible_data_provider` abstract interfaces with exact signatures

### Sibling Implementation (Phase 6 output)
- `src/include/data/convertible_data_batch.hpp` — `convertible_data_batch` and `convertible_data_batch_provider` — the pattern to follow for task queue equivalents

### Task Types
- `src/include/pipeline/gpu_pipeline_task.hpp` — `gpu_pipeline_task` and `gpu_pipeline_task_local_state` with `_input_data` (unique_ptr\<operator_data\>)
- `src/include/pipeline/sirius_pipeline_itask.hpp` — `sirius_pipeline_itask` base class
- `src/include/parallel/task.hpp` — `itask` base class with `local_state()` public accessor and `itask_local_state`

### Operator Data
- `src/include/op/sirius_physical_operator.hpp` — `operator_data`, `pipelineable_operator_data` (with `get_data_batches()`), `partitioned_operator_data`

### Queue
- `src/include/exec/inspectable_mpsc.hpp` — `inspectable_mpsc<T>` with `mutable_pop_if`, `push`, queue semantics

### Conversion Pattern (reference)
- `src/downgrade/downgrade_task.cpp` — original save-prev_state / lock / convert / restore pattern
- `src/include/pipeline/batch_lock_utils.hpp` — `lock_or_prepare_batch()` showing in_transit lock pattern

### Converter Registry
- `src/include/data/sirius_converter_registry.hpp` — singleton `converter_registry::get()`

### Data Batch State Machine
- `cucascade/include/cucascade/data/data_batch.hpp` — `data_batch` with `try_to_lock_for_in_transit()`, `try_to_release_in_transit(prev_state)`, `get_state()`, `get_memory_space()`

### Memory Types
- `cucascade/include/cucascade/memory/memory_space.hpp` — `memory_space` class
- `cucascade/include/cucascade/memory/common.hpp` — `Tier` enum, `memory_space_id`

### Test Utilities (must reuse)
- `test/cpp/operator/operator_test_utils.hpp` — memory manager setup, batch creation helpers, GPU space accessor
- `test/cpp/scan/test_utils.hpp` — `drain_data_repo()`, alternative memory manager setup
- `test/cpp/utils/test_validation_utility.hpp` — batch and table comparison utilities
- `test/cpp/utils/utils.hpp` — random data generation, cudf table creation
- `src/include/data/data_batch_utils.hpp` — `make_data_batch()` factory

### Requirements
- `.planning/REQUIREMENTS.md` — TASK-01, TASK-02, TASK-03 with exact behavioral requirements

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `convertible_data_batch::convert()` — complete implementation of the per-batch lock/convert/restore pattern to replicate for each data_batch inside a task
- `convertible_data_batch_provider` — iteration and filtering pattern to adapt for queue-based search via `mutable_pop_if`
- `sirius::converter_registry::get()` — singleton access to cucascade `representation_converter_registry`
- `itask::local_state()` — public accessor returning `itask_local_state*`, enables cast chain to reach data_batches

### Established Patterns
- Header-only in `src/include/data/` (both `convertible_data.hpp` and `convertible_data_batch.hpp` follow this)
- `memory_space*` for all memory space parameters (non-copyable type, project decision)
- `std::unique_ptr<convertible_data>` ownership from providers
- `dynamic_cast` for type-safe downcasting in predicate functions
- RAII for resource management throughout the codebase

### Integration Points
- `convertible_gpu_pipeline_task` wraps `std::unique_ptr<itask>` — extracted from queue via `mutable_pop_if`
- `convertible_gpu_pipeline_task_provider` wraps `inspectable_mpsc<itask>&` — uses `mutable_pop_if(predicate, front_to_back=false)`
- `sirius_memory_reservation_manager` passed to `convert()` for memory reservation (same as Phase 6)
- Logging dependency for destructor warning: `log/logging.hpp` (SIRIUS_LOG_WARN)

</code_context>

<specifics>
## Specific Ideas

- The `dynamic_cast` chain in the predicate should be defensive: any cast failure means "skip this task", not an error. The queue may contain heterogeneous task types
- Per-batch conversion independence means the `convert()` return value needs careful definition — Claude can decide whether true means "all attempted batches converted" or "at least one batch converted"

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 07-task-queue-conversion*
*Context gathered: 2026-04-15*
