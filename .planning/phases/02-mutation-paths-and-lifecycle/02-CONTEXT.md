# Phase 2: Mutation Paths and Lifecycle - Context

**Gathered:** 2026-04-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Update all conversion/downgrade code to use `to_mutable()` for exclusive access, update the result collector to use `read_only_data_batch::clone_to`, wire subscribe/unsubscribe lifecycle into `gpu_pipeline_task`, and remove the old batch_state machine (`task_created`, `in_transit`, `processing`) and its associated lock functions. Operator call-site sweep is Phase 3 scope.

</domain>

<decisions>
## Implementation Decisions

### Conversion locking strategy
- **D-01:** Add a `bool blocking` parameter to `convertible_data_batch::convert` and `convertible_gpu_pipeline_task::convert`. When `blocking=true`, use `to_mutable()` (blocking). When `blocking=false`, use `try_to_mutable()` (returns nullopt if busy).
- **D-02:** Default all current call sites to `blocking=true`. The user will manually review each call site post-implementation to decide which should be non-blocking.
- **D-03:** The `convert()` pattern becomes: acquire `mutable_data_batch` (blocking or try) -> `convert_to()` on the mutable batch -> release (RAII destructor handles transition back to idle).

### Provider filtering criteria
- **D-04:** Both `convertible_data_batch_provider` and `convertible_gpu_pipeline_task_provider` filter candidate batches by checking `batch_state::idle` only. This replaces the old dual-check pattern (idle for repo batches, task_created for pipeline task batches).

### Result collector conversion
- **D-05:** `sirius_physical_materialized_collector::sink` replaces the current clone-then-convert_to pattern with `read_only_data_batch::clone_to`. Acquire a `read_only_data_batch` from the input batch, then call `clone_to()` to clone directly into the target HOST representation. This eliminates the intermediate clone+convert two-step.

### Subscribe/unsubscribe wiring
- **D-06:** `subscribe()` is called in the `gpu_pipeline_task` constructor when it receives its input data batches. `unsubscribe()` is called in the `gpu_pipeline_task` destructor. This centralizes lifecycle management in one place since all operators create tasks through this type.

### Claude's Discretion
- Internal state-save/restore cleanup in convertible classes (the `prev_state` pattern is replaced by RAII)
- Error logging adjustments for the new locking API
- Whether to keep the `bytes_in_space` helper methods or inline them (they also access private members on idle batches, but may be deferred to Phase 3 accessor sweep)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### cucascade new API
- `cucascade/include/cucascade/data/data_batch.hpp` -- Defines `data_batch`, `read_only_data_batch`, `mutable_data_batch`, transition methods (`to_read_only()`, `to_mutable()`, `try_to_mutable()`, `readonly_to_mutable()`, `mutable_to_readonly()`, `to_idle()`), `subscribe()`, `unsubscribe()`, `clone_to()`
- `cucascade/include/cucascade/data/data_repository.hpp` -- Updated signatures: `pop_idle_data_batch()`, `get_data_batch_by_id(id, partition)` (state param removed)

### Sirius conversion/downgrade files (primary targets)
- `src/include/data/convertible_data_batch.hpp` -- `convertible_data_batch::convert` and `convertible_data_batch_provider` (CONV-01)
- `src/include/data/convertible_gpu_pipeline_task.hpp` -- `convertible_gpu_pipeline_task::convert` and provider (CONV-02)
- `src/op/sirius_physical_result_collector.cpp` -- Materialized collector sink with GPU->HOST conversion (CONV-03)
- `src/include/data/convertible_data.hpp` -- Base interface that `convert()` signature change propagates to

### Sirius lifecycle files (subscribe/unsubscribe)
- `src/include/pipeline/gpu_pipeline_task.hpp` -- `gpu_pipeline_task` class where subscribe/unsubscribe is wired (LIFE-01, LIFE-02)
- `src/pipeline/gpu_pipeline_task.cpp` -- Task implementation with constructor/destructor
- `src/downgrade/downgrade_executor.cpp` -- Drives convert flow via providers; consumes the updated convert API

### Phase 1 context (prior decisions)
- `.planning/phases/01-pipeline-data-path/01-CONTEXT.md` -- Established read_only_data_batch flow, type hierarchy, conversion chain pattern

### Project requirements
- `.planning/REQUIREMENTS.md` -- CONV-01 through CONV-03, LIFE-01 through LIFE-04 define acceptance criteria

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `convertible_data` interface (`src/include/data/convertible_data.hpp`): Base class with virtual `convert()` -- signature needs `bool blocking` parameter added
- `convertible_data_provider` interface: Provider pattern for batch discovery -- filtering logic changes but structure stays
- `convertible_data_batch_provider` and `convertible_gpu_pipeline_task_provider`: Two concrete providers that drive the downgrade executor

### Established Patterns
- State save/restore in `convertible_data_batch::convert`: saves `prev_state`, locks, converts, restores on all paths. New API replaces this with RAII -- `mutable_data_batch` destructor handles state transition back to idle
- Provider iteration: back-to-front for downgrade (evict most recently added data first), with `get_all_convertible` snapshot approach
- `dynamic_cast` chain in `convertible_gpu_pipeline_task::get_pipelineable_data`: navigates itask -> gpu_pipeline_task -> local_state -> pipelineable_operator_data

### Integration Points
- `downgrade_executor::processing_loop` -- Calls `provider.get_all_convertible()` then `candidate->convert()` for each. Will pass `blocking=true` to convert.
- 10+ operator files reference `batch_state::task_created` in `pop_data_batch` / `pop_data_batch_by_id` / `get_data_batch_by_id` calls (Phase 3 sweep, but LIFE-03 removes the state enum values these reference)
- `gpu_pipeline_task` constructor/destructor -- Integration point for subscribe/unsubscribe wiring

</code_context>

<specifics>
## Specific Ideas

- The `bool blocking` parameter on convert is intentionally set to `true` everywhere initially -- the user will manually audit each call site post-implementation to determine which should be non-blocking (`try_to_mutable()`)
- Result collector uses `read_only_data_batch::clone_to` -- this is a cucascade d9dc331 API method that clones directly into a target representation, avoiding the clone+convert two-step
- The old `prev_state` save/restore pattern in convert methods is entirely replaced by RAII semantics of `mutable_data_batch`

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 02-mutation-paths-and-lifecycle*
*Context gathered: 2026-04-22*
