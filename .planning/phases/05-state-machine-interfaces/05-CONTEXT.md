# Phase 5: State Machine & Interfaces - Context

**Gathered:** 2026-04-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend `data_batch` state transitions to support `task_created -> in_transit` and define abstract conversion contracts (`convertible_data`, `convertible_data_provider`). No concrete implementations — those are Phase 6 (batch) and Phase 7 (task queue).

</domain>

<decisions>
## Implementation Decisions

### Interface Location
- **D-01:** Abstract interfaces (`convertible_data`, `convertible_data_provider`) live in `src/include/data/` within the `sirius` namespace
- **D-02:** Single header file: `src/include/data/convertible_data.hpp` containing both interfaces. Provider depends on convertible_data, so they belong together (same pattern as `data_batch.hpp` containing both `data_batch` and `data_batch_processing_handle`)

### State Machine Changes
- **D-03:** Modify `data_batch` directly in the cucascade submodule — update `try_to_lock_for_in_transit()` to accept `task_created` state (not just `idle`), and validate `try_to_release_in_transit()` supports `task_created` as target state
- **D-04:** `task_created_count` is preserved across the in_transit round-trip. When a batch transitions `task_created -> in_transit`, the pending task count stays intact. When releasing back to `task_created`, the count is restored. The pending task remains valid — we just moved the data

### Testing
- **D-05:** Compile-only verification for the abstract interfaces — verify headers compile and interfaces can be subclassed. No mock implementations in this phase
- **D-06:** Unit tests (Catch2) for the new state machine transitions: `task_created -> in_transit` succeeds, `in_transit -> task_created` restores correctly, `task_created_count` preserved across round-trip

### Claude's Discretion
- Exact state transition validation logic (how strictly to check preconditions)
- State transition diagram update formatting in data_batch.hpp comments
- Test file organization within test/cpp/

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### State Machine
- `cucascade/include/cucascade/data/data_batch.hpp` — data_batch class with state machine, try_to_lock_for_in_transit(), try_to_release_in_transit(), batch_state enum
- `cucascade/include/cucascade/data/data_repository.hpp` — data_repository that holds data_batches, used by convertible_data_provider in Phase 6

### Conversion Pattern (reference implementation)
- `src/downgrade/downgrade_task.cpp` — existing save-prev_state/lock/convert/restore pattern that convertible_data::convert() will generalize
- `src/include/pipeline/batch_lock_utils.hpp` — lock_or_prepare_batch() showing the in_transit lock pattern with prev_state save/restore

### Converter Registry
- `src/include/data/sirius_converter_registry.hpp` — singleton converter registry used by convert() implementations

### Memory Types
- `cucascade/include/cucascade/memory/memory_space.hpp` — memory_space class (convert() parameter type)
- `cucascade/include/cucascade/memory/common.hpp` — memory_space_id, Tier enum

### Requirements
- `.planning/REQUIREMENTS.md` — STATE-01, STATE-02, IFACE-01, IFACE-02 with exact function signatures

### Task Queue (context for Phase 7 compatibility)
- `src/include/pipeline/gpu_pipeline_task.hpp` — gpu_pipeline_task and gpu_pipeline_task_local_state that Phase 7 will wrap
- `src/include/exec/inspectable_mpsc.hpp` — the queue that Phase 7's provider will search via mutable_pop_if

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `downgrade_task::execute()` — the save/lock/convert/restore pattern that convertible_data::convert() will generalize
- `batch_lock_utils::lock_or_prepare_batch()` — demonstrates in_transit locking from both idle and task_created states (already handles prev_state)
- `sirius::converter_registry` — singleton access to cucascade representation_converter_registry

### Established Patterns
- Header-only templates in `src/include/` (inspectable_mpsc.hpp pattern)
- `memory_space*` for all memory space parameters (non-copyable type, project decision)
- `std::unique_ptr<T>` for ownership transfer, `std::shared_ptr<T>` for shared ownership
- Pure virtual interfaces with virtual destructor (`= default`) in cucascade (e.g., `idata_representation`, `idata_batch_probe`)

### Integration Points
- `data_batch` state machine in cucascade submodule — direct modification needed
- `sirius_memory_reservation_manager&` in convert() signature ties interface to sirius namespace
- Phase 6 will implement `convertible_data_batch` wrapping `shared_ptr<data_batch>`
- Phase 7 will implement `convertible_gpu_pipeline_task` wrapping `unique_ptr<itask>`

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 05-state-machine-interfaces*
*Context gathered: 2026-04-15*
