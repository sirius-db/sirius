# Phase 1: Foundation - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Decouple `downgrade_executor` from `itask_executor` inheritance. Give it its own `bounded_thread_pool`, request queue, and processing thread. Requests are processed sequentially. Candidate selection logic is preserved verbatim. This phase establishes the structural skeleton — Phase 2 adds predicate-driven execution and the full public API.

</domain>

<decisions>
## Implementation Decisions

### Request struct shape
- **D-01:** `downgrade_request` has the full skeleton from day one: `std::function<bool()> predicate`, `std::promise<size_t> result`, and `size_t target_bytes`. Only `target_bytes` is exercised in Phase 1; predicate and promise are present but unused until Phase 2 wires them up.

### Queue + processing thread
- **D-02:** Request queue uses `exec::interruptible_mpmc<downgrade_request>` — reuses the proven primitive, supports interrupt/resume for drain semantics.
- **D-03:** Processing thread uses collect-then-dispatch: runs candidate selection first (single-threaded), collects all candidates up to `target_bytes`, dispatches all batch downgrades to the pool at once, then calls `pool->wait_all()`. Phase 2 can evolve to incremental dispatch when it adds predicate-after-each-batch.

### task_completion_message_queue
- **D-04:** Remove `task_completion_message_queue` from the downgrade path entirely. The processing thread uses `pool->wait_all()` to track completion — no need to notify `task_creator`. This also removes the `_message_queue` member from `downgrade_executor` and the `_message_queue` reference from `downgrade_task_global_state` (which itself is being removed per D-06).

### Decoupling strategy
- **D-05:** Direct composition, no base class. `downgrade_executor` owns a `bounded_thread_pool`, `interruptible_mpmc<downgrade_request>`, processing thread, monitor thread, and `atomic<bool> _running` as direct members. Implements its own `start()/stop()/drain()`. No inheritance from `itask_executor`, no virtual dispatch.
- **D-06:** `downgrade_task` becomes a plain struct with direct members (`shared_ptr<data_batch> batch`, `sirius_memory_reservation_manager& res_mgr`) and an `execute(rmm::cuda_stream_view)` method. The `itask` base class, `downgrade_task_global_state`, and `downgrade_task_local_state` are all removed from the downgrade path. No polymorphism, no `cast<>()` ceremony.

### Candidate selection
- **D-07:** The candidate selection and ordering logic from `run_downgrade_pass` is ported verbatim, not redesigned. This includes: repo scoring (partitioned first, descending tier data size), two-pass partition walk (non-active last-to-first, then active last-to-first), `collect_candidates_from_partition` (idle batches on source space up to max_bytes), `is_partition_active` (checks task_created/processing states), and `get_repo_data_size_on_tier`. The static helpers move into the new class as-is.

### Claude's Discretion
- Exact start()/stop()/drain() implementation details (interrupt sequencing, thread join order)
- Whether static helpers remain static or become private methods
- Internal error handling within the processing thread loop

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Downgrade executor (current implementation)
- `src/downgrade/downgrade_executor.cpp` — Current implementation with candidate selection, monitor loop, manager loop
- `src/include/downgrade/downgrade_executor.hpp` — Current class definition inheriting from itask_executor
- `src/downgrade/downgrade_task.cpp` — Current task execute() logic (GPU->HOST conversion via converter_registry)
- `src/include/downgrade/downgrade_task.hpp` — Current task types (downgrade_task, global_state, local_state)

### Base class being decoupled from
- `src/include/parallel/task_executor.hpp` — itask_executor base class with start/stop/schedule/drain_and_wait
- `src/include/parallel/task.hpp` — itask, itask_global_state, itask_local_state base types

### Infrastructure being composed
- `src/include/exec/bounded_thread_pool.hpp` — Thread pool with reserve/dispatch/interrupt/resume/wait_all
- `src/include/exec/interruptible_mpmc.hpp` — MPMC queue with interrupt support (for request queue)
- `src/include/exec/config.hpp` — thread_pool_config used by constructor

### Callers
- `src/sirius_context.cpp` — Creates downgrade_executor, calls start/stop/drain/get_space_id

### Requirements
- `.planning/REQUIREMENTS.md` — EXEC-01, EXEC-02, CAND-01, CAND-02 (Phase 1 requirements)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `exec::bounded_thread_pool` — Already has reserve/dispatch/interrupt/resume/wait_all; can be composed directly
- `exec::interruptible_mpmc<T>` — Template queue with interrupt support; reusable for `downgrade_request`
- `downgrade_task::execute()` logic — GPU->HOST conversion via converter_registry, in-transit locking, reservation manager; core logic preserved in new plain struct

### Established Patterns
- `bounded_thread_pool` slot-based dispatch: `reserve()` -> `dispatch(slot, lambda)` -> slot auto-releases on completion
- `interruptible_mpmc` interrupt/resume for drain patterns (used by itask_executor today)
- CUDA stream lifecycle: create on start, destroy on stop (current on_start/on_stopped pattern)
- `cudaSetDevice` per-thread init via pool's `per_thread_init` callback

### Integration Points
- `SiriusContext::Initialize()` — constructs downgrade_executor per GPU memory space; constructor signature must remain compatible
- `SiriusContext` calls `start()`, `stop()`, `drain()`, `get_space_id()` — public API surface preserved
- `monitor_loop` calls `should_downgrade_memory()` and `get_amount_to_downgrade()` on `memory_space` — unchanged
- `run_downgrade_pass_all_repos` iterates `data_repo_mgr.for_each_repository()` — unchanged

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-foundation*
*Context gathered: 2026-04-06*
