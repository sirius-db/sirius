# Phase 2: Request Execution and API - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the predicate-driven execution engine inside `downgrade_executor` and expose the full public API surface: `request_free_memory(bytes)`, `request_free_memory_and_wait(bytes)`, and `request_downgrade(predicate)`. All downgrade work flows through the request queue — no bypass path. Lifecycle integration (start/stop/drain wiring into SiriusContext) and monitor loop migration are Phase 3.

</domain>

<decisions>
## Implementation Decisions

### Dispatch loop redesign
- **D-01:** The processing loop evolves from collect-all/dispatch-all/wait_all to incremental dispatch: dispatch up to pool-width batches concurrently, check predicate after each batch completion via `atomic<bool> satisfied` flag and condition variable, stop dispatching new batches when predicate is satisfied. In-flight batches finish naturally via `pool->wait_all()` after dispatch loop exits.
- **D-02:** Every request always has a predicate — there is no null-predicate path. Byte-based requests construct a default predicate that checks `bytes_freed >= target_bytes`. The dispatch loop always calls `req.predicate()` after each batch completion, no conditional.

### bytes_freed accounting
- **D-03:** `downgrade_request` gains an `atomic<size_t> bytes_freed{0}` member. Each dispatch lambda adds `batch->get_data()->get_size_in_bytes()` after successful `task.execute()`. The default byte-predicate captures a reference to this counter.
- **D-04:** The final `bytes_freed` value (including in-flight batches that finish after predicate is satisfied) is set into `req.result` (the promise) after `pool->wait_all()` returns.

### Public API surface
- **D-05:** Three separate public methods with distinct signatures:
  - `std::future<size_t> request_free_memory(size_t bytes)` — async, byte-based
  - `size_t request_free_memory_and_wait(size_t bytes)` — blocking, byte-based
  - `std::future<size_t> request_downgrade(std::function<bool()> predicate)` — async, predicate-based
- **D-06:** All three methods build a `downgrade_request`, push it to `_request_queue`, and return. The blocking variant calls `.get()` on the future. No direct dispatch — everything goes through the queue and the processing thread.

### run_downgrade_pass removal
- **D-07:** Remove both `run_downgrade_pass(repos, bytes)` and `run_downgrade_pass_all_repos(bytes)` from the public and private API. All downgrade work flows through the request queue via the new API methods. The candidate collection logic (`collect_all_candidates`) remains as a private helper called by the processing loop.

### Claude's Discretion
- Exact condition variable / notification mechanism for dispatch-thread wakeup after batch completion
- Whether `request_downgrade(predicate)` also takes a `target_bytes` hint for candidate collection, or collects all available candidates
- Internal error handling within dispatch lambdas (log-and-continue is established from Phase 1)
- Exact thread synchronization details in `processing_loop` between dispatch and completion tracking

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Current implementation (Phase 1 output)
- `src/downgrade/downgrade_executor.cpp` — Current processing_loop (collect-all/dispatch-all/wait_all), monitor_loop, candidate collection, run_downgrade_pass methods to be removed
- `src/include/downgrade/downgrade_executor.hpp` — Current class definition with downgrade_request struct, public API surface to be extended
- `src/downgrade/downgrade_task.cpp` — Task execute() logic (unchanged in Phase 2)
- `src/include/downgrade/downgrade_task.hpp` — Plain struct task type (unchanged in Phase 2)

### Infrastructure composed
- `src/include/exec/bounded_thread_pool.hpp` — Thread pool with reserve/dispatch/interrupt/resume/wait_all (dispatch target)
- `src/include/exec/interruptible_mpmc.hpp` — MPMC queue with interrupt support (request queue)

### Requirements
- `.planning/REQUIREMENTS.md` — RAPI-01 through RAPI-05, EXEC-03, EXEC-04, EXEC-05 (Phase 2 requirements)

### Phase 1 context
- `.planning/phases/01-foundation/01-CONTEXT.md` — Foundation decisions carried forward (D-01 through D-07)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `exec::bounded_thread_pool::reserve()` — Returns optional slot; blocks when pool is full, giving natural concurrency limiting for incremental dispatch
- `exec::bounded_thread_pool::wait_all()` — Blocks until all dispatched tasks complete; used after dispatch loop exits to let in-flight batches finish
- `collect_all_candidates()` — Already implements the two-pass prioritization (non-active then active partitions); reusable as-is for the new processing loop
- `downgrade_task::execute(stream)` — Plain struct execute method; dispatch lambdas create these inline

### Established Patterns
- `pool->reserve()` returns nullopt when interrupted — dispatch loop uses this as break condition
- Dispatch lambdas capture batch by shared_ptr, res_mgr by reference, stream by value — same pattern continues
- Try/catch around `task.execute()` with SIRIUS_LOG_ERROR — established error handling for EXEC-05

### Integration Points
- `_request_queue` — Public API methods push here; processing thread pops
- `downgrade_request::result` (std::promise) — Processing loop sets value after wait_all; callers receive via std::future
- `monitor_loop` — Currently calls `run_downgrade_pass_all_repos`; after Phase 2 removes that method, monitor_loop must switch to `request_free_memory` (wired in Phase 3, but the method must exist)

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

*Phase: 02-request-execution-and-api*
*Context gathered: 2026-04-06*
