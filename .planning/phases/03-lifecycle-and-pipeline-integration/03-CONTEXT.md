# Phase 3: Lifecycle and Pipeline Integration - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the redesigned downgrade executor into its callers so it becomes a drop-in replacement. SiriusContext manages it via start/stop/drain, the monitor loop uses it for memory pressure response, and gpu_pipeline_executor reclaims memory through it when reservations fall short. LIFE-01 through LIFE-05 are verified via tests (already implemented in Phases 1-2). PIPE-01 through PIPE-03 are the new code changes.

</domain>

<decisions>
## Implementation Decisions

### Pipeline retry strategy
- **D-01:** When `reservation->size() < bytes_needs`, call `request_free_memory_and_wait(bytes_needs - reservation->size())` on the downgrade executor, then retry `make_reservation`. Loop up to 5 attempts total.
- **D-02:** No delay between retry attempts — the blocking `request_free_memory_and_wait` call itself is the wait. Retry immediately after it returns.
- **D-03:** After 5 failed retries (still partial reservation), proceed with the partial reservation and execute the task with reduced memory. Log a warning. This matches the current behavior when `reservation->size() != bytes_needs`.

### Downgrade executor access
- **D-04:** Add `downgrade_executor*` as a constructor parameter to `gpu_pipeline_executor`. Direct, explicit dependency — no runtime lookup via SiriusContext. The constructor already takes `memory_space*`, so adding the executor that manages that space is natural. Update all call sites that construct `gpu_pipeline_executor` to pass the corresponding downgrade executor.

### Lifecycle verification
- **D-05:** LIFE-01 through LIFE-05 are already implemented in Phases 1-2. Phase 3 writes unit tests that exercise these from SiriusContext's perspective: start/stop/drain correctness (LIFE-01), drain shared_ptr guarantee (LIFE-02), monitor loop integration (LIFE-03), concurrent API safety (LIFE-04), and CUDA stream lifecycle (LIFE-05). No code changes unless tests reveal gaps.

### drain() shared_ptr guarantee
- **D-06:** Current `drain()` implementation is sufficient for LIFE-02. `pool->wait_all()` ensures all dispatch lambdas have returned, releasing all `shared_ptr<data_batch>` captures. Queue drain drops pending requests that haven't been dispatched (no batch refs). No explicit ref-counting verification needed.

### Claude's Discretion
- Exact retry loop structure (for loop vs while with counter)
- Whether to release and re-acquire reservation on each retry, or attempt to grow the existing one
- Internal logging verbosity for retry attempts (TRACE vs DEBUG vs WARN)
- Test fixture design for lifecycle tests (mock memory_space vs real)
- Whether `gpu_pipeline_executor` stores `downgrade_executor*` directly or wraps in a helper

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Downgrade executor (current implementation — Phase 2 output)
- `src/downgrade/downgrade_executor.cpp` — processing_loop, monitor_loop, start/stop/drain, collect_all_candidates
- `src/include/downgrade/downgrade_executor.hpp` — Class definition with downgrade_request, public API (request_free_memory, request_free_memory_and_wait, request_downgrade)
- `src/downgrade/downgrade_task.cpp` — Task execute() logic (GPU->HOST conversion)
- `src/include/downgrade/downgrade_task.hpp` — Plain struct task type

### gpu_pipeline_executor (integration target)
- `src/pipeline/gpu_pipeline_executor.cpp` — manager_loop with reservation acquisition (lines 87-117), constructor (lines 36-45)
- `src/include/pipeline/gpu_pipeline_executor.hpp` — Class definition, constructor signature

### SiriusContext (lifecycle caller)
- `src/sirius_context.cpp` — Creates downgrade_executor (lines 181-190), calls drain at QueryEnd (lines 113-115), start at Initialize
- `src/include/sirius_context.hpp` — get_downgrade_executor(space_id), downgrade_executors_ vector

### Infrastructure
- `src/include/exec/bounded_thread_pool.hpp` — Thread pool with reserve/dispatch/interrupt/resume/wait_all
- `src/include/exec/interruptible_mpmc.hpp` — MPMC queue with interrupt support

### Requirements
- `.planning/REQUIREMENTS.md` — LIFE-01 through LIFE-05, PIPE-01 through PIPE-03 (Phase 3 requirements)

### Prior phase context
- `.planning/phases/01-foundation/01-CONTEXT.md` — Foundation decisions (D-01 through D-07)
- `.planning/phases/02-request-execution-and-api/02-CONTEXT.md` — Request execution decisions (D-01 through D-07)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `downgrade_executor::request_free_memory_and_wait(bytes)` — Blocking API ready to use from gpu_pipeline_executor; returns actual bytes freed
- `SiriusContext::get_downgrade_executor(space_id)` — Already exists for looking up executor by memory space; useful at construction sites
- `gpu_pipeline_executor::_memory_space` — Already stores memory_space pointer; space_id available for matching to downgrade executor

### Established Patterns
- Constructor injection: `gpu_pipeline_executor` already takes `memory_space*` and `task_request_publisher` — adding `downgrade_executor*` follows the same pattern
- Warn-and-continue on partial reservation: existing behavior at `gpu_pipeline_executor.cpp:108-116` — retry loop preserves this as the final fallback
- `cudaSetDevice` per-thread init via pool's `per_thread_init` callback — already established in downgrade_executor::start()

### Integration Points
- `gpu_pipeline_executor::manager_loop()` at line 98 (`make_reservation`) — insertion point for retry-with-downgrade loop
- `SiriusContext::Initialize()` around line 183 — where downgrade executors are constructed; also where gpu_pipeline_executors would receive their downgrade_executor pointer
- `task_creator` — constructs gpu_pipeline_executor instances; must be updated to pass downgrade_executor

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

*Phase: 03-lifecycle-and-pipeline-integration*
*Context gathered: 2026-04-06*
