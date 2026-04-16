# Phase 8: API Cleanup + Processing Loop Refactor - Context

**Gathered:** 2026-04-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove `target_bytes` from the downgrade request API and `gpu_pipeline_executor` calculation, and simultaneously replace the `downgrade_executor` processing loop with convertible_data providers using tiered candidate fetching and `convert()`-based conversion. This is a fused phase combining the original Phase 8 (API Cleanup) and Phase 9 (Processing Loop Refactor) — removing `target_bytes` and replacing the collection mechanism are tightly coupled changes.

**Requirements covered:** DAPI-01, DAPI-02, LOOP-01, LOOP-02, LOOP-03, LOOP-04, LOOP-05, LOG-01

</domain>

<decisions>
## Implementation Decisions

### Candidate collection strategy
- **D-01:** Replace `collect_all_candidates` with `convertible_data_batch_provider` per `data_repository`, fetching convertible_datas lazily on a repo-by-repo basis until the predicate is satisfied
- **D-02:** Remove `collect_all_candidates` and all its helpers (`get_repo_data_size_on_tier`, `is_partition_active`, `collect_candidates_from_partition`, `scored_repo` struct) as dead code in this phase — no stale code left behind

### API surface
- **D-03:** `request_downgrade` loses the `target_bytes` parameter — signature becomes `request_downgrade(std::function<bool()> predicate)`
- **D-04:** `downgrade_request` struct loses the `target_bytes` member entirely
- **D-05:** `request_free_memory(size_t bytes)` and `monitor_loop` keep their existing public signatures; internally they build predicates for the new provider-based processing loop without setting `target_bytes`

### Tiered fallback
- **D-06:** Processing loop fetches candidates in order: data_repositories → gpu_pipeline_executor task queue → pipeline_executor task queue (via `convertible_gpu_pipeline_task_provider`)

### downgrade_task elimination
- **D-07:** Eliminate `downgrade_task` entirely — replace with direct `convertible_data::convert()` calls, which already handle state transitions, failure rollback, and converter registry access. Remove `downgrade_task.hpp`/`.cpp` files.

### Predicate threading
- **D-08:** Predicate checks happen in both the dispatch loop (between dispatches) and in thread pool workers (after each `convert()` to set satisfied flag early). Predicate must remain thread-safe — existing contract preserved.

### Logging
- **D-09:** Replace `target_bytes` in the existing DEBUG log with a per-request summary including per-tier breakdown: batches and bytes freed from data_repositories, gpu_pipeline_executor queue, and pipeline_executor queue. Satisfies LOG-01.

### Claude's Discretion
- Queue wiring mechanism — how `downgrade_executor` gets access to gpu_pipeline_executor and pipeline_executor task queues (constructor injection vs setter methods). Claude will analyze executor construction order in `sirius_context`/`sirius_engine` during research.
- Thread pool dispatch strategy for the new provider-based loop
- Exact processing loop structure (iteration, error handling)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Downgrade executor (primary target)
- `src/include/downgrade/downgrade_executor.hpp` — `downgrade_request` struct, `request_downgrade` signature, `request_free_memory` API
- `src/downgrade/downgrade_executor.cpp` — `processing_loop`, `monitor_loop`, `collect_all_candidates` and helpers
- `src/include/downgrade/downgrade_task.hpp` — `downgrade_task` class to be eliminated

### gpu_pipeline_executor (caller-side cleanup)
- `src/pipeline/gpu_pipeline_executor.cpp` — `target_bytes` calculation logic (line ~116) and `request_downgrade` call (line ~138)

### Convertible data abstractions (replacement infrastructure)
- `src/include/data/convertible_data.hpp` — `convertible_data` and `convertible_data_provider` interfaces
- `src/include/data/convertible_data_batch.hpp` — `convertible_data_batch` wrapping `data_batch`
- `src/include/data/convertible_data_batch_provider.hpp` — `convertible_data_batch_provider` wrapping `data_repository`
- `src/include/data/convertible_gpu_pipeline_task.hpp` — `convertible_gpu_pipeline_task` with RAII queue ownership
- `src/include/data/convertible_gpu_pipeline_task_provider.hpp` — `convertible_gpu_pipeline_task_provider` wrapping `inspectable_mpsc<itask>`

### Existing tests
- `test/cpp/downgrade/test_downgrade_executor.cpp` — Existing downgrade executor tests that must continue passing

### Documentation
- `docs/super-sirius/memory-management.md` — Memory management architecture docs (references target_bytes)
- `docs/super-sirius/optimizations.md` — Optimization docs (references target_bytes)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `convertible_data_batch_provider` — wraps `data_repository` with `get_all_convertible()` and `get_next_convertible()`, ready for lazy iteration
- `convertible_gpu_pipeline_task_provider` — wraps `inspectable_mpsc<itask>` with `mutable_pop_if` for task queue tier fallback
- `convertible_data::convert()` — uniform conversion with failure safety, replaces `downgrade_task::execute()`
- `exec::bounded_thread_pool` — existing thread pool used by downgrade_executor, reusable as-is

### Established Patterns
- `request_free_memory` builds predicate from byte target — same pattern continues, just without `target_bytes` on the struct
- `monitor_loop` derives amount from `memory_space->get_amount_to_downgrade()` — same source, predicate-only path
- Nullable constructor params (e.g., `memory_space*` can be nullptr to disable monitor) — applicable for optional queue references

### Integration Points
- `gpu_pipeline_executor::execute_task()` — caller of `request_downgrade` that computes `target_bytes` (line ~116); this calculation is removed
- `downgrade_executor` constructor in `sirius_context` or `sirius_engine` — wiring point for task queue references
- `_request_queue` (interruptible_mpmc) — unchanged, still used for `downgrade_request` enqueueing

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches within the decisions above.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 08-api-cleanup*
*Context gathered: 2026-04-16*
