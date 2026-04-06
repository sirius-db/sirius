# Downgrade Executor Redesign

## What This Is

A thorough redesign of the `downgrade_executor` class and its supporting types (`downgrade_task`, `downgrade_task_local_state`, `downgrade_task_global_state`) in Sirius. The redesign shifts the unit of work from "downgrade a single data_batch" to "free a target amount of memory (or satisfy a predicate) by downgrading data_batches concurrently." This is an internal infrastructure change within the Sirius GPU-native SQL engine.

## Core Value

The downgrade executor must reliably free GPU memory on demand — both asynchronously (fire-and-forget) and synchronously (block until done) — so that upstream components can request memory reclamation with predictable completion semantics.

## Requirements

### Validated

- [x] Own thread pool (drop itask_executor): the downgrade_executor no longer inherits from `itask_executor`; it owns its own `bounded_thread_pool` and request queue — *Validated in Phase 1: Foundation*
- [x] Sequential request processing: requests are queued and executed one at a time (only one thread pool wave active at once) — *Validated in Phase 1: Foundation*
- [x] Candidate selection logic preserved: `collect_candidates_from_partition` and `run_downgrade_pass` selection/prioritization logic (partitioned repos first, non-active partitions first, last-to-first order) remains intact — *Validated in Phase 1: Foundation*
- [x] Predicate-based request API: the fundamental unit of work is a request that takes a lambda `() -> bool` predicate and downgrades data_batches until the predicate returns true or candidates are exhausted — *Validated in Phase 2: Request Execution and API*
- [x] Byte-based convenience API: `request_free_memory(size_t bytes)` wraps the predicate API with a lambda that checks current memory consumption against the target — *Validated in Phase 2: Request Execution and API*
- [x] Blocking API: `request_free_memory_and_wait(size_t bytes)` blocks until the request completes and returns the number of bytes actually freed — *Validated in Phase 2: Request Execution and API*
- [x] Async API: `request_free_memory(size_t bytes)` returns `std::future<size_t>` that the caller can poll or wait on later — *Validated in Phase 2: Request Execution and API*
- [x] Predicate checked after each batch: after every individual data_batch downgrade completes, the predicate is evaluated; if true, no new batches are dispatched (in-flight batches finish naturally) — *Validated in Phase 2: Request Execution and API*
- [x] Concurrent batch downgrades within a request: a thread pool performs multiple batch downgrades simultaneously within a single request — *Validated in Phase 2: Request Execution and API*
- [x] Partial fulfillment: if not enough idle batches exist to satisfy the request, free what's available and return the actual bytes freed — *Validated in Phase 2: Request Execution and API*

### Active

None — all requirements validated.

### Recently Validated (Phase 3)

- [x] Monitor loop preserved: the existing polling loop that checks `should_downgrade_memory()` continues to exist, triggering downgrade passes via the internal request queue — *Validated in Phase 3: Lifecycle and Pipeline Integration*
- [x] Retain start/stop/drain semantics: `start()`, `stop()`, `drain()` methods continue to exist with equivalent behavior to today, used by `SiriusContext` — *Validated in Phase 3: Lifecycle and Pipeline Integration*
- [x] gpu_pipeline_executor integration: retry-with-downgrade loop calls `request_free_memory_and_wait` up to 5 times when reservation falls short — *Validated in Phase 3: Lifecycle and Pipeline Integration*
- [x] SiriusContext initialization order: downgrade executors created before pipeline_executor so pointers are available at construction — *Validated in Phase 3: Lifecycle and Pipeline Integration*

### Out of Scope

- Changing the `itask_executor` base class itself — other executors (`pipeline_executor`, `duckdb_scan_executor`) still use it
- HOST→DISK downgrade — not yet implemented, stays out of scope
- Changing how `SiriusContext` creates/manages downgrade executors (one per GPU memory space) — only the executor's internal design changes
- Retry/timeout semantics for requests — caller handles retries if partial fulfillment is insufficient

## Context

- The downgrade executor is created per GPU memory space in `SiriusContext::Initialize()` (`src/sirius_context.cpp`)
- It is drained at `QueryEnd` to ensure no downgrade tasks hold `shared_ptr<data_batch>` references to batches about to be destroyed
- The current implementation inherits from `itask_executor` which provides a queue-of-tasks model; the new design needs a queue-of-requests model, making the inheritance a poor fit
- Existing files: `src/downgrade/downgrade_executor.cpp`, `src/include/downgrade/downgrade_executor.hpp`, `src/include/downgrade/downgrade_task.hpp`
- Callers: `SiriusContext` (creation, start, stop, drain), monitor_loop (internal), and potentially upstream components that will use the new request APIs
- The `task_completion_message_queue` is currently used by downgrade tasks to notify the task_creator; the redesign may simplify or remove this dependency

## Constraints

- **Thread safety**: All public APIs must be safe to call from any thread; the monitor loop runs on its own thread
- **CUDA device affinity**: Thread pool workers must call `cudaSetDevice` on init (same as today)
- **Non-fatal failures**: Individual batch downgrade failures must not crash the executor — log and continue
- **No breaking SiriusContext**: `SiriusContext` calls `start()`, `stop()`, `drain()`, `get_space_id()` — these must continue to work

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Drop itask_executor inheritance | The base class queue-of-tasks model doesn't fit queue-of-requests; fighting the abstraction adds complexity | ✓ Validated Phase 1 |
| Predicate as fundamental API | More flexible than byte-count-only; byte-based API is a thin wrapper | ✓ Validated Phase 2 |
| std::future for async result | Simple, standard, no callback complexity; caller can poll or block | ✓ Validated Phase 2 |
| Sequential request processing | Avoids contention between concurrent requests competing for the same batches | ✓ Validated Phase 1 |
| Predicate checked after each batch (not per-wave) | Enables earliest possible early-exit, minimizing unnecessary downgrades | ✓ Validated Phase 2 |
| Monitor uses fire-and-forget, not blocking API | Monitor is internal to executor; pushing to its own queue is more efficient than calling the external blocking API | ✓ Validated Phase 3 |
| Constructor injection for downgrade_executor* | nullptr default preserves backward compatibility; matching by space_id ensures correct GPU affinity | ✓ Validated Phase 3 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-06 after Phase 3 completion — all 3 phases complete, milestone v1.0 done. Downgrade executor redesign fully integrated into pipeline and verified with lifecycle tests.*
