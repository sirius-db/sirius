# Phase 2: Request Execution and API - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 02-request-execution-and-api
**Areas discussed:** Dispatch loop redesign, bytes_freed accounting, Public API surface, run_downgrade_pass fate

---

## Dispatch loop redesign

| Option | Description | Selected |
|--------|-------------|----------|
| Incremental dispatch | Dispatch up to pool-width batches concurrently. After each completion (atomic counter + CV), check predicate. Stop dispatching when satisfied, in-flight finish naturally. | ✓ |
| Wave-based dispatch | Dispatch N batches (pool width), wait_all, check predicate, repeat. Coarser — predicate only checked between waves. | |
| Callback-based completion | Each task calls shared completion handler that atomically updates bytes_freed, checks predicate, signals dispatch thread. Most reactive but more complex sync. | |

**User's choice:** Incremental dispatch
**Notes:** None

### Follow-up: Null predicate handling

**User's clarification:** Every request must always have a predicate. If the request is byte-based and the user does not provide an explicit predicate, the API constructs a default predicate that checks `bytes_freed >= target_bytes`. No null-predicate code path.

---

## bytes_freed accounting

| Option | Description | Selected |
|--------|-------------|----------|
| Atomic counter in request | Each request owns atomic<size_t> bytes_freed. Dispatch lambdas add batch size after execute(). Default byte-predicate captures reference to counter. | ✓ |
| Query memory_space after each batch | Call memory_space->get_current_usage() post-completion. More accurate but adds sync cost and coupling. | |
| Pre-dispatch estimation only | Sum batch sizes at collection time. Simplest but can't do early-exit accurately. | |

**User's choice:** Atomic counter in request
**Notes:** None

---

## Public API surface

| Option | Description | Selected |
|--------|-------------|----------|
| Separate methods | request_free_memory(bytes), request_free_memory_and_wait(bytes), request_downgrade(predicate) as distinct public methods. Clear intent. | ✓ |
| Single method with optional predicate | request_free_memory(bytes, predicate = nullptr). Fewer methods but muddier intent. | |
| Overloaded request_free_memory | Two overloads by argument type. Compiler picks. Blocking variant only for byte-based. | |

**User's choice:** Separate methods
**Notes:** None

---

## run_downgrade_pass fate

| Option | Description | Selected |
|--------|-------------|----------|
| Remove both | All downgrade work flows through request queue. No bypass path. Single code path. | ✓ |
| Keep as internal helpers | Make private. Processing loop uses collect_all_candidates internally. Less churn. | |
| Keep public, deprecate later | Leave for Phase 3 callers. Mark with comment. Remove when monitor_loop migrates. | |

**User's choice:** Remove both
**Notes:** None

---

## Claude's Discretion

- Exact condition variable / notification mechanism for dispatch-thread wakeup
- Whether request_downgrade(predicate) takes a target_bytes hint for candidate collection
- Internal error handling details within dispatch lambdas
- Thread synchronization details in processing_loop

## Deferred Ideas

None — discussion stayed within phase scope
