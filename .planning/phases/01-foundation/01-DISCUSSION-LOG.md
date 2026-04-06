# Phase 1: Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 01-foundation
**Areas discussed:** Request struct shape, Queue + processing thread, task_completion_message_queue, Decoupling strategy

---

## Request struct shape

| Option | Description | Selected |
|--------|-------------|----------|
| Full skeleton now | downgrade_request has predicate, promise, and target_bytes — all present but only target_bytes used in Phase 1 | ✓ |
| Bytes-only for now | Phase 1 request is just target_bytes + completion signal. Phase 2 adds predicate and future | |
| Predicate-first, no bytes | Struct takes predicate only. target_bytes is a convenience wrapper | |

**User's choice:** Full skeleton now
**Notes:** User wants Phase 2 to wire up what's already there, not restructure.

---

## Queue + processing thread

### Queue primitive

| Option | Description | Selected |
|--------|-------------|----------|
| interruptible_mpmc | Reuse existing template with downgrade_request. Proven, supports interrupt/resume | ✓ |
| mutex + condvar + deque | Simple std::deque protected by mutex + condvar. More explicit, no dependency on interruptible_mpmc | |
| You decide | Let Claude pick | |

**User's choice:** interruptible_mpmc
**Notes:** Consistent with existing patterns.

### Processing thread model

| Option | Description | Selected |
|--------|-------------|----------|
| Collect then dispatch | Selection single-threaded, then dispatch all to pool, then wait_all | ✓ |
| Incremental dispatch | Collect and dispatch incrementally while pool works | |

**User's choice:** Collect then dispatch
**Notes:** Clean separation. Phase 2 can evolve to incremental for predicate-after-each-batch.

---

## task_completion_message_queue

| Option | Description | Selected |
|--------|-------------|----------|
| Remove it | Processing thread uses pool->wait_all(). No need to notify task_creator | ✓ |
| Keep as dead code | Leave wired but unused. Remove in Phase 2 or 3 | |
| You decide | Let Claude determine | |

**User's choice:** Remove it
**Notes:** Clean break.

---

## Decoupling strategy

### Class structure

| Option | Description | Selected |
|--------|-------------|----------|
| Compose directly | Own bounded_thread_pool + interruptible_mpmc as members. Own start/stop. No base class | ✓ |
| Thin base class | Extract minimal base that both executors derive from | |
| You decide | Let Claude determine | |

**User's choice:** Compose directly
**Notes:** No virtual dispatch overhead, clean ownership.

### Task types

| Option | Description | Selected |
|--------|-------------|----------|
| Plain struct + method | downgrade_task as plain struct with batch + res_mgr + execute(). No itask base | ✓ |
| Dispatch lambdas | No downgrade_task class at all, inline as lambdas | |
| Keep itask inheritance | Leave task type hierarchy as-is for Phase 1 | |

**User's choice:** Plain struct + method
**Notes:** User asked for clarification on what global/local state are used for before deciding. After reviewing that execute() only needs batch + reservation_manager + stream, and that task_id is only for the completion message (being removed), chose plain struct.

### Candidate selection

**User's choice:** Port candidate selection logic verbatim from run_downgrade_pass — not redesigned.
**Notes:** User explicitly asked for this to be captured as a decision. Includes repo scoring, two-pass partition walk, all static helpers.

---

## Claude's Discretion

- Exact start()/stop()/drain() implementation details
- Whether static helpers remain static or become private methods
- Internal error handling within the processing thread loop

## Deferred Ideas

None — discussion stayed within phase scope.
