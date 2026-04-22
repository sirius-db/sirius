# Phase 2: Mutation Paths and Lifecycle - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-22
**Phase:** 02-mutation-paths-and-lifecycle
**Areas discussed:** Conversion locking, Provider filtering, Result collector, Subscribe wiring

---

## Conversion Locking

| Option | Description | Selected |
|--------|-------------|----------|
| try_to_mutable() (Recommended) | Preserves existing skip-busy semantics. Downgrade executor iterates candidates and skips locked batches. | |
| Blocking to_mutable() | Simpler code, matches PROJECT.md constraint. Changes downgrade behavior. | |
| Hybrid approach | try_to_mutable() in downgrade/provider paths, blocking to_mutable() in direct-conversion paths. | |
| Other (user's choice) | Add bool blocking parameter to convert() | ✓ |

**User's choice:** Add a `bool blocking` parameter to convert(). Use blocking `to_mutable()` when true, `try_to_mutable()` when false. Default all current call sites to `blocking=true`. User will manually review each site later.
**Notes:** User wants the flexibility to choose per-call-site but defaults to blocking for now.

---

## Provider Filtering

| Option | Description | Selected |
|--------|-------------|----------|
| Subscriber count (Recommended) | Check subscriber_count() == 0 for free idle, > 0 for assigned. | |
| Try-lock and skip | Don't pre-filter, just try_to_mutable() and skip on failure. | |
| Remove provider filtering | Providers return all batches in target space, let convert() handle contention. | |
| Other (user's choice) | Check batch_state::idle for both providers | ✓ |

**User's choice:** For both providers, just check `batch_state::idle`. Simple and direct.
**Notes:** None.

---

## Result Collector

| Option | Description | Selected |
|--------|-------------|----------|
| to_mutable() required (Recommended) | clone -> to_mutable() -> convert_to() -> to_idle(). Consistent with other paths. | |
| Direct convert_to if API allows | Skip mutable wrapper for exclusively-owned clones. | |
| You decide | Claude determines based on actual d9dc331 API. | |
| Other (user's choice) | Use read_only_data_batch::clone_to | ✓ |

**User's choice:** Use `read_only_data_batch::clone_to` function instead of clone-then-convert pattern.
**Notes:** Eliminates the two-step process entirely by cloning directly into the target representation.

---

## Subscribe Wiring

| Option | Description | Selected |
|--------|-------------|----------|
| In gpu_pipeline_task (Recommended) | subscribe() in constructor, unsubscribe() in destructor. Centralizes lifecycle. | ✓ |
| In task_creator | subscribe() at task assembly, unsubscribe() at task completion. | |
| In operator source methods | Each operator's create_source_tasks subscribes, matched by cleanup. | |

**User's choice:** In gpu_pipeline_task (Recommended)
**Notes:** Centralizes lifecycle management since all operators create tasks through this type.

---

## Claude's Discretion

- Internal state-save/restore cleanup (RAII replaces manual save/restore)
- Error logging adjustments
- bytes_in_space helper method handling (may defer to Phase 3)

## Deferred Ideas

None.
