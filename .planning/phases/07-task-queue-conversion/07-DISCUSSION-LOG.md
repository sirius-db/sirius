# Phase 7: Task Queue Conversion - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-15
**Phase:** 07-task-queue-conversion
**Areas discussed:** RAII ownership & queue return, Task data batch access, Convert scope per task

---

## RAII Ownership & Queue Return

### Queue return on interrupted queue

| Option | Description | Selected |
|--------|-------------|----------|
| Silent drop | If push() returns false, task is destroyed. Queue interruption = shutdown | |
| Log warning and drop | Log a warning before destroying. Gives visibility into shutdown behavior | ✓ |
| You decide | Claude chooses during implementation | |

**User's choice:** Log warning and drop
**Notes:** Adds spdlog dependency to header via logging.hpp

### Queue parameter type

| Option | Description | Selected |
|--------|-------------|----------|
| Raw pointer | inspectable_mpsc\<itask\>*. Matches Phase 6 pattern | |
| Reference | inspectable_mpsc\<itask\>&. Prevents null | ✓ |
| You decide | Claude chooses | |

**User's choice:** Reference
**Notes:** Departs from Phase 6's raw pointer convention but prevents null queue

### Move semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Move-only | Delete copy, enable move. Allows returning from factories | ✓ |
| Non-copyable, non-movable | Delete both. Forces unique_ptr\<convertible_data\> usage | |
| You decide | Claude chooses | |

**User's choice:** Move-only

---

## Task Data Batch Access

### Handling non-gpu_pipeline_task tasks

| Option | Description | Selected |
|--------|-------------|----------|
| Skip non-matching | dynamic_cast in predicate, return false on failure | ✓ |
| Assert gpu_pipeline_task | static_cast + assert, fail fast | |
| You decide | Claude chooses | |

**User's choice:** Skip non-matching (dynamic_cast chain)
**Notes:** User specified both pipelineable_operator_data and partitioned_operator_data should be handled. Since partitioned_operator_data extends pipelineable_operator_data, a single dynamic_cast suffices.

### Local state access

| Option | Description | Selected |
|--------|-------------|----------|
| Public accessor | Add get_local_state() method | ✓ |
| Cast and access directly | Use existing protected/public members | |
| You decide | Claude chooses | |

**User's choice:** Public accessor
**Notes:** Discovered `itask::local_state()` already exists as a public accessor — no new method needed.

---

## Convert Scope Per Task

### Which batches to convert

| Option | Description | Selected |
|--------|-------------|----------|
| Only matching batches | Skip batches not in target memory_space | ✓ |
| All batches in task | Convert every batch regardless | |

**User's choice:** Only matching batches

### Partial failure handling

| Option | Description | Selected |
|--------|-------------|----------|
| Roll back all | Atomic: all revert if any fails | |
| Keep successful, revert failed | Per-batch independence | |

**User's choice:** Nuanced — per-batch independence with specifics:
- Successful conversions: data stays in new tier, batch_state restored to pre-conversion value
- Failed conversions: data unchanged (conversion didn't happen), batch_state restored
- Each batch is independent — failure on one doesn't affect others

---

## Claude's Discretion

- Exact test case structure and Catch2 tag naming
- How to construct gpu_pipeline_task instances for testing
- Internal helper methods
- bytes_in_space() scope (all batches vs matching-state batches)
- convert() return value semantics on partial success

## Deferred Ideas

None — discussion stayed within phase scope.
