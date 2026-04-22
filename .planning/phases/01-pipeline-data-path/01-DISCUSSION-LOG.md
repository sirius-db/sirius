# Phase 1: Pipeline Data Path - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-21
**Phase:** 01-pipeline-data-path
**Areas discussed:** Conversion flow, Type hierarchy, Data flow through compute_task, run_one_operator signature

---

## Conversion flow in `lock_or_prepare_batch`

| Option | Description | Selected |
|--------|-------------|----------|
| `to_read_only()` first, then `readonly_to_mutable()` if mismatch | Acquire read lock first, upgrade only if conversion needed, downgrade back after. No idle transitions. | ✓ |
| `to_mutable()` first for all cases | Always acquire exclusive lock, convert if needed, then `mutable_to_readonly()`. Simpler but blocks other readers unnecessarily. | |
| Separate lock cycles through idle | `to_mutable()` → convert → `to_idle()` → `to_read_only()`. Two separate lock acquisitions with gap. | |

**User's choice:** Acquire `read_only_data_batch` first via `to_read_only()`, check memory space. If mismatch, `readonly_to_mutable()` → `convert_to` → `mutable_to_readonly()`. No idle transitions.

---

## New type hierarchy

| Option | Description | Selected |
|--------|-------------|----------|
| Extend `operator_data` directly | `read_only_pipelineable_operator_data` → `operator_data`, sibling to existing types. `read_only_partitioned_operator_data` → `read_only_pipelineable_operator_data`. | ✓ |
| Extend `pipelineable_operator_data` | New types inherit from existing types, adding read-only semantics. | |
| Standalone types (no inheritance) | Completely separate class hierarchy. | |

**User's choice:** Extend `operator_data` directly. `read_only_pipelineable_operator_data` inherits from `operator_data`. `read_only_partitioned_operator_data` inherits from `read_only_pipelineable_operator_data`.

---

## Data flow through `compute_task`

| Option | Description | Selected |
|--------|-------------|----------|
| `prepare_for_processing` returns `read_only_pipelineable_operator_data` | Direct return of the new type, encapsulating the lock acquisition. | ✓ |
| `prepare_for_processing` returns raw `vector<read_only_data_batch>` | Caller constructs the wrapper type. More flexible but less encapsulated. | |
| `compute_task` constructs wrapper from raw vector | Intermediate step: prepare returns vector, compute_task wraps it. | |

**User's choice:** `prepare_for_processing` returns `optional<read_only_pipelineable_operator_data>` directly.

---

## `run_one_operator` signature

| Option | Description | Selected |
|--------|-------------|----------|
| Takes `read_only_pipelineable_operator_data` | Specific type, preserves polymorphism within the read-only hierarchy. | ✓ |
| Takes `vector<read_only_data_batch>` | Raw vector, breaks operator_data abstraction. | |
| Takes `const operator_data&` (unchanged) | Keep generic, operators cast internally. Less type safety at the call site. | |

**User's choice:** `run_one_operator` takes `read_only_pipelineable_operator_data`.

---

## Claude's Discretion

- Internal helper functions and logging adjustments
- Error message wording for lock failures
- Whether to simplify the retry loop given new blocking API

## Deferred Ideas

None.
