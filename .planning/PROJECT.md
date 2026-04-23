# Sirius data_batch API Refactoring

## What This Is

A refactoring of the Sirius GPU SQL engine to adopt cucascade's new 3-class data_batch API (commit d9dc331). The old API exposed data/memory/tier directly on data_batch with manual state machine transitions (idle, task_created, in_transit, processing). The new API makes data_batch an opaque idle handle, requiring RAII accessor types — `read_only_data_batch` (shared lock) or `mutable_data_batch` (exclusive lock) — to access or mutate data. This affects ~32 files and ~94 call sites across Sirius's pipeline, operator, and downgrade subsystems.

## Core Value

Sirius compiles cleanly against cucascade commit d9dc331 with the new 3-class data_batch API, preserving the existing execution semantics.

## Requirements

### Validated

- ✓ cucascade submodule updated to d9dc331 — existing on data_batch_refactor branch
- ✓ `lock_or_prepare_batch` rewritten to use `to_read_only()` / `readonly_to_mutable()` / `mutable_to_readonly()` transitions — Validated in Phase 1-2
- ✓ `pipelineable_operator_data::prepare_for_processing` returns `optional<vector<read_only_data_batch>>` — Validated in Phase 1
- ✓ New `read_only_pipelineable_operator_data` and `read_only_partitioned_operator_data` classes created — Validated in Phase 1
- ✓ `gpu_pipeline_task::compute_task` and `run_one_operator` receive `vector<read_only_data_batch>` input — Validated in Phase 1
- ✓ All operators cast to `read_only_pipelineable_operator_data` or `read_only_partitioned_operator_data` — Validated in Phase 3 (OPER-01)
- ✓ `convertible_data_batch::convert` and `convertible_gpu_pipeline_task::convert` use `to_mutable()` — Validated in Phase 1
- ✓ All idle `batch->get_data()` / `get_memory_space()` / `get_current_tier()` replaced with `to_read_only()` — Validated in Phase 3 (ACCS-01..04)
- ✓ All `pop_data_batch(batch_state::task_created)` replaced with `pop_idle_data_batch()` — Validated in Phase 3 (OPER-02)
- ✓ Repository signatures updated (state param removed) — Validated in Phase 3 (OPER-03, OPER-04)
- ✓ Subscriber count management via `subscribe()` / `unsubscribe()` — Validated in Phase 1
- ✓ `data_batch_processing_handle`, old `batch_state` values, old lock methods removed entirely — Validated in Phase 3
- ✓ `gpu_pipeline_task_local_state` estimation methods use `to_read_only()` — Validated in Phase 3
- ✓ `result_collector` convert_to uses `to_mutable()` pattern — Validated in Phase 1
- ✓ Project compiles cleanly with the new cucascade API — Validated in Phase 3 (BILD-01)

### Active

(None — all requirements validated)

### Out of Scope

- Test fixes beyond compilation — we target clean build first, tests will be fixed incrementally
- Performance optimization of the new locking patterns
- Refactoring operator logic beyond what's needed for API compatibility
- Changes to data_repository usage patterns beyond pop/get signature updates

## Context

- Branch: `data_batch_refactor` (already exists with cucascade submodule updated)
- cucascade commit chain: `66cc4b8 → 63f834c → 604a1fb → d9dc331` (4 breaking changes)
- Key API removals: `data_batch_processing_handle`, `lock_for_processing`, `try_to_lock_for_in_transit`, `try_to_release_in_transit`, `batch_state::task_created`, `batch_state::in_transit`, `batch_state::processing`
- Key API additions: `read_only_data_batch`, `mutable_data_batch`, `to_read_only()`, `to_mutable()`, `try_to_read_only()`, `try_to_mutable()`, `readonly_to_mutable()`, `mutable_to_readonly()`, `to_idle()`, `subscribe()`, `unsubscribe()`
- `data_batch::get_data()`, `get_memory_space()`, `get_current_tier()` are now private — only accessible through accessor types
- `data_repository::get_data_batch_by_id` lost its optional `batch_state` parameter
- `data_repository::pop_data_batch(state)` replaced by `pop_idle_data_batch()`, `pop_read_only_data_batch()`, `pop_mutable_data_batch()`

## Constraints

- **API compatibility**: Must use cucascade commit d9dc331 exactly — no modifications to cucascade
- **Semantic preservation**: Existing execution flow (pipeline → operator → data flow) stays the same, just using new accessor types
- **Brownfield**: This is a targeted refactoring within an active codebase — minimize changes outside the data_batch API boundary
- **Blocking pattern**: Use blocking `to_mutable()` in downgrade/convert paths (not try-based)
- **Non-blocking reads**: Use `to_read_only()` for all read-only data access on idle batches

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use `to_read_only()` for all data access on idle batches | New API makes get_data/get_memory_space private | ✓ Adopted across all operators and pipeline code |
| Use blocking `to_mutable()` for convert paths | Simplifies conversion flow vs try-based approach | ✓ Used in convertible_data_batch, convertible_gpu_pipeline_task, downgrade |
| Replace `task_created` state with subscriber count | New API removed task_created; subscribe/unsubscribe is the replacement mechanism | ✓ Implemented in pipeline task lifecycle |
| Replace `pop_data_batch(task_created)` with `pop_idle_data_batch()` | task_created state gone; subscriber count distinguishes assigned vs free | ✓ All 9 operator call sites migrated |
| Target compilation only, not test correctness | Allows incremental progress on a large refactoring | ✓ Clean build achieved; test compilation also fixed |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
## Current State

Phase 3 complete — all 3 phases of the data_batch API refactoring milestone are done. Sirius compiles cleanly against cucascade d9dc331 with the new 3-class API. All operator, pipeline, and accessor migration requirements validated.

*Last updated: 2026-04-23 after Phase 3 completion*
