# Sirius data_batch API Refactoring

## What This Is

A refactoring of the Sirius GPU SQL engine to adopt cucascade's new 3-class data_batch API (commit d9dc331). The old API exposed data/memory/tier directly on data_batch with manual state machine transitions (idle, task_created, in_transit, processing). The new API makes data_batch an opaque idle handle, requiring RAII accessor types — `read_only_data_batch` (shared lock) or `mutable_data_batch` (exclusive lock) — to access or mutate data. This affects ~32 files and ~94 call sites across Sirius's pipeline, operator, and downgrade subsystems.

## Core Value

Sirius compiles cleanly against cucascade commit d9dc331 with the new 3-class data_batch API, preserving the existing execution semantics.

## Requirements

### Validated

- ✓ cucascade submodule updated to d9dc331 — existing on data_batch_refactor branch

### Active

- [ ] `lock_or_prepare_batch` rewritten to use `to_read_only()` / `readonly_to_mutable()` / `mutable_to_readonly()` transitions, returning `read_only_data_batch` instead of `data_batch_processing_handle`
- [ ] `pipelineable_operator_data::prepare_for_processing` returns `optional<vector<read_only_data_batch>>` instead of `optional<vector<data_batch_processing_handle>>`
- [ ] New `read_only_pipelineable_operator_data` class holding `vector<read_only_data_batch>` created
- [ ] New `read_only_partitioned_operator_data` class extending `read_only_pipelineable_operator_data` with partition index
- [ ] `gpu_pipeline_task::compute_task` receives `vector<read_only_data_batch>` input (passed from prepare_for_processing, not from local state)
- [ ] `run_one_operator` takes `vector<read_only_data_batch>` input
- [ ] All operators internally cast to `read_only_pipelineable_operator_data` or `read_only_partitioned_operator_data` as appropriate
- [ ] `convertible_data_batch::convert` uses `to_mutable()` to acquire exclusive lock, then calls `convert_to` on `mutable_data_batch`
- [ ] `convertible_gpu_pipeline_task::convert` follows same `to_mutable()` pattern
- [ ] All `batch->get_data()` / `batch->get_memory_space()` / `batch->get_current_tier()` calls on idle data_batch replaced with `to_read_only()` accessor pattern
- [ ] All `pop_data_batch(batch_state::task_created)` calls replaced with `pop_idle_data_batch()`
- [ ] All `get_data_batch_by_id(id, std::nullopt, partition)` calls updated to `get_data_batch_by_id(id, partition)` (optional state param removed)
- [ ] `pop_data_batch_by_id(id, batch_state::task_created, partition)` calls updated to `pop_data_batch_by_id(id, partition)` (state param removed)
- [ ] Subscriber count management: `subscribe()` called at task creation, `unsubscribe()` in task destructor for all input data_batches
- [ ] `data_batch_processing_handle` references removed entirely (type is obsolete)
- [ ] Old `batch_state::task_created` / `batch_state::in_transit` references removed
- [ ] Old `try_to_lock_for_in_transit` / `try_to_release_in_transit` / `wait_to_lock_for_processing` / `try_to_lock_for_processing` calls removed
- [ ] `gpu_pipeline_task_local_state` methods (`get_task_consumption_basis`, `get_estimated_bytes_to_materialize_input`) use `to_read_only()` for data access
- [ ] `result_collector` convert_to calls use `to_mutable()` pattern
- [ ] Project compiles cleanly with the new cucascade API

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
| Use `to_read_only()` for all data access on idle batches | New API makes get_data/get_memory_space private | — Pending |
| Use blocking `to_mutable()` for convert paths | Simplifies conversion flow vs try-based approach | — Pending |
| Replace `task_created` state with subscriber count | New API removed task_created; subscribe/unsubscribe is the replacement mechanism | — Pending |
| Replace `pop_data_batch(task_created)` with `pop_idle_data_batch()` | task_created state gone; subscriber count distinguishes assigned vs free | — Pending |
| Target compilation only, not test correctness | Allows incremental progress on a large refactoring | — Pending |

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
*Last updated: 2026-04-21 after initialization*
