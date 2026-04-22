# Roadmap: Sirius data_batch API Refactoring

## Overview

This refactoring migrates Sirius from the old cucascade data_batch API (with manual state machine transitions and `data_batch_processing_handle`) to the new 3-class API (opaque idle handle + RAII accessor types). Work proceeds in three coarse phases: first the pipeline data path is rerouted to flow `read_only_data_batch` end-to-end; then mutation paths (downgrade/convert) and lifecycle management (subscribe/unsubscribe) are updated; finally all operators and accessor call sites are swept, culminating in a clean build.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Pipeline Data Path** - Reroute the pipeline's core data path to use `read_only_data_batch` end-to-end and introduce the two new RAII wrapper types
- [ ] **Phase 2: Mutation Paths and Lifecycle** - Update all `to_mutable()` conversion paths and remove the old batch_state lifecycle machinery
- [ ] **Phase 3: Operator Sweep and Clean Build** - Migrate all operator call sites and accessor usages, then achieve a clean compilation against cucascade d9dc331

## Phase Details

### Phase 1: Pipeline Data Path
**Goal**: The pipeline core can pass `read_only_data_batch` through `lock_or_prepare_batch` → `prepare_for_processing` → `compute_task` → `run_one_operator`, with the old `data_batch_processing_handle` type fully removed
**Depends on**: Nothing (first phase)
**Requirements**: PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05, TYPE-01, TYPE-02
**Success Criteria** (what must be TRUE):
  1. `lock_or_prepare_batch` compiles returning `read_only_data_batch` with no reference to `data_batch_processing_handle`
  2. `pipelineable_operator_data::prepare_for_processing` returns `optional<vector<read_only_data_batch>>`
  3. `gpu_pipeline_task::compute_task` receives its input batches from `prepare_for_processing` as `vector<read_only_data_batch>`
  4. `run_one_operator` signature accepts `vector<read_only_data_batch>`
  5. `read_only_pipelineable_operator_data` and `read_only_partitioned_operator_data` types exist and are compilable
**Plans:** 2 plans

Plans:
- [x] 01-01-PLAN.md — Define new RAII types and rewrite lock_or_prepare_batch
- [ ] 01-02-PLAN.md — Wire read_only_pipelineable_operator_data through pipeline execution path

### Phase 2: Mutation Paths and Lifecycle
**Goal**: All conversion and downgrade code uses `to_mutable()` for exclusive access, and the old batch_state machine (`task_created`, `in_transit`, `processing`) and its associated lock functions are fully removed
**Depends on**: Phase 1
**Requirements**: CONV-01, CONV-02, CONV-03, LIFE-01, LIFE-02, LIFE-03, LIFE-04
**Success Criteria** (what must be TRUE):
  1. `convertible_data_batch::convert` and `convertible_gpu_pipeline_task::convert` both acquire a `mutable_data_batch` via `to_mutable()` before calling `convert_to`
  2. `result_collector` convert_to calls use the `to_mutable()` pattern
  3. `subscribe()` is called at task creation and `unsubscribe()` is called in the task destructor for all input batches
  4. No references to `batch_state::task_created`, `batch_state::in_transit`, or `batch_state::processing` remain in conversion and lifecycle files (`convertible_data.hpp`, `convertible_data_batch.hpp`, `convertible_gpu_pipeline_task.hpp`, `gpu_pipeline_task.hpp`, `gpu_pipeline_task.cpp`, `downgrade_executor.cpp`, `sirius_physical_result_collector.cpp`). Operator-level references (`pop_data_batch(batch_state::task_created)` in `src/op/` files) are deferred to Phase 3 (OPER-02).
  5. `try_to_lock_for_in_transit`, `try_to_release_in_transit`, and `wait_to_lock_for_processing` calls are all removed
**Plans:** 3 plans

Plans:
- [x] 02-01-PLAN.md — Rewrite conversion/downgrade path to use to_mutable() RAII and remove old state machine
- [x] 02-02-PLAN.md — Result collector clone_to pattern and subscribe/unsubscribe lifecycle wiring
- [ ] 02-03-PLAN.md — Gap closure: scope SC4 and fix [[nodiscard]] to_idle() discards

### Phase 3: Operator Sweep and Clean Build
**Goal**: Every operator casts to the correct new type, every legacy accessor call site on idle batches uses `to_read_only()`, and the project compiles cleanly against cucascade d9dc331
**Depends on**: Phase 2
**Requirements**: OPER-01, OPER-02, OPER-03, OPER-04, ACCS-01, ACCS-02, ACCS-03, ACCS-04, BILD-01
**Success Criteria** (what must be TRUE):
  1. All operators that previously used `data_batch_processing_handle` now cast to `read_only_pipelineable_operator_data` or `read_only_partitioned_operator_data` as appropriate
  2. All `pop_data_batch(batch_state::task_created)` calls are replaced with `pop_idle_data_batch()`
  3. All `get_data_batch_by_id` and `pop_data_batch_by_id` calls use the updated signatures without the state parameter
  4. All `batch->get_data()`, `batch->get_memory_space()`, and `batch->get_current_tier()` calls on idle batches — including estimation methods in `gpu_pipeline_task_local_state` — go through a `to_read_only()` accessor
  5. `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make` completes with zero errors against cucascade d9dc331
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Pipeline Data Path | 0/2 | Planning complete | - |
| 2. Mutation Paths and Lifecycle | 2/3 | Gap closure planned | - |
| 3. Operator Sweep and Clean Build | 0/TBD | Not started | - |
