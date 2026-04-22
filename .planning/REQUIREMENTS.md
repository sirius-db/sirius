# Requirements: Sirius data_batch API Refactoring

**Defined:** 2026-04-21
**Core Value:** Sirius compiles cleanly against cucascade commit d9dc331 with the new 3-class data_batch API

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Pipeline Core

- [ ] **PIPE-01**: `lock_or_prepare_batch` returns `read_only_data_batch` instead of `data_batch_processing_handle`
- [ ] **PIPE-02**: `pipelineable_operator_data::prepare_for_processing` returns `optional<vector<read_only_data_batch>>`
- [ ] **PIPE-03**: `gpu_pipeline_task::compute_task` receives `vector<read_only_data_batch>` from prepare_for_processing
- [ ] **PIPE-04**: `run_one_operator` takes `vector<read_only_data_batch>` input
- [ ] **PIPE-05**: `data_batch_processing_handle` references removed entirely

### New Types

- [ ] **TYPE-01**: `read_only_pipelineable_operator_data` class created holding `vector<read_only_data_batch>`
- [ ] **TYPE-02**: `read_only_partitioned_operator_data` class created with partition index

### Operators

- [ ] **OPER-01**: All operators internally cast to `read_only_pipelineable_operator_data` or `read_only_partitioned_operator_data`
- [ ] **OPER-02**: All `pop_data_batch(batch_state::task_created)` calls replaced with `pop_idle_data_batch()`
- [ ] **OPER-03**: All `get_data_batch_by_id(id, std::nullopt, partition)` calls updated to `get_data_batch_by_id(id, partition)`
- [ ] **OPER-04**: All `pop_data_batch_by_id(id, state, partition)` calls updated to `pop_data_batch_by_id(id, partition)`

### Accessor Migration

- [ ] **ACCS-01**: All `batch->get_data()` calls on idle data_batch use `to_read_only()` accessor
- [ ] **ACCS-02**: All `batch->get_memory_space()` calls on idle data_batch use `to_read_only()` accessor
- [ ] **ACCS-03**: All `batch->get_current_tier()` calls on idle data_batch use `to_read_only()` accessor
- [ ] **ACCS-04**: `gpu_pipeline_task_local_state` estimation methods use `to_read_only()` for data access

### Conversion/Downgrade

- [ ] **CONV-01**: `convertible_data_batch::convert` uses `to_mutable()` then `convert_to` on `mutable_data_batch`
- [ ] **CONV-02**: `convertible_gpu_pipeline_task::convert` uses same `to_mutable()` pattern
- [ ] **CONV-03**: `result_collector` convert_to calls use `to_mutable()` pattern

### Lifecycle

- [ ] **LIFE-01**: `subscribe()` called at task creation for all input data_batches
- [ ] **LIFE-02**: `unsubscribe()` called in task destructor for all input data_batches
- [ ] **LIFE-03**: Old `batch_state::task_created` / `batch_state::in_transit` / `batch_state::processing` references removed
- [ ] **LIFE-04**: Old `try_to_lock_for_in_transit` / `try_to_release_in_transit` / `wait_to_lock_for_processing` calls removed

### Build

- [ ] **BILD-01**: Project compiles cleanly against cucascade d9dc331

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Test Correctness

- **TEST-01**: All existing unit tests pass with new API
- **TEST-02**: All TPC-H SQL logic tests pass with new API

### Performance

- **PERF-01**: No measurable regression in TPC-H query times from locking overhead

## Out of Scope

| Feature | Reason |
|---------|--------|
| Test fixes beyond compilation | Incremental approach — compilation first |
| Performance optimization of locking patterns | Separate concern after correctness |
| Operator logic refactoring beyond API compat | Minimize blast radius of this change |
| cucascade modifications | Must use d9dc331 as-is |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PIPE-01 | Phase 1 | Pending |
| PIPE-02 | Phase 1 | Pending |
| PIPE-03 | Phase 1 | Pending |
| PIPE-04 | Phase 1 | Pending |
| PIPE-05 | Phase 1 | Pending |
| TYPE-01 | Phase 1 | Pending |
| TYPE-02 | Phase 1 | Pending |
| CONV-01 | Phase 2 | Pending |
| CONV-02 | Phase 2 | Pending |
| CONV-03 | Phase 2 | Pending |
| LIFE-01 | Phase 2 | Pending |
| LIFE-02 | Phase 2 | Pending |
| LIFE-03 | Phase 2 | Pending |
| LIFE-04 | Phase 2 | Pending |
| OPER-01 | Phase 3 | Pending |
| OPER-02 | Phase 3 | Pending |
| OPER-03 | Phase 3 | Pending |
| OPER-04 | Phase 3 | Pending |
| ACCS-01 | Phase 3 | Pending |
| ACCS-02 | Phase 3 | Pending |
| ACCS-03 | Phase 3 | Pending |
| ACCS-04 | Phase 3 | Pending |
| BILD-01 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-21*
*Last updated: 2026-04-21 after initial definition*
