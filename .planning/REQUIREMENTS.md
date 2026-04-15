# Requirements: inspectable_mpsc & Convertible Data

**Defined:** 2026-04-15
**Core Value:** Uniform, failure-safe data conversion across memory tiers

## v2.0 Requirements

Requirements for Convertible Data Abstraction milestone. Each maps to roadmap phases.

### Abstract Interfaces

- [ ] **IFACE-01**: `convertible_data` abstract interface with `bool convert(const std::vector<memory_space*>&, rmm::cuda_stream_view, sirius_memory_reservation_manager&)` and `size_t bytes_in_space(memory_space*)`
- [ ] **IFACE-02**: `convertible_data_provider` abstract interface with `std::unique_ptr<convertible_data> get_next_convertible(memory_space*, bool)`, `std::vector<std::unique_ptr<convertible_data>> get_all_convertible(memory_space*, bool)`, and `size_t get_bytes_in_space(memory_space*)`

### Batch Conversion

- [ ] **BATCH-01**: `convertible_data_batch` wraps `shared_ptr<data_batch>`, `convert()` emulates `downgrade_task::execute` — locks for in_transit, requests HOST reservation, converts via converter registry singleton, restores prev_state
- [ ] **BATCH-02**: `convertible_data_batch_provider` wraps `shared_data_repository*`, iterates partitions last-to-first and within each partition iterates data_batches last-to-first, filters by `idle` state and matching `memory_space`, returns batches wrapped as `convertible_data_batch`
- [ ] **BATCH-03**: On conversion failure or exception in `convertible_data_batch::convert()`, batch retains original `idata_representation` and `batch_state` is restored via `try_to_release_in_transit(prev_state)` — never left in `in_transit`

### Task Queue Conversion

- [ ] **TASK-01**: `convertible_gpu_pipeline_task` wraps `unique_ptr<itask>` with RAII ownership — constructor takes `(unique_ptr<itask>, inspectable_mpsc<itask>*)`, destructor pushes task back to queue
- [ ] **TASK-02**: `convertible_gpu_pipeline_task_provider` wraps `inspectable_mpsc<itask>*`, `get_next_convertible` uses `mutable_pop_if` with `front_to_back=false`, predicate inspects `gpu_pipeline_task_local_state` data_batches for matching memory_space and `batch_state::task_created`
- [ ] **TASK-03**: On conversion failure or exception in `convertible_gpu_pipeline_task::convert()`, all `data_batch` objects inside `operator_data` retain original `idata_representation` and `batch_state`; task is always returned to queue via destructor

### State Machine Extension

- [ ] **STATE-01**: Extend `data_batch::try_to_lock_for_in_transit()` to allow transition from `task_created` state (currently only `idle` is permitted)
- [ ] **STATE-02**: `try_to_release_in_transit()` can restore to `task_created` via the existing `prev_state` optional parameter

## Future Requirements

None currently deferred.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Lock-free conversion | Mutex+cv is appropriate for inspection/iteration requirements |
| Async conversion pipeline | Adds complexity beyond what's needed; synchronous convert() is sufficient |
| Converting data in `processing` state | Batches being actively processed should not be interrupted for conversion |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| IFACE-01 | — | Pending |
| IFACE-02 | — | Pending |
| BATCH-01 | — | Pending |
| BATCH-02 | — | Pending |
| BATCH-03 | — | Pending |
| TASK-01 | — | Pending |
| TASK-02 | — | Pending |
| TASK-03 | — | Pending |
| STATE-01 | — | Pending |
| STATE-02 | — | Pending |

**Coverage:**
- v2.0 requirements: 10 total
- Mapped to phases: 0
- Unmapped: 10 ⚠️

---
*Requirements defined: 2026-04-15*
*Last updated: 2026-04-15 after initial definition*
