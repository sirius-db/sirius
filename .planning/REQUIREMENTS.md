# Requirements: inspectable_mpsc & Convertible Data

**Defined:** 2026-04-16
**Core Value:** Thread-safe queue with predicate-based inspection; uniform, failure-safe data conversion across memory tiers

## v3.0 Requirements

Requirements for Downgrade Executor Integration. Each maps to roadmap phases.

### Processing Loop

- [ ] **LOOP-01**: Processing loop creates a `convertible_data_batch_provider` per `data_repository` and fetches downgrade candidates lazily via `get_all_convertible`
- [ ] **LOOP-02**: Processing loop falls back to `gpu_pipeline_executor` task queue via `convertible_gpu_pipeline_task_provider` when data_repositories are exhausted
- [ ] **LOOP-03**: Processing loop falls back to `pipeline_executor` task queue via `convertible_gpu_pipeline_task_provider` when gpu_pipeline_executor queue is exhausted
- [ ] **LOOP-04**: Each downgrade candidate is converted via `convertible_data::convert()` — `downgrade_task` struct eliminated if trivial
- [ ] **LOOP-05**: Processing loop stops when existing memory pressure predicate is satisfied

### API Cleanup

- [ ] **DAPI-01**: `target_bytes` parameter removed from `downgrade_executor::request_downgrade` and `target_bytes` member removed from `downgrade_request`
- [ ] **DAPI-02**: `target_bytes` calculation logic removed from `gpu_pipeline_executor`

### Observability

- [ ] **LOG-01**: Trace logging reports downgrade counts per source tier (data_repositories, gpu_pipeline_executor queue, pipeline_executor queue)

### Batch Lock Exploration

- [ ] **LOCK-01**: Functional diff analysis of `lock_or_prepare_batch` vs `convertible_data_batch::convert()` completed during discussion phase
- [ ] **LOCK-02**: Conditional refactor of `lock_or_prepare_batch` to use `convertible_data_batch::convert()` (go/no-go based on LOCK-01 analysis)

## v2.0 Requirements (Shipped)

### Abstract Interfaces

- [x] **IFACE-01**: `convertible_data` abstract interface — Shipped Phase 5
- [x] **IFACE-02**: `convertible_data_provider` abstract interface — Shipped Phase 5

### Batch Conversion

- [x] **BATCH-01**: `convertible_data_batch` wrapping `data_batch` — Shipped Phase 6
- [x] **BATCH-02**: `convertible_data_batch_provider` wrapping `data_repository` — Shipped Phase 6
- [x] **BATCH-03**: Failure safety for batch conversion — Shipped Phase 6

### Task Queue Conversion

- [x] **TASK-01**: `convertible_gpu_pipeline_task` with RAII ownership — Shipped Phase 7
- [x] **TASK-02**: `convertible_gpu_pipeline_task_provider` with `mutable_pop_if` — Shipped Phase 7
- [x] **TASK-03**: Failure safety for task conversion — Shipped Phase 7

### State Machine Extension

- [x] **STATE-01**: `task_created → in_transit` transition — Shipped Phase 5
- [x] **STATE-02**: `try_to_release_in_transit()` restore to `task_created` — Shipped Phase 5

## Future Requirements

None currently deferred.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Lock-free downgrade queue | Mutex+cv is appropriate for the downgrade executor's workload |
| Refactoring interruptible_mpmc in pipeline_executor/task_creator | Out of scope — only the downgrade path is being refactored |
| Changing the memory pressure predicate logic | Existing predicate stays; only the candidate fetching changes |
| Lock-free conversion | Mutex+cv is appropriate for inspection/iteration requirements |
| Async conversion pipeline | Adds complexity beyond what's needed; synchronous convert() is sufficient |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DAPI-01 | Phase 8 | Pending |
| DAPI-02 | Phase 8 | Pending |
| LOOP-01 | Phase 9 | Pending |
| LOOP-02 | Phase 9 | Pending |
| LOOP-03 | Phase 9 | Pending |
| LOOP-04 | Phase 9 | Pending |
| LOOP-05 | Phase 9 | Pending |
| LOG-01 | Phase 9 | Pending |
| LOCK-01 | Phase 10 | Pending |
| LOCK-02 | Phase 10 | Pending |

**Coverage:**
- v3.0 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0

---
*Requirements defined: 2026-04-16*
*Last updated: 2026-04-16 after v3.0 roadmap creation*
