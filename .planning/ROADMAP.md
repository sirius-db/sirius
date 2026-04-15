# Roadmap: inspectable_mpsc & Convertible Data

## Milestones

- ✅ **v1.0 MVP** — Phases 1-2 (shipped 2026-04-14)
- ✅ **v1.1 Task Queue Refactor** — Phases 3-4 (shipped 2026-04-14)
- 🚧 **v2.0 Convertible Data Abstraction** — Phases 5-7 (in progress)

## Phases

<details>
<summary>v1.0 MVP (Phases 1-2) — SHIPPED 2026-04-14</summary>

- [x] Phase 1: Core Queue (2/2 plans) — completed 2026-04-14
- [x] Phase 2: Predicate Inspection (1/1 plan) — completed 2026-04-14

</details>

<details>
<summary>v1.1 Task Queue Refactor (Phases 3-4) — SHIPPED 2026-04-14</summary>

- [x] Phase 3: Dead Code Removal (1/1 plan) — completed 2026-04-14
- [x] Phase 4: Queue Integration (1/1 plan) — completed 2026-04-14

</details>

### v2.0 Convertible Data Abstraction

- [ ] **Phase 5: State Machine & Interfaces** - Extend data_batch state transitions and define abstract conversion contracts
- [ ] **Phase 6: Batch Conversion** - Concrete convertible_data implementation wrapping data_batch and data_repository
- [ ] **Phase 7: Task Queue Conversion** - Concrete convertible_data implementation wrapping gpu_pipeline_task and inspectable_mpsc

## Phase Details

### Phase 5: State Machine & Interfaces
**Goal**: data_batch supports task_created-to-in_transit transitions and abstract conversion contracts are defined
**Depends on**: Phase 4 (inspectable_mpsc is the production task queue)
**Requirements**: STATE-01, STATE-02, IFACE-01, IFACE-02
**Success Criteria** (what must be TRUE):
  1. `data_batch::try_to_lock_for_in_transit()` succeeds when batch is in `task_created` state, not only `idle`
  2. `try_to_release_in_transit(prev_state)` can restore a batch to `task_created` state
  3. `convertible_data` declares pure virtual `convert()` and `bytes_in_space()` that compile and can be subclassed
  4. `convertible_data_provider` declares pure virtual `get_next_convertible()`, `get_all_convertible()`, and `get_bytes_in_space()` that compile and can be subclassed
**Plans:** 2 plans
Plans:
- [x] 05-01-PLAN.md — Formalize state machine transitions and add round-trip tests
- [x] 05-02-PLAN.md — Define convertible_data and convertible_data_provider abstract interfaces

### Phase 6: Batch Conversion
**Goal**: Data batches in a repository can be discovered by memory space and converted with failure safety
**Depends on**: Phase 5
**Requirements**: BATCH-01, BATCH-02, BATCH-03
**Success Criteria** (what must be TRUE):
  1. `convertible_data_batch::convert()` locks the batch for in_transit, requests a HOST reservation, converts via the converter registry, and restores prev_state on completion
  2. `convertible_data_batch_provider` iterates a `shared_data_repository` partitions last-to-first, batches last-to-first, filtering by `idle` state and matching `memory_space`, returning wrapped `convertible_data_batch` instances
  3. On conversion failure or exception, the batch retains its original `idata_representation` and `batch_state` is restored via `try_to_release_in_transit(prev_state)` — never left in `in_transit`
  4. `bytes_in_space()` returns the correct byte size for a wrapped data_batch in the given memory space
**Plans:** 2 plans
Plans:
- [x] 06-01-PLAN.md — Implement convertible_data_batch and convertible_data_batch_provider
- [x] 06-02-PLAN.md — GPU integration tests for batch conversion and provider discovery

### Phase 7: Task Queue Conversion
**Goal**: Queued pipeline tasks can be discovered by memory space, temporarily owned for conversion, and safely returned to the queue
**Depends on**: Phase 5
**Requirements**: TASK-01, TASK-02, TASK-03
**Success Criteria** (what must be TRUE):
  1. `convertible_gpu_pipeline_task` takes ownership of a `unique_ptr<itask>` via constructor and its destructor pushes the task back to the `inspectable_mpsc<itask>` queue
  2. `convertible_gpu_pipeline_task_provider::get_next_convertible()` uses `mutable_pop_if` with `front_to_back=false`, matching tasks whose `gpu_pipeline_task_local_state` data_batches are in the target `memory_space` and `batch_state::task_created`
  3. On conversion failure or exception, all `data_batch` objects inside `operator_data` retain their original `idata_representation` and `batch_state`; the task is always returned to the queue via RAII destructor
  4. `bytes_in_space()` returns the total byte size across all data_batches in the task's operator_data for the given memory space
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 5 -> 6 -> 7

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Core Queue | v1.0 | 2/2 | Complete | 2026-04-14 |
| 2. Predicate Inspection | v1.0 | 1/1 | Complete | 2026-04-14 |
| 3. Dead Code Removal | v1.1 | 1/1 | Complete | 2026-04-14 |
| 4. Queue Integration | v1.1 | 1/1 | Complete | 2026-04-14 |
| 5. State Machine & Interfaces | v2.0 | 0/2 | Planning complete | - |
| 6. Batch Conversion | v2.0 | 0/2 | Planning complete | - |
| 7. Task Queue Conversion | v2.0 | 0/? | Not started | - |

---
*Full v1.0 details archived to `.planning/milestones/v1.0-ROADMAP.md`*
*Full v1.1 details archived to `.planning/milestones/v1.1-ROADMAP.md`*
