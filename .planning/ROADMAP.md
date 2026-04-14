# Roadmap: inspectable_mpsc

## Milestones

- ✅ **v1.0 MVP** — Phases 1-2 (shipped 2026-04-14)
- 🚧 **v1.1 Task Queue Refactor** — Phases 3-4 (in progress)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-2) — SHIPPED 2026-04-14</summary>

- [x] Phase 1: Core Queue (2/2 plans) — completed 2026-04-14
- [x] Phase 2: Predicate Inspection (1/1 plan) — completed 2026-04-14

</details>

### 🚧 v1.1 Task Queue Refactor (In Progress)

**Milestone Goal:** Replace legacy queue infrastructure with inspectable_mpsc and remove dead queue code.

- [ ] **Phase 3: Dead Code Removal** - Verify and remove unused legacy queue classes from the codebase
- [ ] **Phase 4: Queue Integration** - Replace interruptible_mpmc with inspectable_mpsc in itask_executor

## Phase Details

### Phase 3: Dead Code Removal
**Goal**: Legacy unused queue classes are verified unused and removed from the codebase
**Depends on**: Phase 2 (v1.0 shipped inspectable_mpsc)
**Requirements**: CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04
**Success Criteria** (what must be TRUE):
  1. The four legacy queue classes (gpu_pipeline_queue, pipeline_queue, duckdb_scan_task_queue, itask_queue) no longer exist in the source tree
  2. No references to any of the removed classes remain anywhere in the codebase (headers, source, CMake)
  3. The project builds successfully and all existing tests pass after removal
**Plans:** 1 plan
Plans:
- [x] 03-01-PLAN.md — Delete legacy queue files, clean references, build and test

### Phase 4: Queue Integration
**Goal**: itask_executor and all its implementations use inspectable_mpsc instead of interruptible_mpmc
**Depends on**: Phase 3
**Requirements**: INTG-01, INTG-02
**Success Criteria** (what must be TRUE):
  1. The itask_executor interface declares inspectable_mpsc (not interruptible_mpmc) for its queue type
  2. All classes implementing itask_executor compile and link with inspectable_mpsc queues
  3. No references to interruptible_mpmc remain in itask_executor or any of its implementations
  4. The project builds successfully and all existing tests pass after the queue replacement
**Plans:** 1 plan
Plans:
- [x] 04-01-PLAN.md — Replace interruptible_mpmc with inspectable_mpsc in itask_executor, build and test

## Progress

**Execution Order:** Phases execute in numeric order: 3 -> 4

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Core Queue | v1.0 | 2/2 | Complete | 2026-04-14 |
| 2. Predicate Inspection | v1.0 | 1/1 | Complete | 2026-04-14 |
| 3. Dead Code Removal | v1.1 | 0/1 | Planned | - |
| 4. Queue Integration | v1.1 | 0/1 | Planned | - |

---
*Full v1.0 details archived to `.planning/milestones/v1.0-ROADMAP.md`*
