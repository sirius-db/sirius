# Roadmap: inspectable_mpsc & Convertible Data

## Milestones

- ✅ **v1.0 MVP** — Phases 1-2 (shipped 2026-04-14)
- ✅ **v1.1 Task Queue Refactor** — Phases 3-4 (shipped 2026-04-14)
- ✅ **v2.0 Convertible Data Abstraction** — Phases 5-7 (shipped 2026-04-16)
- 🚧 **v3.0 Downgrade Executor Integration** — Phases 8-10 (in progress)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-2) — SHIPPED 2026-04-14</summary>

- [x] Phase 1: Core Queue (2/2 plans) — completed 2026-04-14
- [x] Phase 2: Predicate Inspection (1/1 plan) — completed 2026-04-14

</details>

<details>
<summary>✅ v1.1 Task Queue Refactor (Phases 3-4) — SHIPPED 2026-04-14</summary>

- [x] Phase 3: Dead Code Removal (1/1 plan) — completed 2026-04-14
- [x] Phase 4: Queue Integration (1/1 plan) — completed 2026-04-14

</details>

<details>
<summary>✅ v2.0 Convertible Data Abstraction (Phases 5-7) — SHIPPED 2026-04-16</summary>

- [x] Phase 5: State Machine & Interfaces (2/2 plans) — completed 2026-04-15
- [x] Phase 6: Batch Conversion (2/2 plans) — completed 2026-04-15
- [x] Phase 7: Task Queue Conversion (2/2 plans) — completed 2026-04-16

</details>

### 🚧 v3.0 Downgrade Executor Integration (In Progress)

**Milestone Goal:** Refactor the downgrade executor to use convertible_data abstractions with lazy, tiered candidate fetching and simplified API.

- [ ] **Phase 8: API Cleanup** - Remove target_bytes from downgrade request path and gpu_pipeline_executor calculation
- [ ] **Phase 9: Processing Loop Refactor** - Replace downgrade_executor processing loop with convertible_data providers and tiered fallback
- [ ] **Phase 10: Batch Lock Exploration** - Analyze and conditionally refactor batch_lock_utils to use convertible_data_batch

## Phase Details

### Phase 8: API Cleanup
**Goal**: Downgrade requests no longer carry or compute target_bytes, simplifying the interface before the processing loop refactor
**Depends on**: Phase 7
**Requirements**: DAPI-01, DAPI-02
**Success Criteria** (what must be TRUE):
  1. `request_downgrade` accepts no `target_bytes` parameter and `downgrade_request` has no `target_bytes` member
  2. `gpu_pipeline_executor` contains no target_bytes calculation logic for downgrade requests
  3. All existing tests pass with the simplified downgrade API (zero regressions)
**Plans:** 1 plan
Plans:
- [ ] 08-01-PLAN.md — Remove target_bytes from downgrade API and gpu_pipeline_executor

### Phase 9: Processing Loop Refactor
**Goal**: Downgrade executor uses convertible_data providers with tiered candidate fetching (repos, then gpu_pipeline_executor queue, then pipeline_executor queue) and convert()-based conversion
**Depends on**: Phase 8
**Requirements**: LOOP-01, LOOP-02, LOOP-03, LOOP-04, LOOP-05, LOG-01
**Success Criteria** (what must be TRUE):
  1. Processing loop iterates data_repositories lazily via `convertible_data_batch_provider`, one repository at a time
  2. When data_repositories are exhausted, processing loop fetches candidates from gpu_pipeline_executor task queue via `convertible_gpu_pipeline_task_provider`
  3. When gpu_pipeline_executor queue is exhausted, processing loop fetches candidates from pipeline_executor task queue via `convertible_gpu_pipeline_task_provider`
  4. Each candidate is converted via `convertible_data::convert()` and `downgrade_task` struct is eliminated (or justified if retained)
  5. Trace logging reports downgrade counts per source tier (data_repositories, gpu_pipeline_executor queue, pipeline_executor queue)
**Plans**: TBD

### Phase 10: Batch Lock Exploration
**Goal**: Determine whether batch_lock_utils can benefit from convertible_data_batch and apply the refactor if analysis supports it
**Depends on**: Phase 9
**Requirements**: LOCK-01, LOCK-02
**Success Criteria** (what must be TRUE):
  1. A functional diff analysis of `lock_or_prepare_batch` vs `convertible_data_batch::convert()` is documented with a clear go/no-go decision
  2. If go: `lock_or_prepare_batch` uses `convertible_data_batch::convert()` and all existing tests pass
  3. If no-go: rationale is documented in PROJECT.md Key Decisions with specific behavioral differences that prevent unification
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 8 → 9 → 10

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Core Queue | v1.0 | 2/2 | Complete | 2026-04-14 |
| 2. Predicate Inspection | v1.0 | 1/1 | Complete | 2026-04-14 |
| 3. Dead Code Removal | v1.1 | 1/1 | Complete | 2026-04-14 |
| 4. Queue Integration | v1.1 | 1/1 | Complete | 2026-04-14 |
| 5. State Machine & Interfaces | v2.0 | 2/2 | Complete | 2026-04-15 |
| 6. Batch Conversion | v2.0 | 2/2 | Complete | 2026-04-15 |
| 7. Task Queue Conversion | v2.0 | 2/2 | Complete | 2026-04-16 |
| 8. API Cleanup | v3.0 | 0/1 | Planned | - |
| 9. Processing Loop Refactor | v3.0 | 0/0 | Not started | - |
| 10. Batch Lock Exploration | v3.0 | 0/0 | Not started | - |

---
*Full v1.0 details archived to `.planning/milestones/v1.0-ROADMAP.md`*
*Full v1.1 details archived to `.planning/milestones/v1.1-ROADMAP.md`*
*Full v2.0 details archived to `.planning/milestones/v2.0-ROADMAP.md`*
