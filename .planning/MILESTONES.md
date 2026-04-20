# Milestones

## v1.0 — Multi-GPU Execution (Foundation + Scheduling + NUMA) *(unmerged baseline)*

**Branch:** `refs/remotes/felipe-ssh/feature/multi-gpu-execution` (23 commits not on `dev`)
**Status:** Implemented and tested, never landed on `dev`. Carried forward as the behavioral baseline this milestone re-integrates.
**Completed plans:** 5 / 7

### What shipped on that branch

- **Phase 1 — Multi-GPU Foundation** (3/3 plans)
  - Plan 01-01: NUMA-aware downgrade, multi-device terminate sync, P2P access enablement
  - Plan 01-02: Device-guard audit + multi-GPU foundation validation tests
  - Plan 01-03: NUMA-aware downgrade tests + GPU-to-GPU transfer validation
  - **Requirements cleared:** FOUND-02, FOUND-03, FOUND-05, CUCS-03, CUCS-04, MEM-03

- **Phase 2 — Data-Locality Task Scheduling** (2/2 plans)
  - Plan 02-01: Data-locality computation in `task_creator` + locality-aware routing in `management_eventloop` (push-model dispatch, `preferred_device_id` plumbing)
  - Plan 02-02: Cross-GPU scan distribution + integration tests
  - **Requirements cleared:** SCHED-01, SCHED-02, SCHED-03, SCHED-04, SCHED-05

- **Phase 3 — NUMA-Aware Memory + Transfer Optimization** (1/2 plans)
  - Plan 03-01: NUMA downgrade ordering verification (MEM-01, MEM-02)
  - Plan 03-02: P2P transfer + adaptive scan distribution — **PENDING** (MEM-04, MEM-05)

### Gaps left open

- **Not cleared:** FOUND-01 (runtime topology discovery), FOUND-04 (single-GPU no-regression), FOUND-06 (device-guard enforcement across all threads), CUCS-01 (GPU↔GPU converter registration), CUCS-02 (per-NUMA host allocator), MEM-04 (P2P direct), MEM-05 (adaptive scan).
- **Never merged to `dev`** — 47 commits landed on `dev` after the branch diverged (sirius-native types, YAML config, AST expression executor, hive partitioning, row group pruning).

### Why v1.0 didn't ship

Dev drift. The Sirius type system was refactored (`logical_type` / `type_id`), libconfig++ was replaced with YAML, DuckDB vocabulary types were removed from the core engine — touching nearly every file the multi-GPU work modified. Merging produced conflicts across ~20 files; a clean replay on top of `dev` is cheaper than conflict resolution.

---

## v1.1 — Multi-GPU Re-integration + Cucascade I/O Migration *(current)*

**Branch:** `feature/single-node-multi-gpu2`
**Status:** Initialized 2026-04-20
**Goal:** Land the v1.0 multi-GPU behavior on top of current `dev`, replace kvikio-backed parquet I/O with cucascade's pluggable io_backend, and bump cucascade to `origin/main`.

See: `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`.

---

*Maintained at milestone boundaries. Current milestones live in `PROJECT.md` under Active / Out of Scope.*
