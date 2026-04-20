# Sirius — GPU-Native SQL Engine (Multi-GPU)

## What This Is

Sirius is a GPU-native SQL engine that runs as a DuckDB extension (`sirius.duckdb_extension`). It intercepts DuckDB's physical plan and routes supported operators to GPU execution via cuDF / RMM / cuCascade, falling back to DuckDB's CPU engine for unsupported cases. The engine is being extended to execute across multiple GPUs on a single node, with data-locality-aware task scheduling and NUMA-aware memory management.

## Core Value

Any query can transparently execute across every GPU on the node — tasks are scheduled to the GPU where their input data already resides, memory pressure is absorbed by downgrading to the correct NUMA domain, and parquet I/O is routed through a multi-GPU-safe backend.

## Requirements

### Validated

<!-- Shipped and validated on feature/multi-gpu-execution (v1.0, unmerged — carried as the behavioral baseline this milestone re-integrates on top of current `dev`). -->

- ✓ **FOUND-02** — per-GPU executor with independent CUDA context, stream pool, thread pool *(v1.0, unmerged)*
- ✓ **FOUND-03** — per-GPU reservation tracking (OOM on GPU 0 ≠ block GPU 1) *(v1.0, unmerged)*
- ✓ **FOUND-05** — per-GPU downgrade executor monitors its own memory space *(v1.0, unmerged)*
- ✓ **MEM-01** — GPU→HOST downgrade prefers pinned host memory on the same NUMA domain *(v1.0, unmerged)*
- ✓ **MEM-02** — fallback to cross-NUMA host memory when NUMA-local is exhausted *(v1.0, unmerged)*
- ✓ **MEM-03** — GPU→GPU data transfer via host staging using cucascade converters *(v1.0, unmerged)*
- ✓ **CUCS-03** — NUMA-aware ordering in downgrade strategy *(v1.0, unmerged)*
- ✓ **CUCS-04** — multi-GPU memory-space configuration tested with N>1 GPUs *(v1.0, unmerged)*
- ✓ **SCHED-01..SCHED-05** — data-locality task routing + cross-GPU scan distribution *(v1.0, unmerged)*

> "Unmerged" = implemented and tested on `feature/multi-gpu-execution` but never landed on `dev`. This milestone re-integrates them on top of `dev`'s 47 intervening commits.

### Active

<!-- Milestone v1.1: Multi-GPU Re-integration + Cucascade I/O Migration. Scoped in REQUIREMENTS.md. -->

- [ ] Re-integrate 23 multi-GPU commits on top of current `dev`
- [ ] Replace kvikio-backed parquet I/O with cucascade's `disk_io_backend` + `io_backend_registry`
- [ ] Bump cucascade submodule from 942c0bf → `origin/main` (f47de0b, includes PR #96 file-downgrade)
- [ ] Complete Phase 3 Plan 2 — MEM-04 (cudaMemcpyPeerAsync P2P) + MEM-05 (adaptive scan partitioning by available GPU memory)
- [ ] Close remaining FOUND-01, FOUND-04, FOUND-06, CUCS-01, CUCS-02 gaps exposed by re-integration

### Out of Scope

- **Distributed multi-node execution** — different problem domain (network serialization, fault tolerance).
- **GPU-Direct RDMA (network GDS)** — only relevant for multi-node; would re-introduce kvikio/GDS dependency we're removing.
- **KvikIO / cuFile backends for parquet** — explicitly replaced by cucascade io_backend; per-GPU CUDA-context scoping makes kvikio unsafe for multi-GPU scheduling.
- **Heterogeneous GPUs** — assume homogeneous GPUs (DGX/HGX configurations).
- **Query-optimizer-level GPU placement** — routing happens at task dispatch with actual data sizes, not plan time with estimates.
- **Data repartitioning / shuffle exchange** — single-node batch-level scheduling avoids global shuffle.
- **Changes to legacy `namespace duckdb` code path** — multi-GPU targets Super Sirius (`namespace sirius`) only.

## Context

- **Worktree branch:** `feature/single-node-multi-gpu2` (fresh worktree with no prior `.planning/`). Sibling to `feature/single-node-multi-gpu` which is at `dev` head.
- **Prior work:** `refs/remotes/felipe-ssh/feature/multi-gpu-execution` implemented Phases 1–3 (partial) of multi-GPU execution, landed 23 commits that never merged to `dev`.
- **Dev drift:** 47 commits on `dev` since the multi-gpu branch diverged, including: sirius-native type system (PR #643), YAML config replacing libconfig++ (#565), hive partition columns (#570), AST expression executor (#531), refactors removing DuckDB vocabulary types (#564/#626/#628), row group pruning (#363).
- **cucascade:** pinned to 942c0bf in the worktree; PR #96 (`Feature/file downgrade`) introduced `disk_io_backend`, `io_backend_registry`, `disk_data_representation`, `disk_file_format`. Additional commits on `origin/main` through f47de0b (NVML drop, stream sync, benchmark bump).
- **Parquet I/O surface in src/:** `hybrid_scan_reader` (used in `host_parquet_representation.{hpp,cpp}`, `host_parquet_representation_converters.cpp`) and direct `cudf::io::parquet_reader_options` in `op/scan/{parquet,iceberg}_scan_task.cpp` and `sirius_parquet_metadata_scan_operator.cpp`. cuDF internally uses kvikio for GPU-direct storage when available.

## Constraints

- **Tech stack:** CUDA 13+, C++20, CUDA std 20, separable compilation. GPU arches 75–120 (Turing → Blackwell).
- **Build:** pixi-driven, `pixi run make -jN`. Never use `pixi run` directly from Claude — route through `mcp__project-commands__run_command` per user preference.
- **Streams:** No `rmm::cuda_stream_default` — every allocation/copy/kernel uses an explicit stream (user rule).
- **cuCascade API:** All disk I/O and tier conversion must go through cucascade's converter + io_backend registries. No hand-rolled kvikio/cuFile/GDS calls anywhere in `src/`.
- **Super Sirius only:** Multi-GPU work targets `namespace sirius`. Legacy `gpu_processing` path (`namespace duckdb`) is frozen.
- **Fallback-first:** Any GPU path that can't run multi-GPU-safely must downgrade through the existing fallback mechanism, not crash.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Replace kvikio with cucascade io_backend | kvikio/cuFile bind to a single CUDA context → unsafe for multi-GPU task dispatch; cucascade's registry supports per-backend factories and is already the tier-conversion authority. | — Pending |
| Cucascade pinned to `origin/main` | PR #96 introduces the `disk_io_backend` + `io_backend_registry` we depend on; staying on 942c0bf would require backporting. | — Pending |
| Re-integrate as a new milestone (v1.1), not merge | 47 dev commits include type-system and config refactors that touch files the multi-gpu work modifies; a fresh plan-by-plan re-integration is cheaper than conflict resolution. | — Pending |
| Push-model task dispatch for locality routing | Pull model (wait for task_request) couldn't use data-locality info; push (pop task first, route by preferred_device_id) enables SCHED-01..04. | ✓ Validated in v1.0 (02-01-SUMMARY) |
| preferred_device_id on both local_state + global_state | Per-task override with pipeline-level default covers both scan distribution and inherited locality. | ✓ Validated in v1.0 (02-01-SUMMARY) |
| NUMA-aware downgrade via cucascade `any_memory_space_in_tier_with_preference` | Avoids bespoke NUMA logic in Sirius; cucascade owns tier selection. | ✓ Validated in v1.0 (01-01, 03-01) |
| KvikIO/GDS explicitly out of scope | Removing the dependency is the milestone; re-adding it for any code path defeats the purpose. | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-20 — v1.1 milestone initialized (Multi-GPU Re-integration + Cucascade I/O Migration)*
