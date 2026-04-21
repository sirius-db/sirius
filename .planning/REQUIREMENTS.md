# Requirements: Sirius Multi-GPU

**Core Value:** Any query can transparently execute across every GPU on the node — tasks scheduled to the GPU where their data resides, memory pressure absorbed by NUMA-aware downgrade, and parquet I/O routed through a multi-GPU-safe backend.

---

## Milestone v1.1 Requirements (current) — Multi-GPU Re-integration + Cucascade I/O Migration

**Defined:** 2026-04-20
**Goal:** Land the v1.0 multi-GPU behavior on top of current `dev` (47 intervening commits), replace kvikio-backed parquet I/O with cucascade's `idisk_io_backend`, bump the cuCascade submodule to `origin/main`, and close the pending Phase 3 plan.

### Port — Re-integrate v1.0 Multi-GPU Commits onto Current `dev`

- [x] **PORT-01**: 23 multi-GPU commits from `refs/remotes/felipe-ssh/feature/multi-gpu-execution` applied (via cherry-pick or replay) on top of current `dev` HEAD with clean compilation on this branch
- [x] **PORT-02**: Re-integrated code compiles against sirius-native type system (`logical_type` / `type_id`, PR #643) — no residual uses of removed DuckDB vocabulary types
- [x] **PORT-03**: Multi-GPU runtime settings (GPU count, per-GPU memory budgets, NUMA policy) are read from YAML config (PR #565) rather than libconfig++
- [x] **PORT-04**: Push-model task dispatch + `preferred_device_id` plumbing from v1.0 Phase 2 preserved (`task_creator`, `management_eventloop`, `pipeline_task_states`)
- [x] **PORT-05**: Existing multi-GPU test suites pass: `multi_gpu_foundation`, downgrade executor NUMA tests, `test_gpu_execution_locality` integration tests

### Bump — cuCascade Submodule Update

- [x] **BUMP-01**: cuCascade submodule pointer updated from 942c0bf to `origin/main` (f47de0b)
- [x] **BUMP-02**: Sirius builds cleanly against new cuCascade surface — absorbs PR #96 (file downgrade / `idisk_io_backend` / `io_backend_registry`), PR #100 (memory_space underflow fix), PR #103 (stream sync on GPU representation destroy), PR #104 (NVML link drop)
- [x] **BUMP-03**: All pre-existing cucascade-integration tests (`downgrade`, `reservation`, `converter`) pass after bump with no new flakes

### IO — Replace kvikio with cuCascade `idisk_io_backend` for Parquet I/O

- [x] **IO-01**: `sirius::io::cucascade_datasource` subclass of `cudf::io::datasource` ships in `src/io/`, backed by cuCascade's `idisk_io_backend` via `io_backend_registry` factory
- [x] **IO-02**: `cucascade_datasource` declares `supports_device_read() == false` so cuDF host-stages reads and issues memcpys on the caller's explicit stream (no GDS, no cuFile, no kvikio)
- [x] **IO-03**: `host_read` returns pinned host memory allocated from cucascade's host-memory resource so cuDF's `cuda_memcpy_async` stays truly asynchronous
- [x] **IO-04**: Per-GPU `idisk_io_backend` instances cached in `SiriusContext`, created once per device under `rmm::cuda_set_device_raii` so each instance owns streams/pinned buffers in its GPU's context
- [x] **IO-05**: `cudf::io::datasource::create(filepath)` removed from `src/op/scan/parquet_scan_task.cpp:312`, `:699` and `src/op/scan/sirius_parquet_metadata_scan_operator.cpp:251` — all three routed through the new factory
- [x] **IO-06**: Iceberg delete-file reads at `src/op/scan/iceberg_scan_task.cpp:57-58` and `:120-121` pass `source_info{ds.get()}` with a cucascade-backed datasource instead of `source_info{filepath}`
- [x] **IO-07**: `prefetched_data_source` fallback datasource is cucascade-backed at `src/data/host_parquet_representation_converters.cpp:82-83` and at the construction site `src/op/scan/parquet_scan_task.cpp:769`
- [x] **IO-08**: `grep -rnw 'datasource::create' src/` returns zero hits — no Sirius code creates a kvikio-backed datasource
- [x] **IO-09**: TPC-H SF1 all queries produce results identical to pre-migration baseline (correctness)
- [x] **IO-10**: TPC-H SF10 parquet scan wall-clock regression vs kvikio-compat baseline ≤ 30%; any larger delta filed as cuCascade upstream issue and documented in the phase summary *(Phase-4 regression comparison deferred to future optimization work per user directive 2026-04-21; absolute Phase-5 SF10 wall-clock captured on real N=2 hardware in `05-06-MULTIGPU-VALIDATION.md`)*
- [x] **IO-11**: Parquet scan validated on multi-GPU hardware — one `idisk_io_backend` per GPU, cross-GPU reads work, no CUDA-context leak between devices (verified with compute-sanitizer or `cudaGetDevice` logging)

### MGPU — Close v1.0 Multi-GPU Gaps

These are requirements from v1.0 that were defined but never cleared on `feature/multi-gpu-execution`. They re-surface as active in v1.1.

- [ ] **MGPU-01** *(formerly FOUND-01)*: Runtime topology discovery via cucascade `topology_discovery` — GPU count, NUMA domains, GPU↔NUMA mapping
- [ ] **MGPU-02** *(formerly FOUND-04)*: Single-GPU systems run with zero behavior or performance regression vs the pre-milestone baseline (baseline = current `dev` HEAD TPC-H SF10 timings)
- [x] **MGPU-03** *(formerly FOUND-06)*: Device-guard conventions enforced on every execution thread — validated via compute-sanitizer `--tool memcheck --require-cuda-init` on a 2+ GPU system
- [x] **MGPU-04** *(formerly CUCS-01)*: GPU↔GPU representation converter registered in cucascade converter registry (feeds MEM-04 P2P path)
- [ ] **MGPU-05** *(formerly CUCS-02)*: Per-NUMA host memory spaces configured with `numa_region_pinned_host_allocator`
- [ ] **MGPU-06** *(formerly MEM-04)*: GPU-direct peer-to-peer transfer via `cudaMemcpyPeerAsync` when P2P access is available — measurably faster than host staging (completes 03-02 plan)
- [ ] **MGPU-07** *(formerly MEM-05)*: Adaptive scan partitioning — scan batches distributed across GPUs proportional to available GPU memory, not round-robin (completes 03-02 plan)

### HYG — Hygiene Fixes Adjacent to Touched Code

- [x] **HYG-01**: `rmm::cuda_stream_default` removed from `src/op/scan/parquet_scan_task.cpp:468` — explicit stream plumbed from task global state (user rule: never use `cuda_stream_default`)
- [x] **HYG-02**: Any other `rmm::cuda_stream_default` callsite introduced or left behind by the v1.0 re-integration is replaced with an explicit stream before phase sign-off

---

## Out of Scope (v1.1)

| Feature | Reason |
|---------|--------|
| Distributed multi-node execution | Different problem domain — needs network serialization, fault tolerance |
| GPU-Direct RDMA / network GDS | Multi-node only; would re-introduce a kvikio-like dependency we're removing |
| **KvikIO / cuFile / cudf-default datasource for parquet** | **Single-CUDA-context binding breaks multi-GPU dispatch — this is the reason for the milestone** |
| S3 / HTTP parquet sources | cuCascade's built-in `"pipeline"` backend is local-filesystem only; remote URIs fall back to `cudf::io::datasource::create` which is banned by IO-08. If remote is ever needed, add a new cuCascade backend, not a kvikio escape hatch. |
| Custom NVLink/NVSwitch protocols | CUDA runtime handles interconnect routing transparently |
| Query-optimizer-level GPU placement | Scheduling happens at task dispatch with actual data sizes |
| Data repartitioning / shuffle exchange | Single-node batch-level scheduling avoids global shuffle |
| Heterogeneous GPU support | Homogeneous GPUs assumed (DGX/HGX configs) |
| Changes to legacy `namespace duckdb` code path | Multi-GPU targets Super Sirius (`namespace sirius`) only |
| `cudf::io::datasource::create` fallback for any reason | Defeats the milestone — enforced by grep check in IO-08 |

---

## v1.0 Requirements (inherited — unmerged, behavior baseline only)

Requirements defined on `refs/remotes/felipe-ssh/feature/multi-gpu-execution`. Most are implemented but never landed on `dev`. Re-validation happens through the PORT-* + MGPU-* requirements above.

### Foundation

- [x] **FOUND-02** *(v1.0)* Per-GPU executor with independent CUDA context, stream pool, thread pool — validated in Phase 01-02
- [x] **FOUND-03** *(v1.0)* Per-GPU reservation tracking (OOM isolation across GPUs) — validated in Phase 01-02
- [x] **FOUND-05** *(v1.0)* Per-GPU downgrade executor — validated in Phase 01-01 / 01-03
- [ ] **FOUND-01** *(v1.0, open)* → tracked as **MGPU-01** in v1.1
- [ ] **FOUND-04** *(v1.0, open)* → tracked as **MGPU-02** in v1.1
- [ ] **FOUND-06** *(v1.0, open)* → tracked as **MGPU-03** in v1.1

### Task Scheduling

- [x] **SCHED-01..SCHED-05** *(v1.0)* Data-locality routing + cross-GPU scan distribution — validated in Phase 02-01 / 02-02. Re-validated by PORT-04 + PORT-05 once rebased.

### Memory Management

- [x] **MEM-01, MEM-02** *(v1.0)* NUMA-ordered downgrade — validated in Phase 03-01
- [x] **MEM-03** *(v1.0)* GPU↔GPU host-staged transfer — validated in Phase 01-03
- [ ] **MEM-04** *(v1.0, open)* → tracked as **MGPU-06** in v1.1
- [ ] **MEM-05** *(v1.0, open)* → tracked as **MGPU-07** in v1.1

### cucascade Integration

- [x] **CUCS-03, CUCS-04** *(v1.0)* NUMA-aware strategy + multi-GPU memory-space config — validated in Phase 01-01
- [ ] **CUCS-01** *(v1.0, open)* → tracked as **MGPU-04** in v1.1
- [ ] **CUCS-02** *(v1.0, open)* → tracked as **MGPU-05** in v1.1

---

## Deferred / Future (v2.0)

- **OPT-01** — Coordinated multi-GPU OOM handling (migrate to peer GPU before downgrading to host)
- **OPT-02** — Topology-aware per-GPU telemetry (utilization, data-movement volume, scheduling decisions)
- **OPT-03** — Hash-partitioned scan routing by join key for join co-location
- **OPT-04** — Automatic data rebalancing across GPUs
- **OPT-05** — Remote / object-store parquet sources through a new cuCascade backend

---

## Traceability

*Filled in by roadmapper 2026-04-20.*

### v1.1 Requirements → Phase Mapping

| Requirement | Phase | Status |
|-------------|-------|--------|
| PORT-01 | Phase 4 | Complete |
| PORT-02 | Phase 4 | Complete |
| PORT-03 | Phase 4 | Complete |
| PORT-04 | Phase 4 | Complete |
| PORT-05 | Phase 4 | Complete |
| BUMP-01 | Phase 4 | Complete |
| BUMP-02 | Phase 4 | Complete |
| BUMP-03 | Phase 4 | Complete |
| IO-01 | Phase 5 | Complete |
| IO-02 | Phase 5 | Complete |
| IO-03 | Phase 5 | Complete |
| IO-04 | Phase 5 | Complete |
| IO-05 | Phase 5 | Complete |
| IO-06 | Phase 5 | Complete |
| IO-07 | Phase 5 | Complete |
| IO-08 | Phase 5 | Complete |
| IO-09 | Phase 5 | Complete |
| IO-10 | Phase 5 | Complete (Phase-4 regression comparison deferred per user directive) |
| IO-11 | Phase 5 | Complete |
| HYG-01 | Phase 5 | Complete |
| HYG-02 | Phase 5 | Complete |
| MGPU-01 | Phase 6 | Pending |
| MGPU-02 | Phase 6 | Pending |
| MGPU-03 | Phase 6 | Complete |
| MGPU-04 | Phase 6 | Complete |
| MGPU-05 | Phase 6 | Pending |
| MGPU-06 | Phase 7 | Pending |
| MGPU-07 | Phase 7 | Pending |

**Coverage:**
- v1.1 requirements: 28 total (5 PORT + 3 BUMP + 11 IO + 7 MGPU + 2 HYG)
- Mapped to phases: 28 / 28 (100%)
- Unmapped: 0
- Phases: 4, 5, 6, 7 (continuing numbering from v1.0 prior milestone which ended at Phase 3)

### v1.0 Requirements (inherited — not re-mapped)

v1.0 validated requirements (FOUND-02/03/05, MEM-01/02/03, CUCS-03/04, SCHED-01..05) are **re-validated transitively via PORT-05** in Phase 4 rather than re-mapped as explicit v1.1 line items. Open v1.0 requirements have been renamed to MGPU-* ids and mapped above.

---

*v1.0 requirements defined: 2026-04-02*
*v1.1 requirements defined: 2026-04-20*
*Last updated: 2026-04-21 — Phase 5 complete: IO-08/09/10/11 + HYG-02 closed (IO-10 Phase-4 regression comparison deferred to future optimization per user directive)*
