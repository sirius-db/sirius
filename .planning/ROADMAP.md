# Roadmap: Sirius Multi-GPU v1.1 — Re-integration + Cucascade I/O Migration

## Overview

This roadmap lands the v1.0 multi-GPU behavior on top of the 47-commit-drifted `dev`, replaces kvikio-backed parquet I/O with cucascade's `idisk_io_backend` registry, and closes the multi-GPU gaps (topology discovery, device-guard enforcement, P2P, adaptive scan) that were left open on `feature/multi-gpu-execution`. Phase 4 bumps cucascade and re-integrates the 23 multi-GPU commits as a single unit because the port can't compile without PR #96 headers. Phase 5 migrates parquet I/O to cucascade — the load-bearing change that enables multi-GPU-safe scan dispatch. Phase 6 closes the structural multi-GPU gaps (topology, device guards, per-NUMA host allocator, GPU↔GPU converter) that unblock Phase 7's P2P transfer + adaptive scan work (finishes the pending v1.0 03-02 plan).

## Prior Milestone

**v1.0 Multi-GPU Execution** (unmerged, behavioral baseline) lived at `refs/remotes/felipe-ssh/feature/multi-gpu-execution` and delivered Phases 1–3 partially. Its phase directories are preserved for reference:
- `.planning/phases/01-multi-gpu-foundation/` (complete)
- `.planning/phases/02-data-locality-task-scheduling/` (complete)
- `.planning/phases/03-numa-aware-memory-and-transfer-optimization/` (1/2 plans)

These are **history, not active work**. All v1.0 validated behavior is re-validated in v1.1 transitively through PORT-05. See `.planning/MILESTONES.md` for the v1.0 summary.

## Phases

**Phase Numbering:**
- v1.1 continues from v1.0's Phase 3, so v1.1 phases start at **Phase 4**.
- Integer phases (4, 5, 6, 7): Planned milestone work
- Decimal phases (e.g., 4.1): Urgent insertions (via `/gsd:insert-phase`)

- [x] **Phase 4: cuCascade Bump + v1.0 Re-integration** — Submodule bump to `origin/main`; replay 23 multi-GPU commits onto current `dev` so they compile against sirius-native types, YAML config, and PR #96 headers
- [ ] **Phase 5: Cucascade-Backed Parquet I/O Migration** — Ship `sirius::io::cucascade_datasource`, replace every `cudf::io::datasource::create(path)` call-site, remove the `rmm::cuda_stream_default` hygiene debt adjacent to the touched scan code
- [ ] **Phase 6: Multi-GPU Gap Closure (Topology, Device Safety, Host Memory, GPU↔GPU Converter)** — Close the structural v1.0 gaps (FOUND-01/04/06, CUCS-01/02) that never cleared on `feature/multi-gpu-execution`
- [ ] **Phase 7: P2P Direct Transfer + Adaptive Scan Partitioning** — Complete the pending v1.0 03-02 plan (MEM-04 P2P via `cudaMemcpyPeerAsync`, MEM-05 memory-proportional scan distribution)

## Phase Details

### Phase 4: cuCascade Bump + v1.0 Re-integration
**Goal**: The 23-commit v1.0 multi-GPU branch is re-applied onto current `dev` and compiles cleanly against the bumped cucascade (f47de0b) — the `dev`-era type system, YAML config, and PR #96 headers.
**Depends on**: Nothing (first phase of v1.1). Prior milestone v1.0 phases 1–3 supply the source commits to replay but are not runtime dependencies.
**Requirements**: BUMP-01, BUMP-02, BUMP-03, PORT-01, PORT-02, PORT-03, PORT-04, PORT-05
**Success Criteria** (what must be TRUE):
  1. `git -C cucascade rev-parse HEAD` reports `f47de0b`; `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make` succeeds end-to-end against the bumped submodule with no new warnings in sirius translation units (BUMP-01, BUMP-02).
  2. Pre-existing cucascade integration tests (`downgrade`, `reservation`, `converter` tags) pass post-bump with zero new flakes when run 5 times back-to-back: `build/release/extension/sirius/test/cpp/sirius_unittest "[downgrade],[reservation],[converter]"` (BUMP-03).
  3. `git log --oneline dev..HEAD` shows all 23 v1.0 multi-GPU commits (or a squash that includes them) on the current branch, with zero residual references to removed DuckDB vocabulary types: `grep -rnE 'LogicalType::(INTEGER|BIGINT|VARCHAR)' src/` returns zero hits in files touched by the port (PORT-01, PORT-02).
  4. Multi-GPU runtime configuration (`gpu_count`, per-GPU memory budgets, NUMA policy) round-trips through the YAML config parser introduced by PR #565 — the legacy `libconfig++` symbols are absent: `grep -rn 'libconfig' src/` returns zero hits (PORT-03).
  5. Push-model dispatch plumbing (`task_creator` locality computation, `management_eventloop` preferred-device routing, `preferred_device_id` on local and global pipeline task states) is present and exercised by the v1.0 locality test: `build/release/extension/sirius/test/cpp/sirius_unittest "[multi_gpu_foundation],[test_gpu_execution_locality]"` passes on a ≥2-GPU host (or reports `WARN+return` on single-GPU dev boxes per the v1.0 Catch2-v2 convention) (PORT-04, PORT-05).
**Plans**: 5 plans
- [x] 04-01-PLAN.md — cuCascade submodule bump (942c0bf -> f47de0b) + build/test gate (BUMP-01/02/03)
- [x] 04-02-PLAN.md — Cherry-pick 5 v1.0 code commits (preferred_device_id, locality score, push-model routing, scan distribution, integration test) onto dev; carve out downgrade_executor hunks (PORT-01/02/04 partial)
- [x] 04-03-PLAN.md — Re-author NUMA-aware downgrade on dev PR #579 shape + re-author 3 downgrade test commits (PORT-01/04 completion); includes human-verify checkpoint
- [x] 04-04-PLAN.md — PORT-03 YAML config verification + full pre-commit run (PORT-03)
- [x] 04-05-PLAN.md — Full unit-test gate + explicit hidden-tag invocation + structural grep gates + phase summary (PORT-05); includes phase sign-off checkpoint

### Phase 5: Cucascade-Backed Parquet I/O Migration
**Goal**: All Sirius parquet I/O flows through a Sirius-owned `cudf::io::datasource` that delegates to a per-GPU `cucascade::idisk_io_backend`. `cudf::io::datasource::create(path)` disappears from `src/`, and the adjacent `rmm::cuda_stream_default` hygiene debt is cleaned up.
**Depends on**: Phase 4 (needs cuCascade f47de0b headers for `idisk_io_backend` / `io_backend_registry`; needs re-integrated per-GPU memory spaces for backend instantiation under `rmm::cuda_set_device_raii`)
**Requirements**: IO-01, IO-02, IO-03, IO-04, IO-05, IO-06, IO-07, IO-08, IO-09, IO-10, IO-11, HYG-01, HYG-02
**Success Criteria** (what must be TRUE):
  1. `src/io/cucascade_datasource.{hpp,cpp}` exists, subclasses `cudf::io::datasource`, reports `supports_device_read() == false`, delegates `host_read` to `idisk_io_backend::read(path, host_ptr, size, offset)`, and returns pinned host memory allocated from cucascade's host memory resource so cuDF's `cuda_memcpy_async` stays asynchronous (IO-01, IO-02, IO-03).
  2. `SiriusContext` owns a per-GPU cache of `shared_ptr<cucascade::idisk_io_backend>` populated once per device under `rmm::cuda_set_device_raii` during context init (co-located with the existing `converter_registry::initialize()` call at `src/sirius_extension.cpp:1053`); backend lookup at scan time resolves by `preferred_device_id` (IO-04, IO-11).
  3. `grep -rnw 'datasource::create' src/` and `grep -rn 'source_info{[^}]*filepath[^}]*}' src/op/scan/iceberg_scan_task.cpp` both return zero hits — every prior call-site is migrated: `parquet_scan_task.cpp:312`, `parquet_scan_task.cpp:699`, `parquet_scan_task.cpp:769`, `parquet_scan_task.cpp:863` (via the new `_datasource`), `sirius_parquet_metadata_scan_operator.cpp:251`, `iceberg_scan_task.cpp:57-58`, `iceberg_scan_task.cpp:120-121`, `host_parquet_representation_converters.cpp:82-83` (IO-05, IO-06, IO-07, IO-08).
  4. TPC-H SF1 all 22 queries produce bitwise-identical result sets to the pre-migration baseline on the same hardware: `build/release/test/unittest --test-dir . test/sql/tpch-sirius.test` passes; TPC-H SF10 parquet scan wall-clock regression vs the Phase 4 kvikio-compat baseline is ≤30% (if it exceeds, the delta is filed upstream against cuCascade and recorded in the phase summary) (IO-09, IO-10).
  5. Parquet scan validated on a 2+ GPU host with one `idisk_io_backend` instance per GPU: `compute-sanitizer --tool memcheck build/release/test/unittest --test-dir . test/sql/tpch-sirius.test` reports zero CUDA-context errors, and a manual 2-GPU scan run logs distinct `cudaGetDevice()` values per backend instance (IO-04, IO-11).
  6. `grep -rn 'cuda_stream_default' src/` returns zero hits in any file touched by the v1.1 migration — specifically `src/op/scan/parquet_scan_task.cpp:468` (the `filter_row_groups_with_stats` call) now receives an explicit stream threaded from the scan task's global state (HYG-01, HYG-02).
**Plans**: 6 plans
- [x] 05-01-PLAN.md — Baseline capture + sirius::io::cucascade_datasource header + CMakeLists registration (IO-01 skeleton)
- [x] 05-02-PLAN.md — Adapter implementation + Catch2 unit tests with mock idisk_io_backend (IO-01, IO-02, IO-03)
- [ ] 05-03-PLAN.md — SiriusContext io_backend_registry + per-GPU backend cache under rmm::cuda_set_device_raii (IO-04, IO-11 infra)
- [ ] 05-04-PLAN.md — parquet_scan_task.cpp migration (lines 312 + 699) + HYG-01 explicit stream fix at line 468 (IO-05, IO-07, HYG-01)
- [ ] 05-05-PLAN.md — sirius_parquet_metadata_scan_operator.cpp:251 + iceberg_scan_task.cpp:57/120 migrations (IO-05, IO-06)
- [ ] 05-06-PLAN.md — IO-08 global grep gate + HYG-02 sweep + SF1 diff + IO-11 compute-sanitizer + IO-10 SF10 measurement + phase sign-off checkpoint + phase SUMMARY (IO-08, IO-09, IO-10, IO-11, HYG-02)

### Phase 6: Multi-GPU Gap Closure (Topology, Device Safety, Host Memory, GPU↔GPU Converter)
**Goal**: The five structural v1.0 gaps that never cleared on `feature/multi-gpu-execution` — runtime topology discovery, single-GPU no-regression guarantee, device-guard enforcement across every thread, GPU↔GPU converter registration, and per-NUMA pinned host memory spaces — are closed on `dev`-rebased code.
**Depends on**: Phase 4 (re-integrated per-GPU executor + memory-space plumbing is the substrate these gaps plug into). Can run in parallel with Phase 5 if plan-level file conflicts permit; the natural ordering is 4 → 5 → 6 → 7 because Phase 5 touches `SiriusContext::initialize` which Phase 6 also modifies.
**Requirements**: MGPU-01, MGPU-02, MGPU-03, MGPU-04, MGPU-05
**Success Criteria** (what must be TRUE):
  1. `SiriusContext::initialize()` calls `cucascade::topology_discovery` exactly once at startup; the resulting GPU count, NUMA domains, and GPU→NUMA mapping are logged at `info` level and are the sole source of truth for downstream memory-space and executor construction — no hand-rolled `cudaGetDeviceCount` / `numa_*` calls remain in `src/`: `grep -rn 'cudaGetDeviceCount\|numa_node_of_cpu' src/` returns zero hits outside the cucascade bridge (MGPU-01).
  2. On a single-GPU host, TPC-H SF10 end-to-end wall-clock is within 5% of the current `dev` HEAD baseline (measured 3-run median, same build flags) — `python3 test/tpch_performance/performance_test.py 10` output is captured in the phase summary and compared directly (MGPU-02).
  3. A 2+ GPU `compute-sanitizer --tool memcheck --require-cuda-init` run over the multi-GPU test suite (`build/release/extension/sirius/test/cpp/sirius_unittest "[multi_gpu_foundation]"`) reports zero "invalid device" or "context mismatch" errors, proving device-guard conventions hold on every execution thread (MGPU-03).
  4. The cucascade converter registry contains a GPU→GPU entry after `SiriusContext::initialize()` — verified by a unit test that calls `converter_registry::instance().has_converter(Tier::GPU, Tier::GPU)` and expects `true`; a round-trip GPU0→GPU1 conversion via the registered converter produces data equal to the input (feeds Phase 7's P2P path) (MGPU-04).
  5. Per-NUMA host memory spaces are constructed with `numa_region_pinned_host_allocator` — verified by inspecting `memory_manager_->get_memory_spaces_for_tier(Tier::HOST)` at context init and confirming the count equals the NUMA-node count reported by topology discovery; allocations from each host space land on the correct NUMA node (validated via `numactl --show` + `/proc/PID/numa_maps`) (MGPU-05).
**Plans**: TBD

### Phase 7: P2P Direct Transfer + Adaptive Scan Partitioning
**Goal**: Complete the pending v1.0 Phase 03-02 plan. GPU↔GPU data transfer uses `cudaMemcpyPeerAsync` directly (skipping host staging) when P2P access is available, and scan batches are distributed across GPUs proportional to available GPU memory rather than round-robin.
**Depends on**: Phase 6 (P2P needs the GPU↔GPU converter from MGPU-04; adaptive scan needs topology discovery from MGPU-01 to know per-GPU capacity)
**Requirements**: MGPU-06, MGPU-07
**Success Criteria** (what must be TRUE):
  1. On a host where `cudaDeviceCanAccessPeer(0, 1)` returns true, a GPU0→GPU1 data transfer uses `cudaMemcpyPeerAsync` on an explicit stream (no host round-trip) — verified by an nsys trace that shows exactly one device-to-device copy between the two GPUs and zero pinned-host staging allocations for the transfer; measured bandwidth exceeds the host-staged path from Phase 4 by ≥1.5× on the same hardware (MGPU-06).
  2. When P2P access is unavailable (`cudaDeviceCanAccessPeer` returns false), the converter falls back to the existing host-staged path without error — verified on a host with P2P explicitly disabled via `CUDA_VISIBLE_DEVICES` or equivalent (MGPU-06).
  3. A large scan run against a 2-GPU host with asymmetric available memory (e.g., GPU 0 pre-loaded to 80% capacity, GPU 1 idle) distributes batches proportional to free memory — the resulting batch counts per GPU differ by ≥2× and match the ratio of free capacity within 10% (MGPU-07).
  4. Integration test `test_gpu_execution_locality` extended with the P2P + adaptive-scan scenarios passes on a ≥2-GPU host; on single-GPU dev boxes the same test emits `WARN+return` per Catch2-v2 convention and does not fail (MGPU-06, MGPU-07).
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 4 -> 5 -> 6 -> 7. Phase 6 may run partially in parallel with Phase 5 at plan-granularity if conflicts permit; Phase 7 strictly follows Phase 6 (P2P needs MGPU-04).

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 4. cuCascade Bump + v1.0 Re-integration | 5/5 | Complete | 2026-04-20 |
| 5. Cucascade-Backed Parquet I/O Migration | 0/6 | Not started | - |
| 6. Multi-GPU Gap Closure | 0/TBD | Not started | - |
| 7. P2P Direct Transfer + Adaptive Scan Partitioning | 0/TBD | Not started | - |

## Coverage

v1.1 requirements: 28 total (5 PORT + 3 BUMP + 11 IO + 7 MGPU + 2 HYG) — 100% mapped.

| Phase | Requirements | Count |
|-------|--------------|-------|
| 4 | BUMP-01, BUMP-02, BUMP-03, PORT-01, PORT-02, PORT-03, PORT-04, PORT-05 | 8 |
| 5 | IO-01, IO-02, IO-03, IO-04, IO-05, IO-06, IO-07, IO-08, IO-09, IO-10, IO-11, HYG-01, HYG-02 | 13 |
| 6 | MGPU-01, MGPU-02, MGPU-03, MGPU-04, MGPU-05 | 5 |
| 7 | MGPU-06, MGPU-07 | 2 |
| **Total** | | **28** |

v1.0 "inherited" requirements (FOUND-02/03/05, MEM-01/02/03, CUCS-03/04, SCHED-01..05) are **not** mapped to v1.1 phases — they are re-validated transitively through PORT-05.
