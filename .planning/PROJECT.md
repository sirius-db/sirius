# Sirius — GPU-Native SQL Engine (Multi-GPU)

## What This Is

Sirius is a GPU-native SQL engine that runs as a DuckDB extension (`sirius.duckdb_extension`). It intercepts DuckDB's physical plan and routes supported operators to GPU execution via cuDF / RMM / cuCascade, falling back to DuckDB's CPU engine for unsupported cases. As of v1.1, Sirius executes transparently across multiple GPUs on a single node, with data-locality-aware task scheduling, NUMA-aware memory management, driver-level P2P peer access, and a multi-GPU-safe parquet I/O path built on cucascade's `idisk_io_backend` (no kvikio).

## Core Value

Any query can transparently execute across every GPU on the node — tasks are scheduled to the GPU where their input data already resides, memory pressure is absorbed by downgrading to the correct NUMA domain, and parquet I/O is routed through a multi-GPU-safe backend.

## Current State

**v1.2 shipped 2026-04-28** — Multi-GPU SQL Pipeline Fix.
- 3 phases / 18 plans / 39 tasks / 11 v1.2 requirements satisfied (8 fully + 3 partial via proxy)
- TPC-H SF100 Q1 num_gpus=2: 5.70s wall-clock, byte-identical to 1-GPU baseline (5.45s); 71 scan batches distributed GPU0=42 / GPU1=29 with cross-GPU intersection=0
- HYG-02 improved 41 → 40 (`rmm::cuda_stream_default` count) via Phase 10-03 stream-use-after-destroy fix
- Branch: `feature/single-node-multi-gpu2`
- Archive: `.planning/milestones/v1.2-*`
- Open Phase-11 candidate: pre-existing `[mgpu-audit]` SIGSEGV at `test_gpu_execution_tpch_mgpu_audit.cpp:200` in the `attach_integration_duckdb` path (orthogonal to v1.2 fixes; documented in `v1.2-MILESTONE-AUDIT.md`)

**v1.1 shipped 2026-04-21** — Multi-GPU Re-integration + Cucascade I/O Migration.
- 4 phases / 19 plans / 44 tasks / 28 requirements cleared
- Full test suite: 979/979 pass on N=2 hardware (2× RTX 6000 Ada, driver 595.58.03, CUDA 13.2)
- Archive: `.planning/milestones/v1.1-*`

## Requirements

### Validated

Shipped and validated in v1.1.

- ✓ **BUMP-01..03** — cucascade submodule bumped 942c0bf → f47de0b (PR #96 file-downgrade + io_backend_registry, PR #100 underflow fix, PR #103 stream sync, PR #104 NVML drop) — *v1.1*
- ✓ **PORT-01..05** — 23 v1.0 multi-GPU commits re-landed on current `dev`; push-model dispatch + `preferred_device_id` plumbing preserved; YAML config replaces libconfig++ — *v1.1*
- ✓ **IO-01..11** — `sirius::io::cucascade_datasource` replaces every `cudf::io::datasource::create(path)` call-site; per-GPU `idisk_io_backend` cache on `SiriusContext` under `rmm::cuda_set_device_raii`; kvikio removed; SF1 correctness preserved — *v1.1*
- ✓ **MGPU-01** — runtime topology discovery via cucascade (fail-hard on zero-GPU; 3-line startup log) — *v1.1*
- ✓ **MGPU-02** — single-GPU SF10 no-regression (absolute Phase-6 timings captured) — *v1.1*
- ✓ **MGPU-03** — device-guard `cudaSetDevice` enforcement with `spdlog::error` in both `noexcept` per-thread init callbacks; compute-sanitizer memcheck 0 errors — *v1.1*
- ✓ **MGPU-04** — GPU↔GPU converter registered in `sirius::converter_registry::initialize()`; forward-leg + return-leg round-trip PASS on N=2 — *v1.1*
- ✓ **MGPU-05** — per-NUMA host memory spaces via `numa_region_pinned_host_allocator` — *v1.1*
- ✓ **MGPU-06** — GPU-direct P2P via `cudaMemcpyPeerAsync`; `cudaDeviceEnablePeerAccess` loop at init; Sirius-side `sirius_p2p_converter_factory` override works around cucascade's cross-stream race — *v1.1*
- ✓ **MGPU-07** — adaptive scan partitioning proportional to free GPU memory (3.08× ratio → batch-count skew within 10% tolerance) — *v1.1*
- ✓ **HYG-01/02** — `rmm::cuda_stream_default` removed from `parquet_scan_task.cpp:468` and every Phase-5-modified file — *v1.1*

### Active

<!-- v1.3 not yet scoped. Run /gsd:new-milestone to define. -->

(No active requirements — next milestone awaiting scoping via `/gsd:new-milestone`.)

**v1.2 deliverables (now Validated):**
- ✓ **FIX-01..04** — cross-device stream-correctness fixes (Pattern 2 idiom): per-GPU stream pool in `duckdb_scan_executor`, Sirius-side `host→gpu` converter override, per-GPU filter translation at plan time, `translated_expression::owned_stream` for scalar lifetime correctness — *v1.2*
- ✓ **TEST-01..04** — TPC-H integration parameterized on `num_gpus∈{1,2}` via Catch2 GENERATE; `integration-2gpu.yaml` fixture; SF1 22 queries × {1,2} GPUs all PASS; SF10 Q1/Q6/Q12 PASS — *v1.2*
- ✓ **AUDIT-01..03** — `[mgpu-audit]` payload extended with `task_id`/`batch_id`; AUDIT TEST_CASE wired in default unit-tests run; Phase 9 disjointedness REQUIRE (`std::set_intersection(scan_ids) == ∅`) fires in `tpch_q1_sf10_2gpu` — *v1.2 (canonical TEST_CASE blocked by pre-existing SIGSEGV; substantive evidence via SF100 + SF10 proxy runs)*

## Deferred to Future Milestones

- **`[mgpu-audit]` per-GPU distribution AUDIT TEST_CASE SIGSEGV** at `test_gpu_execution_tpch_mgpu_audit.cpp:200` (`attach_integration_duckdb` path; pre-existing on base before v1.2; orthogonal to parquet filter translation path; Phase 11 candidate, < 50 LOC expected)
- Upstream cucascade `convert_gpu_to_gpu` cross-stream fix (drop Sirius override once upstream lands)
- Phase-5 vs Phase-4 parquet I/O regression comparison
- Phase-6 vs Phase-5 single-GPU SF10 regression comparison
- Cucascade `idisk_io_backend` file-handle cache (research pitfall P1)
- `cudaDeviceDisablePeerAccess` on explicit teardown
- TPC-H Q4 parquet intermittent flake investigation

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
| Replace kvikio with cucascade io_backend | kvikio/cuFile bind to a single CUDA context → unsafe for multi-GPU task dispatch. | ✓ Good — v1.1 shipped; `grep -rnw 'datasource::create' src/` returns zero hits |
| Cucascade pinned to `origin/main` (f47de0b) | PR #96 introduces the `idisk_io_backend` + `io_backend_registry` Sirius depends on. | ✓ Good — all 28 v1.1 requirements validated on this pin |
| Re-integrate as a new milestone (v1.1), not merge | 47 dev commits include type-system and config refactors — fresh plan-by-plan replay cheaper than conflict resolution. | ✓ Good — v1.1 shipped in 4 phases; replay strategy validated |
| Push-model task dispatch for locality routing | Pull model couldn't use data-locality info; push (pop task first, route by preferred_device_id) enables SCHED-01..04. | ✓ Good — validated end-to-end in v1.1 |
| preferred_device_id on both local_state + global_state | Per-task override with pipeline-level default covers both scan distribution and inherited locality. | ✓ Good — v1.1 integration tests PASS |
| NUMA-aware downgrade via cucascade `any_memory_space_in_tier_with_preference` | Avoids bespoke NUMA logic in Sirius; cucascade owns tier selection. | ✓ Good — v1.1 re-authored onto dev's PR #579 shape; downgrade tests PASS on N=2 |
| KvikIO/GDS explicitly out of scope | Removing the dependency was the milestone goal; re-adding for any path defeats the purpose. | ✓ Good — v1.1 grep gate enforced |
| Sirius-side `sirius_p2p_converter` override for GPU↔GPU | cucascade's `convert_gpu_to_gpu` has a cross-stream race (`cudaMemcpyPeerAsync` on caller stream vs post-copy table on target_stream). Override issues peer copy on target_stream. | ⚠️ Revisit — works around upstream; upstream PR is tech debt in v1.2 |
| Consume `cudaGetLastError()` after `cudaDeviceEnablePeerAccess` | CUDA leaves the return code in thread-local error slot; subsequent unrelated calls fail spuriously with same code. | ✓ Good — pattern established for future CUDA-state-mutation code |
| `supports_device_read() == false` in cucascade_datasource | Host-stage via pinned memory + `cuda_memcpy_async` on caller's stream stays truly async and avoids GDS entirely. | ✓ Good — v1.1 IO-02/03 validated |
| Adaptive scan via existing `select_target_gpu` (no code change needed) | `duckdb_scan_executor::select_target_gpu` was already memory-proportional since v1.0 Phase 2; Phase 7 MGPU-07 scope was test-authoring only. | ✓ Good — 3.08× free-memory ratio test proves proportional skew within 10% tolerance |
| Per-GPU filter translation at plan time (Phase 8 residual closure 93fea6f) | `sirius_physical_parquet_scan` originally translated DuckDB filter expressions ONCE, binding scalars to the planner's current device. Tasks dispatched to other GPUs faulted. Build one tree per configured GPU at plan time, select per-task at converter time. | ✓ Good — closes the v1.2 ship-blocker on parquet TPC-H Q1 num_gpus=2 |
| `_batch_gpu_affinity` map records ownership but does NOT consult at dispatch time (Phase 9 minimum-viable) | Recording is sufficient for the disjointedness REQUIRE regression gate; consultation-at-dispatch was deferred to keep scope tight. Affinity is implicitly preserved because `_scan_round_robin` is monotonic. | ✓ Good — disjointedness REQUIRE fires green at SF10 + SF100; cross-GPU intersection=0 |
| `translated_expression::owned_stream` declared BEFORE `owned_literals` (Phase 10-03) | C++ reverse-destruction order: scalars `cudaFreeAsync` first (using stream handle), then stream destroys. Without this ordering, `cudaFreeAsync(ptr, stale_handle)` SIGSEGVs at next QueryBegin. | ✓ Good — closes the test-ordering-dependent SIGSEGV that 09-04 exposed; HYG-02 improved 41→40 |
| Run all integration/SF100 tests via MCP on this host (no human-delegated checkpoints) | 2026-04-24 host-capability discovery: `mcp__project-commands__run_command nvidia-smi` shows 2× RTX 6000 Ada visible; agent can run the full v1.2 ship-gate autonomously. | ✓ Good — Phase 9-04 + Phase 10-04 ship-gates ran fully autonomously via MCP |

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
*Last updated: 2026-04-28 — v1.2 milestone shipped (Multi-GPU SQL Pipeline Fix)*
