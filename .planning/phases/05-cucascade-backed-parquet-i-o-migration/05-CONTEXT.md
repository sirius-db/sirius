# Phase 5: Cucascade-Backed Parquet I/O Migration - Context

**Gathered:** 2026-04-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Every Sirius parquet I/O path flows through a Sirius-owned `cudf::io::datasource` adapter that delegates to a per-GPU `cucascade::idisk_io_backend`. The adapter (`sirius::op::scan::io_backend_datasource` or equivalent) reports `supports_device_read() == false` so cuDF host-stages reads through pinned memory and issues `cuda_memcpy_async` on the caller's explicit stream — preserving async semantics end-to-end without GDS/cuFile/kvikio anywhere in `src/`. `cudf::io::datasource::create(path)` disappears from `src/` (grep returns zero hits), and the adjacent `rmm::cuda_stream_default` callsite at `src/op/scan/parquet_scan_task.cpp:468` is cleaned up as part of the migration.

**In scope:** adapter class, per-GPU backend cache owned by `SiriusContext`, replacement of 7 `datasource::create` / filepath-based `source_info` call sites, HYG-01/02 for touched files, TPC-H SF1 correctness + SF10 ≤30% regression budget, compute-sanitizer memcheck on 2+ GPU host.

**Out of scope:** topology discovery, GPU↔GPU converter registration, per-NUMA host allocator, device-guard audit, single-GPU perf regression budget (all Phase 6); P2P direct transfer, adaptive scan partitioning (Phase 7); remote URIs (s3://, http://, hdfs://); file-handle caching upstream; cuCascade upstream API changes.

</domain>

<decisions>
## Implementation Decisions

### Adapter Semantics & Concurrency
- `supports_device_read()` returns **false** — cuDF host-stages + issues `cuda_memcpy_async` on the caller's explicit stream. Matches success criterion 1 and research §Integration model (CUCASCADE-IO.md).
- `device_read_async` wraps the backend call in `std::async(std::launch::async, …)` with a per-datasource thread pool so multiple async reads can queue concurrently to the backend (research Open Q5). Accept single-in-flight serialization per backend instance for v1.1.
- **One `idisk_io_backend` instance per GPU** to start — measure before expanding. If concurrent-scan bottleneck surfaces, file upstream cucascade issue rather than growing the cache.
- **Remote URIs (`s3://`, `http://`, `hdfs://`) are rejected hard** — out of scope per PROJECT.md + REQUIREMENTS; no kvikio fallback anywhere. Adapter constructor validates the path scheme is local filesystem.

### Registry / Lifetime / Ownership
- `io_backend_registry` lives as a **`SiriusContext` member** (not process-global), alongside the per-GPU backend cache. This matches per-GPU backend lifetimes with GPU context lifetime.
- Per-GPU backend cache: **`std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>>` keyed by `device_id`**.
- Initialization happens in **`SiriusContext::initialize()` immediately after `memory_manager_` construction** (src/sirius_context.cpp:163-230). Each backend is created under `rmm::cuda_set_device_raii{device_id}` so streams/pinned buffers are bound to the correct CUDA context (research §Per-GPU backend ownership).
- **Teardown order in `~SiriusContext`: destroy per-GPU backend cache and `io_backend_registry` BEFORE `memory_manager_.reset()`** — mirrors the existing `downgrade_executors_` teardown pattern and avoids the `cudaErrorInvalidResourceHandle` hazard documented in research P7.

### Migration Scope
- **Accept per-read `open`/`close` cost** in `pipeline_io_backend` for v1.1 (research P1). Profile during/after the phase; if it dominates, file upstream cucascade issue — do not work around in Sirius.
- **`rmm::cuda_stream_default` removal scope is limited to files touched by the migration** — HYG-01 covers the explicit documented callsite (`src/op/scan/parquet_scan_task.cpp:468`, `filter_row_groups_with_stats`); HYG-02 sweeps any other `cuda_stream_default` left in any file this phase modifies. No repo-wide sweep.
- **Iceberg delete-file scans**: construct `io_backend_datasource` inline in `read_positional_delete_file` (iceberg_scan_task.cpp:57-58) and `read_equality_delete_file` (iceberg_scan_task.cpp:120-121), selecting the backend by the caller's preferred GPU device_id. Do not share the parquet-scan task's datasource.
- **SF10 regression >30% handling**: file upstream cucascade issue, document in phase summary, do **NOT** block phase completion. Success criterion 4 already specifies this escalation path.

### Testing & Validation
- **Correctness baseline**: run `build/release/test/unittest --test-dir . test/sql/tpch-sirius.test` pre-migration and capture SF1 per-query result hashes. Diff post-migration — bitwise identical required.
- **Multi-GPU validation**: `compute-sanitizer --tool memcheck build/release/test/unittest --test-dir . test/sql/tpch-sirius.test` on a 2+ GPU host reports zero "invalid device" / "context mismatch" errors. A manual 2-GPU scan run logs `cudaGetDevice()` per backend construction to confirm each backend is pinned to a distinct context.
- **Unit tests for the adapter**: new `test/cpp/io/test_io_backend_datasource.cpp` with a mock `idisk_io_backend` verifying `host_read` (both `(offset,size)` and `(offset,size,dst)` overloads), `device_read` / `device_read_async` behavior, and `size()` reporting. Integration path covered by `tpch-sirius.test` + iceberg tests.
- **Preserve Phase 4 deferrals**: the `test_downgrade_executor.cpp:813 TODO(MGPU-06)` cross-GPU converter regression stays in place. Phase 5 does not attempt to fix GPU1→GPU0 converter — that's Phase 7 (MGPU-06 P2P direct). The `[.][multi_gpu_transfer]` + `[.][mem_04_p2p_transfer]` hidden tests remain off-by-default.

</decisions>

<code_context>
## Existing Code Insights

### Adapter pattern is already established
- `src/op/scan/prefetched_data_source.{hpp,cpp}` already implements a custom `cudf::io::datasource` subclass. Follow its construction pattern for `io_backend_datasource`. Key reference: `prefetched_data_source::device_read_async` uses `std::async(std::launch::deferred, …)` — Phase 5 will use `std::launch::async` instead for the adapter to preserve overlap.

### Migration target call sites (from CUCASCADE-IO.md §Sirius migration targets)
- `src/op/scan/parquet_scan_task.cpp:312` — footer pre-read during task planning (once per file, planning-time).
- `src/op/scan/parquet_scan_task.cpp:699` — per-task `_datasource` construction (hot path; route by `preferred_device_id`).
- `src/op/scan/parquet_scan_task.cpp:769` — `prefetched_data_source` fallback construction (pick up the new datasource transparently).
- `src/op/scan/parquet_scan_task.cpp:863` — `read_range_into_allocation` uses `_datasource->host_read_async`; no change needed beyond ensuring the adapter implements it correctly.
- `src/op/scan/parquet_scan_task.cpp:468` — HYG-01: `filter_row_groups_with_stats` currently passes `rmm::cuda_stream_default`; thread explicit stream from scan task global state.
- `src/op/scan/sirius_parquet_metadata_scan_operator.cpp:251` — planning-time metadata scan.
- `src/op/scan/iceberg_scan_task.cpp:57-58, 120-121` — delete-file reads (positional + equality) currently pass `source_info{filepath}` to `cudf::io::read_parquet`.
- `src/data/host_parquet_representation_converters.cpp:82-83` — already wraps `prefetched_data_source`; becomes cucascade-backed transitively once the scan task's datasource is migrated.

### SiriusContext integration points
- `src/sirius_extension.cpp:1053` — `converter_registry::initialize()` is the architectural sibling where `io_backend_registry` hooks belong.
- `src/sirius_context.cpp:163-230` — `SiriusContext::initialize()`. Per-GPU backend cache populated right after `memory_manager_` construction.
- `src/include/data/sirius_converter_registry.hpp` — pattern to mirror for the new io-backend registry wrapper (if adding a thin Sirius wrapper) or direct use of `cucascade::io_backend_registry`.

### Stream discipline
- User rule: never use `rmm::cuda_stream_default`. Every adapter method that takes `rmm::cuda_stream_view` must thread it through to the backend without substitution.
- HYG-02 scope: grep `cuda_stream_default` in every file touched by this phase before sign-off; replace each with an explicit stream.

### Test infrastructure
- Catch2 v2 + `[.]` hidden tags for multi-GPU tests that require N>1. Use `WARN+return` convention on single-GPU hosts.
- SQLLogicTest harness: `build/release/test/unittest --test-dir . test/sql/tpch-sirius.test`.
- Build via `mcp__project-commands__run_command` — never direct `pixi run` / `make`.

### Preserved v1.0 patterns the migration must not disturb
- Push-model task dispatch + `preferred_device_id` on local_state + global_state (local wins) — adapter must look up backend by local_state's preferred device, falling back to global_state.
- NUMA-aware downgrade via cucascade `any_memory_space_in_tier_with_preference` — unchanged; parquet scan sits above the tier system (research §Integration model).

</code_context>

<specifics>
## Specific Ideas

- **Adapter file placement**: `src/io/cucascade_datasource.{hpp,cpp}` per REQUIREMENTS IO-01 wording. Research suggested `src/op/scan/io_backend_datasource.*` — follow REQUIREMENTS wording to be consistent with success criteria text.
- **Pinned host buffer return**: `host_read` returning `std::unique_ptr<buffer>` must back the buffer with pinned host memory allocated from cucascade's host-memory resource so cuDF's `cuda_memcpy_async` stays truly asynchronous (IO-03).
- **Adapter thread pool for `device_read_async`**: per-datasource `std::async(std::launch::async, …)` — do not share across datasources to keep concurrency bounded by the number of active scans.
- **Baseline capture ordering**: Task 1 should checkpoint SF1 result hashes from current (Phase 4) HEAD BEFORE touching any migration code, so the post-migration diff is trivially reproducible.
- **Plan ordering suggestion** (for planner):
  1. Baseline capture + adapter header/skeleton + unit test scaffolding (no behavior change yet).
  2. Adapter implementation + `io_backend_registry` wiring in `SiriusContext::initialize()` + teardown ordering.
  3. Migrate 4 `parquet_scan_task.cpp` + metadata scan callsites; HYG-01 fix for line 468; run SF1 tpch-sirius.
  4. Migrate iceberg delete-file callsites; verify iceberg tests.
  5. Multi-GPU compute-sanitizer validation + SF10 regression measurement + HYG-02 sweep of touched files.

</specifics>

<deferred>
## Deferred Ideas

- **File-handle caching inside `pipeline_io_backend`** — research P1 flags it as a likely bottleneck on FUSE/HDFS/networked FS. Deferred to upstream cucascade issue; accept per-read `open`/`close` cost in v1.1.
- **Multi-worker `pipeline_io_backend` for concurrent in-flight reads** — research P3 / Open Q1. If profiling shows serialization is a bottleneck, file upstream issue. Do not fork cucascade.
- **GDS-aware backend variant** for NVMe-GDS systems (research P2) — v1.2 exploration; goal of v1.1 is multi-GPU safety, not peak bandwidth.
- **S3 / HTTP / HDFS parquet sources** — out of scope; add a new cucascade backend in a future milestone rather than a kvikio escape hatch.
- **Cross-GPU converter return-leg fix** (`GPU1→GPU0` via cucascade converter, test_downgrade_executor.cpp:813 TODO(MGPU-06)) — Phase 7 scope, not this phase.
- **Configurable `PIPELINE_BUF_SIZE`** (research P6) — upstream request; 64 MB × 2 × N_gpus is acceptable on DGX-class memory.

</deferred>
