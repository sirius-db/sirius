# Phase 5: Cucascade-Backed Parquet I/O Migration - Research

**Researched:** 2026-04-20
**Domain:** cuDF parquet I/O + cucascade `idisk_io_backend` adapter integration
**Confidence:** HIGH — all critical claims verified against current working tree (Phase 4 HEAD) and the two pre-existing project-level research docs (CUCASCADE-IO.md, CUDF-DATASOURCE.md)

## Summary

Phase 5 builds a Sirius-owned `cudf::io::datasource` adapter that delegates to per-GPU `cucascade::idisk_io_backend` instances, fully replacing every `cudf::io::datasource::create(path)` + filepath-based `source_info{…}` callsite in `src/`. The architecture is already specified in detail by the two project-level research docs and locked by CONTEXT.md; this phase-level research **(1)** verifies that every file:line cited in CONTEXT.md / CUCASCADE-IO.md / CUDF-DATASOURCE.md still matches the Phase-4-HEAD working tree, **(2)** distills the call-site set into a plan-ready ordered change list, and **(3)** confirms HYG-01/HYG-02 scope by grepping `cuda_stream_default` across every file this phase will touch.

**All 7 migration call-sites verified at the exact line numbers CONTEXT.md cited.** The only `cuda_stream_default` occurrence in any Super Sirius file this phase touches is the documented `parquet_scan_task.cpp:468` (HYG-01). `src/op/sirius_physical_parquet_scan.cpp:92` has a `cuda_stream_default` that is adjacent-but-unrelated (expression translator construction at plan time, not I/O) — **out of HYG-02 scope** because Phase 5 does not modify that file. All other hits are in `src/legacy/` and `src/cuda/` (frozen legacy path, not this phase).

**Primary recommendation:** Follow the CONTEXT.md 5-task ordering (baseline → adapter → parquet scan migration → iceberg migration → multi-GPU validation). The adapter class name is `sirius::io::cucascade_datasource` per REQUIREMENTS IO-01 wording (NOT `io_backend_datasource` from the research draft). Use `supports_device_read() == false` and let cuDF host-stage + `cuda_memcpy_async` through pinned memory on the caller's explicit stream — exactly the architecture in CUCASCADE-IO.md §"Recommended architecture". Per-GPU backend cache lives on `SiriusContext` as `std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>>` keyed by `device_id`, created in `initialize()` immediately after `memory_manager_` construction (line 169) under `rmm::cuda_set_device_raii`, destroyed in `terminate()` **before** `memory_manager_->shutdown()` (mirror of existing `downgrade_executors_` teardown order at sirius_context.cpp:265-268).

## Project Constraints (from CLAUDE.md)

Load-bearing directives the planner MUST honor:

| Directive | Source | Enforcement point |
|---|---|---|
| Never use `rmm::cuda_stream_default` | CLAUDE.md "Streams" + user memory | Adapter header + HYG-01/HYG-02 grep gates |
| Build/test via `mcp__project-commands__run_command` — NOT `pixi run`, NOT `make` directly | CLAUDE.md + user memory | All executor task actions |
| Multi-GPU work targets Super Sirius (`namespace sirius`) only; legacy `namespace duckdb` (gpu_processing) is frozen | CLAUDE.md + PROJECT.md | Adapter placement under `src/io/` or `src/op/scan/`, NOT in `src/legacy/` |
| CUDA 13+, C++20, CUDA std 20 | CLAUDE.md | CMakeLists.txt additions inherit from project defaults |
| Run `/module-context <task>` before implementation tasks that touch GPU operators, memory management, or data I/O | CLAUDE.md | Planner-level — include `/module-context "cucascade datasource parquet migration"` as prelude |
| No `cudf::io::datasource::create` fallback for any reason (defeats milestone) | REQUIREMENTS.md + PROJECT.md Out of Scope | IO-08 grep gate: `grep -rnw 'datasource::create' src/` → 0 hits |
| Pre-commit hooks: clang-format, black, cmake-format, codespell. Run `pre-commit run -a` before commit | CLAUDE.md | Final cleanup step per plan |
| Unit tests use Catch2 v2; `WARN+return` skip idiom (not `SKIP()`); `[.]` prefix + tag for hidden multi-GPU tests | CLAUDE.md + STATE.md | Adapter unit test + multi-GPU validation test |

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Adapter Semantics & Concurrency**
- `supports_device_read()` returns **false** — cuDF host-stages + issues `cuda_memcpy_async` on the caller's explicit stream. Matches success criterion 1 and research §Integration model (CUCASCADE-IO.md).
- `device_read_async` wraps the backend call in `std::async(std::launch::async, …)` with a per-datasource thread pool so multiple async reads can queue concurrently to the backend (research Open Q5). Accept single-in-flight serialization per backend instance for v1.1.
- **One `idisk_io_backend` instance per GPU** to start — measure before expanding. If concurrent-scan bottleneck surfaces, file upstream cucascade issue rather than growing the cache.
- **Remote URIs (`s3://`, `http://`, `hdfs://`) are rejected hard** — out of scope per PROJECT.md + REQUIREMENTS; no kvikio fallback anywhere. Adapter constructor validates the path scheme is local filesystem.

**Registry / Lifetime / Ownership**
- `io_backend_registry` lives as a **`SiriusContext` member** (not process-global), alongside the per-GPU backend cache. This matches per-GPU backend lifetimes with GPU context lifetime.
- Per-GPU backend cache: **`std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>>` keyed by `device_id`**.
- Initialization happens in **`SiriusContext::initialize()` immediately after `memory_manager_` construction** (src/sirius_context.cpp:163-230, specifically after line 169). Each backend is created under `rmm::cuda_set_device_raii{device_id}` so streams/pinned buffers are bound to the correct CUDA context (research §Per-GPU backend ownership).
- **Teardown order in `~SiriusContext`: destroy per-GPU backend cache and `io_backend_registry` BEFORE `memory_manager_.reset()`** — mirrors the existing `downgrade_executors_` teardown pattern and avoids the `cudaErrorInvalidResourceHandle` hazard documented in research P7.

**Migration Scope**
- **Accept per-read `open`/`close` cost** in `pipeline_io_backend` for v1.1 (research P1). Profile during/after the phase; if it dominates, file upstream cucascade issue — do not work around in Sirius.
- **`rmm::cuda_stream_default` removal scope is limited to files touched by the migration** — HYG-01 covers the explicit documented callsite (`src/op/scan/parquet_scan_task.cpp:468`, `filter_row_groups_with_stats`); HYG-02 sweeps any other `cuda_stream_default` left in any file this phase modifies. No repo-wide sweep.
- **Iceberg delete-file scans**: construct `cucascade_datasource` inline in `read_positional_delete_file` (iceberg_scan_task.cpp:57-58) and `read_equality_delete_file` (iceberg_scan_task.cpp:120-121), selecting the backend by the caller's preferred GPU device_id. Do not share the parquet-scan task's datasource.
- **SF10 regression >30% handling**: file upstream cucascade issue, document in phase summary, do **NOT** block phase completion. Success criterion 4 already specifies this escalation path.

**Testing & Validation**
- **Correctness baseline**: run `build/release/test/unittest --test-dir . test/sql/tpch-sirius.test` pre-migration and capture SF1 per-query result hashes. Diff post-migration — bitwise identical required.
- **Multi-GPU validation**: `compute-sanitizer --tool memcheck build/release/test/unittest --test-dir . test/sql/tpch-sirius.test` on a 2+ GPU host reports zero "invalid device" / "context mismatch" errors. A manual 2-GPU scan run logs `cudaGetDevice()` per backend construction to confirm each backend is pinned to a distinct context.
- **Unit tests for the adapter**: new `test/cpp/io/test_cucascade_datasource.cpp` with a mock `idisk_io_backend` verifying `host_read` (both `(offset,size)` and `(offset,size,dst)` overloads), `device_read` / `device_read_async` behavior, and `size()` reporting. Integration path covered by `tpch-sirius.test` + iceberg tests.
- **Preserve Phase 4 deferrals**: the `test_downgrade_executor.cpp:813 TODO(MGPU-06)` cross-GPU converter regression stays in place. Phase 5 does not attempt to fix GPU1→GPU0 converter — that's Phase 7 (MGPU-06 P2P direct). The `[.][multi_gpu_transfer]` + `[.][mem_04_p2p_transfer]` hidden tests remain off-by-default.

### Claude's Discretion

- Exact class-layout shape of `cucascade_datasource` (fields, initializer list, constructor validation) — CONTEXT locks semantics but not the private-member layout. See `## Code Examples` for the recommended shape.
- Whether to introduce a small `sirius::io::io_backend_registry` **wrapper** class around the raw `cucascade::io_backend_registry` or use `cucascade::io_backend_registry` directly as a SiriusContext member. Recommendation: use `cucascade::io_backend_registry` directly (no wrapper) for minimal surface area; the per-GPU `std::unordered_map<int, shared_ptr<…>>` cache is separate and gives us the Sirius-specific lookup primitive.
- Exact `/module-context` prelude content for implementation tasks.
- Whether `cucascade_datasource` gets a factory helper (`make_cucascade_datasource(path, device_id)`) or callers construct it directly via constructor. Recommendation: factory helper on `SiriusContext` (`get_io_backend_for(device_id)` + `cucascade_datasource(backend, path, file_size)` ctor) — keeps per-GPU selection concern out of callsites.
- Thread-pool implementation detail for `device_read_async` — `std::async(std::launch::async)` is locked, but pool sizing and reuse strategy are discretionary. Recommendation: fresh `std::async` per call for v1.1 simplicity; measure before introducing a bounded thread pool.

### Deferred Ideas (OUT OF SCOPE)

- **File-handle caching inside `pipeline_io_backend`** — research P1. Deferred to upstream cucascade issue; accept per-read `open`/`close` cost in v1.1.
- **Multi-worker `pipeline_io_backend` for concurrent in-flight reads** — research P3 / Open Q1. File upstream if profiling shows serialization bottleneck.
- **GDS-aware backend variant** for NVMe-GDS systems (research P2) — v1.2 exploration.
- **S3 / HTTP / HDFS parquet sources** — add a new cucascade backend in a future milestone, not a kvikio escape hatch.
- **Cross-GPU converter return-leg fix** (`GPU1→GPU0` via cucascade converter, test_downgrade_executor.cpp:813 TODO(MGPU-06)) — Phase 7 scope.
- **Configurable `PIPELINE_BUF_SIZE`** (research P6) — upstream request; 64 MB × 2 × N_gpus acceptable on DGX-class memory.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| IO-01 | `sirius::io::cucascade_datasource` subclass of `cudf::io::datasource` ships in `src/io/`, backed by cuCascade's `idisk_io_backend` via `io_backend_registry` factory | Adapter class spec + file placement verified (§Verified File Layout, §Code Examples) |
| IO-02 | `cucascade_datasource` declares `supports_device_read() == false` so cuDF host-stages reads and issues memcpys on the caller's explicit stream (no GDS, no cuFile, no kvikio) | CUDF-DATASOURCE.md §3 confirms cuDF host-fallback path; CONTEXT.md locks this |
| IO-03 | `host_read` returns pinned host memory allocated from cucascade's host-memory resource so cuDF's `cuda_memcpy_async` stays truly asynchronous | CUCASCADE-IO.md Pitfall 3; allocator source: `cucascade::memory::fixed_size_host_memory_resource` already in use at `parquet_scan_task.cpp:652-658` |
| IO-04 | Per-GPU `idisk_io_backend` instances cached in `SiriusContext`, created once per device under `rmm::cuda_set_device_raii` so each instance owns streams/pinned buffers in its GPU's context | `SiriusContext::initialize()` entry point verified at sirius_context.cpp:163; insert after line 169 (memory_manager_) and before line 193 (data_repository_manager_). Teardown before line 290 (memory_manager_.reset()) |
| IO-05 | `cudf::io::datasource::create(filepath)` removed from `src/op/scan/parquet_scan_task.cpp:312`, `:699` and `src/op/scan/sirius_parquet_metadata_scan_operator.cpp:251` | Verified: all 3 call-sites at exact lines CONTEXT.md cites. Only 3 such hits in `src/` |
| IO-06 | Iceberg delete-file reads at `src/op/scan/iceberg_scan_task.cpp:57-58` and `:120-121` pass `source_info{ds.get()}` with a cucascade-backed datasource instead of `source_info{filepath}` | Verified: both `source_info{delete_file_path}` calls exist at cited lines; no other filepath-based `source_info{…}` in `src/` |
| IO-07 | `prefetched_data_source` fallback datasource is cucascade-backed at `src/data/host_parquet_representation_converters.cpp:82-83` and at the construction site `src/op/scan/parquet_scan_task.cpp:769` | Transitive: migrating IO-05 line 699 makes `_datasource` cucascade-backed; it flows as `fallback_datasource` into `host_parquet_representation` (line 769) and onward to `prefetched_data_source` (converters.cpp:83). No direct edit needed — verify by inspection post-migration |
| IO-08 | `grep -rnw 'datasource::create' src/` returns zero hits | Current count: 3 hits (all migrated by IO-05). Grep gate is the acceptance signal |
| IO-09 | TPC-H SF1 all queries produce results identical to pre-migration baseline (correctness) | Baseline captured BEFORE any code change (Task 1 in plan ordering). Diff post-migration bitwise |
| IO-10 | TPC-H SF10 parquet scan wall-clock regression vs kvikio-compat baseline ≤ 30%; any larger delta filed as cuCascade upstream issue and documented in the phase summary | CONTEXT locks escalation path — file upstream issue, do NOT block phase |
| IO-11 | Parquet scan validated on multi-GPU hardware — one `idisk_io_backend` per GPU, cross-GPU reads work, no CUDA-context leak between devices (verified with compute-sanitizer or `cudaGetDevice` logging) | `compute-sanitizer --tool memcheck` + manual `cudaGetDevice()` logging per backend construction. Hidden `[.][multi_gpu]` tag if added |
| HYG-01 | `rmm::cuda_stream_default` removed from `src/op/scan/parquet_scan_task.cpp:468` — explicit stream plumbed from task global state (user rule: never use `cuda_stream_default`) | Verified: line 468 `row_group_indices = readers[file_idx]->filter_row_groups_with_stats(row_group_indices, _reader_options, rmm::cuda_stream_default);`. Called from `initialize_from_files()`; needs an explicit stream plumbed through. The sibling `sirius_parquet_metadata_scan_operator.cpp:285` already uses explicit `stream` (precedent for how to thread it) |
| HYG-02 | Any other `rmm::cuda_stream_default` callsite introduced or left behind by the v1.0 re-integration is replaced with an explicit stream before phase sign-off | Grep across files this phase will touch (see §HYG-02 Scope Audit): **only HYG-01 hit** is in-scope. `src/op/sirius_physical_parquet_scan.cpp:92` is adjacent but NOT touched by this phase — out of HYG-02 scope |

## Standard Stack

Phase 5 adds no new dependencies. All library versions are pinned by Phase 4's cucascade bump (f47de0b) + cuDF 26.04 (pixi.toml).

### Core

| Library | Version | Purpose | Why Standard |
|---|---|---|---|
| cucascade | f47de0b (origin/main) | `idisk_io_backend`, `io_backend_registry`, `register_builtin_io_backends`, `pipeline_io_backend` | Already pinned by Phase 4 (BUMP-01). PR #96 introduced the API Phase 5 consumes |
| libcudf | 26.04.* | `cudf::io::datasource` interface (custom subclass), `source_info{datasource*}`, `read_parquet`, `parquet::fetch_footer_to_host`, `hybrid_scan_reader` | Already pinned (pixi.toml). `CUDF_VERSION_NUM >= 2604` gates already in code |
| RMM | via cudf | `rmm::cuda_stream_view`, `rmm::cuda_set_device_raii`, `rmm::cuda_device_id` | Already in use |
| Catch2 | v2.13.x | Unit-test framework with `[.]` hidden-test convention and `WARN+return` skip idiom | Project-wide standard, STATE.md lock |

### Supporting

| Library | Version | Purpose | When to Use |
|---|---|---|---|
| `cucascade::memory::fixed_size_host_memory_resource` | bundled | Pinned host allocations for `host_read` buffer returns (IO-03) | Inside `cucascade_datasource::host_read(offset, size)` to back the returned `datasource::buffer` with pinned memory |
| `std::async(std::launch::async, …)` | STL | `device_read_async` implementation so concurrent reads queue into the backend | Inside adapter `device_read_async` only. Not `std::launch::deferred` (prefetched_data_source does deferred — we explicitly differ) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|---|---|---|
| Custom `cudf::io::datasource` subclass | `KVIKIO_COMPAT_MODE` env var | **NOT VIABLE** — CUDF-DATASOURCE.md §Option 2 proves the env var only toggles GDS-vs-POSIX; kvikio::FileHandle still holds CUDA context. Rejected |
| Per-GPU backend cache on `SiriusContext` | Process-global singleton | CONTEXT locks per-SiriusContext — matches GPU context lifetime, avoids dangling after extension unload/reload (CUCASCADE-IO.md P7) |
| `io_backend_datasource` naming (research-doc draft) | `cucascade_datasource` (REQUIREMENTS IO-01 wording) | **REQUIREMENTS wording wins** — success criterion text is load-bearing |
| `src/op/scan/` placement | `src/io/` placement | **REQUIREMENTS IO-01 wording wins**: "`sirius::io::cucascade_datasource` … ships in `src/io/`". Fresh directory; planner creates it |

**No package installs or version bumps required.**

## Architecture Patterns

### Recommended Project Structure

New files (4 total):

```
src/
├── io/                                      # NEW directory
│   ├── cucascade_datasource.hpp             # IO-01 — cudf::io::datasource adapter declaration
│   └── cucascade_datasource.cpp             # IO-01 — adapter implementation
├── include/
│   └── io/
│       └── cucascade_datasource.hpp         # OR: header in src/include/io/ if project convention
│                                            # prefers include/ mirror (check existing patterns)
```

**Decision on header placement:** Project uses `src/include/` as the canonical include tree (see `src/include/op/scan/prefetched_data_source.hpp` + impl at `src/op/scan/prefetched_data_source.cpp`). Follow that pattern: put the adapter header at `src/include/io/cucascade_datasource.hpp` and impl at `src/io/cucascade_datasource.cpp`. CMakeLists.txt additions go in the `SIRIUS_SRC_FILES` list near line 174-175 (where `prefetched_data_source.cpp` is already listed — check `parquet_scan_task.cpp` neighbors).

Modified files (7):

```
src/
├── sirius_context.cpp                       # io_backend_registry member + per-GPU cache
├── include/sirius_context.hpp               # header declarations for new members
├── sirius_extension.cpp                     # (decision deferred — may not need changes if
│                                            #  registry is a SiriusContext member, not global)
├── op/scan/
│   ├── parquet_scan_task.cpp                # 3 sites: :312 (IO-05), :468 (HYG-01), :699 (IO-05)
│   ├── sirius_parquet_metadata_scan_operator.cpp   # :251 (IO-05)
│   └── iceberg_scan_task.cpp                # :57-58, :120-121 (IO-06)
└── include/op/scan/parquet_scan_task.hpp    # may need plumbing for explicit stream (HYG-01)

test/cpp/
├── io/                                      # NEW test directory (mirror src/io/)
│   └── test_cucascade_datasource.cpp        # Unit test with mock idisk_io_backend
CMakeLists.txt                               # Register new src + test files
```

### Pattern 1: Adapter class — mirror `prefetched_data_source`

**What:** `cucascade_datasource` subclasses `cudf::io::datasource`. Construction takes a `shared_ptr<cucascade::idisk_io_backend>`, a filesystem path, and a cached file size.
**When to use:** Every place that today calls `cudf::io::datasource::create(path)` or `source_info{path}`.
**Precedent:** `src/op/scan/prefetched_data_source.{hpp,cpp}` already implements a custom `cudf::io::datasource`. Follow its file layout, namespace placement (`sirius::op::scan`), and method shape. Adapter difference: `cucascade_datasource` reports `supports_device_read() == false` (prefetched_data_source returns `true`) so cuDF takes the host-staging path.

### Pattern 2: Per-GPU backend cache with context-pinning

**What:** `SiriusContext` holds `cucascade::io_backend_registry io_backend_registry_` + `std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> gpu_io_backends_`. Initialized in `initialize()` by iterating GPU memory spaces and creating a backend under `rmm::cuda_set_device_raii{device_id}`. Destroyed in `terminate()` before `memory_manager_->shutdown()`.
**When to use:** Anywhere the adapter needs a backend — accessed via `SiriusContext::get_io_backend_for(int device_id)`.
**Why:** `pipeline_io_backend` constructor calls `cudaMallocHost` + `cudaStreamCreate` + `cudaEventCreateWithFlags`, all bound to the CUDA context current at that moment (CUCASCADE-IO.md §"Stream ownership"). One instance per GPU is required.

### Pattern 3: Stream-threaded row-group pruning (HYG-01 fix)

**What:** `parquet_scan_task_global_state::initialize_from_files()` currently passes `rmm::cuda_stream_default` to `filter_row_groups_with_stats` on line 468. Replace with an explicit stream threaded from a scan-task-accessible source.
**When to use:** HYG-01 remediation.
**Reference implementation:** `src/op/scan/sirius_parquet_metadata_scan_operator.cpp:285` already passes `stream` (threaded from the operator's `execute(rmm::cuda_stream_view stream)` signature). The same shape works for `parquet_scan_task`: thread a `rmm::cuda_stream_view` into `initialize_from_files()` (called from the global-state constructor). Two viable sources:
  1. **Create a throwaway `rmm::cuda_stream` inside `initialize_from_files()`** — cheap, local-scope, no plumbing. Recommended for Phase 5 because `initialize_from_files()` is called once at scan-plan time and the stream is only used for a single `filter_row_groups_with_stats` call.
  2. **Plumb from a caller's context** — more invasive (`parquet_scan_task_global_state` constructor signature change). Reject for Phase 5; revisit if operator-level streams become canonical.

### Anti-Patterns to Avoid

- **Do NOT extend `prefetched_data_source` or `cucascade_datasource` to handle remote schemes.** Remote URIs (`s3://`, `http://`, `hdfs://`) are out of scope (PROJECT.md). Adapter constructor must validate local-filesystem scheme and throw on remote.
- **Do NOT share one `pipeline_io_backend` instance across GPUs.** Each instance is pinned to the CUDA context current at construction. One-per-GPU is locked by CONTEXT + research.
- **Do NOT wrap `backend.read()` in `std::async(std::launch::deferred, …)` for `device_read_async`.** `prefetched_data_source` uses `deferred` because it has a CUDA event to wait on; `cucascade_datasource`'s backend call is genuinely blocking at issue time, so `deferred` would collapse concurrency. Use `std::launch::async` (CONTEXT lock).
- **Do NOT register `io_backend_registry` as a process-global singleton alongside `converter_registry`.** CONTEXT locks it as a `SiriusContext` member. `converter_registry` is global because converters are stateless; backends hold CUDA resources and must follow context lifetime.
- **Do NOT use `rmm::cuda_stream_default` in any new code added by this phase.** HYG-02 is explicit.
- **Do NOT delete or modify `prefetched_data_source.{hpp,cpp}`.** It remains the caching layer above the adapter (fallback chain: cached ranges → `cucascade_datasource` via `fallback_`). CUCASCADE-IO.md Open Q6 traced this path and confirmed no changes needed.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---|---|---|---|
| Disk byte-range read primitive | Custom `pread`/`fopen` wrapper | `cucascade::idisk_io_backend` via `io_backend_registry.create_default_backend()` | Backend handles pinned host staging, D2H overlap, O_DIRECT option, per-backend stream ownership |
| Pinned host memory for `host_read` return buffer | `cudaMallocHost` + manual `cudaFreeHost` | `cucascade::memory::fixed_size_host_memory_resource` already owned by `SiriusContext::memory_manager_` | Pool reuse, NUMA-awareness, accounting against HOST tier budget |
| Async `device_read` fanout | Dedicated thread pool, lock-free queue | `std::async(std::launch::async, …)` per call | Simple; per-datasource concurrency bound is governed by backend (serialized internally by `io_worker`). Measure before optimizing (research Open Q1) |
| Custom remote-URL handling | URL parser + HTTP client integration | Hard reject in adapter constructor — out of scope | Re-introduces kvikio-like dependency, defeats milestone |
| File-handle caching | LRU of open `fd`s around backend | **Nothing** — accept per-read `open`/`close` cost | Deferred to upstream cucascade (CONTEXT + research P1) |
| Cross-GPU forwarding in adapter | Adapter detects context mismatch and redispatches | **Nothing** — callers (scan tasks) select the right backend by `preferred_device_id` before construction | Keeps adapter simple; device selection is a task-scheduler concern |

**Key insight:** Every concern in this phase that could become a custom Sirius primitive already has a cucascade or cuDF primitive — the phase is strictly about adapting the seam between cuDF's `datasource` interface and cucascade's `idisk_io_backend` interface. The two project-level research docs spent most of their time proving this — we should resist the urge to add "helper" abstractions.

## Runtime State Inventory

Phase 5 is a **pure code refactor** — no stored data, no service config, no OS-level state, no secrets, no build artifacts carry the changed strings. This table confirms that by category:

| Category | Items Found | Action Required |
|---|---|---|
| Stored data | None — no database/datastore keys, collection names, or user_ids reference `datasource::create` or the to-be-replaced call pattern. Verified by grep: nothing in `.planning/`, `test/sql/`, or config files references the renamed APIs | none |
| Live service config | None — no external services (n8n, Datadog, Tailscale, etc.) observe parquet I/O call patterns | none |
| OS-registered state | None — no systemd units, cron jobs, task-scheduler entries, or pm2 processes reference the changed code paths | none |
| Secrets / env vars | None — `KVIKIO_COMPAT_MODE`, `KVIKIO_NTHREADS`, `LIBCUDF_MMAP_ENABLED` remain read by cuDF for any residual kvikio use outside our adapter path, but **this phase's change removes kvikio from Sirius's code path** regardless of those env vars. No secret key renames. No `.env` changes | none |
| Build artifacts / installed packages | None — the Sirius extension rebuilds fully from source; no egg-info, wheel, or installed-package metadata references the changed APIs. `build/` directory is ignored and regenerated on each build | none — standard `rm -rf build && make` if anything feels stale |

**Canonical answer:** After the file edits land, no runtime system has the old `datasource::create(path)` pattern cached, stored, or registered. Rebuild + run tests is sufficient.

## Verified File Layout

All line numbers verified against `feature/single-node-multi-gpu2` @ Phase 4 HEAD (commit 13e4322).

### Migration call sites — exact line numbers

| Req | File | Line | Current code | After migration |
|---|---|---|---|---|
| IO-05 | `src/op/scan/parquet_scan_task.cpp` | 312 | `auto datasource = cudf::io::datasource::create(file_path);` | Construct `cucascade_datasource` via `SiriusContext::get_io_backend_for(…)`. Planning-time call — runs once per file per scan. Use any GPU's backend (caller-supplied default), or introduce a scan-plan-level device preference |
| IO-05 | `src/op/scan/parquet_scan_task.cpp` | 699 | `_datasource = cudf::io::datasource::create(g_state.get_file_path(l_state.get_file_idx()));` | Construct `cucascade_datasource` using the backend for the task's `preferred_device_id`. Hot path — runs per task. Use `gpu_pipeline_task::get_preferred_device_id()` (see §Preferred-device plumbing) |
| IO-05 | `src/op/scan/sirius_parquet_metadata_scan_operator.cpp` | 251 | `result->datasources.push_back(cudf::io::datasource::create(file_path));` | Construct `cucascade_datasource` per file. Planning-time. Datasource flows into `parquet_scan_data->datasource` (see `sirius_gpu_parquet_scan_operator.cpp:151, 167`) which eventually lands on a task's `_datasource` — type is `std::shared_ptr<cudf::io::datasource>`, polymorphism already in place |
| IO-06 | `src/op/scan/iceberg_scan_task.cpp` | 57 | `cudf::io::parquet_reader_options::builder(cudf::io::source_info{delete_file_path}).build();` | Build `cucascade_datasource ds{…}` first; pass `cudf::io::source_info{&ds}` (raw pointer — `source_info` takes non-owning `datasource*`, `ds` must outlive `read_parquet`). Adapter stays on stack until `result.tbl` is materialized |
| IO-06 | `src/op/scan/iceberg_scan_task.cpp` | 120 | same shape as line 57 | same treatment |
| HYG-01 | `src/op/scan/parquet_scan_task.cpp` | 468 | `row_group_indices = readers[file_idx]->filter_row_groups_with_stats(row_group_indices, _reader_options, rmm::cuda_stream_default);` | Thread a `rmm::cuda_stream` local to `initialize_from_files()` (or plumb from caller) and pass instead of `rmm::cuda_stream_default` |

### Transitive / confirmation call sites (no direct edit needed)

| Req | File | Line | Note |
|---|---|---|---|
| IO-07 | `src/op/scan/parquet_scan_task.cpp` | 769 | `_datasource` is passed as `fallback_datasource` into `host_parquet_representation`. Once line 699 is migrated, this line automatically carries the cucascade-backed type. **No edit** |
| IO-07 | `src/data/host_parquet_representation_converters.cpp` | 82-83 | `prefetched_data_source` constructed with `host_src.get_fallback_datasource()`. Transitively cucascade-backed once upstream migrates. **No edit** |
| IO-07 | `src/op/scan/sirius_gpu_parquet_scan_operator.cpp` | 167, 171 | Reads `scan_data->datasource` and passes to `opts.set_source(source_info{datasource.get()})`. Already polymorphic. Transitively cucascade-backed once line 251 is migrated. **No edit** |
| IO-05 context | `src/op/scan/parquet_scan_task.cpp` | 863 | `_datasource->host_read_async(current_offset, bytes_to_read, buffer_ptr)` in `read_range_into_allocation`. Polymorphic — no edit. Adapter must implement `host_read_async` correctly (default base-class `std::async(deferred)` wrapper is OK; CUCASCADE-IO.md Open Q5 recommends async wrapper for overlap — **CONTEXT discretion**; recommend wrap with `std::launch::async`) |

### New files

| Path | Purpose |
|---|---|
| `src/include/io/cucascade_datasource.hpp` | Adapter class declaration |
| `src/io/cucascade_datasource.cpp` | Adapter class implementation |
| `test/cpp/io/test_cucascade_datasource.cpp` | Unit test with mock `idisk_io_backend` |

### SiriusContext integration points

| File | Line | Change |
|---|---|---|
| `src/include/sirius_context.hpp` | ~174 (after `downgrade_executors_`) | Add `cucascade::io_backend_registry io_backend_registry_;` + `std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> gpu_io_backends_;` + public accessor `get_io_backend_for(int device_id)` |
| `src/sirius_context.cpp` | 169 → 193 | Between memory_manager_ construction (line 169) and data_repository_manager_ construction (line 193): call `register_builtin_io_backends(io_backend_registry_)`, iterate GPU memory spaces via `memory_manager_->get_memory_spaces_for_tier(Tier::GPU)`, for each space call `rmm::cuda_set_device_raii guard{space->get_device_id()}` and `gpu_io_backends_[space->get_device_id()] = io_backend_registry_.create_default_backend()` |
| `src/sirius_context.cpp` | 265-290 | `terminate()`: destroy `gpu_io_backends_` (clear the map) and clear `io_backend_registry_` BEFORE `memory_manager_->shutdown()`. Each backend's destructor needs its owning CUDA context alive, which is true as long as `memory_manager_` is alive. Mirror `downgrade_executors_` ordering at lines 265-268 |

### Preferred-device plumbing (line 699 hot path)

The scan task already has access to its preferred device via `gpu_pipeline_task::get_preferred_device_id()` (inherited from base; verified at `src/include/pipeline/gpu_pipeline_task.hpp:87-191`). That method checks local_state first, then global_state — matches Phase 4 locked semantics. Usage at line 699:

```cpp
// Before:
_datasource = cudf::io::datasource::create(g_state.get_file_path(l_state.get_file_idx()));

// After:
auto const device_id = get_preferred_device_id().value_or(/* scan-plan default device */);
auto backend = SiriusContext::instance().get_io_backend_for(device_id);
_datasource = std::make_shared<sirius::io::cucascade_datasource>(
  std::move(backend), g_state.get_file_path(l_state.get_file_idx()), g_state.get_file_size(l_state.get_file_idx()));
```

Open question — how `parquet_scan_task` accesses `SiriusContext`: the scan task currently doesn't hold a context pointer. Options:
1. **Inject backend directly**: look up the backend at task-creation time (in `task_creator`) and pass it into `parquet_scan_task` via the local_state. Preserves current separation of concerns.
2. **Global accessor**: add a `SiriusContext::current()` or equivalent process-singleton pointer. CLAUDE.md / STATE.md don't explicitly forbid this, but the existing pattern (extension callback stores `context_` on `ClientContext::registered_state`) argues against global statistics.
3. **Plumb via task_creator**: `task_creator_` in SiriusContext already exists; `task_creator` could accept a reference to the per-GPU backend cache and seed each parquet_scan_task's local_state with its resolved backend.

**Recommendation for planner:** Option 3 — `task_creator` seeds local_state with the resolved backend, keeping SiriusContext reference out of the scan task. This also matches how `preferred_device_id` is resolved at task-creation time (Phase 4 locked pattern).

### HYG-02 Scope Audit

Files this phase modifies (from §Verified File Layout):
- `src/include/io/cucascade_datasource.hpp` — NEW
- `src/io/cucascade_datasource.cpp` — NEW
- `src/include/sirius_context.hpp` — MODIFIED
- `src/sirius_context.cpp` — MODIFIED
- `src/op/scan/parquet_scan_task.cpp` — MODIFIED (IO-05 + HYG-01)
- `src/op/scan/sirius_parquet_metadata_scan_operator.cpp` — MODIFIED (IO-05)
- `src/op/scan/iceberg_scan_task.cpp` — MODIFIED (IO-06)
- `test/cpp/io/test_cucascade_datasource.cpp` — NEW
- `CMakeLists.txt` — MODIFIED (add new src + test files)
- `src/include/op/scan/parquet_scan_task.hpp` — POSSIBLY MODIFIED (stream plumbing for HYG-01, or a plumbing point for injected backend)

`grep cuda_stream_default` on this exact file set (performed during research):

| File | `cuda_stream_default` hits |
|---|---|
| `src/op/scan/parquet_scan_task.cpp` | 1 hit at line 468 — **HYG-01** |
| `src/op/scan/sirius_parquet_metadata_scan_operator.cpp` | 0 hits (line 285 uses explicit `stream`) |
| `src/op/scan/iceberg_scan_task.cpp` | 0 hits |
| `src/sirius_context.cpp` | 0 hits |
| `src/op/scan/prefetched_data_source.cpp` (not touched but referenced) | 0 hits |
| `src/data/host_parquet_representation_converters.cpp` (not touched but referenced) | 0 hits |

**HYG-02 verdict: no additional hits inside the modified file set.** Complete HYG-02 by re-running the grep on the final diff before phase sign-off. Any hit that the new code accidentally introduces must be fixed on that pass.

**Adjacent but OUT OF SCOPE** (documented here so the planner doesn't stumble into them):
- `src/op/sirius_physical_parquet_scan.cpp:92` — `gpu_expression_translator translator(rmm::cuda_stream_default, …)` at plan-builder time. NOT an I/O concern; not touched by this phase. Leave for a future hygiene sweep.
- `src/legacy/` + `src/cuda/` — 20+ hits. Legacy `namespace duckdb` (gpu_processing) path, frozen per CLAUDE.md.

## Common Pitfalls

### Pitfall 1: `pipeline_io_backend` is context-bound at construction (CUCASCADE-IO.md P7)

**What goes wrong:** `cudaMallocHost`, `cudaStreamCreate`, `cudaEventCreateWithFlags` all bind resources to whatever CUDA context is current when `create_default_backend()` runs. If that's the wrong context (e.g., GPU 0's when we wanted GPU 2's), the backend works but all its streams and pinned buffers live on GPU 0 — defeats per-GPU isolation.
**Why it happens:** Silently — no error until teardown or cross-device use.
**How to avoid:** Wrap every `create_default_backend()` call in `rmm::cuda_set_device_raii{device_id}`. Do this per GPU in `SiriusContext::initialize()` and also document that callers of `get_io_backend_for(device_id)` need to be on the matching CUDA context before using the returned backend (or the backend sets it internally — verify once via `cudaGetDevice()` logging in the adapter's first real use).
**Warning signs:** `cudaErrorInvalidResourceHandle` during extension unload. compute-sanitizer memcheck "invalid device" / "context mismatch" errors.

### Pitfall 2: `host_read` buffer must be pinned or `cuda_memcpy_async` silently serializes (CUDF-DATASOURCE.md Pitfall 3)

**What goes wrong:** cuDF's fallback path (when `supports_device_read() == false`) calls `cudf::detail::cuda_memcpy_async(device_span, host_span, stream)` with the host buffer returned by our `host_read`. If that host buffer is a plain `std::vector<uint8_t>` (pageable), `cuda_memcpy_async` silently falls back to synchronous — defeating the async benefit we designed the adapter around.
**Why it happens:** `std::vector` is pageable; cuDF doesn't validate.
**How to avoid:** In `cucascade_datasource::host_read(offset, size)`, allocate the returned buffer from `cucascade::memory::fixed_size_host_memory_resource` (pinned — already used at `parquet_scan_task.cpp:652-658`). Wrap in `datasource::buffer::create(pinned_owner)` so ownership releases the pinned block on destruction.
**Warning signs:** SF10 regression >> 30%; nsys profile shows synchronous H2D where we expected overlap.

### Pitfall 3: `device_read_async` with `std::launch::deferred` collapses concurrency

**What goes wrong:** Copying `prefetched_data_source`'s shape (`std::async(std::launch::deferred, …)`) and wrapping `backend.read()` in it. cuDF calls `future.get()` later; deferred execution runs on the caller's thread — serializes all reads.
**Why it happens:** `prefetched_data_source` uses `deferred` because its device-copy is already issued on the stream (event-based sync). Our adapter's backend call is blocking at issue time, so the future must run on a worker thread.
**How to avoid:** Use `std::async(std::launch::async, …)` per CONTEXT. Understand this differs from `prefetched_data_source`'s pattern intentionally.
**Warning signs:** nsys shows I/O-CPU-I/O-CPU serialization instead of overlap.

### Pitfall 4: `source_info{datasource_ptr}` takes a non-owning raw pointer

**What goes wrong:** Constructing `cudf::io::source_info{ds.get()}` where `ds` goes out of scope before `read_parquet` completes → dangling pointer → crash or UB inside cuDF.
**Why it happens:** `source_info` documented in cuDF header as non-owning; the call is synchronous so in practice `ds` just needs to outlive the synchronous `read_parquet` call. Easy in iceberg (local variable), easy in parquet_scan_task (`_datasource` is a member with task lifetime). Still a code-review trap.
**How to avoid:** Keep the `cucascade_datasource` in a named local (for iceberg) or a member (for parquet_scan_task). Never construct as a temporary inside the `source_info{…}` call.
**Warning signs:** Segfault during `cudf::io::read_parquet`; use-after-free reported by ASan/valgrind.

### Pitfall 5: Iceberg delete-file scans create two independent datasource instances

**What goes wrong:** Two call sites (iceberg_scan_task.cpp:57 + :120) each construct their own `cucascade_datasource`. If we share a single static-lifetime adapter, we collide on `_file_size` (per-file member).
**Why it happens:** Natural temptation to "cache" the adapter, especially if profiling shows delete-file opens are frequent.
**How to avoid:** Construct `cucascade_datasource` fresh per call (on the stack). Each delete file is a different path; caching would require a `path → datasource` map. Deferred as an optimization.
**Warning signs:** Wrong file size reported; cuDF reads garbage at the end of the "wrong" file.

### Pitfall 6: Planning-time `cucascade_datasource` (line 312, 251) doesn't have a natural `preferred_device_id`

**What goes wrong:** These call sites run at scan-plan time, before tasks exist. The `preferred_device_id` isn't known yet. Picking GPU 0 arbitrarily wastes one read on the wrong GPU; picking "any available" is fine but must be deterministic.
**Why it happens:** Planning vs. execution asymmetry. Research §Integration model discusses this; CONTEXT leaves it to Claude's discretion.
**How to avoid:** Planning-time call sites use `gpu_io_backends_[/*first available device_id*/]`. The reads here are (a) footer pre-read, small, (b) metadata only — they don't populate row-group allocations on GPU, so context mismatch is irrelevant. Document the choice in a comment.
**Warning signs:** None at runtime; this is a correctness-neutral choice. Comment intent so future maintainers don't "optimize" it away.

### Pitfall 7: Phase-4 hidden-test failures might appear to regress

**What goes wrong:** Plan 04-05 left `[.][multi_gpu_transfer]` + `[.][mem_04_p2p_transfer]` FAILING on GPU1→GPU0 return leg, with explicit deferral to Phase 6/7. Phase 5's multi-GPU compute-sanitizer run may observe these failures and be mistaken for a Phase 5 regression.
**Why it happens:** Two adjacent multi-GPU failure modes (Phase-4-deferred vs. Phase-5-introduced) could be confused.
**How to avoid:** Re-run the two hidden tests BEFORE starting Phase-5 code changes to establish the Phase-4 fail baseline. Compare exit codes and error shapes post-migration; differences = Phase 5 regression.
**Warning signs:** New "invalid device" errors in traces that didn't appear in the Phase 4 baseline.

### Pitfall 8: TPC-H Q4 parquet flake (pre-existing, root-cause scoped to Phase 5)

**What goes wrong:** Q4 fails intermittently in TPC-H tests (Plans 04-01/02/05 each observed it; retry green). STATE.md scopes root-cause to Phase 5 because the parquet I/O migration touches the responsible code paths.
**Why it happens:** Unknown — pre-existing, not blamed on any commit.
**How to avoid:** Phase 5 does NOT promise a fix, but the planner should add a Task to investigate whether the migration incidentally stabilizes Q4 (or whether a new flake emerges). If the root cause is e.g. a race in `host_read_async` under contention, this phase's adapter boundary may surface a real fix.
**Warning signs:** Q4 flake recurs post-migration → record in phase summary, escalate root-cause investigation.

## Code Examples

### Adapter header skeleton

```cpp
// Source: pattern cloned from src/include/op/scan/prefetched_data_source.hpp,
// adapter shape from CUCASCADE-IO.md §"Recommended architecture" + CONTEXT locks.
// File: src/include/io/cucascade_datasource.hpp

#pragma once

#include <cucascade/data/disk_io_backend.hpp>

#include <cudf/io/datasource.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <filesystem>
#include <future>
#include <memory>

namespace sirius::io {

/// cudf::io::datasource adapter that delegates byte-range reads to a
/// cucascade::idisk_io_backend. Supports host_read only; reports
/// supports_device_read() == false so cuDF host-stages through pinned
/// memory and issues cuda_memcpy_async on the caller's explicit stream.
///
/// One instance per (backend, path). Planner must ensure the backend's
/// CUDA context is the one appropriate for the consuming GPU task.
class cucascade_datasource : public cudf::io::datasource {
 public:
  cucascade_datasource(std::shared_ptr<cucascade::idisk_io_backend> backend,
                       std::filesystem::path path,
                       std::size_t file_size);

  ~cucascade_datasource() override;

  // Host reads — backing buffer allocated from cucascade's
  // fixed_size_host_memory_resource so cuDF's cuda_memcpy_async stays async.
  [[nodiscard]] std::unique_ptr<cudf::io::datasource::buffer> host_read(size_t offset,
                                                                         size_t size) override;
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  // Device reads disabled — cuDF takes the host + cuda_memcpy_async path.
  [[nodiscard]] bool supports_device_read() const override { return false; }
  [[nodiscard]] bool is_device_read_preferred(size_t /*size*/) const override { return false; }

  // host_read_async: fresh std::async(std::launch::async, …) per call so
  // multiple concurrent reads can queue into the backend concurrently.
  std::future<std::unique_ptr<cudf::io::datasource::buffer>> host_read_async(size_t offset,
                                                                              size_t size) override;
  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override;

  [[nodiscard]] size_t size() const override { return _file_size; }

 private:
  std::shared_ptr<cucascade::idisk_io_backend> _backend;
  std::filesystem::path _path;
  std::size_t _file_size;
};

}  // namespace sirius::io
```

### Per-GPU backend cache init in SiriusContext::initialize()

```cpp
// Source: pattern from CUCASCADE-IO.md §"Per-GPU backend ownership",
// insertion point verified at src/sirius_context.cpp:169 (after
// memory_manager_ construction) and before line 193 (data_repository_manager_).

// ... existing memory_manager_ construction at line 169 ...

// NEW: register built-in io backends and construct one per GPU under
// the matching CUDA context. Each backend binds its internal cudaStream_t
// + pinned host buffers to the context current at construction time
// (cucascade pipeline_io_backend.cpp:143-149).
cucascade::register_builtin_io_backends(io_backend_registry_);
for (auto* gpu_space :
     memory_manager_->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)) {
  auto const device_id = gpu_space->get_device_id();
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
  gpu_io_backends_[device_id] = io_backend_registry_.create_default_backend();
  SIRIUS_LOG_INFO("SiriusContext: io_backend created for GPU {} (current device={})",
                   device_id, /* cudaGetDevice() readback for IO-11 audit */);
}

// ... existing data_repository_manager_ construction at line 193 ...
```

### Teardown order in SiriusContext::terminate()

```cpp
// Source: mirror of downgrade_executors_ teardown at sirius_context.cpp:265-268.
// Insert BEFORE memory_manager_->shutdown() at line 289.

// NEW: destroy per-GPU backends + registry BEFORE memory_manager_ shutdown.
// Each backend's destructor calls cudaFreeHost / cudaStreamDestroy /
// cudaEventDestroy on resources bound to its GPU's CUDA context; that
// context must still be alive, which is true as long as memory_manager_
// hasn't been shut down. Clearing the registry drops the last shared_ptr
// to each backend, triggering destruction.
gpu_io_backends_.clear();
// io_backend_registry_ has no allocated CUDA state; can be cleared anytime
// or rely on SiriusContext destruction.

// ... existing cudaDeviceSynchronize() + small_pinned_allocator_.reset() + ...
// memory_manager_->shutdown();
// memory_manager_.reset();
```

### Iceberg delete-file migration pattern

```cpp
// Source: migration of src/op/scan/iceberg_scan_task.cpp:57 per CONTEXT
// + CUDF-DATASOURCE.md §"Iceberg delete files are small and frequent".

void read_positional_delete_file(std::string const& delete_file_path,
                                 std::unordered_map<std::string, std::vector<int64_t>>& out_map)
{
  auto stream = cudf::get_default_stream();

  // Pick a backend — iceberg delete-file scans run on the calling task's
  // GPU, which is SiriusContext::current_gpu or the task-wide default.
  auto const device_id = /* resolve via caller's preferred_device_id */;
  auto backend = /* SiriusContext accessor */.get_io_backend_for(device_id);

  // Size the file via std::filesystem (cucascade backend has no size() API;
  // adapter accepts file_size as a constructor arg).
  auto const file_size = std::filesystem::file_size(delete_file_path);

  sirius::io::cucascade_datasource ds{std::move(backend), delete_file_path, file_size};

  auto opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{&ds}).build();
  auto result = cudf::io::read_parquet(opts, stream);

  // ... rest of the function unchanged ...
}
```

### HYG-01 fix — threading explicit stream into filter_row_groups_with_stats

```cpp
// Source: src/op/scan/parquet_scan_task.cpp:468 (current) + precedent
// from src/op/scan/sirius_parquet_metadata_scan_operator.cpp:285.

void parquet_scan_task_global_state::initialize_from_files()
{
  // ... existing footer reads, hybrid_scan_reader construction ...

  // NEW: local stream for planning-time filter_row_groups_with_stats.
  // Use a throwaway rmm::cuda_stream; the call is one-shot per file.
  rmm::cuda_stream planning_stream;

  for (std::size_t file_idx = 0; file_idx < _file_paths.size(); ++file_idx) {
    auto row_group_indices = readers[file_idx]->all_row_groups(_reader_options);
    if (_translated_filter) {
      // ... logging ...
      row_group_indices = readers[file_idx]->filter_row_groups_with_stats(
        row_group_indices, _reader_options, planning_stream.view());  // was rmm::cuda_stream_default
      // ... logging ...
    }
    // ... rest unchanged ...
  }
}
```

## State of the Art

Nothing genuinely new since Phase 4 (2026-04-20). The project-level research docs are current as of the phase start.

| Old Approach | Current Approach | When Changed | Impact |
|---|---|---|---|
| `cudf::io::datasource::create(path)` → kvikio `file_source` (single CUDA context binding) | Custom `cucascade_datasource` → `cucascade::idisk_io_backend` (per-GPU instance under `rmm::cuda_set_device_raii`) | Phase 5 (this phase) | Unblocks multi-GPU parquet scan. Potential bandwidth regression if on GDS-enabled NVMe (H2D staging vs. GPUDirect). Budget ≤30% for SF10 (IO-10) |
| `rmm::cuda_stream_default` at `parquet_scan_task.cpp:468` | Explicit `rmm::cuda_stream` threaded through `initialize_from_files()` | Phase 5 (HYG-01) | Hygiene; no behavior change expected. Aligns with user rule |
| Kvikio-backed `datasource` created implicitly by filepath-based `source_info{path}` in iceberg | Explicit `cucascade_datasource` passed as `source_info{&ds}` | Phase 5 (IO-06) | Removes second kvikio entry point; same multi-GPU safety benefit |

**Deprecated/outdated:**
- **Option 2 (env-var disable of kvikio)** — CUDF-DATASOURCE.md proves NOT VIABLE in cuDF 26.04. `KVIKIO_COMPAT_MODE=ON` already the default; it only toggles GDS-vs-POSIX inside kvikio, doesn't remove kvikio from the code path.
- **Option 3 (hybrid: keep kvikio for footer, cucascade for row-groups)** — CUDF-DATASOURCE.md rejects because the metadata-pass call sites still create kvikio `FileHandle` which defeats the multi-GPU goal.

## Open Questions

1. **How does `parquet_scan_task` resolve its preferred device's backend?** CONTEXT locks the per-GPU cache shape but doesn't prescribe the accessor path. Options enumerated in §Preferred-device plumbing; recommend Option 3 (task_creator seeds local_state with the resolved backend).
   - What we know: `gpu_pipeline_task::get_preferred_device_id()` already works (Phase 4). `task_creator` already resolves preferred device at task-creation time. `parquet_scan_task` currently has no `SiriusContext` pointer.
   - What's unclear: whether planner prefers Option 3 plumbing or accepts a new `SiriusContext` accessor.
   - Recommendation: defer to planner; Option 3 keeps architectural separation cleanest.

2. **What's the planning-time device selection for line 312 + 251?** Footer/metadata reads happen before tasks exist. Arbitrary choice (first GPU) is correctness-neutral; matters only for CPU-side completion time. Document choice in comment.
   - Recommendation: use `gpu_io_backends_.begin()->second` with a comment explaining it's a planning-time read.

3. **Does `cucascade::idisk_io_backend` expose a size() API?** Research says no — `disk_io_backend.hpp` header confirmed: only `read`/`write`/`write_batch`/`read_batch`, no size. Caller must `std::filesystem::file_size(path)` before adapter construction.
   - Recommendation: adapter accepts `file_size` as constructor arg; callers use `std::filesystem::file_size(path)` (throws on missing file — appropriate).

4. **TPC-H Q4 flake — fixed incidentally or still present?** STATE.md / Phase-4 deferred the root cause to Phase 5. Plan a measurement task that runs TPC-H SF1 post-migration and records Q4 pass/fail across N=10 runs.
   - Recommendation: include in phase summary, not a blocking criterion.

5. **Should the adapter implement `device_read` and `device_read_async` even though `supports_device_read() == false`?** cuDF contractually won't call them because of the flag, but base class `CUDF_FAIL` would be jarring in debug. Practical behavior:
   - cuDF reads the `supports_device_read()` flag and never calls device_read* — base-class `CUDF_FAIL` never triggers.
   - Leaving them un-overridden is safe but emits "override keyword missing" warnings if we expose them in the header.
   - Recommendation: don't override them. Trust the flag.

6. **How do we detect kvikio residue post-migration?** `grep -rnw 'datasource::create' src/` is the IO-08 gate. Also grep `source_info\{` and filter for the filepath-variant. Also spot-check with `strings build/release/extension/sirius/sirius.duckdb_extension | grep kvikio` — if kvikio symbols appear, cuDF still links kvikio but Sirius source code no longer uses it (cuDF's kvikio use is internal; Phase 5 doesn't touch that, and it's not loaded on our code paths).
   - Recommendation: document the grep gates in the phase summary as acceptance artifacts.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|---|---|---|---|---|
| cucascade submodule @ f47de0b | IO-01, IO-04, entire phase | ✓ | f47de0b (pinned by Phase 4 BUMP-01) | — |
| libcudf 26.04.* | IO-02, IO-05 custom datasource seam | ✓ | 26.04.* (pixi pin) | — |
| libkvikio (transitive via cudf) | Nothing we call directly; still linked by cuDF for internal use | ✓ | 26.04.00 | N/A — we remove Sirius-level use |
| `std::filesystem` | `file_size(path)` at adapter construction | ✓ (C++17/20 std) | — | — |
| Catch2 v2 | Unit tests for adapter | ✓ (already used) | v2.13.x | — |
| `compute-sanitizer` CLI | IO-11 multi-GPU validation | needs verification on target host | — | If unavailable: manual `cudaGetDevice()` logging per backend construction (CONTEXT already allows this as alt) |
| Multi-GPU hardware (N≥2) | IO-11 validation | Available on the Phase-4 verification host (per 04-SUMMARY) | — | If unavailable: WARN+return per Catch2 v2 convention (CLAUDE.md + STATE.md) |
| pixi + MCP project-commands server | Build/test invocation | ✓ (CLAUDE.md establishes) | — | — |
| `pre-commit` tools (clang-format, black, cmake-format, codespell) | CLAUDE.md final-pass formatting | ✓ (Phase 4 ran successfully) | — | — |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** `compute-sanitizer` — if unavailable on the validation host, CONTEXT explicitly allows manual `cudaGetDevice()` logging as proof of IO-11. Document the choice in the phase summary.

## Sources

### Primary (HIGH confidence)

- **`.planning/research/CUCASCADE-IO.md`** — exhaustive research on `idisk_io_backend` API, `pipeline_io_backend` internals, `io_backend_registry`, per-GPU ownership model, 7 pitfalls P1–P7, 6 open questions Q1–Q6. Referenced verbatim for API semantics.
- **`.planning/research/CUDF-DATASOURCE.md`** — exhaustive research on cuDF 26.04 parquet I/O paths, `datasource::create` internals, `kvikio_source` / `file_source` chain, `supports_device_read()` dispatch in parquet reader, 8 pitfalls, 6 open questions. Referenced verbatim for cuDF integration.
- **`.planning/phases/05-cucascade-backed-parquet-i-o-migration/05-CONTEXT.md`** — locked decisions from `/gsd:discuss-phase` 2026-04-20. Authoritative for all Phase 5 user constraints.
- **`.planning/REQUIREMENTS.md`** — REQUIREMENTS IO-01..IO-11 + HYG-01..HYG-02 wording. Authoritative for acceptance criteria and file placement (e.g., "`src/io/`" in IO-01).
- **`.planning/phases/04-cucascade-bump-v1-0-re-integration/04-SUMMARY.md`** — cucascade SHA f47de0b confirmed; Phase 4 deferrals enumerated (test_downgrade_executor.cpp:813 TODO(MGPU-06), Q4 flake to Phase 5).
- **Phase-4-HEAD working tree (commit 13e4322)** — all file:line references verified via direct Grep on the live tree. Every line number in CONTEXT.md + CUCASCADE-IO.md + CUDF-DATASOURCE.md confirmed present at the cited line.
- **`cucascade/include/cucascade/data/disk_io_backend.hpp`** — `idisk_io_backend` interface signatures read directly from f47de0b pinned submodule.
- **`cucascade/include/cucascade/data/io_backend_registry.hpp`** — `io_backend_registry` + `register_builtin_io_backends` signatures read directly.
- **`src/include/op/scan/prefetched_data_source.hpp` + `.cpp`** — precedent custom `cudf::io::datasource` subclass in the Sirius codebase; adapter design mirrors this class's shape.
- **`src/sirius_context.cpp` + `src/include/sirius_context.hpp`** — exact insertion points for per-GPU backend cache verified.

### Secondary (MEDIUM confidence)

- **CLAUDE.md** — project-wide rules (no `cuda_stream_default`, MCP builds, Super Sirius namespace, Catch2 conventions). Authoritative for constraints but not in itself a research finding.
- **STATE.md** — Phase 4 deferrals + TPC-H Q4 flake scoping to Phase 5.

### Tertiary (LOW confidence)

- None — Phase 5 research leans entirely on project-level docs + direct code verification. No WebSearch or external refs needed; Phase 4 + CUCASCADE-IO.md + CUDF-DATASOURCE.md already did the deep research.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions pinned by Phase 4; no new deps
- Architecture: HIGH — locked by CONTEXT; adapter shape precedent (prefetched_data_source) in-tree
- Migration call sites (line:number): HIGH — every line verified against Phase-4-HEAD working tree
- Pitfalls: HIGH — inherited from CUCASCADE-IO.md + CUDF-DATASOURCE.md with adapter-context annotations
- HYG-02 scope: HIGH — grep performed on the exact file set this phase modifies; only HYG-01 hit is in scope
- Preferred-device plumbing (Option 3 recommendation): MEDIUM — three viable options enumerated; planner should confirm

**Research date:** 2026-04-20
**Valid until:** 2026-05-20 (30 days) — cucascade pin is stable (Phase 4 locked); cuDF 26.04 is pinned. Main risk to validity: a cucascade upstream change to `idisk_io_backend` API would invalidate Adapter header skeleton. Revalidate if `cucascade` submodule SHA changes.
