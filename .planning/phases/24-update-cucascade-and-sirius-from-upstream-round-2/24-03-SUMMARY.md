---
phase: 24
plan: 03
subsystem: merge
tags: [merge, upstream, sirius, conflict-resolution, pin-table-host, kvikio-bypass, pin-mgpu-01]
dependency_graph:
  requires: [24-01, 24-02]
  provides: [origin/dev merged, host-tier pin_table, repository_wiring materializer]
  affects: [sirius_extension.cpp, scan_manager, cached_split_provider, sirius_pipeline_converter]
tech_stack:
  added: [host_table_allocation::create, cucascade::host_data_representation]
  patterns: [D-01 upstream-favored, D-04 atomic commit, D-05 gitlink ours-wins, integrate-both for parallel code paths]
key_files:
  created: []
  modified:
    - src/sirius_extension.cpp
    - src/include/scan_manager/cached_split_provider.hpp
    - src/scan_manager/cached_split_provider.cpp
    - src/include/scan_manager/sirius_scan_manager.hpp
    - src/scan_manager/sirius_scan_manager.cpp
    - src/pipeline/sirius_pipeline_converter.cpp
    - src/include/op/result/host_table_chunk_reader.hpp
    - src/op/result/host_table_chunk_reader.cpp
    - src/include/memory/multiple_blocks_allocation_accessor.hpp
    - test/cpp/memory/test_host_table_utils.cpp
decisions:
  - "D-05 ours-wins for cucascade gitlink: fork HEAD 5203de5 beats upstream 96bfea1"
  - "Integrate-both for parallel code paths (PIN-MGPU-01 + HOST tier) — D-09 no unification"
  - "D-04 Commit D separate from merge commit: gpu_table_representation missing stream_view arg"
  - "Drop log_pipeline_debug_info() from converter (ports not attached until after convert returns)"
metrics:
  duration: 82min
  completed: 2026-05-13
---

# Phase 24 Plan 03: Sirius origin/dev Merge Summary

**One-liner:** Merged 2 upstream commits (repository_wiring split + host-tier pin_table) into feature/single-node-multi-gpu2, resolving 9 conflict files by integrating upstream features with our PIN-MGPU-01 and kvikio-bypass invariants.

## Tasks Completed

| Task | Description | Commit | Status |
|------|-------------|--------|--------|
| 1a | D-02 gate: read full diffs (ba5ed27 + 2e197c6), write UPSTREAM-DIFFS.md | `8b2a774` | DONE |
| 1b | Merge `git merge --no-ff origin/dev` | `ff04f31` | DONE |
| 2 | Resolve all conflicts, verify build + invariant gates | `90fad83` | DONE |

## Commits

| SHA | Type | Description |
|-----|------|-------------|
| `8b2a774` | docs | D-02 upstream sirius diff triage (ba5ed27 + 2e197c6) |
| `ff04f31` | merge | origin/dev into feature/single-node-multi-gpu2 (Commit C) |
| `90fad83` | fix | D-04 Commit D: gpu_table_representation requires stream_view arg |
| `c9aa166` | docs | origin/dev merge complete + invariant gates verified |

## What Was Merged

### ba5ed27 — Repository wiring split
- Removes `insert_repository()` overloads from sirius_engine
- Splits `wire_data_repositories()` into plan-time descriptors + runtime `materialize_repository_wiring()` free function
- New files: `repository_wiring.hpp`, `repository_wiring_materializer.cpp`
- Impact on our branch: `sirius_engine.cpp` **auto-merged cleanly** (drain_after_error is in execute(), ba5ed27 only touches initialize_internal())

### 2e197c6 — Host-tier pin_table
- Adds `tier='host'` parameter to pin_table DuckDB function
- New cucascade API: `host_table_allocation::create()` factory, `shared_ptr<multiple_blocks_allocation>`
- New scan path: `insert_pinned_entry_host()` + HOST constructor in `cached_split_provider`
- New test: `[pin_table_host]` integration test
- Bumps cucascade gitlink to `96bfea1` (D-05 conflict: our `5203de5` wins)

## Conflict Resolution (9 files)

All 9 conflicts were driven by 2e197c6's cucascade API changes colliding with our `ff06fac` pre-adaptation:

| File | Strategy | Rationale |
|------|----------|-----------|
| `cucascade` gitlink | OURS-WINS | D-05: our fork `5203de5` is descendant of `96bfea1` |
| `multiple_blocks_allocation_accessor.hpp` | UPSTREAM | D-01: upstream's 3-line comment before template |
| `host_table_chunk_reader.hpp` | INTEGRATE | Upstream method sigs + our value-type private field (avoids dangling ref) |
| `host_table_chunk_reader.cpp` | INTEGRATE | Upstream shared_ptr params + our flexible make_duckdb_strings template |
| `cached_split_provider.hpp` | INTEGRATE BOTH | `_chunk_memory_spaces` (PIN-MGPU-01) + upstream HOST-tier fields |
| `cached_split_provider.cpp` | INTEGRATE BOTH | Our GPU ctor + upstream's new HOST ctor |
| `sirius_scan_manager.hpp` | INTEGRATE BOTH | Our `chunk_memory_spaces` + upstream `host_chunks`/`tier`/`memory_space` |
| `sirius_scan_manager.cpp` | INTEGRATE BOTH | Our `chunk_memory_spaces` move + upstream `tier=GPU` assignment |
| `sirius_pipeline_converter.cpp` | PARTIAL OURS | Keep `configure_partition_min_partitions()`, drop `log_pipeline_debug_info()` (upstream correct) |
| `sirius_extension.cpp` | INTEGRATE BOTH | Our per-file kvikio loop + upstream HOST-tier D2H conversion inside loop |
| `test_host_table_utils.cpp` | UPSTREAM | D-01: formatting only |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Missing stream_view argument in gpu_table_representation constructor**
- **Found during:** Task 2 — MCP build
- **Issue:** Merge resolution added `gpu_table_representation(std::move(chunk.tbl), gpu_mem_space)` (2 args) but the constructor requires 3: `(unique_ptr<cudf::table>, memory_space&, cuda_stream_view)`. The `stream_view` variable was in scope but omitted.
- **Fix:** Added `stream_view` as third argument per D-04 Commit D discipline (separate fix-up, not amending merge commit).
- **Files modified:** `src/sirius_extension.cpp:896`
- **Commit:** `90fad83`

### Structural Issues Fixed During Merge Resolution

**2. [Rule 3 - Blocking] Orphaned `<<<<<<< HEAD` conflict marker**
- Left behind when resolving the large sirius_extension.cpp conflict block.
- Removed with targeted Edit before commit.

**3. [Rule 3 - Blocking] `if (!file_paths.empty())` erroneously nested inside per-file for-loop**
- Redundant guard inside `for (auto const& path : file_paths)` made the `chunked_parquet_reader` construction unreachable in the iteration body.
- Fixed by removing the guard wrapper.

## Invariant Gates (all PASS post-Commit D)

| Gate | Count | Status |
|------|-------|--------|
| drain_after_error | 6 | PASS |
| SCHED-RR (configure_partition_min_partitions) | 4 | PASS |
| CTE producer_types | 2 | PASS |
| downgrade tier gate | 5 | PASS |
| HYG-02 cuda_stream_default | 40 (≤40 limit) | PASS |
| kvikio-free (comment only) | 1 | PASS |
| chunk_memory_spaces (PIN-MGPU-01) | 42 | PASS |
| D-05 gitlink | `5203de5` | PASS |

## Unit Tests

| Tag | Tests | Assertions | Result |
|-----|-------|-----------|--------|
| `[pin_table]` | 1 | 51 | PASS |
| `[pin_table_host]` | 1 | 51 | PASS |
| `[mgpu]` | 16 | 79,091 | PASS |
| `[sirius]` | 3 | 15 | PASS |

## Known Stubs

None — all code paths are wired end-to-end. The HOST-tier pin_table path (`tier='host'`) reads parquet files via kvikio-bypass, converts to host_data_representation via cucascade's D2H converter, and serves scans through the HOST-tier `cached_split_provider` constructor.

## Self-Check: PASSED

Files exist:
- `src/sirius_extension.cpp` — FOUND
- `src/include/scan_manager/cached_split_provider.hpp` — FOUND
- `src/scan_manager/cached_split_provider.cpp` — FOUND
- `src/include/scan_manager/sirius_scan_manager.hpp` — FOUND
- `src/scan_manager/sirius_scan_manager.cpp` — FOUND

Commits exist:
- `8b2a774` — FOUND (D-02 triage gate)
- `ff04f31` — FOUND (merge commit C)
- `90fad83` — FOUND (D-04 Commit D fix-up)
- `c9aa166` — FOUND (docs commit)
