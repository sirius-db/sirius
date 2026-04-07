---
phase: 01-infrastructure-and-metadata-inspection
plan: 01
subsystem: debug-utilities
tags: [infrastructure, metadata, nulls, schema, cuda-stream, logging]
dependency_graph:
  requires: []
  provides: [debug_schema, debug_nulls, copy_null_mask_to_host, host_column_nulls, SIRIUS_DIAG_prefix]
  affects: [src/include/debug_utils.hpp, src/debug_utils.cpp, CMakeLists.txt]
tech_stack:
  added: []
  patterns: [stream-scoped-sync, single-string-log-buffering, tier-guard, try-catch-safety]
key_files:
  created:
    - src/include/debug_utils.hpp
    - src/debug_utils.cpp
  modified:
    - CMakeLists.txt
decisions:
  - Used get_batch_id() instead of get_id() (cucascade::data_batch API uses get_batch_id)
  - Placed debug_utils.cpp in EXTENSION_SOURCES (not CUDA_SOURCES) so SIRIUS_LOG macros produce output
metrics:
  duration: 11m 17s
  completed: 2026-04-07T02:10:10Z
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 1
---

# Phase 01 Plan 01: Debug Utility Infrastructure and Metadata Inspection Summary

Debug utility header and implementation with stream-scoped sync, tier guards, try/catch safety, [SIRIUS_DIAG] log routing, and two metadata inspection functions (debug_schema, debug_nulls) using cudf metadata APIs with no GPU kernel launches.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create debug_utils.hpp header and register in CMakeLists.txt | e63bd823 | src/include/debug_utils.hpp, CMakeLists.txt |
| 2 | Implement debug_schema, debug_nulls, and copy_null_mask_to_host | c22841da | src/debug_utils.cpp |

## Implementation Details

### Task 1: Header and CMake Registration

Created `src/include/debug_utils.hpp` with:
- `host_column_nulls` struct: holds host-side null bitmask with `is_null(row)` method
- `copy_null_mask_to_host()`: async device-to-host bitmask copy via cudaMemcpyAsync
- `debug_schema()`: logs column names, types, null counts, row count as [SIRIUS_DIAG] block
- `debug_nulls()`: logs per-column null count and percentage using metadata only
- All functions accept `rmm::cuda_stream_view` for stream-scoped synchronization
- Forward-declares `cucascade::data_batch` to avoid heavy header inclusion

Registered `src/debug_utils.cpp` in `EXTENSION_SOURCES` in `CMakeLists.txt` (alphabetically between `src/cpu_cache.cpp` and `src/creator/task_creator.cpp`).

### Task 2: Implementation

Created `src/debug_utils.cpp` implementing all declared functions with full compliance to all 6 INFRA requirements:

- **INFRA-01 (stream-scoped sync):** `stream.synchronize()` used in all 3 functions; zero occurrences of `cudaDeviceSynchronize`
- **INFRA-02 (null-aware host copy):** `copy_null_mask_to_host` uses `cudf::bitmask_allocation_size_bytes` and `cudaMemcpyAsync` with stream sync
- **INFRA-03 (type dispatch):** `cudf::type_to_name(col.type())` for human-readable type strings
- **INFRA-04 (log routing):** All output via `SIRIUS_LOG_DEBUG`/`SIRIUS_LOG_WARN` with `[SIRIUS_DIAG]` prefix on every line
- **INFRA-05 (single-string buffering):** Entire table output built in `std::string`, emitted in one atomic `SIRIUS_LOG_DEBUG` call
- **INFRA-06 (try/catch safety):** Both `debug_schema` and `debug_nulls` wrapped in `try { ... } catch (std::exception) { ... } catch (...) { ... }`

Additional safety:
- Tier guard helper (`is_gpu_tier`) logs warning and returns safely for non-GPU-tier or null-data batches
- Negative `null_count()` guarded (UNKNOWN_NULL_COUNT sentinel)
- Division-by-zero guarded in null percentage calculation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed get_id() to get_batch_id()**
- **Found during:** Task 2 (build verification)
- **Issue:** Plan specified `batch.get_id()` but `cucascade::data_batch` API uses `get_batch_id()`
- **Fix:** Changed both call sites to `batch.get_batch_id()`
- **Files modified:** src/debug_utils.cpp
- **Commit:** c22841da

## Verification

- Extension builds successfully with `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make`
- `debug_utils.cpp` compiles in both static and loadable extension targets
- Zero occurrences of `cudaDeviceSynchronize` in new code
- Zero occurrences of `printf` or `std::cout` in new code
- 14 occurrences of `[SIRIUS_DIAG]` prefix across all output lines
- 3 occurrences of `stream.synchronize()` (one per function)
- 2 try blocks, 4 catch blocks for complete exception safety

## Self-Check: PASSED

- [x] src/include/debug_utils.hpp exists
- [x] src/debug_utils.cpp exists
- [x] 01-01-SUMMARY.md exists
- [x] Commit e63bd823 exists (Task 1)
- [x] Commit c22841da exists (Task 2)
