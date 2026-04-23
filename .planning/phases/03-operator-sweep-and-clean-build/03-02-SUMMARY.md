---
phase: "03"
plan: "02"
subsystem: "test"
tags: [cucascade-api, data_batch, clean-build, test-migration]
dependency_graph:
  requires: ["03-01"]
  provides: ["clean-build-confirmed"]
  affects: ["test/cpp/data", "test/cpp/downgrade", "test/cpp/memory", "test/cpp/operator"]
tech_stack:
  added: []
  patterns:
    - "RAII mutable accessor pattern: { auto mut = batch->to_mutable(); mut.convert_to<>(...); }"
    - "Replace direct get_data() access with get_cudf_table_view() or to_read_only()"
    - "Remove data_batch_processing_handle entirely; batches now start idle and stay idle"
key_files:
  created: []
  modified:
    - "src/include/data/convertible_gpu_pipeline_task.hpp"
    - "test/cpp/data/test_convertible_data_batch.cpp"
    - "test/cpp/data/test_convertible_gpu_pipeline_task.cpp"
    - "test/cpp/downgrade/test_downgrade_disk.cpp"
    - "test/cpp/downgrade/test_downgrade_lifecycle.cpp"
    - "test/cpp/memory/test_host_table_utils.cpp"
    - "test/cpp/operator/aggregate/test_gpu_merge_impl.cpp"
    - "test/cpp/operator/test_gpu_partition_impl.cpp"
    - "test/cpp/operator/test_host_table_chunk_reader.cpp"
    - "test/cpp/operator/test_physical_concat.cpp"
    - "test/cpp/operator/test_physical_result_collector.cpp"
    - "test/cpp/utils/test_validation_utility.hpp"
    - "src/legacy/expression_executor/gpu_expression_executor.cpp"
decisions:
  - "Drop const from validate function signatures to satisfy get_cudf_table_view(data_batch&)"
  - "Implement convertible_gpu_pipeline_task_provider using mutable_pop_if() loop"
  - "RAII mutable wrapper pattern for convert_to: acquire to_mutable(), call convert_to, let RAII release"
metrics:
  duration: "~60min"
  completed: "2026-04-22"
  tasks_completed: 2
  files_changed: 13
---

# Phase 03 Plan 02: Gap Closure — Legacy Executor and Clean Build Summary

Migrated the last legacy `get_data()` call site in production code, implemented the stub
`convertible_gpu_pipeline_task_provider`, and fixed all residual test compilation errors
to achieve a clean build against cucascade commit d9dc331.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Migrate legacy expression executor to to_read_only() | b21799b8 |
| 2 | Clean build — fix all residual compilation errors | c241d3fa |

## What Was Done

### Task 1: Legacy Expression Executor (b21799b8)

Migrated `src/legacy/expression_executor/gpu_expression_executor.cpp` to use
`to_read_only()` RAII accessor, removing the last production `get_data()` direct call.

### Task 2: Clean Build — Residual Error Fixes (c241d3fa)

The build revealed compilation errors across 12 test files, all in the same category:
old cucascade API calls (`try_to_create_task`, `try_to_lock_for_processing`,
`data_batch_processing_handle`, `batch->convert_to<>()`, `batch->get_data()`).

**Files fixed and their error patterns:**

| File | Error Pattern | Fix Applied |
|------|--------------|-------------|
| `test_convertible_data_batch.cpp` | `try_to_create_task`, `batch_state::task_created` | Hold `to_read_only()` to simulate non-idle; use non-blocking `convert()` for lock-contention test |
| `test_convertible_gpu_pipeline_task.cpp` | `try_to_create_task`, `get_state()::task_created`, `get_data()` | Rewrite with new batch helpers; hold `to_mutable()` to simulate non-idle |
| `convertible_gpu_pipeline_task.hpp` | Provider stub returned nullptr/empty | Implement `get_next_convertible` and `get_all_convertible` via `mutable_pop_if()` |
| `test_downgrade_disk.cpp` | Stray `\1` from previous bad sed | Fix variable names in `get_batch_tier()` calls |
| `test_downgrade_lifecycle.cpp` | Stray `\1` from previous bad sed | Restore correct variable names per test case context |
| `test_gpu_merge_impl.cpp` | `data_batch_processing_handle`, `try_to_lock_for_processing` | Remove handle struct field and all lock/unlock boilerplate |
| `test_host_table_utils.cpp` | `batch->convert_to<>()`, `try_to_create_task` | Wrap `convert_to` in `to_mutable()` RAII block |
| `test_validation_utility.hpp` | `batch->get_data()` direct access | Replace with `get_cudf_table_view()` helper; add `data_batch_utils.hpp` include |
| `test_host_table_chunk_reader.cpp` | `try_to_create_task`, `try_to_lock_for_processing` | Remove lock boilerplate; `get_cudf_table_view()` works on idle batches directly |
| `test_gpu_partition_impl.cpp` | `data_batch_processing_handle`, structured binding `auto [batch, handle]` | Return just `shared_ptr<data_batch>` from helper; remove structured binding second element |
| `test_physical_concat.cpp` | `out_table.view().column(0)` | `table_view` has no `.view()` method; use `out_table.column(0)` directly |
| `test_physical_result_collector.cpp` | `batch->convert_to<>()`, `try_to_create_task` | Same RAII mutable wrapper pattern |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Implementation] convertible_gpu_pipeline_task_provider was a stub**
- **Found during:** Task 2 (test_convertible_gpu_pipeline_task.cpp tests expected working provider)
- **Issue:** `get_next_convertible()` returned `nullptr`, `get_all_convertible()` returned empty vector — provider extracted no tasks from queue
- **Fix:** Implemented both methods using `_queue.mutable_pop_if()` with the existing `has_matching_batches()` predicate
- **Files modified:** `src/include/data/convertible_gpu_pipeline_task.hpp`
- **Commit:** c241d3fa

**2. [Rule 1 - Bug] cudf::table_view does not have .view() method**
- **Found during:** Task 2 (test_physical_concat.cpp compilation error)
- **Issue:** `out_table.view().column(0)` — `cudf::table_view` is already a view, `.view()` doesn't exist
- **Fix:** Changed to `out_table.column(0)` directly
- **Files modified:** `test/cpp/operator/test_physical_concat.cpp`
- **Commit:** c241d3fa

**3. [Rule 1 - Bug] const data_batch& prevents get_cudf_table_view() binding**
- **Found during:** Task 2 (test_gpu_partition_impl.cpp)
- **Issue:** `validate_hash_partition(const data_batch& input_batch, ...)` calls `sirius::get_cudf_table_view(data_batch&)` — const ref can't bind to non-const ref parameter
- **Fix:** Dropped `const` from validate function signatures
- **Files modified:** `test/cpp/operator/test_gpu_partition_impl.cpp`
- **Commit:** c241d3fa

## Build Verification

Clean build confirmed: `CMAKE_BUILD_PARALLEL_LEVEL=8 make` completed with no errors against cucascade commit d9dc331. The build produces:
- `build/release/extension/sirius/sirius.duckdb_extension`
- `build/release/extension/sirius/sirius_unittest`

## Known Stubs

None. All previously stubbed provider methods have been implemented.

## Self-Check: PASSED

- Task 1 commit b21799b8: confirmed present in git log
- Task 2 commit c241d3fa: confirmed present in git log
- SUMMARY.md created at `.planning/phases/03-operator-sweep-and-clean-build/03-02-SUMMARY.md`
- Build passes: no errors in `make` output
