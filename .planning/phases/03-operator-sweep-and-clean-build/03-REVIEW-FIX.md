---
phase: 03-operator-sweep-and-clean-build
fixed_at: 2026-04-23T15:35:00Z
review_path: .planning/phases/03-operator-sweep-and-clean-build/03-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-04-23T15:35:00Z
**Source review:** .planning/phases/03-operator-sweep-and-clean-build/03-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 5
- Fixed: 5
- Skipped: 0

## Fixed Issues

### CR-01: Use-after-move in `request_downgrade()`

**Files modified:** `src/downgrade/downgrade_executor.cpp`
**Commit:** a9a63798
**Applied fix:** Removed the `req->result.set_value(0)` call on the moved-from `unique_ptr`. When `_request_queue.push()` rejects the request (queue inactive), the code now creates a separate `std::promise<size_t>`, sets its value to 0, and returns its future. The original `future` variable (obtained before the move) is tied to the moved promise which may never be fulfilled, so a fresh resolved future is returned on the failure path instead.

### CR-02: Mutating `release_table()` called through read-only accessor

**Files modified:** `src/op/sirius_physical_grouped_aggregate_merge.cpp`, `src/op/sirius_physical_table_scan.cpp`
**Commit:** 48066070
**Applied fix:** Changed both call sites from `to_read_only()` to `to_mutable()` when calling `release_table()`, which is a mutating operation that moves ownership of the internal `unique_ptr<cudf::table>`. In `sirius_physical_grouped_aggregate_merge.cpp`, renamed `merged_ro` to `merged_mut` and used `to_mutable()`. In `sirius_physical_table_scan.cpp`, renamed `output_ro` to `output_mut` and used `to_mutable()`. This matches the existing correct pattern in `sirius_gpu_parquet_scan_operator.cpp`.

### WR-01: `get_cudf_table_view(data_batch&)` returns table_view after lock release

**Files modified:** `src/include/data/data_batch_utils.hpp`
**Commit:** 538dc270
**Applied fix:** Replaced the docstring with an explicit `@warning` block documenting that the returned `table_view` can become dangling if another thread downgrades or mutates the batch concurrently. The warning advises callers to only use this overload when they have exclusive ownership of the batch, and to prefer the `get_cudf_table_view(const read_only_data_batch&)` overload when the caller can hold the lock.

### WR-02: `const_cast` on `read_only_pipelineable_operator_data` in sort operators

**Files modified:** `src/op/sirius_physical_sort_partition.cpp`, `src/op/sirius_physical_sort_sample.cpp`, `src/op/sirius_physical_result_collector.cpp`
**Commit:** 80621721
**Applied fix:** Added safety documentation comments at each `const_cast` call site explaining why the cast is safe: each task has exclusive ownership of its input data, so no other thread accesses the object. The `execute()` signature takes `const&` but the task owns the data. This documents the invariant for future maintainers without requiring a broader API signature change.

### WR-03: Commented-out code block in `sirius_physical_grouped_aggregate.cpp`

**Files modified:** `src/op/sirius_physical_grouped_aggregate.cpp`
**Commit:** ea30b6ce
**Applied fix:** Removed 93 lines of commented-out code: the `create_group_chunk_types` static function (lines 28-48) and the grouping sets initialization block in the constructor body (lines 89-158). The grouping sets implementation can be recovered from git history when needed. The live code (`convert_duckdb_aggregates_to_cudf` and subsequent member initialization) remains intact.

---

_Fixed: 2026-04-23T15:35:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
