---
phase: 03-operator-sweep-and-clean-build
reviewed: 2026-04-23T15:22:48Z
depth: standard
files_reviewed: 63
files_reviewed_list:
  - src/creator/task_creator.cpp
  - src/cuda/print.cu
  - src/debug_utils.cpp
  - src/downgrade/downgrade_executor.cpp
  - src/expression_executor/gpu_expression_executor.cpp
  - src/include/data/convertible_data.hpp
  - src/include/data/convertible_data_batch.hpp
  - src/include/data/convertible_gpu_pipeline_task.hpp
  - src/include/data/data_batch_utils.hpp
  - src/include/debug_utils.hpp
  - src/include/downgrade/downgrade_executor.hpp
  - src/include/expression_executor/gpu_expression_executor.hpp
  - src/include/op/sirius_physical_operator.hpp
  - src/include/pipeline/gpu_pipeline_task.hpp
  - src/include/pipeline/pipeline_executor.hpp
  - src/include/print.hpp
  - src/legacy/expression_executor/gpu_expression_executor.cpp
  - src/op/scan/cpu_source_task.cpp
  - src/op/scan/duckdb_scan_executor.cpp
  - src/op/scan/duckdb_scan_task.cpp
  - src/op/scan/parquet_scan_task.cpp
  - src/op/scan/sirius_gpu_parquet_scan_operator.cpp
  - src/op/sirius_physical_concat.cpp
  - src/op/sirius_physical_cte.cpp
  - src/op/sirius_physical_delim_join.cpp
  - src/op/sirius_physical_filter.cpp
  - src/op/sirius_physical_grouped_aggregate.cpp
  - src/op/sirius_physical_grouped_aggregate_merge.cpp
  - src/op/sirius_physical_hash_join.cpp
  - src/op/sirius_physical_limit.cpp
  - src/op/sirius_physical_merge_sort.cpp
  - src/op/sirius_physical_nested_loop_join.cpp
  - src/op/sirius_physical_operator.cpp
  - src/op/sirius_physical_order.cpp
  - src/op/sirius_physical_partition.cpp
  - src/op/sirius_physical_projection.cpp
  - src/op/sirius_physical_result_collector.cpp
  - src/op/sirius_physical_sort_partition.cpp
  - src/op/sirius_physical_sort_sample.cpp
  - src/op/sirius_physical_table_scan.cpp
  - src/op/sirius_physical_top_n.cpp
  - src/op/sirius_physical_ungrouped_aggregate.cpp
  - src/pipeline/gpu_pipeline_executor.cpp
  - src/pipeline/gpu_pipeline_task.cpp
  - src/sirius_context.cpp
  - src/sirius_engine.cpp
  - test/cpp/downgrade/test_downgrade_executor.cpp
  - test/cpp/expression_executor/test_gpu_expression_executor.cpp
  - test/cpp/operator/aggregate/test_physical_grouped_aggregate.cpp
  - test/cpp/operator/operator_test_utils.hpp
  - test/cpp/operator/test_physical_filter.cpp
  - test/cpp/operator/test_physical_limit.cpp
  - test/cpp/operator/test_physical_mark_join.cpp
  - test/cpp/operator/test_physical_merge_sort.cpp
  - test/cpp/operator/test_physical_order.cpp
  - test/cpp/operator/test_physical_partition.cpp
  - test/cpp/operator/test_physical_projection.cpp
  - test/cpp/operator/test_physical_table_scan.cpp
  - test/cpp/operator/test_physical_top_n.cpp
  - test/cpp/operator/test_physical_ungrouped_aggregate.cpp
  - test/cpp/pipeline/test_gpu_pipeline_disk_readback.cpp
  - test/cpp/pipeline/test_gpu_pipeline_task_history.cpp
  - test/cpp/scan/test_utils.hpp
findings:
  critical: 2
  warning: 3
  info: 3
  total: 8
status: issues_found
---

# Phase 03: Code Review Report

**Reviewed:** 2026-04-23T15:22:48Z
**Depth:** standard
**Files Reviewed:** 63 (2 files from the original 65 do not exist: `src/downgrade/downgrade_task.cpp`, `src/include/downgrade/downgrade_task.hpp`)
**Status:** issues_found

## Summary

This review covers the data_batch API migration in Phase 03, where Sirius operators were updated to use cucascade's new 3-class data_batch API (idle handle, `read_only_data_batch` shared lock, `mutable_data_batch` exclusive lock). The migration is largely correct across operators, pipeline infrastructure, and test code. Most operators correctly receive `read_only_pipelineable_operator_data`, use `clone()` to obtain idle handles where needed, and use `data_batch::to_idle()` to convert read-only accessors back to idle state.

Two critical issues were found: a use-after-move bug in `downgrade_executor::request_downgrade()` and mutating `release_table()` calls made through read-only accessor locks. Three warnings relate to the `get_cudf_table_view(data_batch&)` overload that returns a `cudf::table_view` after the temporary read-only lock is released, creating a dangling reference risk.

## Critical Issues

### CR-01: Use-after-move in `request_downgrade()`

**File:** `src/downgrade/downgrade_executor.cpp:388-391`
**Issue:** After `std::move(req)` into `_request_queue.push()`, the code accesses `req->result.set_value(0)` on the moved-from unique_ptr. If `push()` returns false (queue inactive), `req` is in a moved-from state and dereferencing it is undefined behavior.
**Fix:**
```cpp
std::future<size_t> downgrade_executor::request_downgrade(std::function<bool()> predicate)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = std::move(predicate);
  auto future    = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request");
    // req has been moved -- create a new promise to fulfill the future
    std::promise<size_t> fallback_promise;
    auto fallback_future = fallback_promise.get_future();
    fallback_promise.set_value(0);
    return fallback_future;
  }
  return future;
}
```

Note: the `future` variable was obtained from `req->result` before the move, so it is still valid. However, the `set_value(0)` call on the moved `req` is the problem. An alternative simpler fix: capture `future` before the push and, on failure, use a standalone promise:

```cpp
  auto future = req->result.get_future();
  auto* promise_ptr = &req->result;  // save raw pointer before move
  if (!_request_queue.push(std::move(req))) {
    // If push consumed req (moved ownership), promise_ptr is dangling.
    // If push rejected without consuming, req is still valid.
    // Safe approach: always use a separate promise for the failure path.
    ...
  }
```

The cleanest fix is to get the future before the push and, on failure, set a separate promise:

```cpp
std::future<size_t> downgrade_executor::request_downgrade(std::function<bool()> predicate)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = std::move(predicate);
  auto future    = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request");
    // future is already connected to the moved promise.
    // We cannot set_value on it. Return a resolved future instead.
    std::promise<size_t> p;
    p.set_value(0);
    return p.get_future();
  }
  return future;
}
```

### CR-02: Mutating `release_table()` called through read-only accessor

**File:** `src/op/sirius_physical_grouped_aggregate_merge.cpp:237-241`
**Issue:** `gpu_rep.release_table()` is called on a `gpu_table_representation` obtained through a `read_only_data_batch` accessor (`merged_ro`). `release_table()` is a mutating operation that moves ownership of the internal `unique_ptr<cudf::table>` out of the representation, leaving it in an empty state. This violates the semantic contract of a read-only lock -- other concurrent readers (if any) would see a null table. While cucascade's `get_data()` returns a non-const pointer even from `read_only_data_batch`, calling mutating methods through it breaks the data_batch API's safety guarantees.

**File:** `src/op/sirius_physical_table_scan.cpp:223-227`
**Issue:** Same pattern. `gpu_rep.release_table()` is called inside a `to_read_only()` scope. The comment even says "Re-acquire read lock to extract table and metadata" but then calls a mutating method.

**Fix:** Use `to_mutable()` instead of `to_read_only()` when you intend to call `release_table()`:

For `sirius_physical_grouped_aggregate_merge.cpp:237`:
```cpp
  // Acquire EXCLUSIVE lock since release_table() is a mutating operation
  auto merged_mut    = merged->to_mutable();
  auto* space        = merged_mut.get_memory_space();
  auto mr            = space->get_default_allocator();
  auto& gpu_rep      = merged_mut.get_data()->cast<cucascade::gpu_table_representation>();
  auto merged_cols   = gpu_rep.release_table()->release();
```

For `sirius_physical_table_scan.cpp:223`:
```cpp
      {
        auto output_mut = output_batch->to_mutable();
        auto& gpu_rep   = output_mut.get_data()->cast<cucascade::gpu_table_representation>();
        space           = output_mut.get_memory_space();
        table           = gpu_rep.release_table();
      }  // exclusive lock released here
```

Note: `sirius_gpu_parquet_scan_operator.cpp:194-196` already does this correctly -- it explicitly comments "Need mutable access to release_table() (mutating op)" and uses `to_mutable()`.

## Warnings

### WR-01: `get_cudf_table_view(data_batch&)` returns table_view after lock release

**File:** `src/include/data/data_batch_utils.hpp:71-77`
**Issue:** The `get_cudf_table_view(data_batch&)` overload acquires a temporary `read_only_data_batch`, extracts the `cudf::table_view`, then returns it after the RAII lock is released. The returned `table_view` references GPU memory that is only guaranteed stable while the read-only lock is held. If another thread downgrades or mutates the batch between lock release and table_view usage, the view could reference freed/moved memory. The docstring acknowledges this ("valid as long as the data_batch is alive") but this is weaker than "valid as long as the lock is held."

**Callers affected:**
- `src/pipeline/gpu_pipeline_task.cpp:52` (`validate_operator_output_types`) - diagnostic only, low risk
- `src/pipeline/gpu_pipeline_task.cpp:108-113` (`log_operator_data`) - diagnostic only, low risk
- `test/cpp/operator/operator_test_utils.hpp:116` (`concatenate_batches_horizontal`) - test-only, low risk

**Fix:** For production code, prefer the `get_cudf_table_view(const read_only_data_batch&)` overload which requires the caller to hold the lock. For the diagnostic functions (`validate_operator_output_types`, `log_operator_data`), the risk is low because these run synchronously within the pipeline task while the batch is still exclusively owned by the task. Consider adding a comment acknowledging the assumption, or refactoring to accept `read_only_data_batch&`.

### WR-02: `const_cast` on `read_only_pipelineable_operator_data` in sort operators

**File:** `src/op/sirius_physical_sort_partition.cpp:66`
**File:** `src/op/sirius_physical_sort_sample.cpp:91`
**File:** `src/op/sirius_physical_result_collector.cpp:67`
**Issue:** These operators use `const_cast<read_only_pipelineable_operator_data&>(input).release_read_only_batches()` to release the read-only batches from the input. The `const_cast` is needed because the input is received as `const operator_data&` but `release_read_only_batches()` is non-const (it moves the internal vector). While this works correctly because each task has exclusive ownership of its input data, `const_cast` bypasses const-correctness and could mask bugs if the execution model changes.

**Fix:** Consider making `release_read_only_batches()` available through a non-const path. The cleanest approach would be to change `execute()` signature to take `operator_data&` (non-const), or to provide a consuming overload that takes `operator_data&&`. Since this is a broader API design change, it can be deferred -- but document the assumption that const_cast is safe here because the operator has unique access to the input.

### WR-03: Commented-out code block in `sirius_physical_grouped_aggregate.cpp`

**File:** `src/op/sirius_physical_grouped_aggregate.cpp:28-48` and `89-158`
**Issue:** Large block of commented-out code (70+ lines) for grouping sets implementation that is flagged as "TODO: for now commenting out this code because we are not using grouping sets yet." This represents dead code that increases maintenance burden and makes the file harder to read. The same commented-out function `create_group_chunk_types` appears in `sirius_physical_grouped_aggregate_merge.cpp:33-52` as live code, suggesting the merge file has diverged.

**Fix:** Remove the commented-out code from `sirius_physical_grouped_aggregate.cpp`. The grouping sets implementation can be recovered from git history when needed. If the `create_group_chunk_types` function in the merge file is also unused, remove it there as well.

## Info

### IN-01: Unused helper functions in `sirius_physical_grouped_aggregate_merge.cpp`

**File:** `src/op/sirius_physical_grouped_aggregate_merge.cpp:33-80`
**Issue:** Three static helper functions (`create_group_chunk_types`, `copy_expressions`, `convert_grouping_functions`) are defined but never called within this file. They appear to be scaffolding for future grouping sets support.

**Fix:** Remove unused static functions or add `[[maybe_unused]]` annotations. These can be recovered from version control when grouping sets are implemented.

### IN-02: Duplicate `pipelineable_operator_data` used where `read_only_pipelineable_operator_data` is expected

**File:** `src/op/sirius_physical_grouped_aggregate.cpp:88`
**File:** `test/cpp/operator/aggregate/test_physical_grouped_aggregate.cpp:88,144,204`
**Issue:** In both the operator's `execute()` and the test code, `pipelineable_operator_data` (containing idle `shared_ptr<data_batch>`) is passed to `execute()`. Inside `execute()`, the code calls `dynamic_cast<const read_only_pipelineable_operator_data&>(input_data)` which would fail at runtime if the actual type is `pipelineable_operator_data`, not `read_only_pipelineable_operator_data`. However, looking at the pipeline execution path in `gpu_pipeline_task.cpp:396-399`, the input is converted to `read_only_pipelineable_operator_data` before calling `execute()`. The test code appears to pass `pipelineable_operator_data` directly, which means the `dynamic_cast` either succeeds (if the type hierarchy allows it) or the tests exercise a different code path. This inconsistency should be verified.

**Fix:** Verify that operator unit tests correctly exercise the `read_only_pipelineable_operator_data` path that production code uses. If `pipelineable_operator_data` inherits from `read_only_pipelineable_operator_data`, document this relationship. Otherwise, update tests to wrap input in `read_only_pipelineable_operator_data`.

### IN-03: Test helper uses unsafe `get_cudf_table_view(data_batch&)` overload

**File:** `test/cpp/operator/operator_test_utils.hpp:116`
**Issue:** `concatenate_batches_horizontal()` calls `sirius::get_cudf_table_view(*batch)` which uses the temporary-lock overload. The returned `table_view` is used to copy columns, which involves GPU memory access. In a single-threaded test context this is safe, but it sets a pattern that could be copied into production code.

**Fix:** Consider acquiring `to_read_only()` explicitly in the test helper to hold the lock during column access:
```cpp
for (const auto& batch : batches) {
  auto ro = batch->to_read_only();
  auto table_view = sirius::get_cudf_table_view(ro);
  for (cudf::size_type i = 0; i < table_view.num_columns(); ++i) {
    all_columns.push_back(std::make_unique<cudf::column>(table_view.column(i), stream, mr));
  }
}
```

---

_Reviewed: 2026-04-23T15:22:48Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
