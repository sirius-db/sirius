---
phase: 03-operator-sweep-and-clean-build
reviewed: 2026-04-22T19:45:00Z
depth: standard
files_reviewed: 49
files_reviewed_list:
  - src/include/data/data_batch_utils.hpp
  - src/expression_executor/gpu_expression_executor.cpp
  - src/include/expression_executor/gpu_expression_executor.hpp
  - src/include/pipeline/gpu_pipeline_task.hpp
  - src/debug_utils.cpp
  - src/op/sirius_physical_operator.cpp
  - src/op/sirius_physical_filter.cpp
  - src/op/sirius_physical_projection.cpp
  - src/op/sirius_physical_hash_join.cpp
  - src/op/sirius_physical_grouped_aggregate_merge.cpp
  - src/op/sirius_physical_table_scan.cpp
  - src/op/sirius_physical_concat.cpp
  - src/op/sirius_physical_column_data_scan.cpp
  - src/op/sirius_physical_cte.cpp
  - src/op/sirius_physical_delim_join.cpp
  - src/op/sirius_physical_result_collector.cpp
  - src/op/sirius_physical_limit.cpp
  - src/op/sirius_physical_grouped_aggregate.cpp
  - src/op/sirius_physical_order.cpp
  - src/op/sirius_physical_merge_sort.cpp
  - src/op/sirius_physical_partition.cpp
  - src/op/sirius_physical_nested_loop_join.cpp
  - src/op/sirius_physical_sort_partition.cpp
  - src/op/sirius_physical_sort_sample.cpp
  - src/op/sirius_physical_ungrouped_aggregate.cpp
  - src/op/sirius_physical_ungrouped_aggregate_merge.cpp
  - src/op/sirius_physical_top_n.cpp
  - src/op/sirius_physical_top_n_merge.cpp
  - src/op/scan/cpu_source_task.cpp
  - src/op/scan/duckdb_scan_executor.cpp
  - src/op/scan/duckdb_scan_task.cpp
  - src/op/scan/parquet_scan_task.cpp
  - src/op/scan/sirius_gpu_parquet_scan_operator.cpp
  - src/pipeline/gpu_pipeline_executor.cpp
  - src/pipeline/gpu_pipeline_task.cpp
  - src/downgrade/downgrade_executor.cpp
  - src/include/downgrade/downgrade_executor.hpp
  - src/include/pipeline/batch_lock_utils.hpp
  - src/include/op/sirius_physical_operator.hpp
  - src/creator/task_creator.cpp
  - src/sirius_context.cpp
  - src/sirius_engine.cpp
  - src/include/data/convertible_data_batch.hpp
  - src/include/data/convertible_data.hpp
  - src/include/data/convertible_gpu_pipeline_task.hpp
  - src/include/pipeline/pipeline_executor.hpp
  - src/include/pipeline/sirius_pipeline_task_states.hpp
  - src/include/pipeline/sirius_plan_printer.hpp
  - src/include/parallel/task_executor.hpp
  - src/parallel/task_executor.cpp
  - src/include/exec/inspectable_mpsc.hpp
findings:
  critical: 2
  warning: 4
  info: 2
  total: 8
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-04-22T19:45:00Z
**Depth:** standard
**Files Reviewed:** 49
**Status:** issues_found

## Summary

Reviewed 49 source files involved in the cucascade 3-class `data_batch` API migration.
The migration is well-structured overall: operators correctly cast to
`read_only_pipelineable_operator_data`, the RAII lock-release pattern via
`data_batch::to_idle(std::move(ro))` is consistently applied, and the new accessor
types (`read_only_data_batch`, `mutable_data_batch`) are used appropriately in the
pipeline infrastructure (`batch_lock_utils.hpp`, `gpu_pipeline_task.cpp`,
`convertible_data_batch.hpp`).

Two critical issues were found: a use-after-move bug in `downgrade_executor.cpp` and
a mutation-through-read-only-lock pattern in `sirius_physical_grouped_aggregate_merge.cpp`.
Several warning-level issues were also identified around dangling `cudf::table_view`
references and a `const_cast` pattern used to release read-only batches.

## Critical Issues

### CR-01: Use-after-move on `downgrade_request` in `request_downgrade()`

**File:** `src/downgrade/downgrade_executor.cpp:391-393`
**Issue:** After `_request_queue.push(std::move(req))` on line 391, the error-handling
path on lines 392-393 accesses `req->result.set_value(0)`. Because `req` was moved
into the queue, this dereferences a null `unique_ptr`, causing undefined behavior
(likely a segfault). The `push()` call returns `false` only when the queue is inactive,
so the moved-from `req` is accessed on the error path.
**Fix:**
```cpp
std::future<size_t> downgrade_executor::request_downgrade(std::function<bool()> predicate)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = std::move(predicate);
  auto future    = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request");
    // req has been moved -- create a fresh promise to fulfill the future
    std::promise<size_t> fallback;
    fallback.set_value(0);
    return fallback.get_future();
  }
  return future;
}
```

### CR-02: Mutation through read-only lock in grouped aggregate merge

**File:** `src/op/sirius_physical_grouped_aggregate_merge.cpp:237-241`
**Issue:** Lines 237-241 acquire a `read_only_data_batch` via `merged->to_read_only()`,
then call `gpu_rep.release_table()` through the read-only accessor. `release_table()`
is a mutating operation that moves ownership of the underlying `cudf::table` out of
the `gpu_table_representation`. Performing this through a shared (read-only) lock
violates the cucascade API contract: read-only accessors guarantee the data will not
be mutated, and other concurrent readers (e.g., the downgrade executor scanning for
candidates) may see a null table pointer, causing a crash.

Since `merged` is a locally-created idle batch that is not shared with any repository
at this point, this is unlikely to trigger in practice, but it is a correctness
violation that will break if the batch is ever published before this projection step.
**Fix:**
```cpp
// Instead of read-only lock, acquire a mutable lock for the projection step
auto merged_mut   = merged->to_mutable();
auto* space       = merged_mut.get_memory_space();
auto mr           = space->get_default_allocator();
auto& gpu_rep     = merged_mut.get_data()->cast<cucascade::gpu_table_representation>();
auto merged_cols  = gpu_rep.release_table()->release();
// ... rest of projection ...
// Convert back to idle after building the projected batch
```

## Warnings

### WR-01: Dangling `cudf::table_view` from temporary read-only lock

**File:** `src/include/data/data_batch_utils.hpp:71-77`
**Issue:** The `get_cudf_table_view(cucascade::data_batch& batch)` overload acquires a
temporary `read_only_data_batch`, extracts a `cudf::table_view` from it, and returns
the view after the `read_only_data_batch` goes out of scope (releasing the lock). The
returned `table_view` is a non-owning pointer into GPU memory. If the downgrade
executor moves the batch to host memory between the lock release and the caller's use
of the view, the GPU memory backing the view may be freed, producing corrupt results
or a CUDA error.

This overload is called from `validate_operator_output_types()` in
`gpu_pipeline_task.cpp:52` and from `debug_utils.cpp`. In the validation path, the
batch is idle (the lock was just released by `run_one_operator`), making it eligible
for downgrade.
**Fix:** Callers should hold a `read_only_data_batch` for the duration of their use
of the `table_view`. Consider deprecating this overload or having it return a pair
of `(read_only_data_batch, table_view)` to enforce lock lifetime.

### WR-02: Mutation through read-only lock in table scan projection

**File:** `src/op/sirius_physical_table_scan.cpp:213-215`
**Issue:** Lines 213-214 call `gpu_rep.release_table()` through `output_ro` (a
`read_only_data_batch` accessor). Line 215 assigns `output_ro = {}` to release the
lock, but the mutation has already occurred while the read lock was held. This has
the same API contract violation as CR-02. Since `output_batch` is locally created and
not yet published, the practical risk is low, but the pattern is incorrect.
**Fix:** Either acquire a `mutable_data_batch` instead of a `read_only_data_batch` for
the projection path, or release the read lock first and then re-acquire as mutable:
```cpp
// Release read lock
output_ro = {};
// Acquire mutable lock for projection
auto output_mut = output_batch->to_mutable();
auto* space     = output_mut.get_memory_space();
auto& gpu_rep   = output_mut.get_data()->cast<cucascade::gpu_table_representation>();
auto table      = gpu_rep.release_table();
auto columns    = table->release();
```

### WR-03: `const_cast` to release read-only batches from const input

**File:** `src/op/sirius_physical_sort_partition.cpp:66`
**Issue:** The pattern `const_cast<read_only_pipelineable_operator_data&>(input).release_read_only_batches()`
casts away const on the `input_data` reference (which is `const operator_data&`).
This same pattern appears in `sirius_physical_sort_sample.cpp:91`,
`sirius_physical_table_scan.cpp:117`, and several other operators.

While functionally this works because the operator's `execute()` is the sole consumer
of the input data, `const_cast` bypasses the type system's guarantee that the input
is not modified. If the framework ever changes to allow multiple consumers of the same
input, this will silently corrupt shared state.
**Fix:** Change the `execute()` signature to take `operator_data&` (non-const) instead
of `const operator_data&`, or provide a `release_read_only_batches()` method on the
const interface that returns a copy. The `const_cast` pattern should be replaced once
the API contract allows it.

### WR-04: `validate_operator_output_types` uses dangling-reference-risk overload

**File:** `src/pipeline/gpu_pipeline_task.cpp:52`
**Issue:** `validate_operator_output_types()` calls `get_cudf_table_view(*batch)` on
idle `data_batch` objects. This invokes the `data_batch&` overload (WR-01) which
acquires and immediately releases a read lock, returning a `table_view` that is valid
only as long as the underlying GPU memory is not moved. Between `run_one_operator()`
returning and `validate_operator_output_types()` running, the batch is idle and
eligible for downgrade. The `table_view` is then used to check column counts and types
on potentially freed memory.

The same risk applies to the `log_operator_data()` function at line 109, which acquires
a `to_read_only()` lock per batch but correctly holds it within scope while accessing
the view.
**Fix:** Hold a `read_only_data_batch` for the duration of the validation loop:
```cpp
for (size_t batch_index = 0; batch_index < batches.size(); batch_index++) {
  const auto& batch = batches[batch_index];
  if (!batch) { continue; }
  auto ro = batch->to_read_only();
  cudf::table_view tbl = get_cudf_table_view(ro);  // use the read_only overload
  // ... validation ...
}
```

## Info

### IN-01: `convertible_gpu_pipeline_task_provider` methods always return empty/nullptr

**File:** `src/include/data/convertible_gpu_pipeline_task.hpp:247-282`
**Issue:** `get_next_convertible()`, `get_all_convertible()`, and `get_bytes_in_space()`
all return stub values (nullptr, empty vector, 0). The TIER 2 downgrade path in
`downgrade_executor.cpp:234-287` uses the provider directly, so pipeline-queue-level
downgrade is effectively disabled. The `has_matching_batches()` static method at line
296 is defined but never called.
**Fix:** Either implement the provider using `inspectable_mpsc::pop_if()` /
`mutable_get_if()`, or remove the dead `has_matching_batches()` method and add a
comment to the TIER 2 block in `downgrade_executor.cpp` noting that pipeline queue
downgrade is intentionally disabled.

### IN-02: TODO/FIXME comments marking incomplete implementations

**File:** `src/include/op/sirius_physical_operator.hpp:407-419`
**Issue:** `can_create_more_tasks()` and `has_processed_all_tasks()` both contain
`WSM TODO implement this` comments and throw `std::runtime_error` unconditionally.
These are virtual methods on the base class. If any derived operator calls the base
implementation, it will crash at runtime.
**Fix:** Either implement these methods or convert them to pure virtual (`= 0`) so
the compiler enforces that derived classes provide implementations.

---

_Reviewed: 2026-04-22T19:45:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
