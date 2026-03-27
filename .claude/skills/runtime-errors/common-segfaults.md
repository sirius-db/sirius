# Known Segfault Patterns in Sirius

## cuDF Column Lifetime Issues

**Pattern:** Crash when accessing a `cudf::column_view` after the owning `cudf::table` or `cudf::column` has been destroyed.

**Symptoms:** Segfault in cuDF operations (join, filter, aggregate) when accessing column data. The column_view is a non-owning reference.

**Fix:** Ensure the owning `cudf::table` or `std::unique_ptr<cudf::column>` outlives all `column_view` references. Check for moves or resets that transfer ownership prematurely.

## DuckDB Vector Invalidation

**Pattern:** Crash when reading data from a DuckDB `Vector` after it has been recycled or invalidated during a scan.

**Symptoms:** Segfault during scan operations, corrupted data in scanned batches.

**Fix:** Copy data from DuckDB vectors before the scan advances to the next chunk. Don't hold references across scan iterations.

## GPU Memory Access Violation (Illegal Address)

**Pattern:** `CUDA error: an illegal memory access was encountered` followed by segfault.

**Symptoms:** Crash during GPU kernel execution or cuDF operation. May be intermittent.

**Causes:**
- Device pointer used after `cudaFree` or RMM deallocation
- Out-of-bounds access in GPU kernel (row index >= num_rows)
- Accessing host memory from device code or vice versa

**Fix:** Use `compute-sanitizer --tool memcheck` to identify the exact kernel and memory access. Check buffer sizes and pointer lifetimes.

## Null Data Batch

**Pattern:** Crash when processing a `DataBatch` that contains null or empty data.

**Symptoms:** Segfault in operator execution, often in the pipeline executor.

**Fix:** Add null/empty checks before processing DataBatch. Check the Data Repository for conditions where a null batch could be produced.

## Thread Pool Use-After-Free

**Pattern:** Crash when a GPU thread pool task accesses data that has been freed by another thread.

**Symptoms:** Intermittent segfault in pipeline execution, especially under high concurrency.

**Fix:** Ensure proper synchronization between task completion and data cleanup. Use shared_ptr for shared data, or synchronize with barriers/events.

## Memory Reservation Underflow

**Pattern:** Crash when a memory reservation returns more memory than was allocated.

**Symptoms:** Assertion failure or segfault in the Memory Reservation Manager.

**Fix:** Track reservation amounts carefully. Ensure release calls match acquire calls exactly.

## cuCascade Tier Migration Crash

**Pattern:** Crash during data movement between GPU/CPU/disk tiers.

**Symptoms:** Segfault in cuCascade allocation or deallocation functions, especially under memory pressure.

**Fix:** Check cuCascade allocation handles for validity. Ensure data is not accessed during migration. Verify memory region sizes are sufficient.

## Stack Overflow in Recursive Plan Generation

**Pattern:** Crash during physical plan generation for deeply nested queries (many subqueries or CTEs).

**Symptoms:** Segfault with a very deep call stack in `gpu_physical_plan_generator.cpp`.

**Fix:** Convert recursion to iteration, or increase stack size. Check for infinite recursion in plan generation.

## String Column Buffer Overflow

**Pattern:** Crash when processing VARCHAR columns with strings exceeding expected buffer sizes.

**Symptoms:** Segfault in string operations, especially during hash computation or comparison.

**Fix:** Use cuDF's string utilities which handle variable-length strings correctly. Don't assume fixed buffer sizes for string data.
