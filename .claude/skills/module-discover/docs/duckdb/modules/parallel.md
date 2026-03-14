# parallel

**Status**: USED
**Path**: `duckdb/src/include/duckdb/parallel/`
**Headers we include**:
- `duckdb/parallel/task_executor.hpp`
- `duckdb/parallel/task_scheduler.hpp`
- `duckdb/parallel/thread_context.hpp`

## Summary

The `parallel` module provides DuckDB's multi-threaded execution infrastructure. Sirius uses it to coordinate with DuckDB's thread pool for CPU-side table scanning — DuckDB's parallel scan framework handles multi-threaded parquet reading while Sirius manages its own GPU thread pool.

## API Reference

### ThreadContext

**Header**: `duckdb/parallel/thread_context.hpp`
**Signature**:
```cpp
class ThreadContext {
public:
    explicit ThreadContext(ClientContext &context);
    ClientContext &context;
    // Thread-local profiling data
};
```

**Description**: Per-thread execution state, primarily for profiling and thread-local storage.

**Our usage**:
- `src/operator/gpu_physical_table_scan.cpp` — Create thread contexts for DuckDB scan threads
- `src/op/scan/duckdb_scan_task.cpp` — Thread context for scan task execution
- `test/cpp/scan/test_scan_executor.cpp` — Test thread context

### TaskScheduler

**Header**: `duckdb/parallel/task_scheduler.hpp`
**Signature**:
```cpp
class TaskScheduler {
public:
    static TaskScheduler &GetScheduler(ClientContext &context);
    idx_t NumberOfThreads();
    void ExecuteForever(atomic<bool> *marker);
};
```

**Description**: DuckDB's global task scheduler for parallel execution.

**Our usage**:
- `src/operator/gpu_physical_table_scan.cpp` — Access scheduler for parallel scan coordination

### TaskExecutor

**Header**: `duckdb/parallel/task_executor.hpp`
**Signature**:
```cpp
class TaskExecutor {
public:
    TaskExecutor(ClientContext &context);
    void ExecuteTask();
    bool HasError();
};
```

**Our usage**:
- `src/operator/gpu_physical_table_scan.cpp` — Execute DuckDB tasks for parallel scanning

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `Pipeline` | `pipeline.hpp` | DuckDB's pipeline execution framework (Sirius has its own) |
| `MetaPipeline` | `meta_pipeline.hpp` | Pipeline dependency management |
| `Event` | `event.hpp` | Pipeline execution events |
| `InterruptState` | `interrupt.hpp` | Query interruption handling |
