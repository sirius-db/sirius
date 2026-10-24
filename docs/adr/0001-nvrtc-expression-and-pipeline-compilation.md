# Direct NVRTC `__global__` Compilation with Data-Centric `produce`/`consume` and Tiered GPU Memory Schedules

We compile Sirius Expression Trees (and future relational pipelines) directly into standalone CUDA `__global__` kernels via NVRTC using Neumann's data-centric `produce`/`consume` model over `(value, valid, overflow)` Register Values and TQP++'s tiered GPU memory schedules (Thread-Private Registers, Block-Shared Memory, and Global HBM), rather than lowering to `cudf::ast` or wrapping `__device__` callbacks in `cudf::multi_transform`.

## Considered Options

- **`cudf::ast` (`compute_column` / `compute_column_jit`)**: Lacks support for conditionals (`CASE`, `COALESCE`), `IN` lists, `TRY`/`TRY_CAST`, arbitrary `CAST` targets, string matching (`LIKE`/`CONTAINS`), and checked overflow, forcing frequent Materialization Boundaries across multiple kernel launches.
- **`cudf::multi_transform` (`__device__` UDF callback)**: Exposes only a per-thread scalar callback (`cuda::std::optional<T>`), which hides block/warp coordination and prevents intra-kernel `Sparse-to-Dense Compaction` for filters (single-pass, with a decoupled look-back over row tiles), block-cooperative schedules in general, and future fused `Map-Reduce` aggregation pipelines.
- **Direct NVRTC `__global__` compilation with `produce`/`consume` and pluggable Sink Consumers**: Emits full `__global__` kernels compiled and cached via Sirius's embedded-CCCL NVRTC pipeline, giving full control over registers, shared memory, warp intrinsics, and pipeline fusion.

## Consequences

- **Unified expression and pipeline seam**: Expression nodes act as register producers feeding a pluggable `Sink Consumer` (`ProjectSink` for 1-to-1 projection with warp-ballot validity bitmasks, `FilterSink` for 3-tier `Sparse-to-Dense Compaction`, and future `AggregateSink` / `HashBuildSink` for full pipeline compilation).
- **In-register Carrier Restoration and parameter caching**: Narrowed physical carriers are widened inline in registers without temporary column allocations, and runtime scalar constants are passed as kernel arguments so NVRTC `.cubin` cache entries are reused across literal changes.
- **Branch-safe SQL NULL and overflow semantics**: Lazy `if`/`else` emission prevents unselected `CASE`/`COALESCE`/`AND`/`OR` arms from executing, while the per-value `overflow` poison flag ensures arithmetic or cast overflows only raise `sirius::out_of_range_exception` (`duckdb::OutOfRangeException`) when a poisoned value survives to the `Sink Consumer` (and are cleanly converted to `NULL` by `TRY` / `TRY_CAST`).
