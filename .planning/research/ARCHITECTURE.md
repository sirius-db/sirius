# Architecture Research

**Domain:** GPU SQL engine debugging utilities (CUDA/cuDF operator pipeline)
**Researched:** 2026-04-06
**Confidence:** HIGH — based on direct inspection of existing codebase

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CALL SITE (operator .cpp files)                   │
│   debug_head(batch, N)   debug_stats(batch)   debug_schema(batch)   │
│   debug_nulls(batch)     debug_checksum(batch) debug_diff(a, b)      │
└────────────────────────────┬─────────────────────────────────────────┘
                             │  (pure C++ calls, no CUDA in .cpp)
┌────────────────────────────▼─────────────────────────────────────────┐
│               FORMATTING LAYER  (print.hpp / print.cu)               │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │  debug_* functions — build std::string output, emit via      │   │
│   │  SIRIUS_LOG_DEBUG("[SIRIUS_DIAG] ...")                        │   │
│   │  Aligned-column format   CSV format                          │   │
│   └──────────────────────────────┬───────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────  │
                                   │  delegates GPU-side work
┌──────────────────────────────────▼───────────────────────────────────┐
│                  DATA EXTRACTION LAYER  (print.cu)                   │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │  extract_column_to_host<T>()   — cudaMemcpy device → host    │   │
│   │  extract_strings_to_host()     — cudf::strings::to_host()    │   │
│   │  extract_null_mask_to_host()   — cudaMemcpy bitmask          │   │
│   │  compute_column_stats()        — cudf::minmax / reduce       │   │
│   │  compute_column_checksum()     — cudf::reduce (sum of hashes)│   │
│   └──────────────────────────────┬───────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                                   │  reads through
┌──────────────────────────────────▼───────────────────────────────────┐
│               GPU DATA LAYER  (cucascade + cuDF)                     │
│   cucascade::data_batch                                              │
│     └── gpu_table_representation                                     │
│           └── cudf::table  ──►  cudf::table_view                    │
│                                   └── cudf::column_view (per col)   │
└──────────────────────────────────────────────────────────────────────┘
                                   │  logged to
┌──────────────────────────────────▼───────────────────────────────────┐
│                     LOG SINK  (spdlog / SIRIUS_LOG)                  │
│   SIRIUS_LOG_DEBUG("[SIRIUS_DIAG] ...")  →  sirius.log file          │
│   Parseable by /validate and /runtime-errors skills via grep         │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | File Location |
|-----------|----------------|---------------|
| Public API (debug_* functions) | Accept `cucascade::data_batch` or `cudf::table_view`, coordinate extraction and formatting, emit to SIRIUS_LOG | `src/include/print.hpp` (declarations) |
| Data extraction layer | Copy GPU memory to host buffers; handle type dispatch; compute per-column statistics and checksums; return host-side plain C++ values | `src/cuda/print.cu` (implementation) |
| Formatting layer | Build aligned-column and CSV string representations from host values; embed `[SIRIUS_DIAG]` tags and batch metadata | `src/cuda/print.cu` (same file, CPU-only helpers) |
| Type dispatch | Route per-type extraction logic via switch on `cudf::type_id`; mirror the existing `print_one_column` switch structure | `src/cuda/print.cu` |
| Log sink | Receive formatted strings via `SIRIUS_LOG_DEBUG` / `SIRIUS_LOG_TRACE` macros; controlled by `SIRIUS_LOG_LEVEL` env var | `src/include/log/logging.hpp` (existing) |
| Operator call sites | Insert `debug_*(batch)` calls at operator entry/exit for diagnosis; call from `.cpp` files only (not `.cu`) | Operator `.cpp` files in `src/op/` |

## Recommended Project Structure

No new files needed. Extend the existing two files:

```
src/
├── include/
│   └── print.hpp              # Add debug_head(), debug_stats(), debug_schema(),
│                              # debug_checksum(), debug_diff(), debug_nulls()
│                              # declarations alongside existing print_table_contents()
└── cuda/
    └── print.cu               # Implement all new functions here
                               # GPU extraction helpers stay in anonymous namespace
                               # Formatting helpers (CPU-only) also in anonymous namespace
                               # Public functions exposed under namespace sirius
```

### Structure Rationale

- **Extend print.hpp/print.cu rather than create new files:** PROJECT.md records this as an explicit key decision. The existing file already owns the GPU-to-host copy pattern and `namespace sirius` scope. New files would require CMakeLists changes and risk duplicating the namespace/include setup.
- **All implementation in the .cu file:** The `logging.hpp` header already no-ops the `SIRIUS_LOG_*` macros under `__CUDACC__`, so formatting code that calls spdlog must live in the `.cu` file but in CPU-path code (not in `__global__` functions or device-compiled translation units). The `.cu` file compiles both device and host code, so host-side C++ that uses spdlog works there with the guard already in place.
- **Anonymous namespace for internal helpers:** The existing code uses an anonymous namespace for `print_column_values_signed`, `print_column_values_unsigned`, etc. All new per-type extraction helpers should follow the same pattern — they are implementation details invisible to call sites.

## Architectural Patterns

### Pattern 1: cudaDeviceSynchronize Before Host Copy

**What:** Call `cudaDeviceSynchronize()` once per top-level debug function entry before any `cudaMemcpy` device-to-host calls.
**When to use:** Every `debug_*` entry point. The operator pipeline is task-based and asynchronous; data may still be in flight when debug functions are called.
**Trade-offs:** Adds synchronization overhead, but debug functions are off the hot path by definition. The existing `print_table_contents` already does this.

**Example:**
```cpp
void debug_head(cucascade::data_batch const& batch, cudf::size_type n) {
    cudaDeviceSynchronize();           // ensure prior kernels are done
    cudf::table_view tv = get_cudf_table_view(batch);
    // ... extract and format
}
```

### Pattern 2: Type Dispatch via switch on cudf::type_id

**What:** A single `switch (col.type().id())` covers all supported cudf type IDs. Each arm calls a typed template helper (e.g., `extract_signed<int32_t>`, `extract_unsigned<uint64_t>`).
**When to use:** Anywhere per-column extraction or formatting differs by type. Mirrors existing `print_one_column` exactly.
**Trade-offs:** Verbose but explicit. The alternative (`cudf::type_dispatcher`) is cleaner for large type lists but introduces template complexity that the existing code deliberately avoids.

**Type coverage required:**

| cudf type_id | Handling |
|---|---|
| INT8, INT16, INT32, INT64 | `cudaMemcpy` into `std::vector<T>`, format as decimal |
| UINT8, UINT16, UINT32, UINT64 | `cudaMemcpy` into `std::vector<T>`, format as decimal |
| FLOAT32 | `cudaMemcpy` into `std::vector<float>`, format with `%.6g` |
| FLOAT64 | `cudaMemcpy` into `std::vector<double>`, format with `%.6g` |
| BOOL8 | Treat as INT8, format as `true`/`false` |
| STRING | Use `cudf::strings_column_view` + `cudf::strings::to_host()` to get `std::vector<std::string>` |
| TIMESTAMP_DAYS, TIMESTAMP_SECONDS, TIMESTAMP_MICROSECONDS, TIMESTAMP_NANOSECONDS | Treat as INT64 (epoch units), format with unit suffix |
| DECIMAL32, DECIMAL64, DECIMAL128 | Treat as INT32/INT64/__int128_t, apply `col.type().scale()` for decimal point rendering |
| Everything else | Emit `"(unsupported type: <name>)"` |

### Pattern 3: Null Mask Extraction

**What:** After extracting data values, also copy the null bitmask from device to host and check bit-i for each row.
**When to use:** All debug functions that render row values (`debug_head`, `debug_diff`). Null-aware display uses `"NULL"` instead of the raw value.
**Trade-offs:** Requires an extra `cudaMemcpy` for the bitmask (typically small). Without it, null rows silently show garbage.

**Example:**
```cpp
// bitmask is ceil(n/8) bytes; null bit = 0 means null
cudf::size_type bitmask_bytes = cudf::bitmask_allocation_size_bytes(n);
std::vector<cudf::bitmask_type> host_mask(bitmask_bytes / sizeof(cudf::bitmask_type));
if (col.has_nulls()) {
    cudaMemcpy(host_mask.data(), col.null_mask(),
               bitmask_bytes, cudaMemcpyDeviceToHost);
}
// check bit i: (host_mask[i/32] >> (i%32)) & 1  → 1 = valid, 0 = null
```

### Pattern 4: Output Format — Aligned Columns + CSV

**What:** Every debug function builds two string representations:
1. Aligned-column (pandas-style): column names right-padded to fixed width, values right-padded per column.
2. CSV: header row + one data row per printed row, comma-separated, quoted where needed.

Both formats are emitted as separate `SIRIUS_LOG_DEBUG` lines prefixed with `[SIRIUS_DIAG]`.
**When to use:** Always. The `/validate` skill greps for `[SIRIUS_DIAG]` to extract structured data.
**Trade-offs:** Two formats per call doubles log lines but gives humans aligned output and machines parseable output.

### Pattern 5: Statistics via cuDF Reduction (No Custom Kernels)

**What:** `debug_stats` computes per-column min/max/sum using `cudf::reduce` or `cudf::minmax`. Results are `cudf::scalar` objects that are materialized to host via `->to_host()`.
**When to use:** `debug_stats` and `debug_checksum`. Avoids writing custom CUDA reduction kernels.
**Trade-offs:** Requires one `cudf::reduce` call per statistic per column — acceptable for a debug path. For checksums, use `cudf::reduce` with a `cudf::make_sum_aggregation<cudf::reduce_aggregation>()` after casting the column through `cudf::hash` (or XOR of values for simplicity).

## Data Flow

### debug_head Flow

```
operator .cpp
    └── debug_head(batch, N)
          │  (pull cudf::table_view)
          ├── get_cudf_table_view(batch)
          │     └── batch.get_data()->cast<gpu_table_representation>().get_table()
          │
          ├── cudaDeviceSynchronize()
          │
          ├── for each column [0..num_columns):
          │     ├── extract N rows to host std::vector<T>   (cudaMemcpy)
          │     ├── extract null bitmask to host            (cudaMemcpy if has_nulls)
          │     └── build formatted string for this column
          │
          ├── assemble aligned-column string (all columns, up to N rows)
          ├── assemble CSV string
          │
          ├── SIRIUS_LOG_DEBUG("[SIRIUS_DIAG] batch_id={} head (aligned):\n{}", ...)
          └── SIRIUS_LOG_DEBUG("[SIRIUS_DIAG] batch_id={} head (csv):\n{}", ...)
```

### debug_stats Flow

```
debug_stats(batch)
    ├── get_cudf_table_view(batch)
    ├── cudaDeviceSynchronize()
    ├── for each numeric column:
    │     ├── cudf::minmax(col) → pair<scalar, scalar> on device
    │     ├── min_scalar->to_host_value()  (materializes to host)
    │     ├── max_scalar->to_host_value()
    │     └── cudf::reduce(col, sum_agg) → scalar → to_host_value()
    ├── for STRING columns: report char count, not min/max/sum
    └── SIRIUS_LOG_DEBUG("[SIRIUS_DIAG] batch_id={} stats:\n{}", ...)
```

### debug_diff Flow

```
debug_diff(batch_a, batch_b)
    ├── extract both to host (full table, up to max configurable limit)
    ├── compare row counts — log mismatch if differ
    ├── compare column counts — log mismatch if differ
    ├── for each (row, col) in min(rowcount_a, rowcount_b):
    │     compare string representations of values
    │     collect differing (row, col) positions
    └── SIRIUS_LOG_DEBUG("[SIRIUS_DIAG] diff: N diffs found\n{row,col list}")
```

### Key Data Flows

1. **GPU memory to host:** `data_batch` → `gpu_table_representation` → `cudf::table` → `cudf::table_view` → `cudf::column_view` → `cudaMemcpy` → `std::vector<T>` on host. The batch is read-only; no state transitions are triggered on the `data_batch` (no need to call `try_to_create_task`).
2. **String extraction:** `cudf::column_view` (STRING type) → `cudf::strings_column_view` → `cudf::strings::to_host()` → `std::vector<std::string>`. This is the canonical cuDF API for string host materialization and already appears in the codebase for string length inspection (`src/op/merge/gpu_merge_impl.cpp`).
3. **Formatted string to log:** `std::string` built on host → `SIRIUS_LOG_DEBUG(...)` macro → spdlog `default_logger_raw()` → daily file sink → `sirius.log`.

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `print.hpp` ↔ operator `.cpp` files | Direct function call; operator includes `print.hpp` | Operators already include this header (see `src/cuda/print.cu` line 24) |
| `print.cu` ↔ `data_batch_utils.hpp` | `get_cudf_table_view()` inline helper; already used in existing `print_data_batch_contents` | No new dependencies |
| `print.cu` ↔ `logging.hpp` | `SIRIUS_LOG_DEBUG` macro; already included in `print.cu` | No-ops under `__CUDACC__` so only the CPU-path code in `.cu` can use it |
| `print.cu` ↔ cuDF reduction API | `cudf::reduce`, `cudf::minmax` from `<cudf/reduction.hpp>`; already in `cudf_utils.hpp` aggregate header | New include of `<cudf/reduction.hpp>` needed in print.cu if not already transitive |
| `print.cu` ↔ cuDF strings API | `cudf::strings::to_host()` from `<cudf/strings/convert/convert_fixed_point.hpp>` or `<cudf/strings_column_view.hpp>` | New include needed; verify exact header with cudf version in pixi.toml |
| `/validate` skill ↔ log output | Grep for `[SIRIUS_DIAG]` prefix in `sirius.log` | Structural contract: all debug output must carry this prefix |
| `/runtime-errors` skill ↔ debug functions | Skill inserts `debug_head(batch, 5)` / `debug_schema(batch)` calls at suspected fault points | Skill depends on these function signatures being stable |

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| spdlog | Via existing `SIRIUS_LOG_*` macros only | Never call spdlog directly from debug functions; use macros for consistency with log level filtering |
| CUDA runtime | `cudaMemcpy`, `cudaDeviceSynchronize` | Use `cudaError_t` return values and emit warning log on failure rather than throwing; printing should not crash the pipeline |
| cuDF | `cudf::reduce`, `cudf::minmax`, `cudf::strings_column_view`, `cudf::strings::to_host` | cuDF calls may throw `cudf::logic_error`; wrap in try/catch and log the error string as part of debug output |

## Build Order

The build order is determined by what depends on what:

```
1. print.hpp (header only — declare new function signatures)
        ↓ (no dependency on implementation; call sites compile immediately)
2. print.cu (implement all debug_* bodies)
        ↓ (depends on: data_batch_utils.hpp, logging.hpp, cudf reduction/strings headers)
3. Operator .cpp call sites (insert debug_* calls during diagnosis)
        ↓ (just include print.hpp; no build order constraint beyond print.cu existing)
```

No CMakeLists changes needed. `print.cu` is already in the build as `src/cuda/print.cu`.

## Anti-Patterns

### Anti-Pattern 1: Calling SIRIUS_LOG_* from a .cu Device Kernel

**What people do:** Put `SIRIUS_LOG_DEBUG(...)` inside a `__global__` kernel or in a function compiled by nvcc in device mode.
**Why it's wrong:** `logging.hpp` deliberately no-ops all `SIRIUS_LOG_*` macros under `__CUDACC__` (line 21-26 of `logging.hpp`). The log calls silently vanish. spdlog also cannot function in device code.
**Do this instead:** Copy data to host first (via `cudaMemcpy`), then call `SIRIUS_LOG_*` from host-path code in the same `.cu` file. The distinction is: device kernels (annotated `__global__` or `__device__`) cannot log; plain C++ functions in `.cu` files can, after `cudaDeviceSynchronize()`.

### Anti-Pattern 2: Creating a New .cu File for Debug Utilities

**What people do:** Create `src/cuda/debug.cu` and `src/include/debug.hpp` to separate debug from print.
**Why it's wrong:** Requires CMakeLists additions, duplicates the `get_cudf_table_view` include chain, and splits a cohesive "GPU inspection" concern across two files. PROJECT.md records extending the existing print files as an explicit decision.
**Do this instead:** Add to `print.hpp` and `print.cu`. If the file grows unwieldy, the split can be revisited after MVP.

### Anti-Pattern 3: Throwing Exceptions from Debug Functions

**What people do:** Let cuDF API exceptions propagate out of `debug_*` calls — e.g., `cudf::logic_error` if the table view is invalid.
**Why it's wrong:** Debug functions may be inserted in operator code paths during a query. An exception from the debug call would mask the original bug being investigated and crash the pipeline differently.
**Do this instead:** Wrap all cuDF and CUDA calls in `try/catch` at the top-level `debug_*` entry points. On error, emit `SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_head failed: {}", e.what())` and return. Never rethrow.

### Anti-Pattern 4: Performing Row-by-Row cudaMemcpy in a Loop

**What people do:** Copy one row (or one scalar) at a time with separate `cudaMemcpy` calls to avoid allocating a host buffer.
**Why it's wrong:** Each `cudaMemcpy` has fixed launch overhead (~5-20 us). For 20 rows, that is 20 separate kernel launches before the actual copy. Batching all N rows into one `cudaMemcpy` is the existing pattern (`print_column_values_signed` does this).
**Do this instead:** Allocate `std::vector<T>(n)` on host, one `cudaMemcpy` for all N elements, then iterate the vector on host.

### Anti-Pattern 5: Using printf Instead of SIRIUS_LOG

**What people do:** Use `std::printf(...)` for debug output as the existing `print_table_contents` does.
**Why it's wrong:** `printf` output goes to stdout and bypasses the `sirius.log` file that the `/validate` and `/runtime-errors` skills parse. The existing `print_table_contents` is being superseded precisely because it uses `printf` and isn't skill-parseable.
**Do this instead:** All new `debug_*` functions must use `SIRIUS_LOG_DEBUG("[SIRIUS_DIAG] ...")`. The `[SIRIUS_DIAG]` prefix is the grep target for both skills.

## Sources

- Direct inspection of `src/include/print.hpp` and `src/cuda/print.cu` (existing implementation)
- Direct inspection of `src/include/log/logging.hpp` (spdlog macro layer)
- Direct inspection of `src/include/data/data_batch_utils.hpp` (`get_cudf_table_view` entry point)
- Direct inspection of `cucascade/include/cucascade/data/data_batch.hpp` (batch state machine)
- Direct inspection of `cucascade/include/cucascade/data/gpu_data_representation.hpp` (`gpu_table_representation`)
- Direct inspection of `src/include/op/sirius_physical_operator.hpp` (`operator_data` container)
- Direct inspection of `src/include/cudf/cudf_utils.hpp` (type mapping, existing cuDF includes)
- Direct inspection of `.planning/PROJECT.md` (requirements, constraints, key decisions)
- Pattern reference: `src/op/merge/gpu_merge_impl.cpp` lines 197-201 (existing `cudf::strings_column_view` usage)
- Pattern reference: `src/expression_executor/gpu_expression_executor.cpp` line 314 (existing `.null_count()` usage)

---
*Architecture research for: GPU data inspection utilities for Sirius debug pipeline*
*Researched: 2026-04-06*
