# Stack Research

**Domain:** GPU SQL engine debug-print utilities (CUDA/cuDF codebase)
**Researched:** 2026-04-06
**Confidence:** HIGH — all findings verified against installed headers in pixi environment and existing Sirius source patterns

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| cudf (libcudf) | 26.02.x | GPU data access, stats, hashing | Already a Sirius dependency; provides `cudf::reduce`, `cudf::hashing::xxhash_64`, `cudf::column_view::null_count()`, and `cudf::strings_column_view` — no new deps required |
| spdlog | 1.8.5 | Log output via `SIRIUS_LOG_*` macros | Already used throughout Sirius; all debug output must route through `SIRIUS_LOG_DEBUG`/`SIRIUS_LOG_TRACE` — see `src/include/log/logging.hpp` |
| fmt (bundled in spdlog) | 7.1.3 | String formatting for aligned-column tables | spdlog 1.8 bundles fmt 7.1 at `<spdlog/fmt/fmt.h>`; `fmt::format` with width specifiers (`{:<20}`) produces aligned log output without adding a dependency |
| CUDA Runtime | 12+/13+ | `cudaMemcpy(DeviceToHost)` for GPU-to-host transfer | Already used in `src/cuda/print.cu` and throughout operators; direct `cudaMemcpy` is the correct primitive for controlled, bounded copies |
| C++20 stdlib | C++20 | `std::string`, `std::vector`, `std::format` | Project already requires C++20; `std::format` is available but spdlog's bundled fmt is preferred for consistency with `SIRIUS_LOG_*` macros |

### Supporting Libraries (all already present — zero new dependencies)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `<cudf/hashing.hpp>` | 26.02.x | Per-column and whole-table fingerprinting | `debug_checksum`: use `cudf::hashing::xxhash_64(table_view)` — operates on GPU, returns a UINT64 column of per-row hashes; then reduce with SUM or XOR on host |
| `<cudf/reduction.hpp>` | 26.02.x | Per-column min/max/sum/null-count stats | `debug_stats`: call `cudf::reduce(col, cudf::make_min_aggregation<cudf::reduce_aggregation>(), output_type)` — returns a `unique_ptr<cudf::scalar>` that can be extracted via `.value()` after one stream sync |
| `<cudf/strings/strings_column_view.hpp>` | 26.02.x | Extracting VARCHAR column data to host | `debug_head` for STRING columns: use `cudf::strings_column_view` to get `chars_begin(stream)` + `offsets()` — this is the established Sirius pattern from `src/op/scan/iceberg_scan_task.cpp` |
| `<cudf/types.hpp>` / `cudf::type_dispatcher` | 26.02.x | Runtime type dispatch for formatting | Dispatch all type-specific logic through `cudf::type_dispatcher` or a `switch(col.type().id())` — already used in `src/cuda/print.cu` |
| `<cudf/copying.hpp>` | 26.02.x | Slicing first-N rows before copy | `debug_head(batch, N)`: call `cudf::slice(table, {0, N})` before memcpy to avoid copying the full table; `cudf::slice` is zero-copy on the device side |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `cudf::get_default_stream()` | Provides the default CUDA stream for all cudf API calls | Always pass this explicitly — do not use `0` (default stream) as Sirius uses per-pipeline streams and mixing causes subtle synchronization bugs |
| `cudaDeviceSynchronize()` | Synchronize before host-side formatting | Call once before starting any `debug_*` function body; existing `print_table_contents` already does this — follow that pattern |
| `SIRIUS_DIAG` tag prefix | Makes diagnostic output grep-able | All `SIRIUS_LOG_*` calls in debug utilities must prefix with `[SIRIUS_DIAG]` — the `/validate` and `/runtime-errors` skills grep for this tag |

---

## Implementation Patterns by Function

### `debug_head(batch, N)` — Print first N rows

**GPU-to-host strategy:** Use `cudf::slice` (zero-copy view slicing) to get a table_view of rows [0, N), then copy only that slice to host.

```cpp
// 1. Slice to limit: zero-copy, no device memory allocation
auto sliced = cudf::slice(table, {0, std::min(N, table.num_rows())});

// 2. For numeric columns: direct cudaMemcpy into std::vector<T>
std::vector<int64_t> host(n);
cudaMemcpy(host.data(), col.data<int64_t>(), n * sizeof(int64_t), cudaMemcpyDeviceToHost);

// 3. For STRING columns: use strings_column_view pattern from iceberg_scan_task.cpp
cudf::strings_column_view sv(col);
auto chars_bytes = sv.chars_size(stream);
std::vector<char> host_chars(chars_bytes);
cudaMemcpy(host_chars.data(), sv.chars_begin(stream), chars_bytes, cudaMemcpyDeviceToHost);
std::vector<int32_t> host_offsets(n + 1);
cudaMemcpy(host_offsets.data(), sv.offsets().data<int32_t>(), (n+1)*sizeof(int32_t), cudaMemcpyDeviceToHost);
// Then reconstruct strings: std::string(host_chars.data() + host_offsets[i], host_offsets[i+1] - host_offsets[i])
```

**Formatting:** Use `fmt::format` (via `<spdlog/fmt/fmt.h>`) with width specifiers for aligned-column output. Build each row as a string, log via `SIRIUS_LOG_DEBUG`. For CSV output, build a second pass with comma separators.

```cpp
// Aligned format (pandas-style):
// col_0        col_1        col_2
// -----------  -----------  -----------
// 42           hello        3.14

// CSV format:
// col_0,col_1,col_2
// 42,hello,3.14
```

**Column width:** Compute `max(col_name.size(), max_value_width)` for each column during the first pass over host data — do not hardcode widths.

### `debug_stats(batch)` — Per-column min/max/sum

**GPU-side reduction:** Call `cudf::reduce` for each column:

```cpp
// MIN and MAX supported for all arithmetic + timestamp + string types
auto min_scalar = cudf::reduce(col, *cudf::make_min_aggregation<cudf::reduce_aggregation>(), col.type());
auto max_scalar = cudf::reduce(col, *cudf::make_max_aggregation<cudf::reduce_aggregation>(), col.type());

// SUM only for arithmetic types (INT*, UINT*, FLOAT*, DOUBLE)
auto sum_scalar = cudf::reduce(col, *cudf::make_sum_aggregation<cudf::reduce_aggregation>(), cudf::data_type{cudf::type_id::INT64});

// Extract value: scalar is already a host-accessible object after reduce() returns
// For numeric_scalar<T>: static_cast<cudf::numeric_scalar<int64_t>*>(min_scalar.get())->value()
// For string_scalar: static_cast<cudf::string_scalar*>(min_scalar.get())->to_string()
```

**Important:** `cudf::reduce` is synchronous — it returns after the GPU completes. No extra `cudaDeviceSynchronize()` needed after `reduce()`.

**Sum for DECIMAL:** Cast to INT64/FLOAT64 output type. DECIMAL128 requires careful handling — sum into DECIMAL128 output type, then format with scale.

**Sum for STRING:** Not meaningful — log "N/A" instead of calling reduce with SUM.

### `debug_schema(batch)` — Column names, types, null counts, row count

**Null count:** `col.null_count()` is available on `cudf::column_view` directly — no GPU kernel needed, it is stored as metadata. If the column has no null mask, `null_count()` returns 0.

```cpp
SIRIUS_LOG_DEBUG("[SIRIUS_DIAG] schema: {} rows, {} cols", table.num_rows(), table.num_columns());
for (int c = 0; c < table.num_columns(); ++c) {
    auto const& col = table.column(c);
    SIRIUS_LOG_DEBUG("[SIRIUS_DIAG]   col[{}]: type={}, rows={}, nulls={} ({:.1f}%)",
        c, cudf::type_to_name(col.type()), col.size(),
        col.null_count(),
        col.size() > 0 ? 100.0 * col.null_count() / col.size() : 0.0);
}
```

Column names come from the caller — `cudf::table_view` does not store names. The `debug_schema` signature should accept `std::vector<std::string> const& names` as an optional parameter.

### `debug_checksum(batch)` — Per-column fingerprint

**Recommended approach:** `cudf::hashing::xxhash_64` operating on a single-column table, then sum the resulting UINT64 column on host.

```cpp
#include <cudf/hashing.hpp>

// Hash each column independently to get a per-column fingerprint
for (int c = 0; c < table.num_columns(); ++c) {
    cudf::table_view col_as_table({table.column(c)});
    auto hash_col = cudf::hashing::xxhash_64(col_as_table);
    // hash_col is a UINT64 column of per-row hashes — copy to host and XOR/sum for fingerprint
    int n = hash_col->size();
    std::vector<uint64_t> host_hashes(n);
    cudaMemcpy(host_hashes.data(), hash_col->view().data<uint64_t>(), n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    uint64_t fingerprint = 0;
    for (auto h : host_hashes) { fingerprint ^= h; }
    SIRIUS_LOG_DEBUG("[SIRIUS_DIAG] checksum col[{}]: 0x{:016x}", c, fingerprint);
}
```

**Why xxhash_64 not md5/sha256:** xxhash_64 is the fastest of the available options (all in `<cudf/hashing.hpp>`) and produces 64-bit values that are representable as a hex integer in log output. MD5 produces string output which is harder to compare programmatically.

**Why per-row XOR not SUM:** XOR is order-independent and wraps cleanly at 64 bits. SUM would overflow and lose information. XOR preserves identity for comparing entire-column fingerprints across two pipeline stages.

### `debug_nulls(batch)` — Null counts and percentages

This is a subset of `debug_schema` — `col.null_count()` on `cudf::column_view` is free (no GPU kernel). No separate implementation needed; `debug_schema` already covers this. `debug_nulls` can be a thin wrapper that calls `debug_schema` with null-focused formatting.

### `debug_diff(batch_a, batch_b)` — Row-level comparison

**GPU strategy:** Use `cudf::binary_operation` with `EQUAL` to produce a bool column per-column pair, then count false values. This is the most expensive function — limit scope to small batches.

```cpp
// For each column pair:
auto eq_col = cudf::binary_operation(col_a, col_b, cudf::binary_operator::EQUAL, cudf::data_type{cudf::type_id::BOOL8});
auto diff_count_scalar = cudf::reduce(*eq_col, *cudf::make_sum_aggregation<cudf::reduce_aggregation>(), cudf::data_type{cudf::type_id::INT64});
// Rows with differences = total_rows - sum_of_trues
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `cudf::hashing::xxhash_64` | `cudf::hashing::murmurhash3_x86_32` | When 32-bit fingerprint is sufficient and you want to match hash_partition seeds — murmurhash3 is what Sirius uses in `gpu_partition_impl.cpp` with `cudf::hash_id::HASH_MURMUR3` |
| `cudf::reduce` for stats | `cudf::groupby` for multi-column stats | Use groupby when computing all stats for all columns in a single kernel pass; for debug utilities this is over-engineering — separate reduce calls per column is simpler and fast enough |
| `fmt::format` (spdlog bundled) | `std::format` (C++20 stdlib) | `std::format` is available (C++20), but `SIRIUS_LOG_*` macros already use spdlog's fmt under the hood — mixing would require `#include <format>` and the two are not always identical on this compiler |
| Direct `cudaMemcpy` for host copy | `cudf::copy_to_host` / Arrow interop | Arrow interop exists (`<cudf/interop.hpp>`) but introduces Arrow dependency and serialization overhead. `cudaMemcpy` is what the existing `print.cu` already uses and what `iceberg_scan_task.cpp` uses for strings — stay consistent |
| `cudf::slice` before copying | Copy full column then truncate | `cudf::slice` produces a zero-copy device view of the first N rows — never copy the full column when only N rows are needed for display |

## What NOT to Do

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Copying entire tables for stats | `debug_stats` only needs min/max/sum — copying millions of rows to compute these on CPU wastes PCIe bandwidth and can be 1000x slower | `cudf::reduce` on GPU for each scalar stat, then copy the single scalar result to host |
| `printf` / `std::cout` for output | Bypasses the spdlog pipeline; output is not captured by log level controls or written to the daily log file that skills parse | `SIRIUS_LOG_DEBUG` / `SIRIUS_LOG_TRACE` exclusively |
| `cudaDeviceSynchronize` inside loops | One synchronization per column serialize the entire GPU | One `cudaDeviceSynchronize()` before the entire debug function, then batch all copies |
| Adding new library dependencies | spdlog, cudf, CUDA runtime cover everything needed | Use what is already linked — do not add tabulate, nlohmann_json, or other formatting libs |
| Implementing in `.cu` files | The `#ifdef __CUDACC__` guard in `logging.hpp` no-ops all `SIRIUS_LOG_*` macros in nvcc-compiled translation units — you cannot log from a `.cu` file | Implement all debug utilities in `.cpp` files; only CUDA kernels live in `.cu` |
| `cudf::reduce` with SUM on STRING columns | cudf::reduce with SUM is not defined for STRING type — it will throw at runtime | For STRING stats, use only MIN/MAX reduction, or skip SUM and log "N/A" |
| Storing column names in `cudf::table_view` | `cudf::table_view` does not carry column names — trying to retrieve them from the view will not compile | Accept `std::vector<std::string> const& names` as a separate parameter; callers pass names from their operator context |

---

## Stack Patterns by Context

**In `.cpp` files (operators, pipeline):**
- Full `SIRIUS_LOG_*` macros available
- Use `cudf::reduce`, `cudf::hashing::xxhash_64`, `cudf::slice` freely
- Call `cudaDeviceSynchronize()` once at the start of each debug function

**In `.cu` files (CUDA kernels):**
- `SIRIUS_LOG_*` is a no-op (see `logging.hpp` guard)
- Debug utilities must NOT be called from device code
- All debug utility implementations belong in `.cpp` or a `.cu` file compiled as C++ (not possible in this build system)
- Keep all new debug code in `src/cuda/print.cu` (already compiled as CUDA) ONLY for device helper kernels if needed; the logging wrappers go in a `.cpp` counterpart

**Thread safety:**
- `SIRIUS_LOG_*` via spdlog is thread-safe (spdlog uses MT sinks)
- `cudf::reduce` and `cudf::hashing` are safe to call from multiple threads with different streams
- `cudaDeviceSynchronize()` is device-wide — it blocks all streams. Use `cudaStreamSynchronize(stream)` per-stream if calling from pipeline task threads to avoid blocking other in-flight pipelines

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| spdlog 1.8.5 | fmt 7.1.3 (bundled) | spdlog 1.8 bundles its own fmt — do not `#include <fmt/format.h>` from an external fmt install; use `#include <spdlog/fmt/fmt.h>` to access the bundled version |
| libcudf 26.02.x | CUDA 12+/13+ | `cudf::hashing::xxhash_64` and the `cudf::hashing` namespace were introduced in cudf 23.x — confirmed present in installed headers at `.pixi/envs/default/include/cudf/hashing.hpp` |
| cudf 26.02.x | `cudf::column_view::null_count()` | Returns stored metadata, not a kernel call — safe to call without stream sync |
| cudf 26.02.x | `cudf::strings_column_view::chars_begin(stream)` | Requires passing a stream; `chars_size(stream)` similarly — both confirmed in installed headers and used in `src/op/scan/iceberg_scan_task.cpp` |

---

## Sources

- Installed headers at `.pixi/envs/default/include/cudf/hashing.hpp` — `cudf::hashing::xxhash_64`, `murmurhash3_x86_32`, `md5`, `sha*` signatures verified (HIGH confidence)
- Installed headers at `.pixi/envs/default/include/cudf/reduction.hpp` — `cudf::reduce` signature, type restrictions table verified (HIGH confidence)
- Installed headers at `.pixi/envs/default/include/cudf/scalar/scalar.hpp` — `numeric_scalar<T>::value()`, `string_scalar::to_string()` verified (HIGH confidence)
- Installed headers at `.pixi/envs/default/include/cudf/column/column_view.hpp` — `null_count()` as stored metadata, not a kernel call (HIGH confidence)
- `src/cuda/print.cu` — existing per-column `cudaMemcpy` pattern, `cudf::type_dispatcher` switch, `print_column_values_signed/unsigned` (HIGH confidence — authoritative existing code)
- `src/op/scan/iceberg_scan_task.cpp` — established Sirius pattern for extracting STRING column data from GPU to host via `cudf::strings_column_view` (HIGH confidence)
- `src/op/merge/gpu_merge_impl.cpp` + `src/op/aggregate/gpu_aggregate_impl.cpp` — established patterns for `cudf::reduce` with `make_min/max/sum_aggregation` (HIGH confidence)
- `src/op/partition/gpu_partition_impl.cpp` — `cudf::hash_partition` with `cudf::hash_id::HASH_MURMUR3` usage (HIGH confidence)
- `pixi.toml` — `libcudf = "26.02.*"`, `spdlog = "1.8.*"` as explicit pinned dependencies (HIGH confidence)
- `CMakeLists.txt` — C++20, CUDA 20, no tabulate/json/external fmt in linked libraries (HIGH confidence)
- `src/include/log/logging.hpp` — `#ifdef __CUDACC__` no-op guard confirmed, spdlog MT macros confirmed (HIGH confidence)

---

*Stack research for: Sirius GPU debug-print utilities*
*Researched: 2026-04-06*
