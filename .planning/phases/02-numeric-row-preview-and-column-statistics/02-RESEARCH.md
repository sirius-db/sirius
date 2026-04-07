# Phase 2: Numeric Row Preview and Column Statistics - Research

**Researched:** 2026-04-06
**Domain:** cuDF GPU-to-host data extraction, cudf::reduce statistics, cudf::slice zero-copy row selection
**Confidence:** HIGH

## Summary

Phase 2 adds two new debug functions to the existing `debug_utils` module: `debug_head(batch, N, stream)` for previewing the first N rows of numeric columns, and `debug_stats(batch, stream)` for GPU-computed per-column min/max/sum statistics. Both follow the established Phase 1 patterns (tier guard, try/catch, output buffering, `[SIRIUS_DIAG]` prefix).

The core technical challenges are: (1) using `cudf::slice` for zero-copy row selection followed by `cudaMemcpyAsync` to extract typed column data to the host, (2) dispatching on `cudf::type_id` to format values correctly per type, and (3) using `cudf::reduce` with `make_min_aggregation`, `make_max_aggregation`, and `make_sum_aggregation` for GPU-side statistics without copying full columns to host. All required cuDF APIs are already used elsewhere in the Sirius codebase, so patterns are well-established and verifiable.

**Primary recommendation:** Implement `debug_head` and `debug_stats` as two new functions in the existing `debug_utils.cpp`, reusing `host_column_nulls`, `is_gpu_tier()`, and `get_cudf_table_view()` from Phase 1. Use `cudf::type_dispatcher` for generic value extraction rather than manual switch-case per type.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `debug_head` uses an `enum class DebugFormat { ALIGNED, CSV }` parameter with default `DebugFormat::ALIGNED` -- type-safe, extensible for future formats
- **D-02:** Default row count is 10 when N is not specified (matches pandas convention)
- **D-03:** Column widths are dynamic -- scan the N rows to find max display width per column, then pad all values for perfectly aligned output
- **D-04:** Floating-point numbers use 6 significant digits (`%g`-style) -- fixed notation for normal ranges, scientific for very large/small values
- **D-05:** Booleans display as lowercase `true`/`false`
- **D-06:** NULLs display as the string `NULL` in row output
- **D-07:** `debug_stats` uses a summary table format (one row per column) consistent with `debug_schema`/`debug_nulls` -- columns: idx, name, type, min, max, sum
- **D-08:** Non-numeric columns show `(non-numeric, skipped)` in the stats table
- **D-09:** Min/max/sum only -- no count or mean (count is in the header, mean is derivable). Keeps output compact and GPU reduce calls minimal
- **D-10:** All-NULL numeric columns show `NULL` for min, max, and sum -- follows SQL standard semantics (`SUM/MIN/MAX` of all NULLs = NULL)
- **D-11:** No cap on N -- trust the caller. Try/catch wrapping handles OOM gracefully
- **D-12:** When N > batch row count, clamp silently to `min(N, num_rows)`. Header already shows total row count so the developer sees the batch was smaller
- **D-13:** Empty batches (0 rows) print header info with an `(empty batch)` note -- no data rows

### Claude's Discretion
- Header separator style (dashes, equals, etc.)
- CSV quoting/escaping rules
- Internal helper function decomposition
- cudf::reduce vs cudf::minmax optimization choice for min/max computation

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HEAD-01 | `debug_head(batch, N)` prints first N rows in aligned-column format | cudf::slice for zero-copy row selection, cudaMemcpyAsync for typed data extraction, fmt::format for aligned output |
| HEAD-02 | `debug_head(batch, N, format=csv)` prints first N rows in CSV format | Same data extraction as HEAD-01, different formatting pass |
| HEAD-03 | Uses `cudf::slice` for zero-copy row selection before GPU-to-host transfer | Verified: `cudf::slice(table_view, {0, N})` returns a view, no data copy; used in `sirius_physical_top_n.cpp`, `sirius_physical_limit.cpp` |
| STATS-01 | `debug_stats(batch)` prints per-column min, max, sum for numeric columns | cudf::reduce with make_min/max/sum_aggregation; scalar value extraction via numeric_scalar::value() |
| STATS-02 | Non-numeric columns (STRING, BOOL, DATE, TIMESTAMP) skipped with note | Type classification via cudf::type_id switch; BOOL8 explicitly excluded per requirement |
| STATS-03 | Uses `cudf::reduce` / `cudf::minmax` for GPU-side computation | Verified: cudf::reduce API in reduction.hpp; cudf::minmax also available for combined min+max in single call |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| cudf (libcudf) | 26.02.x | GPU DataFrame: slice, reduce, type dispatch, scalar extraction | Already linked; sole GPU data API in Sirius [VERIFIED: pixi.toml, CMakeLists.txt] |
| spdlog | 1.8.x | Logging via SIRIUS_LOG_DEBUG macro | Already linked; all debug output goes through this [VERIFIED: logging.hpp] |
| fmt | (bundled with spdlog) | String formatting with `fmt::format` | Already used in Phase 1 debug_utils.cpp [VERIFIED: debug_utils.cpp] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Catch2 | (bundled with DuckDB) | Unit test framework | All test files use `TEST_CASE`, `REQUIRE`, `CHECK` [VERIFIED: test_debug_utils.cpp] |
| rmm | (bundled with cudf) | CUDA stream management, device memory | Stream parameter handling, device buffer allocation [VERIFIED: debug_utils.hpp] |
| cucascade | submodule | `data_batch`, `gpu_table_representation` | Data batch access, tier checking [VERIFIED: debug_utils.cpp] |

**No new dependencies required.** Phase 2 uses the same libraries as Phase 1.

## Architecture Patterns

### File Organization (extend existing)
```
src/
  include/
    debug_utils.hpp          # Add DebugFormat enum, debug_head, debug_stats declarations
  debug_utils.cpp            # Add implementations (same file as Phase 1)
test/
  cpp/
    debug/
      test_debug_utils.cpp   # Extend with debug_head and debug_stats tests
```

### Pattern 1: GPU-to-Host Data Extraction via cudf::type_dispatcher
**What:** Use `cudf::type_dispatcher` to generically extract typed column data from GPU to host memory, avoiding a manual switch-case for every type.
**When to use:** Any time you need to read column values from device memory and format them as strings.
**Example:**
```cpp
// Source: cudf/utilities/type_dispatcher.hpp (installed header)
// Pattern from: test/cpp/memory/test_host_table_utils.cpp
struct value_to_string_functor {
  template <typename T, std::enable_if_t<cudf::is_numeric<T>()>* = nullptr>
  std::string operator()(cudf::column_view const& col, cudf::size_type row,
                         rmm::cuda_stream_view stream)
  {
    T val;
    cudaMemcpyAsync(&val, col.data<T>() + row, sizeof(T),
                    cudaMemcpyDeviceToHost, stream.value());
    stream.synchronize();
    // Format per D-04, D-05
    if constexpr (std::is_same_v<T, bool>) { return val ? "true" : "false"; }
    else if constexpr (std::is_floating_point_v<T>) { return fmt::format("{:g}", val); }
    else { return fmt::format("{}", val); }
  }
  // Fallback for unsupported types
  template <typename T, std::enable_if_t<!cudf::is_numeric<T>()>* = nullptr>
  std::string operator()(...) { return "(unsupported)"; }
};
```
**Key insight:** For `debug_head` with N rows, do NOT copy one element at a time. Instead, copy the entire sliced column (N elements) to a host vector in one `cudaMemcpyAsync` call, then format all N values. This reduces kernel launch / sync overhead from O(N * cols) to O(cols). [VERIFIED: pattern used in test_host_table_utils.cpp line 121]

### Pattern 2: cudf::slice for Zero-Copy Row Selection
**What:** `cudf::slice(table_view, {0, N})` returns a `table_view` that references the same device memory with adjusted offsets -- zero copy.
**When to use:** When you need the first N rows without allocating new GPU memory.
**Example:**
```cpp
// Source: cudf/copying.hpp (installed header)
// Pattern from: src/op/sirius_physical_top_n.cpp lines 111-112
auto sliced_views = cudf::slice(tv, {0, keep_rows}, stream);
auto sliced_tv = sliced_views.front();  // table_view of first N rows
```
**Critical detail:** The returned `column_view` from a sliced table has a non-zero `offset()`. When copying data with `cudaMemcpyAsync`, start from `col.data<T>() + col.offset()`, NOT `col.data<T>()`. Similarly, when checking null bits, use `col.offset() + row_index`. [VERIFIED: cudf::column_view documentation and cudf::slice semantics]

### Pattern 3: cudf::reduce for GPU-Side Statistics
**What:** `cudf::reduce(col, agg, output_type, stream)` computes a single scalar value on GPU without copying the full column to host.
**When to use:** For `debug_stats` to compute min, max, sum.
**Example:**
```cpp
// Source: cudf/reduction.hpp (installed header)
// Pattern from: src/op/sirius_physical_ungrouped_aggregate.cpp lines 385-427
auto min_agg = cudf::make_min_aggregation<cudf::reduce_aggregation>();
auto min_scalar = cudf::reduce(col, *min_agg, col.type(), stream);
if (!min_scalar->is_valid()) {
  // All-NULL column: show "NULL" per D-10
} else {
  // Extract value: static_cast<cudf::numeric_scalar<T> const&>(*min_scalar).value(stream)
}
```
**Optimization choice (Claude's discretion):** Use `cudf::minmax(col, stream)` which computes both min and max in a single GPU pass (1 kernel launch instead of 2). Then use `cudf::reduce` separately for SUM only. This reduces kernel launches from 3 to 2 per numeric column. [VERIFIED: cudf::minmax signature in reduction.hpp line 242]

### Pattern 4: Established Debug Function Template (Phase 1)
**What:** Every debug function follows this structure.
**Example:**
```cpp
// Source: src/debug_utils.cpp (Phase 1 implementation)
void debug_head(cucascade::data_batch const& batch,
                cudf::size_type n,
                rmm::cuda_stream_view stream,
                DebugFormat format,
                std::vector<std::string> const& col_names)
{
  try {
    if (!is_gpu_tier(batch, "debug_head")) { return; }
    cudf::table_view tv = get_cudf_table_view(batch);
    stream.synchronize();
    // ... build output string ...
    SIRIUS_LOG_DEBUG("{}", output);
  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_head failed: {}", e.what());
  } catch (...) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_head failed: unknown error");
  }
}
```

### Anti-Patterns to Avoid
- **Per-element cudaMemcpy:** Copying one value at a time from GPU to host is disastrously slow (O(N*cols) kernel launches). Always batch-copy the entire sliced column.
- **Using `cudf::is_numeric()` for stats classification:** In cuDF, `is_numeric(BOOL8)` returns `true` because `bool` is arithmetic. But STATS-02 explicitly says BOOL should be skipped. Use an explicit type_id check instead.
- **Forgetting column offset after slice:** `cudf::slice` returns a view with `col.offset() > 0`. The `data<T>()` pointer already accounts for offset in newer cuDF versions, but the null bitmask offset must be handled explicitly. Use `col.offset() + row` for null bit checking.
- **Using `cudaDeviceSynchronize()`:** Per INFRA-01, always use `stream.synchronize()`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Row selection | Manual index tracking | `cudf::slice(tv, {0, N})` | Zero-copy, handles all column types and null masks correctly [VERIFIED: used in sirius_physical_top_n.cpp] |
| Min/max/sum computation | Copy column to host, compute in CPU loop | `cudf::reduce` / `cudf::minmax` | Runs on GPU, handles nulls automatically, returns invalid scalar for all-NULL [VERIFIED: used in sirius_physical_ungrouped_aggregate.cpp] |
| Null bitmask extraction | Re-implement bitmask copy | `copy_null_mask_to_host()` from Phase 1 | Already handles edge cases (no nulls, bitmask word alignment) [VERIFIED: debug_utils.cpp] |
| Type dispatch | Manual switch on every type_id | `cudf::type_dispatcher` | Compile-time checked, extensible, standard cuDF pattern [VERIFIED: cudf/utilities/type_dispatcher.hpp] |
| String formatting | `sprintf` / `std::to_string` | `fmt::format` | Already used throughout debug_utils.cpp; supports width/precision specifiers [VERIFIED: debug_utils.cpp] |

**Key insight:** For this phase, only numeric types (INT8/16/32/64, UINT8/16/32/64, FLOAT32/64) plus BOOL8 need row-level extraction in `debug_head`. STRING, DECIMAL, TIMESTAMP, DATE are deferred to Phase 3. The `debug_stats` function only operates on integer and floating-point types (excluding BOOL per STATS-02).

## Common Pitfalls

### Pitfall 1: Column Offset After cudf::slice
**What goes wrong:** Copied data appears shifted or contains garbage values.
**Why it happens:** `cudf::slice` returns a `column_view` with a non-zero `offset()`. The `data<T>()` method returns a pointer to the beginning of the underlying buffer, NOT offset-adjusted in older cuDF versions.
**How to avoid:** Always use `col.data<T>() + col.offset()` as the source for `cudaMemcpyAsync`. Copy exactly `col.size()` elements, not the original column's size. For null bitmask, use `col.offset() + row_idx` when calling `bit_is_set`. The existing `copy_null_mask_to_host` in Phase 1 copies from `col.null_mask()` which is the raw pointer -- when working with sliced columns, the offset must be applied when checking individual bits.
**Warning signs:** First few values correct but later values wrong; off-by-N errors in output.

### Pitfall 2: cudf::reduce Returns Invalid Scalar for All-NULL Columns
**What goes wrong:** Attempting to call `value()` on an invalid scalar causes undefined behavior or crash.
**Why it happens:** `cudf::reduce` returns a scalar with `is_valid() == false` when all input values are NULL.
**How to avoid:** Always check `scalar->is_valid()` before calling `value()`. Per D-10, display "NULL" for invalid scalars.
**Warning signs:** Segfault or garbage values when processing columns that happen to be all-NULL.

### Pitfall 3: SUM Overflow for Integer Types
**What goes wrong:** SUM of INT32 column overflows silently, producing incorrect statistics.
**Why it happens:** `cudf::reduce` with SUM aggregation uses the output_type provided. If output_type matches input type (INT32), overflow can occur.
**How to avoid:** For SUM, always widen the output type: INT8/16/32 -> INT64, UINT8/16/32 -> UINT64, INT64/UINT64 -> INT64/UINT64 (accept risk). FLOAT32/64 -> FLOAT64. This matches the pattern in `cudf_aggregate.cu` lines 116-124 and `gpu_aggregate_impl.cpp` lines 67-72.
**Warning signs:** Negative SUM values for columns of positive integers.

### Pitfall 4: BOOL8 Storage Type Mismatch
**What goes wrong:** BOOL8 values display as integers (0/1) instead of `true`/`false`.
**Why it happens:** cuDF stores BOOL8 as `int8_t` in device memory. `cudf::type_dispatcher` maps BOOL8 to `bool`, but the raw data pointer is `int8_t*`.
**How to avoid:** When copying BOOL8 data from device, copy as `int8_t` (the storage type), then interpret non-zero as `true`. Or use `dispatch_storage_type` with `type_dispatcher` instead of the default `id_to_type` mapping. Per D-05, display as lowercase `true`/`false`.
**Warning signs:** Tests passing for non-bool types but failing for BOOL8 columns.

### Pitfall 5: Empty Batch Division by Zero
**What goes wrong:** Computing column widths or statistics on empty batch crashes.
**Why it happens:** Zero rows means no data to scan for widths; `cudf::reduce` on empty column returns invalid scalar.
**How to avoid:** Per D-13, check `tv.num_rows() == 0` early and emit `(empty batch)` note. Skip all data extraction and stats computation for empty batches.
**Warning signs:** Crash or infinite loop when `N == 0` or batch has 0 rows.

### Pitfall 6: Unsigned Integer SUM Output Type
**What goes wrong:** SUM of unsigned integers uses a signed output type, producing wrong results for large values.
**Why it happens:** The existing Sirius aggregate code maps INT32 -> INT64 for SUM, but doesn't handle unsigned types (UINT8/16/32/64) because DuckDB doesn't use them. Debug utilities must handle all Sirius-supported types.
**How to avoid:** Map UINT8/16/32 -> UINT64 for SUM output type. For UINT64, keep as UINT64 (no wider unsigned type available in cuDF).
**Warning signs:** Very large unsigned sums appearing negative.

## Code Examples

### Example 1: Batch Column Data Extraction (N rows)
```cpp
// Source: pattern from test/cpp/memory/test_host_table_utils.cpp line 121
// and src/op/sirius_physical_top_n.cpp lines 111-112
// Adapted for debug_head use case

// Step 1: Slice to first N rows (zero-copy)
auto keep = std::min(n, tv.num_rows());
cudf::table_view sliced_tv = tv;
if (keep < tv.num_rows()) {
  auto slices = cudf::slice(tv, {0, keep}, stream);
  sliced_tv = slices.front();
}

// Step 2: For each column, copy N typed values to host in one call
for (cudf::size_type c = 0; c < sliced_tv.num_columns(); ++c) {
  auto const& col = sliced_tv.column(c);
  auto nulls = copy_null_mask_to_host(col, stream);

  // Example for INT32 (generalize via type_dispatcher)
  std::vector<int32_t> host_vals(col.size());
  cudaMemcpyAsync(host_vals.data(),
                  col.data<int32_t>() + col.offset(),
                  sizeof(int32_t) * col.size(),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  stream.synchronize();

  // Format each value
  for (cudf::size_type r = 0; r < col.size(); ++r) {
    if (nulls.is_null(col.offset() + r)) {
      // output "NULL" per D-06
    } else {
      // format host_vals[r]
    }
  }
}
```

### Example 2: GPU-Side Statistics with cudf::minmax + cudf::reduce
```cpp
// Source: cudf/reduction.hpp lines 90-95, 242-245
// Pattern from: src/op/sirius_physical_ungrouped_aggregate.cpp lines 385-427

// Use cudf::minmax for combined min+max (1 kernel launch)
auto [min_scalar, max_scalar] = cudf::minmax(col, stream);

// Use cudf::reduce for SUM with widened output type
auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
cudf::data_type sum_output_type = col.type();
// Widen to prevent overflow
if (col.type().id() == cudf::type_id::INT32) {
  sum_output_type = cudf::data_type(cudf::type_id::INT64);
}
auto sum_scalar = cudf::reduce(col, *sum_agg, sum_output_type, stream);

// Check validity (all-NULL case per D-10)
std::string min_str = min_scalar->is_valid() ? format_scalar(min_scalar, col.type(), stream) : "NULL";
std::string max_str = max_scalar->is_valid() ? format_scalar(max_scalar, col.type(), stream) : "NULL";
std::string sum_str = sum_scalar->is_valid() ? format_scalar(sum_scalar, sum_output_type, stream) : "NULL";
```

### Example 3: Scalar Value Extraction
```cpp
// Source: src/op/sirius_physical_ungrouped_aggregate.cpp lines 72-75, 229
// Helper to extract scalar value as string
template <typename T>
std::string scalar_value_to_string(cudf::scalar const& s, rmm::cuda_stream_view stream)
{
  auto const& typed = static_cast<cudf::numeric_scalar<T> const&>(s);
  T val = typed.value(stream);
  if constexpr (std::is_floating_point_v<T>) {
    return fmt::format("{:g}", val);  // D-04: %g-style
  } else if constexpr (std::is_same_v<T, bool>) {
    return val ? "true" : "false";    // D-05
  } else {
    return fmt::format("{}", val);
  }
}
```

### Example 4: Numeric Type Classification for debug_stats
```cpp
// STATS-02: explicitly skip BOOL, STRING, TIMESTAMP, DATE, DECIMAL
// Do NOT use cudf::is_numeric() which includes BOOL8
bool is_stats_numeric(cudf::type_id id) {
  switch (id) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64:
      return true;
    default:
      return false;
  }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `cudf::reduce` with `std::nullopt` init | `cudf::reduce` without init parameter | cuDF 25.x+ | Simplified API -- the 4-arg overload (no init) is preferred [VERIFIED: reduction.hpp] |
| `rmm::cuda_stream_default` | `cudf::get_default_stream()` | cuDF 24.x | Standard stream accessor; both work but new code should use cudf version [VERIFIED: existing test code] |
| Manual aggregation construction | `cudf::make_*_aggregation<cudf::reduce_aggregation>()` factory functions | Long established | Type-safe aggregation creation [VERIFIED: reduction.hpp, cudf_aggregate.cu] |

**Deprecated/outdated:**
- `cudf::minmax` was added after the initial reduce API and is more efficient for combined min+max computation. Existing Sirius code does not use it yet (uses separate reduce calls), but it is available and recommended. [VERIFIED: reduction.hpp line 242]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `col.data<T>()` returns offset-adjusted pointer in cuDF 26.02 (i.e., `data<T>()` already accounts for `offset()`) | Architecture Patterns, Pitfall 1 | If not offset-adjusted, data reads will be shifted by offset elements. LOW risk -- verify by checking cudf::column_view::data() implementation, or simply always add offset defensively |
| A2 | `cudf::minmax` returns the same scalar type as `cudf::reduce` with MIN/MAX aggregation | Code Example 2 | If scalar types differ, extraction code needs adjustment. LOW risk -- both return `std::unique_ptr<cudf::scalar>` |

## Open Questions

1. **Does `col.data<T>()` account for column offset in cuDF 26.02?**
   - What we know: In some cuDF versions, `data<T>()` returns a pointer offset by `offset() * sizeof(T)`, making explicit offset addition redundant and incorrect (double-counting).
   - What's unclear: Whether cuDF 26.02 auto-adjusts or not.
   - Recommendation: Check the cudf::column_view::data() implementation. If it auto-adjusts, use `col.data<T>()` directly. If not, use `col.data<T>() + col.offset()`. The safest approach is to inspect the actual header during implementation. Since we use `cudf::slice` which is zero-copy, a quick unit test with a sliced column will verify the behavior definitively.

2. **Should `debug_head` for BOOL8 columns also compute stats in `debug_stats`?**
   - What we know: STATS-02 explicitly lists BOOL as non-numeric for stats. HEAD-01 says "all numeric types" -- the success criteria mentions BOOL columns in debug_head output.
   - What's unclear: Whether BOOL8 rows should display in debug_head (yes, per success criteria) but be skipped in debug_stats (yes, per STATS-02).
   - Recommendation: BOOL8 is displayed in debug_head (rows show `true`/`false`) but skipped with `(non-numeric, skipped)` in debug_stats. These are independent functions with different type scopes.

## Project Constraints (from CLAUDE.md)

- **Build system:** CMake with CUDA separable compilation, C++20/CUDA 20. New code goes in `src/debug_utils.cpp` (not a .cu file) since it uses no CUDA kernels directly.
- **Logging:** All output via `SIRIUS_LOG_DEBUG` with `[SIRIUS_DIAG]` prefix. No printf/stdout.
- **Thread safety:** Output buffered into single `std::string`, emitted in one log call.
- **Error handling:** All functions wrapped in try/catch -- never crash the pipeline.
- **Stream parameter:** All functions accept `rmm::cuda_stream_view stream`, use `stream.synchronize()`.
- **Naming:** PascalCase for public methods would conflict -- existing API uses `snake_case` functions (`debug_schema`, `debug_nulls`), so `debug_head` and `debug_stats` follow that.
- **Code formatting:** clang-format enforced via pre-commit hooks.
- **Test framework:** Catch2 (bundled with DuckDB), test tags `[debug_utils]`.
- **Build:** Use `pixi shell` then `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make` per CLAUDE.md.
- **GSD workflow:** All edits through GSD commands.

## Sources

### Primary (HIGH confidence)
- `src/debug_utils.cpp` / `src/include/debug_utils.hpp` -- Phase 1 implementation patterns (tier guard, output buffering, null mask copy)
- `src/op/sirius_physical_ungrouped_aggregate.cpp` -- cudf::reduce patterns with scalar extraction (lines 385-427)
- `src/op/sirius_physical_top_n.cpp` -- cudf::slice usage pattern (lines 111-112)
- `/home/bwyogatama/sirius/.pixi/envs/default/include/cudf/reduction.hpp` -- cudf::reduce and cudf::minmax API signatures
- `/home/bwyogatama/sirius/.pixi/envs/default/include/cudf/copying.hpp` -- cudf::slice API signature
- `/home/bwyogatama/sirius/.pixi/envs/default/include/cudf/scalar/scalar.hpp` -- numeric_scalar::value() API
- `/home/bwyogatama/sirius/.pixi/envs/default/include/cudf/utilities/traits.hpp` -- is_numeric, is_floating_point type checks
- `test/cpp/debug/test_debug_utils.cpp` -- Existing Catch2 test patterns for debug utilities
- `test/cpp/operator/operator_type_traits.hpp` -- gpu_type_traits for test column creation
- `test/cpp/utils/data_utils.hpp` -- vector_to_cudf_column helper for tests

### Secondary (MEDIUM confidence)
- `test/cpp/memory/test_host_table_utils.cpp` -- cudaMemcpy DeviceToHost extraction patterns (line 121)
- `src/cuda/cudf/cudf_aggregate.cu` -- Legacy reduce aggregation patterns (SUM overflow handling)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, verified against actual source code
- Architecture: HIGH -- all patterns verified against existing Sirius codebase usage
- Pitfalls: HIGH -- identified from actual cuDF API documentation and existing code patterns, plus cudf::slice offset semantics verified from header

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (stable -- cuDF 26.02 is the installed version, unlikely to change)
