# Phase 4: Diff, Sampling, and Skill Integration - Research

**Researched:** 2026-04-08
**Domain:** C++20/CUDA debug utilities (batch comparison, random sampling), Claude Code skill documentation
**Confidence:** HIGH

## Summary

Phase 4 adds two new debug utility functions (`debug_diff` and `debug_sample`) to the existing `debug_utils.hpp`/`.cpp` module, and updates two Claude Code skill files (`/validate` and `/runtime-errors`) to reference the complete debug utility API. All implementation follows established patterns from Phases 1-3 -- tier guards, try/catch wrapping, `[SIRIUS_DIAG]` log prefix, single-string buffered output via `SIRIUS_LOG_DEBUG`.

`debug_diff` performs host-side element-by-element comparison of two `data_batch` objects after copying both to host memory. Schema validation (column count, types) occurs first, followed by row count comparison, then per-column value comparison up to a configurable row limit (default 10M rows) to prevent OOM. `debug_sample` generates random row indices on the host using `std::mt19937`, uses `cudf::gather` to select those rows from the GPU batch, then formats output using the same pipeline as `debug_head` (cell extraction, width computation, aligned/CSV formatting).

**Primary recommendation:** Implement both functions in `src/debug_utils.cpp` extending the existing type dispatch and formatting infrastructure. Extract the `debug_head` formatting pipeline into a shared helper that both `debug_head` and `debug_sample` can call, avoiding code duplication of the cell extraction + output formatting logic (~200 lines).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `debug_diff` reports per-column diff count + first N differing row indices. Format: `col[0] diffs: 3/1000 rows [idx: 42, 187, 501]`
- **D-02:** Number of differing row indices shown per column is configurable via `max_diff_rows` parameter with default 10
- **D-03:** Row count limit defaults to 10,000,000 rows. Batches exceeding this log a warning and skip value comparison (DIFF-05). This is a host memory guard -- both batches are copied to host for comparison
- **D-04:** Comparison is host-side: copy both batches to host, then compare element-by-element in C++. Simpler code, full control over per-type comparison logic
- **D-05:** Exact equality for all types including FLOAT32/FLOAT64 -- no epsilon tolerance. Debug tool should catch every bit flip; developers understand GPU rounding
- **D-06:** Schema mismatch check first (column count, types) before any value comparison (DIFF-02). Row count mismatch also reported before values (DIFF-03)
- **D-07:** `debug_sample` generates random row indices on host via `std::mt19937`, then uses `cudf::gather` to extract those rows from GPU. No cuRAND dependency
- **D-08:** Optional `seed` parameter. Default uses `std::random_device` for different rows each call. Caller can pass explicit seed for reproducible sampling and unit tests
- **D-09:** Output uses the same formatting as `debug_head` -- aligned columns or CSV, same `DebugFormat` enum and `max_string_len` parameter
- **D-10:** Both `/validate` and `/runtime-errors` SKILL.md get a "Debug Utilities" section with full function signatures, parameter descriptions, and 2-3 usage examples per function
- **D-11:** `/validate` Phase 2 replaces existing ad-hoc `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] operator_name checksum: sum={}, max={}, first_row={}", ...)` patterns with `debug_checksum`, `debug_stats`, `debug_head` calls
- **D-12:** `/runtime-errors` references `debug_schema`, `debug_head`, `debug_nulls` for data inspection at suspected fault points

### Claude's Discretion
- Schema mismatch error message wording
- Internal helper function decomposition for host-side comparison
- Whether `debug_sample` clamps N to batch size or returns fewer rows silently (follow debug_head pattern: clamp silently per D-12 of Phase 2)
- Whether debug_diff header line includes batch_id comparison
- Skill documentation section placement within existing SKILL.md structure

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DIFF-01 | `debug_diff(batch_a, batch_b)` compares two data batches and logs which rows and columns differ | Host-side comparison using existing type dispatch and null mask copy patterns from `debug_head` |
| DIFF-02 | Reports schema mismatches (different column count, types) before attempting value comparison | Use `cudf::table_view::num_columns()`, `column_view::type()` for schema checks |
| DIFF-03 | Reports row count mismatch | `table_view::num_rows()` comparison before value loop |
| DIFF-04 | For matching schemas, reports per-column diff count and first N differing row indices | Per-column host-side loop with counter + bounded index collection (D-01 output format) |
| DIFF-05 | Guards behind configurable row count limit to prevent OOM on large batches | `max_rows` parameter defaulting to 10M; warn+return when exceeded (D-03) |
| SAMPLE-01 | `debug_sample(batch, N)` prints N randomly selected rows from the batch | `std::mt19937` for index generation, `cudf::gather` for row extraction |
| SAMPLE-02 | Uses the same output formatting as `debug_head` (aligned columns + CSV options) | Extract shared formatting helper from `debug_head` internals |
| SAMPLE-03 | Useful for catching bugs that don't appear in first rows | Random seed via `std::random_device` by default (D-08) |
| SKILL-01 | `/validate` SKILL.md references debug utilities with named function calls | Replace Phase 2 ad-hoc `SIRIUS_LOG_TRACE` patterns with `debug_checksum`/`debug_stats`/`debug_head` |
| SKILL-02 | `/runtime-errors` SKILL.md references debug utilities for data inspection | Add `debug_schema`/`debug_head`/`debug_nulls` references at fault inspection points |
| SKILL-03 | Both skills document the function signatures and usage examples | Full signatures from `debug_utils.hpp` + 2-3 usage examples per function |
</phase_requirements>

## Standard Stack

No new libraries or dependencies are introduced in this phase. All implementation uses existing includes already present in `debug_utils.cpp`.

### Core (already available)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| cudf | 26.02.x | `cudf::gather` for random row selection, `cudf::table_view` for schema comparison | Already linked, gather API used extensively in join/sort operators [VERIFIED: codebase grep] |
| C++ `<random>` | C++20 stdlib | `std::mt19937`, `std::random_device`, `std::uniform_int_distribution` for index generation | C++20 required by build; `std::mt19937_64` already used in `test/cpp/utils/utils.cpp` [VERIFIED: codebase grep] |
| spdlog/fmt | 1.8.x | `SIRIUS_LOG_DEBUG` output, `fmt::format` for string formatting | Already included and used throughout `debug_utils.cpp` [VERIFIED: codebase grep] |

### No New Dependencies
All needed headers are already included in `debug_utils.cpp`:
- `<cudf/copying.hpp>` -- contains `cudf::gather` and `cudf::slice`
- `<cudf/types.hpp>` -- `cudf::type_id`, `cudf::data_type`, `cudf::size_type`
- `<cudf/table/table_view.hpp>` -- `cudf::table_view`
- `<cuda_runtime.h>` -- `cudaMemcpyAsync`

Only new include needed: `<random>` for `std::mt19937` / `std::random_device` / `std::uniform_int_distribution`.

## Architecture Patterns

### Recommended Approach

```
src/include/debug_utils.hpp    -- Add debug_diff and debug_sample declarations
src/debug_utils.cpp            -- Add implementations + extract shared formatting helper
test/cpp/debug/test_debug_utils.cpp  -- Add ~10-12 new test cases
.claude/skills/validate/SKILL.md     -- Add Debug Utilities section, update Phase 2
.claude/skills/runtime-errors/SKILL.md -- Add Debug Utilities section, update Phase 2
```

### Pattern 1: Shared Formatting Helper (Code Deduplication)

**What:** Extract the cell-extraction + formatting pipeline from `debug_head` into a reusable internal helper, so `debug_sample` can call the same code without duplicating ~200 lines.

**When to use:** Both `debug_head` and `debug_sample` need to: (1) copy column data to host, (2) format each cell as a string, (3) compute column widths, (4) output in ALIGNED or CSV format.

**Example:**
```cpp
// Source: Extracted from existing debug_head (lines 521-840 of debug_utils.cpp)
namespace {

// Shared helper: format a table_view's rows into log output.
// Called by both debug_head (after cudf::slice) and debug_sample (after cudf::gather).
void format_rows_to_output(
    std::string& output,
    cudf::table_view const& tv,
    rmm::cuda_stream_view stream,
    DebugFormat format,
    std::vector<std::string> const& col_names,
    cudf::size_type max_string_len)
{
    // ... cell extraction, width computation, output formatting
    // (the ~200 lines currently inside debug_head between cudf::slice and SIRIUS_LOG_DEBUG)
}

}  // namespace
```
[ASSUMED -- internal decomposition is at Claude's discretion per CONTEXT.md]

### Pattern 2: debug_diff Host-Side Comparison Loop

**What:** Schema check first, then row count check, then per-column element-by-element comparison on host.

**When to use:** Comparing two `data_batch` objects for value differences.

**Example:**
```cpp
// Source: Based on D-01 through D-06 from CONTEXT.md
// Pseudocode for the comparison loop:
for (each column c) {
    auto nulls_a = copy_null_mask_to_host(col_a, stream);
    auto nulls_b = copy_null_mask_to_host(col_b, stream);
    // Copy both column data to host (same pattern as debug_head extract_numeric)
    int diff_count = 0;
    std::vector<cudf::size_type> diff_indices;
    for (row r = 0; r < num_rows; ++r) {
        bool a_null = nulls_a.is_null(col_a.offset() + r);
        bool b_null = nulls_b.is_null(col_b.offset() + r);
        if (a_null != b_null || (!a_null && !b_null && values_a[r] != values_b[r])) {
            diff_count++;
            if (diff_indices.size() < max_diff_rows) {
                diff_indices.push_back(r);
            }
        }
    }
    // Output: "col[0] diffs: 3/1000 rows [idx: 42, 187, 501]"
}
```
[VERIFIED: Pattern follows existing `debug_head` host-side extraction in `debug_utils.cpp` lines 521-783]

### Pattern 3: debug_sample Random Index Generation + cudf::gather

**What:** Generate sorted random indices on host, build a device column from them, call `cudf::gather` to extract rows.

**When to use:** Selecting N random rows from a GPU batch.

**Example:**
```cpp
// Source: D-07, D-08 from CONTEXT.md; cudf::gather usage from src/op/sirius_physical_top_n.cpp:76
std::mt19937 gen(seed.has_value() ? *seed : std::random_device{}());
std::uniform_int_distribution<cudf::size_type> dist(0, num_rows - 1);
std::vector<cudf::size_type> indices(keep);
for (auto& idx : indices) { idx = dist(gen); }
std::sort(indices.begin(), indices.end());  // sorted for locality

// Copy indices to GPU column
rmm::device_buffer dev_indices(indices.data(), keep * sizeof(cudf::size_type), stream);
auto indices_col = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32}, keep,
    std::move(dev_indices), rmm::device_buffer{}, 0);

// Gather selected rows
auto gathered = cudf::gather(tv, indices_col->view(),
    cudf::out_of_bounds_policy::DONT_CHECK, stream);
```
[VERIFIED: `cudf::gather` signature from `src/op/sirius_physical_top_n.cpp` line 76-77]

### Anti-Patterns to Avoid
- **Duplicating the entire debug_head formatting pipeline in debug_sample:** Extract the shared portion into a helper instead. Currently ~200 lines of cell extraction and formatting code.
- **Using cuRAND for random index generation:** Decision D-07 explicitly excludes cuRAND. Use `std::mt19937` on host.
- **Using epsilon tolerance for float comparison in debug_diff:** Decision D-05 requires exact bitwise equality. No tolerance.
- **Comparing values on GPU with cudf::binaryop:** Decision D-04 specifies host-side comparison for simplicity and per-type control.
- **Forgetting to handle STRING comparison specially in debug_diff:** STRING data requires the two-buffer (offsets + chars) extraction pattern from `debug_head`, then string comparison on host.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Random row selection from GPU table | Manual CUDA kernel for row extraction | `cudf::gather(tv, indices_col->view(), ...)` | Handles all types, null masks, string columns automatically [VERIFIED: existing usage in top_n, hash_join, order operators] |
| Zero-copy row slicing | Manual pointer arithmetic | `cudf::slice(tv, {start, end}, stream)` | Already used in `debug_head` for first-N rows [VERIFIED: debug_utils.cpp line 506] |
| Type name to string | Manual type_id switch | `cudf::type_to_name(col.type())` | Already used in `debug_schema` and `debug_stats` [VERIFIED: debug_utils.cpp line 405] |
| Null mask host copy | Manual bitmask arithmetic | `copy_null_mask_to_host(col, stream)` + `host_column_nulls::is_null(row)` | Already implemented in Phase 1 [VERIFIED: debug_utils.hpp line 28-46] |
| Formatted output buffering | Manual string concatenation | `fmt::format(...)` into `std::string` + single `SIRIUS_LOG_DEBUG` | Established pattern across all debug functions [VERIFIED: debug_utils.cpp] |

**Key insight:** The existing `debug_head` implementation already contains 90% of the infrastructure needed for both `debug_diff` (host-side type dispatch, null handling, per-type value extraction) and `debug_sample` (cell formatting, aligned/CSV output). The main engineering work is refactoring to share this code, not writing new extraction/formatting logic.

## Common Pitfalls

### Pitfall 1: STRING Column Comparison in debug_diff
**What goes wrong:** STRING values stored as offsets + chars buffers cannot be compared element-by-element with `memcmp` -- different strings can have different lengths, and the offset array must be decoded first.
**Why it happens:** All other types have fixed-width elements that can be compared with `==`, but STRING is variable-length.
**How to avoid:** Use the existing two-buffer extraction pattern from `debug_head` (lines 588-627): copy offsets, compute char ranges, copy chars, reconstruct `std::string` per row, then use `std::string::operator==` for comparison.
**Warning signs:** Garbage comparisons, always showing 100% diffs on string columns.

### Pitfall 2: DECIMAL Comparison Requires Same Scale
**What goes wrong:** Two DECIMAL columns with different scales (e.g., DECIMAL(10,2) vs DECIMAL(10,4)) have different raw integer representations of the same logical value.
**How to avoid:** Schema mismatch check (D-06) compares `col.type()` which includes scale. If types don't match exactly, it's reported as a schema mismatch before value comparison begins. The raw integer comparison is correct only when scales match.
**Warning signs:** False diffs on DECIMAL columns that should be equal.

### Pitfall 3: cudf::gather Requires INT32 Index Column
**What goes wrong:** `cudf::gather` expects the gather map (indices column) to have type `INT32`. Passing `INT64` or unsigned types causes a cudf exception.
**Why it happens:** cudf row indices are `cudf::size_type` which is `int32_t`.
**How to avoid:** Generate random indices as `cudf::size_type` (int32_t), create the device column with `cudf::type_id::INT32`.
**Warning signs:** cudf exception about invalid gather map type.

### Pitfall 4: Random Index Duplicates
**What goes wrong:** When N is close to the batch row count, `std::uniform_int_distribution` can produce duplicate indices. `cudf::gather` with duplicates works fine (it just copies the same row twice), but the output may show repeated rows.
**Why it happens:** Random sampling with replacement is the simplest approach.
**How to avoid:** This is acceptable behavior for a debug utility. If N >= num_rows, just display all rows (clamp to num_rows, like `debug_head`). For N << num_rows, duplicates are rare and harmless. Document that sampling is with replacement.
**Warning signs:** Seeing the same row values repeated in sample output when N is large relative to batch size.

### Pitfall 5: Row Limit Guard Must Check BOTH Batches
**What goes wrong:** The 10M row limit (D-03) is a host memory guard. If only one batch is checked, the other could still cause OOM.
**Why it happens:** Developer checks `batch_a.num_rows() > max_rows` but forgets `batch_b`.
**How to avoid:** Check `std::max(num_rows_a, num_rows_b) > max_rows` and skip value comparison for both if either exceeds the limit.
**Warning signs:** OOM when one small batch is compared against a huge batch.

### Pitfall 6: col.offset() for Sliced Column Views
**What goes wrong:** When comparing values from sliced column views, `col.data<T>()` is offset-adjusted in cuDF 26.02, but `null_mask()` is NOT offset-adjusted.
**Why it happens:** cuDF 26.02 changed the behavior of `data<T>()` to account for offset.
**How to avoid:** Follow the existing pattern in `debug_head`: use `col.data<T>()` directly for value access, but use `col.offset() + r` for null mask checks via `is_null()`. This pattern is already correctly implemented and tested.
**Warning signs:** Null flags misaligned with actual data positions.

## Code Examples

### debug_diff Function Signature
```cpp
// Source: Based on D-01 through D-06 from CONTEXT.md
void debug_diff(cucascade::data_batch const& batch_a,
                cucascade::data_batch const& batch_b,
                rmm::cuda_stream_view stream,
                cudf::size_type max_diff_rows = 10,
                cudf::size_type max_rows = 10'000'000,
                std::vector<std::string> const& col_names = {});
```
[ASSUMED -- exact parameter ordering is implementer's choice]

### debug_sample Function Signature
```cpp
// Source: Based on D-07 through D-09 from CONTEXT.md
void debug_sample(cucascade::data_batch const& batch,
                  cudf::size_type n,
                  rmm::cuda_stream_view stream,
                  DebugFormat format = DebugFormat::ALIGNED,
                  std::vector<std::string> const& col_names = {},
                  cudf::size_type max_string_len = 50,
                  std::optional<uint64_t> seed = std::nullopt);
```
[ASSUMED -- exact parameter ordering is implementer's choice; seed type could be uint32_t or uint64_t]

### cudf::gather Usage for debug_sample
```cpp
// Source: Verified from src/op/sirius_physical_top_n.cpp:76-77
// cudf::gather signature: gather(table_view, column_view, out_of_bounds_policy, stream, mr)
auto gathered = cudf::gather(
    tv,
    indices_col->view(),
    cudf::out_of_bounds_policy::DONT_CHECK,
    stream);
// gathered is std::unique_ptr<cudf::table>
// Use gathered->view() for the formatting pipeline
```
[VERIFIED: exact API from codebase usage]

### Host-Side Type Comparison Pattern for debug_diff
```cpp
// Source: Adapted from debug_head extract_numeric pattern (debug_utils.cpp line 532-553)
auto compare_numeric = [&]<typename T>() {
    std::vector<T> host_a(num_rows), host_b(num_rows);
    cudaMemcpyAsync(host_a.data(), col_a.data<T>(),
                    sizeof(T) * num_rows, cudaMemcpyDeviceToHost, stream.value());
    cudaMemcpyAsync(host_b.data(), col_b.data<T>(),
                    sizeof(T) * num_rows, cudaMemcpyDeviceToHost, stream.value());
    stream.synchronize();
    for (cudf::size_type r = 0; r < num_rows; ++r) {
        bool a_null = nulls_a.is_null(col_a.offset() + r);
        bool b_null = nulls_b.is_null(col_b.offset() + r);
        bool differs = (a_null != b_null) ||
                       (!a_null && !b_null && host_a[r] != host_b[r]);
        if (differs) {
            diff_count++;
            if (diff_indices.size() < static_cast<size_t>(max_diff_rows)) {
                diff_indices.push_back(r);
            }
        }
    }
};
```
[VERIFIED: Pattern follows existing debug_head extract_numeric lambda]

### Skill Documentation Example (for /validate)
```markdown
## Debug Utilities

Sirius provides structured debug utility functions in `src/include/debug_utils.hpp`.
Use these instead of ad-hoc `SIRIUS_LOG_TRACE` checksum patterns.

### Function Signatures

```cpp
#include "debug_utils.hpp"

// Schema and null inspection
sirius::debug_schema(batch, stream, col_names);
sirius::debug_nulls(batch, stream, col_names);

// Row preview (first N rows)
sirius::debug_head(batch, N, stream, format, col_names, max_string_len);

// Column statistics (GPU-side min/max/sum)
sirius::debug_stats(batch, stream, col_names);

// Per-column xxhash_64 fingerprint
sirius::debug_checksum(batch, stream, col_names);

// Two-batch comparison
sirius::debug_diff(batch_a, batch_b, stream, max_diff_rows, max_rows, col_names);

// Random row sampling
sirius::debug_sample(batch, N, stream, format, col_names, max_string_len, seed);
```
```
[ASSUMED -- exact markdown formatting and placement within SKILL.md is at Claude's discretion]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Ad-hoc `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] checksum: sum={}, max={}, first_row={}", ...)` in validate skill | `debug_checksum()`, `debug_stats()`, `debug_head()` named function calls | Phase 4 (this phase) | Skills reference stable API; Claude produces consistent diagnostic code |
| No batch comparison capability | `debug_diff(batch_a, batch_b, stream)` | Phase 4 (this phase) | Enables bisection of faulty operators by comparing input/output batches |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `debug_diff` parameter ordering: stream before optional params | Code Examples | Low -- parameter order is flexible; just needs to be consistent |
| A2 | `debug_sample` seed type as `std::optional<uint64_t>` | Code Examples | Low -- `uint32_t` also works for `std::mt19937`; `uint64_t` is more flexible |
| A3 | Shared formatting helper extracted from `debug_head` internals | Architecture Patterns | Low -- internal decomposition at Claude's discretion; duplication also works |
| A4 | Skill documentation placed as a new section within existing SKILL.md | Code Examples | Low -- section placement is at Claude's discretion per CONTEXT.md |

## Open Questions (RESOLVED)

1. **debug_diff on cudf::gather'd / sliced table views**
   - What we know: `debug_diff` receives `data_batch` objects which contain full `cudf::table` references. The function extracts `cudf::table_view` via `get_cudf_table_view()`.
   - What's unclear: Whether callers will ever pass batches that contain sliced views (with non-zero `col.offset()`). The existing null mask handling already accounts for offset, but value comparison needs to be offset-aware too.
   - RESOLVED: Follow existing `debug_head` pattern: `col.data<T>()` is offset-adjusted in cuDF 26.02, null mask uses `col.offset() + r`. This handles both full and sliced views correctly.

2. **debug_sample memory resource for cudf::gather**
   - What we know: `cudf::gather` accepts an optional `rmm::device_async_resource_ref` parameter. Most codebase usages pass an explicit allocator from `memory_space.get_default_allocator()`.
   - What's unclear: Whether `debug_sample` should use the default device resource (`cudf::get_current_device_resource_ref()`) or require a memory resource parameter.
   - RESOLVED: Use `cudf::get_current_device_resource_ref()` (same as `debug_checksum` does on line 942) to keep the API simple. Debug utilities are diagnostic-only and don't need explicit memory management.

## Environment Availability

Step 2.6: SKIPPED (no external dependencies identified). All implementation uses existing C++20 stdlib and cuDF libraries already linked in the build system. No new tools, services, or runtimes required.

## Project Constraints (from CLAUDE.md)

- **Build system:** Use pixi for all builds (`pixi run -e clang make release`). Never bare `make`.
- **Testing:** Run C++ unit tests via `build/release/extension/sirius/test/cpp/sirius_unittest "[debug_utils]"` for the debug tag.
- **Code formatting:** `pre-commit run -a` enforces clang-format, black, cmake-format, codespell.
- **Logging:** All output via `SIRIUS_LOG_DEBUG` / `SIRIUS_LOG_TRACE` with `[SIRIUS_DIAG]` prefix. No printf/stdout.
- **Error handling:** All debug functions wrapped in try/catch. Must never crash the pipeline.
- **Thread safety:** Output buffered into single `std::string`, emitted in one `SIRIUS_LOG_DEBUG` call.
- **CUDA streams:** Use `stream.synchronize()`, never `cudaDeviceSynchronize()`.
- **Naming conventions:** snake_case for functions and variables, PascalCase for public methods following DuckDB conventions.
- **Memory:** Use `duckdb::make_uniq<T>()` for DuckDB allocations, `std::make_unique<T>()` for standard allocations.
- **License headers:** Apache 2.0 on all source files.
- **Module context:** Run `/module-context` before implementing to load cudf/rmm API docs.

## Sources

### Primary (HIGH confidence)
- `src/debug_utils.cpp` (1006 lines) -- Full existing implementation with type dispatch, formatting, checksum patterns
- `src/include/debug_utils.hpp` -- Current API: debug_schema, debug_nulls, debug_head, debug_stats, debug_checksum
- `test/cpp/debug/test_debug_utils.cpp` (967 lines, 31 test cases) -- Established test patterns with batch creation, null masks, multi-type batches
- `src/op/sirius_physical_top_n.cpp:76-77` -- `cudf::gather` API usage pattern
- `src/op/order/gpu_order_impl.cpp:57-61` -- `cudf::gather` with stream and memory resource parameters
- `test/cpp/utils/utils.cpp:339-342` -- `std::mt19937_64` and `std::random_device` usage in codebase
- `.claude/skills/validate/SKILL.md` -- Current validate skill with ad-hoc SIRIUS_LOG_TRACE patterns (Phase 2, line 56-59)
- `.claude/skills/runtime-errors/SKILL.md` -- Current runtime-errors skill structure

### Secondary (MEDIUM confidence)
- `CMakeLists.txt:57,278` -- `debug_utils.cpp` and test file already registered in build system; no CMake changes needed

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies; all libraries verified in codebase
- Architecture: HIGH -- extends well-established patterns from Phases 1-3; all code patterns verified
- Pitfalls: HIGH -- derived from hands-on reading of existing implementation and type dispatch code

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (stable -- internal utility library, no external API drift risk)
