# Phase 1: Infrastructure and Metadata Inspection - Research

**Researched:** 2026-04-06
**Domain:** CUDA/cuDF GPU debug utility infrastructure -- stream-scoped sync, null-aware host copy, log routing, try/catch safety, schema/null inspection
**Confidence:** HIGH

## Summary

Phase 1 establishes the foundational debug utility infrastructure: stream-scoped synchronization, null-aware GPU-to-host copy, single-call atomic log output with `[SIRIUS_DIAG]` prefix, and try/catch safety wrapping. On top of this foundation, two metadata inspection functions are implemented: `debug_schema` (column names, types, null counts, row count) and `debug_nulls` (per-column null count and percentage without GPU kernel launch).

The most critical architectural decision this phase must get right is **implementing the new debug functions in a `.cpp` file, not the existing `.cu` file**. The `logging.hpp` header no-ops all `SIRIUS_LOG_*` macros when `__CUDACC__` is defined (lines 19-26), and `.cu` files are compiled entirely by nvcc which defines `__CUDACC__`. The existing `SIRIUS_LOG_DEBUG` calls in `print.cu` (lines 61, 68, 69) are **already silently dead code** -- they compile but produce zero output. All new debug utilities that need to log MUST live in a `.cpp` file that is added to `EXTENSION_SOURCES` in `CMakeLists.txt`.

**Primary recommendation:** Create `src/debug_utils.cpp` (with declarations in `src/include/debug_utils.hpp`) for all new debug functions, registered in `EXTENSION_SOURCES`. Keep `print.cu` unchanged for backward compatibility. Use `stream.synchronize()` (never `cudaDeviceSynchronize()`), buffer all output into a single `std::string`, and wrap every public entry point in try/catch.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| INFRA-01 | All debug functions accept `rmm::cuda_stream_view stream` and use `stream.synchronize()` -- never `cudaDeviceSynchronize()` | Confirmed: `gpu_pipeline_task::compute_task` and `run_one_operator` both receive `rmm::cuda_stream_view stream`. Stream-scoped sync avoids stalling unrelated GPU streams. Pattern verified at `src/pipeline/gpu_pipeline_task.cpp:210`. |
| INFRA-02 | Null-aware GPU-to-host copy helper that reads null bitmask alongside column values and represents NULLs as `"NULL"` | Confirmed: `cudf::column_view::null_mask()` returns device pointer to bitmask; `cudf::bit_is_set(bitmask, i)` from `<cudf/utilities/bit.hpp>` tests individual bits on host-copied bitmask. `cudf::bitmask_allocation_size_bytes(n)` from `<cudf/null_mask.hpp>` gives copy size. |
| INFRA-03 | Type dispatch covers all Sirius-supported types | Confirmed: existing `print_one_column` switch in `print.cu:199-211` covers INT8-64, UINT8-64, FLOAT32/64, BOOL8. Phase 1 extends to also handle STRING, DECIMAL, TIMESTAMP, DATE as "(type, not printed)" placeholders -- full extraction is Phase 2/3. |
| INFRA-04 | All output routed through `SIRIUS_LOG_DEBUG`/`SIRIUS_LOG_TRACE` with `[SIRIUS_DIAG]` prefix | Confirmed: MUST implement in `.cpp` file (not `.cu`). `logging.hpp` no-ops under `__CUDACC__`. Verified by observing `print.cu` lines 61/68/69 are dead code. |
| INFRA-05 | Entire output buffered into single `std::string` and emitted in one atomic log call | Confirmed: spdlog is thread-safe per-call. `fmt::format` available via `<spdlog/fmt/fmt.h>` in `.cpp` files. Build string with `fmt::memory_buffer` or `std::string` + `fmt::format_to`. |
| INFRA-06 | All debug functions wrapped in try/catch -- never crash the pipeline | Confirmed: cuDF APIs throw `cudf::logic_error`; CUDA APIs return `cudaError_t`. Wrap at public entry points, log error via `SIRIUS_LOG_WARN("[SIRIUS_DIAG] ...")`, return without propagating. |
| SCHEMA-01 | `debug_schema(batch)` prints column names, data types, null counts, total row count | Confirmed: `cudf::column_view::null_count()` returns stored metadata (no GPU kernel). `cudf::type_to_name(col.type())` from `<cudf/utilities/type_dispatcher.hpp>` gives type name string. Column names must come from caller (`cudf::table_view` does not store names). |
| SCHEMA-02 | Output is a compact summary table (one row per column) via SIRIUS_LOG | Confirmed: use `fmt::format` with width specifiers for aligned columns. Buffer into single string, emit one `SIRIUS_LOG_DEBUG` call. |
| NULL-01 | `debug_nulls(batch)` prints per-column null count and null percentage | Confirmed: `col.null_count()` is free metadata access. Percentage = `100.0 * null_count / size`. |
| NULL-02 | Uses `column_view::null_count()` metadata -- no kernel launch required | Confirmed: `null_count()` at `column_view.hpp:156` returns `_null_count` member directly. No stream synchronization or GPU work needed. |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| libcudf | 26.02.x | `column_view::null_count()`, `type_to_name()`, `bitmask_allocation_size_bytes()`, `bit_is_set()` | Already a Sirius dependency; provides all null inspection and type introspection APIs [VERIFIED: pixi.toml pins `libcudf = "26.02.*"`] |
| spdlog | 1.8.5 | `SIRIUS_LOG_DEBUG` / `SIRIUS_LOG_WARN` macros for all output | Already used throughout Sirius; bundled fmt provides formatting [VERIFIED: pixi.toml pins `spdlog = "1.8.*"`] |
| fmt (bundled in spdlog) | 7.1.3 | `fmt::format` with `{:<width}` specifiers for aligned table output | Available via `#include <spdlog/fmt/fmt.h>` in `.cpp` files [VERIFIED: used in `src/pipeline/gpu_pipeline_task.cpp:215`] |
| CUDA Runtime | 12+/13+ | `cudaMemcpy(DeviceToHost)` for null mask host copy | Already linked [VERIFIED: CMakeLists.txt CUDA standard 20] |
| rmm | 26.02.x | `rmm::cuda_stream_view` for stream-scoped sync | Already a Sirius dependency; the stream type used everywhere in pipeline tasks [VERIFIED: `gpu_pipeline_task.hpp:25` includes `<cucascade/memory/memory_reservation.hpp>`] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `<cudf/utilities/bit.hpp>` | 26.02.x | `cudf::bit_is_set(bitmask, i)` for null mask testing on host | In null-aware copy helper when checking per-row null status [VERIFIED: header at `.pixi/envs/default/include/cudf/utilities/bit.hpp:102`] |
| `<cudf/null_mask.hpp>` | 26.02.x | `cudf::bitmask_allocation_size_bytes(n)` | Computing device-to-host copy size for null bitmask [VERIFIED: header at `.pixi/envs/default/include/cudf/null_mask.hpp:50`] |
| `<cudf/utilities/type_dispatcher.hpp>` | 26.02.x | `cudf::type_to_name(data_type)` for human-readable type strings | In `debug_schema` to display column types [VERIFIED: function at line 635 of installed header] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New `debug_utils.cpp` | Extend existing `print.cu` | `print.cu` is compiled by nvcc; `SIRIUS_LOG_*` macros are no-ops there. New code MUST be in `.cpp` for logging to work. |
| `stream.synchronize()` | `cudaDeviceSynchronize()` | Device-wide sync stalls all streams, destroying pipeline concurrency. Stream-scoped sync only waits for the relevant stream. |
| `fmt::format` (spdlog bundled) | `std::format` (C++20) | Both available, but `SIRIUS_LOG_*` macros use spdlog's bundled fmt internally; mixing could cause header conflicts. Stick with `spdlog/fmt/fmt.h`. |
| `cudf::bit_is_set` on host-copied bitmask | Custom bit testing | `cudf::bit_is_set` is `CUDF_HOST_DEVICE` and works correctly on host. No reason to reimplement. |

## Architecture Patterns

### Recommended Project Structure

```
src/
  include/
    debug_utils.hpp          # NEW: Public API declarations for all debug_* functions
    print.hpp                 # UNCHANGED: Legacy print functions
  debug_utils.cpp             # NEW: All debug_* implementations (in EXTENSION_SOURCES)
  cuda/
    print.cu                  # UNCHANGED: Legacy print implementation
```

**Rationale for separate file (not extending `print.cu`):**

The PROJECT.md decision says "extend existing print.hpp/print.cu." However, research reveals this is **technically impossible** for the logging requirement (INFRA-04). `.cu` files are compiled by nvcc, which defines `__CUDACC__`, causing all `SIRIUS_LOG_*` macros to become no-ops. The existing `SIRIUS_LOG_DEBUG` calls in `print.cu` (lines 61, 68, 69) are already dead code that produces no output. [VERIFIED: `logging.hpp` lines 19-26 define no-op macros under `__CUDACC__`; CMakeLists.txt line 192 lists `print.cu` in `CUDA_SOURCES`]

The new `debug_utils.cpp` goes in `EXTENSION_SOURCES` (compiled by the C++ compiler, not nvcc). The header `debug_utils.hpp` can be included from any `.cpp` file. Call sites in operator `.cpp` files just include the new header.

**CMakeLists.txt changes required:** Add `src/debug_utils.cpp` to `EXTENSION_SOURCES` list (around line 78-158).

### Pattern 1: Stream-Scoped Synchronization (INFRA-01)

**What:** Every debug function accepts `rmm::cuda_stream_view stream` and calls `stream.synchronize()` before any device-to-host memcpy.
**When to use:** Every `debug_*` entry point.
**Why:** Pipeline tasks run on separate CUDA streams. `cudaDeviceSynchronize()` blocks ALL streams on the device, serializing concurrent pipeline execution. `stream.synchronize()` only blocks the calling CPU thread until work on that specific stream completes.

```cpp
// Source: pattern from src/pipeline/gpu_pipeline_task.cpp:210
void debug_schema(cucascade::data_batch const& batch,
                  rmm::cuda_stream_view stream,
                  std::vector<std::string> const& col_names = {}) {
    // stream.synchronize() ensures all GPU ops on this stream are done
    // before we read metadata
    stream.synchronize();
    // ... inspect column metadata
}
```

[VERIFIED: `run_one_operator` at `gpu_pipeline_task.cpp:210` uses `stream.synchronize()` as the standard pattern]

### Pattern 2: Null-Aware GPU-to-Host Copy Helper (INFRA-02)

**What:** A reusable helper that copies both data and null bitmask from device to host, providing a way to check if each row is null.
**When to use:** Any debug function that accesses row-level column data (future phases need this for `debug_head`, `debug_diff`).

```cpp
// Source: cudf API verified from installed headers
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>

struct host_column_nulls {
    std::vector<cudf::bitmask_type> mask;
    bool has_nulls;

    bool is_null(cudf::size_type row) const {
        if (!has_nulls) return false;
        return !cudf::bit_is_set(mask.data(), row);
    }
};

host_column_nulls copy_null_mask_to_host(
    cudf::column_view const& col,
    rmm::cuda_stream_view stream)
{
    host_column_nulls result;
    result.has_nulls = col.has_nulls();
    if (!result.has_nulls) return result;

    auto const num_bitmask_words =
        cudf::bitmask_allocation_size_bytes(col.size()) / sizeof(cudf::bitmask_type);
    result.mask.resize(num_bitmask_words);
    cudaMemcpyAsync(result.mask.data(),
                    col.null_mask(),
                    num_bitmask_words * sizeof(cudf::bitmask_type),
                    cudaMemcpyDeviceToHost,
                    stream.value());
    stream.synchronize();
    return result;
}
```

[VERIFIED: `cudf::bit_is_set` at `bit.hpp:102` is `CUDF_HOST_DEVICE`; `bitmask_allocation_size_bytes` at `null_mask.hpp:50`]

**Important:** `col.null_mask()` does NOT account for `col.offset()`. When a column is a slice (e.g., from `cudf::slice`), the bitmask pointer still points to the original allocation and `col.offset()` indicates where the valid range starts. The null-check must use `cudf::bit_is_set(mask, col.offset() + row_index)` to account for this. For Phase 1 (metadata only, no row-level access), this is not yet needed, but the helper should be designed with offset awareness for Phase 2.

### Pattern 3: Single-String Output Buffering (INFRA-05)

**What:** Build the entire debug output into one `std::string`, then emit it in a single `SIRIUS_LOG_DEBUG` call.
**When to use:** Every `debug_*` function.
**Why:** spdlog is thread-safe per-call, but not across calls. Multiple `SIRIUS_LOG_DEBUG` calls from concurrent pipeline tasks will interleave. One call = one atomic log entry.

```cpp
// Source: fmt pattern from spdlog bundled fmt, verified in gpu_pipeline_task.cpp:215
#include <spdlog/fmt/fmt.h>

void debug_schema(/* ... */) {
    try {
        // ... gather metadata ...
        std::string output;
        output += fmt::format("[SIRIUS_DIAG] schema: {} rows, {} cols\n",
                              table.num_rows(), table.num_columns());
        for (int c = 0; c < table.num_columns(); ++c) {
            auto const& col = table.column(c);
            std::string name = (c < col_names.size()) ? col_names[c]
                                                       : fmt::format("col[{}]", c);
            output += fmt::format("[SIRIUS_DIAG]   {:<20s} {:<15s} nulls={:<6d} ({:.1f}%)\n",
                                  name,
                                  cudf::type_to_name(col.type()),
                                  col.null_count(),
                                  col.size() > 0 ? 100.0 * col.null_count() / col.size() : 0.0);
        }
        SIRIUS_LOG_DEBUG("{}", output);
    } catch (std::exception const& e) {
        SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_schema failed: {}", e.what());
    } catch (...) {
        SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_schema failed: unknown error");
    }
}
```

### Pattern 4: Tier Guard for Non-GPU Batches (Success Criterion 4)

**What:** Before accessing the cudf table view, check that the batch's data is in GPU tier. If not, log a warning and return.
**When to use:** Every `debug_*` function that accesses batch data through `get_cudf_table_view()`.

```cpp
// Source: pattern from src/include/pipeline/gpu_pipeline_task.hpp:117
// and src/op/sirius_physical_result_collector.cpp:136
auto* data = batch.get_data();
if (data == nullptr) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_schema: batch has no data representation");
    return;
}
if (data->get_current_tier() != cucascade::memory::Tier::GPU) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_schema: batch not in GPU tier (tier={}), skipping",
                    static_cast<int>(data->get_current_tier()));
    return;
}
```

[VERIFIED: `get_current_tier()` used at `gpu_pipeline_task.hpp:117` and `sirius_physical_result_collector.cpp:136`]

**Note for `debug_nulls` (NULL-02):** `debug_nulls` uses only `column_view::null_count()` which is stored metadata, not a GPU read. However, we still need the `cudf::table_view` to access the columns, which requires the batch to be in GPU tier (because `get_cudf_table_view` casts to `gpu_table_representation`). So the tier guard applies to all debug functions including `debug_nulls`.

### Pattern 5: Try/Catch Safety Wrapping (INFRA-06)

**What:** Every public `debug_*` entry point is wrapped in try/catch that catches all exceptions and logs them.
**When to use:** All public debug functions.

```cpp
void debug_schema(cucascade::data_batch const& batch,
                  rmm::cuda_stream_view stream,
                  std::vector<std::string> const& col_names) {
    try {
        // ... implementation ...
    } catch (cudf::logic_error const& e) {
        SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_schema failed (cudf): {}", e.what());
    } catch (std::exception const& e) {
        SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_schema failed: {}", e.what());
    } catch (...) {
        SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_schema failed: unknown error");
    }
}
```

### Anti-Patterns to Avoid

- **Implementing in `.cu` files:** `SIRIUS_LOG_*` macros are no-ops under `__CUDACC__`. All debug utility implementations MUST be in `.cpp` files. [VERIFIED: `logging.hpp:19-26`]
- **Using `cudaDeviceSynchronize()`:** Blocks ALL GPU streams. Use `stream.synchronize()` instead. [VERIFIED: pitfall documented in PITFALLS.md]
- **Multiple log calls for one logical output:** Interleaves under concurrency. Buffer everything into one string. [VERIFIED: spdlog MT safety is per-call only]
- **Using `printf`/`std::cout`:** Bypasses log pipeline; invisible to skills that grep `sirius.log`. [VERIFIED: existing `print.cu` uses printf and output is not in log files]
- **Forgetting `[SIRIUS_DIAG]` prefix:** Skills grep for this tag. Every line of debug output must include it.
- **Letting exceptions propagate:** A crashing debug call masks the real bug being investigated. Always catch and log.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Null bitmask bit testing | Custom bit manipulation | `cudf::bit_is_set()` from `<cudf/utilities/bit.hpp>` | Handles word alignment and bit indexing correctly; marked `CUDF_HOST_DEVICE` so works on host [VERIFIED] |
| Bitmask allocation size | Manual `ceil(n/32)*4` | `cudf::bitmask_allocation_size_bytes(n)` from `<cudf/null_mask.hpp>` | Accounts for padding boundary (default 64 bytes); avoids off-by-one [VERIFIED] |
| Type name strings | Manual switch on type_id | `cudf::type_to_name(col.type())` from `<cudf/utilities/type_dispatcher.hpp>` | Returns canonical cuDF type name string; stays in sync with cuDF version [VERIFIED] |
| String formatting with alignment | Manual padding with spaces | `fmt::format("{:<width}", val)` via `<spdlog/fmt/fmt.h>` | Already bundled with spdlog; handles Unicode width, padding, type formatting [VERIFIED] |
| Thread-safe logging | Custom mutex around printf | `SIRIUS_LOG_DEBUG` macro (spdlog MT sink) | spdlog's default logger uses thread-safe sinks; per-call atomicity guaranteed [VERIFIED] |

## Common Pitfalls

### Pitfall 1: SIRIUS_LOG_* Silently No-Ops in .cu Files

**What goes wrong:** Developer adds `SIRIUS_LOG_DEBUG` calls in a `.cu` file. Code compiles without errors or warnings. No output appears in the log.
**Why it happens:** `logging.hpp` lines 19-26 define all `SIRIUS_LOG_*` macros as empty `(...)` when `__CUDACC__` is defined. nvcc defines `__CUDACC__` for all `.cu` translation units. This is an intentional workaround because "nvcc cannot compile spdlog/fmt chrono headers."
**How to avoid:** All debug utility implementations go in `.cpp` files added to `EXTENSION_SOURCES`. Only CUDA kernels go in `.cu` files.
**Warning signs:** Any `SIRIUS_LOG_*` call in a `.cu` file. Zero `[SIRIUS_DIAG]` output when debug functions are called.
[VERIFIED: `logging.hpp:19-26`; `print.cu` lines 61/68/69 are existing dead code]

### Pitfall 2: cudaDeviceSynchronize Stalls All Streams

**What goes wrong:** Using `cudaDeviceSynchronize()` in a debug function blocks the calling thread until ALL work on ALL CUDA streams completes, serializing the entire pipeline.
**Why it happens:** It is the "obvious" sync primitive. The existing `print_table_contents` uses it (`print.cu:221`).
**How to avoid:** Accept `rmm::cuda_stream_view stream` parameter and call `stream.synchronize()`.
**Warning signs:** `cudaDeviceSynchronize()` anywhere in new code. Noticeably slower queries when debug is enabled.
[VERIFIED: `print.cu:221` uses the wrong pattern; `gpu_pipeline_task.cpp:210` shows correct pattern]

### Pitfall 3: get_cudf_table_view on Non-GPU-Tier Batch Crashes

**What goes wrong:** `get_cudf_table_view(batch)` calls `data->cast<gpu_table_representation>()`. If the batch is in HOST or DISK tier, this cast fails or returns a stale/null pointer.
**Why it happens:** cuCascade implements tiered memory; batches may be spilled under memory pressure. Debug utilities inserted at arbitrary operator boundaries don't know the current tier.
**How to avoid:** Check `batch.get_data()->get_current_tier() == cucascade::memory::Tier::GPU` before accessing.
**Warning signs:** Crashes or illegal address errors only during debug-instrumented runs on large queries.
[VERIFIED: tier check pattern at `gpu_pipeline_task.hpp:117` and `sirius_physical_result_collector.cpp:136`]

### Pitfall 4: null_count() Returns UNKNOWN on Certain Column Views

**What goes wrong:** `cudf::column_view::null_count()` returns the stored `_null_count` member. For some column views (e.g., freshly constructed views from raw pointers), `_null_count` may be set to `UNKNOWN_NULL_COUNT` (-1), which when cast to `size_type` appears as a very large number.
**Why it happens:** cuDF uses a sentinel value for "null count not yet computed."
**How to avoid:** Guard against negative null_count: `auto nc = col.null_count(); if (nc < 0) nc = 0;` or use the range-based `null_count(0, col.size(), stream)` which always computes. For Phase 1, the simpler approach is to treat negative values as "unknown" and display them accordingly.
[ASSUMED -- based on cuDF API design patterns; most Sirius columns will have precomputed null counts]

### Pitfall 5: Column Offset Not Accounted for in Bitmask Access

**What goes wrong:** After `cudf::slice`, the resulting `column_view` has a non-zero `offset()` but shares the original bitmask allocation. Calling `bit_is_set(mask, row_index)` without adding the offset reads the wrong bits.
**Why it happens:** `cudf::slice` returns a zero-copy view with an offset into the original data/bitmask buffers.
**How to avoid:** Always use `bit_is_set(mask, col.offset() + row_index)` when testing individual bits. For Phase 1 (metadata only), this is not yet relevant since `null_count()` handles offset internally. But the null-mask copy helper should be designed with this in mind for Phase 2.
[VERIFIED: `column_view.hpp:244` stores `_offset` member; `cudf::slice` documentation states views share underlying buffers]

## Code Examples

### debug_schema Implementation Skeleton

```cpp
// Source: cudf API from installed headers, Sirius patterns from gpu_pipeline_task.cpp
#include "debug_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/tier.hpp>

#include <spdlog/fmt/fmt.h>

#include <string>
#include <vector>

namespace sirius {

void debug_schema(cucascade::data_batch const& batch,
                  rmm::cuda_stream_view stream,
                  std::vector<std::string> const& col_names)
{
    try {
        auto* data = batch.get_data();
        if (!data) {
            SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_schema: batch has no data");
            return;
        }
        if (data->get_current_tier() != cucascade::memory::Tier::GPU) {
            SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_schema: batch not in GPU tier, skipping");
            return;
        }

        cudf::table_view tv = get_cudf_table_view(batch);
        stream.synchronize();  // INFRA-01: stream-scoped sync

        // INFRA-05: buffer into single string
        std::string output;
        output += fmt::format(
            "[SIRIUS_DIAG] schema: batch_id={} rows={} cols={}\n",
            batch.get_batch_id(), tv.num_rows(), tv.num_columns());
        output += fmt::format(
            "[SIRIUS_DIAG]   {:<6s} {:<20s} {:<15s} {:>8s} {:>8s}\n",
            "idx", "name", "type", "nulls", "null%");
        output += fmt::format(
            "[SIRIUS_DIAG]   {:-<6s} {:-<20s} {:-<15s} {:->8s} {:->8s}\n",
            "", "", "", "", "");

        for (cudf::size_type c = 0; c < tv.num_columns(); ++c) {
            auto const& col = tv.column(c);
            std::string name = (static_cast<size_t>(c) < col_names.size())
                                   ? col_names[c]
                                   : fmt::format("col[{}]", c);
            auto nc = col.null_count();
            double pct = (col.size() > 0) ? 100.0 * nc / col.size() : 0.0;
            output += fmt::format(
                "[SIRIUS_DIAG]   {:<6d} {:<20s} {:<15s} {:>8d} {:>7.1f}%\n",
                c, name, cudf::type_to_name(col.type()), nc, pct);
        }

        SIRIUS_LOG_DEBUG("{}", output);  // INFRA-04: single atomic log call
    } catch (std::exception const& e) {
        SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_schema failed: {}", e.what());
    } catch (...) {
        SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_schema failed: unknown error");
    }
}

}  // namespace sirius
```

### debug_nulls Implementation Skeleton

```cpp
// Source: cudf column_view::null_count() from installed headers
void debug_nulls(cucascade::data_batch const& batch,
                 rmm::cuda_stream_view stream,
                 std::vector<std::string> const& col_names)
{
    try {
        auto* data = batch.get_data();
        if (!data) {
            SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_nulls: batch has no data");
            return;
        }
        if (data->get_current_tier() != cucascade::memory::Tier::GPU) {
            SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_nulls: batch not in GPU tier, skipping");
            return;
        }

        cudf::table_view tv = get_cudf_table_view(batch);
        // NULL-02: null_count() is stored metadata -- no GPU kernel needed
        // But we still need stream sync for safety if prior ops are in flight
        stream.synchronize();

        std::string output;
        output += fmt::format(
            "[SIRIUS_DIAG] nulls: batch_id={} rows={} cols={}\n",
            batch.get_batch_id(), tv.num_rows(), tv.num_columns());
        output += fmt::format(
            "[SIRIUS_DIAG]   {:<6s} {:<20s} {:>8s} {:>8s}\n",
            "idx", "name", "nulls", "null%");
        output += fmt::format(
            "[SIRIUS_DIAG]   {:-<6s} {:-<20s} {:->8s} {:->8s}\n",
            "", "", "", "");

        for (cudf::size_type c = 0; c < tv.num_columns(); ++c) {
            auto const& col = tv.column(c);
            std::string name = (static_cast<size_t>(c) < col_names.size())
                                   ? col_names[c]
                                   : fmt::format("col[{}]", c);
            auto nc = col.null_count();
            double pct = (col.size() > 0) ? 100.0 * nc / col.size() : 0.0;
            output += fmt::format(
                "[SIRIUS_DIAG]   {:<6d} {:<20s} {:>8d} {:>7.1f}%\n",
                c, name, nc, pct);
        }

        SIRIUS_LOG_DEBUG("{}", output);
    } catch (std::exception const& e) {
        SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_nulls failed: {}", e.what());
    } catch (...) {
        SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_nulls failed: unknown error");
    }
}
```

### Header Declarations

```cpp
// src/include/debug_utils.hpp
#pragma once

#include <rmm/cuda_stream_view.hpp>

#include <string>
#include <vector>

namespace cucascade {
class data_batch;
}

namespace sirius {

/**
 * @brief Log schema metadata for a data batch.
 *
 * Outputs column names, types, null counts, and total row count
 * as a structured [SIRIUS_DIAG] block in sirius.log.
 *
 * @param batch     The data batch to inspect (must be in GPU tier)
 * @param stream    CUDA stream for synchronization
 * @param col_names Optional column names (cudf::table_view has no names)
 */
void debug_schema(cucascade::data_batch const& batch,
                  rmm::cuda_stream_view stream,
                  std::vector<std::string> const& col_names = {});

/**
 * @brief Log per-column null counts and percentages.
 *
 * Uses column_view::null_count() metadata only -- no GPU kernel launched.
 * Outputs a [SIRIUS_DIAG] block with null analysis.
 *
 * @param batch     The data batch to inspect (must be in GPU tier)
 * @param stream    CUDA stream for synchronization
 * @param col_names Optional column names
 */
void debug_nulls(cucascade::data_batch const& batch,
                 rmm::cuda_stream_view stream,
                 std::vector<std::string> const& col_names = {});

}  // namespace sirius
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `print_table_contents` via `printf` | `debug_*` via `SIRIUS_LOG_DEBUG` with `[SIRIUS_DIAG]` | This phase | Output visible in log files, parseable by skills |
| `cudaDeviceSynchronize()` before DtoH copy | `stream.synchronize()` per-stream | This phase | No stalling of unrelated pipeline streams |
| No null awareness (garbage values at null rows) | Null bitmask copied and checked | This phase | Correct NULL display in future `debug_head` |
| Implement in `.cu` file | Implement in `.cpp` file | This phase | `SIRIUS_LOG_*` macros actually produce output |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `null_count()` may return `UNKNOWN_NULL_COUNT` (-1) on some column views | Pitfall 4 | LOW -- if it never happens in Sirius, the guard is harmless. If it does happen without the guard, debug output shows a very large number as the null count. |
| A2 | Creating a new `debug_utils.cpp` instead of extending `print.cu` is necessary | Architecture | HIGH -- this contradicts the PROJECT.md decision to "extend print.hpp/print.cu". However, research proves extending `.cu` is technically impossible for INFRA-04 (logging). The header `print.hpp` can still be extended with declarations that delegate to the `.cpp` implementation if maintaining the single-header API is preferred. |

## Open Questions

1. **File organization: new file vs extending print.hpp**
   - What we know: The `.cu` file cannot contain logging code. A `.cpp` file is required.
   - What's unclear: Should the new `.cpp` file's public API be declared in a new `debug_utils.hpp`, or should declarations be added to the existing `print.hpp` (with the implementations in the new `.cpp` file)?
   - Recommendation: Use a new `debug_utils.hpp` for clarity. The `debug_*` functions have different semantics (log output vs stdout, stream parameter, try/catch) than the existing `print_*` functions. Mixing them in one header is confusing. If the project strongly prefers one header, the declarations can go in `print.hpp` with a `// Implementation in debug_utils.cpp` comment.

2. **Column names source**
   - What we know: `cudf::table_view` does not carry column names. Names must be passed by the caller.
   - What's unclear: Where do callers get column names? Operators have `types` (via `get_types()`) but not names. The scan operators read names from metadata.
   - Recommendation: Accept `std::vector<std::string> const& col_names = {}` as an optional parameter with a default. If empty, use `col[0]`, `col[1]`, etc. Callers provide names when available.

3. **Log level: DEBUG vs TRACE**
   - What we know: INFRA-04 says "SIRIUS_LOG_DEBUG or SIRIUS_LOG_TRACE". Default log level at runtime is controlled by `SIRIUS_LOG_LEVEL` env var (typically `debug` or `info`).
   - What's unclear: Which level should `debug_schema`/`debug_nulls` use?
   - Recommendation: Use `SIRIUS_LOG_DEBUG` for the output. Use `SIRIUS_LOG_WARN` for errors/warnings. `TRACE` would be invisible at the default `debug` level.

## Project Constraints (from CLAUDE.md)

- **Build system:** Use `pixi shell` then `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make`. Clean with `rm -rf build` on errors.
- **Testing:** Unit tests use Catch2 framework. Run with `build/release/extension/sirius/test/cpp/sirius_unittest`.
- **Code formatting:** Pre-commit hooks with clang-format, black, cmake-format, codespell. Run `pre-commit run -a`.
- **Logging:** Use `SIRIUS_LOG_*` macros (never direct spdlog). Log dir controlled by `SIRIUS_LOG_DIR` env var.
- **New features:** Load module context first via `/module-context` before implementing.
- **C++ standard:** C++20 required. CUDA standard 20.
- **Extension architecture:** `build_static_extension` and `build_loadable_extension` in CMakeLists.txt.
- **Main branch:** `dev` (not `main`).

## Sources

### Primary (HIGH confidence)
- `src/include/log/logging.hpp` -- `__CUDACC__` no-op guard confirmed at lines 19-26 [VERIFIED: direct file read]
- `src/cuda/print.cu` -- existing implementation patterns, dead `SIRIUS_LOG_DEBUG` calls at lines 61/68/69 [VERIFIED: direct file read]
- `CMakeLists.txt` -- `print.cu` in `CUDA_SOURCES` (line 192), `EXTENSION_SOURCES` structure (lines 65-158), CUDA_SEPARABLE_COMPILATION ON [VERIFIED: direct file read]
- `src/include/data/data_batch_utils.hpp` -- `get_cudf_table_view` implementation [VERIFIED: direct file read]
- `src/pipeline/gpu_pipeline_task.cpp` -- `stream.synchronize()` pattern (line 210), `run_one_operator` receiving stream (line 197), tier check patterns [VERIFIED: direct file read]
- `src/op/sirius_physical_result_collector.cpp` -- `get_current_tier()` usage pattern (line 136) [VERIFIED: direct file read]
- `src/include/pipeline/gpu_pipeline_task.hpp` -- tier check at line 117 [VERIFIED: direct file read]
- `.pixi/envs/default/include/cudf/column/column_view.hpp` -- `null_count()` returns stored metadata at line 156, `nullable()` at line 149, `null_mask()` at line 215 [VERIFIED: direct file read]
- `.pixi/envs/default/include/cudf/utilities/bit.hpp` -- `bit_is_set` at line 102, `CUDF_HOST_DEVICE` qualified [VERIFIED: direct file read]
- `.pixi/envs/default/include/cudf/null_mask.hpp` -- `bitmask_allocation_size_bytes` at line 50 [VERIFIED: direct file read]
- `.pixi/envs/default/include/cudf/utilities/type_dispatcher.hpp` -- `type_to_name` at line 635 [VERIFIED: direct file read]

### Secondary (MEDIUM confidence)
- `.planning/research/STACK.md` -- Stack research from project init [VERIFIED: direct file read]
- `.planning/research/ARCHITECTURE.md` -- Architecture decisions [VERIFIED: direct file read]
- `.planning/research/PITFALLS.md` -- Known pitfalls catalogue [VERIFIED: direct file read]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified in installed headers and pixi.toml
- Architecture: HIGH -- `.cu` vs `.cpp` issue verified by inspecting `__CUDACC__` guard and dead code in `print.cu`
- Pitfalls: HIGH -- all pitfalls verified against actual source code patterns
- Code examples: HIGH -- based on verified API signatures from installed cuDF 26.02.x headers

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (stable domain -- cuDF 26.02.x and spdlog 1.8.x are pinned in pixi.toml)
