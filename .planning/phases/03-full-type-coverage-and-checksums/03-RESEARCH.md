# Phase 3: Full Type Coverage and Checksums - Research

**Researched:** 2026-04-08
**Domain:** cuDF string extraction, fixed-point decimal formatting, timestamp/date conversion, GPU hashing with XOR reduce
**Confidence:** HIGH

## Summary

Phase 3 extends `debug_head` to handle the remaining Sirius-supported data types (STRING, DECIMAL32/64/128, TIMESTAMP_SECONDS/MILLISECONDS/MICROSECONDS/NANOSECONDS, TIMESTAMP_DAYS) and introduces a new `debug_checksum` function that produces per-column xxhash_64 fingerprints for cross-pipeline data comparison.

The STRING extraction uses `cudf::strings_column_view` with the two-buffer pattern (offsets column + chars pointer) that is already used in 5+ source files in the Sirius codebase. DECIMAL formatting requires reading raw integer storage (`int32_t`, `int64_t`, or `__int128_t` depending on DECIMAL32/64/128), then inserting a decimal point at position `|scale|` from the right. TIMESTAMP/DATE formatting converts raw epoch values to broken-down calendar components using integer arithmetic (no `<chrono>` or `gmtime` needed since the values are simple epoch offsets). The checksum implementation uses `cudf::hashing::xxhash_64` (returns one UINT64 per row) combined with `cudf::reduce` using `make_bitwise_aggregation(bitwise_op::XOR)` to collapse each column's hashes to a single 64-bit value entirely on GPU.

**Primary recommendation:** Extend the existing switch statement in `debug_head` with new cases for STRING, DECIMAL, and TIMESTAMP/DATE types. Add `debug_checksum` as a new function following the same patterns (tier guard, try/catch, single-string output buffering). Add `max_string_len` parameter to `debug_head` with default value 50 for backward compatibility.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** STRING columns extracted via `cudf::strings_column_view` with the standard two-buffer pattern (offsets + chars) for GPU-to-host copy
- **D-02:** Long strings truncated with configurable `max_string_len` parameter on `debug_head`, default 50 characters. Truncated strings get `"..."` suffix. Pass 0 for no truncation
- **D-03:** The `max_string_len` parameter is a function parameter (not a global config setting), following the existing API pattern of explicit parameters with defaults
- **D-04:** DECIMAL values displayed in fixed-point format with exactly `|scale|` decimal places (e.g., scale=-2: `123.45`, scale=-4: `1.2345`). Trailing zeros preserved (e.g., `10.00` not `10`) to match SQL DECIMAL semantics exactly
- **D-05:** Works with DECIMAL32, DECIMAL64, and DECIMAL128 types -- scale read from `col.type().scale()`
- **D-06:** TIMESTAMP values displayed in SQL-style format: `2024-01-15 08:30:00` (space separator, no T)
- **D-07:** DATE values displayed as `2024-01-15` (date only, no time component)
- **D-08:** Sub-second fractional seconds shown only when non-zero (e.g., `08:30:00.123` for ms, `08:30:00.123456` for us). When fractional part is `.000...`, it is omitted to keep output clean
- **D-09:** All temporal types treated as UTC -- no timezone conversion (matches cuDF's storage model)
- **D-10:** `debug_checksum` computes per-column fingerprint using `cudf::hashing::xxhash_64` to hash all rows, then `cudf::reduce(XOR)` to collapse to a single 64-bit value per column. Stays entirely on GPU
- **D-11:** Output format: `col[N] checksum: 0xABCD1234EF567890 nulls=2` -- includes null_count per column to catch null-handling bugs where hashes match but null counts differ
- **D-12:** Deterministic -- same data in same order produces same checksum across runs. Order-dependent (different row order = different checksum)

### Claude's Discretion
- Internal helper function decomposition for type extraction
- Whether to add `max_string_len` to `debug_stats` as well (stats does not display string values, so probably not needed)
- xxhash_64 seed value (0 is standard default)
- Whether `debug_checksum` should also log row_count in the header line (likely yes, following existing pattern)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HEAD-04 | STRING columns extracted via `cudf::strings_column_view` with proper two-buffer (offsets + chars) host copy | Verified: `cudf::strings_column_view` API confirmed in installed cudf 26.02, with `chars_begin(stream)`, `chars_size(stream)`, and `offsets()` accessors. Pattern used in `iceberg_scan_task.cpp` lines 88-108 |
| HEAD-05 | DECIMAL columns display with correct scale factor from `col.type().scale()` | Verified: `cudf::data_type::scale()` returns `int32_t` scale. DECIMAL32 uses `int32_t`, DECIMAL64 uses `int64_t`, DECIMAL128 uses `__int128_t` storage. Scale is negative in cudf convention (e.g., `-2` means 2 decimal places) |
| HEAD-06 | TIMESTAMP and DATE columns display as human-readable calendar format (not raw epoch integers) | Verified: TIMESTAMP_DAYS uses `int32_t` (days since epoch), TIMESTAMP_SECONDS/MS/US/NS use `int64_t`. DuckDB TIMESTAMP maps to TIMESTAMP_MICROSECONDS, DATE maps to TIMESTAMP_DAYS |
| CHKSUM-01 | `debug_checksum(batch)` computes and logs per-column hash fingerprint | Verified: `cudf::hashing::xxhash_64(table_view, seed, stream, mr)` returns `unique_ptr<column>` of UINT64 hashes |
| CHKSUM-02 | Uses `cudf::hashing::xxhash_64` for consistent cross-run comparison | Verified: API in `cudf/hashing.hpp`, takes `table_view` and `uint64_t seed`, returns UINT64 column |
| CHKSUM-03 | Output format enables easy diff between two log files | Design: `col[N] checksum: 0xHEX nulls=N` format per D-11, with header showing batch_id, rows, cols |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| libcudf | 26.02.x | strings_column_view, hashing::xxhash_64, reduce with bitwise XOR | Already installed and used throughout Sirius [VERIFIED: pixi env] |
| spdlog/fmt | 1.8.x | String formatting for decimal display, hex output | Already used in debug_utils.cpp [VERIFIED: codebase] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| cudf/hashing.hpp | 26.02.x | xxhash_64 function for per-row hashing | debug_checksum implementation [VERIFIED: header read] |
| cudf/aggregation.hpp | 26.02.x | make_bitwise_aggregation with bitwise_op::XOR | XOR reduction of hash column [VERIFIED: header read] |
| cudf/strings/strings_column_view.hpp | 26.02.x | String column accessor (offsets + chars) | STRING type in debug_head [VERIFIED: header read] |
| cudf/fixed_point/fixed_point.hpp | 26.02.x | decimal32/64/128 type definitions | Understanding DECIMAL storage [VERIFIED: header read] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual epoch-to-calendar arithmetic | `<chrono>` with `sys_days`/`year_month_day` | chrono is cleaner but adds C++20 calendar dependency; manual arithmetic is simpler for this limited use case and avoids potential cross-platform issues |
| `cudf::hashing::xxhash_64` | `cudf::hashing::murmurhash3_x86_32` | murmur3 is 32-bit only; xxhash_64 provides 64-bit fingerprints with lower collision probability |
| Cast `__int128_t` to double for display | Manual integer-to-string with decimal insertion | Double cast loses precision for large DECIMAL128 values; integer-to-string preserves exact values |

## Architecture Patterns

### String Extraction Pattern (from iceberg_scan_task.cpp)
**What:** Two-phase GPU-to-host copy for string columns using `cudf::strings_column_view`
**When to use:** Any time string column data needs to be read on the host
**Example:**
```cpp
// Source: src/op/scan/iceberg_scan_task.cpp lines 88-108 [VERIFIED: codebase]
cudf::strings_column_view sv(col);
auto const chars_bytes = sv.chars_size(stream);
std::vector<char> host_chars(chars_bytes);
if (chars_bytes > 0) {
  cudaMemcpyAsync(host_chars.data(), sv.chars_begin(stream),
                  chars_bytes, cudaMemcpyDeviceToHost, stream.value());
}
auto const& offsets_col = sv.offsets();
std::vector<int32_t> host_offsets(num_rows + 1);
cudaMemcpyAsync(host_offsets.data(), offsets_col.data<int32_t>(),
                (num_rows + 1) * sizeof(int32_t),
                cudaMemcpyDeviceToHost, stream.value());
stream.synchronize();
// Then extract each string:
for (int i = 0; i < num_rows; ++i) {
  auto start = host_offsets[i];
  auto end   = host_offsets[i + 1];
  std::string value(host_chars.data() + start, end - start);
}
```

### Decimal Fixed-Point Formatting
**What:** Convert raw integer storage to fixed-point string representation
**When to use:** DECIMAL32/64/128 column display in debug_head
**Example:**
```cpp
// Source: Manual implementation based on cudf decimal storage model [ASSUMED]
// For DECIMAL64 with scale=-2, raw value 12345 displays as "123.45"
// scale is negative in cudf convention: col.type().scale() returns e.g. -2
int32_t scale = col.type().scale();  // negative: -2 means 2 decimal places
int64_t raw_value;  // from GPU memcpy
// Convert to string:
// 1. Get absolute scale
int abs_scale = std::abs(scale);
// 2. Format with integer part and fractional part
// For DECIMAL128, use __int128_t and manual division
```

### Timestamp/Date Calendar Formatting
**What:** Convert epoch-based timestamps to human-readable YYYY-MM-DD HH:MM:SS format
**When to use:** TIMESTAMP and DATE column display in debug_head
**Pattern:**
```cpp
// Source: Standard Unix epoch conversion [ASSUMED]
// TIMESTAMP_DAYS: int32_t days since 1970-01-01
// TIMESTAMP_MICROSECONDS: int64_t microseconds since 1970-01-01 00:00:00 UTC
// Convert to broken-down date/time using civil_from_days algorithm
// (Howard Hinnant's algorithm, public domain, widely used)
```

### Checksum Pipeline (GPU-only)
**What:** Per-column xxhash_64 hash + XOR reduce to produce a single 64-bit fingerprint per column
**When to use:** `debug_checksum` function
**Example:**
```cpp
// Source: cudf/hashing.hpp [VERIFIED: header read]
// Step 1: Hash single column (wrap in table_view for xxhash_64)
cudf::table_view single_col_tv({col});
auto hash_col = cudf::hashing::xxhash_64(single_col_tv, 0, stream, mr);

// Step 2: XOR reduce to single value
// Source: cudf/aggregation.hpp [VERIFIED: header grep]
auto xor_agg = cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(
    cudf::bitwise_op::XOR);
auto result = cudf::reduce(hash_col->view(), *xor_agg,
    cudf::data_type{cudf::type_id::UINT64}, stream, mr);

// Step 3: Extract scalar value
auto& scalar = static_cast<cudf::numeric_scalar<uint64_t> const&>(*result);
uint64_t checksum = scalar.value(stream);
```

### Anti-Patterns to Avoid
- **Copying entire string chars buffer when only N rows are needed:** For `debug_head`, only copy the chars bytes spanned by the first N rows' offsets, not the entire chars buffer. Use offset[0] and offset[N] to compute the byte range.
- **Using `gmtime`/`localtime` for timestamp conversion:** These are not thread-safe. Use pure integer arithmetic (civil_from_days algorithm) instead.
- **Using `std::to_string` for `__int128_t`:** There is no standard library support for formatting `__int128_t`. Must implement manual integer-to-string conversion.
- **Forgetting offset adjustment for sliced string columns:** After `cudf::slice`, the offsets column still contains absolute byte positions. Must subtract `offsets[0]` to get relative positions within the chars buffer starting from `chars_begin(stream)`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-row hashing | Custom hash kernel | `cudf::hashing::xxhash_64` | Handles all cudf types including strings, nulls, decimals; deterministic and optimized |
| Hash column reduction | Custom XOR kernel | `cudf::reduce` with `make_bitwise_aggregation(bitwise_op::XOR)` | Single API call, handles empty columns, returns scalar |
| String extraction from GPU | Manual pointer arithmetic on column buffers | `cudf::strings_column_view` | Handles offsets, slicing, and chars buffer access correctly |
| Days-since-epoch to calendar | `gmtime`/`localtime` (not thread-safe) | Pure integer arithmetic (civil_from_days) | Thread-safe, no system call overhead, well-understood algorithm |

**Key insight:** The checksum implementation should stay entirely on GPU. Hashing per-row and reducing with XOR avoids any large GPU-to-host data transfer, which is critical for batches with millions of rows.

## Common Pitfalls

### Pitfall 1: Sliced String Column Offset Handling
**What goes wrong:** After `cudf::slice`, string offsets are relative to the original column, not the sliced view. Reading `chars_begin(stream)` gives the start of the original chars buffer, but offsets may point beyond the slice.
**Why it happens:** `cudf::slice` creates a view with an `offset()` but does not adjust the chars buffer pointer or the offsets values.
**How to avoid:** For the sliced view, read `num_rows + 1` offsets starting from `offsets_col.data<int32_t>() + col.offset()`. The chars pointer from `chars_begin(stream)` is already the base. Subtract the first offset value to get the relative byte range: `chars_ptr + offsets[0]` to `chars_ptr + offsets[N]`.
**Warning signs:** Strings contain garbage characters, wrong lengths, or crash with out-of-bounds access.

### Pitfall 2: `__int128_t` Has No fmt Format Specifier
**What goes wrong:** `fmt::format("{}", int128_value)` fails to compile because there is no formatter for `__int128_t`.
**Why it happens:** `__int128_t` is a compiler extension, not a standard C++ type. Neither `<iostream>` nor `fmtlib` support it natively.
**How to avoid:** Implement a manual `int128_to_string` helper that repeatedly divides by 10 and builds digits in reverse. This is straightforward for the fixed-point decimal case where you just need the raw digits.
**Warning signs:** Compilation error mentioning "cannot format argument" or "no matching overload".

### Pitfall 3: Negative DECIMAL Values with Decimal Point Insertion
**What goes wrong:** Inserting a decimal point at position `|scale|` from the right fails for negative values if the sign is not handled first.
**Why it happens:** The raw integer representation of `-123` with scale=-2 should display as `-1.23`, but naive string manipulation may produce `1.-23` or `-1.2-3`.
**How to avoid:** Extract and strip the sign first, format the absolute value with decimal point insertion, then prepend the minus sign.
**Warning signs:** Negative decimals display incorrectly or have misplaced decimal points.

### Pitfall 4: DECIMAL Values Smaller Than 1.0
**What goes wrong:** Raw value `5` with scale=-2 should display as `0.05`, but naive formatting produces `5` or `.05` (missing leading zero).
**Why it happens:** When the raw integer has fewer digits than `|scale|`, leading zeros must be added.
**How to avoid:** After converting to string, if the string length is less than or equal to `|scale|`, pad with leading zeros to ensure at least one digit before the decimal point.
**Warning signs:** Values like `0.05` display as `.05` or `5`.

### Pitfall 5: Negative Timestamps (Pre-1970 Dates)
**What goes wrong:** Dates before 1970-01-01 have negative epoch values. Integer division and modulo behave differently for negative values in C++.
**Why it happens:** C++ truncates toward zero for negative division, so `-1 / 86400` is `0`, not `-1`.
**How to avoid:** Use floor division for splitting epoch values into days and sub-day components. For the civil_from_days algorithm, negative day counts are handled correctly by the algorithm itself.
**Warning signs:** Pre-1970 dates show wrong year/month/day or wrong time-of-day component.

### Pitfall 6: XOR Reduce of Empty or All-NULL Column
**What goes wrong:** `cudf::reduce` with bitwise XOR on an empty column or all-NULL column returns an invalid scalar.
**Why it happens:** Bitwise aggregation with no valid input has no identity element in cudf's implementation.
**How to avoid:** Check `col.size() == 0` or `col.null_count() == col.size()` before calling reduce. For empty/all-null columns, output a special marker like `0x0000000000000000` with a note.
**Warning signs:** Segfault or assertion failure from accessing an invalid scalar.

### Pitfall 7: `debug_head` Signature Change Breaks Backward Compatibility
**What goes wrong:** Adding `max_string_len` parameter changes the function signature.
**Why it happens:** Existing call sites do not pass the new parameter.
**How to avoid:** Add `max_string_len` as a defaulted parameter (`cudf::size_type max_string_len = 50`) AFTER the existing parameters. Since existing callers already use default values for later parameters, the new parameter must be positioned to maintain backward compatibility. Place it after `format` but before `col_names`, or as the last parameter with a default.
**Warning signs:** Compilation errors at existing call sites.

## Code Examples

### STRING Extraction for debug_head
```cpp
// Source: Pattern from iceberg_scan_task.cpp adapted for debug_head [VERIFIED: codebase pattern]
auto extract_string = [&](cudf::size_type max_str_len) {
  cudf::strings_column_view scv(col);
  // Copy offsets for the sliced range (col.offset() adjusts for sliced views)
  auto const num_offsets = num_rows + 1;
  std::vector<int32_t> host_offsets(num_offsets);
  cudaMemcpyAsync(host_offsets.data(),
                  scv.offsets().data<int32_t>() + col.offset(),
                  num_offsets * sizeof(int32_t),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  // Copy only the chars we need
  stream.synchronize();
  auto const chars_start = host_offsets[0];
  auto const chars_end   = host_offsets[num_rows];
  auto const chars_bytes = chars_end - chars_start;
  std::vector<char> host_chars(chars_bytes);
  if (chars_bytes > 0) {
    cudaMemcpyAsync(host_chars.data(),
                    scv.chars_begin(stream) + chars_start,
                    chars_bytes,
                    cudaMemcpyDeviceToHost,
                    stream.value());
    stream.synchronize();
  }
  for (cudf::size_type r = 0; r < num_rows; ++r) {
    if (nulls.is_null(col.offset() + r)) {
      cells[c][r] = "NULL";
    } else {
      auto const start = host_offsets[r] - chars_start;
      auto const end   = host_offsets[r + 1] - chars_start;
      std::string val(host_chars.data() + start, end - start);
      // Truncate if needed (D-02)
      if (max_str_len > 0 && val.size() > static_cast<std::size_t>(max_str_len)) {
        val.resize(max_str_len);
        val += "...";
      }
      cells[c][r] = std::move(val);
    }
  }
};
```

### DECIMAL Formatting
```cpp
// Source: Manual implementation for fixed-point display [ASSUMED pattern, verified storage types]
// DECIMAL32: int32_t storage, DECIMAL64: int64_t storage, DECIMAL128: __int128_t storage
// col.type().scale() is negative: -2 means 2 decimal places

auto extract_decimal = [&]<typename T>() {
  int32_t scale = col.type().scale();  // negative, e.g. -2
  int abs_scale = std::abs(scale);
  std::vector<T> host_vals(num_rows);
  cudaMemcpyAsync(host_vals.data(), col.data<T>(),
                  sizeof(T) * num_rows, cudaMemcpyDeviceToHost,
                  stream.value());
  stream.synchronize();
  for (cudf::size_type r = 0; r < num_rows; ++r) {
    if (nulls.is_null(col.offset() + r)) {
      cells[c][r] = "NULL";
    } else {
      cells[c][r] = format_decimal(host_vals[r], abs_scale);
    }
  }
};

// Helper: format_decimal for int32_t/int64_t
std::string format_decimal(int64_t raw, int abs_scale) {
  if (abs_scale == 0) return fmt::format("{}", raw);
  bool negative = raw < 0;
  // Use unsigned for magnitude to handle INT64_MIN correctly
  uint64_t magnitude = negative ? static_cast<uint64_t>(-raw) : static_cast<uint64_t>(raw);
  std::string digits = fmt::format("{}", magnitude);
  // Pad with leading zeros if digits shorter than abs_scale
  while (digits.size() <= static_cast<std::size_t>(abs_scale)) {
    digits.insert(digits.begin(), '0');
  }
  // Insert decimal point
  auto decimal_pos = digits.size() - abs_scale;
  digits.insert(decimal_pos, ".");
  return (negative ? "-" : "") + digits;
}
```

### Timestamp/Date Formatting
```cpp
// Source: Howard Hinnant's civil_from_days algorithm (public domain) [CITED: https://howardhinnant.github.io/date_algorithms.html]
// Convert days since Unix epoch to year/month/day
struct civil_date { int year; unsigned month; unsigned day; };
civil_date civil_from_days(int32_t days) {
  // Shift epoch from 1970-01-01 to 0000-03-01 for simpler leap year math
  int z = days + 719468;
  int era = (z >= 0 ? z : z - 146096) / 146097;
  unsigned doe = static_cast<unsigned>(z - era * 146097);
  unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
  int y = static_cast<int>(yoe) + era * 400;
  unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);
  unsigned mp = (5*doy + 2)/153;
  unsigned d = doy - (153*mp+2)/5 + 1;
  unsigned m = mp + (mp < 10 ? 3 : -9);
  y += (m <= 2);
  return {y, m, d};
}

// For TIMESTAMP_MICROSECONDS (int64_t us since epoch):
int64_t raw_us = ...; // from GPU
int64_t total_seconds = (raw_us >= 0) ? raw_us / 1'000'000
                                       : (raw_us - 999'999) / 1'000'000;
int32_t days = static_cast<int32_t>((total_seconds >= 0) ? total_seconds / 86400
                                                          : (total_seconds - 86399) / 86400);
int seconds_in_day = static_cast<int>(total_seconds - static_cast<int64_t>(days) * 86400);
int frac_us = static_cast<int>(raw_us - total_seconds * 1'000'000);
auto [y, m, d] = civil_from_days(days);
int hh = seconds_in_day / 3600;
int mm = (seconds_in_day % 3600) / 60;
int ss = seconds_in_day % 60;
// Format: "2024-01-15 08:30:00" or "2024-01-15 08:30:00.123456"
std::string result = fmt::format("{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}",
                                  y, m, d, hh, mm, ss);
if (frac_us != 0) {
  result += fmt::format(".{:06d}", frac_us);
  // Trim trailing zeros from fractional part
  while (result.back() == '0') result.pop_back();
}
```

### debug_checksum Implementation
```cpp
// Source: cudf/hashing.hpp [VERIFIED: header] + cudf/aggregation.hpp [VERIFIED: header grep]
void debug_checksum(cucascade::data_batch const& batch,
                    rmm::cuda_stream_view stream) {
  // ... tier guard, try/catch ...
  cudf::table_view tv = get_cudf_table_view(batch);
  stream.synchronize();
  auto mr = cudf::get_current_device_resource_ref();

  std::string output;
  output += fmt::format("[SIRIUS_DIAG] checksum: batch_id={} rows={} cols={}\n",
                        batch.get_batch_id(), tv.num_rows(), tv.num_columns());

  for (cudf::size_type c = 0; c < tv.num_columns(); ++c) {
    auto const& col = tv.column(c);
    std::string name = fmt::format("col[{}]", c);
    auto nc = col.null_count();
    if (nc < 0) nc = 0;

    if (col.size() == 0 || col.null_count() == col.size()) {
      output += fmt::format("[SIRIUS_DIAG]   {} checksum: 0x{:016X} nulls={}\n",
                            name, uint64_t{0}, static_cast<int>(nc));
      continue;
    }

    // Hash the single column
    cudf::table_view single_col_tv({col});
    auto hash_col = cudf::hashing::xxhash_64(single_col_tv, 0, stream, mr);

    // XOR reduce
    auto xor_agg = cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(
        cudf::bitwise_op::XOR);
    auto result = cudf::reduce(hash_col->view(), *xor_agg,
        cudf::data_type{cudf::type_id::UINT64}, stream, mr);

    auto& scalar = static_cast<cudf::numeric_scalar<uint64_t> const&>(*result);
    uint64_t checksum = scalar.is_valid(stream) ? scalar.value(stream) : 0;

    output += fmt::format("[SIRIUS_DIAG]   {} checksum: 0x{:016X} nulls={}\n",
                          name, checksum, static_cast<int>(nc));
  }

  SIRIUS_LOG_DEBUG("{}", output);
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `cudf::strings_column_view::chars()` method | `chars_begin(stream)` / `chars_size(stream)` with stream parameter | cuDF 24.x | Must use stream-aware accessors; old `chars()` no longer exists in cuDF 26.02 [VERIFIED: header read] |
| Separate `offsets_begin()`/`offsets_end()` iterators | `offsets()` returns `column_view`, use `.data<int32_t>()` | cuDF 26.02 | No dedicated iterator method; use column_view data accessor [VERIFIED: header read] |

**Deprecated/outdated:**
- `cudf::strings_column_view::chars()` (no stream parameter): Removed in cuDF 24.x+. Use `chars_begin(stream)` instead. [VERIFIED: header shows only stream-based API]

## Project Constraints (from CLAUDE.md)

- All output via `SIRIUS_LOG_DEBUG` or `SIRIUS_LOG_TRACE` with `[SIRIUS_DIAG]` prefix
- Buffer entire output into single `std::string` before emitting one log call (thread safety)
- Use `pixi shell` for builds; build with `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make`
- Run unit tests with `build/release/extension/sirius/test/cpp/sirius_unittest "[debug_utils]"`
- Code must pass `pre-commit run -a` (clang-format, clang-tidy, codespell)
- All functions wrapped in try/catch -- debug call must never crash the pipeline
- C++20 and CUDA 20 standards
- Follow existing naming conventions: PascalCase for public methods, snake_case for functions and variables

## Assumptions Log

> List all claims tagged `[ASSUMED]` in this research.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Howard Hinnant's civil_from_days algorithm works correctly for negative epoch days (pre-1970) | Code Examples - Timestamp | Would produce wrong dates for pre-1970 timestamps; algorithm is well-tested but implementation must be verified |
| A2 | `cudf::reduce` with `make_bitwise_aggregation(bitwise_op::XOR)` works on UINT64 columns | Architecture Patterns - Checksum | Aggregation docs say "supports only integral types" which includes UINT64; if it fails, would need a custom kernel or alternative approach |
| A3 | DECIMAL formatting via manual string manipulation with `__int128_t` handles all edge cases | Code Examples - DECIMAL | Edge cases like `INT128_MIN`, zero scale, and very large values need testing |

**If this table is empty:** N/A -- 3 assumptions identified above.

## Open Questions

1. **`max_string_len` parameter position in `debug_head` signature**
   - What we know: D-03 says it is a function parameter with default 50. Current signature has `(batch, n, stream, format, col_names)`.
   - What is unclear: Should it go after `format` and before `col_names`? Or as the very last parameter?
   - Recommendation: Place after `col_names` to avoid breaking existing call sites that use positional args for `col_names`. Since `col_names` has a default, adding after it is safe: `..., col_names = {}, max_string_len = 50)`.

2. **Should `debug_checksum` accept `col_names` parameter?**
   - What we know: Other functions (debug_schema, debug_head, debug_stats) all accept optional col_names.
   - What is unclear: Whether checksum output benefits from column names vs. just col[N] indices.
   - Recommendation: Include `col_names` for consistency. The planner can decide, but it costs nothing and aids readability.

3. **`debug_checksum` memory resource parameter**
   - What we know: `xxhash_64` and `reduce` both accept a memory resource. The function allocates temporary GPU memory for the hash column.
   - What is unclear: Whether to use `cudf::get_current_device_resource_ref()` (global default) or require an explicit `mr` parameter.
   - Recommendation: Use `cudf::get_current_device_resource_ref()` internally since debug utilities are diagnostic-only and the temporary allocation is small (one UINT64 per row per column).

## Sources

### Primary (HIGH confidence)
- `cudf/hashing.hpp` (installed at `.pixi/envs/default/include/cudf/hashing.hpp`) -- xxhash_64 API signature, seed parameter, return type
- `cudf/aggregation.hpp` (installed at `.pixi/envs/default/include/cudf/aggregation.hpp`) -- `make_bitwise_aggregation`, `bitwise_op::XOR`, `reduce_aggregation`
- `cudf/strings/strings_column_view.hpp` (installed at `.pixi/envs/default/include/cudf/strings/strings_column_view.hpp`) -- `chars_begin(stream)`, `chars_size(stream)`, `offsets()` API
- `cudf/fixed_point/fixed_point.hpp` -- DECIMAL32/64/128 type aliases, scale_type definition
- `cudf/wrappers/timestamps.hpp` / `durations.hpp` -- Timestamp storage types (int32_t for days, int64_t for s/ms/us/ns)
- `cudf/reduction.hpp` -- reduce API, BITWISE_AGG row in documentation table
- `src/include/cudf/cudf_utils.hpp` -- GetCudfType mapping from DuckDB types to cudf types (DATE->TIMESTAMP_DAYS, TIMESTAMP->TIMESTAMP_MICROSECONDS)
- `src/op/scan/iceberg_scan_task.cpp` lines 88-108 -- Verified string extraction pattern using strings_column_view
- `src/op/merge/gpu_merge_impl.cpp` line 199 -- Verified `chars_size(stream)` usage pattern
- `src/op/aggregate/gpu_aggregate_impl.cpp` lines 81-88 -- Verified DECIMAL type handling pattern
- `test/cpp/utils/data_utils.hpp` -- Test utility: `vector_to_cudf_column` already supports strings, decimals, timestamps
- `test/cpp/operator/operator_type_traits.hpp` -- Test type traits: `string_tag`, `decimal64_tag`, `timestamp_us_tag`, `date32_tag` ready for use

### Secondary (MEDIUM confidence)
- Howard Hinnant's date algorithms -- civil_from_days algorithm for epoch-to-calendar conversion (widely used, public domain)

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all cuDF APIs verified against installed headers in pixi env
- Architecture: HIGH -- string extraction pattern verified in 5+ codebase files; checksum API verified in headers
- Pitfalls: HIGH -- blocker issues identified from codebase patterns and cuDF API constraints

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (30 days; cuDF 26.02 API is stable)
