# Phase 3: Full Type Coverage and Checksums - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend `debug_head` type dispatch to STRING, DECIMAL, TIMESTAMP, and DATE columns so all Sirius-supported data types render correctly, and implement `debug_checksum` with per-column xxhash_64 fingerprints for cross-pipeline data comparison. All output continues through `[SIRIUS_DIAG]` log prefix.

</domain>

<decisions>
## Implementation Decisions

### STRING Extraction
- **D-01:** STRING columns extracted via `cudf::strings_column_view` with the standard two-buffer pattern (offsets + chars) for GPU-to-host copy
- **D-02:** Long strings truncated with configurable `max_string_len` parameter on `debug_head`, default 50 characters. Truncated strings get `"..."` suffix. Pass 0 for no truncation
- **D-03:** The `max_string_len` parameter is a function parameter (not a global config setting), following the existing API pattern of explicit parameters with defaults

### DECIMAL Display
- **D-04:** DECIMAL values displayed in fixed-point format with exactly `|scale|` decimal places (e.g., scale=-2: `123.45`, scale=-4: `1.2345`). Trailing zeros preserved (e.g., `10.00` not `10`) to match SQL DECIMAL semantics exactly
- **D-05:** Works with DECIMAL32, DECIMAL64, and DECIMAL128 types — scale read from `col.type().scale()`

### TIMESTAMP/DATE Format
- **D-06:** TIMESTAMP values displayed in SQL-style format: `2024-01-15 08:30:00` (space separator, no T)
- **D-07:** DATE values displayed as `2024-01-15` (date only, no time component)
- **D-08:** Sub-second fractional seconds shown only when non-zero (e.g., `08:30:00.123` for ms, `08:30:00.123456` for us). When fractional part is `.000...`, it's omitted to keep output clean
- **D-09:** All temporal types treated as UTC — no timezone conversion (matches cuDF's storage model)

### Checksum Design
- **D-10:** `debug_checksum` computes per-column fingerprint using `cudf::hashing::xxhash_64` to hash all rows, then `cudf::reduce(XOR)` to collapse to a single 64-bit value per column. Stays entirely on GPU
- **D-11:** Output format: `col[N] checksum: 0xABCD1234EF567890 nulls=2` — includes null_count per column to catch null-handling bugs where hashes match but null counts differ
- **D-12:** Deterministic — same data in same order produces same checksum across runs. Order-dependent (different row order = different checksum)

### Claude's Discretion
- Internal helper function decomposition for type extraction
- Whether to add `max_string_len` to `debug_stats` as well (stats doesn't display string values, so probably not needed)
- xxhash_64 seed value (0 is standard default)
- Whether `debug_checksum` should also log row_count in the header line (likely yes, following existing pattern)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` — HEAD-04, HEAD-05, HEAD-06, CHKSUM-01, CHKSUM-02, CHKSUM-03 define the exact requirements for this phase

### Phase 2 Implementation (extends)
- `src/include/debug_utils.hpp` — Current API with DebugFormat enum, debug_head, debug_stats, debug_schema, debug_nulls
- `src/debug_utils.cpp` — Phase 2 implementation with numeric type dispatch in debug_head, cudf::minmax/reduce in debug_stats

### Existing strings_column_view Usage
- `src/op/merge/gpu_merge_impl.cpp` (line 199) — `cudf::strings_column_view` pattern for extracting string data
- `src/op/aggregate/gpu_aggregate_impl.cpp` (line 149) — Another strings_column_view usage pattern

### Phase 2 Context (prior decisions)
- `.planning/phases/02-numeric-row-preview-and-column-statistics/02-CONTEXT.md` — D-03 (dynamic widths), D-04 (float format), D-06 (NULL display), D-13 (empty batch), D-14 (bulk copy)

### Test Patterns
- `test/cpp/debug/test_debug_utils.cpp` — 19 existing Catch2 tests showing batch creation, null mask setup, and assertion patterns

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cudf::strings_column_view`: Already used in 4+ source files for string extraction — well-understood pattern
- `host_column_nulls` + `copy_null_mask_to_host`: Null handling already works for all types
- `is_gpu_tier()`, `get_cudf_table_view()`: Tier guard and table extraction — reuse unchanged
- `scalar_to_string()`: Helper in debug_utils.cpp for formatting scalar values — may need extension for new types
- `is_stats_numeric()`, `sum_output_type()`: Stats helpers that may need updates if DECIMAL should be included in stats

### Established Patterns
- Type dispatch via switch on `cudf::type_id` in debug_head — extend with new cases for STRING, DECIMAL, TIMESTAMP_*, DATE32
- Output buffered into single `std::string`, emitted via one `SIRIUS_LOG_DEBUG("{}", output)` call
- All debug functions: try/catch wrapping, tier guard check, stream sync
- `"(unsupported)"` placeholder in current debug_head for these types — replace with actual implementations

### Integration Points
- `debug_head` in `src/debug_utils.cpp` — add STRING/DECIMAL/TIMESTAMP/DATE cases to existing switch statement
- `debug_checksum` — new function added to `debug_utils.hpp` (declaration) and `debug_utils.cpp` (implementation)
- `max_string_len` parameter changes `debug_head` signature — existing callers need default value for backward compatibility
- New includes needed: `cudf/strings/strings_column_view.hpp`, `cudf/hashing.hpp` or `cudf/hashing/xxhash_64.hpp`

</code_context>

<specifics>
## Specific Ideas

- Checksum output should enable easy `diff` between two log files — the `col[N] checksum: 0xHEX nulls=N` format is designed for this
- SQL-style timestamp format chosen to match DuckDB output for familiarity

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-full-type-coverage-and-checksums*
*Context gathered: 2026-04-08*
