# Phase 2: Numeric Row Preview and Column Statistics - Context

**Gathered:** 2026-04-06 (updated 2026-04-06)
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement `debug_head(batch, N, stream)` for numeric types with aligned-column and CSV output, and `debug_stats(batch, stream)` with GPU-side min/max/sum reduction. All output routed through `[SIRIUS_DIAG]` log prefix. STRING, DECIMAL, TIMESTAMP, and DATE type support deferred to Phase 3.

</domain>

<decisions>
## Implementation Decisions

### Format Selection API
- **D-01:** `debug_head` uses an `enum class DebugFormat { ALIGNED, CSV }` parameter with default `DebugFormat::ALIGNED` — type-safe, extensible for future formats
- **D-02:** Default row count is 10 when N is not specified (matches pandas convention)

### Display Formatting
- **D-03:** Column widths are dynamic — scan the N rows to find max display width per column, then pad all values for perfectly aligned output
- **D-04:** Floating-point numbers use 6 significant digits (`%g`-style) — fixed notation for normal ranges, scientific for very large/small values
- **D-05:** Booleans display as lowercase `true`/`false`
- **D-06:** NULLs display as the string `NULL` in row output

### Stats Output
- **D-07:** `debug_stats` uses a summary table format (one row per column) consistent with `debug_schema`/`debug_nulls` — columns: idx, name, type, min, max, sum
- **D-08:** Non-numeric columns show `(non-numeric, skipped)` in the stats table
- **D-09:** Min/max/sum only — no count or mean (count is in the header, mean is derivable). Keeps output compact and GPU reduce calls minimal
- **D-10:** All-NULL numeric columns show `NULL` for min, max, and sum — follows SQL standard semantics (`SUM/MIN/MAX` of all NULLs = NULL)

### GPU-to-Host Data Transfer
- **D-14:** Bulk copy per column: one `cudaMemcpyAsync` per column after `cudf::slice`, issue all async copies first, then a single `stream.synchronize()` at the end — avoids per-column sync overhead

### Error Edge Cases
- **D-11:** No cap on N — trust the caller. Try/catch wrapping handles OOM gracefully
- **D-12:** When N > batch row count, clamp silently to `min(N, num_rows)`. Header already shows total row count so the developer sees the batch was smaller
- **D-13:** Empty batches (0 rows) print header info with an `(empty batch)` note — no data rows

### Claude's Discretion
- Header separator style (dashes, equals, etc.)
- CSV quoting/escaping rules
- Internal helper function decomposition
- cudf::reduce vs cudf::minmax optimization choice for min/max computation

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` — HEAD-01, HEAD-02, HEAD-03, STATS-01, STATS-02, STATS-03 define the exact requirements for this phase

### Phase 1 Implementation (foundation)
- `src/include/debug_utils.hpp` — Current debug utility API with `host_column_nulls`, `copy_null_mask_to_host`, `debug_schema`, `debug_nulls`
- `src/debug_utils.cpp` — Implementation patterns: tier guard, output buffering, try/catch wrapping, `[SIRIUS_DIAG]` log routing

### Existing cudf::reduce Usage
- `src/cuda/cudf/cudf_aggregate.cu` — Existing `cudf::reduce` patterns for MIN/MAX/SUM with `make_reduce_aggregation<>` helpers

### Test Patterns
- `test/cpp/debug/test_debug_utils.cpp` — Phase 1 Catch2 tests showing how to create GPU-backed data batches for testing

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `host_column_nulls` + `copy_null_mask_to_host`: Already handles null bitmask copy from GPU to host — reuse for per-row null checking in debug_head
- `is_gpu_tier()`: Internal tier guard helper — reuse for debug_head and debug_stats
- `get_cudf_table_view(batch)`: Extracts `cudf::table_view` from `cucascade::data_batch`
- `cudf::type_to_name(col.type())`: Converts cudf type id to human-readable string

### Established Patterns
- Output buffered into single `std::string`, emitted via one `SIRIUS_LOG_DEBUG("{}", output)` call
- All debug functions: try/catch wrapping, tier guard check, stream sync
- Column naming: use `col_names` vector if provided, fallback to `col[N]` format
- Header line: `[SIRIUS_DIAG] <func_name>: batch_id=<id> rows=<count> cols=<count>`

### Integration Points
- New functions added to `src/include/debug_utils.hpp` (declaration) and `src/debug_utils.cpp` (implementation)
- `DebugFormat` enum also declared in `debug_utils.hpp`
- Tests added to `test/cpp/debug/test_debug_utils.cpp`
- `cudf::slice` for zero-copy row selection (HEAD-03) — used elsewhere in codebase

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-numeric-row-preview-and-column-statistics*
*Context gathered: 2026-04-06*
