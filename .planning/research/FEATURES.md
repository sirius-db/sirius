# Feature Research

**Domain:** C++ debug inspection utilities for a GPU SQL engine (Sirius / cuDF / cucascade)
**Researched:** 2026-04-06
**Confidence:** HIGH (derived primarily from codebase analysis and first-party docs)

## Context: What Exists Today

The current `print.hpp`/`print.cu` baseline is narrow:
- Only numeric/bool types printed; STRING, TIMESTAMP, DECIMAL, DATE fall through to `(unprinted type ...)`
- Output goes to `std::printf` (stdout), bypassing the `SIRIUS_LOG_*` pipeline that skills parse
- No column names — output is `col[0]`, `col[1]`, etc.
- No null awareness — null bitmask is ignored; null rows print garbage values
- No statistics, no checksums, no diffing, no schema inspection
- Only `cudf::table_view` and `cucascade::data_batch` overloads; no batch ID in table variant

The `/validate` skill currently constructs ad-hoc `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] ...")` with manually written `sum()`, `max()`, `head(1)` computations. The `/runtime-errors` skill does the same for variable dumps and data characterization. Both skills would benefit from calling named, reusable functions instead.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features a debug toolkit for a GPU pipeline engine must have. Missing any of these means the engineer has to go back to ad-hoc printf debugging, which is the exact problem being solved.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| `debug_head(batch, N)` — first N rows, aligned columns + CSV | Every SQL engine debugger expects "show me the data." pandas `.head()`, DuckDB `.show()`, Spark `.show()` all provide this. The `/validate` skill explicitly calls for `head(1)`. | MEDIUM | Requires GPU-to-host copy, null-aware value formatting, fixed-width column alignment, and CSV mode for programmatic parsing. STRING columns need `strings_column_view` (offset buffer + chars buffer, not `col.data<char*>()`). DECIMAL needs scale. TIMESTAMP/DATE need epoch decoding. Output goes through `SIRIUS_LOG_DEBUG` with `[SIRIUS_DIAG]` prefix. |
| `debug_schema(batch)` — column names, types, null counts, row count | Schema inspection is prerequisite to any other debugging. Knowing type mismatch, unexpected nullability, or wrong column count identifies entire classes of bugs before looking at values. | LOW | `cudf::column_view::type()`, `col.null_count()`, `table.num_rows()`. Column names must come from caller (cudf table_view has no names); accept `std::vector<std::string>` names param with sensible default (`col[i]`). |
| `debug_nulls(batch)` — per-column null count and null percentage | Null propagation bugs are extremely common in SQL engines. Knowing which columns have unexpected nulls narrows the search immediately. `/validate` skill documents checking null counts as part of data validation. | LOW | Straightforward: `col.null_count()` / `col.size()`. No GPU-to-host copy needed for the counts. Can leverage cudf null_count() which is maintained by cuDF on modification. |
| `debug_stats(batch)` — per-column min, max, sum | The `/validate` skill explicitly lists `sum()` and `max()` as the standard checksum proxies it currently hand-codes. Making this a named function removes the most common ad-hoc pattern. | MEDIUM | Uses `cudf::reduce()` with `make_min_aggregation<reduce_aggregation>()`, `make_max_aggregation<>()`, `make_sum_aggregation<>()`. STRING columns: skip sum, provide min/max lexicographically. BOOL: sum = count of true. TIMESTAMP/DATE: min/max as epoch values. DECIMAL: must pass correct output_dtype with scale. Numeric overflow on sum is acceptable for debug use — it's a fingerprint, not an exact value. |
| `debug_checksum(batch)` — per-column hash fingerprint for cross-run comparison | Checksums are the primary tool for detecting data divergence across pipeline runs without transferring full data. `/validate` Phase 2 is entirely checksum-driven. BlazingSQL, Spark Deequ, and SQL Server's `CHECKSUM_AGG` all use this pattern. cuDF provides `murmurhash3_x86_32()` in the hashing module. | MEDIUM | Use `cudf::hashing::murmurhash3_x86_32()` to hash all columns, then reduce to a single 32-bit integer per column via XOR-fold or sum. Alternatively use `cudf::reduce(SUM)` on the hash output column. Output as hex for readability. Must handle STRING (hash the string data) and DECIMAL/TIMESTAMP/DATE (hash raw representation). |
| Log routing via `SIRIUS_LOG_*` macros with `[SIRIUS_DIAG]` prefix | Without log routing, debug output is lost to stdout during multi-threaded runs and is invisible to the skills that parse `$SIRIUS_LOG_DIR/*.log`. The `[SIRIUS_DIAG]` prefix is already the established convention in both skills. | LOW | Replace all `std::printf` in new/updated functions with `SIRIUS_LOG_DEBUG("[SIRIUS_DIAG] ...")`. Note: `logging.hpp` no-ops all macros under `__CUDACC__`, so debug functions must be in `.cpp` files, not `.cu` kernels. The existing `print.cu` already does the GPU-to-host copy in CPU code, so this is compatible. |
| Full Sirius type coverage | Existing code leaves STRING, TIMESTAMP, DATE, DECIMAL printing as `(unprinted type ...)`. Any debug function that silently skips supported types is useless at exactly the moment when the suspicious column is of that type. | MEDIUM | STRING: copy via `strings_column_view.chars()` and offsets. DECIMAL: `fixed_point_scalar.value()` — need scale to display as `1.23` not `123`. TIMESTAMP variants: decode epoch + units (days/seconds/microseconds) to ISO string. DATE (TIMESTAMP_DAYS): decode as calendar date. |

### Differentiators (Competitive Advantage)

Features that go beyond what any debug tool must have, making this toolkit genuinely useful beyond manual log insertion.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| `debug_diff(batch_a, batch_b)` — row- and column-level diff of two batches | Directly answers the `/validate` Phase 2 question "where do checksums first diverge?" with precise row-level detail rather than requiring multiple manual checksum comparisons. The `data-diff` open-source ecosystem demonstrates this pattern is high-value for pipeline debugging. | HIGH | Cannot use SQL OUTER JOIN inside a C++ debug utility without a DuckDB context. Practical approach: copy both batches to host, sort both by all columns, then walk row-by-row comparing. Only show first N differing rows. Report column mask of which columns differ per row. Requires careful null handling (null == null in diff context). Hard requirement: batch_a and batch_b must have identical schemas (same column types and count); validate and log a clear error if not. |
| Batch ID and call-site tagging on every output line | When multiple operators run concurrently and log interleaves, messages without batch IDs are impossible to correlate. The existing `print_data_batch_contents` already logs `id=%llu` — making this systematic across all functions lets skills grep for specific batches in the pipeline. | LOW | Thread ID + batch_id header per debug function call: `[SIRIUS_DIAG][batch=42][thread=7] debug_schema:`. Use `std::this_thread::get_id()`. |
| Consistent multi-format output per function (aligned + CSV) | Aligned columns are readable by humans reviewing log files. CSV mode allows skills to programmatically parse and compare values without fragile regex against a pretty-printed table. DuckDB's `BoxRenderer` is the model for aligned output (already used in the legacy `printGPUTable` path). | LOW | Two output modes controlled by a parameter or a build flag. Aligned mode: compute column widths from max value string length. CSV mode: comma-separated, no padding. The `/validate` skill's Phase 2 compares values programmatically — CSV makes this reliable. |
| `debug_sample(batch, N)` — N random rows (not just head) | For large batches, the first N rows are often all from the same partition or scan chunk and unrepresentative of distribution. Random sampling is standard in pandas `.sample()` and Spark `.sample()`. Useful when the bug only manifests in specific value ranges. | MEDIUM | Use `cudf::sample()` or manual index generation via RMM + cudaMemcpy. Alternatively: compute stride = row_count / N, gather every stride-th row. True random requires a random index column — cudf can generate one. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Full table dump to a file (not via SIRIUS_LOG) | "I want the entire batch as a CSV file for offline analysis" sounds useful | Defeats the log pipeline that skills are designed to parse. Creates uncontrolled disk usage during multi-GB batch debugging. Creates a parallel output channel that requires separate grep tooling. The PROJECT.md explicitly calls this out as out of scope. | Use `debug_head(batch, 1000)` with CSV mode to a log file that skills already parse. If truly needed, implement as a standalone tool outside the debug library. |
| Python wrappers for interactive REPL debugging | "I want to call these from a Jupyter notebook" | Requires building a separate Python extension, maintaining pybind11 bindings, and handling GPU context management across Python GC boundaries. This is a different product from C++ debug utilities. PROJECT.md explicitly marks this out of scope. | The `/validate` and `/runtime-errors` skills already orchestrate debugging from Claude Code, which can read log files. No Python wrapper needed for the use case. |
| GUI or web-based data viewer | "I want to browse data interactively" | Requires a web server, frontend tooling, and a persistent process — a major scope expansion for a debugging utility that runs in a C++ pipeline task thread. | DuckDB's CLI already provides interactive query + table display. For interactive inspection, dump a batch to DuckDB format and browse with the existing CLI. |
| Runtime performance profiling inside debug functions | "While I have the batch, let me also measure kernel time" | Conflates two concerns: data correctness inspection (this project) vs performance profiling (already handled by `/profile-analyzer`). Performance hooks inside debug functions would interfere with the very benchmarks they're meant to profile. | Use `/profile-analyzer` and nsys for performance. Debug functions are correctness tools. |
| Automatic debug insertion without code changes | "Inject debug_head() at every operator boundary without modifying source" | Requires DuckDB/Sirius plugin hooks or source transformation. The engineering cost vastly exceeds the benefit vs the current manual-insertion model. Automatic injection also produces enormous log volume that obscures the signal. | The skills already know which operators to instrument — targeted manual insertion is faster and produces less noise. |
| Floating-point equality in debug_diff | "Show me rows where float columns differ by any amount" | Float equality is undefined without an epsilon. Two runs of the same GPU kernel may produce slightly different float results due to non-deterministic reduction order. This leads to false positives that drown out real differences. | In `debug_diff`, use approximate equality for FLOAT32/FLOAT64 columns (e.g., relative tolerance 1e-6). Report the actual values in differing rows so the engineer can judge materiality. |

---

## Feature Dependencies

```
debug_head(batch, N)
    requires──> GPU-to-host copy layer (per-type specializations)
    requires──> Log routing (SIRIUS_LOG_DEBUG + [SIRIUS_DIAG] prefix)
    requires──> Full type coverage (STRING, DECIMAL, TIMESTAMP, DATE)

debug_stats(batch)
    requires──> GPU-to-host copy layer
    requires──> Log routing
    requires──> Full type coverage (for per-type aggregation dispatch)
    requires──> cudf::reduce() integration (SUM, MIN, MAX per column)

debug_checksum(batch)
    requires──> GPU-to-host copy layer
    requires──> Log routing
    requires──> cudf::hashing::murmurhash3_x86_32() (or reduce-based fingerprint)
    enhances──> debug_diff (diff can use checksums as pre-filter)

debug_schema(batch)
    requires──> Log routing
    (no GPU-to-host copy needed — metadata only)

debug_nulls(batch)
    requires──> Log routing
    requires──> debug_schema (null counts are part of schema output; can be standalone or merged)

debug_diff(batch_a, batch_b)
    requires──> GPU-to-host copy layer
    requires──> Full type coverage (for value comparison)
    requires──> debug_schema (must validate schemas match before diffing)
    enhances──> debug_checksum (run checksum first to fast-path the no-diff case)

debug_sample(batch, N)
    requires──> GPU-to-host copy layer
    requires──> Full type coverage
    requires──> debug_head (same formatting logic; sample just uses different row selection)

Batch ID + call-site tagging
    enhances──> all debug_* functions (add as a header to every output block)

Aligned + CSV dual output mode
    enhances──> debug_head, debug_diff, debug_sample (tabular functions benefit most)
```

### Dependency Notes

- **GPU-to-host copy layer** is the foundational primitive everything else builds on. It must handle all 16+ type IDs, null bitmask checking, and CUDA error reporting. This layer exists in partial form in `print.cu` today but needs to be extended for STRING, DECIMAL, TIMESTAMP, and DATE.
- **Full type coverage** blocks any function that reads values. Cannot ship `debug_stats` that silently skips STRING or DECIMAL columns — the bug is often in exactly that column.
- **`debug_schema` has no GPU-to-host dependency** — it only reads column metadata available on the CPU side of `cudf::column_view`. This makes it the cheapest function to call and the safest to insert in tight paths.
- **`debug_diff` depends on `debug_schema`** because it must validate schemas match before attempting row comparison. A schema mismatch should produce a clear error, not a crash or silent wrong output.
- **`debug_checksum` enhances `debug_diff`**: run checksums first; if all column checksums match, skip the expensive row-level diff. This is the data-diff library's standard optimization strategy.

---

## MVP Definition

### Launch With (v1)

Minimum to make the `/validate` and `/runtime-errors` skills replace their ad-hoc log patterns with named functions.

- [ ] `debug_schema(batch, names)` — column names, types, null counts, row count via SIRIUS_LOG with [SIRIUS_DIAG]
- [ ] `debug_nulls(batch, names)` — per-column null count and percentage via SIRIUS_LOG
- [ ] `debug_head(batch, N, names)` — first N rows in aligned + CSV format via SIRIUS_LOG, full type coverage including STRING / DECIMAL / TIMESTAMP / DATE
- [ ] `debug_stats(batch, names)` — per-column min, max, sum via SIRIUS_LOG, full type coverage
- [ ] `debug_checksum(batch, names)` — per-column hash fingerprint via SIRIUS_LOG
- [ ] All functions accept both `cudf::table_view` and `cucascade::data_batch` overloads to match the existing print.hpp API surface
- [ ] All output tagged with `[SIRIUS_DIAG]` prefix and batch ID where available
- [ ] Extend existing `print.hpp`/`print.cu` rather than new files (PROJECT.md decision)

### Add After Validation (v1.x)

Add once the five core functions are confirmed useful in real debugging sessions.

- [ ] `debug_diff(batch_a, batch_b, names)` — row-level diff; add when `/validate` skill explicitly requests it or when engineers find checksum-then-hunt workflow too slow
- [ ] `debug_sample(batch, N, names)` — random N rows; add when first N rows are demonstrably unrepresentative in actual bugs

### Future Consideration (v2+)

- [ ] Batch ID + thread ID tagging as a structured header on every output block — useful at scale, low urgency for initial validation
- [ ] Dual aligned/CSV mode as explicit parameter — CSV mode is the differentiator but aligned mode suffices for initial skill integration

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| `debug_schema` | HIGH — prerequisite for understanding any batch | LOW | P1 |
| `debug_nulls` | HIGH — null bugs are frequent, cheap to check | LOW | P1 |
| `debug_head` (full types) | HIGH — replaces most ad-hoc printf patterns | MEDIUM | P1 |
| `debug_stats` (min/max/sum) | HIGH — replaces `/validate` Phase 2 hand-coding | MEDIUM | P1 |
| `debug_checksum` | HIGH — `/validate` Phase 2 explicitly needs this | MEDIUM | P1 |
| Full type coverage (STRING/DECIMAL/TIMESTAMP) | HIGH — bugs often manifest in these exact types | MEDIUM | P1 |
| `[SIRIUS_DIAG]` log routing | HIGH — skills cannot parse stdout | LOW | P1 |
| Batch ID tagging in output | MEDIUM — useful for concurrent runs | LOW | P2 |
| Aligned + CSV dual output | MEDIUM — CSV enables programmatic skill parsing | LOW | P2 |
| `debug_diff` | MEDIUM — valuable but builds on checksums | HIGH | P2 |
| `debug_sample` | LOW — niche, head() usually sufficient | MEDIUM | P3 |

**Priority key:**
- P1: Must have for the milestone to deliver value
- P2: Should add if time permits; enhances skill automation
- P3: Defer; nice to have but not blocking any current use case

---

## Competitor / Ecosystem Feature Analysis

| Feature | pandas / cuDF Python API | Spark DataFrame | DuckDB CLI | Sirius (current) | Sirius (target) |
|---------|--------------------------|-----------------|------------|-----------------|-----------------|
| Head N rows | `.head(N)` / `.show()` | `.show(N)` | `.show()` / `LIMIT N` | `print_table_contents` (numeric only, stdout) | `debug_head` (all types, SIRIUS_LOG) |
| Schema / types | `.dtypes` / `.info()` | `.printSchema()` | `DESCRIBE` | Not available | `debug_schema` |
| Null counts | `.isnull().sum()` | `df.select([count(when(isnan(c),c)).alias(c)])` | `COUNT(*) - COUNT(col)` | Not available | `debug_nulls` |
| Column stats | `.describe()` | `.describe()` | `SUMMARIZE` | Not available | `debug_stats` |
| Checksum / fingerprint | Not built-in (Deequ, great_expectations) | Spark Deequ library | `md5(col)` aggregate | Not available | `debug_checksum` |
| Data diff | `data-diff` library | `data-diff`, dbt test | Manual SQL | Not available | `debug_diff` (v1.x) |
| Log routing | N/A (Python context) | Spark UI / log4j | N/A (interactive) | stdout printf | SIRIUS_LOG + [SIRIUS_DIAG] |
| Type coverage | All pandas dtypes | All Spark types | All DuckDB types | INT/UINT/FLOAT/BOOL only | All Sirius types |

The pattern across all ecosystems is consistent: schema inspection, null counting, basic statistics, and row preview are the canonical four primitives every data debugging tool provides. Checksums are the "advanced" feature used in pipeline validation contexts (Deequ, data-diff). Diffing is specialized tooling added on top.

---

## Sources

- PROJECT.md: explicit feature requirements and constraints for this milestone
- `src/include/print.hpp`, `src/cuda/print.cu`: existing implementation — baseline gaps identified by reading the code
- `.claude/skills/validate/SKILL.md`: `/validate` skill workflow — Phase 2 explicitly lists `sum()`, `max()`, `head(1)` as the manual patterns to replace
- `.claude/skills/runtime-errors/SKILL.md`: `/runtime-errors` skill workflow — Phase 2 uses `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] ...")` for variable dumps and data characterization
- `.claude/skills/module-discover/docs/cudf/modules/aggregation.md`: `cudf::reduce()` API for SUM/MIN/MAX
- `.claude/skills/module-discover/docs/cudf/modules/hashing.md`: `cudf::hashing::murmurhash3_x86_32()` for checksum implementation
- `.claude/skills/module-discover/docs/cudf/modules/strings.md`: `cudf::strings_column_view` — required for correct STRING column access
- `.claude/skills/module-discover/docs/cudf/modules/types_core.md`: full type_id enum including DECIMAL/TIMESTAMP/DATE variants
- `.claude/skills/module-discover/docs/cucascade/modules/data.md`: `cucascade::data_batch`, `gpu_table_representation`, `get_cudf_table_view()`
- `src/include/log/logging.hpp`: `SIRIUS_LOG_*` macro definitions — confirmed no-op under `__CUDACC__`, meaning debug functions must live in `.cpp` not `.cu`
- `tools/parse_pipeline_log.py`: the existing pipeline log parser — confirms that skills operate on `SIRIUS_LOG_DIR/*.log` files, not stdout
- data-diff library (datafold/data-diff): checksum-based diff strategy for cross-pipeline table comparison (MEDIUM confidence — open-source library pattern)
- Spark Deequ / great_expectations: industry pattern for column statistics + checksum as data quality primitives (MEDIUM confidence — well-known libraries)

---
*Feature research for: GPU SQL engine debug inspection utilities (Sirius)*
*Researched: 2026-04-06*
