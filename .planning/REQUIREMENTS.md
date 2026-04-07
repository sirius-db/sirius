# Requirements: Sirius Debug Utilities

**Defined:** 2026-04-06
**Core Value:** Enable fast, accurate identification of faulty operators by providing consistent, pretty-printed data inspection at any point in the GPU execution pipeline.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Infrastructure

- [ ] **INFRA-01**: All debug functions accept `rmm::cuda_stream_view stream` parameter and use `stream.synchronize()` — never `cudaDeviceSynchronize()`
- [ ] **INFRA-02**: Null-aware GPU-to-host copy helper that reads null bitmask alongside column values and represents NULLs as `"NULL"` in output
- [ ] **INFRA-03**: Type dispatch covers all Sirius-supported types: INT8/16/32/64, UINT8/16/32/64, FLOAT32/64, BOOL8, STRING, DECIMAL, TIMESTAMP, DATE
- [ ] **INFRA-04**: All output routed through `SIRIUS_LOG_DEBUG` or `SIRIUS_LOG_TRACE` with `[SIRIUS_DIAG]` prefix — not printf/stdout
- [ ] **INFRA-05**: Entire table/output is buffered into a single `std::string` and emitted in one atomic log call for thread safety
- [ ] **INFRA-06**: All debug functions wrapped in try/catch — a crashing debug call must never crash the pipeline

### Schema Inspection

- [ ] **SCHEMA-01**: `debug_schema(batch)` prints column names (if available), data types, null counts, and total row count
- [ ] **SCHEMA-02**: Output is a compact summary table (one row per column) via SIRIUS_LOG

### Null Analysis

- [ ] **NULL-01**: `debug_nulls(batch)` prints per-column null count and null percentage
- [ ] **NULL-02**: Uses `column_view::null_count()` metadata (zero GPU cost) — no kernel launch required

### Row Preview

- [ ] **HEAD-01**: `debug_head(batch, N)` prints first N rows in aligned-column format (fixed-width, pandas-style)
- [ ] **HEAD-02**: `debug_head(batch, N, format=csv)` prints first N rows in CSV format
- [ ] **HEAD-03**: Uses `cudf::slice` for zero-copy row selection before GPU-to-host transfer
- [ ] **HEAD-04**: STRING columns extracted via `cudf::strings_column_view` with proper two-buffer (offsets + chars) host copy
- [ ] **HEAD-05**: DECIMAL columns display with correct scale factor from `col.type().scale()`
- [ ] **HEAD-06**: TIMESTAMP and DATE columns display as human-readable calendar format (not raw epoch integers)

### Column Statistics

- [ ] **STATS-01**: `debug_stats(batch)` prints per-column min, max, sum for numeric columns only
- [ ] **STATS-02**: Non-numeric columns (STRING, BOOL, DATE, TIMESTAMP) are skipped with a note (e.g., `"(non-numeric, skipped)"`)
- [ ] **STATS-03**: Uses `cudf::reduce` / `cudf::minmax` for GPU-side computation — no full column copy to host

### Column Checksum

- [ ] **CHKSUM-01**: `debug_checksum(batch)` computes and logs per-column hash fingerprint
- [ ] **CHKSUM-02**: Uses `cudf::hashing::xxhash_64` for consistent cross-run comparison
- [ ] **CHKSUM-03**: Output format enables easy diff between two log files (e.g., `col[0] checksum: 0xABCD1234`)

### Batch Diff

- [ ] **DIFF-01**: `debug_diff(batch_a, batch_b)` compares two data batches and logs which rows and columns differ
- [ ] **DIFF-02**: Reports schema mismatches (different column count, types) before attempting value comparison
- [ ] **DIFF-03**: Reports row count mismatch
- [ ] **DIFF-04**: For matching schemas, reports per-column diff count and first N differing row indices
- [ ] **DIFF-05**: Guards behind configurable row count limit to prevent OOM on large batches

### Random Sampling

- [ ] **SAMPLE-01**: `debug_sample(batch, N)` prints N randomly selected rows from the batch
- [ ] **SAMPLE-02**: Uses the same output formatting as `debug_head` (aligned columns + CSV options)
- [ ] **SAMPLE-03**: Useful for catching bugs that don't appear in first rows

### Skill Integration

- [ ] **SKILL-01**: `/validate` SKILL.md references debug utilities and instructs Claude to use `debug_checksum`, `debug_stats`, `debug_head`, `debug_diff` instead of ad-hoc SIRIUS_LOG_TRACE checksum lines
- [ ] **SKILL-02**: `/runtime-errors` SKILL.md references debug utilities and instructs Claude to use `debug_schema`, `debug_head`, `debug_nulls` for data inspection at suspected fault points
- [ ] **SKILL-03**: Both skills document the function signatures and usage examples

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extended Utilities

- **EXT-01**: `debug_histogram(batch, col_idx)` — value distribution for a single column
- **EXT-02**: `debug_memory(batch)` — GPU memory usage, batch tier (GPU/HOST/STORAGE), allocation size
- **EXT-03**: Conditional debug macros that compile out entirely in release builds
- **EXT-04**: Python bindings for interactive debugging via Jupyter

## Out of Scope

| Feature | Reason |
|---------|--------|
| GUI/web-based data viewer | Over-engineering for a log-based debug toolkit |
| Persistent data dumping to files | SIRIUS_LOG already writes to files; no separate dump path needed |
| Runtime performance profiling | Covered by `/profile-analyzer` skill |
| Production-safe data sampling | Debug utilities are diagnostic-only, not production code paths |
| stdout/stderr output mode | Skills parse log files only; stdout is invisible to them |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | TBD | Pending |
| INFRA-02 | TBD | Pending |
| INFRA-03 | TBD | Pending |
| INFRA-04 | TBD | Pending |
| INFRA-05 | TBD | Pending |
| INFRA-06 | TBD | Pending |
| SCHEMA-01 | TBD | Pending |
| SCHEMA-02 | TBD | Pending |
| NULL-01 | TBD | Pending |
| NULL-02 | TBD | Pending |
| HEAD-01 | TBD | Pending |
| HEAD-02 | TBD | Pending |
| HEAD-03 | TBD | Pending |
| HEAD-04 | TBD | Pending |
| HEAD-05 | TBD | Pending |
| HEAD-06 | TBD | Pending |
| STATS-01 | TBD | Pending |
| STATS-02 | TBD | Pending |
| STATS-03 | TBD | Pending |
| CHKSUM-01 | TBD | Pending |
| CHKSUM-02 | TBD | Pending |
| CHKSUM-03 | TBD | Pending |
| DIFF-01 | TBD | Pending |
| DIFF-02 | TBD | Pending |
| DIFF-03 | TBD | Pending |
| DIFF-04 | TBD | Pending |
| DIFF-05 | TBD | Pending |
| SAMPLE-01 | TBD | Pending |
| SAMPLE-02 | TBD | Pending |
| SAMPLE-03 | TBD | Pending |
| SKILL-01 | TBD | Pending |
| SKILL-02 | TBD | Pending |
| SKILL-03 | TBD | Pending |

**Coverage:**
- v1 requirements: 33 total
- Mapped to phases: 0
- Unmapped: 33

---
*Requirements defined: 2026-04-06*
*Last updated: 2026-04-06 after initial definition*
