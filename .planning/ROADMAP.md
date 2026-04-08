# Roadmap: Sirius Debug Utilities

## Overview

This project delivers a structured GPU debug inspection library by extending the existing `print.hpp`/`print.cu` files. The build proceeds in dependency order: infrastructure invariants first (so every feature inherits correct stream sync, null handling, and log routing), then numeric data extraction, then full type coverage and checksums, then the highest-complexity utilities (diff and sampling) plus skill documentation. Every phase ends with a verifiable, callable function that replaces an ad-hoc debug pattern in the codebase.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Infrastructure and Metadata Inspection** - Establish foundational invariants (stream sync, null-aware copy, log routing, try/catch) and ship `debug_schema` + `debug_nulls` as the first integration test
- [ ] **Phase 2: Numeric Row Preview and Column Statistics** - Implement `debug_head` for numeric types with aligned and CSV output, and `debug_stats` with GPU-side min/max/sum reduction
- [ ] **Phase 3: Full Type Coverage and Checksums** - Extend type dispatch to STRING, DECIMAL, TIMESTAMP, DATE for `debug_head`, and implement `debug_checksum` with xxhash_64
- [ ] **Phase 4: Diff, Sampling, and Skill Integration** - Implement `debug_diff`, `debug_sample`, and update both Claude Code skills to reference the complete utility API

## Phase Details

### Phase 1: Infrastructure and Metadata Inspection
**Goal**: The foundational infrastructure is correct and callable — stream-scoped sync, null-aware host copy, single-call output buffering, `[SIRIUS_DIAG]` log routing, and try/catch wrapping are all in place; `debug_schema` and `debug_nulls` are callable and produce structured output in the log file
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-06, SCHEMA-01, SCHEMA-02, NULL-01, NULL-02
**Success Criteria** (what must be TRUE):
  1. Calling `debug_schema(batch, stream)` from a pipeline task produces a `[SIRIUS_DIAG]` block in `sirius.log` with one row per column showing name, type, null count, and total row count
  2. Calling `debug_nulls(batch, stream)` produces a `[SIRIUS_DIAG]` block showing per-column null count and null percentage with no GPU kernel launch
  3. All debug functions accept `rmm::cuda_stream_view` and use stream-scoped sync — `cudaDeviceSynchronize` does not appear in any new code
  4. A debug function called on a non-GPU-tier batch logs a warning and returns without crashing
  5. A debug function that encounters an internal exception logs the error and returns without propagating the exception to the caller
**Plans:** 2 plans
Plans:
- [x] 01-01-PLAN.md — Create debug_utils header, implementation with all infrastructure invariants, and CMake integration
- [x] 01-02-PLAN.md — Comprehensive Catch2 unit tests for debug_schema, debug_nulls, and copy_null_mask_to_host

### Phase 2: Numeric Row Preview and Column Statistics
**Goal**: Developers can call `debug_head(batch, N, stream)` and see the first N rows in aligned-column and CSV format for all numeric types, and call `debug_stats(batch, stream)` to see GPU-computed min, max, and sum per numeric column — all output routed through `[SIRIUS_DIAG]`
**Depends on**: Phase 1
**Requirements**: HEAD-01, HEAD-02, HEAD-03, STATS-01, STATS-02, STATS-03
**Success Criteria** (what must be TRUE):
  1. `debug_head(batch, 5, stream)` on a batch with INT32, BIGINT, FLOAT, DOUBLE, and BOOL columns prints five rows in fixed-width aligned-column format with correct values and `NULL` for null positions
  2. `debug_head(batch, 5, stream, format=csv)` prints the same five rows in CSV format
  3. `debug_stats(batch, stream)` prints per-column min, max, and sum for numeric columns; non-numeric columns appear as `(non-numeric, skipped)` in the output
  4. `debug_stats` uses `cudf::reduce` — no full column is copied to host for statistics computation
**Plans:** 2 plans
Plans:
- [x] 02-01-PLAN.md — Implement debug_head and debug_stats functions in debug_utils module
- [x] 02-02-PLAN.md — Comprehensive Catch2 unit tests for debug_head and debug_stats

### Phase 3: Full Type Coverage and Checksums
**Goal**: `debug_head` handles all Sirius-supported data types including STRING, DECIMAL (with correct scale), TIMESTAMP, and DATE (as human-readable calendar format), and `debug_checksum` produces a stable per-column xxhash_64 fingerprint that can be compared across two log files to detect data divergence
**Depends on**: Phase 2
**Requirements**: HEAD-04, HEAD-05, HEAD-06, CHKSUM-01, CHKSUM-02, CHKSUM-03
**Success Criteria** (what must be TRUE):
  1. `debug_head` on a batch containing VARCHAR columns shows correct string values (not raw pointers or garbage), extracted via `cudf::strings_column_view`
  2. `debug_head` on a DECIMAL column shows values with the correct decimal point position derived from `col.type().scale()`, not raw integer values
  3. `debug_head` on TIMESTAMP and DATE columns shows human-readable calendar format (e.g., `2024-01-15 08:30:00`), not raw epoch integers
  4. `debug_checksum(batch, stream)` produces a `col[N] checksum: 0xXXXXXXXX` line per column, and running the same query twice yields identical checksum values for identical data
**Plans:** 2 plans
Plans:
- [ ] 03-01-PLAN.md — Extend debug_head with STRING, DECIMAL, TIMESTAMP, DATE type support and unit tests
- [ ] 03-02-PLAN.md — Implement debug_checksum with xxhash_64 + XOR reduce and unit tests

### Phase 4: Diff, Sampling, and Skill Integration
**Goal**: `debug_diff` compares two batches and reports schema mismatches and per-column row differences; `debug_sample` prints N randomly selected rows using the same formatting as `debug_head`; both Claude Code skills document the complete utility API so Claude uses named functions instead of ad-hoc `SIRIUS_LOG_TRACE` patterns
**Depends on**: Phase 3
**Requirements**: DIFF-01, DIFF-02, DIFF-03, DIFF-04, DIFF-05, SAMPLE-01, SAMPLE-02, SAMPLE-03, SKILL-01, SKILL-02, SKILL-03
**Success Criteria** (what must be TRUE):
  1. `debug_diff(batch_a, batch_b, stream)` on two batches with different schemas logs a schema mismatch error and returns without attempting value comparison
  2. `debug_diff` on two batches with identical schemas and some differing rows logs the per-column diff count and the first N differing row indices
  3. `debug_diff` on a batch exceeding the configurable row limit logs a warning and skips value comparison rather than attempting an OOM copy
  4. `debug_sample(batch, N, stream)` prints N randomly selected rows in the same aligned-column format as `debug_head`, with different rows visible on repeated calls
  5. The `/validate` and `/runtime-errors` skill files instruct Claude to call `debug_checksum`, `debug_stats`, `debug_head`, `debug_schema`, `debug_nulls`, and `debug_diff` by name, with function signatures and usage examples documented
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Infrastructure and Metadata Inspection | 2/2 | Complete | 2026-04-07 |
| 2. Numeric Row Preview and Column Statistics | 2/2 | Complete | 2026-04-07 |
| 3. Full Type Coverage and Checksums | 0/2 | Planning complete | - |
| 4. Diff, Sampling, and Skill Integration | 0/TBD | Not started | - |
