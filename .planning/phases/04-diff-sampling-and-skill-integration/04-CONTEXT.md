# Phase 4: Diff, Sampling, and Skill Integration - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement `debug_diff` for two-batch comparison with schema and value mismatch reporting, `debug_sample` for random row selection using the same formatting as `debug_head`, and update `/validate` and `/runtime-errors` Claude Code skills to reference the complete debug utility API (replacing ad-hoc SIRIUS_LOG_TRACE checksum patterns).

</domain>

<decisions>
## Implementation Decisions

### Diff Output Format
- **D-01:** `debug_diff` reports per-column diff count + first N differing row indices. Format: `col[0] diffs: 3/1000 rows [idx: 42, 187, 501]`
- **D-02:** Number of differing row indices shown per column is configurable via `max_diff_rows` parameter with default 10
- **D-03:** Row count limit defaults to 10,000,000 rows. Batches exceeding this log a warning and skip value comparison (DIFF-05). This is a host memory guard — both batches are copied to host for comparison

### Diff Comparison Scope
- **D-04:** Comparison is host-side: copy both batches to host, then compare element-by-element in C++. Simpler code, full control over per-type comparison logic
- **D-05:** Exact equality for all types including FLOAT32/FLOAT64 — no epsilon tolerance. Debug tool should catch every bit flip; developers understand GPU rounding
- **D-06:** Schema mismatch check first (column count, types) before any value comparison (DIFF-02). Row count mismatch also reported before values (DIFF-03)

### Random Sampling
- **D-07:** `debug_sample` generates random row indices on host via `std::mt19937`, then uses `cudf::gather` to extract those rows from GPU. No cuRAND dependency
- **D-08:** Optional `seed` parameter. Default uses `std::random_device` for different rows each call. Caller can pass explicit seed for reproducible sampling and unit tests
- **D-09:** Output uses the same formatting as `debug_head` — aligned columns or CSV, same `DebugFormat` enum and `max_string_len` parameter

### Skill Integration
- **D-10:** Both `/validate` and `/runtime-errors` SKILL.md get a "Debug Utilities" section with full function signatures, parameter descriptions, and 2-3 usage examples per function
- **D-11:** `/validate` Phase 2 replaces existing ad-hoc `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] operator_name checksum: sum={}, max={}, first_row={}", ...)` patterns with `debug_checksum`, `debug_stats`, `debug_head` calls
- **D-12:** `/runtime-errors` references `debug_schema`, `debug_head`, `debug_nulls` for data inspection at suspected fault points

### Claude's Discretion
- Schema mismatch error message wording
- Internal helper function decomposition for host-side comparison
- Whether `debug_sample` clamps N to batch size or returns fewer rows silently (follow debug_head pattern: clamp silently per D-12 of Phase 2)
- Whether debug_diff header line includes batch_id comparison
- Skill documentation section placement within existing SKILL.md structure

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` — DIFF-01 through DIFF-05, SAMPLE-01 through SAMPLE-03, SKILL-01 through SKILL-03 define the exact requirements

### Phase 3 Implementation (extends)
- `src/include/debug_utils.hpp` — Current API: debug_schema, debug_nulls, debug_head, debug_stats, debug_checksum
- `src/debug_utils.cpp` — Full implementation with all type dispatch, helper functions, established patterns

### Existing Comparison Patterns
- `src/cuda/operator/cuda_helper.cuh` — Contains `curand.h` include (cuRAND usage in codebase — NOT used for debug_sample, but shows library is available)

### Skill Files to Update
- `.claude/skills/validate/SKILL.md` — Current /validate skill with ad-hoc SIRIUS_LOG_TRACE patterns to replace
- `.claude/skills/runtime-errors/SKILL.md` — Current /runtime-errors skill to update with debug utility references
- `.claude/skills/_shared/build-and-query.md` — Shared infrastructure referenced by both skills

### Prior Phase Decisions
- `.planning/phases/02-numeric-row-preview-and-column-statistics/02-CONTEXT.md` — D-03 (dynamic widths), D-06 (NULL display), D-11 (no cap on N), D-13 (empty batch), D-14 (bulk copy)
- `.planning/phases/03-full-type-coverage-and-checksums/03-CONTEXT.md` — D-02 (max_string_len), D-09 (UTC timestamps)

### Test Patterns
- `test/cpp/debug/test_debug_utils.cpp` — 31 existing Catch2 tests showing batch creation, null mask setup, multi-type batches

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `debug_head` formatting logic: `debug_sample` reuses the same row formatting pipeline — extract values to `cells[][]`, compute widths, format output
- `host_column_nulls` + `copy_null_mask_to_host`: Null handling for host-side diff comparison
- `is_gpu_tier()`, `get_cudf_table_view()`: Tier guard and table extraction — reuse unchanged
- All type dispatch helpers from Phase 3: `format_decimal_value`, `format_timestamp_us`, `civil_from_days`, etc.
- `cudf::gather`: Used elsewhere in codebase for row selection by index

### Established Patterns
- Output buffered into single `std::string`, emitted via one `SIRIUS_LOG_DEBUG("{}", output)` call
- All debug functions: try/catch wrapping, tier guard check, stream sync
- `[SIRIUS_DIAG]` prefix on all diagnostic output
- Configurable parameters with sensible defaults as last function parameters

### Integration Points
- `debug_diff` and `debug_sample` added to `debug_utils.hpp` (declarations) and `debug_utils.cpp` (implementations)
- Tests added to `test/cpp/debug/test_debug_utils.cpp`
- Skill files at `.claude/skills/validate/SKILL.md` and `.claude/skills/runtime-errors/SKILL.md`

</code_context>

<specifics>
## Specific Ideas

- debug_diff output format designed for `diff log_a.txt log_b.txt` workflow — grep-friendly per-column lines
- Host-side comparison chosen over GPU-side cudf::binaryop for simplicity and full per-type control
- Seed parameter on debug_sample enables reproducible unit tests while defaulting to random behavior

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-diff-sampling-and-skill-integration*
*Context gathered: 2026-04-08*
