# Null-data tests — known GPU/CPU divergences (issue #1095)

> **Temporary tracking doc — delete before merge.** These are real Sirius GPU-vs-DuckDB
> correctness bugs surfaced while adding null-data test coverage on this branch. Each is
> quarantined behind a Catch2 `[!shouldfail]` tag (so CI stays green) and documented inline in
> the test with a `TODO(#1095 follow-up)` note. File one issue per item, then remove the tag,
> the inline TODO, and this file.

## Open divergences

### 1. Three-valued `OR` drops rows
- **Symptom:** GPU evaluates `TRUE OR NULL` as non-TRUE (naive NULL propagation) instead of
  `TRUE`, so a row where one `OR` branch is TRUE and the other is NULL is wrongly filtered out.
- **Repro:** `SELECT id FROM nt WHERE i = 10 OR b = 200` → 3 rows on GPU vs 4 on DuckDB
  (drops the `i=10, b=NULL` row).
- **Suspected area:** GPU predicate / conjunction evaluator.
- **Note:** `AND` is unaffected — both `FALSE AND NULL` and `NULL` exclude the row, so
  membership never changes.
- **Test:** `test_gpu_execution_filter_nulls.cpp` → "three-valued OR with NULL operand [known divergence]".

### 2. `concat()` propagates NULL
- **Symptom:** GPU `concat(NULL, '_x')` returns NULL; DuckDB returns `'_x'` (DuckDB's `concat()`
  ignores NULL arguments — only the `||` operator propagates). GPU `concat` is implementing
  `||` semantics.
- **Repro:** `SELECT id, concat(s, '_x') FROM nt` mismatches on rows with `s IS NULL`.
- **Suspected area:** GPU `concat` scalar function.
- **Test:** `test_gpu_execution_filter_nulls.cpp` → "concat NULL-handling [known divergence]".

### 3. Aggregates over a wholly-NULL column read sentinel values
- **Symptom:** A column that is entirely NULL loses its validity mask in the GPU native scan and
  is read as sentinel `INT_MAX`, so aggregates see fake data. `SUM(allnull)` returns `8*INT_MAX`
  (`17179869176`) and `COUNT(allnull)` returns the row count (`8`) instead of NULL / 0.
- **Repro:** `SELECT SUM(allnull), COUNT(allnull) FROM agg_n` (column `allnull` is all NULL).
- **Scope:** Only *wholly-NULL columns*. All-NULL *groups* of a normally-nullable column are
  correct (the `GROUP BY g` case with all-NULL group `g=3` passes).
- **Suspected area:** GPU DuckDB-native scan — validity mask dropped for all-null column segments.
- **Test:** `test_gpu_execution_aggregate_nulls.cpp` → "aggregates over a wholly-NULL column [known divergence]".

### 4. Ungrouped `AVG` divides by row count, not non-null count
- **Symptom:** Ungrouped `AVG` over a column with NULLs uses the total row count as the
  denominator. `AVG(v)` returns `335/8 = 41.875` instead of `335/5 = 67`. `SUM` and `COUNT` are
  individually correct; **grouped** `AVG` is correct — bug is isolated to the ungrouped aggregate.
- **Repro:** `SELECT AVG(v) FROM agg_n`.
- **Suspected area:** ungrouped aggregate operator (`sirius_physical_ungrouped_aggregate`).
- **Test:** `test_gpu_execution_aggregate_nulls.cpp` → "ungrouped AVG denominator counts NULL rows [known divergence]".

### 5. `COUNT(DISTINCT)` runtime-falls-back to CPU
- **Symptom:** `COUNT(DISTINCT ...)` errors on the GPU and falls back to DuckDB CPU at runtime
  (`runtime_fallbacks` increments) — it does not execute on-device. Result is correct via fallback.
- **Repro:** `SELECT COUNT(DISTINCT v) FROM agg_n`.
- **Suspected area:** GPU count-distinct aggregate path (possibly NULL-input specific).
- **Test:** `test_gpu_execution_aggregate_nulls.cpp` → "COUNT(DISTINCT) runtime-falls-back to CPU [known divergence]".

## Confirmed correct on GPU (no divergence)
- **Filters/expressions:** `IS [NOT] NULL`, three-valued comparison filtering (`= <> < <= > >=`),
  `IS [NOT] DISTINCT FROM`, three-valued `AND` / `NOT`, `BETWEEN`, `IN`, `NOT IN` (with NULL probe),
  `COALESCE` / `NULLIF` / `CASE`, and NULL propagation through arithmetic, `CAST`, `length`,
  `substring`, and date functions.
- **Aggregates:** `COUNT(*)` and `COUNT(col)` (partially-null column), ungrouped `SUM`/`MIN`/`MAX`
  skipping NULLs, `GROUP BY` on a NULL key (groups NULLs together), all-NULL *groups* of a
  nullable column (→ NULL), and grouped `SUM`/`AVG`.

<!-- Append commit 4 (join nulls) findings below as they surface. -->
