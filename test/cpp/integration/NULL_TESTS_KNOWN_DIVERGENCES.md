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

## Confirmed correct on GPU (no divergence)
`IS [NOT] NULL`, three-valued comparison filtering (`= <> < <= > >=`), `IS [NOT] DISTINCT FROM`,
three-valued `AND` / `NOT`, `BETWEEN`, `IN`, `NOT IN` (with NULL probe), `COALESCE` / `NULLIF` /
`CASE`, and NULL propagation through arithmetic, `CAST`, `length`, `substring`, and date functions.

<!-- Append commit 3 (aggregate nulls) and commit 4 (join nulls) findings below as they surface. -->
