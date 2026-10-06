# Doris translation semantic gaps

The translator refuses or reshapes whatever Doris and the DuckDB Substrait consumer / Sirius do
not agree on. Each case has a number (`G-nn`) that the refusal reasons, the module docs, the gap
probes under `sql/gaps` and the tests use:

| # | Doris | What the translator does |
|---|---|---|
| G-01 | `LARGEINT` (128-bit integer) | rejected: Sirius narrows 128-bit integers to 64 bits silently |
| G-02 | `concat` is NULL-strict | rejected: DuckDB's `concat` ignores NULL arguments |
| G-03 | `LIKE` escapes with `\` | only a constant pattern without a backslash is translated |
| G-04 | `substring` follows MySQL for `pos <= 0` and negative positions | only `(expr, constant start > 0, constant length > 0)` is translated |
| G-05 | `DECIMAL256` | rejected: exceeds the 128-bit decimal carrier |
| G-06 | `DECIMAL(p <= 4)` | slots rejected (DuckDB stores them as `INT16`, which has no cuDF carrier); literals are widened to precision 5 |
| G-07 | `HLL` / `BITMAP` / `QUANTILE_STATE` / `AGG_STATE` | rejected |
| G-08 | `JSONB` / `VARIANT` | rejected (Sirius would carry them as strings) |
| G-09 | `BINARY` / `VARBINARY` / `IPV4` / `IPV6` / `TIMEV2` / `TIMESTAMPTZ` | rejected: no mapping |
| G-10 | `ARRAY` / `MAP` / `STRUCT` | rejected as a whole (Sirius only passes nested columns through) |
| G-11 | window functions (`ANALYTIC_EVAL_NODE`) | rejected by node type |
| G-12 | `UNION` / `INTERSECT` / `EXCEPT` (`UNION_NODE`) | rejected by node type (the consumer's `SetRel` takes two inputs only) |
| G-13 | `SELECT DISTINCT`, compiled by Nereids into a two-phase group-by with no aggregate | folded by the stitcher into one aggregate |
| G-14 | multi-phase aggregation (update → exchange → merge) | a lone phase is refused; the stitcher folds a query's phases into one aggregate |
| G-15 | scalar functions outside the allowlist (`upper`, `lower`, `trim`, `round`, `abs`, `replace`, …) | rejected |
| G-16 | aggregates outside the allowlist (`stddev`, `median`, `approx_count_distinct`, …) | rejected; `count(DISTINCT)` is supported |
| G-17 | legacy `DATE` / `DATETIME` (v1) and `DECIMALV2` | rejected: only `DATEV2` / `DATETIMEV2` / `DECIMAL32/64/128I` |
| G-18 | untyped `NULL` (`NULL_TYPE`) | rejected; typed `NULL_LITERAL`s are wrapped in a cast so the consumer does not turn them into `SQLNULL` |
| G-19 | DECIMAL result types and DOUBLE→DECIMAL rounding | DuckDB re-derives expression types, so every projection and measure is cast back to its Doris slot type; the GPU's DOUBLE→DECIMAL cast truncates the last digit where DuckDB rounds (the validator has column-specific allowances for Q1 `avg_*` and Q8 `mkt_share`) |
