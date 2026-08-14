# Vendored test databases

Pre-generated DuckDB files that the C++ integration tests attach **read-only**, so
the tests need no network access or DuckDB extensions at runtime and run in CI.

These files are intentionally large and are therefore exempt from `.gitignore`'s
`*.duckdb` rule and from the `check-added-large-files` pre-commit hook (see
`.pre-commit-config.yaml`).

## `tpcds.duckdb`

TPC-DS at **scale factor 0.01** (~12 MB). Used by
`test_gpu_execution_tpcds_nulls.cpp` — the GPU-vs-CPU NULL differential suite.
Attached via `get_tpcds_db_path()`; override the path with the
`SIRIUS_TPCDS_TEST_DB_PATH` environment variable.

### Regenerating

Only regenerate when necessary — e.g. the DuckDB storage format changes and the
test fails to `ATTACH` the file, or the schema/scale needs to change. Generate with
a DuckDB whose storage format matches the build's (run inside the pixi env). This
needs network once to install the `tpcds` extension:

```bash
TPCDS_SF=0.01 pixi run bash test/cpp/integration/data/duckdb/generate_tpcds_duckdb.sh \
  test/cpp/integration/data/duckdb/tpcds.duckdb
```

Then commit the regenerated file. A precondition case in the test asserts the data
still contains NULLs in every column under test, so a bad regeneration fails loudly.

## `integration.duckdb`

TPC-H data (~27 MB) — the eight standard tables (`lineitem`, `orders`, `customer`,
`nation`, `region`, `part`, `partsupp`, `supplier`). Used by
`test_gpu_execution_tpch.cpp`, attached read-only as `tpch` via `get_tpch_db_path()`;
override with `SIRIUS_INTEGRATION_TEST_DB_PATH`. (The separate, larger SF10 smoke
tests read external parquet via `SIRIUS_TEST_SF10_PATH` and do **not** use this
file.)
