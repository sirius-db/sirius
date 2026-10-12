Run the translator tests from `experimental/starrocks`:

```sh
pixi run -e cn cargo test -p starrocks-plan-translator
```

The optional execution test runs the translated plan in DuckDB. It checks distinct
BIGINT sum/count values through sort materialization, a projection, and repeated
and computed final outputs. It tests Substrait execution on CPU; it does not require
a Sirius build or audit GPU operators.

Install DuckDB in a Python environment and install its Substrait extension once:

```sh
python3 -c 'import duckdb; duckdb.connect().execute("INSTALL substrait FROM community")'
PYTHON=/path/to/python3 pixi run -e cn cargo test -p starrocks-plan-translator \
  execute_same_type_aggregate_sort_and_project_outputs -- --ignored
```
