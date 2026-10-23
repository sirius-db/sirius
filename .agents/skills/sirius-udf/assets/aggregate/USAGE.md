# Synthetic aggregate-to-SQL project

This project contains two independently synthetic CPU originals in `original.py`,
an executable `oracle.py` adapter, SQL macros, and correctness cases. It is an
infrastructure example. Replace the originals, adapter, SQL, and cases in this
external directory before claiming validation of customer logic; update
`project.json` to identify the actual source reference and set `synthetic` false.

## Validate

Run from the compatible Sirius/provider workspace containing `pixi.toml`. Use a
provider's generic `udf_host` built from this skill's native host against the
installed SDK and matching DuckDB shared library. No UDF plugin is needed:

```bash
pixi run --locked -e cuda12 python /absolute/project/run.py \
  --host /absolute/provider/udf_host --rows 1000000 --validate-only \
  --report /absolute/private/aggregate-validation.json
```

If that host is unavailable, build only its executable from a disposable native
scaffold. These commands do not build or load the scaffold's synthetic plugin:

```bash
pixi run --locked -e cuda12 python .agents/skills/sirius-udf/scripts/scaffold.py --target cudf \
  --output /absolute/private/host-source
pixi run --locked -e cuda12 cmake -S /absolute/private/host-source \
  -B /absolute/private/host-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/absolute/provider/sirius-sdk \
  -DSIRIUS_UDF_DUCKDB_LIBRARY=/absolute/provider/libduckdb.so
pixi run --locked -e cuda12 cmake --build /absolute/private/host-build \
  --target udf_host -j2
```

Alternatively, pass `--extension /absolute/sirius.duckdb_extension` when a matching
DuckDB Python client is available. The default Pixi Python package may differ from
Sirius's pinned DuckDB revision; a mismatched package cannot load the extension.

The runner loads each case into an isolated checkpointed database, exports all
ordered source inputs, executes the original adapter, compares complete CPU SQL
and GPU SQL results, and checks Sirius execution counters with fallback disabled.
The original adapter performs update, partial-state merge, and finalization; it
does not evaluate the rewritten SQL. Empty cases check semantics separately and
may be optimized away. Required nonempty cases must show positive GPU execution
and zero fallbacks. Validation must pass before any timing starts.

Exact types, NULLs, values, and floating-point bit patterns are the default.
`weighted_mean_fractional_global` explicitly permits `1e-12` absolute and relative
error for its DOUBLE result; its floating-point sum can change with reduction
order. The generated large cases instead use bounded integer-valued doubles so
partial sums remain exact at the supplied row counts. Do not relax a customer's
source contract to make a rewrite pass.

## Performance

Removing `--validate-only` runs query wall-time measurements after the complete
case suite passes. `--warmups 3 --iterations 7` controls the sample counts. CPU SQL
is the performance baseline; the original UDAF is the correctness reference.
These timings include result materialization and are not kernel-only timing.

For operator comparisons, use `test/udf_performance/run_stages.py` from the Sirius
checkout as described in that harness's README. Report input preparation,
projection, aggregation, and output materialization as accumulated operator work.
Retain generated SQL, raw samples, configuration, versions, and GPU counters;
report regressions and suppress ratios when pressure or timing precision prevents
a reliable comparison.

## Deploy

Execute `registration.sql` through the ordinary provider SQL connection, then use
`sum_squares(value)` or `weighted_mean(value, weight)` in global/grouped queries.
Macros expand into existing SQL operators; they do not register native UDAFs and
need no native package or Sirius rebuild. A complete rewritten query is preferable
when a macro cannot preserve the source invocation. Unsupported operators or
unproven behavior mean the original implementation stays in use.

Keep fixtures, original source, outputs, logs, and reports outside the repository.
The suite's JSON/adapter protocol is documented in the skill's aggregate reference.
