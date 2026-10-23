# Synthetic SQL UDF project

This is a working infrastructure example, not a conversion of customer logic.
Replace the macro, reference, setup, and comparison data with the external
conversion before reporting customer validation.

Use Sirius's matching DuckDB Python package and built extension. Run commands
through `pixi run` from the compatible Sirius/provider workspace containing
`pixi.toml`; the external project does not contain its own Pixi environment:

```bash
pixi run python /absolute/project/run.py --extension /absolute/sirius.duckdb_extension
pixi run python /absolute/project/run.py --extension /absolute/sirius.duckdb_extension \
  --rows 1000000 --warmups 3 --iterations 5 --report /absolute/private/results.json
```

The default Pixi DuckDB Python client is for data generation; it cannot load an
extension built from the pinned v2 source. If a matching Python client is absent,
use the provider's generic `udf_host` built against the installed SDK (the same
host included in cuDF/CUDA projects):

```bash
pixi run python /absolute/project/run.py --host /absolute/provider/udf_host \
  --rows 1000000 --warmups 3 --iterations 5 --report /absolute/private/results.json
```

No native UDF plugin is needed for this SQL path. The host loads Sirius once,
executes the macro registration, and applies the same correctness and GPU checks.

The runner uses a temporary file-backed DuckDB database, loads `registration.sql`,
runs the reference on CPU, verifies the candidate on CPU and GPU, compares logical
types and complete results, and checks execution counters with fallback disabled.
Use an isolated database for validation: execution counters are shared by
connections to the same Sirius runtime.
The runner checkpoints setup tables and routes Sirius logs to its temporary
validation directory, overriding inherited log paths without changing the caller's
working directory.
Benchmarks start only after that comparison passes. Timings are query wall time
including fetching results, not device-only time. Keep generated data and reports
outside Git and CI.

For deployment, execute `registration.sql` through the ordinary SQL connection.
The selected engine must support all expressions used after macro expansion.
