# Synthetic native GPU UDF project

The generated package contains an independently authored `DOUBLE -> DOUBLE`
example, CPU fallback, GPU implementation, and registration hook. Replace these
with the external conversion and its reference suite. Sirius must already be
built with the UDF SDK; build each additional function separately:

Run these commands from the Sirius/provider workspace containing the compatible
`pixi.toml`. The generated external project does not contain its own Pixi environment.
CMake 4.0 or later is required by the pinned cuDF/RMM packages; Pixi supplies it.

```bash
pixi run cmake -S /absolute/project -B /absolute/project/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/absolute/sirius-sdk-prefix \
  -DSIRIUS_UDF_DUCKDB_LIBRARY=/absolute/matching/libduckdb.so
pixi run cmake --build /absolute/project/build --parallel
pixi run python /absolute/project/run.py --host /absolute/project/build/udf_host \
  --plugin /absolute/project/build/sirius_udf_example.so
pixi run python /absolute/project/run.py --host /absolute/project/build/udf_host \
  --plugin /absolute/project/build/sirius_udf_example.so \
  --rows 1000000 --warmups 3 --iterations 5 --report /absolute/private/results.json
```

Use the exact DuckDB revision, compiler ABI, libcudf/RMM, and CUDA toolchain used
by the installed Sirius SDK. The host needs matching DuckDB headers from the SDK
and a matching shared DuckDB library; an unrelated Python wheel is insufficient.
Before loading the plugin, the host compares DuckDB's source commit to the SDK's
`SIRIUS_UDF_DUCKDB_REVISION` identity, accepting matching abbreviated commit IDs.
An absent SDK identity is an error; the provider must set the pinned revision when
building the SDK, including source-tarball builds.
For CUDA set `CMAKE_CUDA_ARCHITECTURES` for the deployment GPU when cross-building.
Keep provider runtime configuration available through `SIRIUS_CONFIG_FILE`.

The generic embedding host loads this plugin and calls
`register_sirius_udf(duckdb::ExtensionLoader&)`; the UDF is then callable through
ordinary SQL. Providers can invoke the same hook from their existing extension
loader or embedding host. The `.so` is an embedding plugin, not a DuckDB file
with extension metadata: do not use `LOAD 'sirius_udf_example.so'`. A provider
that exposes DuckDB `LOAD` packages this hook with its matching DuckDB extension
tooling. This scaffold deliberately requires no engine rebuild for a new UDF.
The same `udf_host` executable can validate generated SQL macro projects with
`run.py --host /absolute/udf_host`; SQL projects require no native plugin.

The runner checks CPU reference, CPU candidate, and GPU candidate before timing.
It requires positive GPU execution counts and zero fallbacks. Its timings are
wall time including result consumption; use a device profiler for kernel time.
Use an isolated database for validation: execution counters are shared by
connections to the same Sirius runtime.
The runner checkpoints setup tables and routes Sirius logs to its temporary
validation directory, overriding inherited log paths without changing the caller's
working directory.
Floating comparisons check IEEE bits, including NaN and signed zero. If the
original contract allows a tolerance or canonical NaNs, explicitly encode that
contract in the external comparator rather than silently ignoring differences.
The timed CPU baseline is `reference.sql`; adapt it to call the original CPU UDF
before describing the comparison as a speedup over that UDF. This synthetic
example's independent SQL reference demonstrates mathematical equivalence only.
The host retains the plugin until process exit so bound callbacks stay valid.
Production hosts must likewise retain code while catalog entries or plans refer
to it. Native callbacks are trusted provider code; use the provider's existing
build, review, isolation, and deployment process.
