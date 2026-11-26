---
name: validate
description: Diagnose Sirius query results that differ from the DuckDB CPU baseline.
argument-hint: [sql-query-or-file]
disable-model-invocation: true
---

# Validation Error Analyzer

Compare the supplied query against DuckDB CPU and confirm that the Sirius run actually used the
GPU: transparent execution can fall back to CPU. Compare results as multisets unless the query
specifies ordering; preserve duplicates and account for types and floating-point tolerances.
Use the matching built wrapper as described in `sirius-duckdb/README.md`.

This example uses the default release wrapper. Substitute the reproduction's
database and SQL, and retain the printed directory:

```bash
result_dir=$(mktemp -d)
printf 'Results and logs: %s\n' "$result_dir"
pixi run env SIRIUS_DISABLE=1 sirius-duckdb/build/release/duckdb <db_path> -c "SELECT ...;" > "$result_dir/baseline_result.txt"
export SIRIUS_LOG_BACKEND=spdlog
export SIRIUS_LOG_LEVEL=trace
export SIRIUS_LOG_DIR="$result_dir/log"
mkdir -p "$SIRIUS_LOG_DIR"
pixi run sirius-duckdb/build/release/duckdb -unsigned -cmd "LOAD 'sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension';" <db_path> -c "SELECT ...;" > "$result_dir/gpu_result.txt"
```

For intermittent failures, repeat the entire block (up to 10 runs), retaining each new directory
and labelling good and bad runs. `pixi run python3 tools/parse_pipeline_log.py <result_dir>/log/sirius_*.log` can summarize
row counts, but first check that its patterns recognize the current logs. Empty output does not
establish that counts match; inspect logs or use temporary instrumentation instead. Compare
equivalent operator boundaries: divergence narrows the investigation but does not prove cause.

For data inspection, `src/debug_utils.hpp` provides checksum, statistics, row display, and batch
comparison helpers logging at debug level with the `[SIRIUS_DIAG]` prefix. These require GPU-tier
batches; compare equivalent batches at the same boundary, accounting for row ordering and batch
boundaries. CPU results are not directly comparable as batches, and checksums from different
algorithms cannot be compared. A matching DuckDB CPU subquery can supply row counts and statistics
for deterministic failures. Remove temporary instrumentation after the investigation.

Intermittent or garbage data can result from stream synchronization: synchronizing the caller's
stream does not synchronize a different stream used by an operation. `utils/stream_check/`
intercepts `cudf::get_default_stream()` on monitored threads and writes call stacks to
`default_stream_traces.log`. Sirius builds in the hook that enables monitoring on
pipeline executor threads. `LD_PRELOAD` installs the interposer for a shared cuDF
build, and `SIRIUS_STREAM_CHECK_LIB` gives the hook the path to the helper library.
Preload an existing compatible `libstream_check.so` without rebuilding Sirius.

Set `build_dir` and `wrapper_build_dir` to the actual build directories. If
you need to build the helper, record the previous `ENABLE_STREAM_CHECK` cache
value and restore it afterward. Use a helper path without spaces or colons in
`LD_PRELOAD`. A relative path from the launch directory works even when the
checkout path contains spaces. When needed, point `stream_check_lib` to a
symlink at a separator-free path in a private temporary directory.

```bash
build_dir="${build_dir:-build/release}"
wrapper_build_dir="${wrapper_build_dir:-sirius-duckdb/build/release}"
database=/path/to/reproduction.duckdb

# Run these two commands if the helper needs to be built.
pixi run cmake -S . -B "$build_dir" -DENABLE_STREAM_CHECK=ON
pixi run cmake --build "$build_dir" --target stream_check

stream_check_lib="$build_dir/utils/stream_check/libstream_check.so"
pixi run env LD_PRELOAD="$stream_check_lib" SIRIUS_STREAM_CHECK_LIB="$stream_check_lib" \
  "$wrapper_build_dir/duckdb" -unsigned \
  -cmd "LOAD '${wrapper_build_dir//\'/\'\'}/extension/sirius/sirius.duckdb_extension';" "$database" -c "SELECT ...;"
```

`default_stream_traces.log` is append-only. When running serially, record its
size before starting the query and inspect the appended bytes. Treat a default
stream trace as a diagnostic lead. Temporarily inserting
`cudaDeviceSynchronize()` can test a synchronization or timing hypothesis. A changed result alone
cannot establish the cause. Remove the diagnostic synchronization before finishing.
