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

For example, substitute the reproduction's database and SQL, and retain the printed directory:

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
`default_stream_traces.log`; Sirius pipeline executor threads enable monitoring through
`src/util/stream_check_wrapper.cpp`. Record the prior `ENABLE_STREAM_CHECK` cache value
and restore it after the investigation. In an already configured build tree:

```bash
pixi run cmake -S . -B build/release -DENABLE_STREAM_CHECK=ON
pixi run cmake --build build/release --target stream_check
pixi run env LD_PRELOAD=<absolute-path-to-libstream_check.so> sirius-duckdb/build/release/duckdb -unsigned -cmd "LOAD 'sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension';" <db_path> -c "SELECT ...;"
```

The library is normally `build/release/utils/stream_check/libstream_check.so`. A default stream
trace is a lead, not proof of a defect. Temporarily inserting `cudaDeviceSynchronize()` can test a
synchronization or timing hypothesis, but a changed result does not prove the cause; remove the
diagnostic synchronization before finishing.
