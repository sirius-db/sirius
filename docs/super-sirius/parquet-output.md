# Parquet output

Sirius can export a query result to a **local directory containing no files** of Parquet parts:

```sql
COPY (
    SELECT * FROM table_a ORDER BY col_a
) TO '/existing-parent/new-results' (
    FORMAT PARQUET,
    PER_THREAD_OUTPUT true,
    COMPRESSION ZSTD
);
```

The completion result is the number of rows written, as with DuckDB's `COPY`.
The destination may be absent or an existing directory tree containing no files.
Preexisting empty subdirectories are allowed and preserved. Files, symlinks, and
nonempty directory trees are rejected. If the destination is absent, its parent
must already exist. Relative local paths are resolved at planning time.
Remote destinations are not supported by the GPU writer.

## Options and fallback

The initial option allowlist is `FORMAT PARQUET`, `PER_THREAD_OUTPUT true`, and
`COMPRESSION`. The multi-file flag is required; omitting it requests DuckDB's
single-file behavior, which Sirius does not implement. Sirius writes **one part per
arriving nonempty batch**, rather than one part per thread. It neither combines nor
splits batches to target a file size. Encoded bytes will differ from GPU batch bytes.

Explicit unsupported options, codecs, formats, and result types decline GPU planning
with a logged reason. The original DuckDB plan runs when fallback is enabled; with
`SET enable_duckdb_fallback = false`, the statement fails instead. DuckDB binding
errors remain binding errors. MAP and SQL types without a lossless export mapping
are currently CPU cases. LIST, STRUCT, ARRAY, and supported scalar types carry
recursive names and decimal precision metadata into the writer.

Available codecs depend on the installed cuDF Parquet writer and its runtime
configuration. Sirius recognizes uncompressed, Snappy, Gzip, Brotli, Zstd, and LZ4
(including `none` and `lz4_raw` aliases), and checks cuDF's Parquet-writing
capabilities before accepting a request. This is separate from spill/pin compression.

The default is configured through Sirius YAML:

```yaml
sirius:
  output:
    parquet:
      compression: snappy
```

An explicit SQL `COMPRESSION` overrides the config; the built-in default is Snappy.
The default is captured into the query's owned specification. CPU fallback retains
DuckDB's own defaults when SQL did not explicitly specify a codec.

## Parallelism and order

The terminal writer is pipelineable: it consumes arriving batches on the task's
GPU without gathering the entire output or converting it to DuckDB CPU result rows.
Independent tasks and GPUs can write different parts concurrently. Existing safe
grouped-merge fusion can feed it directly; additional sort-merge fusion is not
required or enabled by this feature.

The first input with rows selects positioned or unordered mode. Empty inputs do
not select a mode. Positioned filenames contain a zero-padded 20-digit position
and a batch-within-input ordinal. Unordered filenames use an operator-local atomic
index. Sparse positions are allowed; duplicate positions from separate inputs and
mixed positioned/unpositioned nonempty inputs fail the export. Retried inputs use
the same file identities and cannot duplicate successful rows.

Filename sorting reconstructs the **encoded part sequence** for positioned results.
A consumer must honor that sequence and the order inside each part. An ordinary
parallel Parquet directory scan does not itself guarantee SQL row order.

## Failure and cleanup

Like DuckDB's multi-file COPY, Sirius writes parts directly into the destination.
Files can be visible, incomplete, or missing their Parquet footer while COPY is
running. There is no atomic publication of the entire directory; consumers should
wait for COPY to succeed before reading it. A wholly empty result produces one
schema-only part with zero rows.

Every part is first reserved with exclusive creation. An existing filename causes
an error without overwriting it. The export tracks only files it successfully
reserved. Failed encoding attempts remove their owned parts before retrying; a
completed input is recorded only once. Concurrent exports into the same directory
are not a supported coordination mechanism: they can fail directory validation or
filename reservation, but failed reservations do not grant ownership of another
writer's file.

On runtime failure or cancellation, Sirius retires asynchronous work, then attempts
to remove all files owned by the failed export, including completed parts. It removes
the destination directory only if this export created it and it is still empty.
Preexisting directories and unrelated files are preserved; cleanup never recursively
removes the destination. Cleanup is best effort and is retried during destruction.
Runtime failures **are not replayed on the CPU**, even when fallback is enabled.

`OVERWRITE`, `OVERWRITE_OR_IGNORE`, and `APPEND` remain unsupported GPU options and
follow the existing planning-time CPU fallback policy. The GPU writer never deletes
or replaces preexisting output. These modes are deliberately not emulated by the
cleanup mechanism.

Once COPY succeeds, cleanup preserves the output. SQL transaction rollback does not
undo a successful export. Process crashes or cleanup failures can leave partial
output behind; there is no crash-recovery service or `fsync` durability guarantee.

## Implementation boundary

`src/planner/sirius_plan_copy.cpp` is the DuckDB adapter. It validates bound options
and builds `sirius::output::parquet_export_spec`, including a recursive Sirius-owned
schema. Runtime writer/session interfaces contain no DuckDB values, logical types,
client contexts, or registered-state lookups. The physical sink uses the engine's
existing operator/pipeline infrastructure; another frontend can supply the same
export specification. `sirius_interface.cpp` converts only the completion metadata
back into a DuckDB result.

The export session owns filename allocation, retry accounting, and owned-file cleanup.
The cuDF adapter owns Parquet metadata and encoding. The sink uses prepared batch
read handles and the task stream, with a conservative encoding-workspace estimate.

## Benchmarks

`test/tpch_performance/performance_test.py`, `run_tpch_parquet.sh`,
`run_tpch_duckdb.sh`, and `benchmark_and_validate.sh` accept:

```text
--result-format parquet --parquet-compression snappy
```

The Python runner's existing `--output` selects the benchmark root. Shell runners
use `OUTPUT_DIR`, or an engine-specific `tpch-parquet-output` directory. Every
query/iteration writes into a fresh `q<N>/parquet_iter<I>` directory. SQL setup and
cleanup around a query (notably Q15) are not put inside `COPY`. Export timing covers
query execution, file production, and completion, not subsequent readback or cleanup.

GPU export benchmarks disable CPU fallback, so unsupported GPU cases fail rather
than recording CPU work under a GPU label. Both engines receive the same explicit
codec. Input cache/pinning and session reuse are retained. Footer-only inspection between iterations avoids warming the entire output in the OS
cache; full value comparison runs after timing. GPU runs also verify cuDF writer
attribution in each Parquet footer. Sidecar JSON records rows,
parts, bytes, and requested codec; generated COPY SQL is retained. Python
`--validation` and the shell comparison validate Parquet schemas and exact row
multisets, including multiplicity, for all iterations; equal COPY counts alone do
not constitute validation. This exact comparison does not use the text validator's
floating-point tolerance. Reusing `--duckdb-results` is currently text-only.

CPU/GPU file counts and sizes may differ because the engines batch differently.
Export measurements must not be mixed with query-only timings or interpreted as
isolated writer cost. The default benchmark mode still returns normal client rows.
