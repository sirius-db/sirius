# Parquet output

Sirius can export a query result to a **new local directory** of Parquet parts:

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
The destination must not exist, including as an empty directory or a symlink.
Its parent must already exist. Relative local paths are resolved at planning time.
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

## Failure and publication

Each query creates a private staging directory beside its destination. Failed
encoding attempts remove their provisional parts. A completed input is recorded
only once. The query retires asynchronous work before preparing completion and
publishing the directory with a Linux atomic, no-replacement rename. A competing
destination causes publication to fail without replacing it. A wholly empty result
produces one schema-only part with zero rows.

Runtime export failures and cancellation clean up private staging on normal process
unwind and **are not replayed on the CPU**. After publication the directory is owned
by the caller and is never removed by cleanup. Process crashes can leave private
`.sirius-parquet-*` staging directories; there is no crash-recovery service. Atomic
visibility is not a durability guarantee: the writer does not promise `fsync` of
all files and directories. Publication is external to DuckDB transaction rollback.

## Implementation boundary

`src/planner/sirius_plan_copy.cpp` is the DuckDB adapter. It validates bound options
and builds `sirius::output::parquet_export_spec`, including a recursive Sirius-owned
schema. Runtime writer/session interfaces contain no DuckDB values, logical types,
client contexts, or registered-state lookups. The physical sink uses the engine's
existing operator/pipeline infrastructure; another frontend can supply the same
export specification. `sirius_interface.cpp` converts only the completion metadata
back into a DuckDB result.

The export session owns filename allocation, retry accounting, and publication.
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
the query and file production/publication, not subsequent readback or cleanup.

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
