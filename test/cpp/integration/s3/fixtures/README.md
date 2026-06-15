# S3 Perf Benchmark Fixture

The hidden `[!benchmark][perf][bench]` Catch2 benchmark uses an SF=10 `lineitem`
Parquet fixture. It is now generated and uploaded **by the test binary itself**
(`test/cpp/utils/s3_backend.*`) when `SIRIUS_TEST_S3_LARGE=1`, which:

1. replicates the SF1 `lineitem` parquet 10x with the in-tree DuckDB CLI
   (`build/release/duckdb`, or `SIRIUS_TEST_DUCKDB`) to reach ~SF10 scale, reading
   from `test_datasets/tpch_parquet_sf1/lineitem.parquet` (or
   `SIRIUS_TEST_S3_LARGE_SOURCE`) and caching the result across invocations, and
2. uploads it from the host via Sirius's SigV4 signer + libcurl (no `mc`).

The in-tree DuckDB build has no `tpch` extension (and cannot download one for its
dev version), so `CALL dbgen` is unavailable. The large gate only needs the
*scale* of the object (>50M rows, ~GB across many row groups, enough to push
Sirius into GPU memory tiering) — every correctness check compares the
GPU-over-S3 result against a CPU read of this same file — so a 10x replica of SF1
(~6M rows -> ~60M) suffices and is built with DuckDB's core `range()`. Generate
the SF1 source with `scripts/tpch_to_parquet.sql`.

The benchmark reads `SIRIUS_BENCH_S3_*` and falls back to the harness-published
`SIRIUS_TEST_S3_*`, with `SIRIUS_BENCH_S3_KEY` defaulting to
`tpch/lineitem_sf10.parquet`.

For the local SeaweedFS backend, just run:

```sh
make s3-bench
```

`make s3-bench` sets `SIRIUS_TEST_S3_AUTO=1 SIRIUS_TEST_S3_LARGE=1`, so the
SeaweedFS backend is auto-managed and the SF10 fixture is prepared in-process —
no separate fixture/up step.

AWS portability uses the same benchmark test with the auto-managed path off:

```sh
SIRIUS_BENCH_BACKEND=aws-s3
SIRIUS_BENCH_AWS_S3_REGION=us-east-1
SIRIUS_BENCH_AWS_S3_BUCKET=<bucket>
SIRIUS_BENCH_AWS_S3_KEY=tpch/lineitem_sf10.parquet
SIRIUS_BENCH_AWS_S3_ACCESS_KEY=<access-key>
SIRIUS_BENCH_AWS_S3_SECRET_KEY=<secret-key>
```
