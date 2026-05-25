# S3 Perf Benchmark Fixture

`test/cpp/integration/s3/fixtures.sh --perf` prepares the hidden
`[!benchmark][perf][bench]` Catch2 benchmark fixture:

1. Uses DuckDB's `tpch` extension to generate SF=10 data.
2. Exports the full `lineitem` table used by the benchmark to Parquet.
3. Uploads the Parquet file to a MinIO/S3-compatible bucket with the same
   dockerized `minio/mc` wrapper used by the regular S3 fixtures.

The benchmark defaults are:

```sh
SIRIUS_BENCH_S3_ENDPOINT=http://127.0.0.1:9000
SIRIUS_BENCH_S3_BUCKET=${SIRIUS_TEST_S3_BUCKET:-sirius-test}
SIRIUS_BENCH_S3_KEY=tpch/lineitem_sf10.parquet
```

For local MinIO, run:

```sh
make s3-up
make s3-bench-fixtures
make s3-bench
```

`make s3-up` only uploads the standard small fixtures. The SF10 parquet
generation stays behind `make s3-bench-fixtures` / `fixtures.sh --perf`, so the
regular `[s3]` test gate does not pay the benchmark setup cost.

The `--perf` path uses `${DUCKDB:-build/release/duckdb}`. Run `make release`
first, or pass `DUCKDB=/path/to/duckdb`. The script sets
`SIRIUS_CONFIG_FILE=test/cpp/integration/s3/sirius.yaml` by default so the
autoloaded Sirius extension uses the small integration-test GPU reservation.

The in-tree DuckDB build includes the `tpch` extension, so the fixture script
uses `LOAD tpch` without `INSTALL tpch` and does not require outbound network.

AWS portability uses the same benchmark test with:

```sh
SIRIUS_BENCH_BACKEND=aws-s3
SIRIUS_BENCH_AWS_S3_REGION=us-east-1
SIRIUS_BENCH_AWS_S3_BUCKET=<bucket>
SIRIUS_BENCH_AWS_S3_KEY=tpch/lineitem_sf10.parquet
SIRIUS_BENCH_AWS_S3_ACCESS_KEY=<access-key>
SIRIUS_BENCH_AWS_S3_SECRET_KEY=<secret-key>
```
