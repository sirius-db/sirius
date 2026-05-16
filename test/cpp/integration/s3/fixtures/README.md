# S3 Perf Benchmark Fixture

`generate_perf_dataset.sh` prepares the hidden `[!benchmark][perf][bench]`
Catch2 benchmark fixture:

1. Uses DuckDB's `tpch` extension to generate SF=10 data.
2. Exports the full `lineitem` table used by the benchmark to Parquet.
3. Uploads the Parquet file to a MinIO/S3-compatible bucket with `mc`.

The benchmark defaults are:

```sh
SIRIUS_BENCH_S3_ENDPOINT=http://127.0.0.1:9000
SIRIUS_BENCH_S3_BUCKET=${SIRIUS_TEST_S3_BUCKET:-sirius-test}
SIRIUS_BENCH_S3_KEY=tpch/lineitem_sf10.parquet
```

For local MinIO, run `make s3-up` first, then run the generator. The planned
`make s3-bench` target should call both steps before invoking `sirius_unittest`
with the hidden benchmark tag.

DuckDB may need network access the first time it installs the `tpch` extension.
On CI hosts without outbound network, pre-populate `~/.duckdb/extensions/` or
install the extension cache before running this fixture generator.

AWS portability uses the same benchmark test with:

```sh
SIRIUS_BENCH_BACKEND=aws-s3
SIRIUS_BENCH_AWS_S3_REGION=us-east-1
SIRIUS_BENCH_AWS_S3_BUCKET=<bucket>
SIRIUS_BENCH_AWS_S3_KEY=tpch/lineitem_sf10.parquet
SIRIUS_BENCH_AWS_S3_ACCESS_KEY=<access-key>
SIRIUS_BENCH_AWS_S3_SECRET_KEY=<secret-key>
```
