# M9gd CPU benchmarks

The [DuckDB 2.0.0 S3 recipe](duckdb-2.0-s3/README.md) preserves the November 24,
2026 SF1000, SF3000 and SF10000 measurements on `m9gd.24xlarge` in North Virginia.
Each query runs twice consecutively; DuckDB restarts and Linux caches are cleared
between query pairs. Both external-file and Parquet-metadata caches are enabled.

See the [results and complete iteration sums](duckdb-2.0-s3/results/20261124/RESULTS.md),
exact settings, independently validated answers' proof records, and original
measured scripts. These are CPU DuckDB results; Sirius is not loaded. The M9gd
recipe is separate from the [G7e](../g7e/README.md) and
[GB300](../gb300/README.md) GPU protocols.
