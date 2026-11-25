# M9gd CPU benchmarks

The [DuckDB 2.0.0 S3 recipe](duckdb-2.0-s3/README.md) preserves the SF1000,
SF3000 and SF10000 measurements on `m9gd.24xlarge` in North Virginia,
with additional SF100 and SF300 measurements.
Each query runs twice consecutively; DuckDB restarts and Linux caches are cleared
between query pairs. Both external-file and Parquet-metadata caches are enabled.

See the [selected results and iteration sums](duckdb-2.0-s3/results/SELECTED.md),
exact settings, independently validated answers' proof records, and original
measured scripts. These are CPU DuckDB results; Sirius is not loaded. The M9gd
recipe is separate from the [G7e](../g7e/README.md) and
[GB300](../gb300/README.md) GPU protocols.

The November 25 SF100–SF10000 selections use a native Release build and a
64 KiB page kernel. SF10000 combines original measurements and retries;
**Q12 uses a 256 GiB memory limit**, while other queries use 296.7 GiB.
That selection is not a uniform-memory full-suite run. The earlier 4 KiB
baseline remains preserved in its dated folder.
