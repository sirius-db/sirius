# S3 integration tests

Catch2 tests cover REST range reads, retries, cache reads, scan-manager
`create_datasource`, `describe_parquet`, and SQL over S3. Loopback routing
tests run in the default unit suite.

## Running the gates

`SIRIUS_BUILD_S3_TESTS` defaults to `ON`, including the MinIO harness. CMake
fetches and patches testcontainers-native at configure time and builds its
Go c-archive (`cmake/testcontainers_native.cmake`). The vcpkg presets turn
S3 tests off.

| Command | Selection and environment |
|---|---|
| `make test` | Default unit suite; does not start Docker unless opted in |
| `make s3-test` | Non-large, non-AWS S3 integration cases; AUTO and STRICT enabled |
| `make s3-test-large` | Two processes: cache-enabled SF10 cases plus SF1 TPC-H and glob-scale, then cache-disabled SF10 cases |
| `make s3-test-aws` | Manual AWS cases; STRICT enabled; supply the endpoint, bucket and temporary credentials |
| `make s3-tpch` | Deprecated; still enables TPCH and runs both tiny and SF1 suites |
| `make s3-test-aws-sigv4` | Deprecated; forwards to `s3-test-aws` |
| `make s3-test-aws-broker` | Deprecated; prints a warning and runs nothing |

The manual equivalent of `make s3-test` is:

```bash
SIRIUS_TEST_S3_AUTO=1 SIRIUS_TEST_S3_STRICT=1 \
  build/release/extension/sirius/test/cpp/sirius_unittest "[s3][integration]~[large]~[aws]"
```

Catch2's `[.]` hides a case from an unfiltered run, not from a matching
tag selector. Ten hidden cases currently run in `s3-test`: the three SQL
join/nested-column cases, the REST LIST-scale case, five HTTP filesystem
cases (LIST caps, positional reads and ETag validation), and the tiny
TPC-H suite.

## MinIO lifecycle

The binary starts two MinIO containers on dynamically mapped ports, one
HTTP and one HTTPS. It generates a self-signed certificate with `openssl`,
uploads fixtures from the host with SigV4 and libcurl, and publishes the
environment used by the tests. No separate setup script is needed.

An existing `SIRIUS_TEST_S3_ENDPOINT` is used as-is. Otherwise,
`SIRIUS_TEST_S3_AUTO=1` opts into container startup. SQL, httpfs and TPC-H
tests pass missing required `SIRIUS_TEST_S3_*` settings to
`skip_or_fail_unless`: a skip, or a failure with `SIRIUS_TEST_S3_STRICT=1`.
REST, describe_parquet and kvikio tests use that helper only for
`ensure_s3_container_env`. Once an endpoint is set, missing BUCKET,
ACCESS_KEY or SECRET_KEY fails those tests regardless of STRICT.

The LARGE, TPCH and GLOB_SCALE switches use the same skip-or-fail rule.
Live HEAD/GET and query errors fail regardless of STRICT, except that
SF10 tests report a failed describe of the SF10 object as a skip unless
STRICT is set. The two tests that PUT objects into managed MinIO skip when
the endpoint is externally managed; device tests also report unavailable CUDA.

`unittest.cpp` calls explicit container shutdown before exiting. The
library's default Ryuk reaper is best effort; a killed process can leave
containers running.

The working directory, `<tmp>/sirius-s3-testcontainers`, is reused across
runs and shared by users on the host. Do not run conflicting fixture
generators there concurrently.

Requirements: a reachable Docker daemon, the Pixi Go toolchain for the
bridge build, Python 3.9+ and `openssl`. Both SF10 and SF1 generation need
the built DuckDB CLI or `SIRIUS_TEST_DUCKDB`.

The image is pinned in `s3_container.cpp`:

| Image | Tag |
|---|---|
| `minio/minio` | `RELEASE.2025-09-07T16-13-09Z-cpuv1` |

After changing `kMinioImage`, run `make s3-test`.

## Fixtures

`generate_fixtures.py` creates deterministic blobs and copies the
committed parquet fixtures. The harness adds the edge-type and special-key
files. `SIRIUS_TEST_S3_LOCAL_DIR` points to the uploaded local copy used by
CPU oracles. `MANIFEST.sha256` is written next to that directory and is
not uploaded.

| Object group | Contents | Upload |
|---|---|---|
| `hello.txt` | 16 bytes; HEAD and tiny reads | HTTP + HTTPS |
| `small.bin` | 20 KiB; exact-byte reads | HTTP + HTTPS |
| `medium.bin` | 8 MiB; ranges at odd offsets | HTTP + HTTPS |
| `parquet/*` | Committed parquet files plus runtime-generated `edge_types.parquet` | HTTP + HTTPS |
| `glob/multi/*` | Two nation copies and `region.parquet` | HTTP + HTTPS |
| `glob/hive/*` | Hive partition directories | HTTP + HTTPS |
| `root_a.parquet`, `root_b.parquet` | Bucket-root glob inputs | HTTP + HTTPS |
| `glob-enc/*` | 12 keys covering percent bytes, spaces, slashes, query/fragment delimiters and partition directories | HTTP + HTTPS |
| `tpch/lineitem_sf10.parquet` | SF10 lineitem; requires LARGE | HTTP + HTTPS |
| `tpch/sf1/*` | Eight SF1 tables; requires TPCH | HTTP + HTTPS |
| `glob-scale/part_*.parquet` | 1001 nation copies; requires GLOB_SCALE | HTTP only |
| Objects written during tests | ETag overwrite and kvikio stream-ordering inputs | HTTP only |

The SF10 cache was 2,223,320,375 bytes (about 2.07 GiB), measured on
2026-05-21, before the DuckDB v1.5.6 pin.
`maybe_upload_large_fixture` generates it with DuckDB's
`CALL dbgen(sf=10)` and `COPY ... (FORMAT PARQUET)`; size depends on the
generator version and encoding. It is reused when non-empty, not regenerated
on every run.

`nation.parquet` has 25 rows, keys 0-24, and five nations per region.
The HTTPS tests use the generated CA bundle so `rest_ioctx` checks the
certificate. MinIO uses region `us-east-1`.

## Environment

User inputs:

| Variable | Purpose |
|---|---|
| `SIRIUS_TEST_S3_AUTO` | Start managed MinIO when no endpoint is supplied |
| `SIRIUS_TEST_S3_STRICT` | Fail on missing prerequisites or failed startup |
| `SIRIUS_TEST_S3_ENDPOINT` | Use an existing endpoint instead of starting containers |
| `SIRIUS_TEST_S3_SESSION_TOKEN` | Session token for temporary credentials |
| `SIRIUS_TEST_S3_LARGE` | Generate/upload SF10 lineitem |
| `SIRIUS_TEST_S3_TPCH` | Generate/upload all eight SF1 tables |
| `SIRIUS_BENCH_S3_TPCH` | Deprecated; generates/uploads SF1 only, without enabling the SF1 test case |
| `SIRIUS_TEST_S3_GLOB_SCALE` | Upload the 1001-object glob fixture |
| `SIRIUS_TEST_S3_PARQUET_SOURCE` | Override the source parquet directory |
| `SIRIUS_TEST_DUCKDB` | Override the CLI used for SF10/SF1 generation |
| `SIRIUS_BENCH_S3_KEY` | Override the SF10 object key used for upload and reads |

Published by managed startup; supply the applicable values yourself for an
external endpoint:

| Variable | Value |
|---|---|
| `SIRIUS_TEST_S3_ENDPOINT` | HTTP endpoint |
| `SIRIUS_TEST_S3_HTTPS_ENDPOINT` | HTTPS endpoint |
| `SIRIUS_TEST_S3_REGION` | Signing region |
| `SIRIUS_TEST_S3_ACCESS_KEY`, `SIRIUS_TEST_S3_SECRET_KEY` | Credentials |
| `SIRIUS_TEST_S3_BUCKET` | Fixture bucket |
| `SIRIUS_TEST_S3_LOCAL_DIR` | Local oracle root |
| `SIRIUS_TEST_S3_CA_BUNDLE` | Generated certificate |
| `SIRIUS_TEST_S3_TPCH_LOCAL_DIR` | SF1 oracle directory, when TPCH is enabled |
| `SIRIUS_PR6_LARGE_LOCAL_PARQUET` | SF10 oracle file, when LARGE is enabled |
| `SIRIUS_TEST_S3_KEY` | Default object key; no current test reads it |

Deprecated manual overrides retain their precedence and warn only when used:

| Variable | Replacement |
|---|---|
| `SIRIUS_PR6_LARGE_S3_KEY` | `SIRIUS_BENCH_S3_KEY`, also used by the uploader |
| `SIRIUS_BENCH_WORK_DIR` | `SIRIUS_PR6_LARGE_LOCAL_PARQUET` pointing to the file |

`SIRIUS_BENCH_WORK_DIR` has lower priority than
`SIRIUS_PR6_LARGE_LOCAL_PARQUET`. Managed LARGE startup always sets the
latter, so it silently ignores `SIRIUS_BENCH_WORK_DIR`.
