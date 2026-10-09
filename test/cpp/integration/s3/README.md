# S3 integration tests

Catch2 tests cover REST range reads, retries, cache reads, scan-manager
`create_datasource`, `describe_parquet`, and SQL over S3. Loopback routing
tests run in the default unit suite.

## Running the gates

The S3 harness is included whenever the C++ tests are built. The Pixi
environment provides the prebuilt `weed` executable; no Docker daemon or Go
toolchain is required. Server startup is opt-in through `SIRIUS_TEST_S3_AUTO`.

| Command | Selection and environment |
|---|---|
| `make test` | Default unit suite; does not start a server unless opted in |
| `make s3-test` | Non-large, non-AWS S3 integration cases; AUTO and STRICT enabled |
| `make s3-test-large` | Two processes: cache-enabled SF10 cases plus SF1 TPC-H and glob-scale, then cache-disabled SF10 cases |
| `make s3-test-aws` | Manual AWS cases; STRICT enabled; supply the endpoint, bucket and temporary credentials |
| `make s3-tpch` | Deprecated; still enables TPCH and runs both tiny and SF1 suites |
| `make s3-test-aws-sigv4` | Deprecated; forwards to `s3-test-aws` |
| `make s3-test-aws-broker` | Deprecated; prints a warning and runs nothing |

The S3 make targets pass `--order decl` to run cases in declaration order.
The manual equivalent of `make s3-test` is:

```bash
SIRIUS_TEST_S3_AUTO=1 SIRIUS_TEST_S3_STRICT=1 \
  build/release/extension/sirius/test/cpp/sirius_unittest --order decl "[s3][integration]~[large]~[aws]"
```

## Tags and gates

Every S3 case carries `[s3]`. Gate membership comes from the full
Makefile selector, not from any one tag:

| Gate | Selector |
|---|---|
| `make s3-test` | `[s3][integration]~[large]~[aws]` |
| `make s3-test-large`, first process | `[s3][sql][large][large-cache],[s3][integration][sql][tpch][large],[s3][large][glob-scale]` |
| `make s3-test-large`, second process | `[s3][sql][large][large-nocache]` |
| `make s3-test-aws` | `[s3][aws]` |
| `make s3-tpch` (deprecated) | `[s3][integration][sql][tpch]` |

Adjacent tags mean AND; commas separate OR alternatives. Keep the
gate tags when retagging a case. `[integration]` also tells the listener
in `test/cpp/unittest.cpp` to resume the shared integration DuckDB
environment.

The S3 tag vocabulary is:

| Role | Tags |
|---|---|
| Gate selection | `[s3]`, `[integration]`, `[sql]`, `[large]`, `[large-cache]`, `[large-nocache]`, `[tpch]`, `[glob-scale]`, `[aws]` |
| Execution path | `[transparent]` marks the `SET gpu_execution` path |
| Topics | `[rest]`, `[sigv4]`, `[list]`, `[filesystem]`, `[glob]`, `[routing]`, `[describe_parquet]`, `[config]`, `[footerbind]`, `[pushdown]`, `[nested]`, `[fallback]`, `[kvikio]` |
| File topics | `[uri_parser]`, `[object_store_config]` |
| Slow cases | `[stress]` |

Use at most two topic tags per case. No make target selects on the
topic, file-topic or stress tags. Unit cases without `[integration]`,
`[large]` or `[aws]` run in `make test` unless hidden.

The large gate selects three SF10 cases with `[large-cache]` and three
with `[large-nocache]`. The tiny `[tpch]` suite runs in `s3-test`;
`[tpch][large]` selects SF1 in the first large process, alongside the
1001-object `[glob-scale]` case. `[aws]` cases belong to the manual
real-AWS gate.

Without an external endpoint, SeaweedFS-backed cases skip when
`SIRIUS_TEST_S3_AUTO` is unset. The gates set
`SIRIUS_TEST_S3_STRICT=1` so missing prerequisites fail instead.
SF10, SF1 TPC-H and glob-scale also require their LARGE, TPCH and
GLOB_SCALE switches; the targets export them for the appropriate
process. The three harness-PUT cases skip against an external endpoint,
even in strict mode.

Catch2's `[.]` hides a case from an unfiltered run only. A hidden case
still runs when it matches an explicit positive name or tag selector,
subject to that selector's exclusions. These 15 hidden cases run in
`make s3-test`:

- `DuckDB external file cache invalidates an overwritten S3 range by ETag`
- `gpu_execution rejects operations on nested S3 parquet columns cleanly`
- `gpu_execution S3 nested parquet projections match local DuckDB CPU`
- `rest_ioctx generated LIST scale obeys the default safety caps`
- `S3 pushdown non-pruned aggregate still matches the local parquet oracle`
- `S3 pushdown selective filters still match the local parquet oracle`
- `S3 pushdown shape-C zero-side joins match the local parquet oracle`
- `S3 pushdown zero-input grouped aggregate still emits no groups`
- `S3 pushdown zero-input ungrouped aggregates emit SQL identity and null values`
- `S3 pushdown zero-input ungrouped count emits the aggregate identity row`
- `sirius_httpfs exposes S3 ETags as DuckDB version tags`
- `sirius_httpfs glob helper throws instead of silently truncating matched files`
- `sirius_httpfs opens through FileOpener and reads positional ranges`
- `sirius_httpfs positional reads fail on short reads and negative sizes`
- `transparent S3 TPC-H Q1-Q22 match the tiny local CPU oracle`

Before changing tags, compare the case-name lists for every gate
selector above. Run one command per selector, keeping the entire
selector in one quoted argument:

```bash
bin=build/release/extension/sirius/test/cpp/sirius_unittest
spec='[s3][integration]~[large]~[aws]'

# Catch2 v2
"$bin" --list-test-names-only "$spec"

# Catch2 v3
"$bin" --list-tests --verbosity quiet "$spec"
```

The gate lists contain 105, 5, 3 and 3 cases respectively; the deprecated
TPC-H target selects two. `--list-tags "[s3]"` lists the 26 tags above
plus `[.]`.

## SeaweedFS lifecycle

The binary starts one SeaweedFS process with HTTP and HTTPS listeners sharing
one backend, using dynamically selected loopback ports. It generates a
self-signed certificate with `openssl`, uploads fixtures with SigV4 and libcurl,
and publishes the environment used by the tests.

An existing `SIRIUS_TEST_S3_ENDPOINT` is used as-is. Otherwise,
`SIRIUS_TEST_S3_AUTO=1` opts into server startup. SQL, httpfs and TPC-H
tests pass missing required `SIRIUS_TEST_S3_*` settings to
`skip_or_fail_unless`: a skip, or a failure with `SIRIUS_TEST_S3_STRICT=1`.
REST, describe_parquet and kvikio tests use that helper only for
`ensure_s3_test_env`. Once an endpoint is set, missing BUCKET,
ACCESS_KEY or SECRET_KEY fails those tests regardless of STRICT.

The LARGE, TPCH and GLOB_SCALE switches use the same skip-or-fail rule.
Live HEAD/GET and query errors fail regardless of STRICT, except that
SF10 tests report a failed describe of the SF10 object as a skip unless
STRICT is set. The three tests that PUT objects into managed SeaweedFS skip when
the endpoint is externally managed; device tests also report unavailable CUDA.
The PUT cases cover ETag invalidation, kvikio stream ordering, and
`transparent S3 glob refuses semantic type drift before decoding`.

`unittest.cpp` terminates and reaps the server before exiting. On Linux the
server also receives SIGKILL if the test process dies. Each process has its own
working directory under `<tmp>/sirius-s3-seaweedfs-<uid>`, removed at shutdown.

Requirements: `weed` (provided and locked by Pixi), Python 3.9+, and `openssl`.
Set `SIRIUS_TEST_WEED` to use another executable. SF1 and SF10 generation need the built
DuckDB CLI or `SIRIUS_TEST_DUCKDB`, which loads the TPC-H extension to generate
the data. The SF1 and SF10 fixtures are cached across processes; file locks serialize
concurrent generation. Set `TMPDIR` to a disk with several GB free for large
fixtures and the server data. SeaweedFS also requires the disk to remain above
its default 1% free-space reserve.

After changing the SeaweedFS version, run `make s3-test`.

## Fixtures

`generate_fixtures.py` creates deterministic blobs and copies the
committed parquet fixtures. The harness adds the edge-type and special-key
files. `SIRIUS_TEST_S3_LOCAL_DIR` points to the uploaded local copy used by
CPU oracles. `MANIFEST.sha256` is written next to that directory and is
not uploaded.

Fixtures are uploaded once and are available through both HTTP and HTTPS.

| Object group | Contents |
|---|---|
| `hello.txt` | 16 bytes; HEAD and tiny reads |
| `small.bin` | 20 KiB; exact-byte reads |
| `medium.bin` | 8 MiB; ranges at odd offsets |
| `parquet/*` | Committed parquet files plus runtime-generated `edge_types.parquet` |
| `glob/multi/*` | Two nation copies and `region.parquet` |
| `glob/hive/*` | Hive partition directories |
| `root_a.parquet`, `root_b.parquet` | Bucket-root glob inputs |
| `glob-enc/*` | 12 keys covering percent bytes, spaces, slashes, query/fragment delimiters and partition directories |
| `tpch/lineitem_sf10.parquet` | SF10 lineitem; requires LARGE |
| `tpch/sf1/*` | Eight SF1 tables; requires TPCH |
| `glob-scale/part_*.parquet` | 1001 nation copies; requires GLOB_SCALE |
| Objects written during tests | ETag overwrite, kvikio stream-ordering inputs and schema-drift parquet pair |

The SF10 cache was 2,223,320,375 bytes (about 2.07 GiB), measured on
2026-05-21, before the DuckDB v1.5.6 pin.
`maybe_upload_large_fixture` generates it with DuckDB's
`CALL dbgen(sf=10)` and `COPY ... (FORMAT PARQUET)`; size depends on the
generator version and encoding. It is reused when non-empty, not regenerated
on every run.

`nation.parquet` has 25 rows, keys 0-24, and five nations per region.
The HTTPS tests use the generated CA bundle so `rest_ioctx` checks the
certificate. SeaweedFS uses region `us-east-1`.

## Environment

User inputs:

| Variable | Purpose |
|---|---|
| `SIRIUS_TEST_S3_AUTO` | Start managed SeaweedFS when no endpoint is supplied |
| `SIRIUS_TEST_WEED` | Override the `weed` executable |
| `SIRIUS_TEST_S3_STRICT` | Fail on missing prerequisites or failed startup |
| `SIRIUS_TEST_S3_ENDPOINT` | Use an existing endpoint instead of starting a server |
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
