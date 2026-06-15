# S3 integration tests

Catch2 `[s3]` tests for the S3 backend — from the lower-level `s3_ioctx` and
retry/cache paths through `scan_manager`, parquet split-provider routing,
`describe_parquet`, and the SQL-over-S3 surface.

## How the S3 backend is managed

The S3 backend is a **SeaweedFS `weed` process started by the test binary
itself** — no Docker. The harness lives in `test/cpp/utils/s3_backend.*` and is
driven from `unittest.cpp`'s `main()`. There is **no** `make s3-up`/`s3-down`,
no `docker-compose.yml`, and no `env.sh` to source — when opted in, the binary:

1. spawns a single `weed server` that serves the S3 API over **both** plain HTTP
   (`-s3.port`) and self-signed TLS (`-s3.port.https`) at once — both listeners
   share one filer backend — on **dynamically-chosen free ports**,
2. generates a self-signed cert in-process and points `weed` at it, and enforces
   the test access/secret keys via an `-s3.config` identities file,
3. generates the local fixtures (`generate_fixtures.py`) and uploads them
   **once** (over HTTP) using Sirius's own SigV4 signer + libcurl, and
4. publishes the `SIRIUS_TEST_S3_*` env vars the tests already consume.

The process is torn down at process exit; on Linux it also inherits a
parent-death signal (`PR_SET_PDEATHSIG`) so a crashed test binary takes it down
too. The `weed` binary is resolved from `PATH` (the pixi `seaweedfs` package);
override with `SIRIUS_TEST_WEED`.

### Opt-in

Bring-up is gated by `SIRIUS_TEST_S3_AUTO=1` so the default `make test` suite
never spawns a server. The behavior:

- `SIRIUS_TEST_S3_ENDPOINT` already set → used as-is, no server started
  (this is how the real-AWS `[s3][aws]` gates and manual runs work).
- `SIRIUS_TEST_S3_AUTO` not set → tests skip, exactly as before.
- `SIRIUS_TEST_S3_AUTO=1` → the server comes up. Failure skips, unless
  `SIRIUS_TEST_S3_STRICT=1`, which makes a bring-up failure abort the run
  (non-zero exit) instead of silently going green.

## Requirements

- The `weed` binary on `PATH` — provided by the pixi `seaweedfs` package (or set
  `SIRIUS_TEST_WEED` to a `weed` of your choice). No Docker.
- Python 3.9+ (stdlib only) and `openssl` at run time, both from the pixi env.
- For the large gate: the in-tree `build/release/duckdb` CLI (or
  `SIRIUS_TEST_DUCKDB`) and an SF1 `lineitem` parquet at
  `test_datasets/tpch_parquet_sf1/lineitem.parquet` (or `SIRIUS_TEST_S3_LARGE_SOURCE`),
  which the harness replicates 10x to reach ~SF10 scale. Generate the SF1 source
  with `scripts/tpch_to_parquet.sql`. The DuckDB build has no `tpch` extension, so
  `CALL dbgen` is not used.

## Typical flow

```bash
# Default Catch2 suite — does not start the S3 backend.
make test

# Standard S3 correctness gate (auto-managed SeaweedFS, strict mode), runs every
# non-large, non-aws [s3][integration] test including the SQL-over-S3 subset.
make s3-test

# Opt-in large-SF10 SQL-over-S3 gate. The harness generates + uploads
# lineitem_sf10.parquet once (cached across the per-case invocations) and runs
# the [s3][sql][large] tests. Much slower than s3-test.
make s3-test-large

# Manually, without the Makefile:
SIRIUS_TEST_S3_AUTO=1 SIRIUS_TEST_S3_STRICT=1 \
  build/release/extension/sirius/test/cpp/sirius_unittest "[s3]~[large]~[aws]"
```

## SeaweedFS version

The `weed` version is pinned by the pixi `seaweedfs` dependency (`pixi.toml`).
To bump it, edit that pin and confirm `make s3-test` still passes.

## Fixtures

`generate_fixtures.py` writes deterministic blobs (seeded PRNG) plus copies of
the standard TPCH Parquet fixtures, so a second run produces identical files and
the sha256 manifest stays stable. The harness writes them under a temp dir and
points `SIRIUS_TEST_S3_LOCAL_DIR` at it.

| file | size | purpose |
|---|---|---|
| `hello.txt` | 16 B | HEAD + tiny-range read |
| `small.bin` | 20 KiB | bit-equal full-object read via `datasource_factory` |
| `medium.bin` | 8 MiB | multi-range reads at odd offsets |
| `parquet/*.parquet` | varies | standard TPCH Parquet fixtures reused by the S3 datasource, scan-manager, split-provider, and SQL-over-S3 tests. |
| `tpch/lineitem_sf10.parquet` | ~1.5 GiB | opt-in large fixture, generated only when `SIRIUS_TEST_S3_LARGE=1` (`make s3-test-large`). |

A single `weed` filer backend serves both the HTTP and HTTPS endpoints, so the
fixtures are uploaded once and visible on both. The HTTPS path uses the generated
CA bundle (`SIRIUS_TEST_S3_CA_BUNDLE`) so `s3_ioctx` exercises TLS verification
rather than disabling certificate checks.

The S3 Parquet tests use `parquet/nation.parquet`, whose TPCH contents are fixed
and small enough for direct row-level assertions: 25 nations, keys 0-24, and
five nations per region.

## Notes

- The backend uses region `us-east-1`; `SIRIUS_TEST_S3_REGION` matches.
- When `SIRIUS_TEST_S3_*` is not set the tests `SUCCEED`/`WARN` with a skip
  message rather than failing — intentional so the default `sirius_unittest` run
  stays green where the S3 backend was not brought up.
- When `SIRIUS_TEST_S3_STRICT=1`, once the env is present, live failures (e.g.
  `HEAD` or `datasource_factory::create` errors) fail the test instead of
  skipping. `make s3-test` enables this.
- SQL-over-S3 tests cover `sirius_read_parquet('s3://...')` directly and the
  `gpu_execution('... read_parquet("s3://...") ...')` rewrite path. The large
  variants are tagged `[s3][sql][large]` and hidden from the default run.
