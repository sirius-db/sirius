# S3 integration tests

Catch2 `[s3]` tests for the S3 backend — from the lower-level `s3_ioctx` and
retry/cache paths through `scan_manager`, parquet split-provider routing,
`describe_parquet`, and the SQL-over-S3 surface.

## How MinIO is managed

MinIO is started **by the test binary itself** via the vendored, patched
testcontainers-native bridge (`third_party/testcontainers-native`). The harness
lives in `test/cpp/utils/s3_container.*` and is driven from `unittest.cpp`'s
`main()`. There is **no** `make s3-up`/`s3-down`, no `docker-compose.yml`, and no
`env.sh` to source — when opted in, the binary:

1. starts two MinIO containers (plain HTTP + self-signed TLS) on **dynamically
   mapped** host ports,
2. generates a self-signed cert in-process and mounts it into the TLS instance,
3. generates the local fixtures (`generate_fixtures.py`) and uploads them to
   both endpoints **from the host** using Sirius's own SigV4 signer + libcurl
   (no `mc` container, no docker network), and
4. publishes the `SIRIUS_TEST_S3_*` env vars the tests already consume.

Containers are torn down at process exit; testcontainers' Ryuk sidecar
guarantees cleanup even on a crash.

### Opt-in

Bring-up is gated by `SIRIUS_TEST_S3_AUTO=1` so the default `make test` suite
never touches Docker. The behavior:

- `SIRIUS_TEST_S3_ENDPOINT` already set → used as-is, no container started
  (this is how the real-AWS `[s3][aws]` gates and manual runs work).
- `SIRIUS_TEST_S3_AUTO` not set → tests skip, exactly as before.
- `SIRIUS_TEST_S3_AUTO=1` → containers come up. Failure skips, unless
  `SIRIUS_TEST_S3_STRICT=1`, which makes a bring-up failure abort the run
  (non-zero exit) instead of silently going green.

## Requirements

- Docker (reachable daemon).
- A Go toolchain (provided by pixi) and network access on the first
  configure/build: CMake fetches + patches upstream testcontainers-native and
  builds it as a Go c-archive (see `cmake/testcontainers_native.cmake`).
- Python 3.9+ (stdlib only) and `openssl` at run time, both from the pixi env.
- For the large gate: the in-tree `build/release/duckdb` CLI (or
  `SIRIUS_TEST_DUCKDB`) to generate the SF10 fixture.

## Typical flow

```bash
# Default Catch2 suite — does not start MinIO.
make test

# Standard S3 correctness gate (auto-managed MinIO, strict mode), runs every
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

## Pinned image version

The MinIO image is pinned to an exact release tag (in `s3_container.cpp`) rather
than `:latest` so the same Sirius commit is reproducible over time:

| image | tag |
|---|---|
| `minio/minio` | `RELEASE.2025-09-07T16-13-09Z-cpuv1` |

To bump it, edit `kMinioImage` in `test/cpp/utils/s3_container.cpp` and confirm
`make s3-test` still passes.

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

The same standard fixtures are uploaded to both the HTTP and HTTPS instances.
The HTTPS path uses the generated CA bundle (`SIRIUS_TEST_S3_CA_BUNDLE`) so
`s3_ioctx` exercises TLS verification rather than disabling certificate checks.

The S3 Parquet tests use `parquet/nation.parquet`, whose TPCH contents are fixed
and small enough for direct row-level assertions: 25 nations, keys 0-24, and
five nations per region.

## Notes

- MinIO signs with `us-east-1`; `SIRIUS_TEST_S3_REGION` matches.
- When `SIRIUS_TEST_S3_*` is not set the tests `SUCCEED`/`WARN` with a skip
  message rather than failing — intentional so the default `sirius_unittest` run
  stays green on runners without Docker.
- When `SIRIUS_TEST_S3_STRICT=1`, once the env is present, live failures (e.g.
  `HEAD` or `datasource_factory::create` errors) fail the test instead of
  skipping. `make s3-test` enables this.
- SQL-over-S3 tests cover `sirius_read_parquet('s3://...')` directly and the
  `gpu_execution('... read_parquet("s3://...") ...')` rewrite path. The large
  variants are tagged `[s3][sql][large]` and hidden from the default run.
