# Implementation Plan: Closing the Multi-Backend Datasource Gaps

**Baseline:** `feature-newdatasourcesupport`
**Reference:** [gap.md](gap.md), [sirius-multidatasource-proposal.md](sirius-multidatasource-proposal.md) §9
**Date:** 2026-04-17

Each PR closes **exactly one** gap (unless two gaps are too tightly coupled to split), can be checked in independently, passes CI on its own, and can be reverted in isolation. PRs are ordered by dependency — a later PR only becomes feasible after its prerequisite lands.

---

## Rules

1. **One gap per PR.** The "Closes" row names at most one gap; when two must ship together, the "Rationale" row explains the hard coupling.
2. **Independently checkin-able.** After each PR lands:
   - `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make` succeeds
   - `sirius_unittest` is green
   - `make test` is green
   - `pre-commit run -a` emits no new warnings
3. **Every PR ships its own tests.** At least one Catch2 unit test or SQLLogicTest must directly exercise the new code path.
4. **No cross-PR TODOs.** If a gap needs follow-up, shrink the current PR to a shippable subset rather than leaving `// TODO: wire in PR5` comments behind.

---

## PR sequence

### PR0 — Link `liburing`
| | |
|---|---|
| **Closes** | Gap #1 (B1) |
| **Touches** | [`CMakeLists.txt`](../../CMakeLists.txt) |
| **Diff size** | ~5 lines |

**Changes**
- Add `pkg_check_modules(URING REQUIRED IMPORTED_TARGET liburing)` near the top of `CMakeLists.txt`.
- Append `PkgConfig::URING` to the `target_link_libraries(sirius_extension ...)` call at [`CMakeLists.txt:322-323`](../../CMakeLists.txt#L322-L323).

**Tests**
- `test/cpp/io/test_uring_link.cpp`: a minimal test that calls `io_uring_queue_init` + `io_uring_queue_exit` to guarantee the symbols are linked into the extension.
- Catch2 tag `[io_link]`, runs in the CI smoke stage.

**Verify**
- `nm build/release/extension/sirius/sirius.duckdb_extension | grep io_uring_` shows `U` or `T` entries.
- `ldd` lists `liburing.so.2`.

---

### PR1 — `datasource_factory` + `datasource_registry` (scaffolding)
| | |
|---|---|
| **Closes** | Gap #3 |
| **Touches** | `src/include/io/datasource_factory.hpp`, `src/io/datasource_factory.cpp` (new) |
| **Diff size** | ~200 lines |

**Changes**
- New type `datasource_registry`: `std::unordered_map<std::string, std::shared_ptr<sirius_ioctx>>`, guarded by `shared_mutex`.
- New type `datasource_factory`: `static std::unique_ptr<io_datasource> create(std::string_view uri, datasource_registry&, sirius_config const&)`. In this PR it does URI scheme extraction + registry lookup + `ioctx->make_datasource(io_object)` dispatch, supporting only `file://` and bare paths → `uring_ioctx`.
- No call-site changes; no `sirius_engine.cpp` changes. Pure additive scaffolding.

**Tests**
- `test/cpp/io/test_datasource_factory.cpp` tag `[datasource_factory]`:
  - `factory_returns_null_for_unknown_scheme`
  - `factory_parses_bare_path_as_file_scheme`
  - `factory_dispatches_to_registered_uring_ioctx` (the test wires up a `uring_ioctx` into a locally constructed registry)
  - `registry_is_thread_safe_for_concurrent_lookup`

**Verify**
- gcov coverage for the two new files ≥ 80%.

---

### PR2 — Construct registry in `sirius_engine`
| | |
|---|---|
| **Closes** | Gap #5 (D1) |
| **Depends on** | PR1 |
| **Touches** | [`src/sirius_engine.cpp`](../../src/sirius_engine.cpp), [`src/include/sirius_engine.hpp`](../../src/include/sirius_engine.hpp) |
| **Diff size** | ~60 lines |

**Changes**
- `sirius_engine` owns a `datasource_registry` (`shared_ptr`) and populates it with a default `uring_ioctx` at construction.
- Expose `sirius_engine::datasource_registry()` accessor.
- **Call sites not yet switched** — that's the next PR. This PR just stands the registry up and must survive the existing test suite untouched.

**Tests**
- `test/cpp/engine/test_registry_bootstrap.cpp` tag `[engine]`:
  - `engine_bootstrap_populates_registry_with_uring_ioctx`
  - `engine_destruction_releases_ioctx_cleanly` (asserts `shared_ptr::use_count` drops to zero)

**Verify**
- Existing SQLLogicTests all green (regression guard).

---

### PR3 — Replace 3 scan call sites with the factory
| | |
|---|---|
| **Closes** | Gap #4 |
| **Depends on** | PR2 |
| **Touches** | [`src/op/scan/parquet_scan_task.cpp`](../../src/op/scan/parquet_scan_task.cpp), [`src/op/scan/sirius_parquet_metadata_scan_operator.cpp`](../../src/op/scan/sirius_parquet_metadata_scan_operator.cpp) |
| **Diff size** | ~40 lines |

**Changes**
- [`parquet_scan_task.cpp:265`](../../src/op/scan/parquet_scan_task.cpp#L265) and [`:491`](../../src/op/scan/parquet_scan_task.cpp#L491): replace `cudf::io::datasource::create(path)` with `datasource_factory::create(path, engine.datasource_registry(), engine.config())`.
- [`sirius_parquet_metadata_scan_operator.cpp:251`](../../src/op/scan/sirius_parquet_metadata_scan_operator.cpp#L251): same substitution.
- For local-file paths the factory returns a thin wrapper around a `uring_ioctx`-backed datasource; its behavior must be equivalent to `cudf::io::datasource::create(path)` from the parquet reader's perspective.

**Tests**
- Full SQLLogicTest regression via `test/sql/tpch-sirius.test`.
- `test/cpp/operator/test_parquet_scan_via_factory.cpp` tag `[parquet_scan]`:
  - `scan_local_parquet_via_factory_equivalent_to_cudf_direct` (run the same parquet through both paths, assert `cudf::table` equality)
  - `scan_handles_missing_file_via_factory_error_path`

**Verify**
- `sirius_unittest "[parquet_scan]"` all green.
- TPC-H SF1 runtime within +3% of the `dev` baseline (factory indirection must not add observable overhead).

---

### PR4 — Fix `is_device_read_preferred` hard-coding
| | |
|---|---|
| **Closes** | Gap #8 (C1) |
| **Depends on** | PR1 (independent of PR2/3 but logically sits here) |
| **Touches** | [`src/include/io/types.hpp`](../../src/include/io/types.hpp), [`src/io/sirius_datasource.cpp`](../../src/io/sirius_datasource.cpp), [`src/include/io/uring/uring_ioctx.hpp`](../../src/include/io/uring/uring_ioctx.hpp), [`src/io/uring/uring_ioctx.cpp`](../../src/io/uring/uring_ioctx.cpp) |
| **Diff size** | ~40 lines |

**Changes**
- `sirius_ioctx` gains `virtual bool supports_device_read() const = 0;` and `virtual bool is_device_read_preferred(size_t) const = 0;`.
- `sirius_datasource::supports_device_read()` / `is_device_read_preferred()` forward to `ioctx_`.
- `uring_ioctx`: `supports_device_read() = true`, **`is_device_read_preferred() = false`** (the current path is CPU bounce, not real GDS).

**Tests**
- `test/cpp/io/test_device_read_flags.cpp` tag `[io_caps]`:
  - `uring_ioctx_does_not_prefer_device_read`
  - `sirius_datasource_forwards_caps_from_ioctx` (mock ioctx, toggle both return values, assert forwarding)

**Verify**
- No upstream caller regresses from this flip (grep `is_device_read_preferred` to confirm the branch currently has zero callers).

---

### PR5 — Migrate `read_range_into_allocation` to `host_read_ranges_async`
| | |
|---|---|
| **Closes** | Gap #6 (A1) |
| **Depends on** | PR3 |
| **Touches** | [`src/op/scan/parquet_scan_task.cpp`](../../src/op/scan/parquet_scan_task.cpp#L634-L660) |
| **Diff size** | ~80 lines |

**Changes**
- Replace the `host_read_async` loop in `read_range_into_allocation` with a single `host_read_ranges_async` call over a `std::vector<host_read_req>`.
- Note: `host_read_ranges_async` only exists on `io_datasource` (`sirius_datasource`), not on the base `cudf::io::datasource`. Either `dynamic_cast` at the call site or tighten the factory return type. We pick the latter: the factory's declared return type becomes `std::unique_ptr<io_datasource>`.

**Tests**
- `test/cpp/operator/test_parquet_scan_batch_read.cpp` tag `[parquet_scan_batch]`:
  - `batch_read_returns_same_bytes_as_loop` (run the old loop and new batch path against the same parquet, compare buffer bytes)
  - `batch_read_handles_partial_failure` (mock ioctx, fail the second chunk, assert exception aggregation)
- Optional micro-benchmark: `test/cpp/bench/bench_parquet_batch_read.cpp` records baseline vs. batched wall time.

**Verify**
- No TPC-H SF1 regression; ideally scan-heavy queries like Q6/Q12 improve by 5-15%.

---

### PR6 — `gds_ioctx` via KvikIO
| | |
|---|---|
| **Closes** | Gap #7 (D2) |
| **Depends on** | PR4 (virtualized caps), PR2 (registry) |
| **Touches** | `src/include/io/gds/` (new), `src/io/gds/` (new), `CMakeLists.txt`, `src/sirius_engine.cpp` |
| **Diff size** | ~400 lines |

**Changes**
- New `gds_io_object`: holds a `kvikio::FileHandle`; `raw_file_cache_id()` returns device id + inode.
- New `gds_ioctx`: `supports_device_read() = true`; `is_device_read_preferred()` depends on `kvikio::defaults::compat_mode()`. `host_read_async` rides KvikIO's POSIX fallback; `device_read_async` goes through cuFile.
- `gds_ioctx::host_read_ranges_async` aggregates `std::vector<std::future<size_t>>` from KvikIO.
- `datasource_factory`: if the path's underlying filesystem is NVMe and CUDA is available, dispatch to `gds_ioctx`; otherwise keep `uring_ioctx`. Detection is simplified in this PR to a new DuckDB variable `SET enable_gds='auto|off'`.
- `sirius_engine.cpp` registers `gds_ioctx` in the registry.

**Tests**
- `test/cpp/io/test_gds_ioctx.cpp` tag `[gds]` (enabled on CI runners with cuFile; skipped otherwise):
  - `gds_host_read_matches_posix`
  - `gds_device_read_dma_into_rmm_buffer` (post-read `cudaMemcpy` back to host and compare)
  - `gds_compat_mode_falls_back_to_posix`
- SQLLogicTest `test/sql/datasource/gds_read.test`: run `read_parquet('local.parquet')` with `SET enable_gds='auto'` and `='off'`, assert identical results.

**Verify**
- On a GDS-capable host, `nvidia-smi dmon -s u` shows DMA traffic on the GPU.
- CI on non-GDS hosts falls back to compat mode and tests still pass.

---

### PR7 — `sirius_config.object_store_config` + `SET s3_transport`
| | |
|---|---|
| **Closes** | Gap #11 (D3) |
| **Depends on** | PR2 |
| **Touches** | [`src/include/config.hpp`](../../src/include/config.hpp), [`src/config.cpp`](../../src/config.cpp), [`src/sirius_extension.cpp`](../../src/sirius_extension.cpp) |
| **Diff size** | ~120 lines |

**Changes**
- `sirius_config` grows `object_store_config { std::string endpoint; std::string region; std::string access_key; std::string secret_key; enum class transport { AUTO, HTTP, RDMA } s3_transport = AUTO; }`.
- DuckDB session variables (`SET s3_transport='rdma|http|auto'`, `SET s3_endpoint=...`, etc.) register via `ExtensionLoader` / `DBConfig::AddExtensionOption`, and a thin mapping layer pulls values into `sirius_config`.
- **No S3 ioctx added in this PR.** The factory reads the config but has no backend yet to consume it. This is intentional: wire the config plumbing first so PR9 can land an S3 backend independently.

**Tests**
- `test/cpp/config/test_object_store_config.cpp` tag `[config]`:
  - `set_s3_transport_rdma_updates_config`
  - `set_s3_endpoint_updates_config`
  - `unknown_s3_transport_value_rejected`
- SQLLogicTest `test/sql/datasource/set_s3_transport.test`: `SELECT current_setting('s3_transport')` round-trip.

**Verify**
- No existing-test regressions; `SHOW VARIABLES LIKE 's3_%'` lists the new variables.

---

### PR8 — URI normalization rules in the factory
| | |
|---|---|
| **Closes** | Gap A2 |
| **Depends on** | PR1 |
| **Touches** | `src/io/datasource_factory.cpp`, `src/include/io/uri_parser.hpp` (new) |
| **Diff size** | ~150 lines |

**Changes**
- New header `uri_parser.hpp`: `struct parsed_uri { std::string scheme; std::string host; std::string path; std::unordered_map<string, string> query; }` with a free `parse(std::string_view)` function (implementation in `uri_parser.cpp` if it grows).
- Supported shapes: `s3://bucket/key`, `s3://bucket/key?region=us-west-2`, `file:///abs/path`, bare paths, `gs://`, `azure://`.
- Normalization rules per scheme: reject empty keys, collapse `//bucket` vs `/bucket`, decode query strings.
- Factory swaps its placeholder URI handling for the parser.

**Tests**
- `test/cpp/io/test_uri_parser.cpp` tag `[uri]`, at least 20 cases:
  - well-formed paths, relative-path rejection, region query extraction, percent-encoded characters, trailing-slash handling
- Fuzzy tests (10k random URIs) asserting the parser never crashes.

**Verify**
- PR1 factory unit tests stay green (parser is backward-compatible).

---

### PR9 — `s3_ioctx` (libcurl + SigV4)
| | |
|---|---|
| **Closes** | Gap #10 |
| **Depends on** | PR7 (config), PR8 (URI parser), PR4 (caps) |
| **Touches** | `src/include/io/s3/`, `src/io/s3/`, `CMakeLists.txt`, `vcpkg.json` |
| **Diff size** | ~600 lines |

**Changes**
- New `s3_io_object`: holds bucket, key, endpoint, credentials.
- New `s3_ioctx`: libcurl easy-handle pool doing async HTTP Range GET; SigV4 signing (~200 lines hand-written, or pull `aws-sdk-cpp-core-only`; we lean toward hand-written libcurl to control the dependency footprint).
- `supports_device_read() = false` (S3 over HTTP always lands on host memory).
- `host_read_ranges_async` fans out N parallel Range requests (N sourced from config).
- Factory: `s3://` + `s3_transport != RDMA` → `s3_ioctx`.
- `sirius_engine.cpp` registers `s3_ioctx` on demand (only when config supplies an endpoint).

**Tests**
- `test/cpp/io/test_s3_ioctx.cpp` tag `[s3]` (a local MinIO container spun up by CI):
  - `s3_host_read_returns_exact_bytes`
  - `s3_range_read_correct_offset`
  - `s3_batch_range_reads_parallel`
  - `s3_sigv4_signature_matches_aws_test_vectors`
  - `s3_404_reports_error_via_request_context`
- SQLLogicTest `test/sql/datasource/s3_read.test`: `read_parquet('s3://bucket/file.parquet')` end-to-end.
- **This PR unblocks the motivating query from §1.**

**Verify**
- Bit-exact match against the AWS official [SigV4 test suite](https://docs.aws.amazon.com/general/latest/gr/signature-v4-test-suite.html) canonical request / signature vectors.
- MinIO TPC-H SF1: runtime ≤ 5x the local NVMe number (network-bound).

---

### PR10 — `rdma_s3_ioctx`
| | |
|---|---|
| **Closes** | Gap #12 |
| **Depends on** | PR9 |
| **Touches** | `src/include/io/s3/rdma/`, `src/io/s3/rdma/`, `CMakeLists.txt` |
| **Diff size** | ~700 lines |

**Changes**
- New `rdma_s3_io_object`, `rdma_s3_ioctx`.
- Adds `libibverbs` + `librdmacm` to `pixi.toml`.
- `device_read_async` uses RDMA_WRITE straight into a GPU buffer (MR registration via `ibv_reg_mr` + `IBV_ACCESS_ON_DEMAND` or `IBV_EXP_ACCESS_GPU`).
- `supports_device_read() = true`; `is_device_read_preferred() = true` when a supported cluster is detected.
- Factory: `s3://` + `s3_transport == RDMA` → `rdma_s3_ioctx`.

**Tests**
- `test/cpp/io/test_rdma_s3_ioctx.cpp` tag `[rdma_s3]` (CI requires an RNIC-equipped runner; skip + tag when unavailable):
  - `rdma_s3_device_read_target_gpu_buffer`
  - `rdma_s3_falls_back_to_http_when_rdma_unavailable`
- Integration test against the research team's VAST Data testbed.

**Verify**
- `ibv_devinfo` shows successful MR registration.
- Sequential large-range reads ≥ 2x the PR9 HTTP path throughput.

---

### PR11 — Iceberg delete reads via the factory
| | |
|---|---|
| **Closes** | Gap #13 (D4 subset) |
| **Depends on** | PR3 |
| **Touches** | [`src/op/scan/iceberg_scan_task.cpp:58`](../../src/op/scan/iceberg_scan_task.cpp#L58) / `:121` |
| **Diff size** | ~80 lines |

**Changes**
- `read_positional_delete_file` / `read_equality_delete_file` stop calling `cudf::io::read_parquet(path)` directly. Instead: `auto ds = datasource_factory::create(path, registry, config); cudf::io::parquet_reader_options::builder(cudf::io::source_info{ds.get()})`.
- Delete files now transparently work against S3 once PR9 is in.

**Tests**
- `test/cpp/operator/test_iceberg_delete_via_factory.cpp` tag `[iceberg]`:
  - `positional_delete_on_local_parquet_still_correct` (fixtures already exist under `test/sql/iceberg/`)
  - `equality_delete_on_local_parquet_still_correct`
- SQLLogicTest: all existing Iceberg V2 tests green.

**Verify**
- Zero Iceberg-test regressions.

---

### PR12 — Iceberg Avro manifest reads via the factory
| | |
|---|---|
| **Closes** | Gap #14 (D4 subset) |
| **Depends on** | PR3 |
| **Touches** | [`src/op/scan/iceberg_avro_reader.cpp:552`](../../src/op/scan/iceberg_avro_reader.cpp#L552) / `:631` |
| **Diff size** | ~100 lines |

**Changes**
- Replace `std::ifstream(path)` with an `io_datasource` sourced from the factory.
- The Avro parser needs a sequential byte stream, so either wrap `io_datasource::host_read` in a `std::streambuf` adapter, or do a one-shot `host_read(0, size)` into a memory buffer backing a `std::stringstream` (manifests are typically < 10 MB, so the one-shot variant is acceptable).

**Tests**
- `test/cpp/operator/test_iceberg_avro_via_factory.cpp` tag `[iceberg_avro]`:
  - `avro_manifest_read_local_matches_ifstream`
  - `avro_manifest_read_handles_empty_file_error`
- SQLLogicTest: existing Iceberg scan regressions.

**Verify**
- Iceberg SF1 TPC-H (if enabled) runtime does not regress.

---

### PR13 — Plumb reactor tuning through `sirius_config`
| | |
|---|---|
| **Closes** | Gap C2 |
| **Depends on** | PR7 |
| **Touches** | `src/include/config.hpp`, [`src/include/io/uring/uring_ioctx.hpp`](../../src/include/io/uring/uring_ioctx.hpp), `src/sirius_engine.cpp` |
| **Diff size** | ~60 lines |

**Changes**
- `sirius_config` grows `struct uring_config { size_t host_ring_depth=16; size_t ring_entries=64; size_t n_reactors=4; size_t bounce_slot_size=1<<20; }`.
- `uring_ioctx` constructor takes `uring_config const&`.
- DuckDB SETs: `SET uring_host_ring_depth=...`, etc.
- `sirius_engine.cpp` injects the config when constructing the registry.

**Tests**
- `test/cpp/config/test_uring_config.cpp` tag `[config_uring]`:
  - `custom_ring_depth_applied_to_ioctx`
  - `invalid_values_clamped_or_rejected`
- Micro-benchmark: scan-the-same-parquet wall time at `host_ring_depth=16` vs. `64`.

**Verify**
- TPC-H SF1 with tuned params ≥ default-params run.

---

### PR14 (optional) — Re-enable `KvikIO_REMOTE_SUPPORT`
| | |
|---|---|
| **Closes** | Gap #9 (B2) |
| **Depends on** | PR6, PR9 |
| **Trigger** | Only pursue if we decide to swap PR9's hand-written libcurl/SigV4 stack for `kvikio::RemoteHandle`. Otherwise §9 already resolved this gap by keeping the flag OFF. |
| **Diff size** | ~30 lines (port), but drags libcurl in transitively |

Tentatively **not scheduled**. Revisit only if PR9's SigV4 maintenance burden proves too high.

---

### PR15 — Integration-test scaffolding for all backends
| | |
|---|---|
| **Closes** | Gap #2 (B3 — test infrastructure) |
| **Depends on** | PR6, PR9, PR10 |
| **Touches** | `test/cpp/io/`, `test/sql/datasource/`, `.github/workflows/` |
| **Diff size** | ~300 lines |

**Changes**
- Per-backend fixtures: generate a 1 GB parquet, upload to MinIO, copy to NVMe, etc.
- SQLLogicTest matrix: run the same query through local / gds / s3 / rdma_s3 and assert bit-equal results.
- CI runner matrix: cpu-only, gpu-nvme, gpu-rnic; each tier skips the tags it can't satisfy.

**Tests**
- This PR *is* the test scaffolding — no additional tests required.
- Acceptance: merged CI is green and each backend tag produces at least one log line per run.

**Verify**
- Every backend's Catch2 tag logs in CI after merge.

---

## Dependency graph

```
PR0 (liburing link)
  ↓
PR1 (factory scaffold)
  ↓
  ├→ PR2 (registry in engine)
  │    ↓
  │    PR3 (replace call sites)
  │      ├→ PR5 (batch read)
  │      ├→ PR11 (Iceberg delete)
  │      └→ PR12 (Iceberg Avro)
  │
  ├→ PR4 (cap virtualization) ────┐
  │                                ↓
  └→ PR8 (URI parser) ─────→ PR6 (gds_ioctx)
                       ↘
                        PR7 (config wiring)
                          ↓
                          PR9 (s3_ioctx)
                            ↓
                            PR10 (rdma_s3_ioctx)
                            ↓
                            PR13 (reactor tuning)
                            ↓
                            PR15 (integration matrix)
```

---

## Checkpoints

- **After PR3 lands:** zero user-visible behavior change, but all scan IO flows through the abstraction. First demo-able milestone.
- **After PR6 lands:** GDS path live on local NVMe; GPU-side DMA is observable.
- **After PR9 lands:** §1's motivating query (`read_parquet('s3://...')`) works for the first time.
- **After PR10 lands:** VAST Data POC possible.
- **After PR15 lands:** feature-complete; ready to merge back to `dev`.

---

## Risk register

| Risk | Mitigation |
|---|---|
| PR6 KvikIO cuFile fails to initialize on CI runners | `kvikio::defaults::compat_mode()` auto-detects; `[gds]` tag passes in compat mode too |
| PR9 SigV4 has subtle differences vs. AWS / MinIO / Ceph | Separate fixtures per vendor; AWS official test vectors cover the unit layer |
| PR10 RDMA stack is fragile | HTTP fallback is always available; `SET s3_transport='auto'` downgrades when RDMA detection fails |
| PR3 triggers TPC-H regressions | Every PR runs TPC-H SF1; > +3% triggers a rollback |
| Review bandwidth across many PRs | Each PR stays small (≤ 700 lines), so they can be reviewed in parallel |
