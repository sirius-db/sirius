# Gap Analysis: Multi-Backend Datasource

**Baseline:** `feature-newdatasourcesupport` (rebased aminaramoon/add_sirius_datasource onto `dev`)
**Reference:** [sirius-multidatasource-proposal.md](sirius-multidatasource-proposal.md)
**Date:** 2026-04-17

This document enumerates what is still missing between the current branch state and the proposal's design goals. Each gap has a status, location, and effort tag (**S** = small, **M** = medium, **L** = large). Gaps are grouped by proposal section.

---

## Summary table

| # | Gap | Status | Blocks |
|---|---|---|---|
| 1 | `liburing` not linked in CMake | ❌ likely build-break | all |
| 2 | No tests for the new IO layer | ❌ | all |
| 3 | `datasource_factory` + `datasource_registry` not implemented | ❌ | PR1 |
| 4 | 3 scan call sites still call `cudf::io::datasource::create()` directly | ❌ | PR1 |
| 5 | `sirius_engine.cpp` never constructs any `sirius_ioctx` | ❌ | PR1 |
| 6 | `read_range_into_allocation` still loops `host_read_async` | ❌ | PR2 |
| 7 | `gds_ioctx` + `gds_io_object` (KvikIO) not implemented | ❌ | PR3 |
| 8 | `sirius_datasource::is_device_read_preferred()` hard-coded to `true` | ⚠️ incorrect | PR3 |
| 9 | `KvikIO_REMOTE_SUPPORT=OFF` — KvikIO remote I/O disabled in vcpkg port | ⚠️ | PR3/PR4 |
| 10 | `s3_ioctx` / `s3_io_object` not implemented | ❌ | PR4 |
| 11 | `sirius_config.object_store_config` + `SET s3_transport` not wired | ❌ | PR4 |
| 12 | `rdma_s3_ioctx` / `rdma_s3_io_object` not implemented | ❌ | PR5 |
| 13 | Iceberg V2 delete reads still bypass datasource (`cudf::io::read_parquet(path)`) | ❌ | PR6 |
| 14 | Iceberg Avro manifest reader uses `std::ifstream` directly | ❌ | PR6 |

Legend: ❌ not started · ⚠️ partially present but wrong/insufficient · ✅ complete

---

## 1. What the branch *does* provide (baseline)

These are the only pieces currently on `feature-newdatasourcesupport`:

- [`src/include/io/types.hpp`](../../src/include/io/types.hpp) — abstract interfaces `io_datasource`, `sirius_ioctx`, `sirius_io_object`, `sirius_io_reactor`, plus `request_context` / `device_read_req` / `host_read_req` POD descriptors.
- [`src/include/io/sirius_datasource.hpp`](../../src/include/io/sirius_datasource.hpp) + [`src/io/sirius_datasource.cpp`](../../src/io/sirius_datasource.cpp) — generic thin delegate.
- [`src/include/io/uring/uring_ioctx.hpp`](../../src/include/io/uring/uring_ioctx.hpp) + [`src/io/uring/uring_ioctx.cpp`](../../src/io/uring/uring_ioctx.cpp) — one concrete backend (O_DIRECT + io_uring + 1 MiB bounce buffers).
- [`src/include/io/uring/uring_reactor.hpp`](../../src/include/io/uring/uring_reactor.hpp) + [`src/io/uring/uring_reactor.cpp`](../../src/io/uring/uring_reactor.cpp) — per-reactor submission/completion loop.
- `CMakeLists.txt` compiles the four `.cpp` files (lines 202-204).
- `pixi.toml` declares `liburing >=2,<3` as a dependency.

That's the full delta. Nothing above this layer touches the abstraction yet.

---

## 2. Design-goal gaps

### G1 — Broad coverage (local FS + object stores)

Only the **uring local-POSIX** backend exists. All object-store work is unimplemented.

| Proposal backend (§8) | Status |
|---|---|
| `cudf::io::datasource` (existing) — Local / NFS / Lustre / GPFS | ✅ still works (used directly by scan tasks) |
| `gds_datasource` — Local NVMe via KvikIO | ❌ |
| `s3_datasource` — S3 / MinIO / GCS / Azure / Ceph via HTTP | ❌ |
| `rdma_s3_datasource` — VAST / MinIO-RDMA | ❌ |

**Motivating query from §1 still fails:** `FROM read_parquet('s3://...')` throws at `cudf::io::datasource::create()` before any bytes move.

### G2 — GPU Direct when available

The uring reactor's `device_read_async` path is **O_DIRECT → pinned host bounce → `cudaMemcpyAsync`** (see [`uring_reactor.cpp`](../../src/io/uring/uring_reactor.cpp)). This is CPU-mediated — it is **not** GPUDirect Storage. Yet `sirius_datasource::supports_device_read()` and `is_device_read_preferred(size_t)` both hard-return `true` at [`src/io/sirius_datasource.cpp:43-45`](../../src/io/sirius_datasource.cpp#L43-L45).

**Impact:** if a future caller honors `is_device_read_preferred()` it will believe GDS is active when it isn't.

**Fix path:** introduce `gds_ioctx` (KvikIO) per decision §9.5; tighten `uring_ioctx`'s `is_device_read_preferred()` to `false`.

### G3 — Minimal CPU-path integration (factory over 3 call sites)

**Not integrated.** All three call sites still create a `cudf::io::datasource` directly:

| Site | Current | Needed |
|---|---|---|
| [`parquet_scan_task.cpp:265`](../../src/op/scan/parquet_scan_task.cpp#L265) | `cudf::io::datasource::create(file_path)` | `datasource_factory::create(file_path, registry, config)` |
| [`parquet_scan_task.cpp:491`](../../src/op/scan/parquet_scan_task.cpp#L491) | same | same |
| [`sirius_parquet_metadata_scan_operator.cpp:251`](../../src/op/scan/sirius_parquet_metadata_scan_operator.cpp#L251) | same | same |

`datasource_factory` and `datasource_registry` do not exist anywhere under `src/` (confirmed by grep).

### G4 — Flat, simple design

The branch's four-layer abstraction (`ioctx / io_object / reactor / datasource`) is **more** nested than the proposal's original flat factory-returns-subclass design. §9 resolved this tension by adopting the branch's layering and building the factory *on top of* it. No new gap, but the proposal and the branch disagreed here until §9 reconciled them — future reviewers should read §9 before the earlier sections.

---

## 3. §9 integration-decision gaps

### D1 — ioctx registry in `sirius_engine.cpp`

[`src/sirius_engine.cpp`](../../src/sirius_engine.cpp) has **zero** references to `sirius_ioctx`, `uring_ioctx`, `sirius_datasource`, or `io_datasource`. No registry exists; no `ioctx` is ever constructed.

**Needed:** a `datasource_registry` type holding `map<scheme, shared_ptr<sirius_ioctx>>`, constructed at engine startup and injected into the factory.

### D2 — GDS via KvikIO in dedicated `gds_ioctx`

Not started. KvikIO is present in the dep tree (cudf 26.04) and has a vcpkg port at [`vcpkg_ports/kvikio/portfile.cmake`](../../vcpkg_ports/kvikio/portfile.cmake), but:

- No `src/io/gds/` directory exists.
- No `gds_io_object` / `gds_ioctx` classes.
- Factory dispatch for "NVMe + GDS available → `gds_ioctx`" isn't wired because the factory doesn't exist.

### D3 — Factory-level config wiring

[`src/config.*`](../../src/config.cpp) has no `object_store_config` struct. There is no `SET s3_transport='rdma'` DuckDB variable and no mapping from DuckDB session variables into `sirius_config` for this field. Both ends are missing.

### D4 — Iceberg delete / Avro manifest via factory

Still bypass the datasource layer entirely:

- [`iceberg_scan_task.cpp:58`](../../src/op/scan/iceberg_scan_task.cpp#L58) — `read_positional_delete_file` calls `cudf::io::read_parquet(opts)` directly with a file path (no datasource).
- [`iceberg_scan_task.cpp:121`](../../src/op/scan/iceberg_scan_task.cpp#L121) — `read_equality_delete_file` same pattern.
- [`iceberg_avro_reader.cpp:552`](../../src/op/scan/iceberg_avro_reader.cpp#L552) + [`:631`](../../src/op/scan/iceberg_avro_reader.cpp#L631) — Avro manifest reads use `std::ifstream(path, std::ios::binary)` directly.

These will need to construct an `io_datasource` via the factory and feed bytes to the parquet / Avro parser (for Avro, via an adapter around `cudf::io::datasource::buffer` or a `std::streambuf` wrapper). This is mostly mechanical once the factory exists.

---

## 4. Build / dependency gaps

### B1 — `liburing` not linked

`pixi.toml` declares `liburing` and [`CMakeLists.txt:203-204`](../../CMakeLists.txt#L203-L204) adds `uring_ioctx.cpp` / `uring_reactor.cpp` to the sources, but there is **no** `pkg_check_modules(URING REQUIRED liburing)` and no `target_link_libraries(sirius_extension ... uring)`. `sirius_extension`'s link line is [`CMakeLists.txt:322-323`](../../CMakeLists.txt#L322-L323) and lists only `PkgConfig::NUMA yaml-cpp::yaml-cpp absl::any_invocable`.

Likely consequences:
- `io_uring_queue_init`, `io_uring_submit` etc. fail to resolve at link time, or
- The symbols happen to resolve via transitive deps but the build is fragile.

Either way, an explicit `pkg_check_modules(URING REQUIRED IMPORTED_TARGET liburing)` + link is needed before any scan task actually uses the uring path. This is the *first* gap to close — without it, nothing downstream can be tested.

### B2 — `KvikIO_REMOTE_SUPPORT=OFF`

[`vcpkg_ports/kvikio/portfile.cmake:77`](../../vcpkg_ports/kvikio/portfile.cmake#L77) sets `KvikIO_REMOTE_SUPPORT=OFF`. The proposal §7 discussed this and recommended `libcurl + SigV4` for S3, so this is fine **if** we build S3 on libcurl directly. If we later want to use `kvikio::RemoteHandle` (proposal §7 alternative), this flag has to flip and libcurl needs to be pulled transitively.

### B3 — No tests

Zero test files reference `sirius_datasource`, `sirius_ioctx`, `uring_ioctx`, or `io_datasource` (grep across `test/` returns nothing). Without unit tests covering at minimum (a) basic `host_read` / `host_read_async` round-trip, (b) `host_read_ranges_async` batching, (c) failure paths via `request_context::chunk_failed`, regressions will be invisible during subsequent PRs.

---

## 5. API-adoption gaps

### A1 — `host_read_ranges_async` has no caller

The batched-range API is declared ([`types.hpp`](../../src/include/io/types.hpp)) and implemented in `uring_ioctx`, but grep shows **no caller anywhere in the codebase**. The natural consumer is [`parquet_scan_task::read_range_into_allocation`](../../src/op/scan/parquet_scan_task.cpp#L634-L660), which still loops over `host_read_async`. Until migrated, the batched API exists as dead code.

### A2 — `sirius_io_object` subclassing surface

The interface has `raw_file_cache_id()` and `size()` only. For object-store backends (§D3) we'll need credentials / endpoint / bucket-key on the concrete `s3_io_object`. That's a subclass concern (fine), but the factory will need to parse URIs and construct the right subclass — which means the factory spec should define URI normalization rules (bucket+key vs. full URL, query strings, region detection). Not a code gap today, but a design gap to close **before** PR4.

---

## 6. Correctness gaps in existing code

### C1 — Over-claiming device-read support ([`src/io/sirius_datasource.cpp:43-45`](../../src/io/sirius_datasource.cpp#L43-L45))

```cpp
bool sirius_datasource::supports_device_read() const { return true; }
bool sirius_datasource::is_device_read_preferred(size_t) const { return true; }
```

These must come from the underlying `sirius_ioctx`, not be hard-coded. The uring backend currently has no true GDS path, so it should answer `false` to `is_device_read_preferred`. A future `gds_ioctx` would answer `true` when `kvikio::defaults::compat_mode()` is off.

**Fix:** make `supports_device_read()` / `is_device_read_preferred()` virtual on `sirius_ioctx`, have `sirius_datasource` forward to the ioctx.

### C2 — Hard-coded reactor tuning in [`uring_ioctx.hpp`](../../src/include/io/uring/uring_ioctx.hpp)

Constructor defaults (`host_ring_depth=16`, `ring_entries=64`, `n_reactors=4`, `bounce_slot_size=1 MiB`) are hard-coded. The proposal says the factory / engine owns lifecycle, so these should be plumbed from `sirius_config` at registry construction — not a correctness bug today, but a maintainability gap that will bite once we tune for real workloads.

---

## 7. Recommended order of attack

This re-states §9.7 of the proposal but annotates each PR with its gap-closure scope:

| PR | Gaps closed | Why first/later |
|---|---|---|
| **PR0 (prereq)** | B1, B3 (test skeleton) | Nothing works without linking liburing; CI needs at least smoke-tests on the uring path. |
| **PR1** | #3, #4, #5, D1 | Factory + registry + replace 3 call sites. Behavior identical to today but all IO routes through the abstraction. |
| **PR2** | #6, A1 | Replace `read_range_into_allocation` loop with `host_read_ranges_async`. Benchmark locally. |
| **PR3** | #7, #8, C1, D2 | Add `gds_ioctx` (KvikIO). Fix `is_device_read_preferred` semantics. |
| **PR4** | #10, #11, D3, A2 | Add `s3_ioctx` + config/SET wiring. Unblocks the motivating query from §1. |
| **PR5** | #12 | Add `rdma_s3_ioctx`. |
| **PR6** | #13, #14, D4 | Migrate Iceberg V2 delete + Avro manifest reads through the factory. |
| **Opportunistic** | C2 | Plumb reactor tuning through `sirius_config` once we have benchmark numbers. |

---

## 8. What to verify before declaring "done"

- `CALL gpu_execution('SELECT ... FROM read_parquet(''s3://...'')')` runs end-to-end.
- `SET s3_transport='rdma'` switches to `rdma_s3_ioctx` (observable via metrics / logs).
- Local-file TPC-H numbers on `feature-newdatasourcesupport` match `dev` within noise (no regression from the factory indirection).
- On a GDS-capable host, `gds_ioctx` is selected for NVMe paths, and `is_device_read_preferred()` returns `true`; on a non-GDS host, KvikIO's POSIX fallback kicks in transparently.
- Iceberg V2 tables with positional + equality deletes return correct results when backed by S3.
- `sirius_unittest` has coverage for each `ioctx` backend (host, ranges, device, failure propagation).
