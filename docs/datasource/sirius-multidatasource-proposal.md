# Proposal: Multi-Backend Datasource Layer for Sirius

**Author:** @ranyuan_jin(Yu Teng)
**Status:** Draft
**Created:** 2026-04-08
**Target branch:** `dev`

---

## 1. Motivation

Sirius currently supports three scan backends:

- **Local Parquet / Iceberg** — GPU-accelerated path via `parquet_scan_task` and `iceberg_scan_task`.
- **DuckDB `seq_scan`** — CPU path that delegates to DuckDB's native table functions, used for in-memory tables and `.duckdb` database files.

What is missing is **any direct S3 (and other object store) support**. Today, if a query references an `s3://` URI through `read_parquet` or `iceberg_scan`, Sirius fails immediately — `cudf::io::datasource::create(path)` inside `parquet_scan_task` only accepts local file paths and throws on an S3 URI, before any data is read. There is no CPU fallback path for this case. Users who want GPU-accelerated queries over S3 data must first copy files to local disk, which defeats the purpose of a cloud data lake.

Beyond filling this gap, there is a structural problem worth fixing at the same time. The two `cudf::io::datasource::create()` call sites in `parquet_scan_task.cpp` are the only place where storage backend details leak into scan logic. Everything above those two lines — footer parsing, row group partitioning, byte range scheduling, the `prefetched_data_source` caching layer, GPU decompression, Iceberg delete hooks — is completely storage-agnostic. Yet today those two lines hard-code a single backend. Adding S3 support by branching inside `parquet_scan_task` would spread storage-awareness through scan code that has no business knowing where bytes come from.

A datasource factory solves both problems at once. It moves all backend selection behind a single call, so:

- Scan tasks never see storage details — they call `datasource_factory::create(uri, config)` and get back a `cudf::io::datasource`, regardless of whether the bytes are on local NVMe, NFS, S3, or an RDMA-capable object store.
- Adding a new storage backend in the future means adding one new `cudf::io::datasource` subclass and one dispatch branch in the factory — zero changes to scan tasks, pipelines, or converters.
- The GPU Direct path (GDS for local NVMe, RDMA for object stores) fits naturally as a factory variant, isolated behind the same interface.

```sql
-- Today: throws at initialization, before any data is read
CALL gpu_execution('
    SELECT l_returnflag, l_linestatus, SUM(l_quantity)
    FROM read_parquet(''s3://my-bucket/tpch/lineitem/*.parquet'')
    GROUP BY 1, 2 ORDER BY 1, 2
');

-- With this proposal: works, GPU-accelerated, no local copy required
CALL gpu_execution('
    SELECT l_returnflag, l_linestatus, SUM(l_quantity)
    FROM read_parquet(''s3://my-bucket/tpch/lineitem/*.parquet'')
    GROUP BY 1, 2 ORDER BY 1, 2
');

-- On-prem RDMA object storage (GPU Direct)
SET s3_transport = 'rdma';
CALL gpu_execution('
    SELECT l_returnflag, l_linestatus, SUM(l_quantity)
    FROM read_parquet(''s3://fast-store/warehouse/lineitem/*.parquet'')
    GROUP BY 1, 2 ORDER BY 1, 2
');
```

---

## 2. Where We Are Today

The scan pipeline has a clean two-layer split. The **datasource** layer reads bytes at offsets — it knows nothing about data formats. The **scan task** layer understands Parquet structure: it reads the footer, figures out which byte ranges correspond to the needed columns and row groups, and issues parallel reads through the datasource.

This separation is what makes the proposal cheap to implement. The scan task never calls `open()` or `pread()` directly. It always goes through `cudf::io::datasource`, which is a virtual interface with `host_read()`, `host_read_async()`, and optional `device_read_async()` methods. Swap the datasource implementation, and the scan task reads from a different storage backend without knowing the difference.

The current flow:

```
parquet_scan_task_global_state
  ├── Read footer via datasource  →  parse metadata, partition row groups
  └── Spawn parquet_scan_tasks

parquet_scan_task  (runs on CPU scan executor)
  ├── Create datasource for the file
  ├── Compute byte ranges for selected columns + row groups
  ├── host_read_async() for each range in parallel
  ├── Package into host_parquet_representation
  └── Publish data_batch to shared_data_repository

gpu_pipeline_task  (downstream)
  ├── Wrap cached bytes in prefetched_data_source
  ├── cudf::io::read_parquet() decompresses on GPU
  └── Filter / project → gpu_table_representation
```

All datasource creation goes through 2 call sites in `parquet_scan_task.cpp`. Everything downstream — the `prefetched_data_source` caching layer, `cudf::io::read_parquet()`, the converter registry, the GPU pipeline operators — only sees the `cudf::io::datasource` interface. None of it cares where the bytes came from.

Both `parquet_scan_task` and `iceberg_scan_task` share these same 2 call sites (`iceberg_scan_task_global_state` inherits from `parquet_scan_task_global_state`). Iceberg adds metadata discovery and post-convert delete hooks on top, but the I/O path is identical. This means any datasource implementation that satisfies the interface automatically works for both Parquet and Iceberg scans on the CPU read path with no scan-level code changes.

---

## 3. Design Goals

1. **Broad coverage.** Local FS, network FS, parallel FS, Object Storage.
2. **GPU Direct when available.** Bypass host memory where hardware allows; fall back to CPU reads otherwise.
3. **Minimal integration for the CPU read path.** Replace the 2 datasource creation call sites with a factory; scan tasks, pipelines, and converters are untouched. The GPU Direct path (GDS, RDMA) additionally requires a branch in compute_task() and a new gpu_parquet_representation type, scoped to Section 6
4. **Flat, simple design.** Each backend is a self-contained `cudf::io::datasource` subclass. No deep abstraction stacks.

---

## 4. Storage Classification

We split storage systems into two categories based on their access interface. The key question is whether the system exposes a POSIX file path or an HTTP/HTTPS endpoint — this determines which datasource implementation to use and whether GPU Direct I/O is feasible.

### 4.1 POSIX-Compatible (File Interface)

These systems present a mountable file path and support standard `open`/`pread` semantics, either natively or through a kernel/FUSE mount. For Sirius, the important distinction is whether the underlying block device supports GPU Direct Storage. GDS needs a local or fabric-attached NVMe controller that can DMA straight to GPU memory. Distributed and network filesystems go through the kernel VFS layer, so the CPU always mediates — even if the network transport itself uses RDMA (as in NFS over RDMA or Lustre over InfiniBand).

| Storage | CPU Read | GPU Direct |
|---------|----------|:----------:|
| **Local FS** (ext4, XFS, ZFS) | POSIX pread | GDS via KvikIO (NVMe only) |
| **NFS / NFS over RDMA** | POSIX through VFS | No (kernel-mediated) |
| **NVMe-oF** | POSIX through block layer | GDS possible (fabric NVMe appears local) |
| **Lustre** | POSIX through VFS | No (kernel-mediated) |
| **GPFS / Spectrum Scale** | POSIX through VFS | No (kernel-mediated) |

### 4.2 Object Storage (S3 Interface)

Object storage expose an HTTP/HTTPS API and don't mount as a filesystem. All of them support CPU reads via HTTP Range GET. GPU Direct is only possible when the storage system implements S3 over RDMA — a protocol where the HTTP channel carries control messages (the S3 request plus an RDMA token) and the storage server uses RDMA_WRITE to push data directly into GPU memory registered by the client.

| Storage | CPU Read | GPU Direct |
|---------|----------|:----------:|
| **AWS S3** | HTTP Range GET | No |
| **MinIO** | HTTP Range GET | S3 over RDMA |
| **VAST Data** | HTTP Range GET | S3 over RDMA |
| **Google GCS** | HTTP Range GET | No |
| **Azure Blob** | HTTP Range GET | No |
| **Ceph (RGW)** | HTTP Range GET | No |

---

## 5. Datasource Factory

The central idea is a factory that replaces the 2 hardcoded `cudf::io::datasource::create(file_path)` calls. The factory looks at the URI scheme and the runtime configuration to decide which datasource to create. Local paths get the existing cudf datasource (or the GDS variant on NVMe). S3 URIs get the HTTP/HTTPS or RDMA datasource depending on what the storage system supports.
sirius_config will be extended with an object_store_config struct carrying endpoint, credentials, and transport type (AUTO / HTTP / RDMA). The SET s3_transport DuckDB variable maps to this field via the existing config update path.

```cpp
class datasource_factory {
public:
  static std::unique_ptr<cudf::io::datasource> create(
    std::string const& uri,
    sirius_config const& config);
};
```

Dispatch logic:

```
datasource_factory::create(uri, config)
  ├── "/path" or "file:///path"  →  cudf::io::datasource (or gds_datasource if NVMe + GDS)
  └── "s3://bucket/key"          →  config.s3_transport == RDMA?
                                       Yes → rdma_s3_datasource
                                       No  → s3_datasource
```

The 2 call sites in `parquet_scan_task.cpp` change from `cudf::io::datasource::create(path)` to `datasource_factory::create(path, config)`. Nothing else in the pipeline changes — `host_parquet_representation`, `prefetched_data_source`, the converter registry, Iceberg delete hooks all continue to work because they only depend on the `cudf::io::datasource` interface.

Iceberg scans inherit the same path. `prefetch_iceberg_metadata()` resolves data file and delete file URIs before any datasource is created, so by the time the factory runs it receives fully resolved URIs — local or remote.
Note: Iceberg V2 delete file reads (read_positional_delete_file, read_equality_delete_file) and Avro manifest reads currently use cudf::io::read_parquet and std::ifstream directly and do not go through the factory. These require separate fixes to support S3 URIs and are out of scope for this proposal.

---

## 6. POSIX Datasource

### CPU Read (existing, unchanged)

For any POSIX-mountable path — local disk, NFS, Lustre, GPFS — the existing `cudf::io::datasource::create(path)` already works. It wraps a file descriptor and reads via `pread`. This is the current production path and remains the default.

```
Disk / NFS / Lustre / GPFS / NVMe-oF
  └── pread → pinned host memory → host_parquet_representation → GPU decompression
```

### GPU Direct (new: `gds_datasource`)

For local NVMe drives (and NVMe-oF targets that present as local block devices), we add a `gds_datasource` that wraps KvikIO's `FileHandle`. KvikIO is already in the dependency tree via libcudf 26.04 — it wraps NVIDIA's cuFile API and provides automatic fallback to POSIX when GDS hardware or the cuFile driver is absent.

The datasource implements the `cudf::io::datasource` GPU-read interface:

```cpp
class gds_datasource : public cudf::io::datasource {
  kvikio::FileHandle _handle;
public:
  bool supports_device_read() const override { return _gds_available; }
  bool is_device_read_preferred(size_t size) const override;

  std::future<size_t> device_read_async(
    size_t offset, size_t size, uint8_t* d_dst,
    rmm::cuda_stream_view stream) override;

  // host_read() still works for small metadata reads (footer parsing)
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  size_t size() const override;
};
```

```
Local NVMe / NVMe-oF
  └── gds_datasource
        ├── device_read_async() → cuFile DMA → GPU HBM directly
        └── host_read() → POSIX fallback for metadata (footer)
```

The same `gds_datasource` code works for NVMe-oF (NVMe commands travel over RDMA fabric, but the GDS driver handles it transparently) and for Lustre/GPFS clusters that ship a GDS plugin — the filesystem presents a local path, and KvikIO's `FileHandle` opens it the same way.

### Scan Task Change

Today `parquet_scan_task::compute_task()` always calls `host_read_async()` and writes compressed Parquet bytes into pinned host memory. To take advantage of GPU Direct datasources, it needs to check whether the datasource prefers device reads and branch accordingly:

```
if device read preferred:
  device_read_async() → GPU buffer → gpu_parquet_representation (new)
  → in-place GPU decompression → gpu_table_representation
else:
  host_read_async() → host buffer → host_parquet_representation (existing)
  → prefetched_data_source → cudf::io::read_parquet() → gpu_table_representation
```

The GPU Direct path needs a new `gpu_parquet_representation` data type — compressed Parquet bytes already resident in GPU memory — and a corresponding converter that decompresses in-place on GPU, bypassing the `prefetched_data_source` host-to-device copy step that the existing path uses.

---

## 7. S3 Datasource

### CPU Read (new: `s3_datasource`)

For S3-compatible object stores, we need a new `cudf::io::datasource` implementation that translates `host_read_async(offset, size, dst)` into HTTP Range GET requests. The datasource wraps an object key and a connection to the storage endpoint. It fetches the object size via HTTP HEAD on construction, then serves read calls by issuing `Range: bytes=offset-(offset+size-1)` requests and writing the response body into the caller's pinned host buffer.

```cpp
class s3_datasource : public cudf::io::datasource {
public:
  explicit s3_datasource(std::string const& url);

  size_t size() const override;
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override;
};
```

```
S3 / MinIO / GCS / Azure / Ceph
  └── s3_datasource
        ├── size() → HTTP HEAD
        └── host_read_async() → HTTP Range GET → pinned host memory
```

From the scan task's perspective, this looks identical to the local file datasource — it calls `host_read_async()` and gets bytes back. The parquet row group partitioning, parallel byte-range scheduling, `host_parquet_representation` packaging, and GPU decompression all work unchanged.

Implementation options for the underlying HTTP client:

| Option | Pros | Cons |
|--------|------|------|
| **DuckDB httpfs** | Already in DuckDB, supports S3 | Row-oriented DataChunk path; bypasses Sirius GPU decode pipeline |
| **kvikio::RemoteHandle** | In dep tree but disabled (KvikIO_REMOTE_SUPPORT=OFF); S3/GCS/HTTP/WebHDFS, env-var auth if enabled | Coupled to kvikio threading |
| **AWS SDK C++** | Production-grade, auto-retry, SigV4 | Heavy dependency |
| **libcurl + SigV4** | Lightweight, full control | Manual auth/retry |

Recommended starting point: libcurl + AWS Signature V4. kvikio is already in the dependency tree, but remote I/O support is explicitly disabled in Sirius's vcpkg build (KvikIO_REMOTE_SUPPORT=OFF, CUDF_KVIKIO_REMOTE_IO=OFF). Enabling it would pull in libcurl anyway — so using libcurl directly gives the same HTTP client without the kvikio dependency toggle, avoids the cuFile SDK version constraints that triggered the original patch, and keeps the build self-contained. libcurl is widely available, supports connection pooling and async I/O out of the box, and the SigV4 signing layer is a standalone ~200-line implementation with no additional dependencies.
An alternative worth discussing with the team: enable KvikIO_REMOTE_SUPPORT in the vcpkg portfile and use kvikio::RemoteHandle directly. This would consolidate S3 I/O under one library that already handles credential resolution, retries, and GCS/Azure support — at the cost of pulling in libcurl transitively and maintaining the portfile change.

### GPU Direct (new: `rdma_s3_datasource`)

Some high-performance on-prem object Storage — VAST Data, MinIO with RDMA extensions— support a protocol where the S3 HTTP channel carries control messages but the actual data travels via RDMA. The client registers a GPU memory region with the RDMA NIC and sends the resulting token (containing the buffer address, rkey, and NIC addressing info) as an HTTP header alongside the normal S3 GET request. The storage server then uses RDMA_WRITE to push data directly into that GPU buffer, bypassing the client's CPU and host memory entirely.

```cpp
class rdma_s3_datasource : public cudf::io::datasource {
public:
  bool supports_device_read() const override { return true; }
  bool is_device_read_preferred(size_t size) const override { return true; }

  std::future<size_t> device_read_async(
    size_t offset, size_t size, uint8_t* d_dst,
    rmm::cuda_stream_view stream) override;

  // host_read() for small metadata reads (footer)
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  size_t size() const override;
};
```

The flow for a single GET:

```
Client                                     Storage Server
  │
  ├── Register GPU buffer with NIC → get RDMA token
  ├── HTTP GET + <vendor RDMA token header> ──────→
  │                                          Server parses token
  │                                          RDMA_WRITE → client GPU buffer
  ←══════ data arrives directly in GPU HBM ══════
  ←── HTTP 200 OK ─────────────────────────────
```

When an object is erasure-coded or replicated across multiple storage nodes, the coordinator forwards the client's RDMA token (with adjusted base address offsets) to peer nodes. Each node RDMA_WRITEs its block directly to the corresponding offset in the client's GPU buffer. All transfers happen in parallel, because the RDMA token describes a memory region — not a connection — and InfiniBand DC transport allows any number of server initiators to write concurrently:

```
Client GPU Buffer
┌──────────────────────────────────────────────┐
│  Block 0 (Node A)  │  Block 1 (Node B)  │  Block 2 (Node C)  │
└────────┬───────────┴────────┬────────────┴────────┬───────────┘
         │ RDMA_WRITE         │ RDMA_WRITE          │ RDMA_WRITE
     Node A               Node B                Node C
     (all parallel, same token, different offsets)
```

This is the highest-throughput path in the design. At 100 Gbps InfiniBand, with data landing directly in GPU HBM, the bottleneck shifts from I/O to GPU-side Parquet decompression.

---

## 8. Summary

Four datasource implementations cover the full spectrum, from cloud S3 to on-prem RDMA:

| Datasource | Backends | Data Flow | GPU Direct |
|------------|----------|-----------|:----------:|
| `cudf::io::datasource` (existing) | Local FS, NFS, Lustre, GPFS, NVMe-oF | pread → host → cudaMemcpy → GPU | No |
| `gds_datasource` | Local NVMe, NVMe-oF | cuFile DMA → GPU | **Yes** |
| `s3_datasource` | S3, MinIO, GCS, Azure, Ceph | HTTP Range GET → host → cudaMemcpy → GPU | No |
| `rdma_s3_datasource` | VAST, MinIO (RDMA) | HTTP control + RDMA_WRITE → GPU | **Yes** |

The factory selects the right one at runtime:

```
File path?
  ├── Yes → NVMe + GDS available? → gds_datasource
  │                           No  → cudf::io::datasource
  └── No (s3://) → config.s3_transport == RDMA? → rdma_s3_datasource
                                          No  → s3_datasource
```

### Parquet & Iceberg Adoption

Both Parquet and Iceberg scans flow through the same 2 datasource creation call sites in `parquet_scan_task.cpp`. Replacing those with `datasource_factory::create()` is the only integration change. The Iceberg V2 delete pipeline — positional deletes via binary search + boolean mask, equality deletes via pre-built GPU hash join — operates on `cudf::table` after decompression and works identically regardless of whether the bytes arrived from local disk, S3, or RDMA.

---

## 9. Integration with the `sirius_datasource` Abstraction

An in-flight WIP branch (`aminaramoon/add_sirius_datasource`, rebased into `feature-newdatasourcesupport`) introduces a lower-level IO abstraction layer that this proposal should build on top of rather than parallel to. The two efforts are complementary: that branch answers **how** to read bytes efficiently (io_uring reactors, O_DIRECT, pinned bounce buffers, batched range reads); this proposal answers **where** the bytes live (local FS, NVMe-GDS, S3, S3-over-RDMA) and how URIs route to the right backend.

**Decisions locked in for this integration:**

1. **ioctx registry**. Multi-instance. `datasource_registry` owns a `map<scheme, shared_ptr<sirius_ioctx>>`, constructed once at engine startup in [`src/sirius_engine.cpp`](src/sirius_engine.cpp) and injected into the factory.
2. **GDS backend**. Use **KvikIO in a dedicated `gds_ioctx`**, not an extension of `uring_reactor`. KvikIO is already in the libcudf 26.04 dep tree and provides built-in POSIX fallback.
3. **Configuration wiring**. `sirius_config.object_store_config` lives at the factory level. `SET s3_transport='rdma'` updates this field; the factory consults it during scheme dispatch. Individual `ioctx` instances stay stateless w.r.t. runtime-mutable settings.
4. **Iceberg direct-read paths**. The Iceberg V2 delete file reads (`read_positional_delete_file`, `read_equality_delete_file`) and Avro manifest reads — flagged out-of-scope in §5 — will also route through the factory. The interface is already uniform, so there is no reason to keep a second direct-IO path.

### 9.1 What the abstraction already provides

The `sirius_datasource` branch adds (in [`src/include/io/types.hpp`](src/include/io/types.hpp)):

- **`io_datasource`** — extends `cudf::io::datasource` with a **batched multi-range read API**:

  ```cpp
  virtual std::future<size_t> host_read_ranges_async(
    std::vector<cudf::io::text::byte_range_info> const& ranges,
    std::span<cudf::host_span<std::byte>> dst) = 0;
  ```

- **`sirius_io_object`** — abstract per-file handle carrying file identity (path, cache key, size). Decouples "what file" from "how to read."
- **`sirius_ioctx`** — abstract shared context (thread pools, ring pools, reactor threads). All read APIs are parameterized by `sirius_io_object`, so one `ioctx` can serve many files.
- **`sirius_datasource`** — generic thin delegate: every read forwards to `_io_ctx->op(*_io_object, ...)`. No backend-specific logic lives here; subclassing `io_object` + `ioctx` is enough to add a new backend.

The concrete `uring_ioctx` / `uring_io_object` / `uring_reactor` implementation ships the local-NVMe-focused engine. It uses `O_DIRECT + io_uring` with a 1 MiB pinned-buffer pool; device reads currently land in bounce buffers and are `cudaMemcpyAsync`'d to GPU (not true GDS yet, but the interface is compatible).

### 9.2 Recommended layered architecture

Rather than introducing a parallel `datasource_factory` hierarchy, the factory from §5 becomes a **URI-dispatch layer above `sirius_ioctx`**:

```
datasource_factory::create(uri, registry, config)
        │
        ├─ /path or file://          → uring_ioctx   + uring_io_object(path)
        ├─ file:// + NVMe + GDS      → gds_ioctx     + gds_io_object(path)        [KvikIO / cuFile]
        ├─ s3://                     → s3_ioctx      + s3_io_object(url, creds)
        └─ s3:// + transport=RDMA    → rdma_s3_ioctx + rdma_s3_io_object(url, creds, nic)
```

A small `datasource_registry` maps URI scheme → shared `sirius_ioctx` instance, constructed once at engine startup in [`src/sirius_engine.cpp`](src/sirius_engine.cpp). Each `ioctx` owns its own resources (io_uring pools, libcurl connection pool, RDMA QPs). The factory never allocates per-file state; it just picks the right `ioctx` and asks it to `make_datasource(io_object)`.

This preserves every abstraction the branch already added, and the four datasource implementations from §8 become four `(ioctx, io_object)` pairs instead of four `cudf::io::datasource` subclasses.

### 9.3 Integration points in current code

Three call sites in today's scan path need to flow through the factory:

| Location | Today | After |
| --- | --- | --- |
| [`parquet_scan_task.cpp:265`](src/op/scan/parquet_scan_task.cpp#L265) | `cudf::io::datasource::create(file_path)` (footer read in global state) | `factory::create(file_path, registry, config)` |
| [`parquet_scan_task.cpp:491`](src/op/scan/parquet_scan_task.cpp#L491) | same, in per-task `compute_task()` | same |
| [`sirius_parquet_metadata_scan_operator.cpp:251`](src/op/scan/sirius_parquet_metadata_scan_operator.cpp#L251) | same, in metadata scan op | same |

Iceberg scans inherit automatically because `iceberg_scan_task_global_state` extends `parquet_scan_task_global_state`. The Iceberg V2 delete file reads and Avro manifest reads (originally flagged out-of-scope in §5) should also migrate to the factory — since `sirius_ioctx` already abstracts ioctxs by file, there is no extra cost to routing them uniformly.

### 9.4 Absorbing the batched-range API

`read_range_into_allocation` in [`parquet_scan_task.cpp:634-660`](src/op/scan/parquet_scan_task.cpp#L634-L660) currently issues one `host_read_async()` call per coalesced byte range inside a loop. This loop should be replaced with a single `io_datasource::host_read_ranges_async(byte_ranges, dst_spans)` call:

- **Local path** (uring): the reactor already submits multiple SQEs per ring; batching lets it saturate the submission queue without per-range lock contention.
- **S3 path**: multiple Range GETs can be issued concurrently on a shared HTTP/2 connection, and (longer term) coalesced into a single multipart-range request when the server supports it.
- **RDMA path**: one RDMA registration + one RDMA_WRITE per range with all tokens bundled in a single request, eliminating per-range round-trips.

This promotes the "Multi-range S3 reads" item from §10 (formerly §9) to a core part of the interface: the scan task no longer decides how to batch — the datasource does.

### 9.5 GPU Direct inside this layering

The branch's `uring_reactor` currently implements `device_read_async` via O_DIRECT → pinned bounce buffer → `cudaMemcpyAsync`. This is CPU-mediated; it is **not** true GDS. True NVMe-to-GPU DMA goes through `cuFile`.

**Decision**: introduce a **separate `gds_ioctx`** built on `KvikIO::FileHandle`, rather than bolting a cuFile branch onto the existing uring reactor. Rationale:

- KvikIO is already in the libcudf 26.04 dependency tree — no new third-party dependency.
- KvikIO internally wraps `cuFile` and provides automatic POSIX fallback when GDS hardware or the cuFile driver is unavailable, so a single `gds_ioctx` covers both the fast path (NVMe + cuFile) and the fallback (pread) without extra branching.
- The uring reactor and cuFile have very different submission / completion models; keeping them in one reactor would force awkward unions of state. Two `ioctx`es are simpler to reason about and each can tune its own thread / pool sizing independently.
- The factory already dispatches by scheme + config, so picking `gds_ioctx` vs `uring_ioctx` for a local path is a one-line decision in the factory — no cost at the scan-task layer.

Concrete shape:

- `gds_io_object(path)` — opens the file via `kvikio::FileHandle`, exposes `size()` and the `FileHandle` itself to the reactor.
- `gds_ioctx` — implements the `sirius_ioctx` interface. `device_read_async` → `FileHandle::pread` into GPU memory (cuFile DMA when available). `host_read_*` still supported for footer / metadata reads.
- `supports_device_read()` / `is_device_read_preferred()` report truthfully based on `kvikio::defaults::compat_mode()`. The existing `uring_ioctx` should tighten these to `false` on `device_read_preferred` since its device path is just a bounce-buffer `cudaMemcpyAsync`.

Factory dispatch for local paths becomes: if the path lives on an NVMe block device and KvikIO reports GDS is available → `gds_ioctx`; otherwise → `uring_ioctx`.

### 9.6 Configuration wiring

`sirius_config.object_store_config` (endpoint / credentials / transport mode) lives at the factory level, not inside any `ioctx`. The DuckDB `SET s3_transport='rdma'` path updates this field; the factory reads it when dispatching `s3://` URIs. Individual `ioctx` instances remain configuration-free once constructed — keeping them stateless w.r.t. runtime-mutable settings simplifies lifecycle.

### 9.7 Suggested PR sequence

To land this without a single mega-PR:

1. **PR1** — Introduce `datasource_factory` + `datasource_registry`; populate with only the `uring_ioctx` backend; replace the 3 call sites. Behavior is identical to today, but all IO now flows through the factory.
2. **PR2** — Replace the `read_range_into_allocation` loop with `host_read_ranges_async`. Benchmark locally to confirm io_uring batching wins.
3. **PR3** — Add `gds_io_object` + `gds_ioctx` (KvikIO `FileHandle`). Tighten `is_device_read_preferred()` in `uring_ioctx` to `false`. Factory picks `gds_ioctx` for local paths on NVMe when KvikIO reports GDS available.
4. **PR4** — Add `s3_io_object` + `s3_ioctx` (libcurl + SigV4). Enable `s3://` URIs end-to-end.
5. **PR5** — Add `rdma_s3_io_object` + `rdma_s3_ioctx` for on-prem GPU Direct object stores.
6. **PR6** — Migrate Iceberg delete file and Avro manifest reads to the factory (closes the §5 out-of-scope note).

---

## 10. Future Optimizations

**Multi-range S3 reads.** Today `parquet_scan_task` issues one `host_read_async()` per column chunk byte range. For local files this is cheap (just a `pread`), but for S3 each call becomes a separate HTTP round-trip with TLS handshake and SigV4 signing overhead. Coalescing nearby byte ranges into larger requests — reading a few unused gap bytes rather than paying for an extra round-trip — would help. Longer term, proposing an S3 batch-range-read extension (returning multiple disjoint ranges in a single response) would be the ideal solution for columnar storage workloads where many small, scattered ranges are the norm.

**Pure RDMA S3 control path.** The S3 over RDMA model still uses HTTP for the control plane — each request pays ~50-100 us for the TCP round-trip. For metadata-heavy workloads (many small files, frequent footer reads), this dominates latency. Replacing HTTP with InfiniBand DC Send/Recv for control messages would bring request latency down to ~1-5 us and eliminate the TCP stack entirely. 
