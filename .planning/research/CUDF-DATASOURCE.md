# cuDF parquet I/O — kvikio dependency + extension points

**Researched:** 2026-04-20
**Overall confidence:** HIGH (cuDF and kvikio sources read directly from the pixi env and v26.04.00 GitHub tag)
**Scope:** Answer — which pieces of Sirius's parquet pipeline invoke kvikio, and what is the minimum-surface change to route I/O through cucascade's io_backend_registry instead.

---

## cuDF version in use

| Pin | Source | Value |
|-----|--------|-------|
| `libcudf` | `pixi.toml` lines 47, 59 | `26.04.*` (both cuda12 and cuda13 features) |
| `libkvikio` (transitive) | `pixi.lock` lines 96, 413, 1041, 5079+ | `26.04.00` (built against cuda12_260408_5aba1d10 / cuda13_260408_5aba1d10) |
| Source tag used for spec verification | `github.com/rapidsai/cudf` | `v26.04.00` (verified present in tag list) |
| Local header proof | `.pixi/envs/default/include/cudf/io/datasource.hpp` | Matches `v26.04.00` source |

Sirius also carries `CUDF_VERSION_NUM >= 2604` feature gates in code (`src/op/scan/parquet_scan_task.cpp:51`, `src/op/scan/sirius_parquet_metadata_scan_operator.cpp:26`). The pre-26.04 fallback path is retained for back-porting but not currently executed because `libcudf 26.04.*` is pinned. The plan to bump to 26.06 nightly is already wired in the `nightly-runner` pixi environment (`pixi.toml` lines 89-94); Sirius's hybrid_scan and `parquet_io_utils.hpp` usages assume 26.04+ APIs.

HIGH confidence on the version. The datasource interface, `fetch_footer_to_host`, `fetch_byte_ranges_to_device_async`, and hybrid_scan_reader signatures are verified both against the installed headers and the tagged source on GitHub.

---

## How cuDF uses kvikio (parquet reader internals)

### 1. Where the kvikio dependency is injected

cuDF wraps kvikio inside a concrete implementation of the `cudf::io::datasource` interface. The kvikio linkage lives in **one file**:

- `cpp/src/io/utilities/datasource.cpp` (verified from `github.com/rapidsai/cudf/v26.04.00`)

Relevant classes:

| cuDF class | kvikio object held | Where it lives |
|------------|--------------------|----------------|
| `kvikio_source<HandleT>` (CRTP base, file-scope anonymous namespace) | `HandleT _kvikio_handle;` | `datasource.cpp` |
| `file_source : kvikio_source<kvikio::FileHandle>` | `kvikio::FileHandle` | `datasource.cpp` |
| `memory_mapped_source : kvikio_source<kvikio::MmapHandle>` | `kvikio::MmapHandle` | `datasource.cpp` |
| `remote_file_source : kvikio_source<kvikio::RemoteHandle>` (when `CUDF_KVIKIO_REMOTE_IO` is set) | `kvikio::RemoteHandle` | `datasource.cpp` |
| `device_buffer_source`, `host_buffer_source`, `user_datasource_wrapper` | **NO kvikio** | `datasource.cpp` |

`datasource::create(std::string const& filepath, ...)` dispatches:

1. If `LIBCUDF_MMAP_ENABLED=ON` → `memory_mapped_source` (kvikio MmapHandle)
2. If URL scheme (`s3://`, `http://`, ...) → `remote_file_source` (kvikio RemoteHandle)
3. Otherwise → `file_source` (kvikio FileHandle) — **the default local-filesystem path**

All three paths open a kvikio handle inside a single CUDA context. This is the multi-GPU failure mode: once a `kvikio::FileHandle` has performed even one `device_read_async`, the registered stream, bounce buffers, and cuFile driver state are tied to whichever CUDA context was active at creation. A second task dispatched to a different device's CUDA context cannot safely share that handle.

### 2. What happens inside `file_source`

From `datasource.cpp` (v26.04.00):

```cpp
class kvikio_source : public datasource {
  // ...
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override {
    return host_read_async(offset, size, dst).get();
  }
  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override {
    auto const read_size = std::min(size, this->size() - offset);
    return _kvikio_handle.pread(dst, read_size, offset);
  }
  [[nodiscard]] bool supports_device_read() const override { return true; }
  [[nodiscard]] bool is_device_read_preferred(size_t size) const override {
    return supports_device_read();
  }
  std::future<size_t> device_read_async(size_t offset, size_t size,
                                        uint8_t* dst, rmm::cuda_stream_view stream) override {
    stream.synchronize();
    return _kvikio_handle.pread(dst, read_size, offset);
  }
  // ...
};

class file_source : public kvikio_source<kvikio::FileHandle> {
  // Overrides device_read_async to pass task_size, gds_threshold, sync_default_stream=false
  std::future<size_t> device_read_async(...) override {
    stream.synchronize();
    return _kvikio_handle.pread(dst, read_size, offset,
                                kvikio::defaults::task_size(),
                                kvikio::defaults::gds_threshold(),
                                false);
  }
};
```

Three key observations from source:

1. **Every call** to `host_read` / `host_read_async` / `device_read` / `device_read_async` on cuDF's built-in file source goes through `kvikio::FileHandle::pread`. There is no non-kvikio code path inside `file_source`.
2. `file_source::device_read_async` calls `stream.synchronize()` before issuing the kvikio pread. The returned future is a **kvikio future**, not stream-ordered.
3. `supports_device_read()` always returns `true` for `file_source`, which steers cuDF's parquet reader toward the device-read path (see "How cuDF chooses the read path" below).

### 3. How cuDF's parquet reader selects host vs device reads

Two call sites drive the entire parquet I/O: both use the same predicate `is_device_read_preferred(io_size)`.

**Call site A — classic `cudf::io::read_parquet` / `parquet_reader`**
`cpp/src/io/parquet/reader_impl_preprocess_utils.cu::read_column_chunks_async` (lines 217-231):

```cpp
if (source->is_device_read_preferred(io_size)) {
  auto fut = source->device_read_async(io_offset, io_size, dest, stream);
  read_tasks.emplace_back(std::move(fut));
} else {
  read_tasks.emplace_back(std::async(std::launch::deferred, [source, ...]() {
    auto const read_buffer = source.get().host_read(io_offset, io_size);
    cudf::detail::cuda_memcpy_async(
      cudf::device_span<uint8_t>{static_cast<uint8_t*>(dest), io_size},
      cudf::host_span<uint8_t const>{read_buffer->data(), io_size},
      stream);
    return io_size;
  }));
}
```

**Call site B — hybrid_scan / `fetch_byte_ranges_to_device_async`**
`cpp/src/io/parquet/io_utils/parquet_io_utils.cpp::fetch_byte_ranges_to_device_async` (lines ~126-150):

```cpp
if (datasource.supports_device_read() and datasource.is_device_read_preferred(io_size)) {
  device_read_tasks.emplace_back(
    datasource.device_read_async(io_offset, io_size, dest, stream));
} else {
  host_read_tasks.emplace_back(std::async(std::launch::deferred, [...]() {
    auto host_buffer = datasource.host_read(io_offset, io_size);
    cudf::detail::cuda_memcpy_async(
      cudf::device_span<uint8_t>{dest, io_size},
      cudf::host_span<uint8_t const>{host_buffer->data(), io_size}, stream);
    return io_size;
  }));
}
```

**Implication for us:** both paths already have a clean host-read-plus-memcpy fallback. A custom `cudf::io::datasource` that reports `supports_device_read() == false` (or `is_device_read_preferred() == false`) will steer **all** parquet-reader I/O to `host_read` — which is exactly where cucascade's disk_io_backend wants to live.

### 4. The `set_up_kvikio` hook

`cudf::io::kvikio_integration::set_up_kvikio()` is called from inside every kvikio_source's constructor (`kvikio_initializer _;` member, constructed before `_kvikio_handle`). Source (verified at `cpp/src/io/utilities/config_utils.cpp`, v26.04.00):

```cpp
void set_up_kvikio() {
  static std::once_flag flag{};
  std::call_once(flag, [] {
    cudaFree(nullptr);  // workaround for rapidsai/cudf#14140
    auto const compat_mode = kvikio::getenv_or("KVIKIO_COMPAT_MODE", kvikio::CompatMode::ON);
    kvikio::defaults::set_compat_mode(compat_mode);
    auto const nthreads = getenv_or<unsigned int>("KVIKIO_NTHREADS", 4u);
    kvikio::defaults::set_thread_pool_nthreads(nthreads);
  });
}
```

Two noteworthy facts from this:

1. **cuDF 26.04 defaults `KVIKIO_COMPAT_MODE` to `ON`**, meaning cuDF forces kvikio into POSIX compatibility mode by default. GDS / cuFile is **not** used unless the user explicitly sets `KVIKIO_COMPAT_MODE=OFF` or `KVIKIO_COMPAT_MODE=AUTO`. This means Sirius today is almost certainly not using GDS, even though cuDF calls `device_read_async`.
2. The `cudaFree(nullptr)` workaround initializes the current CUDA context. This is what ties the kvikio state to whatever GPU was current at the moment the first `datasource::create(...)` ran in the process. Source: `#14140` comment in `set_up_kvikio`.

### 5. Summary of kvikio invocation points

In cuDF 26.04, **every** `cudf::io::datasource` created from a filepath is kvikio-backed. The paths into kvikio are:

| Reader path | Entry point | Datasource call | kvikio call |
|-------------|-------------|-----------------|-------------|
| `cudf::io::read_parquet` (classic, includes nested types, chunked reader) | `reader_impl_preprocess_utils.cu:read_column_chunks_async` | `device_read_async` or `host_read` | `FileHandle::pread` |
| `hybrid_scan_reader` footer/page-index reads | `parquet::fetch_footer_to_host` / `fetch_page_index_to_host` | `host_read` (sync, into `datasource::buffer`) | `FileHandle::pread` |
| `hybrid_scan_reader` column chunk / secondary filter reads | `parquet::fetch_byte_ranges_to_device_async` | `device_read_async` or `host_read` | `FileHandle::pread` |

`hybrid_scan_reader` itself **never** calls the datasource directly — it only consumes buffers that the caller fetches via `fetch_footer_to_host` / `fetch_page_index_to_host` / `fetch_byte_ranges_to_device_async`. See `cpp/src/io/parquet/experimental/hybrid_scan.cpp` (v26.04.00): the class constructor takes `cudf::host_span<uint8_t const> footer_bytes` and `materialize_*_columns` accept `cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data`. No datasource reference is stored inside the reader impl.

**Conclusion:** kvikio is only reached **through** a `cudf::io::datasource`. Replacing the datasource is sufficient to fully eliminate kvikio from Sirius's parquet pipeline.

---

## `cudf::io::datasource` interface (exact signatures)

Verified from `.pixi/envs/default/include/cudf/io/datasource.hpp` (matches v26.04.00 source):

```cpp
class datasource {
 public:
  class buffer {
   public:
    [[nodiscard]] virtual size_t size() const = 0;
    [[nodiscard]] virtual uint8_t const* data() const = 0;
    virtual ~buffer() = default;
    operator cudf::host_span<uint8_t const>() const;
    template <typename Container>
    static std::unique_ptr<buffer> create(Container&& data_owner);
  };

  // Factories (all wrap kvikio for filepath variant)
  static std::unique_ptr<datasource> create(std::string const& filepath,
                                            size_t offset            = 0,
                                            size_t max_size_estimate = 0);
  static std::unique_ptr<datasource> create(cudf::host_span<std::byte const> buffer);
  static std::unique_ptr<datasource> create(cudf::device_span<std::byte const> buffer);
  static std::unique_ptr<datasource> create(datasource* source);   // user-implemented wrapper
  template <typename T>
  static std::vector<std::unique_ptr<datasource>> create(std::vector<T> const& args);

  virtual ~datasource() = default;

  // --- HOST READS (mandatory; must override both) ---
  virtual std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size) = 0;
  virtual size_t host_read(size_t offset, size_t size, uint8_t* dst) = 0;

  // --- HOST READS (optional, default impls wrap sync host_read in std::async deferred) ---
  virtual std::future<std::unique_ptr<datasource::buffer>> host_read_async(size_t offset,
                                                                            size_t size);
  virtual std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst);

  // --- DEVICE READS (optional; default supports_device_read() returns false) ---
  [[nodiscard]] virtual bool supports_device_read() const { return false; }
  [[nodiscard]] virtual bool is_device_read_preferred(size_t size) const {
    return supports_device_read();
  }
  virtual std::unique_ptr<datasource::buffer> device_read(size_t offset, size_t size,
                                                          rmm::cuda_stream_view stream) {
    CUDF_FAIL("datasource classes that support device_read must override it.");
  }
  virtual size_t device_read(size_t offset, size_t size, uint8_t* dst,
                             rmm::cuda_stream_view stream) {
    CUDF_FAIL("datasource classes that support device_read must override it.");
  }
  virtual std::future<size_t> device_read_async(size_t offset, size_t size, uint8_t* dst,
                                                rmm::cuda_stream_view stream) {
    CUDF_FAIL("datasource classes that support device_read_async must override it.");
  }

  // --- METADATA ---
  [[nodiscard]] virtual size_t size() const = 0;
  [[nodiscard]] virtual bool is_empty() const { return size() == 0; }
};
```

### What cuDF actually calls on the datasource

For a custom datasource that reports `supports_device_read() == false`, the parquet reader will only ever call:

| Method | When |
|--------|------|
| `host_read(offset, size)` → `unique_ptr<buffer>` | footer reads (`fetch_footer_to_host`), page-index reads (`fetch_page_index_to_host`), column chunk fallback path |
| `host_read(offset, size, dst)` | `kvikio_source` uses this internally but the parquet reader path only uses the returning variant. Safe to implement by wrapping `host_read_async(...).get()` if desired. |
| `host_read_async(offset, size, dst)` | Used by some datasource wrappers (see user_datasource_wrapper forwarding); default deferred-async impl is fine if sync is cheap. |
| `size()` | Called by `fetch_footer_to_host` to locate the tail. |

If we return `supports_device_read() == true` but `is_device_read_preferred(size) == false`, the parquet reader will still fall through to `host_read` for column chunks. See `reader_impl_preprocess_utils.cu:217`: the predicate is **only** `is_device_read_preferred(io_size)`, not `supports_device_read()`. But `fetch_byte_ranges_to_device_async` at `parquet_io_utils.cpp:126` uses **both** conditions (`supports_device_read() AND is_device_read_preferred()`), so returning `false` from either is sufficient for the hybrid_scan path.

**Minimum override surface for a kvikio-free custom datasource:**
- `size_t size() const override` — return file size
- `std::unique_ptr<buffer> host_read(size_t, size_t) override` — allocating host read
- `size_t host_read(size_t, size_t, uint8_t*) override` — copy-into-buffer host read
- (optional) `std::future<size_t> host_read_async(...)` — if the backend has native async; otherwise the base-class deferred wrapper is correct

Anything else (`device_read*`, `supports_device_read`) can be left at the default (false / CUDF_FAIL) because cuDF never calls device_read unless the datasource opts in.

---

## Environment variables / options that affect kvikio usage

Verified against `kvikio/defaults.hpp` (26.04.00, in `.pixi/envs/default/include/kvikio/defaults.hpp`) and `cudf/io/config_utils.cpp` (v26.04.00):

| Env var | Owner | Default (cuDF 26.04) | Meaning | Kill-switch potential |
|---------|-------|---------------------|---------|-----------------------|
| `KVIKIO_COMPAT_MODE` | kvikio, set by cuDF | `ON` (forced by `set_up_kvikio`) | `ON` = POSIX `pread`, no cuFile; `OFF` = cuFile/GDS; `AUTO` = infer | Already `ON` by default → GDS is not in use. This does NOT disable kvikio, only its GDS path. |
| `KVIKIO_NTHREADS` | kvikio, set by cuDF | `4` (forced by `set_up_kvikio`) | Thread pool size inside kvikio | Reduces concurrency but does not decouple from single-context kvikio state |
| `KVIKIO_TASK_SIZE` | kvikio | 4 MiB | Parallel-read chunk size | None |
| `KVIKIO_GDS_THRESHOLD` | kvikio | 1 MiB | Below this size, use POSIX even when compat=OFF | None (POSIX path still kvikio-owned) |
| `KVIKIO_BOUNCE_BUFFER_SIZE` | kvikio | 16 MiB | Host bounce buffer for GDS | None |
| `LIBCUDF_MMAP_ENABLED` | cuDF | `OFF` | `ON` uses `kvikio::MmapHandle` instead of `FileHandle` | Doesn't help: still kvikio-owned |
| `LIBCUDF_IO_REROUTE_LOCAL_DIR_PATTERN` / `..._REMOTE_DIR_PATTERN` | cuDF | unset | Rewrites local paths to remote URLs → kvikio RemoteHandle | Unrelated, pushes us further into kvikio |
| `LIBCUDF_NVCOMP_POLICY` | cuDF | `STABLE` | nvCOMP decompression kernel choice | Unrelated to I/O |
| `LIBCUDF_INTEGRATED_MEMORY_OPTIMIZATION` | cuDF | `AUTO` | Unified memory opt on integrated GPUs (Grace-Hopper) | Unrelated to I/O |
| `LIBCUDF_CUFILE_POLICY` | (historical) | — | **Does not exist in cuDF 26.04.** The historical knob was renamed/folded into `KVIKIO_COMPAT_MODE`. No references in v26.04.00 source. | N/A |
| `CUDF_HOST_MEMORY_RESOURCE` | (historical) | — | **Does not exist in cuDF 26.04.** No references in v26.04.00 source. cuDF 26.04 uses the pinned/pageable host resource selected via `cudf::set_host_memory_resource` and nothing environment-driven. | N/A |

**Critical negative finding (HIGH confidence):** cuDF 26.04 has **no public API or environment variable that replaces the default kvikio-backed file_source with a non-kvikio backend**. The only way to bypass kvikio is to construct an explicit `cudf::io::datasource*` and wrap it in `cudf::io::source_info{datasource_ptr}`. Confirmed by:
- `datasource.cpp` (v26.04.00) — `datasource::create(std::string const&, ...)` hard-codes `file_source` / `memory_mapped_source` / `remote_file_source` dispatch.
- `config_utils.cpp` (v26.04.00) — Only sets kvikio compat mode and thread count, no global off switch.
- Searching the v26.04.00 tree for `LIBCUDF_CUFILE_POLICY` and `CUDF_HOST_MEMORY_RESOURCE` yields zero hits (tree fetched via GitHub API).

Therefore, **option 1 (custom datasource) is the only viable path** for cuDF 26.04. Env-var mitigation is ruled out.

---

## `hybrid_scan_reader` — origin and kvikio coupling

### Origin

`cudf::io::parquet::experimental::hybrid_scan_reader` is a **public** (but `experimental` namespace) cuDF API. It is declared in a shipped header:

- Header: `cudf/io/experimental/hybrid_scan.hpp` (verified at `.pixi/envs/default/include/cudf/io/experimental/hybrid_scan.hpp`)
- Source: `cpp/src/io/parquet/experimental/hybrid_scan.cpp` (v26.04.00)
- Impl: `cpp/src/io/parquet/experimental/hybrid_scan_impl.{cpp,hpp}` (not a public header)

The class is stable enough for cuDF to ship an example (`cpp/examples/hybrid_scan_io/`) and benchmarks (`cpp/benchmarks/io/parquet/experimental/hybrid_scan/`). Sirius already treats it as a supported API (`host_parquet_representation.cpp:47`, `host_parquet_representation_converters.cpp:171`). The `experimental` namespace means cuDF reserves the right to change the interface between minor versions; concretely, Sirius should track cuDF changelog when bumping to 26.06+.

### Kvikio coupling

**None, directly.** `hybrid_scan_reader` takes:

- Constructor: `cudf::host_span<uint8_t const> footer_bytes` (or pre-populated `FileMetaData`)
- `setup_page_index(cudf::host_span<uint8_t const> page_index_bytes)` — host buffer in
- `materialize_*_columns(... cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data, ...)` — device buffers in

Every byte that `hybrid_scan_reader` decodes was pre-fetched by the caller. The I/O happens outside the reader via:

- `cudf::io::parquet::fetch_footer_to_host(datasource&)` → `datasource.host_read(...)`
- `cudf::io::parquet::fetch_page_index_to_host(datasource&, byte_range_info)` → `datasource.host_read(...)`
- `cudf::io::parquet::fetch_byte_ranges_to_device_async(datasource&, byte_ranges, stream, mr)` → `datasource.device_read_async` or `datasource.host_read` + `cuda_memcpy_async`

All three utilities live in `cudf/io/parquet_io_utils.hpp` (public) and are implemented in `cpp/src/io/parquet/io_utils/parquet_io_utils.cpp`. None of them directly references kvikio; they dispatch through the datasource vtable. This means **swapping the datasource is sufficient to redirect both the classic and hybrid paths** — no need to modify hybrid_scan_reader itself.

### Current Sirius usage of hybrid_scan_reader

The reader does four things for Sirius:

1. **Parse the footer** (`parquet_metadata()`) — `src/op/scan/parquet_scan_task.cpp:355`, `src/op/scan/sirius_parquet_metadata_scan_operator.cpp:264`. Zero I/O once the footer buffer is passed in.
2. **Row group pruning with stats** (`filter_row_groups_with_stats`) — `parquet_scan_task.cpp:467-468`, `sirius_parquet_metadata_scan_operator.cpp:285`. Zero I/O — purely filter evaluation on footer stats.
3. **Report column-chunk byte ranges** (`all_column_chunks_byte_ranges`) — `parquet_scan_task.cpp:731-732`. Zero I/O — pure footer arithmetic.
4. **Re-create a reader for a future materialization pass** (`std::make_unique<hybrid_scan_reader>(parquet_metadata(), options)`) — `host_parquet_representation.cpp:47,69`, `host_parquet_representation_converters.cpp:171`. Zero I/O.

Sirius currently does **not** call `materialize_filter_columns` / `materialize_payload_columns` / `materialize_all_columns`. The GPU materialization step goes back through `cudf::io::read_parquet(opts, stream, mr_ref)` in `host_parquet_representation_converters.cpp:92`, feeding the packed column chunk bytes as a `prefetched_data_source` (already a custom datasource). The hybrid_scan API is only used for metadata and byte-range extraction.

---

## Sirius call-sites and required changes

Every place that can instantiate a kvikio-backed datasource today. Grouped by the two kinds of work needed.

### Group A — places that currently call `cudf::io::datasource::create(filepath)`

Each of these constructs a `file_source` (kvikio) today. These need to be replaced with a cucascade-backed datasource factory.

| File | Line | Current call | Notes |
|------|------|--------------|-------|
| `src/op/scan/parquet_scan_task.cpp` | 312 | `cudf::io::datasource::create(file_path)` | Per-file footer read at scan-plan time (`initialize_from_files`). Called once per file. |
| `src/op/scan/parquet_scan_task.cpp` | 699 | `cudf::io::datasource::create(g_state.get_file_path(...))` | Per-task datasource. This becomes `_datasource` on the task and is also used as `_fallback_datasource` for `prefetched_data_source`. |
| `src/op/scan/sirius_parquet_metadata_scan_operator.cpp` | 251 | `cudf::io::datasource::create(file_path)` | Metadata scan operator — per-file footer read. |

Additionally:

- `src/op/scan/parquet_scan_task.cpp:863` uses the datasource via `_datasource->host_read_async(current_offset, bytes_to_read, buffer_ptr)`. That call is the hot host-read path that actually streams column-chunk bytes into the host allocation. **This is the highest-bandwidth kvikio invocation in Sirius today.** It feeds the `fixed_size_host_memory_resource` allocation that is later consumed by `prefetched_data_source`.
- `src/data/host_parquet_representation_converters.cpp:83` constructs a `prefetched_data_source` with `host_src.get_fallback_datasource()` as the fallback — the fallback is the same kvikio-backed datasource created at scan-plan time. If any row-group byte range escapes the cache (see "Pitfalls"), cuDF reads it via kvikio.
- `src/op/scan/iceberg_scan_task.cpp:57-58, 120-121` constructs a **filepath-based `source_info`** and calls `cudf::io::read_parquet(opts, stream)`. This is the iceberg delete-file read path. Internally, `cudf::io::read_parquet` calls `datasource::create(filepath)` → `file_source` → kvikio. This is a **second** kvikio entry point that does not go through the prefetched-data-source machinery. It needs the same treatment.

### Group B — places that call `cudf::io::read_parquet` with a `source_info{cudf::io::datasource*}`

These are already using a custom datasource; the question is only whether the underlying pre-fetch and fallback paths are kvikio-free.

| File | Line | Current call | Status |
|------|------|--------------|--------|
| `src/data/host_parquet_representation_converters.cpp` | 87, 92 | `opts.set_source(cudf::io::source_info{data_source.get()})` then `cudf::io::read_parquet(opts, stream, mr_ref)` — `data_source` is `sirius::op::scan::prefetched_data_source` | Custom datasource, but its **fallback** field (`prefetched_data_source::fallback_`) is still a kvikio-backed `cudf::io::datasource`. Needs fallback swap. |
| `src/op/scan/sirius_gpu_parquet_scan_operator.cpp` | 171, 175 | `opts.set_source(cudf::io::source_info{datasource.get()})` then `cudf::io::read_parquet(opts, stream)` | `datasource` here is `scan_data->datasource` which is the `cudf::io::datasource` owned by `parquet_scan_data` / created in `sirius_parquet_metadata_scan_operator.cpp:251`. Today it is a kvikio `file_source`. Needs swap. |

### Group C — options-building sites that are unaffected once the datasource is swapped

These places never create a datasource directly — they only build options. Included for completeness; no change required as long as the `source_info` is set to a custom datasource by the time `read_parquet` runs.

| File | Line | Call |
|------|------|------|
| `src/op/scan/parquet_scan_task.cpp` | 333 | `cudf::io::parquet_reader_options::builder().build()` |
| `src/op/scan/sirius_parquet_metadata_scan_operator.cpp` | 194-195 | `std::make_shared<cudf::io::parquet_reader_options>(cudf::io::parquet_reader_options::builder().build())` |
| `src/op/scan/iceberg_scan_task.cpp` | 57, 120 | `cudf::io::parquet_reader_options::builder(cudf::io::source_info{delete_file_path}).build()` — **but** this implicitly passes a filepath, which is kvikio. Either switch to a custom datasource (Group A) or migrate the delete-file loader to the cucascade path explicitly. |

### Summary table — one row per thing that touches kvikio today

| Change | File:line | Fix |
|--------|-----------|-----|
| Swap default datasource factory for parquet scan | `src/op/scan/parquet_scan_task.cpp:312` | Replace `cudf::io::datasource::create(file_path)` with a new `sirius::io::make_cucascade_datasource(file_path)` helper |
| Same | `src/op/scan/parquet_scan_task.cpp:699` | Same as above |
| Same | `src/op/scan/sirius_parquet_metadata_scan_operator.cpp:251` | Same as above |
| Remove kvikio from per-task host read path | `src/op/scan/parquet_scan_task.cpp:863` | Replace `_datasource->host_read_async(...)` with a call through cucascade's `idisk_io_backend` (routed by `io_backend_registry`). Alternatively, the new `sirius::io::make_cucascade_datasource` already returns a datasource whose `host_read_async` is cucascade-backed — in that case this line is unchanged in shape, only the underlying implementation changes. |
| Iceberg delete-file read | `src/op/scan/iceberg_scan_task.cpp:57-58` | Build a custom datasource via `sirius::io::make_cucascade_datasource` and use `source_info{ds.get()}` instead of `source_info{delete_file_path}` |
| Iceberg equality-delete read | `src/op/scan/iceberg_scan_task.cpp:120-121` | Same treatment as above |
| Fallback inside prefetched_data_source | `src/data/host_parquet_representation_converters.cpp:82-83` (and the `_fallback_datasource` field populated from `parquet_scan_task.cpp:769`) | Ensure `prefetched_data_source`'s `fallback_source` is also a cucascade-backed datasource, not a kvikio one |

All other call-sites (options builders, hybrid_scan_reader usage) are transitively fixed by swapping the datasource.

---

## Migration options (custom datasource vs env-var disable vs hybrid)

### Option 1 — Custom `cudf::io::datasource` wrapping cucascade's `idisk_io_backend`

**Viability:** HIGH. This is the only approach that fully decouples from kvikio in cuDF 26.04.

**Shape of the implementation (pseudocode):**

```cpp
namespace sirius::io {

class cucascade_datasource : public cudf::io::datasource {
 public:
  cucascade_datasource(std::string filepath,
                       cucascade::io_backend_registry& registry)
    : _size(/* stat() or backend->open()->size() */),
      _file_handle(registry.get_backend_for(filepath)->open(filepath)) {}

  // --- HOST READS ---
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override {
    std::vector<uint8_t> v(std::min(size, _size - offset));
    _file_handle->pread(v.data(), v.size(), offset).get();
    return buffer::create(std::move(v));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override {
    auto const read_size = std::min(size, _size - offset);
    return _file_handle->pread(dst, read_size, offset).get();
  }

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override {
    auto const read_size = std::min(size, _size - offset);
    return _file_handle->pread(dst, read_size, offset);   // cucascade returns std::future<size_t>
  }

  // --- DEVICE READS: intentionally disabled ---
  [[nodiscard]] bool supports_device_read() const override { return false; }
  // (device_read / device_read_async inherit base-class CUDF_FAIL impl;
  //  cuDF will never call them because supports_device_read() is false.)

  [[nodiscard]] size_t size() const override { return _size; }

 private:
  size_t _size;
  std::unique_ptr<cucascade::disk_file_handle> _file_handle;  // name approximate; PR96 surface
};

inline std::unique_ptr<cudf::io::datasource>
make_cucascade_datasource(std::string const& filepath) {
  return std::make_unique<cucascade_datasource>(
    filepath, cucascade::io_backend_registry::instance());
}

}  // namespace sirius::io
```

Return `supports_device_read() == false` so cuDF uses the host_read + `cuda_memcpy_async` path. This keeps stream-semantics clean: the memcpy happens on the user's explicit stream (already in line with the Sirius "no cuda_stream_default" rule), the actual disk I/O runs in cucascade's thread pool, and there's no implicit CUDA context coupling.

**Why not offer `device_read`?** Three reasons:
1. The multi-GPU safety story is simpler when I/O is always host-staged. Disk → pinned host → `cudaMemcpyAsync(dest_device, src_host, size, stream)` works regardless of which GPU owns the stream.
2. cucascade's `disk_io_backend` (per PR #96) is a disk-side abstraction, not a GPUDirect Storage abstraction. Routing it through `device_read` would require cucascade to also own device-side bounce buffers per GPU, which is outside the PR #96 surface.
3. Sirius already has a cache layer (`prefetched_data_source`) that owns packed host buffers and performs its own `cudaMemcpyBatchAsync` (see `prefetched_data_source.cpp:152`). Duplicating that work inside the datasource would make the stream ordering story harder.

**Required downstream change:** `prefetched_data_source::fallback_` must be a `cucascade_datasource`, not a `cudf::io::datasource` from `create(filepath)`. The type is already `std::shared_ptr<cudf::io::datasource>` so the polymorphism works unchanged.

### Option 2 — Disable kvikio globally via env var

**Viability:** NOT VIABLE. `KVIKIO_COMPAT_MODE=ON` is already the cuDF 26.04 default; it only disables GDS/cuFile, it does **not** remove kvikio::FileHandle from the code path. There is no env var or `parquet_reader_options` field that swaps the underlying read implementation. Verified by reading `datasource.cpp` and `config_utils.cpp` at v26.04.00 — the only env knobs (see section "Environment variables / options") affect kvikio's behavior, not whether it is used.

Sub-option: `LIBCUDF_MMAP_ENABLED=ON` switches to `kvikio::MmapHandle` — still kvikio. No help.

### Option 3 — Hybrid: keep kvikio for footer/metadata reads, cucascade for bulk row-group bytes

**Viability:** Possible but not recommended.

Today Sirius's `prefetched_data_source` already operates in a "hybrid" mode: it serves cached ranges from its own host buffers, and falls through to a kvikio-backed fallback for anything not cached. Keeping the fallback kvikio-based works **if** the cache is always complete — which Sirius tries to guarantee by pre-fetching the header + all column chunk ranges + the footer (`parquet_scan_task.cpp:735-740`).

However, (a) the metadata-pass (`parquet_scan_task.cpp:312`, `sirius_parquet_metadata_scan_operator.cpp:251`) still creates a raw `datasource::create(filepath)` for the footer read, which opens a `kvikio::FileHandle`; and (b) the hot `host_read_async` loop at `parquet_scan_task.cpp:863` is the **primary** source of kvikio I/O volume, not a fallback.

Therefore a hybrid option still leaves kvikio anchored to the CUDA context that opened the handle in scenario (a), defeating the milestone goal. Rejected.

### Recommendation

**Option 1.** Implement `sirius::io::cucascade_datasource` with `supports_device_read() == false`. Replace every `cudf::io::datasource::create(filepath)` call-site with a factory that returns the cucascade variant. Migrate the iceberg `source_info{filepath}` builders to `source_info{ds.get()}` with the new datasource. Make the prefetched-data-source fallback use the new type.

After this change, the only kvikio calls remaining in the process are any that cucascade itself chooses to make — and PR #96 is explicitly a non-kvikio disk backend, so that count should be zero.

---

## Pitfalls

### Pitfall 1 — `prefetched_data_source` threading

**Issue:** Today the per-task host-read loop at `parquet_scan_task.cpp:857-872` issues multiple concurrent `host_read_async` calls against the same datasource. Any replacement must handle concurrent calls from the Sirius task thread pool. kvikio's `FileHandle::pread` is thread-safe; cucascade's `disk_io_backend` needs to offer the same guarantee.
**Prevention:** Confirm thread-safety on `disk_file_handle::pread` in PR #96 review. If cucascade serializes per-handle, wrap the handle in an object pool sized to the thread pool.

### Pitfall 2 — `cudf::io::datasource::buffer` ownership semantics

**Issue:** `host_read(offset, size)` returns `std::unique_ptr<buffer>`. cuDF stores that unique_ptr for a brief window (footer parsing, page-index parsing, column-chunk host-read fallback in `fetch_byte_ranges_to_device_async`), then calls `buffer::data()` / `buffer::size()`. The underlying memory must survive until the `unique_ptr` is destroyed. If we return a buffer pointing into a cucascade-owned cache that might be evicted concurrently, we get use-after-free.
**Prevention:** In the cucascade-backed datasource, always allocate fresh pinned host memory for each `host_read` call (as `kvikio_source::clamped_read_to_vector` does with `std::vector<uint8_t>`). Do not return spans into shared buffers from the datasource itself — that role is reserved for `prefetched_data_source`.

### Pitfall 3 — `cuda_memcpy_async` host-source pinning requirement

**Issue:** cuDF's fallback path (`reader_impl_preprocess_utils.cu:223`, `parquet_io_utils.cpp:142`) calls `cudf::detail::cuda_memcpy_async` with the host buffer from `host_read`. For the async path to actually be async on the GPU (rather than synchronizing), the host buffer needs to be **pinned** (page-locked). Plain `std::vector<uint8_t>` is pageable → cuDF will silently serialize.
**Prevention:** Have `cucascade_datasource::host_read` allocate from cucascade's `fixed_size_host_memory_resource` (pinned) rather than `std::vector<uint8_t>`. Sirius already uses pinned host memory via `cucascade::memory::fixed_size_host_memory_resource` elsewhere (`parquet_scan_task.cpp:652-658`). Re-use the same allocator for datasource returns.

### Pitfall 4 — Compression codec decoding is independent of kvikio

**Issue:** SNAPPY / ZSTD / GZIP decompression inside cuDF's parquet reader runs on GPU via nvCOMP (see `cudf::io::nvcomp_integration`). This has **no** dependency on kvikio. Changing the datasource does not affect decompression correctness or performance.
**Prevention:** No special handling needed. `LIBCUDF_NVCOMP_POLICY` controls which codecs are GPU-accelerated vs CPU-decompressed; leave it at `STABLE` (default).

### Pitfall 5 — Footer length assumption in the pre-26.04 fallback

**Issue:** `parquet_scan_task.cpp:71-93` and `sirius_parquet_metadata_scan_operator.cpp:48-66` implement a manual footer read for cuDF < 26.04 by calling `datasource.host_read(len - 8, 8)` then `datasource.host_read(footer_offset, footer_len)`. Any custom datasource must support reads at arbitrary offsets (not only row-group-aligned).
**Prevention:** Datasource API contract already requires this. Test with both empty files and very small files to catch offset-bounds bugs.

### Pitfall 6 — `hybrid_scan_reader::materialize_*` future work

**Issue:** Sirius today does not call `materialize_filter_columns` / `materialize_payload_columns` but may in the future (per PR #96 planning docs elsewhere in `.planning/`). Those functions take device spans of column-chunk data. The caller (Sirius) owns the device-side lifetime. If we route column chunks through `fetch_byte_ranges_to_device_async` + our custom datasource, the datasource reports `supports_device_read() == false` → cuDF uses `host_read` + `cuda_memcpy_async`. The device buffer returned by `fetch_byte_ranges_to_device_async` is then sirius-owned. No correctness issue but worth noting: the I/O path is CPU-staged even for the hybrid reader.
**Prevention:** If/when materialize_* is adopted, rely on `fetch_byte_ranges_to_device_async` (public API) + custom datasource. Do not write a parallel "copy cached host bytes to device" code path — `prefetched_data_source::device_read*` already does this correctly.

### Pitfall 7 — Iceberg delete files are small and frequent

**Issue:** Iceberg positional/equality delete parquet files (`iceberg_scan_task.cpp:52-139`) are read via `cudf::io::read_parquet(source_info{filepath}, ...)` — one datasource per file, created implicitly inside cuDF. Under heavy delete-file workloads, this creates many short-lived `kvikio::FileHandle`s (one per call). Handle open cost is relatively low but the CUDA-context coupling issue is the same.
**Prevention:** Replace both call sites (lines 57-58 and 120-121) with `source_info{custom_datasource.get()}`. Consider caching the cucascade datasource instance per `delete_file_path` to amortize `disk_file_handle` open cost if profiling shows it's a hotspot.

### Pitfall 8 — `rmm::cuda_stream_default` in `parquet_scan_task.cpp:468`

**Issue:** The current code uses `rmm::cuda_stream_default` in `reader->filter_row_groups_with_stats(..., rmm::cuda_stream_default)`. This violates the project rule (user preference: "never use `rmm::cuda_stream_default`, always use explicit streams"). Not directly a kvikio issue but adjacent code that will be touched by this milestone.
**Prevention:** Plumb an explicit stream through from the scan-task global state. Mention in ROADMAP as a hygiene fix that can land with the datasource migration.

---

## Open questions

1. **Does cucascade's PR #96 `idisk_io_backend` expose a `pread(dst, size, offset) -> std::future<size_t>` API?** The research prompt states PR #96 adds `idisk_io_backend` + `io_backend_registry` but the exact method signatures weren't provided. The custom-datasource pseudocode above assumes `pread`-shaped semantics; if the real API is `read(offset, size, buffer) -> cucascade::read_future` or similar, the adapter layer is marginally more complex but the overall design is unchanged.

2. **Does cucascade's backend offer pinned-host-memory allocations directly, or does the caller allocate and pass buffers in?** This affects whether `cucascade_datasource::host_read` allocates from a cucascade pool or from Sirius's existing `fixed_size_host_memory_resource`. Pitfall 3 applies regardless.

3. **What is the threading model of cucascade's `disk_io_backend`?** Critical for Pitfall 1. PR #96 review should surface this.

4. **Does cuDF 26.06 introduce any env-var or reader-option level kvikio switch?** This research only confirmed absence in 26.04. If 26.06 adds a switch, the migration could potentially be staged (env-var for 26.06, custom datasource as the stable long-term answer). Worth a quick check of rapidsai/cudf branch-26.06 changelog when the nightly bump happens. Confidence LOW on this sub-question — not directly verified.

5. **Does `cudf::io::read_parquet` with `source_info{datasource*}` ever call `datasource::create(string)` internally?** Answered NO for v26.04.00 based on reading the top of `datasource.cpp` and `read_parquet` dispatch, but worth a spot-check during implementation — if cuDF opens auxiliary sidecar files (e.g., partitioned dataset metadata) via filepath, they would slip through as kvikio. Sirius's single-file-per-source usage makes this unlikely to matter in practice.

6. **What happens when `file_size_` is unknown at `cucascade_datasource` construction?** The current `prefetched_data_source::size()` (`prefetched_data_source.cpp:200-205`) falls back to `ranges_->max_offset()` or `fallback_->size()`. The new custom datasource will typically know the size from the backend's stat-equivalent; if not, we must `stat(filepath)` before construction. Trivial but worth pinning down.
