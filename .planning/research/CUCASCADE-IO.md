# Cucascade IO Backend API — for Sirius parquet migration

**Researched:** 2026-04-20
**Cucascade version locked to:** `origin/main` @ f47de0b (contains PR #96 @ 9833849)
**Overall confidence:** HIGH — all signatures and behavior copy-pasted from `git show origin/main:...` on the pinned SHA; Sirius integration points cross-referenced against the actual working tree.

---

## TL;DR for the roadmap

**Critical finding:** PR #96's `idisk_io_backend` is a **cuCascade-columnar-snapshot** I/O backend, not a parquet I/O backend. It writes flat `{ptr, size, file_offset}` buffers to disk and reads them back; it does **not** understand parquet, compression codecs, or row-group layout. The built-in `pipeline_io_backend` is a pinned-memory double-buffered `pread/pwrite(O_DIRECT optional)` implementation driven by its own internally-created `cudaStream_t` plus a background I/O thread.

**What this means for Sirius' goal of replacing kvikio:**

1. `idisk_io_backend::read(path, dev_ptr, size, file_offset, stream)` gives Sirius exactly the primitive it needs to fetch arbitrary byte ranges from a parquet file directly into a device buffer, with an explicit RMM stream. This is the **only piece of the PR #96 API that Sirius will call from the parquet scan path.**
2. `disk_data_representation` / `disk_file_format` / `disk_table_allocation` are **not useful** for parquet — they're for the cucascade disk-tier downgrade path (GPU-table → on-disk columnar snapshot → GPU-table). Sirius' parquet scan pipeline stays parquet on disk.
3. `io_backend_registry` is the right place to register per-GPU backend factories; but Sirius needs to **invent its own integration point** (a custom `cudf::io::datasource` adapter on top of `idisk_io_backend`) because cuDF's parquet reader takes a `datasource`, not an `idisk_io_backend`.
4. `pipeline_io_backend` creates its own internal `cudaStream_t` + event at construction — so the CUDA context that owns the backend is whatever context was current at `make_pipeline_io_backend()` time. **One instance per GPU is required** for multi-GPU safety. This is consistent with the registry's factory model ("each call should return a new instance").

**Migration target:** Build `sirius::op::scan::io_backend_datasource` — a `cudf::io::datasource` that wraps a `cucascade::idisk_io_backend` shared_ptr from the per-GPU memory space, translating `host_read` / `device_read` / `device_read_async` into `backend.read(path, …)` calls.

---

## Public API (exact signatures)

### `idisk_io_backend`

Source: `cucascade/include/cucascade/data/disk_io_backend.hpp` (origin/main, PR #96).

```cpp
namespace cucascade {

/// Descriptor for a single I/O operation in a batch.
struct io_batch_entry {
  const void* ptr;          // Device or host memory pointer
  std::size_t size;         // Number of bytes
  std::size_t file_offset;  // Byte offset in the file
};

class idisk_io_backend {
 public:
  virtual ~idisk_io_backend() = default;

  // Device write (GPU → file). 'stream' orders against the caller's kernels.
  virtual void write(const std::filesystem::path& path,
                     const void* dev_ptr,
                     std::size_t size,
                     std::size_t file_offset,
                     rmm::cuda_stream_view stream) = 0;

  // Device read (file → GPU). 'stream' orders subsequent kernels.
  virtual void read(const std::filesystem::path& path,
                    void* dev_ptr,
                    std::size_t size,
                    std::size_t file_offset,
                    rmm::cuda_stream_view stream) = 0;

  // Host write (host → file). No CUDA involvement.
  virtual void write(const std::filesystem::path& path,
                     const void* host_ptr,
                     std::size_t size,
                     std::size_t file_offset) = 0;

  // Host read (file → host). No CUDA involvement.
  virtual void read(const std::filesystem::path& path,
                    void* host_ptr,
                    std::size_t size,
                    std::size_t file_offset) = 0;

  // Batched device writes — default impl is sequential write() calls.
  virtual void write_batch(const std::filesystem::path& path,
                           const std::vector<io_batch_entry>& entries,
                           rmm::cuda_stream_view stream);

  // Batched device reads — default impl is sequential read() calls.
  virtual void read_batch(const std::filesystem::path& path,
                          const std::vector<io_batch_entry>& entries,
                          rmm::cuda_stream_view stream);
};

}  // namespace cucascade
```

**Semantics / threading model (verified against `pipeline_io_backend.cpp`):**

- All methods are **blocking from the caller's thread's perspective** (they return after issuing the work, but the device-variant methods still require `stream.synchronize()` on the caller side before reading the device buffer — see the `read()` pipeline, which performs `cudaStreamSynchronize(_copy_stream)` inside the backend between H2D stages, but does **not** sync the caller's `stream` at the end).
- `stream` is used to **order against the caller's kernels**: on device-write, `cudaEventRecord(_order_event, stream)` + `cudaStreamWaitEvent(_copy_stream, _order_event)` to guarantee prior GPU work completes before D2H starts; on device-read, the same pattern protects the destination buffer from being overwritten while prior kernels still use it.
- The backend owns its own `_copy_stream` (internal non-blocking) and an `io_worker` background thread that serializes `pwrite`/`pread` system calls via `std::future`. Per-backend-instance concurrency is 1 in-flight read and 1 in-flight write (double-buffered). **Callers wanting N concurrent reads into N separate device buffers must create N backend instances** (or extend the pipeline backend — not in the default).
- Host overloads open the file with `O_CREAT|O_WRONLY` / `O_RDONLY` (no `O_DIRECT`) and issue a single `pwrite`/`pread` synchronously on the calling thread.
- **File open/close happens inside every `read()`/`write()`** call — there is no persistent file-handle cache. For Sirius, which issues hundreds of byte-range reads per parquet file, this is a **performance concern** (see Pitfalls §P3).

**Stream ownership (critical for multi-GPU):**

- `pipeline_io_backend` constructor (src/data/pipeline_io_backend.cpp:143-149) runs:
  - `cudaMallocHost(&_buf[0], 64MB)` — pinned host buffer, bound to current CUDA context
  - `cudaMallocHost(&_buf[1], 64MB)`
  - `cudaStreamCreate(&_copy_stream)` — stream bound to current CUDA context
  - `cudaEventCreateWithFlags(&_order_event, …)` — event bound to current CUDA context
- Destructor calls `cudaFreeHost` + `cudaStreamDestroy` + `cudaEventDestroy` in the same context (problematic if the backend outlives the context).
- **Conclusion:** A `pipeline_io_backend` instance is context-bound at construction. `io_backend_registry` creates a **fresh instance per call** to `create_backend()` — this is documented at `io_backend_registry.hpp:32-33` ("Each call should return a new instance. Backends may have internal state (staging buffers, CUDA streams) that should not be shared across unrelated contexts.").

### `io_backend_registry`

Source: `cucascade/include/cucascade/data/io_backend_registry.hpp`.

```cpp
namespace cucascade {

using io_backend_factory_fn = std::function<std::shared_ptr<idisk_io_backend>()>;

class io_backend_registry {
 public:
  io_backend_registry() = default;
  // Non-copyable, non-movable.

  // Throws std::runtime_error on duplicate name.
  void register_backend(const std::string& name, io_backend_factory_fn factory);

  [[nodiscard]] bool has_backend(const std::string& name) const;

  // Throws std::runtime_error if not registered.
  [[nodiscard]] std::shared_ptr<idisk_io_backend> create_backend(const std::string& name) const;

  bool unregister_backend(const std::string& name);
  void clear();  // Also resets default to "pipeline".

  // Throws std::runtime_error if name not registered.
  void set_default(const std::string& name);
  [[nodiscard]] std::string get_default_name() const;  // initially "pipeline"
  [[nodiscard]] std::shared_ptr<idisk_io_backend> create_default_backend() const;
};

// Registers "pipeline" only. No other built-ins.
void register_builtin_io_backends(io_backend_registry& registry);

}  // namespace cucascade
```

**Behavior (from `src/data/io_backend_registry.cpp`):**

- Thread-safe via a single `std::mutex _mutex` — every operation takes the lock. Registry-level contention is trivial (registration happens at init; lookup is rare).
- `create_backend()` calls the factory under the lock, returns `shared_ptr`. **The factory closure captures whatever CUDA context is current at the time `create_backend()` is called**, which is what makes per-GPU scoping possible: call `cudaSetDevice(i)` then `registry.create_backend("pipeline")` and the returned backend is pinned to GPU `i`.
- `register_builtin_io_backends()` registers exactly **one** backend: `"pipeline"`, factory `make_pipeline_io_backend()` (with `direct_io=false` default; the `direct_io=true` variant is not registered and must be registered by the caller if needed).

### `disk_data_representation` / `disk_file_format`

Source: `cucascade/include/cucascade/data/disk_data_representation.hpp`, `disk_file_format.hpp`.

```cpp
namespace cucascade {

// Alignment boundary for column data in cucascade disk files. 4KB = DMA-safe.
static constexpr std::size_t DISK_FILE_ALIGNMENT = 4096u;

class disk_data_representation : public idata_representation {
 public:
  disk_data_representation(std::unique_ptr<memory::disk_table_allocation> disk_table,
                           memory::memory_space& memory_space);
  ~disk_data_representation() noexcept;  // Deletes the backing file (best-effort).

  [[nodiscard]] std::size_t get_size_in_bytes() const override;
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override;

  // ALWAYS throws — disk representations cannot be cloned in place.
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;

  [[nodiscard]] const memory::disk_table_allocation& get_disk_table() const;
};

}  // namespace cucascade
```

```cpp
// include/cucascade/memory/disk_table.hpp
struct disk_table_allocation {
  std::string file_path;                 // Absolute path to the batch file on disk
  std::vector<column_metadata> columns;  // Per-column metadata
  std::size_t data_size;                 // Total bytes written to the file
};

std::string generate_disk_file_path(std::string_view base_path);
```

**Critical observation for Sirius:** `disk_data_representation` is **NOT a parquet-on-disk representation**. It is a cuCascade-native columnar snapshot format — a header-less file of 4KB-aligned raw column buffers with metadata held in memory (not in the file). The destructor deletes the file, which is correct for ephemeral downgrade snapshots but **incompatible with persistent user parquet files that must not be deleted**.

Sirius should **NOT** extend `disk_data_representation` for parquet. The integration Sirius needs is strictly at the `idisk_io_backend` primitive level.

### Memory-space changes from PR #96 (`memory_space.hpp`)

```cpp
class memory_space {
 public:
  // Existing.
  explicit memory_space(const gpu_memory_space_config& config);
  explicit memory_space(const host_memory_space_config& config);
  explicit memory_space(const disk_memory_space_config& config);

  // NEW in PR #96.
  memory_space(const disk_memory_space_config& config,
               std::shared_ptr<idisk_io_backend> io_backend);

  [[nodiscard]] std::string_view get_disk_mount_path() const;  // Throws if not DISK tier.
  [[nodiscard]] idisk_io_backend& get_io_backend() const;      // Throws if not DISK tier.
  // ... rest unchanged ...
 protected:
  // NEW: I/O backend for DISK tier (null for other tiers).
  std::shared_ptr<idisk_io_backend> _io_backend;
};

// config.hpp
struct disk_memory_space_config {
  int disk_id{-1};
  std::size_t memory_capacity{0};
  std::string mount_paths;   // REQUIRED — used by disk_access_limiter and generate_disk_file_path()
  Tier tier() const { return Tier::DISK; }
  // ...
};
```

**Implication:** `idisk_io_backend` is **stored on and retrieved from DISK-tier memory spaces only**. A `gpu_memory_space_config` or `host_memory_space_config` has **no** io_backend slot. For Sirius to resolve the right backend for a parquet scan on GPU `i`, it cannot go through the memory_space; it has to maintain its own mapping (or construct one) from `device_id → shared_ptr<idisk_io_backend>`.

---

## Built-in backends

**Only one built-in: `"pipeline"`.**

Source: `cucascade/src/data/pipeline_io_backend.cpp` (registered by `register_builtin_io_backends()` with `direct_io=false`).

**Architecture:**
- Two 64 MB pinned host buffers (`cudaMallocHost`) used as a ping-pong staging area.
- One internal `_copy_stream` (non-blocking) for all D2H/H2D copies.
- One `_order_event` to synchronize against the caller's stream.
- One dedicated background `io_worker` thread with a `std::future<void>` handoff for each `pread`/`pwrite` system call — avoids `std::async` spawn overhead per chunk.

**Write path (device → disk):**
1. `cudaEventRecord` + `cudaStreamWaitEvent` to order against the caller's stream.
2. `open(path, O_CREAT|O_WRONLY [|O_DIRECT])`.
3. For each 64 MB chunk: `cudaMemcpyAsync` D2H on `_copy_stream`, `cudaStreamSynchronize`, wait on previous `io_worker` future, submit new `pwrite` future, swap buffers.
4. Final `io_worker.get()`, `close(fd)`.

**Read path (disk → device):**
1. `cudaEventRecord` + `cudaStreamWaitEvent` (protect destination from prior kernels).
2. `open(path, O_RDONLY [|O_DIRECT])`.
3. Pre-read first chunk into buffer 0 synchronously on the calling thread.
4. For each subsequent chunk: start H2D copy from current buffer on `_copy_stream` + in parallel submit `pread` of next chunk to `io_worker`, sync copy stream, wait on pread, swap buffers.
5. `close(fd)`.

**Performance characteristics:**

| Dimension | `pipeline` backend | cuDF's default (kvikio) |
|---|---|---|
| Chunk size | 64 MB fixed | kvikio-default thread pool |
| Concurrency | 1 in-flight pread + 1 in-flight H2D per backend | kvikio thread pool (default 4 threads, env `KVIKIO_NTHREADS`) |
| O_DIRECT | Optional (disabled by default; not wired into the built-in factory) | `KVIKIO_COMPAT_MODE` controls cuFile vs POSIX |
| cuFile/GDS | **No** — goes H2D through pinned host buffer | Yes (when GDS + cuFile available + NVMe) |
| PCIe vs NVMe overlap | Explicit — D2H runs in parallel with NVMe write via background worker | Via kvikio thread pool |
| File handle reuse | **No** — every call re-opens the file | kvikio caches file handles |
| Multi-GPU safe | Yes if one instance per GPU (explicit per-device factory call) | **No** — cuFile driver state ties to a single primary context |

**Correctness characteristics:**
- `O_DIRECT` path pads writes up to 512-byte alignment and zeros the padding (`align_up_dio(n)`). Reads request `first_read_sz = align_up_dio(first_chunk)` but only copy `first_chunk` bytes to device. Host write/read overloads deliberately do **not** use `O_DIRECT` (the comment says "O_DIRECT requires 4KB-aligned buffers and sizes which metadata typically isn't").
- No GPU-direct path — all device reads go through pinned host memory. This removes the kvikio/cuFile multi-context fragility but leaves ~10-15 GB/s PCIe staging bandwidth on the table vs direct NVMe-to-GPU on systems with GDS.
- No prefetch / read-ahead across multiple row-group ranges in a single call — the batch API (`read_batch`) flattens entries into a sequential chunk stream but still serializes through the single `_copy_stream` and single `io_worker`.

**No other built-ins.** S3, HTTP, memory-mapped, async-filesystem backends are **not** provided. `register_builtin_io_backends(registry)` only calls `registry.register_backend("pipeline", …)`.

---

## Integration model for cuDF parquet

This is the load-bearing section for Sirius migration planning.

### The fundamental shape mismatch

| What cuDF parquet expects | What `idisk_io_backend` provides |
|---|---|
| `cudf::io::datasource*` via `source_info{datasource*}` | `read(path, buf, size, offset, stream)` |
| Byte-range reads keyed by `(offset, size)` | Byte-range reads keyed by `(offset, size)` — ✓ compatible |
| `host_read(offset, size)`, `host_read(offset, size, uint8_t* dst)` | `read(path, host_ptr, size, offset)` — ✓ compatible |
| `device_read(offset, size, dst, stream)` + `device_read_async(…)` | `read(path, dev_ptr, size, offset, stream)` — ✓ compatible but sync |
| `size()` (total file size) | **Not provided** — backend has no file-metadata concept |
| `supports_device_read()`, `is_device_read_preferred()` | implicit — backend always supports both; Sirius decides policy |

**cuDF parquet scan accepts a user-implemented `cudf::io::datasource`** via `source_info{datasource*}` (`cudf/io/types.hpp:406-419`). This is the integration seam. Sirius already uses this pattern in `src/op/scan/prefetched_data_source.cpp`.

### Recommended architecture: `io_backend_datasource`

Build a thin adapter class (proposed name: `sirius::op::scan::io_backend_datasource`) that implements `cudf::io::datasource` by delegating to a `cucascade::idisk_io_backend`:

```cpp
namespace sirius::op::scan {

class io_backend_datasource : public cudf::io::datasource {
 public:
  io_backend_datasource(std::shared_ptr<cucascade::idisk_io_backend> backend,
                        std::filesystem::path path,
                        std::size_t file_size);

  // host_read variants — delegate to backend.read(path, host_ptr, size, offset)
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override;
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  // device_read variants — delegate to backend.read(path, dev_ptr, size, offset, stream)
  bool supports_device_read() const override { return true; }
  bool is_device_read_preferred(size_t size) const override { return true; }
  std::unique_ptr<buffer> device_read(size_t offset, size_t size,
                                      rmm::cuda_stream_view stream) override;
  size_t device_read(size_t offset, size_t size, uint8_t* dst,
                     rmm::cuda_stream_view stream) override;
  std::future<size_t> device_read_async(size_t offset, size_t size, uint8_t* dst,
                                        rmm::cuda_stream_view stream) override;

  size_t size() const override { return _file_size; }

 private:
  std::shared_ptr<cucascade::idisk_io_backend> _backend;
  std::filesystem::path _path;
  std::size_t _file_size;
};

}  // namespace sirius::op::scan
```

### Per-GPU backend ownership

Because `pipeline_io_backend` is context-bound at construction and Sirius is single-node multi-GPU:

1. At SiriusContext init (where `converter_registry::initialize()` runs — `src/sirius_extension.cpp:1053`), also initialize an `io_backend_registry`:
   ```cpp
   cucascade::register_builtin_io_backends(io_backend_registry_);
   ```
2. Create **one backend instance per GPU** by setting the CUDA device before each `create_backend()` call, and cache the `shared_ptr<idisk_io_backend>` indexed by `device_id`:
   ```cpp
   for (auto* gpu_space : memory_manager_->get_memory_spaces_for_tier(Tier::GPU)) {
     rmm::cuda_set_device_raii guard{gpu_space->get_device_id()};
     gpu_io_backends_[gpu_space->get_device_id()] =
       io_backend_registry_.create_default_backend();
   }
   ```
3. At parquet scan time, select the backend for the preferred device (see SCHED-01..05 in PROJECT.md for where `preferred_device_id` comes from) and wrap it in `io_backend_datasource`.

### Where cucascade's `disk_data_representation` fits (answer: not here)

`disk_data_representation` + the converters registered by `register_builtin_converters()` (HOST↔DISK and GPU↔DISK) compose a **downgrade tier** — they let Sirius spill a materialized cuDF `gpu_table_representation` to a cucascade-format columnar file and later re-hydrate it. This is useful when a query's intermediate results exceed HOST memory and must spill to local NVMe. It is **orthogonal to parquet scan**. The parquet file lives in the filesystem as a first-class persistent artifact and is not a cucascade disk-tier allocation.

Parquet scan sits **above** the tier system: it reads from the filesystem directly (via the wrapped io_backend) into whatever memory tier the scan task's reservation targets (typically HOST for `host_parquet_representation`, then GPU via the existing host→gpu converter).

---

## Sirius migration targets (file:line)

All file paths are absolute. Line references are to the working tree (cucascade at 942c0bf, sirius at `feature/single-node-multi-gpu2` tip).

### Direct `cudf::io::datasource::create(path)` call sites (these become `io_backend_datasource`)

1. **`/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/src/op/scan/parquet_scan_task.cpp:312`** — `parquet_scan_task_global_state::initialize_from_files()`, footer pre-read during task planning. Per-file, once per query. Currently host-only (`datasource->host_read`). Migration: construct one `io_backend_datasource` per file using the global (not per-GPU) backend since this runs during planning, not per-task.
2. **`/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/src/op/scan/parquet_scan_task.cpp:699`** — `parquet_scan_task::compute_task()`, per-task datasource creation for `_datasource`. This is the hot path — one call per row-group partition. Migration: select the per-GPU backend based on `preferred_device_id` from the local_state and construct an `io_backend_datasource`.
3. **`/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/src/op/scan/parquet_scan_task.cpp:863`** — `read_range_into_allocation()` uses `_datasource->host_read_async(…)` to prefetch column-chunk ranges into a host multiple-blocks allocation. Already goes through the `_datasource` abstraction, so it picks up whatever `_datasource` was set to on line 699. **No code change required** here beyond making sure the new datasource implements `host_read_async` correctly (the base class provides a default via `std::async`, but that bypasses the io_backend — see Pitfalls §P3).
4. **`/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/src/op/scan/sirius_parquet_metadata_scan_operator.cpp:251`** — `execute()`, parallel to call site #1. Same treatment: construct `io_backend_datasource` via global backend for planning-time metadata parsing.

### Indirect kvikio use via `cudf::io::read_parquet`

5. **`/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/src/op/scan/iceberg_scan_task.cpp:58`** (`read_positional_delete_file`) — `cudf::io::read_parquet(opts, stream)` where `opts` is built from `source_info{delete_file_path}` — this path goes directly through cuDF's filesystem datasource (and hence kvikio when available). Migration: replace the `source_info{path}` with `source_info{io_backend_datasource*}` to route through cucascade.
6. **`/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/src/op/scan/iceberg_scan_task.cpp:120`** (`read_equality_delete_file`) — same pattern. Same migration.
7. **`/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/src/data/host_parquet_representation_converters.cpp:82-92`** — this one is already OK: `opts.set_source(cudf::io::source_info{data_source.get()})` where `data_source` is a `prefetched_data_source`. Data has already been pulled to HOST by the scan task. **No migration needed** — the prefetched data source is not a kvikio path.

### Memory-space / context integration points

8. **`/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/src/sirius_extension.cpp:1053`** (`converter_registry::initialize()`) — add a sibling `io_backend_registry` singleton and `cucascade::register_builtin_io_backends(...)` call here.
9. **`/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/src/sirius_context.cpp:163-230`** (`SiriusContext::initialize()`) — after `memory_manager_` is constructed (line 169), iterate GPU memory spaces and create one `idisk_io_backend` per GPU under `rmm::cuda_set_device_raii`. Store in a `std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>>` on `SiriusContext`.
10. **`/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/src/include/data/sirius_converter_registry.hpp`** — likely the right place to add a `sirius::io_backend_registry` parallel singleton (or fold it into a new `sirius::io_registry`). Keep initialization co-located with `converter_registry::initialize()`.

### Task/local-state plumbing

11. **`/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/src/include/op/scan/parquet_scan_task.hpp:432-526`** — `parquet_scan_task_local_state` already carries a `_reservation` which has a `memory_space&` (line 650 in `.cpp`). The HOST space is used for the compressed-block allocation. The GPU space for the final table is resolved later in the converter via `target_memory_space->get_device_id()`. The per-GPU io_backend needs to be selected by the **GPU device id that the scan task is targeting**, which is surfaced through `preferred_device_id` on local_state (v1.0 multi-gpu work, SCHED-01..04 per PROJECT.md). Proposed: thread `io_backend_datasource* datasource` through the local_state or pull it from `SiriusContext` at the point of datasource construction in `compute_task()`.

### Summary of source file changes

| File | Change |
|---|---|
| New: `src/include/op/scan/io_backend_datasource.hpp` | Declare the `cudf::io::datasource` adapter |
| New: `src/op/scan/io_backend_datasource.cpp` | Implement the adapter |
| New: `src/include/sirius_io_registry.hpp` | Singleton wrapping `cucascade::io_backend_registry` + per-GPU backend cache |
| `src/op/scan/parquet_scan_task.cpp` | Replace 2 `datasource::create(path)` calls with `io_backend_datasource` construction |
| `src/op/scan/sirius_parquet_metadata_scan_operator.cpp` | Replace 1 `datasource::create(path)` call |
| `src/op/scan/iceberg_scan_task.cpp` | Replace 2 `source_info{path}` constructions |
| `src/sirius_extension.cpp` | Initialize io_backend_registry at extension load |
| `src/sirius_context.cpp` | Create per-GPU backend instances during context init |
| `CMakeLists.txt` / `extension_config.cmake` | Bump cucascade submodule pointer to f47de0b |
| `cucascade` submodule pointer | 942c0bf → f47de0b |

---

## Pitfalls

Specific to **parquet reads through `idisk_io_backend`**, not generic file I/O.

### P1 — `pipeline_io_backend` has no file-handle cache; parquet scans issue hundreds of small reads

Parquet scans issue one `datasource.host_read(offset, size, dst)` per merged column-chunk byte range. After `merge_byte_ranges()` a typical TPC-H-SF10 parquet file still produces 20-60 reads per row-group partition. The current `pipeline_io_backend::read(path, host_ptr, size, offset)` host overload does `open` / `pread` / `close` on **every call** (pipeline_io_backend.cpp:310-330). Each open on NVMe is O(10-100 µs) but on HDFS / FUSE / networked filesystems it can be milliseconds.

- **Detection:** profile shows `open`/`close` dominate I/O time for small-file-per-range access patterns.
- **Mitigation:** either extend `pipeline_io_backend` with a caching LRU of open fds (needs a cucascade upstream patch), or in `io_backend_datasource` keep the fd open via a `pread` wrapper that bypasses the backend for subsequent reads on the same path (defeats the abstraction — avoid). **Recommendation:** file a cucascade issue to add persistent file-handle caching to `pipeline_io_backend`; short-term, accept the per-read open cost (NVMe-local) and measure.

### P2 — No compression codec support; no parquet structure awareness

The io_backend reads raw bytes. All parquet-specific concerns — decompressing Snappy/Zstd/Gzip column chunks, parsing page headers, decoding dictionary pages, applying deletes — happen on the GPU inside `cudf::io::read_parquet`. This is correct separation of concerns (the io_backend is a byte provider) but means:

- **Detection:** comparing to a cuFile/GDS baseline will show worse end-to-end parquet-read throughput because kvikio+cuFile can push decompressed bytes directly to GPU, bypassing host staging. `pipeline_io_backend` **always** stages through pinned host buffers.
- **Mitigation:** the 10-15 GB/s PCIe ceiling this imposes is acceptable for v1.1 (the goal is multi-GPU safety, not bandwidth). Flag for v1.2 to explore a GDS-aware backend variant if benchmark regresses >2x vs kvikio on NVMe-GDS systems.

### P3 — `device_read_async` semantics differ: backend is synchronous, cuDF expects async

`idisk_io_backend::read(path, dev_ptr, size, offset, stream)` returns after the read is **issued on `stream`** but the implementation in `pipeline_io_backend` does `cudaStreamSynchronize(_copy_stream)` between each 64 MB H2D chunk (pipeline_io_backend.cpp:248) — meaning the caller's `stream` does not actually return immediately for large reads; the call blocks on the backend's internal pipeline completion.

cuDF's `datasource::device_read_async(offset, size, dst, stream)` expects a **fire-and-forget** submission that returns a `std::future<size_t>` the caller can wait on later. Naïvely wrapping `backend.read(…)` in `std::async(std::launch::deferred, …)` (as `prefetched_data_source::device_read_async` does at src/op/scan/prefetched_data_source.cpp:194) is correct but loses the async benefit because the deferred task only runs when the future is waited on — at which point `backend.read` blocks the caller.

- **Detection:** parquet reader throughput regresses vs the kvikio path on multi-row-group reads where cuDF would naturally overlap multiple `device_read_async` calls.
- **Mitigation:** in `io_backend_datasource::device_read_async`, submit the backend call to a shared `std::async(std::launch::async, …)` with a per-datasource thread pool, so multiple async reads can queue up to the backend concurrently. Even so, **the backend serializes** — per-instance concurrency is 1 (see `io_worker` in pipeline_io_backend.cpp:46-104). For true concurrency, Sirius would need N backend instances per GPU (N ≈ number of concurrent scan tasks targeting that GPU). This conflicts with the registry's "one instance per context" model. **Recommendation:** for v1.1, accept serialization per GPU and measure; if it's a bottleneck, file an upstream cucascade issue requesting a multi-worker pipeline backend.

### P4 — Memory-mapped and S3/HTTP sources are not supported

cuDF's default `datasource::create(path)` detects `s3://`, `http://`, `hdfs://` URIs and dispatches to appropriate backends (via kvikio remote I/O or custom datasources). `pipeline_io_backend` only handles local `std::filesystem::path`. Any Iceberg or Delta Lake catalog that points at S3 URIs will break.

- **Detection:** attempts to pass `s3://...` paths to the backend will fail in `::open()`.
- **Mitigation:** in `io_backend_datasource::io_backend_datasource(path, …)`, detect non-local schemes and fall back to `cudf::io::datasource::create(path)` (wrapping the cuDF-provided datasource instead of the cucascade backend) for remote paths. This preserves kvikio-dependency for remote I/O only. Alternatively, defer remote-source support to v1.2. **Recommendation:** fallback path (2-3 LOC in the adapter constructor); remote I/O is not in Milestone v1.1's scope per PROJECT.md §Out of Scope ("GPU-Direct RDMA (network GDS)").

### P5 — Large files > 2 GB or offsets > INT_MAX require size_t correctness

All `idisk_io_backend` methods use `std::size_t` for `size` and `file_offset`, and `pwrite`/`pread` use `off_t`. The backend casts `static_cast<off_t>(file_offset)` (pipeline_io_backend.cpp:199, 232). On 64-bit Linux `off_t` is 64-bit — safe. On other platforms (none in Sirius's target matrix, but worth noting) this could truncate. Not a concern for Sirius but worth flagging for the upstream backend.

### P6 — `pipeline_io_backend` allocates 128 MB of pinned host memory per instance

At 8 backend instances (one per GPU on DGX), that's **1 GB of pinned host memory** just for the staging buffers. If the pool is shared with Sirius's fixed-size host memory resource (`cucascade::memory::fixed_size_host_memory_resource`), this competes for the same NUMA-local pinned allocation budget.

- **Detection:** `numactl --hardware` + `cat /proc/meminfo | grep HugePages` / pinned memory accounting before and after init.
- **Mitigation:** the 64 MB buffer size is a compile-time constant (`PIPELINE_BUF_SIZE`). Either (a) accept the cost (cucascade's memory accounting already counts pinned memory in the HOST tier budget), (b) file an upstream issue to make `PIPELINE_BUF_SIZE` configurable via the factory, or (c) register a custom factory that instantiates a smaller-buffered variant. For v1.1 **(a) is fine** — 1 GB on DGX-H100 systems with 1-2 TB RAM is noise.

### P7 — Backend construction and destruction must happen with the correct CUDA device current

Since the constructor calls `cudaMallocHost` / `cudaStreamCreate` / `cudaEventCreateWithFlags` and the destructor calls their counterparts, both must execute with the device_id that the backend is intended for. The Sirius init sequence must wrap backend creation in `rmm::cuda_set_device_raii` (as shown in the §Integration model above) and the destruction (via `shared_ptr` release) must happen **before** the CUDA contexts are torn down at extension unload.

- **Detection:** `cudaErrorInvalidResourceHandle` during extension unload.
- **Mitigation:** destroy the `io_backend_registry` and per-GPU cache in `SiriusContext` destructor **before** `memory_manager_.reset()`. Mirror the teardown order used for `downgrade_executors_` in the current `SiriusContext` destructor.

### P8 — `disk_data_representation::~disk_data_representation` deletes files unconditionally

**Not a Sirius concern for the parquet migration** — Sirius is not using `disk_data_representation` for parquet. But if Sirius ever adopts cucascade's disk-tier downgrade, any `disk_data_representation` constructed around a persistent user file path would delete it on destruction. Always use `memory::generate_disk_file_path(mount_path)` for disk-tier allocations, never user-supplied paths.

---

## Open questions

1. **Async path optimization.** Does Sirius need to extend `pipeline_io_backend` for multiple concurrent in-flight reads per GPU, or is 1 in-flight sufficient given that Sirius's scan pipeline already has N concurrent tasks-per-GPU at the task-creator level? **To resolve:** measure parquet scan throughput on a 16-GPU system with 1 backend-per-GPU vs N backends-per-GPU. Defer until v1.1 has end-to-end plumbing and benchmarks exist.
2. **Fallback-to-kvikio for remote URIs.** Should `io_backend_datasource` always wrap `idisk_io_backend`, or should it detect scheme and fall back to `cudf::io::datasource::create(uri)` for `s3://` / `http://`? P4 suggests the fallback is 2-3 LOC. **To resolve:** ask the milestone owner whether Iceberg-on-S3 is in scope; PROJECT.md does not explicitly list it but also doesn't exclude it.
3. **Benchmark regression tolerance.** How much parquet-read throughput regression is acceptable vs the current kvikio baseline to gain multi-GPU safety? The pipeline backend's H2D-staging design will regress vs cuFile/GDS on NVMe-with-GDS systems. **To resolve:** baseline TPC-H SF100 parquet scan on DGX with GDS enabled (current `dev`), then compare to `io_backend_datasource` path. Propose ≤30% regression as acceptable.
4. **Should the io_backend_registry be stateful on SiriusContext, or a process-global singleton like converter_registry?** Converter_registry is global. Backends, because they hold CUDA resources, arguably should be per-context. **To resolve:** the `io_backend_registry` itself is cheap — make it global. The per-GPU `shared_ptr<idisk_io_backend>` cache needs to be per-context because contexts can be torn down and rebuilt.
5. **Host-side `host_read_async` semantics.** cuDF's base-class `host_read_async` does a `std::async(std::launch::deferred, …)` wrapping the sync `host_read`. `idisk_io_backend` has no `host_read_async`; the backend's host read is fully synchronous. For Sirius's current `read_range_into_allocation` pattern (src/op/scan/parquet_scan_task.cpp:863) which issues many `host_read_async` calls to overlap I/O, we need an **adapter-level thread pool** to preserve that overlap. **To resolve:** include a small `std::async(std::launch::async, …)` wrapper in the adapter; do not rely on the base-class default.
6. **Does swapping to `io_backend_datasource` break the prefetched_data_source flow?** No — `prefetched_data_source` wraps an already-staged host allocation, and its `fallback_datasource` (src/op/scan/prefetched_data_source.cpp:113) is only hit on uncovered ranges. The fallback is `_fallback_datasource` set from `parquet_scan_task::_datasource` — which we're replacing with an `io_backend_datasource`. So the fallback path naturally benefits from the migration with no additional work. Confirmed by tracing src/op/scan/parquet_scan_task.cpp:769 → src/data/host_parquet_representation.hpp:119 → src/data/host_parquet_representation_converters.cpp:83.

---

## References (with confidence)

| Claim | Source | Confidence |
|---|---|---|
| `idisk_io_backend` API signatures | `cucascade` origin/main `include/cucascade/data/disk_io_backend.hpp` | HIGH — copied verbatim from `git show` |
| `io_backend_registry` API signatures | `cucascade` origin/main `include/cucascade/data/io_backend_registry.hpp` + `src/data/io_backend_registry.cpp` | HIGH — copied verbatim |
| `pipeline_io_backend` threading model, buffer sizes, O_DIRECT semantics | `cucascade` origin/main `src/data/pipeline_io_backend.cpp` | HIGH — read in full |
| `disk_data_representation` deletes backing file in dtor | `cucascade` origin/main `src/data/disk_data_representation.cpp:43` | HIGH |
| `register_builtin_io_backends` registers only "pipeline" | `cucascade` origin/main `src/data/io_backend_registry.cpp:87-91` | HIGH |
| `register_builtin_converters` registers 4 disk converters (HOST↔DISK, GPU↔DISK) | `cucascade` origin/main `src/data/representation_converter.cpp:1461-1506` | HIGH |
| `disk_memory_space` stores an optional `shared_ptr<idisk_io_backend>` | `cucascade` origin/main `include/cucascade/memory/memory_space.hpp:96-100, 151` | HIGH |
| cuDF accepts user `cudf::io::datasource*` via `source_info` | `.pixi/envs/default/include/cudf/io/types.hpp:406-419` | HIGH |
| cuDF parquet uses kvikio_integration for file paths | `.pixi/envs/default/include/cudf/io/config_utils.hpp:11-29` | HIGH |
| Sirius currently constructs `cudf::io::datasource::create(path)` at 3 call sites | `src/op/scan/parquet_scan_task.cpp:312, 699` + `src/op/scan/sirius_parquet_metadata_scan_operator.cpp:251` | HIGH — grep result |
| Sirius ice-scan read_parquet goes through filesystem datasource | `src/op/scan/iceberg_scan_task.cpp:58, 120` | HIGH |
| `prefetched_data_source` already uses the custom-datasource pattern | `src/op/scan/prefetched_data_source.{hpp,cpp}` | HIGH |
| Multi-GPU branch has no prior io_backend work | `git log --all --grep="io_backend"` returned zero matches | HIGH |
