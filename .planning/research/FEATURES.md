# Feature Research — v1.4 New API Surface

**Domain:** GPU SQL engine API rebase (cucascade + Sirius origin/dev)
**Researched:** 2026-05-04
**Confidence:** HIGH — all findings sourced from actual commit diffs; no training-data assertions

---

## PR Dependency Order

Before digging into individual PRs, the dependency graph matters for phase sequencing:

```
cucascade #116 (get_table_view ctor)
    └── required before #739 can compile
cucascade #112 (bandwidth profiler)
    └── required before #739 can compile (bumped cucascade to 0cd4a6a)
cucascade #117 (DataBatch RAII)
    └── supersedes #116+#112's data_batch API surface entirely
    └── BREAKS everything #739 migrated (different API — see below)

Sirius #675 (IO framework)
    └── standalone; no dependency on other PRs

Sirius #731 (Scan Manager)
    └── standalone; deletes sirius_parquet_metadata_scan_operator.hpp
    └── must land before #721

Sirius #721 (Pin Tables)
    └── requires #731 (cached_split_provider derives from split_provider)

Sirius #739 (cucascade compat)
    └── adapts to cucascade #112+#116 shape (0cd4a6a pin)
    └── does NOT adapt to cucascade #117 (#117 post-dates #739's pin)
    └── migration done in #739 is PARTIAL — serves as reference only
```

**Critical finding:** #739's migration targets cucascade pin `0cd4a6a` (= #112 + #116, but NOT #117).
The v1.4 rebase must land cucascade #117 as well, which re-breaks every site #739 touched — the
#739 diff is a reference for "what files need touching," but the actual per-site recipe changes
again under #117's accessor model. See the migration recipe section below.

---

## PR #117 — Cucascade DataBatch RAII Model

**Commit:** `73d00c4` (cucascade)
**Classification:** TABLE-STAKES — must absorb to compile. `batch_state`, `data_batch_processing_handle`,
`idata_batch_probe`, and all old transition methods are deleted. Code that calls any of them will not compile.

### Public API Surface

**`data_batch`** (`include/cucascade/data/data_batch.hpp`)

Non-copyable, non-movable. Managed exclusively by `shared_ptr`. Inherits `enable_shared_from_this`.

Lock-free public methods (safe without any accessor):
```cpp
uint64_t get_batch_id() const;
void subscribe();
void unsubscribe();
size_t get_subscriber_count() const;
batch_state get_state() const;          // atomic load; states: idle/read_only/mutable_locked
size_t get_read_only_count() const;     // count of active read_only_data_batch instances
```

Locking transitions (block until lock acquired; caller's `shared_ptr` is NOT consumed):
```cpp
[[nodiscard]] read_only_data_batch to_read_only();
[[nodiscard]] mutable_data_batch   to_mutable();
[[nodiscard]] std::optional<read_only_data_batch> try_to_read_only();   // non-blocking
[[nodiscard]] std::optional<mutable_data_batch>   try_to_mutable();     // non-blocking
```

Static transitions (consume accessors via move, produce a different type):
```cpp
[[nodiscard]] static std::shared_ptr<data_batch> to_idle(read_only_data_batch&& accessor);
[[nodiscard]] static std::shared_ptr<data_batch> to_idle(mutable_data_batch&& accessor);
[[nodiscard]] static mutable_data_batch readonly_to_mutable(read_only_data_batch&& accessor);
[[nodiscard]] static read_only_data_batch mutable_to_readonly(mutable_data_batch&& accessor);
```

Private (only accessible via friend accessor classes):
```cpp
idata_representation* get_data() const;
memory::Tier get_current_tier() const;
memory::memory_space* get_memory_space() const;
void set_data(std::unique_ptr<idata_representation> data);
```

**`read_only_data_batch`** — RAII shared lock, copyable

Holds `shared_lock<shared_mutex>`. Copy acquires a new shared lock; move transfers it.
Members declared in order: `_batch` first (destroyed second), `_lock` first (destroyed first) —
this ordering is load-bearing for mutex destruction safety.

Read accessors (delegate to `data_batch` private interface):
```cpp
uint64_t get_batch_id() const;
memory::Tier get_current_tier() const;
idata_representation* get_data() const;
memory::memory_space* get_memory_space() const;
```

Clone operations:
```cpp
[[nodiscard]] std::shared_ptr<data_batch> clone(uint64_t new_batch_id, rmm::cuda_stream_view stream) const;
template <typename TargetRepresentation>
[[nodiscard]] std::shared_ptr<data_batch> clone_to(representation_converter_registry&, uint64_t new_batch_id,
                                                   const memory::memory_space*, rmm::cuda_stream_view) const;
```

**`mutable_data_batch`** — RAII exclusive lock, move-only

Holds `unique_lock<shared_mutex>`. Same read methods as `read_only_data_batch`, plus:
```cpp
void set_data(std::unique_ptr<idata_representation> data);
template <typename TargetRepresentation>
void convert_to(representation_converter_registry&, const memory::memory_space*, rmm::cuda_stream_view);
[[nodiscard]] std::shared_ptr<data_batch> clone(uint64_t new_batch_id, rmm::cuda_stream_view) const;
template <typename TargetRepresentation>
[[nodiscard]] std::shared_ptr<data_batch> clone_to(...) const;
```

**`data_repository` API changes**

Old: `pop_data_batch(batch_state target)` — blocks until batch reaches target state.
New: `pop_next_data_batch()` — non-blocking FIFO pop, any state.
`pop_data_batch_by_id()` and `get_data_batch_by_id()` no longer take `target_state` and no longer block.

### Threading / Lock Model

Underlying primitive: `std::shared_mutex _rw_mutex` inside `data_batch`.

- Shared lock (`read_only_data_batch`): allows concurrent readers; blocks writers.
- Exclusive lock (`mutable_data_batch`): blocks all readers and writers.
- `_state` and `_read_only_count` are `std::atomic` — safe to read lock-free at any time.
- `readonly_to_mutable` transition is NOT atomic — it releases the shared lock, then acquires
  the exclusive lock. Another writer could interpose between the two operations.

No internal condition variable — the old blocking `wait_to_*` API is entirely deleted. All waiting
is now caller's responsibility (spin on `try_to_*`, or block on `to_read_only()`/`to_mutable()`).

### Invariants Callers Must Guarantee

1. Every `data_batch` MUST be managed by `shared_ptr` before any `to_read_only()` / `to_mutable()`
   call — these methods call `shared_from_this()`, which throws `bad_weak_ptr` on a stack object.
2. Accessor lifetime must not outlive the `data_batch` it was obtained from. (`shared_ptr` in
   the accessor keeps the batch alive — this is automatic when used correctly.)
3. The `readonly_to_mutable` transition is not atomic. Callers must not rely on exclusive ownership
   being maintained through the gap between releasing the shared lock and acquiring the exclusive lock.
4. `to_idle(accessor&&)` consumes the accessor — calling any method on it afterward is UB.

### Deleted API (compile-breaking removal list)

- `batch_state` values `task_created`, `processing`, `in_transit` → replaced by `read_only`, `mutable_locked`
- `idata_batch_probe` class (removed entirely)
- `data_batch_processing_handle` class (removed entirely)
- `lock_for_processing_status` enum (removed entirely)
- `lock_for_processing_result` struct (removed entirely)
- `data_batch::wait_to_create_task()`
- `data_batch::wait_to_cancel_task()`
- `data_batch::wait_to_lock_for_processing(memory_space_id)`
- `data_batch::wait_to_lock_for_in_transit()`
- `data_batch::wait_to_release_in_transit(optional<batch_state>)`
- `data_batch::try_to_create_task()`
- `data_batch::try_to_cancel_task()`
- `data_batch::try_to_lock_for_processing(memory_space_id)`
- `data_batch::try_to_lock_for_in_transit()`
- `data_batch::try_to_release_in_transit(optional<batch_state>)`
- `data_batch::set_state_change_cv(condition_variable*)`
- `data_batch::get_processing_count()`
- `data_batch::get_task_created_count()`
- `data_batch::convert_to<T>(...)` (moved to `mutable_data_batch`)
- `data_batch::clone(...)` (moved to accessor classes)
- `data_batch::get_data()` (now private, behind RAII)
- `data_batch::get_memory_space()` (now private, behind RAII)
- `data_batch::get_current_tier()` (now private, behind RAII)
- Static entry-point `data_batch::to_read_only(PtrType&&)` variants
- Static entry-point `data_batch::to_mutable(PtrType&&)` variants
- `data_repository::pop_data_batch(batch_state, partition_index)` → replaced by `pop_next_data_batch()`

### Sirius Call-Site Migration Scope

Based on `grep` of current `feature/single-node-multi-gpu2` src/:

| Category | Count | Files |
|----------|-------|-------|
| `pop_data_batch(batch_state::task_created)` calls | 10 | `sirius_physical_operator.cpp`, `sirius_physical_hash_join.cpp` (+4 sites), `sirius_physical_merge_sort.cpp`, `sirius_physical_grouped_aggregate_merge.cpp`, `sirius_physical_table_scan.cpp`, `sirius_physical_ungrouped_aggregate.cpp`, `sirius_physical_top_n.cpp`, `sirius_physical_concat.cpp` (+2 sites) |
| `lock_for_processing` / old in-transit calls | 11 | `batch_lock_utils.hpp` (entire implementation), `convertible_data_batch.hpp` (convert path), `gpu_pipeline_executor.cpp` (try_to_create_task) |
| `get_data()` direct access | ~45 call sites across 26 files | All src/op/*.cpp, src/pipeline/*.cpp, src/creator/*.cpp, data_batch_utils.hpp |
| Old `batch_state` enum values | ~10 sites | Same files as `pop_data_batch` |

**Key migration hubs:**
- `src/include/pipeline/batch_lock_utils.hpp` (129 LOC): entire `lock_or_prepare_batch` must be rewritten for `to_mutable()` + `mutable_data_batch::convert_to()`.
- `src/include/data/convertible_data_batch.hpp`: `try_to_lock_for_in_transit()` replaced by acquiring `mutable_data_batch` via `to_mutable()`.
- `src/include/data/data_batch_utils.hpp` (`get_cudf_table_view`): wraps `batch.get_data()` — must wrap `batch.to_read_only()` instead (note: this is a function taking `const data_batch&`; post-#117 it needs a `shared_ptr<data_batch>` to call `to_read_only()`).
- `src/pipeline/gpu_pipeline_executor.cpp`: `batch->try_to_create_task()` → deleted; post-#117 the concept of "task creation" no longer exists at the cucascade level.

### Migration Recipe

**Pattern 1: Reading batch data (was `get_data()` without lock)**
```cpp
// BEFORE (pre-#117, also what #739 did — still needs lock upgrade for #117)
auto* data = batch->get_data();
auto view = data->cast<cucascade::gpu_table_representation>().get_table_view();

// AFTER (#117)
auto ro = batch->to_read_only();
auto view = ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
// ro released at end of scope
```

**Pattern 2: Mutating batch data (was `convert_to` on data_batch directly)**
```cpp
// BEFORE
batch->convert_to<cucascade::gpu_table_representation>(registry, space, stream);

// AFTER (#117)
auto mut = batch->to_mutable();
mut.convert_to<cucascade::gpu_table_representation>(registry, space, stream);
// lock released when mut goes out of scope
```

**Pattern 3: `pop_data_batch(task_created)` (was blocking until batch hits task_created state)**
```cpp
// BEFORE
auto batch = repo->pop_data_batch(cucascade::batch_state::task_created);

// AFTER (#117)
auto batch = repo->pop_next_data_batch();   // non-blocking; any state
// State filtering is now caller's responsibility
// If the operator needs to wait for data, it must poll or use a different synchronization mechanism
```

**Pattern 4: `lock_or_prepare_batch` rewrite**

The entire `batch_lock_utils.hpp` `lock_or_prepare_batch` function must be rewritten.
The old flow: `try_to_create_task` + `wait_to_lock_for_processing` + handle RAII.
The new flow: `to_mutable()` to get exclusive access, `convert_to<>()` if space mismatch,
release via destructor. The old "processing lock" concept maps cleanly to `mutable_data_batch`.

**Pattern 5: `data_batch_utils.hpp::get_cudf_table_view` (const data_batch& parameter)**

Post-#117, `batch.get_data()` is private. The function signature must change from
`const cucascade::data_batch&` to `read_only_data_batch&` (caller holds the lock):
```cpp
// AFTER
inline cudf::table_view get_cudf_table_view(const cucascade::read_only_data_batch& ro) {
  auto* data = ro.get_data();
  if (!data) { throw std::runtime_error("data_batch has no data representation"); }
  return data->cast<cucascade::gpu_table_representation>().get_table_view();
}
```

### Integration Points

- Cucascade internal: `representation_converter.cpp` updated to use new batch types.
- Sirius: every operator in `src/op/`, `batch_lock_utils.hpp`, `convertible_data_batch.hpp`,
  `gpu_pipeline_executor.cpp`, `task_creator.cpp`, `data_batch_utils.hpp` must be updated.

---

## PR #116 — `gpu_data_representation` from `cudf::table_view`

**Commit:** `47e430e` (cucascade)
**Classification:** TABLE-STAKES (indirectly — #739 depends on it; also `get_table()` → `get_table_view()` API change)

### What Changed

`gpu_table_representation` gains a second constructor:
```cpp
template <typename Owner>
gpu_table_representation(cudf::table_view table_view,
                         Owner&& owner,
                         std::size_t alloc_size,
                         cucascade::memory::memory_space& memory_space);
```

The internal `_table` member changes from `std::unique_ptr<cudf::table>` to:
```cpp
std::variant<std::unique_ptr<cudf::table>, owning_table_view>
```
where `owning_table_view` carries `std::any owner`, `std::size_t alloc_size`, and `cudf::table_view view`.

**Breaking API changes in this PR:**
- `get_table() -> const cudf::table&` removed
- `get_table_view() -> cudf::table_view` added (returns view from either variant arm)
- `release_table()` signature changes: `release_table()` → `release_table(rmm::cuda_stream_view stream)`

**Migration recipe (also what #739 performed):**
```cpp
// BEFORE
auto& table = rep.get_table();
auto view   = table.view();
auto table  = rep.release_table();

// AFTER
auto view  = rep.get_table_view();          // view is already cudf::table_view
auto table = rep.release_table(stream);     // now requires stream for sync
```

All `table.view()` calls on the old `const cudf::table&` become direct use of the `table_view`.

**Scope in current src/:** `grep` shows 14 `.get_table()` sites and 3 `release_table()` sites across
approximately 10 operator `.cpp` files and `data_batch_utils.hpp`. PR #739 already shows the per-site
transformation — the recipe is mechanical.

Note: #116 must land before #117 in the cucascade rebase. The origin/main history confirms this:
`47e430e` (#116) precedes `73d00c4` (#117). Post-#117 the methods on `gpu_table_representation` are
only reachable through `read_only_data_batch::get_data()` or `mutable_data_batch::get_data()`, but
the methods themselves (`get_table_view`, `release_table(stream)`) come from #116.

---

## PR #112 — Memory-Space Bandwidth Profiler

**Commit:** `0cd4a6a` (cucascade)
**Classification:** DIFFERENTIATOR — additive new API; zero Sirius call-sites today; optional uplift

### Public API Surface

`include/cucascade/data/bandwidth_profiler.hpp` (new file, `cucascade::data` namespace):

```cpp
// Result types
struct bandwidth_sample { double gbps; double mean_seconds; size_t bytes_transferred; size_t iterations_timed; };
struct bandwidth_pair_result { memory_space_id src; memory_space_id dst; map<size_t, bandwidth_sample> per_size; bandwidth_sample summary; bool converter_available; string unavailable_reason; };
struct bandwidth_profile { vector<bandwidth_pair_result> pairs;
  double gbps(memory_space_id src, memory_space_id dst) const;
  optional<bandwidth_sample> sample(src, dst, size_bytes) const;
};

// Config
struct bandwidth_profile_config {
  vector<size_t> test_sizes_bytes;       // default: {1, 16, 64, 256} MiB
  size_t warmup_iterations = 3;
  size_t timed_iterations = 10;
  bool measure_disk_pairs = true;
  bool drop_page_cache_between_iters = true;
};

// Entry point
[[nodiscard]] bandwidth_profile measure_bandwidth(
  std::span<memory::memory_space* const> spaces,
  representation_converter_registry& registry,
  bandwidth_profile_config config = {});
```

Companion additions:
- `chunked_resource_info` mixin (new header `include/cucascade/memory/chunked_resource_info.hpp`):
  allocators that hand out fixed-size chunks may inherit this to advertise `max_chunk_bytes()`.
  `fixed_size_host_memory_resource` inherits it. `memory_space::get_chunked_resource_info()` exposes
  the probe.
- `pipeline_io_backend`: internal change to per-device stream/event cache (lazy-created on first use
  per GPU context). This fixes a cross-context cudaErrorInvalidResourceHandle when GPU N > 0 uses
  the disk backend. No public API change; internal fix that unblocks multi-GPU disk I/O.

### Threading / Lock Model

`measure_bandwidth` is a pure init-time function. It is not thread-safe for concurrent invocations
on the same `registry`. Intended to be called once at startup, result stored and passed to routing.

### Integration Points

No Sirius call-sites today. Relevant for v1.4 only if Sirius wants to use bandwidth info for routing
decisions. The `pipeline_io_backend` per-device fix is consumed transparently by any code that calls
cucascade disk converters from GPU N > 0 — this matters for the multi-GPU parquet path.

---

## PR #675 — IO / Prefetching / Caching Framework

**Commit:** `4c0f1ac` (Sirius)
**Classification:** TABLE-STAKES — retires `cucascade_datasource` (the v1.1 stopgap). Not compiling
in isolation with this PR absent does not block the build (old datasource still compiles), but the
v1.4 goal is to retire the stopgap.

### Public API Surface

**`src/include/io/types.hpp`** — constants and base types (`namespace sirius::io`):

```cpp
static constexpr size_t CHUNK_SIZE    = 1UL << 20;   // 1 MiB bounce buffer
static constexpr size_t NUM_CHUNKS    = 32;           // bounce slots per reactor
static constexpr size_t IO_BLOCK_SIZE = 4096;         // O_DIRECT alignment

using io_completion_handler = std::function<void(size_t bytes_transferred, std::exception_ptr)>;

class sirius_io_object : public std::enable_shared_from_this<sirius_io_object> {
  virtual const std::string& raw_file_cache_id() const noexcept = 0;
  virtual size_t size() const noexcept = 0;
};

struct request_context { atomic pending; size_t total_bytes; atomic failed; exception_ptr exc; void chunk_done(); void chunk_failed(exception_ptr); };
template <typename Handle> struct device_read_req { Handle handle; size_t file_off; size_t io_size; size_t data_off; size_t data_size; uint8_t* dst; cudaStream_t stream; int device_id; shared_ptr<request_context> ctx; };
template <typename Handle> struct host_read_req { Handle handle; size_t offset; size_t size; uint8_t* dst; shared_ptr<request_context> ctx; };
```

**`sirius_ioctx`** — abstract shared context:

```cpp
class sirius_ioctx : public enable_shared_from_this<sirius_ioctx> {
  virtual void shutdown() = 0;
  virtual unique_ptr<cudf::io::datasource> make_datasource(shared_ptr<sirius_io_object>) = 0;
  void initialize_cache(buffer_pool& pool, size_t inflight_budget_chunks = 2048);
  prefetching_cache* cache() noexcept;
  virtual size_t host_read(sirius_io_object&, size_t offset, size_t size, uint8_t* dst) = 0;
  virtual unique_ptr<datasource::buffer> host_read(sirius_io_object&, size_t offset, size_t size) = 0;
  // + device_read*, host_read_async, device_read_async, host_read_ranges*
};
```

**`sirius_datasource`** (`src/include/io/sirius_datasource.hpp`):

```cpp
class sirius_datasource : public io_datasource {
  explicit sirius_datasource(shared_ptr<sirius_ioctx> io_ctx, shared_ptr<sirius_io_object> io_object);
  bool supports_device_read() const override;      // always true
  bool is_device_read_preferred(size_t) const override;
  size_t size() const override;
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  unique_ptr<buffer> host_read(size_t, size_t) override;
  future<size_t> host_read_async(size_t, size_t, uint8_t*) override;
  unique_ptr<buffer> device_read(size_t, size_t, rmm::cuda_stream_view) override;
  size_t device_read(size_t, size_t, uint8_t*, rmm::cuda_stream_view) override;
  future<size_t> device_read_async(size_t, size_t, uint8_t*, rmm::cuda_stream_view) override;
};
```

**`uring_ioctx`** — concrete backend (io_uring):
- `uring_io_object` opens two fds per file (buffered fd + O_DIRECT fd)
- One reactor thread per reactor instance, one io_uring ring per reactor
- `SINGLE_ISSUER | DEFER_TASKRUN` attempted, falls back to plain flags
- Device reads go through pinned `cudaHostAllocPortable` bounce slots

**C++20 concepts** (`src/include/io/templated_ioctx.hpp`):
```cpp
template <typename O, typename Handle>
concept io_object_c = derived_from<O, sirius_io_object> && /* host_handle, device_handle */;

template <typename R>
concept io_reactor_c = /* native_handle_type, io_object_type, device_read_req_type, host_read_req_type,
                          enqueue_bulk, host_read, host_read_async, shutdown, static align_to_physical */;
```

**`prefetching_cache`** — pinned-memory, chunk-based:
- 5-bucket tiered LRU; background `jthread` worker + evictor
- Lock-free per-entry atomic state machine: 4-bit state + 28-bit pin count in one `atomic<uint32_t>`
- `admission_control`: RAII slot, fixed budget (default 2 GiB)
- Locking hierarchy: `prefetching_cache::_map_mtx` → `file_entry::mtx` → lock-free `cache_entry` atomics

### Threading Model

- 1 reactor thread per `uring_reactor` instance (owns the io_uring ring)
- 1 background worker `jthread` + 1 evictor `jthread` in `prefetching_cache`
- `device_read_req.device_id` carries the CUDA device index; reactor thread calls `cudaSetDevice(device_id)` before the H2D copy — this is the multi-GPU safety hook
- `cudaHostAllocPortable` bounce slabs — reachable from any CUDA context

### Multi-GPU Adaptation Required

The current `sirius_datasource` is single-ioctx. For multi-GPU:
- Reactor pools should be per-GPU (or per-NUMA node) to avoid cross-device `cudaSetDevice` thrashing
- Prefetching-cache scoping: a shared cache is valid (bounce slabs are Portable), but eviction
  priority should be per-GPU if data locality matters
- Every `device_read` already carries `device_id` in `device_read_req` — the reactor honors it

### Integration Points

Replaces `src/include/io/sirius_cucascade_datasource.hpp` (v1.1 stopgap). Factory site:
`parquet_scan_task.cpp` — wherever `datasource::create(path)` was replaced with
`cucascade_datasource` in v1.1, it now becomes `sirius_datasource` via `uring_ioctx::make_datasource()`.

---

## PR #731 — Scan Manager

**Commit:** `aa0f29a` (Sirius)
**Classification:** TABLE-STAKES — deletes `sirius_parquet_metadata_scan_operator.hpp` (214 LOC);
Sirius will fail to compile if any include of that file remains.

### Public API Surface

**`split_provider`** (abstract, `src/include/scan_manager/split_provider.hpp`):
```cpp
class split_provider {
  virtual std::future<void> start(exec::thread_pool& pool, split_connector& connector) = 0;
};
```

**`split_connector`** (`src/include/scan_manager/split_connector.hpp`):
```cpp
class split_connector {
  void push_split(unique_ptr<op::operator_data> split);   // producer side
  void close();                                            // idempotent
  optional<unique_ptr<op::operator_data>> get_next_split(); // consumer; BLOCKS until available or closed
  bool is_closed() const;
  bool has_more_splits() const;
};
```
Internal: `mutex + condition_variable + deque`. `get_next_split()` blocks — consumer calls it from
a task_creator worker thread.

**`parquet_split_provider`** (`src/include/scan_manager/parquet_split_provider.hpp`):
```cpp
class parquet_split_provider : public split_provider {
  parquet_split_provider(
    vector<sirius::logical_type> returned_types,
    vector<string> file_paths,
    vector<duckdb::ColumnIndex> column_ids,
    vector<duckdb::idx_t> projection_ids,
    vector<string> names,
    size_t scan_output_arity,
    unique_ptr<duckdb::TableFilterSet> table_filter_set,
    vector<duckdb::HivePartitioningIndex> partition_indices,
    size_t approximate_batch_size = config::DEFAULT_SCAN_TASK_BATCH_SIZE,
    size_t max_file_processed     = DEFAULT_MAX_FILE_PROCESSED);

  op::scan::partition_inject_fn_t take_partition_inject_fn();  // move-out; returns empty fn for trivial scans
  future<void> start(exec::thread_pool& pool, split_connector& connector) override;
};
```

**`sirius_scan_manager`** (`src/include/scan_manager/sirius_scan_manager.hpp`):
```cpp
class sirius_scan_manager {
  explicit sirius_scan_manager(exec::thread_pool_config config);

  void prepare_for_query(const sirius::planner::query& query);
  void reset();
  void start();
  void stop();
  void close_all_connectors();       // MUST be called before operators are destroyed on error path
  std::exception_ptr take_driver_error();

 private:
  exec::thread_pool_config _config;
  unique_ptr<exec::thread_pool> _thread_pool;
  vector<pair<sirius_gpu_parquet_scan_operator*, unique_ptr<split_provider>>> _providers;
  thread _driver_thread;
  exception_ptr _driver_error;
};
```

`prepare_for_query` walks query pipelines, creates a `parquet_split_provider` per parquet scan op,
installs a fresh `split_connector` on each operator, appends (op, provider) pair to `_providers`.
`_driver_thread` runs providers SEQUENTIALLY: provider[0].start().get(), then provider[1], etc.

### Threading Model

- `sirius_scan_manager._driver_thread`: one `std::thread` per query, runs providers sequentially.
- Each provider `start()` dispatches N file-batch tasks into `exec::thread_pool`.
- Consumer: `gpu parquet scan operator` calls `split_connector::get_next_split()` — blocks on
  `condition_variable` until splits arrive or connector closes.
- Lifetime hazard: `close_all_connectors()` must be called BEFORE the gpu scan operators are
  destroyed. Do NOT call from `SiriusContext::terminate()` (operators are already torn down by then).

### What It Replaces

Deletes `src/include/op/scan/sirius_parquet_metadata_scan_operator.hpp` (214 LOC removed).
The metadata scan logic moves into `parquet_split_provider::run_batch()`.

### Multi-GPU Impact

Phase 9's `_batch_gpu_affinity` recording and Phase 14's SCHED-RR counter (current branch) live
in `parquet_scan_task.cpp` and `duckdb_scan_executor.cpp`. These are NOT in the deleted file.
MGPU-07 adaptive scan lives in `duckdb_scan_executor::select_target_gpu` — also unaffected.

However, Phase 13's stream-lineage (`writer_stream` / `writer_event`) was anchored in
`sirius_parquet_metadata_scan_operator.cpp`. That file is deleted by #731. Stream-lineage
must be re-anchored in `parquet_split_provider::run_batch()` during v1.4 integration.

---

## PR #721 — Pin Tables in GPU Memory

**Commit:** `cdd6864` (Sirius)
**Classification:** DIFFERENTIATOR — builds on #731; additive DDL feature

### Public API Surface

**`duckdb::PinTableArgs`** (`src/include/pin_table.hpp`):
```cpp
struct PinTableArgs { string path; string tier; string name; optional<vector<string>> cols; optional<int64_t> n_rows; };
void pin_table_to(const PinTableArgs& args);
void unpin_table_to(const std::string& name);
```

SQL DDL (registered in `sirius_extension.cpp`):
```sql
CALL pin_table('/path/to/file.parquet', name='myname', tier='gpu', cols=['col1', 'col2']);
CALL unpin_table('myname');
```

**`cached_split_provider`** (`src/include/scan_manager/cached_split_provider.hpp`):
```cpp
class cached_split_provider : public split_provider {
  cached_split_provider(
    vector<vector<shared_ptr<cudf::column>>> columns_per_request,  // columns_per_request[d] = chunk vector for D-position d
    cucascade::memory::memory_space& memory_space,
    shared_ptr<duckdb::Expression> filter_expression,
    op::scan::partition_inject_fn_t inject_fn);

  future<void> start(exec::thread_pool& pool, split_connector& connector) override;
};
```

`sirius_scan_manager` is extended with:
- `_pinned_entries` map (file_paths → column chunks + memory_space pointer)
- `create_provider_for()` checks `_pinned_entries` first; falls through to `parquet_split_provider` if:
  - no entry matches the file paths, OR
  - the pinned entry has hive partitions (not supported for cached path today)

### Threading Model

`cached_split_provider::start()` runs synchronously on the caller thread (or pool thread) — no
metadata scan needed. Produces one `scan_cached_operator_data` per chunk, zero-copy (view-backed
`gpu_data_representation` via the PR #116 constructor).

### Dependency on #731

`cached_split_provider` inherits from `split_provider`, which is defined in #731.
`sirius_scan_manager` extended scan manager introduced in #731.
PR #721 cannot compile without #731 already landed.

### Multi-GPU Impact

Current `cached_split_provider` always pins to "the first GPU memory space." In multi-GPU,
per-GPU pinned copies or pointer-sharing with `cudaIpcMemHandle` would be needed. This is
documented as a current limitation in the PR commit message ("tier is accepted; current
implementation always pins to the first GPU memory space").

---

## PR #739 — Cucascade GPU Table Compat (Reference Migration)

**Commit:** `468f6e1` (Sirius)
**Classification:** Reference only for migration scope; NOT the target state for v1.4

**Critical caveat:** #739 pins cucascade to `0cd4a6a` (#112 + #116, NOT #117). It adapts Sirius
to the #116 API change (`get_table()` → `get_table_view()`, `release_table()` → `release_table(stream)`)
but leaves `batch->get_data()` calls intact (still compiles because #117 has not landed yet).

### What #739 Did (reference for mechanical migrations)

Scope: 12 src files + 16 test files, 149 insertions / 160 deletions.

**In `data_batch_utils.hpp:54`:**
```cpp
// BEFORE
return data->cast<cucascade::gpu_table_representation>().get_table();
// AFTER
return data->cast<cucascade::gpu_table_representation>().get_table_view();
```

**In operator `.cpp` files (all 10 operators):**
```cpp
// BEFORE pattern
auto& table = batch->get_data()->cast<cucascade::gpu_table_representation>().get_table();
auto view   = table.view();

// AFTER pattern (#739, still pre-#117)
auto view = batch->get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
```

**`release_table` migration (3 sites):**
```cpp
auto table = gpu_rep.release_table();        // BEFORE
auto table = gpu_rep.release_table(stream);  // AFTER (#739)
```

**In tests:** `.get_table()` → `.get_table_view()`, `.view()` calls removed since `table_view` is already a view.

### v1.4 Migration: What #739 Started, #117 Finishes

After absorbing #117, every `batch->get_data()` call in the #739-migrated files needs a further
upgrade to acquire the appropriate accessor first. The #739 diff identifies WHICH files need
touching; the recipes above in the #117 section describe HOW.

---

## Feature Dependencies (Phase Ordering Constraints)

```
[cucascade #116] ──required-by──> [cucascade #117]   (117 lands after 116 in origin/main)
[cucascade #112] ──required-by──> [cucascade #117]   (117 lands after 112 in origin/main)

[cucascade rebase: #112 + #116 + #117]
    └── must complete before DataBatch API migration (Phase 18 work)

[Sirius #739 reference scan]
    └── identifies files; must be done before committing Phase 18

[Sirius #731 (Scan Manager)] ──required-by──> [Sirius #721 (Pin Tables)]

[Sirius #675 (IO Framework)] ──independent──> can land in parallel with #731/#721

[DataBatch API migration (Phase 18)]
    └── depends on cucascade rebase (new accessor types must exist)
    └── blocks IO Framework adoption (sirius_datasource creates batches; those batches need the new API)
    └── does NOT block Scan Manager landing (scan manager doesn't touch batch data directly)
```

### Phase 18 vs Phase 19 Ordering

**Phase 18 (DataBatch API migration) should precede Phase 19 (IO Framework adoption).**

Rationale:
- `sirius_datasource::device_read()` produces data that becomes `data_batch` objects
- Code that creates batches from IO reads will call `mutable_data_batch` setters post-#117
- If IO Framework lands on top of un-migrated batch API, the result doesn't compile
- Conversely, DataBatch migration can be validated independently (batch creation from existing
  parquet path, existing operators) before IO plumbing changes

### Can #731 + #721 Land Together?

Yes — #721 depends on #731 but adds no new compile-breaking dependencies. A single phase covering
both is feasible. The only risk is that `cached_split_provider`'s multi-GPU limitation (always
pins to GPU 0) creates a correctness gap on multi-GPU hardware; this should be flagged in the phase
plan as a known limitation requiring a follow-up.

---

## Feature Prioritization Matrix

| Feature | Compile Blocking | Migration Effort | Phase Dependency |
|---------|-----------------|-----------------|-----------------|
| Cucascade #116 (`get_table_view`) | YES — fails if `get_table()` called on post-#116 cucascade | LOW — mechanical 14 sites (PR #739 shows recipe) | Must precede #117 |
| Cucascade #112 (bandwidth profiler) | NO — additive | NONE | Must precede #117 in cucascade history |
| Cucascade #117 (DataBatch RAII) | YES — all 21 old API calls removed | HIGH — ~21+ call sites, batch_lock_utils.hpp full rewrite, convertible_data_batch rework, 26 files with get_data() | After #116 and #112 |
| Sirius #675 (IO Framework) | NO — old datasource still compiles | MEDIUM — retire cucascade_datasource, adapt uring_ioctx for multi-GPU | After #117 (batches it creates need new accessor model) |
| Sirius #731 (Scan Manager) | YES — deletes metadata scan operator hpp | MEDIUM — 25 files changed; Phase 13 stream-lineage must be re-anchored | After cucascade rebase; before #721 |
| Sirius #721 (Pin Tables) | NO — additive (depends on #731) | LOW — 22 files; multi-GPU limitation accepted | After #731 |
| Sirius #739 (reference) | Not landing as-is — reference only | N/A | Use as migration reference for #117 phase |

---

## Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|---|---|---|
| Landing #739 as-is onto v1.4 | It targets pre-#117 cucascade. After #117, every `batch->get_data()` call it left in place breaks. | Use #739 as a file/site index; apply the full #117 accessor pattern from scratch at each site. |
| Sharing `sirius_ioctx` across GPUs without per-device reactor pools | Reactor thread calls `cudaSetDevice` per request — heavy overhead at scale | Create per-GPU `uring_ioctx` instances (one reactor pool per GPU) |
| Calling `close_all_connectors()` from `SiriusContext::terminate()` | DuckDB tears down the physical plan (operators) before calling terminate; the scan operator pointers in `_providers` dangle | Call `close_all_connectors()` from the engine's catch/drain path, before the task pool joins |
| Caching tables to GPU 0 only in multi-GPU | `pin_table` always pins to "first GPU" — creates a GPU-0 hot-spot | Follow-up: per-GPU pinned copies or lazy cache-on-demand |

---

## Sources

- `git -C cucascade show 73d00c4` — PR #117 full diff
- `git -C cucascade show 47e430e` — PR #116 full diff
- `git -C cucascade show 0cd4a6a` — PR #112 full diff
- `git show 4c0f1ac` — PR #675 full diff
- `git show aa0f29a` — PR #731 full diff
- `git show cdd6864` — PR #721 full diff
- `git show 468f6e1` — PR #739 full diff (migration reference)
- `find/grep` on `feature/single-node-multi-gpu2` src/ for call-site counts

---
*Feature research for: v1.4 "Rebase After DataBatch Changes" — new API surfaces*
*Researched: 2026-05-04*
