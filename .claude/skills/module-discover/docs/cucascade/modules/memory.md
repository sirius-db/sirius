# Memory Module

**Status**: USED
**Path**: `cucascade/include/cucascade/memory/`
**Headers we include**: `common.hpp`, `config.hpp`, `fixed_size_host_memory_resource.hpp`, `host_table.hpp`, `memory_reservation.hpp`, `memory_reservation_manager.hpp`, `memory_space.hpp`, `numa_region_pinned_host_allocator.hpp`, `reservation_aware_resource_adaptor.hpp`, `reservation_manager_configurator.hpp`, `small_pinned_host_memory_resource.hpp`, `stream_pool.hpp`, `topology_discovery.hpp`

## Summary

The memory module implements cuCascade's tiered memory management system. It provides memory spaces (GPU, Host, Disk), a reservation system to prevent OOM, and resource adaptors that integrate with RMM. Sirius uses this module to manage all GPU and pinned host memory, schedule downgrades when GPU memory is constrained, and provide per-stream memory tracking.

## API Reference

### `cucascade::memory::Tier` (enum)

**Header**: `cucascade/memory/common.hpp`

```cpp
enum class Tier : int32_t { GPU, HOST, DISK, SIZE };
```

**Description**: Identifies the memory tier. `SIZE` is a sentinel for iteration.

**Our usage**:
- `src/memory/sirius_memory_reservation_manager.cpp:35` — `get_memory_spaces_for_tier(Tier::GPU)` to find GPU spaces
- Used throughout pipeline code to check data batch tier

---

### `cucascade::memory::memory_space_id`

**Header**: `cucascade/memory/common.hpp`

```cpp
struct memory_space_id {
  Tier tier;
  int32_t device_id;
  explicit memory_space_id(Tier t, int32_t d_id);
  auto operator<=>(const memory_space_id&) const noexcept = default;
  std::size_t uuid() const noexcept;
};
```

**Description**: Unique identifier for a memory space (tier + device ID). Supports three-way comparison and hashing.

**Our usage**:
- `src/pipeline/gpu_pipeline_task.cpp:27` — passed to `wait_to_lock_for_processing()` to request specific memory space
- `src/include/pipeline/task_request.hpp:24` — task requests specify target memory space

---

### `cucascade::memory::memory_space`

**Header**: `cucascade/memory/memory_space.hpp`

```cpp
class memory_space {
public:
  explicit memory_space(const gpu_memory_space_config& config);
  explicit memory_space(const host_memory_space_config& config);
  explicit memory_space(const disk_memory_space_config& config);

  memory_space_id get_id() const noexcept;
  Tier get_tier() const noexcept;
  int get_device_id() const noexcept;

  // Reservation creation
  std::unique_ptr<reservation> make_reservation_or_null(size_t size);
  std::unique_ptr<reservation> make_reservation_upto(size_t size);
  std::unique_ptr<reservation> make_reservation(size_t size);

  // Stream acquisition
  rmm::cuda_stream_view acquire_stream() const;

  // State queries
  std::size_t get_active_reservation_count() const;
  bool should_downgrade_memory() const;
  bool should_stop_downgrading_memory() const;
  size_t get_amount_to_downgrade() const;
  size_t get_available_memory(rmm::cuda_stream_view stream) const;
  size_t get_available_memory() const;
  size_t get_total_reserved_memory() const;
  size_t get_max_memory() const noexcept;

  // Allocator access
  rmm::mr::device_memory_resource* get_default_allocator() const noexcept;
  template <typename T> T* get_memory_resource_as() const noexcept;
  template <Tier TIER> auto* get_memory_resource_of() const noexcept;

  std::string to_string() const;
  void shutdown();
};
```

**Description**: Represents a single memory region (e.g., one GPU, one NUMA node). Owns the underlying memory resource and provides reservation/query interface. The `should_downgrade_memory()` method signals the downgrade executor.

**Our usage**:
- `src/pipeline/gpu_pipeline_task.cpp:27` — accesses memory space from data batches
- `src/include/pipeline/gpu_pipeline_executor.hpp:30` — pipeline executor queries available memory
- `src/data/host_parquet_representation_converters.cpp:27` — gets memory space for host allocations

---

### `cucascade::memory::reservation`

**Header**: `cucascade/memory/memory_reservation.hpp`

```cpp
class reservation {
public:
  static std::unique_ptr<reservation> create(memory_space& space,
                                             std::unique_ptr<reserved_arena> arena);
  size_t size() const noexcept;
  Tier tier() const noexcept;
  int device_id() const noexcept;

  rmm::mr::device_memory_resource* get_memory_resource() const noexcept;
  const memory_space& get_memory_space() const noexcept;

  template <typename T>
    requires std::derived_from<T, rmm::mr::device_memory_resource>
  T* get_memory_resource_as() const noexcept;

  template <Tier TIER>
  auto* get_memory_resource_of() const noexcept;

  bool grow_by(size_t additional_bytes);
  void shrink_to_fit();
};
```

**Description**: A lease on memory within a memory space. Wraps a `reserved_arena` and provides access to the underlying memory resource for allocation. Automatically releases the reservation on destruction.

**Our usage**:
- `src/include/pipeline/gpu_pipeline_task.hpp:28` — each pipeline task holds a reservation
- `src/include/op/scan/duckdb_scan_task.hpp:39` — scan tasks hold reservations
- `src/include/parallel/task.hpp:25` — base task class includes reservation
- `src/include/pipeline/gpu_pipeline_executor.hpp:29` — executor manages reservations

---

### `cucascade::memory::memory_reservation_manager`

**Header**: `cucascade/memory/memory_reservation_manager.hpp`

```cpp
class memory_reservation_manager {
public:
  explicit memory_reservation_manager(std::vector<memory_space_config> configs);
  ~memory_reservation_manager();

  // Reservation interface
  std::unique_ptr<reservation> request_reservation(
    const reservation_request_strategy& request, size_t size);

  // Memory space access
  const memory_space* get_memory_space(Tier tier, int32_t device_id) const;
  memory_space* get_memory_space(Tier tier, int32_t device_id);
  std::span<const memory_space*> get_memory_spaces_for_tier(Tier tier) const;
  std::span<const memory_space*> get_all_memory_spaces() const noexcept;

  // Aggregated queries
  size_t get_available_memory_for_tier(Tier tier) const;
  size_t get_total_reserved_memory_for_tier(Tier tier) const;
  size_t get_active_reservation_count_for_tier(Tier tier) const;
  size_t get_total_available_memory() const;
  size_t get_total_reserved_memory() const;
  size_t get_active_reservation_count() const;
  void shutdown();
};
```

**Description**: Top-level manager for all memory spaces. Constructs memory spaces from configs, handles reservation requests using pluggable strategies, and provides aggregate memory queries.

**Reservation request strategies** (same header):
- `any_memory_space_in_tier` — any space in a given tier
- `any_memory_space_in_tier_with_preference` — prefer a specific device
- `any_memory_space_in_tiers` — try multiple tiers in order
- `specific_memory_space` — exact (tier, device_id)
- `any_memory_space_to_downgrade` — find downgrade target
- `any_memory_space_to_upgrade` — find upgrade target

**Our usage**:
- `src/memory/sirius_memory_reservation_manager.cpp:33` — `sirius_memory_reservation_manager` inherits from this
- `src/memory/sirius_memory_reservation_manager.cpp:35` — `get_memory_spaces_for_tier(Tier::GPU)` at init
- `src/op/scan/parquet_scan_task.cpp:31` — scan tasks request reservations
- `src/include/op/scan/duckdb_scan_executor.hpp:29` — scan executor holds manager reference

---

### `cucascade::memory::reservation_aware_resource_adaptor`

**Header**: `cucascade/memory/reservation_aware_resource_adaptor.hpp`

```cpp
class reservation_aware_resource_adaptor : public rmm::mr::device_memory_resource {
public:
  explicit reservation_aware_resource_adaptor(
    memory_space_id space_id, rmm::device_async_resource_ref upstream,
    std::size_t capacity,
    std::unique_ptr<reservation_limit_policy> policy = nullptr,
    std::unique_ptr<oom_handling_policy> oom_policy = nullptr,
    AllocationTrackingScope tracking = AllocationTrackingScope::PER_STREAM,
    cudaMemPool_t pool_handle = nullptr);

  // Memory queries
  rmm::device_async_resource_ref get_upstream_resource() const noexcept;
  std::size_t get_available_memory() const noexcept;
  std::size_t get_available_memory(rmm::cuda_stream_view stream) const noexcept;

  // Allocation tracking
  std::size_t get_allocated_bytes(rmm::cuda_stream_view stream) const;
  std::size_t get_peak_allocated_bytes(rmm::cuda_stream_view stream) const;
  std::size_t get_total_allocated_bytes() const;
  std::size_t get_total_reserved_bytes() const;

  // Reservation management
  std::unique_ptr<reserved_arena> reserve(std::size_t bytes,
                                          std::unique_ptr<event_notifier> notifier = nullptr);
  std::unique_ptr<reserved_arena> reserve_upto(std::size_t bytes,
                                               std::unique_ptr<event_notifier> notifier = nullptr);
  bool attach_reservation_to_tracker(rmm::cuda_stream_view stream,
                                     std::unique_ptr<reservation> reserved,
                                     std::unique_ptr<reservation_limit_policy> policy = nullptr,
                                     std::unique_ptr<oom_handling_policy> oom_policy = nullptr);
  void reset_stream_reservation(rmm::cuda_stream_view stream);
};
```

**Description**: RMM memory resource adaptor that enforces reservation-based allocation. Wraps an upstream allocator and tracks per-stream (or per-thread) allocations against reserved amounts. When allocation exceeds reservation, the `reservation_limit_policy` is invoked. On OOM from upstream, the `oom_handling_policy` handles it.

**Our usage**:
- `src/pipeline/gpu_pipeline_task.cpp:28` — pipeline tasks use this for reservation-aware GPU allocation
- `test/cpp/pipeline/test_gpu_pipeline_executor.cpp:28` — tests create adaptors
- `test/cpp/pipeline/test_oom_reschedule.cpp:31` — OOM handling tests

---

### `cucascade::memory::fixed_size_host_memory_resource`

**Header**: `cucascade/memory/fixed_size_host_memory_resource.hpp`

```cpp
class fixed_size_host_memory_resource : public rmm::mr::device_memory_resource {
public:
  explicit fixed_size_host_memory_resource(
    int device_id, rmm::device_async_resource_ref upstream_mr,
    std::size_t mem_limit, std::size_t capacity,
    std::size_t block_size = 1 << 20,    // 1MB default
    std::size_t pool_size = 128,
    std::size_t initial_pools = 4);

  // Queries
  std::size_t get_total_allocated_bytes() const noexcept;
  std::size_t get_available_memory() const noexcept;
  std::size_t get_block_size() const noexcept;
  std::size_t get_free_blocks() const noexcept;
  std::size_t get_total_blocks() const noexcept;
  std::size_t get_total_reserved_bytes() const noexcept;

  // Reservations
  std::unique_ptr<reserved_arena> reserve(std::size_t bytes,
                                          std::unique_ptr<event_notifier> notifier = nullptr);
  std::unique_ptr<reserved_arena> reserve_upto(std::size_t bytes,
                                               std::unique_ptr<event_notifier> notifier = nullptr);

  // Block allocation
  fixed_multiple_blocks_allocation allocate_multiple_blocks(std::size_t total_bytes,
                                                           reservation* res = nullptr);
};

using fixed_multiple_blocks_allocation = std::unique_ptr<multiple_blocks_allocation>;
```

**Description**: Fixed-block-size host memory allocator. Allocates pinned host memory in fixed-size blocks (default 1MB) from pools. The `multiple_blocks_allocation` type provides a scatter-gather view over multiple blocks. Used for all host/CPU data storage.

**Our usage**:
- `src/sirius_context.cpp:31` — creates the host memory resource at startup
- `src/data/host_parquet_representation_converters.cpp:26` — uses for host allocation during conversion
- `src/include/data/host_parquet_representation.hpp:21` — host parquet data stored in blocks
- `src/include/op/scan/duckdb_scan_task.hpp:37` — scan tasks allocate host blocks
- `src/include/op/result/host_table_chunk_reader.hpp:35` — result reader uses block allocator

---

### `cucascade::memory::small_pinned_host_memory_resource`

**Header**: `cucascade/memory/small_pinned_host_memory_resource.hpp`

```cpp
class small_pinned_host_memory_resource : public rmm::mr::device_memory_resource {
public:
  static constexpr std::size_t MAX_SLAB_SIZE = 8192;
  static constexpr std::array<std::size_t, 5> SLAB_SIZES{512, 1024, 2048, 4096, 8192};

  explicit small_pinned_host_memory_resource(fixed_size_host_memory_resource& upstream);
  static std::size_t slab_index_for(std::size_t bytes) noexcept;
};
```

**Description**: Slab allocator for small pinned host allocations (≤8KB). Carves small allocations from larger blocks obtained from `fixed_size_host_memory_resource`. Reduces fragmentation for metadata and small buffers.

**Our usage**:
- `src/sirius_context.cpp:32` — created alongside main host memory resource
- `test/cpp/data/test_host_parquet_representation.cpp:34` — used in host data tests

---

### `cucascade::memory::reservation_manager_configurator`

**Header**: `cucascade/memory/reservation_manager_configurator.hpp`

```cpp
class reservation_manager_configurator {
public:
  // GPU config (builder pattern — all return *this)
  auto& set_number_of_gpus(std::size_t n_gpus);
  auto& set_gpu_ids(const std::vector<int>& gpu_ids);
  auto& set_gpu_usage_limit(std::size_t bytes);
  auto& set_usage_limit_ratio_per_gpu(double fraction);
  auto& set_reservation_limit_per_gpu(size_t bytes);
  auto& set_reservation_fraction_per_gpu(double fraction);
  auto& set_downgrade_fractions_per_gpu(double start, double end);
  auto& track_reservation_per_stream(bool enable);
  auto& set_gpu_memory_resource_factory(DeviceMemoryResourceFactoryFn fn);

  // Host config
  auto& use_host_per_gpu();
  auto& use_host_per_numa();
  auto& set_total_host_capacity(std::size_t bytes);
  auto& set_per_host_capacity(std::size_t bytes);
  auto& set_downgrade_fractions_per_host(double start, double end);
  auto& set_host_memory_resource_factory(DeviceMemoryResourceFactoryFn fn);
  auto& set_host_pool_features(std::size_t chunk_size, std::size_t block_size,
                               std::size_t initial_block_count);

  // Disk config
  auto& set_disk_mounting_point(int uuid, std::size_t capacity, std::string mounting_point);

  // Build
  std::vector<memory_space_config> build(const system_topology_info& topology) const;
  std::vector<memory_space_config> build() const;
};
```

**Description**: Builder pattern for constructing `memory_space_config` vectors to initialize `memory_reservation_manager`. Supports multi-GPU, NUMA-aware host memory, and disk tiers.

**Our usage**:
- `src/sirius_config.cpp:23` — builds memory configs from Sirius runtime settings
- `test/cpp/operator/operator_test_utils.hpp:38` — tests use configurator to set up memory
- `test/cpp/creator/test_task_creator.cpp:27` — task creator tests
- `test/cpp/expression_executor/test_gpu_expression_executor.cpp:23` — expression executor tests

---

### `cucascade::memory::exclusive_stream_pool`

**Header**: `cucascade/memory/stream_pool.hpp`

```cpp
class borrowed_stream {
public:
  rmm::cuda_stream_view get() const noexcept;
  operator rmm::cuda_stream_view() const;
  void reset() noexcept;
};

class exclusive_stream_pool {
public:
  enum class stream_acquire_policy { GROW, BLOCK };
  static constexpr std::size_t default_size{16};

  explicit exclusive_stream_pool(rmm::cuda_device_id device_id = {},
                                 std::size_t pool_size = default_size,
                                 rmm::cuda_stream::flags flags = rmm::cuda_stream::flags::non_blocking);
  borrowed_stream acquire_stream(stream_acquire_policy policy = stream_acquire_policy::BLOCK) noexcept;
  std::size_t size() const noexcept;
};
```

**Description**: Pool of exclusive CUDA streams. `acquire_stream()` returns a `borrowed_stream` RAII handle that returns the stream to the pool on destruction. Supports blocking or growing policy when pool is exhausted.

**Our usage**:
- `src/pipeline/gpu_pipeline_executor.cpp:20` — pipeline executor acquires streams for GPU work
- `src/include/op/scan/duckdb_scan_executor.hpp:30` — scan executor uses stream pool

---

### `cucascade::memory::topology_discovery`

**Header**: `cucascade/memory/topology_discovery.hpp`

```cpp
struct gpu_topology_info {
  unsigned int id;
  std::string name, pci_bus_id, uuid;
  int numa_node;
  std::string cpu_affinity_list;
  std::vector<int> cpu_cores, memory_binding;
  std::vector<std::string> network_devices;
};

struct system_topology_info {
  std::string hostname;
  unsigned int num_gpus;
  int num_numa_nodes, num_network_devices;
  std::vector<gpu_topology_info> gpus;
  std::vector<network_device_info> network_devices;
  std::vector<storage_device_info> storage_devices;
};

class topology_discovery {
public:
  bool discover();
  system_topology_info const& get_topology() const;
  bool is_discovered() const;
};
```

**Description**: Discovers system hardware topology — GPUs, NUMA nodes, NIC devices, and storage. Used at startup to configure memory spaces based on actual hardware.

**Our usage**:
- `src/include/sirius_config.hpp:25` — sirius config uses topology for memory setup
- `src/include/pipeline/task_scheduler.hpp:33` — task scheduler reads topology

---

### `cucascade::memory::host_table_allocation` / `column_metadata`

**Header**: `cucascade/memory/host_table.hpp`

```cpp
struct column_metadata {
  cudf::type_id type_id;
  cudf::size_type num_rows, null_count;
  int32_t scale;
  bool has_null_mask;
  std::size_t null_mask_offset, null_mask_size;
  bool has_data;
  std::size_t data_offset, data_size;
  std::vector<column_metadata> children;
  bool is_synthetic_empty_offsets = false;
};

struct host_table_allocation {
  fixed_multiple_blocks_allocation allocation;
  std::vector<column_metadata> columns;
  std::size_t data_size;
};
```

**Description**: Describes a cuDF table stored in host memory blocks. `column_metadata` stores per-column layout information (offsets, sizes, null masks). The actual data lives in the `multiple_blocks_allocation`.

**Our usage**:
- `src/include/op/scan/duckdb_scan_task.hpp:38` — scan tasks create host tables
- `src/include/op/result/host_table_chunk_reader.hpp:36` — reads host table chunks for output

---

### Memory Config Types

**Header**: `cucascade/memory/config.hpp`

```cpp
struct gpu_memory_space_config {
  int device_id{-1};
  double reservation_limit_fraction{0.9};
  double downgrade_trigger_fraction{0.75};
  double downgrade_stop_fraction{0.65};
  std::size_t memory_capacity{0};
  bool per_stream_reservation{true};
  DeviceMemoryResourceFactoryFn mr_factory_fn{nullptr};

  Tier tier() const;
  std::size_t reservation_limit() const;
  std::size_t downgrade_trigger_threshold() const;
  std::size_t downgrade_stop_threshold() const;
};

struct host_memory_space_config { /* similar fields for host */ };
struct disk_memory_space_config { /* disk tier config */ };

using memory_space_config = std::variant<std::monostate,
                                         gpu_memory_space_config,
                                         host_memory_space_config,
                                         disk_memory_space_config>;
```

**Description**: Configuration structs for each memory tier. Downgrade fractions control when the downgrade executor triggers (at 75% usage) and stops (at 65% usage).

**Our usage**:
- `src/sirius_config.cpp:22` — builds configs from runtime settings
- `src/include/sirius_config.hpp:24` — stores config as member

---

### Reservation Limit Policies

**Header**: `cucascade/memory/memory_reservation.hpp`

```cpp
class reservation_limit_policy { /* abstract */ };
class ignore_reservation_limit_policy : public reservation_limit_policy { /* no-op */ };
class fail_reservation_limit_policy : public reservation_limit_policy { /* throws */ };
class increase_reservation_limit_policy : public reservation_limit_policy {
public:
  increase_reservation_limit_policy();
  explicit increase_reservation_limit_policy(double padding_factor, bool allow_beyond_limit = false);
};

std::unique_ptr<reservation_limit_policy> make_default_reservation_limit_policy();
```

**Description**: Policies for what happens when a stream's allocation exceeds its reservation. `increase_reservation_limit_policy` tries to grow the reservation with optional padding.

---

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `disk_access_limiter` | `disk_access_limiter.hpp` | Reservation-based disk I/O limiter with file-backed arenas |
| `null_device_memory_resource` | `null_device_memory_resource.hpp` | No-op memory resource (all allocations return nullptr) |
| `oom_handling_policy` | `oom_handling_policy.hpp` | Abstract OOM handler; `throw_on_oom_policy` rethrows (used internally by `reservation_aware_resource_adaptor` but not directly included by Sirius) |
| `notification_channel` | `notification_channel.hpp` | Thread-safe notification primitive with `wait()`/`post()` semantics (used internally by reservation system) |
| `host_table_packed_allocation` | `host_table_packed.hpp` | Packed cuDF table in host blocks (used via `host_data_packed_representation` but header not directly included) |
| `cucascade_out_of_memory` | `error.hpp` | Custom OOM exception with allocation context metadata |
