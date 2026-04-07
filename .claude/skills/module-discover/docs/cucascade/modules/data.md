# Data Module

**Status**: USED
**Path**: `cucascade/include/cucascade/data/`
**Headers we include**: `common.hpp`, `cpu_data_representation.hpp`, `data_batch.hpp`, `data_repository.hpp`, `data_repository_manager.hpp`, `gpu_data_representation.hpp`, `representation_converter.hpp`

## Summary

The data module provides the core data abstraction layer for cuCascade. It defines how data is represented across memory tiers (GPU tables, CPU host tables), how batches of data are managed with thread-safe state machines, and how repositories store and dispatch batches between pipeline stages. This is the most heavily used cuCascade module in Sirius — virtually every operator and pipeline component touches `data_batch` or `gpu_table_representation`.

## API Reference

### `cucascade::idata_representation` (abstract base)

**Header**: `cucascade/data/common.hpp`

```cpp
class idata_representation {
public:
  idata_representation(cucascade::memory::memory_space& memory_space);
  virtual ~idata_representation() = default;

  memory::Tier get_current_tier() const;
  int get_device_id() const;
  cucascade::memory::memory_space& get_memory_space();
  const cucascade::memory::memory_space& get_memory_space() const;

  virtual std::size_t get_size_in_bytes() const = 0;
  virtual std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) = 0;

  template <class TargetType>
    requires std::derived_from<TargetType, idata_representation>
  TargetType& cast();

  template <class TargetType>
    requires std::derived_from<TargetType, idata_representation>
  const TargetType& cast() const;
};
```

**Description**: Base class for all data representations. Tracks which memory tier and space the data lives in. Use `cast<T>()` to downcast to concrete types.

**Our usage**:
- `src/expression_executor/gpu_expression_executor.cpp:253` — `input_batch->get_data()->cast<cucascade::gpu_table_representation>()` to access cudf::table
- `src/pipeline/gpu_pipeline_task.cpp` — checks tier and casts between GPU/CPU representations

---

### `cucascade::gpu_table_representation`

**Header**: `cucascade/data/gpu_data_representation.hpp`

```cpp
class gpu_table_representation : public idata_representation {
public:
  gpu_table_representation(std::unique_ptr<cudf::table> table,
                           cucascade::memory::memory_space& memory_space);

  std::size_t get_size_in_bytes() const override;
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;
  const cudf::table& get_table() const;
  std::unique_ptr<cudf::table> release_table();
};
```

**Description**: Wraps a `cudf::table` on GPU memory. Primary data representation for all GPU pipeline operations.

**Our usage**:
- `src/expression_executor/gpu_expression_executor.cpp:253` — `cast<gpu_table_representation>()` then `.get_table()` to access cudf columns
- `src/expression_executor/gpu_expression_executor.cpp:283` — constructs new `gpu_table_representation` with output table
- `src/op/scan/duckdb_scan_executor.cpp` — wraps scanned data as gpu_table_representation
- `src/op/sirius_physical_table_scan.cpp:25` — accesses GPU data from batches
- Nearly every operator uses this to read input and produce output

---

### `cucascade::host_data_representation`

**Header**: `cucascade/data/cpu_data_representation.hpp`

```cpp
class host_data_representation : public idata_representation {
public:
  host_data_representation(std::unique_ptr<memory::host_table_allocation> host_table,
                           memory::memory_space* memory_space);

  std::size_t get_size_in_bytes() const override;
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;
  const std::unique_ptr<memory::host_table_allocation>& get_host_table() const;
};
```

**Description**: Wraps a host (CPU) table allocation. Used when data is downgraded from GPU to CPU memory.

**Our usage**:
- `src/downgrade/downgrade_task.cpp:24` — converts GPU data to host representation during downgrade
- `src/op/scan/duckdb_scan_executor.cpp:20` — creates host representation from scanned data
- `src/op/sirius_physical_result_collector.cpp:29` — reads host data for final result output

---

### `cucascade::host_data_packed_representation`

**Header**: `cucascade/data/cpu_data_representation.hpp`

```cpp
class host_data_packed_representation : public idata_representation {
public:
  host_data_packed_representation(
    std::unique_ptr<cucascade::memory::host_table_packed_allocation> host_table,
    cucascade::memory::memory_space* memory_space);

  std::size_t get_size_in_bytes() const override;
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;
  const std::unique_ptr<cucascade::memory::host_table_packed_allocation>& get_host_table() const;
};
```

**Description**: Packed (serialized) host table format. Uses cuDF's packed table format for efficient GPU↔CPU transfers. Used as an intermediate during representation conversion.

---

### `cucascade::data_batch`

**Header**: `cucascade/data/data_batch.hpp`

```cpp
enum class batch_state { idle, task_created, processing, in_transit };

enum class lock_for_processing_status {
  success, task_not_created, invalid_state,
  memory_space_mismatch, missing_data, not_attempted
};

struct lock_for_processing_result {
  bool success{false};
  data_batch_processing_handle handle{};
  lock_for_processing_status status{lock_for_processing_status::not_attempted};
};

class data_batch : public std::enable_shared_from_this<data_batch> {
public:
  data_batch(uint64_t batch_id, std::unique_ptr<idata_representation> data);

  // State queries
  memory::Tier get_current_tier() const;
  uint64_t get_batch_id() const;
  batch_state get_state() const;
  size_t get_processing_count() const;
  idata_representation* get_data() const;
  cucascade::memory::memory_space* get_memory_space() const;

  // State transitions (blocking)
  void wait_to_create_task();
  void wait_to_cancel_task();
  lock_for_processing_result wait_to_lock_for_processing(memory::memory_space_id requested);
  void wait_to_lock_for_in_transit();
  void wait_to_release_in_transit(std::optional<batch_state> target = std::nullopt);

  // State transitions (non-blocking)
  bool try_to_create_task();
  lock_for_processing_result try_to_lock_for_processing(memory::memory_space_id requested);
  bool try_to_lock_for_in_transit();
  bool try_to_release_in_transit(std::optional<batch_state> target = std::nullopt);

  // Data manipulation
  void set_data(std::unique_ptr<idata_representation> data);
  std::shared_ptr<data_batch> clone(uint64_t new_batch_id, rmm::cuda_stream_view stream);

  template <typename TargetRepresentation>
  void convert_to(representation_converter_registry& registry,
                  const memory::memory_space* target, rmm::cuda_stream_view stream);

  template <typename TargetRepresentation>
  std::shared_ptr<data_batch> clone_to(representation_converter_registry& registry,
                                       uint64_t new_batch_id,
                                       const memory::memory_space* target,
                                       rmm::cuda_stream_view stream);
};
```

**Description**: Thread-safe container for a single batch of data flowing through the pipeline. Implements a state machine (`idle` → `task_created` → `processing` → `idle`) with mutex-protected transitions. The `data_batch_processing_handle` RAII type automatically releases the processing lock on destruction.

**Our usage**:
- `src/pipeline/gpu_pipeline_task.cpp:24-28` — core pipeline task operates on data_batch, locks for processing, reads/writes data
- `src/op/scan/duckdb_scan_executor.cpp:21` — creates new data_batch from scanned data
- `src/op/sirius_physical_result_collector.cpp:30` — reads batch data for result output
- `src/creator/task_creator.hpp:32` — task creator queries batch states to schedule work
- `src/expression_executor/gpu_expression_executor.cpp:242` — accepts and returns data_batch

---

### `cucascade::idata_repository<PtrType>` / `shared_data_repository`

**Header**: `cucascade/data/data_repository.hpp`

```cpp
template <typename PtrType>
class idata_repository {
public:
  virtual void add_data_batch(PtrType batch, size_t partition_idx = 0);
  void notify_state_change();

  virtual PtrType pop_data_batch(batch_state target_state, size_t partition_idx = 0);
  virtual PtrType pop_data_batch_by_id(uint64_t batch_id, std::optional<batch_state> target,
                                       size_t partition_idx = 0);
  virtual PtrType get_data_batch_by_id(uint64_t batch_id, std::optional<batch_state> target,
                                       size_t partition_idx = 0);
  virtual std::vector<uint64_t> get_batch_ids(size_t partition_idx = 0) const;

  std::size_t size(size_t partition_idx = 0) const;
  bool empty(size_t partition_idx = 0) const;
  std::size_t total_size() const;
  bool all_empty() const;
  std::size_t num_partitions() const;
};

using shared_data_repository = idata_repository<std::shared_ptr<data_batch>>;
using unique_data_repository = idata_repository<std::unique_ptr<data_batch>>;
```

**Description**: Thread-safe container for data batches, partitioned by index. Supports popping batches that match a target state (for scheduling). `shared_data_repository` (shared_ptr) is used in pipelines where multiple consumers read the same batch; `unique_data_repository` for exclusive ownership.

**Our usage**:
- `src/include/op/sirius_physical_operator.hpp:34` — operators read from upstream repository
- `src/include/pipeline/gpu_pipeline_task.hpp:26` — pipeline tasks reference repositories
- `src/include/creator/task_creator.hpp:33` — task creator polls repositories for available data
- `test/cpp/operator/test_physical_concat.cpp:309` — tests create `shared_data_repository` instances

---

### `cucascade::data_repository_manager<PtrType>` / `shared_data_repository_manager`

**Header**: `cucascade/data/data_repository_manager.hpp`

```cpp
template <typename PtrType>
class data_repository_manager {
public:
  void add_new_repository(size_t operator_id, std::string_view port_id,
                          std::unique_ptr<repository_type> repository);
  void add_data_batch(PtrType batch, std::vector<std::pair<size_t, std::string_view>> ops);
  std::unique_ptr<repository_type>& get_repository(size_t operator_id, std::string_view port_id);
  uint64_t get_next_data_batch_id();
  std::vector<leaked_repository_info> clear_all_repositories();
  std::vector<PtrType> get_data_batches_for_downgrade(memory_space_id space, size_t amount);
  void for_each_repository(std::function<void(repository_type*)> visitor);
};

using shared_data_repository_manager = data_repository_manager<std::shared_ptr<data_batch>>;
```

**Description**: Manages all repositories across the query plan, keyed by `(operator_id, port_id)`. Each operator's input/output port gets its own repository. The `get_next_data_batch_id()` provides globally unique batch IDs.

**Our usage**:
- `src/sirius_engine.cpp:44` — engine owns the `data_repository_manager`
- `src/include/sirius_engine.hpp:29` — declared as member of `sirius_engine`
- `src/include/expression_executor/gpu_expression_executor.hpp:38` — expression executor references it
- `src/include/downgrade/downgrade_executor.hpp:25` — downgrade executor uses it to find batches to downgrade

---

### `cucascade::representation_converter_registry`

**Header**: `cucascade/data/representation_converter.hpp`

```cpp
class representation_converter_registry {
public:
  template <typename SourceType, typename TargetType>
  void register_converter(representation_converter_fn converter);

  template <typename SourceType, typename TargetType>
  bool has_converter() const;

  template <typename TargetType>
  std::unique_ptr<TargetType> convert(idata_representation& source,
                                      const memory::memory_space* target,
                                      rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;
};

void register_builtin_converters(representation_converter_registry& registry);
```

**Description**: Registry of type-erased conversion functions between data representations. `register_builtin_converters()` registers GPU↔Host converters.

**Our usage**:
- `src/include/data/sirius_converter_registry.hpp:19` — Sirius defines its own converter registry
- `src/include/data/host_parquet_representation_converters.hpp:20` — registers parquet-specific converters
- `test/cpp/data/test_host_parquet_representation.cpp:30` — tests use converter registry

## APIs Available but Not Used

All APIs in this module are used by our codebase.
