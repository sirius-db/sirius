# Device Management

**Status**: USED
**Path**: `rmm/cuda_device.hpp`
**Headers we include**: `<rmm/cuda_device.hpp>`

## Summary

CUDA device identification and query utilities. `cuda_device_id` is a strong type wrapper for GPU device IDs. Utility functions query available memory and device count.

## API Reference

### `rmm::cuda_device_id`

**Header**: `<rmm/cuda_device.hpp>`
```cpp
struct cuda_device_id {
  using value_type = int;
  cuda_device_id() noexcept;               // current device
  explicit constexpr cuda_device_id(value_type dev_id) noexcept;
  constexpr value_type value() const noexcept;
  friend bool operator==(cuda_device_id const&, cuda_device_id const&) noexcept;
  friend bool operator!=(cuda_device_id const&, cuda_device_id const&) noexcept;
};
```

**Description**: Strong type for CUDA device IDs. Default constructor captures the current device.

**Our usage**:
- `src/include/cudf/cudf_utils.hpp:52` — Device management utilities
- `src/pipeline/gpu_pipeline_executor.cpp:29` — Pipeline executor device management
- `src/memory/sirius_memory_reservation_manager.cpp:24` — Memory reservation per device
- `src/data/host_parquet_representation_converters.cpp:35` — Device queries during conversion
- `src/op/scan/duckdb_scan_executor.cpp:34` — Scan executor device management

### `rmm::get_current_cuda_device()`

**Header**: `<rmm/cuda_device.hpp>`
```cpp
cuda_device_id get_current_cuda_device();
```

**Description**: Returns a `cuda_device_id` for the currently active CUDA device.

### `rmm::available_device_memory()`

**Header**: `<rmm/cuda_device.hpp>`
```cpp
std::pair<std::size_t, std::size_t> available_device_memory();
```

**Description**: Returns `{free_bytes, total_bytes}` for the current device.

### `rmm::cuda_set_device_raii`

**Header**: `<rmm/cuda_device.hpp>`
```cpp
struct cuda_set_device_raii {
  explicit cuda_set_device_raii(cuda_device_id dev_id);
  ~cuda_set_device_raii() noexcept;  // restores previous device
};
```

**Description**: RAII guard that sets CUDA device on construction and restores the previous device on destruction.

## APIs Available but Not Used

| API | Brief Description |
|-----|-------------------|
| `get_num_cuda_devices()` | Return total number of CUDA devices |
| `percent_of_free_device_memory()` | Calculate percentage of free GPU memory |
