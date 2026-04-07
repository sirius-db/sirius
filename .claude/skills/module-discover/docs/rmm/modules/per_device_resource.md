# Per-Device Resources

**Status**: USED
**Path**: `rmm/mr/per_device_resource.hpp`
**Headers we include**: `<rmm/mr/per_device_resource.hpp>`

## Summary

Global per-device resource management. Maps CUDA device IDs to default memory resources. Provides thread-safe get/set functions. The initial default for any device is a `cuda_memory_resource` (raw cudaMalloc).

## API Reference

### `rmm::mr::get_current_device_resource()`

**Header**: `<rmm/mr/per_device_resource.hpp>`
```cpp
inline device_memory_resource* get_current_device_resource();
```

**Description**: Returns the `device_memory_resource*` for the current CUDA device. Thread-safe.

**Our usage**:
- `test/cpp/utils/data_utils.hpp:60` — Default MR for test data construction
- `test/cpp/utils/test_validation_utility.hpp:220,277` — `auto mr = rmm::mr::get_current_device_resource()`

### `rmm::mr::get_current_device_resource_ref()`

**Header**: `<rmm/mr/per_device_resource.hpp>`
```cpp
inline device_async_resource_ref get_current_device_resource_ref();
```

**Description**: Returns a `device_async_resource_ref` for the current CUDA device. Thread-safe. This is the default `mr` parameter value for `device_buffer` and `device_uvector` constructors.

**Our usage**:
- Implicitly used as default parameter in `device_buffer` and `device_uvector` constructors throughout the codebase

## APIs Available but Not Used

| API | Brief Description |
|-----|-------------------|
| `set_current_device_resource()` | Set the default MR for current device |
| `set_per_device_resource()` | Set MR for a specific device ID |
| `get_per_device_resource()` | Get MR for a specific device ID |
| `set_current_device_resource_ref()` | Set resource ref for current device |
| `reset_current_device_resource_ref()` | Reset to initial cuda MR |
