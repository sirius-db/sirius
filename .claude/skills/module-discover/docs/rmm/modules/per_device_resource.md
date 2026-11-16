# Per-Device Resources

**Status**: USED
**Path**: `rmm/mr/per_device_resource.hpp`
**Headers we include**: `<rmm/mr/per_device_resource.hpp>`

## Summary

Global per-device resource management. Maps CUDA device IDs to owned default memory resources (`cuda::mr::any_resource`); getters return `device_async_resource_ref`. Provides thread-safe get/set functions. The initial default for any device is a `cuda_memory_resource` (raw cudaMalloc).

## API Reference

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
| `set_current_device_resource(any_resource)` | Set the default MR for current device; returns the previous one as an owning `any_resource` |
| `set_per_device_resource(cuda_device_id, any_resource)` | Set MR for a specific device ID |
| `get_per_device_resource_ref(cuda_device_id)` | Get MR ref for a specific device ID |
| `reset_current_device_resource()` | Reset to initial cuda MR |
