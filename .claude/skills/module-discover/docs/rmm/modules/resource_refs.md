# Resource Refs

**Status**: USED
**Path**: `rmm/resource_ref.hpp`
**Headers we include**: `<rmm/resource_ref.hpp>`

## Summary

Type-erased resource reference types based on CCCL's `cuda::mr::resource_ref`. These are the modern way to pass memory resources without template parameters. `device_async_resource_ref` is the standard allocator parameter type throughout Sirius operators and kernels.

## API Reference

### `rmm::device_async_resource_ref`

**Header**: `<rmm/resource_ref.hpp>`
```cpp
using device_async_resource_ref =
  detail::cccl_async_resource_ref<cuda::mr::resource_ref<cuda::mr::device_accessible>>;
```

**Description**: Type-erased reference to an async device memory resource. Constructible from any `device_memory_resource*` or `device_memory_resource&`. This is the standard parameter type for passing allocators.

**Our usage**:
- `src/include/sirius_context.hpp:28` — Stored in `sirius_context` for operator access
- `src/include/operator/empty_str_check.cuh:22` — Kernel parameter type
- `src/include/operator/strlen_from_offsets.cuh:22` — Kernel parameter type
- `src/include/expression_executor/gpu_dispatcher.hpp:23` — Expression executor allocator
- `src/include/expression_executor/gpu_expression_executor.hpp:45` — Expression executor allocator
- `src/op/sirius_physical_top_n.cpp:30` — Top-N operator allocator
- `src/op/sirius_physical_ungrouped_aggregate.cpp:36` — Aggregate operator allocator
- `src/op/sirius_physical_nested_loop_join.cpp:38` — NLJ operator allocator
- `test/cpp/operator/operator_test_utils.hpp:86` — `get_resource_ref()` helper returns this type

### `rmm::to_device_async_resource_ref_checked`

**Header**: `<rmm/resource_ref.hpp>`
```cpp
template <class Resource>
device_async_resource_ref to_device_async_resource_ref_checked(Resource* res);
```

**Description**: Converts a raw pointer to a memory resource into a `device_async_resource_ref`, throwing `std::logic_error` if the pointer is null.

**Our usage**:
- `test/cpp/operator/operator_test_utils.hpp:88` — `return rmm::to_device_async_resource_ref_checked(space.get_default_allocator())`
- `test/cpp/expression_executor/test_gpu_expression_executor.cpp:78` — Same pattern

### `rmm::host_device_async_resource_ref`

**Header**: `<rmm/resource_ref.hpp>`
```cpp
using host_device_async_resource_ref = detail::cccl_async_resource_ref<
  cuda::mr::resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>>;
```

**Description**: Reference to a resource that provides both host and device accessible memory.

**Our usage**:
- `test/cpp/data/test_host_parquet_representation.cpp:484` — Used in parquet representation tests

## Other Type Aliases (Available but Less Used)

| API | Brief Description |
|-----|-------------------|
| `device_resource_ref` | Synchronous device resource reference |
| `host_resource_ref` | Synchronous host resource reference |
| `host_async_resource_ref` | Async host resource reference |
| `host_device_resource_ref` | Synchronous host+device resource reference |
