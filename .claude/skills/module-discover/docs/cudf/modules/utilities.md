# Utilities

**Status**: USED
**Path**: `cudf/utilities/`
**Headers we include**: `cudf/utilities/default_stream.hpp`, `cudf/utilities/memory_resource.hpp`, `cudf/utilities/pinned_memory.hpp`, `cudf/utilities/type_dispatcher.hpp`, `cudf/utilities/span.hpp`, `cudf/utilities/bit.hpp`, `cudf/utilities/error.hpp`

## Summary

Utility functions for CUDA stream management, RMM memory resource control, type dispatch, and memory helpers. These are used pervasively throughout Sirius for GPU resource management.

## API Reference

### `cudf::get_default_stream()`

**Header**: `cudf/utilities/default_stream.hpp`
```cpp
rmm::cuda_stream_view get_default_stream();
```

**Our usage** (20+ sites):
- `src/parallel/task_executor.cpp:19` — Stream for task execution
- `src/include/parallel/task.hpp:21` — Default stream for tasks
- `test/cpp/utils/data_utils.hpp:26` — Test utilities

### `cudf::get_current_device_resource_ref()` / `cudf::set_current_device_resource()`

**Header**: `cudf/utilities/memory_resource.hpp`
```cpp
rmm::device_async_resource_ref get_current_device_resource_ref();
rmm::mr::device_memory_resource* set_current_device_resource(rmm::mr::device_memory_resource* new_mr);
```

**Our usage**:
- `src/memory/sirius_memory_reservation_manager.cpp:42` — Swapping memory resources for reservation-aware allocation
- `src/op/sirius_physical_ungrouped_aggregate.cpp` — Passing resource to cuDF operations
- `src/op/sirius_physical_hash_join.cpp:26` — Resource for join allocation

### Pinned Memory Configuration

**Header**: `cudf/utilities/pinned_memory.hpp`
```cpp
size_t get_allocate_host_as_pinned_threshold();
void set_allocate_host_as_pinned_threshold(size_t threshold);
rmm::host_async_resource_ref get_pinned_memory_resource();
void set_pinned_memory_resource(rmm::host_async_resource_ref mr);
```

**Our usage**:
- `src/sirius_context.cpp:27` — Configuring pinned memory for GPU↔CPU transfers at startup

### `cudf::type_dispatcher`

**Header**: `cudf/utilities/type_dispatcher.hpp`
```cpp
template <typename Functor, typename... Ts>
decltype(auto) type_dispatcher(data_type dtype, Functor f, Ts&&... args);
```

**Description**: Dispatches to a functor's `operator()` template based on runtime `data_type`. Enables type-generic code without switch statements.

**Our usage**:
- `src/expression_executor/gpu_expression_executor.cpp:35` — Type-based expression dispatch
- `src/cuda/print.cu:28` — Debug printing of columns
- `test/cpp/utils/test_validation_utility.hpp:28` — Test result validation

### `cudf::host_span<T>` / `cudf::device_span<T>`

**Header**: `cudf/utilities/span.hpp`
```cpp
template <typename T>
class host_span {
    host_span(T* data, size_t size);
    T* data() const;
    size_t size() const;
};
```

**Our usage**:
- `src/data/host_parquet_representation_converters.cpp:32` — Span over host parquet data
- `src/include/memory/host_table_utils.hpp` — Host table data references

### Bit Utilities

**Header**: `cudf/utilities/bit.hpp`
```cpp
bool bit_is_set(bitmask_type const* bitmask, size_type bit_index);
size_type word_index(size_type bit_index);
size_type intra_word_index(size_type bit_index);
```

**Our usage**:
- `test/cpp/operator/test_host_table_chunk_reader.cpp:30` — Null mask inspection in tests
- `test/cpp/memory/test_host_table_utils.cpp:38` — Bit manipulation in tests

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::prefetch()` | `prefetch.hpp` | Memory prefetch hints |
| `cudf::is_fixed_width()` | `traits.hpp` | Check if type is fixed-width |
| `cudf::is_numeric()` | `traits.hpp` | Check if type is numeric |
| `CUDF_EXPECTS()` | `error.hpp` | Assertion macro with message |
| `CUDF_FAIL()` | `error.hpp` | Unconditional failure macro |
