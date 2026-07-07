# Memory Resources

**Status**: USED
**Path**: `rmm/mr/device_memory_resource.hpp`, `rmm/mr/cuda_memory_resource.hpp`, `rmm/mr/pool_memory_resource.hpp`
**Headers we include**: `<rmm/mr/device_memory_resource.hpp>`, `<rmm/mr/cuda_memory_resource.hpp>`, `<rmm/mr/pool_memory_resource.hpp>`

## Summary

Polymorphic memory resource classes for GPU memory allocation. Sirius uses `cuda_memory_resource` (raw cudaMalloc) as the upstream allocator and wraps it in `pool_memory_resource` for the GPU processing memory pool.

## API Reference

### `rmm::mr::device_memory_resource`

**Header**: `<rmm/mr/device_memory_resource.hpp>`
```cpp
class device_memory_resource {
public:
  virtual ~device_memory_resource() = default;

  void* allocate(cuda_stream_view stream, std::size_t bytes,
                 std::size_t alignment = CUDA_ALLOCATION_ALIGNMENT);
  void deallocate(cuda_stream_view stream, void* ptr, std::size_t bytes,
                  std::size_t alignment = CUDA_ALLOCATION_ALIGNMENT) noexcept;
  void* allocate_sync(std::size_t bytes, std::size_t alignment = CUDA_ALLOCATION_ALIGNMENT);
  void deallocate_sync(void* ptr, std::size_t bytes,
                       std::size_t alignment = CUDA_ALLOCATION_ALIGNMENT) noexcept;
  bool is_equal(device_memory_resource const& other) const noexcept;
};
```

**Description**: Abstract base class for all device memory allocators. Stream-ordered allocation API. **Important (v26.x)**: `allocate(stream, bytes)` and `deallocate(stream, ptr, bytes)` — stream is the FIRST parameter.

**Our usage**:
- `src/include/memory/sirius_memory_reservation_manager.hpp:20` — Type used for memory resource pointers
- `src/gpu_buffer_manager.cpp` — Pool MR's upstream; `mr->deallocate(rmm::cuda_stream_view{}, ptr, size)`

### `rmm::mr::cuda_memory_resource`

**Header**: `<rmm/mr/cuda_memory_resource.hpp>`
```cpp
class cuda_memory_resource final : public device_memory_resource {
public:
  cuda_memory_resource() = default;
  // Internally uses cudaMalloc / cudaFree
};
```

**Description**: Simplest MR — wraps `cudaMalloc`/`cudaFree`. Stream argument is ignored (allocation is synchronous).

**Our usage**:
- `src/gpu_buffer_manager.cpp:171` — `cuda_mr = new rmm::mr::cuda_memory_resource()`
- `test/cpp/data/test_host_parquet_representation.cpp:78` — Creates cuda MR in test setup

### `rmm::mr::pool_memory_resource<Upstream>`

**Header**: `<rmm/mr/pool_memory_resource.hpp>`
```cpp
template <typename Upstream>
class pool_memory_resource final : public /* stream_ordered_memory_resource */ {
public:
  explicit pool_memory_resource(Upstream* upstream_mr,
                                std::size_t initial_pool_size,
                                std::optional<std::size_t> maximum_pool_size = std::nullopt);
  explicit pool_memory_resource(device_async_resource_ref upstream_mr,
                                std::size_t initial_pool_size,
                                std::optional<std::size_t> maximum_pool_size = std::nullopt);
  ~pool_memory_resource() override;  // releases all upstream memory

  device_async_resource_ref get_upstream_resource() const noexcept;
  std::size_t pool_size() const noexcept;
};
```

**Description**: Coalescing best-fit suballocator. Allocates a large pool from upstream and sub-allocates from it. Thread-safe. Grows geometrically up to `maximum_pool_size`. On destruction, returns all memory to upstream.

**Our usage**:
- `src/gpu_buffer_manager.cpp:172` — `mr = new rmm::mr::pool_memory_resource(cuda_mr, processing_size_per_gpu, processing_size_per_gpu)` — Fixed-size pool for GPU processing memory

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `pool_memory_resource::pool_size()` | `pool_memory_resource.hpp` | Query current pool size |
| `pool_memory_resource::get_upstream_resource()` | `pool_memory_resource.hpp` | Get upstream resource ref |
| `device_memory_resource::allocate_sync()` | `device_memory_resource.hpp` | Synchronous allocation |
