# Memory Resources

**Status**: USED
**Path**: `rmm/mr/cuda_memory_resource.hpp`, `rmm/mr/pool_memory_resource.hpp`
**Headers we include**: `<rmm/mr/cuda_memory_resource.hpp>`, `<rmm/mr/pool_memory_resource.hpp>`

## Summary

Device memory resources. RMM has no polymorphic `device_memory_resource` base class: resources satisfy the CCCL `cuda::mr` resource concept and are passed as `rmm::device_async_resource_ref` (non-owning) or `cuda::mr::any_resource<cuda::mr::device_accessible>` (owning). Sirius uses `cuda_memory_resource` (raw cudaMalloc) as the upstream allocator and wraps it in `pool_memory_resource` for the GPU processing memory pool.

## API Reference

### Resource concept

Every resource provides:
```cpp
void* allocate(cuda::stream_ref stream, std::size_t bytes,
               std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);
void deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes,
                std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);
void deallocate_sync(void* ptr, std::size_t bytes,
                     std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;
bool operator==(Resource const&) const noexcept;
friend void get_property(Resource const&, cuda::mr::device_accessible) noexcept;
```

The stream is the FIRST parameter of `allocate`/`deallocate`.

### `rmm::mr::cuda_memory_resource`

**Header**: `<rmm/mr/cuda_memory_resource.hpp>`
```cpp
class cuda_memory_resource final {
public:
  cuda_memory_resource() = default;
  // Resource concept members; internally uses cudaMalloc / cudaFree
};
```

**Description**: Simplest MR — wraps `cudaMalloc`/`cudaFree`. Stream argument is ignored (allocation is synchronous).

**Our usage**:
- `test/cpp/data/test_host_parquet_representation.cpp:78` — Creates cuda MR in test setup

### `rmm::mr::pool_memory_resource`

**Header**: `<rmm/mr/pool_memory_resource.hpp>`
```cpp
// Copyable; copies share the pool.
class pool_memory_resource
  : public cuda::mr::shared_resource<detail::pool_memory_resource_impl> {
public:
  explicit pool_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                                std::size_t initial_pool_size,
                                std::optional<std::size_t> maximum_pool_size = std::nullopt);

  device_async_resource_ref get_upstream_resource() const noexcept;
  std::size_t pool_size() const noexcept;
};
```

**Description**: Coalescing best-fit suballocator. Allocates a large pool from upstream and sub-allocates from it. Thread-safe. Grows geometrically up to `maximum_pool_size`. Returns all memory to upstream when the last copy is destroyed.

**Our usage**:

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `pool_memory_resource::pool_size()` | `pool_memory_resource.hpp` | Query current pool size |
| `pool_memory_resource::get_upstream_resource()` | `pool_memory_resource.hpp` | Get upstream resource ref |
