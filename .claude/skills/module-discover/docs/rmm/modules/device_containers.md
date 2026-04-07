# Device Containers

**Status**: USED
**Path**: `rmm/device_buffer.hpp`, `rmm/device_uvector.hpp`
**Headers we include**: `<rmm/device_buffer.hpp>`, `<rmm/device_uvector.hpp>`

## Summary

RAII containers for uninitialized device memory. `device_buffer` holds raw bytes; `device_uvector<T>` is a typed vector that skips initialization for performance. Both are move-only and stream-aware. These are used throughout Sirius for GPU data storage in operators and CUDA kernels.

## API Reference

### `rmm::device_buffer`

**Header**: `<rmm/device_buffer.hpp>`
```cpp
class device_buffer {
public:
  device_buffer();  // empty buffer
  explicit device_buffer(std::size_t size, cuda_stream_view stream,
                         device_async_resource_ref mr = mr::get_current_device_resource_ref());
  device_buffer(void const* source_data, std::size_t size, cuda_stream_view stream,
                device_async_resource_ref mr = mr::get_current_device_resource_ref());
  device_buffer(device_buffer const& other, cuda_stream_view stream,
                device_async_resource_ref mr = mr::get_current_device_resource_ref());
  device_buffer(device_buffer&& other) noexcept;
  device_buffer& operator=(device_buffer&& other) noexcept;
  ~device_buffer() noexcept;

  void reserve(std::size_t new_capacity, cuda_stream_view stream);
  void resize(std::size_t new_size, cuda_stream_view stream);
  void shrink_to_fit(cuda_stream_view stream);

  void const* data() const noexcept;
  void* data() noexcept;
  std::size_t size() const noexcept;
  std::int64_t ssize() const noexcept;
  bool is_empty() const noexcept;
  std::size_t capacity() const noexcept;
  cuda_stream_view stream() const noexcept;
  void set_stream(cuda_stream_view stream) noexcept;
  device_async_resource_ref memory_resource() noexcept;
};
```

**Description**: Untyped, uninitialized device memory with RAII. Supports host-to-device copy construction. Copy ctor/assignment deleted (must specify stream explicitly).

**Our usage**:
- `src/op/sirius_physical_top_n.cpp:175` — Creates null mask buffers: `std::make_unique<rmm::device_buffer>(std::move(new_mask))`
- `src/cuda/operator/empty_str_check.cu:69` — Empty buffer for null mask: `rmm::device_buffer(0, stream, mr)`
- `src/cuda/operator/strlen_from_offsets.cu:69` — Same pattern for column construction
- `test/cpp/utils/data_utils.hpp:95` — Creates char buffers for string column construction
- `test/cpp/operator/operator_test_utils.hpp:249` — Allocates char data for test string columns
- `test/cpp/memory/test_host_table_utils.cpp:280` — Copies host mask to device

### `rmm::device_uvector<T>`

**Header**: `<rmm/device_uvector.hpp>`
```cpp
template <typename T>
class device_uvector {
  static_assert(std::is_trivially_copyable_v<T>);
public:
  explicit device_uvector(size_type size, cuda_stream_view stream,
                          device_async_resource_ref mr = mr::get_current_device_resource_ref());
  explicit device_uvector(device_uvector const& other, cuda_stream_view stream,
                          device_async_resource_ref mr = mr::get_current_device_resource_ref());
  device_uvector(device_uvector&&) noexcept = default;
  device_uvector() = delete;  // no default ctor

  void set_element_async(size_type idx, value_type const& value, cuda_stream_view stream);
  void set_element(size_type idx, T const& value, cuda_stream_view stream);
  value_type element(size_type idx, cuda_stream_view stream) const;

  void reserve(size_type new_capacity, cuda_stream_view stream);
  void resize(size_type new_size, cuda_stream_view stream);
  device_buffer release() noexcept;

  pointer data() noexcept;
  const_pointer data() const noexcept;
  size_type size() const noexcept;
  bool is_empty() const noexcept;
  size_type capacity() const noexcept;

  iterator begin() noexcept;
  iterator end() noexcept;
  // + const/reverse iterators

  operator cuda::std::span<T>() noexcept;
  operator cuda::std::span<T const>() const noexcept;
  device_async_resource_ref memory_resource() noexcept;
  cuda_stream_view stream() const noexcept;
  void set_stream(cuda_stream_view stream) noexcept;
};
```

**Description**: Typed, uninitialized device vector. Only supports trivially copyable types. Unlike `thrust::device_vector`, does not default-initialize elements (better performance). All allocation/copy ops take a stream parameter.

**Our usage**:
- `src/cuda/operator/empty_str_check.cu:56` — `rmm::device_uvector<bool> output(num_rows, stream, mr)` for kernel output
- `src/cuda/operator/strlen_from_offsets.cu:56` — `rmm::device_uvector<int32_t> output(num_rows, stream, mr)` for string length calculation
- `src/expression_executor/specializations/gpu_execute_operator.cpp:28` — Used in expression evaluation
- `src/cuda/expression_executor/gpu_dispatch_materialize.cu:21` — Used in materialization kernels

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `device_uvector::front_element()` | `device_uvector.hpp` | Get first element (sync) |
| `device_uvector::back_element()` | `device_uvector.hpp` | Get last element (sync) |
| `device_uvector::set_element_to_zero_async()` | `device_uvector.hpp` | Optimized zero-set via memset |
| `device_buffer::ssize()` | `device_buffer.hpp` | Signed size |
