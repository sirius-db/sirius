# CUDA Streams

**Status**: USED
**Path**: `rmm/cuda_stream.hpp`, `rmm/cuda_stream_view.hpp`
**Headers we include**: `<rmm/cuda_stream.hpp>`, `<rmm/cuda_stream_view.hpp>`

## Summary

Provides strongly-typed CUDA stream wrappers. `cuda_stream_view` is a non-owning view (like `string_view`) used as the standard stream parameter across all Sirius GPU operations. `cuda_stream` is an owning RAII wrapper used when a dedicated stream is needed.

## API Reference

### `rmm::cuda_stream_view`

**Header**: `<rmm/cuda_stream_view.hpp>`
```cpp
class cuda_stream_view {
public:
  cuda_stream_view() = default;
  cuda_stream_view(cudaStream_t stream) noexcept;
  cuda_stream_view(cuda::stream_ref stream) noexcept;

  cudaStream_t value() const noexcept;
  operator cudaStream_t() const noexcept;
  operator cuda::stream_ref() const noexcept;

  bool is_per_thread_default() const noexcept;
  bool is_default() const noexcept;
  void synchronize() const;
  void synchronize_no_throw() const noexcept;
};
```

**Description**: Non-owning view of a CUDA stream. Implicitly convertible to/from `cudaStream_t` and `cuda::stream_ref`. Default-constructed view wraps the null/default stream.

**Our usage**:
- `src/include/pipeline/sirius_pipeline_itask.hpp:25` — Stream parameter for pipeline task interface
- `src/include/expression_executor/gpu_expression_executor.hpp:44` — Stream for GPU expression execution
- `src/include/operator/empty_str_check.cuh:21` — Stream parameter for CUDA kernel launches
- Virtually every operator and pipeline file uses this as the stream parameter type

### `rmm::cuda_stream`

**Header**: `<rmm/cuda_stream.hpp>`
```cpp
class cuda_stream {
public:
  enum class flags : unsigned int {
    sync_default = cudaStreamDefault,
    non_blocking = cudaStreamNonBlocking,
  };

  cuda_stream(flags f = flags::sync_default);
  ~cuda_stream() = default;  // RAII cleanup
  cuda_stream(cuda_stream&&) = default;
  cuda_stream(cuda_stream const&) = delete;

  bool is_valid() const;
  cudaStream_t value() const;
  explicit operator cudaStream_t() const noexcept;
  cuda_stream_view view() const;
  operator cuda_stream_view() const;  // implicit conversion
  void synchronize() const;
  void synchronize_no_throw() const noexcept;
};
```

**Description**: RAII wrapper that creates and destroys a CUDA stream. Move-only. Implicitly converts to `cuda_stream_view`.

**Our usage**:
- `src/downgrade/downgrade_executor.cpp:22` — Owns streams for downgrade operations
- `test/cpp/operator/test_host_table_chunk_reader.cpp:250` — Creates streams that must outlive data batches
- `test/cpp/scan/test_parquet_scan_task.cpp:323` — Owns streams for scan test operations

### `rmm::cuda_stream_default`

**Header**: `<rmm/cuda_stream_view.hpp>`
```cpp
static constexpr cuda_stream_view cuda_stream_default{};
```

**Description**: Pre-defined view of the default CUDA stream (stream 0).

**Our usage**:
- `src/operator/gpu_physical_ungrouped_aggregate.cpp:109` — Default stream for legacy operator execution
- `src/operator/gpu_physical_result_collector.cpp:338` — Default stream for result collection
- `src/operator/gpu_physical_nested_loop_join.cpp:389` — Default stream for legacy NLJ
- `test/cpp/data/test_host_parquet_representation.cpp:309` — Default stream for cloning in tests

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cuda_stream_legacy` | `cuda_stream_view.hpp` | View of `cudaStreamLegacy` |
| `cuda_stream_per_thread` | `cuda_stream_view.hpp` | View of `cudaStreamPerThread` |
| `cuda_stream::flags::non_blocking` | `cuda_stream.hpp` | Create non-blocking stream |
