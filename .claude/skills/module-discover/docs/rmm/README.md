# RMM (RAPIDS Memory Manager) — Module Reference

**Version**: 26.2.0
**Location**: `.pixi/envs/default/include/rmm/`
**Namespace**: `rmm`, `rmm::mr`
**CMake target**: `rmm::rmm`

## Module Map

| Module | Status | Description | Key APIs Used |
|--------|--------|-------------|---------------|
| CUDA Streams | USED | CUDA stream wrappers (owning & non-owning) | `cuda_stream_view`, `cuda_stream`, `cuda_stream_default` |
| Device Containers | USED | GPU memory containers (buffer, uvector) | `device_buffer`, `device_uvector` |
| Memory Resources | USED | Polymorphic device memory allocators | `device_memory_resource`, `cuda_memory_resource`, `pool_memory_resource` |
| Resource Refs | USED | Type-erased resource references (CCCL-based) | `device_async_resource_ref`, `to_device_async_resource_ref_checked` |
| Per-Device Resources | USED | Global per-device resource management | `get_current_device_resource`, `get_current_device_resource_ref` |
| Device Management | USED | CUDA device ID and queries | `cuda_device_id`, `get_current_cuda_device`, `available_device_memory` |
| Alignment Utilities | USED | Memory alignment helpers | `align_up`, `CUDA_ALLOCATION_ALIGNMENT` |
| Error Handling | USED | Exception types and CUDA error macros | `out_of_memory`, `bad_alloc`, `RMM_CUDA_TRY` |
| Arena MR | UNUSED | Arena-based memory resource | — |
| Async MRs | UNUSED | CUDA async memory resources (cudaMallocAsync) | — |
| Managed Memory | UNUSED | Managed/unified memory resources | — |
| Logging/Stats MRs | UNUSED | Logging, statistics, tracking adaptors | — |
| Pinned Host Memory | UNUSED | Pinned host memory resource | — |
| Other MR Adaptors | UNUSED | Binning, fixed-size, limiting, callback, etc. | — |
| Prefetch | UNUSED | Memory prefetch utilities | — |
| Exec Policy | UNUSED | Thrust execution policy with stream | — |
| Device Scalar/Vector | UNUSED | Higher-level device containers | — |

## Our Usage Summary

We use **8 of 17** modules. Primary integration points:
- **Stream management**: `rmm::cuda_stream_view` is the standard stream parameter type across all GPU operators, pipeline tasks, and expression executors
- **Device memory**: `rmm::device_buffer` and `rmm::device_uvector` for GPU memory allocations in operators and CUDA kernels
- **Memory resources**: `rmm::mr::pool_memory_resource<cuda_memory_resource>` for Sirius's GPU processing memory pool
- **Resource refs**: `rmm::device_async_resource_ref` as the standard allocator parameter type in operators and kernels
- **Error handling**: `rmm::out_of_memory` caught in pipeline OOM reschedule logic

## Files That Reference This Library

| Source File | Modules Used | Key APIs |
|-------------|-------------|----------|
| `src/gpu_buffer_manager.cpp` | Memory Resources, Alignment, Streams | `pool_memory_resource`, `cuda_memory_resource`, `align_up` |
| `src/include/cudf/cudf_utils.hpp` | Device Mgmt, Memory Resources | `cuda_device_id`, `cuda_memory_resource`, `pool_memory_resource` |
| `src/include/sirius_context.hpp` | Resource Refs | `device_async_resource_ref` |
| `src/include/pipeline/sirius_pipeline_itask.hpp` | Streams | `cuda_stream_view` |
| `src/include/memory/sirius_memory_reservation_manager.hpp` | Memory Resources | `device_memory_resource` |
| `src/cuda/operator/empty_str_check.cu` | Containers, Resource Refs, Streams | `device_uvector`, `device_buffer`, `device_async_resource_ref` |
| `src/cuda/operator/strlen_from_offsets.cu` | Containers, Resource Refs, Streams | `device_uvector`, `device_buffer` |
| `src/expression_executor/specializations/gpu_execute_operator.cpp` | Containers | `device_uvector` |
| `src/op/scan/prefetched_data_source.cpp` | Error Handling, Containers | `rmm::detail::error`, `device_buffer` |
| `src/downgrade/downgrade_executor.cpp` | Streams | `cuda_stream` |
| `src/pipeline/gpu_pipeline_executor.cpp` | Device Mgmt | `cuda_device_id` |
| `src/data/host_parquet_representation_converters.cpp` | Device Mgmt, Streams, Resource Refs | `cuda_device_id`, `cuda_stream_view` |
| `src/op/sirius_physical_top_n.cpp` | Containers, Resource Refs | `device_buffer` |
| `src/op/sirius_physical_ungrouped_aggregate.cpp` | Resource Refs | `device_async_resource_ref` |
| `src/op/sirius_physical_nested_loop_join.cpp` | Resource Refs | `device_async_resource_ref` |
| `src/operator/gpu_physical_ungrouped_aggregate.cpp` | Streams | `cuda_stream_default` |
| `src/operator/gpu_physical_result_collector.cpp` | Streams | `cuda_stream_default` |
| `src/operator/gpu_physical_nested_loop_join.cpp` | Streams | `cuda_stream_default` |
