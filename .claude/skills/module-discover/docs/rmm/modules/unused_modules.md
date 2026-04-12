# Unused RMM Modules

## Arena Memory Resource

**Status**: UNUSED
**Path**: `rmm/mr/arena_memory_resource.hpp`

Arena-based GPU allocator that pre-allocates large arenas and sub-allocates from them. Uses a different strategy than `pool_memory_resource` — better for workloads with many similar-sized allocations.

**Potential Relevance**: Could be an alternative to the pool MR for Sirius's processing memory if allocation patterns are uniform.

## CUDA Async Memory Resources

**Status**: UNUSED
**Path**: `rmm/mr/cuda_async_memory_resource.hpp`, `rmm/mr/cuda_async_managed_memory_resource.hpp`, `rmm/mr/cuda_async_view_memory_resource.hpp`

Wrappers around `cudaMallocAsync`/`cudaFreeAsync` (CUDA 11.2+). Leverages CUDA's built-in memory pool with stream-ordered semantics.

**Potential Relevance**: Could simplify memory management if CUDA driver-level pooling is sufficient. May offer better performance than RMM's pool MR on newer drivers.

## Managed Memory Resource

**Status**: UNUSED
**Path**: `rmm/mr/managed_memory_resource.hpp`, `rmm/mr/sam_headroom_memory_resource.hpp`, `rmm/mr/system_memory_resource.hpp`

Managed (unified) memory resources using `cudaMallocManaged`. Includes System Addressable Memory (SAM) support.

**Potential Relevance**: Not applicable — Sirius explicitly manages GPU/CPU tiers via cuCascade.

## Logging/Statistics/Tracking MR Adaptors

**Status**: UNUSED
**Path**: `rmm/mr/logging_resource_adaptor.hpp`, `rmm/mr/statistics_resource_adaptor.hpp`, `rmm/mr/tracking_resource_adaptor.hpp`

Wrapper MRs that log allocations, collect statistics, or track outstanding allocations.

- `logging_resource_adaptor` — Logs every allocate/deallocate call
- `statistics_resource_adaptor` — Tracks peak/current memory usage
- `tracking_resource_adaptor` — Tracks outstanding allocations for leak detection

**Potential Relevance**: Useful for debugging memory issues. Could wrap the pool MR to diagnose allocation patterns or detect leaks.

## Other Unused MR Adaptors

**Status**: UNUSED

| Resource | Path | Description |
|----------|------|-------------|
| `aligned_resource_adaptor` | `rmm/mr/aligned_resource_adaptor.hpp` | Enforces alignment > 256 bytes |
| `binning_memory_resource` | `rmm/mr/binning_memory_resource.hpp` | Routes allocations to size-specific pools |
| `callback_memory_resource` | `rmm/mr/callback_memory_resource.hpp` | Delegates to user-provided callbacks |
| `failure_callback_resource_adaptor` | `rmm/mr/failure_callback_resource_adaptor.hpp` | Calls a callback on allocation failure |
| `fixed_size_memory_resource` | `rmm/mr/fixed_size_memory_resource.hpp` | Pool for fixed-size allocations |
| `limiting_resource_adaptor` | `rmm/mr/limiting_resource_adaptor.hpp` | Enforces a maximum allocation limit |
| `owning_wrapper` | `rmm/mr/owning_wrapper.hpp` | Wraps MR with ownership semantics |
| `pinned_host_memory_resource` | `rmm/mr/pinned_host_memory_resource.hpp` | Pinned (page-locked) host memory |
| `prefetch_resource_adaptor` | `rmm/mr/prefetch_resource_adaptor.hpp` | Prefetches managed memory to device |
| `thread_safe_resource_adaptor` | `rmm/mr/thread_safe_resource_adaptor.hpp` | Adds mutex to non-thread-safe MR |
| `thrust_allocator_adaptor` | `rmm/mr/thrust_allocator_adaptor.hpp` | Adapts MR for use with Thrust |
| `polymorphic_allocator` | `rmm/mr/polymorphic_allocator.hpp` | C++ PMR-compatible allocator |

**Potential Relevance**: `failure_callback_resource_adaptor` could be useful for OOM handling; `limiting_resource_adaptor` could enforce per-query memory budgets.

## Prefetch

**Status**: UNUSED
**Path**: `rmm/prefetch.hpp`

Utility for prefetching managed memory to a specific device.

**Potential Relevance**: Not applicable — Sirius doesn't use managed memory.

## Exec Policy

**Status**: UNUSED
**Path**: `rmm/exec_policy.hpp`

Thrust execution policy that uses a specified CUDA stream.

**Potential Relevance**: Useful if Sirius needs stream-aware Thrust operations (currently uses cudf's stream-aware APIs).

## Device Scalar / Device Vector

**Status**: UNUSED
**Path**: `rmm/device_scalar.hpp`, `rmm/device_vector.hpp`

- `device_scalar<T>` — Single value in device memory
- `device_vector<T>` — Thrust-compatible device vector (initializes elements, unlike `device_uvector`)

**Potential Relevance**: `device_scalar` could be useful for single-value GPU results; `device_vector` is generally superseded by `device_uvector` for performance.

## CUDA Stream Pool

**Status**: UNUSED
**Path**: `rmm/cuda_stream_pool.hpp`

Pool of reusable CUDA streams.

**Potential Relevance**: Sirius manages its own stream-per-thread model in the GPU thread pool, so this is redundant.
