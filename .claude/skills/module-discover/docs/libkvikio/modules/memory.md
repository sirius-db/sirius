# Memory Management

**Status**: UNUSED
**Path**: `kvikio/buffer.hpp`, `kvikio/bounce_buffer.hpp`

## Summary
Provides GPU buffer registration with cuFile (required for GDS) and a thread-safe bounce buffer pool for staging I/O through pinned host memory. Multiple allocator strategies are available (page-aligned, CUDA pinned, or both).

## Key APIs
- `buffer_register()` / `buffer_deregister()` — Register device memory with cuFile for DMA
- `memory_register()` / `memory_deregister()` — Register device memory allocations
- `BounceBufferPool<Allocator>` — Singleton thread-safe pool of reusable bounce buffers
- `BounceBufferPool::Buffer` — RAII wrapper for a borrowed bounce buffer
- `CudaPinnedAllocator` — Allocates CUDA pinned host memory
- `PageAlignedAllocator` — Allocates page-aligned host memory

## Potential Relevance
Low. Sirius already manages GPU memory via cuCascade/RMM. These APIs would only be relevant if integrating GDS directly.
