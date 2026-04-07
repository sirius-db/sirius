# Memory-Mapped I/O

**Status**: UNUSED
**Path**: `kvikio/mmap.hpp`

## Summary
Provides memory-mapped file access with parallel read support. Maps file regions into memory for efficient sequential access, with thread pool-based parallel reads.

## Key APIs
- `MmapHandle` — Handle for memory-mapped files; supports sequential and parallel reads
- `MmapHandle::read()` — Sequential read from mapped region
- `MmapHandle::pread()` — Parallel read using thread pool

## Potential Relevance
Low. Memory-mapped I/O is CPU-oriented; Sirius's GPU execution model favors direct file-to-GPU transfers via GDS or explicit reads.
