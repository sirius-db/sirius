# Batch I/O

**Status**: UNUSED
**Path**: `kvikio/batch.hpp`

## Summary
Provides batched cuFile I/O operations for submitting multiple read/write requests in a single call. Useful for reading many file regions into GPU memory with reduced overhead.

## Key APIs
- `BatchHandle` — Handle for batch I/O operations
- `BatchOp` — Struct describing a single I/O operation (file, offset, size, device pointer)
- `BatchHandle::submit()` — Submit a vector of `BatchOp` operations
- `BatchHandle::status()` — Poll for completion events
- `BatchHandle::cancel()` — Cancel pending operations

## Potential Relevance
Medium. Could be useful if Sirius implements its own Parquet reader that needs to read many row groups/column chunks in parallel from a single file.
