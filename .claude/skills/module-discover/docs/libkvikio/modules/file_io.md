# File I/O

**Status**: UNUSED
**Path**: `kvikio/file_handle.hpp`, `kvikio/stream.hpp`

## Summary
Core module providing GPU-accelerated file I/O via NVIDIA cuFile/GDS. Supports synchronous, asynchronous (CUDA stream-based), and parallel (thread pool-based) reads and writes between files and device memory. Falls back to POSIX I/O when cuFile is unavailable (controlled by `CompatMode`).

## Key APIs
- `FileHandle` — Main file handle class; open files, read/write with sync/async/parallel modes
- `FileHandle::read()` / `FileHandle::write()` — Synchronous GPU↔file transfers
- `FileHandle::pread()` / `FileHandle::pwrite()` — Parallel I/O using thread pool, returns `std::future`
- `FileHandle::read_async()` / `FileHandle::write_async()` — CUDA stream-based async I/O, returns `StreamFuture`
- `StreamFuture` — Future for async I/O operations; `check_bytes_done()` synchronizes stream
- `stream_register()` / `stream_deregister()` — Register CUDA streams with cuFile for async I/O

## Potential Relevance
High. If Sirius ever implements direct GPU file reading (bypassing DuckDB's CPU-based Parquet scanner), `FileHandle` with GDS could significantly reduce data loading latency by enabling direct storage-to-GPU transfers without CPU staging.
