# libkvikio — Module Reference

**Version**: 26.02.00
**Location**: `.pixi/envs/default/include/kvikio/`
**Namespace**: `kvikio`
**Conda package**: `libkvikio` (from rapidsai channel)

## Overview

KvikIO (pronounced "quick I/O") is a RAPIDS library providing GPU-accelerated file I/O. It wraps NVIDIA's cuFile/GDS (GPUDirect Storage) APIs with a C++ header-only interface that supports synchronous, asynchronous, parallel, and batched I/O between files and GPU memory. It also supports POSIX fallback, remote file access (S3, HTTP), and memory-mapped I/O.

## Module Map

| Module | Status | Description | Key APIs Used |
|--------|--------|-------------|---------------|
| file_io | UNUSED | Core file I/O via cuFile/GDS with sync, async, and parallel reads/writes | — |
| batch | UNUSED | Batched cuFile I/O operations | — |
| memory | UNUSED | GPU buffer registration and bounce buffer pools | — |
| remote | UNUSED | Remote file access (S3, HTTP, WebHDFS) | — |
| mmap | UNUSED | Memory-mapped file access | — |
| config | UNUSED | Global defaults, compatibility mode, and driver configuration | — |
| shim | UNUSED | Dynamic loading shims for CUDA and cuFile libraries | — |

## Our Usage Summary

This is a historical module inventory for the version listed above. Sirius and
cuCascade no longer provide a kvikIO backend and use **0 of 7 modules** directly.
libkvikio remains a **transitive libcudf dependency** in the RAPIDS conda ecosystem;
the vcpkg cudf/cuvs ports also retain their kvikio dependency and its build/cache
configuration. These package references do not enable a Sirius kvikIO backend.

### Why it's installed
- libcudf depends on libkvikio for its Parquet/ORC I/O subsystem (`cudf::io`)
- The dependency is declared in the pixi lock file but not in our direct dependencies

### Current Sirius I/O
- Local GDS reads and writes use cuCascade's cuFile backend and private cuFile loader.
- Object-store I/O uses cuCascade's native io_uring REST backend with TLS.
- `scan_manager.backend: kvikio` and the removed kvikIO configuration block are rejected.

## Files That Reference This Library

| Source File | Context | Notes |
|-------------|---------|-------|
| `pixi.lock` | Transitive dependency | Pulled in by libcudf |
| `test/tpch_performance/pixi.lock` | Transitive dependency | Same |
| `cucascade/.clang-format` | Include ordering regex | Historical RAPIDS include grouping |
| `sirius-duckdb/vcpkg_ports/{cudf,cuvs}/vcpkg.json` | Transitive dependency | Their upstream implementations still require kvikio |
| `sirius-duckdb/vcpkg_ports/kvikio/` | Transitive port | Builds the dependency for libcudf/cuvs, not a Sirius backend |
| `sirius-duckdb/vcpkg_triplets/{x64,arm64}-linux.cmake` | Binary cache key | Tracks the shared RAPIDS input used by the transitive kvikio port |
